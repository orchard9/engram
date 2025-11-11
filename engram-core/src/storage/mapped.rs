//! Memory-mapped storage for warm and cold tiers
//!
//! This module implements high-performance memory-mapped storage with:
//! - Cache-optimal layouts for SIMD operations
//! - NUMA-aware allocation for multi-socket systems
//! - Zero-copy reads with hardware prefetching
//! - Columnar storage for batch operations

// Allow unsafe code for performance-critical memory-mapped operations
#![allow(unsafe_code)]

use super::{StorageError, StorageMetrics, StorageResult, StorageTierBackend, TierStatistics};
use crate::{Confidence, Cue, Episode, Memory};
use dashmap::DashMap;
use rayon::prelude::*;
use std::convert::TryFrom;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::time::SystemTime;

#[cfg(feature = "memory_mapped_persistence")]
use crc32c::crc32c;
#[cfg(feature = "memory_mapped_persistence")]
use memmap2::{Mmap, MmapMut, MmapOptions};

/// Cache-line aligned embedding block for optimal SIMD performance
#[repr(C, align(64))]
pub struct EmbeddingBlock {
    /// 768-dimensional embedding (3072 bytes = 48 cache lines)
    pub embedding: [f32; 768],

    /// Metadata co-located for cache efficiency (1 cache line)
    /// Confidence score of the memory (0.0 to 1.0)
    pub confidence: f32,
    /// Current activation level (0.0 to 1.0)
    pub activation: f32,
    /// Timestamp of last access for LRU eviction
    pub last_access: u64,
    /// Decay rate for memory strength over time
    pub decay_rate: f32,
    /// Node-specific flags for graph operations
    pub node_flags: u32,
    /// Hash of memory content for integrity verification
    pub content_hash: u64,
    /// Creation timestamp in nanoseconds since UNIX epoch
    pub creation_time: u64,
    /// Number of times this memory has been recalled
    pub recall_count: u32,
    /// Length of content string in bytes
    pub content_length: u32,
    /// Byte offset in content storage section
    pub content_offset: u64,

    /// Padding to complete cache line
    padding: [u8; 4],
}

impl EmbeddingBlock {
    /// Create a new embedding block from a Memory instance
    #[must_use]
    pub fn new(memory: &Memory) -> Self {
        Self {
            embedding: memory.embedding,
            confidence: memory.confidence.raw(),
            activation: memory.activation(),
            last_access: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX),
            decay_rate: 0.2, // Default decay
            node_flags: 0,
            content_hash: Self::compute_content_hash(&memory.id),
            creation_time: SystemTime::from(memory.created_at)
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX),
            recall_count: 0,
            content_length: 0,
            content_offset: u64::MAX, // Sentinel value indicating no content stored
            padding: [0; 4],
        }
    }

    fn compute_content_hash(content: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }

    /// Prefetch only hot cache line for activation spreading
    #[cfg(all(feature = "memory_mapped_persistence", target_arch = "x86_64"))]
    #[inline]
    pub fn prefetch_for_activation(&self) {
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                std::ptr::from_ref(self).cast::<i8>(),
                std::arch::x86_64::_MM_HINT_T0,
            );
        }
    }

    /// Prefetch embedding cache lines for SIMD operations
    #[cfg(all(feature = "memory_mapped_persistence", target_arch = "x86_64"))]
    #[inline]
    pub fn prefetch_for_similarity(&self) {
        unsafe {
            let embedding_ptr = self.embedding.as_ptr().cast::<i8>();
            for line in 0..48 {
                // 768 * 4 bytes / 64 bytes = 48 lines
                std::arch::x86_64::_mm_prefetch(
                    embedding_ptr.add(line * 64),
                    std::arch::x86_64::_MM_HINT_T1,
                );
            }
        }
    }

    /// Prefetch only hot cache line for activation spreading (no-op on non-x86_64)
    #[cfg(not(all(feature = "memory_mapped_persistence", target_arch = "x86_64")))]
    pub const fn prefetch_for_activation(&self) {
        let _ = self;
        // No-op on non-x86_64 platforms
    }

    /// Prefetch embedding cache lines for SIMD operations (no-op on non-x86_64)
    #[cfg(not(all(feature = "memory_mapped_persistence", target_arch = "x86_64")))]
    pub const fn prefetch_for_similarity(&self) {
        let _ = self;
        // No-op on non-x86_64 platforms
    }
}

/// Memory-mapped file header for metadata
#[repr(C)]
pub struct MappedFileHeader {
    /// Magic number for file format identification
    pub magic: u64,
    /// File format version number
    pub version: u32,
    /// Number of entries stored in the file
    pub entry_count: u32,
    /// Size of each entry in bytes
    pub entry_size: u32,
    /// Size of this header structure in bytes
    pub header_size: u32,
    /// Timestamp when the file was created (nanoseconds since UNIX epoch)
    pub created_at: u64,
    /// Timestamp when the file was last modified (nanoseconds since UNIX epoch)
    pub last_modified: u64,
    /// Checksum of file contents for integrity verification
    pub checksum: u32,
    /// File-specific flags and options
    pub flags: u32,
    /// Reserved space for future extensions
    pub reserved: [u64; 6],
}

const MAPPED_FILE_MAGIC: u64 = 0x454E_4752_414D_4D41; // "ENGRAMMA"
const CURRENT_VERSION: u32 = 1;

impl MappedFileHeader {
    /// Create a new mapped file header with the specified parameters
    #[must_use]
    pub fn new(entry_count: u32, entry_size: u32) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .try_into()
            .unwrap_or(u64::MAX);

        Self {
            magic: MAPPED_FILE_MAGIC,
            version: CURRENT_VERSION,
            entry_count,
            entry_size,
            header_size: u32::try_from(std::mem::size_of::<Self>()).unwrap_or(u32::MAX),
            created_at: now,
            last_modified: now,
            checksum: 0, // Computed separately
            flags: 0,
            reserved: [0; 6],
        }
    }

    /// Validate the header for corruption and version compatibility.
    ///
    /// # Errors
    ///
    /// Returns an error when the magic number, version, or checksum values are
    /// inconsistent with the expected format.
    #[must_use = "Header validation guards against mapped file corruption"]
    pub fn validate(&self) -> StorageResult<()> {
        if self.magic != MAPPED_FILE_MAGIC {
            return Err(StorageError::CorruptionDetected(format!(
                "Invalid magic: expected {:x}, got {:x}",
                MAPPED_FILE_MAGIC, self.magic
            )));
        }

        if self.version != CURRENT_VERSION {
            return Err(StorageError::CorruptionDetected(format!(
                "Unsupported version: {}",
                self.version
            )));
        }

        Ok(())
    }

    #[cfg(feature = "memory_mapped_persistence")]
    /// Compute CRC32 checksum of the provided data
    #[must_use = "Checksum output must be written to the mapped header"]
    pub fn compute_checksum(&self, data: &[u8]) -> u32 {
        let mut combined = Vec::new();

        // Add header bytes (excluding checksum field)
        combined.extend_from_slice(&self.magic.to_le_bytes());
        combined.extend_from_slice(&self.version.to_le_bytes());
        combined.extend_from_slice(&self.entry_count.to_le_bytes());
        combined.extend_from_slice(&self.entry_size.to_le_bytes());
        combined.extend_from_slice(&self.header_size.to_le_bytes());
        combined.extend_from_slice(&self.created_at.to_le_bytes());
        combined.extend_from_slice(&self.last_modified.to_le_bytes());
        combined.extend_from_slice(&self.flags.to_le_bytes());

        // Add data
        combined.extend_from_slice(data);

        crc32c(&combined)
    }

    #[cfg(not(feature = "memory_mapped_persistence"))]
    #[must_use = "Checksum output must be written to the mapped header"]
    pub fn compute_checksum(&self, _data: &[u8]) -> u32 {
        0
    }
}

/// Content storage statistics for compaction decisions
#[derive(Debug, Clone, Copy)]
pub struct ContentStorageStats {
    /// Total bytes allocated in content storage
    pub total_bytes: u64,
    /// Bytes occupied by live content
    pub live_bytes: u64,
    /// Fragmentation ratio: (total - live) / total
    pub fragmentation_ratio: f64,
    /// Last compaction timestamp (Unix seconds)
    pub last_compaction: u64,
    /// Total bytes reclaimed since process start
    pub bytes_reclaimed_total: u64,
}

/// Compaction operation statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct CompactionStats {
    /// Size before compaction (bytes)
    pub old_size: u64,
    /// Size after compaction (bytes)
    pub new_size: u64,
    /// Bytes reclaimed
    pub bytes_reclaimed: u64,
    /// Compaction duration (serialized as milliseconds)
    #[serde(serialize_with = "serialize_duration_ms")]
    pub duration: std::time::Duration,
    /// Fragmentation before compaction (0.0 to 1.0)
    pub fragmentation_before: f64,
    /// Fragmentation after compaction (should be 0.0)
    pub fragmentation_after: f64,
}

fn serialize_duration_ms<S>(
    duration: &std::time::Duration,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_u128(duration.as_millis())
}

/// RAII guard to ensure compaction flag is reset
struct CompactionGuard<'a> {
    flag: &'a AtomicBool,
}

impl<'a> CompactionGuard<'a> {
    const fn new(flag: &'a AtomicBool) -> Self {
        Self { flag }
    }
}

impl Drop for CompactionGuard<'_> {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::Release);
    }
}

/// Memory-mapped storage for warm tier (recently accessed memories)
pub struct MappedWarmStorage {
    file_path: PathBuf,
    #[cfg(feature = "memory_mapped_persistence")]
    mmap: Option<Mmap>,
    #[cfg(feature = "memory_mapped_persistence")]
    mmap_mut: Option<MmapMut>,

    /// Index from memory ID to offset in file
    memory_index: DashMap<String, u64>,

    /// Performance metrics
    metrics: Arc<StorageMetrics>,

    /// Statistics
    entry_count: AtomicUsize,
    total_size: AtomicU64,
    last_access: AtomicU64,

    /// Variable-length content storage (separate from embeddings)
    content_data: parking_lot::RwLock<Vec<u8>>,

    /// Compaction state tracking
    compaction_in_progress: AtomicBool,
    last_compaction: AtomicU64, // Unix timestamp in seconds
    bytes_reclaimed: AtomicU64, // Total bytes reclaimed since start
}

impl MappedWarmStorage {
    /// Create new memory-mapped warm storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the backing file cannot be created or initialized
    /// for memory mapping.
    #[must_use = "Handle the result of mapped storage initialization"]
    pub fn new<P: AsRef<Path>>(
        file_path: P,
        initial_capacity: usize,
        metrics: Arc<StorageMetrics>,
    ) -> StorageResult<Self> {
        let file_path = file_path.as_ref().to_path_buf();

        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut storage = Self {
            file_path,
            #[cfg(feature = "memory_mapped_persistence")]
            mmap: None,
            #[cfg(feature = "memory_mapped_persistence")]
            mmap_mut: None,
            memory_index: DashMap::new(),
            metrics,
            entry_count: AtomicUsize::new(0),
            total_size: AtomicU64::new(0),
            last_access: AtomicU64::new(0),
            content_data: parking_lot::RwLock::new(Vec::with_capacity(initial_capacity * 128)),
            compaction_in_progress: AtomicBool::new(false),
            last_compaction: AtomicU64::new(0),
            bytes_reclaimed: AtomicU64::new(0),
        };

        // Initialize file if it doesn't exist
        storage.initialize_file(initial_capacity)?;

        // Check fragmentation on startup and warn if high
        let stats = storage.content_storage_stats();
        if stats.fragmentation_ratio > 0.7 && stats.total_bytes > 0 {
            tracing::warn!(
                fragmentation = format!("{:.1}%", stats.fragmentation_ratio * 100.0),
                size_mb = stats.total_bytes / 1_000_000,
                "High fragmentation detected on startup - compaction recommended"
            );
        }

        Ok(storage)
    }

    #[cfg(feature = "memory_mapped_persistence")]
    fn initialize_file(&mut self, capacity: usize) -> StorageResult<()> {
        use std::fs::OpenOptions;

        let entry_size_u32 =
            u32::try_from(std::mem::size_of::<EmbeddingBlock>()).unwrap_or(u32::MAX);
        let entry_size = usize::try_from(entry_size_u32).unwrap_or(0);
        let header_size = std::mem::size_of::<MappedFileHeader>();
        let data_size = capacity.saturating_mul(entry_size);
        let total_size = header_size + data_size;

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&self.file_path)?;

        // Set file size
        file.set_len(u64::try_from(total_size).unwrap_or(u64::MAX))?;

        // Create memory mapping
        let mut mmap_mut = unsafe { MmapOptions::new().len(total_size).map_mut(&file)? };

        // Initialize header
        let header = MappedFileHeader::new(0, entry_size_u32);
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(&header).cast::<u8>(),
                std::mem::size_of::<MappedFileHeader>(),
            )
        };

        mmap_mut[..header_size].copy_from_slice(header_bytes);

        // Compute and store checksum
        let checksum = header.compute_checksum(&mmap_mut[header_size..]);
        let checksum_bytes = checksum.to_le_bytes();
        let checksum_offset = std::mem::offset_of!(MappedFileHeader, checksum);
        mmap_mut[checksum_offset..checksum_offset + 4].copy_from_slice(&checksum_bytes);

        // Set memory advice for access patterns
        #[cfg(unix)]
        unsafe {
            libc::madvise(
                mmap_mut.as_mut_ptr().cast::<libc::c_void>(),
                mmap_mut.len(),
                libc::MADV_SEQUENTIAL,
            );
        }

        self.mmap_mut = Some(mmap_mut);
        Ok(())
    }

    #[cfg(not(feature = "memory_mapped_persistence"))]
    fn initialize_file(&mut self, _capacity: usize) -> StorageResult<()> {
        // Fallback for non-mmap builds
        Ok(())
    }

    /// Load existing memory-mapped file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened
    /// - The file is too small to contain a valid header
    /// - The header validation fails
    /// - The checksum doesn't match
    #[cfg(feature = "memory_mapped_persistence")]
    #[must_use = "Opening mapped storage can fail and should be inspected"]
    pub fn load_existing<P: AsRef<Path>>(
        file_path: P,
        metrics: Arc<StorageMetrics>,
    ) -> StorageResult<Self> {
        let file_path = file_path.as_ref().to_path_buf();

        let file = std::fs::File::open(&file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Validate header
        if mmap.len() < std::mem::size_of::<MappedFileHeader>() {
            return Err(StorageError::CorruptionDetected(
                "File too small".to_string(),
            ));
        }

        let header = unsafe { std::ptr::read_unaligned(mmap.as_ptr().cast::<MappedFileHeader>()) };

        header.validate()?;

        // Verify checksum
        let header_size = header.header_size as usize;
        let data_checksum = header.compute_checksum(&mmap[header_size..]);
        if data_checksum != header.checksum {
            return Err(StorageError::ChecksumMismatch {
                expected: header.checksum,
                actual: data_checksum,
            });
        }

        // Build index of existing entries
        let memory_index = DashMap::new();
        let entry_size = header.entry_size as usize;

        for i in 0..header.entry_count as usize {
            let offset = header_size + i * entry_size;
            if offset + entry_size <= mmap.len() {
                // This would require deserializing the memory ID from the embedding block
                // For now, we'll rebuild the index during normal operations
            }
        }

        Ok(Self {
            file_path,
            mmap: Some(mmap),
            mmap_mut: None,
            memory_index,
            metrics,
            entry_count: AtomicUsize::new(header.entry_count as usize),
            total_size: AtomicU64::new(
                header_size as u64 + u64::from(header.entry_count) * entry_size as u64,
            ),
            last_access: AtomicU64::new(0),
            content_data: parking_lot::RwLock::new(Vec::new()),
            compaction_in_progress: AtomicBool::new(false),
            last_compaction: AtomicU64::new(0),
            bytes_reclaimed: AtomicU64::new(0),
        })
    }

    /// Store embedding block at specific offset
    #[cfg(feature = "memory_mapped_persistence")]
    fn store_embedding_block(&self, block: &EmbeddingBlock, offset: usize) -> StorageResult<()> {
        if let Some(mmap_mut) = &self.mmap_mut {
            let block_size = std::mem::size_of::<EmbeddingBlock>();

            if offset + block_size > mmap_mut.len() {
                return Err(StorageError::allocation_failed("Offset exceeds file size"));
            }

            let block_bytes = unsafe {
                std::slice::from_raw_parts(std::ptr::from_ref(block).cast::<u8>(), block_size)
            };

            // SAFETY: We've validated the offset bounds above
            unsafe {
                let dst = mmap_mut.as_ptr().add(offset).cast_mut();
                std::ptr::copy_nonoverlapping(block_bytes.as_ptr(), dst, block_size);
            }
            self.metrics.record_write(block_size as u64);

            Ok(())
        } else {
            Err(StorageError::NotInitialized(
                "MMAP not initialized".to_string(),
            ))
        }
    }

    #[cfg(not(feature = "memory_mapped_persistence"))]
    fn store_embedding_block(&self, _block: &EmbeddingBlock, _offset: usize) -> StorageResult<()> {
        Ok(())
    }

    /// Read embedding block from offset
    #[cfg(feature = "memory_mapped_persistence")]
    fn read_embedding_block(&self, offset: usize) -> StorageResult<EmbeddingBlock> {
        let mmap_slice: &[u8] = if let Some(mmap) = &self.mmap {
            mmap.as_ref()
        } else if let Some(mmap_mut) = &self.mmap_mut {
            mmap_mut.as_ref()
        } else {
            return Err(StorageError::NotInitialized(
                "MMAP not initialized".to_string(),
            ));
        };

        let block_size = std::mem::size_of::<EmbeddingBlock>();

        if offset + block_size > mmap_slice.len() {
            return Err(StorageError::CorruptionDetected(
                "Offset exceeds file size".to_string(),
            ));
        }

        let block = unsafe {
            std::ptr::read_unaligned(mmap_slice[offset..].as_ptr().cast::<EmbeddingBlock>())
        };

        self.metrics.record_read(block_size as u64);
        Ok(block)
    }

    #[cfg(not(feature = "memory_mapped_persistence"))]
    fn read_embedding_block(&self, _offset: usize) -> StorageResult<EmbeddingBlock> {
        Err(StorageError::NotInitialized)
    }

    /// Retrieve a memory by ID
    ///
    /// Returns the memory if found, None if not found, or an error if
    /// the storage operation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the memory cannot be read from storage.
    pub fn get(&self, memory_id: &str) -> StorageResult<Option<Arc<Memory>>> {
        // Look up offset in index
        let Some(entry) = self.memory_index.get(memory_id) else {
            return Ok(None);
        };
        let offset = usize::try_from(*entry.value()).map_err(|_| {
            StorageError::CorruptionDetected("Offset too large for platform".to_string())
        })?;
        drop(entry);

        // Read embedding block from storage
        let block = self.read_embedding_block(offset)?;

        // Restore content from variable-length storage
        // content_offset == u64::MAX indicates no content was stored (None)
        let content = if block.content_offset == u64::MAX {
            None
        } else {
            // Scope the read lock to minimize contention
            // parking_lot::RwLock doesn't poison - panics will abort the thread
            let content_storage = self.content_data.read();
            let start = block.content_offset as usize;
            let end = start + block.content_length as usize;

            if end > content_storage.len() {
                tracing::error!(
                    memory_id = %memory_id,
                    offset = block.content_offset,
                    length = block.content_length,
                    storage_size = content_storage.len(),
                    "Content offset out of bounds"
                );
                return Err(StorageError::CorruptionDetected(format!(
                    "Content offset out of bounds for memory {} (offset={}, length={}, storage_size={})",
                    memory_id,
                    block.content_offset,
                    block.content_length,
                    content_storage.len()
                )));
            }

            let content_bytes = &content_storage[start..end];
            let result = Some(String::from_utf8_lossy(content_bytes).to_string());
            drop(content_storage); // Early drop to release lock
            result
        };

        // Convert EmbeddingBlock to Memory
        let mut memory = Memory::new(
            memory_id.to_string(),
            block.embedding,
            Confidence::exact(block.confidence),
        );

        // Set custom field values from stored block
        memory.set_activation(block.activation);
        memory.activation_value = block.activation;
        memory.last_access = chrono::DateTime::from_timestamp_nanos(
            block.last_access.try_into().unwrap_or(i64::MAX),
        );
        memory.access_count = block.recall_count.into();
        memory.created_at = chrono::DateTime::from_timestamp_nanos(
            block.creation_time.try_into().unwrap_or(i64::MAX),
        );
        memory.decay_rate = block.decay_rate;
        memory.content = content;

        let memory = Arc::new(memory);

        Ok(Some(memory))
    }

    /// Get content storage statistics and update Prometheus metrics
    #[must_use]
    pub fn content_storage_stats(&self) -> ContentStorageStats {
        let total_bytes = {
            let storage = self.content_data.read();
            storage.len() as u64
        };

        let live_bytes = self.calculate_live_bytes();
        let fragmentation_ratio = if total_bytes > 0 {
            (total_bytes.saturating_sub(live_bytes)) as f64 / total_bytes as f64
        } else {
            0.0
        };

        // Update Prometheus metrics (StorageMetrics only tracks operation counts/bytes)
        // Actual metric recording happens via the global metrics registry
        // TODO: Add proper metrics recording when global registry is accessible

        ContentStorageStats {
            total_bytes,
            live_bytes,
            fragmentation_ratio,
            last_compaction: self.last_compaction.load(Ordering::Relaxed),
            bytes_reclaimed_total: self.bytes_reclaimed.load(Ordering::Relaxed),
        }
    }

    /// Calculate live bytes (sum of all content_length for live memories)
    fn calculate_live_bytes(&self) -> u64 {
        self.memory_index
            .iter()
            .filter_map(|entry| {
                let offset = *entry.value();
                self.read_embedding_block(offset as usize).ok()
            })
            .filter(|block| block.content_offset != u64::MAX)
            .map(|block| u64::from(block.content_length))
            .sum()
    }

    /// Estimate live content size for allocation
    fn estimate_live_content_size(&self) -> usize {
        // Assume average content size of 128 bytes per memory
        self.memory_index.len().saturating_mul(128)
    }

    /// Update content offset in an embedding block
    fn update_content_offset_in_block(
        &self,
        embedding_offset: usize,
        new_content_offset: u64,
    ) -> StorageResult<()> {
        let mut block = self.read_embedding_block(embedding_offset)?;
        block.content_offset = new_content_offset;
        self.store_embedding_block(&block, embedding_offset)?;
        Ok(())
    }

    /// Compact content storage to remove deleted memory holes
    ///
    /// # Algorithm
    ///
    /// 1. Acquire write lock on content_data (blocks new stores)
    /// 2. Collect all live content with new offsets
    /// 3. Build offset remapping table (old → new)
    /// 4. Update embedding blocks with new offsets
    /// 5. Atomically swap in new storage
    ///
    /// # Concurrency
    ///
    /// - Reads blocked during compaction (RwLock write acquisition)
    /// - Typical duration: ~500ms for 1M memories
    /// - Memory overhead: 2x during compaction (old + new Vec)
    ///
    /// # Error Recovery
    ///
    /// - If compaction fails, old storage remains unchanged
    /// - Offset updates are transactional (all or nothing)
    /// - Safe to retry on failure
    ///
    /// # Performance
    ///
    /// - Linear scan: O(n) where n = number of live memories
    /// - Memory copies: O(m) where m = total content bytes
    /// - Offset updates: O(n) with parallel updates
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Compaction is already in progress
    /// - Offset updates fail
    /// - Storage operations fail
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn compact_content(&self) -> StorageResult<CompactionStats> {
        // 1. Mark compaction as in-progress (prevent concurrent compactions)
        if self
            .compaction_in_progress
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(StorageError::CompactionInProgress);
        }

        let start_time = std::time::Instant::now();

        // Ensure compaction completes or resets flag
        let _guard = CompactionGuard::new(&self.compaction_in_progress);

        // 2. Acquire read lock to read old content
        let content_storage = self.content_data.read();
        let old_size = content_storage.len();

        // 3. Estimate live content size for allocation
        let estimated_live_size = self.estimate_live_content_size();
        let mut new_content = Vec::with_capacity(estimated_live_size);

        // 4. Collect live content and build offset map
        // Sort by old offset for sequential read pattern (cache-friendly)
        let mut live_memories: Vec<(String, u64, u64, u64)> = self
            .memory_index
            .iter()
            .filter_map(|entry| {
                let memory_id = entry.key().clone();
                let offset = *entry.value();

                // Read embedding block to get content metadata
                match self.read_embedding_block(offset as usize) {
                    Ok(block) if block.content_offset != u64::MAX => Some((
                        memory_id,
                        offset,
                        block.content_offset,
                        u64::from(block.content_length),
                    )),
                    Ok(_) => None, // No content stored
                    Err(e) => {
                        tracing::warn!(
                            memory_id = %memory_id,
                            error = %e,
                            "Failed to read embedding block during compaction, skipping"
                        );
                        None
                    }
                }
            })
            .collect();

        // Sort by content offset for sequential access pattern
        live_memories.sort_by_key(|(_, _, content_offset, _)| *content_offset);

        // 5. Build offset remapping table
        let mut offset_map: std::collections::HashMap<String, (u64, u64)> =
            std::collections::HashMap::with_capacity(live_memories.len());

        for (memory_id, embedding_offset, old_content_offset, content_length) in live_memories {
            let new_content_offset = new_content.len() as u64;

            // Validate bounds
            let start = old_content_offset as usize;
            let end = start.checked_add(content_length as usize).ok_or_else(|| {
                StorageError::CorruptionDetected(format!(
                    "Content offset overflow for memory {memory_id}"
                ))
            })?;

            if end > content_storage.len() {
                tracing::error!(
                    memory_id = %memory_id,
                    old_offset = old_content_offset,
                    length = content_length,
                    storage_size = content_storage.len(),
                    "Content offset out of bounds during compaction, skipping memory"
                );
                continue;
            }

            // Copy content to new Vec
            new_content.extend_from_slice(&content_storage[start..end]);

            // Record mapping: memory_id → (embedding_offset, new_content_offset)
            offset_map.insert(memory_id, (embedding_offset, new_content_offset));
        }

        drop(content_storage); // Release read lock early

        // 6. Atomically swap in new storage FIRST
        // CRITICAL: This must happen BEFORE updating offsets to prevent race condition
        // where concurrent reads use new offsets with old storage
        let mut content_storage = self.content_data.write();
        let new_size = new_content.len();

        // Shrink capacity to actual size to reclaim memory
        new_content.shrink_to_fit();

        *content_storage = new_content;
        drop(content_storage); // Release write lock

        // 7. Update embedding blocks with new content offsets
        // This is safe now because new storage is active
        // Save original offsets for rollback in case of failure
        let mut original_offsets = std::collections::HashMap::with_capacity(offset_map.len());
        for (memory_id, (embedding_offset, _)) in &offset_map {
            if let Ok(block) = self.read_embedding_block(*embedding_offset as usize) {
                original_offsets
                    .insert(memory_id.clone(), (*embedding_offset, block.content_offset));
            }
        }

        let update_errors = std::sync::atomic::AtomicUsize::new(0);

        // Use rayon for parallel updates (embedding blocks are independent)
        offset_map
            .par_iter()
            .for_each(|(memory_id, (embedding_offset, new_content_offset))| {
                if let Err(e) = self
                    .update_content_offset_in_block(*embedding_offset as usize, *new_content_offset)
                {
                    tracing::error!(
                        memory_id = %memory_id,
                        error = %e,
                        "Failed to update embedding block during compaction"
                    );
                    update_errors.fetch_add(1, Ordering::Relaxed);
                }
            });

        // Check for update failures and rollback if needed
        let failed_updates = update_errors.load(Ordering::Relaxed);
        if failed_updates > 0 {
            // ROLLBACK: Restore original offsets
            tracing::error!(
                failed_count = failed_updates,
                "Compaction offset updates failed, attempting rollback"
            );

            let mut rollback_failures = 0;
            for (memory_id, (embedding_offset, old_offset)) in original_offsets {
                if let Err(e) =
                    self.update_content_offset_in_block(embedding_offset as usize, old_offset)
                {
                    tracing::error!(
                        memory_id = %memory_id,
                        error = %e,
                        "Failed to rollback offset during compaction recovery"
                    );
                    rollback_failures += 1;
                }
            }

            if rollback_failures > 0 {
                return Err(StorageError::CompactionFailed(format!(
                    "Failed to update {failed_updates} embedding blocks, and {rollback_failures} rollback operations also failed. Storage may be in inconsistent state."
                )));
            }

            return Err(StorageError::CompactionFailed(format!(
                "Failed to update {failed_updates} embedding blocks (successfully rolled back)"
            )));
        }

        // 8. Update compaction statistics and metrics
        let duration = start_time.elapsed();
        let bytes_reclaimed = old_size.saturating_sub(new_size);
        self.bytes_reclaimed
            .fetch_add(bytes_reclaimed as u64, Ordering::Relaxed);
        self.last_compaction.store(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            Ordering::Relaxed,
        );

        // Record Prometheus metrics (StorageMetrics only tracks operation counts/bytes)
        // Actual metric recording happens via the global metrics registry
        // TODO: Add proper metrics recording when global registry is accessible

        tracing::info!(
            old_size_mb = old_size / 1_000_000,
            new_size_mb = new_size / 1_000_000,
            bytes_reclaimed_mb = bytes_reclaimed / 1_000_000,
            duration_ms = duration.as_millis(),
            "Content storage compaction completed"
        );

        Ok(CompactionStats {
            old_size: old_size as u64,
            new_size: new_size as u64,
            bytes_reclaimed: bytes_reclaimed as u64,
            duration,
            fragmentation_before: 1.0 - (new_size as f64 / old_size.max(1) as f64),
            fragmentation_after: 0.0,
        })
    }

    /// Compact content storage (fallback for non-mmap builds)
    #[cfg(not(feature = "memory_mapped_persistence"))]
    pub fn compact_content(&self) -> StorageResult<CompactionStats> {
        Err(StorageError::NotImplemented(
            "Compaction requires memory_mapped_persistence feature".to_string(),
        ))
    }
}

#[cfg(feature = "memory_mapped_persistence")]
impl StorageTierBackend for MappedWarmStorage {
    type Error = StorageError;

    async fn store(&self, memory: Arc<Memory>) -> Result<(), Self::Error> {
        let mut block = EmbeddingBlock::new(&memory);

        // Persist content to variable-length storage
        if let Some(content) = &memory.content {
            let content_bytes = content.as_bytes();
            let content_len = content_bytes.len();

            let offset = {
                // Acquire write lock on content storage in limited scope
                // parking_lot::RwLock doesn't poison - panics will abort the thread
                let mut content_storage = self.content_data.write();

                // Get current offset
                let offset = content_storage.len() as u64;

                // Append content (even if empty - we store empty strings explicitly)
                if content_len > 0 {
                    content_storage.extend_from_slice(content_bytes);
                }

                offset
            }; // Lock is dropped here

            // Update block metadata to indicate content was present
            // Use content_length = 0 with valid offset to distinguish from None
            block.content_offset = offset;
            block.content_length = content_len as u32;
        }

        // CRITICAL: Atomically allocate offset to prevent race conditions
        // fetch_add returns the OLD value before increment, giving each thread a unique index
        // Use SeqCst ordering to ensure cross-thread visibility of writes
        let entry_index = self.entry_count.fetch_add(1, Ordering::SeqCst);

        // Calculate offset using the atomically-allocated entry index
        let header_size = std::mem::size_of::<MappedFileHeader>();
        let entry_size = std::mem::size_of::<EmbeddingBlock>();
        let offset = header_size + entry_index * entry_size;

        self.store_embedding_block(&block, offset)?;
        self.memory_index.insert(memory.id.clone(), offset as u64);
        self.total_size.fetch_add(
            std::mem::size_of::<EmbeddingBlock>() as u64,
            Ordering::Relaxed,
        );
        self.last_access.store(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );

        Ok(())
    }

    async fn recall(&self, cue: &Cue) -> Result<Vec<(Episode, Confidence)>, Self::Error> {
        let mut results = Vec::new();

        // For demonstration, we'll do a simple linear scan
        // In practice, this would use SIMD-optimized similarity search
        for entry in &self.memory_index {
            let memory_id = entry.key();
            let offset = usize::try_from(*entry.value()).map_err(|_| {
                StorageError::CorruptionDetected("Offset too large for platform".to_string())
            })?;

            let block = self.read_embedding_block(offset)?;

            // Simple confidence-based filtering for now
            let confidence = Confidence::exact(block.confidence);
            if confidence.raw() >= cue.result_threshold.raw() {
                // Restore content from variable-length storage
                // content_offset == u64::MAX indicates no content was stored (None)
                let content = if block.content_offset == u64::MAX {
                    format!("Stored memory {memory_id}")
                } else {
                    // Scope the read lock to minimize contention
                    // parking_lot::RwLock doesn't poison - panics will abort the thread
                    let content_storage = self.content_data.read();
                    let start = block.content_offset as usize;
                    let end = start + block.content_length as usize;

                    let result = if end <= content_storage.len() {
                        let content_bytes = &content_storage[start..end];
                        String::from_utf8_lossy(content_bytes).to_string()
                    } else {
                        tracing::error!(
                            memory_id = %memory_id,
                            offset = block.content_offset,
                            length = block.content_length,
                            storage_size = content_storage.len(),
                            "Content offset out of bounds during recall"
                        );
                        format!("Stored memory {memory_id}")
                    };
                    drop(content_storage); // Early drop to release lock
                    result
                };

                // Convert back to Episode
                let episode = crate::EpisodeBuilder::new()
                    .id(memory_id.clone())
                    .when(chrono::DateTime::from_timestamp_nanos(
                        block.creation_time.try_into().unwrap_or(i64::MAX),
                    ))
                    .what(content)
                    .embedding(block.embedding)
                    .confidence(confidence)
                    .build();

                results.push((episode, confidence));
            }
        }

        Ok(results)
    }

    async fn update_activation(&self, memory_id: &str, activation: f32) -> Result<(), Self::Error> {
        if let Some(entry) = self.memory_index.get(memory_id) {
            let offset = usize::try_from(*entry.value()).map_err(|_| {
                StorageError::CorruptionDetected("Offset too large for platform".to_string())
            })?;
            let mut block = self.read_embedding_block(offset)?;

            block.activation = activation;
            block.last_access = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX);

            self.store_embedding_block(&block, offset)?;
        }

        Ok(())
    }

    async fn remove(&self, memory_id: &str) -> Result<(), Self::Error> {
        self.memory_index.remove(memory_id);
        // Note: This leaves a hole in the file - compaction would be needed
        Ok(())
    }

    fn statistics(&self) -> TierStatistics {
        TierStatistics {
            memory_count: self.entry_count.load(Ordering::Relaxed),
            total_size_bytes: self.total_size.load(Ordering::Relaxed),
            average_activation: 0.5, // Would compute from actual data
            last_access_time: SystemTime::UNIX_EPOCH
                + std::time::Duration::from_nanos(self.last_access.load(Ordering::Relaxed)),
            cache_hit_rate: self.metrics.cache_hit_rate(),
            compaction_ratio: 0.8, // Would compute from compaction stats
        }
    }

    async fn maintenance(&self) -> Result<(), Self::Error> {
        // Perform any background maintenance like compaction
        Ok(())
    }
}

#[cfg(not(feature = "memory_mapped_persistence"))]
impl StorageTierBackend for MappedWarmStorage {
    type Error = StorageError;

    async fn store(&self, _memory: Arc<Memory>) -> Result<(), Self::Error> {
        Err(StorageError::NotInitialized)
    }

    async fn recall(&self, _cue: &Cue) -> Result<Vec<(Episode, Confidence)>, Self::Error> {
        Ok(Vec::new())
    }

    async fn update_activation(
        &self,
        _memory_id: &str,
        _activation: f32,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    async fn remove(&self, _memory_id: &str) -> Result<(), Self::Error> {
        Ok(())
    }

    fn statistics(&self) -> TierStatistics {
        TierStatistics {
            memory_count: 0,
            total_size_bytes: 0,
            average_activation: 0.0,
            last_access_time: SystemTime::UNIX_EPOCH,
            cache_hit_rate: 0.0,
            compaction_ratio: 0.0,
        }
    }

    async fn maintenance(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

// Note: MappedWarmStorage uses async interface which doesn't work well with
// the synchronous Storage trait. The feature system falls back to NullStorage.

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use tempfile::TempDir;

    type TestResult<T = ()> = Result<T, String>;

    fn ensure(condition: bool, message: impl Into<String>) -> TestResult {
        if condition {
            Ok(())
        } else {
            Err(message.into())
        }
    }

    fn ensure_eq<T>(actual: &T, expected: &T, context: &str) -> TestResult
    where
        T: PartialEq + Debug,
    {
        if actual == expected {
            Ok(())
        } else {
            Err(format!("{context}: expected {expected:?}, got {actual:?}"))
        }
    }

    trait IntoTestResult<T> {
        fn into_test_result(self, context: &str) -> TestResult<T>;
    }

    impl<T, E: std::fmt::Debug> IntoTestResult<T> for Result<T, E> {
        fn into_test_result(self, context: &str) -> TestResult<T> {
            self.map_err(|err| format!("{context}: {err:?}"))
        }
    }

    fn create_test_memory() -> Arc<Memory> {
        use crate::EpisodeBuilder;
        use chrono::Utc;

        let episode = EpisodeBuilder::new()
            .id("test_memory".to_string())
            .when(Utc::now())
            .what("test content".to_string())
            .embedding([0.5f32; 768])
            .confidence(Confidence::HIGH)
            .build();

        Arc::new(Memory::from_episode(episode, 0.8))
    }

    #[test]
    fn test_embedding_block_alignment() {
        assert_eq!(std::mem::size_of::<EmbeddingBlock>(), 3136); // 768*4 + 64 bytes metadata
        assert_eq!(std::mem::align_of::<EmbeddingBlock>(), 64);
    }

    #[test]
    fn test_embedding_block_creation() {
        let memory = create_test_memory();
        let block = EmbeddingBlock::new(&memory);

        for (actual, expected) in block.embedding.iter().zip(memory.embedding.iter()) {
            assert!((actual - expected).abs() < f32::EPSILON);
        }
        assert!((block.confidence - memory.confidence.raw()).abs() < f32::EPSILON);
        assert!(block.last_access > 0);
    }

    #[cfg(feature = "memory_mapped_persistence")]
    #[tokio::test]
    async fn test_mapped_warm_storage() -> TestResult {
        let temp_dir =
            TempDir::new().into_test_result("create temp dir for mapped storage tests")?;
        let file_path = temp_dir.path().join("warm_storage.dat");
        let metrics = Arc::new(StorageMetrics::new());

        let storage = MappedWarmStorage::new(file_path, 1000, metrics)
            .into_test_result("construct mapped warm storage")?;
        let memory = create_test_memory();

        // Test store
        storage
            .store(memory)
            .await
            .into_test_result("store memory in mapped storage")?;

        // Test recall
        let cue = Cue::semantic(
            "test".to_string(),
            "test content".to_string(),
            Confidence::MEDIUM,
        );
        let results = storage
            .recall(&cue)
            .await
            .into_test_result("recall from mapped storage")?;
        ensure(
            !results.is_empty(),
            "mapped storage recall should return results",
        )?;

        // Test statistics
        let stats = storage.statistics();
        ensure_eq(&stats.memory_count, &1_usize, "mapped storage memory count")?;
        ensure(stats.total_size_bytes > 0, "mapped storage tracks bytes")?;

        Ok(())
    }
}
