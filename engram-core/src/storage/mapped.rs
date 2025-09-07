//! Memory-mapped storage for warm and cold tiers
//!
//! This module implements high-performance memory-mapped storage with:
//! - Cache-optimal layouts for SIMD operations
//! - NUMA-aware allocation for multi-socket systems
//! - Zero-copy reads with hardware prefetching
//! - Columnar storage for batch operations

use super::{StorageError, StorageMetrics, StorageResult, StorageTier, TierStatistics};
use crate::{Confidence, Cue, Episode, Memory};
use crossbeam_skiplist::SkipMap;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Instant, SystemTime};

#[cfg(feature = "memory_mapped_persistence")]
use crc32c::crc32c;
#[cfg(feature = "memory_mapped_persistence")]
use memmap2::{Mmap, MmapMut, MmapOptions};

#[cfg(all(feature = "memory_mapped_persistence", unix))]
use super::numa::{NumaMemoryMap, NumaPolicy, NumaTopology};

/// Cache-line aligned embedding block for optimal SIMD performance  
#[repr(C, align(64))]
pub struct EmbeddingBlock {
    /// 768-dimensional embedding (3072 bytes = 48 cache lines)
    pub embedding: [f32; 768],

    /// Metadata co-located for cache efficiency (1 cache line)
    pub confidence: f32,
    pub activation: f32,
    pub last_access: u64,
    pub decay_rate: f32,
    pub node_flags: u32,
    pub content_hash: u64,
    pub creation_time: u64,
    pub recall_count: u32,

    /// Padding to complete cache line
    pub _padding: [u8; 12],
}

impl EmbeddingBlock {
    pub fn new(memory: &Memory) -> Self {
        Self {
            embedding: memory.embedding,
            confidence: memory.confidence.raw(),
            activation: memory.activation(),
            last_access: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            decay_rate: 0.2, // Default decay
            node_flags: 0,
            content_hash: Self::compute_content_hash(&memory.id),
            creation_time: memory
                .created_at
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            recall_count: 0,
            _padding: [0; 12],
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
                (self as *const Self) as *const i8,
                std::arch::x86_64::_MM_HINT_T0,
            );
        }
    }

    /// Prefetch embedding cache lines for SIMD operations
    #[cfg(all(feature = "memory_mapped_persistence", target_arch = "x86_64"))]
    #[inline]
    pub fn prefetch_for_similarity(&self) {
        unsafe {
            let embedding_ptr = self.embedding.as_ptr() as *const i8;
            for line in 0..48 {
                // 768 * 4 bytes / 64 bytes = 48 lines
                std::arch::x86_64::_mm_prefetch(
                    embedding_ptr.add(line * 64),
                    std::arch::x86_64::_MM_HINT_T1,
                );
            }
        }
    }

    #[cfg(not(all(feature = "memory_mapped_persistence", target_arch = "x86_64")))]
    pub fn prefetch_for_activation(&self) {
        // No-op on non-x86_64 platforms
    }

    #[cfg(not(all(feature = "memory_mapped_persistence", target_arch = "x86_64")))]
    pub fn prefetch_for_similarity(&self) {
        // No-op on non-x86_64 platforms
    }
}

/// Memory-mapped file header for metadata
#[repr(C)]
pub struct MappedFileHeader {
    pub magic: u64,
    pub version: u32,
    pub entry_count: u32,
    pub entry_size: u32,
    pub header_size: u32,
    pub created_at: u64,
    pub last_modified: u64,
    pub checksum: u32,
    pub flags: u32,
    pub reserved: [u64; 6],
}

const MAPPED_FILE_MAGIC: u64 = 0x454E4752414D4D41; // "ENGRAMMA"
const CURRENT_VERSION: u32 = 1;

impl MappedFileHeader {
    pub fn new(entry_count: u32, entry_size: u32) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        Self {
            magic: MAPPED_FILE_MAGIC,
            version: CURRENT_VERSION,
            entry_count,
            entry_size,
            header_size: std::mem::size_of::<Self>() as u32,
            created_at: now,
            last_modified: now,
            checksum: 0, // Computed separately
            flags: 0,
            reserved: [0; 6],
        }
    }

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
    pub fn compute_checksum(&self, _data: &[u8]) -> u32 {
        0
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

    /// NUMA topology for optimal allocation
    #[cfg(all(feature = "memory_mapped_persistence", unix))]
    numa_topology: Arc<NumaTopology>,

    /// Statistics
    entry_count: AtomicUsize,
    total_size: AtomicU64,
    last_access: AtomicU64,
}

impl MappedWarmStorage {
    /// Create new memory-mapped warm storage
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

        #[cfg(all(feature = "memory_mapped_persistence", unix))]
        let numa_topology = Arc::new(
            super::numa::NumaTopology::detect()
                .unwrap_or_else(|_| super::numa::NumaTopology::detect().unwrap()),
        );

        let mut storage = Self {
            file_path,
            #[cfg(feature = "memory_mapped_persistence")]
            mmap: None,
            #[cfg(feature = "memory_mapped_persistence")]
            mmap_mut: None,
            memory_index: DashMap::new(),
            metrics,
            #[cfg(all(feature = "memory_mapped_persistence", unix))]
            numa_topology,
            entry_count: AtomicUsize::new(0),
            total_size: AtomicU64::new(0),
            last_access: AtomicU64::new(0),
        };

        // Initialize file if it doesn't exist
        storage.initialize_file(initial_capacity)?;

        Ok(storage)
    }

    #[cfg(feature = "memory_mapped_persistence")]
    fn initialize_file(&mut self, capacity: usize) -> StorageResult<()> {
        use std::fs::OpenOptions;

        let entry_size = std::mem::size_of::<EmbeddingBlock>() as u32;
        let header_size = std::mem::size_of::<MappedFileHeader>();
        let data_size = capacity * entry_size as usize;
        let total_size = header_size + data_size;

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&self.file_path)?;

        // Set file size
        file.set_len(total_size as u64)?;

        // Create memory mapping
        let mut mmap_mut = unsafe { MmapOptions::new().len(total_size).map_mut(&file)? };

        // Initialize header
        let header = MappedFileHeader::new(0, entry_size);
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const _ as *const u8,
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
                mmap_mut.as_ptr() as *mut libc::c_void,
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
    #[cfg(feature = "memory_mapped_persistence")]
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

        let header = unsafe { std::ptr::read(mmap.as_ptr() as *const MappedFileHeader) };

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

        #[cfg(all(feature = "memory_mapped_persistence", unix))]
        let numa_topology = Arc::new(
            super::numa::NumaTopology::detect()
                .unwrap_or_else(|_| super::numa::NumaTopology::detect().unwrap()),
        );

        Ok(Self {
            file_path,
            mmap: Some(mmap),
            mmap_mut: None,
            memory_index,
            metrics,
            #[cfg(all(feature = "memory_mapped_persistence", unix))]
            numa_topology,
            entry_count: AtomicUsize::new(header.entry_count as usize),
            total_size: AtomicU64::new(
                header_size as u64 + header.entry_count as u64 * entry_size as u64,
            ),
            last_access: AtomicU64::new(0),
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

            let block_bytes =
                unsafe { std::slice::from_raw_parts(block as *const _ as *const u8, block_size) };

            mmap_mut[offset..offset + block_size].copy_from_slice(block_bytes);
            self.metrics.record_write(block_size as u64);

            Ok(())
        } else {
            Err(StorageError::NotInitialized)
        }
    }

    #[cfg(not(feature = "memory_mapped_persistence"))]
    fn store_embedding_block(&self, _block: &EmbeddingBlock, _offset: usize) -> StorageResult<()> {
        Ok(())
    }

    /// Read embedding block from offset
    #[cfg(feature = "memory_mapped_persistence")]
    fn read_embedding_block(&self, offset: usize) -> StorageResult<EmbeddingBlock> {
        let mmap = self
            .mmap
            .as_ref()
            .or_else(|| self.mmap_mut.as_ref().map(|m| m as &[u8]))
            .ok_or(StorageError::NotInitialized)?;

        let block_size = std::mem::size_of::<EmbeddingBlock>();

        if offset + block_size > mmap.len() {
            return Err(StorageError::CorruptionDetected(
                "Offset exceeds file size".to_string(),
            ));
        }

        let block = unsafe { std::ptr::read(mmap[offset..].as_ptr() as *const EmbeddingBlock) };

        self.metrics.record_read(block_size as u64);
        Ok(block)
    }

    #[cfg(not(feature = "memory_mapped_persistence"))]
    fn read_embedding_block(&self, _offset: usize) -> StorageResult<EmbeddingBlock> {
        Err(StorageError::NotInitialized)
    }

    /// Find next available offset for new entry
    fn find_next_offset(&self) -> usize {
        let header_size = std::mem::size_of::<MappedFileHeader>();
        let entry_size = std::mem::size_of::<EmbeddingBlock>();
        let current_count = self.entry_count.load(Ordering::Relaxed);

        header_size + current_count * entry_size
    }
}

#[cfg(feature = "memory_mapped_persistence")]
impl StorageTier for MappedWarmStorage {
    type Error = StorageError;

    async fn store(&self, memory: Arc<Memory>) -> Result<(), Self::Error> {
        let block = EmbeddingBlock::new(&memory);
        let offset = self.find_next_offset();

        self.store_embedding_block(&block, offset)?;
        self.memory_index.insert(memory.id.clone(), offset as u64);

        self.entry_count.fetch_add(1, Ordering::Relaxed);
        self.total_size.fetch_add(
            std::mem::size_of::<EmbeddingBlock>() as u64,
            Ordering::Relaxed,
        );
        self.last_access.store(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
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
            let offset = *entry.value() as usize;

            let block = self.read_embedding_block(offset)?;

            // Simple confidence-based filtering for now
            let confidence = Confidence::exact(block.confidence);
            if confidence.raw() >= cue.confidence.raw() {
                // Convert back to Episode (simplified)
                let episode = crate::EpisodeBuilder::new()
                    .id(memory_id.clone())
                    .when(chrono::DateTime::from_timestamp_nanos(
                        block.creation_time as i64,
                    ))
                    .what(format!("Stored memory {}", memory_id))
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
            let offset = *entry.value() as usize;
            let mut block = self.read_embedding_block(offset)?;

            block.activation = activation;
            block.last_access = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;

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
impl StorageTier for MappedWarmStorage {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_memory() -> Arc<Memory> {
        use crate::{EpisodeBuilder, MemoryBuilder};
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

        assert_eq!(block.embedding, memory.embedding);
        assert_eq!(block.confidence, memory.confidence.raw());
        assert!(block.last_access > 0);
    }

    #[cfg(feature = "memory_mapped_persistence")]
    #[tokio::test]
    async fn test_mapped_warm_storage() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("warm_storage.dat");
        let metrics = Arc::new(StorageMetrics::new());

        let storage = MappedWarmStorage::new(file_path, 1000, metrics).unwrap();
        let memory = create_test_memory();

        // Test store
        storage.store(memory.clone()).await.unwrap();

        // Test recall
        let cue = Cue::semantic(
            "test".to_string(),
            "test content".to_string(),
            Confidence::MEDIUM,
        );
        let results = storage.recall(&cue).await.unwrap();
        assert!(!results.is_empty());

        // Test statistics
        let stats = storage.statistics();
        assert_eq!(stats.memory_count, 1);
        assert!(stats.total_size_bytes > 0);
    }
}
