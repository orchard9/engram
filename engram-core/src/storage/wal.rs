//! Crash-consistent write-ahead log for durability guarantees
//!
//! This module implements a high-performance WAL with:
//! - Sub-10ms P99 write latency including fsync
//! - CRC32C hardware-accelerated checksums
//! - Batch group commits for throughput optimization  
//! - Lock-free writer with bounded ring buffer
//! - Automatic recovery with corruption detection

// Allow unsafe code for performance-critical WAL operations
#![allow(unsafe_code)]

use super::{FsyncMode, StorageError, StorageMetrics, StorageResult};
use crate::{Episode, Memory};
use crossbeam_queue::SegQueue;
use std::convert::TryFrom;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Instant, SystemTime};

#[cfg(feature = "memory_mapped_persistence")]
use crc32c::crc32c;

/// Write-ahead log entry header (exactly 64 bytes = 1 cache line)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WalEntryHeader {
    /// Magic number for corruption detection
    pub magic: u32,
    /// Monotonic sequence number
    pub sequence: u64,
    /// Wall clock timestamp as nanoseconds since epoch
    pub timestamp: u64,
    /// Entry type discriminant
    pub entry_type: u32,
    /// Payload size in bytes
    pub payload_size: u32,
    /// CRC32C checksum of payload data
    pub payload_crc: u32,
    /// CRC32C checksum of header fields
    pub header_crc: u32,
    /// Reserved for future extensions - pad to exactly 64 bytes
    pub reserved: [u8; 20], // Account for alignment padding
}

const WAL_MAGIC: u32 = 0xDEAD_BEEF;
const HEADER_SIZE: usize = std::mem::size_of::<WalEntryHeader>();

impl WalEntryHeader {
    /// Create a new header with computed checksums
    ///
    /// # Errors
    ///
    /// Returns an error when the payload exceeds the addressable size for the
    /// header format.
    #[must_use = "Inspect WAL header creation failures to avoid accepting invalid entries"]
    pub fn new(entry_type: WalEntryType, payload: &[u8], sequence: u64) -> StorageResult<Self> {
        let raw_timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let capped_timestamp = raw_timestamp.min(u128::from(u64::MAX));
        let timestamp = u64::try_from(capped_timestamp).unwrap_or(u64::MAX);

        let payload_size = u32::try_from(payload.len()).map_err(|_| {
            StorageError::Configuration("WAL payload exceeds 4GiB limit".to_string())
        })?;

        let mut header = Self {
            magic: WAL_MAGIC,
            sequence,
            timestamp,
            entry_type: entry_type as u32,
            payload_size,
            header_crc: 0,
            payload_crc: 0,
            reserved: [0; 20],
        };

        #[cfg(feature = "memory_mapped_persistence")]
        {
            header.payload_crc = crc32c(payload);
            header.header_crc = header.compute_header_crc();
        }

        Ok(header)
    }

    /// Compute CRC32C of header fields (excluding `header_crc` itself)
    #[cfg(feature = "memory_mapped_persistence")]
    #[must_use]
    fn compute_header_crc(&self) -> u32 {
        let mut bytes = Vec::with_capacity(HEADER_SIZE - 4);

        bytes.extend_from_slice(&self.magic.to_le_bytes());
        bytes.extend_from_slice(&self.sequence.to_le_bytes());
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes.extend_from_slice(&self.entry_type.to_le_bytes());
        bytes.extend_from_slice(&self.payload_size.to_le_bytes());
        bytes.extend_from_slice(&self.payload_crc.to_le_bytes());
        bytes.extend_from_slice(&self.reserved);

        crc32c(&bytes)
    }

    /// Validate header integrity
    ///
    /// # Errors
    ///
    /// Returns an error if the magic value is corrupted or the stored
    /// checksum does not match the computed checksum.
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn validate(&self) -> StorageResult<()> {
        if self.magic != WAL_MAGIC {
            return Err(StorageError::CorruptionDetected(format!(
                "Invalid magic number: expected {:x}, got {:x}",
                WAL_MAGIC, self.magic
            )));
        }

        let expected_crc = self.compute_header_crc();
        if self.header_crc != expected_crc {
            return Err(StorageError::ChecksumMismatch {
                expected: expected_crc,
                actual: self.header_crc,
            });
        }

        Ok(())
    }

    /// Validate payload checksum
    ///
    /// # Errors
    ///
    /// Returns an error if the payload size differs from the header-specified
    /// size or the payload checksum mismatches.
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn validate_payload(&self, payload: &[u8]) -> StorageResult<()> {
        if payload.len() != self.payload_size as usize {
            return Err(StorageError::CorruptionDetected(format!(
                "Payload size mismatch: expected {}, got {}",
                self.payload_size,
                payload.len()
            )));
        }

        let actual_crc = crc32c(payload);
        if self.payload_crc != actual_crc {
            return Err(StorageError::ChecksumMismatch {
                expected: self.payload_crc,
                actual: actual_crc,
            });
        }

        Ok(())
    }

    #[cfg(not(feature = "memory_mapped_persistence"))]
    pub fn validate(&self) -> StorageResult<()> {
        Ok(())
    }

    #[cfg(not(feature = "memory_mapped_persistence"))]
    pub fn validate_payload(&self, _payload: &[u8]) -> StorageResult<()> {
        Ok(())
    }

    /// Serialize header to bytes in little-endian format
    #[must_use]
    pub fn as_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        let mut offset = 0;

        // Write each field in little-endian format
        bytes[offset..offset + 4].copy_from_slice(&self.magic.to_le_bytes());
        offset += 4;
        bytes[offset..offset + 8].copy_from_slice(&self.sequence.to_le_bytes());
        offset += 8;
        bytes[offset..offset + 8].copy_from_slice(&self.timestamp.to_le_bytes());
        offset += 8;
        bytes[offset..offset + 4].copy_from_slice(&self.entry_type.to_le_bytes());
        offset += 4;
        bytes[offset..offset + 4].copy_from_slice(&self.payload_size.to_le_bytes());
        offset += 4;
        bytes[offset..offset + 4].copy_from_slice(&self.payload_crc.to_le_bytes());
        offset += 4;
        bytes[offset..offset + 4].copy_from_slice(&self.header_crc.to_le_bytes());
        offset += 4;
        // Write reserved bytes
        bytes[offset..offset + 20].copy_from_slice(&self.reserved);

        bytes
    }

    /// Deserialize header from bytes in little-endian format
    #[must_use]
    pub fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> Self {
        let mut offset = 0;

        let magic = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        let sequence = u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        offset += 8;

        let timestamp = u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        offset += 8;

        let entry_type = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        let payload_size = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        let payload_crc = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        let header_crc = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;

        let mut reserved = [0u8; 20];
        reserved.copy_from_slice(&bytes[offset..offset + 20]);

        Self {
            magic,
            sequence,
            timestamp,
            entry_type,
            payload_size,
            payload_crc,
            header_crc,
            reserved,
        }
    }
}

/// Types of WAL entries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WalEntryType {
    /// Episode storage operation
    EpisodeStore = 1,
    /// Memory update operation
    MemoryUpdate = 2,
    /// Memory deletion operation
    MemoryDelete = 3,
    /// Memory consolidation operation
    Consolidation = 4,
    /// Checkpoint marker
    Checkpoint = 5,
    /// Log compaction marker
    CompactionMarker = 6,
}

impl From<u32> for WalEntryType {
    fn from(value: u32) -> Self {
        match value {
            2 => Self::MemoryUpdate,
            3 => Self::MemoryDelete,
            4 => Self::Consolidation,
            5 => Self::Checkpoint,
            6 => Self::CompactionMarker,
            _ => Self::EpisodeStore, // Default fallback
        }
    }
}

/// WAL entry with variable-size payload
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Entry header with metadata
    pub header: WalEntryHeader,
    /// Variable-size payload data
    pub payload: Vec<u8>,
}

impl WalEntry {
    /// Create entry for episode storage
    ///
    /// # Errors
    ///
    /// Returns an error if the episode cannot be serialized or the payload
    /// exceeds the WAL header limits.
    #[must_use = "Handle WAL episode serialization errors before continuing"]
    pub fn new_episode(episode: &Episode) -> StorageResult<Self> {
        let payload = bincode::serialize(episode)
            .map_err(|e| StorageError::CorruptionDetected(format!("Serialization failed: {e}")))?;

        let header = WalEntryHeader::new(WalEntryType::EpisodeStore, &payload, 0)?; // sequence filled later

        Ok(Self { header, payload })
    }

    /// Create entry for memory update
    ///
    /// # Errors
    ///
    /// Returns an error if the memory cannot be serialized or the payload
    /// exceeds the WAL header limits.
    #[must_use = "Handle WAL memory update serialization errors before continuing"]
    pub fn new_memory_update(memory: &Memory) -> StorageResult<Self> {
        let payload = bincode::serialize(memory)
            .map_err(|e| StorageError::CorruptionDetected(format!("Serialization failed: {e}")))?;

        let header = WalEntryHeader::new(WalEntryType::MemoryUpdate, &payload, 0)?;

        Ok(Self { header, payload })
    }

    /// Create entry for memory deletion
    ///
    /// # Errors
    ///
    /// Returns an error if the payload exceeds the WAL header limits.
    #[must_use = "Check WAL memory deletion serialization failures before discarding results"]
    pub fn new_memory_delete(memory_id: &str) -> StorageResult<Self> {
        let payload = memory_id.as_bytes().to_vec();

        let header = WalEntryHeader::new(WalEntryType::MemoryDelete, &payload, 0)?;

        Ok(Self { header, payload })
    }

    /// Create checkpoint entry
    ///
    /// # Errors
    ///
    /// Returns an error if the payload exceeds the WAL header limits.
    #[must_use = "Inspect WAL checkpoint creation failures to maintain durability"]
    pub fn new_checkpoint(sequence: u64) -> StorageResult<Self> {
        let payload = sequence.to_le_bytes().to_vec();

        let header = WalEntryHeader::new(WalEntryType::Checkpoint, &payload, 0)?;

        Ok(Self { header, payload })
    }

    /// Get total serialized size
    #[must_use]
    pub const fn serialized_size(&self) -> usize {
        HEADER_SIZE + self.payload.len()
    }

    /// Validate entry integrity
    ///
    /// # Errors
    ///
    /// Returns an error when header or payload verification fails.
    pub fn validate(&self) -> StorageResult<()> {
        self.header.validate()?;
        self.header.validate_payload(&self.payload)?;
        Ok(())
    }
}

/// Batch of entries for group commit
#[derive(Debug)]
pub struct WalBatch {
    /// Batch of WAL entries
    pub entries: Vec<WalEntry>,
    /// Total size of all entries in bytes
    pub total_size: usize,
    /// When this batch was created
    pub created_at: Instant,
}

impl WalBatch {
    /// Create a new empty WAL batch
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_size: 0,
            created_at: Instant::now(),
        }
    }

    /// Add an entry to the batch
    pub fn add_entry(&mut self, entry: WalEntry) {
        self.total_size += entry.serialized_size();
        self.entries.push(entry);
    }

    /// Check if the batch is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of entries in the batch
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }
}

impl Default for WalBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from WAL compaction
#[derive(Debug, Clone, Copy)]
pub struct CompactionStats {
    /// Number of episodes written to new WAL
    pub entries_written: usize,
    /// Total bytes written
    pub bytes_written: u64,
    /// Bytes reclaimed by removing old WAL files
    pub bytes_reclaimed: u64,
}

/// High-performance write-ahead log implementation
pub struct WalWriter {
    /// Root directory where WAL files are written
    wal_dir: PathBuf,
    /// File handle for WAL
    file: Arc<parking_lot::Mutex<BufWriter<File>>>,

    /// Global sequence counter
    sequence_counter: AtomicU64,

    /// Timestamp (nanoseconds since epoch) of last successful write
    last_write_timestamp_ns: Arc<AtomicU64>,

    /// Entry queue for batching
    entry_queue: Arc<SegQueue<WalEntry>>,

    /// Batch commit settings
    fsync_mode: FsyncMode,
    max_batch_size: usize,
    max_batch_delay: std::time::Duration,

    /// Performance metrics
    metrics: Arc<StorageMetrics>,

    /// Writer thread handle (uses interior mutability for shutdown)
    writer_thread: Arc<parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>>,

    /// Shutdown signal
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl WalWriter {
    /// Create a new WAL writer
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL directory cannot be created or the log file
    /// fails to open with the requested permissions.
    #[must_use = "Handle WAL writer initialization errors to preserve durability"]
    pub fn new<P: AsRef<Path>>(
        wal_dir: P,
        fsync_mode: FsyncMode,
        metrics: Arc<StorageMetrics>,
    ) -> StorageResult<Self> {
        let wal_dir = wal_dir.as_ref().to_path_buf();

        // Create WAL directory if needed
        std::fs::create_dir_all(&wal_dir)?;

        // Generate WAL file name with timestamp
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let wal_filename = format!("wal-{timestamp:016x}.log");
        let current_file = wal_dir.join(wal_filename);

        // Open WAL file with appropriate flags
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&current_file)?;

        // Set file permissions for security
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = file.metadata()?.permissions();
            perms.set_mode(0o600); // Owner read/write only
            std::fs::set_permissions(&current_file, perms)?;
        }

        let buf_writer = BufWriter::with_capacity(64 * 1024, file); // 64KB buffer

        // Initialize last write timestamp to current time
        let now_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let init_timestamp = u64::try_from(now_ns.min(u128::from(u64::MAX))).unwrap_or(u64::MAX);

        Ok(Self {
            wal_dir,
            file: Arc::new(parking_lot::Mutex::new(buf_writer)),
            sequence_counter: AtomicU64::new(0),
            last_write_timestamp_ns: Arc::new(AtomicU64::new(init_timestamp)),
            entry_queue: Arc::new(SegQueue::new()),
            fsync_mode,
            max_batch_size: 1000,
            max_batch_delay: std::time::Duration::from_millis(10),
            metrics,
            writer_thread: Arc::new(parking_lot::Mutex::new(None)),
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Start background writer thread
    ///
    /// # Errors
    ///
    /// Returns an error if spawning the writer thread fails.
    pub fn start(&self) -> StorageResult<()> {
        {
            let writer_thread = self.writer_thread.lock();
            if writer_thread.is_some() {
                return Ok(()); // Already started
            }
        }

        let entry_queue = self.entry_queue.clone();
        let file = Arc::clone(&self.file);
        let metrics = Arc::clone(&self.metrics);
        let shutdown = Arc::clone(&self.shutdown);
        let last_write_timestamp_ns = Arc::clone(&self.last_write_timestamp_ns);
        let fsync_mode = self.fsync_mode;
        let max_batch_size = self.max_batch_size;
        let max_batch_delay = self.max_batch_delay;

        let handle = std::thread::Builder::new()
            .name("wal-writer".to_string())
            .spawn(move || {
                Self::writer_loop(
                    &entry_queue,
                    &file,
                    &metrics,
                    &shutdown,
                    &last_write_timestamp_ns,
                    fsync_mode,
                    max_batch_size,
                    max_batch_delay,
                );
            })
            .map_err(|e| {
                StorageError::wal_failed(&format!("Failed to start writer thread: {e}"))
            })?;

        *self.writer_thread.lock() = Some(handle);
        Ok(())
    }

    /// Return the WAL directory used by this writer
    #[must_use]
    pub fn wal_directory(&self) -> &Path {
        &self.wal_dir
    }

    /// Calculate WAL lag in milliseconds
    ///
    /// Returns the time difference between the current time and the timestamp
    /// of the last successfully written entry. A value of 0.0 indicates the WAL
    /// is fully caught up with writes.
    #[must_use]
    pub fn lag_ms(&self) -> f64 {
        let now_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        let last_write_ns = u128::from(self.last_write_timestamp_ns.load(Ordering::Relaxed));

        if now_ns <= last_write_ns {
            return 0.0;
        }

        let lag_ns = now_ns.saturating_sub(last_write_ns);

        // Convert nanoseconds to milliseconds
        lag_ns as f64 / 1_000_000.0
    }

    /// Write entry asynchronously
    pub fn write_async(&self, entry: WalEntry) {
        self.entry_queue.push(entry);
    }

    /// Write entry synchronously with immediate fsync
    ///
    /// # Errors
    ///
    /// Returns an error when any write, flush, or fsync operation fails.
    #[must_use = "Propagate WAL write failures to trigger recovery"]
    pub fn write_sync(&self, mut entry: WalEntry) -> StorageResult<u64> {
        let sequence = self.sequence_counter.fetch_add(1, Ordering::SeqCst);
        entry.header.sequence = sequence;

        // Recompute header CRC after updating sequence
        #[cfg(feature = "memory_mapped_persistence")]
        {
            entry.header.header_crc = entry.header.compute_header_crc();
        }

        let mut file = self.file.lock();

        // Write header using proper serialization
        let header_bytes = entry.header.as_bytes();
        file.write_all(&header_bytes)?;

        // Write payload
        file.write_all(&entry.payload)?;

        // Fsync for durability
        file.flush()?;
        file.get_mut().sync_all()?;
        drop(file);

        self.metrics
            .record_write((HEADER_SIZE + entry.payload.len()) as u64);
        self.metrics.record_fsync();

        // Update last write timestamp for lag tracking
        self.last_write_timestamp_ns
            .store(entry.header.timestamp, Ordering::Relaxed);

        Ok(sequence)
    }

    /// Background writer loop
    fn writer_loop(
        entry_queue: &Arc<SegQueue<WalEntry>>,
        file: &Arc<parking_lot::Mutex<BufWriter<File>>>,
        metrics: &Arc<StorageMetrics>,
        shutdown: &Arc<std::sync::atomic::AtomicBool>,
        last_write_timestamp_ns: &Arc<AtomicU64>,
        fsync_mode: FsyncMode,
        max_batch_size: usize,
        max_batch_delay: std::time::Duration,
    ) {
        let mut batch = WalBatch::new();
        let mut sequence_counter = 0u64;

        while !shutdown.load(Ordering::Relaxed) {
            // Collect entries into batch
            let batch_start = Instant::now();
            let mut entries_collected = 0;

            while entries_collected < max_batch_size && batch_start.elapsed() < max_batch_delay {
                if let Some(mut entry) = entry_queue.pop() {
                    entry.header.sequence = sequence_counter;
                    sequence_counter += 1;
                    #[cfg(feature = "memory_mapped_persistence")]
                    {
                        entry.header.header_crc = entry.header.compute_header_crc();
                    }
                    batch.add_entry(entry);
                    entries_collected += 1;
                } else {
                    // No entries available, short sleep
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
            }

            if batch.is_empty() {
                continue;
            }

            // Write batch
            if let Err(e) = Self::write_batch(file, &batch, metrics, fsync_mode) {
                tracing::error!("WAL batch write failed: {}", e);
                // Could implement retry logic here
            } else {
                // Update last write timestamp with the latest entry in the batch
                if let Some(last_entry) = batch.entries.last() {
                    last_write_timestamp_ns.store(last_entry.header.timestamp, Ordering::Relaxed);
                }
            }

            batch = WalBatch::new();
        }

        // Flush remaining entries on shutdown
        while let Some(mut entry) = entry_queue.pop() {
            entry.header.sequence = sequence_counter;
            sequence_counter += 1;
            #[cfg(feature = "memory_mapped_persistence")]
            {
                entry.header.header_crc = entry.header.compute_header_crc();
            }
            batch.add_entry(entry);
        }

        if !batch.is_empty() && Self::write_batch(file, &batch, metrics, fsync_mode).is_ok() {
            // Update last write timestamp with the latest entry in the final batch
            if let Some(last_entry) = batch.entries.last() {
                last_write_timestamp_ns.store(last_entry.header.timestamp, Ordering::Relaxed);
            }
        }
    }

    /// Write a batch of entries
    ///
    /// # Errors
    ///
    /// Returns an error if writing any entry or flushing the backing file
    /// fails.
    fn write_batch(
        file: &Arc<parking_lot::Mutex<BufWriter<File>>>,
        batch: &WalBatch,
        metrics: &StorageMetrics,
        fsync_mode: FsyncMode,
    ) -> StorageResult<()> {
        let mut file = file.lock();
        let start = Instant::now();

        for entry in &batch.entries {
            // Write header using proper serialization
            let header_bytes = entry.header.as_bytes();
            file.write_all(&header_bytes)?;

            // Write payload
            file.write_all(&entry.payload)?;
        }

        // Flush buffer
        file.flush()?;

        // Fsync based on mode
        match fsync_mode {
            FsyncMode::PerWrite | FsyncMode::PerBatch => {
                file.get_mut().sync_all()?;
                metrics.record_fsync();
            }
            FsyncMode::Timer(_) | FsyncMode::None => {
                // Deferred or disabled fsync handling
            }
        }

        drop(file);
        metrics.record_write(batch.total_size as u64);

        let duration = start.elapsed();
        if duration.as_millis() > 10 {
            tracing::warn!(
                "WAL batch write took {}ms, target <10ms",
                duration.as_millis()
            );
        }

        Ok(())
    }

    /// Forcefully sync all pending writes
    ///
    /// # Errors
    ///
    /// Returns an error if flushing or syncing the file fails.
    pub fn fsync(&self) -> StorageResult<()> {
        let mut file = self.file.lock();
        file.flush()?;
        file.get_mut().sync_all()?;
        drop(file);
        self.metrics.record_fsync();
        Ok(())
    }

    /// Shutdown writer and ensure all entries are flushed
    ///
    /// # Errors
    ///
    /// Returns an error if the writer thread panics or final fsync fails.
    pub fn shutdown(&self) -> StorageResult<()> {
        self.shutdown.store(true, Ordering::SeqCst);

        let handle = self.writer_thread.lock().take();
        if let Some(handle) = handle {
            handle
                .join()
                .map_err(|_| StorageError::wal_failed("Writer thread panicked during shutdown"))?;
        }

        // Final fsync
        self.fsync()?;

        Ok(())
    }

    /// Enhanced WAL operations with proper error recovery.
    /// These methods replace unsafe `unwrap()` calls with recovery strategies.
    ///
    /// # Errors
    ///
    /// Returns an error when the write-ahead log cannot persist the entry even
    /// after applying the configured recovery strategy.
    #[must_use = "Propagate WAL write failures to trigger recovery"]
    pub fn write_entry_with_recovery(&self, entry: WalEntry) -> crate::error::Result<u64> {
        use crate::error::{EngramError, RecoveryStrategy};

        self.write_sync(entry)
            .map_err(|e| EngramError::WriteAheadLog {
                operation: "write_entry".to_string(),
                source: match e {
                    StorageError::Io(io_err) => io_err,
                    _ => std::io::Error::other(e.to_string()),
                },
                recovery: RecoveryStrategy::Retry {
                    max_attempts: 3,
                    backoff_ms: 100,
                },
                can_continue: false,
            })
    }

    /// Write episode with proper error handling instead of `unwrap()`.
    ///
    /// # Errors
    ///
    /// Returns an error when the episode fails serialization or the entry
    /// cannot be persisted to the write-ahead log.
    #[must_use = "Propagate WAL write failures to trigger recovery"]
    pub fn write_episode_with_recovery(&self, episode: &Episode) -> crate::error::Result<u64> {
        use crate::error::{EngramError, RecoveryStrategy};

        let entry = WalEntry::new_episode(episode).map_err(|e| EngramError::WriteAheadLog {
            operation: "serialize_episode".to_string(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            recovery: RecoveryStrategy::RequiresIntervention {
                action: "Check episode data validity".to_string(),
            },
            can_continue: true,
        })?;

        self.write_entry_with_recovery(entry)
    }

    /// Batch write with recovery handling.
    ///
    /// # Errors
    ///
    /// Returns an error if any entry write fails and recovery strategies cannot
    /// continue the batch safely.
    #[must_use = "Propagate WAL write failures to trigger recovery"]
    pub fn write_batch_with_recovery(&self, batch: WalBatch) -> crate::error::Result<Vec<u64>> {
        use crate::error::{EngramError, RecoveryStrategy};

        let mut sequences = Vec::new();

        for entry in batch.entries {
            match self.write_entry_with_recovery(entry) {
                Ok(seq) => sequences.push(seq),
                Err(e) if e.is_recoverable() => {
                    tracing::warn!("WAL write failed but recoverable: {}", e);
                    // For batch operations, we might want to continue with partial success
                    match e.recovery_strategy() {
                        RecoveryStrategy::PartialResult { .. } => {
                            tracing::info!("Continuing batch with partial results");
                            break;
                        }
                        _ => return Err(e),
                    }
                }
                Err(e) => return Err(e),
            }
        }

        if sequences.is_empty() {
            Err(EngramError::WriteAheadLog {
                operation: "write_batch".to_string(),
                source: std::io::Error::new(std::io::ErrorKind::WriteZero, "No entries written"),
                recovery: RecoveryStrategy::RequiresIntervention {
                    action: "Check WAL writer state and disk space".to_string(),
                },
                can_continue: false,
            })
        } else {
            Ok(sequences)
        }
    }

    /// Compact WAL by rewriting only valid current state.
    ///
    /// This creates a new WAL file containing only episodes currently in memory,
    /// effectively removing all corrupt entries and deleted episodes.
    ///
    /// # Safety
    ///
    /// Must be called AFTER recover_from_wal() completes, with all episodes
    /// successfully loaded into memory. Otherwise, data loss will occur.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Cannot create new WAL file
    /// - Writing episodes fails
    /// - Cannot atomically rename files
    /// - Cannot remove old WAL files
    ///
    /// If an error occurs mid-compaction, old WAL files are preserved.
    #[must_use = "WAL compaction failures can cause unbounded disk growth"]
    pub fn compact_from_memory<I>(&self, episodes: I) -> StorageResult<CompactionStats>
    where
        I: Iterator<Item = Episode>,
    {
        // 1. Create new temporary WAL file
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let temp_filename = format!("wal-{timestamp:016x}.log.tmp");
        let temp_path = self.wal_dir.join(&temp_filename);

        let temp_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&temp_path)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = temp_file.metadata()?.permissions();
            perms.set_mode(0o600);
            std::fs::set_permissions(&temp_path, perms)?;
        }

        let mut buf_writer = BufWriter::with_capacity(64 * 1024, temp_file);

        // 2. Write all valid episodes to new WAL
        let mut entries_written = 0usize;
        let mut bytes_written = 0u64;

        for (sequence, episode) in episodes.enumerate() {
            let mut entry = WalEntry::new_episode(&episode)?;
            entry.header.sequence = sequence as u64;

            #[cfg(feature = "memory_mapped_persistence")]
            {
                entry.header.header_crc = entry.header.compute_header_crc();
            }

            let header_bytes = entry.header.as_bytes();
            buf_writer.write_all(&header_bytes)?;
            buf_writer.write_all(&entry.payload)?;

            bytes_written += (HEADER_SIZE + entry.payload.len()) as u64;
            entries_written += 1;
        }

        // 3. Fsync new WAL
        buf_writer.flush()?;
        buf_writer.get_mut().sync_all()?;
        drop(buf_writer);

        // 4. Get size of old WAL files for metrics
        let mut old_bytes = 0u64;
        for entry in std::fs::read_dir(&self.wal_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "log")
                && !path.to_str().unwrap_or("").contains(".tmp")
            {
                if let Ok(metadata) = path.metadata() {
                    old_bytes += metadata.len();
                }
            }
        }

        // 5. Atomically rename new WAL (this is the commit point)
        let final_filename = format!("wal-{timestamp:016x}.log");
        let final_path = self.wal_dir.join(&final_filename);
        std::fs::rename(&temp_path, &final_path)?;

        // 6. Remove old WAL files (safe after atomic rename)
        for entry in std::fs::read_dir(&self.wal_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path != final_path && path.extension().is_some_and(|ext| ext == "log") {
                std::fs::remove_file(&path)?;
                tracing::info!("Removed old WAL file: {}", path.display());
            }
        }

        let stats = CompactionStats {
            entries_written,
            bytes_written,
            bytes_reclaimed: old_bytes.saturating_sub(bytes_written),
        };

        tracing::info!(
            entries_written = entries_written,
            bytes_written = bytes_written,
            bytes_reclaimed = stats.bytes_reclaimed,
            "WAL compaction completed"
        );

        Ok(stats)
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// WAL reader for recovery operations
pub struct WalReader {
    wal_dir: PathBuf,
    metrics: Arc<StorageMetrics>,
}

impl WalReader {
    /// Create a new WAL reader for the given directory
    pub fn new<P: AsRef<Path>>(wal_dir: P, metrics: Arc<StorageMetrics>) -> Self {
        Self {
            wal_dir: wal_dir.as_ref().to_path_buf(),
            metrics,
        }
    }

    /// Scan all WAL files and return entries in sequence order.
    ///
    /// # Errors
    ///
    /// Returns an error if directory iteration or WAL file deserialization
    /// fails for reasons other than expected end-of-file conditions.
    pub fn scan_all(&self) -> StorageResult<Vec<WalEntry>> {
        let mut all_entries = Vec::new();

        // Find all WAL files
        let mut wal_files = Vec::new();
        for entry in std::fs::read_dir(&self.wal_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "log")
                && path
                    .file_name()
                    .is_some_and(|name| name.to_str().unwrap_or("").starts_with("wal-"))
            {
                wal_files.push(path);
            }
        }

        // Sort by filename (timestamp)
        wal_files.sort();

        // Read entries from each file
        for wal_file in wal_files {
            let entries = self.read_wal_file(&wal_file)?;
            all_entries.extend(entries);
        }

        // Sort by sequence number
        all_entries.sort_by_key(|entry| entry.header.sequence);

        Ok(all_entries)
    }

    /// Read entries from a single WAL file
    fn read_wal_file(&self, path: &Path) -> StorageResult<Vec<WalEntry>> {
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut entries = Vec::new();

        loop {
            // Read header
            let mut header_bytes = [0u8; HEADER_SIZE];
            match file.read_exact(&mut header_bytes) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            // Parse header using proper deserialization
            let header = WalEntryHeader::from_bytes(&header_bytes);

            // Validate header
            if let Err(e) = header.validate() {
                tracing::warn!("Corrupted header in {}: {}", path.display(), e);
                println!("Header validation failed: {e}");
                break; // Skip rest of file
            }

            // Read payload
            let mut payload = vec![0u8; header.payload_size as usize];
            file.read_exact(&mut payload)?;

            // Validate payload
            if let Err(e) = header.validate_payload(&payload) {
                tracing::warn!("Corrupted payload in {}: {}", path.display(), e);
                println!("Payload validation failed: {e}");
                continue; // Skip this entry but continue reading
            }

            let payload_len = payload.len();
            entries.push(WalEntry { header, payload });
            self.metrics.record_read((HEADER_SIZE + payload_len) as u64);
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Confidence;
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

    fn create_test_episode() -> Episode {
        use crate::EpisodeBuilder;
        use chrono::Utc;

        EpisodeBuilder::new()
            .id("test_episode".to_string())
            .when(Utc::now())
            .what("test content".to_string())
            .embedding([0.5f32; 768])
            .confidence(Confidence::HIGH)
            .build()
    }

    #[test]
    fn test_wal_header_size() {
        assert_eq!(std::mem::size_of::<WalEntryHeader>(), 64);
        // Alignment should be natural (8 bytes for u64 fields)
        assert!(std::mem::align_of::<WalEntryHeader>() >= 8);
    }

    #[test]
    fn test_wal_entry_creation() -> TestResult {
        let episode = create_test_episode();
        let entry = WalEntry::new_episode(&episode).into_test_result("create wal entry")?;

        ensure_eq(&entry.header.magic, &WAL_MAGIC, "wal header magic")?;
        ensure_eq(
            &entry.header.entry_type,
            &(WalEntryType::EpisodeStore as u32),
            "wal entry type",
        )?;
        ensure(
            !entry.payload.is_empty(),
            "wal entry payload should not be empty",
        )?;

        Ok(())
    }

    #[test]
    fn test_wal_writer_sync() -> TestResult {
        let temp_dir = TempDir::new().into_test_result("create temp dir for wal writer sync")?;
        let metrics = Arc::new(StorageMetrics::new());
        let wal = WalWriter::new(temp_dir.path(), FsyncMode::PerWrite, metrics)
            .into_test_result("construct wal writer")?;

        let episode = create_test_episode();
        let entry = WalEntry::new_episode(&episode).into_test_result("create wal entry")?;

        let sequence = wal
            .write_sync(entry)
            .into_test_result("write wal entry synchronously")?;
        ensure_eq(&sequence, &0_u64, "wal sequence")?;

        Ok(())
    }

    #[test]
    fn test_wal_write_and_read() -> TestResult {
        let temp_dir = TempDir::new().into_test_result("create wal temp dir")?;
        let metrics = Arc::new(StorageMetrics::new());
        let wal = WalWriter::new(temp_dir.path(), FsyncMode::PerWrite, Arc::clone(&metrics))
            .into_test_result("construct wal writer for write/read test")?;

        let episode = create_test_episode();
        let entry = WalEntry::new_episode(&episode).into_test_result("serialize episode")?;

        wal.write_sync(entry)
            .into_test_result("write wal entry for readback")?;
        wal.shutdown()
            .into_test_result("shutdown wal writer after write")?;

        let reader = WalReader::new(temp_dir.path(), metrics);
        let entries = reader
            .scan_all()
            .into_test_result("scan wal entries for readback")?;

        ensure_eq(&entries.len(), &1_usize, "wal entry count after readback")?;
        let recovered = entries
            .first()
            .ok_or_else(|| "missing wal entry".to_string())?;
        recovered
            .validate()
            .into_test_result("validate wal entry during readback")?;

        Ok(())
    }

    #[test]
    fn test_wal_reader_recovery() -> TestResult {
        let temp_dir = TempDir::new().into_test_result("create temp dir for wal reader")?;
        let metrics = Arc::new(StorageMetrics::new());

        // Write some entries
        let wal = WalWriter::new(temp_dir.path(), FsyncMode::PerWrite, metrics.clone())
            .into_test_result("construct wal writer for recovery test")?;

        for _i in 0..5 {
            let episode = create_test_episode();
            let entry = WalEntry::new_episode(&episode).into_test_result("create wal entry")?;
            wal.write_sync(entry)
                .into_test_result("write wal entry during recovery test")?;
        }

        // Explicitly drop to ensure all data is flushed
        wal.shutdown()
            .into_test_result("shutdown wal writer during recovery test")?;
        drop(wal);

        // Read them back
        let reader = WalReader::new(temp_dir.path(), metrics);

        // Debug: List files in the directory
        println!("Files in WAL directory:");
        if let Ok(dir_entries) = std::fs::read_dir(temp_dir.path()) {
            for entry in dir_entries.flatten() {
                let path = entry.path();
                println!("  - {}", path.display());
                if let Ok(metadata) = std::fs::metadata(&path) {
                    println!("    Size: {} bytes", metadata.len());
                }
            }
        }

        let entries = reader.scan_all().into_test_result("scan wal entries")?;
        println!("Found {} entries, expected 5", entries.len());
        ensure_eq(&entries.len(), &5_usize, "wal entry count")?;
        for (i, entry) in entries.iter().enumerate() {
            ensure_eq(&entry.header.sequence, &(i as u64), "wal entry sequence")?;
            ensure_eq(
                &entry.header.entry_type,
                &(WalEntryType::EpisodeStore as u32),
                "wal entry type",
            )?;
            entry.validate().into_test_result("validate wal entry")?;
        }

        Ok(())
    }
}
