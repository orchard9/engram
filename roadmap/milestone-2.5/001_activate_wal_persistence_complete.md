# Task 001: Activate WAL Persistence

## Status: Complete
## Priority: P0 - Critical (Blocks Production)
## Estimated Effort: 2 days
## Dependencies: None

## Objective

Wire the existing WAL writer into MemoryStore so every store operation writes to disk with crash-consistent durability guarantees.

## Current State

**Infrastructure EXISTS but INACTIVE:**
- ✅ `engram-core/src/storage/wal.rs:1-400` - Full WAL implementation
- ✅ `WalWriter` struct with batching, fsync modes, CRC32C checksums
- ✅ `WalEntryHeader` with 64-byte cache-aligned format
- ❌ MemoryStore doesn't call it - persistence spawns async but doesn't write WAL

**Relevant Code:**
```rust
// engram-core/src/store.rs:203-207 (CURRENT)
if let Some(ref backend) = self.persistent_backend {
    let backend = Arc::clone(backend);
    let memory = Arc::clone(memory_arc);
    Self::spawn_persistence_task(backend, memory);  // ← Calls backend but NOT WAL!
}
```

## Implementation Steps

### Step 1: Add WAL Writer to MemoryStore (30 min)

**File**: `engram-core/src/store.rs`

**Line 115** - Add WAL writer field:
```rust
/// Write-ahead log for durability
#[cfg(feature = "memory_mapped_persistence")]
wal_writer: Option<Arc<WalWriter>>,
```

**Line 245** - Initialize in constructor:
```rust
#[cfg(feature = "memory_mapped_persistence")]
wal_writer: None,
```

**Line 262-275** - Add constructor method:
```rust
/// Enable WAL-based persistence
#[cfg(feature = "memory_mapped_persistence")]
#[must_use]
pub fn with_wal<P: AsRef<Path>>(mut self, wal_dir: P, fsync_mode: FsyncMode) -> Result<Self, StorageError> {
    use crate::storage::wal::WalWriter;

    let metrics = Arc::clone(&self.storage_metrics);
    let wal = WalWriter::new(wal_dir, fsync_mode, metrics)?;
    self.wal_writer = Some(Arc::new(wal));
    Ok(self)
}
```

### Step 2: Write to WAL on Store Operations (1 hour)

**File**: `engram-core/src/store.rs`

**Line 453-456** - Replace current WAL buffer insert with real WAL write:
```rust
// REMOVE THIS:
self.wal_buffer.insert(memory_id.clone(), wal_episode);

// REPLACE WITH:
#[cfg(feature = "memory_mapped_persistence")]
if let Some(ref wal) = self.wal_writer {
    use crate::storage::wal::{WalEntry, WalEntryType};

    let entry = WalEntry {
        entry_type: WalEntryType::MemoryStore,
        payload: bincode::serialize(&wal_episode)
            .map_err(|e| tracing::warn!("WAL serialization failed: {}", e))
            .unwrap_or_default(),
    };

    if let Err(e) = wal.write(entry) {
        tracing::error!("WAL write failed: {:?}", e);
        // Graceful degradation: continue without WAL
    }
}
```

### Step 3: Implement WAL Recovery on Startup (2 hours)

**File**: `engram-core/src/store.rs`

**Line 280-310** - Add recovery method:
```rust
/// Recover memories from WAL on startup
#[cfg(feature = "memory_mapped_persistence")]
pub fn recover_from_wal(&mut self) -> Result<usize, StorageError> {
    use crate::storage::wal::WalReader;

    let Some(ref wal_writer) = self.wal_writer else {
        return Ok(0);
    };

    let wal_dir = wal_writer.wal_directory();
    let reader = WalReader::new(wal_dir)?;
    let mut recovered = 0;

    for entry in reader.iter() {
        let entry = entry?;

        if entry.header.entry_type == WalEntryType::MemoryStore as u32 {
            match bincode::deserialize::<WalEpisode>(&entry.payload) {
                Ok(wal_episode) => {
                    // Convert WalEpisode back to Memory
                    let memory = Arc::new(Memory::from_wal_episode(wal_episode));
                    let id = memory.id.clone();

                    self.hot_memories.insert(id.clone(), Arc::clone(&memory));
                    self.memory_count.fetch_add(1, Ordering::Relaxed);
                    recovered += 1;
                }
                Err(e) => {
                    tracing::warn!("Skipping corrupted WAL entry: {}", e);
                }
            }
        }
    }

    tracing::info!("Recovered {} memories from WAL", recovered);
    Ok(recovered)
}
```

### Step 4: Add WAL Reader (3 hours)

**File**: `engram-core/src/storage/wal.rs`

**Line 400** - Add WalReader implementation:
```rust
/// WAL reader for recovery
pub struct WalReader {
    wal_dir: PathBuf,
    current_file: Option<File>,
}

impl WalReader {
    pub fn new<P: AsRef<Path>>(wal_dir: P) -> StorageResult<Self> {
        Ok(Self {
            wal_dir: wal_dir.as_ref().to_path_buf(),
            current_file: None,
        })
    }

    pub fn iter(&self) -> WalIterator {
        WalIterator::new(self.wal_dir.clone())
    }
}

pub struct WalIterator {
    wal_files: Vec<PathBuf>,
    current_index: usize,
    current_reader: Option<BufReader<File>>,
}

impl WalIterator {
    fn new(wal_dir: PathBuf) -> Self {
        let mut wal_files: Vec<PathBuf> = std::fs::read_dir(&wal_dir)
            .unwrap_or_else(|_| std::fs::read_dir(".").unwrap())
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("log"))
            .collect();

        wal_files.sort(); // Process in chronological order

        Self {
            wal_files,
            current_index: 0,
            current_reader: None,
        }
    }
}

impl Iterator for WalIterator {
    type Item = StorageResult<WalEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try to read from current file
            if let Some(ref mut reader) = self.current_reader {
                match self.read_entry(reader) {
                    Ok(Some(entry)) => return Some(Ok(entry)),
                    Ok(None) => {
                        // End of file, move to next
                        self.current_reader = None;
                        self.current_index += 1;
                    }
                    Err(e) => return Some(Err(e)),
                }
            } else {
                // Open next file
                if self.current_index >= self.wal_files.len() {
                    return None; // No more files
                }

                match File::open(&self.wal_files[self.current_index]) {
                    Ok(file) => {
                        self.current_reader = Some(BufReader::new(file));
                    }
                    Err(e) => {
                        return Some(Err(StorageError::IoError {
                            operation: "open WAL file".to_string(),
                            expected: "readable file".to_string(),
                            suggestion: "check file permissions".to_string(),
                            example: "chmod 644 wal-*.log".to_string(),
                            source: e,
                        }));
                    }
                }
            }
        }
    }
}

impl WalIterator {
    fn read_entry(&self, reader: &mut BufReader<File>) -> StorageResult<Option<WalEntry>> {
        use std::io::Read;

        // Read header (64 bytes)
        let mut header_bytes = [0u8; HEADER_SIZE];
        match reader.read_exact(&mut header_bytes) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None); // End of file
            }
            Err(e) => return Err(StorageError::IoError {
                operation: "read WAL header".to_string(),
                expected: "64 bytes".to_string(),
                suggestion: "WAL file may be corrupted".to_string(),
                example: "delete and recover from backup".to_string(),
                source: e,
            }),
        }

        let header: WalEntryHeader = unsafe {
            std::ptr::read(header_bytes.as_ptr() as *const WalEntryHeader)
        };

        // Validate magic number
        if header.magic != WAL_MAGIC {
            return Err(StorageError::Configuration(
                format!("Invalid WAL magic: 0x{:08x}", header.magic)
            ));
        }

        // Verify header CRC
        #[cfg(feature = "memory_mapped_persistence")]
        {
            let expected_crc = header.compute_header_crc();
            if header.header_crc != expected_crc {
                return Err(StorageError::Configuration(
                    "WAL header checksum mismatch".to_string()
                ));
            }
        }

        // Read payload
        let mut payload = vec![0u8; header.payload_size as usize];
        reader.read_exact(&mut payload)?;

        // Verify payload CRC
        #[cfg(feature = "memory_mapped_persistence")]
        {
            let payload_crc = crc32c(&payload);
            if header.payload_crc != payload_crc {
                return Err(StorageError::Configuration(
                    "WAL payload checksum mismatch".to_string()
                ));
            }
        }

        Ok(Some(WalEntry {
            entry_type: WalEntryType::from_u32(header.entry_type),
            payload,
        }))
    }
}
```

### Step 5: Update Main to Call Recovery (15 min)

**File**: `engram-cli/src/main.rs`

**Line 150-157** - Replace memory store initialization:
```rust
// Initialize memory store with HNSW indexing
use engram_core::MemoryStore;

#[cfg(feature = "hnsw_index")]
let mut memory_store = MemoryStore::new(100_000).with_hnsw_index();

#[cfg(not(feature = "hnsw_index"))]
let mut memory_store = MemoryStore::new(100_000);

// Enable WAL persistence
#[cfg(feature = "memory_mapped_persistence")]
{
    use engram_core::storage::FsyncMode;

    memory_store = memory_store
        .with_wal("./data/wal", FsyncMode::PerBatch)
        .expect("Failed to initialize WAL");

    // Recover from WAL on startup
    match memory_store.recover_from_wal() {
        Ok(count) => tracing::info!("Recovered {} memories from WAL", count),
        Err(e) => tracing::warn!("WAL recovery failed: {:?}", e),
    }
}

let memory_store = Arc::new(memory_store);
```

### Step 6: Add WAL Write Method (2 hours)

**File**: `engram-core/src/storage/wal.rs`

**Line 150-200** - Add write method to WalWriter:
```rust
impl WalWriter {
    /// Write an entry to the WAL
    pub fn write(&self, entry: WalEntry) -> StorageResult<()> {
        let sequence = self.sequence_counter.fetch_add(1, Ordering::SeqCst);

        let header = WalEntryHeader::new(
            entry.entry_type,
            &entry.payload,
            sequence,
        )?;

        // Queue for batch processing
        self.entry_queue.push(WalQueuedEntry { header, payload: entry.payload });

        // Trigger flush if batch size reached
        if self.entry_queue.len() >= self.max_batch_size {
            self.flush_batch()?;
        }

        Ok(())
    }

    fn flush_batch(&self) -> StorageResult<()> {
        let mut file = self.file.lock();
        let mut batch_size = 0;

        while let Some(queued) = self.entry_queue.pop() {
            // Write header
            let header_bytes = unsafe {
                std::slice::from_raw_parts(
                    &queued.header as *const _ as *const u8,
                    HEADER_SIZE,
                )
            };
            file.write_all(header_bytes)?;

            // Write payload
            file.write_all(&queued.payload)?;

            batch_size += 1;
            if batch_size >= self.max_batch_size {
                break;
            }
        }

        // Fsync based on mode
        match self.fsync_mode {
            FsyncMode::PerWrite => file.flush()?,
            FsyncMode::PerBatch => file.flush()?,
            FsyncMode::Timer => {
                // Flush will happen on timer thread
            }
            FsyncMode::None => {}
        }

        self.metrics.record_batch_commit(batch_size);
        Ok(())
    }
}

struct WalQueuedEntry {
    header: WalEntryHeader,
    payload: Vec<u8>,
}
```

## Testing Strategy

### Unit Tests

**File**: `engram-core/src/storage/wal.rs`

Add at line 600:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_wal_write_and_read() {
        let dir = tempdir().unwrap();
        let metrics = Arc::new(StorageMetrics::new());
        let wal = WalWriter::new(dir.path(), FsyncMode::PerWrite, metrics).unwrap();

        let entry = WalEntry {
            entry_type: WalEntryType::MemoryStore,
            payload: b"test payload".to_vec(),
        };

        wal.write(entry.clone()).unwrap();
        wal.shutdown();

        // Read back
        let reader = WalReader::new(dir.path()).unwrap();
        let entries: Vec<_> = reader.iter().collect();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].as_ref().unwrap().payload, b"test payload");
    }
}
```

### Integration Test

**File**: `engram-core/tests/wal_recovery_test.rs` (create new)

```rust
use engram_core::{MemoryStore, Episode, Confidence};
use chrono::Utc;
use tempfile::tempdir;

#[test]
fn test_wal_recovery() {
    let wal_dir = tempdir().unwrap();

    // Create store and write data
    {
        let mut store = MemoryStore::new(100)
            .with_wal(wal_dir.path(), FsyncMode::PerWrite)
            .unwrap();

        let episode = Episode::new(
            "test_1".to_string(),
            Utc::now(),
            "test content".to_string(),
            [0.5; 768],
            Confidence::HIGH,
        );

        store.store(episode);
    }

    // Recover in new store
    let mut new_store = MemoryStore::new(100)
        .with_wal(wal_dir.path(), FsyncMode::PerWrite)
        .unwrap();

    let recovered = new_store.recover_from_wal().unwrap();
    assert_eq!(recovered, 1);
    assert_eq!(new_store.count(), 1);
}
```

## Acceptance Criteria

- [ ] `MemoryStore::with_wal()` initializes WalWriter
- [ ] Every `store(episode)` writes to WAL with CRC32C checksums
- [ ] `recover_from_wal()` restores memories on startup
- [ ] WAL recovery test passes (kill process, restart, verify data)
- [ ] Fsync modes configurable (PerWrite, PerBatch, Timer, None)
- [ ] P99 write latency <10ms with PerBatch mode
- [ ] Corrupted entries skipped with warning log

## Performance Targets

- WAL write overhead: <1ms per entry (batched)
- Recovery speed: >10,000 entries/sec
- Disk usage: ~1KB per memory (header + serialized episode)
- Fsync P99: <10ms (PerBatch mode)

## Files to Modify

1. `engram-core/src/store.rs` - Add wal_writer field, write on store, recovery method
2. `engram-core/src/storage/wal.rs` - Add write(), WalReader, WalIterator
3. `engram-cli/src/main.rs` - Initialize WAL and call recovery on startup
4. `engram-core/tests/wal_recovery_test.rs` - Create integration test

## Files to Read First

- `engram-core/src/storage/wal.rs:1-150` - Existing WAL structures
- `engram-core/src/store.rs:203-240` - Current persistence code
- `engram-core/src/storage/mod.rs:10-30` - Storage traits and types
