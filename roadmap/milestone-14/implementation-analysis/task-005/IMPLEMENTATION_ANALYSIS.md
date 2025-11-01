# Task 005: Replication Protocol Implementation Analysis

## Executive Summary

This detailed report provides implementation guidance for Task 005 (Replication Protocol for Episodic Memories) in Engram Milestone 14. The analysis covers exact file locations, current codebase patterns, integration points, and concrete code examples ready for implementation.

---

## 1. WAL IMPLEMENTATION ANALYSIS

### Location and Structure
**Primary File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/wal.rs`

### Current WAL Entry Format

The existing WAL implementation uses a **64-byte cache-line-aligned header** followed by variable-size payload:

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WalEntryHeader {
    pub magic: u32,           // 4 bytes: 0xDEAD_BEEF for corruption detection
    pub sequence: u64,        // 8 bytes: Monotonic sequence number
    pub timestamp: u64,       // 8 bytes: Wall clock nanoseconds since epoch
    pub entry_type: u32,      // 4 bytes: WalEntryType discriminant
    pub payload_size: u32,    // 4 bytes: Payload size (max 4GiB)
    pub payload_crc: u32,     // 4 bytes: CRC32C of payload
    pub header_crc: u32,      // 4 bytes: CRC32C of header fields
    pub reserved: [u8; 20],   // 20 bytes: Future extensions
}
// Total: 64 bytes (perfect cache line)

const HEADER_SIZE: usize = 64;
const WAL_MAGIC: u32 = 0xDEAD_BEEF;
```

**File location for header definition**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/wal.rs:29-49`

### WalEntry Type Enumeration

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WalEntryType {
    EpisodeStore = 1,      // Episode storage operation
    MemoryUpdate = 2,      // Memory update operation
    MemoryDelete = 3,      // Memory deletion operation
    Consolidation = 4,     // Memory consolidation operation
    Checkpoint = 5,        // Checkpoint marker
    CompactionMarker = 6,  // Log compaction marker
}
```

**File location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/wal.rs:286-315`

### Core WalEntry Struct

```rust
#[derive(Debug, Clone)]
pub struct WalEntry {
    pub header: WalEntryHeader,
    pub payload: Vec<u8>,
}
```

**File location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/wal.rs:318-324`

### Key Methods for Replication Integration

1. **Entry Creation** (lines 334-385):
   - `WalEntry::new_episode(&Episode)` - Creates episode entry
   - `WalEntry::new_memory_update(&Memory)` - Creates memory update entry
   - `WalEntry::new_memory_delete(&str)` - Creates deletion entry
   - `WalEntry::new_checkpoint(u64)` - Creates checkpoint entry

2. **Serialization** (lines 175-199):
   ```rust
   pub fn as_bytes(&self) -> [u8; HEADER_SIZE] {
       // Serializes header to 64 bytes in little-endian format
   }
   
   pub fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> Self {
       // Deserializes 64-byte header from little-endian
   }
   ```

3. **Validation** (lines 398-402):
   ```rust
   pub fn validate(&self) -> StorageResult<()> {
       self.header.validate()?;
       self.header.validate_payload(&self.payload)?;
       Ok(())
   }
   ```

### WalWriter Public Methods for Replication Hooks

**Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/wal.rs:464-1052`

Critical methods to hook into:

1. **write_sync()** (lines 645-678):
   - **Purpose**: Synchronous write with fsync
   - **Parameters**: `entry: WalEntry`
   - **Returns**: `StorageResult<u64>` (sequence number)
   - **Hook Point**: After write_sync returns, entry needs replication
   - **Key line**: Line 677 returns sequence number for tracking

2. **writer_loop()** (lines 681-750):
   - **Purpose**: Background batch writing
   - **Location**: Spawned in start() at line 584
   - **Hook Point**: Line 720 where batch is written - can trigger replication
   - **Timing**: Collects entries for max_batch_delay (default 10ms)

3. **write_async()** (lines 635-637):
   - **Purpose**: Queue entry for background write
   - **Hook Point**: Entry queued to entry_queue (SegQueue)

### WalReader for Fetching Entries

**Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/wal.rs:1061-1158`

Key method for replication shipper:

```rust
pub fn scan_all(&self) -> StorageResult<Vec<WalEntry>> {
    // Returns all WAL entries in sequence order
    // Line 1100-1110: Reads all wal-*.log files, sorts by filename (timestamp)
}

fn read_wal_file(&self, path: &Path) -> StorageResult<Vec<WalEntry>> {
    // Line 1115-1157: Reads single WAL file
    // Handles corruption with tracing (lines 1134-1138)
    // Returns entries in file order
}
```

---

## 2. STORAGE LAYER ARCHITECTURE

### Location: Storage Module
**Base path**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/`

### Directory Structure

```
storage/
├── mod.rs                     (Main storage module, error types, config)
├── wal.rs                     (Write-ahead log implementation)
├── persistence.rs             (Per-space persistence handle)
├── recovery.rs                (Crash recovery coordinator)
├── cache.rs                   (Memory-mapped cache optimization)
├── compact.rs                 (WAL compaction)
├── index.rs                   (Storage indexing)
├── confidence.rs              (Confidence-based tiering)
├── warm_tier.rs               (Warm tier storage)
├── cold_tier.rs               (Cold tier storage)
├── hot_tier.rs                (Hot tier storage)
├── deduplication.rs           (Semantic deduplication)
├── content_addressing.rs      (Content-based addressing)
├── numa.rs                    (NUMA-aware storage)
└── mapped.rs                  (Memory-mapped storage)
```

### Storage Write Paths

**Episodic Memory Storage Flow**:

1. **Entry Point**: `MemorySpacePersistence` (persistence.rs:44-97)
   - Line 45: `wal_writer: Arc<WalWriter>` - The WAL writer instance
   - Line 46: `storage_metrics: Arc<StorageMetrics>` - Metrics collector

2. **WAL Writer Access** (persistence.rs:105-107):
   ```rust
   pub fn wal_writer(&self) -> Arc<WalWriter> {
       Arc::clone(&self.wal_writer)
   }
   ```

3. **Memory Space ID** (types.rs:144-150):
   ```rust
   #[derive(Clone, Debug, Eq, PartialEq, Hash)]
   pub struct MemorySpaceId(Arc<str>);
   ```
   - Used as key for tenant isolation in replication

### Write Operations Tracking

**StorageMetrics** (mod.rs:335-416):

```rust
pub struct StorageMetrics {
    pub writes_total: AtomicU64,
    pub bytes_written: AtomicU64,
    pub fsync_count: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub page_faults: AtomicU64,
    pub compactions: AtomicU64,
}

// Recording methods:
pub fn record_write(&self, bytes: u64) {
    self.writes_total.fetch_add(1, Ordering::Relaxed);
    self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
}
```

**File location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/mod.rs:335-416`

### Persistence Configuration

**File location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/persistence.rs:16-38`

```rust
#[derive(Clone, Debug)]
pub struct PersistenceConfig {
    pub hot_capacity: usize,           // Hot tier max memories
    pub warm_capacity: usize,          // Warm tier max memories
    pub cold_capacity: usize,          // Cold tier max memories
    pub fsync_mode: FsyncMode,         // PerWrite, PerBatch, Timer, None
}

// Default configuration:
// - hot_capacity: 100,000
// - warm_capacity: 1,000,000
// - cold_capacity: 10,000,000
// - fsync_mode: FsyncMode::PerBatch
```

### Storage Error Handling

**File location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/mod.rs:241-329`

Key error types to extend for replication:

```rust
pub enum StorageError {
    Io(#[from] std::io::Error),
    CorruptionDetected(String),
    ChecksumMismatch { expected: u32, actual: u32 },
    WalFailed(String),
    Configuration(String),
    // ... others
}
```

---

## 3. gRPC AND ASYNC PATTERNS

### Proto Files Location

**Service Definition**: `/Users/jordan/Workspace/orchard9/engram/proto/engram/v1/service.proto`

**Memory Types**: `/Users/jordan/Workspace/orchard9/engram/proto/engram/v1/memory.proto`

### Existing gRPC Service Structure

**File**: `proto/engram/v1/service.proto:1-88`

Current service includes memory operations:
```proto
service EngramService {
  rpc Remember(RememberRequest) returns (RememberResponse);
  rpc Recall(RecallRequest) returns (RecallResponse);
  rpc Experience(ExperienceRequest) returns (ExperienceResponse);
  rpc Consolidate(ConsolidateRequest) returns (ConsolidateResponse);
  rpc Stream(StreamRequest) returns (stream StreamResponse);
  rpc StreamingRemember(stream RememberRequest) returns (stream RememberResponse);
  // ... more operations
}
```

### Streaming Infrastructure (Milestone 11)

**File**: `proto/engram/v1/service.proto:570-676`

Already has streaming patterns for replication foundation:

```proto
// Observation stream: continuous memory formation
message ObservationRequest {
  string memory_space_id = 1;
  oneof operation {
    StreamInit init = 2;
    Episode observation = 3;
    FlowControl flow = 4;
    StreamClose close = 5;
  }
  string session_id = 10;
  uint64 sequence_number = 11;
}

message ObservationResponse {
  oneof result {
    StreamInitAck init_ack = 1;
    ObservationAck ack = 2;
    StreamStatus status = 3;
  }
  string session_id = 10;
  uint64 sequence_number = 11;
}
```

**Key insight**: Proto already has session tracking (session_id, sequence_number) - replication can reuse this pattern!

### Async Runtime Patterns in Codebase

**Found in multiple locations**:
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/auth/jwt.rs` - tokio::spawn patterns
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/security/vault.rs` - async trait patterns
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/persistence.rs` - Arc<RwLock<T>> patterns

**Standard patterns used**:
```rust
use tokio::sync::RwLock;
use dashmap::DashMap;
use std::sync::Arc;

// Lock-free concurrent data structures:
let map: Arc<DashMap<K, V>> = Arc::new(DashMap::new());

// Async mutexes:
let state: Arc<RwLock<State>> = Arc::new(RwLock::new(state));
```

### Metrics Recording Integration

**File location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/metrics/mod.rs:1-150`

Standard metric names follow pattern: `engram_<subsystem>_<metric>_<unit>`

Examples:
```rust
const SPREADING_ACTIVATIONS_TOTAL: &str = "engram_spreading_activations_total";
const WAL_RECOVERY_SUCCESSES_TOTAL: &str = "engram_wal_recovery_successes_total";
const WAL_RECOVERY_DURATION_SECONDS: &str = "engram_wal_recovery_duration_seconds";
const WAL_COMPACTION_RUNS_TOTAL: &str = "engram_wal_compaction_runs_total";
const WAL_COMPACTION_BYTES_RECLAIMED: &str = "engram_wal_compaction_bytes_reclaimed";
```

---

## 4. METRICS INTEGRATION PATTERNS

### Metrics Module Structure

**Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/metrics/`

Files:
- `mod.rs` - Main metrics definitions and constants
- `cognitive.rs` - Cognitive metrics
- `health.rs` - Health check metrics
- `streaming.rs` - Stream-based metrics
- `prometheus.rs` - Prometheus export

### Naming Convention for Replication Metrics

Following Engram patterns, replication metrics should use:

```
engram_replication_<metric>_<unit>
```

Examples needed:
```
engram_replication_lag_ms              # Per-replica lag
engram_replication_lag_alert_total     # Lag alert count
engram_replication_batch_size_bytes    # Batch size
engram_replication_throughput_entries_per_sec
engram_replication_batches_sent_total
engram_replication_batches_failed_total
engram_replication_catchup_entries_replayed_total
engram_replication_connection_pool_active
engram_replication_replica_status      # Gauge: 0=failed, 1=degraded, 2=lagging, 3=healthy
```

### Label Support for Multi-Tenant

**File location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/metrics/mod.rs:131-135`

Already has multi-tenant label support:

```rust
pub fn with_space(space_id: &MemorySpaceId) -> Vec<(&'static str, String)> {
    vec![("memory_space", space_id.to_string())]
}
```

Replication metrics should include:
- `memory_space` - The space being replicated
- `replica_id` - Which replica
- `primary_id` - Primary node ID

---

## 5. ERROR HANDLING PATTERNS

### Storage Error Type

**Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/mod.rs:241-329`

```rust
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Corruption detected: {0}")]
    CorruptionDetected(String),
    
    #[error("Checksum verification failed: expected {expected:x}, got {actual:x}")]
    ChecksumMismatch { expected: u32, actual: u32 },
    
    #[error("WAL operation failed: {0}")]
    WalFailed(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
}

impl StorageError {
    pub fn wal_failed(msg: &str) -> Self {
        Self::WalFailed(msg.to_string())
    }
}

pub type StorageResult<T> = Result<T, StorageError>;
```

### Memory Space Error Pattern

**Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/registry/error.rs:11-63`

Follows cognitive error pattern with context:

```rust
#[derive(Debug, Error)]
pub enum MemorySpaceError {
    #[error(
        "Memory space '{id}' not found\n  Expected: Previously created memory space id\n  Suggestion: Create the space via registry.create_or_get before use\n  Example: registry.create_or_get(&MemorySpaceId::try_from(\"tenant_a\")?)"
    )]
    NotFound { id: MemorySpaceId },
    
    #[error(
        "Failed to prepare persistence directory '{path}' for space '{id}'\n  Expected: Writable filesystem path\n  Suggestion: Ensure Engram has permissions for the data root\n  Example: chmod +w {path}"
    )]
    Persistence {
        id: MemorySpaceId,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}
```

### Replication Error Structure (From Task File)

**File**: `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/005_replication_protocol_expansion.md:1109-1151`

```rust
#[derive(Debug, Error)]
pub enum ReplicationError {
    #[error("Unsupported replication protocol version: {0}")]
    UnsupportedVersion(u16),
    
    #[error("Empty replication batch")]
    EmptyBatch,
    
    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: u32, actual: u32 },
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Connection timeout")]
    ConnectionTimeout,
    
    #[error("Maximum retries exceeded: {0}")]
    MaxRetriesExceeded(usize),
}
```

---

## 6. EXISTING ASYNC PATTERNS AND BEST PRACTICES

### Tokio Integration Points

Found in codebase:

```rust
// From storage/persistence.rs and similar files
use tokio::sync::RwLock;
use std::sync::Arc;

// Pattern 1: Async spawn with Arc cloning
let handle = std::thread::Builder::new()
    .name("wal-writer".to_string())
    .spawn(move || {
        // Background work
    })?;

// Pattern 2: RwLock for shared state
let state = Arc::new(RwLock::new(initial_state));
let state_clone = state.clone();
tokio::spawn(async move {
    let read = state_clone.read().await;
    // Use read-only access
});

// Pattern 3: DashMap for lock-free concurrent maps
let map: Arc<DashMap<K, V>> = Arc::new(DashMap::new());
```

### Timeout Pattern

From task file (line 781-784):

```rust
let timeout = tokio::time::Duration::from_secs(5);
let results = tokio::time::timeout(
    timeout,
    futures::future::join_all(handles),
).await;
```

### Tracing Integration

Used throughout for logging:

```rust
use tracing::{debug, error, info, warn};

info!("Shipped batch of {} entries", batch_entries.len());
error!("WAL batch write failed: {}", e);
warn!("Some replicas failed: {:?}", failures);
```

---

## 7. CONCRETE INTEGRATION APPROACH

### Hook Points in WAL Writer

**Location 1**: After successful write in `write_sync()` (wal.rs:677)

```rust
pub fn write_sync(&self, mut entry: WalEntry) -> StorageResult<u64> {
    let sequence = self.sequence_counter.fetch_add(1, Ordering::SeqCst);
    // ... write to file ...
    self.metrics.record_write((HEADER_SIZE + entry.payload.len()) as u64);
    self.metrics.record_fsync();
    
    // UPDATE last_write_timestamp
    self.last_write_timestamp_ns
        .store(entry.header.timestamp, Ordering::Relaxed);
    
    // HOOK: Trigger replication here if registered
    // if let Some(repl) = &self.replication_shipper {
    //     repl.append_entry(entry).await;
    // }
    
    Ok(sequence)
}
```

**Location 2**: In writer_loop after batch write (wal.rs:720)

```rust
if let Err(e) = Self::write_batch(file, &batch, metrics, fsync_mode) {
    tracing::error!("WAL batch write failed: {}", e);
} else {
    // HOOK: Trigger replication with batch
    // replication_shipper.queue_batch(batch.entries).await;
}
```

### WAL Reader Integration

**Location**: For replication shipper to fetch entries

```rust
let reader = WalReader::new(wal_dir, metrics);
let entries = reader.scan_all()?;  // Gets all entries in order

// Replication shipper can filter by sequence:
let last_shipped = 100u64;
let new_entries: Vec<_> = entries
    .into_iter()
    .filter(|entry| entry.header.sequence > last_shipped)
    .collect();
```

### MemorySpacePersistence Integration

**Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/persistence.rs:59-97`

Already provides access to WAL writer:

```rust
impl MemorySpacePersistence {
    pub fn wal_writer(&self) -> Arc<WalWriter> {
        Arc::clone(&self.wal_writer)
    }
    
    pub fn storage_metrics(&self) -> Arc<StorageMetrics> {
        Arc::clone(&self.storage_metrics)
    }
}

// Replication coordinator can obtain these for a space:
// let persistence = registry.get_space(space_id)?;
// let wal_writer = persistence.wal_writer();
// let metrics = persistence.storage_metrics();
```

---

## 8. PROTO EXTENSION STRATEGY

### Replication Service Addition

Add to `proto/engram/v1/service.proto` after existing services:

```proto
// Replication Protocol Service (internal cluster communication)
// Not exposed to clients, used for primary-to-replica communication

service ReplicationService {
  // Ship WAL batch to replica
  rpc ShipBatch(ShipBatchRequest) returns (ShipBatchResponse);
  
  // Get replication status
  rpc GetStatus(GetStatusRequest) returns (GetStatusResponse);
  
  // Establish replication stream
  rpc ReplicationStream(stream ReplicationStreamRequest) 
    returns (stream ReplicationStreamResponse);
}
```

### Message Definitions

Add to `proto/engram/v1/memory.proto`:

```proto
// Replication batch for network transmission
message ReplicationBatch {
  bytes batch_header = 1;    // ReplicationBatchHeader (64 bytes)
  repeated bytes wal_entries = 2;  // Serialized WalEntry entries
}

message ShipBatchRequest {
  string replica_id = 1;
  ReplicationBatch batch = 2;
}

message ShipBatchResponse {
  uint64 applied_sequence = 1;    // Latest sequence applied by replica
  google.protobuf.Timestamp committed_at = 2;
}
```

---

## 9. FILES TO CREATE - EXACT LOCATIONS AND DEPENDENCIES

### Module Creation Order (dependency chain):

**Phase 1: Error Types and Data Structures**
```
engram-core/src/cluster/                          (NEW)
├── replication/
│   ├── mod.rs                                    (NEW)
│   └── error.rs                                  (NEW)
```

**Phase 2: State Management**
```
engram-core/src/cluster/
├── replication/
│   ├── state.rs                                  (NEW)
│   └── wal_format.rs                             (NEW)
```

**Phase 3: Core Replication Logic**
```
engram-core/src/cluster/
├── replication/
│   ├── connection_pool.rs                        (NEW)
│   ├── wal_shipper.rs                            (NEW)
│   ├── receiver.rs                               (NEW)
│   ├── lag_monitor.rs                            (NEW)
│   ├── catchup.rs                                (NEW)
│   └── promotion.rs                              (NEW)
```

**Phase 4: Integration**
```
engram-core/src/
├── cluster/
│   └── mod.rs                                    (NEW)
├── storage/
│   └── mod.rs                                    (MODIFY)
├── metrics/
│   └── mod.rs                                    (MODIFY)
└── lib.rs                                        (MODIFY)
```

### Detailed File Creation List

**1. engram-core/src/cluster/mod.rs** (NEW)
- Purpose: Module root, re-exports all replication types
- Size estimate: 50 lines
- Key items to export: ReplicationCoordinator, ReplicationState, WalShipper, all error types

**2. engram-core/src/cluster/replication/mod.rs** (NEW)
- Purpose: Replication submodule root
- Size estimate: 30 lines
- Re-export all: error, state, wal_format, wal_shipper, connection_pool, lag_monitor, receiver

**3. engram-core/src/cluster/replication/error.rs** (NEW)
- Purpose: Error types for replication
- Size estimate: 50 lines
- From task file lines 1109-1151

**4. engram-core/src/cluster/replication/wal_format.rs** (NEW)
- Purpose: ReplicationBatchHeader and ReplicationBatch
- Size estimate: 250 lines
- From task file lines 94-371

**5. engram-core/src/cluster/replication/state.rs** (NEW)
- Purpose: ReplicationState and ReplicationCoordinator
- Size estimate: 200 lines
- From task file lines 374-637

**6. engram-core/src/cluster/replication/connection_pool.rs** (NEW)
- Purpose: ReplicaConnectionPool and connection management
- Size estimate: 150 lines
- From task file lines 839-1041

**7. engram-core/src/cluster/replication/wal_shipper.rs** (NEW)
- Purpose: WalShipper - ships WAL batches to replicas
- Size estimate: 180 lines
- From task file lines 640-836

**8. engram-core/src/cluster/replication/lag_monitor.rs** (NEW)
- Purpose: LagMonitor - tracks replication lag
- Size estimate: 80 lines
- From task file lines 1043-1107

**9. engram-core/src/cluster/replication/receiver.rs** (NEW)
- Purpose: Replica-side WAL receiver
- Size estimate: 150 lines
- Counterpart to WalShipper, handles incoming batches

**10. engram-core/src/cluster/replication/catchup.rs** (NEW)
- Purpose: Catchup mechanism for lagged replicas
- Size estimate: 120 lines
- Implements logical playback from sequence gap

**11. engram-core/src/cluster/replication/promotion.rs** (NEW)
- Purpose: Replica promotion to primary on failure
- Size estimate: 100 lines
- Handles replica election and takeover

### Files to Modify

**1. engram-core/src/storage/wal.rs**
- Add optional replication shipper field to WalWriter
- Hook write_sync() to notify replication shipper
- Hook writer_loop() to batch-notify replication shipper
- Lines 464-492 (WalWriter struct): Add `replication_shipper` field
- Lines 645-678 (write_sync): Add hook after line 677

**2. engram-core/src/storage/mod.rs**
- Re-export cluster::replication types at top level
- Add replication metrics constants
- Estimated changes: 50 lines

**3. engram-core/src/metrics/mod.rs**
- Add replication metric constants (5-10 new const definitions)
- Add `with_replica()` label helper similar to `with_space()`
- Estimated changes: 30 lines

**4. engram-core/src/lib.rs**
- Add `pub mod cluster;` at module root
- Estimated changes: 2 lines

**5. engram-core/Cargo.toml**
- Add `socket2 = "0.5"` dependency for TCP keepalive
- `futures` already present
- `tokio` already present
- `dashmap` already present
- Estimated changes: 2 lines

---

## 10. TESTING INFRASTRUCTURE

### Unit Test Locations

From task file lines 1179-1241, tests should be in:
- `engram-core/src/cluster/replication/wal_format.rs` - inline tests
- `engram-core/src/cluster/replication/state.rs` - inline tests
- `engram-core/src/cluster/replication/connection_pool.rs` - inline tests

### Integration Test Location

**File**: `engram-core/tests/replication_integration.rs` (NEW)

Size estimate: 300+ lines
Contains: Primary-to-replica flow, lag recovery, promotion scenarios

---

## 11. IMPLEMENTATION CHECKLIST

### Phase 1: Foundation (Day 1)
- [ ] Create cluster module structure
- [ ] Implement ReplicationError type
- [ ] Implement ReplicationBatchHeader and ReplicationBatch
- [ ] Implement serialization/deserialization with CRC32C

### Phase 2: State Management (Day 1-2)
- [ ] Implement ReplicationState with DashMap
- [ ] Implement ReplicaProgress tracking
- [ ] Implement ReplicationCoordinator

### Phase 3: Async Communication (Day 2-3)
- [ ] Implement ReplicaConnectionPool with Tokio
- [ ] Implement connection lifecycle (connect, keepalive, cleanup)
- [ ] Implement WalShipper with batch batching
- [ ] Integrate WalShipper with WalWriter

### Phase 4: Monitoring & Recovery (Day 3-4)
- [ ] Implement LagMonitor with metrics integration
- [ ] Implement ReplicationReceiver (replica-side)
- [ ] Implement Catchup mechanism
- [ ] Implement Promotion logic
- [ ] Add replication metrics to metrics/mod.rs
- [ ] Modify WAL writer to hook replication

### Phase 5: Testing & Integration
- [ ] Unit tests for batch serialization
- [ ] Unit tests for state tracking
- [ ] Integration tests for primary-to-replica flow
- [ ] Integration test for lag recovery
- [ ] `make quality` - fix all clippy warnings
- [ ] Verify with engram_diagnostics.sh

---

## 12. DEPENDENCY GRAPH

```
ReplicationError (error.rs)
  ↓
ReplicationBatchHeader, ReplicationBatch (wal_format.rs)
  ↓
ReplicationState, ReplicaProgress (state.rs)
  ↓
ReplicaConnection, ReplicaConnectionPool (connection_pool.rs)
  ↓
WalShipper (wal_shipper.rs) + LagMonitor (lag_monitor.rs)
  ↓
ReplicationReceiver (receiver.rs) + Catchup (catchup.rs) + Promotion (promotion.rs)
  ↓
Integration with WalWriter, MemorySpacePersistence, StorageMetrics
```

---

## 13. CRITICAL INTEGRATION POINTS SUMMARY

| Component | Integration Point | File Location | Method/Field |
|-----------|------------------|------------------|---------|
| WAL Entry Creation | Post-write notification | wal.rs:645-678 | write_sync() return |
| WAL Batch Writing | Batch ready signal | wal.rs:681-750 | writer_loop after write_batch |
| Sequence Tracking | Progress tracking | wal.rs:471 | sequence_counter |
| Metrics Recording | Replication metrics | metrics/mod.rs | New constants + recording |
| Storage Config | Replication config | storage/mod.rs | StorageConfig extension |
| Memory Space Isolation | Space-aware replication | registry/ | MemorySpaceId propagation |
| Error Propagation | Replication failures | storage/mod.rs | StorageError extension |

---

## 14. PERFORMANCE CONSIDERATIONS

From existing codebase patterns:

1. **Lock-Free Structures**: Use `dashmap::DashMap` for concurrent access (no locks)
2. **Atomic Operations**: Use `AtomicU64` for counters with `Relaxed` ordering for hot path
3. **Arc Cloning**: Cheap operation, use liberally for ownership sharing
4. **RwLock Over Mutex**: For read-heavy (lag monitoring) vs write-heavy (state updates) decision
5. **Batch Operations**: Always batch network sends (see WalBatch pattern in wal.rs:405-450)

---

## Summary

This analysis provides exact file paths, line numbers, concrete code examples, and integration points needed to implement Task 005 efficiently within the existing Engram architecture. The implementation follows established patterns from the codebase (error handling, async patterns, metrics, storage integration) while adding specialized replication logic.

**Key files to study before implementing**:
1. `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/wal.rs` - WAL format and writer
2. `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/persistence.rs` - Storage integration
3. `/Users/jordan/Workspace/orchard9/engram/engram-core/src/metrics/mod.rs` - Metrics patterns
4. `/Users/jordan/Workspace/orchard9/engram/proto/engram/v1/service.proto` - gRPC patterns
5. `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/005_replication_protocol_expansion.md` - Complete specification

