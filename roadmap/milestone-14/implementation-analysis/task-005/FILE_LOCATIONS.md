# Task 005 Implementation - Exact File Locations and Line Numbers

## Key Source Files (Read-Only - Study Only)

### WAL Implementation
- **File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/wal.rs`
  - WalEntryHeader struct: lines 29-49
  - WalEntryHeader magic constant: line 51 (0xDEAD_BEEF)
  - WalEntryType enum: lines 286-315
  - WalEntry struct: lines 318-324
  - WalEntry::new_episode(): lines 334-341
  - WalEntry::new_memory_update(): lines 350-357
  - WalEntry::new_memory_delete(): lines 365-371
  - WalEntry::new_checkpoint(): lines 379-385
  - WalEntry::validate(): lines 398-402
  - WalEntryHeader::as_bytes(): lines 175-199
  - WalEntryHeader::from_bytes(): lines 201-283
  - WalWriter struct: lines 464-492
  - WalWriter::write_sync(): lines 645-678
  - WalWriter::writer_loop(): lines 681-750
  - WalWriter::write_batch(): lines 758-802
  - WalReader::scan_all(): lines 1081-1112
  - WalReader::read_wal_file(): lines 1115-1157

### Storage Layer
- **File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/mod.rs`
  - StorageConfig struct: lines 143-176
  - StorageError enum: lines 241-329
  - StorageResult type alias: line 332
  - StorageMetrics struct: lines 335-416
  - StorageMetrics::record_write(): lines 365-370
  - StorageMetrics::record_fsync(): lines 381-384

- **File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/persistence.rs`
  - PersistenceConfig struct: lines 16-38
  - MemorySpacePersistence struct: lines 44-97
  - MemorySpacePersistence::wal_writer(): lines 105-107
  - MemorySpacePersistence::storage_metrics(): lines 110-112
  - PersistenceError enum: lines 189-222

- **File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/recovery.rs`
  - CrashRecoveryCoordinator struct: lines 8-15
  - Recovery/validation API pattern

### Types and Identifiers
- **File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/types.rs`
  - MemorySpaceId struct: lines 144-150
  - MemorySpaceIdError enum: lines 112-139

### Metrics
- **File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/metrics/mod.rs`
  - Metric name constants: lines 42-106
  - with_space() function: lines 131-135
  - encode_metric_name_with_labels(): lines 149-150 (continuation)

### Error Handling
- **File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/registry/error.rs`
  - MemorySpaceError enum: lines 11-63
  - Cognitive error pattern examples

### Proto Definitions
- **File**: `/Users/jordan/Workspace/orchard9/engram/proto/engram/v1/service.proto`
  - EngramService definition: lines 15-88
  - Streaming patterns (Milestone 11): lines 570-676
  - ObservationRequest message: lines 575-587
  - ObservationResponse message: lines 589-599

- **File**: `/Users/jordan/Workspace/orchard9/engram/proto/engram/v1/memory.proto`
  - Confidence message: lines 15-34
  - Memory message: lines 42-59
  - Episode message: lines 72-94
  - Cue message: lines 111-129

### Task Specification
- **File**: `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/005_replication_protocol_expansion.md`
  - ReplicationBatchHeader spec: lines 100-154
  - ReplicationBatch spec: lines 256-285
  - ReplicationState spec: lines 386-506
  - ReplicationCoordinator spec: lines 545-637
  - WalShipper spec: lines 652-836
  - ReplicaConnectionPool spec: lines 851-1041
  - LagMonitor spec: lines 1054-1106
  - ReplicationError spec: lines 1116-1150
  - Unit tests spec: lines 1179-1241
  - Integration tests spec: lines 1247-1310

---

## Files to Create (New Implementation)

### Module Root
- **Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/cluster/`
- **Create**: `mod.rs` (Module exports)
- **Create**: `replication/mod.rs` (Replication submodule root)

### Replication Core
- **Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/cluster/replication/`
- **Create**: `error.rs` (Error types, ~50 lines)
- **Create**: `wal_format.rs` (Batch headers/serialization, ~250 lines)
- **Create**: `state.rs` (State tracking, ~200 lines)
- **Create**: `connection_pool.rs` (Connection management, ~150 lines)
- **Create**: `wal_shipper.rs` (WAL batch shipping, ~180 lines)
- **Create**: `lag_monitor.rs` (Lag monitoring, ~80 lines)
- **Create**: `receiver.rs` (Replica-side receiver, ~150 lines)
- **Create**: `catchup.rs` (Catchup mechanism, ~120 lines)
- **Create**: `promotion.rs` (Replica promotion, ~100 lines)

### Tests
- **Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/tests/`
- **Create**: `replication_integration.rs` (Integration tests, ~300 lines)

---

## Files to Modify (Existing Implementation)

### WAL Writer Integration
- **File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/wal.rs`
- **Modify**: WalWriter struct (lines 464-492)
  - Add optional field: `replication_shipper: Option<Arc<WalShipper>>`
- **Modify**: write_sync() method (lines 645-678)
  - Add hook after line 677 (after returning sequence)
- **Modify**: writer_loop() method (lines 681-750)
  - Add hook after successful batch write (around line 720)

### Storage Module
- **File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/storage/mod.rs`
- **Modify**: Add pub use statements for cluster::replication types
- **Modify**: Add replication metric constants (similar to WAL_RECOVERY_SUCCESSES_TOTAL pattern)
- **Modify**: StorageError enum - consider adding ReplicationError variant

### Metrics Module
- **File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/metrics/mod.rs`
- **Modify**: Add replication metric constants (lines after WAL_COMPACTION_BYTES_RECLAIMED)
  - engram_replication_lag_ms
  - engram_replication_batches_sent_total
  - engram_replication_batches_failed_total
  - engram_replication_replica_status
  - engram_replication_throughput_entries_per_sec
- **Modify**: Add with_replica() helper function (after with_space())

### Library Root
- **File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/lib.rs`
- **Modify**: Add `pub mod cluster;` at module level

### Dependencies
- **File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/Cargo.toml`
- **Modify**: Add dependency: `socket2 = "0.5"`
- **Note**: Following dependencies already present:
  - tokio (async runtime)
  - dashmap (lock-free maps)
  - bincode (serialization)
  - crc32c (checksums)
  - uuid (node IDs)
  - thiserror (error types)
  - tracing (logging)

---

## Integration Test Patterns

### Location
- `/Users/jordan/Workspace/orchard9/engram/engram-core/tests/replication_integration.rs`

### Test Scenarios from Spec
1. Primary to replica replication (lines 1250-1275 of task file)
2. Replication lag recovery (lines 1278-1309 of task file)

### Unit Test Locations (Inline)
- Batch serialization: `cluster/replication/wal_format.rs`
- State tracking: `cluster/replication/state.rs`
- Connection pool: `cluster/replication/connection_pool.rs`

---

## Implementation Dependency Chain

```
1. engram-core/src/cluster/replication/error.rs
   |
   v
2. engram-core/src/cluster/replication/wal_format.rs
   |
   v
3. engram-core/src/cluster/replication/state.rs
   |
   v
4. engram-core/src/cluster/replication/connection_pool.rs
   |
   +---> engram-core/src/cluster/replication/wal_shipper.rs
   |
   +---> engram-core/src/cluster/replication/lag_monitor.rs
   |
   v
5. engram-core/src/cluster/replication/receiver.rs
   
6. engram-core/src/cluster/replication/catchup.rs
   
7. engram-core/src/cluster/replication/promotion.rs
   |
   v
8. Modify: engram-core/src/storage/wal.rs (hook integration)
   
9. Modify: engram-core/src/metrics/mod.rs (metric definitions)
```

---

## Key Constants and Values

### WAL Constants
- HEADER_SIZE: 64 bytes (cache line aligned)
- WAL_MAGIC: 0xDEAD_BEEF
- Max batch size: 1000 entries
- Max batch delay: 10ms
- Default fsync mode: FsyncMode::PerBatch

### Storage Defaults
- Hot tier capacity: 100,000 memories
- Warm tier capacity: 1,000,000 memories
- Cold tier capacity: 10,000,000 memories
- Compaction threshold: 0.7

### Replication Defaults (from spec)
- Replica count: 2
- Batch size: 100 entries
- Batch delay: 10ms
- Lag alert threshold: 1000ms
- Lag degraded threshold: 5000ms
- Connection timeout: 5s
- Read/Write timeout: 10s
- Keepalive interval: 10s
- Max retries: 3

### Replica Status Values
- Healthy: Caught up (lag = 0)
- Lagging: lag < 1000ms
- Degraded: lag >= 1000ms
- Failed: 3+ consecutive failures

---

## Quick Reference

### To Hook WAL Write
```rust
// In WalWriter::write_sync(), after line 677:
if let Some(shipper) = &self.replication_shipper {
    shipper.append_entry(entry).await;
}
```

### To Record Metrics
```rust
// Pattern from existing code:
let labels = vec![
    ("memory_space", space_id.to_string()),
    ("replica_id", replica_id.clone()),
];
metrics::gauge("engram_replication_lag_ms", lag_ms, labels);
```

### To Create Replication Error
```rust
// Use thiserror pattern:
#[error("message")]
ErrorVariant { field: Type },
```

### Testing
```bash
# Run tests
cargo test --test replication_integration

# Check quality
make quality

# Check diagnostics
./scripts/engram_diagnostics.sh
```

