# Task 005 Expansion Summary

## What Was Done

Expanded Task 005 (Replication Protocol for Episodic Memories) from a 30-line summary to a comprehensive 700+ line implementation specification following the same structure and depth as Tasks 001-003.

## Expansion Details

### Research Foundation (Lines 17-144)
- **WAL-Based Replication Patterns**: PostgreSQL streaming replication, FoundationDB zero-copy I/O, RocksDB Raft integration, MySQL binlog
- **Async vs Sync Tradeoffs**: Detailed analysis of why async replication fits episodic memory tier (1-10ms writes vs 20-100ms for sync)
- **Zero-Copy I/O with io_uring**: Linux 5.1+ optimization for <5ms P99 latency, with fallback strategies for macOS/Windows
- **Connection Pooling**: Apache Kafka pattern for pre-established connections with circuit breakers

### Technical Specification (Lines 146-550)

#### 1. WAL File Format for Replication
- `ReplicationBatchHeader` (64 bytes, cache-line aligned)
- `ReplicationBatch` with serialization/deserialization
- CRC32C checksums for corruption detection
- Compression hooks (zstd/lz4, deferred to optimization phase)

#### 2. Core Data Structures
- `ReplicationState`: Per-space replication tracking
- `ReplicaProgress`: Per-replica lag and health status
- `ReplicationCoordinator`: Global coordination across all spaces
- `ReplicaStatus`: Healthy/Lagging/Degraded/Failed states

#### 3. WAL Shipper
- Batching: 100 entries or 10ms delay (whichever comes first)
- Parallel shipping to all replicas
- Automatic retry with exponential backoff
- Integration with existing `engram-core/src/storage/wal.rs`

#### 4. Replica Connection Pool
- Pre-established TCP connections with keepalive
- Circuit breaker pattern (3 failures → mark degraded)
- Timeout handling: 5s connect, 10s read/write
- Exponential backoff: 100ms → 1s → 10s → 60s

#### 5. Lag Monitor
- Polls every 5s for lagging replicas
- Alert thresholds: 1s (warning), 5s (critical)
- Metrics integration for Prometheus/Grafana
- Automatic degradation marking

### Files to Create (10 new files)
1. `replication/mod.rs` - Module structure
2. `replication/wal_format.rs` - Batch format (316 lines)
3. `replication/state.rs` - State tracking (211 lines)
4. `replication/wal_shipper.rs` - Shipping logic (183 lines)
5. `replication/connection_pool.rs` - Connection management (182 lines)
6. `replication/lag_monitor.rs` - Monitoring (65 lines)
7. `replication/error.rs` - Error types (31 lines)
8. `replication/receiver.rs` - Replica-side receiver (TODO)
9. `replication/catchup.rs` - Catchup for lagged replicas (TODO)
10. `replication/promotion.rs` - Replica promotion logic (TODO)

### Files to Modify (6 existing files)
1. `cluster/mod.rs` - Export replication module
2. `storage/wal.rs` - Add replication hooks
3. `registry/memory_space.rs` - Track primary/replica assignments
4. `cluster.rs` - Start replication services
5. `metrics/mod.rs` - Add replication metrics
6. `Cargo.toml` - Add dependencies (socket2, futures)

### Testing Strategy (Lines 639-726)
- **Unit tests**: Batch serialization, state tracking, header validation
- **Integration tests**: Primary-to-replica replication, lag recovery, catchup
- **Property-based tests**: Deferred to Task 011 (Jepsen)

### Acceptance Criteria (Lines 741-751)
- Write latency <10ms (P99)
- Replication lag <1s under normal load
- Replica promotion <5s on primary failure
- Batch efficiency: 100 entries/batch typical
- Connection pool reuse: no handshake per batch
- Lag alerting: >1s warning, >5s critical
- Catchup: handles 10K entry gaps
- Checksums: detect all corruption in transit

### Performance Targets (Lines 753-761)
- P99 write latency: <10ms
- Replication throughput: 100K entries/sec per stream
- Network bandwidth: <100MB/sec per replica (future compression)
- Connection pool overhead: <1ms per batch
- Lag monitoring: <0.01% CPU
- Memory: <100MB per replica stream

## Integration Instructions

### Option 1: Replace Entire Task 005 Section
Replace lines 25-54 in `004-012_remaining_tasks_pending.md` with the contents of `005_replication_protocol_expansion.md`.

### Option 2: Create Standalone Task File
Move `005_replication_protocol_expansion.md` to:
```
roadmap/milestone-14/005_replication_protocol_pending.md
```

Then update `004-012_remaining_tasks_pending.md` to reference it:
```markdown
## Task 005: Replication Protocol for Episodic Memories (4 days)

See detailed specification: `005_replication_protocol_pending.md`
```

### Option 3: Keep Both (Recommended)
1. Keep the summary in `004-012_remaining_tasks_pending.md` for quick reference
2. Keep the expansion in `005_replication_protocol_expansion.md` for implementation
3. Add a link in the summary pointing to the expansion

## Key Design Decisions

### 1. Async Replication (Not Sync)
**Rationale**: Episodic memories are reconstructible from source data. 1-10ms write latency is critical for real-time ingestion. Async replication provides 10x better throughput with acceptable durability risk.

### 2. WAL Shipping (Not State Shipping)
**Rationale**: Existing WAL infrastructure already guarantees crash consistency. Shipping WAL entries reuses proven durability mechanism. Easier to implement than state-based replication.

### 3. Batching (100 entries or 10ms)
**Rationale**: Individual entry shipping wastes network bandwidth. Batching amortizes TCP overhead. 10ms delay prevents lag buildup. PostgreSQL uses similar batching.

### 4. Connection Pooling (Not Per-Request)
**Rationale**: TCP handshake adds 1-10ms RTT. Pre-established connections eliminate handshake overhead. Apache Kafka proved this pattern scales to 1000s of replicas.

### 5. Circuit Breaker (3 Failures → Degraded)
**Rationale**: Prevents cascading failures. Failing replicas don't block primary writes. Manual intervention required for recovery (runbook in Task 012).

## Biological Alignment

Replication mimics **memory consolidation from hippocampus to neocortex**:
- Hippocampus (episodic tier) = primary node with WAL
- Neocortex (semantic tier) = replicas with consolidated state
- Sleep replay = WAL shipping batches
- Gradual strengthening = catchup mechanism for lagged replicas
- Forgetting = WAL compaction after successful replication

Async replication reflects biological reality: memories can be lost if not consolidated quickly (primary failure before replication). Trade availability for perfect durability, just like biological systems.

## What Still Needs Design

The expansion includes TODOs for 3 additional components (to be designed in separate passes):

1. **Replica Receiver** (`receiver.rs`):
   - Listen for replication batches
   - Apply WAL entries to local storage
   - Send acknowledgments to primary
   - Handle duplicate/out-of-order batches

2. **Catchup Mechanism** (`catchup.rs`):
   - Detect severely lagged replicas (>10K entries behind)
   - Transfer snapshot + incremental WAL
   - Resume normal replication after catchup
   - Throttling to avoid overwhelming primary

3. **Replica Promotion** (`promotion.rs`):
   - Detect primary failure (via SWIM)
   - Elect new primary (highest sequence number)
   - Notify all replicas of new primary
   - Redirect writes to new primary

These will be expanded in a follow-up pass after Task 005 core implementation is reviewed.

## References

- PostgreSQL Streaming Replication: https://www.postgresql.org/docs/current/warm-standby.html
- FoundationDB Architecture: https://apple.github.io/foundationdb/architecture.html
- io_uring Benchmarks: https://kernel.dk/io_uring.pdf
- Apache Kafka Replication: https://kafka.apache.org/documentation/#replication
- RocksDB Replication (TiKV): https://tikv.org/deep-dive/scalability/replication/
