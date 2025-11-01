# Task 005: Replication Protocol Implementation Guide

This directory contains comprehensive analysis and implementation guidance for Task 005 (Replication Protocol for Episodic Memories) in Engram Milestone 14.

## Deliverables

### 1. TASK_005_IMPLEMENTATION_ANALYSIS.md (28KB, 913 lines)
**Primary comprehensive guide** containing:
- WAL implementation analysis (exact line numbers, code structure)
- Storage layer architecture (write paths, persistence config, error handling)
- gRPC and async patterns (proto files, streaming infrastructure, async runtime)
- Metrics integration patterns (naming conventions, multi-tenant support)
- Error handling patterns (StorageError, ReplicationError, cognitive error patterns)
- Async patterns in use (tokio, Arc, DashMap, RwLock, timeouts, tracing)
- Concrete integration approach (WAL writer hooks, WalReader usage)
- Proto extension strategy (ReplicationService messages)
- Files to create (11 new files with sizes and line counts)
- Files to modify (5 existing files with exact locations)
- Testing infrastructure (unit test patterns, integration test locations)
- Implementation checklist (5-phase approach, day-by-day breakdown)
- Dependency graph (module creation order)
- Integration points table (component to code mapping)
- Performance considerations (lock-free patterns, atomic operations)

### 2. TASK_005_FILE_LOCATIONS.md (9KB, 280 lines)
**Quick reference guide** with:
- Key source files organized by module (read-only study references)
  - WAL implementation (17 specific line ranges)
  - Storage layer (7 specific line ranges)
  - Types and identifiers (2 specific line ranges)
  - Metrics module (3 specific line ranges)
  - Error handling (1 specific line range)
  - Proto definitions (2 files with ranges)
- Task specification file cross-references (lines 100-1310)
- Files to create (exact paths and locations for 11 files)
- Files to modify (exact paths and 5 target files)
- Integration test patterns
- Implementation dependency chain (visual graph)
- Key constants and values (WAL, storage, replication defaults)
- Quick reference (code patterns for WAL hooks, metrics, errors, testing)

## How to Use These Documents

### For Initial Understanding
1. Read: **TASK_005_IMPLEMENTATION_ANALYSIS.md** sections 1-3
   - Understand WAL format (64-byte header, sequence tracking)
   - Understand storage integration points
   - Understand existing proto/async patterns

### For Architecture Decisions
2. Read: **TASK_005_IMPLEMENTATION_ANALYSIS.md** sections 4-6
   - Metrics naming and patterns
   - Error handling patterns
   - Async runtime patterns in use

### For Integration Planning
3. Read: **TASK_005_IMPLEMENTATION_ANALYSIS.md** sections 7-10
   - Concrete integration approach
   - Proto extensions needed
   - File creation and modification plan
   - Testing infrastructure

### For Implementation
4. Reference: **TASK_005_FILE_LOCATIONS.md**
   - Know exactly which files to create and where
   - Know exactly which lines to hook in existing files
   - Follow dependency chain for implementation order
   - Use quick reference for code patterns

### For Detailed Specifications
5. Original source: `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/005_replication_protocol_expansion.md`
   - Lines 100-154: ReplicationBatchHeader specification
   - Lines 256-285: ReplicationBatch specification
   - Lines 386-506: ReplicationState specification
   - Lines 545-637: ReplicationCoordinator specification
   - Lines 652-836: WalShipper specification
   - Lines 851-1041: ReplicaConnectionPool specification
   - Lines 1054-1106: LagMonitor specification
   - Lines 1116-1150: ReplicationError specification
   - Lines 1179-1241: Unit tests specification
   - Lines 1247-1310: Integration tests specification

## Key Findings Summary

### WAL is Already Ready for Replication
- 64-byte cache-line-aligned header with CRC32C checksums
- Monotonic sequence numbers for tracking
- Multiple entry types (Episode, Memory update, deletion, consolidation, checkpoint)
- Serialization already supports little-endian deserialization
- WalReader can scan all entries in sequence order

### Storage Integration is Clean
- MemorySpacePersistence provides direct access to WalWriter and StorageMetrics
- Storage metrics use atomic counters (lock-free, no overhead)
- Error handling follows cognitive error patterns (context + suggestion + example)
- Memory space ID provides tenant isolation key

### Proto Infrastructure Supports Replication
- Streaming patterns already exist (ObservationRequest/Response from Milestone 11)
- Session tracking (session_id, sequence_number) can be reused
- Multi-tenant support already in place

### Metrics System Ready
- Pattern: engram_<subsystem>_<metric>_<unit>
- Multi-tenant labels already supported (with_space helper)
- Atomic recording methods for lock-free operation

### Async Patterns Established
- Tokio with RwLock for async-compatible shared state
- DashMap for lock-free concurrent maps
- Arc cloning for cheap ownership sharing
- Tracing for structured logging
- Timeout patterns with tokio::time::timeout

## Implementation Path

### Phase 1: Foundation (Day 1)
- Create cluster module structure
- Implement ReplicationError type
- Implement ReplicationBatchHeader and ReplicationBatch
- Implement serialization/deserialization with CRC32C

### Phase 2: State Management (Day 1-2)
- Implement ReplicationState with DashMap
- Implement ReplicaProgress tracking
- Implement ReplicationCoordinator

### Phase 3: Async Communication (Day 2-3)
- Implement ReplicaConnectionPool with Tokio
- Implement connection lifecycle
- Implement WalShipper with batching
- Integrate with WalWriter

### Phase 4: Monitoring & Recovery (Day 3-4)
- Implement LagMonitor with metrics
- Implement ReplicationReceiver
- Implement Catchup mechanism
- Implement Promotion logic

### Phase 5: Testing & Integration
- Unit tests for serialization and state
- Integration tests for primary-to-replica flow
- `make quality` - fix clippy warnings
- `./scripts/engram_diagnostics.sh` - verify integration

## Files You'll Create
1. engram-core/src/cluster/mod.rs (50 lines)
2. engram-core/src/cluster/replication/mod.rs (30 lines)
3. engram-core/src/cluster/replication/error.rs (50 lines)
4. engram-core/src/cluster/replication/wal_format.rs (250 lines)
5. engram-core/src/cluster/replication/state.rs (200 lines)
6. engram-core/src/cluster/replication/connection_pool.rs (150 lines)
7. engram-core/src/cluster/replication/wal_shipper.rs (180 lines)
8. engram-core/src/cluster/replication/lag_monitor.rs (80 lines)
9. engram-core/src/cluster/replication/receiver.rs (150 lines)
10. engram-core/src/cluster/replication/catchup.rs (120 lines)
11. engram-core/src/cluster/replication/promotion.rs (100 lines)
12. engram-core/tests/replication_integration.rs (300+ lines)

**Total new code**: ~1,500+ lines (mostly from task specification)

## Files You'll Modify
1. engram-core/src/storage/wal.rs - Add replication shipper hook
2. engram-core/src/storage/mod.rs - Re-export types, add metrics
3. engram-core/src/metrics/mod.rs - Add replication metrics
4. engram-core/src/lib.rs - Add pub mod cluster;
5. engram-core/Cargo.toml - Add socket2 dependency

**Total modifications**: ~100 lines across 5 files

## Dependencies

### New Dependency
- socket2 = "0.5" (for TCP keepalive configuration)

### Already Present
- tokio (async runtime)
- dashmap (lock-free maps)
- bincode (serialization)
- crc32c (checksums)
- uuid (node IDs)
- thiserror (error types)
- tracing (logging)
- futures (async utilities)

## Key Integration Points

| Component | Location | Method | Line |
|-----------|----------|--------|------|
| WAL sequence tracking | wal.rs | write_sync() | 645-678 |
| Batch write completion | wal.rs | writer_loop() | 681-750 |
| WAL entry fetching | wal.rs | WalReader::scan_all() | 1081-1112 |
| Metrics recording | mod.rs | StorageMetrics | 335-416 |
| Space isolation | types.rs | MemorySpaceId | 144-150 |
| Error propagation | mod.rs | StorageError | 241-329 |

## Performance Targets

- Write latency P99: <10ms (async, no network wait)
- Replication throughput: 100K entries/sec per replica
- Network bandwidth: <100MB/sec per replica
- Connection pool overhead: <1ms per batch
- Lag monitoring overhead: <0.01% CPU
- Memory overhead: <100MB per replica stream

## Testing Requirements

### Unit Tests (inline in modules)
- Batch serialization round-trip
- State tracking and progress updates
- Connection pool lifecycle

### Integration Tests
- Primary-to-replica full flow
- Lag recovery with delayed replica
- Replica promotion on failure

### Quality Gates
- `make quality` - all clippy warnings must be fixed
- `./scripts/engram_diagnostics.sh` - diagnostics check
- Integration tests pass
- 1+ second test runtime acceptable for integration tests

## Next Steps

1. Open TASK_005_IMPLEMENTATION_ANALYSIS.md and read sections 1-3
2. Study the key files listed in TASK_005_FILE_LOCATIONS.md
3. Reference the original task file for specification details
4. Follow the implementation dependency chain
5. Create modules in order (error -> wal_format -> state -> connection_pool -> ...)
6. Add metrics constants to metrics/mod.rs
7. Add hooks to storage/wal.rs write_sync() and writer_loop()
8. Write unit tests inline in each module
9. Write integration tests in tests/replication_integration.rs
10. Run `make quality` and fix all clippy warnings
11. Run `./scripts/engram_diagnostics.sh` to verify

## Document Structure

```
README_TASK_005.md (this file)
├── TASK_005_IMPLEMENTATION_ANALYSIS.md (detailed guide)
├── TASK_005_FILE_LOCATIONS.md (quick reference)
└── Original: roadmap/milestone-14/005_replication_protocol_expansion.md
```

All three documents are complementary and should be used together for complete understanding.

