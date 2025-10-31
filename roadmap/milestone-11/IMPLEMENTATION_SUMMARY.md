# Milestone 11 Tasks 005-007: Executive Summary

**Architect:** Margo Seltzer
**Date:** 2025-10-30
**Analysis Status:** Complete

## TL;DR

**Status:** 60% infrastructure complete, 40% implementation needed
**Time Estimate:** 8 days (3+2+3) for single engineer
**Critical Blocker:** Task 003 (Worker Pool) must complete first

## What's Already Done

### Excellent Foundation (90-100% complete):
1. **Protocol Definition** - protobuf messages fully specified
2. **Session Management** - lock-free session tracking with monotonic sequences
3. **Observation Queue** - lock-free 3-lane priority queue with backpressure detection
4. **Queue Metrics** - comprehensive observability already built

### Files Ready to Use:
- `/Users/jordan/Workspace/orchard9/engram/proto/engram/v1/service.proto` (lines 78-677)
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/session.rs` (530 lines)
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/observation_queue.rs` (930 lines)

## What Needs Building

### Task 005: Bidirectional gRPC Streaming (3 days)

**Gap:** No gRPC handlers - protocol defined but not wired up

**Create:**
- `engram-cli/src/handlers/streaming.rs` (~400 lines)
  - `handle_observe_stream()` - client → server streaming
  - `handle_recall_stream()` - server → client streaming
  - `handle_memory_stream()` - bidirectional

**Modify:**
- `engram-cli/src/grpc.rs` - Add streaming method impls to `MemoryService`

**Key Challenge:** Mapping protobuf enums to Rust types, async stream lifecycle management

**Tests:**
- Integration test: stream 1K observations, verify acks
- Sequence validation: send gaps/duplicates, verify rejection
- Flow control: pause/resume, verify correctness

### Task 006: Backpressure & Admission Control (2 days)

**Gap:** Backpressure detection exists but not proactively emitted to clients

**Create:**
- `engram-core/src/streaming/backpressure.rs` (~300 lines)
  - `BackpressureMonitor` - periodic queue depth checker
  - `BackpressureState` enum (Normal/Warning/Critical/Overloaded)
  - Adaptive batch sizing based on pressure

**Modify:**
- Streaming handlers to subscribe to backpressure state changes
- Worker pool (Task 003) to use adaptive batching

**Key Challenge:** Calculating accurate retry-after based on drain rate

**Tests:**
- Fill queue to 85%, verify StreamStatus::BACKPRESSURE sent
- Admission control: verify RESOURCE_EXHAUSTED when queue full
- Monitor overhead: verify < 0.1% CPU at 100K ops/sec

### Task 007: Incremental Recall with Snapshot Isolation (3 days)

**Gap:** Recall works but not snapshot-aware or streaming

**Create:**
- `engram-core/src/streaming/recall.rs` (~400 lines)
  - `IncrementalRecallStream` - yields batches of results
  - `SnapshotRecallConfig` - captures committed generation
  - Visibility filter for HNSW search

**Modify:**
- `engram-core/src/index/hnsw_node.rs` - Add `generation: u64` field
- `engram-core/src/index/hnsw_search.rs` - Add `search_with_filter()`
- `engram-core/src/streaming/observation_queue.rs` - Track current_generation

**Key Challenge:** Maintaining snapshot consistency without blocking concurrent insertions

**Tests:**
- Snapshot isolation: only committed observations visible
- Incremental streaming: first result < 10ms, full results < 100ms
- Visibility latency: P99 < 100ms (observation → recall)

## Critical Dependencies

```
Task 003 (Worker Pool) BLOCKS everything
    │
    ├──► Task 005 (gRPC Streaming) - needs queue consumer
    │         │
    │         ├──► Task 006 (Backpressure) - needs handlers
    │         └──► Task 007 (Recall) - needs generation tracking
    │
    └──► All tasks feed into Task 010 (Performance Benchmarking)
```

**Task 003 Status:** PENDING - this is the critical path blocker

## Performance Architecture

### Bottleneck Analysis:

| Component | Current Throughput | Target | Gap |
|-----------|-------------------|--------|-----|
| Single HNSW insert | ~10K/sec | 100K/sec | Need 10x |
| Queue enqueue | 4M/sec | 100K/sec | ✓ Sufficient |
| Session validation | 1M/sec | 100K/sec | ✓ Sufficient |

**Solution:** Task 003 (Parallel Worker Pool) provides 10x via 8-10 workers

### Memory Footprint:

| Component | Per-Item Cost | At 100K items | At 1M items |
|-----------|--------------|---------------|-------------|
| QueuedObservation | 64 bytes | 6.4 MB | 64 MB |
| Session tracking | 128 bytes/session | 128 MB (1K sessions) | N/A |
| HNSW node overhead | 96 bytes | 9.6 MB | 96 MB |

**Total at 100K obs/sec sustained:** ~150 MB RAM (acceptable)

## Lock-Free Guarantees

All streaming components maintain lock-freedom:

1. **ObservationQueue:** `SegQueue` (Michael & Scott algorithm)
2. **SessionManager:** `DashMap` (lock-free concurrent hashmap)
3. **Backpressure Monitor:** Read-only atomic loads
4. **HNSW Worker Pool:** Per-worker queues, no cross-worker coordination

**Performance win:** 92x throughput vs mutex-based queue (4.8M vs 52K ops/sec)

## Temporal Ordering Model

**Guarantees:**
- ✓ Intra-stream: Total order via sequence numbers
- ✓ Cross-stream: Undefined order (biological memory model)
- ✓ Snapshot isolation: Eventual consistency with bounded staleness (P99 < 100ms)

**Non-guarantees:**
- ✗ Linearizability (not needed for cognitive memory)
- ✗ Cross-stream causal consistency (too expensive)

## Risk Assessment

### High Risk (Probability × Impact):

1. **Lock contention in HNSW insert** (HIGH × CRITICAL)
   - Mitigation: Per-space sharding in Task 003
   - Validation: Microbenchmark before implementing Task 005

2. **Visibility latency > 100ms P99** (MEDIUM × HIGH)
   - Mitigation: Fast-path for recent observations in Task 007
   - Validation: Measure in Task 007 tests

### Medium Risk:

3. **Backpressure causes client disconnects** (MEDIUM × MEDIUM)
   - Mitigation: Graceful degradation, not hard disconnect
   - Validation: Chaos testing in Task 009

### Low Risk:

4. **gRPC stream overhead** (LOW × LOW)
   - Mitigation: Batch responses, reuse buffers
   - Measurement: Baseline in Task 010

## Implementation Strategy

### Week 1: Foundation
- **Day 1-3:** Complete Task 003 (Worker Pool) - CRITICAL PATH
- **Day 4-5:** Prototype Task 005 (streaming handlers)

### Week 2: Integration
- **Day 6-8:** Complete Task 005 (gRPC Streaming)
- **Day 9-10:** Complete Task 006 (Backpressure)

### Week 3: Advanced Features
- **Day 11-13:** Complete Task 007 (Incremental Recall)
- **Day 14:** Integration testing across all three tasks

### Parallelization Opportunity:
- **Engineer A:** Tasks 003 → 005 (infrastructure focus)
- **Engineer B:** Tasks 004 → 007 (query focus)
- **Convergence:** Task 006 (both engineers), Tasks 009-010 (validation)

**Time savings:** 18 days → 12 days with 2 engineers

## Key Metrics to Track

### During Development:
- [ ] Queue depth P50/P99/P99.9
- [ ] Backpressure activation frequency
- [ ] Observation → ack latency distribution
- [ ] Visibility latency (observation → recall)
- [ ] Worker utilization (idle vs busy)

### Production Readiness:
- [ ] 100K observations/sec sustained for 60s
- [ ] 10 concurrent recalls/sec with < 20ms P99
- [ ] Zero data loss in 10-minute chaos test
- [ ] Memory usage < 2GB for 1M observations
- [ ] CPU usage < 80% under sustained load

## Code Quality Standards

### Must Pass Before Task Complete:
- [ ] `make quality` - zero clippy warnings
- [ ] All unit tests passing
- [ ] Integration tests demonstrating acceptance criteria
- [ ] Performance benchmarks meeting targets
- [ ] Documentation updated (inline + README)

### Rust Edition 2024 Compliance:
- Use `if let` chains for nested conditions
- Prefer `?` operator over explicit match on Result
- Use `#[must_use]` on all getters and constructors
- Avoid `unwrap()` in production code (tests OK)

## Questions for Implementer

Before starting Task 005, answer:

1. **Session timeout:** 5 minutes sufficient? (Matches biological working memory)
2. **Batch size:** Start with 10/100/1000? (Tune in Task 010)
3. **Backpressure threshold:** 80% correct? (Could be 70% or 90%)
4. **Incremental batch size:** 10 results per batch? (Could be 50 or 100)

Before starting Task 007, validate:

5. **Visibility latency:** Can we achieve < 100ms P99 with current HNSW?
   - Run microbenchmark: insert rate vs search latency
6. **Generation tracking overhead:** Is 8 bytes per node acceptable?
   - Calculate: 1M nodes × 8 bytes = 8MB (acceptable)

## References

**Codebase Files:**
- Protocol: `/Users/jordan/Workspace/orchard9/engram/proto/engram/v1/service.proto`
- Session: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/session.rs`
- Queue: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/observation_queue.rs`
- gRPC base: `/Users/jordan/Workspace/orchard9/engram/engram-cli/src/grpc.rs`
- HNSW search: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/index/hnsw_search.rs`

**Documentation:**
- Milestone 11 README: `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-11/README.md`
- Detailed guidance: `TASK_005_006_007_IMPLEMENTATION_GUIDANCE.md`

**Research Papers:**
- Lock-free queues: Michael & Scott (1996)
- Snapshot isolation: Berenson et al. (1995)
- HNSW: Malkov & Yashunin (2018)

---

**Ready to implement.** Start with Task 003 validation, then proceed to Task 005.
