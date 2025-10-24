# Milestone 11: Streaming Interface - Implementation Specification

## Executive Summary

Build bidirectional streaming for continuous memory observation and real-time recall with lock-free incremental indexing. Target: sustained 100K observations/second with concurrent recalls, < 10ms P99 latency, and chaos-tested correctness.

**Duration:** 18 days with 2 engineers (31 days single engineer)

**Critical Path:** Protocol (3d) → Queue (2d) → Workers (4d) → Batch HNSW (3d) → gRPC (3d) → Backpressure (2d) → Snapshot Recall (3d)

## Consistency Model: Eventual with Bounded Staleness

NOT linearizable. NOT strong consistency. **Eventual consistency with probabilistic temporal bounds.**

### Invariants That Must Hold

1. **Intra-stream total ordering:** All observations within a single stream session are totally ordered by sequence number
2. **Atomic visibility:** Index updates are atomic at item granularity (no partial observations visible)
3. **Bounded staleness:** Observation visible in recall within 100ms P99
4. **No silent drops:** System either accepts observation or returns error (never silently discards)
5. **Backpressure preserves order:** FIFO ordering maintained even under load

### What We Sacrifice

- **Cross-stream ordering undefined:** Observations from different streams may interleave probabilistically
- **No linearizability:** Concurrent recalls may see different snapshots
- **Eventual index visibility:** Recent observations may not be immediately visible (bounded by 100ms P99)

**Rationale:** Biological memory formation is probabilistic. Human episodic memory does not provide linearizable guarantees. We model cognitive reality, not transactional databases.

## Architecture Overview

```
┌─────────────┐
│   Client    │
│ (gRPC/WS)   │
└──────┬──────┘
       │
       │ ObservationRequest (sequence number, episode)
       ▼
┌─────────────────────────────────────────┐
│     Streaming Service (Task 001/005)    │
│  • Session management                   │
│  • Sequence validation                  │
│  • Flow control                         │
└──────┬──────────────────────────────────┘
       │
       │ Enqueue (lock-free)
       ▼
┌─────────────────────────────────────────┐
│   Observation Queue (Task 002)          │
│  • Priority lanes (High/Normal/Low)     │
│  • SegQueue (unbounded)                 │
│  • Backpressure detection               │
└──────┬──────────────────────────────────┘
       │
       │ Dequeue batch (100-500 items)
       ▼
┌─────────────────────────────────────────┐
│   Worker Pool (Task 003)                │
│  • 4-8 parallel workers                 │
│  • Space-based sharding                 │
│  • Work stealing                        │
└──────┬──────────────────────────────────┘
       │
       │ Batch insert
       ▼
┌─────────────────────────────────────────┐
│   HNSW Index (Task 004)                 │
│  • Batch-aware insertion                │
│  • Lock-free concurrent updates         │
│  • Snapshot isolation for reads         │
└─────────────────────────────────────────┘
       │
       │ Search (snapshot-isolated)
       ▼
┌─────────────────────────────────────────┐
│   Recall Stream (Task 007)              │
│  • Incremental result streaming         │
│  • Confidence scoring for recent obs    │
└─────────────────────────────────────────┘
```

## Task Breakdown

### Critical Path (18 days)

1. **Protocol Foundation (3d):** Define protobuf messages, session management, sequence numbers
2. **Lock-Free Queue (2d):** Replace ArrayQueue with SegQueue, priority lanes, backpressure detection
3. **Worker Pool (4d):** Multi-threaded HNSW workers with space sharding and work stealing
4. **Batch HNSW (3d):** Optimize HNSW for batch insertions (3-5x speedup)
5. **gRPC Streaming (3d):** Bidirectional stream handlers with flow control
6. **Backpressure (2d):** Adaptive admission control and batching under load
7. **Snapshot Recall (3d):** Recall with snapshot isolation and incremental streaming

### Parallel Tracks

8. **WebSocket (2d):** Browser-friendly streaming (parallel with Task 9)
9. **Chaos Testing (3d):** Fault injection, eventual consistency validation (parallel with Task 8)
10. **Performance Tuning (2d):** Benchmark and optimize for 100K ops/sec
11. **Monitoring (2d):** Prometheus metrics, Grafana dashboards, alerting
12. **Integration (2d):** End-to-end tests, client examples, operational runbook

## Technical Decisions

### 1. Why SegQueue Instead of ArrayQueue?

**Problem:** ArrayQueue has fixed capacity, blocks when full.

**Solution:** SegQueue is unbounded, lock-free push/pop.

**Trade-off:** Unbounded growth risk mitigated by admission control (soft capacity limits trigger backpressure).

### 2. Why Space-Based Sharding for Workers?

**Insight:** Memory spaces are independent - no cross-space HNSW operations.

**Benefit:** Zero contention between workers (each owns subset of spaces).

**Scaling:** Linear up to number of active spaces. Example: 100 active spaces, 8 workers → each worker handles ~12 spaces with zero contention.

### 3. Why Work Stealing Instead of Global Queue?

**Problem:** Hash-based sharding can cause load imbalance (one space gets 10x traffic).

**Solution:** Workers steal from overloaded peers when idle.

**Complexity:** Adds ~100 lines of coordination code.

**Benefit:** Maintains load balance < 20% imbalance under realistic workloads.

### 4. Why Batch HNSW Instead of Single Inserts?

**Observation:** HNSW insert has fixed overhead (graph lock, entry point lookup).

**Measurement:** Single insert: ~100μs. Batch of 100: ~3ms = 30μs each (3x speedup).

**Strategy:** Adaptive batching - small batches (10) under low load for latency, large batches (500) under high load for throughput.

### 5. Why Eventual Consistency Instead of Linearizability?

**Biological justification:** Human memory formation is inherently probabilistic and asynchronous.

**Performance:** Linearizability would require global coordination (locks or consensus), killing throughput.

**Acceptable for workload:** Memory recall is naturally fuzzy. Missing a 10ms-old observation in a recall is acceptable if it appears in next recall.

**Bounded staleness:** Guarantee visibility within 100ms P99 provides practical consistency.

## Performance Architecture

### Bottleneck Analysis

| Component | Single-thread | Bottleneck | Parallelization Strategy |
|-----------|---------------|------------|-------------------------|
| gRPC receive | 100K ops/sec | CPU decode | Tokio async (100+ concurrent streams) |
| Queue enqueue | 5M ops/sec | None (lock-free) | N/A |
| HNSW insert | 10K ops/sec | CPU graph traversal | 4-8 workers = 40-80K ops/sec |
| Recall search | 50K ops/sec | CPU vector similarity | Concurrent searches (HNSW read-lock-free) |

**Critical bottleneck:** HNSW insertion at 10K/sec single-threaded.

**Solution:** 8 parallel workers → 80K ops/sec, exceeds 100K target with headroom.

### Zero-Copy Pipeline

```rust
// No intermediate copies
gRPC receive → Arc<Episode> → SegQueue → Worker → HNSW
```

**Memory allocation:**
- Episode allocated once at gRPC receive
- Wrapped in Arc (reference counting, not cloning)
- Queue stores Arc (no copy)
- Worker receives Arc (no copy)
- HNSW stores Arc reference (no copy)

**Result:** 768-dim embedding vector copied 0 times (single allocation).

### Memory Pool (Optional Optimization)

**Observation:** At 100K ops/sec, allocating 100K Episodes/sec stresses allocator.

**Solution:** Pre-allocate pool of 10K Episode objects, reuse after indexing.

**Complexity:** Adds lifecycle management (return to pool after HNSW insert).

**Decision:** Implement if profiling shows allocation overhead > 10%. Otherwise, defer to future optimization.

## Chaos Testing Strategy

### Failure Modes to Test

1. **Network delays:** Random 0-100ms delays on observation path
2. **Packet loss:** Drop random observations (client must detect via sequence gap and retry)
3. **Worker crashes:** Kill random HNSW worker, verify others take over
4. **Queue overflow:** Exceed capacity, verify admission control rejects gracefully
5. **Clock skew:** Simulate time jumps, verify timestamp handling robust

### Invariant Validation

**After 10 min chaos run with continuous failures:**

1. **No data loss:** All accepted observations (ack received) eventually indexed
2. **No corruption:** HNSW graph validation passes (connectivity, layer invariants)
3. **Bounded staleness:** 99% of observations visible within 100ms
4. **Performance degradation:** P99 latency < 100ms even under chaos

**Testing harness:**

```rust
#[test]
fn chaos_test_10min_sustained() {
    let chaos = ChaosConfig {
        delay: UniformDist(0ms, 100ms),
        drop_rate: 0.01,  // 1% packet loss
        worker_kill_rate: 0.1,  // Kill worker every 10s
        queue_overflow: true,
    };

    let server = spawn_test_server_with_chaos(chaos);
    let client = StreamingClient::connect(server.addr()).await.unwrap();

    let mut acked = HashSet::new();
    let start = Instant::now();

    while start.elapsed() < Duration::from_secs(600) {  // 10 min
        let episode = random_episode();
        match client.observe(episode.clone()).await {
            Ok(ack) => {
                acked.insert(episode.id);
            }
            Err(StreamError::Overload) => {
                // Expected during chaos - retry
                continue;
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    // Wait for staleness bound
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Validate all acked observations are indexed
    let recalled = client.recall_all().await.unwrap();
    let recalled_ids: HashSet<_> = recalled.iter().map(|e| e.id.clone()).collect();

    for id in &acked {
        assert!(recalled_ids.contains(id), "Observation {} was acked but not indexed", id);
    }

    // Validate HNSW integrity
    assert!(server.validate_hnsw_integrity(), "HNSW graph corrupted");
}
```

## Risk Analysis

### Risk 1: Lock Contention in HNSW Insert (CRITICAL)

**Probability:** HIGH
**Impact:** CRITICAL (blocks 100K ops/sec target)

**Validation:** Microbenchmark before Task 003.

```rust
#[bench]
fn bench_concurrent_hnsw_insert(b: &mut Bencher) {
    let index = Arc::new(CognitiveHnswIndex::new());

    b.iter(|| {
        // 8 threads inserting concurrently
        let handles: Vec<_> = (0..8).map(|t| {
            let idx = Arc::clone(&index);
            std::thread::spawn(move || {
                for i in 0..1000 {
                    let memory = random_memory(t * 1000 + i);
                    idx.insert_memory(Arc::new(memory)).unwrap();
                }
            })
        }).collect();

        for h in handles {
            h.join().unwrap();
        }
    });
}

// Target: 8K insertions in < 1 second (8K ops/sec with 8 threads)
```

**If contention detected:**
- Fallback 1: Per-layer locks (reduce contention scope)
- Fallback 2: Optimistic concurrency (retry on conflict)
- Fallback 3: Space partitioning (already planned, natural sharding)

### Risk 2: Unbounded Queue Growth (HIGH)

**Probability:** MEDIUM (only if admission control fails)
**Impact:** HIGH (OOM crash)

**Mitigation:**
1. Soft capacity limits trigger backpressure before hard limit
2. Hard capacity limit: enqueue fails with error (never unbounded growth)
3. Monitoring: alert when queue depth > 80%
4. Load shedding: prioritize recalls over observations when overloaded

**Testing:**

```rust
#[test]
fn test_queue_never_exceeds_capacity() {
    let config = QueueConfig { normal_capacity: 10_000, .. };
    let queue = ObservationQueue::new(config);

    // Try to enqueue 20K (2x capacity)
    let mut accepted = 0;
    for i in 0..20_000 {
        if queue.enqueue(space_id(), episode(i), i, ObservationPriority::Normal).is_ok() {
            accepted += 1;
        }
    }

    // Should accept at most capacity
    assert!(accepted <= 10_000);
    assert!(queue.total_depth() <= 10_000);
}
```

### Risk 3: Temporal Ordering Violations (MEDIUM)

**Probability:** MEDIUM (async processing can reorder)
**Impact:** HIGH (correctness bug)

**Mitigation:**
1. Sequence numbers provide total order within stream
2. Reject out-of-order observations with error
3. Property testing: verify no reordering under concurrent load

**Property test:**

```rust
proptest! {
    #[test]
    fn prop_no_reordering(
        observations in prop::collection::vec(any::<Episode>(), 100..1000)
    ) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let client = StreamingClient::test_client().await;

            // Send observations with sequence numbers
            for (seq, episode) in observations.iter().enumerate() {
                client.observe_with_sequence(episode.clone(), seq as u64).await.unwrap();
            }

            // Verify server received in order
            let acks = client.get_acks().await;
            for (i, ack) in acks.iter().enumerate() {
                assert_eq!(ack.sequence, i as u64, "Sequence mismatch at position {}", i);
            }
        });
    }
}
```

### Risk 4: Snapshot Isolation Complexity (MEDIUM)

**Probability:** MEDIUM (subtle concurrency bugs)
**Impact:** MEDIUM (degraded recall quality)

**Mitigation:**
1. Start simple: "all committed before timestamp"
2. Measure visibility latency in Task 007
3. If visibility > 100ms P99, add fast-path for recent observations

**Fallback:** If snapshot isolation too complex, ship without it (eventual consistency only). Document that recalls may miss very recent observations (<100ms).

## Success Metrics

### Performance (Quantitative)

- **Throughput:** Sustained 100K observations/sec for 60s on 4-core CPU ✓
- **Concurrent recalls:** 10 recalls/sec with < 20ms P99 latency ✓
- **Index visibility:** Observation → recall latency < 100ms P99 ✓
- **Memory usage:** < 2GB for 1M observations ✓
- **CPU usage:** < 80% during sustained 100K ops/sec load ✓

### Correctness (Qualitative)

- **No data loss:** Chaos test (10 min) with 0 lost observations ✓
- **No corruption:** HNSW graph validation passes throughout chaos ✓
- **Temporal ordering:** No reordering within stream (property test) ✓
- **Graceful degradation:** Backpressure activates before OOM ✓

### Production Readiness (Operational)

- **Monitoring:** Grafana dashboard for streaming health ✓
- **Alerting:** Queue depth and backpressure alerts configured ✓
- **Client examples:** Rust, Python, TypeScript examples functional ✓
- **Runbook:** Operators can configure and troubleshoot streaming ✓

## Implementation Schedule

### Week 1-2: Foundation (8 days)

- **Day 1-3:** Task 001 - Protocol foundation
- **Day 4-5:** Task 002 - Lock-free queue
- **Day 6-9:** Task 003 - Worker pool (parallel with Task 004)
- **Day 6-8:** Task 004 - Batch HNSW (parallel with Task 003)

**Milestone:** Core streaming pipeline functional (no flow control yet)

### Week 3: Streaming Interfaces (5 days)

- **Day 9-11:** Task 005 - gRPC streaming
- **Day 12-13:** Task 006 - Backpressure
- **Day 14-16:** Task 007 - Snapshot recall

**Milestone:** Full bidirectional streaming with flow control

### Week 4: Validation and Production (5 days)

- **Day 16-17:** Task 008 - WebSocket (parallel with Task 009)
- **Day 16-18:** Task 009 - Chaos testing (parallel with Task 008)
- **Day 18-19:** Task 010 - Performance tuning
- **Day 19-20:** Task 011 - Monitoring
- **Day 20-21:** Task 012 - Integration and docs

**Milestone:** Production-ready with validated correctness

### Parallelization Strategy (2 Engineers)

**Engineer A (Backend):**
- Week 1: Tasks 001, 002, 003 (foundation)
- Week 2: Tasks 006, 007 (flow control, recall)
- Week 3: Tasks 009, 010 (chaos, performance)

**Engineer B (Interfaces):**
- Week 1: Task 004 (batch HNSW)
- Week 2: Tasks 005, 008 (gRPC, WebSocket)
- Week 3: Tasks 011, 012 (monitoring, docs)

**Total:** 18 days (3.6 weeks) with 2 engineers

## Files Modified Summary

### New Files (12 files, ~3500 lines)

- `proto/engram/v1/streaming.proto` (200 lines)
- `engram-core/src/streaming/mod.rs` (50 lines)
- `engram-core/src/streaming/observation_queue.rs` (400 lines)
- `engram-core/src/streaming/worker_pool.rs` (600 lines)
- `engram-core/src/streaming/work_stealing.rs` (250 lines)
- `engram-core/src/streaming/session.rs` (200 lines)
- `engram-core/src/streaming/metrics.rs` (150 lines)
- `engram-server/src/grpc/streaming.rs` (500 lines)
- `engram-server/src/http/websocket.rs` (350 lines)
- `engram-core/tests/chaos/streaming_chaos.rs` (400 lines)
- `docs/operations/streaming.md` (200 lines)
- `examples/streaming/rust_client.rs` (200 lines)

### Modified Files (6 files, ~300 line changes)

- `proto/engram/v1/service.proto` (+150 lines)
- `engram-core/src/store.rs` (~100 line changes)
- `engram-core/src/index/hnsw_construction.rs` (+80 lines batch insert)
- `engram-core/src/index/hnsw_search.rs` (+50 lines snapshot isolation)
- `engram-server/src/grpc/service.rs` (+20 lines wire streaming)
- `Cargo.toml` (verify crossbeam dependency)

**Total:** ~3800 new lines, ~300 modified lines

## Next Steps After Milestone 11

**Milestone 12 (GPU Acceleration):**
- CUDA kernels for batch embedding similarity
- Parallel activation spreading on GPU
- Unified memory for zero-copy CPU-GPU transfers

**Milestone 13 (Cognitive Patterns):**
- Priming effects in recall
- Interference detection during consolidation
- Reconsolidation triggered by recall

**Milestone 14 (Distribution):**
- Partition memory across nodes
- Gossip-based consolidation sync
- Transparent distribution (API unchanged)
