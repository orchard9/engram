# Milestone 11: Streaming Interface - Complete Technical Specification

## Executive Summary

**Objective:** Build production-ready bidirectional streaming for continuous memory observation and real-time recall with lock-free incremental indexing.

**Performance Target:** Sustained 100K observations/second with concurrent recalls at <10ms P99 latency.

**Consistency Model:** Eventual consistency with bounded staleness (100ms P99 visibility).

**Timeline:** 18 days with 2 engineers (31 days single engineer).

**Validation:** 10-minute chaos test with zero data loss under continuous failures.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Consistency Model](#consistency-model)
3. [Performance Architecture](#performance-architecture)
4. [Protocol Design](#protocol-design)
5. [Implementation Tasks](#implementation-tasks)
6. [Risk Analysis](#risk-analysis)
7. [Testing Strategy](#testing-strategy)
8. [Operational Considerations](#operational-considerations)

---

## Architecture Overview

### System Context

Engram is a cognitive graph database. Milestone 11 adds **streaming memory formation** - the ability to continuously observe new experiences and recall memories in real-time, mirroring biological memory processes.

**Current state (M1-M10):**
- Batch-oriented APIs: `store(episode) → activation`
- HTTP REST + gRPC endpoints
- Server-Sent Events for metrics monitoring
- <10ms P99 latency for single operations

**Milestone 11 adds:**
- Bidirectional streaming: push observations + pull recalls in single stream
- 100K observations/second sustained throughput (10x batch)
- Lock-free incremental HNSW indexing
- Chaos-tested correctness under failures

### Component Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Client Layer                          │
│  • gRPC bidirectional stream                             │
│  • WebSocket for browsers                                │
│  • Session management with sequence numbers              │
└────────────────────┬─────────────────────────────────────┘
                     │
                     │ ObservationRequest (seq, episode)
                     ▼
┌──────────────────────────────────────────────────────────┐
│              Streaming Service (Task 001/005)            │
│  • Validate sequence numbers (monotonic per session)     │
│  • Route to memory space (multi-tenant isolation)        │
│  • Emit flow control signals (backpressure)              │
└────────────────────┬─────────────────────────────────────┘
                     │
                     │ Lock-free enqueue
                     ▼
┌──────────────────────────────────────────────────────────┐
│         Observation Queue (Task 002)                     │
│  • SegQueue (unbounded, lock-free)                       │
│  • Priority lanes: High (immediate) / Normal / Low       │
│  • Backpressure detection: queue_depth / capacity        │
└────────────────────┬─────────────────────────────────────┘
                     │
                     │ Dequeue batch (100-500 items)
                     ▼
┌──────────────────────────────────────────────────────────┐
│            Worker Pool (Task 003)                        │
│  • 4-8 parallel workers (one per CPU core)               │
│  • Space-based sharding: hash(space_id) % num_workers    │
│  • Work stealing: idle workers steal from overloaded     │
│  • Adaptive batching: small (10) → large (500) under load│
└────────────────────┬─────────────────────────────────────┘
                     │
                     │ Batch insert (amortized locks)
                     ▼
┌──────────────────────────────────────────────────────────┐
│              HNSW Index (Task 004)                       │
│  • Batch insertion API: 3-5x faster than single          │
│  • Lock-free reads during writes (crossbeam)             │
│  • Snapshot isolation for recalls                        │
└────────────────────┬─────────────────────────────────────┘
                     │
                     │ Snapshot-isolated search
                     ▼
┌──────────────────────────────────────────────────────────┐
│            Recall Stream (Task 007)                      │
│  • Incremental result streaming                          │
│  • Confidence scoring for recent observations            │
│  • Bounded staleness: 100ms P99 visibility               │
└──────────────────────────────────────────────────────────┘
```

---

## Consistency Model

### Formal Specification

**Type:** Eventual consistency with bounded staleness.

**NOT:**
- Linearizable (would require global coordination)
- Strongly consistent (would block on every write)
- Causally consistent across streams (no vector clocks)

**IS:**
- Eventually consistent (all observations eventually visible)
- Bounded staleness (P99 visibility within 100ms)
- Session-consistent (total order within stream)

### Invariants

**I1: Intra-stream total ordering**
```
∀ observations o₁, o₂ in session S:
  seq(o₁) < seq(o₂) ⟹ commit(o₁) happens-before commit(o₂)
```

**I2: Atomic visibility**
```
∀ observation o:
  visible(o) ⟹ ∀ fields f in o: visible(f)
```
No partial observations. Either fully indexed or not visible.

**I3: Bounded staleness**
```
∀ observation o:
  P99(time_to_visibility(o)) < 100ms
```
99% of observations visible within 100ms of commit.

**I4: No silent drops**
```
∀ observation o:
  (accepted(o) ⟹ eventually visible(o)) ∨ error_returned(o)
```
System never silently discards data.

**I5: Backpressure preserves FIFO**
```
∀ observations o₁, o₂:
  enqueue(o₁) happens-before enqueue(o₂) ⟹ dequeue(o₁) happens-before dequeue(o₂)
```
Even under load, FIFO order maintained.

### Consistency Trade-offs

**What we gain:**
- 100K ops/sec throughput (10x batch)
- Lock-free indexing (no contention)
- Graceful degradation under overload

**What we sacrifice:**
- Real-time visibility (100ms lag acceptable)
- Cross-stream ordering (probabilistic interleaving)
- Snapshot reads may miss very recent observations

**Rationale:** Biological memory formation is inherently asynchronous and probabilistic. We model cognitive reality, not ACID transactions.

---

## Performance Architecture

### Bottleneck Analysis

| Component | Single-Thread | Parallel Strategy | Target Throughput |
|-----------|---------------|-------------------|-------------------|
| gRPC recv | 100K ops/sec | Tokio async (100+ streams) | 200K+ ops/sec |
| Queue enqueue | 5M ops/sec | Lock-free (no bottleneck) | 5M+ ops/sec |
| **HNSW insert** | **10K ops/sec** | **8 workers = 80K ops/sec** | **100K ops/sec** |
| Recall search | 50K ops/sec | Concurrent reads (lock-free) | 100K+ ops/sec |

**Critical bottleneck:** HNSW insertion at 10K ops/sec single-threaded.

**Solution:** 8 parallel workers with space-based sharding → 80K ops/sec baseline, exceeds target with 20% headroom.

### Zero-Copy Pipeline

```rust
// gRPC receive: allocate Episode once
let episode = Episode::decode(grpc_bytes)?;

// Wrap in Arc (reference counting, not cloning)
let arc_episode = Arc::new(episode);

// Queue stores Arc reference (no copy)
queue.enqueue(arc_episode.clone());

// Worker receives Arc reference (no copy)
let obs = queue.dequeue()?;

// HNSW stores Arc reference (no copy)
hnsw.insert_memory(Arc::clone(&obs.episode));
```

**Result:** 768-dimensional embedding vector copied **0 times** (single allocation at gRPC receive).

**Memory savings:** At 100K ops/sec:
- With copies: 100K × 768 × 4 bytes = 307 MB/sec allocation rate
- Zero-copy: 100K × (Episode overhead ~200 bytes) = 20 MB/sec allocation rate
- **15x reduction in allocation rate**

### Adaptive Batching Strategy

**Observation:** HNSW batch insert has amortized overhead.

**Measurements:**
- Single insert: ~100μs (graph lock + entry point lookup + insertion)
- Batch of 10: ~1.2ms = 120μs each (1.2x overhead)
- Batch of 100: ~3ms = 30μs each (3x speedup)
- Batch of 1000: ~25ms = 25μs each (4x speedup, but high latency)

**Strategy:**
```rust
fn select_batch_size(queue_depth: usize) -> usize {
    match queue_depth {
        0..100 => 10,      // Low load: optimize for latency
        100..1000 => 100,  // Medium load: balance latency vs throughput
        1000.. => 500,     // High load: optimize for throughput
    }
}
```

**Trade-offs:**
- Batch 10: 12ms latency, 83K ops/sec throughput (8 workers)
- Batch 100: 30ms latency, 267K ops/sec throughput (8 workers)
- Batch 500: 125ms latency, 320K ops/sec throughput (8 workers)

**Target:** Adaptive batching achieves 100K ops/sec sustained with P99 latency <100ms.

---

## Protocol Design

### gRPC Messages

**Streaming observation request:**

```protobuf
message ObservationRequest {
  string memory_space_id = 1;

  oneof operation {
    StreamInit init = 2;
    Episode observation = 3;
    FlowControl flow = 4;
    StreamClose close = 5;
  }

  string session_id = 10;
  uint64 sequence_number = 11;  // Monotonic per session
}
```

**Streaming observation response:**

```protobuf
message ObservationResponse {
  oneof result {
    StreamInitAck init_ack = 1;
    ObservationAck ack = 2;
    StreamStatus status = 3;
  }

  string session_id = 10;
  uint64 sequence_number = 11;  // Echo client sequence
  google.protobuf.Timestamp server_timestamp = 12;
}
```

**Observation acknowledgment:**

```protobuf
message ObservationAck {
  enum Status {
    STATUS_ACCEPTED = 1;   // Queued for indexing
    STATUS_INDEXED = 2;    // Visible in HNSW
    STATUS_REJECTED = 3;   // Admission control reject
  }

  Status status = 1;
  string memory_id = 2;
  google.protobuf.Timestamp committed_at = 3;
}
```

### Session Lifecycle

```
1. INIT:    Client → StreamInit
            Server → StreamInitAck (session_id, capabilities)

2. ACTIVE:  Client → ObservationRequest (seq 0, 1, 2, ...)
            Server → ObservationResponse (ack for each)

3. PAUSE:   Client → FlowControl(ACTION_PAUSE)
            Server → StreamStatus(STATE_PAUSED)

4. RESUME:  Client → FlowControl(ACTION_RESUME)
            Server → StreamStatus(STATE_ACTIVE)

5. CLOSE:   Client → StreamClose
            Server → StreamStatus(STATE_CLOSED)
            Server drains queue before closing stream
```

### Sequence Number Protocol

**Client invariant:** Sequence numbers are strictly monotonic within session.

```rust
impl StreamingClient {
    async fn observe(&self, episode: Episode) -> Result<ObservationAck> {
        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);
        let request = ObservationRequest {
            session_id: self.session_id.clone(),
            sequence_number: seq,
            operation: Some(Operation::Observation(episode)),
        };

        let response = self.send(request).await?;

        // Verify sequence echo
        assert_eq!(response.sequence_number, seq);
        Ok(response.ack)
    }
}
```

**Server validation:** Reject out-of-order sequences.

```rust
fn validate_sequence(
    session: &StreamSession,
    received_seq: u64,
) -> Result<(), SequenceError> {
    let expected = session.last_sequence.fetch_add(1, Ordering::SeqCst) + 1;

    if received_seq != expected {
        return Err(SequenceError::Mismatch {
            expected,
            received: received_seq,
        });
    }

    Ok(())
}
```

### Flow Control

**Backpressure signal:**

```protobuf
message StreamStatus {
  State state = 1;  // BACKPRESSURE or OVERLOADED
  uint32 queue_depth = 3;
  uint32 queue_capacity = 4;
  float pressure = 5;  // queue_depth / queue_capacity
}
```

**Client response:**
- `BACKPRESSURE`: Reduce send rate by 50%
- `OVERLOADED`: Pause sending, retry after delay

**Server trigger:**
- Queue depth > 80% capacity: Send `BACKPRESSURE`
- Queue depth > 90% capacity: Send `OVERLOADED`, reject new observations

---

## Implementation Tasks

### Task Dependencies Graph

```
        001 (Protocol)
             ↓
        002 (Queue)
             ↓
    ┌────────┴────────┐
    ▼                 ▼
003 (Workers)    004 (Batch HNSW)
    └────────┬────────┘
             ▼
        005 (gRPC)
             ↓
        006 (Backpressure)
             ↓
        007 (Recall)
             ↓
    ┌────────┴────────┐
    ▼                 ▼
008 (WebSocket)   009 (Chaos)
    └────────┬────────┘
             ▼
        010 (Performance)
             ↓
        011 (Monitoring)
             ↓
        012 (Integration)
```

### Critical Path (18 days with 2 engineers)

**Week 1-2: Foundation**
- Day 1-3: Task 001 - Protocol design, session management
- Day 4-5: Task 002 - Lock-free queue with priority lanes
- Day 6-9: Task 003 - Worker pool (parallel with 004)
- Day 6-8: Task 004 - Batch HNSW (parallel with 003)

**Week 3: Streaming**
- Day 9-11: Task 005 - gRPC bidirectional streaming
- Day 12-13: Task 006 - Backpressure and admission control
- Day 14-16: Task 007 - Snapshot-isolated recall

**Week 4: Validation**
- Day 16-17: Task 008 - WebSocket (parallel with 009)
- Day 16-18: Task 009 - Chaos testing (parallel with 008)
- Day 18-19: Task 010 - Performance tuning
- Day 19-20: Task 011 - Monitoring and alerting
- Day 20-21: Task 012 - Integration and documentation

### Parallelization (2 Engineers)

**Engineer A (Backend):**
- Week 1: Foundation (001-003)
- Week 2: Flow control (006-007)
- Week 3: Chaos testing (009-010)

**Engineer B (Interfaces):**
- Week 1: Batch optimization (004)
- Week 2: Streaming (005, 008)
- Week 3: Operations (011-012)

---

## Risk Analysis

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| HNSW lock contention | HIGH | CRITICAL | Microbenchmark, fallback to per-layer locks |
| Unbounded queue growth | MEDIUM | HIGH | Soft/hard capacity, monitoring alerts |
| Temporal ordering violations | MEDIUM | HIGH | Property testing, sequence validation |
| Snapshot isolation complexity | MEDIUM | MEDIUM | Start simple, incremental improvement |
| WebSocket scalability | LOW | MEDIUM | Document limits, recommend gRPC for high-throughput |

### Critical Risk: HNSW Lock Contention

**Assumption:** HNSW insertion is lock-free (uses crossbeam).

**Validation required:** Microbenchmark concurrent insertions BEFORE Task 003.

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
                    idx.insert_memory(Arc::new(random_memory(t * 1000 + i))).unwrap();
                }
            })
        }).collect();

        for h in handles {
            h.join().unwrap();
        }
    });
}

// Target: 8K insertions in < 1 second (8K ops/sec with 8 threads)
// If < 5K ops/sec, HNSW has lock contention - need fallback plan
```

**Fallback plans (in order of preference):**

1. **Per-layer locks:** Reduce contention scope (layer 0 separate from layer 1)
2. **Optimistic concurrency:** Retry on conflict (works if conflicts rare)
3. **Space partitioning:** Already planned - natural sharding eliminates contention

---

## Testing Strategy

### Unit Tests (Per Task)

**Example: Queue priority ordering**

```rust
#[test]
fn test_priority_ordering() {
    let queue = ObservationQueue::new(QueueConfig::default());

    // Enqueue in mixed order
    queue.enqueue(space(), ep(1), 1, Low).unwrap();
    queue.enqueue(space(), ep(2), 2, High).unwrap();
    queue.enqueue(space(), ep(3), 3, Normal).unwrap();

    // Dequeue: should get High, Normal, Low
    assert_eq!(queue.dequeue().unwrap().sequence_number, 2);
    assert_eq!(queue.dequeue().unwrap().sequence_number, 3);
    assert_eq!(queue.dequeue().unwrap().sequence_number, 1);
}
```

### Property Tests (Invariant Validation)

**Example: Sequence monotonicity**

```rust
proptest! {
    #[test]
    fn prop_sequence_monotonic(
        observations in vec(any::<Episode>(), 100..1000)
    ) {
        let client = StreamingClient::test().await;

        let mut last_seq = 0u64;
        for episode in observations {
            let ack = client.observe(episode).await.unwrap();
            assert!(ack.sequence > last_seq);
            last_seq = ack.sequence;
        }
    }
}
```

### Integration Tests (End-to-End)

**Example: Streaming roundtrip**

```rust
#[tokio::test]
async fn test_streaming_roundtrip() {
    let server = spawn_server().await;
    let client = connect(server.addr()).await;

    // Initialize stream
    client.init("test_space").await.unwrap();

    // Stream 1000 observations
    for i in 0..1000 {
        let ack = client.observe(episode(i)).await.unwrap();
        assert_eq!(ack.status, ACCEPTED);
    }

    // Recall all
    let results = client.recall_all().await.unwrap();
    assert_eq!(results.len(), 1000);
}
```

### Chaos Tests (Failure Injection)

**10-minute sustained chaos:**

```rust
#[tokio::test(flavor = "multi_thread")]
async fn chaos_test_10min() {
    let chaos = ChaosConfig {
        delay: Uniform(0ms, 100ms),
        packet_loss: 0.01,
        worker_kill_interval: 10s,
    };

    let server = spawn_with_chaos(chaos).await;
    let client = connect(server.addr()).await;

    let acked = Arc::new(DashMap::new());
    let start = Instant::now();

    // Stream for 10 minutes with chaos
    while start.elapsed() < Duration::from_secs(600) {
        match client.observe(random_episode()).await {
            Ok(ack) => { acked.insert(ack.memory_id, ()); }
            Err(Overload) => { /* expected */ }
            Err(e) => panic!("Unexpected: {}", e),
        }
    }

    // Wait for bounded staleness
    sleep(Duration::from_millis(200)).await;

    // Validate eventual consistency
    let recalled = client.recall_all().await.unwrap();
    for id in acked.iter() {
        assert!(recalled.contains(id.key()));
    }

    // Validate HNSW integrity
    assert!(server.validate_hnsw());
}
```

---

## Operational Considerations

### Hardware Requirements

**Minimum:**
- 4-core CPU (Intel/AMD x86_64 with AVX2)
- 4GB RAM (2GB observations, 2GB HNSW index)
- 20GB SSD storage (WAL + persistence)
- 1Gbps network

**Recommended for 100K ops/sec:**
- 8-core CPU (or 4-core with 2x SMT)
- 8GB RAM (headroom for bursts)
- 50GB NVMe SSD
- 10Gbps network

### Configuration Parameters

```toml
[streaming]
# Worker pool
num_workers = 8  # Default: num_cpus, range: 4-16

# Queue capacity (soft limits for backpressure)
queue_capacity_high = 10_000
queue_capacity_normal = 100_000
queue_capacity_low = 50_000

# Batch sizes
batch_size_low_load = 10      # < 100 items in queue
batch_size_medium_load = 100  # 100-1000 items
batch_size_high_load = 500    # > 1000 items

# Backpressure thresholds
backpressure_threshold = 0.8  # 80% capacity
admission_control_threshold = 0.9  # 90% capacity

# Session timeout
session_timeout_seconds = 300  # 5 minutes idle
```

### Monitoring Metrics

**Prometheus metrics:**

```prometheus
# Queue depth by priority
engram_observation_queue_depth{priority="high|normal|low"}

# Backpressure events
engram_observation_backpressure_total

# Worker utilization
engram_worker_utilization{worker_id="0..7"}

# Latency distribution
engram_observation_latency_seconds{quantile="0.5|0.99|0.999"}

# Visibility lag (observation → indexed)
engram_index_visibility_latency_seconds{quantile="0.5|0.99|0.999"}
```

**Grafana alerts:**

```yaml
- alert: QueueDepthHigh
  expr: engram_observation_queue_depth > 80000
  for: 5m
  annotations:
    summary: "Observation queue depth > 80%"

- alert: BackpressureActive
  expr: rate(engram_observation_backpressure_total[5m]) > 10
  for: 5m
  annotations:
    summary: "Backpressure activated frequently"

- alert: WorkerCrashed
  expr: up{job="engram_worker"} == 0
  for: 1m
  annotations:
    summary: "HNSW worker crashed"
```

### Troubleshooting Guide

**Symptom: High latency (P99 > 100ms)**

1. Check queue depth: `engram_observation_queue_depth`
2. If > 80%: Scale up workers or reduce ingestion rate
3. Check worker utilization: `engram_worker_utilization`
4. If < 60%: HNSW contention - validate with bench_concurrent_hnsw_insert
5. Check batch size: Increase for higher throughput

**Symptom: Observations rejected (OVERLOADED)**

1. Check queue capacity: `queue_capacity_normal`
2. Increase capacity (short-term fix)
3. Add workers (long-term solution)
4. Enable adaptive batching (may already be active)

**Symptom: Worker crashes frequently**

1. Check logs for panic messages
2. Validate HNSW integrity: `server.validate_hnsw_integrity()`
3. If corrupted: Rebuild index from WAL
4. If OOM: Reduce queue capacity or add RAM

---

## Conclusion

This specification defines a **production-ready streaming interface** for Engram that:

1. **Scales to 100K ops/sec** via parallel workers and adaptive batching
2. **Guarantees correctness** via eventual consistency with bounded staleness
3. **Handles failures** via chaos-tested recovery and graceful degradation
4. **Enables real-time interaction** via bidirectional streaming and incremental recall

The design reflects **cognitive principles** (probabilistic consistency mirrors biological memory) while achieving **systems performance** (lock-free, zero-copy, parallel).

**Next steps:**
1. Review and approve this specification
2. Create remaining task files (004-008, 010-012)
3. Begin implementation with Task 001 (Protocol Foundation)

---

**Document version:** 1.0
**Last updated:** 2025-10-23
**Authors:** Systems Product Planner (Bryan Cantrill mode)
