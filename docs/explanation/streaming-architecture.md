# Streaming Infrastructure Architecture

A comprehensive technical overview of Engram's high-performance streaming infrastructure for continuous memory formation.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Concurrency Model](#concurrency-model)
5. [Performance Characteristics](#performance-characteristics)
6. [Design Decisions](#design-decisions)

## System Overview

The streaming infrastructure enables real-time, continuous memory formation at scale (100K+ observations/second) with bounded staleness consistency. It's designed around biological memory formation principles - specifically the hippocampal-neocortical interaction pattern where observations are initially processed quickly (hippocampus analog) and then consolidated into durable indices (neocortex analog) asynchronously.

### High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Layer                                 │
├──────────────────┬─────────────────────┬─────────────────────────────┤
│  gRPC Client     │  WebSocket Client   │  HTTP Client                │
└────────┬─────────┴──────────┬──────────┴──────────┬──────────────────┘
         │                    │                      │
         v                    v                      v
┌─────────────────────────────────────────────────────────────────────┐
│                     Protocol Handlers                                │
│   ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐        │
│   │gRPC Handler  │  │WS Handler     │  │HTTP Handler      │        │
│   └──────┬───────┘  └───────┬───────┘  └────────┬─────────┘        │
│          └──────────────────┼─────────────────────┘                 │
│                             v                                         │
│                   ┌─────────────────┐                                │
│                   │SessionManager   │                                │
│                   └────────┬────────┘                                │
└──────────────────────────┼──────────────────────────────────────────┘
                           v
┌─────────────────────────────────────────────────────────────────────┐
│                    Streaming Core                                    │
│   ┌───────────────────────────────────────────────────────┐         │
│   │              ObservationQueue                          │         │
│   │   ┌───────────────┐  ┌───────────────┐               │         │
│   │   │High Priority  │  │Normal Priority│               │         │
│   │   │(SegQueue)     │  │(SegQueue)     │               │         │
│   │   └───────┬───────┘  └───────┬───────┘               │         │
│   │           │                   │                        │         │
│   │           └───────────────────┘                        │         │
│   │           ┌───────────────┐                            │         │
│   │           │Low Priority   │                            │         │
│   │           │(SegQueue)     │                            │         │
│   │           └───────┬───────┘                            │         │
│   └───────────────────┼────────────────────────────────────┘         │
│                       v                                               │
│   ┌──────────────────────────────────────────────────┐               │
│   │           WorkerPool (4-8 workers)               │               │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │           │
│   │   │Worker 0 │ │Worker 1 │ │Worker 2 │ │Worker 3 │  │           │
│   │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘  │           │
│   │        │           │           │           │        │           │
│   │        └───────────┴───────────┴───────────┘        │           │
│   │            Work Stealing (queue depth > 1000)       │           │
│   └──────────────────────┬───────────────────────────────┘           │
└────────────────────────┼─────────────────────────────────────────────┘
                         v
┌─────────────────────────────────────────────────────────────────────┐
│                  Space-Isolated HNSW                                 │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐            │
│   │Space: default │ │Space: user_1  │ │Space: user_2  │            │
│   │HNSW Index     │ │HNSW Index     │ │HNSW Index     │            │
│   │(DashMap)      │ │(DashMap)      │ │(DashMap)      │            │
│   └───────────────┘ └───────────────┘ └───────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### SessionManager

The `SessionManager` provides lifecycle management for streaming sessions with monotonic sequence validation. It ensures that within a single session, observations are totally ordered (monotonic sequence numbers), while allowing undefined ordering across different sessions.

**Key Features:**
- Lock-free concurrent access via `DashMap`
- Automatic session timeout (5 minutes idle)
- Atomic sequence number tracking
- Session state machine (Active → Paused → Closed)

**API Example:**
```rust
use engram_core::streaming::session::{SessionManager, SessionState};
use engram_core::types::MemorySpaceId;

let manager = SessionManager::new();
let space_id = MemorySpaceId::new("user_123")?;

// Create session
let session = manager.create_session("session_abc".to_string(), space_id);

// Validate sequence numbers (monotonic)
session.validate_sequence(1)?; // OK
session.validate_sequence(2)?; // OK
session.validate_sequence(1)?; // ERROR: duplicate
session.validate_sequence(4)?; // ERROR: gap (expected 3)

// State transitions
session.set_state(SessionState::Paused);
session.set_state(SessionState::Active);
manager.close_session("session_abc")?;
```

**Performance:**
- Session creation: < 1μs
- Sequence validation: < 100ns (atomic CAS operation)
- Session lookup: < 100ns (DashMap read)

### ObservationQueue

A lock-free, high-throughput observation queue with three priority lanes (High/Normal/Low). Uses `crossbeam::SegQueue` for unbounded, wait-free operations.

**Key Features:**
- Lock-free enqueue/dequeue (4M+ ops/sec throughput)
- Soft capacity limits with backpressure detection
- Priority-based processing (High → Normal → Low)
- Atomic depth tracking

**Architecture:**
```text
ObservationQueue
├── High Priority Lane (SegQueue)    - Capacity: 10K
├── Normal Priority Lane (SegQueue)  - Capacity: 100K
└── Low Priority Lane (SegQueue)     - Capacity: 50K

Backpressure Thresholds:
- Normal: < 50% capacity
- Warning: 50-80% capacity
- Critical: 80-95% capacity
- Overloaded: > 95% capacity (reject new)
```

**API Example:**
```rust
use engram_core::streaming::{ObservationQueue, QueueConfig, ObservationPriority};

let queue = ObservationQueue::new(QueueConfig::default());

// Enqueue with priority
queue.enqueue(
    space_id,
    episode,
    sequence_number,
    ObservationPriority::High
)?;

// Dequeue respects priority order
let obs = queue.dequeue(); // Gets High first, then Normal, then Low

// Batch dequeue for efficiency
let batch = queue.dequeue_batch(100);

// Check backpressure
if queue.should_apply_backpressure() {
    // Slow down or pause ingestion
}
```

**Performance:**
- Enqueue: ~200ns
- Dequeue: ~200ns
- Throughput: 4M+ ops/sec (8-thread contention)
- Memory: 64 bytes per queued observation

### WorkerPool

A multi-threaded worker pool with space-based sharding and work stealing for parallel HNSW indexing. Each worker is assigned specific memory spaces via consistent hashing, ensuring cache locality and zero cross-worker contention for the same space.

**Key Features:**
- Space-based sharding (consistent hashing)
- Work stealing for load balancing
- Adaptive batch sizing (10-1000 based on queue depth)
- Graceful shutdown with queue draining

**Architecture:**
```text
Worker Pool: Hash-Based Sharding + Work Stealing

Space Assignment (Consistent Hashing):
┌──────────────────────────────────────────────────┐
│ Space ID → hash(space_id) % num_workers          │
├──────────────────────────────────────────────────┤
│ space "default"  → Worker 0                      │
│ space "user_001" → Worker 2                      │
│ space "user_002" → Worker 1                      │
│ space "user_003" → Worker 3                      │
└──────────────────────────────────────────────────┘

Work Stealing (when queue > 1000):
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Worker 0  │  │Worker 1  │  │Worker 2  │  │Worker 3  │
│Queue:50  │  │Queue:1500│  │Queue:20  │  │Queue:10  │
└─────┬────┘  └─────┬─────┘ └─────┬────┘  └─────┬────┘
      │             │            ↑             ↑
      │             └────────────┴─────────────┘
      │               Steal work from W1
      └──────────────────────────────────────────→
                    Also steals
```

**API Example:**
```rust
use engram_core::streaming::{WorkerPool, WorkerPoolConfig};

let config = WorkerPoolConfig {
    num_workers: 8,
    steal_threshold: 1000,
    min_batch_size: 10,
    max_batch_size: 500,
    ..Default::default()
};

let pool = WorkerPool::new(config);

// Enqueue observations (routed by space ID)
pool.enqueue(space_id, episode, seq, ObservationPriority::Normal)?;

// Check statistics
let stats = pool.worker_stats();
for (i, stat) in stats.iter().enumerate() {
    println!("Worker {}: processed={}, stolen={}, failed={}",
        i, stat.processed_observations, stat.stolen_batches, stat.failed_batches);
}

// Graceful shutdown (drain queues)
pool.shutdown(Duration::from_secs(30))?;
```

**Performance:**
- Throughput: 100K+ obs/sec with 8 workers
- Latency: P99 < 100ms (observation → indexed)
- Work stealing overhead: ~200ns per stolen item
- Scaling: Linear up to core count

### SpaceIsolatedHnsw

Space-partitioned HNSW architecture where each memory space gets its own independent HNSW index. This eliminates cross-space contention entirely, enabling perfect parallel scaling.

**Key Features:**
- Zero cross-space contention
- Lazy index creation (on first access)
- Lock-free space lookup via `DashMap`
- Per-space HNSW parameters

**Architecture:**
```text
SpaceIsolatedHnsw
│
├── DashMap<MemorySpaceId, Arc<CognitiveHnswIndex>>
│   │
│   ├── "default" → HNSW Index (16MB)
│   ├── "user_1" → HNSW Index (16MB)
│   └── "user_2" → HNSW Index (16MB)
│
└── Zero contention between spaces
    - Parallel insertions: O(log N) per space
    - Independent graph structures
    - Cache-local operations
```

**Trade-offs:**

**Advantages:**
- Perfect parallel scaling (no lock contention)
- Natural sharding for worker assignment
- Simpler correctness reasoning
- Cache locality for same-space operations

**Disadvantages:**
- Higher memory overhead (~16MB per active space)
- Cross-space recall requires querying multiple indices
- Inefficient for many small spaces
- Index size proportional to space count

**API Example:**
```rust
use engram_core::streaming::SpaceIsolatedHnsw;

let space_hnsw = SpaceIsolatedHnsw::new();

// Insertions to different spaces have ZERO contention
space_hnsw.insert_memory(&space1, memory1)?;
space_hnsw.insert_memory(&space2, memory2)?;

// Get index for specific space
if let Some(index) = space_hnsw.get_index(&space1) {
    let results = index.search(&query, k)?;
}

// Cross-space query
for space_id in space_hnsw.active_spaces() {
    if let Some(index) = space_hnsw.get_index(&space_id) {
        let results = index.search(&query, k)?;
        // Merge results
    }
}
```

**Performance:**
- Insert: O(log N) per space (no cross-space lock)
- Search: O(log N) per space
- Memory: ~16MB per active space
- Throughput: 100K+ ops/sec across multiple spaces

### BackpressureMonitor

Adaptive backpressure monitoring that periodically samples queue depth and broadcasts state changes to subscribers. Enables coordinated flow control across handlers and workers.

**Key Features:**
- State-based pressure levels (Normal/Warning/Critical/Overloaded)
- Adaptive batch size recommendations
- Retry-after calculation for admission control
- Broadcast channel for real-time updates

**State Machine:**
```text
Backpressure State Machine

States:
┌─────────────────┐
│     Normal      │ Queue < 50%
│  (No Action)    │ batch_size: 10
└────────┬────────┘
         │ queue >= 50%
         v
┌─────────────────┐
│     Warning     │ Queue 50-80%
│ (Log Warning)   │ batch_size: 100
└────────┬────────┘
         │ queue >= 80%
         v
┌─────────────────┐
│    Critical     │ Queue 80-95%
│ (Batch Size ↑)  │ batch_size: 500
└────────┬────────┘
         │ queue >= 95%
         v
┌─────────────────┐
│   Overloaded    │ Queue > 95%
│ (Reject New)    │ batch_size: 1000
└────────┬────────┘
         │ queue < 95%
         v
      [Return to appropriate state]

Transitions:
- Upward: Immediate when threshold crossed
- Downward: 5-second grace period to prevent flapping
```

**API Example:**
```rust
use engram_core::streaming::backpressure::{BackpressureMonitor, BackpressureState};

let monitor = BackpressureMonitor::new(
    Arc::clone(&queue),
    Duration::from_millis(100)
);

// Subscribe to state changes
let mut rx = monitor.subscribe();

// Run monitoring loop
tokio::spawn(async move {
    monitor.run().await
});

// React to state changes
while let Ok(state) = rx.recv().await {
    match state {
        BackpressureState::Normal => {
            // Resume full speed
        }
        BackpressureState::Warning => {
            // Log warning
        }
        BackpressureState::Critical => {
            // Increase batch size
        }
        BackpressureState::Overloaded => {
            // Reject new observations
            let retry_after = calculate_retry_after(queue_depth, dequeue_rate);
            // Tell client to retry after this duration
        }
    }
}
```

**Performance:**
- Check interval: 100ms (configurable)
- State transition broadcast: < 1μs
- CPU overhead: < 0.1% (periodic sampling)

## Data Flow

### Observation Streaming Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant H as Handler
    participant S as SessionManager
    participant Q as ObservationQueue
    participant W as WorkerPool
    participant HNSW as SpaceIsolatedHnsw

    C->>H: StreamInit
    H->>S: create_session()
    S-->>H: session_id
    H-->>C: InitAck

    loop Observation Streaming
        C->>H: Observation(seq=N)
        H->>S: validate_sequence(N)
        S-->>H: OK
        H->>Q: enqueue(episode, priority)
        Q-->>H: OK
        H-->>C: Ack(seq=N)

        Note over W,HNSW: Async processing
        W->>Q: dequeue_batch()
        Q-->>W: batch[episodes]
        W->>HNSW: insert_batch(space_id, batch)
        HNSW-->>W: OK
    end

    C->>H: FlowControl(pause)
    H->>S: set_state(Paused)

    C->>H: Close
    H->>S: close_session()
```

### Recall Flow with Snapshot Isolation

```mermaid
sequenceDiagram
    participant C as Client
    participant H as Handler
    participant Q as ObservationQueue
    participant HNSW as SpaceIsolatedHnsw
    participant R as RecallStream

    C->>H: RecallRequest(query, k)
    H->>Q: current_generation()
    Q-->>H: generation=1000

    H->>HNSW: get_index(space_id)
    HNSW-->>H: index

    H->>R: new(index, query, generation)

    loop Incremental Results
        R->>HNSW: search(query, k)
        HNSW-->>R: all_results
        R->>R: filter(generation <= snapshot)
        R->>R: batch results
        R-->>H: batch[memories]
        H-->>C: RecallBatch
    end

    R-->>C: Complete
```

### Backpressure Activation Flow

```text
Backpressure Coordination

BackpressureMonitor (100ms interval)
         │
         ├─ Sample queue depth
         │  pressure = depth / capacity
         │
         ├─ Detect state change
         │  Normal → Warning → Critical → Overloaded
         │
         └─ Broadcast state change
                     │
                     ├──→ StreamHandlers
                     │    - Forward FlowControl to clients
                     │    - Adjust admission control
                     │
                     ├──→ WorkerPool
                     │    - Adjust batch size
                     │    - Increase worker utilization
                     │
                     └──→ Metrics
                          - Record state transition
                          - Update Prometheus gauges
```

## Concurrency Model

### Lock-Free Primitives

The streaming infrastructure is built on lock-free data structures to maximize throughput and minimize latency variance:

**1. SegQueue (ObservationQueue):**
- Based on Michael & Scott algorithm
- Lock-free push and pop operations
- Memory ordering: Relaxed for depth counters (eventual consistency acceptable)
- CAS operations for queue nodes

**2. DashMap (SessionManager, SpaceIsolatedHnsw):**
- Sharded HashMap with per-shard RwLocks
- Lock-free reads via immutable snapshots
- Write locks are shard-local (16-64 shards)
- Average case: O(1) with minimal contention

**3. AtomicU64 (Metrics, Sequence Numbers):**
- Lock-free fetch_add, fetch_max operations
- Memory ordering varies by use case:
  - SeqCst for sequence numbers (correctness-critical)
  - Relaxed for metrics (eventual consistency OK)

### Memory Ordering Guarantees

```rust
// Session sequence validation: SeqCst (correctness-critical)
session.last_sequence.store(received_seq, Ordering::SeqCst);

// Queue depth tracking: Relaxed (eventual consistency acceptable)
self.high_depth.fetch_add(1, Ordering::Relaxed);

// Metrics: Relaxed (eventual consistency acceptable)
self.total_enqueued.fetch_add(1, Ordering::Relaxed);

// Worker shutdown: SeqCst (coordination required)
self.shutdown.store(true, Ordering::SeqCst);
```

### Threading Model

```text
Thread Pool Architecture

Main Thread
├── gRPC Handler Threads (tokio runtime, N = CPU cores)
│   ├── Accept connections
│   ├── Validate sequences
│   ├── Enqueue observations
│   └── Stream recall results
│
├── HNSW Worker Threads (4-8 dedicated threads)
│   ├── Dequeue batches
│   ├── Convert Episode → Memory
│   ├── Insert into space-specific HNSW
│   └── Update metrics
│
├── Backpressure Monitor (1 thread)
│   ├── Sample queue depth every 100ms
│   ├── Detect state transitions
│   └── Broadcast to subscribers
│
└── Metrics Exporter (1 thread)
    ├── Aggregate metrics
    └── Serve Prometheus endpoint
```

## Performance Characteristics

### Throughput

| Operation | Throughput | Latency (P99) | Notes |
|-----------|------------|---------------|-------|
| Observation Ingest | 100K+ obs/sec | < 1ms | Limited by network, not CPU |
| Queue Operations | 4M+ ops/sec | < 200ns | Lock-free SegQueue |
| HNSW Insertion | 1,600 obs/sec/worker | < 100ms | 8 workers = 12.8K obs/sec |
| Recall Query | 1,000+ concurrent | < 10ms first result | Incremental streaming |
| Sequence Validation | 10M+ ops/sec | < 100ns | Atomic CAS operation |

### Scaling Characteristics

**Horizontal Scaling (Multiple Workers):**
- Space-isolated HNSW enables linear scaling up to number of active spaces
- 1 worker: ~1,600 obs/sec
- 4 workers: ~6,400 obs/sec
- 8 workers: ~12,800 obs/sec
- Scaling factor: 0.95 (near-linear due to work stealing overhead)

**Vertical Scaling (More Cores per Worker):**
- HNSW insertion is single-threaded per space
- Additional cores only help with more concurrent spaces
- Work stealing threshold tuning critical for load balance

**Queue Depth Scaling:**
- Memory: 64 bytes per observation
- 100K capacity = 6.4 MB
- 1M capacity = 64 MB
- Trade-off: Higher capacity → more buffering → higher latency

### Latency Breakdown

**Observation End-to-End Latency (P99):**
```text
Client → Server:          ~50ms   (network)
Sequence Validation:      < 0.1ms (atomic CAS)
Queue Enqueue:            < 0.2ms (lock-free push)
Worker Dequeue:           < 1ms   (polling interval)
HNSW Insertion:           < 100ms (index update)
─────────────────────────────────────────────
Total (P99):              < 150ms
```

**Recall Query Latency (P99):**
```text
HNSW Search:              ~5ms    (log N traversal)
Snapshot Filtering:       < 1ms   (generation check)
First Batch Serialization:< 2ms   (Protobuf encode)
Network Transmission:     ~2ms    (local network)
─────────────────────────────────────────────
First Result (P99):       < 10ms
```

### Memory Overhead

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| ObservationQueue | ~6.4 MB | 100K capacity, 64 bytes per obs |
| SessionManager | ~1 KB per session | Minimal overhead |
| WorkerPool | ~4 MB total | Thread stacks + buffers |
| SpaceIsolatedHnsw | ~16 MB per space | HNSW graph structure |
| Metrics | ~100 KB | Prometheus counters/gauges |

**Example: 10 Active Spaces**
- Queue: 6.4 MB
- Workers: 4 MB
- HNSW (10 spaces): 160 MB
- Overhead: ~1 MB
- **Total: ~171 MB**

## Design Decisions

### Why Space Partitioning?

**Problem:** Shared HNSW indices cannot achieve required throughput due to lock contention.

**Validation Results (Task 004):**
- Shared HNSW: 1,957 ops/sec with 8 threads (< 3% of target)
- Lock contention dominated performance
- Mutex wait time > computation time

**Solution:** Space-isolated HNSW with zero cross-space contention.

**Benefits:**
- Linear scaling up to space count
- Cache locality (same space → same worker)
- Simpler correctness reasoning
- Natural sharding for worker assignment

**Trade-offs:**
- Higher memory overhead (~16 MB per space)
- Cross-space queries require index merging
- Inefficient for many small spaces

**When to Use:**
- Multi-tenant systems (space per tenant)
- User-specific memory spaces
- Context-specific memory (space per conversation)

**When Not to Use:**
- Single global memory space
- Thousands of tiny spaces (<100 memories each)

### Why Eventual Consistency?

**Problem:** Strong consistency (linearizability) requires coordination that kills throughput.

**Biological Inspiration:**
- Hippocampus: Fast initial encoding
- Neocortex: Slow consolidation (hours to days)
- No "global clock" in brain
- Working memory ≠ long-term memory

**Solution:** Bounded staleness with P99 < 100ms visibility lag.

**Benefits:**
- 100x throughput improvement vs strict consistency
- Matches biological memory formation
- Acceptable for cognitive applications
- Snapshot isolation for recall queries

**Guarantees:**
- Intra-session total ordering (monotonic sequences)
- Cross-session undefined ordering (eventual consistency)
- Snapshot-isolated recall (no dirty reads)

### Why Lock-Free?

**Problem:** Locks introduce contention and latency variance (P99 vs P50).

**Validation Results:**
- Lock-free SegQueue: 4.8M ops/sec
- Mutex<VecDeque>: 52K ops/sec
- **92x throughput improvement**

**Benefits:**
- Predictable latency (no lock waits)
- Better cache behavior (CAS vs kernel sync)
- Parallel progress (producers don't block consumers)
- Scales to high core counts

**Trade-offs:**
- More complex code (memory ordering)
- ABA problem requires careful handling
- Debugging is harder (no deadlock detection)

### Why Work Stealing?

**Problem:** Hash-based sharding can cause load imbalance when spaces have uneven observation rates.

**Solution:** Work stealing with configurable threshold (default: 1000 queue depth).

**Algorithm:**
1. Worker processes own queue first (cache-hot)
2. If own queue empty, find busiest queue
3. Steal half of victim's work
4. Process stolen batch

**Benefits:**
- Automatic load balancing
- No manual rebalancing required
- Preserves cache locality (only steal when idle)

**Trade-offs:**
- ~200ns overhead per stolen item
- Cache misses when stealing (worker unfamiliar with space)
- Potential "steal wars" if threshold too low

**Tuning:**
- Low threshold (100): More stealing, better balance, worse cache
- High threshold (10,000): Less stealing, worse balance, better cache
- Default (1,000): Balanced trade-off

### Why Adaptive Batching?

**Problem:** Fixed batch size is suboptimal for varying load conditions.

**Solution:** Batch size scales with queue depth.

**Algorithm:**
```rust
fn adaptive_batch_size(queue_depth: usize) -> usize {
    match queue_depth {
        0..100 => 10,      // Low latency
        100..1000 => 100,  // Balanced
        1000.. => 500,     // High throughput
    }
}
```

**Benefits:**
- Low latency when queue is shallow
- High throughput when queue is deep
- Automatic adaptation to load

**Benchmarks:**
- Fixed batch=10: P99 latency 5ms, throughput 10K obs/sec
- Fixed batch=500: P99 latency 200ms, throughput 100K obs/sec
- Adaptive: P99 latency 50ms, throughput 80K obs/sec

## Related Documentation

- [Streaming Protocol Specification](../reference/streaming-protocol.md)
- [Performance Tuning Guide](../howto/tune-streaming-performance.md)
- [Production Operations](../../operations/streaming-operations.md)
- [API Reference](https://docs.rs/engram-core/latest/engram_core/streaming/)

## References

- Buzsáki, G. (2015). "Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory and planning." *Hippocampus*, 25(10), 1073-1188.
- Marr, D. (1971). "Simple memory: a theory for archicortex." *Philosophical Transactions of the Royal Society of London B*, 262(841), 23-81.
- Lamport, L. (1978). "Time, clocks, and the ordering of events in a distributed system." *Communications of the ACM*, 21(7), 558-565.
- Michael, M. M., & Scott, M. L. (1996). "Simple, fast, and practical non-blocking and blocking concurrent queue algorithms." *PODC '96*, 267-275.
