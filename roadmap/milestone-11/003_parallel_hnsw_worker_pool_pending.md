# Task 003: Parallel HNSW Worker Pool

**Status:** Pending
**Estimated Effort:** 4 days
**Dependencies:** Task 002 (observation queue)
**Priority:** CRITICAL PATH

## Objective

Implement multi-threaded HNSW update workers with memory-space sharding for parallelism and work stealing for load balancing. Target: 40K-100K insertions/sec with 4-8 worker threads.

## Architecture

### Single-Threaded Bottleneck Analysis

**Current limitation:**
- `MemoryStore` has one background HNSW worker thread
- HNSW insertion is CPU-bound: ~10K insertions/sec per core
- Streaming target: 100K observations/sec
- **Gap:** 10x throughput needed

**Solution: Parallel workers with space-based sharding**

### Space-Based Sharding

**Key insight:** Memory spaces are independent - no cross-space HNSW operations.

```
Observation → hash(memory_space_id) % N → Worker[i] → HNSW index for that space
```

**Advantages:**
- Zero contention between workers (no shared HNSW index)
- Natural load distribution (spaces are independent tenants)
- Scales linearly with worker count (up to number of active spaces)

**Challenge:** Load imbalance when one space gets 10x more observations.

**Solution:** Work stealing.

### Work Stealing for Load Balancing

**Problem scenario:**
- 4 workers, 10 memory spaces
- Space A gets 50K obs/sec, other spaces get 1K obs/sec each
- Worker assigned to Space A is saturated, others idle

**Work stealing protocol:**

1. Each worker has own queue (space-sharded)
2. When worker idle (own queue empty), check other queues
3. If other queue depth > threshold, steal half its work
4. Process stolen work, then check own queue again

**Implementation:**

```rust
pub struct WorkerPool {
    workers: Vec<Worker>,
    queues: Vec<Arc<ObservationQueue>>,
    shutdown: Arc<AtomicBool>,
}

struct Worker {
    id: usize,
    own_queue: Arc<ObservationQueue>,
    all_queues: Arc<Vec<Arc<ObservationQueue>>>,  // For work stealing
    hnsw_index: Arc<CognitiveHnswIndex>,
    memory_store: Arc<MemoryStore>,
    stats: WorkerStats,
}

impl Worker {
    fn run(&self) {
        while !self.shutdown.load(Ordering::Relaxed) {
            // 1. Check own queue first
            if let Some(batch) = self.own_queue.dequeue_batch(100) {
                self.process_batch(batch);
                continue;
            }

            // 2. Own queue empty - try work stealing
            if let Some(batch) = self.steal_work() {
                self.stats.stolen_batches.fetch_add(1, Ordering::Relaxed);
                self.process_batch(batch);
                continue;
            }

            // 3. No work anywhere - sleep briefly
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn steal_work(&self) -> Option<Vec<QueuedObservation>> {
        // Find queue with highest depth
        let mut max_depth = 0usize;
        let mut victim_idx = None;

        for (i, queue) in self.all_queues.iter().enumerate() {
            if i == self.id {
                continue; // Skip own queue
            }

            let depth = queue.total_depth();
            if depth > max_depth && depth > STEAL_THRESHOLD {
                max_depth = depth;
                victim_idx = Some(i);
            }
        }

        // Steal half of victim's work
        if let Some(idx) = victim_idx {
            let steal_count = max_depth / 2;
            Some(self.all_queues[idx].dequeue_batch(steal_count))
        } else {
            None
        }
    }

    fn process_batch(&self, batch: Vec<QueuedObservation>) {
        let start = Instant::now();

        // Convert to Memory objects
        let memories: Vec<Arc<Memory>> = batch.iter()
            .map(|obs| Arc::new(Memory::from_episode(&obs.episode)))
            .collect();

        // Batch insert into HNSW (Task 004)
        if let Err(e) = self.hnsw_index.insert_batch(&memories) {
            eprintln!("HNSW batch insert failed: {}", e);
            self.stats.failed_batches.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Update stats
        let elapsed = start.elapsed();
        self.stats.processed_observations.fetch_add(batch.len() as u64, Ordering::Relaxed);
        self.stats.total_processing_time_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }
}

const STEAL_THRESHOLD: usize = 1000;  // Only steal if victim has > 1K items

struct WorkerStats {
    processed_observations: AtomicU64,
    stolen_batches: AtomicU64,
    failed_batches: AtomicU64,
    total_processing_time_ns: AtomicU64,
}
```

### Space-to-Worker Assignment

**Hash-based sharding:**

```rust
fn assign_worker(memory_space_id: &MemorySpaceId, num_workers: usize) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    memory_space_id.hash(&mut hasher);
    let hash = hasher.finish();

    (hash as usize) % num_workers
}
```

**Ensures:**
- Same space always maps to same worker (cache locality)
- Load distributes evenly across workers (hash uniformity)
- No coordination needed (deterministic assignment)

### Batch Coalescing

**Observation:** Small batches (< 10 items) have high overhead.

**Strategy:** Adaptive batching based on queue depth.

```rust
fn select_batch_size(&self) -> usize {
    let depth = self.own_queue.total_depth();

    if depth < 100 {
        10      // Low load: small batches for low latency
    } else if depth < 1000 {
        100     // Medium load: balance latency vs throughput
    } else {
        500     // High load: maximize throughput
    }
}
```

**Trade-off:**
- Small batch: Low latency (10ms), moderate throughput (10K/sec)
- Large batch: Higher latency (50ms), high throughput (100K/sec)

### Graceful Shutdown

**Requirement:** Drain all queued observations before shutdown.

```rust
impl WorkerPool {
    pub fn shutdown(&self, timeout: Duration) -> Result<(), ShutdownError> {
        // Signal shutdown
        self.shutdown.store(true, Ordering::SeqCst);

        let deadline = Instant::now() + timeout;

        // Wait for queues to drain
        loop {
            let total_depth: usize = self.queues.iter()
                .map(|q| q.total_depth())
                .sum();

            if total_depth == 0 {
                // All queues empty - safe to shutdown
                break;
            }

            if Instant::now() > deadline {
                return Err(ShutdownError::Timeout {
                    remaining_items: total_depth,
                });
            }

            std::thread::sleep(Duration::from_millis(100));
        }

        // Join worker threads
        for worker in &self.workers {
            worker.thread_handle.join().unwrap();
        }

        Ok(())
    }
}
```

## Files to Create

- `engram-core/src/streaming/worker_pool.rs` (600 lines)
- `engram-core/src/streaming/work_stealing.rs` (250 lines)
- `engram-core/src/streaming/worker_stats.rs` (150 lines)

## Files to Modify

- `engram-core/src/store.rs` (replace single worker with pool, ~100 line changes)
- `engram-core/src/streaming/mod.rs` (export worker pool types)

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_space_to_worker_assignment() {
    let space1 = MemorySpaceId::new("space1");
    let space2 = MemorySpaceId::new("space2");

    // Same space always maps to same worker
    assert_eq!(assign_worker(&space1, 4), assign_worker(&space1, 4));

    // Different spaces may map to different workers
    // (Not guaranteed, but likely with good hash function)
    let workers1: Vec<_> = (0..100)
        .map(|i| assign_worker(&MemorySpaceId::new(&format!("space{}", i)), 4))
        .collect();

    // Should distribute across all 4 workers
    assert!(workers1.contains(&0));
    assert!(workers1.contains(&1));
    assert!(workers1.contains(&2));
    assert!(workers1.contains(&3));
}

#[test]
fn test_work_stealing() {
    let pool = WorkerPool::new(4, QueueConfig::default());

    // Load one queue heavily
    for i in 0..5000 {
        pool.enqueue(MemorySpaceId::new("space0"), episode(i), i, ObservationPriority::Normal);
    }

    // Give workers time to steal work
    std::thread::sleep(Duration::from_secs(1));

    // Check that work was distributed
    let stats = pool.worker_stats();
    let total_stolen: u64 = stats.iter().map(|s| s.stolen_batches).sum();
    assert!(total_stolen > 0, "Work stealing should have occurred");
}

#[test]
fn test_graceful_shutdown() {
    let pool = WorkerPool::new(4, QueueConfig::default());

    // Enqueue 10K observations
    for i in 0..10_000 {
        pool.enqueue(MemorySpaceId::new(&format!("space{}", i % 10)), episode(i), i, ObservationPriority::Normal);
    }

    // Shutdown with 5s timeout
    let result = pool.shutdown(Duration::from_secs(5));
    assert!(result.is_ok(), "Should complete within timeout");

    // Verify all processed
    let stats = pool.worker_stats();
    let total_processed: u64 = stats.iter().map(|s| s.processed_observations).sum();
    assert_eq!(total_processed, 10_000);
}
```

### Performance Benchmarks

```rust
#[bench]
fn bench_worker_pool_throughput(b: &mut Bencher) {
    let pool = WorkerPool::new(4, QueueConfig::default());
    let spaces: Vec<_> = (0..10).map(|i| MemorySpaceId::new(&format!("space{}", i))).collect();

    b.iter(|| {
        // Simulate streaming load
        for i in 0..10_000 {
            let space = &spaces[i % 10];
            pool.enqueue(space.clone(), episode(i), i, ObservationPriority::Normal);
        }

        // Wait for processing
        while pool.total_queue_depth() > 0 {
            std::thread::yield_now();
        }
    });

    // Target: 10K observations processed in < 1 second (100K/sec with 4 workers = 40K/sec per worker)
}

#[bench]
fn bench_work_stealing_overhead(b: &mut Bencher) {
    let pool = WorkerPool::new(4, QueueConfig::default());

    b.iter(|| {
        // Trigger work stealing by loading one queue
        for i in 0..1000 {
            pool.enqueue(MemorySpaceId::new("space0"), episode(i), i, ObservationPriority::Normal);
        }

        // Measure time until work distributed
        let start = Instant::now();
        while pool.queues[0].total_depth() > 100 {
            std::thread::yield_now();
        }
        black_box(start.elapsed());
    });

    // Target: Work stealing activates within 10ms
}
```

### Load Tests

```rust
#[test]
fn load_test_sustained_100k_ops() {
    let pool = WorkerPool::new(4, QueueConfig::default());
    let start = Instant::now();
    let target_duration = Duration::from_secs(60);
    let target_rate = 100_000; // 100K obs/sec

    let mut count = 0u64;
    while start.elapsed() < target_duration {
        let space = MemorySpaceId::new(&format!("space{}", count % 100));
        pool.enqueue(space, episode(count), count, ObservationPriority::Normal);
        count += 1;

        // Rate limiting: sleep if ahead of target
        let expected_count = (start.elapsed().as_secs_f64() * target_rate as f64) as u64;
        if count > expected_count {
            std::thread::sleep(Duration::from_micros(10));
        }
    }

    // Wait for queue to drain
    while pool.total_queue_depth() > 0 {
        std::thread::sleep(Duration::from_millis(100));
    }

    let elapsed = start.elapsed();
    let actual_rate = count as f64 / elapsed.as_secs_f64();

    println!("Processed {} observations in {:?} ({:.0} obs/sec)", count, elapsed, actual_rate);

    // Verify target achieved
    assert!(actual_rate >= 100_000.0, "Should sustain 100K obs/sec");

    // Verify worker utilization
    let stats = pool.worker_stats();
    for (i, stat) in stats.iter().enumerate() {
        let processed = stat.processed_observations;
        println!("Worker {}: {} observations", i, processed);
        assert!(processed > 0, "Worker {} should have processed observations", i);
    }

    // Load imbalance should be < 20%
    let max_processed = stats.iter().map(|s| s.processed_observations).max().unwrap();
    let min_processed = stats.iter().map(|s| s.processed_observations).min().unwrap();
    let imbalance = (max_processed as f64 / min_processed as f64) - 1.0;
    assert!(imbalance < 0.2, "Load imbalance should be < 20%, got {:.1}%", imbalance * 100.0);
}
```

## Acceptance Criteria

1. 4-worker pool sustains 40K insertions/sec (10K per worker)
2. 8-worker pool sustains 80K insertions/sec (linear scaling)
3. Load imbalance < 20% (max_worker_throughput / min_worker_throughput < 1.2)
4. Work stealing activates when one queue > 1000 and others < 100
5. Graceful shutdown: all queued observations processed within 5s
6. No memory leaks (valgrind clean after 1M observations)
7. Worker stats accurate (processed count matches enqueued count)

## Performance Targets

- Throughput: 40K obs/sec with 4 workers, 80K with 8 workers
- Latency: P99 observation → indexed < 100ms
- Work stealing overhead: < 10ms to detect imbalance and steal
- Shutdown time: < 5s for 10K queued observations

## Dependencies

- Task 002 (observation queue provides work items)
- Task 004 (batch HNSW insert, can develop in parallel)

## Next Steps

- Task 004 implements the batch HNSW insert that workers use
- Task 006 integrates worker pool stats with backpressure detection
