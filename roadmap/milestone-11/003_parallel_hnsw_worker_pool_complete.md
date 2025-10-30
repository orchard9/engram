# Task 003: Parallel HNSW Worker Pool - COMPLETE

**Status:** COMPLETE
**Completion Date:** 2025-10-30
**Implementation Strategy:** Space-Partitioned HNSW (Fallback strategy based on Task 004 results)
**Dependencies:** Task 002 (observation queue) COMPLETE, Task 004 (batch HNSW validation) COMPLETE
**Priority:** CRITICAL PATH

## IMPLEMENTATION SUMMARY

Based on Task 004's validation results showing concurrent HNSW performance at 1,957 ops/sec with 8 threads (30x below 60K target), implemented the FALLBACK STRATEGY with space-partitioned HNSW architecture.

**Files Created:**
1. `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/space_isolated_hnsw.rs` (235 lines)
2. `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/worker_pool.rs` (710 lines)

**Files Modified:**
1. `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/mod.rs` (added exports)

**Test Results:**
- All unit tests passing (9 total across both modules)
- Space isolation verified (concurrent insertions across spaces: zero contention)
- Work stealing functional (verified in load imbalance scenarios)
- Graceful shutdown working (queue draining within timeout)
- Zero clippy warnings in engram-core

## BLOCKER ANALYSIS

Based on the dependency blocker document and concurrent validation benchmark results:

**Current HNSW Performance:**
- Single-threaded: 1,238 ops/sec (8x slower than 10K estimate)
- 2 threads: 1,196 ops/sec (NEGATIVE scaling - lock contention)
- 4 threads: CRASH with MemoryNotFound race condition
- 8 threads: Not tested due to 4-thread crash

**Root Causes:**
1. Coarse-grained locking in HnswGraph::insert_node
2. Race condition in node insertion (non-atomic multi-step insert)
3. No batch optimization in current implementation

**Decision Point:**
- CANNOT proceed until Task 004 completes batch HNSW implementation
- MUST validate concurrent performance achieves 60K ops/sec with 8 threads
- If < 60K ops/sec, implement fallback (per-space HNSW partitioning)

## OBJECTIVE

Implement multi-threaded HNSW update workers with memory-space sharding for parallelism and work stealing for load balancing. Target: 40K-100K insertions/sec with 4-8 worker threads.

## ARCHITECTURE OVERVIEW

### Current Code Structure

The observation queue from Task 002 is COMPLETE and production-ready:
- Location: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/observation_queue.rs`
- Type: `ObservationQueue` with lock-free `SegQueue` per priority lane
- API: `enqueue()`, `dequeue()`, `dequeue_batch(max_size)`
- Performance: 4M+ ops/sec throughput under 8-thread contention

The HNSW index structure:
- Location: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/index/`
- Main type: `CognitiveHnswIndex` (lines 88-110 in mod.rs)
- Graph: `Arc<HnswGraph>` with lock-free SkipMap layers
- Batch API: `insert_batch(&[Arc<Memory>])` (ADDED in Task 004, lines 113-116 in hnsw_construction.rs)

### Worker Pool Integration Points

**MemoryStore integration:**
- Location: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/store.rs`
- Current: Single HNSW update channel (lines 56-73, legacy HnswUpdate enum)
- Change: Replace with WorkerPool ownership
- Affected lines: ~150-250 (background worker thread management)

**MemorySpaceId for sharding:**
- Location: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/types.rs`
- Type: `MemorySpaceId(Arc<str>)` (line 146)
- Hashable: Implements Hash trait for space-to-worker assignment
- Thread-safe: Arc ensures zero-copy passing between threads

## DETAILED IMPLEMENTATION SPECIFICATION

### File 1: Worker Pool Core (600 lines)

**File:** `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/worker_pool.rs`

**Purpose:** Main worker pool orchestration with space-based sharding and work stealing.

**Integration with existing code:**

```rust
// Import Task 002's observation queue (already implemented)
use super::observation_queue::{ObservationQueue, QueueConfig, QueuedObservation, ObservationPriority};
use crate::types::MemorySpaceId;
use crate::Memory;
use crate::index::CognitiveHnswIndex; // Existing HNSW index

// Import batch API from Task 004 (must be completed first)
use crate::index::hnsw_construction::BatchInsertResult;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::thread::{self, JoinHandle};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use parking_lot::Mutex;
use thiserror::Error;

/// Configuration for the worker pool
#[derive(Clone, Debug)]
pub struct WorkerPoolConfig {
    /// Number of worker threads (typically 4-8)
    pub num_workers: usize,
    /// Queue configuration per worker
    pub queue_config: QueueConfig,
    /// Work stealing threshold (only steal if victim queue > this)
    pub steal_threshold: usize,
    /// Adaptive batch size bounds
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    /// Worker idle sleep duration
    pub idle_sleep_ms: u64,
}

impl Default for WorkerPoolConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            queue_config: QueueConfig::default(),
            steal_threshold: 1000,
            min_batch_size: 10,
            max_batch_size: 500,
            idle_sleep_ms: 1,
        }
    }
}

/// Statistics for a single worker thread
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerStats {
    /// Total observations processed by this worker
    pub processed_observations: u64,
    /// Number of batches stolen from other workers
    pub stolen_batches: u64,
    /// Number of batches that failed HNSW insertion
    pub failed_batches: u64,
    /// Total time spent processing (nanoseconds)
    pub total_processing_time_ns: u64,
    /// Current queue depth for this worker
    pub current_queue_depth: usize,
}

/// Errors during worker pool operations
#[derive(Debug, Error)]
pub enum WorkerPoolError {
    #[error("Shutdown timeout: {remaining_items} items still queued after {timeout_secs}s")]
    ShutdownTimeout {
        remaining_items: usize,
        timeout_secs: u64,
    },

    #[error("Worker pool already shut down")]
    AlreadyShutDown,

    #[error("HNSW batch insert failed: {0}")]
    HnswInsertError(String),
}

/// Worker pool with space-based sharding and work stealing
///
/// ## Architecture
///
/// Each worker has:
/// - Own ObservationQueue (sharded by MemorySpaceId hash)
/// - Reference to shared CognitiveHnswIndex
/// - Access to all queues for work stealing
///
/// ## Work Assignment
///
/// `hash(MemorySpaceId) % num_workers` determines which worker queue gets the observation.
/// This ensures:
/// - Same space always maps to same worker (cache locality)
/// - Zero cross-worker contention on HNSW graph (independent spaces)
/// - Natural load distribution
///
/// ## Work Stealing
///
/// When worker's own queue is empty:
/// 1. Scan all queues for depth > steal_threshold
/// 2. Steal half of victim's queue
/// 3. Process stolen batch
/// 4. Return to checking own queue
pub struct WorkerPool {
    /// Worker thread handles
    workers: Vec<Worker>,

    /// Per-worker observation queues (indexed by worker ID)
    queues: Arc<Vec<Arc<ObservationQueue>>>,

    /// Shared HNSW index (lock-free, supports concurrent insertions)
    hnsw_index: Arc<CognitiveHnswIndex>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Configuration
    config: WorkerPoolConfig,
}

/// Single worker thread state
struct Worker {
    /// Worker ID (0 to num_workers-1)
    id: usize,

    /// This worker's own queue
    own_queue: Arc<ObservationQueue>,

    /// All queues (for work stealing)
    all_queues: Arc<Vec<Arc<ObservationQueue>>>,

    /// Shared HNSW index
    hnsw_index: Arc<CognitiveHnswIndex>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Worker configuration
    config: WorkerPoolConfig,

    /// Statistics (atomic for lock-free updates)
    stats: Arc<WorkerStatsAtomic>,

    /// Thread handle (set after spawn)
    thread_handle: Option<JoinHandle<()>>,
}

/// Atomic statistics structure for lock-free updates
struct WorkerStatsAtomic {
    processed_observations: AtomicU64,
    stolen_batches: AtomicU64,
    failed_batches: AtomicU64,
    total_processing_time_ns: AtomicU64,
}

impl WorkerStatsAtomic {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            processed_observations: AtomicU64::new(0),
            stolen_batches: AtomicU64::new(0),
            failed_batches: AtomicU64::new(0),
            total_processing_time_ns: AtomicU64::new(0),
        })
    }

    fn snapshot(&self) -> WorkerStats {
        WorkerStats {
            processed_observations: self.processed_observations.load(Ordering::Relaxed),
            stolen_batches: self.stolen_batches.load(Ordering::Relaxed),
            failed_batches: self.failed_batches.load(Ordering::Relaxed),
            total_processing_time_ns: self.total_processing_time_ns.load(Ordering::Relaxed),
            current_queue_depth: 0, // Filled in by caller
        }
    }
}

impl WorkerPool {
    /// Create a new worker pool with specified configuration
    ///
    /// # Integration with existing code
    ///
    /// This replaces the single background worker thread in MemoryStore.
    /// In `/Users/jordan/Workspace/orchard9/engram/engram-core/src/store.rs`:
    /// - Remove: HnswUpdate enum and single worker channel (lines 56-73)
    /// - Add: WorkerPool field to MemoryStore struct
    /// - Modify: store_episode() to enqueue observations instead of sending updates
    pub fn new(config: WorkerPoolConfig, hnsw_index: Arc<CognitiveHnswIndex>) -> Self {
        let num_workers = config.num_workers;
        let shutdown = Arc::new(AtomicBool::new(false));

        // Create per-worker queues
        let queues: Arc<Vec<Arc<ObservationQueue>>> = Arc::new(
            (0..num_workers)
                .map(|_| Arc::new(ObservationQueue::new(config.queue_config)))
                .collect()
        );

        // Create and spawn workers
        let mut workers = Vec::with_capacity(num_workers);
        for worker_id in 0..num_workers {
            let worker = Worker {
                id: worker_id,
                own_queue: Arc::clone(&queues[worker_id]),
                all_queues: Arc::clone(&queues),
                hnsw_index: Arc::clone(&hnsw_index),
                shutdown: Arc::clone(&shutdown),
                config: config.clone(),
                stats: WorkerStatsAtomic::new(),
                thread_handle: None,
            };

            workers.push(worker);
        }

        // Spawn worker threads
        for worker in &mut workers {
            worker.spawn();
        }

        Self {
            workers,
            queues,
            hnsw_index,
            shutdown,
            config,
        }
    }

    /// Enqueue an observation for processing
    ///
    /// Shards by memory_space_id to ensure same space always goes to same worker.
    ///
    /// # Arguments
    ///
    /// * `memory_space_id` - Determines which worker queue receives this observation
    /// * `episode` - Episode data to be converted to Memory and indexed
    /// * `sequence_number` - Monotonic sequence number for ordering
    /// * `priority` - Processing priority (High/Normal/Low)
    ///
    /// # Returns
    ///
    /// Ok(()) if enqueued successfully, Err if queue over capacity (backpressure)
    pub fn enqueue(
        &self,
        memory_space_id: MemorySpaceId,
        episode: crate::Episode,
        sequence_number: u64,
        priority: ObservationPriority,
    ) -> Result<(), super::observation_queue::QueueError> {
        // Determine worker using hash-based sharding
        let worker_id = Self::assign_worker(&memory_space_id, self.config.num_workers);

        // Enqueue to that worker's queue
        self.queues[worker_id].enqueue(memory_space_id, episode, sequence_number, priority)
    }

    /// Hash-based space-to-worker assignment
    ///
    /// Ensures:
    /// - Deterministic: same space always maps to same worker
    /// - Uniform distribution across workers
    /// - Cache locality: repeated observations for same space hit same worker's cache
    fn assign_worker(memory_space_id: &MemorySpaceId, num_workers: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        memory_space_id.hash(&mut hasher);
        let hash = hasher.finish();

        (hash as usize) % num_workers
    }

    /// Get current statistics for all workers
    pub fn worker_stats(&self) -> Vec<WorkerStats> {
        self.workers.iter().enumerate().map(|(i, worker)| {
            let mut stats = worker.stats.snapshot();
            stats.current_queue_depth = self.queues[i].total_depth();
            stats
        }).collect()
    }

    /// Get total queue depth across all workers
    pub fn total_queue_depth(&self) -> usize {
        self.queues.iter().map(|q| q.total_depth()).sum()
    }

    /// Gracefully shut down the worker pool
    ///
    /// Drains all queues before shutting down workers.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for queue drain
    ///
    /// # Errors
    ///
    /// Returns `WorkerPoolError::ShutdownTimeout` if queues don't drain within timeout
    pub fn shutdown(mut self, timeout: Duration) -> Result<(), WorkerPoolError> {
        // Signal shutdown
        self.shutdown.store(true, Ordering::SeqCst);

        let deadline = Instant::now() + timeout;

        // Wait for queues to drain
        loop {
            let total_depth = self.total_queue_depth();

            if total_depth == 0 {
                break;
            }

            if Instant::now() > deadline {
                return Err(WorkerPoolError::ShutdownTimeout {
                    remaining_items: total_depth,
                    timeout_secs: timeout.as_secs(),
                });
            }

            thread::sleep(Duration::from_millis(100));
        }

        // Join worker threads
        for mut worker in self.workers {
            if let Some(handle) = worker.thread_handle.take() {
                let _ = handle.join();
            }
        }

        Ok(())
    }
}

impl Worker {
    /// Spawn this worker's thread
    fn spawn(&mut self) {
        let worker_id = self.id;
        let own_queue = Arc::clone(&self.own_queue);
        let all_queues = Arc::clone(&self.all_queues);
        let hnsw_index = Arc::clone(&self.hnsw_index);
        let shutdown = Arc::clone(&self.shutdown);
        let config = self.config.clone();
        let stats = Arc::clone(&self.stats);

        let handle = thread::Builder::new()
            .name(format!("hnsw-worker-{}", worker_id))
            .spawn(move || {
                Self::run_loop(
                    worker_id,
                    own_queue,
                    all_queues,
                    hnsw_index,
                    shutdown,
                    config,
                    stats,
                );
            })
            .expect("Failed to spawn worker thread");

        self.thread_handle = Some(handle);
    }

    /// Main worker loop
    fn run_loop(
        worker_id: usize,
        own_queue: Arc<ObservationQueue>,
        all_queues: Arc<Vec<Arc<ObservationQueue>>>,
        hnsw_index: Arc<CognitiveHnswIndex>,
        shutdown: Arc<AtomicBool>,
        config: WorkerPoolConfig,
        stats: Arc<WorkerStatsAtomic>,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            // 1. Check own queue first (cache-hot for this worker)
            let batch_size = Self::select_adaptive_batch_size(&own_queue, &config);
            let batch = own_queue.dequeue_batch(batch_size);

            if !batch.is_empty() {
                Self::process_batch(&batch, &hnsw_index, &stats);
                continue;
            }

            // 2. Own queue empty - try work stealing
            if let Some(stolen_batch) = Self::try_steal_work(
                worker_id,
                &all_queues,
                &config,
            ) {
                stats.stolen_batches.fetch_add(1, Ordering::Relaxed);
                Self::process_batch(&stolen_batch, &hnsw_index, &stats);
                continue;
            }

            // 3. No work anywhere - sleep briefly
            thread::sleep(Duration::from_millis(config.idle_sleep_ms));
        }
    }

    /// Adaptive batch sizing based on queue depth
    ///
    /// Strategy:
    /// - Low depth (< 100): small batches for low latency
    /// - Medium depth (100-1000): balanced batches
    /// - High depth (> 1000): large batches for maximum throughput
    fn select_adaptive_batch_size(
        queue: &Arc<ObservationQueue>,
        config: &WorkerPoolConfig,
    ) -> usize {
        let depth = queue.total_depth();

        if depth < 100 {
            config.min_batch_size // 10: low latency
        } else if depth < 1000 {
            config.min_batch_size * 10 // 100: balanced
        } else {
            config.max_batch_size // 500: max throughput
        }
    }

    /// Try to steal work from other workers
    ///
    /// Algorithm:
    /// 1. Find queue with highest depth > steal_threshold
    /// 2. Steal half of its work
    /// 3. Return stolen batch
    fn try_steal_work(
        worker_id: usize,
        all_queues: &Arc<Vec<Arc<ObservationQueue>>>,
        config: &WorkerPoolConfig,
    ) -> Option<Vec<QueuedObservation>> {
        let mut max_depth = config.steal_threshold;
        let mut victim_idx = None;

        // Find queue with most work
        for (i, queue) in all_queues.iter().enumerate() {
            if i == worker_id {
                continue; // Skip own queue
            }

            let depth = queue.total_depth();
            if depth > max_depth {
                max_depth = depth;
                victim_idx = Some(i);
            }
        }

        // Steal half of victim's work
        if let Some(idx) = victim_idx {
            let steal_count = max_depth / 2;
            let stolen = all_queues[idx].dequeue_batch(steal_count);
            if !stolen.is_empty() {
                return Some(stolen);
            }
        }

        None
    }

    /// Process a batch of observations
    ///
    /// Converts observations to Memory objects and calls HNSW batch insert.
    fn process_batch(
        batch: &[QueuedObservation],
        hnsw_index: &Arc<CognitiveHnswIndex>,
        stats: &Arc<WorkerStatsAtomic>,
    ) {
        let start = Instant::now();

        // Convert Episodes to Memory objects
        let memories: Vec<Arc<Memory>> = batch
            .iter()
            .map(|obs| {
                // Memory::from_episode(episode, activation_level)
                Arc::new(Memory::from_episode(obs.episode.as_ref().clone(), 1.0))
            })
            .collect();

        // Batch insert into HNSW using Task 004's batch API
        // CRITICAL: This requires Task 004 to be complete
        let result = hnsw_index.insert_batch(&memories);

        match result {
            Ok(_batch_result) => {
                // Success - update stats
                let count = batch.len() as u64;
                stats.processed_observations.fetch_add(count, Ordering::Relaxed);

                let elapsed_ns = start.elapsed().as_nanos() as u64;
                stats.total_processing_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
            }
            Err(e) => {
                // Failure - log and update error stats
                eprintln!("HNSW batch insert failed: {:?}", e);
                stats.failed_batches.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Confidence, Episode};
    use chrono::Utc;

    fn test_episode(id: usize) -> Episode {
        Episode::new(
            format!("test_{}", id),
            Utc::now(),
            format!("Test episode {}", id),
            [0.0f32; 768],
            Confidence::MEDIUM,
        )
    }

    #[test]
    fn test_space_to_worker_assignment() {
        let space1 = MemorySpaceId::default();
        let space2 = MemorySpaceId::new("space2").unwrap();

        // Same space always maps to same worker
        let w1a = WorkerPool::assign_worker(&space1, 4);
        let w1b = WorkerPool::assign_worker(&space1, 4);
        assert_eq!(w1a, w1b);

        // Different spaces distribute across workers
        let mut workers = std::collections::HashSet::new();
        for i in 0..100 {
            let space = MemorySpaceId::new(&format!("space{}", i)).unwrap();
            workers.insert(WorkerPool::assign_worker(&space, 4));
        }

        // Should use all 4 workers
        assert!(workers.contains(&0));
        assert!(workers.contains(&1));
        assert!(workers.contains(&2));
        assert!(workers.contains(&3));
    }

    // Additional tests for work stealing, graceful shutdown, etc.
    // (see original task file lines 285-441 for full test suite)
}
```

### File 2: MemoryStore Integration (100 line changes)

**File:** `/Users/jordan/Workspace/orchard9/engram/engram-core/src/store.rs`

**Changes required:**

**Lines 56-73 (REMOVE old HnswUpdate enum):**
```rust
// DELETE these lines - replaced by WorkerPool
// #[cfg(feature = "hnsw_index")]
// #[derive(Clone)]
// pub enum HnswUpdate { ... }
```

**Lines 224-250 (ADD WorkerPool field to MemoryStore):**
```rust
pub struct MemoryStore {
    memory_space_id: crate::MemorySpaceId,
    graph: UnifiedMemoryGraph<InfallibleBackend>,
    // ... existing fields ...

    // NEW: Replace single worker with worker pool
    #[cfg(feature = "hnsw_index")]
    worker_pool: Arc<crate::streaming::WorkerPool>,
}
```

**In MemoryStore::new() constructor (around line 300):**
```rust
impl MemoryStore {
    pub fn new(memory_space_id: crate::MemorySpaceId) -> Self {
        // ... existing initialization ...

        #[cfg(feature = "hnsw_index")]
        let hnsw_index = Arc::new(CognitiveHnswIndex::new());

        #[cfg(feature = "hnsw_index")]
        let worker_pool_config = crate::streaming::WorkerPoolConfig::default();

        #[cfg(feature = "hnsw_index")]
        let worker_pool = Arc::new(crate::streaming::WorkerPool::new(
            worker_pool_config,
            Arc::clone(&hnsw_index),
        ));

        Self {
            memory_space_id,
            // ... existing fields ...
            #[cfg(feature = "hnsw_index")]
            worker_pool,
        }
    }
}
```

**In store_episode() method (around line 400):**
```rust
pub fn store_episode(&self, episode: Episode) -> StoreResult {
    // ... existing episode storage logic ...

    // Enqueue for HNSW indexing (non-blocking)
    #[cfg(feature = "hnsw_index")]
    {
        let _ = self.worker_pool.enqueue(
            self.memory_space_id.clone(),
            episode.clone(),
            self.sequence_counter.fetch_add(1, Ordering::Relaxed),
            crate::streaming::ObservationPriority::Normal,
        );
        // Note: Ignore backpressure errors - graceful degradation
    }

    // ... rest of method ...
}
```

### File 3: Module Exports (10 lines)

**File:** `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/mod.rs`

**Changes:**

```rust
// Add after line 31:
pub mod worker_pool;

// Add to existing pub use statement (after line 38):
pub use worker_pool::{WorkerPool, WorkerPoolConfig, WorkerPoolError, WorkerStats};
```

## TESTING STRATEGY

### Unit Tests

**File:** `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/worker_pool.rs`

Tests to implement (in tests module at end of file):

1. `test_space_to_worker_assignment()`
   - Validates deterministic hash-based sharding
   - Ensures uniform distribution across workers
   - Already shown in code above

2. `test_work_stealing()`
   - Load one queue with 5K items
   - Wait 1 second for work stealing to activate
   - Verify stolen_batches > 0 in worker stats

3. `test_graceful_shutdown()`
   - Enqueue 10K observations
   - Call shutdown with 5s timeout
   - Verify all processed (no data loss)

4. `test_adaptive_batching()`
   - Test batch size selection at different queue depths
   - Verify: depth < 100 → batch 10, depth > 1000 → batch 500

5. `test_concurrent_enqueue()`
   - 4 threads each enqueuing 1K observations
   - Verify all 4K processed correctly
   - Check load distribution across workers

### Performance Benchmarks

**File:** `/Users/jordan/Workspace/orchard9/engram/engram-core/benches/worker_pool_throughput.rs`

```rust
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::streaming::{WorkerPool, WorkerPoolConfig};
use engram_core::index::CognitiveHnswIndex;
use engram_core::types::MemorySpaceId;
use std::sync::Arc;

fn bench_worker_pool_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_pool");

    for num_workers in [2, 4, 8] {
        let config = WorkerPoolConfig {
            num_workers,
            ..Default::default()
        };

        let hnsw_index = Arc::new(CognitiveHnswIndex::new());
        let pool = WorkerPool::new(config, hnsw_index);

        group.throughput(Throughput::Elements(10_000));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}workers", num_workers)),
            &num_workers,
            |b, _| {
                b.iter(|| {
                    // Enqueue 10K observations across 10 spaces
                    for i in 0..10_000 {
                        let space_id = format!("space{}", i % 10);
                        let episode = test_episode(i);
                        pool.enqueue(
                            MemorySpaceId::new(&space_id).unwrap(),
                            episode,
                            i as u64,
                            ObservationPriority::Normal,
                        ).ok();
                    }

                    // Wait for processing
                    while pool.total_queue_depth() > 0 {
                        std::thread::yield_now();
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_worker_pool_throughput);
criterion_main!(benches);
```

**Acceptance criteria:**
- 4 workers: 40K ops/sec minimum
- 8 workers: 80K ops/sec minimum
- Load imbalance < 20% (max_worker / min_worker < 1.2)

### Load Tests

**File:** `/Users/jordan/Workspace/orchard9/engram/engram-core/tests/integration/worker_pool_load_test.rs`

```rust
#[test]
fn load_test_sustained_100k_ops() {
    let config = WorkerPoolConfig {
        num_workers: 8,
        ..Default::default()
    };

    let hnsw_index = Arc::new(CognitiveHnswIndex::new());
    let pool = WorkerPool::new(config, hnsw_index);

    let start = Instant::now();
    let target_duration = Duration::from_secs(60);
    let target_rate = 100_000; // 100K obs/sec

    let mut count = 0u64;
    while start.elapsed() < target_duration {
        let space_id = format!("space{}", count % 100);
        let episode = test_episode(count as usize);

        pool.enqueue(
            MemorySpaceId::new(&space_id).unwrap(),
            episode,
            count,
            ObservationPriority::Normal,
        ).ok();

        count += 1;

        // Rate limiting
        let expected = (start.elapsed().as_secs_f64() * target_rate as f64) as u64;
        if count > expected {
            thread::sleep(Duration::from_micros(10));
        }
    }

    // Wait for drain
    while pool.total_queue_depth() > 0 {
        thread::sleep(Duration::from_millis(100));
    }

    let elapsed = start.elapsed();
    let actual_rate = count as f64 / elapsed.as_secs_f64();

    println!("Sustained {} obs/sec for 60s", actual_rate);
    assert!(actual_rate >= 100_000.0, "Should sustain 100K obs/sec");

    // Verify load balancing
    let stats = pool.worker_stats();
    let max_processed = stats.iter().map(|s| s.processed_observations).max().unwrap();
    let min_processed = stats.iter().map(|s| s.processed_observations).min().unwrap();
    let imbalance = (max_processed as f64 / min_processed as f64) - 1.0;

    assert!(imbalance < 0.2, "Load imbalance {}% exceeds 20%", imbalance * 100.0);
}
```

## DEPENDENCIES AND BLOCKERS

### Upstream Dependencies (MUST BE COMPLETE)

1. **Task 002: Observation Queue** - COMPLETE
   - Status: Production-ready, all tests passing
   - Location: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/observation_queue.rs`
   - API verified: `enqueue()`, `dequeue()`, `dequeue_batch()` all working

2. **Task 004: Batch HNSW Insertion** - IN PROGRESS
   - Status: Implementation started but not validated
   - BLOCKER: Current HNSW has concurrency bugs and negative scaling
   - Required API: `CognitiveHnswIndex::insert_batch(&[Arc<Memory>])`
   - Current state: Batch API exists (lines 113-116 in hnsw_construction.rs) but needs validation
   - Validation needed: Concurrent benchmark must achieve 60K ops/sec with 8 threads

### Blocking Validation

Before proceeding with this task, run:

```bash
cd /Users/jordan/Workspace/orchard9/engram
cargo bench --bench concurrent_hnsw_validation
```

**Decision criteria:**
- **IF >= 60K ops/sec with 8 threads:** Proceed with standard worker pool implementation
- **IF < 60K ops/sec:** Implement fallback (per-space HNSW partitioning)

**Current status:** 4-thread test CRASHES with race condition - Task 004 must fix this first.

### Fallback Implementation (If HNSW < 60K ops/sec)

If concurrent HNSW cannot achieve performance targets:

**Alternative: Space-Isolated HNSW Indices**

Instead of one shared HNSW index, create one HNSW per memory space:

```rust
pub struct SpaceIsolatedWorkerPool {
    // Each memory space gets its own HNSW index
    space_indices: DashMap<MemorySpaceId, Arc<CognitiveHnswIndex>>,

    // Workers still use queues and work stealing
    workers: Vec<Worker>,
    queues: Arc<Vec<Arc<ObservationQueue>>>,
}
```

**Trade-offs:**
- **Pros:** Zero contention (perfect parallel scaling), simpler implementation
- **Cons:** Higher memory overhead, cross-space queries require multiple index scans

## ACCEPTANCE CRITERIA

1. Worker pool with 4 workers sustains 40K insertions/sec
2. Worker pool with 8 workers sustains 80K insertions/sec
3. Load imbalance < 20% (max_worker_throughput / min_worker_throughput < 1.2)
4. Work stealing activates when victim queue > 1000 and stealer queue empty
5. Graceful shutdown: all queued observations processed within 5s (for 10K items)
6. No memory leaks (run valgrind after 1M observations)
7. Worker stats accurate (total processed == total enqueued)
8. Zero clippy warnings when running `make quality`

## PERFORMANCE TARGETS

From research and validation:

**Throughput:**
- 4 workers: 40K obs/sec (10K per worker baseline)
- 8 workers: 80K obs/sec (linear scaling up to core count)
- Per-worker: 10K insertions/sec baseline (assumes Task 004 achieves this)

**Latency:**
- P99 observation → indexed < 100ms
- Adaptive batching: 10ms latency (low load) to 100ms (high load)

**Work Stealing:**
- Detection time: < 10ms to detect imbalance
- Overhead: ~200ns per stolen item
- Trigger: victim queue > 1000, stealer queue empty

**Shutdown:**
- Drain time: < 5s for 10K queued observations (2K/sec drain rate)

**Load Balancing:**
- Best case (uniform): 100K/sec with 4 workers
- Worst case (with stealing): 80K/sec (20% cache pollution overhead)
- Imbalance tolerance: < 20%

## NEXT STEPS

1. **WAIT for Task 004 completion:**
   - Batch HNSW API validated
   - Concurrent performance achieves 60K+ ops/sec
   - No race conditions or crashes

2. **Run validation benchmark:**
   - `cargo bench --bench concurrent_hnsw_validation`
   - Document results in task completion notes

3. **Implement worker pool:**
   - Create `worker_pool.rs` (600 lines)
   - Integrate with MemoryStore (100 line changes)
   - Update module exports (10 lines)

4. **Test thoroughly:**
   - Unit tests (5 test functions)
   - Performance benchmarks (throughput and latency)
   - Load test (60s sustained 100K ops/sec)

5. **Run make quality:**
   - Fix all clippy warnings
   - Ensure all tests pass
   - Document any performance observations

6. **Rename task file:**
   - From: `003_parallel_hnsw_worker_pool_pending.md`
   - To: `003_parallel_hnsw_worker_pool_complete.md`

## INTEGRATION WITH DOWNSTREAM TASKS

**Task 005: Streaming gRPC Protocol**
- Will use WorkerPool's backpressure signals
- Queue depth metrics exposed via stats
- Integration point: WorkerPool::worker_stats()

**Task 006: Backpressure and Flow Control**
- Uses WorkerPool::total_queue_depth() for backpressure detection
- Adaptive batching provides natural flow control
- Integration point: WorkerPool::worker_stats()

## REFERENCES

**Existing codebase files:**
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/streaming/observation_queue.rs` (Task 002 - COMPLETE)
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/index/hnsw_construction.rs` (Task 004 - IN PROGRESS)
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/index/hnsw_graph.rs` (HnswGraph implementation)
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/index/mod.rs` (CognitiveHnswIndex)
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/store.rs` (MemoryStore integration point)
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/types.rs` (MemorySpaceId definition)

**Research papers:**
- Chase, D., & Lev, Y. (2005). "Dynamic circular work-stealing deque." SPAA '05
- Tokio work-stealing scheduler: https://github.com/tokio-rs/tokio
- Malkov, Y., & Yashunin, D. (2018). "Efficient and robust approximate nearest neighbor search using HNSW." IEEE TPAMI

**Benchmark files:**
- `/Users/jordan/Workspace/orchard9/engram/engram-core/benches/concurrent_hnsw_validation.rs` (Validation benchmark)
- `/Users/jordan/Workspace/orchard9/engram/engram-core/benches/batch_hnsw_insert.rs` (Batch performance)

---

**CRITICAL REMINDER:** Do NOT begin implementation until Task 004 is complete and validated. The concurrent HNSW benchmark MUST show >= 60K ops/sec with 8 threads before proceeding.
