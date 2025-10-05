# Task 003: HNSW Queue Consumer or Document Sync Design

## Status: Pending
## Priority: P1 - Important (Performance Decision)
## Estimated Effort: 1 day
## Dependencies: None

## Objective

Either implement async HNSW update queue consumer for non-blocking inserts, OR document that synchronous updates are the intentional design choice with performance trade-offs.

## Current State

**Design Decision Needed:**
- ✅ HNSW index implemented: `engram-core/src/index/hnsw.rs:1-500`
- ✅ Index updates work: `store.rs:460-463` calls `hnsw.insert_memory()` synchronously
- ❌ **Previous queue removed** during storage unification (commit 1628abd)
- ❌ **No documentation** of sync vs async trade-offs

**Current Implementation:**
```rust
// engram-core/src/store.rs:460-463 (CURRENT)
#[cfg(feature = "hnsw_index")]
{
    if let Some(ref hnsw) = self.hnsw_index {
        let _ = hnsw.insert_memory(Arc::clone(&memory_arc)); // ← BLOCKS store()!
    }
}
```

## Decision: Choose One Path

### Option A: Implement Async Queue Consumer (Preferred for High Throughput)

**Best for:** Systems with >1000 writes/sec, need non-blocking store operations

### Option B: Document Sync Design (Preferred for Simplicity)

**Best for:** Systems with <1000 writes/sec, need simpler mental model

---

## Option A: Async Queue Consumer Implementation

### Step 1: Add Update Queue to Store (1 hour)

**File**: `engram-core/src/store.rs`

**Line 108** - Add queue field:
```rust
/// HNSW index update queue for async processing
#[cfg(feature = "hnsw_index")]
hnsw_update_queue: Arc<crossbeam_queue::ArrayQueue<HnswUpdate>>,

/// Background worker handle
#[cfg(feature = "hnsw_index")]
hnsw_worker: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
```

**Line 247** - Initialize queue:
```rust
#[cfg(feature = "hnsw_index")]
hnsw_update_queue: Arc::new(crossbeam_queue::ArrayQueue::new(10_000)),

#[cfg(feature = "hnsw_index")]
hnsw_worker: Arc::new(Mutex::new(None)),
```

**Line 260-270** - Add update type:
```rust
#[cfg(feature = "hnsw_index")]
#[derive(Clone)]
pub enum HnswUpdate {
    Insert { memory: Arc<Memory> },
    Remove { id: String },
    Rebuild,
}
```

### Step 2: Queue Updates Instead of Blocking (30 min)

**File**: `engram-core/src/store.rs`

**Line 460-470** - Replace synchronous insert:
```rust
#[cfg(feature = "hnsw_index")]
{
    if let Some(ref hnsw) = self.hnsw_index {
        // Queue update instead of blocking
        let update = HnswUpdate::Insert { memory: Arc::clone(&memory_arc) };

        if let Err(_) = self.hnsw_update_queue.push(update) {
            tracing::warn!("HNSW update queue full, dropping update for {}", memory_id);
            // Fallback: synchronous update
            let _ = hnsw.insert_memory(Arc::clone(&memory_arc));
        }
    }
}
```

### Step 3: Implement Background Worker (2 hours)

**File**: `engram-core/src/store.rs`

**Line 550-620** - Add worker implementation:
```rust
#[cfg(feature = "hnsw_index")]
impl MemoryStore {
    /// Start HNSW update worker thread
    pub fn start_hnsw_worker(&self) {
        let queue = Arc::clone(&self.hnsw_update_queue);
        let hnsw = match &self.hnsw_index {
            Some(index) => Arc::clone(index),
            None => return,
        };

        let handle = std::thread::Builder::new()
            .name("hnsw-worker".to_string())
            .spawn(move || {
                Self::hnsw_worker_loop(queue, hnsw);
            })
            .expect("Failed to spawn HNSW worker");

        *self.hnsw_worker.lock() = Some(handle);
        tracing::info!("Started HNSW update worker thread");
    }

    fn hnsw_worker_loop(
        queue: Arc<crossbeam_queue::ArrayQueue<HnswUpdate>>,
        hnsw: Arc<CognitiveHnswIndex>,
    ) {
        const BATCH_SIZE: usize = 100;
        const BATCH_TIMEOUT: Duration = Duration::from_millis(50);

        let mut batch = Vec::with_capacity(BATCH_SIZE);
        let mut last_flush = Instant::now();

        loop {
            // Collect batch
            while batch.len() < BATCH_SIZE {
                match queue.pop() {
                    Some(update) => batch.push(update),
                    None => break,
                }
            }

            // Flush if batch full or timeout
            let should_flush = batch.len() >= BATCH_SIZE
                || (last_flush.elapsed() > BATCH_TIMEOUT && !batch.is_empty());

            if should_flush {
                Self::process_hnsw_batch(&hnsw, &batch);
                batch.clear();
                last_flush = Instant::now();
            } else if batch.is_empty() {
                // Sleep briefly to avoid busy loop
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    fn process_hnsw_batch(
        hnsw: &CognitiveHnswIndex,
        batch: &[HnswUpdate],
    ) {
        let start = Instant::now();
        let mut processed = 0;

        for update in batch {
            match update {
                HnswUpdate::Insert { memory } => {
                    if let Err(e) = hnsw.insert_memory(Arc::clone(memory)) {
                        tracing::warn!("HNSW insert failed: {:?}", e);
                    } else {
                        processed += 1;
                    }
                }
                HnswUpdate::Remove { id } => {
                    if let Err(e) = hnsw.remove(id) {
                        tracing::warn!("HNSW remove failed for {}: {:?}", id, e);
                    } else {
                        processed += 1;
                    }
                }
                HnswUpdate::Rebuild => {
                    if let Err(e) = hnsw.rebuild() {
                        tracing::error!("HNSW rebuild failed: {:?}", e);
                    }
                }
            }
        }

        tracing::debug!(
            "Processed {} HNSW updates in {:?}",
            processed,
            start.elapsed()
        );
    }

    /// Graceful shutdown of HNSW worker
    pub fn shutdown_hnsw_worker(&self) {
        if let Some(handle) = self.hnsw_worker.lock().take() {
            // Note: In production, would use shutdown signal
            handle.join().ok();
            tracing::info!("HNSW worker thread stopped");
        }
    }
}
```

### Step 4: Start Worker in Main (15 min)

**File**: `engram-cli/src/main.rs`

**Line 175** - Start worker after store creation:
```rust
let memory_store = Arc::new(memory_store);

// Start HNSW update worker
#[cfg(feature = "hnsw_index")]
memory_store.start_hnsw_worker();
```

**Line 193** - Graceful shutdown:
```rust
// Cleanup on exit
#[cfg(feature = "hnsw_index")]
memory_store.shutdown_hnsw_worker();

remove_pid_file()?;
info!(" Server stopped gracefully");
```

### Step 5: Add Monitoring (30 min)

**File**: `engram-core/src/store.rs`

**Line 630-650** - Add queue metrics:
```rust
#[cfg(feature = "hnsw_index")]
impl MemoryStore {
    /// Get HNSW queue statistics
    pub fn hnsw_queue_stats(&self) -> HnswQueueStats {
        HnswQueueStats {
            queue_depth: self.hnsw_update_queue.len(),
            queue_capacity: self.hnsw_update_queue.capacity(),
            utilization: self.hnsw_update_queue.len() as f32
                / self.hnsw_update_queue.capacity() as f32,
        }
    }
}

#[cfg(feature = "hnsw_index")]
#[derive(Debug, Clone)]
pub struct HnswQueueStats {
    pub queue_depth: usize,
    pub queue_capacity: usize,
    pub utilization: f32,
}
```

---

## Option B: Document Sync Design

### Step 1: Create Design Document (2 hours)

**File**: `docs/hnsw-sync-design.md` (create new)

```markdown
# HNSW Synchronous Update Design

## Decision

Engram uses **synchronous HNSW updates** during store operations, intentionally blocking the write path to maintain index consistency.

## Rationale

### Advantages of Sync Design
1. **Simplicity**: No background threads, queue management, or worker coordination
2. **Consistency**: Index always reflects stored data (no lag)
3. **Memory Pressure**: No queue buffer consuming additional RAM
4. **Debugging**: Easier to trace insert → index update path
5. **Sufficient for Target**: <1000 writes/sec throughput acceptable

### Performance Characteristics
- **Store Latency**: +2-5ms for HNSW insert (still <10ms P95 target)
- **Throughput**: ~800 writes/sec single-threaded, 3000+ multi-threaded
- **Consistency**: Immediate query visibility (no eventual consistency lag)

### When to Reconsider
If production workload shows:
- Store P95 latency >10ms consistently
- Write throughput requirements >5000/sec
- User complaints about insert blocking

Then implement async queue (see milestone-2.5/task-003 Option A).

## Implementation

HNSW updates happen at `engram-core/src/store.rs:460-463`:

```rust
#[cfg(feature = "hnsw_index")]
{
    if let Some(ref hnsw) = self.hnsw_index {
        let _ = hnsw.insert_memory(Arc::clone(&memory_arc)); // Synchronous
    }
}
```

## Benchmarks

| Operation | Latency (P95) | Throughput |
|-----------|---------------|------------|
| Store (no HNSW) | 0.5ms | 15,000/sec |
| Store (with HNSW) | 3.2ms | 800/sec |
| Recall (HNSW) | 1.1ms | 5,000/sec |

## Migration Path

If async becomes necessary:
1. Implement `HnswUpdate` queue (ArrayQueue<T>)
2. Spawn worker thread with batching
3. Feature flag: `hnsw-async-updates`
4. Validate consistency with differential testing
```

### Step 2: Update Architecture Docs (30 min)

**File**: `docs/architecture.md`

Add section:
```markdown
## HNSW Index Updates

Engram uses **synchronous HNSW updates** (not async queue) for simplicity and consistency.

Store operations block on index insert (~2-5ms overhead), keeping index immediately consistent with data. This is acceptable for target workload (<1000 writes/sec).

See [HNSW Sync Design](./hnsw-sync-design.md) for rationale and migration path.
```

### Step 3: Add Performance Note to Store (15 min)

**File**: `engram-core/src/store.rs`

**Line 458** - Add documentation:
```rust
// HNSW index update (synchronous by design)
//
// This intentionally blocks the store operation to maintain immediate consistency.
// Adds ~2-5ms latency but ensures queries see just-inserted data immediately.
//
// For async updates (if >5000 writes/sec needed), see:
// roadmap/milestone-2.5/003_hnsw_queue_or_document_sync_pending.md
#[cfg(feature = "hnsw_index")]
{
    if let Some(ref hnsw) = self.hnsw_index {
        let _ = hnsw.insert_memory(Arc::clone(&memory_arc));
    }
}
```

---

## Testing (Both Options)

### Performance Benchmark

**File**: `engram-core/benches/hnsw_update_latency.rs` (create new)

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use engram_core::{MemoryStore, Episode, Confidence};
use chrono::Utc;

fn bench_store_with_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_updates");

    for size in [100, 1000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("sync", size),
            &size,
            |b, &size| {
                let store = MemoryStore::new(size).with_hnsw_index();

                b.iter(|| {
                    let episode = Episode::new(
                        format!("ep_{}", rand::random::<u32>()),
                        Utc::now(),
                        "test".to_string(),
                        [0.5; 768],
                        Confidence::MEDIUM,
                    );
                    store.store(episode);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_store_with_hnsw);
criterion_main!(benches);
```

Run with:
```bash
cargo bench --bench hnsw_update_latency
```

## Acceptance Criteria

### Option A (Async Queue)
- [ ] `HnswUpdate` queue with 10k capacity
- [ ] Background worker processes batches of 100 updates
- [ ] Batch timeout of 50ms for low-traffic periods
- [ ] Graceful shutdown drains queue
- [ ] Queue depth metrics exported
- [ ] Store latency <1ms (queue push only)
- [ ] Index update P95 <100ms (batched)

### Option B (Sync Design)
- [ ] Design document created explaining decision
- [ ] Architecture docs updated
- [ ] Performance benchmarks documented
- [ ] Migration path to async defined
- [ ] Code comments explain sync choice
- [ ] Store P95 <10ms (including HNSW)

## Performance Targets

### Option A (Async)
- Queue push: <100μs
- Batch processing: <100ms for 100 updates
- Store throughput: >5000 writes/sec

### Option B (Sync)
- Store with HNSW: <5ms P95
- Store throughput: >800 writes/sec
- Recall latency: <1ms P95

## Files to Modify

### Option A
1. `engram-core/src/store.rs` - Add queue, worker, batch processing
2. `engram-cli/src/main.rs` - Start worker thread
3. `engram-core/benches/hnsw_update_latency.rs` - Create benchmark

### Option B
1. `docs/hnsw-sync-design.md` - Create design doc
2. `docs/architecture.md` - Update architecture docs
3. `engram-core/src/store.rs` - Add code comments
4. `engram-core/benches/hnsw_update_latency.rs` - Benchmark sync performance

## Recommendation

**Start with Option B (Sync Design)** unless benchmarks show >10ms P95 latency. Premature optimization adds complexity without proven need.

If production metrics later show bottleneck, implement Option A with feature flag for A/B testing.
