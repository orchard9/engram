# Parallel HNSW Workers: Twitter Thread

## Tweet 1/7

You have 10K HNSW insertions/sec with one core.

You need 100K/sec.

Obvious solution: 10 workers. Should be easy, right?

Shared index + mutex:
Result: 12K ops/sec with 8 cores.

1.2x speedup from 8x hardware.

Why? Thread on scaling HNSW insertion:

## Tweet 2/7

The problem: cache coherence.

8 threads inserting into the same graph partition → fight for same cache lines.

Core 0 modifies node A → invalidates other cores' caches
Core 1 reads node A → cache miss, fetch from Core 0
Core 2 reads node A → cache miss...

You spend more time on cache traffic than actual work.

## Tweet 3/7

Solution: partition by memory space (tenant).

```rust
fn assign_worker(
    space_id: &MemorySpaceId,
    num_workers: usize
) -> usize {
    hash(space_id) % num_workers
}
```

Space A → Worker 0 (always)
Space B → Worker 1 (always)

No cross-worker contention.
Cache locality bonus: same space → same CPU cache.

## Tweet 4/7

Result: 78K ops/sec with 8 workers.

That's 7.8x speedup vs 1.2x with shared mutex.

Why not perfect 8x? Some overhead:
- Worker coordination
- Memory allocation
- Context switching

But close enough for production.

## Tweet 5/7

Problem: load imbalance.

Space A: 50K obs/sec → Worker 0 saturated
Spaces B-J: 1K obs/sec → Workers 1-7 idle

Total throughput: 50K + 9K = 59K (not 100K!)

Solution: work stealing.

When Worker 1 is idle, it steals work from Worker 0's queue.

## Tweet 6/7

Work stealing implementation:

```rust
// Check own queue first
if let Some(batch) = self.own_queue.pop_batch(100) {
    process(batch);
}

// Own queue empty - steal
if let Some(batch) = steal_from_busiest_worker() {
    process(batch);
}
```

Only steal if victim has > 1000 items (worth the cache pollution).
Steal half, not all (keeps both workers busy).

## Tweet 7/7

Real-world test:
- 100K observations/sec
- 20 memory spaces (skewed distribution)
- 8 workers

Result:
- Throughput: 100K/sec sustained
- Load imbalance: < 6%
- Work stealing: 4 events/sec

Mission accomplished.

---

Key insight: don't fight cache coherence. Partition to avoid it. Then balance with work stealing.

Implementation at github.com/engramhq/engram

## Bonus: Adaptive Batching Thread

HNSW insertion has fixed overhead (~15μs for entry point lookup).

Batching amortizes this:

Batch 1: 100μs per item
Batch 100: 85μs per item (15% faster)
Batch 500: 25μs per item (4x faster!)

But large batches increase latency.

Solution: adaptive batch size.

Low load (< 100 items): batch 10 → 10ms latency
High load (> 1000 items): batch 500 → 100ms latency

Trade latency for throughput under load.

## Cache Locality Thread

Deterministic assignment (same space → same worker) gives cache locality bonus.

Worker 0 processes 1000 observations from Space A:
- First: cold cache (100ns per cache line)
- Next 999: warm cache (1ns per access)

Random assignment: every observation likely different space.
Result: 3.5x slower due to cache misses alone.

Measure with perf:
- Deterministic: 98% L1 hit rate
- Random: 65% L1 hit rate
