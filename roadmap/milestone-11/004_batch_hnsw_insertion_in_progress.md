# Task 004: Batch HNSW Insertion - IN PROGRESS

## Status: IN PROGRESS
**Started:** 2025-10-25
**Assigned to:** rust-graph-engine-architect (Jon Gjengset mode)
**Priority:** CRITICAL - Blocks Task 003 (Worker Pool)

## Objective

Design and implement batch insertion API for HNSW index to achieve 3-5x speedup through amortized lock acquisition and optimized entry point selection. This task must complete BEFORE Task 003 to validate concurrent performance assumptions.

## Performance Targets

- **Batch of 100:** 3x faster than 100 individual inserts (target: 30μs per item vs 100μs)
- **Batch of 500:** 4x faster (target: 25μs per item)
- **Concurrent benchmark:** 8 threads must achieve 80K ops/sec (10K per thread)
- **Fallback trigger:** If concurrent performance < 60K ops/sec, implement per-layer locks

## Technical Specification

### Batch API Design

```rust
impl CognitiveHnswIndex {
    /// Insert a batch of memories with amortized locks
    ///
    /// Achieves 3-5x speedup by:
    /// - Amortizing entry point lookup across batch
    /// - Holding write lock once for all insertions
    /// - Optimizing neighbor selection within batch
    pub fn insert_batch(&self, memories: &[Arc<Memory>]) -> Result<Vec<u32>, HnswError> {
        // 1. Pre-allocate node IDs for entire batch
        // 2. Compute layers probabilistically for all nodes
        // 3. Acquire write lock ONCE
        // 4. Batch insert into graph structure
        // 5. Update entry point if needed
        // 6. Return node IDs
    }
}
```

### Implementation Strategy

1. **Node ID Pre-allocation:** Allocate all node IDs atomically before graph modification
2. **Layer Selection:** Compute layers for all nodes using fast LCG RNG
3. **Lock Amortization:** Single write lock acquisition for entire batch
4. **Entry Point Optimization:** Reuse entry point across batch, update only if higher layer found
5. **Neighbor Selection:** Optimize candidate selection within batch context

### Files to Modify

- `engram-core/src/index/hnsw_construction.rs`: Add batch insertion API (~80 lines)
- `engram-core/src/index/hnsw_graph.rs`: Add batch commit method (~60 lines)
- `engram-core/benches/batch_hnsw_insert.rs`: New benchmark file (~200 lines)

## Critical Validation (MUST RUN)

### Concurrent Performance Benchmark

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
```

**Target:** 8K insertions in < 1 second (8K ops/sec with 8 threads)
**Minimum acceptable:** 5K ops/sec (indicates acceptable contention)
**Fallback trigger:** < 5K ops/sec (indicates high lock contention)

## Fallback Plans (Priority Order)

If concurrent benchmark shows < 60K ops/sec:

1. **Per-layer locks:** Reduce contention scope
   - Layer 0 has separate lock from higher layers
   - 2x reduction in contention for typical workloads

2. **Optimistic concurrency:** Retry on conflict
   - Works if conflicts are rare (<10%)
   - Minimal overhead for happy path

3. **Space partitioning:** Natural sharding (Task 003)
   - Already planned in worker pool design
   - Eliminates inter-space contention

## Acceptance Criteria

- [ ] Batch of 100: ≥3x faster than sequential inserts
- [ ] Batch of 500: ≥4x faster than sequential inserts
- [ ] Concurrent 8-thread benchmark: ≥60K ops/sec (preferably 80K)
- [ ] Lock contention analysis documented
- [ ] Recommendation for Task 003 provided (proceed or fallback)
- [ ] Zero clippy warnings
- [ ] All benchmarks pass and results recorded

## Implementation Log

### 2025-10-25 - Starting Implementation

- Task file created
- Beginning batch API design
- Reviewing current HNSW structure for optimization opportunities

## Dependencies

- **Blocks:** Task 003 (Worker Pool) - needs concurrent performance data
- **Depends on:** None (foundation task)

## Notes

This is a CRITICAL PATH task. The concurrent benchmark results will determine if Task 003 can proceed as planned or needs fallback implementation. Accuracy and thoroughness of benchmarking is essential.

The lock-free claims in the codebase need validation under actual concurrent load. HNSW typically uses read-write locks at the graph level, which can become a bottleneck under high concurrent write load.
