# Task 013: Prefetching Effectiveness

**Status**: Pending
**Estimated Duration**: 3-4 days
**Priority**: Medium - Advanced optimization

## Objective

Implement software prefetching for sequential memory access patterns (HNSW traversal, spreading activation). Measure >70% prefetch coverage for sequential scans. Profile with perf to validate prefetch hits.

## Prefetching Strategy

```rust
// HNSW traversal with prefetching
pub fn hnsw_search_with_prefetch(query: &[f32], k: usize) -> Vec<NodeId> {
    let mut candidates = BinaryHeap::new();

    for i in 0..self.nodes.len() {
        // Prefetch next 4 nodes (256 bytes ahead)
        if i + 4 < self.nodes.len() {
            unsafe {
                core::arch::x86_64::_mm_prefetch(
                    self.nodes[i + 4].as_ptr() as *const i8,
                    core::arch::x86_64::_MM_HINT_T0, // L1 cache
                );
            }
        }

        // Process current node
        let distance = cosine_similarity(query, &self.nodes[i].embedding);
        candidates.push((distance, self.nodes[i].id));
    }

    candidates.into_sorted_vec()
}
```

## Measurement

**perf stat**:
```bash
perf stat -e L1-dcache-prefetches,L1-dcache-prefetch-misses ./target/release/engram
```

**Target Metrics**:
- Prefetch coverage: >70% (prefetches / total accesses)
- Prefetch accuracy: >80% (useful prefetches / total prefetches)

## Success Criteria

- **Coverage**: >70% of sequential accesses prefetched
- **Accuracy**: >80% of prefetches hit before use
- **Performance**: 5-10% latency improvement on sequential scans
- **No Regression**: Random access patterns unaffected

## Files

- `engram-core/src/memory/prefetch.rs` (220 lines)
- `scripts/measure_prefetch_effectiveness.sh` (130 lines)
- `engram-core/benches/prefetch_benchmarks.rs` (180 lines)
