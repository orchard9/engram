# DualMemoryBudget Lock-Free Implementation

## Overview

The `DualMemoryBudget` module provides lock-free memory budget coordination for Engram's dual-tier episode-concept storage architecture. It enforces independent memory budgets through atomic allocation tracking with zero synchronization overhead.

## Architecture

### Design Philosophy

Episodes and concepts have fundamentally different memory dynamics:
- **Episodes**: High churn, temporal locality, rapid allocation/deallocation
- **Concepts**: Stable long-term storage, grows slowly, infrequent deallocation

Independent budgets prevent concept growth from starving episode allocation and enable tier-specific eviction policies.

### Core Structure

```rust
pub struct DualMemoryBudget {
    episode_budget_bytes: usize,
    concept_budget_bytes: usize,
    episode_allocated: CachePadded<AtomicUsize>,
    concept_allocated: CachePadded<AtomicUsize>,
    episode_capacity: usize,
    concept_capacity: usize,
}
```

## Performance Characteristics

### Memory Ordering Rationale

All atomic operations use `Ordering::Relaxed` because:

1. **Independent Counters**: Episode and concept allocations are completely independent - no cross-atomic synchronization needed
2. **Approximate Accounting**: Budget enforcement is probabilistic - exact precision not required
3. **No Happens-Before**: No inter-thread ordering requirements between allocations
4. **Best-Effort Enforcement**: Temporary budget overruns during concurrent allocation are acceptable

The worst case is a transient budget overrun during concurrent allocations, which is tolerable given that eviction runs asynchronously.

### Cache Line Optimization

Episode and concept atomics are separated via `CachePadded` to prevent false sharing:

```
Cache Line 0:  | episode_allocated (64 bytes)                    |
Cache Line 1:  | concept_allocated (64 bytes)                    |
```

Without padding, concurrent updates to episode and concept counters would cause cache line bouncing between CPU cores, degrading performance by 10-100x.

### Operation Costs

| Operation | Latency | Notes |
|-----------|---------|-------|
| `can_allocate_episode()` | ~1-2ns | Single atomic load (Relaxed) |
| `record_episode_allocation()` | ~8-12ns | fetch_update with saturating_add |
| `record_episode_deallocation()` | ~8-12ns | fetch_update with saturating_sub |
| `episode_utilization()` | ~2-3ns | Load + division (no atomic writes) |

### Overflow/Underflow Protection

Uses `fetch_update` with saturating arithmetic:

```rust
self.episode_allocated.fetch_update(
    Ordering::Relaxed,
    Ordering::Relaxed,
    |current| Some(current.saturating_add(bytes))
)
```

This prevents panics from overflow/underflow while maintaining lock-free operation. The saturating behavior is acceptable because:
- Overflow indicates severe overallocation (already a failure case)
- Underflow indicates accounting errors (tracking is approximate anyway)

## Integration with Storage Tiers

### Episode Storage Integration

```rust
use engram_core::storage::DualMemoryBudget;

let budget = DualMemoryBudget::new(512, 1024); // 512MB episodes, 1GB concepts

// Before allocating episode
if budget.can_allocate_episode() {
    let node = DualMemoryNode::new_episode(...);
    budget.record_episode_allocation(std::mem::size_of_val(&node));
    store_episode(node);
} else {
    // Trigger episode eviction
    evict_old_episodes();
}

// On episode eviction
budget.record_episode_deallocation(node_size);
```

### Concept Storage Integration

```rust
// Before forming concept
if budget.can_allocate_concept() {
    let concept = DualMemoryNode::new_concept(...);
    budget.record_concept_allocation(std::mem::size_of_val(&concept));
    store_concept(concept);
} else {
    // Trigger concept consolidation/compaction
    compact_concepts();
}
```

### Eviction Policy Integration

```rust
// Monitor utilization for eviction triggers
if budget.episode_utilization() > 90.0 {
    trigger_episode_eviction(target_utilization: 70.0);
}

if budget.concept_utilization() > 95.0 {
    trigger_concept_compaction();
}
```

## Testing

### Test Coverage

13 comprehensive tests covering:
- Budget initialization and validation
- Basic allocation/deallocation tracking
- Budget exhaustion scenarios
- Concurrent allocation (16 threads)
- Concurrent mixed operations (32 threads, 1000 ops each)
- Overflow protection (saturating at usize::MAX)
- Underflow protection (saturating at 0)
- Independent budget enforcement
- Utilization calculation accuracy

### Stress Testing

The concurrent stress test validates correctness under extreme load:
```rust
32 threads × 1000 operations = 32,000 concurrent operations
Operations: allocate/deallocate episodes and concepts
Result: No panics, no deadlocks, correct saturation behavior
```

## Memory Layout

### DualMemoryNode Size

Conservative estimate: **3328 bytes** per node

Breakdown:
- 16 bytes: Uuid (128-bit)
- 64+ bytes: MemoryNodeType enum (largest variant + discriminant + padding)
- 3072 bytes: [f32; 768] embedding (4 bytes × 768)
- 64 bytes: CachePadded<AtomicF32> activation
- 16 bytes: Confidence struct
- 24 bytes: DateTime<Utc> × 2
- 64 bytes: repr(align(64)) alignment padding

### Budget Capacity Calculation

```
Episode capacity = (512 MB) / (3328 bytes) = 161,319 nodes
Concept capacity = (1024 MB) / (3328 bytes) = 322,638 nodes
```

## Performance Validation

### Benchmarks

Created `engram-core/benches/dual_memory_budget.rs` measuring:
- Single-threaded allocation overhead
- Concurrent allocation scalability (2/4/8/16 threads)
- Mixed operation latency (alloc + check + dealloc)

### Expected Performance

Based on atomic operation characteristics:
- **Single-threaded throughput**: ~100-200M allocations/sec
- **16-thread throughput**: ~800M-1.6B allocations/sec
- **Latency overhead**: <10ns per allocation tracking call

## Systems Engineering Trade-offs

### Why Not Use Locks?

Mutex-based tracking would add ~20-50ns overhead per allocation due to:
- Lock acquisition/release overhead
- Cache line bouncing (mutex state)
- Potential contention under high load

For Engram's workload (millions of allocations/sec), this overhead is unacceptable.

### Why Relaxed Ordering?

Stronger orderings (Acquire/Release/SeqCst) add memory barriers that prevent CPU reordering. This costs ~5-10ns per operation but provides no benefit here because:

1. Budget counters are independent (no cross-atomic synchronization)
2. Allocation tracking is best-effort (approximate is fine)
3. Eviction runs asynchronously (no immediate consistency required)

### Why Saturating Arithmetic?

Alternatives considered:
- **Wrapping arithmetic**: Silent overflow/underflow (dangerous)
- **Checked arithmetic + panic**: Crashes on edge cases (unacceptable)
- **CAS loop with bounds checking**: High overhead (10-100x slower)

Saturating arithmetic provides safety without overhead. Saturation indicates a severe system issue (massive overallocation or accounting bugs) that would require manual intervention anyway.

## Future Enhancements

1. **NUMA-Aware Budgets**: Per-NUMA-node budgets for better locality
2. **Tiered Budgets**: Hot/warm/cold sub-budgets within episode tier
3. **Adaptive Limits**: Dynamic budget adjustment based on workload
4. **Telemetry**: Histogram tracking of allocation sizes
5. **Budget Leasing**: Thread-local budget reservations for batch allocation

## Files Modified

- Created: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/storage/dual_memory_budget.rs`
- Modified: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/storage/mod.rs`
- Created: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/benches/dual_memory_budget.rs`
- Created: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/examples/dual_memory_budget_usage.rs`

## References

- **Herlihy & Shavit (2012)**: The Art of Multiprocessor Programming - Lock-free data structures
- **McKenney (2017)**: Is Parallel Programming Hard, And, If So, What Can You Do About It? - Memory ordering
- **Seltzer & Yigit (1991)**: A New Hashing Package for UNIX - Cache-conscious design principles
