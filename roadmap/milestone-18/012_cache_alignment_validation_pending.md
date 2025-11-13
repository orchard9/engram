# Task 012: Cache-Line Alignment Validation

**Status**: Pending
**Estimated Duration**: 4-5 days
**Priority**: High - Core performance optimization

## Objective

Measure false sharing in hot structures using cachegrind and perf c2c. Validate cache-line alignment (64-byte boundaries) for atomic fields. Target <1% cache misses due to false sharing.

## False Sharing Detection

**Problem**: Adjacent fields in different threads cause cache-line bouncing.

```rust
// BAD: activation and confidence in same cache line
#[repr(C)]
pub struct MemoryNode {
    pub activation: AtomicF32,    // Thread A writes
    pub confidence: f32,           // Thread B reads
    // ... false sharing if not padded
}

// GOOD: Cache-line aligned atomics
#[repr(C, align(64))]
pub struct MemoryNode {
    pub activation: AtomicF32,    // Isolated cache line
    _pad1: [u8; 60],               // Pad to 64 bytes

    pub confidence: f32,           // Different cache line
    // ...
}
```

## Measurement Tools

1. **perf c2c**: Cache-to-cache data transfers
   ```bash
   perf c2c record -F 60000 ./target/release/engram
   perf c2c report --stdio
   ```

2. **cachegrind**: Cache miss rates
   ```bash
   valgrind --tool=cachegrind ./target/release/engram
   cg_annotate cachegrind.out.<pid>
   ```

## Success Criteria

- **False Sharing**: <1% of cache misses due to sharing
- **L1 Hit Rate**: >95% for hot-tier access
- **L3 Hit Rate**: >80% for warm-tier access
- **Alignment**: All atomic fields 64-byte aligned (static assert)

## Files

- `engram-core/src/memory/aligned_node.rs` (180 lines)
- `scripts/measure_cache_performance.sh` (220 lines)
- `docs/performance/cache_optimization.md` (300 lines)
