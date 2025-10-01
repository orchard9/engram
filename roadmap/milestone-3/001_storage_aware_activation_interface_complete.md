# Task 001: Storage-Aware Activation Interface

## Objective
Integrate activation spreading with three-tier storage by creating storage-aware activation records that respect tier latencies and confidence adjustments.

## Priority
P0 (Critical Path)

## Effort Estimate
1 day

## Dependencies
None

## Technical Approach

### Implementation Details

#### 1. Core Data Structure (`engram-core/src/activation/storage_aware.rs`)
```rust
#[repr(C, align(64))] // Cache-line aligned
pub struct StorageAwareActivation {
    // Hot cache line - frequently accessed
    pub memory_id: String,
    pub activation_level: AtomicF32,
    pub confidence: AtomicF32,
    pub hop_count: AtomicU16,
    pub storage_tier: StorageTier,
    pub flags: ActivationFlags,

    // Warm cache line - timing and metadata
    pub creation_time: Instant,
    pub last_update: AtomicU64, // timestamp in nanos
    pub access_latency: Duration,
    pub tier_confidence_factor: f32,
}
```

#### 2. Storage Tier Enumeration
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTier {
    Hot = 0,   // In-memory HotTier
    Warm = 1,  // SSD-based WarmTier
    Cold = 2,  // Columnar ColdTier
}
```

#### 3. Tier Characteristics
- **Hot Tier**: 0.01 threshold, 100μs latency budget, 1.0 confidence factor
- **Warm Tier**: 0.05 threshold, 1ms latency budget, 0.95 confidence factor
- **Cold Tier**: 0.1 threshold, 10ms latency budget, 0.9 confidence factor

#### 4. Integration with Existing ActivationRecord
- Extend existing `ActivationRecord` in `engram-core/src/activation/mod.rs`
- Add `storage_tier: Option<StorageTier>` field
- Add method `to_storage_aware() -> StorageAwareActivation`

### Files to Create/Modify
- `engram-core/src/activation/storage_aware.rs` - New module for storage-aware types
- `engram-core/src/activation/mod.rs` - Add `pub mod storage_aware;` and extend ActivationRecord
- `engram-core/src/activation/latency_budget.rs` - New file for LatencyBudgetManager
- `engram-core/src/storage/mod.rs` - Export StorageTier enum

### Integration Points

#### With Storage Tiers
- Import `HotTier`, `WarmTier`, `ColdTier` from `engram-core/src/storage/`
- Use existing `Confidence` type from `engram-core/src/confidence.rs`
- Integrate with `MemoryStore` for tier determination

#### With Activation System
- Extend existing `ActivationRecord` (line 27 in activation/mod.rs)
- Integrate with `ActivationTask` for tier-aware processing
- Work with `PriorityActivationQueue` for tier-based scheduling

#### Specific Code Locations
- `ActivationRecord` struct at `engram-core/src/activation/mod.rs:27`
- `Activation` type at `engram-core/src/store.rs:43`
- Storage modules at `engram-core/src/storage/{hot_tier,warm_tier,cold_tier}.rs`

## Acceptance Criteria
- [ ] `StorageAwareActivation` includes tier metadata and confidence tracking
- [ ] Tier-specific activation thresholds properly configured (0.01, 0.05, 0.1)
- [ ] Latency budgets respect tier characteristics (hot: 100μs, warm: 1ms, cold: 10ms)
- [ ] Integration tests validate tier-aware activation behavior
- [ ] Performance impact <5% overhead for activation processing

## Testing Approach

### Unit Tests (`engram-core/src/activation/storage_aware.rs`)
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_storage_aware_activation_creation() {
        // Test creation with different tiers
    }

    #[test]
    fn test_tier_threshold_values() {
        // Verify thresholds: hot=0.01, warm=0.05, cold=0.1
    }

    #[test]
    fn test_confidence_adjustment_per_tier() {
        // Verify confidence factors: hot=1.0, warm=0.95, cold=0.9
    }

    #[test]
    fn test_atomic_activation_updates() {
        // Test concurrent activation updates
    }
}
```

### Integration Tests (`engram-core/tests/storage_aware_activation_test.rs`)
- Test activation spreading across all three tiers
- Verify latency budget enforcement
- Test confidence propagation through tiers
- Validate tier migration during activation

### Performance Benchmarks
```rust
#[bench]
fn bench_storage_aware_vs_regular_activation(b: &mut Bencher) {
    // Compare overhead of storage-aware vs regular activation
    // Target: <5% overhead
}
```

## Implementation Steps

1. **Create StorageTier enum** in `engram-core/src/storage/mod.rs`
2. **Implement StorageAwareActivation struct** in new file `engram-core/src/activation/storage_aware.rs`
3. **Add LatencyBudgetManager** in `engram-core/src/activation/latency_budget.rs`
4. **Extend ActivationRecord** to include optional storage tier field
5. **Implement tier-specific methods**:
   - `tier_threshold()` - returns threshold for current tier
   - `adjust_confidence_for_tier()` - applies tier confidence factor
   - `can_access_within_budget()` - checks latency budget
6. **Add integration with storage modules**:
   - Method to determine tier from MemoryId
   - Tier metadata tracking in activation records
7. **Write comprehensive tests** covering all acceptance criteria
8. **Run benchmarks** to verify <5% overhead

## Risk Mitigation
- **Risk**: Performance overhead from tier tracking
- **Mitigation**: Use atomic operations, cache-line alignment, compact representation
- **Monitoring**: Add metrics for activation creation time, tier access counts

## Implementation Notes

### Based on Codebase Analysis:
- The existing `ActivationRecord` struct uses atomic types for lock-free updates
- Storage tiers (HotTier, WarmTier, ColdTier) are already implemented
- Confidence system exists and can be integrated
- The activation system uses work-stealing queues for parallel processing

### Key Design Decisions:
- Use 64-byte cache-line alignment to prevent false sharing
- Separate hot/warm/cold data into different cache lines
- Use atomic types for concurrent access without locks
- Implement tier thresholds as compile-time constants initially

### Performance Considerations:
- Minimize indirection - embed tier data directly in activation record
- Use bitflags for compact flag representation
- Pre-calculate tier characteristics to avoid runtime lookups
- Consider SIMD-friendly layouts for batch processing

## Notes
This task creates the foundation for tier-aware spreading that prevents cold storage from blocking hot tier operations, essential for maintaining the <10ms P95 latency target while providing comprehensive recall across all storage tiers. The implementation leverages existing atomic primitives and lock-free data structures in the codebase while adding minimal overhead through careful cache-line optimization and compact data representation.