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
- Extend `ActivationRecord` to include storage tier metadata
- Create `StorageAwareActivation` that tracks which tier memories reside in
- Implement tier-specific activation thresholds (hot: 0.01, warm: 0.05, cold: 0.1)
- Add activation latency budgets per tier (hot: 100μs, warm: 1ms, cold: 10ms)

### Files to Create/Modify
- `engram-core/src/activation/storage_aware.rs` - New file for storage-aware activation types
- `engram-core/src/activation/mod.rs` - Export new storage-aware types
- `engram-core/src/activation/record.rs` - Extend ActivationRecord

### Integration Points
- Links with `engram-core/src/storage/` tier implementations
- Uses `StorageTier` enum from existing storage module
- Integrates with confidence system from Milestone 2

## Acceptance Criteria
- [ ] `StorageAwareActivation` includes tier metadata and confidence tracking
- [ ] Tier-specific activation thresholds properly configured
- [ ] Latency budgets respect tier characteristics (hot fastest, cold slowest)
- [ ] Integration tests validate tier-aware activation behavior
- [ ] Performance impact <5% overhead for activation processing

## Testing Approach
- Unit tests for `StorageAwareActivation` construction and methods
- Integration tests with all three storage tiers
- Performance benchmarks comparing with/without storage awareness
- Property tests ensuring activation thresholds maintain tier ordering

## Risk Mitigation
- **Risk**: Performance overhead from tier tracking
- **Mitigation**: Use compact representation, cache tier lookups
- **Monitoring**: Benchmark activation record creation and access

## Notes
This task creates the foundation for tier-aware spreading that prevents cold storage from blocking hot tier operations, essential for maintaining the <10ms P95 latency target while providing comprehensive recall across all storage tiers.