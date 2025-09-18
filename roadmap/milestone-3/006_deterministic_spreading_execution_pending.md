# Task 006: Deterministic Spreading Execution

## Objective
Ensure spreading produces deterministic results given same seed, critical for reproducible debugging and testing.

## Priority
P1 (Quality Critical)

## Effort Estimate
1 day

## Dependencies
- Task 005: Cyclic Graph Protection

## Technical Approach

### Implementation Details
- Implement deterministic work scheduling using seeded random number generator
- Add phase barriers for synchronization between spreading phases
- Create deterministic tie-breaking for equal activation values
- Add determinism validation tests comparing multiple runs with same seed

### Files to Create/Modify
- `engram-core/src/activation/deterministic.rs` - New file for deterministic execution
- `engram-core/src/activation/spreading.rs` - Add deterministic mode support
- `engram-core/src/activation/scheduler.rs` - Integrate deterministic scheduling

### Integration Points
- Uses seeded RNG for deterministic behavior
- Integrates with tier-aware scheduler from Task 003
- Connects to cycle detection from Task 005
- Maintains compatibility with existing spreading infrastructure

## Implementation Details

### Deterministic Execution Framework
```rust
pub struct DeterministicSpreadingEngine {
    rng_seed: u64,
    phase_barriers: Vec<Barrier>,
    tie_breaker: TieBreaker,
    execution_mode: ExecutionMode,
}

pub enum ExecutionMode {
    Deterministic { seed: u64 },
    Performance,  // Non-deterministic but faster
}

impl DeterministicSpreadingEngine {
    pub async fn spread_deterministic(
        &self,
        initial_activations: Vec<StorageAwareActivation>,
    ) -> SpreadingResults {
        // Phase 1: Seed all workers with same RNG state
        // Phase 2: Process activations with deterministic ordering
        // Phase 3: Synchronize results with phase barriers
        // Phase 4: Apply deterministic tie-breaking
    }
}
```

### Deterministic Guarantees
1. **Seeded RNG**: All randomness derived from single seed
2. **Ordered Processing**: Process activations in deterministic order (by memory ID)
3. **Phase Barriers**: Synchronize parallel workers at spreading phase boundaries
4. **Tie Breaking**: Consistent resolution of equal activation values

### Synchronization Strategy
- **Phase Barriers**: Workers synchronize after each spreading hop
- **Ordered Queues**: Process activations in sorted order within each phase
- **Atomic Counters**: Track phase completion across worker threads
- **Result Aggregation**: Merge results in deterministic order

## Acceptance Criteria
- [ ] Identical results produced for same seed across multiple runs
- [ ] Deterministic behavior maintained under concurrent execution
- [ ] Phase barriers properly synchronize parallel workers
- [ ] Tie-breaking consistently resolves equal activation values
- [ ] Performance impact <10% when deterministic mode enabled
- [ ] Non-deterministic mode maintains performance optimization
- [ ] Validation tests confirm bit-identical results

## Testing Approach
- Property tests running same spreading 10+ times with same seed
- Differential tests comparing deterministic vs performance modes
- Concurrency tests ensuring determinism under parallel execution
- Performance benchmarks measuring deterministic mode overhead
- Stress tests with high tie-breaking scenarios

## Risk Mitigation
- **Risk**: Performance degradation from synchronization overhead
- **Mitigation**: Make deterministic mode optional, optimize for common case
- **Monitoring**: Track performance difference between modes

- **Risk**: Determinism breaks under future code changes
- **Mitigation**: Automated regression tests, clear deterministic invariants
- **Testing**: Continuous validation of deterministic properties

- **Risk**: Race conditions in phase barrier implementation
- **Mitigation**: Use proven synchronization primitives, extensive testing
- **Validation**: Stress test phase barriers under high contention

## Implementation Strategy

### Phase 1: Basic Determinism
- Implement seeded RNG for all spreading randomness
- Add simple ordering for activation processing
- Basic validation tests

### Phase 2: Parallel Determinism
- Add phase barriers for worker synchronization
- Implement deterministic tie-breaking
- Comprehensive parallel testing

### Phase 3: Performance Optimization
- Optimize phase barrier implementation
- Add fast path for non-deterministic mode
- Performance validation

## Notes
Deterministic execution is crucial for debugging cognitive behavior and validating spreading algorithms. Unlike traditional databases with predetermined execution plans, cognitive spreading involves emergent behavior that must be reproducible for scientific validation and system debugging. This capability is essential for research applications and production troubleshooting.