# Task 003: Tier-Aware Spreading Scheduler

## Objective
Implement spreading scheduler that respects storage tier latencies and prevents cold storage from blocking hot tier spreading.

## Priority
P0 (Critical Path)

## Effort Estimate
2 days

## Dependencies
- Task 001: Storage-Aware Activation Interface
- Task 002: Vector-Similarity Activation Seeding

## Technical Approach

### Implementation Details
- Create `TierAwareSpreadingScheduler` with separate queues per tier
- Implement async spreading with tier-specific timeouts
- Add hot tier priority scheduling (process hot tier activations first)
- Create tier bypass for time-critical spreading (skip cold if budget exceeded)

### Files to Create/Modify
- `engram-core/src/activation/scheduler.rs` - New file for tier-aware scheduling
- `engram-core/src/activation/mod.rs` - Export scheduler functionality
- `engram-core/src/activation/spreading.rs` - Modify to use tier-aware scheduling

### Integration Points
- Integrates with three-tier storage system from Milestone 2
- Uses async runtime for concurrent tier access
- Leverages work-stealing scheduler from existing activation module
- Connects to storage-aware activation from Task 001

## Implementation Details

### TierAwareSpreadingScheduler Structure
```rust
pub struct TierAwareSpreadingScheduler {
    hot_queue: lockfree::queue::Queue<StorageAwareActivation>,
    warm_queue: lockfree::queue::Queue<StorageAwareActivation>,
    cold_queue: lockfree::queue::Queue<StorageAwareActivation>,

    tier_timeouts: HashMap<StorageTier, Duration>,
    priority_hot_tier: bool,
    max_concurrent_per_tier: usize,
}

impl TierAwareSpreadingScheduler {
    pub async fn spread_activation(
        &self,
        initial_activations: Vec<StorageAwareActivation>,
        cycle_detector: &CycleDetector,
    ) -> SpreadingResults {
        // Distribute activations to tier-specific queues
        // Process hot tier with priority
        // Apply tier-specific timeouts
        // Aggregate results maintaining confidence
    }
}
```

### Tier Processing Strategy
- **Hot Tier**: Process immediately with highest priority, shortest timeout (100μs)
- **Warm Tier**: Process concurrently with medium timeout (1ms), yield to hot tier
- **Cold Tier**: Process in background with longest timeout (10ms), skip if budget exceeded

### Queue Management
- Lock-free queues for each tier to prevent blocking
- Work-stealing between worker threads within same tier
- Priority preemption allowing hot tier to interrupt warm/cold processing

## Acceptance Criteria
- [ ] Separate queues and processing for each storage tier
- [ ] Hot tier processing prioritized over warm/cold tiers
- [ ] Tier-specific timeouts properly enforced
- [ ] Cold tier bypass works when time budget exceeded
- [ ] No deadlocks or starvation between tier processing
- [ ] Spreading results maintain confidence tracking across tiers
- [ ] Performance meets <10ms P95 latency for single-hop activation

## Testing Approach
- Unit tests for queue management and tier prioritization
- Integration tests with realistic tier latency distributions
- Stress tests with high load on mixed storage tiers
- Chaos tests with artificial tier failures and delays
- Performance benchmarks measuring tier scheduling overhead

## Risk Mitigation
- **Risk**: Cold tier blocking hot tier processing
- **Mitigation**: Strict timeout enforcement, tier bypass mechanisms
- **Monitoring**: Track tier processing latency distributions

- **Risk**: Work-stealing overhead between tier queues
- **Mitigation**: Separate worker pools per tier, minimize cross-tier stealing
- **Testing**: Benchmark single-tier vs multi-tier processing overhead

- **Risk**: Priority inversion causing hot tier starvation
- **Mitigation**: Priority inheritance, explicit hot tier preemption
- **Validation**: Stress test with heavy cold tier load while measuring hot tier latency

## Notes
This task is crucial for maintaining the cognitive database's responsiveness. Unlike traditional databases that process requests uniformly, cognitive recall must prioritize recent/active memories (hot tier) while still exploring deep memory (cold tier) when time permits. The scheduler design directly impacts user-perceived recall latency.