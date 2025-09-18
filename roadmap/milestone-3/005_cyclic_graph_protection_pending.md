# Task 005: Cyclic Graph Protection

## Objective
Implement cycle detection and prevention to ensure spreading terminates in bounded time without infinite loops.

## Priority
P0 (Critical Path)

## Effort Estimate
1.5 days

## Dependencies
- Task 003: Tier-Aware Spreading Scheduler

## Technical Approach

### Implementation Details
- Extend existing `CycleDetector` to work with storage-aware spreading
- Implement visited set with hop count limits per memory
- Add cycle breaking with confidence penalty (reduce activation when cycles detected)
- Create cycle analysis metrics for debugging pathological graphs

### Files to Create/Modify
- `engram-core/src/activation/cycle_detection.rs` - Extend existing cycle detection
- `engram-core/src/activation/spreading.rs` - Integrate cycle protection
- `engram-core/src/activation/mod.rs` - Export enhanced cycle detection

### Integration Points
- Extends existing cycle detection from activation module
- Integrates with tier-aware spreading scheduler from Task 003
- Uses storage-aware activation from Task 001
- Connects to confidence system for penalty application

## Implementation Details

### Enhanced CycleDetector Structure
```rust
pub struct CycleDetector {
    visited: DashMap<MemoryId, VisitRecord>,
    max_hop_count: u16,
    cycle_penalty: f32,
    detection_strategy: CycleDetectionStrategy,
}

#[derive(Debug, Clone)]
pub struct VisitRecord {
    hop_count: u16,
    activation_level: f32,
    first_visit_tier: StorageTier,
    visit_count: u8,
}

pub enum CycleDetectionStrategy {
    HopLimit,           // Simple hop count limit
    ActivationThreshold, // Stop when activation drops below threshold
    AdaptiveDecay,      // Reduce activation on revisit
}
```

### Cycle Prevention Strategies
1. **Hop Count Limits**: Maximum hops per memory (default: 5)
2. **Activation Threshold**: Stop spreading when activation < 0.01
3. **Visit Tracking**: Detect and penalize repeated visits to same memory
4. **Confidence Penalties**: Reduce confidence by `cycle_penalty` factor on cycles

### Integration with Storage Tiers
- **Hot Tier**: Aggressive cycle detection (short hop limits)
- **Warm Tier**: Moderate cycle detection (medium hop limits)
- **Cold Tier**: Relaxed cycle detection (longer hop limits for deep exploration)

### Performance Considerations
- Lock-free `DashMap` for concurrent visit tracking
- Memory pool for `VisitRecord` allocation
- Bloom filter for fast negative lookups before `DashMap` access

## Acceptance Criteria
- [ ] Spreading always terminates in bounded time (<100ms worst case)
- [ ] Cycles detected and handled gracefully without infinite loops
- [ ] Hop count limits properly enforced per storage tier
- [ ] Confidence penalties applied correctly for cyclic paths
- [ ] Visit tracking maintains thread safety under concurrent spreading
- [ ] Performance impact <2% overhead for cycle detection
- [ ] Debugging metrics available for cycle analysis

## Testing Approach
- Unit tests with known cyclic graph topologies
- Property tests ensuring spreading always terminates
- Stress tests with pathological graphs (high connectivity, many cycles)
- Performance tests measuring cycle detection overhead
- Integration tests with realistic memory association patterns

## Risk Mitigation
- **Risk**: Cycle detection becomes performance bottleneck
- **Mitigation**: Bloom filter pre-filtering, memory pooling, lock-free data structures
- **Monitoring**: Track cycle detection latency and memory usage

- **Risk**: Over-aggressive cycle detection prevents valid deep exploration
- **Mitigation**: Tier-specific hop limits, adaptive thresholds based on activation level
- **Testing**: Validate deep memory exploration still works with reasonable hop limits

- **Risk**: Memory leaks from visit tracking in long-running systems
- **Mitigation**: Periodic cleanup of old visit records, LRU eviction
- **Monitoring**: Track visit record count and memory usage over time

## Implementation Phases

### Phase 1: Basic Cycle Detection
- Implement hop count limits with simple visit tracking
- Integrate with existing spreading infrastructure
- Basic termination guarantees

### Phase 2: Advanced Strategies
- Add activation threshold detection
- Implement confidence penalties for cycles
- Tier-specific detection parameters

### Phase 3: Performance Optimization
- Add Bloom filter pre-filtering
- Optimize memory allocation patterns
- Concurrent access optimization

## Notes
This task is critical for system stability. Unlike traditional graph databases with predetermined query plans, cognitive spreading must explore dynamically while avoiding infinite loops. The quality of cycle detection directly impacts both system reliability and the cognitive realism of memory exploration patterns.