# Task 004: Confidence Aggregation Engine

## Objective
Implement confidence aggregation for spreading paths, ensuring probabilistic correctness when multiple paths reach the same memory.

## Priority
P1 (Quality Critical)

## Effort Estimate
1 day

## Dependencies
- Task 003: Tier-Aware Spreading Scheduler

## Technical Approach

### Implementation Details
- Create `ConfidenceAggregator` using maximum likelihood estimation
- Implement path-dependent confidence decay based on hop count
- Add confidence source tracking (which tier contributed confidence)
- Validate aggregation maintains [0,1] bounds and probabilistic semantics

### Files to Create/Modify
- `engram-core/src/activation/confidence_aggregation.rs` - New file for confidence aggregation
- `engram-core/src/activation/mod.rs` - Export aggregation functionality
- `engram-core/src/types/confidence.rs` - Extend confidence type with aggregation methods

### Integration Points
- Uses existing `Confidence` type from Milestone 0
- Integrates with confidence calibration from Milestone 2
- Connects to tier-aware spreading from Task 003
- Leverages probabilistic foundations from storage tiers

## Implementation Details

### ConfidenceAggregator Structure
```rust
pub struct ConfidenceAggregator {
    decay_rate: f32,
    min_confidence: f32,
    max_paths: usize,
}

#[derive(Debug, Clone)]
pub struct ConfidencePath {
    confidence: Confidence,
    hop_count: u16,
    source_tier: StorageTier,
    path_weight: f32,
}

impl ConfidenceAggregator {
    pub fn aggregate_paths(&self, paths: Vec<ConfidencePath>) -> Confidence {
        // Apply hop-based decay to each path
        // Use maximum likelihood estimation for aggregation
        // Weight by source tier reliability
        // Ensure result maintains probabilistic bounds
    }
}
```

### Aggregation Strategies
1. **Maximum Likelihood**: `P(correct) = 1 - ∏(1 - P_i(correct))`
2. **Weighted Average**: Weight by `1 / (1 + hop_count)` and tier reliability
3. **Conservative Minimum**: Use minimum confidence for safety-critical paths
4. **Source Tracking**: Maintain which tier/path contributed to final confidence

### Mathematical Foundation
- **Probabilistic Independence**: Assume paths provide independent evidence
- **Decay Function**: `confidence_decayed = confidence * exp(-decay_rate * hop_count)`
- **Bounds Validation**: Ensure `0 ≤ final_confidence ≤ 1`
- **Tier Weighting**: Hot tier = 1.0, Warm tier = 0.95, Cold tier = 0.9

## Acceptance Criteria
- [ ] Multiple paths to same memory aggregate probabilistically correctly
- [ ] Hop count decay properly reduces confidence with distance
- [ ] Tier-specific confidence weighting applied correctly
- [ ] Aggregated confidence maintains [0,1] bounds
- [ ] Source tier tracking available for debugging/analysis
- [ ] Performance overhead <1% for typical spreading operations
- [ ] Statistical validation against known probabilistic outcomes

## Testing Approach
- Unit tests for various aggregation scenarios (2-10 paths)
- Property tests ensuring probabilistic bounds always maintained
- Statistical tests comparing aggregation with theoretical expectations
- Integration tests with realistic spreading patterns
- Performance benchmarks measuring aggregation overhead

## Risk Mitigation
- **Risk**: Incorrect probability aggregation leading to miscalibrated confidence
- **Mitigation**: Extensive mathematical validation, comparison with reference implementations
- **Testing**: Statistical validation against known probability distributions

- **Risk**: Performance overhead from complex aggregation
- **Mitigation**: Optimize for common cases (1-3 paths), use fast approximations
- **Monitoring**: Track aggregation latency and path count distributions

- **Risk**: Numerical instability with many paths or extreme confidence values
- **Mitigation**: Use log-space arithmetic for stability, validate bounds after computation
- **Testing**: Stress test with extreme confidence values and many paths

## Notes
This task ensures that the cognitive database maintains probabilistic correctness when multiple memory associations lead to the same recall result. Unlike traditional databases that return binary results, cognitive systems must properly combine uncertain evidence from multiple sources while preserving the mathematical properties of probability.