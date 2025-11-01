# Task 005: Retroactive Interference and Fan Effect

**Status:** PENDING
**Priority:** P1
**Estimated Duration:** 2 days
**Dependencies:** Task 004 (Proactive Interference)
**Agent Review Required:** memory-systems-researcher

## Overview

Implement retroactive interference (McGeoch 1942) and fan effect (Anderson 1974) to complete the interference modeling trilogy. Retroactive interference is the opposite of proactive - new learning interferes with old memories. Fan effect models retrieval slowdown from associative density.

## Psychology Foundation

### Retroactive Interference
**Source:** McGeoch, J. A. (1942). The psychology of human learning

- **Mechanism:** New learning interferes with old memories
- **Example:** Learn List A → Learn List B → Recall List A is impaired
- **Effect Size:** 15-25% accuracy reduction with 1 interpolated list
- **Factors:** Similarity (quadratic scaling), temporal proximity, retention interval

### Fan Effect
**Source:** Anderson, J. R. (1974). Retrieval of propositional information from long-term memory

- **Mechanism:** More associations to a concept slow retrieval
- **Example:** "The banker is in the park" retrieved faster if banker appears in 1 fact vs 5 facts
- **Effect Size:** 50-150ms RT increase per additional association
- **Linear:** Retrieval time = base + (fan_count × time_per_association)

## Implementation Specifications

### File Structure
```
engram-core/src/cognitive/interference/
├── retroactive.rs (new)
└── fan_effect.rs (new)

engram-core/tests/cognitive/
└── interference_integration_tests.rs (new)
```

### Retroactive Interference Detector

**File:** `/engram-core/src/cognitive/interference/retroactive.rs`

```rust
pub struct RetroactiveInterferenceDetector {
    similarity_threshold: f32,           // Default: 0.7
    retroactive_window: Duration,        // Default: 1 hour after original
    base_interference: f32,              // Default: 0.10 (10%)
    similarity_exponent: f32,            // Default: 2.0 (quadratic)
}

impl RetroactiveInterferenceDetector {
    pub fn detect_interference(
        &self,
        target_episode: &Episode,
        subsequent_episodes: &[Episode],
        graph: &MemoryGraph
    ) -> RetroactiveInterferenceResult;

    fn is_interfering(
        &self,
        target: &Episode,
        subsequent: &Episode,
        graph: &MemoryGraph
    ) -> bool;
}
```

**Key Differences from Proactive:**
- Temporal direction reversed: subsequent episodes interfere with target
- Similarity-weighted sum: interference ∝ similarity²
- Shorter temporal window: 1 hour vs 24 hours
- Quadratic scaling emphasizes highly similar items

### Fan Effect Detector

**File:** `/engram-core/src/cognitive/interference/fan_effect.rs`

```rust
pub struct FanEffectDetector {
    base_retrieval_time_ms: f32,       // Default: 100ms
    time_per_association_ms: f32,      // Default: 50ms (Anderson 1974)
}

impl FanEffectDetector {
    pub fn compute_fan_effect(
        &self,
        node_id: NodeId,
        graph: &MemoryGraph
    ) -> FanEffectResult;
}

pub struct FanEffectResult {
    pub fan_count: usize,
    pub predicted_retrieval_time_ms: f32,
    pub magnitude: f32,  // Relative to baseline
}
```

**Computation:**
```
retrieval_time = base_time + (fan_count × time_per_association)
magnitude = (fan_count × time_per_association) / base_time
```

## Integration Points

**M3 (Activation Spreading):** Apply fan effect as retrieval slowdown
- File: `engram-core/src/activation/recall.rs`
- Mechanism: Higher fan → lower activation per edge

**M8 (Pattern Completion):** Retroactive interference competes with reconstructed patterns
- File: `engram-core/src/completion/mod.rs`
- Mechanism: Recently learned similar items interfere with pattern completion

**Metrics:** Record both interference types independently
- File: `engram-core/src/metrics/cognitive_patterns.rs`

## Testing Strategy

### Retroactive Interference Tests

#### Test 1: McGeoch (1942) Replication
```rust
#[test]
fn test_mcgeoch_1942_replication() {
    // Learn List A
    // Learn List B (similar to A, within 1 hour)
    // Recall List A
    // Expected: 15-25% accuracy reduction ±10%
}
```

#### Test 2: Similarity Weighting
```rust
#[test]
fn test_similarity_quadratic_weighting() {
    // Similar item (0.9): interference = 0.10 × 0.9² = 0.081
    // Dissimilar item (0.5): interference = 0.10 × 0.5² = 0.025
}
```

### Fan Effect Tests

#### Test 1: Anderson (1974) Replication
```rust
#[test]
fn test_anderson_1974_fan_effect() {
    // Node with fan=1: ~150ms retrieval
    // Node with fan=3: ~250ms retrieval
    // Node with fan=5: ~350ms retrieval
    // Linear progression: 50ms per association ±25ms
}
```

#### Test 2: Linear Scaling
```rust
#[test]
fn test_fan_effect_linear_scaling() {
    for fan_count in 1..=10 {
        let result = detector.compute_fan_effect(node_id, &graph);
        let expected = 100.0 + (fan_count as f32 * 50.0);
        assert_eq!(result.predicted_retrieval_time_ms, expected);
    }
}
```

### Integration Test

```rust
#[test]
fn test_all_interference_types_tracked_independently() {
    // Apply proactive interference
    // Apply retroactive interference
    // Apply fan effect
    // Verify metrics distinguish all three types
}
```

## Acceptance Criteria

### Must Have
- [ ] McGeoch (1942) replication: 15-25% accuracy reduction ±10%
- [ ] Retroactive interference uses similarity² weighting
- [ ] Temporal window 1 hour for retroactive interference
- [ ] Anderson (1974) fan effect: 50-150ms per association ±25ms
- [ ] Fan effect linear: retrieval_time = 100 + (fan × 50)
- [ ] Metrics distinguish proactive, retroactive, and fan independently
- [ ] All unit tests pass
- [ ] `make quality` passes

### Should Have
- [ ] Performance: interference detection <100μs
- [ ] Fan effect computation <50μs (hot path)
- [ ] Retroactive interference capped at 50% max reduction

### Nice to Have
- [ ] Visualization of interference sources
- [ ] Configurable similarity exponent (not just quadratic)

## Implementation Checklist

- [ ] Create `retroactive.rs` with similarity² weighting
- [ ] Create `fan_effect.rs` with linear scaling
- [ ] Update `mod.rs` to export new types
- [ ] Implement McGeoch (1942) replication test
- [ ] Implement Anderson (1974) replication test
- [ ] Implement integration test for all three interference types
- [ ] Add metrics recording for retroactive and fan
- [ ] Integrate with M3 activation spreading
- [ ] Run `make quality` and fix warnings
- [ ] Verify performance benchmarks

## Risks and Mitigations

**Risk 1:** Quadratic similarity weighting too aggressive
- **Mitigation:** Configurable exponent (default 2.0)
- **Mitigation:** Parameter sweep if validation fails

**Risk 2:** Fan effect 50ms per association doesn't match empirical data
- **Mitigation:** Value from Anderson (1974), but validate with real graphs
- **Mitigation:** Make time_per_association configurable

**Risk 3:** Three interference types interact unexpectedly
- **Mitigation:** Track independently in metrics
- **Mitigation:** Integration test validates independent operation

## References

1. McGeoch, J. A. (1942). The psychology of human learning: An introduction.
2. Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. *Cognitive Psychology*, 6(4), 451-474.
3. Anderson, M. C., & Neely, J. H. (1996). Interference and inhibition in memory retrieval. *Memory*, 125-153.
