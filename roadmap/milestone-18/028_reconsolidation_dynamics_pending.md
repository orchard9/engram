# Task 028: Reconsolidation Dynamics Validation

**Status**: Pending
**Duration**: 4 days
**Priority**: Medium
**Dependencies**: M17 Task 006 (consolidation integration), existing reconsolidation module

## Objective

Validate that Engram's reconsolidation mechanism exhibits proper memory updating dynamics: a 3-6 hour reactivation window where memories become labile and can be enhanced (10-20% boost) or disrupted (30-50% impairment) depending on post-retrieval interventions.

## Background

Reconsolidation (Lee, 2008; Nader et al., 2000) demonstrates that retrieved memories temporarily return to a labile state, allowing updating before re-stabilizing. This window:
- Opens immediately upon retrieval
- Lasts 3-6 hours
- Allows memory strengthening (with additional learning)
- Allows memory weakening (with interference or blockers)
- Then closes, with memories reconsolidated

In Engram, this validates that M17's consolidation system integrates with existing reconsolidation mechanisms and preserves retrieval-induced plasticity during dual memory operation.

## Key Metrics

| Metric | Target Range | Source |
|--------|--------------|--------|
| **Reactivation window duration** | 3-6 hours | Lee 2008, Nader et al. 2000 |
| **Enhancement with additional learning** | 10-20% retention boost | Lee 2008 Figure 3 |
| **Disruption during window** | 30-50% impairment | Nader et al. 2000 |
| **No effect outside window** | <5% difference | Window is time-limited |
| **Binding strength update** | 10-30% increase | M17 binding system |

## Validation Criteria

### Pass Criteria
1. Reactivation window lasts 3-6 hours after retrieval
2. Additional learning during window: 10-20% retention boost
3. Interference during window: 30-50% impairment relative to no-reactivation control
4. Intervention outside window: <5% effect (window closed)
5. Binding strength increases 10-30% with enhancement paradigm
6. Performance regression <5% from M17 baseline

### Fail Criteria
- Window duration <2 hours or >8 hours (wrong timescale)
- Enhancement <5% or >30% (mechanism miscalibrated)
- Interference <20% or >70% (disruption too weak/strong)
- Window never closes (reconsolidation failure)
- No binding strength update (integration with M17 broken)

## Test Design

### Paradigm 1: Memory Enhancement
1. **Day 1 - Initial learning**: Study 50 word pairs
2. **Day 2 - Reactivation**: Test on 25 pairs (reactivated group)
3. **Day 2 - Enhancement**: Additional study of reactivated pairs (various delays)
4. **Day 7 - Final test**: Assess retention for all 50 pairs

**Conditions**:
- **Enhancement-immediate** (0h delay): Additional study immediately after reactivation
- **Enhancement-early** (2h delay): Additional study 2 hours post-reactivation
- **Enhancement-window** (4h delay): Additional study within window
- **Enhancement-late** (8h delay): Additional study after window closes
- **Control** (no reactivation): No Day 2 testing, only Day 7 test

### Paradigm 2: Memory Disruption
1. **Day 1 - Initial learning**: Study 50 word pairs
2. **Day 2 - Reactivation**: Test on 25 pairs
3. **Day 2 - Interference**: Present competing word pairs (various delays)
4. **Day 7 - Final test**: Assess retention

**Conditions**:
- **Disruption-immediate** (0h delay): Interference immediately after reactivation
- **Disruption-early** (2h delay): Interference 2 hours post-reactivation
- **Disruption-window** (4h delay): Interference within window
- **Disruption-late** (8h delay): Interference after window closes
- **Control** (no reactivation): No Day 2 testing

## Implementation Approach

### Test Structure
```rust
// engram-core/tests/cognitive_validation/reconsolidation_dynamics.rs

#[test]
fn test_reconsolidation_enhancement() {
    let graph = Arc::new(MemoryGraph::new());

    // Day 1: Initial learning
    let word_pairs = load_word_pairs(50);
    for pair in &word_pairs {
        store_episode(graph, pair, timestamp);
    }
    advance_time(Duration::from_days(1));

    // Day 2: Reactivation + enhancement at various delays
    let delays = vec![
        Duration::ZERO,              // Immediate
        Duration::from_hours(2),     // Early window
        Duration::from_hours(4),     // Late window
        Duration::from_hours(8),     // After window
    ];

    let mut results = HashMap::new();

    for delay in delays {
        // Reactivate subset
        let reactivated_pairs = &word_pairs[0..25];
        for pair in reactivated_pairs {
            let _retrieved = query_memory(graph, pair.cue);
        }

        // Wait for delay
        advance_time(delay);

        // Additional learning (enhancement)
        for pair in reactivated_pairs {
            store_episode(graph, pair, timestamp);  // Re-encode
        }

        advance_time(Duration::from_days(5));

        // Day 7: Final test
        let retention = test_cued_recall(graph, reactivated_pairs);
        results.insert(delay, retention);
    }

    // Validate enhancement within window
    assert_in_range(
        results[&Duration::ZERO] / results[&Duration::from_hours(8)],
        1.10,
        1.20
    );
}

#[test]
fn test_reconsolidation_disruption() {
    // Similar structure, but interference paradigm
    // Present competing associations during reactivation window
    // Expect 30-50% impairment for within-window interference
}

#[test]
fn test_binding_strength_update() {
    let graph = Arc::new(MemoryGraph::new());

    // Create episodic memory with initial binding
    let episode = store_episode(graph, content, timestamp);
    let initial_strength = get_binding_strength(graph, episode);

    // Reactivate
    query_memory(graph, content.cue);

    // Additional learning during window
    advance_time(Duration::from_hours(2));
    store_episode(graph, content, timestamp);

    // Check binding strength increased
    let updated_strength = get_binding_strength(graph, episode);
    let increase = (updated_strength - initial_strength) / initial_strength;
    assert_in_range(increase, 0.10, 0.30);
}
```

## Dataset

- **Word pairs**: 50 moderately associated pairs per paradigm
- **Interference pairs**: 25 competing pairs (e.g., table-cloud vs table-river)
- **Control groups**: No-reactivation baseline for each paradigm

Dataset file: `engram-core/tests/cognitive_validation/datasets/reconsolidation_word_pairs.json`

## Statistical Analysis

### Enhancement Effect
```rust
let enhancement_ratio = retention_within_window / retention_outside_window;
assert_in_range(enhancement_ratio, 1.10, 1.20);

// Time course analysis
let window_curve = fit_time_decay(
    x: delay_hours,
    y: enhancement_magnitude
);
assert_gt(window_curve.half_life, 3.0);  // Window lasts >3 hours
assert_lt(window_curve.half_life, 6.0);  // Window closes <6 hours
```

### Disruption Effect
```rust
let disruption_ratio = retention_no_interference / retention_with_interference;
assert_in_range(disruption_ratio, 1.30, 1.50);  // 30-50% impairment

// Verify window-specific effect
let late_disruption_ratio = retention_no_interference
    / retention_late_interference;
assert_lt(late_disruption_ratio, 1.05);  // <5% effect outside window
```

## Expected Results

### Enhancement Time Course
```
Retention Boost vs Delay:
    0h (immediate):    +18%  ────┐
    2h (early):        +15%      │ Within window
    4h (late):         +12%      │ (enhancement effective)
    ──────────────────────────────┘
    6h (closing):      +6%
    8h (closed):       +2%   ──── Outside window (no effect)
```

### Disruption Time Course
```
Impairment vs Delay:
    0h (immediate):    -42%  ────┐
    2h (early):        -38%      │ Within window
    4h (late):         -31%      │ (disruption effective)
    ──────────────────────────────┘
    6h (closing):      -12%
    8h (closed):       -3%   ──── Outside window (no effect)
```

## Performance Requirements

- **Reactivation latency**: <20ms per memory query
- **Update latency**: <50ms for binding strength modification
- **Memory usage**: <300MB for 50 pairs + interference
- **Time simulation**: Fast-forward through multi-day paradigm
- **Regression**: <5% from M17 baseline

## Deliverables

1. Test implementation: `reconsolidation_dynamics.rs`
2. Dataset file: `reconsolidation_word_pairs.json`
3. Time-course analysis module: Decay curve fitting
4. Validation report: Enhancement/disruption curves
5. Integration validation: M17 binding system updates during reconsolidation

## Success Validation

Run test with:
```bash
cargo test --test cognitive_validation reconsolidation_dynamics -- --nocapture

# Output should show:
# Reactivation window duration: 3-6 hours ✓
# Enhancement within window: +15% (target: 10-20%) ✓
# Disruption within window: -38% (target: 30-50%) ✓
# No effect outside window: +2% (target: <5%) ✓
# Binding strength increase: +18% (target: 10-30%) ✓
# PASSED
```

## References

### Primary Papers
- Lee, J. L. (2008). Memory reconsolidation mediates the strengthening of memories by additional learning. *Nature Neuroscience*, 11(11), 1264-1266.
- Nader, K., et al. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature*, 406(6797), 722-726.

### Theoretical Context
- Dudai, Y. (2006). Reconsolidation: The advantage of being refocused. *Current Opinion in Neurobiology*, 16(2), 174-178.
- Alberini, C. M., & LeDoux, J. E. (2013). Memory reconsolidation. *Current Biology*, 23(17), R746-R750.

### Updating Mechanisms
- Sara, S. J. (2000). Retrieval and reconsolidation: Toward a neurobiology of remembering. *Learning & Memory*, 7(2), 73-84.
- Inda, M. C., et al. (2011). Memory retrieval and the passage of time: From reconsolidation and strengthening to extinction. *Journal of Neuroscience*, 31(5), 1635-1643.

### Boundary Conditions
- Suzuki, A., et al. (2004). Memory reconsolidation and extinction have distinct temporal and biochemical signatures. *Journal of Neuroscience*, 24(20), 4787-4795.

## Integration with M17

This task validates:
- **M17 Task 006**: Consolidation integration preserves reconsolidation dynamics
- **M17 Task 005**: Binding strength updates during reactivation window
- **M17 Task 009**: Blended recall triggers reactivation for both episodes and concepts
- **Existing reconsolidation module**: Integration with dual memory architecture

Reconsolidation should work seamlessly with dual memory - episodic retrieval triggers both episode and associated concept reactivation, allowing updating of both memory types.

## Task Completion Checklist

- [ ] Implement test in `reconsolidation_dynamics.rs`
- [ ] Create `reconsolidation_word_pairs.json` dataset
- [ ] Implement time-course decay analysis
- [ ] Run test with M17 performance baseline check
- [ ] Validate 3-6 hour reactivation window
- [ ] Validate 10-20% enhancement, 30-50% disruption
- [ ] Validate binding strength updates
- [ ] Generate validation report with time-course curves
- [ ] Run `make quality` - zero clippy warnings
- [ ] Update task file: `_pending` → `_in_progress` → `_complete`
- [ ] Commit with validation results
