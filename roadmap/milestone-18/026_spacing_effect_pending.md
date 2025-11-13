# Task 026: Spacing Effect Validation

**Status**: Pending
**Duration**: 3 days
**Priority**: Medium
**Dependencies**: M17 Tasks 001-006 (dual memory, consolidation integration)

## Objective

Validate that Engram's consolidation process produces the spacing effect: distributed practice leads to 15-30% better retention than massed practice at 1-week delay. This demonstrates proper interaction between rehearsal timing and memory consolidation.

## Background

The spacing effect (Cepeda et al., 2006 meta-analysis) shows that studying material distributed across time produces better long-term retention than massed practice. Optimal spacing is approximately 10-20% of the retention interval, following an inverted-U function.

This validates that Engram's consolidation mechanism benefits from distributed rehearsal and that temporal dynamics match human memory principles.

## Key Metrics

| Metric | Target Range | Source |
|--------|--------------|--------|
| **Spacing benefit at 1-week** | 15-30% retention advantage | Cepeda et al. 2006 |
| **Optimal spacing** | 10-20% of retention interval | Cepeda meta-analysis |
| **Inverted-U function** | r² > 0.70 for quadratic fit | Spacing × retention curve |
| **Overnight consolidation boost** | 5-10% improvement | Sleep-dependent consolidation |

## Validation Criteria

### Pass Criteria
1. Distributed practice shows 15-30% retention advantage over massed at 1-week test
2. Optimal spacing interval falls in 10-20% range (14-34 hours for 1-week retention)
3. Inverted-U function: r² > 0.70 for quadratic fit across spacing intervals
4. Overnight consolidation shows 5-10% boost relative to same-duration awake period
5. Performance regression <5% from M17 baseline

### Fail Criteria
- Spacing benefit <10% or >40% (mechanism miscalibrated)
- Optimal spacing outside 5-30% range (wrong timescale)
- Monotonic relationship (missing inverted-U characteristic)
- No overnight consolidation benefit (consolidation not working)

## Test Design

### Conditions
1. **Massed practice**: 3 presentations at 0 seconds spacing
2. **Short spacing**: 3 presentations at 1 minute spacing
3. **Medium spacing**: 3 presentations at 5 minutes spacing
4. **Long spacing**: 3 presentations at 1 hour spacing
5. **Very long spacing**: 3 presentations at 1 day spacing

### Study Phase
- 50 word pairs per condition (250 total)
- Each pair presented 3 times at condition-specific intervals
- Encoding: Create episodic memories for each presentation
- Consolidation: Allow memory system to consolidate between presentations

### Test Phase
- 1-week delay after final presentation
- Cued recall: Given first word, recall second word
- Measure: Recall accuracy (exact match required)
- Compute: Retention rate per condition

### Overnight Consolidation Sub-Test
- **Awake condition**: Study at 8am, test at 8pm (12 hours awake)
- **Sleep condition**: Study at 8pm, test at 8am (12 hours with sleep)
- Compare retention rates to validate sleep-dependent consolidation

## Implementation Approach

### Test Structure
```rust
// engram-core/tests/cognitive_validation/spacing_effect.rs

#[test]
fn test_spacing_effect_retention() {
    let graph = Arc::new(MemoryGraph::new());

    let conditions = vec![
        SpacingCondition::Massed(Duration::ZERO),
        SpacingCondition::Short(Duration::from_secs(60)),
        SpacingCondition::Medium(Duration::from_secs(300)),
        SpacingCondition::Long(Duration::from_hours(1)),
        SpacingCondition::VeryLong(Duration::from_days(1)),
    ];

    for condition in conditions {
        // Study phase with spacing
        for pair in word_pairs(condition) {
            for presentation in 0..3 {
                store_episode(graph, pair, timestamp);
                advance_time(condition.spacing);
            }
        }
    }

    // 1-week retention interval
    advance_time(Duration::from_days(7));

    // Test phase
    let results = test_cued_recall(graph, all_word_pairs);

    // Validate spacing effect
    assert_gt(results.distributed_retention, results.massed_retention * 1.15);
    assert_inverted_u_function(results, 0.70);
}

#[test]
fn test_overnight_consolidation_boost() {
    // Awake condition: 12 hours daytime
    let awake_retention = study_and_test_with_delay(
        study_time: 8am,
        test_time: 8pm,
        allow_consolidation: true
    );

    // Sleep condition: 12 hours with overnight sleep
    let sleep_retention = study_and_test_with_delay(
        study_time: 8pm,
        test_time: 8am,
        allow_consolidation: true
    );

    // Sleep should show 5-10% boost
    let boost = (sleep_retention - awake_retention) / awake_retention;
    assert_in_range(boost, 0.05, 0.10);
}
```

## Dataset

- **Word pairs**: 250 weakly associated pairs (e.g., table-cloud, river-book)
- **Weak associations**: Prevents semantic priming confound
- **Counterbalancing**: Rotate pairs across conditions

Dataset file: `engram-core/tests/cognitive_validation/datasets/spacing_word_pairs.json`

## Statistical Analysis

### Primary Analysis
```rust
// Spacing benefit calculation
let spacing_benefit = (distributed_retention - massed_retention)
    / massed_retention;

// Inverted-U quadratic fit
let fit = quadratic_regression(
    x: spacing_intervals,
    y: retention_rates
);
let r_squared = fit.r_squared();

// Optimal spacing calculation
let optimal_spacing = fit.vertex_x();  // Maximum of parabola
let pct_of_retention_interval = optimal_spacing / (7 * 24 * 3600);
```

### Expected Pattern
```
Retention Rate vs Spacing Interval (1-week test):
    Massed (0s):     45%
    Short (1m):      52%
    Medium (5m):     58%
    Long (1h):       62%  ← Optimal (~0.6% of retention interval)
    Very Long (1d):  55%
```

## Performance Requirements

- **Test duration**: <15 minutes (fast-forwarded time)
- **Memory usage**: <300MB for 250 pairs × 3 presentations
- **Consolidation cycles**: Must run on accelerated timeline
- **Regression**: <5% from M17 baseline

## Deliverables

1. Test implementation: `spacing_effect.rs`
2. Dataset file: `spacing_word_pairs.json`
3. Quadratic regression module for inverted-U analysis
4. Validation report with spacing curve graphs
5. Performance metrics and regression check

## Success Validation

Run test with:
```bash
cargo test --test cognitive_validation spacing_effect -- --nocapture

# Output should show:
# Spacing benefit (distributed vs massed): 24% (target: 15-30%) ✓
# Optimal spacing: 16% of retention interval (target: 10-20%) ✓
# Inverted-U fit: r² = 0.78 (target: >0.70) ✓
# Overnight consolidation boost: 7% (target: 5-10%) ✓
# PASSED
```

## References

### Primary Source
- Cepeda, N. J., et al. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin*, 132(3), 354-380.

### Theoretical Context
- Bjork, R. A., & Allen, T. W. (1970). The spacing effect: Consolidation or differential encoding? *Journal of Verbal Learning and Verbal Behavior*, 9(5), 567-572.
- Hintzman, D. J. (1974). Theoretical implications of the spacing effect. In *Theories in cognitive psychology* (pp. 77-99).

### Consolidation Mechanisms
- Walker, M. P., & Stickgold, R. (2006). Sleep, memory, and plasticity. *Annual Review of Psychology*, 57, 139-166.
- Dudai, Y., et al. (2015). The consolidation and transformation of memory. *Neuron*, 88(1), 20-32.

## Integration with M17

This task validates:
- **M17 Task 006**: Consolidation integration strengthens memories over time
- **M17 Task 005**: Repeated binding formation with spacing enhances retention
- **M17 Task 004**: Concept formation benefits from distributed presentations
- **Time-dependent consolidation**: Validates that consolidation timescale matches human data

## Task Completion Checklist

- [ ] Implement test in `spacing_effect.rs`
- [ ] Create `spacing_word_pairs.json` dataset
- [ ] Run test with M17 performance baseline check
- [ ] Validate 15-30% spacing benefit at 1 week
- [ ] Validate inverted-U function (r² > 0.70)
- [ ] Validate overnight consolidation boost
- [ ] Generate validation report with spacing curves
- [ ] Run `make quality` - zero clippy warnings
- [ ] Update task file: `_pending` → `_in_progress` → `_complete`
- [ ] Commit with validation results
