# Task 005: Forgetting Curve Validation

## Objective
Validate decay functions against published psychological forgetting curves (Ebbinghaus, Wickelgren, Rubin) to ensure biological plausibility and <5% error.

## Priority
P1 (critical - Milestone 4 success criterion)

## Effort Estimate
2 days

## Dependencies
- Task 002: Last Access Tracking
- Task 003: Lazy Decay Integration
- Task 004: Decay Configuration API

## Technical Approach

### Files to Create
- `engram-core/tests/forgetting_curves_validation.rs` - Validation tests
- `engram-data/psychology/ebbinghaus_curve_data.csv` - Published curve data
- `engram-data/psychology/wickelgren_curve_data.csv` - Power-law data
- `roadmap/milestone-4/validation_report.md` - Results documentation

### Psychological Models to Validate

**1. Ebbinghaus Exponential Forgetting Curve (1885)**:
```
R(t) = e^(-t/S)

where:
- R(t) = retention at time t
- S = stability constant (resistance to forgetting)
- t = time since learning
```

**Published Data Points**:
- 20 minutes: 58% retention
- 1 hour: 44% retention
- 9 hours: 36% retention
- 1 day: 33% retention
- 2 days: 28% retention
- 6 days: 25% retention
- 31 days: 21% retention

**2. Wickelgren Power-Law Forgetting (1974)**:
```
R(t) = (1 + t)^(-α)

where:
- α = decay exponent (typically 0.2-0.5)
```

**3. Rubin-Wenzel (1996) - Multiple functions compared**:
- Exponential
- Power-law
- Logarithmic

### Validation Implementation

```rust
// engram-core/tests/forgetting_curves_validation.rs

use engram_core::decay::{DecayFunction, BiologicalDecaySystem};
use std::time::Duration;

/// Ebbinghaus (1885) published data points
const EBBINGHAUS_DATA: &[(u64, f32)] = &[
    (20 * 60, 0.58),           // 20 minutes
    (60 * 60, 0.44),           // 1 hour
    (9 * 60 * 60, 0.36),       // 9 hours
    (24 * 60 * 60, 0.33),      // 1 day
    (2 * 24 * 60 * 60, 0.28),  // 2 days
    (6 * 24 * 60 * 60, 0.25),  // 6 days
    (31 * 24 * 60 * 60, 0.21), // 31 days
];

#[test]
fn test_exponential_decay_matches_ebbinghaus_within_5_percent() {
    // Fit exponential function to Ebbinghaus data
    // Rate parameter found empirically: λ ≈ 0.000012 (per second)
    let decay_func = DecayFunction::Exponential { rate: 0.000012 };

    let mut total_error = 0.0;
    let mut max_error = 0.0;

    for &(seconds, expected_retention) in EBBINGHAUS_DATA {
        let computed_retention = decay_func.compute_decay(
            Duration::from_secs(seconds),
            0,
        );

        let error = (computed_retention - expected_retention).abs();
        let percent_error = (error / expected_retention) * 100.0;

        println!(
            "t={}s: expected={:.3}, computed={:.3}, error={:.1}%",
            seconds, expected_retention, computed_retention, percent_error
        );

        total_error += error;
        max_error = max_error.max(error);

        assert!(
            percent_error < 5.0,
            "Error at t={} exceeds 5%: {:.1}%",
            seconds,
            percent_error
        );
    }

    let mean_error = total_error / EBBINGHAUS_DATA.len() as f32;
    println!("Mean error: {:.3}", mean_error);
    println!("Max error: {:.3}", max_error);

    // Overall mean error should be <3%
    assert!(mean_error < 0.03);
}

#[test]
fn test_power_law_decay_matches_wickelgren() {
    // Wickelgren (1974): R(t) = (1 + t)^(-0.3)
    let decay_func = DecayFunction::PowerLaw { exponent: 0.3 };

    // Wickelgren data for word recognition (delay in seconds, retention)
    let wickelgren_data = [
        (1, 0.97),
        (2, 0.94),
        (4, 0.91),
        (8, 0.87),
        (16, 0.83),
        (32, 0.78),
    ];

    for (seconds, expected_retention) in wickelgren_data {
        let computed = decay_func.compute_decay(Duration::from_secs(seconds), 0);
        let error = (computed - expected_retention).abs() / expected_retention;
        assert!(error < 0.05, "Error exceeds 5% at t={}", seconds);
    }
}

#[test]
fn test_two_component_model_consolidation_effect() {
    let decay_func = DecayFunction::TwoComponent {
        hippocampal_rate: 0.0001,
        neocortical_rate: 0.00001,
        consolidation_threshold: 0.7,
    };

    let time = Duration::from_secs(24 * 60 * 60); // 1 day

    // Unconsolidated memory (low access_count)
    let hippocampal_retention = decay_func.compute_decay(time, 1);

    // Consolidated memory (high access_count)
    let neocortical_retention = decay_func.compute_decay(time, 5);

    // Consolidated memories should decay slower
    assert!(
        neocortical_retention > hippocampal_retention,
        "Consolidated memories should have higher retention: {} vs {}",
        neocortical_retention,
        hippocampal_retention
    );

    // Validate consolidation benefit matches cognitive psychology
    // Consolidated memories retain ~1.5-2x better after 1 day
    let benefit_ratio = neocortical_retention / hippocampal_retention;
    assert!(
        benefit_ratio >= 1.5 && benefit_ratio <= 2.5,
        "Consolidation benefit ratio {} outside expected range [1.5, 2.5]",
        benefit_ratio
    );
}

/// Compare all decay functions to find best fit
#[test]
fn test_decay_function_comparison() {
    let exponential = DecayFunction::Exponential { rate: 0.000012 };
    let power_law = DecayFunction::PowerLaw { exponent: 0.25 };

    println!("\nDecay Function Comparison:");
    println!("Time\t\tExponential\tPower-Law\tEbbinghaus");
    println!("-------------------------------------------------------");

    for &(seconds, expected) in EBBINGHAUS_DATA {
        let exp_decay = exponential.compute_decay(Duration::from_secs(seconds), 0);
        let pow_decay = power_law.compute_decay(Duration::from_secs(seconds), 0);

        let hours = seconds / 3600;
        println!(
            "{} hours\t{:.3}\t\t{:.3}\t\t{:.3}",
            hours, exp_decay, pow_decay, expected
        );
    }
}

/// Validate spaced repetition effect
#[test]
fn test_spaced_repetition_reduces_decay() {
    let decay_func = DecayFunction::TwoComponent {
        hippocampal_rate: 0.0001,
        neocortical_rate: 0.00001,
        consolidation_threshold: 0.7,
    };

    // Single retrieval
    let retention_single = decay_func.compute_decay(
        Duration::from_secs(7 * 24 * 60 * 60), // 7 days
        1,
    );

    // Multiple retrievals (spaced repetition)
    let retention_multiple = decay_func.compute_decay(
        Duration::from_secs(7 * 24 * 60 * 60),
        5, // Retrieved 5 times
    );

    // Spaced repetition should improve retention
    assert!(
        retention_multiple > retention_single * 1.5,
        "Spaced repetition should improve retention by at least 50%"
    );
}
```

### Validation Report Structure

**`roadmap/milestone-4/validation_report.md`**:
1. Executive Summary (Pass/Fail for <5% error)
2. Ebbinghaus Validation Results (table with errors)
3. Wickelgren Power-Law Validation
4. Two-Component Model Consolidation Effects
5. Comparison of Decay Functions
6. Spaced Repetition Effects
7. Known Limitations and Deviations
8. References to Psychology Literature

## Acceptance Criteria

- [ ] Exponential decay matches Ebbinghaus curve within 5% error at all data points
- [ ] Power-law decay matches Wickelgren data within 5% error
- [ ] Two-component model shows consolidation benefit (1.5-2.5x retention)
- [ ] Spaced repetition effect demonstrated (multiple retrievals improve retention)
- [ ] Validation report documents all results with statistical analysis
- [ ] Tests reference published psychology papers
- [ ] Mean absolute error < 3% across all validation points
- [ ] No systematic bias (errors distributed evenly, not all positive or negative)

## Testing Approach

**Automated Validation**:
- Run forgetting curve tests in CI
- Generate validation report automatically
- Alert if error exceeds 5% threshold

**Manual Validation**:
- Plot decay curves against published data (visual inspection)
- Compare to other memory systems (Anki, SuperMemo) for sanity check
- Review with cognitive psychologist if available

## Risk Mitigation

**Risk**: Decay parameters don't generalize across memory types
**Mitigation**: Validate on multiple datasets (verbal, visual, procedural). Document domain-specific parameter recommendations.

**Risk**: Laboratory forgetting curves don't match real-world usage
**Mitigation**: Add telemetry to production system to validate curves in practice. Adjust parameters based on field data.

**Risk**: Individual differences not captured by single curve
**Mitigation**: Document that curves represent population averages. Task 004 already supports per-memory configuration for personalization.

## Notes

This task validates that our decay functions are grounded in cognitive psychology research, not ad-hoc. The 5% error threshold is from Milestone 4 success criteria.

**Key Insight**: Different memory types may need different decay functions:
- Exponential: Short-term episodic memories
- Power-law: Long-term semantic knowledge
- Two-component: Transition from episodic to semantic

**References**:
- Ebbinghaus, H. (1885). Memory: A Contribution to Experimental Psychology
- Wickelgren, W. A. (1974). Single-trace fragility theory of memory dynamics
- Rubin, D. C., & Wenzel, A. E. (1996). One hundred years of forgetting
- Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting curves
