# Task 009: Spacing Effect Validation

**Status:** PENDING
**Priority:** P1 (Validation)
**Estimated Duration:** 1 day
**Dependencies:** M4 (Temporal Dynamics)
**Agent Review Required:** verification-testing-lead

## Overview

Validate that Engram's temporal dynamics replicate the spacing effect from Cepeda et al. (2006) meta-analysis. The spacing effect is one of the most robust findings in cognitive psychology: distributed practice produces better retention than massed practice.

## Psychology Foundation

**Source:** Cepeda, N. J., et al. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin*, 132(3), 354.

**Phenomenon:** Distributed practice > Massed practice

**Experimental Design:**
- **Massed:** Study items 3 times consecutively
- **Distributed:** Study items 3 times with 1-hour spacing
- **Test:** Retention after 24 hours
- **Expected:** 20-40% better retention for distributed vs massed

**Acceptance Range:** ±10% → [10%, 50%] retention improvement

## Implementation Specifications

### File Structure
```
engram-core/tests/psychology/
├── spacing_effect.rs (new)
└── test_materials.json (word lists for testing)
```

### Test Implementation

**File:** `/engram-core/tests/psychology/spacing_effect.rs`

```rust
use engram_core::MemoryEngine;
use chrono::Duration;

#[test]
fn test_spacing_effect_replication() {
    let engine = MemoryEngine::new();

    let study_items = generate_random_facts(50);

    // Group 1: Massed practice (3 consecutive exposures)
    let massed_group = &study_items[0..25];
    for item in massed_group {
        for _ in 0..3 {
            engine.store_episode(item.clone());
        }
    }

    // Group 2: Distributed practice (3 exposures, 1 hour apart)
    // Note: Use accelerated time for testing
    let distributed_group = &study_items[25..50];
    for item in distributed_group {
        engine.store_episode(item.clone());
        engine.advance_time(Duration::hours(1));  // Simulated time
        engine.store_episode(item.clone());
        engine.advance_time(Duration::hours(1));
        engine.store_episode(item.clone());
    }

    // Retention test after 24 hours
    engine.advance_time(Duration::hours(24));

    let massed_accuracy = test_retention(&engine, massed_group);
    let distributed_accuracy = test_retention(&engine, distributed_group);

    let improvement = (distributed_accuracy - massed_accuracy) / massed_accuracy;

    // Acceptance: 20-40% ±10% = [10%, 50%]
    assert!(
        improvement >= 0.10 && improvement <= 0.50,
        "Spacing effect {:.1}% outside [10%, 50%] acceptance range (Cepeda 2006)",
        improvement * 100.0
    );

    // Log results
    println!("Spacing Effect Validation:");
    println!("  Massed accuracy: {:.1}%", massed_accuracy * 100.0);
    println!("  Distributed accuracy: {:.1}%", distributed_accuracy * 100.0);
    println!("  Improvement: {:.1}%", improvement * 100.0);
    println!("  Target range: 10-50% (Cepeda et al. 2006)");
}

fn test_retention(engine: &MemoryEngine, items: &[Episode]) -> f32 {
    let mut correct = 0;
    for item in items {
        let recall_result = engine.recall_by_cue(&item.cue);
        if recall_result.is_successful() && recall_result.matches(item) {
            correct += 1;
        }
    }
    correct as f32 / items.len() as f32
}
```

## Integration Points

**M4 (Temporal Dynamics):** Relies on forgetting curves
- File: `engram-core/src/decay/mod.rs`
- Mechanism: Distributed practice benefits from retrieval practice effect

**M6 (Consolidation):** Consolidation between study sessions
- File: `engram-core/src/decay/consolidation.rs`
- Mechanism: Distributed practice allows consolidation between exposures

**Existing Tests:** Similar to forgetting curve validation
- File: `engram-core/tests/forgetting_curves_validation.rs`
- Reuse: Time simulation utilities

## Testing Strategy

### Statistical Validation
- **Sample Size:** 50 items (25 massed, 25 distributed)
- **Repetitions:** Run test 10 times for stability
- **Statistical Test:** Paired t-test, p < 0.05
- **Power:** >0.80 (sufficient to detect 20% difference)

### Edge Cases
1. Zero spacing (degrades to massed)
2. Very long spacing (>24 hours)
3. Unequal number of exposures

## Acceptance Criteria

### Must Have
- [ ] Cepeda et al. (2006) replication: 20-40% improvement ±10%
- [ ] Statistical significance: p < 0.05 (paired t-test)
- [ ] Massed vs distributed comparison with matched items
- [ ] Test passes consistently (8/10 runs minimum)
- [ ] Results logged with statistical details

### Should Have
- [ ] Multiple spacing intervals tested (1h, 4h, 12h)
- [ ] Replication with different content types (words, facts, etc.)

### Nice to Have
- [ ] Optimal spacing interval identified
- [ ] Visualization of retention curves

## Implementation Checklist

- [ ] Create `spacing_effect.rs` test file
- [ ] Implement massed practice condition
- [ ] Implement distributed practice condition (1-hour spacing)
- [ ] Implement retention test after 24 hours
- [ ] Add statistical significance calculation (t-test)
- [ ] Run test 10 times and verify consistency
- [ ] Verify acceptance range [10%, 50%]
- [ ] Log results with target range
- [ ] Document in validation report

## Risks and Mitigations

**Risk:** Test fails due to forgetting curve parameters
- **Mitigation:** M4 forgetting curves already validated
- **Mitigation:** If fails, tune consolidation parameters
- **Mitigation:** Budget +0.5 day for tuning

**Risk:** Time simulation affects results
- **Mitigation:** Use M4's time advancement utilities
- **Mitigation:** Verify time simulation doesn't introduce artifacts

## References

1. Cepeda, N. J., et al. (2006). Distributed practice in verbal recall tasks. *Psychological Bulletin*, 132(3), 354.
2. Bjork, R. A., & Bjork, E. L. (1992). A new theory of disuse. *From Learning Processes to Cognitive Processes*, 2, 35-67.
