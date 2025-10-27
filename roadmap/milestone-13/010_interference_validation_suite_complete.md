# Task 010: Interference Validation Suite

**Status:** PENDING
**Priority:** P1 (Validation)
**Estimated Duration:** 1 day
**Dependencies:** Tasks 004, 005 (All interference types implemented)
**Agent Review Required:** verification-testing-lead

## Overview

Comprehensive validation suite that verifies all three interference types (proactive, retroactive, fan effect) replicate published empirical data within acceptance criteria.

## Validation Targets

### Proactive Interference (Underwood 1957)
- **Target:** 20-30% accuracy reduction with 5+ prior lists
- **Acceptance:** ±10% → [10%, 40%] accuracy reduction
- **Mechanism:** Similar prior memories interfere with new learning

### Retroactive Interference (McGeoch 1942)
- **Target:** 15-25% accuracy reduction with 1 interpolated list
- **Acceptance:** ±10% → [5%, 35%] accuracy reduction
- **Mechanism:** New learning interferes with old memories

### Fan Effect (Anderson 1974)
- **Target:** 50-150ms RT increase per additional association
- **Acceptance:** ±25ms → [25ms, 175ms] per association
- **Mechanism:** More associations slow retrieval

## Implementation Specifications

### File Structure
```
engram-core/tests/psychology/
├── interference_validation.rs (new)
└── interference_test_data.json (standard test materials)
```

### Validation Suite

**File:** `/engram-core/tests/psychology/interference_validation.rs`

```rust
mod proactive_validation {
    #[test]
    fn test_underwood_1957_validation() {
        // Exactly replicates Underwood (1957) experimental design
        // Expected: 20-30% ±10% accuracy reduction
    }
}

mod retroactive_validation {
    #[test]
    fn test_mcgeoch_1942_validation() {
        // Exactly replicates McGeoch (1942) experimental design
        // Expected: 15-25% ±10% accuracy reduction
    }
}

mod fan_effect_validation {
    #[test]
    fn test_anderson_1974_validation() {
        // Exactly replicates Anderson (1974) experimental design
        // Expected: 50-150ms ±25ms per association
    }
}

mod comprehensive_validation {
    #[test]
    fn test_all_interference_types_comprehensive() {
        // Run all three validations in sequence
        // Generate validation report with all statistics
    }
}
```

## Acceptance Criteria

### Must Have
- [ ] Underwood (1957) PI validation within ±10%
- [ ] McGeoch (1942) RI validation within ±10%
- [ ] Anderson (1974) fan effect within ±25ms
- [ ] All tests pass with statistical significance (p < 0.05)
- [ ] Validation report generated with all statistics
- [ ] Tests run in CI pipeline

### Should Have
- [ ] Effect size (Cohen's d) calculated for each
- [ ] Multiple replications (n=30) for stability
- [ ] Correlation with published data >0.80

### Nice to Have
- [ ] Visualization of interference effects
- [ ] Comparison table with published data
- [ ] Sensitivity analysis for parameters

## Implementation Checklist

- [ ] Create `interference_validation.rs`
- [ ] Implement Underwood (1957) replication
- [ ] Implement McGeoch (1942) replication
- [ ] Implement Anderson (1974) replication
- [ ] Add statistical significance tests
- [ ] Generate validation report
- [ ] Verify all tests pass acceptance criteria
- [ ] Document results in milestone report

## References

1. Underwood, B. J. (1957). Interference and forgetting. *Psychological Review*, 64(1), 49.
2. McGeoch, J. A. (1942). The psychology of human learning.
3. Anderson, J. R. (1974). Retrieval of propositional information. *Cognitive Psychology*, 6(4), 451-474.
