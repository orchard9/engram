# Task 003: Associative and Repetition Priming - Validation Notes

**Biological Plausibility:** PASS (with corrections)
**Reviewer:** Randy O'Reilly (memory-systems-researcher)

## Critical Parameter Corrections

### 1. Co-occurrence Window: 10s → 30s

**Current (Line 20):** 10 seconds for co-occurrence counting
**Empirical Basis:** McKoon & Ratcliff (1992) used inter-trial intervals of 2-30 seconds. Working memory span governs functional co-occurrence (~7±2 items × 3-4s per item = ~30s total).

**Required Change:**
```rust
/// Temporal window for co-occurrence (default: 30 seconds)
/// Empirical: Working memory span + central executive integration
cooccurrence_window: Duration,  // Duration::seconds(30)
```

### 2. Minimum Co-occurrence: 3 → 2

**Current (Line 21):** 3 co-occurrences before association forms
**Empirical Basis:** Saffran et al. (1996) statistical learning shows reliable associations after ~2 exposures. Single-trial learning exists but is prone to spurious associations.

**Required Change:**
```rust
/// Minimum co-occurrence for reliable association (default: 2)
/// Empirical: Saffran et al. (1996) statistical learning threshold
min_cooccurrence: u64,  // 2
```

### 3. Repetition Priming Parameters: CORRECT

**Current (Lines 28-29):** 5% boost per exposure, 30% ceiling
**Validation:** Matches Tulving & Schacter (1990) perceptual priming data (20-50% RT reduction over 3-6 exposures). **No changes needed.**

### 4. Additive Combination Needs Saturation

**Current (Line 156):** Linear additive: `semantic + associative + repetition`
**Problem:** Can exceed 100% boost with perfect storm of all priming types.

**Required Change:**
```rust
pub fn compute_total_boost(&self, node_id: NodeId) -> f32 {
    let semantic = self.semantic.compute_priming_boost(node_id);
    let associative = self.associative.compute_association_strength(/* ... */);
    let repetition = self.repetition.compute_repetition_boost(node_id);

    // Additive with saturation (diminishing returns)
    let linear_sum = semantic + associative + repetition;
    1.0 - (-linear_sum).exp()  // Never exceeds ~63% even with maximum all types
}
```

**Justification:** Neural firing rate saturation, behavioral ceiling effects in RT studies (Neely & Keefe 1989).

## Additional Validation Tests Required

```rust
#[test]
fn test_cooccurrence_window_matches_working_memory_span() {
    // Items within 30s window should co-activate
    // Items beyond 30s should not (working memory cleared)
}

#[test]
fn test_priming_saturation_prevents_overshoot() {
    // semantic=0.3, associative=0.3, repetition=0.3
    // linear_sum = 0.9 → should saturate to ~0.60, not 0.90
}

#[test]
fn test_minimum_cooccurrence_prevents_spurious_associations() {
    // Single co-occurrence: no association formed
    // Two co-occurrences: association strength > 0
}
```

## Integration with M3/M4 Systems

**Validated:** Priming boosts applied during spreading activation (M3) ✓
**Validated:** Semantic priming decay via M4 temporal dynamics ✓
**Clarification Needed:** Associative and repetition priming marked as "no decay within session" (line 171) - confirm this matches session boundary definition in M4.

## References

- McKoon & Ratcliff (1992): Spreading activation versus compound cue accounts of priming. *Psych Review*, 99(1), 177.
- Tulving & Schacter (1990): Priming and human memory systems. *Science*, 247(4940), 301-306.
- Saffran et al. (1996): Statistical learning by 8-month-old infants. *Science*, 274(5294), 1926-1928.
- Neely & Keefe (1989): Semantic context effects on visual word processing. *Psych of Learning and Motivation*, 24, 207-248.
