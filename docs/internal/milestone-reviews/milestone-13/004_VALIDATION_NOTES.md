# Task 004: Proactive Interference Detection - Validation Notes

**Biological Plausibility:** PASS (with temporal window adjustment)
**Reviewer:** Randy O'Reilly (memory-systems-researcher)

## Critical Parameter Corrections

### 1. Temporal Window: 24 hours → 6 hours

**Current (Line 41):** "Only memories within 24 hours prior count as interfering"
**Problem:** Underwood (1957) used session-based interference (minutes to hours), not day-scale. After 24 hours, consolidation reduces interference.

**Required Change:**
```rust
/// Temporal window for "prior" memories (default: 6 hours before)
/// Empirical: Underwood (1957) session-based interference
/// Justification: Synaptic consolidation (~6h) transitions memories from
/// hippocampal to neocortical representations, reducing interference
prior_memory_window: Duration,  // Duration::hours(6)
```

**Biological Rationale:**
- Synaptic consolidation completes within ~6 hours (protein synthesis window)
- After consolidation, memories shift from hippocampal-dependent → neocortical
- Cross-consolidation-boundary interference is reduced (CLS theory)

### 2. Linear Accumulation: CORRECT

**Current (Lines 42-43):** 5% per similar item, capped at 30%
**Validation:** Matches Underwood (1957) empirical data. With 6 prior lists, recall dropped from ~70% to ~45% (25% reduction). 5% × 5 = 25% ✓

**No changes needed** - linear model is appropriate.

### 3. Similarity Threshold: 0.7 (REASONABLE)

**Current (Line 42):** Embedding similarity ≥ 0.7 for interference
**Assessment:** Threshold is reasonable but needs empirical calibration.

**Required Validation Test:**
```rust
#[test]
fn test_similarity_threshold_calibration_with_category_structure() {
    // Within-category pairs (dog/cat): similarity should be ≥ 0.7
    // Across-category pairs (dog/car): similarity should be < 0.7
    // Validates threshold captures semantic category interference
}
```

### 4. Temporal Direction: CORRECT

**Current (Line 44):** "Only forward-in-time interference (old → new)"
**Validation:** Implementation correctly checks `prior_episode.timestamp >= new_episode.timestamp` (line 150) ✓

## Updated Default Parameters

```rust
impl Default for ProactiveInterferenceDetector {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,          // KEEP - validate with category tests
            prior_memory_window: Duration::hours(6),  // CHANGE from 24h
            interference_per_item: 0.05,        // KEEP - matches Underwood 1957
            max_interference: 0.30,             // KEEP - empirically grounded
        }
    }
}
```

## Additional Validation Tests Required

```rust
#[test]
fn test_consolidation_boundary_reduces_interference() {
    // Prior memory learned >6 hours ago: reduced/no interference
    // Prior memory learned <6 hours ago: full interference
    // Validates synaptic consolidation effect per CLS theory
}

#[test]
fn test_interference_magnitude_matches_underwood_1957() {
    // 5 similar prior lists within 6h window
    // Expected: 20-30% accuracy reduction
    // Validates against Underwood (1957) Figure 2 empirical data
}

#[test]
fn test_temporal_window_enforced_exactly() {
    let detector = ProactiveInterferenceDetector::default();

    // Prior at 5h 59m: should interfere
    // Prior at 6h 01m: should NOT interfere
    // Validates exact boundary per is_interfering() logic
}
```

## Integration with M3/M8 Systems

**M3 (Activation Spreading):** Apply interference during recall ✓
```rust
let adjusted_activation = base_activation * (1.0 - interference.magnitude);
```

**M8 (Pattern Completion):** Interfering memories compete with reconstructed patterns ✓

**Metrics:** Record interference events with magnitude and count ✓

## Complementary Learning Systems Alignment

**Hippocampal System:**
- Proactive interference strongest during rapid encoding phase ✓
- Similar patterns in hippocampus compete for consolidation ✓

**Temporal Boundary:**
- 6-hour window captures within-consolidation interference ✓
- After consolidation, memories are less susceptible (CLS prediction) ✓

**Neocortical System:**
- Long-term interference (>24h) would operate via different mechanisms
- Task correctly focuses on short-term (session-based) interference

## References

- Underwood, B. J. (1957). Interference and forgetting. *Psychological Review*, 64(1), 49.
- Anderson, M. C., & Neely, J. H. (1996). Interference and inhibition in memory retrieval. *Memory*, 125-153.
- McClelland et al. (1995). Why there are complementary learning systems. *Psych Review*, 102(3), 419.
