# Task 006: Completion Confidence Calibration

**Status:** Pending
**Priority:** P0 (Critical Path)
**Estimated Effort:** 2 days
**Dependencies:** Task 002 (CA3 Dynamics), Task 005 (Source Attribution)

## Objective

Implement multi-factor confidence computation for pattern completion combining CA3 convergence speed, attractor basin depth, pattern strength, and source consensus. Achieve <8% calibration error across confidence bins using empirical validation framework from Milestone 5.

## Integration Points

**Uses:**
- `/engram-core/src/query/confidence_calibration.rs` - Calibration framework from M5
- `/engram-core/src/completion/attractor_dynamics.rs` - Convergence stats from Task 002
- `/engram-core/src/completion/source_monitor.rs` - Source attribution from Task 005
- `/engram-core/src/completion/confidence.rs` - MetacognitiveConfidence

**Creates:**
- `/engram-core/src/completion/completion_confidence.rs` - Multi-factor confidence computation
- `/engram-core/src/completion/calibration.rs` - Completion-specific calibration
- `/engram-core/tests/completion_confidence_tests.rs` - Calibration accuracy tests

## Theoretical Foundations from Research

### Brier Score (1950) - Calibration Measurement

Measures accuracy of probabilistic predictions. Penalizes both under-confidence and over-confidence.

**Formula:**
```
BS = (1/N) Σ (forecast - outcome)²
```

**Perfect calibration:** Brier score = 0
**Well-calibrated:** Events predicted at 70% confidence occur 70% of the time

**Engram Application:**
For each completion, record predicted confidence and actual accuracy (ground truth comparison in Task 009). Compute Brier score across all completions.

**Target:** Brier score <0.08 (corresponding to <8% calibration error)

### Calibration Curves (Murphy & Winkler, 1977)

Plot predicted probability vs. observed frequency in bins.

**Perfect calibration:** Diagonal line (predicted = observed)
**Over-confident:** Above diagonal (70% predictions → 50% actual accuracy)
**Under-confident:** Below diagonal (50% predictions → 70% actual accuracy)

**Implementation:**
```
Bin completion confidences: [0-0.1, 0.1-0.2, ..., 0.9-1.0]
For each bin, measure actual reconstruction accuracy
Calibration error = deviation from diagonal
```

**Monitoring (Task 008):** `engram_completion_confidence_calibration_error{memory_space, bin}` tracks per-bin calibration.

### Multi-Factor Confidence Computation

Confidence should aggregate multiple signals, not rely on single score.

**Factors for Pattern Completion (from research):**

**1. Convergence Speed (CA3 iterations):** Faster → higher confidence
```rust
convergence_factor = 1.0 - (iterations as f32 / max_iterations as f32)
weight = 0.3  // convergence_weight
```

**2. Energy Reduction (attractor depth):** Deeper basin → higher confidence
```rust
energy_factor = (energy_delta.abs() / 10.0).min(1.0)
weight = 0.25  // energy_weight
```

**3. Field Consensus (neighbor agreement):** Higher agreement → higher confidence
```rust
field_consensus = weighted_agreement_ratio from Task 001
weight = 0.25  // consensus_weight
```

**4. Plausibility Score (semantic coherence):** More plausible → higher confidence
```rust
plausibility = HNSW neighborhood consistency check
weight = 0.2  // plausibility_weight
```

**Weighted Combination:**
```rust
confidence = (convergence_factor * 0.3) + (energy_factor * 0.25)
           + (field_consensus * 0.25) + (plausibility * 0.2)
```

Weights sum to 1.0. No single factor dominates (all <60%).

### Koriat's Cue-Utilization Framework (1997)

People use multiple cues to assess confidence:

**Intrinsic cues:** Properties of item itself
- Engram: Pattern strength, cue overlap

**Extrinsic cues:** Learning conditions
- Engram: Number of source episodes, pattern age from consolidation

**Mnemonic cues:** Retrieval fluency
- Engram: **Convergence speed** (faster = more fluent), **consensus** (agreement = easier)

**Analog for Completion:**
All four factors map to Koriat's framework, providing theoretically-grounded confidence computation.

### Metacognitive Confidence (Fleming & Dolan, 2012)

Metacognitive confidence = confidence in your confidence.

**Neural substrate:** Prefrontal cortex monitors primary memory systems, assesses reliability.

**Engram Implementation:**
Compare alternative hypotheses (Task 005) for internal consistency:
- **High consistency:** Alternatives agree on key fields → high metacognitive confidence
- **Low consistency:** Alternatives diverge → uncertainty about reconstruction quality

```rust
pub struct MetacognitiveMonitor {
    fn compute_metacognitive_confidence(
        &self,
        completion_confidence: Confidence,
        alternative_hypotheses: &[(Episode, Confidence)],
    ) -> Confidence {
        // Measure agreement across alternatives
        let consistency = self.check_consistency(alternatives);
        // Higher consistency → trust completion_confidence more
    }
}
```

### Calibration Frameworks: Isotonic Regression (Zadrozny & Elkan, 2002)

Post-hoc calibration: Learn monotonic mapping from raw scores to calibrated probabilities.

**Advantages over Platt Scaling:**
- Non-parametric (fewer assumptions)
- Handles non-linear calibration curves
- Better fits empirical calibration data

**Training:**
- Collect 1000+ completions with ground truth (Task 009 validation dataset)
- Learn isotonic mapping from raw confidence → calibrated probability
- Ensure monotonicity (higher raw score → higher calibrated probability)

**Application:**
Transform completion confidence to calibrated probability that matches actual accuracy.

**Engram Choice:** Isotonic regression (implemented in Task 009 parameter tuning)

### Confidence-Accuracy Correlation (Spearman Rank)

Measure correlation between confidence ranking and accuracy ranking.

**Perfect correlation:** ρ = 1.0 (items ranked high confidence are most accurate)
**No correlation:** ρ = 0.0 (confidence uninformative)

**Target:** ρ > 0.80 for well-calibrated system (acceptance criterion)

**Resolution vs Calibration Trade-off:**
- **High resolution, poor calibration:** Over-confident on easy, under-confident on hard
- **Low resolution, good calibration:** All confidences near average (uninformative)
- **Optimal:** High resolution AND good calibration

**Engram Goal:** Both properties through multi-factor confidence + isotonic calibration

## Detailed Specification

### Multi-Factor Confidence Computation

```rust
// /engram-core/src/completion/completion_confidence.rs

use crate::Confidence;

pub struct CompletionConfidenceComputer {
    /// Convergence speed weight (default: 0.3)
    convergence_weight: f32,

    /// Energy reduction weight (default: 0.25)
    energy_weight: f32,

    /// Field consensus weight (default: 0.25)
    consensus_weight: f32,

    /// Plausibility weight (default: 0.2)
    plausibility_weight: f32,
}

impl CompletionConfidenceComputer {
    pub fn compute_completion_confidence(
        &self,
        convergence_stats: &ConvergenceStats,
        field_consensus: f32,
        plausibility_score: f32,
    ) -> Confidence;

    /// Convergence factor: faster = higher confidence
    fn convergence_factor(iterations: usize, max_iterations: usize) -> f32 {
        1.0 - (iterations as f32 / max_iterations as f32)
    }

    /// Energy factor: deeper attractor = higher confidence
    fn energy_factor(energy_delta: f32) -> f32 {
        (energy_delta.abs() / 10.0).min(1.0) // Normalize by typical delta
    }
}

pub struct MetacognitiveMonitor {
    /// Internal consistency checker
    consistency_threshold: f32,
}

impl MetacognitiveMonitor {
    /// Compute metacognitive confidence (confidence in our confidence)
    pub fn compute_metacognitive_confidence(
        &self,
        completion_confidence: Confidence,
        alternative_hypotheses: &[(Episode, Confidence)],
    ) -> Confidence;

    /// Check internal consistency of alternative hypotheses
    fn check_consistency(
        &self,
        alternatives: &[(Episode, Confidence)],
    ) -> f32;
}
```

## Acceptance Criteria

1. **Calibration Error:** <8% calibration error across 10 confidence bins
2. **Correlation:** Confidence correlates >0.80 with reconstruction accuracy (Spearman)
3. **Multi-Factor Balance:** All four factors contribute meaningfully (no single factor >60% weight)
4. **Metacognitive Accuracy:** Metacognitive confidence correlates >0.75 with actual confidence accuracy
5. **Performance:** Confidence computation <200μs

## Testing Strategy

**Calibration Tests:** 1000+ completions with ground truth; measure calibration curve

**Correlation Tests:** Spearman rank correlation on validation sets

**Factor Ablation:** Remove each factor; verify accuracy degrades >10%

## Success Criteria Validation

- [ ] Calibration error <8%
- [ ] Confidence correlation >0.80
- [ ] All factors contribute (no >60% dominance)
- [ ] Metacognitive correlation >0.75
- [ ] Computation <200μs
