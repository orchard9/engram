# Task 004: Global Pattern Integration

**Status:** Pending
**Priority:** P0 (Critical Path)
**Estimated Effort:** 2 days
**Dependencies:** Task 001 (Field Reconstruction), Task 003 (Pattern Retrieval)

## Objective

Implement hierarchical evidence aggregation combining local temporal context (Task 001) with global semantic patterns (Task 003). Use Bayesian evidence combination to produce coherent completions that balance immediate context with learned statistical regularities.

## Integration Points

**Uses:**
- `/engram-core/src/completion/field_reconstruction.rs` - Local field reconstructions from Task 001
- `/engram-core/src/completion/pattern_retrieval.rs` - Global pattern retrieval from Task 003
- `/engram-core/src/query/evidence_aggregator.rs` - Bayesian evidence combination from M5
- `/engram-core/src/query/confidence_calibration.rs` - Confidence propagation from M5

**Creates:**
- `/engram-core/src/completion/evidence_integration.rs` - Hierarchical evidence aggregation
- `/engram-core/src/completion/global_patterns.rs` - Global pattern application logic
- `/engram-core/tests/pattern_integration_tests.rs` - Integration correctness tests

## Theoretical Foundations from Research

### Bayesian Evidence Combination (Pearl, 1988)

Hierarchical integration implements Pearl's framework for combining evidence from multiple independent sources:

**For independent evidence:**
```
P(H | E1, E2) ∝ P(E1 | H) × P(E2 | H) × P(H)
```

**Application to Engram:**
- **H:** True episode being reconstructed
- **E1:** Local context evidence from temporal neighbors (Task 001)
- **E2:** Global pattern evidence from semantic patterns (Task 003)
- **Independence:** Conditionally independent given true episode

**Implementation in bayesian_combine():**
When local and global agree on field value:
```rust
// Agreement → multiplicative confidence boost
let combined_conf = evidence_aggregator.combine_independent(local_conf, global_conf);
```

### Maximum Entropy Principle (Jaynes, 1957)

When multiple distributions consistent with constraints, choose maximum entropy (minimum assumptions).

**Application:** When local and global evidence conflict, weight by strength rather than forcing agreement:
```rust
if local_value != global_value {
    // Choose higher-confidence source (don't artificially merge)
    if local_conf >= global_conf {
        return (local_value.to_string(), local_conf);
    } else {
        return (global_value.to_string(), global_conf);
    }
}
```

Avoids introducing unfounded assumptions by respecting evidence quality.

### Hierarchical Bayesian Models (Hemmer & Steyvers, 2009)

Memory reconstruction uses hierarchical inference:
- **Prior:** Semantic knowledge from global patterns (statistical regularities)
- **Likelihood:** Episodic evidence from local context (specific instances)
- **Posterior:** Reconstructed memory (integrated completion)

**Strength-dependent blending:**
- Weak episodic evidence → rely more on semantic priors (global patterns)
- Strong episodic evidence → rely more on specific neighbors (local context)

**Implementation in compute_adaptive_weights():**
```rust
let local_strength = local_confidence.raw();
let global_strength = global_consensus;

let total_strength = local_strength + global_strength;
let local_weight = local_strength / total_strength;
let global_weight = global_strength / total_strength;
```

Adaptive weighting implements Bayesian posterior proportional to evidence strength.

### Condorcet's Jury Theorem (1785) & Agreement Boosting

When independent sources agree, confidence multiplies:

**Theorem:** If sources have >50% accuracy and are independent, their agreement is more accurate than any individual.

**Modern Application (Ensemble Methods, Breiman 1996):**
- Local and global are diverse by construction (temporal vs statistical)
- Both exceed random baseline (validated in Task 009)
- Agreement signals high-confidence reconstruction

**Expected Boost:** 15-25% confidence increase when sources agree (acceptance criterion)

### Adaptive Weighting Strategy (Kahneman & Tversky, 1974)

Humans adaptively weight evidence based on perceived reliability. High-certainty evidence receives higher weight.

**Normalized weighting formula:**
```rust
weight_local = confidence_local / (confidence_local + confidence_global)
weight_global = confidence_global / (confidence_local + confidence_global)
```

Ensures weights sum to 1.0 while respecting relative evidence strength.

## Detailed Specification

### 1. Hierarchical Evidence Aggregator

```rust
// /engram-core/src/completion/evidence_integration.rs

use crate::Confidence;
use crate::completion::{ReconstructedField, RankedPattern, PartialEpisode};
use crate::query::evidence_aggregator::EvidenceAggregator;

pub struct HierarchicalEvidenceAggregator {
    /// Bayesian evidence aggregator from M5
    evidence_aggregator: EvidenceAggregator,

    /// Local context weight (default: 0.6)
    local_weight: f32,

    /// Global pattern weight (default: 0.4)
    global_weight: f32,

    /// Adaptive weighting based on evidence quality
    adaptive_weighting: bool,
}

impl HierarchicalEvidenceAggregator {
    pub fn new() -> Self;

    /// Integrate local and global evidence for field reconstruction
    pub fn integrate_evidence(
        &self,
        field_name: &str,
        local_reconstruction: Option<&ReconstructedField>,
        global_patterns: &[RankedPattern],
    ) -> IntegratedField;

    /// Compute adaptive weights based on evidence quality
    fn compute_adaptive_weights(
        &self,
        local_confidence: Confidence,
        global_consensus: f32,
    ) -> (f32, f32); // (local_weight, global_weight)

    /// Extract field value from global patterns using consensus
    fn extract_global_field(
        &self,
        field_name: &str,
        patterns: &[RankedPattern],
    ) -> Option<(String, Confidence)>;

    /// Combine local and global field values using Bayesian updating
    fn bayesian_combine(
        &self,
        local_value: &str,
        local_conf: Confidence,
        global_value: &str,
        global_conf: Confidence,
    ) -> (String, Confidence);
}

#[derive(Debug, Clone)]
pub struct IntegratedField {
    pub value: String,
    pub confidence: Confidence,
    pub local_contribution: f32,
    pub global_contribution: f32,
    pub evidence_sources: Vec<EvidenceSource>,
}

#[derive(Debug, Clone)]
pub enum EvidenceSource {
    LocalContext { episode_id: String, weight: f32 },
    GlobalPattern { pattern_id: String, weight: f32 },
}
```

### 2. Adaptive Weighting Strategy

```rust
impl HierarchicalEvidenceAggregator {
    fn compute_adaptive_weights(
        &self,
        local_confidence: Confidence,
        global_consensus: f32,
    ) -> (f32, f32) {
        if !self.adaptive_weighting {
            return (self.local_weight, self.global_weight);
        }

        // High local confidence → favor local
        // High global consensus → favor global
        // Low both → balanced weighting

        let local_strength = local_confidence.raw();
        let global_strength = global_consensus;

        let total_strength = local_strength + global_strength;
        if total_strength < 0.1 {
            // Both weak → balanced
            return (0.5, 0.5);
        }

        let local_weight = local_strength / total_strength;
        let global_weight = global_strength / total_strength;

        (local_weight, global_weight)
    }

    fn bayesian_combine(
        &self,
        local_value: &str,
        local_conf: Confidence,
        global_value: &str,
        global_conf: Confidence,
    ) -> (String, Confidence) {
        if local_value == global_value {
            // Agreement → boost confidence
            let combined_conf = self.evidence_aggregator.combine_independent(
                local_conf,
                global_conf,
            );
            return (local_value.to_string(), combined_conf);
        }

        // Disagreement → choose higher confidence source
        if local_conf >= global_conf {
            (local_value.to_string(), local_conf)
        } else {
            (global_value.to_string(), global_conf)
        }
    }
}
```

## Acceptance Criteria

1. **Integration Accuracy:** Combined local+global achieves >85% field accuracy (vs >80% local-only, >75% global-only)
2. **Adaptive Weighting:** Automatically balances sources based on evidence quality; improves accuracy by >10% vs fixed weighting
3. **Agreement Boosting:** When local and global agree, confidence increases by 15-25%
4. **Disagreement Handling:** Selects higher-confidence source; maintains >80% accuracy in disagreement cases
5. **Performance:** Evidence integration <1ms P95 per field

## Testing Strategy

**Unit Tests:** Adaptive weight computation, Bayesian combination, agreement boosting, disagreement resolution

**Integration Tests:** End-to-end completion with local+global sources, ablation studies (local-only vs global-only vs combined)

**Property Tests:** Confidence always bounded [0,1], weights sum to 1.0, agreement never decreases confidence

## Risk Mitigation

**Risk:** Local and global sources frequently disagree (>40% of cases)
- **Mitigation:** Confidence-based selection; prefer higher-confidence source
- **Contingency:** Three-way voting with CA3 output as tiebreaker

**Risk:** Adaptive weighting introduces latency
- **Mitigation:** Pre-compute weights based on partial episode statistics
- **Contingency:** Disable adaptive weighting (use fixed weights)

## Success Criteria Validation

- [ ] Combined accuracy >85% (exceeds both local-only and global-only)
- [ ] Adaptive weighting improves accuracy by >10% vs fixed
- [ ] Agreement cases show 15-25% confidence boost
- [ ] Evidence integration <1ms P95
- [ ] All tests pass
