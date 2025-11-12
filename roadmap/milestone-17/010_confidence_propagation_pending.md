# Task 010: Confidence Propagation with Formal Verification

## Objective
Implement mathematically sound confidence propagation through the dual memory architecture with comprehensive formal verification, ensuring uncertainty is tracked correctly through concept formation, binding traversal, and blended recall.

## Background
Confidence scores must propagate correctly through all dual memory operations to maintain the probabilistic integrity of the system. The existing confidence system (engram-core/src/lib.rs:88-200) provides bounded probability values with cognitive-friendly operations. This task extends that foundation to handle episodic-to-semantic and semantic-to-episodic propagation with provable correctness.

Existing confidence infrastructure:
- `Confidence` newtype with [0, 1] bounds (lib.rs)
- `ConfidenceAggregator` for multi-path convergence (activation/confidence_aggregation.rs)
- `CalibrationTracker` for empirical validation (query/confidence_calibration.rs)
- `SMTVerificationSuite` for formal proof (query/verification.rs)
- Property-based tests using proptest (tests/confidence_property_tests.rs)

## Requirements
1. Define mathematically sound confidence rules for concept formation
2. Implement provably correct confidence decay through bindings
3. Calculate blended confidence scores with formal guarantees
4. Track confidence through spreading paths without inflation
5. Ensure mathematical consistency via SMT verification
6. Validate empirical accuracy through calibration tracking

## Mathematical Properties to Verify

### Core Invariants (SMT Verification Required)
1. **Monotonicity**: Confidence never increases through propagation
   - ∀ source_conf, decay_rate, binding_strength: propagated_conf ≤ source_conf
   - No information gain from graph traversal

2. **Bounds Preservation**: All confidence values stay in [0, 1]
   - ∀ operation: 0 ≤ output_confidence ≤ 1
   - Type system + runtime checks + SMT proof

3. **No Cycle Inflation**: Repeated propagation through cycles converges to zero
   - For any cycle with decay &lt; 1.0: lim(n→∞) conf_n = 0
   - Prevents confidence accumulation in cycles

4. **Blend Bonus Justification**: Convergent evidence bonus is mathematically valid
   - When episodic_conf > 0.5 AND semantic_conf > 0.5:
     - Bonus = min(1.1 * weighted_avg, 1.0)
     - Justification: Independent evidence from two memory systems
     - Must prove: blended_conf ≤ max(episodic_conf, semantic_conf) * 1.1

5. **Binding Decay Correctness**: Multi-hop attenuation follows exponential decay
   - propagated = source * binding_strength * decay_rate
   - After n hops: conf_n = conf_0 * (binding_strength * decay_rate)^n
   - Verify convergence properties

## Technical Specification

### Files to Create
- `engram-core/src/confidence/dual_memory.rs` - Dual memory confidence rules
- `engram-core/src/confidence/verification.rs` - Formal verification for dual memory
- `engram-core/tests/confidence_propagation_properties.rs` - Property-based tests

### Files to Modify
- `engram-core/src/confidence/mod.rs` - Add dual_memory module
- `engram-core/src/query/verification.rs` - Add dual memory axioms

### Dual Memory Confidence Rules
```rust
pub struct DualMemoryConfidence {
    concept_formation_penalty: f32, // Default 0.9 (information loss in clustering)
    binding_confidence_decay: f32,  // Default 0.95 (per-hop attenuation)
    blend_confidence_bonus: f32,    // Default 1.1 (convergent evidence)
}

impl DualMemoryConfidence {
    /// Calculate concept confidence from clustered episodes
    ///
    /// Formal property: concept_conf ≤ max(episode_confs) * penalty * coherence
    pub fn concept_confidence(
        &self,
        episodes: &[Episode],
        coherence: f32, // [0, 1] from clustering quality
    ) -> Confidence {
        // Average episode confidence weighted by coherence
        let avg_confidence = episodes.iter()
            .map(|e| e.confidence.raw())
            .sum::<f32>() / episodes.len() as f32;

        // Apply formation penalty and coherence weighting
        let concept_conf = avg_confidence * self.concept_formation_penalty * coherence;

        Confidence::exact(concept_conf)
    }

    /// Propagate confidence through bindings
    ///
    /// Formal property: propagated ≤ source (monotonicity)
    pub fn propagate_through_binding(
        &self,
        source_confidence: Confidence,
        binding_strength: f32, // [0, 1]
    ) -> Confidence {
        let propagated = source_confidence.raw()
            * binding_strength
            * self.binding_confidence_decay;

        Confidence::exact(propagated)
    }

    /// Calculate blended recall confidence
    ///
    /// Formal properties:
    /// 1. blended ≤ max(episodic, semantic) * blend_bonus
    /// 2. If no bonus: blended = weighted_avg(episodic, semantic)
    /// 3. Bonus only when both > 0.5 (convergent evidence threshold)
    pub fn blend_confidence(
        &self,
        episodic_conf: Confidence,
        semantic_conf: Confidence,
        episodic_weight: f32,
        semantic_weight: f32,
    ) -> Confidence {
        // Weighted average with blend bonus
        let total_weight = episodic_weight + semantic_weight;
        let base = (episodic_conf.raw() * episodic_weight
                   + semantic_conf.raw() * semantic_weight)
                   / total_weight;

        // Apply blend bonus for convergent evidence
        // Justification: Independent confirmation from hippocampal (episodic)
        // and neocortical (semantic) systems reduces uncertainty
        let blended = if episodic_conf.raw() > 0.5 && semantic_conf.raw() > 0.5 {
            (base * self.blend_confidence_bonus).min(1.0)
        } else {
            base
        };

        Confidence::exact(blended)
    }

    /// Verify no confidence inflation through multi-hop propagation
    ///
    /// Property: For n hops, conf_n ≤ conf_0 * decay^n
    pub fn verify_multi_hop_decay(&self, hops: u32) -> bool {
        let initial = Confidence::exact(0.9);
        let mut current = initial;

        for _ in 0..hops {
            current = self.propagate_through_binding(current, 0.8);
        }

        // Should decay exponentially
        let expected_max = initial.raw() * (0.8 * self.binding_confidence_decay).powi(hops as i32);
        current.raw() <= expected_max + f32::EPSILON
    }
}
```

### SMT Formal Verification
```rust
// In engram-core/src/query/verification.rs

impl SMTVerificationSuite {
    /// Verify dual memory confidence propagation axioms
    pub fn verify_dual_memory_axioms(&self) -> Result<Vec<VerificationProof>, VerificationError> {
        let mut proofs = Vec::new();

        // Axiom: Monotonicity of propagation
        proofs.push(self.verify_propagation_monotonicity()?);

        // Axiom: Blend bonus bounds
        proofs.push(self.verify_blend_bonus_bounds()?);

        // Axiom: Cycle convergence
        proofs.push(self.verify_cycle_convergence()?);

        Ok(proofs)
    }

    fn verify_propagation_monotonicity(&self) -> Result<VerificationProof, VerificationError> {
        let solver = Solver::new(&self.context);

        let source = Real::new_const(&self.context, "source_conf");
        let binding = Real::new_const(&self.context, "binding_strength");
        let decay = Real::from_real(&self.context, 95, 100); // 0.95

        let zero = Real::from_real(&self.context, 0, 1);
        let one = Real::from_real(&self.context, 1, 1);

        // Constraints: valid probability inputs
        solver.assert(&source.ge(&zero));
        solver.assert(&source.le(&one));
        solver.assert(&binding.ge(&zero));
        solver.assert(&binding.le(&one));

        // Propagation formula
        let factors = [&binding, &decay];
        let propagated = Real::mul(&self.context, &[&source, &Real::mul(&self.context, &factors)]);

        // Try to find violation: propagated > source
        solver.assert(&propagated.gt(&source));

        let result = solver.check();
        Ok(VerificationProof {
            property_name: "propagation_monotonicity".to_string(),
            verified: matches!(result, SatResult::Unsat), // UNSAT = property holds
            sat_result: format!("{:?}", result),
            timestamp: std::time::SystemTime::now(),
        })
    }

    fn verify_blend_bonus_bounds(&self) -> Result<VerificationProof, VerificationError> {
        // Verify: blend with bonus ≤ max(episodic, semantic) * 1.1
        let solver = Solver::new(&self.context);

        let episodic = Real::new_const(&self.context, "episodic");
        let semantic = Real::new_const(&self.context, "semantic");
        let ep_weight = Real::from_real(&self.context, 7, 10); // 0.7
        let sem_weight = Real::from_real(&self.context, 3, 10); // 0.3

        let zero = Real::from_real(&self.context, 0, 1);
        let one = Real::from_real(&self.context, 1, 1);
        let bonus = Real::from_real(&self.context, 11, 10); // 1.1

        // Valid inputs above threshold
        solver.assert(&episodic.gt(&Real::from_real(&self.context, 1, 2))); // > 0.5
        solver.assert(&semantic.gt(&Real::from_real(&self.context, 1, 2))); // > 0.5
        solver.assert(&episodic.le(&one));
        solver.assert(&semantic.le(&one));

        // Weighted average
        let weighted_sum = Real::add(&self.context, &[
            &Real::mul(&self.context, &[&episodic, &ep_weight]),
            &Real::mul(&self.context, &[&semantic, &sem_weight])
        ]);
        let total_weight = Real::add(&self.context, &[&ep_weight, &sem_weight]);
        let base = Real::div(&self.context, &weighted_sum, &total_weight);

        // With bonus
        let with_bonus = Real::mul(&self.context, &[&base, &bonus]);

        // Maximum of inputs
        let max_input = Real::ite(
            &episodic.ge(&semantic),
            &episodic,
            &semantic
        );

        // Try to violate: with_bonus > max_input * bonus
        let max_allowed = Real::mul(&self.context, &[&max_input, &bonus]);
        solver.assert(&with_bonus.gt(&max_allowed));

        let result = solver.check();
        Ok(VerificationProof {
            property_name: "blend_bonus_bounds".to_string(),
            verified: matches!(result, SatResult::Unsat),
            sat_result: format!("{:?}", result),
            timestamp: std::time::SystemTime::now(),
        })
    }

    fn verify_cycle_convergence(&self) -> Result<VerificationProof, VerificationError> {
        // Verify: decay < 1.0 ensures convergence
        let solver = Solver::new(&self.context);

        let decay = Real::from_real(&self.context, 95, 100); // 0.95
        let binding = Real::from_real(&self.context, 8, 10); // 0.8
        let combined = Real::mul(&self.context, &[&decay, &binding]);

        let one = Real::from_real(&self.context, 1, 1);

        // Try to violate: combined >= 1.0 (would not converge)
        solver.assert(&combined.ge(&one));

        let result = solver.check();
        Ok(VerificationProof {
            property_name: "cycle_convergence".to_string(),
            verified: matches!(result, SatResult::Unsat),
            sat_result: format!("{:?}", result),
            timestamp: std::time::SystemTime::now(),
        })
    }
}
```

### Property-Based Testing Strategy
```rust
// In engram-core/tests/confidence_propagation_properties.rs

use engram_core::confidence::dual_memory::DualMemoryConfidence;
use engram_core::Confidence;
use proptest::prelude::*;

proptest! {
    /// Property: Confidence never increases through binding propagation
    #[test]
    fn propagation_is_monotonic(
        source in 0.0f32..1.0,
        binding in 0.0f32..1.0
    ) {
        let conf_sys = DualMemoryConfidence::default();
        let source_conf = Confidence::exact(source);
        let propagated = conf_sys.propagate_through_binding(source_conf, binding);

        prop_assert!(propagated.raw() <= source_conf.raw() + f32::EPSILON,
            "Propagation violated monotonicity: {} -> {}", source, propagated.raw());
    }

    /// Property: All confidence values stay in [0, 1]
    #[test]
    fn propagation_preserves_bounds(
        source in 0.0f32..1.0,
        binding in 0.0f32..1.0
    ) {
        let conf_sys = DualMemoryConfidence::default();
        let source_conf = Confidence::exact(source);
        let propagated = conf_sys.propagate_through_binding(source_conf, binding);

        prop_assert!(propagated.raw() >= 0.0);
        prop_assert!(propagated.raw() <= 1.0);
    }

    /// Property: Blend confidence is bounded by max input + bonus
    #[test]
    fn blend_confidence_bounded(
        episodic in 0.0f32..1.0,
        semantic in 0.0f32..1.0
    ) {
        let conf_sys = DualMemoryConfidence::default();
        let ep_conf = Confidence::exact(episodic);
        let sem_conf = Confidence::exact(semantic);

        let blended = conf_sys.blend_confidence(ep_conf, sem_conf, 0.7, 0.3);
        let max_input = episodic.max(semantic);

        // With bonus (1.1x), should not exceed max_input * 1.1
        prop_assert!(blended.raw() <= max_input * 1.1 + f32::EPSILON,
            "Blend exceeded bounds: episodic={}, semantic={}, blended={}",
            episodic, semantic, blended.raw());
    }

    /// Property: Blend bonus only applies when both > 0.5
    #[test]
    fn blend_bonus_threshold(
        episodic in 0.0f32..0.5,
        semantic in 0.0f32..1.0
    ) {
        let conf_sys = DualMemoryConfidence::default();
        let ep_conf = Confidence::exact(episodic);
        let sem_conf = Confidence::exact(semantic);

        let blended = conf_sys.blend_confidence(ep_conf, sem_conf, 0.7, 0.3);
        let weighted_avg = (episodic * 0.7 + semantic * 0.3) / 1.0;

        // No bonus since episodic < 0.5
        prop_assert!((blended.raw() - weighted_avg).abs() < f32::EPSILON,
            "Bonus applied incorrectly when episodic < 0.5");
    }

    /// Property: Multi-hop propagation decays exponentially
    #[test]
    fn multi_hop_exponential_decay(
        initial in 0.5f32..1.0,
        hops in 1u32..10
    ) {
        let conf_sys = DualMemoryConfidence::default();
        let binding_strength = 0.8;

        let mut current = Confidence::exact(initial);
        for _ in 0..hops {
            current = conf_sys.propagate_through_binding(current, binding_strength);
        }

        // Expected: initial * (binding * decay)^hops
        let expected = initial * (binding_strength * 0.95_f32).powi(hops as i32);

        prop_assert!((current.raw() - expected).abs() < 0.01,
            "Multi-hop decay incorrect: expected={}, got={}", expected, current.raw());
    }

    /// Property: Cycle propagation converges to zero
    #[test]
    fn cycle_convergence(
        initial in 0.5f32..1.0
    ) {
        let conf_sys = DualMemoryConfidence::default();
        let binding_strength = 0.8;

        let mut current = Confidence::exact(initial);
        for _ in 0..100 {
            current = conf_sys.propagate_through_binding(current, binding_strength);
        }

        // After 100 hops, should be negligible
        prop_assert!(current.raw() < 0.001,
            "Cycle did not converge: after 100 hops, conf={}", current.raw());
    }
}
```

### Differential Testing Strategy
Test confidence calculation consistency across memory types:

```rust
#[cfg(test)]
mod differential_tests {
    use super::*;

    #[test]
    fn episodic_only_vs_semantic_only_vs_blended() {
        let conf_sys = DualMemoryConfidence::default();

        // Same source confidence
        let source = Confidence::exact(0.8);

        // Episodic-only: direct propagation
        let episodic_only = conf_sys.propagate_through_binding(source, 0.9);

        // Semantic-only: through concept formation then back
        let concept = conf_sys.concept_confidence(&[episode_with_conf(0.8)], 0.95);
        let semantic_only = conf_sys.propagate_through_binding(concept, 0.9);

        // Blended: combine both paths
        let blended = conf_sys.blend_confidence(episodic_only, semantic_only, 0.7, 0.3);

        // Semantic path should have lower confidence (additional formation penalty)
        assert!(semantic_only.raw() < episodic_only.raw());

        // Blended should be between (without bonus) or slightly above (with bonus)
        assert!(blended.raw() >= episodic_only.raw().min(semantic_only.raw()));
        assert!(blended.raw() <= episodic_only.raw() * 1.1);
    }

    #[test]
    fn convergent_evidence_bonus_applied() {
        let conf_sys = DualMemoryConfidence::default();

        // Both sources high confidence (convergent evidence)
        let episodic = Confidence::exact(0.8);
        let semantic = Confidence::exact(0.75);

        let blended = conf_sys.blend_confidence(episodic, semantic, 0.7, 0.3);
        let base = (0.8 * 0.7 + 0.75 * 0.3) / 1.0;

        // Should apply 1.1x bonus
        assert!((blended.raw() - base * 1.1).abs() < f32::EPSILON);
    }

    #[test]
    fn no_bonus_when_evidence_weak() {
        let conf_sys = DualMemoryConfidence::default();

        // One source below threshold
        let episodic = Confidence::exact(0.4);
        let semantic = Confidence::exact(0.8);

        let blended = conf_sys.blend_confidence(episodic, semantic, 0.7, 0.3);
        let base = (0.4 * 0.7 + 0.8 * 0.3) / 1.0;

        // No bonus since episodic < 0.5
        assert!((blended.raw() - base).abs() < f32::EPSILON);
    }
}
```

### Statistical Validation of Confidence-Accuracy Correlation

Add empirical validation that confidence scores correlate with actual retrieval accuracy:

```rust
#[cfg(test)]
mod calibration_validation {
    use engram_core::query::confidence_calibration::CalibrationTracker;

    #[test]
    fn confidence_correlates_with_accuracy() {
        let mut tracker = CalibrationTracker::new(10);
        let conf_sys = DualMemoryConfidence::default();

        // Simulate retrievals with known accuracy
        for _ in 0..1000 {
            let (episode, was_correct, expected_conf) = generate_retrieval_sample();

            let predicted = conf_sys.calculate_confidence(&episode);
            tracker.record_sample(predicted, was_correct);
        }

        let metrics = tracker.compute_metrics();

        // Confidence should correlate with accuracy (Spearman > 0.7)
        assert!(metrics.confidence_accuracy_correlation.unwrap() > 0.7,
            "Confidence does not correlate with accuracy: rho={:?}",
            metrics.confidence_accuracy_correlation);

        // Expected Calibration Error should be low (<0.1)
        assert!(metrics.expected_calibration_error < 0.1,
            "Poor calibration: ECE={}", metrics.expected_calibration_error);
    }
}
```

## Integration with Spreading

```rust
impl SpreadingWithConfidence {
    pub fn spread_with_confidence(
        &self,
        activations: Vec<(NodeId, f32, Confidence)>,
    ) -> Vec<(NodeId, f32, Confidence)> {
        let mut results = Vec::new();
        let conf_rules = DualMemoryConfidence::default();

        for (node_id, activation, confidence) in activations {
            let spread_targets = self.get_spread_targets(&node_id);

            for (target_id, spread_strength) in spread_targets {
                // Propagate both activation and confidence
                let spread_activation = activation * spread_strength;
                let spread_confidence = conf_rules
                    .propagate_through_binding(confidence, spread_strength);

                results.push((target_id, spread_activation, spread_confidence));
            }
        }

        results
    }
}
```

## Implementation Notes
- Use SMT verification for mathematical correctness proofs
- Property-based tests generate thousands of test cases automatically
- Differential testing ensures consistency across memory types
- Statistical validation uses existing CalibrationTracker infrastructure
- Type system enforces confidence bounds at compile time
- Runtime assertions catch any edge cases missed by static analysis

## Testing Approach
1. **Unit tests**: Basic functionality of each method
2. **Property tests**: Verify invariants hold for all inputs (proptest)
3. **SMT verification**: Formal mathematical proofs (Z3)
4. **Differential tests**: Compare episodic-only, semantic-only, blended paths
5. **Statistical validation**: Confidence-accuracy correlation on simulated data
6. **Integration tests**: End-to-end propagation through graph
7. **Benchmarks**: Performance overhead of confidence tracking

## Acceptance Criteria
- [ ] All confidence values stay in [0, 1] range (type system + property tests)
- [ ] Propagation preserves uncertainty monotonicity (SMT proof + property tests)
- [ ] Blended confidence correctly reflects convergent evidence (differential tests)
- [ ] No confidence inflation through cycles (SMT proof + multi-hop tests)
- [ ] Blend bonus mathematically justified (SMT bounds proof)
- [ ] Confidence-accuracy correlation > 0.7 (statistical validation)
- [ ] Expected Calibration Error < 0.1 (empirical validation)
- [ ] Performance overhead < 5% (benchmarks)
- [ ] All SMT axioms verified (formal proofs)
- [ ] Zero clippy warnings
- [ ] All property tests pass (1000+ cases each)

## Dependencies
- Task 009 (Blended Recall) - provides blending context
- Existing confidence system (lib.rs, activation/, query/)
- SMT verification infrastructure (query/verification.rs)
- Property testing framework (proptest dependency)

## Estimated Time
3 days (original) + 1 day (formal verification) = 4 days total

## References
- McClelland et al. (1995): Complementary Learning Systems theory
- Existing confidence aggregation: engram-core/src/activation/confidence_aggregation.rs
- Calibration framework: engram-core/src/query/confidence_calibration.rs
- SMT verification: engram-core/src/query/verification.rs
- Property tests: engram-core/tests/confidence_property_tests.rs
