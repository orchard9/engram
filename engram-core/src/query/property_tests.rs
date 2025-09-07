//! Property-based testing for probabilistic operations
//!
//! This module implements comprehensive property-based testing using QuickCheck/Proptest
//! to verify that all probabilistic operations satisfy fundamental mathematical properties
//! and prevent cognitive biases through systematic validation.

use crate::{Confidence, Activation};
use super::{ConfidenceInterval, evidence::EvidenceAggregator, ProbabilisticQueryResult};
use proptest::prelude::*;
use std::time::Duration;

/// Generate valid confidence values [0.0, 1.0]
fn confidence_strategy() -> impl Strategy<Value = Confidence> {
    (0.0f32..=1.0f32).prop_map(Confidence::exact)
}

/// Generate valid activation values [0.0, 1.0]
fn activation_strategy() -> impl Strategy<Value = Activation> {
    (0.0f32..=1.0f32).prop_map(Activation::new)
}

/// Generate confidence intervals with proper ordering
fn confidence_interval_strategy() -> impl Strategy<Value = ConfidenceInterval> {
    (0.0f32..=1.0f32, 0.0f32..=0.2f32)
        .prop_map(|(center, half_width)| {
            ConfidenceInterval::from_confidence_with_uncertainty(
                Confidence::exact(center),
                half_width,
            )
        })
}

proptest! {
    /// Test fundamental probability axioms
    #[test]
    fn test_probability_axioms_non_negativity(
        p in confidence_strategy()
    ) {
        // Axiom 1: P(A) ≥ 0 for any event A
        prop_assert!(p.raw() >= 0.0);
    }
    
    #[test]
    fn test_probability_axioms_normalization(
        p in confidence_strategy()
    ) {
        // Axiom 2: P(Ω) = 1 for the sample space Ω
        // We verify that probabilities are bounded by 1.0
        prop_assert!(p.raw() <= 1.0);
    }
    
    #[test]
    fn test_probability_axioms_additivity(
        p1 in confidence_strategy(),
        p2 in confidence_strategy()
    ) {
        // Axiom 3: P(A ∪ B) = P(A) + P(B) - P(A ∩ B) 
        // Verify OR operation matches this formula
        let union = p1.or(p2);
        let intersection = p1.and(p2);
        let expected = Confidence::exact(
            (p1.raw() + p2.raw() - intersection.raw()).clamp(0.0, 1.0)
        );
        
        // Allow small numerical errors
        prop_assert!((union.raw() - expected.raw()).abs() < 1e-6);
    }
    
    /// Test conjunction fallacy prevention
    #[test]
    fn test_conjunction_fallacy_prevention(
        p1 in confidence_strategy(),
        p2 in confidence_strategy()
    ) {
        let conjunction = p1.and(p2);
        
        // P(A ∧ B) ≤ P(A) and P(A ∧ B) ≤ P(B)
        prop_assert!(conjunction.raw() <= p1.raw() + 1e-6); // Small tolerance for floating point
        prop_assert!(conjunction.raw() <= p2.raw() + 1e-6);
    }
    
    /// Test that OR operation doesn't exceed unity
    #[test]
    fn test_disjunction_bound(
        p1 in confidence_strategy(),
        p2 in confidence_strategy()
    ) {
        let disjunction = p1.or(p2);
        
        // P(A ∨ B) ≤ 1.0
        prop_assert!(disjunction.raw() <= 1.0);
        
        // P(A ∨ B) ≥ max(P(A), P(B))
        prop_assert!(disjunction.raw() >= p1.raw().max(p2.raw()) - 1e-6);
    }
    
    /// Test De Morgan's laws for logical operations
    #[test]
    fn test_de_morgans_laws(
        p1 in confidence_strategy(),
        p2 in confidence_strategy()
    ) {
        // ¬(A ∧ B) = ¬A ∨ ¬B
        let left_side = p1.and(p2).not();
        let right_side = p1.not().or(p2.not());
        
        prop_assert!((left_side.raw() - right_side.raw()).abs() < 1e-5);
        
        // ¬(A ∨ B) = ¬A ∧ ¬B  
        let left_side2 = p1.or(p2).not();
        let right_side2 = p1.not().and(p2.not());
        
        prop_assert!((left_side2.raw() - right_side2.raw()).abs() < 1e-5);
    }
    
    /// Test confidence interval properties
    #[test]
    fn test_confidence_interval_ordering(
        interval in confidence_interval_strategy()
    ) {
        // Lower ≤ Point ≤ Upper
        prop_assert!(interval.lower.raw() <= interval.point.raw());
        prop_assert!(interval.point.raw() <= interval.upper.raw());
        
        // Width should be non-negative
        prop_assert!(interval.width >= 0.0);
        
        // Width should match bounds
        let computed_width = interval.upper.raw() - interval.lower.raw();
        prop_assert!((interval.width - computed_width).abs() < 1e-6);
    }
    
    /// Test interval arithmetic properties
    #[test]
    fn test_interval_and_monotonicity(
        a in confidence_interval_strategy(),
        b in confidence_interval_strategy()
    ) {
        let result = a.and(&b);
        
        // AND should preserve interval structure
        prop_assert!(result.lower.raw() <= result.point.raw());
        prop_assert!(result.point.raw() <= result.upper.raw());
        
        // AND result should be no greater than either input's lower bound
        // (conservative estimate)
        prop_assert!(result.upper.raw() <= a.lower.raw().max(b.lower.raw()) + 1e-5);
    }
    
    /// Test interval OR properties
    #[test]
    fn test_interval_or_monotonicity(
        a in confidence_interval_strategy(),
        b in confidence_interval_strategy()
    ) {
        let result = a.or(&b);
        
        // OR should preserve interval structure
        prop_assert!(result.lower.raw() <= result.point.raw());
        prop_assert!(result.point.raw() <= result.upper.raw());
        
        // OR result should be no less than max of inputs
        let min_expected = a.upper.raw().max(b.upper.raw());
        prop_assert!(result.lower.raw() >= min_expected - 1e-5);
    }
    
    /// Test evidence combination properties
    #[test]
    fn test_evidence_combination_commutativity(
        _strength1 in confidence_strategy(),
        _strength2 in confidence_strategy(),
        activation1 in activation_strategy(),
        activation2 in activation_strategy()
    ) {
        let mut aggregator1 = EvidenceAggregator::new();
        let mut aggregator2 = EvidenceAggregator::new();
        
        // Create evidence in different orders
        let evidence1 = EvidenceAggregator::evidence_from_activation(
            "episode1".to_string(), activation1, 1
        );
        let evidence2 = EvidenceAggregator::evidence_from_activation(
            "episode2".to_string(), activation2, 1
        );
        
        let id1_a = aggregator1.add_evidence(evidence1.clone());
        let id2_a = aggregator1.add_evidence(evidence2.clone());
        
        let id2_b = aggregator2.add_evidence(evidence2);
        let id1_b = aggregator2.add_evidence(evidence1);
        
        let result1 = aggregator1.combine_evidence(vec![id1_a, id2_a]);
        let result2 = aggregator2.combine_evidence(vec![id2_b, id1_b]);
        
        // Both should succeed or both should fail
        match (result1, result2) {
            (Ok(r1), Ok(r2)) => {
                // Results should be approximately equal (commutative property)
                prop_assert!((r1.confidence.raw() - r2.confidence.raw()).abs() < 1e-5);
            }
            (Err(_), Err(_)) => {
                // Both failing is acceptable
            }
            _ => {
                prop_assert!(false, "Evidence combination should be commutative");
            }
        }
    }
    
    /// Test that evidence combination never violates probability bounds
    #[test]
    fn test_evidence_combination_bounds(
        activations in prop::collection::vec(activation_strategy(), 1..10)
    ) {
        let mut aggregator = EvidenceAggregator::new();
        let mut evidence_ids = Vec::new();
        
        for (i, activation) in activations.iter().enumerate() {
            let evidence = EvidenceAggregator::evidence_from_activation(
                format!("episode_{}", i),
                *activation,
                1
            );
            let id = aggregator.add_evidence(evidence);
            evidence_ids.push(id);
        }
        
        if let Ok(result) = aggregator.combine_evidence(evidence_ids) {
            // Combined confidence must be valid probability
            prop_assert!(result.confidence.raw() >= 0.0);
            prop_assert!(result.confidence.raw() <= 1.0);
            
            // Bounds must be properly ordered
            prop_assert!(result.lower_bound.raw() <= result.confidence.raw());
            prop_assert!(result.confidence.raw() <= result.upper_bound.raw());
        }
    }
    
    /// Test temporal decay properties
    #[test]
    fn test_temporal_decay_monotonicity(
        initial_confidence in confidence_strategy(),
        decay_rate in 0.0f32..0.1f32,
        time_seconds in 0u64..86400u64  // Up to 1 day
    ) {
        let time_elapsed = Duration::from_secs(time_seconds);
        
        let evidence = EvidenceAggregator::evidence_from_decay(
            initial_confidence,
            time_elapsed,
            decay_rate
        );
        
        // Decayed confidence should not exceed initial
        prop_assert!(evidence.strength.raw() <= initial_confidence.raw() + 1e-6);
        
        // Should be non-negative
        prop_assert!(evidence.strength.raw() >= 0.0);
        
        // Zero time should preserve confidence (approximately)
        if time_seconds == 0 {
            prop_assert!((evidence.strength.raw() - initial_confidence.raw()).abs() < 1e-5);
        }
    }
    
    /// Test activation-based evidence degradation
    #[test]
    fn test_activation_path_degradation(
        activation in activation_strategy(),
        path_length in 0u16..20u16
    ) {
        let evidence = EvidenceAggregator::evidence_from_activation(
            "test_episode".to_string(),
            activation,
            path_length
        );
        
        // Path degradation should reduce confidence
        if path_length > 0 {
            let expected_max = activation.value() / (1.0 + path_length as f32 * 0.1);
            prop_assert!(evidence.strength.raw() <= expected_max + 1e-6);
        } else {
            // Zero path length should preserve activation level
            prop_assert!((evidence.strength.raw() - activation.value()).abs() < 1e-6);
        }
        
        // Should never be negative
        prop_assert!(evidence.strength.raw() >= 0.0);
    }
    
    /// Test that overconfidence correction reduces extreme values
    #[test]
    fn test_overconfidence_correction(
        confidence in confidence_strategy()
    ) {
        let corrected = confidence.calibrate_overconfidence();
        
        // Should still be valid probability
        prop_assert!(corrected.raw() >= 0.0);
        prop_assert!(corrected.raw() <= 1.0);
        
        // High confidence values should be reduced
        if confidence.raw() > 0.9 {
            prop_assert!(corrected.raw() <= confidence.raw() + 1e-6);
        }
    }
    
    /// Test weighted combination properties
    #[test]
    fn test_weighted_combination_properties(
        p1 in confidence_strategy(),
        p2 in confidence_strategy(),
        w1 in 0.0f32..10.0f32,
        w2 in 0.0f32..10.0f32
    ) {
        // Skip degenerate case
        prop_assume!(w1 + w2 > 0.0);
        
        let result = p1.combine_weighted(p2, w1, w2);
        
        // Result should be valid probability
        prop_assert!(result.raw() >= 0.0);
        prop_assert!(result.raw() <= 1.0);
        
        // Result should be between the two input values
        let min_input = p1.raw().min(p2.raw());
        let max_input = p1.raw().max(p2.raw());
        prop_assert!(result.raw() >= min_input - 1e-6);
        prop_assert!(result.raw() <= max_input + 1e-6);
        
        // Equal weights should give average
        if (w1 - w2).abs() < 1e-6 {
            let expected_avg = (p1.raw() + p2.raw()) / 2.0;
            prop_assert!((result.raw() - expected_avg).abs() < 1e-5);
        }
    }
}

/// Integration tests with real memory store data patterns
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::{Episode, MemoryStore, CueType, Cue};
    use chrono::Utc;
    
    #[test]
    fn test_probabilistic_query_result_from_episodes() {
        // Create test episodes with different confidence levels
        let episodes = vec![
            (create_test_episode("high_conf"), Confidence::HIGH),
            (create_test_episode("medium_conf"), Confidence::MEDIUM),
            (create_test_episode("low_conf"), Confidence::LOW),
        ];
        
        let result = ProbabilisticQueryResult::from_episodes(episodes.clone());
        
        // Should have correct number of episodes
        assert_eq!(result.episodes.len(), 3);
        
        // Confidence interval should reflect the diversity
        assert!(result.confidence_interval.width > 0.0);
        
        // Should be considered successful due to multiple results
        assert!(result.is_successful());
    }
    
    #[test]
    fn test_empty_query_result() {
        let result = ProbabilisticQueryResult::from_episodes(Vec::new());
        
        assert!(result.is_empty());
        assert!(!result.is_successful());
        assert_eq!(result.confidence_interval.point, Confidence::NONE);
        assert_eq!(result.confidence_interval.width, 0.0);
    }
    
    #[test]
    fn test_single_high_confidence_result() {
        let episodes = vec![
            (create_test_episode("single"), Confidence::HIGH),
        ];
        
        let result = ProbabilisticQueryResult::from_episodes(episodes);
        
        assert!(!result.is_empty());
        assert!(result.is_successful());
        
        // Single result should have zero width (no uncertainty from diversity)
        assert_eq!(result.confidence_interval.width, 0.0);
        assert_eq!(result.confidence_interval.point, Confidence::HIGH);
    }
    
    fn create_test_episode(id: &str) -> Episode {
        Episode {
            id: id.to_string(),
            when: Utc::now(),
            where_location: Some("test_location".to_string()),
            who: Some(vec!["test_person".to_string()]),
            what: format!("test content for {}", id),
            embedding: [0.5f32; 768],
            encoding_confidence: Confidence::HIGH,
            vividness_confidence: Confidence::MEDIUM,
            reliability_confidence: Confidence::HIGH,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 0.1,
        }
    }
}

/// Benchmarks for performance validation
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn bench_evidence_combination_performance() {
        let mut aggregator = EvidenceAggregator::new();
        
        // Add 100 pieces of evidence
        let mut evidence_ids = Vec::new();
        for i in 0..100 {
            let evidence = EvidenceAggregator::evidence_from_activation(
                format!("episode_{}", i),
                Activation::new(0.5 + (i as f32) * 0.005),
                i % 10,
            );
            let id = aggregator.add_evidence(evidence);
            evidence_ids.push(id);
        }
        
        let start = Instant::now();
        let result = aggregator.combine_evidence(evidence_ids);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        
        // Should complete in reasonable time (less than 10ms for 100 evidence items)
        assert!(duration.as_millis() < 10, 
                "Evidence combination took {}ms, expected <10ms", 
                duration.as_millis());
    }
    
    #[test]
    fn bench_confidence_interval_operations() {
        let interval1 = ConfidenceInterval::from_confidence_with_uncertainty(
            Confidence::HIGH, 
            0.1
        );
        let interval2 = ConfidenceInterval::from_confidence_with_uncertainty(
            Confidence::MEDIUM,
            0.15
        );
        
        let start = Instant::now();
        
        // Perform many operations
        for _ in 0..1000 {
            let _and_result = interval1.and(&interval2);
            let _or_result = interval1.or(&interval2);
            let _not_result = interval1.not();
        }
        
        let duration = start.elapsed();
        
        // Should complete quickly (less than 1ms for 3000 operations)
        assert!(duration.as_millis() < 1,
                "Interval operations took {}ms, expected <1ms",
                duration.as_millis());
    }
}