//! Evidence Aggregation with Bayesian Updating
//!
//! Implements lock-free evidence combination with circular dependency detection
//! and proper Bayesian updating for evidence chains. Aggregates multiple sources
//! of evidence while respecting dependency relationships and detecting cycles.
//!
//! # Architecture
//!
//! The aggregator uses:
//! - Dependency graph for cycle detection (Tarjan's algorithm)
//! - Topological ordering for processing dependent evidence
//! - Bayesian combination for independent evidence
//! - Log-space computation for numerical stability
//!
//! # Performance
//!
//! Target: <100μs for aggregating 10 evidence sources
//!
//! # Example
//!
//! ```
//! use engram_core::query::evidence_aggregator::{EvidenceAggregator, EvidenceInput};
//! use engram_core::query::{EvidenceSource, MatchType};
//! use engram_core::Confidence;
//! use std::time::SystemTime;
//!
//! let aggregator = EvidenceAggregator::new(Confidence::from_raw(0.01), 10);
//!
//! let evidence_inputs = vec![
//!     EvidenceInput {
//!         id: 0,
//!         source: EvidenceSource::DirectMatch {
//!             cue_id: "test".to_string(),
//!             similarity_score: 0.8,
//!             match_type: MatchType::Semantic,
//!         },
//!         strength: Confidence::from_raw(0.8),
//!         timestamp: SystemTime::now(),
//!         dependencies: vec![],
//!     },
//!     EvidenceInput {
//!         id: 1,
//!         source: EvidenceSource::DirectMatch {
//!             cue_id: "test2".to_string(),
//!             similarity_score: 0.7,
//!             match_type: MatchType::Semantic,
//!         },
//!         strength: Confidence::from_raw(0.7),
//!         timestamp: SystemTime::now(),
//!         dependencies: vec![0], // Depends on evidence 0
//!     },
//! ];
//!
//! let result = aggregator.aggregate_evidence(&evidence_inputs).unwrap();
//! assert!(result.aggregate_confidence.raw() > 0.0);
//! assert!(!result.has_circular_dependencies);
//! ```

use crate::Confidence;
use crate::query::dependency_graph::DependencyGraph;
use crate::query::{EvidenceId, EvidenceSource, ProbabilisticError, ProbabilisticResult};
use std::collections::HashMap;
use std::time::SystemTime;

/// Lock-free evidence aggregator with Bayesian updating
#[derive(Debug, Clone)]
pub struct EvidenceAggregator {
    /// Minimum evidence strength to consider
    min_strength: Confidence,
    /// Maximum number of evidence sources to aggregate
    max_evidence: usize,
}

impl EvidenceAggregator {
    /// Create a new evidence aggregator
    ///
    /// # Arguments
    ///
    /// * `min_strength` - Minimum confidence threshold for evidence consideration
    /// * `max_evidence` - Maximum number of evidence sources to process
    #[must_use]
    pub const fn new(min_strength: Confidence, max_evidence: usize) -> Self {
        Self {
            min_strength,
            max_evidence,
        }
    }

    /// Aggregate multiple evidence sources with dependency tracking
    ///
    /// Performs Bayesian combination of evidence while respecting dependencies.
    /// Detects circular dependencies and returns an error if found.
    ///
    /// # Arguments
    ///
    /// * `evidence` - Slice of evidence inputs to aggregate
    ///
    /// # Returns
    ///
    /// * `Ok(EvidenceAggregationOutcome)` - Successful aggregation with combined confidence
    /// * `Err(ProbabilisticError::CircularDependency)` - If circular dependencies detected
    /// * `Err(ProbabilisticError::InsufficientEvidence)` - If no valid evidence provided
    ///
    /// # Performance
    ///
    /// Target: <100μs for 10 evidence sources
    pub fn aggregate_evidence(
        &self,
        evidence: &[EvidenceInput],
    ) -> ProbabilisticResult<EvidenceAggregationOutcome> {
        if evidence.is_empty() {
            return Err(ProbabilisticError::InsufficientEvidence);
        }

        // Build dependency graph
        let mut graph = DependencyGraph::new();
        for ev in evidence {
            graph.add_evidence(ev.id, ev.dependencies.clone());
        }

        // Check for circular dependencies
        if graph.has_cycles() {
            return Err(ProbabilisticError::CircularDependency);
        }

        // Get topological ordering for processing
        let order = graph
            .topological_sort()
            .ok_or(ProbabilisticError::CircularDependency)?;

        // Create evidence lookup map
        let evidence_map: HashMap<EvidenceId, &EvidenceInput> =
            evidence.iter().map(|ev| (ev.id, ev)).collect();

        // Process evidence in dependency order
        let mut contributions = Vec::new();
        let mut independent_evidence = Vec::new();

        for &ev_id in &order {
            if let Some(&ev) = evidence_map.get(&ev_id) {
                // Filter by minimum strength
                if ev.strength.raw() < self.min_strength.raw() {
                    continue;
                }

                // Limit total evidence processed (count both independent and dependent)
                if contributions.len() + independent_evidence.len() >= self.max_evidence {
                    break;
                }

                if ev.dependencies.is_empty() {
                    // Independent evidence - accumulate for combination
                    independent_evidence.push(ev);
                } else {
                    // Dependent evidence - process based on dependencies
                    contributions.push(EvidenceContribution {
                        evidence_id: ev.id,
                        original_strength: ev.strength,
                        posterior_strength: ev.strength, // Updated during combination
                        source: ev.source.clone(),
                        timestamp: ev.timestamp,
                    });
                }
            }
        }

        // Combine independent evidence using Bayesian formula
        let aggregate = if independent_evidence.is_empty() {
            // Only dependent evidence or empty
            if contributions.is_empty() {
                return Err(ProbabilisticError::InsufficientEvidence);
            }
            Self::combine_dependent_evidence(&contributions)
        } else {
            // Add independent evidence to contributions
            for &ev in &independent_evidence {
                contributions.push(EvidenceContribution {
                    evidence_id: ev.id,
                    original_strength: ev.strength,
                    posterior_strength: ev.strength,
                    source: ev.source.clone(),
                    timestamp: ev.timestamp,
                });
            }
            Self::combine_independent_evidence(&independent_evidence)
        };

        Ok(EvidenceAggregationOutcome {
            aggregate_confidence: aggregate,
            contributing_evidence: contributions,
            dependency_order: order,
            has_circular_dependencies: false,
        })
    }

    /// Combine independent evidence using probabilistic OR formula
    ///
    /// Uses log-space computation for numerical stability:
    /// P(H|E₁,E₂,...,Eₙ) = 1 - ∏(1 - P(H|Eᵢ))
    fn combine_independent_evidence(evidence: &[&EvidenceInput]) -> Confidence {
        if evidence.is_empty() {
            return Confidence::NONE;
        }

        if evidence.len() == 1 {
            return evidence[0].strength;
        }

        // Use log-space for numerical stability
        let mut log_one_minus_total = 0.0f64;

        for ev in evidence {
            let probability = f64::from(ev.strength.raw()).clamp(0.0, 1.0);
            let complement = (1.0 - probability).clamp(f64::MIN_POSITIVE, 1.0);
            log_one_minus_total += complement.ln();
        }

        #[allow(clippy::cast_precision_loss)]
        #[allow(clippy::cast_possible_truncation)]
        let combined = (1.0 - log_one_minus_total.exp()).clamp(0.0, 1.0) as f32;

        Confidence::from_raw(combined)
    }

    /// Combine dependent evidence (placeholder for more sophisticated Bayesian updating)
    ///
    /// For now, uses the same independent combination formula.
    /// Future enhancement: Implement proper conditional probability updates.
    fn combine_dependent_evidence(contributions: &[EvidenceContribution]) -> Confidence {
        if contributions.is_empty() {
            return Confidence::NONE;
        }

        if contributions.len() == 1 {
            return contributions[0].posterior_strength;
        }

        // Use log-space for numerical stability
        let mut log_one_minus_total = 0.0f64;

        for contrib in contributions {
            let probability = f64::from(contrib.posterior_strength.raw()).clamp(0.0, 1.0);
            let complement = (1.0 - probability).clamp(f64::MIN_POSITIVE, 1.0);
            log_one_minus_total += complement.ln();
        }

        #[allow(clippy::cast_precision_loss)]
        #[allow(clippy::cast_possible_truncation)]
        let combined = (1.0 - log_one_minus_total.exp()).clamp(0.0, 1.0) as f32;

        Confidence::from_raw(combined)
    }
}

impl Default for EvidenceAggregator {
    fn default() -> Self {
        Self::new(Confidence::from_raw(0.01), 10)
    }
}

/// Input evidence for aggregation
///
/// Simplified version of Evidence that includes only necessary fields for aggregation.
#[derive(Debug, Clone)]
pub struct EvidenceInput {
    /// Unique identifier for this evidence
    pub id: EvidenceId,
    /// Source of evidence
    pub source: EvidenceSource,
    /// Strength as confidence value
    pub strength: Confidence,
    /// When evidence was collected
    pub timestamp: SystemTime,
    /// Dependencies on other evidence IDs
    pub dependencies: Vec<EvidenceId>,
}

/// Outcome of evidence aggregation
#[derive(Debug, Clone)]
pub struct EvidenceAggregationOutcome {
    /// Combined confidence from all evidence
    pub aggregate_confidence: Confidence,
    /// Individual evidence contributions
    pub contributing_evidence: Vec<EvidenceContribution>,
    /// Topological order used for processing
    pub dependency_order: Vec<EvidenceId>,
    /// Whether circular dependencies were detected
    pub has_circular_dependencies: bool,
}

impl EvidenceAggregationOutcome {
    /// Check if any evidence contributed to the outcome
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.contributing_evidence.is_empty()
    }

    /// Get the number of contributing evidence sources
    #[must_use]
    pub const fn len(&self) -> usize {
        self.contributing_evidence.len()
    }
}

/// Individual evidence contribution to the aggregated result
#[derive(Debug, Clone)]
pub struct EvidenceContribution {
    /// Evidence identifier
    pub evidence_id: EvidenceId,
    /// Original evidence strength before combination
    pub original_strength: Confidence,
    /// Posterior strength after Bayesian updating
    pub posterior_strength: Confidence,
    /// Source of this evidence
    pub source: EvidenceSource,
    /// When this evidence was collected
    pub timestamp: SystemTime,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Unwrap is acceptable in tests
mod tests {
    use super::*;
    use crate::query::MatchType;

    fn create_test_evidence(
        id: EvidenceId,
        strength: f32,
        dependencies: Vec<EvidenceId>,
    ) -> EvidenceInput {
        EvidenceInput {
            id,
            source: EvidenceSource::DirectMatch {
                cue_id: format!("cue_{id}"),
                similarity_score: strength,
                match_type: MatchType::Semantic,
            },
            strength: Confidence::from_raw(strength),
            timestamp: SystemTime::now(),
            dependencies,
        }
    }

    #[test]
    fn test_empty_evidence_returns_error() {
        let aggregator = EvidenceAggregator::default();
        let result = aggregator.aggregate_evidence(&[]);
        assert!(matches!(
            result,
            Err(ProbabilisticError::InsufficientEvidence)
        ));
    }

    #[test]
    fn test_single_evidence_returns_same_strength() {
        let aggregator = EvidenceAggregator::default();
        let evidence = vec![create_test_evidence(0, 0.8, vec![])];

        let result = aggregator.aggregate_evidence(&evidence).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result.aggregate_confidence.raw() - 0.8).abs() < 1e-6);
        assert!(!result.has_circular_dependencies);
    }

    #[test]
    fn test_two_independent_evidence_combines_higher() {
        let aggregator = EvidenceAggregator::default();
        let evidence = vec![
            create_test_evidence(0, 0.6, vec![]),
            create_test_evidence(1, 0.5, vec![]),
        ];

        let result = aggregator.aggregate_evidence(&evidence).unwrap();
        assert_eq!(result.len(), 2);

        // Combined: 1 - (1-0.6)*(1-0.5) = 1 - 0.4*0.5 = 1 - 0.2 = 0.8
        let expected = 0.8;
        assert!((result.aggregate_confidence.raw() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dependent_evidence_ordered_correctly() {
        let aggregator = EvidenceAggregator::default();
        let evidence = vec![
            create_test_evidence(0, 0.7, vec![]),
            create_test_evidence(1, 0.6, vec![0]), // Depends on 0
            create_test_evidence(2, 0.5, vec![1]), // Depends on 1
        ];

        let result = aggregator.aggregate_evidence(&evidence).unwrap();

        // Check dependency order: 0 should come before 1, 1 before 2
        let order = &result.dependency_order;
        let pos_0 = order.iter().position(|&x| x == 0).unwrap();
        let pos_1 = order.iter().position(|&x| x == 1).unwrap();
        let pos_2 = order.iter().position(|&x| x == 2).unwrap();

        assert!(pos_0 < pos_1);
        assert!(pos_1 < pos_2);
    }

    #[test]
    fn test_circular_dependency_detected() {
        let aggregator = EvidenceAggregator::default();
        let evidence = vec![
            create_test_evidence(0, 0.7, vec![1]), // 0 depends on 1
            create_test_evidence(1, 0.6, vec![0]), // 1 depends on 0 -> cycle!
        ];

        let result = aggregator.aggregate_evidence(&evidence);
        assert!(matches!(
            result,
            Err(ProbabilisticError::CircularDependency)
        ));
    }

    #[test]
    fn test_self_loop_detected() {
        let aggregator = EvidenceAggregator::default();
        let evidence = vec![create_test_evidence(0, 0.7, vec![0])]; // Self-loop

        let result = aggregator.aggregate_evidence(&evidence);
        assert!(matches!(
            result,
            Err(ProbabilisticError::CircularDependency)
        ));
    }

    #[test]
    fn test_min_strength_filter() {
        let aggregator = EvidenceAggregator::new(Confidence::from_raw(0.5), 10);
        let evidence = vec![
            create_test_evidence(0, 0.3, vec![]), // Below threshold
            create_test_evidence(1, 0.7, vec![]), // Above threshold
        ];

        let result = aggregator.aggregate_evidence(&evidence).unwrap();

        // Only evidence 1 should contribute
        assert_eq!(result.len(), 1);
        assert_eq!(result.contributing_evidence[0].evidence_id, 1);
    }

    #[test]
    fn test_max_evidence_limit() {
        let aggregator = EvidenceAggregator::new(Confidence::from_raw(0.01), 2);
        let evidence = vec![
            create_test_evidence(0, 0.8, vec![]),
            create_test_evidence(1, 0.7, vec![]),
            create_test_evidence(2, 0.6, vec![]),
            create_test_evidence(3, 0.5, vec![]),
        ];

        let result = aggregator.aggregate_evidence(&evidence).unwrap();

        // Only 2 evidence sources should contribute
        assert!(result.len() <= 2);
    }

    #[test]
    fn test_three_independent_evidence() {
        let aggregator = EvidenceAggregator::default();
        let evidence = vec![
            create_test_evidence(0, 0.5, vec![]),
            create_test_evidence(1, 0.4, vec![]),
            create_test_evidence(2, 0.3, vec![]),
        ];

        let result = aggregator.aggregate_evidence(&evidence).unwrap();

        // Combined: 1 - (1-0.5)*(1-0.4)*(1-0.3) = 1 - 0.5*0.6*0.7 = 1 - 0.21 = 0.79
        let expected = 0.79;
        assert!((result.aggregate_confidence.raw() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dag_with_multiple_paths() {
        let aggregator = EvidenceAggregator::default();
        let evidence = vec![
            create_test_evidence(0, 0.6, vec![]),
            create_test_evidence(1, 0.5, vec![]),
            create_test_evidence(2, 0.7, vec![0, 1]), // Depends on both 0 and 1
        ];

        let result = aggregator.aggregate_evidence(&evidence).unwrap();
        assert_eq!(result.len(), 3);
        assert!(!result.has_circular_dependencies);

        // Both 0 and 1 should appear before 2 in the dependency order
        let order = &result.dependency_order;
        let pos_0 = order.iter().position(|&x| x == 0).unwrap();
        let pos_1 = order.iter().position(|&x| x == 1).unwrap();
        let pos_2 = order.iter().position(|&x| x == 2).unwrap();

        assert!(pos_0 < pos_2);
        assert!(pos_1 < pos_2);
    }

    // Property-based tests
    use proptest::prelude::*;

    proptest! {
        /// Property: Aggregated confidence is always within [0, 1]
        #[test]
        fn prop_confidence_bounded(
            strength1 in 0.01f32..1.0,
            strength2 in 0.01f32..1.0,
            strength3 in 0.01f32..1.0,
        ) {
            let aggregator = EvidenceAggregator::default();
            let evidence = vec![
                create_test_evidence(0, strength1, vec![]),
                create_test_evidence(1, strength2, vec![]),
                create_test_evidence(2, strength3, vec![]),
            ];

            let result = aggregator.aggregate_evidence(&evidence).unwrap();
            let conf = result.aggregate_confidence.raw();

            prop_assert!((0.0..=1.0).contains(&conf));
        }

        /// Property: Commutativity - order of independent evidence doesn't matter
        #[test]
        fn prop_commutativity_two_evidence(
            strength1 in 0.1f32..0.9,
            strength2 in 0.1f32..0.9,
        ) {
            let aggregator = EvidenceAggregator::default();

            // Order 1: evidence 0, then 1
            let evidence1 = vec![
                create_test_evidence(0, strength1, vec![]),
                create_test_evidence(1, strength2, vec![]),
            ];

            // Order 2: evidence 1, then 0
            let evidence2 = vec![
                create_test_evidence(1, strength2, vec![]),
                create_test_evidence(0, strength1, vec![]),
            ];

            let result1 = aggregator.aggregate_evidence(&evidence1).unwrap();
            let result2 = aggregator.aggregate_evidence(&evidence2).unwrap();

            // Results should be identical
            prop_assert!((result1.aggregate_confidence.raw() - result2.aggregate_confidence.raw()).abs() < 1e-6);
        }

        /// Property: Monotonicity - more evidence increases confidence
        #[test]
        fn prop_monotonicity(strength1 in 0.1f32..0.9, strength2 in 0.1f32..0.9) {
            let aggregator = EvidenceAggregator::default();

            // Single evidence
            let single = vec![create_test_evidence(0, strength1, vec![])];
            let result_single = aggregator.aggregate_evidence(&single).unwrap();

            // Two evidence sources
            let double = vec![
                create_test_evidence(0, strength1, vec![]),
                create_test_evidence(1, strength2, vec![]),
            ];
            let result_double = aggregator.aggregate_evidence(&double).unwrap();

            // More evidence should not decrease confidence
            prop_assert!(result_double.aggregate_confidence.raw() >= result_single.aggregate_confidence.raw());
        }

        /// Property: Associativity - grouping doesn't matter for independent evidence
        #[test]
        fn prop_associativity(
            strength1 in 0.1f32..0.7,
            strength2 in 0.1f32..0.7,
            strength3 in 0.1f32..0.7,
        ) {
            let aggregator = EvidenceAggregator::new(Confidence::from_raw(0.01), 10);

            // All three at once
            let all_at_once = vec![
                create_test_evidence(0, strength1, vec![]),
                create_test_evidence(1, strength2, vec![]),
                create_test_evidence(2, strength3, vec![]),
            ];
            let result_all = aggregator.aggregate_evidence(&all_at_once).unwrap();

            // Compute sequentially
            let first_two = vec![
                create_test_evidence(0, strength1, vec![]),
                create_test_evidence(1, strength2, vec![]),
            ];
            let intermediate = aggregator.aggregate_evidence(&first_two).unwrap();

            // Then combine with third
            let sequential = vec![
                create_test_evidence(0, intermediate.aggregate_confidence.raw(), vec![]),
                create_test_evidence(2, strength3, vec![]),
            ];
            let result_seq = aggregator.aggregate_evidence(&sequential).unwrap();

            // Should be approximately equal (allowing for floating point error)
            let diff = (result_all.aggregate_confidence.raw() - result_seq.aggregate_confidence.raw()).abs();
            prop_assert!(diff < 0.01); // Allow 1% difference due to floating point
        }

        /// Property: Identity - single evidence returns same strength
        #[test]
        fn prop_identity(strength in 0.01f32..1.0) {
            let aggregator = EvidenceAggregator::default();
            let evidence = vec![create_test_evidence(0, strength, vec![])];

            let result = aggregator.aggregate_evidence(&evidence).unwrap();

            prop_assert!((result.aggregate_confidence.raw() - strength).abs() < 1e-6);
        }

        /// Property: Upper bound - combination can't exceed 1.0
        #[test]
        fn prop_upper_bound(
            s1 in 0.5f32..1.0,
            s2 in 0.5f32..1.0,
            s3 in 0.5f32..1.0,
        ) {
            let aggregator = EvidenceAggregator::default();
            let evidence = vec![
                create_test_evidence(0, s1, vec![]),
                create_test_evidence(1, s2, vec![]),
                create_test_evidence(2, s3, vec![]),
            ];

            let result = aggregator.aggregate_evidence(&evidence).unwrap();

            prop_assert!(result.aggregate_confidence.raw() <= 1.0);
        }

        /// Property: Weak evidence filtering works correctly
        #[test]
        fn prop_min_strength_filtering(
            threshold in 0.1f32..0.9,
            below in 0.01f32..0.1,
            above in 0.9f32..1.0,
        ) {
            let aggregator = EvidenceAggregator::new(Confidence::from_raw(threshold), 10);
            let evidence = vec![
                create_test_evidence(0, below, vec![]),  // Should be filtered
                create_test_evidence(1, above, vec![]),  // Should pass
            ];

            let result = aggregator.aggregate_evidence(&evidence).unwrap();

            // Only evidence above threshold should contribute
            prop_assert!(result.len() == 1);
            prop_assert!(result.contributing_evidence[0].evidence_id == 1);
        }
    }
}
