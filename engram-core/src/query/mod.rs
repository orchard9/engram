//! Probabilistic Query Engine with Formal Verification
//!
//! This module implements mathematically sound uncertainty propagation through
//! query operations with comprehensive formal verification, distinguishing
//! "no results" from "low confidence results" while ensuring all probabilistic
//! operations are correct by construction.
//!
//! ## Design Principles
//!
//! 1. **Cognitive Compatibility**: Seamless integration with existing Confidence type
//! 2. **Formal Verification**: All probability operations verified with SMT solvers
//! 3. **Bias Prevention**: Systematic prevention of cognitive biases in reasoning
//! 4. **Performance**: Lock-free operations optimized for cognitive memory workloads
//! 5. **Uncertainty Tracking**: Comprehensive tracking of uncertainty sources

use crate::{Activation, Confidence, Episode};
use std::convert::TryFrom;
use std::sync::Arc;
use std::time::SystemTime;

#[must_use]
#[inline]
pub(crate) const fn clamp_probability_to_f32(value: f64) -> f32 {
    // Confidence values are guaranteed to be within [0,1], so truncation is safe.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    {
        value.clamp(0.0, 1.0) as f32
    }
}

// Parser infrastructure (Milestone 9 Task 001)
pub mod parser;

// Query expansion modules (Milestone 3.6 Task 003)
pub mod expansion;
pub mod lexicon;

// Figurative language interpretation (Milestone 3.6 Task 004)
pub mod analogy;
pub mod figurative;

// Probabilistic query executor (Milestone 5 Task 001)
pub mod executor;

// Evidence aggregation with dependency tracking (Milestone 5 Task 002)
pub mod dependency_graph;
pub mod evidence_aggregator;

// Uncertainty tracking system (Milestone 5 Task 003)
pub mod uncertainty_tracker;

// Confidence calibration framework (Milestone 5 Task 004)
pub mod confidence_calibration;

// Conditional compilation for SMT verification features
#[cfg(feature = "probabilistic_queries")]
pub mod evidence;
#[cfg(feature = "probabilistic_queries")]
pub mod integration;

// Testing and verification modules
#[cfg(all(feature = "probabilistic_queries", test))]
pub mod property_tests;
#[cfg(all(feature = "probabilistic_queries", feature = "smt_verification"))]
pub mod verification;

/// Lock-free probabilistic query result extending existing `MemoryStore::recall` interface
#[derive(Debug, Clone)]
pub struct ProbabilisticQueryResult {
    /// Episodes with confidence scores (compatible with existing recall interface)
    pub episodes: Vec<(Episode, Confidence)>,
    /// Enhanced confidence interval around the point confidence estimates
    pub confidence_interval: ConfidenceInterval,
    /// Evidence chain with dependency tracking for proper Bayesian updating
    pub evidence_chain: Vec<Evidence>,
    /// Uncertainty sources from activation spreading and decay functions
    pub uncertainty_sources: Vec<UncertaintySource>,
}

/// Confidence interval extending the existing Confidence type with interval arithmetic
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ConfidenceInterval {
    /// Lower bound as existing Confidence type
    pub lower: Confidence,
    /// Upper bound as existing Confidence type  
    pub upper: Confidence,
    /// Point estimate (matches existing single Confidence value)
    pub point: Confidence,
    /// Width measure for uncertainty quantification
    pub width: f32,
}

impl ConfidenceInterval {
    /// Create interval from existing Confidence with estimated uncertainty
    #[must_use]
    pub fn from_confidence_with_uncertainty(point: Confidence, uncertainty: f32) -> Self {
        let raw_point = point.raw();
        let half_width = (uncertainty * raw_point).min(raw_point.min(1.0 - raw_point));

        Self {
            lower: Confidence::exact(raw_point - half_width),
            upper: Confidence::exact(raw_point + half_width),
            point,
            width: half_width * 2.0,
        }
    }

    /// Convert to existing Confidence type (backward compatibility)
    #[must_use]
    pub const fn as_confidence(&self) -> Confidence {
        self.point
    }

    /// Create a point interval (no uncertainty) from existing Confidence
    #[must_use]
    pub const fn from_confidence(confidence: Confidence) -> Self {
        Self {
            lower: confidence,
            upper: confidence,
            point: confidence,
            width: 0.0,
        }
    }

    /// Test if interval contains value
    #[must_use]
    pub fn contains(&self, value: f32) -> bool {
        value >= self.lower.raw() && value <= self.upper.raw()
    }

    /// Test if this interval overlaps with another
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.lower.raw() <= other.upper.raw() && other.lower.raw() <= self.upper.raw()
    }

    /// Extend existing Confidence logical operations to intervals
    #[must_use]
    pub fn and(&self, other: &Self) -> Self {
        let point_and = self.point.and(other.point);
        let lower_and = self.lower.and(other.lower);
        let upper_and = self.upper.and(other.upper);

        Self {
            lower: lower_and,
            upper: upper_and,
            point: point_and,
            width: (upper_and.raw() - lower_and.raw()).max(0.0),
        }
    }

    /// Interval OR operation extending existing `Confidence::or`
    #[must_use]
    pub fn or(&self, other: &Self) -> Self {
        let point_or = self.point.or(other.point);
        let lower_or = self.lower.or(other.lower);
        let upper_or = self.upper.or(other.upper);

        Self {
            lower: lower_or,
            upper: upper_or,
            point: point_or,
            width: (upper_or.raw() - lower_or.raw()).max(0.0),
        }
    }

    /// Negation of confidence interval
    #[must_use]
    pub const fn not(&self) -> Self {
        Self {
            lower: self.upper.not(),
            upper: self.lower.not(),
            point: self.point.not(),
            width: self.width,
        }
    }
}

/// Evidence from activation spreading and other uncertainty sources
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Source of evidence (activation spreading, decay function, etc.)
    pub source: EvidenceSource,
    /// Strength as existing Confidence type
    pub strength: Confidence,
    /// Time when evidence was collected
    pub timestamp: SystemTime,
    /// Dependencies on other evidence for circular dependency detection
    pub dependencies: Vec<EvidenceId>,
}

/// Sources of evidence integrated with existing engram-core systems
#[derive(Debug, Clone)]
pub enum EvidenceSource {
    /// From existing `MemoryStore` spreading activation
    SpreadingActivation {
        /// Identifier for the episode the activation originated from.
        source_episode: String,
        /// Activation strength recorded during the spread.
        activation_level: Activation,
        /// Hop count taken to reach the current node.
        path_length: u16,
    },
    /// From decay functions (Task 005 integration)
    TemporalDecay {
        /// Baseline confidence before decay was applied.
        original_confidence: Confidence,
        /// Duration elapsed since the memory was encoded.
        time_elapsed: std::time::Duration,
        /// Decay rate used to adjust the confidence.
        decay_rate: f32,
    },
    /// From direct cue matching (existing recall logic)
    DirectMatch {
        /// Cue identifier that produced the match.
        cue_id: String,
        /// Similarity score returned by the matcher.
        similarity_score: f32,
        /// Type of cue match that was performed.
        match_type: MatchType,
    },
    /// From HNSW index results (Task 002 integration)
    VectorSimilarity(Box<VectorSimilarityEvidence>),
}

/// Evidence collected from vector similarity comparisons.
#[derive(Debug, Clone)]
pub struct VectorSimilarityEvidence {
    /// Query embedding vector used in the search.
    pub query_vector: Arc<[f32; 768]>,
    /// Distance between the query and the retrieved vector.
    pub result_distance: f32,
    /// Confidence derived from index ranking heuristics.
    pub index_confidence: Confidence,
}

/// Types of matches for evidence classification
#[derive(Debug, Clone, Copy)]
pub enum MatchType {
    /// Embedding similarity comparison.
    Embedding,
    /// Semantic or symbolic match detected.
    Semantic,
    /// Temporal overlap or proximity.
    Temporal,
    /// Contextual association match.
    Context,
}

/// Sources of uncertainty in the system
#[derive(Debug, Clone)]
pub enum UncertaintySource {
    /// Elevated system load affecting recall paths.
    SystemPressure {
        /// Severity of the load or contention.
        pressure_level: f32,
        /// Estimated impact on resulting confidence scores.
        effect_on_confidence: f32,
    },
    /// Randomness introduced by spreading activation heuristics.
    SpreadingActivationNoise {
        /// Variance observed across activation samples.
        activation_variance: f32,
        /// Diversity of paths explored during spreading.
        path_diversity: f32,
    },
    /// Uncertainty stemming from incomplete decay models.
    TemporalDecayUnknown {
        /// Time elapsed since the memory was encoded.
        time_since_encoding: std::time::Duration,
        /// Amount of uncertainty in the decay model parameters.
        decay_model_uncertainty: f32,
    },
    /// Instrumentation or measurement noise.
    MeasurementError {
        /// Magnitude of the observed error.
        error_magnitude: f32,
        /// Estimated reduction in confidence due to noise.
        confidence_degradation: f32,
    },
}

/// Unique identifier for evidence tracking
pub type EvidenceId = u64;

/// Error types for probabilistic operations
#[derive(Debug, thiserror::Error)]
pub enum ProbabilisticError {
    /// Provided probability is outside the valid 0..=1 interval.
    #[error("Invalid probability value: {value} (must be in range 0..=1)")]
    InvalidProbability {
        /// Value that violated probability bounds.
        value: f32,
    },

    /// Evidence graph forms a loop that prevents stable inference.
    #[error("Circular dependency detected in evidence chain")]
    CircularDependency,

    /// Not enough supporting data to produce a confident result.
    #[error("Insufficient evidence for reliable probability estimate")]
    InsufficientEvidence,

    /// Numerical operations became unstable or divergent.
    #[error("Numerical instability in probability calculation")]
    NumericalInstability,

    /// SMT solver reported a failure while validating a property.
    #[error("SMT verification failed: {reason}")]
    VerificationFailed {
        /// Explanation reported by the solver.
        reason: String,
    },
}

/// Result type for probabilistic operations
pub type ProbabilisticResult<T> = Result<T, ProbabilisticError>;

// Public API that extends existing MemoryStore functionality
impl ProbabilisticQueryResult {
    /// Create a simple result from existing recall interface
    #[must_use]
    pub fn from_episodes(episodes: Vec<(Episode, Confidence)>) -> Self {
        // Calculate overall confidence interval
        let overall_confidence = if episodes.is_empty() {
            ConfidenceInterval::from_confidence(Confidence::NONE)
        } else {
            let len_u32 = u32::try_from(episodes.len()).unwrap_or(u32::MAX);
            let total_count = f64::from(len_u32).max(1.0);
            let sum_confidence: f64 = episodes.iter().map(|(_, c)| f64::from(c.raw())).sum();
            let avg_confidence = sum_confidence / total_count;

            // Estimate uncertainty from result diversity
            let variance = episodes
                .iter()
                .map(|(_, c)| {
                    let diff = f64::from(c.raw()) - avg_confidence;
                    diff * diff
                })
                .sum::<f64>()
                / total_count;
            let uncertainty = variance.sqrt();

            ConfidenceInterval::from_confidence_with_uncertainty(
                Confidence::exact(clamp_probability_to_f32(avg_confidence)),
                clamp_probability_to_f32(uncertainty),
            )
        };

        Self {
            episodes,
            confidence_interval: overall_confidence,
            evidence_chain: Vec::new(),
            uncertainty_sources: Vec::new(),
        }
    }

    /// Check if result indicates successful query
    #[must_use]
    pub const fn is_successful(&self) -> bool {
        !self.episodes.is_empty() && self.confidence_interval.point.is_high()
    }

    /// Get the number of results
    #[must_use]
    pub const fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Check if result is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Conjunctive query: A AND B
    ///
    /// Combines two query results using intersection semantics:
    /// - Episodes: Returns episodes present in both results
    /// - Confidence: Multiplies intervals (independence assumption)
    /// - Evidence: Merges both chains
    /// - Uncertainty: Combines sources from both queries
    #[must_use]
    pub fn and(&self, other: &Self) -> Self {
        // Intersect episodes: only keep episodes in both results
        let episodes = self.intersect_episodes(&other.episodes);

        // Combine confidence intervals (AND operation)
        let confidence_interval = self.confidence_interval.and(&other.confidence_interval);

        // Merge evidence chains
        let evidence_chain = self.merge_evidence_chains(&other.evidence_chain);

        // Combine uncertainty sources
        let uncertainty_sources = self.combine_uncertainty_sources(&other.uncertainty_sources);

        Self {
            episodes,
            confidence_interval,
            evidence_chain,
            uncertainty_sources,
        }
    }

    /// Disjunctive query: A OR B
    ///
    /// Combines two query results using union semantics:
    /// - Episodes: Returns episodes from either result
    /// - Confidence: Combines intervals using probabilistic OR
    /// - Evidence: Merges both chains
    /// - Uncertainty: Combines sources from both queries
    #[must_use]
    pub fn or(&self, other: &Self) -> Self {
        // Union episodes: keep all unique episodes
        let episodes = self.union_episodes(&other.episodes);

        // Combine confidence intervals (OR operation)
        let confidence_interval = self.confidence_interval.or(&other.confidence_interval);

        // Merge evidence chains
        let evidence_chain = self.merge_evidence_chains(&other.evidence_chain);

        // Combine uncertainty sources
        let uncertainty_sources = self.combine_uncertainty_sources(&other.uncertainty_sources);

        Self {
            episodes,
            confidence_interval,
            evidence_chain,
            uncertainty_sources,
        }
    }

    /// Negation query: NOT A
    ///
    /// Negates the confidence of a query result:
    /// - Episodes: Kept unchanged (negation affects confidence only)
    /// - Confidence: Inverted using NOT operation
    /// - Evidence: Preserved as-is
    /// - Uncertainty: Preserved as-is
    #[must_use]
    pub fn not(&self) -> Self {
        // Episodes remain unchanged (negation is confidence-only)
        let episodes = self.episodes.clone();

        // Negate confidence interval
        let confidence_interval = self.confidence_interval.not();

        // Evidence and uncertainty unchanged
        let evidence_chain = self.evidence_chain.clone();
        let uncertainty_sources = self.uncertainty_sources.clone();

        Self {
            episodes,
            confidence_interval,
            evidence_chain,
            uncertainty_sources,
        }
    }

    /// Helper: Intersect episode sets (for AND operation)
    fn intersect_episodes(
        &self,
        other_episodes: &[(Episode, Confidence)],
    ) -> Vec<(Episode, Confidence)> {
        let other_ids: std::collections::HashSet<_> =
            other_episodes.iter().map(|(ep, _)| &ep.id).collect();

        self.episodes
            .iter()
            .filter(|(ep, _)| other_ids.contains(&ep.id))
            .cloned()
            .collect()
    }

    /// Helper: Union episode sets (for OR operation)
    fn union_episodes(
        &self,
        other_episodes: &[(Episode, Confidence)],
    ) -> Vec<(Episode, Confidence)> {
        let mut result = self.episodes.clone();

        // Collect existing IDs first to avoid borrow issues
        let existing_ids: std::collections::HashSet<_> =
            self.episodes.iter().map(|(ep, _)| &ep.id).collect();

        for (ep, conf) in other_episodes {
            if !existing_ids.contains(&ep.id) {
                result.push((ep.clone(), *conf));
            }
        }

        result
    }

    /// Helper: Merge evidence chains from two queries
    fn merge_evidence_chains(&self, other_evidence: &[Evidence]) -> Vec<Evidence> {
        let mut result = self.evidence_chain.clone();
        result.extend_from_slice(other_evidence);
        result
    }

    /// Helper: Combine uncertainty sources from two queries
    fn combine_uncertainty_sources(
        &self,
        other_uncertainty: &[UncertaintySource],
    ) -> Vec<UncertaintySource> {
        let mut result = self.uncertainty_sources.clone();
        result.extend_from_slice(other_uncertainty);
        result
    }
}

// Extension trait for existing MemoryStore
/// Adds a probabilistic recall API layered on top of deterministic retrieval.
pub trait ProbabilisticRecall {
    /// Enhanced recall with probabilistic uncertainty propagation
    fn recall_probabilistic(&self, cue: crate::Cue) -> ProbabilisticQueryResult;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Episode;

    #[test]
    fn test_confidence_interval_creation() {
        let point = Confidence::exact(0.7);
        let interval = ConfidenceInterval::from_confidence_with_uncertainty(point, 0.1);

        assert!(interval.lower.raw() <= interval.point.raw());
        assert!(interval.point.raw() <= interval.upper.raw());
        assert!(interval.contains(0.7));
        assert!(!interval.contains(0.5));
    }

    #[test]
    fn test_confidence_interval_logical_operations() {
        let a = ConfidenceInterval::from_confidence_with_uncertainty(Confidence::exact(0.8), 0.05);
        let b = ConfidenceInterval::from_confidence_with_uncertainty(Confidence::exact(0.6), 0.1);

        let and_result = a.and(&b);
        let or_result = a.or(&b);

        // AND result should be no greater than either input
        assert!(and_result.upper.raw() <= a.lower.raw() || and_result.upper.raw() <= b.lower.raw());

        // OR result should be no less than either input
        assert!(or_result.lower.raw() >= a.upper.raw() || or_result.lower.raw() >= b.upper.raw());
    }

    #[test]
    fn test_probabilistic_query_result_creation() {
        use chrono::Utc;

        let episode1 = Episode {
            id: "test1".to_string(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: "test episode".to_string(),
            embedding: [0.5f32; 768],
            embedding_provenance: None, // Test episode doesn't need provenance
            encoding_confidence: Confidence::HIGH,
            vividness_confidence: Confidence::HIGH,
            reliability_confidence: Confidence::HIGH,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 0.1,
            decay_function: None, // Use system default
            metadata: std::collections::HashMap::new(),
        };

        let episode2 = Episode {
            id: "test2".to_string(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: "test episode 2".to_string(),
            embedding: [0.3f32; 768],
            embedding_provenance: None, // Test episode doesn't need provenance
            encoding_confidence: Confidence::MEDIUM,
            vividness_confidence: Confidence::MEDIUM,
            reliability_confidence: Confidence::MEDIUM,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 0.1,
            decay_function: None, // Use system default
            metadata: std::collections::HashMap::new(),
        };

        let episode_list = vec![(episode1, Confidence::HIGH), (episode2, Confidence::MEDIUM)];

        let result = ProbabilisticQueryResult::from_episodes(episode_list);

        assert_eq!(result.episodes.len(), 2);
        assert!(result.is_successful());
        assert!(!result.is_empty());

        // Confidence interval should reflect the average
        let expected_avg = f32::midpoint(Confidence::HIGH.raw(), Confidence::MEDIUM.raw());
        assert!((result.confidence_interval.point.raw() - expected_avg).abs() < 1e-6);
    }

    #[test]
    fn test_empty_result() {
        let result = ProbabilisticQueryResult::from_episodes(Vec::new());

        assert!(!result.is_successful());
        assert!(result.is_empty());
        assert_eq!(result.confidence_interval.point, Confidence::NONE);
    }

    // Helper function to create test episodes
    fn create_test_episode(id: &str, confidence: Confidence) -> Episode {
        use chrono::Utc;
        Episode {
            id: id.to_string(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: format!("test episode {id}"),
            embedding: [0.5f32; 768],
            embedding_provenance: None,
            encoding_confidence: confidence,
            vividness_confidence: confidence,
            reliability_confidence: confidence,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 0.1,
            decay_function: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_query_and_operation_intersection_semantics() {
        // Create two query results with overlapping episodes
        let episode1 = create_test_episode("ep1", Confidence::HIGH);
        let episode2 = create_test_episode("ep2", Confidence::HIGH);
        let episode3 = create_test_episode("ep3", Confidence::MEDIUM);

        let result_a = ProbabilisticQueryResult::from_episodes(vec![
            (episode1, Confidence::HIGH),
            (episode2.clone(), Confidence::HIGH),
        ]);

        let result_b = ProbabilisticQueryResult::from_episodes(vec![
            (episode2, Confidence::MEDIUM),
            (episode3, Confidence::MEDIUM),
        ]);

        // AND operation should only keep episode2 (present in both)
        let and_result = result_a.and(&result_b);

        assert_eq!(and_result.len(), 1);
        assert_eq!(and_result.episodes[0].0.id, "ep2");

        // Confidence should be lower due to AND operation (multiplication)
        assert!(
            and_result.confidence_interval.point.raw() <= result_a.confidence_interval.point.raw()
        );
        assert!(
            and_result.confidence_interval.point.raw() <= result_b.confidence_interval.point.raw()
        );
    }

    #[test]
    fn test_query_or_operation_union_semantics() {
        // Create two query results with different episodes
        let episode1 = create_test_episode("ep1", Confidence::HIGH);
        let episode2 = create_test_episode("ep2", Confidence::MEDIUM);
        let episode3 = create_test_episode("ep3", Confidence::MEDIUM);

        let result_a = ProbabilisticQueryResult::from_episodes(vec![
            (episode1, Confidence::HIGH),
            (episode2.clone(), Confidence::MEDIUM),
        ]);

        let result_b = ProbabilisticQueryResult::from_episodes(vec![
            (episode2, Confidence::MEDIUM),
            (episode3, Confidence::MEDIUM),
        ]);

        // OR operation should include all unique episodes
        let or_result = result_a.or(&result_b);

        assert_eq!(or_result.len(), 3);

        // Check all episodes are present
        let ids: Vec<_> = or_result
            .episodes
            .iter()
            .map(|(ep, _)| ep.id.as_str())
            .collect();
        assert!(ids.contains(&"ep1"));
        assert!(ids.contains(&"ep2"));
        assert!(ids.contains(&"ep3"));

        // Confidence should be higher due to OR operation
        assert!(
            or_result.confidence_interval.point.raw()
                >= result_a
                    .confidence_interval
                    .point
                    .raw()
                    .min(result_b.confidence_interval.point.raw())
        );
    }

    #[test]
    fn test_query_not_operation_negation_semantics() {
        let episode1 = create_test_episode("ep1", Confidence::HIGH);
        let episode2 = create_test_episode("ep2", Confidence::MEDIUM);

        let result = ProbabilisticQueryResult::from_episodes(vec![
            (episode1, Confidence::HIGH),
            (episode2, Confidence::MEDIUM),
        ]);

        // NOT operation should negate confidence but keep episodes
        let not_result = result.not();

        // Episodes should be unchanged
        assert_eq!(not_result.len(), result.len());
        assert_eq!(not_result.episodes[0].0.id, result.episodes[0].0.id);
        assert_eq!(not_result.episodes[1].0.id, result.episodes[1].0.id);

        // Confidence should be inverted
        let original_conf = result.confidence_interval.point.raw();
        let negated_conf = not_result.confidence_interval.point.raw();
        assert!((negated_conf - (1.0 - original_conf)).abs() < 1e-6);
    }

    #[test]
    fn test_query_and_with_empty_results() {
        let episode1 = create_test_episode("ep1", Confidence::HIGH);
        let result = ProbabilisticQueryResult::from_episodes(vec![(episode1, Confidence::HIGH)]);
        let empty = ProbabilisticQueryResult::from_episodes(Vec::new());

        let and_result = result.and(&empty);

        // AND with empty should yield empty
        assert!(and_result.is_empty());
    }

    #[test]
    fn test_query_or_with_empty_results() {
        let episode1 = create_test_episode("ep1", Confidence::HIGH);
        let result = ProbabilisticQueryResult::from_episodes(vec![(episode1, Confidence::HIGH)]);
        let empty = ProbabilisticQueryResult::from_episodes(Vec::new());

        let or_result = result.or(&empty);

        // OR with empty should preserve original
        assert_eq!(or_result.len(), 1);
        assert_eq!(or_result.episodes[0].0.id, "ep1");
    }

    #[test]
    fn test_evidence_chain_merging() {
        use std::time::SystemTime;

        let episode1 = create_test_episode("ep1", Confidence::HIGH);
        let episode2 = create_test_episode("ep2", Confidence::HIGH);

        // Create result A with evidence chain
        let mut result_a =
            ProbabilisticQueryResult::from_episodes(vec![(episode1, Confidence::HIGH)]);
        result_a.evidence_chain = vec![Evidence {
            source: EvidenceSource::DirectMatch {
                cue_id: "cue1".to_string(),
                similarity_score: 0.9,
                match_type: MatchType::Embedding,
            },
            strength: Confidence::HIGH,
            timestamp: SystemTime::now(),
            dependencies: vec![],
        }];

        // Create result B with different evidence
        let mut result_b =
            ProbabilisticQueryResult::from_episodes(vec![(episode2, Confidence::MEDIUM)]);
        result_b.evidence_chain = vec![Evidence {
            source: EvidenceSource::DirectMatch {
                cue_id: "cue2".to_string(),
                similarity_score: 0.7,
                match_type: MatchType::Semantic,
            },
            strength: Confidence::MEDIUM,
            timestamp: SystemTime::now(),
            dependencies: vec![],
        }];

        // AND operation should merge evidence chains
        let and_result = result_a.and(&result_b);
        assert_eq!(and_result.evidence_chain.len(), 2);

        // OR operation should also merge evidence chains
        let or_result = result_a.or(&result_b);
        assert_eq!(or_result.evidence_chain.len(), 2);
    }

    #[test]
    fn test_uncertainty_source_combining() {
        let episode1 = create_test_episode("ep1", Confidence::HIGH);
        let episode2 = create_test_episode("ep2", Confidence::HIGH);

        // Create result A with uncertainty source
        let mut result_a =
            ProbabilisticQueryResult::from_episodes(vec![(episode1, Confidence::HIGH)]);
        result_a.uncertainty_sources = vec![UncertaintySource::SystemPressure {
            pressure_level: 0.3,
            effect_on_confidence: 0.1,
        }];

        // Create result B with different uncertainty
        let mut result_b =
            ProbabilisticQueryResult::from_episodes(vec![(episode2, Confidence::MEDIUM)]);
        result_b.uncertainty_sources = vec![UncertaintySource::SpreadingActivationNoise {
            activation_variance: 0.05,
            path_diversity: 0.2,
        }];

        // AND operation should combine uncertainty sources
        let and_result = result_a.and(&result_b);
        assert_eq!(and_result.uncertainty_sources.len(), 2);

        // OR operation should also combine uncertainty sources
        let or_result = result_a.or(&result_b);
        assert_eq!(or_result.uncertainty_sources.len(), 2);
    }

    #[test]
    fn test_conjunction_bound_axiom() {
        // Verify P(A AND B) <= min(P(A), P(B))
        let episode1 = create_test_episode("ep1", Confidence::HIGH);
        let episode2 = create_test_episode("ep2", Confidence::HIGH);

        let result_a =
            ProbabilisticQueryResult::from_episodes(vec![(episode1, Confidence::exact(0.8))]);
        let result_b =
            ProbabilisticQueryResult::from_episodes(vec![(episode2, Confidence::exact(0.6))]);

        let and_result = result_a.and(&result_b);

        // Conjunction bound: P(A ∧ B) ≤ min(P(A), P(B))
        let min_confidence = result_a
            .confidence_interval
            .point
            .raw()
            .min(result_b.confidence_interval.point.raw());
        assert!(and_result.confidence_interval.point.raw() <= min_confidence + 1e-6);
    }

    #[test]
    fn test_disjunction_bound_axiom() {
        // Verify P(A OR B) >= max(P(A), P(B))
        let episode1 = create_test_episode("ep1", Confidence::MEDIUM);
        let episode2 = create_test_episode("ep2", Confidence::LOW);

        let result_a =
            ProbabilisticQueryResult::from_episodes(vec![(episode1, Confidence::exact(0.5))]);
        let result_b =
            ProbabilisticQueryResult::from_episodes(vec![(episode2, Confidence::exact(0.3))]);

        let or_result = result_a.or(&result_b);

        // Disjunction bound: P(A ∨ B) ≥ max(P(A), P(B))
        let max_confidence = result_a
            .confidence_interval
            .point
            .raw()
            .max(result_b.confidence_interval.point.raw());
        assert!(or_result.confidence_interval.point.raw() >= max_confidence - 1e-6);
    }

    #[test]
    fn test_negation_bounds_axiom() {
        // Verify P(NOT A) = 1 - P(A) and stays within [0,1]
        let episode1 = create_test_episode("ep1", Confidence::HIGH);
        let result =
            ProbabilisticQueryResult::from_episodes(vec![(episode1, Confidence::exact(0.75))]);

        let not_result = result.not();

        // Check negation formula
        let expected = 1.0 - result.confidence_interval.point.raw();
        assert!((not_result.confidence_interval.point.raw() - expected).abs() < 1e-6);

        // Check bounds
        assert!(not_result.confidence_interval.point.raw() >= 0.0);
        assert!(not_result.confidence_interval.point.raw() <= 1.0);
    }

    #[test]
    fn test_associativity_of_and_operation() {
        // Verify (A AND B) AND C = A AND (B AND C)
        let ep1 = create_test_episode("ep1", Confidence::HIGH);

        let a = ProbabilisticQueryResult::from_episodes(vec![(ep1.clone(), Confidence::HIGH)]);
        let b = ProbabilisticQueryResult::from_episodes(vec![(ep1.clone(), Confidence::MEDIUM)]);
        let c = ProbabilisticQueryResult::from_episodes(vec![(ep1, Confidence::LOW)]);

        let left = a.and(&b).and(&c);
        let right = a.and(&b.and(&c));

        // Confidence should be approximately equal (within floating point error)
        assert!(
            (left.confidence_interval.point.raw() - right.confidence_interval.point.raw()).abs()
                < 1e-5
        );
    }

    #[test]
    fn test_commutativity_of_and_operation() {
        // Verify A AND B = B AND A
        let ep1 = create_test_episode("ep1", Confidence::HIGH);
        let ep2 = create_test_episode("ep2", Confidence::MEDIUM);

        let a = ProbabilisticQueryResult::from_episodes(vec![(ep1, Confidence::HIGH)]);
        let b = ProbabilisticQueryResult::from_episodes(vec![(ep2, Confidence::MEDIUM)]);

        let ab = a.and(&b);
        let ba = b.and(&a);

        // Results should be equal (order shouldn't matter)
        assert_eq!(ab.len(), ba.len());
        assert!(
            (ab.confidence_interval.point.raw() - ba.confidence_interval.point.raw()).abs() < 1e-6
        );
    }

    #[test]
    fn test_commutativity_of_or_operation() {
        // Verify A OR B = B OR A
        let ep1 = create_test_episode("ep1", Confidence::HIGH);
        let ep2 = create_test_episode("ep2", Confidence::MEDIUM);

        let a = ProbabilisticQueryResult::from_episodes(vec![(ep1, Confidence::HIGH)]);
        let b = ProbabilisticQueryResult::from_episodes(vec![(ep2, Confidence::MEDIUM)]);

        let ab = a.or(&b);
        let ba = b.or(&a);

        // Results should be equal (order shouldn't matter)
        assert_eq!(ab.len(), ba.len());
        assert!(
            (ab.confidence_interval.point.raw() - ba.confidence_interval.point.raw()).abs() < 1e-6
        );
    }

    #[test]
    fn test_double_negation_law() {
        // Verify NOT (NOT A) ≈ A
        let ep1 = create_test_episode("ep1", Confidence::HIGH);
        let result = ProbabilisticQueryResult::from_episodes(vec![(ep1, Confidence::exact(0.7))]);

        let double_negated = result.not().not();

        // Should approximately equal original
        assert!(
            (double_negated.confidence_interval.point.raw()
                - result.confidence_interval.point.raw())
            .abs()
                < 1e-5
        );
    }
}
