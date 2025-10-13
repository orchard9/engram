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

// Query expansion modules (Milestone 3.6 Task 003)
pub mod expansion;
pub mod lexicon;

// Figurative language interpretation (Milestone 3.6 Task 004)
pub mod analogy;
pub mod figurative;

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
}
