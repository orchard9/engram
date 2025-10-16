//! Probabilistic Query Executor
//!
//! Core query execution engine that transforms recall results into probabilistic
//! query results with evidence tracking, uncertainty propagation, and confidence
//! calibration.
//!
//! # Architecture
//!
//! The executor integrates multiple sources of uncertainty:
//! - Spreading activation paths with hop-dependent decay
//! - Temporal decay from forgetting curves
//! - Direct cue matching with similarity scores
//! - Vector similarity from HNSW indices
//!
//! All evidence is combined using mathematically sound probability aggregation
//! from `ConfidenceAggregator`, ensuring results respect probability axioms.
//!
//! # Example
//!
//! ```
//! use engram_core::query::executor::{ProbabilisticQueryExecutor, QueryExecutorConfig};
//! use engram_core::{Confidence, Episode};
//! use chrono::Utc;
//!
//! let config = QueryExecutorConfig::default();
//! let executor = ProbabilisticQueryExecutor::new(config);
//!
//! // Transform recall results into probabilistic query result
//! let episodes = vec![
//!     (Episode::new(
//!         "test1".to_string(),
//!         Utc::now(),
//!         "Test episode".to_string(),
//!         [0.5f32; 768],
//!         Confidence::HIGH,
//!     ), Confidence::HIGH)
//! ];
//!
//! let result = executor.execute(episodes, &[], vec![]);
//! assert!(!result.is_empty());
//! ```

use crate::activation::confidence_aggregation::{
    ConfidenceAggregationOutcome, ConfidenceAggregator, ConfidencePath,
};
use crate::query::{
    ConfidenceInterval, Evidence, EvidenceSource, MatchType, ProbabilisticQueryResult,
    UncertaintySource, VectorSimilarityEvidence,
};
use crate::{Activation, Confidence, Episode};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Configuration for the probabilistic query executor
#[derive(Debug, Clone)]
pub struct QueryExecutorConfig {
    /// Decay rate for hop-dependent confidence reduction
    pub decay_rate: f32,
    /// Minimum confidence threshold for evidence consideration
    pub min_confidence: Confidence,
    /// Maximum number of activation paths to consider
    pub max_paths: usize,
    /// Enable detailed evidence tracking (may impact performance)
    pub track_evidence: bool,
    /// Enable uncertainty source tracking
    pub track_uncertainty: bool,
}

impl Default for QueryExecutorConfig {
    fn default() -> Self {
        Self {
            decay_rate: 0.35,
            min_confidence: Confidence::from_raw(0.001),
            max_paths: 8,
            track_evidence: true,
            track_uncertainty: true,
        }
    }
}

/// Main probabilistic query executor
///
/// Transforms raw recall results into probabilistic query results with
/// comprehensive evidence tracking and uncertainty propagation.
#[derive(Debug, Clone)]
pub struct ProbabilisticQueryExecutor {
    config: QueryExecutorConfig,
    aggregator: ConfidenceAggregator,
}

impl Default for ProbabilisticQueryExecutor {
    fn default() -> Self {
        Self::new(QueryExecutorConfig::default())
    }
}

impl ProbabilisticQueryExecutor {
    /// Create a new query executor with the given configuration
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Can't be const because ConfidenceAggregator::new isn't const
    pub fn new(config: QueryExecutorConfig) -> Self {
        let aggregator =
            ConfidenceAggregator::new(config.decay_rate, config.min_confidence, config.max_paths);

        Self { config, aggregator }
    }

    /// Execute a probabilistic query on recall results
    ///
    /// Takes recall results (episodes with confidence scores), activation paths,
    /// and uncertainty sources, and produces a `ProbabilisticQueryResult` with
    /// comprehensive evidence tracking.
    ///
    /// # Arguments
    ///
    /// * `episodes` - Raw recall results from memory store
    /// * `activation_paths` - Activation spreading paths for evidence extraction
    /// * `uncertainty_sources` - System-level uncertainty sources
    ///
    /// # Returns
    ///
    /// `ProbabilisticQueryResult` with aggregated confidence, evidence chain,
    /// and uncertainty tracking.
    #[must_use]
    pub fn execute(
        &self,
        episodes: Vec<(Episode, Confidence)>,
        activation_paths: &[ActivationPath],
        uncertainty_sources: Vec<UncertaintySource>,
    ) -> ProbabilisticQueryResult {
        // Start with basic result from episodes
        let mut result = ProbabilisticQueryResult::from_episodes(episodes);

        // Extract evidence from activation paths if tracking enabled
        if self.config.track_evidence && !activation_paths.is_empty() {
            let evidence = Self::extract_evidence_from_paths(activation_paths);
            result.evidence_chain = evidence;
        }

        // Add uncertainty sources if tracking enabled
        if self.config.track_uncertainty {
            result.uncertainty_sources = uncertainty_sources;
        }

        // Aggregate confidence from multiple paths if available
        if !activation_paths.is_empty() {
            let confidence_paths = Self::convert_to_confidence_paths(activation_paths);
            let outcome = self.aggregator.aggregate_paths(&confidence_paths);
            result.confidence_interval = Self::create_interval_from_outcome(&outcome);
        }

        result
    }

    /// Extract evidence from activation paths
    fn extract_evidence_from_paths(paths: &[ActivationPath]) -> Vec<Evidence> {
        let mut evidence = Vec::with_capacity(paths.len());
        let now = SystemTime::now();

        for path in paths {
            let source = EvidenceSource::SpreadingActivation {
                source_episode: path.source_episode_id.clone(),
                activation_level: path.activation,
                path_length: path.hop_count,
            };

            evidence.push(Evidence {
                source,
                strength: path.confidence,
                timestamp: now,
                dependencies: vec![], // Will be populated by dependency tracking in Task 002
            });
        }

        evidence
    }

    /// Convert activation paths to confidence paths for aggregation
    fn convert_to_confidence_paths(paths: &[ActivationPath]) -> Vec<ConfidencePath> {
        paths
            .iter()
            .map(|path| {
                ConfidencePath::new(
                    path.confidence,
                    path.hop_count,
                    path.source_tier,
                    path.weight,
                )
            })
            .collect()
    }

    /// Create confidence interval from aggregation outcome
    fn create_interval_from_outcome(outcome: &ConfidenceAggregationOutcome) -> ConfidenceInterval {
        if outcome.is_empty() {
            return ConfidenceInterval::from_confidence(Confidence::NONE);
        }

        // Calculate uncertainty based on path diversity
        let uncertainty = Self::calculate_path_diversity_uncertainty(outcome);

        ConfidenceInterval::from_confidence_with_uncertainty(outcome.aggregate, uncertainty)
    }

    /// Calculate uncertainty from path diversity
    ///
    /// More diverse paths (different tiers, hop counts) indicate higher uncertainty
    fn calculate_path_diversity_uncertainty(outcome: &ConfidenceAggregationOutcome) -> f32 {
        if outcome.contributing_paths.len() <= 1 {
            return 0.0;
        }

        // Calculate variance in confidence across paths
        let mean: f32 = outcome
            .contributing_paths
            .iter()
            .map(|c| c.decayed.raw())
            .sum::<f32>()
            / outcome.contributing_paths.len() as f32;

        let variance: f32 = outcome
            .contributing_paths
            .iter()
            .map(|c| {
                let diff = c.decayed.raw() - mean;
                diff * diff
            })
            .sum::<f32>()
            / outcome.contributing_paths.len() as f32;

        variance.sqrt().min(0.3) // Cap uncertainty at 0.3
    }

    /// Create evidence from direct cue match
    #[must_use]
    pub fn create_direct_match_evidence(
        cue_id: String,
        similarity_score: f32,
        match_type: MatchType,
    ) -> Evidence {
        Evidence {
            source: EvidenceSource::DirectMatch {
                cue_id,
                similarity_score,
                match_type,
            },
            strength: Confidence::from_raw(similarity_score.clamp(0.0, 1.0)),
            timestamp: SystemTime::now(),
            dependencies: vec![],
        }
    }

    /// Create evidence from temporal decay
    #[must_use]
    pub fn create_temporal_decay_evidence(
        original_confidence: Confidence,
        time_elapsed: Duration,
        decay_rate: f32,
    ) -> Evidence {
        let decay_factor = (-decay_rate * time_elapsed.as_secs_f32() / 3600.0).exp();
        let decayed_confidence =
            Confidence::from_raw((original_confidence.raw() * decay_factor).clamp(0.0, 1.0));

        Evidence {
            source: EvidenceSource::TemporalDecay {
                original_confidence,
                time_elapsed,
                decay_rate,
            },
            strength: decayed_confidence,
            timestamp: SystemTime::now(),
            dependencies: vec![],
        }
    }

    /// Create evidence from vector similarity
    #[must_use]
    pub fn create_vector_similarity_evidence(
        query_vector: Arc<[f32; 768]>,
        result_distance: f32,
        index_confidence: Confidence,
    ) -> Evidence {
        Evidence {
            source: EvidenceSource::VectorSimilarity(Box::new(VectorSimilarityEvidence {
                query_vector,
                result_distance,
                index_confidence,
            })),
            strength: index_confidence,
            timestamp: SystemTime::now(),
            dependencies: vec![],
        }
    }
}

/// Activation path information for evidence extraction
///
/// Represents a single activation spreading path from source to target memory.
#[derive(Debug, Clone)]
pub struct ActivationPath {
    /// Source episode ID where activation originated
    pub source_episode_id: String,
    /// Target episode ID where activation arrived
    pub target_episode_id: String,
    /// Activation strength
    pub activation: Activation,
    /// Confidence value for this path
    pub confidence: Confidence,
    /// Number of hops from source
    pub hop_count: u16,
    /// Storage tier of source memory
    pub source_tier: crate::activation::storage_aware::StorageTier,
    /// Path weight (default 1.0)
    pub weight: f32,
}

impl ActivationPath {
    /// Create a new activation path
    #[must_use]
    pub const fn new(
        source_episode_id: String,
        target_episode_id: String,
        activation: Activation,
        confidence: Confidence,
        hop_count: u16,
        source_tier: crate::activation::storage_aware::StorageTier,
        weight: f32,
    ) -> Self {
        Self {
            source_episode_id,
            target_episode_id,
            activation,
            confidence,
            hop_count,
            source_tier,
            weight,
        }
    }

    /// Create an activation path with default weight
    #[must_use]
    pub const fn with_default_weight(
        source_episode_id: String,
        target_episode_id: String,
        activation: Activation,
        confidence: Confidence,
        hop_count: u16,
        source_tier: crate::activation::storage_aware::StorageTier,
    ) -> Self {
        Self::new(
            source_episode_id,
            target_episode_id,
            activation,
            confidence,
            hop_count,
            source_tier,
            1.0,
        )
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::activation::storage_aware::StorageTier;
    use chrono::Utc;

    fn test_episode(id: &str, confidence: Confidence) -> Episode {
        Episode::new(
            id.to_string(),
            Utc::now(),
            format!("Test episode {id}"),
            [0.5f32; 768],
            confidence,
        )
    }

    #[test]
    fn test_executor_creation() {
        let config = QueryExecutorConfig::default();
        let executor = ProbabilisticQueryExecutor::new(config);
        assert!(executor.config.track_evidence);
        assert!(executor.config.track_uncertainty);
    }

    #[test]
    fn test_execute_with_empty_input() {
        let executor = ProbabilisticQueryExecutor::default();
        let result = executor.execute(vec![], &[], vec![]);
        assert!(result.is_empty());
        assert_eq!(result.confidence_interval.point, Confidence::NONE);
    }

    #[test]
    fn test_execute_with_episodes_only() {
        let executor = ProbabilisticQueryExecutor::default();
        let episodes = vec![
            (test_episode("ep1", Confidence::HIGH), Confidence::HIGH),
            (test_episode("ep2", Confidence::MEDIUM), Confidence::MEDIUM),
        ];

        let result = executor.execute(episodes, &[], vec![]);
        assert_eq!(result.len(), 2);
        assert!(result.is_successful());
        assert!(result.confidence_interval.point.raw() > 0.5);
    }

    #[test]
    fn test_execute_with_activation_paths() {
        let executor = ProbabilisticQueryExecutor::default();
        let episodes = vec![(test_episode("ep1", Confidence::HIGH), Confidence::HIGH)];

        let paths = vec![
            ActivationPath::with_default_weight(
                "source1".to_string(),
                "ep1".to_string(),
                Activation::new(0.8),
                Confidence::HIGH,
                1,
                StorageTier::Hot,
            ),
            ActivationPath::with_default_weight(
                "source2".to_string(),
                "ep1".to_string(),
                Activation::new(0.6),
                Confidence::MEDIUM,
                2,
                StorageTier::Warm,
            ),
        ];

        let result = executor.execute(episodes, &paths, vec![]);
        assert_eq!(result.len(), 1);
        assert_eq!(result.evidence_chain.len(), 2);

        // Should have evidence from both activation paths
        for evidence in &result.evidence_chain {
            if let EvidenceSource::SpreadingActivation { path_length, .. } = evidence.source {
                assert!((1..=2).contains(&path_length));
            } else {
                panic!("Expected SpreadingActivation evidence");
            }
        }
    }

    #[test]
    fn test_evidence_extraction_disabled() {
        let mut config = QueryExecutorConfig::default();
        config.track_evidence = false;
        let executor = ProbabilisticQueryExecutor::new(config);

        let episodes = vec![(test_episode("ep1", Confidence::HIGH), Confidence::HIGH)];
        let paths = vec![ActivationPath::with_default_weight(
            "source".to_string(),
            "ep1".to_string(),
            Activation::new(0.8),
            Confidence::HIGH,
            1,
            StorageTier::Hot,
        )];

        let result = executor.execute(episodes, &paths, vec![]);
        assert!(result.evidence_chain.is_empty());
    }

    #[test]
    fn test_uncertainty_tracking() {
        let executor = ProbabilisticQueryExecutor::default();
        let episodes = vec![(test_episode("ep1", Confidence::HIGH), Confidence::HIGH)];

        let uncertainty = vec![UncertaintySource::SystemPressure {
            pressure_level: 0.3,
            effect_on_confidence: 0.05,
        }];

        let result = executor.execute(episodes, &[], uncertainty);
        assert_eq!(result.uncertainty_sources.len(), 1);
    }

    #[test]
    fn test_direct_match_evidence_creation() {
        let evidence = ProbabilisticQueryExecutor::create_direct_match_evidence(
            "cue1".to_string(),
            0.85,
            MatchType::Semantic,
        );

        assert!((evidence.strength.raw() - 0.85).abs() < 1e-6);
        if let EvidenceSource::DirectMatch {
            cue_id,
            similarity_score,
            match_type,
        } = evidence.source
        {
            assert_eq!(cue_id, "cue1");
            assert!((similarity_score - 0.85).abs() < 1e-6);
            matches!(match_type, MatchType::Semantic);
        } else {
            panic!("Expected DirectMatch evidence");
        }
    }

    #[test]
    fn test_temporal_decay_evidence_creation() {
        let original = Confidence::HIGH;
        let elapsed = Duration::from_secs(3600 * 2); // 2 hours
        let decay_rate = 1.0; // 1 hour tau

        let evidence = ProbabilisticQueryExecutor::create_temporal_decay_evidence(
            original, elapsed, decay_rate,
        );

        // After 2 hours with tau=1h, retention = e^(-2) ≈ 0.135
        let expected_retention = (-2.0f32).exp();
        assert!(
            (evidence.strength.raw() - (Confidence::HIGH.raw() * expected_retention)).abs() < 0.01
        );
    }

    #[test]
    fn test_path_diversity_uncertainty() {
        let executor = ProbabilisticQueryExecutor::default();

        // Create paths with high diversity
        let paths = vec![
            ActivationPath::with_default_weight(
                "s1".to_string(),
                "t".to_string(),
                Activation::new(0.9),
                Confidence::from_raw(0.9),
                1,
                StorageTier::Hot,
            ),
            ActivationPath::with_default_weight(
                "s2".to_string(),
                "t".to_string(),
                Activation::new(0.3),
                Confidence::from_raw(0.3),
                5,
                StorageTier::Cold,
            ),
        ];

        let confidence_paths = ProbabilisticQueryExecutor::convert_to_confidence_paths(&paths);
        let outcome = executor.aggregator.aggregate_paths(&confidence_paths);
        let uncertainty =
            ProbabilisticQueryExecutor::calculate_path_diversity_uncertainty(&outcome);

        // High diversity should produce non-zero uncertainty
        assert!(uncertainty > 0.0);
        assert!(uncertainty <= 0.3); // Capped at 0.3
    }
}
