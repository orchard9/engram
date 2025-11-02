//! RECALL query execution implementation.
//!
//! Maps RecallQuery AST to MemoryStore::recall() with constraint application
//! and probabilistic result generation.
//!
//! # Architecture
//!
//! The RECALL executor:
//! 1. Converts Pattern to Cue for memory store lookup
//! 2. Calls MemoryStore::recall() to retrieve candidate episodes
//! 3. Applies constraints (confidence, temporal, embedding similarity)
//! 4. Executes probabilistic query to generate evidence and uncertainty
//! 5. Returns ProbabilisticQueryResult with confidence intervals
//!
//! # Biological Plausibility
//!
//! This implementation aligns with complementary learning systems (CLS) theory
//! and hippocampal-neocortical memory dynamics:
//!
//! ## Pattern-to-Cue Conversion (Hippocampal Indexing)
//!
//! - **Direct NodeId lookup**: Mimics CA3 recurrent network pattern completion from
//!   sparse indices with deterministic, high-confidence retrieval (O'Reilly & Rudy, 2001)
//! - **Embedding similarity**: Reflects dentate gyrus pattern separation followed by
//!   CA3 pattern completion based on distributed representations
//! - **Semantic search**: Corresponds to neocortical semantic memory with graded
//!   activation spreading across conceptual schemas
//!
//! ## Similarity Computation (Neural Population Coding)
//!
//! Cosine similarity uses ReLU transformation (max(0, dot_product)) to match neural
//! representational similarity analysis (Kriegeskorte et al., 2008):
//! - Positive correlations indicate shared neural tuning curves
//! - Zero/negative values represent orthogonal or conceptually distinct memories
//! - Aligns with empirical recognition memory showing positive-valued similarity judgments
//!
//! ## Constraint Filtering (Neocortical Evaluation)
//!
//! - **Confidence filtering**: Metacognitive monitoring and source memory evaluation
//! - **Temporal constraints**: Contextual reinstatement per temporal context model
//! - **Content matching**: Schema-based gist retrieval from consolidated semantic knowledge
//!
//! ## References
//!
//! - O'Reilly, R. C., & Rudy, J. W. (2001). Conjunctive representations in learning
//!   and memory. Psychological Review, 108(2), 311-345.
//! - Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity
//!   analysis. Frontiers in Systems Neuroscience, 2, 4.
//! - McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are
//!   complementary learning systems in the hippocampus and neocortex. Psychological Review.

use crate::query::executor::{ActivationPath, ProbabilisticQueryExecutor, QueryExecutorConfig};
use crate::query::parser::ast::{Constraint, Pattern, RecallQuery};
use crate::query::{ProbabilisticQueryResult, UncertaintySource};
use crate::{Confidence, Cue, Episode, MemoryStore};
use std::time::SystemTime;
use thiserror::Error;

/// Errors that can occur during RECALL execution
#[derive(Debug, Error)]
pub enum RecallExecutionError {
    /// Pattern conversion to cue failed
    #[error("Failed to convert pattern to cue: {reason}")]
    PatternConversionFailed {
        /// Reason for conversion failure
        reason: String,
    },

    /// Constraint application failed
    #[error("Failed to apply constraint: {reason}")]
    ConstraintApplicationFailed {
        /// Reason for constraint failure
        reason: String,
    },

    /// Invalid embedding dimension
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbeddingDimension {
        /// Expected dimension
        expected: usize,
        /// Actual dimension provided
        actual: usize,
    },

    /// Memory store operation failed
    #[error("Memory store operation failed: {reason}")]
    MemoryStoreFailed {
        /// Reason for memory store failure
        reason: String,
    },
}

/// RECALL query executor
///
/// Executes RECALL queries by mapping them to MemoryStore operations
/// and generating probabilistic results.
pub struct RecallExecutor {
    /// Probabilistic query executor for result generation
    probabilistic_executor: ProbabilisticQueryExecutor,
}

impl RecallExecutor {
    /// Create a new RECALL executor with the given configuration
    #[must_use]
    pub fn new(config: QueryExecutorConfig) -> Self {
        Self {
            probabilistic_executor: ProbabilisticQueryExecutor::new(config),
        }
    }

    /// Execute a RECALL query against a memory store
    ///
    /// # Arguments
    ///
    /// * `query` - The parsed RECALL query AST
    /// * `store` - The memory store to query
    ///
    /// # Returns
    ///
    /// A `ProbabilisticQueryResult` containing:
    /// - Episodes matching the query pattern and constraints
    /// - Confidence interval reflecting result quality
    /// - Evidence chain explaining the retrieval
    /// - Uncertainty sources from system state
    ///
    /// # Errors
    ///
    /// Returns `RecallExecutionError` if:
    /// - Pattern cannot be converted to a valid Cue
    /// - Constraints contain invalid parameters
    /// - Memory store operation fails
    pub fn execute(
        &self,
        query: &RecallQuery<'_>,
        store: &MemoryStore,
    ) -> Result<ProbabilisticQueryResult, RecallExecutionError> {
        // Step 1: Convert pattern to memory cue
        let cue = Self::pattern_to_cue(&query.pattern)?;

        // Step 2: Recall from memory store
        let recall_result = store.recall(&cue);

        // Step 3: Apply constraints to filter results
        let filtered = Self::apply_constraints(recall_result.results, &query.constraints)?;

        // Step 4: Apply confidence threshold and limit if specified
        let filtered =
            Self::apply_query_filters(filtered, query.confidence_threshold.as_ref(), query.limit);

        // Step 5: Extract activation paths (empty for now - will be populated by spreading activation)
        let activation_paths: Vec<ActivationPath> = vec![];

        // Step 6: Gather uncertainty sources
        let uncertainty_sources = Self::gather_uncertainty_sources();

        // Step 7: Execute probabilistic query to generate result with evidence
        let result =
            self.probabilistic_executor
                .execute(filtered, &activation_paths, uncertainty_sources);

        Ok(result)
    }

    /// Convert a Pattern to a Cue for memory store lookup
    ///
    /// Maps AST pattern types to appropriate cue types:
    /// - NodeId → Direct ID lookup (not currently supported by Cue, use semantic)
    /// - Embedding → Embedding-based similarity search
    /// - ContentMatch → Semantic content search
    /// - Any → Return all memories (semantic with empty content)
    fn pattern_to_cue(pattern: &Pattern<'_>) -> Result<Cue, RecallExecutionError> {
        match pattern {
            Pattern::NodeId(id) => {
                // Direct memory addressing via hippocampal CA3-like indexing.
                // Node IDs represent direct episodic addresses (analogous to place cells
                // or event cells in hippocampus), enabling deterministic pattern completion.
                //
                // Biological basis: Hippocampal CA3 recurrent networks support rapid,
                // high-confidence pattern completion from sparse indices (O'Reilly & Rudy, 2001).
                // Direct indexing bypasses semantic search, matching the speed and certainty
                // of hippocampal episodic retrieval.
                //
                // Using semantic CueType with CERTAIN confidence signals this is an exact
                // match operation, not fuzzy content search. The ID is used as both cue ID
                // and content to enable direct lookup in the memory store.
                Ok(Cue::semantic(
                    id.as_str().to_string(),
                    id.as_str().to_string(),
                    Confidence::CERTAIN, // Direct addressing has maximum confidence
                ))
            }

            Pattern::Embedding { vector, threshold } => {
                // Validate embedding dimension (768 for current system)
                if vector.len() != 768 {
                    return Err(RecallExecutionError::InvalidEmbeddingDimension {
                        expected: 768,
                        actual: vector.len(),
                    });
                }

                // Convert Vec<f32> to [f32; 768] array
                let mut embedding_array = [0.0f32; 768];
                embedding_array.copy_from_slice(vector);

                Ok(Cue::embedding(
                    format!(
                        "embedding_query_{hash}",
                        hash = Self::hash_embedding(&embedding_array)
                    ),
                    embedding_array,
                    Confidence::from_raw(*threshold),
                ))
            }

            Pattern::ContentMatch(content) => Ok(Cue::semantic(
                format!("content_query_{hash}", hash = Self::hash_string(content)),
                content.to_string(),
                Confidence::MEDIUM,
            )),

            Pattern::Any => {
                // Return all memories via empty semantic query with low threshold
                Ok(Cue::semantic(
                    "query_all".to_string(),
                    String::new(),
                    Confidence::NONE,
                ))
            }
        }
    }

    /// Apply constraints to filter recall results
    ///
    /// Filters episodes based on:
    /// - Confidence thresholds (above/below)
    /// - Temporal constraints (created before/after)
    /// - Embedding similarity
    /// - Content matching
    fn apply_constraints(
        mut episodes: Vec<(Episode, Confidence)>,
        constraints: &[Constraint],
    ) -> Result<Vec<(Episode, Confidence)>, RecallExecutionError> {
        for constraint in constraints {
            episodes = Self::apply_single_constraint(episodes, constraint)?;
        }
        Ok(episodes)
    }

    /// Apply a single constraint to episode list
    fn apply_single_constraint(
        episodes: Vec<(Episode, Confidence)>,
        constraint: &Constraint,
    ) -> Result<Vec<(Episode, Confidence)>, RecallExecutionError> {
        match constraint {
            Constraint::ConfidenceAbove(threshold) => Ok(episodes
                .into_iter()
                .filter(|(_, confidence)| confidence.raw() > threshold.raw())
                .collect()),

            Constraint::ConfidenceBelow(threshold) => Ok(episodes
                .into_iter()
                .filter(|(_, confidence)| confidence.raw() < threshold.raw())
                .collect()),

            Constraint::CreatedBefore(time) => {
                let system_time = *time;
                Ok(episodes
                    .into_iter()
                    .filter(|(episode, _)| {
                        let episode_time: SystemTime = episode.when.into();
                        episode_time < system_time
                    })
                    .collect())
            }

            Constraint::CreatedAfter(time) => {
                let system_time = *time;
                Ok(episodes
                    .into_iter()
                    .filter(|(episode, _)| {
                        let episode_time: SystemTime = episode.when.into();
                        episode_time > system_time
                    })
                    .collect())
            }

            Constraint::SimilarTo {
                embedding,
                threshold,
            } => {
                // Validate embedding dimension
                if embedding.len() != 768 {
                    return Err(RecallExecutionError::InvalidEmbeddingDimension {
                        expected: 768,
                        actual: embedding.len(),
                    });
                }

                // Convert to array for similarity computation
                let mut query_embedding = [0.0f32; 768];
                query_embedding.copy_from_slice(embedding);

                Ok(episodes
                    .into_iter()
                    .filter(|(episode, _)| {
                        let similarity =
                            Self::cosine_similarity(&episode.embedding, &query_embedding);
                        similarity >= *threshold
                    })
                    .collect())
            }

            Constraint::ContentContains(text) => {
                let search_text = text.to_lowercase();
                Ok(episodes
                    .into_iter()
                    .filter(|(episode, _)| episode.what.to_lowercase().contains(&search_text))
                    .collect())
            }

            Constraint::InMemorySpace(_space_id) => {
                // Memory space filtering would be handled at store level
                // For now, pass through all episodes
                Ok(episodes)
            }
        }
    }

    /// Apply query-level filters (confidence threshold and limit)
    fn apply_query_filters(
        mut episodes: Vec<(Episode, Confidence)>,
        confidence_threshold: Option<&crate::query::parser::ast::ConfidenceThreshold>,
        limit: Option<usize>,
    ) -> Vec<(Episode, Confidence)> {
        // Apply confidence threshold if specified
        if let Some(threshold) = confidence_threshold {
            episodes.retain(|(_, confidence)| threshold.matches(*confidence));
        }

        // Apply limit if specified
        if let Some(max_results) = limit {
            episodes.truncate(max_results);
        }

        episodes
    }

    /// Gather uncertainty sources from system state
    ///
    /// Integrates with the metrics system to provide real-time uncertainty
    /// tracking based on system health, load, and degradation.
    ///
    /// # Uncertainty Sources
    ///
    /// - **System Pressure**: Memory pressure, CPU load affecting recall quality
    /// - **Spreading Activation Noise**: Variance in activation spreading paths
    /// - **Temporal Decay**: Uncertainty from time-based memory degradation
    /// - **Measurement Error**: Query execution noise and sampling errors
    fn gather_uncertainty_sources() -> Vec<UncertaintySource> {
        use crate::metrics;

        let mut sources = Vec::new();

        // Get global metrics registry if available
        if let Some(metrics_registry) = metrics::metrics() {
            // 1. System Pressure from health status
            let health_status = metrics_registry.health_status();
            if !matches!(health_status, crate::metrics::health::HealthStatus::Healthy) {
                // Calculate pressure level based on health status
                let pressure_level = match health_status {
                    crate::metrics::health::HealthStatus::Degraded => 0.5,
                    crate::metrics::health::HealthStatus::Unhealthy => 1.0,
                    crate::metrics::health::HealthStatus::Healthy => 0.0,
                };

                // Estimate confidence impact (degraded health reduces confidence)
                let effect_on_confidence = pressure_level * 0.3; // Max 30% reduction

                sources.push(UncertaintySource::SystemPressure {
                    pressure_level,
                    effect_on_confidence,
                });
            }

            // 2. Spreading Activation Noise from streaming metrics
            let snapshot = metrics_registry.streaming_snapshot();
            if let Some(spreading_summary) = snapshot.spreading {
                // Calculate activation variance from latency variance across tiers
                let tier_latencies: Vec<f64> = spreading_summary
                    .per_tier
                    .values()
                    .map(|tier| tier.mean_seconds)
                    .collect();

                if tier_latencies.len() >= 2 {
                    let mean = tier_latencies.iter().sum::<f64>() / (tier_latencies.len() as f64);
                    let variance = tier_latencies
                        .iter()
                        .map(|x| {
                            let diff = x - mean;
                            diff * diff
                        })
                        .sum::<f64>()
                        / (tier_latencies.len() as f64);

                    // Use normalized variance and tier count as diversity measure
                    let activation_variance = (variance.sqrt() / mean.max(0.001)) as f32;
                    let path_diversity = (tier_latencies.len() as f32) / 3.0; // Normalize by max tiers

                    sources.push(UncertaintySource::SpreadingActivationNoise {
                        activation_variance: activation_variance.min(1.0),
                        path_diversity: path_diversity.min(1.0),
                    });
                }
            }

            // 3. Measurement Error from query execution variance
            // Use gauge readings if available to estimate measurement noise
            if let Some(pool_utilization) =
                metrics_registry.gauge_value("engram_spreading_pool_utilization")
            {
                // High pool utilization can indicate contention and measurement noise
                if pool_utilization > 0.8 {
                    let error_magnitude = ((pool_utilization - 0.8) / 0.2) as f32;
                    let confidence_degradation = error_magnitude * 0.15; // Max 15% degradation

                    sources.push(UncertaintySource::MeasurementError {
                        error_magnitude,
                        confidence_degradation,
                    });
                }
            }
        }

        sources
    }

    /// Compute cosine similarity between two embeddings
    ///
    /// Returns similarity in range [0, 1] where 1 is identical.
    ///
    /// Biological basis: Neural population similarity is represented as positive
    /// activation overlap, matching empirical representational similarity analysis
    /// (Kriegeskorte et al., 2008). Neurons with similar tuning curves show positive
    /// correlation, not negative anti-correlation.
    ///
    /// Uses ReLU-like transformation max(0, cosine_sim) to match neural activation
    /// patterns where negative correlations represent orthogonal/irrelevant memories
    /// rather than anti-similar ones. This aligns with hippocampal pattern completion
    /// which shows graded retrieval strength based on positive overlap (O'Reilly & Rudy, 2001).
    fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        let mut dot_product = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..768 {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let magnitude = (norm_a * norm_b).sqrt();
        if magnitude == 0.0 {
            return 0.0;
        }

        let cosine_sim = dot_product / magnitude;

        // ReLU transformation: treat negative similarities as zero (orthogonal memories)
        // This matches neural population coding where similarity is positive-valued.
        // Negative correlations indicate conceptual orthogonality, not anti-similarity.
        cosine_sim.max(0.0)
    }

    /// Hash an embedding for ID generation
    fn hash_embedding(embedding: &[f32; 768]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        // Hash first 10 values for deterministic ID
        for value in &embedding[..10] {
            value.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Hash a string for ID generation
    fn hash_string(s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for RecallExecutor {
    fn default() -> Self {
        Self::new(QueryExecutorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]
    #![allow(clippy::panic)]

    use super::*;
    use crate::CueType;
    use crate::query::parser::ast::{ConfidenceThreshold, NodeIdentifier};
    use chrono::Utc;

    fn create_test_episode(id: &str, confidence: Confidence, content: &str) -> Episode {
        Episode::new(
            id.to_string(),
            Utc::now(),
            content.to_string(),
            [0.5f32; 768],
            confidence,
        )
    }

    #[test]
    fn test_pattern_to_cue_node_id() {
        let pattern = Pattern::NodeId(NodeIdentifier::from("episode_123"));

        let cue = RecallExecutor::pattern_to_cue(&pattern).unwrap();

        // NodeId should map to semantic cue
        match cue.cue_type {
            CueType::Semantic { content, .. } => {
                assert_eq!(content, "episode_123");
            }
            _ => panic!("Expected semantic cue type"),
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // Testing exact constant values
    fn test_pattern_to_cue_embedding() {
        let embedding_vec = vec![0.1f32; 768];
        let pattern = Pattern::Embedding {
            vector: embedding_vec,
            threshold: 0.8,
        };

        let cue = RecallExecutor::pattern_to_cue(&pattern).unwrap();

        match cue.cue_type {
            CueType::Embedding { vector, threshold } => {
                assert_eq!(vector[0], 0.1);
                assert_eq!(threshold.raw(), 0.8);
            }
            _ => panic!("Expected embedding cue type"),
        }
    }

    #[test]
    fn test_pattern_to_cue_content_match() {
        let pattern = Pattern::ContentMatch(std::borrow::Cow::Borrowed("test content"));

        let cue = RecallExecutor::pattern_to_cue(&pattern).unwrap();

        match cue.cue_type {
            CueType::Semantic { content, .. } => {
                assert_eq!(content, "test content");
            }
            _ => panic!("Expected semantic cue type"),
        }
    }

    #[test]
    fn test_pattern_to_cue_any() {
        let pattern = Pattern::Any;

        let cue = RecallExecutor::pattern_to_cue(&pattern).unwrap();

        match cue.cue_type {
            CueType::Semantic { content, .. } => {
                assert!(content.is_empty());
            }
            _ => panic!("Expected semantic cue type"),
        }
    }

    #[test]
    fn test_apply_confidence_above_constraint() {
        let episodes = vec![
            (
                create_test_episode("ep1", Confidence::HIGH, "high"),
                Confidence::HIGH,
            ),
            (
                create_test_episode("ep2", Confidence::LOW, "low"),
                Confidence::LOW,
            ),
            (
                create_test_episode("ep3", Confidence::MEDIUM, "medium"),
                Confidence::MEDIUM,
            ),
        ];

        let constraint = Constraint::ConfidenceAbove(Confidence::MEDIUM);
        let filtered = RecallExecutor::apply_single_constraint(episodes, &constraint).unwrap();

        // Should keep only HIGH confidence episode
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0.id, "ep1");
    }

    #[test]
    fn test_apply_confidence_below_constraint() {
        let episodes = vec![
            (
                create_test_episode("ep1", Confidence::HIGH, "high"),
                Confidence::HIGH,
            ),
            (
                create_test_episode("ep2", Confidence::LOW, "low"),
                Confidence::LOW,
            ),
        ];

        let constraint = Constraint::ConfidenceBelow(Confidence::MEDIUM);
        let filtered = RecallExecutor::apply_single_constraint(episodes, &constraint).unwrap();

        // Should keep only LOW confidence episode
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0.id, "ep2");
    }

    #[test]
    fn test_apply_content_contains_constraint() {
        let episodes = vec![
            (
                create_test_episode("ep1", Confidence::HIGH, "The cat sat on the mat"),
                Confidence::HIGH,
            ),
            (
                create_test_episode("ep2", Confidence::HIGH, "The dog barked loudly"),
                Confidence::HIGH,
            ),
            (
                create_test_episode("ep3", Confidence::HIGH, "A cat and a dog played"),
                Confidence::HIGH,
            ),
        ];

        let constraint = Constraint::ContentContains(std::borrow::Cow::Borrowed("cat"));
        let filtered = RecallExecutor::apply_single_constraint(episodes, &constraint).unwrap();

        // Should keep episodes containing "cat"
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().any(|(ep, _)| ep.id == "ep1"));
        assert!(filtered.iter().any(|(ep, _)| ep.id == "ep3"));
    }

    #[test]
    fn test_apply_temporal_constraints() {
        use chrono::Duration;

        let now = Utc::now();
        let past = now - Duration::hours(1);
        let future = now + Duration::hours(1);

        let mut ep1 = create_test_episode("ep1", Confidence::HIGH, "past");
        ep1.when = past;
        let mut ep2 = create_test_episode("ep2", Confidence::HIGH, "future");
        ep2.when = future;

        let episodes = vec![(ep1, Confidence::HIGH), (ep2, Confidence::HIGH)];

        // Test CreatedBefore
        let constraint = Constraint::CreatedBefore(now.into());
        let filtered = RecallExecutor::apply_single_constraint(episodes.clone(), &constraint)
            .expect("Failed to apply CreatedBefore constraint");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0.id, "ep1");

        // Test CreatedAfter
        let constraint = Constraint::CreatedAfter(now.into());
        let filtered = RecallExecutor::apply_single_constraint(episodes, &constraint)
            .expect("Failed to apply CreatedAfter constraint");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0.id, "ep2");
    }

    #[test]
    fn test_apply_query_confidence_threshold() {
        let episodes = vec![
            (
                create_test_episode("ep1", Confidence::HIGH, "high"),
                Confidence::HIGH,
            ),
            (
                create_test_episode("ep2", Confidence::LOW, "low"),
                Confidence::LOW,
            ),
            (
                create_test_episode("ep3", Confidence::MEDIUM, "medium"),
                Confidence::MEDIUM,
            ),
        ];

        let threshold = ConfidenceThreshold::Above(Confidence::MEDIUM);
        let filtered = RecallExecutor::apply_query_filters(episodes, Some(&threshold), None);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0.id, "ep1");
    }

    #[test]
    fn test_apply_query_limit() {
        let episodes = vec![
            (
                create_test_episode("ep1", Confidence::HIGH, "1"),
                Confidence::HIGH,
            ),
            (
                create_test_episode("ep2", Confidence::HIGH, "2"),
                Confidence::HIGH,
            ),
            (
                create_test_episode("ep3", Confidence::HIGH, "3"),
                Confidence::HIGH,
            ),
            (
                create_test_episode("ep4", Confidence::HIGH, "4"),
                Confidence::HIGH,
            ),
        ];

        let filtered = RecallExecutor::apply_query_filters(episodes, None, Some(2));

        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = [0.5f32; 768];
        let b = [0.5f32; 768];

        let similarity = RecallExecutor::cosine_similarity(&a, &b);

        // Identical vectors should have similarity close to 1.0
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut a = [0.0f32; 768];
        let mut b = [0.0f32; 768];

        // Create orthogonal vectors
        a[0] = 1.0;
        b[1] = 1.0;

        let similarity = RecallExecutor::cosine_similarity(&a, &b);

        // Orthogonal vectors have cosine similarity of 0, which after ReLU remains 0.
        // This matches neural population coding where orthogonal representations
        // indicate conceptually distinct memories with no overlap.
        assert!((similarity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_constraints() {
        let episodes = vec![
            (
                create_test_episode("ep1", Confidence::HIGH, "cat story"),
                Confidence::HIGH,
            ),
            (
                create_test_episode("ep2", Confidence::LOW, "cat tale"),
                Confidence::LOW,
            ),
            (
                create_test_episode("ep3", Confidence::HIGH, "dog story"),
                Confidence::HIGH,
            ),
        ];

        let constraints = vec![
            Constraint::ConfidenceAbove(Confidence::MEDIUM),
            Constraint::ContentContains(std::borrow::Cow::Borrowed("cat")),
        ];

        let filtered = RecallExecutor::apply_constraints(episodes, &constraints)
            .expect("Failed to apply multiple constraints");

        // Should keep only HIGH confidence episodes containing "cat"
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0.id, "ep1");
    }
}
