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
//! Pattern-to-cue conversion mirrors hippocampal pattern separation where
//! query patterns are transformed into sparse memory indices. Constraint
//! application reflects neocortical filtering where retrieved memories are
//! evaluated against contextual expectations. The probabilistic executor
//! implements confidence calibration analogous to metacognitive monitoring
//! in human recall.

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

    /// Create a new RECALL executor with default configuration
    #[must_use]
    pub fn default() -> Self {
        Self::new(QueryExecutorConfig::default())
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
        query: RecallQuery<'_>,
        store: &MemoryStore,
    ) -> Result<ProbabilisticQueryResult, RecallExecutionError> {
        // Step 1: Convert pattern to memory cue
        let cue = self.pattern_to_cue(&query.pattern)?;

        // Step 2: Recall from memory store
        let recall_result = store.recall(&cue);

        // Step 3: Apply constraints to filter results
        let filtered = self.apply_constraints(recall_result.results, &query.constraints)?;

        // Step 4: Apply confidence threshold and limit if specified
        let filtered =
            self.apply_query_filters(filtered, query.confidence_threshold.as_ref(), query.limit);

        // Step 5: Extract activation paths (empty for now - will be populated by spreading activation)
        let activation_paths: Vec<ActivationPath> = vec![];

        // Step 6: Gather uncertainty sources
        let uncertainty_sources = self.gather_uncertainty_sources();

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
    fn pattern_to_cue<'a>(&self, pattern: &Pattern<'a>) -> Result<Cue, RecallExecutionError> {
        match pattern {
            Pattern::NodeId(id) => {
                // Convert node ID to semantic cue for content-based lookup
                // This approximates ID lookup via semantic matching
                Ok(Cue::semantic(
                    id.as_str().to_string(),
                    id.as_str().to_string(),
                    Confidence::HIGH,
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
                    format!("embedding_query_{}", Self::hash_embedding(&embedding_array)),
                    embedding_array,
                    Confidence::from_raw(*threshold),
                ))
            }

            Pattern::ContentMatch(content) => Ok(Cue::semantic(
                format!("content_query_{}", Self::hash_string(content)),
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
        &self,
        mut episodes: Vec<(Episode, Confidence)>,
        constraints: &[Constraint],
    ) -> Result<Vec<(Episode, Confidence)>, RecallExecutionError> {
        for constraint in constraints {
            episodes = self.apply_single_constraint(episodes, constraint)?;
        }
        Ok(episodes)
    }

    /// Apply a single constraint to episode list
    fn apply_single_constraint(
        &self,
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
                let system_time = SystemTime::from(*time);
                Ok(episodes
                    .into_iter()
                    .filter(|(episode, _)| {
                        let episode_time: SystemTime = episode.when.into();
                        episode_time < system_time
                    })
                    .collect())
            }

            Constraint::CreatedAfter(time) => {
                let system_time = SystemTime::from(*time);
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
        &self,
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
    /// Currently returns empty - would be populated with:
    /// - System memory pressure
    /// - Query load
    /// - Store degradation metrics
    fn gather_uncertainty_sources(&self) -> Vec<UncertaintySource> {
        // TODO: Integrate with system metrics to gather real uncertainty sources
        vec![]
    }

    /// Compute cosine similarity between two embeddings
    ///
    /// Returns similarity in range [0, 1] where 1 is identical
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

        // Map from [-1, 1] to [0, 1] for consistency with threshold expectations
        ((dot_product / magnitude) + 1.0) / 2.0
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

#[cfg(test)]
mod tests {
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
        let executor = RecallExecutor::default();
        let pattern = Pattern::NodeId(NodeIdentifier::from("episode_123"));

        let cue = executor.pattern_to_cue(&pattern).unwrap();

        // NodeId should map to semantic cue
        match cue.cue_type {
            CueType::Semantic { content, .. } => {
                assert_eq!(content, "episode_123");
            }
            _ => panic!("Expected semantic cue type"),
        }
    }

    #[test]
    fn test_pattern_to_cue_embedding() {
        let executor = RecallExecutor::default();
        let embedding_vec = vec![0.1f32; 768];
        let pattern = Pattern::Embedding {
            vector: embedding_vec,
            threshold: 0.8,
        };

        let cue = executor.pattern_to_cue(&pattern).unwrap();

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
        let executor = RecallExecutor::default();
        let pattern = Pattern::ContentMatch(std::borrow::Cow::Borrowed("test content"));

        let cue = executor.pattern_to_cue(&pattern).unwrap();

        match cue.cue_type {
            CueType::Semantic { content, .. } => {
                assert_eq!(content, "test content");
            }
            _ => panic!("Expected semantic cue type"),
        }
    }

    #[test]
    fn test_pattern_to_cue_any() {
        let executor = RecallExecutor::default();
        let pattern = Pattern::Any;

        let cue = executor.pattern_to_cue(&pattern).unwrap();

        match cue.cue_type {
            CueType::Semantic { content, .. } => {
                assert!(content.is_empty());
            }
            _ => panic!("Expected semantic cue type"),
        }
    }

    #[test]
    fn test_apply_confidence_above_constraint() {
        let executor = RecallExecutor::default();
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
        let filtered = executor
            .apply_single_constraint(episodes, &constraint)
            .unwrap();

        // Should keep only HIGH confidence episode
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0.id, "ep1");
    }

    #[test]
    fn test_apply_confidence_below_constraint() {
        let executor = RecallExecutor::default();
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
        let filtered = executor
            .apply_single_constraint(episodes, &constraint)
            .unwrap();

        // Should keep only LOW confidence episode
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0.id, "ep2");
    }

    #[test]
    fn test_apply_content_contains_constraint() {
        let executor = RecallExecutor::default();
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
        let filtered = executor
            .apply_single_constraint(episodes, &constraint)
            .unwrap();

        // Should keep episodes containing "cat"
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().any(|(ep, _)| ep.id == "ep1"));
        assert!(filtered.iter().any(|(ep, _)| ep.id == "ep3"));
    }

    #[test]
    fn test_apply_temporal_constraints() {
        use chrono::Duration;

        let executor = RecallExecutor::default();

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
        let filtered = executor
            .apply_single_constraint(episodes.clone(), &constraint)
            .expect("Failed to apply CreatedBefore constraint");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0.id, "ep1");

        // Test CreatedAfter
        let constraint = Constraint::CreatedAfter(now.into());
        let filtered = executor
            .apply_single_constraint(episodes, &constraint)
            .expect("Failed to apply CreatedAfter constraint");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0.id, "ep2");
    }

    #[test]
    fn test_apply_query_confidence_threshold() {
        let executor = RecallExecutor::default();
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
        let filtered = executor.apply_query_filters(episodes, Some(&threshold), None);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0.id, "ep1");
    }

    #[test]
    fn test_apply_query_limit() {
        let executor = RecallExecutor::default();
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

        let filtered = executor.apply_query_filters(episodes, None, Some(2));

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

        // Orthogonal vectors should map to 0.5 in [0,1] range
        assert!((similarity - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_constraints() {
        let executor = RecallExecutor::default();
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

        let filtered = executor
            .apply_constraints(episodes, &constraints)
            .expect("Failed to apply multiple constraints");

        // Should keep only HIGH confidence episodes containing "cat"
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0.id, "ep1");
    }
}
