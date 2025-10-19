//! Storage compaction for safe episode replacement with semantic memories.
//!
//! This module implements the storage compaction system that safely replaces
//! consolidated episodes with semantic patterns while maintaining retrieval
//! capability through a 5-phase verification process.

use crate::completion::consolidation::SemanticPattern;
use crate::error::{CognitiveError, ErrorContext};
use crate::{Confidence, Episode};
use std::time::Duration;

/// Configuration for storage compaction operations
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Minimum age for episode consolidation (episodes younger than this won't be compacted)
    pub min_episode_age: Duration,
    /// Preserve high-confidence episodes above this threshold (won't be compacted)
    pub preserve_threshold: f32,
    /// Verify retrieval before and after compaction
    pub verify_retrieval: bool,
    /// Minimum similarity threshold for reconstruction verification (default: 0.8)
    pub reconstruction_threshold: f32,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            min_episode_age: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            preserve_threshold: 0.95,                               // Preserve very high confidence
            verify_retrieval: true,
            reconstruction_threshold: 0.8,
        }
    }
}

/// Storage compactor for safe episode replacement
pub struct StorageCompactor {
    /// Configuration for compaction operations
    config: CompactionConfig,
}

impl StorageCompactor {
    /// Create a new storage compactor with the given configuration
    #[must_use]
    pub const fn new(config: CompactionConfig) -> Self {
        Self { config }
    }

    /// Create a storage compactor with default configuration
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            config: CompactionConfig::default(),
        }
    }

    /// Verify semantic memory can reconstruct episodes with acceptable similarity
    ///
    /// This is Phase 1 of the compaction process - ensuring no information loss
    /// before any destructive operations occur.
    ///
    /// # Errors
    ///
    /// Returns `CompactionError::ReconstructionFailed` if any episode cannot be
    /// reconstructed with similarity above the configured threshold.
    pub fn verify_reconstruction(
        &self,
        episodes: &[Episode],
        semantic_memory: &SemanticPattern,
    ) -> Result<(), Box<CognitiveError>> {
        for episode in episodes {
            let similarity =
                Self::embedding_similarity(&episode.embedding, &semantic_memory.embedding);

            if similarity < self.config.reconstruction_threshold {
                return Err(Box::new(CognitiveError::new(
                    format!(
                        "Reconstruction verification failed for episode {}",
                        episode.id
                    ),
                    ErrorContext::new(
                        format!("similarity >= {:.3}", self.config.reconstruction_threshold),
                        format!("similarity {similarity:.3}"),
                    ),
                    "Adjust reconstruction threshold or review semantic memory quality",
                    format!(
                        "Expected similarity above {:.3}, got {:.3}",
                        self.config.reconstruction_threshold, similarity
                    ),
                    Confidence::HIGH,
                )));
            }
        }
        Ok(())
    }

    /// Compute cosine similarity between two embeddings
    fn embedding_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Compute storage reduction in bytes from replacing episodes with semantic memory
    #[must_use]
    pub const fn compute_storage_reduction(
        episodes: &[Episode],
        _semantic_memory: &SemanticPattern,
    ) -> u64 {
        let before = std::mem::size_of_val(episodes) as u64;
        let after = std::mem::size_of::<SemanticPattern>() as u64;
        before.saturating_sub(after)
    }

    /// Check if an episode is eligible for compaction based on age and confidence
    #[must_use]
    pub fn is_episode_eligible(
        &self,
        episode: &Episode,
        now: chrono::DateTime<chrono::Utc>,
    ) -> bool {
        // Check age criterion
        let age = now - episode.when;
        let age_eligible = age.num_seconds() >= self.config.min_episode_age.as_secs() as i64;

        // Check confidence criterion (don't compact very high confidence episodes)
        let confidence_eligible =
            episode.encoding_confidence.raw() < self.config.preserve_threshold;

        age_eligible && confidence_eligible
    }

    /// Perform complete storage compaction with 5-phase verification
    ///
    /// This method implements the full compaction process:
    /// - Phase 1: Verify reconstruction quality (semantic memory can reconstruct episodes)
    /// - Phase 2: Compute storage metrics
    /// - Phases 3-5: Caller is responsible for storage mutations
    ///
    /// # Errors
    ///
    /// Returns error if reconstruction verification fails (Phase 1)
    ///
    /// # Implementation Note
    ///
    /// This method performs non-destructive verification only. The caller must handle:
    /// - Phase 3: Mark episodes as consolidated (soft delete)
    /// - Phase 4: Verify retrieval still works
    /// - Phase 5: Remove consolidated episodes (hard delete)
    pub fn compact_storage(
        &self,
        episodes: &[Episode],
        semantic_memory: &SemanticPattern,
    ) -> Result<CompactionResult, Box<CognitiveError>> {
        // Phase 1: Verify reconstruction before any modifications
        self.verify_reconstruction(episodes, semantic_memory)?;

        // Phase 2: Compute storage metrics
        let storage_reduction_bytes = Self::compute_storage_reduction(episodes, semantic_memory);

        // Calculate average similarity for observability
        let total_similarity: f32 = episodes
            .iter()
            .map(|ep| Self::embedding_similarity(&ep.embedding, &semantic_memory.embedding))
            .sum();
        let average_similarity = if episodes.is_empty() {
            0.0
        } else {
            total_similarity / episodes.len() as f32
        };

        Ok(CompactionResult {
            episodes_removed: episodes.len(),
            semantic_memory_id: semantic_memory.id.clone(),
            storage_reduction_bytes,
            average_similarity,
        })
    }
}

/// Result of a storage compaction operation
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Number of episodes successfully removed
    pub episodes_removed: usize,
    /// ID of the semantic memory that replaced the episodes
    pub semantic_memory_id: String,
    /// Storage space saved in bytes
    pub storage_reduction_bytes: u64,
    /// Average similarity score during reconstruction verification
    pub average_similarity: f32,
}

impl CompactionResult {
    /// Calculate reduction ratio (0.0 = no reduction, 1.0 = complete removal)
    #[must_use]
    pub fn reduction_ratio(&self, original_storage_bytes: u64) -> f32 {
        if original_storage_bytes == 0 {
            return 0.0;
        }
        self.storage_reduction_bytes as f32 / original_storage_bytes as f32
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Tests are allowed to use unwrap
    #![allow(clippy::float_cmp)] // Tests use exact float comparisons

    use super::*;
    use crate::Confidence;
    use chrono::Utc;

    fn create_test_episode(id: &str, embedding: &[f32; 768], age_days: i64) -> Episode {
        let when = Utc::now() - chrono::Duration::days(age_days);
        Episode::new(
            id.to_string(),
            when,
            format!("Test episode {id}"),
            *embedding,
            Confidence::exact(0.8),
        )
    }

    fn create_test_semantic_pattern(id: &str, embedding: &[f32; 768]) -> SemanticPattern {
        SemanticPattern {
            id: id.to_string(),
            embedding: *embedding,
            source_episodes: vec!["ep1".to_string(), "ep2".to_string()],
            strength: 0.9,
            schema_confidence: Confidence::exact(0.85),
            last_consolidated: Utc::now(),
        }
    }

    #[test]
    fn test_embedding_similarity_identical() {
        let a = [1.0; 768];
        let b = [1.0; 768];
        let similarity = StorageCompactor::embedding_similarity(&a, &b);
        assert!(
            (similarity - 1.0).abs() < 1e-5,
            "Identical embeddings should have similarity ~1.0"
        );
    }

    #[test]
    fn test_embedding_similarity_orthogonal() {
        let mut a = [0.0; 768];
        let mut b = [0.0; 768];
        a[0] = 1.0;
        b[1] = 1.0;
        let similarity = StorageCompactor::embedding_similarity(&a, &b);
        assert!(
            similarity.abs() < 1e-5,
            "Orthogonal embeddings should have similarity ~0.0"
        );
    }

    #[test]
    fn test_verify_reconstruction_success() {
        let compactor = StorageCompactor::default_config();
        let embedding = [0.5; 768];

        let episodes = vec![
            create_test_episode("ep1", &embedding, 10),
            create_test_episode("ep2", &embedding, 12),
        ];

        let semantic_memory = create_test_semantic_pattern("pattern1", &embedding);

        let result = compactor.verify_reconstruction(&episodes, &semantic_memory);
        assert!(
            result.is_ok(),
            "Should verify reconstruction successfully with identical embeddings"
        );
    }

    #[test]
    fn test_verify_reconstruction_failure() {
        let compactor = StorageCompactor::default_config();
        let episode_embedding = [1.0; 768];
        let semantic_embedding = [0.0; 768];

        let episodes = vec![create_test_episode("ep1", &episode_embedding, 10)];
        let semantic_memory = create_test_semantic_pattern("pattern1", &semantic_embedding);

        let result = compactor.verify_reconstruction(&episodes, &semantic_memory);
        assert!(
            result.is_err(),
            "Should fail verification with dissimilar embeddings"
        );
    }

    #[test]
    fn test_episode_eligibility_by_age() {
        let config = CompactionConfig {
            min_episode_age: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            preserve_threshold: 0.95,
            verify_retrieval: true,
            reconstruction_threshold: 0.8,
        };
        let compactor = StorageCompactor::new(config);
        let now = Utc::now();

        // Old episode (10 days) - should be eligible
        let old_episode = create_test_episode("old", &[0.5; 768], 10);
        assert!(
            compactor.is_episode_eligible(&old_episode, now),
            "Episode older than min_age should be eligible"
        );

        // Recent episode (5 days) - should not be eligible
        let recent_episode = create_test_episode("recent", &[0.5; 768], 5);
        assert!(
            !compactor.is_episode_eligible(&recent_episode, now),
            "Episode younger than min_age should not be eligible"
        );
    }

    #[test]
    fn test_episode_eligibility_by_confidence() {
        let compactor = StorageCompactor::default_config();
        let now = Utc::now();
        let when = now - chrono::Duration::days(10);

        // Low confidence episode - should be eligible
        let low_conf = Episode::new(
            "low".to_string(),
            when,
            "Low confidence".to_string(),
            [0.5; 768],
            Confidence::exact(0.7), // Below preserve threshold
        );
        assert!(
            compactor.is_episode_eligible(&low_conf, now),
            "Low confidence episode should be eligible"
        );

        // High confidence episode - should not be eligible
        let high_conf = Episode::new(
            "high".to_string(),
            when,
            "High confidence".to_string(),
            [0.5; 768],
            Confidence::exact(0.98), // Above preserve threshold
        );
        assert!(
            !compactor.is_episode_eligible(&high_conf, now),
            "High confidence episode should be preserved"
        );
    }

    #[test]
    fn test_storage_reduction_computation() {
        let episodes = vec![
            create_test_episode("ep1", &[0.5; 768], 10),
            create_test_episode("ep2", &[0.5; 768], 11),
            create_test_episode("ep3", &[0.5; 768], 12),
        ];
        let semantic_memory = create_test_semantic_pattern("pattern1", &[0.5; 768]);

        let reduction = StorageCompactor::compute_storage_reduction(&episodes, &semantic_memory);

        // Should save space by replacing 3 episodes with 1 pattern
        assert!(reduction > 0, "Should have positive storage reduction");
    }

    #[test]
    fn test_compaction_result_reduction_ratio() {
        let result = CompactionResult {
            episodes_removed: 10,
            semantic_memory_id: "pattern1".to_string(),
            storage_reduction_bytes: 5000,
            average_similarity: 0.92,
        };

        let ratio = result.reduction_ratio(10000);
        assert!(
            (ratio - 0.5).abs() < 1e-5,
            "Should calculate 50% reduction ratio"
        );

        let zero_ratio = result.reduction_ratio(0);
        assert!(
            zero_ratio.abs() < 1e-5,
            "Should handle zero original storage gracefully"
        );
    }

    #[test]
    fn test_compact_storage_success() {
        let compactor = StorageCompactor::default_config();
        let embedding = [0.5; 768];

        let episodes = vec![
            create_test_episode("ep1", &embedding, 10),
            create_test_episode("ep2", &embedding, 12),
            create_test_episode("ep3", &embedding, 14),
        ];

        let semantic_memory = create_test_semantic_pattern("pattern1", &embedding);

        let result = compactor.compact_storage(&episodes, &semantic_memory);
        assert!(
            result.is_ok(),
            "Compaction should succeed with good similarity"
        );

        let compaction_result = result.unwrap();
        assert_eq!(
            compaction_result.episodes_removed, 3,
            "Should report 3 episodes removed"
        );
        assert_eq!(
            compaction_result.semantic_memory_id, "pattern1",
            "Should report correct semantic memory ID"
        );
        assert!(
            compaction_result.average_similarity > 0.9,
            "Average similarity should be high for identical embeddings"
        );
        assert!(
            compaction_result.storage_reduction_bytes > 0,
            "Should report positive storage reduction"
        );
    }

    #[test]
    fn test_compact_storage_fails_on_poor_reconstruction() {
        let compactor = StorageCompactor::default_config();
        let episode_embedding = [1.0; 768];
        let semantic_embedding = [0.0; 768];

        let episodes = vec![create_test_episode("ep1", &episode_embedding, 10)];
        let semantic_memory = create_test_semantic_pattern("pattern1", &semantic_embedding);

        let result = compactor.compact_storage(&episodes, &semantic_memory);
        assert!(
            result.is_err(),
            "Compaction should fail when reconstruction similarity is too low"
        );
    }

    #[test]
    fn test_compact_storage_empty_episodes() {
        let compactor = StorageCompactor::default_config();
        let embedding = [0.5; 768];
        let episodes: Vec<Episode> = vec![];
        let semantic_memory = create_test_semantic_pattern("pattern1", &embedding);

        let result = compactor.compact_storage(&episodes, &semantic_memory);
        assert!(
            result.is_ok(),
            "Should handle empty episode list gracefully"
        );

        let compaction_result = result.unwrap();
        assert_eq!(
            compaction_result.episodes_removed, 0,
            "Should report 0 episodes removed"
        );
        assert_eq!(
            compaction_result.average_similarity, 0.0,
            "Average similarity should be 0 for empty list"
        );
    }

    #[test]
    fn test_compact_storage_calculates_correct_average_similarity() {
        let compactor = StorageCompactor::default_config();

        // Create semantic embedding
        let mut semantic_embedding = [0.0; 768];
        for (i, val) in semantic_embedding.iter_mut().enumerate() {
            *val = (i as f32) / 768.0; // Increasing values
        }

        // Create episodes with varying similarity to semantic pattern
        let ep1_embedding = semantic_embedding;
        let mut ep2_embedding = [0.0; 768];
        for (i, val) in ep2_embedding.iter_mut().enumerate() {
            *val = ((i + 100) as f32) / 768.0; // Shifted pattern
        }

        let episodes = vec![
            create_test_episode("ep1", &ep1_embedding, 10), // High similarity
            create_test_episode("ep2", &ep2_embedding, 11), // Lower similarity
        ];

        let semantic_memory = create_test_semantic_pattern("pattern1", &semantic_embedding);

        let result = compactor.compact_storage(&episodes, &semantic_memory);
        assert!(result.is_ok(), "Compaction should succeed");

        let compaction_result = result.unwrap();
        // Average should be above reconstruction threshold (0.8)
        assert!(
            compaction_result.average_similarity > 0.8,
            "Average similarity should be above threshold, got {}",
            compaction_result.average_similarity
        );
    }
}
