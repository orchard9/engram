//! Alternative hypothesis generation for preventing confabulation
//!
//! Implements System 2 reasoning (Kahneman, 2011) by generating multiple
//! alternative completions with diversity constraints. Prevents single-path
//! confabulation through metacognitive monitoring (Koriat & Goldsmith, 1996).

use crate::completion::{PartialEpisode, RankedPattern};
use crate::compute::cosine_similarity_768;
use crate::{Confidence, Episode};

/// Generates alternative completion hypotheses to prevent confabulation
///
/// Uses System 2 reasoning to review System 1 completions by varying
/// pattern weights and ensuring diversity among alternatives.
pub struct AlternativeHypothesisGenerator {
    /// Number of hypotheses to generate (default: 3)
    num_hypotheses: usize,

    /// Minimum similarity distance between hypotheses (default: 0.3)
    #[allow(dead_code)]
    // Used in ensure_diversity, which is tested but not yet used in production
    diversity_threshold: f32,

    /// Random seed for reproducible variation
    #[allow(dead_code)]
    // Used in pseudo_random, which is tested but not yet used in production
    seed: u64,
}

impl AlternativeHypothesisGenerator {
    /// Create new hypothesis generator with default parameters
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_hypotheses: 3,
            diversity_threshold: 0.3,
            seed: 42,
        }
    }

    /// Create generator with custom parameters
    #[must_use]
    pub const fn with_params(num_hypotheses: usize, diversity_threshold: f32, seed: u64) -> Self {
        Self {
            num_hypotheses,
            diversity_threshold,
            seed,
        }
    }

    /// Generate alternative completions with diverse pattern weights
    ///
    /// # Arguments
    ///
    /// * `partial` - Partial episode to complete
    /// * `primary_completion` - Primary completion from System 1 (CA3)
    /// * `ranked_patterns` - Semantic patterns for field reconstruction
    ///
    /// # Returns
    ///
    /// Vector of (Episode, Confidence) alternatives, including primary
    #[must_use]
    pub fn generate_alternatives(
        &self,
        _partial: &PartialEpisode,
        primary_completion: &Episode,
        _ranked_patterns: &[RankedPattern],
    ) -> Vec<(Episode, Confidence)> {
        let mut alternatives = Vec::with_capacity(self.num_hypotheses);

        // Always include primary completion as first alternative
        alternatives.push((
            primary_completion.clone(),
            primary_completion.encoding_confidence,
        ));

        // Generate additional alternatives by varying pattern weights
        // Placeholder: Would vary weights and regenerate completions
        // For now, return just primary to satisfy type constraints

        alternatives
    }

    /// Vary pattern weights to produce diverse completions
    ///
    /// Uses pseudo-random perturbations to pattern relevance scores
    /// to explore alternative completion pathways.
    ///
    /// # Arguments
    ///
    /// * `base_weights` - Original pattern relevance weights
    /// * `variation` - Variation index (0 to num_hypotheses-1)
    ///
    /// # Returns
    ///
    /// Perturbed weights for alternative pattern selection
    #[must_use]
    pub fn vary_pattern_weights(&self, base_weights: &[f32], variation: usize) -> Vec<f32> {
        let mut varied = Vec::with_capacity(base_weights.len());

        // Simple deterministic variation based on seed + variation index
        for (idx, &weight) in base_weights.iter().enumerate() {
            // Pseudo-random perturbation [-0.2, +0.2]
            let perturbation = self.pseudo_random(idx, variation) * 0.4 - 0.2;
            let new_weight = (weight + perturbation).clamp(0.0, 1.0);
            varied.push(new_weight);
        }

        varied
    }

    /// Ensure hypotheses are diverse (minimum similarity distance)
    ///
    /// Filters alternatives to maintain >diversity_threshold embedding distance.
    /// Prevents near-duplicate completions that don't provide additional coverage.
    ///
    /// # Arguments
    ///
    /// * `hypotheses` - Candidate alternative completions
    ///
    /// # Returns
    ///
    /// Filtered alternatives meeting diversity constraints
    #[must_use]
    pub fn ensure_diversity(
        &self,
        hypotheses: Vec<(Episode, Confidence)>,
    ) -> Vec<(Episode, Confidence)> {
        if hypotheses.len() <= 1 {
            return hypotheses;
        }

        let mut diverse = Vec::with_capacity(hypotheses.len());

        // Always include first (primary) hypothesis
        diverse.push(hypotheses[0].clone());

        for candidate in hypotheses.iter().skip(1) {
            // Check diversity against all accepted hypotheses
            let is_diverse = diverse.iter().all(|(accepted, _)| {
                let similarity = cosine_similarity_768(&candidate.0.embedding, &accepted.embedding);
                let distance = 1.0 - similarity;
                distance >= self.diversity_threshold
            });

            if is_diverse {
                diverse.push(candidate.clone());
            }

            // Stop if we have enough diverse hypotheses
            if diverse.len() >= self.num_hypotheses {
                break;
            }
        }

        diverse
    }

    /// Compute embedding similarity between two episodes
    ///
    /// Uses cosine similarity to measure distance in semantic space.
    #[must_use]
    pub fn compute_similarity(episode1: &Episode, episode2: &Episode) -> f32 {
        cosine_similarity_768(&episode1.embedding, &episode2.embedding)
    }

    /// Check if alternative coverage meets target threshold
    ///
    /// Validates that ground truth appears in top-N alternatives.
    /// Target: >70% of time (Acceptance Criterion from Task 005).
    ///
    /// # Arguments
    ///
    /// * `ground_truth` - Actual episode
    /// * `alternatives` - Generated hypotheses
    /// * `threshold` - Similarity threshold for match (default: 0.8)
    ///
    /// # Returns
    ///
    /// True if ground truth is sufficiently similar to any alternative
    #[must_use]
    pub fn contains_ground_truth(
        ground_truth: &Episode,
        alternatives: &[(Episode, Confidence)],
        threshold: f32,
    ) -> bool {
        alternatives.iter().any(|(alt, _)| {
            let similarity = Self::compute_similarity(ground_truth, alt);
            similarity >= threshold
        })
    }

    /// Simple pseudo-random number generator for reproducible variation
    ///
    /// Uses linear congruential generator for deterministic perturbations.
    #[allow(dead_code)] // Tested but not yet used in production - planned for future use
    fn pseudo_random(&self, idx: usize, variation: usize) -> f32 {
        #[allow(clippy::cast_possible_truncation)]
        let combined = self
            .seed
            .wrapping_mul(1_103_515_245)
            .wrapping_add((idx as u64).wrapping_mul(2_654_435_761))
            .wrapping_add((variation as u64).wrapping_mul(1_013_904_223))
            .wrapping_add(12_345);

        let x = (combined % (1_u64 << 31)) as u32;

        #[allow(clippy::cast_precision_loss)]
        (x as f32 / (1_u32 << 31) as f32)
    }
}

impl Default for AlternativeHypothesisGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    // Test helper to create episode
    fn create_test_episode(id: &str, embedding: &[f32; 768], what: &str) -> Episode {
        Episode {
            id: id.to_string(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: what.to_string(),
            embedding: *embedding,
            embedding_provenance: None,
            encoding_confidence: Confidence::exact(0.9),
            vividness_confidence: Confidence::exact(0.8),
            reliability_confidence: Confidence::exact(0.85),
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 0.05,
            decay_function: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_generator_creation() {
        let generator = AlternativeHypothesisGenerator::new();

        assert_eq!(generator.num_hypotheses, 3);
        assert!((generator.diversity_threshold - 0.3).abs() < 1e-6);
        assert_eq!(generator.seed, 42);
    }

    #[test]
    fn test_generator_custom_params() {
        let generator = AlternativeHypothesisGenerator::with_params(5, 0.4, 123);

        assert_eq!(generator.num_hypotheses, 5);
        assert!((generator.diversity_threshold - 0.4).abs() < 1e-6);
        assert_eq!(generator.seed, 123);
    }

    #[test]
    fn test_vary_pattern_weights() {
        let generator = AlternativeHypothesisGenerator::new();

        let base_weights = vec![0.5, 0.6, 0.7, 0.8];
        let varied = generator.vary_pattern_weights(&base_weights, 0);

        // Should have same length
        assert_eq!(varied.len(), base_weights.len());

        // All weights should be in valid range [0, 1]
        for &weight in &varied {
            assert!((0.0..=1.0).contains(&weight));
        }

        // Weights should be perturbed (not identical)
        assert_ne!(varied, base_weights);
    }

    #[test]
    fn test_vary_pattern_weights_deterministic() {
        let generator = AlternativeHypothesisGenerator::new();

        let base_weights = vec![0.5, 0.6, 0.7];

        // Same variation should produce same result
        let varied1 = generator.vary_pattern_weights(&base_weights, 0);
        let varied2 = generator.vary_pattern_weights(&base_weights, 0);

        assert_eq!(varied1, varied2);

        // Different variations should differ
        let varied3 = generator.vary_pattern_weights(&base_weights, 1);
        assert_ne!(varied1, varied3);
    }

    #[test]
    fn test_compute_similarity_identical() {
        let episode1 = create_test_episode("1", &[1.0; 768], "coffee");
        let episode2 = create_test_episode("2", &[1.0; 768], "coffee");

        let similarity = AlternativeHypothesisGenerator::compute_similarity(&episode1, &episode2);

        // Identical embeddings should have similarity ~1.0
        assert!((similarity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_similarity_different() {
        // Create embeddings with actual variation
        let mut emb1 = [0.0; 768];
        let mut emb2 = [0.0; 768];
        for i in 0..768 {
            emb1[i] = (i as f32 / 768.0).sin();
            emb2[i] = (i as f32 / 768.0).cos();
        }

        let episode1 = create_test_episode("1", &emb1, "coffee");
        let episode2 = create_test_episode("2", &emb2, "tea");

        let similarity = AlternativeHypothesisGenerator::compute_similarity(&episode1, &episode2);

        // Different embeddings should have similarity <1.0
        assert!(similarity < 1.0);
        assert!(similarity > -1.0);
    }

    #[test]
    fn test_ensure_diversity_empty() {
        let generator = AlternativeHypothesisGenerator::new();

        let hypotheses = vec![];
        let diverse = generator.ensure_diversity(hypotheses);

        assert!(diverse.is_empty());
    }

    #[test]
    fn test_ensure_diversity_single() {
        let generator = AlternativeHypothesisGenerator::new();

        let episode = create_test_episode("1", &[1.0; 768], "coffee");
        let hypotheses = vec![(episode, Confidence::exact(0.9))];

        let diverse = generator.ensure_diversity(hypotheses.clone());

        assert_eq!(diverse.len(), 1);
        assert_eq!(diverse[0].0.id, hypotheses[0].0.id);
    }

    #[test]
    fn test_ensure_diversity_filters_similar() {
        let generator = AlternativeHypothesisGenerator::new();

        // Create actually diverse embeddings
        let mut emb1 = [0.0; 768];
        let mut emb2 = [0.0; 768];
        let mut emb3 = [0.0; 768];
        for i in 0..768 {
            emb1[i] = (i as f32).sin();
            emb2[i] = (i as f32 + 0.1).sin(); // Very similar to emb1
            emb3[i] = -(i as f32).sin(); // Opposite of emb1 (diverse)
        }

        let episode1 = create_test_episode("1", &emb1, "coffee");
        let episode2 = create_test_episode("2", &emb2, "coffee"); // Very similar
        let episode3 = create_test_episode("3", &emb3, "tea"); // Diverse

        let hypotheses = vec![
            (episode1, Confidence::exact(0.9)),
            (episode2, Confidence::exact(0.85)), // Should be filtered (too similar)
            (episode3, Confidence::exact(0.8)),  // Should be included (diverse)
        ];

        let diverse = generator.ensure_diversity(hypotheses);

        // Should include episode1 and episode3, filter out episode2
        assert_eq!(diverse.len(), 2);
        assert_eq!(diverse[0].0.id, "1");
        assert_eq!(diverse[1].0.id, "3");
    }

    #[test]
    fn test_contains_ground_truth_match() {
        let ground_truth = create_test_episode("truth", &[1.0; 768], "coffee");
        let alt1 = create_test_episode("alt1", &[0.95; 768], "coffee");
        let alt2 = create_test_episode("alt2", &[0.5; 768], "tea");

        let alternatives = vec![
            (alt1, Confidence::exact(0.9)),
            (alt2, Confidence::exact(0.8)),
        ];

        assert!(AlternativeHypothesisGenerator::contains_ground_truth(
            &ground_truth,
            &alternatives,
            0.8
        ));
    }

    #[test]
    fn test_contains_ground_truth_no_match() {
        // Create truly different embeddings
        let mut truth_emb = [0.0; 768];
        let mut alt1_emb = [0.0; 768];
        let mut alt2_emb = [0.0; 768];
        for i in 0..768 {
            truth_emb[i] = (i as f32).sin();
            alt1_emb[i] = -(i as f32).cos(); // Orthogonal
            alt2_emb[i] = (i as f32 + std::f32::consts::PI).sin(); // Phase shifted
        }

        let ground_truth = create_test_episode("truth", &truth_emb, "coffee");
        let alt1 = create_test_episode("alt1", &alt1_emb, "tea");
        let alt2 = create_test_episode("alt2", &alt2_emb, "water");

        let alternatives = vec![
            (alt1, Confidence::exact(0.9)),
            (alt2, Confidence::exact(0.8)),
        ];

        assert!(!AlternativeHypothesisGenerator::contains_ground_truth(
            &ground_truth,
            &alternatives,
            0.8
        ));
    }

    #[test]
    fn test_generate_alternatives_includes_primary() {
        let generator = AlternativeHypothesisGenerator::new();

        let primary = create_test_episode("primary", &[1.0; 768], "coffee");

        let partial = PartialEpisode {
            known_fields: std::collections::HashMap::new(),
            partial_embedding: vec![None; 768],
            cue_strength: Confidence::exact(0.7),
            temporal_context: vec![],
        };

        let ranked_patterns = vec![];

        let alternatives = generator.generate_alternatives(&partial, &primary, &ranked_patterns);

        // Should always include primary completion
        assert!(!alternatives.is_empty());
        assert_eq!(alternatives[0].0.id, "primary");
    }

    #[test]
    fn test_pseudo_random_reproducible() {
        let generator = AlternativeHypothesisGenerator::new();

        let val1 = generator.pseudo_random(0, 0);
        let val2 = generator.pseudo_random(0, 0);

        assert!((val1 - val2).abs() < 1e-10);
    }

    #[test]
    fn test_pseudo_random_different_inputs() {
        let generator = AlternativeHypothesisGenerator::new();

        let val1 = generator.pseudo_random(0, 0);
        let val2 = generator.pseudo_random(1, 0);
        let val3 = generator.pseudo_random(0, 1);

        assert!((val1 - val2).abs() > 1e-6);
        assert!((val1 - val3).abs() > 1e-6);
        assert!((val2 - val3).abs() > 1e-6);
    }

    #[test]
    fn test_pseudo_random_range() {
        let generator = AlternativeHypothesisGenerator::new();

        for idx in 0..100 {
            for var in 0..10 {
                let val = generator.pseudo_random(idx, var);
                assert!((0.0..=1.0).contains(&val));
            }
        }
    }
}
