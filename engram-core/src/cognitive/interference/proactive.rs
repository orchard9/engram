//! Proactive interference detector and modeler
//!
//! Implements Underwood (1957) proactive interference dynamics with
//! exact boundary conditions from empirical research.
//!
//! ## Psychology Foundation
//!
//! Proactive interference occurs when old memories interfere with new learning.
//! The classic example is learning List A then List B, where recall of List B
//! is impaired by similar items from List A.
//!
//! **Key Findings (Underwood 1957):**
//! - Effect increases with number of prior lists learned
//! - Similarity between old and new material amplifies interference
//! - Retention interval modulates interference strength
//! - Benchmark: 20-30% accuracy reduction with 5+ prior similar lists
//!
//! ## Temporal Window Correction
//!
//! The temporal window is 6 hours (NOT 24 hours) to align with synaptic
//! consolidation timescale per Dudai et al. (2015). After consolidation,
//! memories shift from hippocampal to neocortical representations, reducing
//! interference (Complementary Learning Systems theory).

use crate::Confidence;
use crate::memory::Episode;
use chrono::Duration;

/// Proactive interference detector and modeler
///
/// Implements Underwood (1957) proactive interference dynamics with
/// exact boundary conditions from empirical research.
///
/// # Temporal Window
///
/// Uses 6-hour window (not 24h) to align with synaptic consolidation
/// boundary per Dudai et al. (2015) and Complementary Learning Systems
/// theory (McClelland et al. 1995).
pub struct ProactiveInterferenceDetector {
    /// Similarity threshold for interference (default: 0.7)
    similarity_threshold: f32,

    /// Temporal window for "prior" memories (default: 6 hours before)
    ///
    /// Empirical: Underwood (1957) session-based interference
    /// Justification: Synaptic consolidation (~6h) transitions memories from
    /// hippocampal to neocortical representations, reducing interference
    prior_memory_window: Duration,

    /// Interference strength per similar prior item (default: 0.05 = 5%)
    interference_per_item: f32,

    /// Maximum interference effect (default: 0.30 = 30% retrieval reduction)
    max_interference: f32,
}

impl Default for ProactiveInterferenceDetector {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            prior_memory_window: Duration::hours(6), // CORRECTED from 24h
            interference_per_item: 0.05,
            max_interference: 0.30,
        }
    }
}

impl ProactiveInterferenceDetector {
    /// Create detector with custom parameters
    #[must_use]
    pub const fn new(
        similarity_threshold: f32,
        prior_memory_window: Duration,
        interference_per_item: f32,
        max_interference: f32,
    ) -> Self {
        Self {
            similarity_threshold,
            prior_memory_window,
            interference_per_item,
            max_interference,
        }
    }

    /// Get similarity threshold
    #[must_use]
    pub const fn similarity_threshold(&self) -> f32 {
        self.similarity_threshold
    }

    /// Get prior memory window duration
    #[must_use]
    pub const fn prior_memory_window(&self) -> Duration {
        self.prior_memory_window
    }

    /// Get interference per item coefficient
    #[must_use]
    pub const fn interference_per_item(&self) -> f32 {
        self.interference_per_item
    }

    /// Get maximum interference cap
    #[must_use]
    pub const fn max_interference(&self) -> f32 {
        self.max_interference
    }

    /// Detect proactive interference for a new episode
    ///
    /// Searches for similar prior memories and computes interference magnitude
    /// based on Underwood (1957) linear accumulation model.
    ///
    /// # Returns
    ///
    /// Interference magnitude in [0, max_interference]
    #[must_use]
    pub fn detect_interference(
        &self,
        new_episode: &Episode,
        prior_episodes: &[Episode],
    ) -> ProactiveInterferenceResult {
        // Find similar prior memories within temporal window
        let interfering_episodes: Vec<&Episode> = prior_episodes
            .iter()
            .filter(|ep| self.is_interfering(new_episode, ep))
            .collect();

        let interference_count = interfering_episodes.len();

        // Compute interference magnitude (linear in similar items)
        #[allow(clippy::cast_precision_loss)]
        let magnitude =
            (interference_count as f32 * self.interference_per_item).min(self.max_interference);

        // Record metrics
        #[cfg(feature = "monitoring")]
        {
            if let Some(metrics) = crate::metrics::cognitive_patterns::cognitive_patterns() {
                metrics.record_interference(
                    crate::metrics::cognitive_patterns::InterferenceType::Proactive,
                    magnitude,
                );
            }
        }

        ProactiveInterferenceResult {
            magnitude,
            interfering_episodes: interfering_episodes
                .into_iter()
                .map(|ep| ep.id.clone())
                .collect(),
            count: interference_count,
        }
    }

    /// Check if prior episode interferes with new episode
    ///
    /// Three requirements:
    /// 1. Temporally prior to new episode
    /// 2. Within temporal window (6 hours before)
    /// 3. Semantically similar (≥ threshold)
    fn is_interfering(&self, new_episode: &Episode, prior_episode: &Episode) -> bool {
        // Must be temporally prior
        if prior_episode.when >= new_episode.when {
            return false;
        }

        // Must be within temporal window (6 hours - CORRECTED)
        let time_diff = new_episode.when - prior_episode.when;
        if time_diff > self.prior_memory_window {
            return false;
        }

        // Must be sufficiently similar (semantic overlap)
        let similarity = cosine_similarity(&new_episode.embedding, &prior_episode.embedding);

        similarity >= self.similarity_threshold
    }

    /// Apply proactive interference to retrieval confidence
    ///
    /// Reduces confidence based on interference magnitude
    #[must_use]
    pub fn apply_interference(
        base_confidence: Confidence,
        interference: &ProactiveInterferenceResult,
    ) -> Confidence {
        let reduction_factor = 1.0 - interference.magnitude;
        Confidence::from_raw(base_confidence.raw() * reduction_factor)
    }
}

/// Result of proactive interference detection
#[derive(Debug, Clone)]
pub struct ProactiveInterferenceResult {
    /// Magnitude of interference in [0, max_interference]
    pub magnitude: f32,

    /// Episode IDs of interfering prior memories
    pub interfering_episodes: Vec<String>,

    /// Count of interfering episodes
    pub count: usize,
}

impl ProactiveInterferenceResult {
    /// Check if interference is significant (>10% magnitude)
    #[must_use]
    pub const fn is_significant(&self) -> bool {
        self.magnitude > 0.10
    }

    /// Predicted accuracy reduction percentage
    #[must_use]
    pub const fn accuracy_reduction_percent(&self) -> f32 {
        self.magnitude * 100.0
    }
}

/// Compute cosine similarity between two embeddings
///
/// This is a standalone implementation to avoid circular dependencies
/// with the graph module.
fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    let mut dot_product = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..768 {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let norm_product = (norm_a * norm_b).sqrt();
    if norm_product == 0.0 {
        return 0.0;
    }

    (dot_product / norm_product).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_episode_with_embedding(
        id: &str,
        embedding: &[f32; 768],
        when: chrono::DateTime<Utc>,
    ) -> Episode {
        Episode {
            id: id.to_string(),
            when,
            where_location: None,
            who: None,
            what: "test episode".to_string(),
            embedding: *embedding,
            embedding_provenance: None,
            encoding_confidence: Confidence::HIGH,
            vividness_confidence: Confidence::HIGH,
            reliability_confidence: Confidence::HIGH,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 0.05,
            decay_function: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    fn create_similar_embedding(base_value: f32) -> [f32; 768] {
        let mut embedding = [0.0f32; 768];
        for (i, item) in embedding.iter_mut().enumerate() {
            *item = base_value + (i as f32 * 0.001);
        }
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut embedding {
            *val /= norm;
        }
        embedding
    }

    #[test]
    fn test_default_parameters() {
        let detector = ProactiveInterferenceDetector::default();
        assert!((detector.similarity_threshold() - 0.7).abs() < f32::EPSILON);
        assert_eq!(detector.prior_memory_window(), Duration::hours(6));
        assert!((detector.interference_per_item() - 0.05).abs() < f32::EPSILON);
        assert!((detector.max_interference() - 0.30).abs() < f32::EPSILON);
    }

    #[test]
    fn test_temporal_direction_enforcement() {
        let detector = ProactiveInterferenceDetector::default();
        let now = Utc::now();

        let new_episode = create_episode_with_embedding("new", &create_similar_embedding(1.0), now);

        let prior_episode = create_episode_with_embedding(
            "prior",
            &create_similar_embedding(1.0),
            now - Duration::hours(1),
        );

        // Prior should interfere with new
        assert!(detector.is_interfering(&new_episode, &prior_episode));

        // New should NOT interfere with prior (wrong direction)
        assert!(!detector.is_interfering(&prior_episode, &new_episode));
    }

    #[test]
    fn test_temporal_window_6_hours() {
        let detector = ProactiveInterferenceDetector::default();
        let now = Utc::now();

        let new_episode = create_episode_with_embedding("new", &create_similar_embedding(1.0), now);

        // Within window (3 hours ago)
        let recent_prior = create_episode_with_embedding(
            "recent",
            &create_similar_embedding(1.0),
            now - Duration::hours(3),
        );

        // Outside window (8 hours ago)
        let old_prior = create_episode_with_embedding(
            "old",
            &create_similar_embedding(1.0),
            now - Duration::hours(8),
        );

        assert!(
            detector.is_interfering(&new_episode, &recent_prior),
            "Should interfere within 6h window"
        );
        assert!(
            !detector.is_interfering(&new_episode, &old_prior),
            "Should NOT interfere outside 6h window"
        );
    }

    #[test]
    fn test_linear_accumulation() {
        let detector = ProactiveInterferenceDetector::default();
        let now = Utc::now();

        let new_episode = create_episode_with_embedding("new", &create_similar_embedding(1.0), now);

        // Test with varying numbers of prior episodes
        for num_prior in 0..=10 {
            let prior_episodes: Vec<Episode> = (0..num_prior)
                .map(|i| {
                    create_episode_with_embedding(
                        &format!("prior_{i}"),
                        &create_similar_embedding(1.0),
                        now - Duration::hours(3),
                    )
                })
                .collect();

            let result = detector.detect_interference(&new_episode, &prior_episodes);

            #[allow(clippy::cast_precision_loss)]
            let expected = (num_prior as f32 * 0.05).min(0.30);
            assert!(
                (result.magnitude - expected).abs() < f32::EPSILON,
                "Interference should scale linearly: {num_prior} prior → {expected}% interference"
            );
            assert_eq!(result.count, num_prior);
        }
    }

    #[test]
    fn test_similarity_threshold() {
        let detector = ProactiveInterferenceDetector::default();
        let now = Utc::now();

        let new_episode = create_episode_with_embedding("new", &create_similar_embedding(1.0), now);

        // Similar episode (same embedding pattern)
        let similar_prior = create_episode_with_embedding(
            "similar",
            &create_similar_embedding(1.0),
            now - Duration::hours(1),
        );

        // Dissimilar episode (different embedding pattern)
        let dissimilar_prior = create_episode_with_embedding(
            "dissimilar",
            &create_similar_embedding(-1.0),
            now - Duration::hours(1),
        );

        let result_similar = detector.detect_interference(&new_episode, &[similar_prior]);
        let result_dissimilar = detector.detect_interference(&new_episode, &[dissimilar_prior]);

        assert!(
            result_similar.magnitude > 0.0,
            "Similar episodes should interfere"
        );
        assert!(
            result_dissimilar.magnitude.abs() < f32::EPSILON,
            "Dissimilar episodes should NOT interfere"
        );
    }

    #[test]
    fn test_apply_interference_to_confidence() {
        let base_confidence = Confidence::exact(0.9);

        // 25% interference
        let interference = ProactiveInterferenceResult {
            magnitude: 0.25,
            interfering_episodes: vec!["prior1".to_string(), "prior2".to_string()],
            count: 5,
        };

        let adjusted =
            ProactiveInterferenceDetector::apply_interference(base_confidence, &interference);

        // 0.9 * (1 - 0.25) = 0.9 * 0.75 = 0.675
        assert!((adjusted.raw() - 0.675).abs() < 0.001);
    }

    #[test]
    fn test_interference_result_helpers() {
        let result = ProactiveInterferenceResult {
            magnitude: 0.25,
            interfering_episodes: vec!["ep1".to_string(), "ep2".to_string()],
            count: 5,
        };

        assert!(result.is_significant());
        assert!((result.accuracy_reduction_percent() - 25.0).abs() < f32::EPSILON);

        let insignificant = ProactiveInterferenceResult {
            magnitude: 0.05,
            interfering_episodes: vec![],
            count: 1,
        };

        assert!(!insignificant.is_significant());
    }

    #[test]
    fn test_cosine_similarity_computation() {
        // Identical embeddings
        let a = create_similar_embedding(1.0);
        let b = create_similar_embedding(1.0);
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "Identical embeddings should have similarity ~1.0"
        );

        // Orthogonal-ish embeddings
        let c = create_similar_embedding(-1.0);
        let sim2 = cosine_similarity(&a, &c);
        assert!(
            sim2 < 0.5,
            "Dissimilar embeddings should have low similarity"
        );
    }
}
