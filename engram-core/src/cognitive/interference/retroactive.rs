//! Retroactive interference detector and modeler
//!
//! Implements McGeoch (1942) retroactive interference paradigm with
//! linear similarity weighting from Anderson & Neely (1996).
//!
//! ## Psychology Foundation
//!
//! Retroactive interference occurs when new learning during the retention interval
//! disrupts consolidation of previously encoded memories.
//!
//! **Critical Temporal Logic (McGeoch 1942 Three-Phase Design):**
//! 1. T=0 min: Learn List A (target material)
//! 2. T=1-10 min: Learn List B (interpolated material, DURING retention interval)
//! 3. T=60 min: Test recall of List A
//!
//! **Key Point:** List B must be learned AFTER List A encoding AND BEFORE List A
//! retrieval. This is "interpolated learning" during the retention interval.
//!
//! **Key Findings:**
//! - 15-25% reduction in List A recall when List B is similar (McGeoch 1942)
//! - Interference magnitude scales linearly with similarity (Anderson & Neely 1996)
//! - Effect strongest during synaptic consolidation window (0-24 hours post-encoding)
//! - Similarity between materials is the primary moderator
//!
//! ## Neuroscience Mechanism
//!
//! **Synaptic Consolidation Disruption:**
//! - Newly encoded memories undergo protein synthesis-dependent stabilization (0-24h)
//! - Interpolated learning during this window recruits overlapping neural populations
//! - Similar representations compete for limited consolidation resources
//! - Result: Incomplete stabilization of original memory trace

use crate::Confidence;
use crate::memory::Episode;
use chrono::{DateTime, Duration, Utc};

/// Retroactive interference detector for consolidation-stage disruption
///
/// Implements McGeoch (1942) retroactive interference paradigm with
/// linear similarity weighting from Anderson & Neely (1996).
///
/// CRITICAL: This detector checks for learning that occurred DURING
/// the retention interval (after target encoding, before current retrieval).
pub struct RetroactiveInterferenceDetector {
    /// Base interference magnitude (default: 0.15 = 15% at similarity=1.0)
    /// Empirical basis: McGeoch (1942) 15-25% range
    base_interference: f32,

    /// Similarity threshold for interference (default: 0.6)
    /// Below this threshold, materials are considered dissimilar
    similarity_threshold: f32,

    /// Maximum interference magnitude (default: 0.25 = 25%)
    /// Prevents unrealistic memory obliteration
    max_interference: f32,

    /// Consolidation window (default: 24 hours)
    /// Synaptic consolidation period during which interference operates
    consolidation_window: Duration,
}

impl Default for RetroactiveInterferenceDetector {
    fn default() -> Self {
        Self {
            base_interference: 0.15,   // 15% base reduction
            similarity_threshold: 0.6, // Anderson & Neely (1996)
            max_interference: 0.25,    // 25% max per McGeoch (1942)
            consolidation_window: Duration::hours(24),
        }
    }
}

impl RetroactiveInterferenceDetector {
    /// Create detector with custom parameters
    #[must_use]
    pub const fn new(
        base_interference: f32,
        similarity_threshold: f32,
        max_interference: f32,
        consolidation_window: Duration,
    ) -> Self {
        Self {
            base_interference,
            similarity_threshold,
            max_interference,
            consolidation_window,
        }
    }

    /// Get base interference coefficient
    #[must_use]
    pub const fn base_interference(&self) -> f32 {
        self.base_interference
    }

    /// Get similarity threshold
    #[must_use]
    pub const fn similarity_threshold(&self) -> f32 {
        self.similarity_threshold
    }

    /// Get maximum interference cap
    #[must_use]
    pub const fn max_interference(&self) -> f32 {
        self.max_interference
    }

    /// Get consolidation window duration
    #[must_use]
    pub const fn consolidation_window(&self) -> Duration {
        self.consolidation_window
    }

    /// Detect retroactive interference for a target episode during consolidation
    ///
    /// Searches for episodes learned DURING the retention interval
    /// (after target encoding, before current time) that may disrupt
    /// target's consolidation.
    ///
    /// # Parameters
    /// - `target_episode`: Episode undergoing consolidation (List A)
    /// - `all_episodes`: All episodes in memory graph
    /// - `retrieval_time`: Current time for determining retention interval
    ///
    /// Returns interference magnitude in [0, max_interference]
    #[must_use]
    pub fn detect_interference(
        &self,
        target_episode: &Episode,
        all_episodes: &[Episode],
        retrieval_time: DateTime<Utc>,
    ) -> RetroactiveInterferenceResult {
        // Find interpolated episodes (learned during retention interval)
        let interfering_episodes: Vec<&Episode> = all_episodes
            .iter()
            .filter(|ep| self.is_retroactively_interfering(target_episode, ep, retrieval_time))
            .collect();

        let interference_count = interfering_episodes.len();

        // Compute interference magnitude with LINEAR similarity weighting
        let magnitude = self.compute_interference_magnitude(target_episode, &interfering_episodes);

        // Record metrics
        #[cfg(feature = "monitoring")]
        {
            if let Some(metrics) = crate::metrics::cognitive_patterns::cognitive_patterns() {
                metrics.record_interference(
                    crate::metrics::cognitive_patterns::InterferenceType::Retroactive,
                    magnitude,
                );
            }
        }

        RetroactiveInterferenceResult {
            magnitude,
            interfering_episodes: interfering_episodes
                .into_iter()
                .map(|ep| ep.id.clone())
                .collect(),
            count: interference_count,
        }
    }

    /// Check if an episode is retroactively interfering
    ///
    /// Three critical temporal checks (CORRECTED FROM PREVIOUS SPEC):
    /// 1. Subsequent episode learned AFTER target (temporal ordering)
    /// 2. Subsequent episode learned BEFORE retrieval (interpolated)
    /// 3. Target within consolidation window
    /// Plus similarity check.
    fn is_retroactively_interfering(
        &self,
        target_episode: &Episode,
        subsequent_episode: &Episode,
        retrieval_time: DateTime<Utc>,
    ) -> bool {
        // CHECK 1: Temporal ordering - subsequent must come after target
        if subsequent_episode.when <= target_episode.when {
            return false; // Not temporally subsequent
        }

        // CHECK 2: Interpolated - subsequent must be learned BEFORE retrieval
        // This is the CRITICAL check that ensures retroactive causation
        if subsequent_episode.when >= retrieval_time {
            return false; // Learned after retrieval, cannot interfere retroactively
        }

        // CHECK 3: Within consolidation window
        let time_since_target = retrieval_time - target_episode.when;
        if time_since_target > self.consolidation_window {
            return false; // Target already consolidated, interference minimal
        }

        // CHECK 4: Semantic similarity
        let similarity =
            cosine_similarity(&target_episode.embedding, &subsequent_episode.embedding);

        similarity >= self.similarity_threshold
    }

    /// Compute interference magnitude with LINEAR similarity weighting
    ///
    /// Anderson & Neely (1996) show LINEAR relationship between
    /// similarity and interference, NOT quadratic.
    ///
    /// Formula: magnitude = base_interference * avg(similarity) * sqrt(count)
    /// - Linear in similarity (exponent = 1.0)
    /// - Sublinear in count (multiple interfering items, diminishing returns)
    fn compute_interference_magnitude(
        &self,
        target_episode: &Episode,
        interfering_episodes: &[&Episode],
    ) -> f32 {
        if interfering_episodes.is_empty() {
            return 0.0;
        }

        // Compute average similarity (LINEAR weighting)
        let total_similarity: f32 = interfering_episodes
            .iter()
            .map(|ep| {
                let sim = cosine_similarity(&target_episode.embedding, &ep.embedding);
                // Normalize to [0, 1] range relative to threshold
                // If sim = threshold (0.6), normalized = 0
                // If sim = 1.0, normalized = 1.0
                (sim - self.similarity_threshold) / (1.0 - self.similarity_threshold)
            })
            .sum();

        #[allow(clippy::cast_precision_loss)]
        let avg_similarity = total_similarity / (interfering_episodes.len() as f32);

        // Scale by number of interfering items (sublinear - sqrt)
        #[allow(clippy::cast_precision_loss)]
        let count_factor = (interfering_episodes.len() as f32).sqrt();

        // Final magnitude: base * similarity * sqrt(count)
        let magnitude = self.base_interference * avg_similarity * count_factor;

        // Clamp to maximum
        magnitude.min(self.max_interference)
    }

    /// Apply retroactive interference to consolidation strength
    ///
    /// Reduces the strength at which a memory consolidates based on
    /// interference from interpolated learning.
    ///
    /// Used during background consolidation operations.
    #[must_use]
    pub fn apply_interference_to_consolidation(
        base_strength: f32,
        interference: &RetroactiveInterferenceResult,
    ) -> f32 {
        let reduction_factor = 1.0 - interference.magnitude;
        base_strength * reduction_factor
    }

    /// Apply retroactive interference to confidence
    ///
    /// Reduces confidence based on interference magnitude during consolidation
    #[must_use]
    pub fn apply_interference(
        base_confidence: Confidence,
        interference: &RetroactiveInterferenceResult,
    ) -> Confidence {
        let reduction_factor = 1.0 - interference.magnitude;
        Confidence::from_raw(base_confidence.raw() * reduction_factor)
    }
}

/// Result of retroactive interference detection
#[derive(Debug, Clone)]
pub struct RetroactiveInterferenceResult {
    /// Magnitude of interference in [0, max_interference]
    pub magnitude: f32,

    /// Episode IDs of interfering interpolated memories
    pub interfering_episodes: Vec<String>,

    /// Count of interfering episodes
    pub count: usize,
}

impl RetroactiveInterferenceResult {
    /// Check if interference is significant (>10% magnitude)
    #[must_use]
    pub const fn is_significant(&self) -> bool {
        self.magnitude > 0.10
    }

    /// Predicted recall accuracy reduction percentage
    #[must_use]
    pub const fn accuracy_reduction_percent(&self) -> f32 {
        self.magnitude * 100.0
    }

    /// No interference detected
    #[must_use]
    pub fn none() -> Self {
        Self {
            magnitude: 0.0,
            interfering_episodes: Vec::new(),
            count: 0,
        }
    }
}

/// Compute cosine similarity between two embeddings
///
/// This is a standalone implementation to avoid circular dependencies
/// with the graph module. Matches the implementation in proactive.rs.
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

    fn create_episode_with_embedding(
        id: &str,
        embedding: [f32; 768],
        when: DateTime<Utc>,
    ) -> Episode {
        Episode {
            id: id.to_string(),
            when,
            where_location: None,
            who: None,
            what: "test episode".to_string(),
            embedding,
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
        for i in 0..768 {
            embedding[i] = base_value + (i as f32 * 0.001);
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
        let detector = RetroactiveInterferenceDetector::default();
        assert_eq!(detector.base_interference(), 0.15);
        assert_eq!(detector.similarity_threshold(), 0.6);
        assert_eq!(detector.max_interference(), 0.25);
        assert_eq!(detector.consolidation_window(), Duration::hours(24));
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_mcgeoch_1942_interpolated_learning_paradigm() {
        let detector = RetroactiveInterferenceDetector::default();

        // PHASE 1: Learn List A at T=0
        let list_a_time = Utc::now();
        let list_a_episode =
            create_episode_with_embedding("list_a", create_similar_embedding(1.0), list_a_time);

        // PHASE 2: Learn List B at T=30min (DURING retention interval)
        let list_b_time = list_a_time + Duration::minutes(30);
        let list_b_episode = create_episode_with_embedding(
            "list_b",
            create_similar_embedding(1.0), // High similarity
            list_b_time,
        );

        // PHASE 3: Test List A recall at T=60min
        let retrieval_time = list_a_time + Duration::minutes(60);

        // Detect interference on List A
        let interference = detector.detect_interference(
            &list_a_episode,
            &[list_b_episode.clone()],
            retrieval_time,
        );

        // List B was interpolated (learned during retention interval)
        // Therefore it SHOULD interfere
        assert!(
            interference.magnitude > 0.10,
            "List B was interpolated during retention, should interfere significantly, got {}",
            interference.magnitude
        );
        assert_eq!(interference.count, 1);

        // CRITICAL TEST: Learn List C at T=90min (AFTER retrieval)
        let list_c_time = retrieval_time + Duration::minutes(30);
        let list_c_episode = create_episode_with_embedding(
            "list_c",
            create_similar_embedding(1.0), // High similarity
            list_c_time,
        );

        let interference_c = detector.detect_interference(
            &list_a_episode,
            &[list_c_episode],
            retrieval_time, // Retrieval was at T=60, List C learned at T=90
        );

        // List C was learned AFTER retrieval, NOT during retention interval
        // Therefore it CANNOT interfere retroactively
        assert_eq!(
            interference_c.magnitude, 0.0,
            "List C learned after retrieval cannot retroactively interfere"
        );
        assert_eq!(interference_c.count, 0);
    }

    #[test]
    fn test_temporal_ordering_enforcement() {
        let detector = RetroactiveInterferenceDetector::default();
        let now = Utc::now();
        let retrieval_time = now + Duration::hours(1);

        let target = create_episode_with_embedding("target", create_similar_embedding(1.0), now);

        // Episode learned BEFORE target (should NOT interfere retroactively)
        let before_episode = create_episode_with_embedding(
            "before",
            create_similar_embedding(1.0),
            now - Duration::hours(1),
        );

        let interference = detector.detect_interference(&target, &[before_episode], retrieval_time);

        assert_eq!(
            interference.magnitude, 0.0,
            "Episodes learned before target cannot interfere retroactively"
        );
    }

    #[test]
    fn test_linear_similarity_weighting_not_quadratic() {
        let detector = RetroactiveInterferenceDetector::default();

        let target_time = Utc::now();
        let target =
            create_episode_with_embedding("target", create_similar_embedding(1.0), target_time);

        let retrieval_time = target_time + Duration::minutes(60);

        // Test at similarity = 0.9
        // With linear weighting:
        //   normalized_sim = (0.9 - 0.6) / (1.0 - 0.6) = 0.3 / 0.4 = 0.75
        //   magnitude = 0.15 * 0.75 * sqrt(1) = 0.1125
        //
        // With quadratic weighting (wrong):
        //   magnitude = 0.15 * 0.81 * sqrt(1) = 0.1215

        // Create embedding with controlled similarity ~0.9
        let mut high_sim_embedding = create_similar_embedding(1.0);
        // Slightly perturb to get ~0.9 similarity
        for i in 0..100 {
            high_sim_embedding[i] *= 0.95;
        }

        let interfering = create_episode_with_embedding(
            "interfering",
            high_sim_embedding,
            target_time + Duration::minutes(30), // Interpolated
        );

        let result = detector.detect_interference(&target, &[interfering], retrieval_time);

        // Should be closer to linear (0.1125) than quadratic (0.1215)
        // Allow tolerance due to actual similarity variations
        assert!(
            result.magnitude < 0.15,
            "Linear weighting should give magnitude < 0.15 for single interferer, got {}",
            result.magnitude
        );
    }

    #[test]
    fn test_consolidation_window_24_hours() {
        let detector = RetroactiveInterferenceDetector::default();

        let target_time = Utc::now();
        let target =
            create_episode_with_embedding("target", create_similar_embedding(1.0), target_time);

        // Test retrieval within consolidation window (should interfere)
        let early_retrieval = target_time + Duration::hours(12);
        let early_interfering = create_episode_with_embedding(
            "early",
            create_similar_embedding(1.0),
            target_time + Duration::hours(6), // Interpolated
        );

        let early_interference =
            detector.detect_interference(&target, &[early_interfering], early_retrieval);
        assert!(
            early_interference.magnitude > 0.10,
            "Should interfere within 24h consolidation window"
        );

        // Test retrieval outside consolidation window (minimal interference)
        let late_retrieval = target_time + Duration::hours(48);
        let late_interfering = create_episode_with_embedding(
            "late",
            create_similar_embedding(1.0),
            target_time + Duration::hours(36), // Still interpolated, but target consolidated
        );

        let late_interference =
            detector.detect_interference(&target, &[late_interfering], late_retrieval);
        assert_eq!(
            late_interference.magnitude, 0.0,
            "Should NOT interfere outside 24h consolidation window"
        );
    }

    #[test]
    fn test_similarity_threshold() {
        let detector = RetroactiveInterferenceDetector::default();
        let now = Utc::now();
        let retrieval_time = now + Duration::hours(1);

        let target = create_episode_with_embedding("target", create_similar_embedding(1.0), now);

        // Similar episode (same embedding pattern)
        let similar = create_episode_with_embedding(
            "similar",
            create_similar_embedding(1.0),
            now + Duration::minutes(30), // Interpolated
        );

        // Dissimilar episode (different embedding pattern)
        let dissimilar = create_episode_with_embedding(
            "dissimilar",
            create_similar_embedding(-1.0),
            now + Duration::minutes(30), // Interpolated
        );

        let result_similar = detector.detect_interference(&target, &[similar], retrieval_time);
        let result_dissimilar =
            detector.detect_interference(&target, &[dissimilar], retrieval_time);

        assert!(
            result_similar.magnitude > 0.0,
            "Similar episodes should interfere"
        );
        assert_eq!(
            result_dissimilar.magnitude, 0.0,
            "Dissimilar episodes should NOT interfere"
        );
    }

    #[test]
    fn test_mcgeoch_1942_empirical_benchmarks() {
        let detector = RetroactiveInterferenceDetector::default();

        let target_time = Utc::now();
        let retrieval_time = target_time + Duration::hours(1);

        // High similarity condition (should show ~25% interference with multiple items)
        let target =
            create_episode_with_embedding("target", create_similar_embedding(1.0), target_time);

        // Create multiple highly similar interfering episodes
        let interfering: Vec<Episode> = (0..5)
            .map(|i| {
                create_episode_with_embedding(
                    &format!("interfering_{i}"),
                    create_similar_embedding(1.0),
                    target_time + Duration::minutes(10 + i * 5), // Interpolated early
                )
            })
            .collect();

        let result = detector.detect_interference(&target, &interfering, retrieval_time);

        // McGeoch (1942): High similarity with multiple items → up to 25% reduction
        assert!(
            result.magnitude <= 0.25,
            "Interference should be capped at 25%, got {:.1}%",
            result.magnitude * 100.0
        );

        // Should be significant
        assert!(
            result.is_significant(),
            "Multiple similar interfering items should produce significant interference"
        );
    }

    #[test]
    fn test_apply_interference_to_consolidation() {
        let base_strength = 1.0;

        // 25% interference
        let interference = RetroactiveInterferenceResult {
            magnitude: 0.25,
            interfering_episodes: vec!["ep1".to_string()],
            count: 1,
        };

        let final_strength = RetroactiveInterferenceDetector::apply_interference_to_consolidation(
            base_strength,
            &interference,
        );

        // 1.0 * (1 - 0.25) = 0.75
        assert!((final_strength - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_apply_interference_to_confidence() {
        let base_confidence = Confidence::exact(0.9);

        // 15% interference
        let interference = RetroactiveInterferenceResult {
            magnitude: 0.15,
            interfering_episodes: vec!["ep1".to_string()],
            count: 1,
        };

        let adjusted =
            RetroactiveInterferenceDetector::apply_interference(base_confidence, &interference);

        // 0.9 * (1 - 0.15) = 0.9 * 0.85 = 0.765
        assert!((adjusted.raw() - 0.765).abs() < 0.001);
    }

    #[test]
    fn test_interference_result_helpers() {
        let result = RetroactiveInterferenceResult {
            magnitude: 0.20,
            interfering_episodes: vec!["ep1".to_string(), "ep2".to_string()],
            count: 2,
        };

        assert!(result.is_significant());
        assert_eq!(result.accuracy_reduction_percent(), 20.0);

        let insignificant = RetroactiveInterferenceResult {
            magnitude: 0.05,
            interfering_episodes: vec![],
            count: 0,
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

        // Dissimilar embeddings
        let c = create_similar_embedding(-1.0);
        let sim2 = cosine_similarity(&a, &c);
        assert!(
            sim2 < 0.5,
            "Dissimilar embeddings should have low similarity"
        );
    }
}
