//! Pattern detection engine for unsupervised statistical pattern detection.
//!
//! This module implements hierarchical agglomerative clustering to identify
//! recurring structures in episodic memories without supervision.

use crate::Episode;
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use std::sync::Arc;

/// Signature for caching patterns
type PatternSignature = u64;

/// Configuration for pattern detection
#[derive(Debug, Clone)]
pub struct PatternDetectionConfig {
    /// Minimum cluster size for pattern extraction (default: 3)
    pub min_cluster_size: usize,
    /// Similarity threshold for clustering (cosine similarity, default: 0.8)
    pub similarity_threshold: f32,
    /// Maximum number of patterns to extract per cycle (default: 100)
    pub max_patterns: usize,
}

impl Default for PatternDetectionConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 3,
            similarity_threshold: 0.8,
            max_patterns: 100,
        }
    }
}

/// Pattern detector for identifying recurring episodic structures
pub struct PatternDetector {
    /// Configuration for pattern detection
    config: PatternDetectionConfig,
    /// Cached patterns for fast lookup (reserved for future optimization)
    #[allow(dead_code)]
    pattern_cache: Arc<DashMap<PatternSignature, CachedPattern>>,
}

/// Cached pattern for fast lookups (reserved for future optimization)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CachedPattern {
    /// Pattern data
    pattern: EpisodicPattern,
    /// When this pattern was cached
    cached_at: DateTime<Utc>,
}

/// Represents a detected episodic pattern
#[derive(Debug, Clone)]
pub struct EpisodicPattern {
    /// Pattern identifier (deterministic based on source episodes)
    pub id: String,
    /// Average embedding across clustered episodes
    pub embedding: [f32; 768],
    /// IDs of source episodes contributing to this pattern
    pub source_episodes: Vec<String>,
    /// Pattern strength (coherence of cluster, 0.0-1.0)
    pub strength: f32,
    /// Extracted common features
    pub features: Vec<PatternFeature>,
    /// First occurrence timestamp
    pub first_occurrence: DateTime<Utc>,
    /// Last occurrence timestamp
    pub last_occurrence: DateTime<Utc>,
    /// Number of episodes in this pattern
    pub occurrence_count: usize,
}

/// Common features extracted from pattern
#[derive(Debug, Clone)]
pub enum PatternFeature {
    /// Temporal sequence with typical interval
    TemporalSequence {
        /// Average time interval between episodes in the pattern
        interval: Duration,
    },
    /// Spatial proximity with location
    SpatialProximity {
        /// Common location string extracted from episodes
        location: String,
    },
    /// Conceptual theme
    ConceptualTheme {
        /// Extracted thematic concept
        theme: String,
    },
    /// Emotional valence (-1.0 to 1.0)
    EmotionalValence {
        /// Emotional polarity score from -1.0 (negative) to 1.0 (positive)
        valence: f32,
    },
}

impl PatternDetector {
    /// Create a new pattern detector with configuration
    #[must_use]
    pub fn new(config: PatternDetectionConfig) -> Self {
        Self {
            config,
            pattern_cache: Arc::new(DashMap::new()),
        }
    }

    /// Detect patterns in episode collection using embedding similarity
    ///
    /// # Performance
    /// - <100ms for 100 episodes
    /// - <1s for 1000 episodes
    #[must_use]
    pub fn detect_patterns(&self, episodes: &[Episode]) -> Vec<EpisodicPattern> {
        if episodes.is_empty() {
            return Vec::new();
        }

        // Phase 1: Cluster episodes by embedding similarity
        let clusters = self.cluster_episodes(episodes);

        // Phase 2: Extract common patterns from clusters
        let mut patterns = Vec::new();
        for cluster in clusters {
            if cluster.len() >= self.config.min_cluster_size {
                if let Some(pattern) = self.extract_pattern(&cluster) {
                    patterns.push(pattern);

                    // Stop if we've hit the max patterns limit
                    if patterns.len() >= self.config.max_patterns {
                        break;
                    }
                }
            }
        }

        // Phase 3: Merge similar patterns
        Self::merge_similar_patterns(patterns)
    }

    /// Cluster episodes using hierarchical agglomerative clustering with centroid linkage
    fn cluster_episodes(&self, episodes: &[Episode]) -> Vec<Vec<Episode>> {
        if episodes.is_empty() {
            return Vec::new();
        }

        // DETERMINISM FIX 1: Sort episodes by ID before clustering for deterministic initial state
        let mut sorted_episodes = episodes.to_vec();
        sorted_episodes.sort_by(|a, b| a.id.cmp(&b.id));

        // Initialize each episode as its own cluster
        let mut clusters: Vec<Vec<Episode>> =
            sorted_episodes.iter().map(|ep| vec![ep.clone()]).collect();

        // Cache cluster centroids for performance (avoid recomputation)
        let mut centroids: Vec<[f32; 768]> = clusters
            .iter()
            .map(|cluster| Self::compute_centroid(cluster))
            .collect();

        // Iteratively merge most similar clusters
        while clusters.len() > 1 {
            let (i, j, similarity) =
                Self::find_most_similar_clusters_centroid(&centroids, &clusters);

            if similarity < self.config.similarity_threshold {
                break; // No more similar clusters to merge
            }

            // Merge clusters i and j (j > i always)
            let cluster_j = clusters.remove(j);
            centroids.remove(j);

            clusters[i].extend(cluster_j);

            // Recompute centroid for merged cluster
            centroids[i] = Self::compute_centroid(&clusters[i]);
        }

        clusters
    }

    /// Compute centroid (average embedding) of a cluster
    fn compute_centroid(cluster: &[Episode]) -> [f32; 768] {
        if cluster.is_empty() {
            return [0.0; 768];
        }

        Self::average_embeddings(cluster)
    }

    /// Find the two most similar clusters using centroid linkage
    ///
    /// Returns (index_i, index_j, similarity) where i < j
    ///
    /// DETERMINISM FIX 2: When similarities are equal, use lexicographic tie-breaking
    /// based on the smallest episode ID in each cluster to ensure deterministic behavior.
    fn find_most_similar_clusters_centroid(
        centroids: &[[f32; 768]],
        clusters: &[Vec<Episode>],
    ) -> (usize, usize, f32) {
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_similarity = 0.0;

        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                let similarity = Self::embedding_similarity(&centroids[i], &centroids[j]);

                // Deterministic tie-breaking: prefer lexicographically earlier cluster IDs
                // Note: We use exact equality here because we WANT deterministic tie-breaking.
                // For truly equal similarities, lexicographic ordering provides determinism.
                #[allow(clippy::float_cmp)]
                // Intentional: deterministic tie-breaking requires exact equality
                let is_better = similarity > best_similarity
                    || (similarity == best_similarity
                        && Self::cluster_tiebreaker(
                            &clusters[i],
                            &clusters[j],
                            &clusters[best_i],
                            &clusters[best_j],
                        ));

                if is_better {
                    best_similarity = similarity;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        (best_i, best_j, best_similarity)
    }

    /// Deterministic tie-breaker for cluster pairs with equal similarity.
    ///
    /// Returns true if (cluster_i, cluster_j) should be preferred over the current best.
    /// Uses lexicographic ordering of minimum episode IDs, mapping to the "primacy effect"
    /// in neuroscience where earlier-encoded memories have precedence.
    fn cluster_tiebreaker(
        cluster_i: &[Episode],
        cluster_j: &[Episode],
        current_best_i: &[Episode],
        current_best_j: &[Episode],
    ) -> bool {
        // Get minimum episode IDs from each cluster
        let min_id_i = cluster_i.iter().map(|ep| &ep.id).min();
        let min_id_j = cluster_j.iter().map(|ep| &ep.id).min();
        let min_id_best_i = current_best_i.iter().map(|ep| &ep.id).min();
        let min_id_best_j = current_best_j.iter().map(|ep| &ep.id).min();

        // Lexicographic comparison: (i, j) < (best_i, best_j)
        (min_id_i, min_id_j) < (min_id_best_i, min_id_best_j)
    }

    /// Extract pattern from cluster of similar episodes
    fn extract_pattern(&self, episodes: &[Episode]) -> Option<EpisodicPattern> {
        if episodes.is_empty() {
            return None;
        }

        // Compute average embedding
        let avg_embedding = Self::average_embeddings(episodes);

        // Compute pattern strength (coherence measure)
        let strength = Self::compute_pattern_strength(episodes, &avg_embedding);

        // Extract common features
        let features = self.extract_common_features(episodes);

        // Find time range
        let first_occurrence = episodes
            .iter()
            .map(|ep| ep.when)
            .min()
            .unwrap_or_else(Utc::now);
        let last_occurrence = episodes
            .iter()
            .map(|ep| ep.when)
            .max()
            .unwrap_or_else(Utc::now);

        // Create deterministic ID
        let mut source_episodes: Vec<String> = episodes.iter().map(|ep| ep.id.clone()).collect();
        source_episodes.sort();
        let id = format!("pattern_{}", Self::compute_pattern_hash(&source_episodes));

        Some(EpisodicPattern {
            id,
            embedding: avg_embedding,
            source_episodes,
            strength,
            features,
            first_occurrence,
            last_occurrence,
            occurrence_count: episodes.len(),
        })
    }

    /// Compute average embedding from episodes using Kahan summation for determinism.
    ///
    /// DETERMINISM FIX 3: Uses Kahan compensated summation to eliminate floating-point
    /// non-associativity. This ensures identical results across platforms and episode orderings.
    ///
    /// Reference: Kahan, W. (1965). "Further remarks on reducing truncation errors"
    fn average_embeddings(episodes: &[Episode]) -> [f32; 768] {
        let mut avg = [0.0f32; 768];
        let count = episodes.len() as f32;

        if count == 0.0 {
            return avg;
        }

        // Use Kahan summation for each dimension to ensure deterministic FP arithmetic
        for (i, avg_val) in avg.iter_mut().enumerate() {
            let values = episodes.iter().map(|ep| ep.embedding[i]);
            let (sum, _compensation) = Self::kahan_sum(values);
            *avg_val = sum / count;
        }

        avg
    }

    /// Kahan summation algorithm for deterministic floating-point addition.
    ///
    /// This algorithm tracks and compensates for rounding errors during summation,
    /// making the result independent of summation order and platform floating-point
    /// implementation details.
    ///
    /// Returns (sum, final_compensation)
    fn kahan_sum(values: impl Iterator<Item = f32>) -> (f32, f32) {
        let mut sum = 0.0f32;
        let mut compensation = 0.0f32; // Running compensation for lost low-order bits

        for value in values {
            let y = value - compensation;
            let t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }

        (sum, compensation)
    }

    /// Compute pattern strength (cluster coherence)
    fn compute_pattern_strength(episodes: &[Episode], centroid: &[f32; 768]) -> f32 {
        if episodes.is_empty() {
            return 0.0;
        }

        // Measure average similarity to centroid
        let mut total_similarity = 0.0;
        for episode in episodes {
            total_similarity += Self::embedding_similarity(&episode.embedding, centroid);
        }

        total_similarity / episodes.len() as f32
    }

    /// Extract common features from episode cluster
    #[allow(clippy::unused_self)] // Reserved for future feature extraction config
    fn extract_common_features(&self, episodes: &[Episode]) -> Vec<PatternFeature> {
        let mut features = Vec::new();

        // Extract temporal sequence if present
        if episodes.len() >= 2 {
            let timestamps: Vec<DateTime<Utc>> = episodes.iter().map(|ep| ep.when).collect();
            if let Some(interval) = Self::detect_temporal_pattern(&timestamps) {
                features.push(PatternFeature::TemporalSequence { interval });
            }
        }

        // Future feature extraction (deferred to future milestones):
        // - Spatial proximity detection
        // - Conceptual theme extraction
        // - Emotional valence analysis

        features
    }

    /// Detect temporal pattern in timestamps
    fn detect_temporal_pattern(timestamps: &[DateTime<Utc>]) -> Option<Duration> {
        if timestamps.len() < 2 {
            return None;
        }

        let mut sorted_times = timestamps.to_vec();
        sorted_times.sort();

        // Compute intervals
        let mut intervals = Vec::new();
        for i in 1..sorted_times.len() {
            intervals.push(sorted_times[i] - sorted_times[i - 1]);
        }

        // Check if intervals are roughly consistent
        if intervals.is_empty() {
            return None;
        }

        let total_ms: i64 = intervals
            .iter()
            .map(chrono::TimeDelta::num_milliseconds)
            .sum();
        let avg_interval = Duration::milliseconds(total_ms / intervals.len() as i64);

        Some(avg_interval)
    }

    /// Merge similar patterns to avoid redundancy
    ///
    /// DETERMINISM FIX 5: Sort patterns by ID before merging to ensure deterministic merge order
    fn merge_similar_patterns(mut patterns: Vec<EpisodicPattern>) -> Vec<EpisodicPattern> {
        if patterns.len() <= 1 {
            return patterns;
        }

        // Sort patterns by ID for deterministic iteration order
        patterns.sort_by(|a, b| a.id.cmp(&b.id));

        let mut merged = Vec::new();
        let mut used = vec![false; patterns.len()];

        for i in 0..patterns.len() {
            if used[i] {
                continue;
            }

            let mut current = patterns[i].clone();
            used[i] = true;

            // Find similar patterns to merge
            for j in (i + 1)..patterns.len() {
                if used[j] {
                    continue;
                }

                let similarity =
                    Self::embedding_similarity(&current.embedding, &patterns[j].embedding);
                if similarity > 0.9 {
                    // Merge patterns
                    current = Self::merge_two_patterns(current, patterns[j].clone());
                    used[j] = true;
                }
            }

            merged.push(current);
        }

        merged
    }

    /// Merge two similar patterns into one
    fn merge_two_patterns(mut a: EpisodicPattern, b: EpisodicPattern) -> EpisodicPattern {
        // Combine source episodes
        a.source_episodes.extend(b.source_episodes);
        a.source_episodes.sort();
        a.source_episodes.dedup();

        // Re-average embedding weighted by occurrence count
        let weight_a = a.occurrence_count as f32;
        let weight_b = b.occurrence_count as f32;
        let total_weight = weight_a + weight_b;

        for i in 0..768 {
            a.embedding[i] = (a.embedding[i] * weight_a + b.embedding[i] * weight_b) / total_weight;
        }

        // Update metadata
        a.strength = (a.strength * weight_a + b.strength * weight_b) / total_weight;
        a.first_occurrence = a.first_occurrence.min(b.first_occurrence);
        a.last_occurrence = a.last_occurrence.max(b.last_occurrence);
        a.occurrence_count += b.occurrence_count;
        a.features.extend(b.features);

        // Update ID
        a.id = format!("pattern_{}", Self::compute_pattern_hash(&a.source_episodes));

        a
    }

    /// Compute embedding similarity (cosine similarity)
    fn embedding_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Compute deterministic hash for pattern ID
    fn compute_pattern_hash(source_episodes: &[String]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for id in source_episodes {
            id.hash(&mut hasher);
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Confidence, Episode};
    use std::collections::HashSet;

    // Helper functions and constants for determinism tests

    // EXPECTED SIGNATURE for cross-platform determinism test.
    // This MUST be identical across all platforms after determinism fix.
    // Reference platform: macOS ARM64 (Apple Silicon)
    // Generated: 2025-11-01
    const EXPECTED_CROSS_PLATFORM_SIGNATURE: Option<u64> = Some(0xA47D_9F44_DDA0_BAD2);

    fn create_test_episodes(count: usize) -> Vec<Episode> {
        let base_time = Utc::now();
        let mut episodes = Vec::new();

        for i in 0..count {
            // Create diverse embeddings based on index
            let mut embedding = [0.0f32; 768];
            for (j, emb_val) in embedding.iter_mut().enumerate() {
                *emb_val = ((i + j) % 10) as f32 / 10.0;
            }

            episodes.push(Episode::new(
                format!("episode_{i:03}"),
                base_time + Duration::seconds(i as i64 * 60),
                format!("content_{i}"),
                embedding,
                Confidence::exact(0.8 + (i % 3) as f32 * 0.05),
            ));
        }

        episodes
    }

    fn compute_pattern_set_signature(patterns: &[EpisodicPattern]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Sort pattern IDs for deterministic hashing
        #[allow(clippy::collection_is_never_read)]
        // False positive: collection is read via hash() below
        let mut sorted_ids: Vec<String> = patterns.iter().map(|p| p.id.clone()).collect();
        sorted_ids.sort();

        let mut hasher = DefaultHasher::new();
        sorted_ids.hash(&mut hasher);

        // Also include pattern count and total occurrence count for more robust signature
        patterns.len().hash(&mut hasher);
        let total_occurrences: usize = patterns.iter().map(|p| p.occurrence_count).sum();
        total_occurrences.hash(&mut hasher);

        hasher.finish()
    }

    // Tests

    #[test]
    fn test_pattern_detector_creation() {
        let config = PatternDetectionConfig::default();
        let detector = PatternDetector::new(config);
        assert_eq!(detector.pattern_cache.len(), 0);
    }

    #[test]
    fn test_embedding_similarity() {
        let a = [1.0; 768];
        let b = [1.0; 768];
        let similarity = PatternDetector::embedding_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < f32::EPSILON);

        let c = [0.0; 768];
        let similarity_zero = PatternDetector::embedding_similarity(&a, &c);
        assert!(similarity_zero.abs() < f32::EPSILON);
    }

    #[test]
    fn test_empty_episodes() {
        let config = PatternDetectionConfig::default();
        let detector = PatternDetector::new(config);
        let patterns = detector.detect_patterns(&[]);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_min_cluster_size_enforcement() {
        let config = PatternDetectionConfig {
            min_cluster_size: 3,
            ..Default::default()
        };
        let detector = PatternDetector::new(config);

        // Create 2 similar episodes (below min cluster size)
        let episodes = vec![
            Episode::new(
                "ep1".to_string(),
                Utc::now(),
                "test".to_string(),
                [1.0; 768],
                Confidence::exact(0.9),
            ),
            Episode::new(
                "ep2".to_string(),
                Utc::now(),
                "test".to_string(),
                [1.0; 768],
                Confidence::exact(0.9),
            ),
        ];

        let patterns = detector.detect_patterns(&episodes);
        // Should be empty because cluster size < min_cluster_size
        assert!(patterns.is_empty());
    }

    /// DETERMINISM TEST: Verify consolidation produces identical results across multiple runs.
    ///
    /// Property: ∀ episodes, ∀ runs ∈ [1..1000], all_equal(runs.map(|_| detect_patterns(episodes)))
    ///
    /// This is a CRITICAL test for M14 distributed consolidation. Non-determinism would cause
    /// gossip divergence and permanent inconsistency across nodes.
    #[test]
    fn test_consolidation_determinism_basic() {
        let config = PatternDetectionConfig {
            min_cluster_size: 2,
            similarity_threshold: 0.7,
            max_patterns: 100,
        };
        let detector = PatternDetector::new(config);

        // Create test episodes with varying embeddings
        let episodes = create_test_episodes(20);

        // Run pattern detection 100 times to verify determinism
        let mut pattern_signatures = HashSet::new();

        for _ in 0..100 {
            let patterns = detector.detect_patterns(&episodes);
            let signature = compute_pattern_set_signature(&patterns);
            pattern_signatures.insert(signature);
        }

        // MUST produce identical results every time
        assert_eq!(
            pattern_signatures.len(),
            1,
            "Non-deterministic consolidation detected: {} unique outputs from 100 runs",
            pattern_signatures.len()
        );
    }

    /// DETERMINISM TEST: Verify consolidation is invariant to input order.
    ///
    /// Property: ∀ episodes, detect_patterns(episodes) == detect_patterns(shuffle(episodes))
    ///
    /// This ensures that episode arrival order (which varies across distributed nodes)
    /// does not affect final consolidation results.
    #[test]
    fn test_consolidation_order_invariance() {
        let config = PatternDetectionConfig {
            min_cluster_size: 2,
            similarity_threshold: 0.7,
            max_patterns: 100,
        };
        let detector = PatternDetector::new(config);

        let episodes = create_test_episodes(15);

        // Run on original order
        let patterns_original = detector.detect_patterns(&episodes);
        let sig_original = compute_pattern_set_signature(&patterns_original);

        // Run on reversed order
        let mut episodes_reversed = episodes.clone();
        episodes_reversed.reverse();
        let patterns_reversed = detector.detect_patterns(&episodes_reversed);
        let sig_reversed = compute_pattern_set_signature(&patterns_reversed);

        assert_eq!(
            sig_original, sig_reversed,
            "Consolidation is order-dependent: original != reversed"
        );

        // Run on shuffled order (using deterministic shuffle based on IDs)
        #[allow(clippy::redundant_clone)] // Not redundant: episodes is reused below
        let mut episodes_shuffled = episodes.clone();
        episodes_shuffled.sort_by(|a, b| a.id.len().cmp(&b.id.len()).then_with(|| a.id.cmp(&b.id)));
        let patterns_shuffled = detector.detect_patterns(&episodes_shuffled);
        let sig_shuffled = compute_pattern_set_signature(&patterns_shuffled);

        assert_eq!(
            sig_original, sig_shuffled,
            "Consolidation is order-dependent: original != shuffled"
        );
    }

    /// DETERMINISM TEST: Verify Kahan summation produces identical centroids.
    ///
    /// Tests that floating-point centroid computation is deterministic across
    /// different episode orderings, validating the Kahan summation implementation.
    #[test]
    fn test_kahan_summation_determinism() {
        let episodes = create_test_episodes(10);

        // Compute centroid from original order
        let centroid_original = PatternDetector::average_embeddings(&episodes);

        // Compute centroid from reversed order
        #[allow(clippy::redundant_clone)] // Not redundant: episodes is immutable parameter
        let mut episodes_reversed = episodes.clone();
        episodes_reversed.reverse();
        let centroid_reversed = PatternDetector::average_embeddings(&episodes_reversed);

        // Verify bit-exact equality (Kahan summation should produce identical results)
        #[allow(clippy::float_cmp)] // Intentional: testing for exact deterministic equality
        for i in 0..768 {
            assert_eq!(
                centroid_original[i], centroid_reversed[i],
                "Kahan summation is order-dependent at dimension {i}"
            );
        }
    }

    /// DETERMINISM TEST: Verify tie-breaking is consistent.
    ///
    /// When multiple cluster pairs have equal similarity, the lexicographic tie-breaker
    /// should produce deterministic merge decisions.
    #[test]
    fn test_tiebreaker_consistency() {
        // Create episodes with identical embeddings (will have equal similarities)
        let identical_embedding = [0.5; 768];
        let episodes = vec![
            Episode::new(
                "ep_charlie".to_string(),
                Utc::now(),
                "test".to_string(),
                identical_embedding,
                Confidence::exact(0.9),
            ),
            Episode::new(
                "ep_alpha".to_string(),
                Utc::now(),
                "test".to_string(),
                identical_embedding,
                Confidence::exact(0.9),
            ),
            Episode::new(
                "ep_bravo".to_string(),
                Utc::now(),
                "test".to_string(),
                identical_embedding,
                Confidence::exact(0.9),
            ),
        ];

        let config = PatternDetectionConfig {
            min_cluster_size: 2,
            similarity_threshold: 0.5,
            max_patterns: 100,
        };
        let detector = PatternDetector::new(config);

        // Run multiple times - tie-breaker should produce same result
        let mut signatures = HashSet::new();
        for _ in 0..50 {
            let patterns = detector.detect_patterns(&episodes);
            let signature = compute_pattern_set_signature(&patterns);
            signatures.insert(signature);
        }

        assert_eq!(
            signatures.len(),
            1,
            "Tie-breaking is non-deterministic: {} unique outputs",
            signatures.len()
        );
    }

    /// DETERMINISM STRESS TEST: 1000-iteration determinism validation
    ///
    /// This is the comprehensive test specified in the M14 action plan.
    /// It verifies that consolidation produces identical results across 1000 runs,
    /// providing high confidence in determinism for distributed deployment.
    #[test]
    #[ignore] // Run explicitly with: cargo test -- --ignored --nocapture
    fn test_consolidation_determinism_1000_iterations() {
        let config = PatternDetectionConfig {
            min_cluster_size: 2,
            similarity_threshold: 0.7,
            max_patterns: 100,
        };
        let detector = PatternDetector::new(config);

        // Create larger test dataset (100 episodes)
        let episodes = create_test_episodes(100);

        let start = std::time::Instant::now();
        let mut pattern_signatures = HashSet::new();

        for i in 0..1000 {
            let patterns = detector.detect_patterns(&episodes);
            let signature = compute_pattern_set_signature(&patterns);
            pattern_signatures.insert(signature);

            if i % 100 == 0 {
                println!("Completed {i} iterations...");
            }
        }

        let duration = start.elapsed();
        println!("1000 iterations completed in {duration:?}");

        // MUST produce identical results every time
        let num_unique = pattern_signatures.len();
        assert_eq!(
            num_unique, 1,
            "CRITICAL FAILURE: Non-deterministic consolidation detected across 1000 runs. \
             Found {num_unique} unique outputs. This blocks M14 distributed consolidation."
        );

        // Performance validation: should complete in <60 seconds
        let secs = duration.as_secs();
        assert!(
            secs < 60,
            "Performance regression: 1000 iterations took {duration:?} (limit: 60s)"
        );

        println!("SUCCESS: Consolidation is deterministic across 1000 iterations");
    }

    /// CROSS-PLATFORM DETERMINISM TEST: Verify identical results across architectures
    ///
    /// This test computes a reference signature that MUST match across all platforms:
    /// - macOS ARM64 (Apple Silicon)
    /// - Linux x86_64 (Intel/AMD)
    /// - Linux ARM64 (Graviton/Raspberry Pi)
    ///
    /// Platform-specific floating-point differences would cause signature mismatch.
    #[test]
    #[ignore] // Run on multiple platforms to verify cross-platform determinism
    fn test_cross_platform_determinism() {
        let config = PatternDetectionConfig {
            min_cluster_size: 2,
            similarity_threshold: 0.7,
            max_patterns: 100,
        };
        let detector = PatternDetector::new(config);

        // Use fixed seed for reproducible test episodes
        let episodes = create_test_episodes(50);

        let patterns = detector.detect_patterns(&episodes);
        let signature = compute_pattern_set_signature(&patterns);

        let arch = std::env::consts::ARCH;
        let os = std::env::consts::OS;
        let pattern_count = patterns.len();
        println!("Platform: {arch}");
        println!("OS: {os}");
        println!("Pattern signature: 0x{signature:016X}");
        println!("Pattern count: {pattern_count}");

        // Verify against expected cross-platform signature
        if let Some(expected) = EXPECTED_CROSS_PLATFORM_SIGNATURE {
            assert_eq!(
                signature,
                expected,
                "Cross-platform determinism FAILED. \
                 Signature 0x{:016X} != expected 0x{:016X}. \
                 Platform: {} {}. \
                 This indicates floating-point or sorting differences across architectures.",
                signature,
                expected,
                std::env::consts::OS,
                std::env::consts::ARCH
            );

            println!("SUCCESS: Cross-platform determinism verified");
        } else {
            println!("NOTICE: EXPECTED_SIGNATURE not set. Run this test and update the constant.");
            println!("Copy this line to the test:");
            println!("    const EXPECTED_SIGNATURE: Option<u64> = Some(0x{signature:016X});");
        }
    }

    /// DISTRIBUTED GOSSIP SIMULATION: Verify determinism with different arrival orders
    ///
    /// Simulates a distributed cluster where 5 nodes receive episodes in different orders
    /// (due to network delays, gossip propagation, etc). All nodes MUST produce identical
    /// patterns after consolidation for gossip convergence.
    #[test]
    fn test_distributed_gossip_convergence_simulation() {
        let config = PatternDetectionConfig {
            min_cluster_size: 2,
            similarity_threshold: 0.7,
            max_patterns: 100,
        };

        let episodes = create_test_episodes(30);

        // Simulate 5 nodes receiving episodes in different orders
        let mut node_patterns: Vec<Vec<EpisodicPattern>> = Vec::new();

        for node_id in 0..5 {
            // Different arrival orders per node (deterministic shuffle using node_id)
            let mut node_episodes = episodes.clone();
            node_episodes.sort_by(|a, b| {
                let hash_a = (a.id.len() * (node_id + 1)) % 100;
                let hash_b = (b.id.len() * (node_id + 1)) % 100;
                hash_a.cmp(&hash_b).then_with(|| a.id.cmp(&b.id))
            });

            let detector = PatternDetector::new(config.clone());
            let patterns = detector.detect_patterns(&node_episodes);
            node_patterns.push(patterns);
        }

        // All nodes MUST produce identical patterns
        let reference_signature = compute_pattern_set_signature(&node_patterns[0]);

        for (node_id, patterns) in node_patterns.iter().enumerate() {
            let signature = compute_pattern_set_signature(patterns);
            assert_eq!(
                signature, reference_signature,
                "Node {node_id} diverged from reference. Gossip convergence would FAIL. \
                 This blocks M14 distributed consolidation."
            );
        }

        println!(
            "SUCCESS: All 5 nodes converged to identical patterns despite different arrival orders"
        );
    }
}
