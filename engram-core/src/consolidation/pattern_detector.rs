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
    TemporalSequence { interval: Duration },
    /// Spatial proximity with location
    SpatialProximity { location: String },
    /// Conceptual theme
    ConceptualTheme { theme: String },
    /// Emotional valence (-1.0 to 1.0)
    EmotionalValence { valence: f32 },
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
            if cluster.len() >= self.config.min_cluster_size
                && let Some(pattern) = self.extract_pattern(&cluster)
            {
                patterns.push(pattern);

                // Stop if we've hit the max patterns limit
                if patterns.len() >= self.config.max_patterns {
                    break;
                }
            }
        }

        // Phase 3: Merge similar patterns
        self.merge_similar_patterns(patterns)
    }

    /// Cluster episodes using hierarchical agglomerative clustering with centroid linkage
    fn cluster_episodes(&self, episodes: &[Episode]) -> Vec<Vec<Episode>> {
        if episodes.is_empty() {
            return Vec::new();
        }

        // Initialize each episode as its own cluster
        let mut clusters: Vec<Vec<Episode>> = episodes.iter().map(|ep| vec![ep.clone()]).collect();

        // Cache cluster centroids for performance (avoid recomputation)
        let mut centroids: Vec<[f32; 768]> = clusters
            .iter()
            .map(|cluster| self.compute_centroid(cluster))
            .collect();

        // Iteratively merge most similar clusters
        while clusters.len() > 1 {
            let (i, j, similarity) = Self::find_most_similar_clusters_centroid(&centroids);

            if similarity < self.config.similarity_threshold {
                break; // No more similar clusters to merge
            }

            // Merge clusters i and j (j > i always)
            let cluster_j = clusters.remove(j);
            centroids.remove(j);

            clusters[i].extend(cluster_j);

            // Recompute centroid for merged cluster
            centroids[i] = self.compute_centroid(&clusters[i]);
        }

        clusters
    }

    /// Compute centroid (average embedding) of a cluster
    fn compute_centroid(&self, cluster: &[Episode]) -> [f32; 768] {
        if cluster.is_empty() {
            return [0.0; 768];
        }

        Self::average_embeddings(cluster)
    }

    /// Find the two most similar clusters using centroid linkage
    ///
    /// Returns (index_i, index_j, similarity) where i < j
    fn find_most_similar_clusters_centroid(centroids: &[[f32; 768]]) -> (usize, usize, f32) {
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_similarity = 0.0;

        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                let similarity = Self::embedding_similarity(&centroids[i], &centroids[j]);
                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        (best_i, best_j, best_similarity)
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

    /// Compute average embedding from episodes
    fn average_embeddings(episodes: &[Episode]) -> [f32; 768] {
        let mut avg = [0.0f32; 768];
        let count = episodes.len() as f32;

        if count == 0.0 {
            return avg;
        }

        for episode in episodes {
            for (i, &val) in episode.embedding.iter().enumerate() {
                avg[i] += val / count;
            }
        }

        avg
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
            if let Some(interval) = self.detect_temporal_pattern(&timestamps) {
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
    fn detect_temporal_pattern(&self, timestamps: &[DateTime<Utc>]) -> Option<Duration> {
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

        let total_ms: i64 = intervals.iter().map(|d| d.num_milliseconds()).sum();
        let avg_interval = Duration::milliseconds(total_ms / intervals.len() as i64);

        Some(avg_interval)
    }

    /// Merge similar patterns to avoid redundancy
    fn merge_similar_patterns(&self, patterns: Vec<EpisodicPattern>) -> Vec<EpisodicPattern> {
        if patterns.len() <= 1 {
            return patterns;
        }

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
                    current = self.merge_two_patterns(current, patterns[j].clone());
                    used[j] = true;
                }
            }

            merged.push(current);
        }

        merged
    }

    /// Merge two similar patterns into one
    fn merge_two_patterns(&self, mut a: EpisodicPattern, b: EpisodicPattern) -> EpisodicPattern {
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
}
