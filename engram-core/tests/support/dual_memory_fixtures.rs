// Support fixtures for dual-memory integration tests
//
// Provides deterministic test data generation for migration, differential testing,
// and load scenarios. All fixtures use seeded RNGs for reproducibility.

use chrono::{Duration, Utc};
use engram_core::{Confidence, Episode, Memory, EMBEDDING_DIM};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Configuration for episode cluster generation
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Number of distinct clusters to generate
    pub cluster_count: usize,
    /// Episodes per cluster
    pub episodes_per_cluster: usize,
    /// Intra-cluster similarity (0.0 = random, 1.0 = identical)
    pub intra_cluster_similarity: f32,
    /// Inter-cluster separation (0.0 = no separation, 1.0 = orthogonal)
    pub inter_cluster_separation: f32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            cluster_count: 5,
            episodes_per_cluster: 10,
            intra_cluster_similarity: 0.85,
            inter_cluster_separation: 0.3,
        }
    }
}

/// Generate deterministic episodes that cluster well
///
/// Creates episodes with controlled similarity for testing concept formation.
/// Uses cluster centroids with added noise to simulate natural episode variation.
#[must_use]
pub fn generate_clusterable_episodes(config: ClusterConfig, seed: u64) -> Vec<Episode> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut episodes = Vec::with_capacity(config.cluster_count * config.episodes_per_cluster);

    // Generate cluster centroids (orthogonal-ish base vectors)
    let centroids = generate_cluster_centroids(config.cluster_count, seed);

    let base_time = Utc::now() - Duration::hours(24);

    for (cluster_idx, centroid) in centroids.iter().enumerate() {
        for episode_idx in 0..config.episodes_per_cluster {
            let episode_id = format!("cluster_{}_episode_{}", cluster_idx, episode_idx);

            // Add controlled noise around centroid
            let embedding = add_controlled_noise(
                centroid,
                1.0 - config.intra_cluster_similarity,
                &mut rng,
            );

            // Vary temporal properties within cluster
            let time_offset = Duration::minutes(rng.gen_range(0..120));
            let when = base_time + time_offset;

            let confidence = Confidence::exact(rng.gen_range(0.7..0.95));

            let episode = Episode {
                id: episode_id,
                when,
                where_location: None,
                who: None,
                what: format!("Episode from cluster {} (semantic group)", cluster_idx),
                embedding,
                embedding_provenance: None,
                encoding_confidence: confidence,
                vividness_confidence: confidence,
                reliability_confidence: confidence,
                last_recall: when,
                recall_count: 0,
                decay_rate: 1.0,
                decay_function: None,
                metadata: std::collections::HashMap::new(),
            };

            episodes.push(episode);
        }
    }

    episodes
}

/// Generate cluster centroids with controlled separation
fn generate_cluster_centroids(count: usize, seed: u64) -> Vec<[f32; EMBEDDING_DIM]> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = Vec::with_capacity(count);

    for cluster_idx in 0..count {
        let mut centroid = [0.0_f32; EMBEDDING_DIM];

        // Use block-sparse pattern to create separation
        let block_start = (cluster_idx * EMBEDDING_DIM / count).min(EMBEDDING_DIM - 64);
        let block_end = (block_start + 128).min(EMBEDDING_DIM);

        for i in block_start..block_end {
            let phase = (i - block_start) as f32 * 0.1;
            centroid[i] = (phase.sin() + rng.gen_range(-0.2..0.2)).clamp(-1.0, 1.0);
        }

        // Normalize to unit length
        let norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut centroid {
                *x /= norm;
            }
        }

        centroids.push(centroid);
    }

    centroids
}

/// Add Gaussian noise to an embedding vector
fn add_controlled_noise(
    base: &[f32; EMBEDDING_DIM],
    noise_scale: f32,
    rng: &mut StdRng,
) -> [f32; EMBEDDING_DIM] {
    let mut result = *base;

    for x in &mut result {
        let noise = rng.gen_range(-noise_scale..noise_scale);
        *x = (*x + noise).clamp(-1.0, 1.0);
    }

    // Re-normalize
    let norm = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut result {
            *x /= norm;
        }
    }

    result
}

/// Generate diverse test episodes for general testing
///
/// Creates episodes with varying:
/// - Temporal distribution (recent to old)
/// - Confidence levels (low to high)
/// - Embedding diversity (random but reproducible)
#[must_use]
pub fn generate_test_episodes(count: usize, seed: u64) -> Vec<Episode> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut episodes = Vec::with_capacity(count);

    let base_time = Utc::now();

    for idx in 0..count {
        let embedding = generate_random_embedding(seed + idx as u64);

        // Exponential time distribution (more recent, some old)
        let hours_ago = rng.gen_range(0..1000);
        let when = base_time - Duration::hours(hours_ago);

        let confidence = Confidence::exact(rng.gen_range(0.5..1.0));

        let episode = Episode {
            id: format!("test_episode_{}", idx),
            when,
            where_location: None,
            who: None,
            what: format!("Test content for episode {}", idx),
            embedding,
            embedding_provenance: None,
            encoding_confidence: confidence,
            vividness_confidence: confidence,
            reliability_confidence: confidence,
            last_recall: when,
            recall_count: 0,
            decay_rate: 1.0,
            decay_function: None,
            metadata: std::collections::HashMap::new(),
        };

        episodes.push(episode);
    }

    episodes
}

/// Generate random but deterministic embedding
fn generate_random_embedding(seed: u64) -> [f32; EMBEDDING_DIM] {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut embedding = [0.0_f32; EMBEDDING_DIM];

    for (idx, slot) in embedding.iter_mut().enumerate() {
        let phase = idx as f32 * 0.013;
        let noise = rng.gen_range(-0.5..0.5);
        *slot = (phase.sin() + noise).clamp(-1.0, 1.0);
    }

    // Normalize
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }

    embedding
}

/// Query set for differential testing
///
/// Returns (query_text, query_embedding, expected_episode_count) tuples
#[must_use]
pub fn generate_test_query_set(seed: u64) -> Vec<(String, [f32; EMBEDDING_DIM], usize)> {
    vec![
        (
            "General query 1".to_string(),
            generate_random_embedding(seed),
            10,
        ),
        (
            "General query 2".to_string(),
            generate_random_embedding(seed + 1),
            10,
        ),
        (
            "Specific query 3".to_string(),
            generate_random_embedding(seed + 2),
            5,
        ),
    ]
}

/// Generate episodes with temporal patterns for consolidation testing
#[must_use]
pub fn generate_temporal_episodes(
    count: usize,
    time_window_hours: i64,
    seed: u64,
) -> Vec<Episode> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut episodes = Vec::with_capacity(count);

    let end_time = Utc::now();
    let start_time = end_time - Duration::hours(time_window_hours);
    let total_seconds = time_window_hours * 3600;

    for idx in 0..count {
        let offset_seconds = rng.gen_range(0..total_seconds);
        let when = start_time + Duration::seconds(offset_seconds);

        let embedding = generate_random_embedding(seed + idx as u64);
        let confidence = Confidence::exact(rng.gen_range(0.6..0.95));

        let episode = Episode {
            id: format!("temporal_episode_{}", idx),
            when,
            where_location: None,
            who: None,
            what: format!("Temporal test content {}", idx),
            embedding,
            embedding_provenance: None,
            encoding_confidence: confidence,
            vividness_confidence: confidence,
            reliability_confidence: confidence,
            last_recall: when,
            recall_count: 0,
            decay_rate: 1.0,
            decay_function: None,
            metadata: std::collections::HashMap::new(),
        };

        episodes.push(episode);
    }

    // Sort by time for realistic insertion order
    episodes.sort_by_key(|e| e.when);
    episodes
}

/// Generate memories for legacy compatibility testing
#[must_use]
pub fn generate_legacy_memories(count: usize, seed: u64) -> Vec<Memory> {
    let mut memories = Vec::with_capacity(count);

    for idx in 0..count {
        let embedding = generate_random_embedding(seed + idx as u64);
        let confidence = Confidence::exact(0.8);

        let memory = Memory::new(
            format!("legacy_memory_{}", idx),
            embedding,
            confidence,
        );

        memories.push(memory);
    }

    memories
}

/// Calculate overlap percentage between two episode ID sets
///
/// Returns value between 0.0 (no overlap) and 1.0 (identical)
#[must_use]
pub fn calculate_overlap(a: &[String], b: &[String]) -> f32 {
    let set_a: std::collections::HashSet<_> = a.iter().collect();
    let set_b: std::collections::HashSet<_> = b.iter().collect();

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        return 1.0; // Both empty = perfect overlap
    }

    intersection as f32 / union as f32
}

/// Assert that two confidence values are within tolerance
pub fn assert_confidence_similar(
    actual: Confidence,
    expected: Confidence,
    tolerance: f32,
    message: &str,
) {
    let diff = (actual.raw() - expected.raw()).abs();
    assert!(
        diff <= tolerance,
        "{}: confidence diff {} exceeds tolerance {} (actual={}, expected={})",
        message,
        diff,
        tolerance,
        actual.raw(),
        expected.raw()
    );
}

/// Assert that overlap meets minimum threshold
pub fn assert_overlap_threshold(
    actual_ids: &[String],
    expected_ids: &[String],
    min_threshold: f32,
    message: &str,
) {
    let overlap = calculate_overlap(actual_ids, expected_ids);
    assert!(
        overlap >= min_threshold,
        "{}: overlap {} below threshold {} ({} vs {} items)",
        message,
        overlap,
        min_threshold,
        actual_ids.len(),
        expected_ids.len()
    );
}
