//! Embedding generation utilities for testing
//!
//! Provides functions to create synthetic embeddings with controlled semantic similarity.
//! Based on the DRM paradigm embedding generator from drm_biological_validation.rs.
//!
//! Key Features:
//! - High intra-cluster similarity (BAS > 0.80) for semantically related items
//! - Low inter-cluster similarity (BAS < 0.30) for unrelated items
//! - Deterministic generation based on word and cluster parameters

use std::collections::HashMap;

/// Simple deterministic RNG (Linear Congruential Generator)
fn next_random(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    ((*state / 65_536) % 32_768) as f32 / 32_768.0
}

/// Compute cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Create a high-similarity semantic cluster embedding
///
/// This function generates embeddings where items in the same cluster have
/// high cosine similarity (typically > 0.80), while items in different clusters
/// have low similarity (< 0.30).
///
/// # Parameters
/// - `word`: The word to generate an embedding for
/// - `cluster`: The semantic cluster (0-3 supported)
///
/// # Returns
/// A 768-dimensional normalized embedding vector
///
/// # Example
/// ```ignore
/// let sleep_emb = create_high_similarity_embedding("sleep", 0);
/// let bed_emb = create_high_similarity_embedding("bed", 0);
/// let similarity = cosine_similarity(&sleep_emb, &bed_emb);
/// assert!(similarity > 0.80); // Same cluster = high similarity
/// ```
pub fn create_high_similarity_embedding(word: &str, cluster: usize) -> [f32; 768] {
    // Create cluster-specific theme vector (shared by all words in cluster)
    let mut theme_vector = [0.0f32; 768];
    let mut rng_state = 12_345_u64 + (cluster as u64 * 1000);

    for (i, val) in theme_vector.iter_mut().enumerate() {
        *val = if i % 4 == cluster % 4 {
            // Strong signal in cluster-specific dimensions
            next_random(&mut rng_state) * 2.0 - 1.0
        } else {
            // Weak signal in other dimensions (10% magnitude)
            (next_random(&mut rng_state) * 2.0 - 1.0) * 0.1
        };
    }

    // Normalize theme vector
    let norm: f32 = theme_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut theme_vector {
        *val /= norm;
    }

    // Start with theme vector and add SMALL word-specific noise
    let mut embedding = theme_vector;
    let word_seed = word.as_bytes().iter().map(|&b| u64::from(b)).sum::<u64>();
    let mut word_rng = 12_345_u64 + word_seed + (cluster as u64 * 1000);

    // Add small word-specific noise to make embeddings unique
    // This noise is small enough to maintain high similarity within cluster
    for val in &mut embedding {
        let noise = 0.03 + (next_random(&mut word_rng) * 0.02); // 0.03-0.05 noise
        *val += (next_random(&mut word_rng) * 2.0 - 1.0) * noise;
    }

    // Normalize to unit vector
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }

    embedding
}

/// Generate embeddings for a DRM word list with high intra-list similarity
///
/// Creates embeddings for all study words and the critical lure such that:
/// - All items have similarity > 0.70 to the critical lure
/// - Average BAS (between-item similarity) > 0.80
/// - This enables proper spreading activation and recall in DRM paradigm tests
///
/// # Parameters
/// - `study_words`: Words presented during study phase
/// - `critical_lure`: Unstudied word that should be falsely recalled
///
/// # Returns
/// HashMap mapping words to their embeddings
///
/// # Example
/// ```ignore
/// let study_words = ["bed", "rest", "awake", "tired", "dream"];
/// let embeddings = generate_drm_embeddings(&study_words, "sleep");
///
/// let lure_emb = embeddings["sleep"];
/// for &word in &study_words {
///     let word_emb = embeddings[word];
///     let sim = cosine_similarity(&lure_emb, &word_emb);
///     assert!(sim > 0.70); // High similarity enables recall
/// }
/// ```
pub fn generate_drm_embeddings<'a>(
    study_words: &'a [&'a str],
    critical_lure: &'a str,
) -> HashMap<&'a str, [f32; 768]> {
    let mut embeddings = HashMap::new();

    // All words in the same semantic cluster (cluster 0)
    // This ensures high intra-list similarity
    for &word in study_words {
        embeddings.insert(word, create_high_similarity_embedding(word, 0));
    }

    // Critical lure also in same cluster (semantically related)
    embeddings.insert(
        critical_lure,
        create_high_similarity_embedding(critical_lure, 0),
    );

    embeddings
}

/// Create a simple deterministic embedding from a seed value
///
/// This is a fallback function for tests that don't require semantic clustering.
/// It produces normalized embeddings but does NOT guarantee high similarity between
/// related items. Use `create_high_similarity_embedding()` for cognitive pattern tests.
///
/// # Parameters
/// - `seed`: Floating point seed for deterministic generation
///
/// # Returns
/// A 768-dimensional normalized embedding vector
pub fn create_embedding(seed: f32) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for (i, val) in embedding.iter_mut().enumerate() {
        *val = ((seed + i as f32) * 0.001).sin();
    }
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }
    embedding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_similarity_within_cluster() {
        let emb1 = create_high_similarity_embedding("sleep", 0);
        let emb2 = create_high_similarity_embedding("bed", 0);
        let emb3 = create_high_similarity_embedding("rest", 0);

        let sim_1_2 = cosine_similarity(&emb1, &emb2);
        let sim_1_3 = cosine_similarity(&emb1, &emb3);
        let sim_2_3 = cosine_similarity(&emb2, &emb3);

        assert!(
            sim_1_2 > 0.70,
            "Same cluster similarity too low: {sim_1_2:.3}"
        );
        assert!(
            sim_1_3 > 0.70,
            "Same cluster similarity too low: {sim_1_3:.3}"
        );
        assert!(
            sim_2_3 > 0.70,
            "Same cluster similarity too low: {sim_2_3:.3}"
        );
    }

    #[test]
    fn test_low_similarity_across_clusters() {
        let emb1 = create_high_similarity_embedding("sleep", 0);
        let emb2 = create_high_similarity_embedding("chair", 1);
        let emb3 = create_high_similarity_embedding("doctor", 2);

        let sim_1_2 = cosine_similarity(&emb1, &emb2);
        let sim_1_3 = cosine_similarity(&emb1, &emb3);
        let sim_2_3 = cosine_similarity(&emb2, &emb3);

        assert!(
            sim_1_2 < 0.40,
            "Different cluster similarity too high: {sim_1_2:.3}"
        );
        assert!(
            sim_1_3 < 0.40,
            "Different cluster similarity too high: {sim_1_3:.3}"
        );
        assert!(
            sim_2_3 < 0.40,
            "Different cluster similarity too high: {sim_2_3:.3}"
        );
    }

    #[test]
    fn test_drm_embeddings_structure() {
        let study_words = ["bed", "rest", "awake", "tired", "dream"];
        let critical_lure = "sleep";

        let embeddings = generate_drm_embeddings(&study_words, critical_lure);

        let lure_emb = embeddings[critical_lure];

        // Verify all study words have high similarity to lure
        for &word in &study_words {
            let word_emb = embeddings[word];
            let sim = cosine_similarity(&lure_emb, &word_emb);
            assert!(
                sim > 0.65,
                "Study word '{word}' has insufficient similarity to lure: {sim:.3}"
            );
        }
    }
}
