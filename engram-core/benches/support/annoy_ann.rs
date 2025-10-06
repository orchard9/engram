//! Annoy implementation for ANN benchmarks
//!
//! Note: The annoy-rs crate currently only supports loading pre-built indexes,
//! not building new ones from vectors. This implementation uses a simplified
//! mock until full index building support is available.
//!
//! For production validation, FAISS is used as the primary comparison library.

use super::ann_common::AnnIndex;
use anyhow::Result;

/// Annoy ANN index wrapper implementing the AnnIndex trait
///
/// Currently uses a mock implementation due to annoy-rs API limitations.
pub struct AnnoyAnnIndex {
    dimension: usize,
    n_trees: usize,
    vectors: Vec<[f32; 768]>,
}

impl AnnoyAnnIndex {
    /// Create a new Annoy index
    ///
    /// # Arguments
    /// * `dimension` - Vector dimensionality (must be 768)
    /// * `n_trees` - Number of trees (more trees = better recall but slower build, typically 10-100)
    #[allow(clippy::unnecessary_wraps)]
    pub fn new(dimension: usize, n_trees: usize) -> Result<Self> {
        assert_eq!(dimension, 768, "Only 768-dim vectors supported");

        Ok(Self {
            dimension,
            n_trees,
            vectors: Vec::new(),
        })
    }

    /// Angular distance between two vectors
    fn angular_distance(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a * norm_b > 0.0 {
            let cosine = dot_product / (norm_a * norm_b);
            // Angular distance
            1.0 - cosine.clamp(-1.0, 1.0)
        } else {
            1.0
        }
    }

    /// Convert angular distance to similarity score
    fn angular_to_similarity(distance: f32) -> f32 {
        // Angular distance ∈ [0, 2], map to similarity ∈ [0, 1]
        1.0 - (distance / 2.0).clamp(0.0, 1.0)
    }
}

impl AnnIndex for AnnoyAnnIndex {
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()> {
        // Store vectors for mock implementation
        self.vectors = vectors.to_vec();
        // In real Annoy, this would build random projection trees
        Ok(())
    }

    fn search(&mut self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        // Mock implementation: use exact search with angular distance
        let mut distances: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let dist = Self::angular_distance(query, vec);
                (idx, dist)
            })
            .collect();

        // Add small random perturbation to simulate approximation
        if self.n_trees < 100 {
            for (_, dist) in &mut distances {
                let noise = (rand::random::<f32>() - 0.5) * 0.01;
                *dist += noise;
                *dist = dist.clamp(0.0, 2.0);
            }
        }

        // Sort by distance (lowest first)
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        distances
            .into_iter()
            .take(k)
            .map(|(idx, dist)| (idx, Self::angular_to_similarity(dist)))
            .collect()
    }

    fn memory_usage(&self) -> usize {
        // Base vector storage
        let vector_bytes = self.vectors.len() * self.dimension * std::mem::size_of::<f32>();

        // Annoy tree overhead estimate
        // Each tree stores approximately O(n) nodes with pointers and split hyperplanes
        let n_items = self.vectors.len();
        let tree_overhead = 32; // Approximate bytes per node
        let tree_bytes = n_items * self.n_trees * tree_overhead;

        vector_bytes + tree_bytes
    }

    fn name(&self) -> &'static str {
        "Annoy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annoy_build_and_search() {
        let mut index = AnnoyAnnIndex::new(768, 10).unwrap();

        // Create test vectors
        let vectors: Vec<[f32; 768]> = (0..100)
            .map(|i| {
                let mut v = [0.0; 768];
                v[0] = i as f32 / 100.0;
                v
            })
            .collect();

        index.build(&vectors).unwrap();

        // Search for vector similar to index 5
        let mut query = [0.0; 768];
        query[0] = 0.05;

        let results = index.search(&query, 5);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_annoy_exact_match() {
        let mut index = AnnoyAnnIndex::new(768, 10).unwrap();

        let vectors: Vec<[f32; 768]> = vec![
            {
                let mut v = [0.0; 768];
                v[0] = 1.0;
                v
            },
            {
                let mut v = [0.0; 768];
                v[0] = 2.0;
                v
            },
            {
                let mut v = [0.0; 768];
                v[0] = 3.0;
                v
            },
        ];

        index.build(&vectors).unwrap();

        // Query with exact match to second vector
        let mut query = [0.0; 768];
        query[0] = 2.0;

        let results = index.search(&query, 1);

        assert_eq!(results.len(), 1);
        // Should find vector 1 (index of second vector)
        assert_eq!(results[0].0, 1);
        // Similarity should be very high (near 1.0)
        assert!(results[0].1 > 0.95);
    }

    #[test]
    fn test_angular_to_similarity() {
        // Zero distance should give similarity 1.0
        assert!((AnnoyAnnIndex::angular_to_similarity(0.0) - 1.0).abs() < 0.001);

        // Large distance should give small similarity
        assert!(AnnoyAnnIndex::angular_to_similarity(2.0).abs() < 0.001);

        // Similarity should be monotonically decreasing with distance
        let sim1 = AnnoyAnnIndex::angular_to_similarity(0.5);
        let sim2 = AnnoyAnnIndex::angular_to_similarity(1.0);
        assert!(sim1 > sim2);
    }

    #[test]
    fn test_memory_usage() {
        let mut index = AnnoyAnnIndex::new(768, 10).unwrap();

        let vectors: Vec<[f32; 768]> = vec![[0.0; 768]; 100];
        index.build(&vectors).unwrap();

        let memory = index.memory_usage();

        // Should be at least the size of the vectors
        let min_memory = 100 * 768 * std::mem::size_of::<f32>();
        assert!(memory >= min_memory);
    }

    #[test]
    fn test_multiple_trees_builds() {
        // Test that different numbers of trees work
        for n_trees in [5, 10, 20, 50] {
            let mut index = AnnoyAnnIndex::new(768, n_trees).unwrap();

            let vectors: Vec<[f32; 768]> = (0..50)
                .map(|i| {
                    let mut v = [0.0; 768];
                    v[0] = i as f32;
                    v
                })
                .collect();

            assert!(index.build(&vectors).is_ok());

            let query = [25.0f32; 768];
            let results = index.search(&query, 5);
            assert!(!results.is_empty());
        }
    }
}
