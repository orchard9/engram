//! Mock Annoy implementation for benchmarking framework testing
//!
//! This is a placeholder that simulates Annoy behavior.
//! In production, you would use actual Annoy bindings.

use crate::ann_comparison::AnnIndex;
use anyhow::Result;

pub struct MockAnnoyIndex {
    vectors: Vec<[f32; 768]>,
    dimension: usize,
    n_trees: usize,
}

impl MockAnnoyIndex {
    pub fn new(dimension: usize, n_trees: usize) -> Result<Self> {
        assert_eq!(dimension, 768, "Only 768-dim vectors supported");

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            n_trees,
        })
    }

    fn angular_distance(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        // Angular distance = 1 - cosine_similarity for normalized vectors
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
}

impl AnnIndex for MockAnnoyIndex {
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()> {
        self.vectors = vectors.to_vec();
        // In real Annoy, this would build random projection trees
        Ok(())
    }

    fn search(&self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        // Simulate Annoy's approximate search
        // In reality, Annoy uses random projection trees

        // For mock, we'll do exact search with some randomization
        let mut distances: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let dist = Self::angular_distance(query, vec);
                // Convert angular distance to similarity
                (idx, 1.0 - dist)
            })
            .collect();

        // Add small random perturbation to simulate approximation
        if self.n_trees < 100 {
            // Less trees = more approximation
            for (_, sim) in &mut distances {
                let noise = (rand::random::<f32>() - 0.5) * 0.01 * (100.0 / self.n_trees as f32);
                *sim += noise;
                *sim = sim.clamp(0.0, 1.0);
            }
        }

        // Sort by similarity (highest first)
        distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        distances.truncate(k);
        distances
    }

    fn memory_usage(&self) -> usize {
        // Estimate: vectors + trees
        let vector_memory = self.vectors.len() * 768 * std::mem::size_of::<f32>();

        // Each tree stores node structure
        let tree_memory = self.n_trees * self.vectors.len() * std::mem::size_of::<usize>() * 2;

        vector_memory + tree_memory
    }

    fn name(&self) -> &str {
        "Annoy (Mock)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_annoy() {
        let mut index = MockAnnoyIndex::new(768, 10).expect("Failed to create");

        let vectors = vec![
            [0.1f32; 768],
            [0.2f32; 768],
            [0.3f32; 768],
            [0.4f32; 768],
        ];

        index.build(&vectors).expect("Failed to build");

        let query = [0.25f32; 768];
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 3);
        // Results should be sorted by similarity
        assert!(results[0].1 >= results[1].1);
    }

    #[test]
    fn test_angular_distance() {
        let a = [1.0f32; 768];
        let b = [1.0f32; 768];

        let dist = MockAnnoyIndex::angular_distance(&a, &b);
        assert!(dist.abs() < 0.001); // Same vectors = 0 distance

        let c = [-1.0f32; 768];
        let dist2 = MockAnnoyIndex::angular_distance(&a, &c);
        assert!((dist2 - 2.0).abs() < 0.001); // Opposite vectors = max distance
    }
}