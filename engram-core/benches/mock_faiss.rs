//! Mock FAISS implementation for benchmarking framework testing
//!
//! This is a placeholder that simulates FAISS behavior.
//! In production, you would use actual FAISS bindings.

use crate::ann_comparison::AnnIndex;
use anyhow::Result;
use std::collections::BTreeMap;

pub struct MockFaissIndex {
    vectors: Vec<[f32; 768]>,
    index_type: String,
    m: usize,
}

impl MockFaissIndex {
    pub fn new_ivf_flat(dimension: usize, nlist: usize) -> Result<Self> {
        assert_eq!(dimension, 768, "Only 768-dim vectors supported");

        Ok(Self {
            vectors: Vec::new(),
            index_type: format!("IVF{},Flat", nlist),
            m: 16,
        })
    }

    pub fn new_hnsw(dimension: usize, m: usize) -> Result<Self> {
        assert_eq!(dimension, 768, "Only 768-dim vectors supported");

        Ok(Self {
            vectors: Vec::new(),
            index_type: format!("HNSW{}", m),
            m,
        })
    }

    fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a * norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

impl AnnIndex for MockFaissIndex {
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()> {
        self.vectors = vectors.to_vec();
        Ok(())
    }

    fn search(&self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)> {
        // Simple brute-force search for mock implementation
        let mut similarities: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| (idx, Self::cosine_similarity(query, vec)))
            .collect();

        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        similarities.truncate(k);
        similarities
    }

    fn memory_usage(&self) -> usize {
        // Estimate: vectors + graph structure
        let vector_memory = self.vectors.len() * 768 * std::mem::size_of::<f32>();
        let graph_memory = if self.index_type.starts_with("HNSW") {
            // HNSW: M edges per node on average
            self.vectors.len() * self.m * std::mem::size_of::<usize>()
        } else {
            // IVF: centroid storage + inverted lists
            self.vectors.len() * std::mem::size_of::<usize>()
        };

        vector_memory + graph_memory
    }

    fn name(&self) -> &str {
        "FAISS (Mock)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_faiss_hnsw() {
        let mut index = MockFaissIndex::new_hnsw(768, 16).expect("Failed to create");

        let vectors = vec![
            [0.1f32; 768],
            [0.2f32; 768],
            [0.3f32; 768],
        ];

        index.build(&vectors).expect("Failed to build");

        let query = [0.15f32; 768];
        let results = index.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Closest should be first vector
    }

    #[test]
    fn test_mock_faiss_ivf() {
        let mut index = MockFaissIndex::new_ivf_flat(768, 100).expect("Failed to create");

        let vectors = vec![[0.5f32; 768]; 10];

        index.build(&vectors).expect("Failed to build");

        let results = index.search(&vectors[0], 5);

        assert_eq!(results.len(), 5);
        // All vectors are identical, so all should have similarity 1.0
        for (_, sim) in results {
            assert!((sim - 1.0).abs() < 0.001);
        }
    }
}