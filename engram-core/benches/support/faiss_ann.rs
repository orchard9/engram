//! Real FAISS implementation for ANN benchmarks
//!
//! Provides wrappers around FAISS indices for benchmarking against Engram.

use super::ann_common::AnnIndex;
use anyhow::Result;
use faiss::index::IndexImpl;
use faiss::{Index, MetricType, index_factory};

/// FAISS index types supported for benchmarking
#[derive(Debug, Clone, Copy)]
pub enum FaissIndexType {
    /// Flat exact search (baseline)
    FlatL2,
    /// HNSW approximate search
    Hnsw { m: usize },
    /// Inverted file with flat encoding
    IVFFlat { nlist: usize },
}

/// FAISS ANN index wrapper implementing the AnnIndex trait
pub struct FaissAnnIndex {
    index: Box<IndexImpl>,
    dimension: usize,
    index_type: FaissIndexType,
}

impl FaissAnnIndex {
    /// Create a FAISS Flat (exact) index for ground truth computation
    pub fn new_flat(dimension: usize) -> Result<Self> {
        let description = "Flat";
        let index = index_factory(dimension as u32, description, MetricType::L2)?;

        Ok(Self {
            index: Box::new(index),
            dimension,
            index_type: FaissIndexType::FlatL2,
        })
    }

    /// Create a FAISS HNSW index (comparable to Engram's HNSW)
    ///
    /// # Arguments
    /// * `dimension` - Vector dimensionality (768 for Engram)
    /// * `m` - Number of connections per node (typically 16-32)
    pub fn new_hnsw(dimension: usize, m: usize) -> Result<Self> {
        let description = format!("HNSW{m}");
        let index = index_factory(dimension as u32, &description, MetricType::L2)?;

        Ok(Self {
            index: Box::new(index),
            dimension,
            index_type: FaissIndexType::Hnsw { m },
        })
    }

    /// Create a FAISS IVF index
    ///
    /// # Arguments
    /// * `dimension` - Vector dimensionality
    /// * `nlist` - Number of clusters (typically sqrt(n) where n is dataset size)
    #[allow(dead_code)]
    pub fn new_ivf_flat(dimension: usize, nlist: usize) -> Result<Self> {
        let description = format!("IVF{nlist},Flat");
        let index = index_factory(dimension as u32, &description, MetricType::L2)?;

        Ok(Self {
            index: Box::new(index),
            dimension,
            index_type: FaissIndexType::IVFFlat { nlist },
        })
    }

    /// Convert L2 distance to similarity score
    ///
    /// Uses 1/(1+distance) mapping to convert distances to [0,1] similarity
    fn l2_to_similarity(distance: f32) -> f32 {
        1.0 / (1.0 + distance)
    }
}

impl AnnIndex for FaissAnnIndex {
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()> {
        // Flatten vectors to FAISS format: [x1,y1,z1, x2,y2,z2, ...]
        let flat_vectors: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        // Train index if needed (IVF requires training)
        if !self.index.is_trained() {
            self.index.train(&flat_vectors)?;
        }

        // Add vectors to index
        self.index.add(&flat_vectors)?;

        Ok(())
    }

    fn search(&mut self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)> {
        // FAISS search returns SearchResult with distances and labels
        match self.index.search(query, k) {
            Ok(search_result) => {
                let distances = search_result.distances;
                let labels = search_result.labels;

                // Convert FAISS labels and distances to (index, similarity) pairs
                // Labels are indices into the original dataset
                let mut results = Vec::with_capacity(k);
                for (i, &dist) in distances.iter().enumerate() {
                    // Get the label at this position
                    // FAISS returns labels as Idx type (usually i64)
                    // We need to convert to usize for our API
                    if i < labels.len() {
                        // Use format/parse as a generic way to convert Idx to i64
                        let label_str = format!("{:?}", labels[i]);
                        if let Ok(label_i64) = label_str.parse::<i64>()
                            && label_i64 >= 0
                        {
                            let similarity = Self::l2_to_similarity(dist);
                            results.push((label_i64 as usize, similarity));
                        }
                    }
                }
                results
            }
            Err(e) => {
                eprintln!("FAISS search failed: {e:?}");
                Vec::new()
            }
        }
    }

    fn memory_usage(&self) -> usize {
        // Get number of vectors
        let n = self.index.ntotal() as usize;

        // Base vector storage
        let vector_bytes = n * self.dimension * std::mem::size_of::<f32>();

        // Index-specific overhead
        let overhead = match self.index_type {
            FaissIndexType::FlatL2 => 0,
            FaissIndexType::Hnsw { m } => {
                // HNSW stores M links per node, each link is ~8 bytes
                n * m * 8
            }
            FaissIndexType::IVFFlat { nlist } => {
                // IVF stores centroids + inverted lists
                nlist * self.dimension * std::mem::size_of::<f32>()
            }
        };

        vector_bytes + overhead
    }

    fn name(&self) -> &'static str {
        match self.index_type {
            FaissIndexType::FlatL2 => "FAISS-Flat",
            FaissIndexType::Hnsw { .. } => "FAISS-HNSW",
            FaissIndexType::IVFFlat { .. } => "FAISS-IVF",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_faiss_flat_exact_search() {
        let mut index = FaissAnnIndex::new_flat(768).unwrap();

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

        assert_eq!(results.len(), 5);
        // Closest should be vector 5
        assert_eq!(results[0].0, 5);
    }

    #[test]
    fn test_faiss_hnsw_approximate_search() {
        let mut index = FaissAnnIndex::new_hnsw(768, 16).unwrap();

        let vectors: Vec<[f32; 768]> = (0..200)
            .map(|i| {
                let mut v = [0.0; 768];
                v[0] = i as f32;
                v
            })
            .collect();

        index.build(&vectors).unwrap();

        // Search
        let mut query = [0.0; 768];
        query[0] = 50.0;

        let results = index.search(&query, 10);
        assert!(!results.is_empty());
        // HNSW should find approximately correct neighbors
        assert!(results.len() <= 10);
    }

    #[test]
    fn test_l2_to_similarity_conversion() {
        // Zero distance should give similarity 1.0
        assert!((FaissAnnIndex::l2_to_similarity(0.0) - 1.0).abs() < 0.001);

        // Large distance should give small similarity
        assert!(FaissAnnIndex::l2_to_similarity(100.0) < 0.01);

        // Similarity should be monotonically decreasing with distance
        let sim1 = FaissAnnIndex::l2_to_similarity(1.0);
        let sim2 = FaissAnnIndex::l2_to_similarity(2.0);
        assert!(sim1 > sim2);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let mut index = FaissAnnIndex::new_hnsw(768, 16).unwrap();

        let vectors: Vec<[f32; 768]> = vec![[0.0; 768]; 100];
        index.build(&vectors).unwrap();

        let memory = index.memory_usage();

        // Should be at least the size of the vectors
        let min_memory = 100 * 768 * std::mem::size_of::<f32>();
        assert!(memory >= min_memory);
    }
}
