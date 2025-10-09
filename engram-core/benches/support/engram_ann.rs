//! Engram ANN index implementation for benchmarking

use super::ann_common::AnnIndex;
use anyhow::Result;

/// Simplified Engram index for benchmarking
///
/// This uses a simplified HNSW-like implementation for benchmarking purposes.
/// In production, this would use the actual `CognitiveHnswIndex` from `engram_core`.
pub struct EngramOptimizedAnnIndex {
    vectors: Vec<[f32; 768]>,
    graph: Vec<Vec<usize>>, // Adjacency list for HNSW graph
    m: usize,               // Number of connections per node
    ef_construction: usize,
    ef_search: usize,
}

/// Convenient alias used by benchmark harnesses.
pub type EngramAnnIndex = EngramOptimizedAnnIndex;

impl EngramOptimizedAnnIndex {
    pub const fn new() -> Self {
        Self {
            vectors: Vec::new(),
            graph: Vec::new(),
            m: 16,
            ef_construction: 200,
            ef_search: 100,
        }
    }

    pub const fn with_params(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            vectors: Vec::new(),
            graph: Vec::new(),
            m,
            ef_construction,
            ef_search,
        }
    }

    fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        // Use SIMD-optimized version if available
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::cosine_similarity_avx2(a, b) };
            }
        }

        // Fallback to scalar
        Self::cosine_similarity_scalar(a, b)
    }

    fn cosine_similarity_scalar(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a * norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn cosine_similarity_avx2(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        use std::arch::x86_64::*;

        let mut dot_product = _mm256_setzero_ps();
        let mut norm_a = _mm256_setzero_ps();
        let mut norm_b = _mm256_setzero_ps();

        // Process 8 floats at a time
        for i in (0..768).step_by(8) {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));

            dot_product = _mm256_add_ps(dot_product, _mm256_mul_ps(va, vb));
            norm_a = _mm256_add_ps(norm_a, _mm256_mul_ps(va, va));
            norm_b = _mm256_add_ps(norm_b, _mm256_mul_ps(vb, vb));
        }

        // Horizontal sum
        let dot_sum = Self::horizontal_sum_avx2(dot_product);
        let norm_a_sum = Self::horizontal_sum_avx2(norm_a).sqrt();
        let norm_b_sum = Self::horizontal_sum_avx2(norm_b).sqrt();

        if norm_a_sum * norm_b_sum > 0.0 {
            dot_sum / (norm_a_sum * norm_b_sum)
        } else {
            0.0
        }
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
        use std::arch::x86_64::*;

        // Sum upper and lower 128-bit lanes
        let sum_128 = _mm_add_ps(_mm256_extractf128_ps(v, 0), _mm256_extractf128_ps(v, 1));

        // Horizontal adds
        let sum_64 = _mm_hadd_ps(sum_128, sum_128);
        let sum_32 = _mm_hadd_ps(sum_64, sum_64);

        _mm_cvtss_f32(sum_32)
    }

    fn build_hnsw_graph(&mut self) {
        // Simplified HNSW construction
        let candidate_cap = self.ef_construction.max(self.m);
        self.graph = vec![Vec::with_capacity(candidate_cap); self.vectors.len()];

        for i in 0..self.vectors.len() {
            // Connect to M nearest neighbors (simplified)
            let mut neighbors: Vec<(usize, f32)> = (0..self.vectors.len())
                .filter(|&j| j != i)
                .map(|j| {
                    let sim = Self::cosine_similarity(&self.vectors[i], &self.vectors[j]);
                    (j, sim)
                })
                .collect();

            neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            neighbors.truncate(
                self.ef_construction
                    .min(self.vectors.len().saturating_sub(1)),
            );

            self.graph[i] = neighbors
                .into_iter()
                .take(self.m)
                .map(|(idx, _)| idx)
                .collect();
        }
    }

    fn search_layer(
        &self,
        query: &[f32; 768],
        entry_points: &[usize],
        ef: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited = vec![false; self.vectors.len()];
        let mut candidates = Vec::new();
        let mut w = Vec::new();

        // Initialize with entry points
        for &point in entry_points {
            let sim = Self::cosine_similarity(query, &self.vectors[point]);
            candidates.push((sim, point));
            w.push((point, sim));
            visited[point] = true;
        }

        // Sort candidates by similarity (highest first)
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        while !candidates.is_empty() {
            let (lowerbound, current) = candidates.remove(0);

            // Sort w to get minimum
            w.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if lowerbound < w[0].1 {
                break;
            }

            // Check neighbors
            for &neighbor in &self.graph[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    let sim = Self::cosine_similarity(query, &self.vectors[neighbor]);

                    if w.is_empty() || sim > w[0].1 || w.len() < ef {
                        candidates.push((sim, neighbor));
                        w.push((neighbor, sim));

                        // Keep w sorted and limited to ef
                        w.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        if w.len() > ef {
                            w.truncate(ef);
                        }

                        // Keep candidates sorted
                        candidates.sort_by(|a, b| {
                            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                }
            }
        }

        // Return results sorted by similarity (highest first)
        w.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        w
    }
}

impl AnnIndex for EngramOptimizedAnnIndex {
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()> {
        self.vectors = vectors.to_vec();
        self.build_hnsw_graph();
        Ok(())
    }

    fn search(&mut self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        // Use first vector as entry point (simplified)
        let entry_points = vec![0];
        let mut results = self.search_layer(query, &entry_points, self.ef_search);

        results.truncate(k);
        results
    }

    fn memory_usage(&self) -> usize {
        // Vectors memory
        let vector_memory = self.vectors.len() * 768 * std::mem::size_of::<f32>();

        // Graph memory
        let graph_memory = self
            .graph
            .iter()
            .map(|neighbors| neighbors.len() * std::mem::size_of::<usize>())
            .sum::<usize>();

        vector_memory + graph_memory
    }

    fn name(&self) -> &'static str {
        "Engram-Optimized"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engram_build_and_search() {
        let mut index = EngramOptimizedAnnIndex::new();

        let vectors = vec![
            [0.1f32; 768],
            [0.2f32; 768],
            [0.3f32; 768],
            [0.4f32; 768],
            [0.5f32; 768],
        ];

        index.build(&vectors).expect("Failed to build");

        let query = [0.25f32; 768];
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 3);
        // Results should be sorted by similarity
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    #[test]
    fn test_with_params_customises_configuration() {
        let index = EngramOptimizedAnnIndex::with_params(8, 64, 32);

        assert_eq!(index.m, 8);
        assert_eq!(index.ef_construction, 64);
        assert_eq!(index.ef_search, 32);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0f32; 768];
        let b = [1.0f32; 768];

        let sim = EngramOptimizedAnnIndex::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);

        let c = [-1.0f32; 768];
        let sim2 = EngramOptimizedAnnIndex::cosine_similarity(&a, &c);
        assert!((sim2 + 1.0).abs() < 0.001);
    }
}
