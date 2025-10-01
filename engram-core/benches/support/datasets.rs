//! Dataset loaders for ANN benchmarking

use super::ann_common::AnnDataset;
use rand::{Rng, SeedableRng};

pub struct DatasetLoader;

impl DatasetLoader {
    /// Generate synthetic dataset for testing
    pub fn generate_synthetic(num_vectors: usize, num_queries: usize) -> AnnDataset {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Generate base vectors
        let vectors: Vec<[f32; 768]> = (0..num_vectors)
            .map(|_| {
                let mut vec = [0.0f32; 768];
                for val in &mut vec {
                    *val = rng.gen_range(-1.0..1.0);
                }
                // Normalize
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for val in &mut vec {
                        *val /= norm;
                    }
                }
                vec
            })
            .collect();

        // Generate query vectors (some overlap with base for ground truth)
        let queries: Vec<[f32; 768]> = (0..num_queries)
            .map(|i| {
                if i < num_queries / 2 {
                    // Half queries are slight perturbations of base vectors
                    let base_idx = rng.gen_range(0..vectors.len());
                    let mut vec = vectors[base_idx];
                    // Add small noise
                    for val in &mut vec {
                        *val += rng.gen_range(-0.01..0.01);
                    }
                    // Renormalize
                    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for val in &mut vec {
                            *val /= norm;
                        }
                    }
                    vec
                } else {
                    // Half are completely random
                    let mut vec = [0.0f32; 768];
                    for val in &mut vec {
                        *val = rng.gen_range(-1.0..1.0);
                    }
                    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for val in &mut vec {
                            *val /= norm;
                        }
                    }
                    vec
                }
            })
            .collect();

        // Compute ground truth using brute force
        let ground_truth = Self::compute_ground_truth(&vectors, &queries);

        AnnDataset {
            name: format!("Synthetic_{num_vectors}_{num_queries}"),
            vectors,
            queries,
            ground_truth,
        }
    }

    /// Load SIFT1M dataset (mock version for testing)
    pub fn load_sift1m_mock() -> AnnDataset {
        // Create a smaller mock dataset that mimics SIFT1M characteristics
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);

        // Generate 10K vectors instead of 1M for testing
        let num_vectors = 10_000;
        let num_queries = 100;

        // SIFT vectors are typically 128-dim, we'll pad to 768
        let vectors: Vec<[f32; 768]> = (0..num_vectors)
            .map(|_| {
                let mut vec = [0.0f32; 768];
                // Fill first 128 dimensions with SIFT-like values
                for val in vec.iter_mut().take(128) {
                    *val = rng.gen_range(0.0..255.0) / 255.0;
                }
                vec
            })
            .collect();

        let queries: Vec<[f32; 768]> = (0..num_queries)
            .map(|_| {
                let mut vec = [0.0f32; 768];
                for val in vec.iter_mut().take(128) {
                    *val = rng.gen_range(0.0..255.0) / 255.0;
                }
                vec
            })
            .collect();

        let ground_truth = Self::compute_ground_truth(&vectors, &queries);

        AnnDataset {
            name: "SIFT1M_Mock".to_string(),
            vectors,
            queries,
            ground_truth,
        }
    }

    /// Load `GloVe` dataset (mock version for testing)
    #[allow(dead_code)]
    pub fn load_glove_mock() -> AnnDataset {
        let mut rng = rand::rngs::StdRng::seed_from_u64(456);

        // Generate 5K vectors for testing
        let num_vectors = 5_000;
        let num_queries = 50;

        // GloVe vectors have specific distribution characteristics
        let vectors: Vec<[f32; 768]> = (0..num_vectors)
            .map(|_| {
                let mut vec = [0.0f32; 768];
                for val in &mut vec {
                    // GloVe vectors typically have mean near 0 and std ~0.4
                    *val = rng.gen_range(-1.5..1.5) * 0.4;
                }
                vec
            })
            .collect();

        let queries = vectors[0..num_queries].to_vec();

        let ground_truth = Self::compute_ground_truth(&vectors, &queries);

        AnnDataset {
            name: "GloVe_Mock".to_string(),
            vectors,
            queries,
            ground_truth,
        }
    }

    /// Compute exact k-NN ground truth using brute force
    fn compute_ground_truth(vectors: &[[f32; 768]], queries: &[[f32; 768]]) -> Vec<Vec<usize>> {
        queries
            .iter()
            .map(|query| {
                let mut distances: Vec<(usize, f32)> = vectors
                    .iter()
                    .enumerate()
                    .map(|(idx, vec)| {
                        let dist = Self::euclidean_distance(query, vec);
                        (idx, dist)
                    })
                    .collect();

                // Sort by distance (lowest first)
                distances
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                // Return top 100 indices
                distances
                    .into_iter()
                    .take(100)
                    .map(|(idx, _)| idx)
                    .collect()
            })
            .collect()
    }

    fn euclidean_distance(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Pad vectors to 768 dimensions
    pub fn pad_to_768(vectors: Vec<Vec<f32>>) -> Vec<[f32; 768]> {
        vectors
            .into_iter()
            .map(|v| {
                let mut arr = [0.0f32; 768];
                let copy_len = v.len().min(768);
                arr[..copy_len].copy_from_slice(&v[..copy_len]);
                arr
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_dataset() {
        let dataset = DatasetLoader::generate_synthetic(100, 10);

        assert_eq!(dataset.vectors.len(), 100);
        assert_eq!(dataset.queries.len(), 10);
        assert_eq!(dataset.ground_truth.len(), 10);

        // Check normalization
        for vec in &dataset.vectors {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.001, "Vector not normalized");
        }
    }

    #[test]
    fn test_ground_truth_computation() {
        let vectors = vec![[1.0f32; 768], [0.5f32; 768], [0.0f32; 768]];

        let queries = vec![[0.6f32; 768]];

        let ground_truth = DatasetLoader::compute_ground_truth(&vectors, &queries);

        assert_eq!(ground_truth.len(), 1);
        assert!(ground_truth[0].len() >= 3);
        // Closest should be the 0.5 vector (index 1)
        assert_eq!(ground_truth[0][0], 1);
    }

    #[test]
    fn test_sift_mock() {
        let dataset = DatasetLoader::load_sift1m_mock();

        assert_eq!(dataset.vectors.len(), 10_000);
        assert_eq!(dataset.queries.len(), 100);

        // Check that first 128 dims are non-zero
        for vec in &dataset.vectors[0..10] {
            let first_128_sum: f32 = vec[0..128].iter().sum();
            assert!(first_128_sum > 0.0);

            // Rest should be zeros
            let rest_sum: f32 = vec[128..].iter().sum();
            assert!(rest_sum.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_pad_to_768() {
        let vectors = vec![vec![1.0f32; 10], vec![2.0f32; 1000]];
        let padded = DatasetLoader::pad_to_768(vectors);

        assert_eq!(padded.len(), 2);
        assert!((padded[0][0] - 1.0).abs() < f32::EPSILON);
        assert!((padded[0][10] - 0.0).abs() < f32::EPSILON);
        // Vector was truncated to 768 dimensions
        // Original had 1000 elements, but we only keep first 768
        assert!((padded[1][767] - 0.0).abs() < f32::EPSILON);
    }
}
