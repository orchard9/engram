//! Scalar reference implementation of vector operations
//!
//! Provides portable, correct implementations that serve as:
//! 1. Fallback for platforms without SIMD support
//! 2. Reference for correctness testing of SIMD implementations
//! 3. Baseline for performance comparisons

use super::VectorOps;

/// Scalar implementation of vector operations
pub struct ScalarVectorOps;

impl ScalarVectorOps {
    /// Create new scalar vector operations instance
    pub fn new() -> Self {
        Self
    }
}

impl VectorOps for ScalarVectorOps {
    fn cosine_similarity_768(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
        let mut dot_product = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        // Single pass through data for better cache locality
        for i in 0..768 {
            let ai = a[i];
            let bi = b[i];
            dot_product += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }

        let norm_a = norm_a.sqrt();
        let norm_b = norm_b.sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            (dot_product / (norm_a * norm_b)).clamp(-1.0, 1.0)
        }
    }

    fn cosine_similarity_batch_768(&self, query: &[f32; 768], vectors: &[[f32; 768]]) -> Vec<f32> {
        // Pre-compute query norm once
        let query_norm = self.l2_norm_768(query);
        if query_norm == 0.0 {
            return vec![0.0; vectors.len()];
        }

        vectors
            .iter()
            .map(|vector| {
                let dot_product = self.dot_product_768(query, vector);
                let vector_norm = self.l2_norm_768(vector);

                if vector_norm == 0.0 {
                    0.0
                } else {
                    (dot_product / (query_norm * vector_norm)).clamp(-1.0, 1.0)
                }
            })
            .collect()
    }

    fn dot_product_768(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..768 {
            sum += a[i] * b[i];
        }
        sum
    }

    fn l2_norm_768(&self, vector: &[f32; 768]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..768 {
            let v = vector[i];
            sum += v * v;
        }
        sum.sqrt()
    }

    fn vector_add_768(&self, a: &[f32; 768], b: &[f32; 768]) -> [f32; 768] {
        let mut result = [0.0f32; 768];
        for i in 0..768 {
            result[i] = a[i] + b[i];
        }
        result
    }

    fn vector_scale_768(&self, vector: &[f32; 768], scale: f32) -> [f32; 768] {
        let mut result = [0.0f32; 768];
        for i in 0..768 {
            result[i] = vector[i] * scale;
        }
        result
    }

    fn weighted_average_768(&self, vectors: &[&[f32; 768]], weights: &[f32]) -> [f32; 768] {
        let mut result = [0.0f32; 768];

        if vectors.is_empty() || weights.is_empty() {
            return result;
        }

        let weight_sum: f32 = weights.iter().sum();
        if weight_sum == 0.0 {
            return result;
        }

        let num_vectors = vectors.len().min(weights.len());

        for i in 0..768 {
            let mut sum = 0.0f32;
            for j in 0..num_vectors {
                sum += vectors[j][i] * weights[j];
            }
            result[i] = sum / weight_sum;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let ops = ScalarVectorOps::new();
        let a = [1.0f32; 768];
        let similarity = ops.cosine_similarity_768(&a, &a);
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let ops = ScalarVectorOps::new();
        let a = [1.0f32; 768];
        let b = [-1.0f32; 768];
        let similarity = ops.cosine_similarity_768(&a, &b);
        assert!((similarity + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let ops = ScalarVectorOps::new();
        let mut a = [0.0f32; 768];
        let mut b = [0.0f32; 768];
        a[0] = 1.0;
        b[1] = 1.0;
        let similarity = ops.cosine_similarity_768(&a, &b);
        assert!(similarity.abs() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let ops = ScalarVectorOps::new();
        let a = [2.0f32; 768];
        let b = [3.0f32; 768];
        let dot = ops.dot_product_768(&a, &b);
        assert!((dot - 768.0 * 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_l2_norm() {
        let ops = ScalarVectorOps::new();
        let mut a = [0.0f32; 768];
        a[0] = 3.0;
        a[1] = 4.0;
        let norm = ops.l2_norm_768(&a);
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_cosine_similarity() {
        let ops = ScalarVectorOps::new();
        let query = [1.0f32; 768];
        let vectors = vec![[1.0f32; 768], [-1.0f32; 768], [0.5f32; 768]];

        let similarities = ops.cosine_similarity_batch_768(&query, &vectors);
        assert_eq!(similarities.len(), 3);
        assert!((similarities[0] - 1.0).abs() < 1e-6);
        assert!((similarities[1] + 1.0).abs() < 1e-6);
        assert!((similarities[2] - 1.0).abs() < 1e-6);
    }
}
