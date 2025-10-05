//! AVX2 optimized vector operations for x86_64
//!
//! Provides SIMD implementations using 256-bit AVX2 instructions
//! with FMA (Fused Multiply-Add) when available.

#![cfg(target_arch = "x86_64")]

use super::VectorOps;
use std::arch::x86_64::*;

/// AVX2 implementation of vector operations
pub struct Avx2VectorOps {
    has_fma: bool,
}

impl Avx2VectorOps {
    /// Create new AVX2 vector operations instance
    pub fn new() -> Self {
        Self {
            has_fma: is_x86_feature_detected!("fma"),
        }
    }
}

impl VectorOps for Avx2VectorOps {
    fn cosine_similarity_768(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
        if self.has_fma {
            unsafe { cosine_similarity_768_avx2_fma(a, b) }
        } else {
            unsafe { cosine_similarity_768_avx2(a, b) }
        }
    }

    fn cosine_similarity_batch_768(&self, query: &[f32; 768], vectors: &[[f32; 768]]) -> Vec<f32> {
        if vectors.is_empty() {
            return Vec::new();
        }

        // Use optimized batch implementation when available
        if self.has_fma {
            unsafe { cosine_similarity_batch_768_avx2_fma(query, vectors) }
        } else {
            unsafe { cosine_similarity_batch_768_avx2(query, vectors) }
        }
    }

    fn dot_product_768(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
        if self.has_fma {
            unsafe { dot_product_768_avx2_fma(a, b) }
        } else {
            unsafe { dot_product_768_avx2(a, b) }
        }
    }

    fn l2_norm_768(&self, vector: &[f32; 768]) -> f32 {
        unsafe { l2_norm_768_avx2(vector) }
    }

    fn vector_add_768(&self, a: &[f32; 768], b: &[f32; 768]) -> [f32; 768] {
        let mut result = [0.0f32; 768];
        unsafe { vector_add_768_avx2(a, b, &mut result) };
        result
    }

    fn vector_scale_768(&self, vector: &[f32; 768], scale: f32) -> [f32; 768] {
        let mut result = [0.0f32; 768];
        unsafe { vector_scale_768_avx2(vector, scale, &mut result) };
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

        unsafe { weighted_average_768_avx2(vectors, weights, weight_sum, &mut result) };
        result
    }

    fn fma_accumulate(&self, column: &[f32], scalar: f32, accumulator: &mut [f32]) {
        unsafe { fma_accumulate_avx2(column, scalar, accumulator) };
    }

    fn gather_f32(&self, base: &[f32], indices: &[usize]) -> Vec<f32> {
        unsafe { gather_f32_avx2(base, indices) }
    }

    fn horizontal_sum(&self, values: &[f32]) -> f32 {
        unsafe { horizontal_sum_avx2(values) }
    }
}

/// AVX2 cosine similarity with FMA
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_similarity_768_avx2_fma(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    let mut dot_product = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();

    // Process 8 f32 elements per iteration (256-bit registers)
    for chunk_idx in (0..768).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(chunk_idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(chunk_idx));

        // Fused multiply-add for all three accumulations
        dot_product = _mm256_fmadd_ps(va, vb, dot_product);
        norm_a = _mm256_fmadd_ps(va, va, norm_a);
        norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
    }

    // Horizontal reduction
    let dot_sum = horizontal_add_ps_256(dot_product);
    let norm_a_sum = horizontal_add_ps_256(norm_a).sqrt();
    let norm_b_sum = horizontal_add_ps_256(norm_b).sqrt();

    if norm_a_sum == 0.0 || norm_b_sum == 0.0 {
        0.0
    } else {
        (dot_sum / (norm_a_sum * norm_b_sum)).clamp(-1.0, 1.0)
    }
}

/// AVX2 cosine similarity without FMA
#[target_feature(enable = "avx2")]
unsafe fn cosine_similarity_768_avx2(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    let mut dot_product = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();

    for chunk_idx in (0..768).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(chunk_idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(chunk_idx));

        dot_product = _mm256_add_ps(dot_product, _mm256_mul_ps(va, vb));
        norm_a = _mm256_add_ps(norm_a, _mm256_mul_ps(va, va));
        norm_b = _mm256_add_ps(norm_b, _mm256_mul_ps(vb, vb));
    }

    let dot_sum = horizontal_add_ps_256(dot_product);
    let norm_a_sum = horizontal_add_ps_256(norm_a).sqrt();
    let norm_b_sum = horizontal_add_ps_256(norm_b).sqrt();

    if norm_a_sum == 0.0 || norm_b_sum == 0.0 {
        0.0
    } else {
        (dot_sum / (norm_a_sum * norm_b_sum)).clamp(-1.0, 1.0)
    }
}

/// AVX2 dot product with FMA
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_768_avx2_fma(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    let mut sum = _mm256_setzero_ps();

    for chunk_idx in (0..768).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(chunk_idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(chunk_idx));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    horizontal_add_ps_256(sum)
}

/// AVX2 dot product without FMA
#[target_feature(enable = "avx2")]
unsafe fn dot_product_768_avx2(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    let mut sum = _mm256_setzero_ps();

    for chunk_idx in (0..768).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(chunk_idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(chunk_idx));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }

    horizontal_add_ps_256(sum)
}

/// AVX2 L2 norm
#[target_feature(enable = "avx2")]
unsafe fn l2_norm_768_avx2(vector: &[f32; 768]) -> f32 {
    let mut sum = _mm256_setzero_ps();

    for chunk_idx in (0..768).step_by(8) {
        let v = _mm256_loadu_ps(vector.as_ptr().add(chunk_idx));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v, v));
    }

    horizontal_add_ps_256(sum).sqrt()
}

/// AVX2 vector addition
#[target_feature(enable = "avx2")]
unsafe fn vector_add_768_avx2(a: &[f32; 768], b: &[f32; 768], result: &mut [f32; 768]) {
    for chunk_idx in (0..768).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(chunk_idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(chunk_idx));
        let sum = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(result.as_mut_ptr().add(chunk_idx), sum);
    }
}

/// AVX2 vector scaling
#[target_feature(enable = "avx2")]
unsafe fn vector_scale_768_avx2(vector: &[f32; 768], scale: f32, result: &mut [f32; 768]) {
    let scale_vec = _mm256_set1_ps(scale);

    for chunk_idx in (0..768).step_by(8) {
        let v = _mm256_loadu_ps(vector.as_ptr().add(chunk_idx));
        let scaled = _mm256_mul_ps(v, scale_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(chunk_idx), scaled);
    }
}

/// AVX2 weighted average
#[target_feature(enable = "avx2")]
unsafe fn weighted_average_768_avx2(
    vectors: &[&[f32; 768]],
    weights: &[f32],
    weight_sum: f32,
    result: &mut [f32; 768],
) {
    let inv_weight_sum = _mm256_set1_ps(1.0 / weight_sum);
    let num_vectors = vectors.len().min(weights.len());

    for chunk_idx in (0..768).step_by(8) {
        let mut sum = _mm256_setzero_ps();

        for j in 0..num_vectors {
            let v = _mm256_loadu_ps(vectors[j].as_ptr().add(chunk_idx));
            let w = _mm256_set1_ps(weights[j]);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(v, w));
        }

        let avg = _mm256_mul_ps(sum, inv_weight_sum);
        _mm256_storeu_ps(result.as_mut_ptr().add(chunk_idx), avg);
    }
}

/// Horizontal addition for AVX2 (256-bit)
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_add_ps_256(v: __m256) -> f32 {
    // Add upper and lower halves
    let sum128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    // Horizontal add within 128-bit vector
    let sum64 = _mm_hadd_ps(sum128, sum128);
    let sum32 = _mm_hadd_ps(sum64, sum64);
    _mm_cvtss_f32(sum32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::scalar::ScalarVectorOps;

    #[test]
    fn test_avx2_cosine_similarity() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 tests - not supported on this CPU");
            return;
        }

        let avx2_ops = Avx2VectorOps::new();
        let scalar_ops = ScalarVectorOps::new();

        let a = [0.5f32; 768];
        let b = [0.7f32; 768];

        let avx2_result = avx2_ops.cosine_similarity_768(&a, &b);
        let scalar_result = scalar_ops.cosine_similarity_768(&a, &b);

        assert!(
            (avx2_result - scalar_result).abs() < 1e-6,
            "AVX2: {}, Scalar: {}",
            avx2_result,
            scalar_result
        );
    }

    #[test]
    fn test_avx2_dot_product() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let avx2_ops = Avx2VectorOps::new();
        let scalar_ops = ScalarVectorOps::new();

        let a = [2.0f32; 768];
        let b = [3.0f32; 768];

        let avx2_result = avx2_ops.dot_product_768(&a, &b);
        let scalar_result = scalar_ops.dot_product_768(&a, &b);

        assert!(
            (avx2_result - scalar_result).abs() < 1e-4,
            "AVX2: {}, Scalar: {}",
            avx2_result,
            scalar_result
        );
    }
}

/// Optimized batch cosine similarity with AVX2 and FMA
/// Processes multiple vectors against a single query with better cache utilization
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_similarity_batch_768_avx2_fma(
    query: &[f32; 768],
    vectors: &[[f32; 768]],
) -> Vec<f32> {
    let mut results = Vec::with_capacity(vectors.len());

    // Pre-compute query norm once
    let mut query_norm_sq = _mm256_setzero_ps();
    for chunk_idx in (0..768).step_by(8) {
        let vq = _mm256_loadu_ps(query.as_ptr().add(chunk_idx));
        query_norm_sq = _mm256_fmadd_ps(vq, vq, query_norm_sq);
    }
    let query_norm = horizontal_add_ps_256(query_norm_sq).sqrt();

    if query_norm == 0.0 {
        return vec![0.0; vectors.len()];
    }

    // Process each vector against the query
    for vector in vectors {
        let mut dot_product = _mm256_setzero_ps();
        let mut vector_norm_sq = _mm256_setzero_ps();

        // Prefetch next vector for better cache performance
        let next_idx = results.len() + 1;
        if next_idx < vectors.len() {
            let prefetch_ptr = vectors[next_idx].as_ptr() as *const i8;
            _mm_prefetch(prefetch_ptr, _MM_HINT_T0);
        }

        for chunk_idx in (0..768).step_by(8) {
            let vq = _mm256_loadu_ps(query.as_ptr().add(chunk_idx));
            let vv = _mm256_loadu_ps(vector.as_ptr().add(chunk_idx));

            dot_product = _mm256_fmadd_ps(vq, vv, dot_product);
            vector_norm_sq = _mm256_fmadd_ps(vv, vv, vector_norm_sq);
        }

        let dot_sum = horizontal_add_ps_256(dot_product);
        let vector_norm = horizontal_add_ps_256(vector_norm_sq).sqrt();

        let similarity = if vector_norm == 0.0 {
            0.0
        } else {
            (dot_sum / (query_norm * vector_norm)).clamp(-1.0, 1.0)
        };

        results.push(similarity);
    }

    results
}

/// Optimized batch cosine similarity with AVX2 without FMA
#[target_feature(enable = "avx2")]
unsafe fn cosine_similarity_batch_768_avx2(query: &[f32; 768], vectors: &[[f32; 768]]) -> Vec<f32> {
    let mut results = Vec::with_capacity(vectors.len());

    // Pre-compute query norm once
    let mut query_norm_sq = _mm256_setzero_ps();
    for chunk_idx in (0..768).step_by(8) {
        let vq = _mm256_loadu_ps(query.as_ptr().add(chunk_idx));
        query_norm_sq = _mm256_add_ps(query_norm_sq, _mm256_mul_ps(vq, vq));
    }
    let query_norm = horizontal_add_ps_256(query_norm_sq).sqrt();

    if query_norm == 0.0 {
        return vec![0.0; vectors.len()];
    }

    // Process each vector against the query
    for vector in vectors {
        let mut dot_product = _mm256_setzero_ps();
        let mut vector_norm_sq = _mm256_setzero_ps();

        // Prefetch next vector for better cache performance
        let next_idx = results.len() + 1;
        if next_idx < vectors.len() {
            let prefetch_ptr = vectors[next_idx].as_ptr() as *const i8;
            _mm_prefetch(prefetch_ptr, _MM_HINT_T0);
        }

        for chunk_idx in (0..768).step_by(8) {
            let vq = _mm256_loadu_ps(query.as_ptr().add(chunk_idx));
            let vv = _mm256_loadu_ps(vector.as_ptr().add(chunk_idx));

            dot_product = _mm256_add_ps(dot_product, _mm256_mul_ps(vq, vv));
            vector_norm_sq = _mm256_add_ps(vector_norm_sq, _mm256_mul_ps(vv, vv));
        }

        let dot_sum = horizontal_add_ps_256(dot_product);
        let vector_norm = horizontal_add_ps_256(vector_norm_sq).sqrt();

        let similarity = if vector_norm == 0.0 {
            0.0
        } else {
            (dot_sum / (query_norm * vector_norm)).clamp(-1.0, 1.0)
        };

        results.push(similarity);
    }

    results
}

/// AVX2 FMA accumulate for columnar operations
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_accumulate_avx2(column: &[f32], scalar: f32, accumulator: &mut [f32]) {
    use std::arch::x86_64::*;

    let scalar_vec = _mm256_set1_ps(scalar);
    let len = column.len().min(accumulator.len());
    let chunks = len / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let col_vec = _mm256_loadu_ps(column.as_ptr().add(offset));
        let acc_vec = _mm256_loadu_ps(accumulator.as_ptr().add(offset));
        let result = _mm256_fmadd_ps(col_vec, scalar_vec, acc_vec);
        _mm256_storeu_ps(accumulator.as_mut_ptr().add(offset), result);
    }

    // Handle remainder with scalar operations
    for i in (chunks * 8)..len {
        accumulator[i] += column[i] * scalar;
    }
}

/// AVX2 gather for non-contiguous access
#[target_feature(enable = "avx2")]
unsafe fn gather_f32_avx2(base: &[f32], indices: &[usize]) -> Vec<f32> {
    let mut result = Vec::with_capacity(indices.len());

    // Note: AVX2 gather requires i32 indices, so we need to convert
    // For now, use scalar fallback as gather with 64-bit indices needs special handling
    for &idx in indices {
        if idx < base.len() {
            result.push(base[idx]);
        } else {
            result.push(0.0);
        }
    }

    result
}

/// AVX2 horizontal sum reduction
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(values: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let chunks = values.len() / 8;

    // Sum all chunks
    for i in 0..chunks {
        let offset = i * 8;
        let vec = _mm256_loadu_ps(values.as_ptr().add(offset));
        sum = _mm256_add_ps(sum, vec);
    }

    // Horizontal sum of the AVX register
    // First, sum upper and lower 128-bit lanes
    let sum_128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));

    // Now do horizontal adds within the 128-bit register
    let sum_64 = _mm_hadd_ps(sum_128, sum_128);
    let sum_32 = _mm_hadd_ps(sum_64, sum_64);

    let mut result = _mm_cvtss_f32(sum_32);

    // Add remainder
    for i in (chunks * 8)..values.len() {
        result += values[i];
    }

    result
}
