//! SIMD-optimized vector operations for Engram memory engine
//!
//! Provides high-performance implementations of vector operations critical to
//! memory retrieval and activation spreading, with runtime CPU feature detection
//! and automatic fallback to scalar implementations.

use std::sync::OnceLock;

pub mod dispatch;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub mod avx512;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(feature = "gpu")]
pub mod cuda;

/// Compute capability detection for optimal implementation selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuCapability {
    #[cfg(target_arch = "x86_64")]
    Avx512F,
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Sse42,
    #[cfg(target_arch = "aarch64")]
    /// ARM NEON SIMD instruction set
    Neon,
    #[cfg(feature = "gpu")]
    /// CUDA GPU acceleration
    Gpu,
    /// Fallback scalar operations without SIMD
    Scalar,
}

/// Trait for vector operations with optimized implementations
pub trait VectorOps: Send + Sync {
    /// Compute cosine similarity between two 768-dimensional vectors
    fn cosine_similarity_768(&self, a: &[f32; 768], b: &[f32; 768]) -> f32;

    /// Batch cosine similarity computation
    fn cosine_similarity_batch_768(&self, query: &[f32; 768], vectors: &[[f32; 768]]) -> Vec<f32>;

    /// Compute dot product of two 768-dimensional vectors
    fn dot_product_768(&self, a: &[f32; 768], b: &[f32; 768]) -> f32;

    /// Compute L2 norm of a 768-dimensional vector
    fn l2_norm_768(&self, vector: &[f32; 768]) -> f32;

    /// Element-wise vector addition
    fn vector_add_768(&self, a: &[f32; 768], b: &[f32; 768]) -> [f32; 768];

    /// Scale vector by scalar
    fn vector_scale_768(&self, vector: &[f32; 768], scale: f32) -> [f32; 768];

    /// Weighted average of multiple vectors
    fn weighted_average_768(&self, vectors: &[&[f32; 768]], weights: &[f32]) -> [f32; 768];

    /// Fused multiply-add for columnar operations
    /// Computes: `accumulator[i] += column[i] * scalar`
    fn fma_accumulate(&self, column: &[f32], scalar: f32, accumulator: &mut [f32]) {
        // Default scalar implementation
        for i in 0..column.len().min(accumulator.len()) {
            accumulator[i] += column[i] * scalar;
        }
    }

    /// SIMD gather for non-contiguous memory access
    fn gather_f32(&self, base: &[f32], indices: &[usize]) -> Vec<f32> {
        // Default scalar implementation
        indices
            .iter()
            .map(|&i| base.get(i).copied().unwrap_or(0.0))
            .collect()
    }

    /// Horizontal sum reduction across values
    fn horizontal_sum(&self, values: &[f32]) -> f32 {
        // Default scalar implementation
        values.iter().sum()
    }

    /// Batch dot product optimized for columnar layout
    fn batch_dot_product_columnar(
        &self,
        query: &[f32; 768],
        columns: &[&[f32]],
        results: &mut [f32],
    ) {
        // Default scalar implementation
        results.fill(0.0);
        for (dim, &column) in columns.iter().enumerate().take(768) {
            let query_val = query[dim];
            for (i, result) in results.iter_mut().enumerate().take(column.len()) {
                *result += column[i] * query_val;
            }
        }
    }
}

/// Runtime CPU feature detection
static CPU_CAPS: OnceLock<CpuCapability> = OnceLock::new();

/// Detect CPU features and return the best available capability
pub fn detect_cpu_features() -> CpuCapability {
    *CPU_CAPS.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
                CpuCapability::Avx512F
            } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                CpuCapability::Avx2Fma
            } else if is_x86_feature_detected!("avx2") {
                CpuCapability::Avx2
            } else if is_x86_feature_detected!("sse4.2") {
                CpuCapability::Sse42
            } else {
                CpuCapability::Scalar
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                CpuCapability::Neon
            } else {
                CpuCapability::Scalar
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            CpuCapability::Scalar
        }
    })
}

/// Create vector operations implementation based on CPU capabilities
#[must_use]
pub fn create_vector_ops() -> Box<dyn VectorOps> {
    #[cfg(feature = "force_scalar_compute")]
    {
        Box::new(scalar::ScalarVectorOps::new())
    }

    #[cfg(not(feature = "force_scalar_compute"))]
    match detect_cpu_features() {
        #[cfg(target_arch = "x86_64")]
        CpuCapability::Avx512F => Box::new(avx512::Avx512VectorOps::new()),
        #[cfg(target_arch = "x86_64")]
        CpuCapability::Avx2Fma | CpuCapability::Avx2 => Box::new(avx2::Avx2VectorOps::new()),
        #[cfg(target_arch = "aarch64")]
        CpuCapability::Neon => Box::new(neon::NeonVectorOps::new()),
        _ => Box::new(scalar::ScalarVectorOps::new()),
    }
}

/// Global vector operations instance
static VECTOR_OPS: OnceLock<Box<dyn VectorOps>> = OnceLock::new();

/// Get the global vector operations instance
pub fn get_vector_ops() -> &'static dyn VectorOps {
    VECTOR_OPS.get_or_init(create_vector_ops).as_ref()
}

/// Convenience function for cosine similarity
#[inline]
#[must_use]
pub fn cosine_similarity_768(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    get_vector_ops().cosine_similarity_768(a, b)
}

/// Convenience function for batch cosine similarity
#[inline]
#[must_use]
pub fn cosine_similarity_batch_768(query: &[f32; 768], vectors: &[[f32; 768]]) -> Vec<f32> {
    get_vector_ops().cosine_similarity_batch_768(query, vectors)
}

/// Runtime validation of implementation correctness
static VALIDATION_PASSED: OnceLock<bool> = OnceLock::new();

/// Validate that SIMD implementations match scalar reference
pub fn validate_implementation() -> bool {
    *VALIDATION_PASSED.get_or_init(|| {
        let test_a = [1.0f32; 768];
        let test_b = [0.5f32; 768];

        let scalar_ops = scalar::ScalarVectorOps::new();
        let scalar_result = scalar_ops.cosine_similarity_768(&test_a, &test_b);
        let simd_result = get_vector_ops().cosine_similarity_768(&test_a, &test_b);

        (scalar_result - simd_result).abs() < 1e-6
    })
}

/// Error types for compute operations
#[derive(Debug, thiserror::Error)]
pub enum ComputeError {
    /// Batch operation buffer has reached maximum capacity
    #[error("Batch capacity exceeded")]
    BatchFull,
    /// Vector has incorrect number of dimensions
    #[error("Invalid vector dimensions: expected 768, got {0}")]
    InvalidDimensions(usize),
    /// Number of weights doesn't match number of vectors in weighted operation
    #[error("Weight and vector count mismatch: {weights} weights for {vectors} vectors")]
    WeightMismatch {
        /// Number of weight values provided
        weights: usize,
        /// Number of vectors provided
        vectors: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_feature_detection() {
        let capability = detect_cpu_features();
        println!("Detected CPU capability: {capability:?}");
        // Should detect something on any platform
        #[cfg(target_arch = "x86_64")]
        {
            assert!(matches!(
                capability,
                CpuCapability::Avx512F
                    | CpuCapability::Avx2Fma
                    | CpuCapability::Avx2
                    | CpuCapability::Sse42
                    | CpuCapability::Scalar
            ));
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert!(matches!(
                capability,
                CpuCapability::Neon | CpuCapability::Scalar
            ));
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            assert!(matches!(capability, CpuCapability::Scalar));
        }
    }

    #[test]
    fn test_implementation_validation() {
        assert!(
            validate_implementation(),
            "SIMD implementation diverged from scalar reference"
        );
    }

    #[test]
    fn test_cosine_similarity_basic() {
        let a = [1.0f32; 768];
        let b = [1.0f32; 768];
        let similarity = cosine_similarity_768(&a, &b);
        assert!(
            (similarity - 1.0).abs() < 1e-6,
            "Identical vectors should have similarity 1.0"
        );

        let c = [-1.0f32; 768];
        let similarity = cosine_similarity_768(&a, &c);
        assert!(
            (similarity + 1.0).abs() < 1e-6,
            "Opposite vectors should have similarity -1.0"
        );
    }
}
