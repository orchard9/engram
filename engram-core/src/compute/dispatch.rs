//! Runtime dispatch for vector operations based on CPU capabilities
//!
//! Provides unified interface that automatically selects the best
//! implementation based on detected CPU features.

use super::scalar::ScalarVectorOps;
use super::{CpuCapability, VectorOps, detect_cpu_features};

#[cfg(target_arch = "x86_64")]
use super::avx2::Avx2VectorOps;

#[cfg(target_arch = "x86_64")]
use super::avx512::Avx512VectorOps;

#[cfg(target_arch = "aarch64")]
use super::neon::NeonVectorOps;

/// Dispatching vector operations implementation
pub struct DispatchVectorOps {
    implementation: Box<dyn VectorOps>,
}

impl DispatchVectorOps {
    /// Create new dispatching implementation based on CPU features
    pub fn new() -> Self {
        let implementation: Box<dyn VectorOps> = match detect_cpu_features() {
            #[cfg(target_arch = "x86_64")]
            CpuCapability::Avx512F => Box::new(Avx512VectorOps::new()),
            #[cfg(target_arch = "x86_64")]
            CpuCapability::Avx2Fma | CpuCapability::Avx2 => Box::new(Avx2VectorOps::new()),
            #[cfg(target_arch = "aarch64")]
            CpuCapability::Neon => Box::new(NeonVectorOps::new()),
            _ => Box::new(ScalarVectorOps::new()),
        };

        Self { implementation }
    }
}

impl VectorOps for DispatchVectorOps {
    #[inline]
    fn cosine_similarity_768(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
        self.implementation.cosine_similarity_768(a, b)
    }

    #[inline]
    fn cosine_similarity_batch_768(&self, query: &[f32; 768], vectors: &[[f32; 768]]) -> Vec<f32> {
        self.implementation
            .cosine_similarity_batch_768(query, vectors)
    }

    #[inline]
    fn dot_product_768(&self, a: &[f32; 768], b: &[f32; 768]) -> f32 {
        self.implementation.dot_product_768(a, b)
    }

    #[inline]
    fn l2_norm_768(&self, vector: &[f32; 768]) -> f32 {
        self.implementation.l2_norm_768(vector)
    }

    #[inline]
    fn vector_add_768(&self, a: &[f32; 768], b: &[f32; 768]) -> [f32; 768] {
        self.implementation.vector_add_768(a, b)
    }

    #[inline]
    fn vector_scale_768(&self, vector: &[f32; 768], scale: f32) -> [f32; 768] {
        self.implementation.vector_scale_768(vector, scale)
    }

    #[inline]
    fn weighted_average_768(&self, vectors: &[&[f32; 768]], weights: &[f32]) -> [f32; 768] {
        self.implementation.weighted_average_768(vectors, weights)
    }
}
