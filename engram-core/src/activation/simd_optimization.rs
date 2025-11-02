#[allow(unsafe_code)]
const LANES: usize = 8;
const TILE_DIM: usize = 96; // 768 / 8 = 96 tiles

/// Cache-aligned batch structure for SIMD processing (AoSoA layout)
/// Stores embeddings in Array-of-Structures-of-Arrays format for optimal SIMD access
#[repr(align(64))]
#[derive(Clone)]
pub struct ActivationBatch {
    /// Embeddings stored in AoSoA layout: [dim][lane]
    /// Each dimension has LANES elements grouped together for SIMD access
    embeddings: Vec<[f32; LANES]>,
    /// Number of valid embeddings in this batch
    count: usize,
}

impl ActivationBatch {
    /// Create new activation batch with specified capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        // Round up to next multiple of LANES for alignment
        let aligned_capacity = capacity.div_ceil(LANES) * LANES;
        Self {
            embeddings: vec![[0.0; LANES]; TILE_DIM * aligned_capacity / LANES],
            count: 0,
        }
    }

    /// Add an embedding to the batch
    /// Returns true if successful, false if batch is full
    pub fn push(&mut self, embedding: &[f32; 768]) -> bool {
        if self.count >= self.capacity() {
            return false;
        }

        let lane_idx = self.count % LANES;
        let batch_offset = (self.count / LANES) * TILE_DIM;

        // Copy embedding into AoSoA layout
        for (dim, &value) in embedding.iter().enumerate() {
            let tile_idx = dim / LANES;
            let tile_offset = batch_offset + tile_idx;
            self.embeddings[tile_offset][lane_idx] = value;
        }

        self.count += 1;
        true
    }

    /// Get the number of embeddings in this batch
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Check if batch is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the capacity of this batch
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.embeddings.len() / TILE_DIM * LANES
    }

    /// Clear the batch for reuse
    pub const fn clear(&mut self) {
        self.count = 0;
    }

    /// Get embeddings in standard [f32; 768] format for batch processing
    #[must_use]
    pub fn as_standard_vectors(&self) -> Vec<[f32; 768]> {
        let mut result = Vec::with_capacity(self.count);

        for i in 0..self.count {
            let mut embedding = [0.0f32; 768];
            let lane_idx = i % LANES;
            let batch_offset = (i / LANES) * TILE_DIM;

            for (dim, item) in embedding.iter_mut().enumerate() {
                let tile_idx = dim / LANES;
                let tile_offset = batch_offset + tile_idx;
                *item = self.embeddings[tile_offset][lane_idx];
            }

            result.push(embedding);
        }

        result
    }

    /// Verify alignment for safe SIMD operations
    #[must_use]
    pub fn is_aligned(&self) -> bool {
        let ptr = self.embeddings.as_ptr() as usize;
        ptr.is_multiple_of(64)
    }

    /// Assert alignment for safety-critical operations
    ///
    /// # Panics
    ///
    /// Panics if the batch is not properly aligned
    pub fn assert_alignment(&self) {
        assert!(
            self.is_aligned(),
            "ActivationBatch not aligned to 64-byte boundary. Pointer: {:p}",
            self.embeddings.as_ptr()
        );
    }
}

/// Determines if SIMD should be used based on storage tier and batch size
#[must_use]
pub const fn should_use_simd_for_tier(
    tier: crate::activation::storage_aware::StorageTier,
    batch_size: usize,
    min_batch_size: usize,
) -> bool {
    use crate::activation::storage_aware::StorageTier;

    // Don't use SIMD for batches smaller than threshold
    if batch_size < min_batch_size {
        return false;
    }

    // Tier-aware SIMD selection
    match tier {
        StorageTier::Hot => true, // Always use SIMD for hot tier
        StorageTier::Warm => batch_size >= min_batch_size * 2, // Higher threshold for warm
        StorageTier::Cold => false, // Never use SIMD for cold tier (bandwidth limited)
    }
}

/// SIMD-aware activation mapper that converts similarity scores into activations.
#[derive(Debug, Clone, Copy)]
pub struct SimdActivationMapper {
    /// Number of SIMD lanes reserved for future vectorized operations metadata
    #[allow(dead_code)]
    lanes: usize,
}

impl SimdActivationMapper {
    /// Creates a new SIMD activation mapper
    #[must_use]
    pub const fn new() -> Self {
        Self { lanes: LANES }
    }

    /// Map similarities into activations using temperature-scaled sigmoid.
    /// Uses SIMD when available for better performance.
    #[must_use]
    pub fn batch_sigmoid_activation(
        similarities: &[f32],
        temperature: f32,
        threshold: f32,
    ) -> Vec<f32> {
        if similarities.is_empty() {
            return Vec::new();
        }

        // Try SIMD path first
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { batch_sigmoid_avx2_fma(similarities, temperature, threshold) };
            }
            if is_x86_feature_detected!("avx2") {
                return unsafe { batch_sigmoid_avx2(similarities, temperature, threshold) };
            }
        }

        // Scalar fallback
        batch_sigmoid_scalar(similarities, temperature, threshold)
    }

    /// Fused multiply-add for confidence aggregation with SIMD acceleration.
    /// Computes: `activations[i] = activations[i] + confidence_weights[i] * path_confidence`
    pub fn fma_confidence_aggregate(
        activations: &mut [f32],
        confidence_weights: &[f32],
        path_confidence: f32,
    ) {
        let len = activations.len().min(confidence_weights.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    fma_confidence_aggregate_avx2(
                        activations,
                        confidence_weights,
                        path_confidence,
                        len,
                    );
                }
                return;
            }
        }

        // Scalar fallback
        for i in 0..len {
            activations[i] += confidence_weights[i] * path_confidence;
        }
    }
}

impl Default for SimdActivationMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// Scalar fallback for batch sigmoid activation
fn batch_sigmoid_scalar(similarities: &[f32], temperature: f32, threshold: f32) -> Vec<f32> {
    let inv_temp = 1.0 / temperature.max(0.05);
    similarities
        .iter()
        .map(|&value| {
            let normalized = (value - threshold) * inv_temp;
            sigmoid(normalized)
        })
        .collect()
}

/// AVX2 implementation with FMA for batch sigmoid activation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn batch_sigmoid_avx2_fma(
    similarities: &[f32],
    temperature: f32,
    threshold: f32,
) -> Vec<f32> {
    use std::arch::x86_64::{
        _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };

    let inv_temp = 1.0 / temperature.max(0.05);
    let inv_temp_vec = _mm256_set1_ps(inv_temp);
    let threshold_vec = _mm256_set1_ps(threshold);

    let mut output = Vec::with_capacity(similarities.len());
    let chunks = similarities.len() / 8;

    // Process 8 elements at a time with SIMD
    for i in 0..chunks {
        let offset = i * 8;
        let sim_vec = unsafe { _mm256_loadu_ps(similarities.as_ptr().add(offset)) };

        // Normalize: (sim - threshold) * inv_temp
        let diff = _mm256_sub_ps(sim_vec, threshold_vec);
        let normalized = _mm256_mul_ps(diff, inv_temp_vec);

        // Compute sigmoid approximation using rational function
        // sigmoid(x) ≈ 1 / (1 + exp(-x))
        // For better performance, use fast approximation
        let neg_norm = _mm256_sub_ps(_mm256_setzero_ps(), normalized);

        // Store and compute sigmoid scalar for now (exp is expensive in SIMD)
        let mut temp = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(temp.as_mut_ptr(), neg_norm) };

        for &val in &temp {
            output.push(1.0 / (1.0 + val.exp()));
        }
    }

    // Handle remainder with scalar operations
    for &value in &similarities[(chunks * 8)..] {
        let normalized = (value - threshold) * inv_temp;
        output.push(sigmoid(normalized));
    }

    output
}

/// AVX2 implementation without FMA for batch sigmoid activation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_sigmoid_avx2(similarities: &[f32], temperature: f32, threshold: f32) -> Vec<f32> {
    // For now, just use scalar fallback as sigmoid exp is complex in SIMD
    // This can be optimized later with polynomial approximations
    batch_sigmoid_scalar(similarities, temperature, threshold)
}

/// AVX2 FMA implementation for confidence aggregation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_confidence_aggregate_avx2(
    activations: &mut [f32],
    confidence_weights: &[f32],
    path_confidence: f32,
    len: usize,
) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    let path_vec = _mm256_set1_ps(path_confidence);
    let chunks = len / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let act_vec = unsafe { _mm256_loadu_ps(activations.as_ptr().add(offset)) };
        let weight_vec = unsafe { _mm256_loadu_ps(confidence_weights.as_ptr().add(offset)) };

        // Fused multiply-add: act + weight * path
        let result = _mm256_fmadd_ps(weight_vec, path_vec, act_vec);
        unsafe { _mm256_storeu_ps(activations.as_mut_ptr().add(offset), result) };
    }

    // Handle remainder
    for i in (chunks * 8)..len {
        activations[i] += confidence_weights[i] * path_confidence;
    }
}

/// Auto-tune SIMD batch size by benchmarking different configurations
/// Returns the optimal batch size based on performance measurements
#[must_use]
pub fn auto_tune_batch_size() -> usize {
    use crate::compute::cosine_similarity_batch_768;
    use std::time::Instant;

    // Test batch sizes
    let batch_sizes = [8, 16, 32];
    let iterations = 100;

    // Generate test data
    let query = [0.5f32; 768];
    let mut best_batch_size = 8;
    let mut best_duration = std::time::Duration::MAX;

    for &batch_size in &batch_sizes {
        let vectors: Vec<[f32; 768]> = (0..batch_size)
            .map(|i| {
                let mut v = [0.0f32; 768];
                for (j, item) in v.iter_mut().enumerate() {
                    let idx = (i + j) as f32;
                    *item = (idx * 0.01).sin();
                }
                v
            })
            .collect();

        // Benchmark this batch size
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cosine_similarity_batch_768(&query, &vectors);
        }
        let duration = start.elapsed();

        // Normalize by batch size (throughput metric)
        let per_vector_duration = duration / (batch_size as u32 * iterations);

        if per_vector_duration < best_duration / best_batch_size as u32 {
            best_duration = per_vector_duration * best_batch_size as u32;
            best_batch_size = batch_size;
        }
    }

    best_batch_size
}

#[cfg(test)]
mod tests {
    use super::SimdActivationMapper;

    #[test]
    fn sigmoid_mapping_produces_bounded_values() {
        let sims = vec![-0.5, 0.0, 0.4, 0.9, 1.2, -0.2, 0.7, 0.3, 0.8];
        let activations = SimdActivationMapper::batch_sigmoid_activation(&sims, 0.5, 0.1);
        assert_eq!(activations.len(), sims.len());
        for value in activations {
            assert!((0.0..=1.0).contains(&value), "value {value} out of bounds");
        }
    }

    #[test]
    fn test_auto_tune_batch_size() {
        let optimal_size = super::auto_tune_batch_size();
        assert!(
            [8, 16, 32].contains(&optimal_size),
            "Auto-tuned batch size should be one of the test sizes"
        );
    }
}
