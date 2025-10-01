const LANES: usize = 8;

/// SIMD-aware activation mapper that converts similarity scores into activations.
#[derive(Debug, Clone, Copy)]
pub struct SimdActivationMapper {
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
        self,
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
    /// Computes: activations[i] = activations[i] + confidence_weights[i] * path_confidence
    pub fn fma_confidence_aggregate(
        &self,
        activations: &mut [f32],
        confidence_weights: &[f32],
        path_confidence: f32,
    ) {
        let len = activations.len().min(confidence_weights.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    fma_confidence_aggregate_avx2(activations, confidence_weights, path_confidence, len);
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
    use std::arch::x86_64::*;

    let inv_temp = 1.0 / temperature.max(0.05);
    let inv_temp_vec = _mm256_set1_ps(inv_temp);
    let threshold_vec = _mm256_set1_ps(threshold);
    let one_vec = _mm256_set1_ps(1.0);

    let mut output = Vec::with_capacity(similarities.len());
    let chunks = similarities.len() / 8;

    // Process 8 elements at a time with SIMD
    for i in 0..chunks {
        let offset = i * 8;
        let sim_vec = _mm256_loadu_ps(similarities.as_ptr().add(offset));

        // Normalize: (sim - threshold) * inv_temp
        let diff = _mm256_sub_ps(sim_vec, threshold_vec);
        let normalized = _mm256_mul_ps(diff, inv_temp_vec);

        // Compute sigmoid approximation using rational function
        // sigmoid(x) ≈ 1 / (1 + exp(-x))
        // For better performance, use fast approximation
        let neg_norm = _mm256_sub_ps(_mm256_setzero_ps(), normalized);

        // Store and compute sigmoid scalar for now (exp is expensive in SIMD)
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), neg_norm);

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
    use std::arch::x86_64::*;

    let path_vec = _mm256_set1_ps(path_confidence);
    let chunks = len / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let act_vec = _mm256_loadu_ps(activations.as_ptr().add(offset));
        let weight_vec = _mm256_loadu_ps(confidence_weights.as_ptr().add(offset));

        // Fused multiply-add: act + weight * path
        let result = _mm256_fmadd_ps(weight_vec, path_vec, act_vec);
        _mm256_storeu_ps(activations.as_mut_ptr().add(offset), result);
    }

    // Handle remainder
    for i in (chunks * 8)..len {
        activations[i] += confidence_weights[i] * path_confidence;
    }
}

#[cfg(test)]
mod tests {
    use super::SimdActivationMapper;

    #[test]
    fn sigmoid_mapping_produces_bounded_values() {
        let mapper = SimdActivationMapper::new();
        let sims = vec![-0.5, 0.0, 0.4, 0.9, 1.2, -0.2, 0.7, 0.3, 0.8];
        let activations = mapper.batch_sigmoid_activation(&sims, 0.5, 0.1);
        assert_eq!(activations.len(), sims.len());
        for value in activations {
            assert!((0.0..=1.0).contains(&value), "value {value} out of bounds");
        }
    }
}
