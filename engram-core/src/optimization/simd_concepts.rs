//! SIMD helpers for concept-centric operations.

use crate::EMBEDDING_DIM;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn horizontal_sum_avx512(v: __m512) -> f32 {
    // SAFETY: Extract all four 128-bit lanes (16 floats total)
    // AVX-512 has 512 bits = 16x 32-bit floats organized as 4x 128-bit lanes
    let lane0 = _mm512_castps512_ps128(v);           // Lanes 0-3
    let lane1 = _mm512_extractf32x4_ps::<1>(v);      // Lanes 4-7
    let lane2 = _mm512_extractf32x4_ps::<2>(v);      // Lanes 8-11
    let lane3 = _mm512_extractf32x4_ps::<3>(v);      // Lanes 12-15

    // Pairwise addition to combine all lanes
    let sum01 = _mm_add_ps(lane0, lane1);
    let sum23 = _mm_add_ps(lane2, lane3);
    let sum_all = _mm_add_ps(sum01, sum23);

    // Horizontal sum within final 128-bit vector
    let temp64 = _mm_hadd_ps(sum_all, sum_all);
    let temp32 = _mm_hadd_ps(temp64, temp64);
    _mm_cvtss_f32(temp32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
/// Compute cosine similarities between a query embedding and concept centroids using AVX-512.
#[must_use]
pub unsafe fn batch_concept_similarity_avx512(
    query_embedding: &[f32; EMBEDDING_DIM],
    concept_centroids: &[&[f32; EMBEDDING_DIM]],
) -> Vec<f32> {
    let mut results = Vec::with_capacity(concept_centroids.len());

    for centroid in concept_centroids {
        let mut dot = 0.0f32;
        let mut query_norm = 0.0f32;
        let mut centroid_norm = 0.0f32;

        for idx in (0..EMBEDDING_DIM).step_by(16) {
            let q = _mm512_loadu_ps(query_embedding.as_ptr().add(idx));
            let c = _mm512_loadu_ps(centroid.as_ptr().add(idx));

            let prod = _mm512_mul_ps(q, c);
            dot += horizontal_sum_avx512(prod);

            query_norm += horizontal_sum_avx512(_mm512_mul_ps(q, q));
            centroid_norm += horizontal_sum_avx512(_mm512_mul_ps(c, c));
        }

        let norm_product = (query_norm * centroid_norm).sqrt();
        if norm_product <= f32::EPSILON {
            results.push(0.0);
        } else {
            results.push(dot / norm_product);
        }
    }

    results
}

#[cfg(not(target_arch = "x86_64"))]
/// Scalar fallback for platforms without AVX-512 support.
#[must_use]
pub fn batch_concept_similarity_avx512(
    query_embedding: &[f32; EMBEDDING_DIM],
    concept_centroids: &[&[f32; EMBEDDING_DIM]],
) -> Vec<f32> {
    fn cosine(a: &[f32; EMBEDDING_DIM], b: &[f32; EMBEDDING_DIM]) -> f32 {
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..EMBEDDING_DIM {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom <= f32::EPSILON {
            0.0
        } else {
            dot / denom
        }
    }

    concept_centroids
        .iter()
        .map(|centroid| cosine(query_embedding, centroid))
        .collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
/// Apply per-binding decay factors eight values at a time using AVX2.
pub unsafe fn batch_binding_decay_avx2(
    binding_strengths: &mut [f32],
    decay_factors: &[f32],
    dt: f32,
) {
    assert_eq!(binding_strengths.len(), decay_factors.len());

    let len = binding_strengths.len();
    let chunk_count = len / 8;

    let dt_vec = _mm256_set1_ps(dt);
    let one = _mm256_set1_ps(1.0);

    for chunk in 0..chunk_count {
        let offset = chunk * 8;
        let strengths = _mm256_loadu_ps(binding_strengths.as_ptr().add(offset));
        let decays = _mm256_loadu_ps(decay_factors.as_ptr().add(offset));

        let decay_term = _mm256_mul_ps(decays, dt_vec);
        let multiplier = _mm256_sub_ps(one, decay_term);
        let updated = _mm256_mul_ps(strengths, multiplier);
        _mm256_storeu_ps(binding_strengths.as_mut_ptr().add(offset), updated);
    }

    for idx in chunk_count * 8..len {
        binding_strengths[idx] *= 1.0 - decay_factors[idx] * dt;
    }
}

#[cfg(not(target_arch = "x86_64"))]
/// Scalar fallback for the binding decay helper.
pub fn batch_binding_decay_avx2(binding_strengths: &mut [f32], decay_factors: &[f32], dt: f32) {
    assert_eq!(binding_strengths.len(), decay_factors.len());
    for (strength, decay) in binding_strengths.iter_mut().zip(decay_factors) {
        *strength *= 1.0 - decay * dt;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Divide activations by sqrt(fan-out) with AVX2 lanes.
pub unsafe fn batch_fan_effect_division_avx2(activations: &mut [f32], fan_out_counts: &[u32]) {
    assert_eq!(activations.len(), fan_out_counts.len());

    let len = activations.len();
    let chunk_count = len / 8;

    for chunk in 0..chunk_count {
        let offset = chunk * 8;
        let acts = _mm256_loadu_ps(activations.as_ptr().add(offset));
        let mut fan_vals = [1.0f32; 8];
        for i in 0..8 {
            fan_vals[i] = (fan_out_counts[offset + i].max(1) as f32).sqrt().max(1.0);
        }
        let fans = _mm256_loadu_ps(fan_vals.as_ptr());
        let result = _mm256_div_ps(acts, fans);
        _mm256_storeu_ps(activations.as_mut_ptr().add(offset), result);
    }

    for idx in chunk_count * 8..len {
        let denom = (fan_out_counts[idx].max(1) as f32).sqrt().max(1.0);
        activations[idx] /= denom;
    }
}

#[cfg(not(target_arch = "x86_64"))]
/// Scalar fallback for fan-effect normalization.
pub fn batch_fan_effect_division_avx2(activations: &mut [f32], fan_out_counts: &[u32]) {
    assert_eq!(activations.len(), fan_out_counts.len());
    for (activation, count) in activations.iter_mut().zip(fan_out_counts) {
        let denom = (*count).max(1) as f32;
        *activation /= denom.sqrt().max(1.0);
    }
}
