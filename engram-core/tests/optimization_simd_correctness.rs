//! SIMD correctness validation for optimization module
//!
//! Differential testing: SIMD implementations must match scalar reference within epsilon

use engram_core::EMBEDDING_DIM;
use engram_core::optimization::simd_concepts;

const EPSILON: f32 = 1e-5; // Relaxed from 1e-6 for floating-point accumulation

/// Scalar reference implementation for cosine similarity
#[cfg(target_arch = "x86_64")]
fn scalar_cosine_similarity(a: &[f32; EMBEDDING_DIM], b: &[f32; EMBEDDING_DIM]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..EMBEDDING_DIM {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let norm_product = (norm_a * norm_b).sqrt();
    if norm_product <= f32::EPSILON {
        0.0
    } else {
        dot / norm_product
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_avx512_concept_similarity_vs_scalar() {
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("Skipping AVX-512 test: CPU does not support avx512f");
        return;
    }

    // Test vectors
    let query = [0.5f32; EMBEDDING_DIM];
    let centroid1 = [0.5f32; EMBEDDING_DIM];
    let centroid2 = [-0.5f32; EMBEDDING_DIM];
    let centroids = vec![centroid1, centroid2];
    let refs: Vec<_> = centroids.iter().collect();

    // SIMD path
    let simd = unsafe { simd_concepts::batch_concept_similarity_avx512(&query, &refs) };

    // Scalar reference
    let scalar: Vec<f32> = centroids
        .iter()
        .map(|c| scalar_cosine_similarity(&query, c))
        .collect();

    assert_eq!(simd.len(), scalar.len(), "Result length mismatch");

    for (i, (s, r)) in simd.iter().zip(scalar.iter()).enumerate() {
        let diff = (s - r).abs();
        assert!(
            diff < EPSILON,
            "Index {}: SIMD={}, scalar={}, diff={} > epsilon={}",
            i,
            s,
            r,
            diff,
            EPSILON
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_avx512_orthogonal_vectors() {
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("Skipping AVX-512 test: CPU does not support avx512f");
        return;
    }

    // Orthogonal vectors should have similarity ~0
    let mut query = [0.0f32; EMBEDDING_DIM];
    let mut centroid = [0.0f32; EMBEDDING_DIM];

    for i in 0..EMBEDDING_DIM / 2 {
        query[i] = 1.0;
    }
    for i in EMBEDDING_DIM / 2..EMBEDDING_DIM {
        centroid[i] = 1.0;
    }

    let centroids = vec![centroid];
    let refs: Vec<_> = centroids.iter().collect();

    let simd = unsafe { simd_concepts::batch_concept_similarity_avx512(&query, &refs) };
    let scalar = scalar_cosine_similarity(&query, &centroid);

    let diff = (simd[0] - scalar).abs();
    assert!(
        diff < EPSILON,
        "Orthogonal vectors: SIMD={}, scalar={}, diff={}",
        simd[0],
        scalar,
        diff
    );

    // Both should be near zero
    assert!(
        simd[0].abs() < EPSILON,
        "Orthogonal similarity should be ~0, got {}",
        simd[0]
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_avx2_binding_decay_vs_scalar() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test: CPU does not support avx2");
        return;
    }

    let mut strengths_simd = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.9, 0.7, 0.5];
    let mut strengths_scalar = strengths_simd.clone();
    let decays = vec![0.1, 0.2, 0.15, 0.05, 0.3, 0.1, 0.25, 0.2];
    let dt = 0.1;

    // SIMD path
    unsafe {
        simd_concepts::batch_binding_decay_avx2(&mut strengths_simd, &decays, dt);
    }

    // Scalar reference
    for (strength, decay) in strengths_scalar.iter_mut().zip(&decays) {
        *strength *= 1.0 - decay * dt;
    }

    for (i, (s, r)) in strengths_simd.iter().zip(&strengths_scalar).enumerate() {
        let diff = (s - r).abs();
        assert!(
            diff < EPSILON,
            "Binding decay index {}: SIMD={}, scalar={}, diff={}",
            i,
            s,
            r,
            diff
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_avx2_fan_effect_vs_scalar() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping AVX2 test: CPU does not support avx2");
        return;
    }

    let mut acts_simd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut acts_scalar = acts_simd.clone();
    let fan_counts = vec![1, 4, 9, 16, 25, 36, 49, 64];

    // SIMD path
    unsafe {
        simd_concepts::batch_fan_effect_division_avx2(&mut acts_simd, &fan_counts);
    }

    // Scalar reference
    for (act, count) in acts_scalar.iter_mut().zip(&fan_counts) {
        let denom = (*count).max(1) as f32;
        *act /= denom.sqrt().max(1.0);
    }

    for (i, (s, r)) in acts_simd.iter().zip(&acts_scalar).enumerate() {
        let diff = (s - r).abs();
        assert!(
            diff < EPSILON,
            "Fan effect index {}: SIMD={}, scalar={}, diff={}",
            i,
            s,
            r,
            diff
        );
    }
}

#[test]
#[allow(clippy::similar_names)] // centroid1/centroid2/centroids is clear
fn test_simd_fallback_paths() {
    // Test that non-x86 platforms compile and run scalar fallbacks
    let query = [0.5f32; EMBEDDING_DIM];
    let centroid1 = [0.5f32; EMBEDDING_DIM];
    let centroid2 = [-0.5f32; EMBEDDING_DIM];
    let centroids = vec![centroid1, centroid2];
    let refs: Vec<_> = centroids.iter().collect();

    let results = simd_concepts::batch_concept_similarity_avx512(&query, &refs);

    assert_eq!(results.len(), 2);
    // Identical vectors should have similarity 1.0
    assert!((results[0] - 1.0).abs() < EPSILON);
    // Opposite vectors should have similarity -1.0
    assert!((results[1] - (-1.0)).abs() < EPSILON);
}

#[test]
fn test_binding_decay_edge_cases() {
    let mut strengths = vec![0.0, 1.0, 0.5];
    let decays = vec![0.1, 0.0, 1.0]; // No decay, full decay
    let dt = 1.0;

    simd_concepts::batch_binding_decay_avx2(&mut strengths, &decays, dt);

    // 0.0 * (1 - 0.1) = 0.0
    assert!((strengths[0] - 0.0).abs() < EPSILON);
    // 1.0 * (1 - 0.0) = 1.0
    assert!((strengths[1] - 1.0).abs() < EPSILON);
    // 0.5 * (1 - 1.0) = 0.0
    assert!((strengths[2] - 0.0).abs() < EPSILON);
}

#[test]
fn test_fan_effect_zero_fan_out() {
    let mut activations = vec![1.0, 2.0, 3.0];
    let fan_counts = vec![0, 1, 4]; // Zero should be treated as 1

    simd_concepts::batch_fan_effect_division_avx2(&mut activations, &fan_counts);

    // 0 fan-out should be clamped to 1, so no division
    assert!((activations[0] - 1.0).abs() < EPSILON);
    // 1 fan-out: sqrt(1) = 1, so no change
    assert!((activations[1] - 2.0).abs() < EPSILON);
    // 4 fan-out: 3.0 / sqrt(4) = 3.0 / 2.0 = 1.5
    assert!((activations[2] - 1.5).abs() < EPSILON);
}
