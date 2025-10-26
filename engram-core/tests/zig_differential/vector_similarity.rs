//! Differential tests for vector similarity (cosine similarity) kernels.
//!
//! Validates that Zig SIMD implementation produces numerically identical results
//! to the Rust baseline implementation across a wide range of inputs.
//!
//! # Test Coverage
//!
//! 1. **Property-Based Tests** - 10,000 random test cases with various dimensions
//! 2. **Edge Cases** - Zero vectors, orthogonal, parallel, opposite, NaN handling
//! 3. **Boundary Conditions** - Single candidate, many candidates, different dimensions
//!
//! # Current Status (Task 002)
//!
//! Zig kernels are currently stubs that return zeros. These tests will FAIL until
//! Task 005 implements the actual SIMD cosine similarity kernel. This is expected
//! and validates that the differential testing framework correctly detects divergence.

use super::{EPSILON_VECTOR_OPS, NUM_PROPTEST_CASES, assert_slices_approx_eq};
use proptest::prelude::*;

// Conditional import - tests work with or without zig-kernels feature
#[cfg(feature = "zig-kernels")]
use engram_core::zig_kernels;

// Mock Zig kernels module when feature is disabled (for testing Rust baseline)
#[cfg(not(feature = "zig-kernels"))]
mod zig_kernels {
    /// Mock vector_similarity for testing without zig-kernels feature
    pub fn vector_similarity(
        _query: &[f32],
        _candidates: &[f32],
        num_candidates: usize,
    ) -> Vec<f32> {
        // Fallback uses stub behavior (zeros) to match Task 002 Zig stubs
        vec![0.0_f32; num_candidates]
    }
}

/// Proptest strategy for generating normalized embeddings
///
/// Generates non-zero vectors in [-100, 100] range and normalizes them.
/// Filters out zero vectors to ensure valid test inputs.
fn prop_embedding(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-100.0_f32..100.0_f32, dim..=dim)
        .prop_filter("non-zero norm", |v| {
            let norm_sq: f32 = v.iter().map(|x| x * x).sum();
            norm_sq > 1e-10 // Filter out near-zero vectors
        })
        .prop_map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter().map(|x| x / norm).collect()
        })
}

/// Proptest strategy for generating unnormalized embeddings (for edge case testing)
fn prop_embedding_unnormalized(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-100.0_f32..100.0_f32, dim..=dim)
}

/// Rust baseline implementation using scalar operations
///
/// This implementation mirrors the Zig kernel's defensive NaN/Inf handling
/// to ensure differential testing validates correctness, not just compatibility.
fn rust_cosine_similarity(query: &[f32], candidates: &[f32], num_candidates: usize) -> Vec<f32> {
    let dim = query.len();
    assert_eq!(candidates.len(), num_candidates * dim);

    let mut scores = Vec::with_capacity(num_candidates);

    for i in 0..num_candidates {
        let candidate = &candidates[i * dim..(i + 1) * dim];

        // Compute dot product
        let mut dot = 0.0;
        for j in 0..dim {
            dot += query[j] * candidate[j];
        }

        // Compute norms
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let candidate_norm: f32 = candidate.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Handle NaN or Infinity in intermediate results (matches Zig behavior)
        if dot.is_nan() || query_norm.is_nan() || candidate_norm.is_nan() {
            scores.push(0.0);
            continue;
        }
        if dot.is_infinite() || query_norm.is_infinite() || candidate_norm.is_infinite() {
            scores.push(0.0);
            continue;
        }

        // Cosine similarity with denormal flushing
        const DENORMAL_THRESHOLD: f32 = 1e-30;
        let similarity = if query_norm < DENORMAL_THRESHOLD || candidate_norm < DENORMAL_THRESHOLD {
            0.0 // Flush denormals to zero (matches Zig behavior)
        } else {
            let result = dot / (query_norm * candidate_norm);

            // Handle NaN from division
            if result.is_nan() {
                0.0
            } else {
                // Clamp to [-1.0, 1.0] to handle numerical precision errors
                result.clamp(-1.0, 1.0)
            }
        };

        scores.push(similarity);
    }

    scores
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(NUM_PROPTEST_CASES))]

    /// Property test: Zig and Rust implementations produce identical scores
    /// for normalized embeddings with dimension 768
    #[test]
    fn prop_vector_similarity_768(
        query in prop_embedding(768),
        candidates_vec in prop::collection::vec(prop_embedding(768), 1..100)
    ) {
        let num_candidates = candidates_vec.len();

        // Flatten candidates for FFI
        let candidates_flat: Vec<f32> = candidates_vec
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();

        // Call Zig kernel
        let zig_scores = zig_kernels::vector_similarity(
            &query,
            &candidates_flat,
            num_candidates,
        );

        // Call Rust baseline
        let rust_scores = rust_cosine_similarity(
            &query,
            &candidates_flat,
            num_candidates,
        );

        // Verify equivalence (use EPSILON_VECTOR_OPS for sqrt/division operations)
        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
    }

    /// Property test: Various embedding dimensions
    ///
    /// This test generates random dimensions and creates query/candidates with that dimension.
    /// Fixed version that properly generates random vectors for the chosen dimension.
    #[test]
    fn prop_vector_similarity_various_dims(
        dim in 1_usize..1024,
        num_candidates in 1_usize..50,
        seed in 0_u64..1_000_000
    ) {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        // Generate random query and candidates with the same dimension
        let mut rng = StdRng::seed_from_u64(seed);

        let query: Vec<f32> = (0..dim)
            .map(|_| rng.gen_range(-100.0..100.0))
            .collect();

        let candidates_flat: Vec<f32> = (0..num_candidates * dim)
            .map(|_| rng.gen_range(-100.0..100.0))
            .collect();

        // Call Zig kernel
        let zig_scores = zig_kernels::vector_similarity(
            &query,
            &candidates_flat,
            num_candidates,
        );

        // Call Rust baseline
        let rust_scores = rust_cosine_similarity(
            &query,
            &candidates_flat,
            num_candidates,
        );

        // Verify equivalence (use EPSILON_VECTOR_OPS for sqrt/division operations)
        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
    }

    /// Property test: Random unnormalized vectors (stress testing)
    #[test]
    fn prop_vector_similarity_unnormalized(
        query in prop_embedding_unnormalized(384),
        candidates_vec in prop::collection::vec(prop_embedding_unnormalized(384), 1..50)
    ) {
        let num_candidates = candidates_vec.len();
        let candidates_flat: Vec<f32> = candidates_vec
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();

        let zig_scores = zig_kernels::vector_similarity(
            &query,
            &candidates_flat,
            num_candidates,
        );

        let rust_scores = rust_cosine_similarity(
            &query,
            &candidates_flat,
            num_candidates,
        );

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let query: Vec<f32> = (0..768).map(|i| i as f32).collect();
        let candidates = query.clone();

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
        // Should be 1.0 (identical vectors)
        assert!(
            (rust_scores[0] - 1.0).abs() < 1e-5,
            "Identical vectors should have similarity ~1.0, got {}",
            rust_scores[0]
        );
    }

    #[test]
    fn test_orthogonal_vectors() {
        let mut query = vec![0.0; 768];
        query[0] = 1.0;

        let mut candidate = vec![0.0; 768];
        candidate[1] = 1.0;

        let zig_scores = zig_kernels::vector_similarity(&query, &candidate, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidate, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
        // Should be 0.0 (orthogonal vectors)
        assert!(
            rust_scores[0].abs() < 1e-5,
            "Orthogonal vectors should have similarity ~0.0, got {}",
            rust_scores[0]
        );
    }

    #[test]
    fn test_opposite_vectors() {
        let query: Vec<f32> = (0..768).map(|i| i as f32).collect();
        let candidate: Vec<f32> = query.iter().map(|x| -x).collect();

        let zig_scores = zig_kernels::vector_similarity(&query, &candidate, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidate, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
        // Should be -1.0 (opposite vectors)
        assert!(
            (rust_scores[0] + 1.0).abs() < 1e-5,
            "Opposite vectors should have similarity ~-1.0, got {}",
            rust_scores[0]
        );
    }

    #[test]
    fn test_zero_query_vector() {
        let query = vec![0.0; 768];
        let candidates = vec![1.0; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
        // Zero vector should produce 0.0 similarity
        assert!(
            rust_scores[0].abs() < 1e-5,
            "Zero query should have similarity ~0.0, got {}",
            rust_scores[0]
        );
    }

    #[test]
    fn test_zero_candidate_vector() {
        let query = vec![1.0; 768];
        let candidates = vec![0.0; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
    }

    #[test]
    fn test_both_zero_vectors() {
        let query = vec![0.0; 768];
        let candidates = vec![0.0; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
    }

    #[test]
    fn test_single_candidate() {
        let query = vec![1.0; 768];
        let candidates = vec![0.5; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_eq!(zig_scores.len(), 1);
        assert_eq!(rust_scores.len(), 1);
        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
    }

    #[test]
    fn test_many_candidates() {
        let query = vec![1.0; 768];
        let num_candidates = 1000;
        let candidates: Vec<f32> = (0..num_candidates * 768)
            .map(|i| (i % 768) as f32 / 768.0)
            .collect();

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, num_candidates);
        let rust_scores = rust_cosine_similarity(&query, &candidates, num_candidates);

        assert_eq!(zig_scores.len(), num_candidates);
        assert_eq!(rust_scores.len(), num_candidates);
        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
    }

    #[test]
    fn test_small_dimension() {
        let query = vec![1.0, 2.0, 3.0];
        let candidates = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 2);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 2);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
    }

    #[test]
    fn test_normalized_vs_unnormalized() {
        // Test that normalization is handled correctly
        let query = vec![10.0; 768]; // Large magnitude
        let candidates = vec![0.1; 768]; // Small magnitude

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
        // Should be 1.0 (same direction despite different magnitudes)
        assert!(
            (rust_scores[0] - 1.0).abs() < 1e-5,
            "Same direction vectors should have similarity ~1.0, got {}",
            rust_scores[0]
        );
    }

    #[test]
    fn test_mixed_signs() {
        let query: Vec<f32> = (0..768)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let candidate: Vec<f32> = (0..768)
            .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
            .collect();

        let zig_scores = zig_kernels::vector_similarity(&query, &candidate, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidate, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
    }

    #[test]
    fn test_very_small_values() {
        let query = vec![1e-10; 768];
        let candidates = vec![1e-10; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
    }

    #[test]
    fn test_very_large_values() {
        let query = vec![1e10; 768];
        let candidates = vec![1e10; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
    }

    /// Test NaN handling: query vector contains NaN
    #[test]
    fn test_nan_in_query() {
        let mut query = vec![1.0; 768];
        query[100] = f32::NAN;
        let candidates = vec![1.0; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        // Both should handle NaN gracefully (return 0.0)
        assert_eq!(rust_scores.len(), 1);
        assert_eq!(zig_scores.len(), 1);
        assert_eq!(
            rust_scores[0], 0.0,
            "Rust should return 0.0 for NaN in query"
        );
        assert_eq!(zig_scores[0], 0.0, "Zig should return 0.0 for NaN in query");
    }

    /// Test NaN handling: candidate vector contains NaN
    #[test]
    fn test_nan_in_candidate() {
        let query = vec![1.0; 768];
        let mut candidates = vec![1.0; 768];
        candidates[50] = f32::NAN;

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        // Both should handle NaN gracefully (return 0.0)
        assert_eq!(
            rust_scores[0], 0.0,
            "Rust should return 0.0 for NaN in candidate"
        );
        assert_eq!(
            zig_scores[0], 0.0,
            "Zig should return 0.0 for NaN in candidate"
        );
    }

    /// Test NaN handling: multiple NaN values
    #[test]
    fn test_multiple_nan_values() {
        let mut query = vec![1.0; 768];
        query[10] = f32::NAN;
        query[20] = f32::NAN;
        query[30] = f32::NAN;

        let mut candidates = vec![1.0; 768];
        candidates[15] = f32::NAN;
        candidates[25] = f32::NAN;

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_eq!(rust_scores[0], 0.0);
        assert_eq!(zig_scores[0], 0.0);
    }

    /// Test Infinity handling: positive infinity in query
    #[test]
    fn test_inf_in_query() {
        let mut query = vec![1.0; 768];
        query[100] = f32::INFINITY;
        let candidates = vec![1.0; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        // Both should handle infinity gracefully (return 0.0)
        assert_eq!(
            rust_scores[0], 0.0,
            "Rust should return 0.0 for Inf in query"
        );
        assert_eq!(zig_scores[0], 0.0, "Zig should return 0.0 for Inf in query");
    }

    /// Test Infinity handling: negative infinity in candidate
    #[test]
    fn test_neg_inf_in_candidate() {
        let query = vec![1.0; 768];
        let mut candidates = vec![1.0; 768];
        candidates[50] = f32::NEG_INFINITY;

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        // Both should handle negative infinity gracefully (return 0.0)
        assert_eq!(
            rust_scores[0], 0.0,
            "Rust should return 0.0 for -Inf in candidate"
        );
        assert_eq!(
            zig_scores[0], 0.0,
            "Zig should return 0.0 for -Inf in candidate"
        );
    }

    /// Test batch processing with NaN in candidates
    #[test]
    fn test_batch_with_nan_candidates() {
        let query = vec![1.0; 768];

        // Three candidates: valid, NaN, mixed
        let mut candidates = vec![1.0; 768 * 3];
        // Second candidate has NaN
        candidates[768 + 100] = f32::NAN;
        // Third candidate has multiple NaN
        candidates[1536 + 50] = f32::NAN;
        candidates[1536 + 150] = f32::NAN;

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 3);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 3);

        // First should be valid (near 1.0)
        assert!(rust_scores[0] > 0.99);
        assert!(zig_scores[0] > 0.99);

        // Second and third should be 0.0 due to NaN
        assert_eq!(rust_scores[1], 0.0);
        assert_eq!(zig_scores[1], 0.0);
        assert_eq!(rust_scores[2], 0.0);
        assert_eq!(zig_scores[2], 0.0);
    }

    /// Test batch processing with infinity in candidates
    #[test]
    fn test_batch_with_inf_candidates() {
        let query = vec![1.0; 768];

        // Three candidates: valid, +Inf, -Inf
        let mut candidates = vec![1.0; 768 * 3];
        candidates[768 + 100] = f32::INFINITY;
        candidates[1536 + 100] = f32::NEG_INFINITY;

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 3);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 3);

        // First should be valid
        assert!(rust_scores[0] > 0.99);
        assert!(zig_scores[0] > 0.99);

        // Second and third should be 0.0 due to infinity
        assert_eq!(rust_scores[1], 0.0);
        assert_eq!(zig_scores[1], 0.0);
        assert_eq!(rust_scores[2], 0.0);
        assert_eq!(zig_scores[2], 0.0);
    }

    /// Test denormal number handling
    #[test]
    fn test_denormal_handling() {
        // Create vector with magnitude in denormal range
        let query = vec![1e-40_f32; 768];
        let candidates = vec![1e-40_f32; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        // Both should handle denormals gracefully
        // Either flush to zero or compute accurately
        assert!(
            !rust_scores[0].is_nan(),
            "Rust should not produce NaN for denormals"
        );
        assert!(
            !zig_scores[0].is_nan(),
            "Zig should not produce NaN for denormals"
        );
        assert!(rust_scores[0] >= -1.0 && rust_scores[0] <= 1.0);
        assert!(zig_scores[0] >= -1.0 && zig_scores[0] <= 1.0);
    }

    /// Test clamping: ensure results stay in [-1, 1] range
    #[test]
    fn test_result_clamping() {
        // Test many random vectors to ensure results are always in bounds
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let query: Vec<f32> = (0..768).map(|_| rng.gen_range(-100.0..100.0)).collect();
            let candidates: Vec<f32> = (0..768).map(|_| rng.gen_range(-100.0..100.0)).collect();

            let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
            let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

            // Results must be in valid range
            assert!(
                rust_scores[0] >= -1.0 && rust_scores[0] <= 1.0,
                "Rust similarity out of bounds: {}",
                rust_scores[0]
            );
            assert!(
                zig_scores[0] >= -1.0 && zig_scores[0] <= 1.0,
                "Zig similarity out of bounds: {}",
                zig_scores[0]
            );
        }
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    /// Regression test for Issue C2: Zero vector generator bug
    ///
    /// Previously, prop_embedding() could generate zero vectors when random
    /// values canceled out. This caused test flakiness and division by zero.
    #[test]
    fn test_regression_zero_vector_handling() {
        let query = vec![0.0; 768];
        let candidates = vec![0.0; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        // Both should handle zero vectors gracefully (return 0.0)
        assert_eq!(rust_scores.len(), 1);
        assert_eq!(zig_scores.len(), 1);
        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
        assert!(
            rust_scores[0].abs() < 1e-6,
            "Zero vectors should produce 0.0 similarity"
        );
    }

    /// Regression test for Issue H1: Various dimensions test was broken
    ///
    /// Previously, prop_vector_similarity_various_dims generated fixed dimension
    /// 128 instead of varying dimensions. This test verifies the fix works.
    #[test]
    fn test_regression_various_dimensions() {
        // Test a few specific dimensions to ensure it works
        for dim in [1, 3, 16, 64, 256, 384, 512, 768, 1024] {
            let query = vec![1.0; dim];
            let candidates = vec![0.5; dim * 3];

            let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 3);
            let rust_scores = rust_cosine_similarity(&query, &candidates, 3);

            assert_eq!(
                rust_scores.len(),
                3,
                "Should have 3 scores for dimension {dim}"
            );
            assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
        }
    }

    /// Regression test for numerical stability with denormal floats
    #[test]
    fn test_regression_denormal_stability() {
        let query = vec![f32::MIN_POSITIVE; 768];
        let candidates = vec![f32::MIN_POSITIVE; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        // Verify no NaN or Inf
        assert!(!rust_scores[0].is_nan(), "Rust should not produce NaN");
        assert!(!rust_scores[0].is_infinite(), "Rust should not produce Inf");
        assert!(!zig_scores[0].is_nan(), "Zig should not produce NaN");
        assert!(!zig_scores[0].is_infinite(), "Zig should not produce Inf");

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON_VECTOR_OPS);
    }
}
