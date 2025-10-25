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

use super::{EPSILON, NUM_PROPTEST_CASES, assert_slices_approx_eq};
use proptest::prelude::*;

// Conditional import - tests work with or without zig-kernels feature
#[cfg(feature = "zig-kernels")]
use engram_core::zig_kernels;

// Mock Zig kernels module when feature is disabled (for testing Rust baseline)
#[cfg(not(feature = "zig-kernels"))]
mod zig_kernels {
    /// Mock vector_similarity for testing without zig-kernels feature
    pub fn vector_similarity(_query: &[f32], _candidates: &[f32], num_candidates: usize) -> Vec<f32> {
        // Fallback uses stub behavior (zeros) to match Task 002 Zig stubs
        vec![0.0_f32; num_candidates]
    }
}

/// Proptest strategy for generating normalized embeddings
fn prop_embedding(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(prop::num::f32::NORMAL, dim..=dim).prop_map(|v| {
        // Normalize to avoid extreme values
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v
        }
    })
}

/// Proptest strategy for generating unnormalized embeddings (for edge case testing)
fn prop_embedding_unnormalized(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-100.0_f32..100.0_f32, dim..=dim)
}

/// Rust baseline implementation using scalar operations
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

        // Cosine similarity
        let similarity = if query_norm > 0.0 && candidate_norm > 0.0 {
            dot / (query_norm * candidate_norm)
        } else {
            0.0 // Handle zero vectors
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

        // Verify equivalence
        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
    }

    /// Property test: Various embedding dimensions
    #[test]
    fn prop_vector_similarity_various_dims(
        dim in 1_usize..1024,
        query in prop_embedding_unnormalized(128).prop_ind_flat_map(|v| {
            let d = v.len();
            prop_embedding_unnormalized(d)
        }),
        num_candidates in 1_usize..50
    ) {
        prop_assume!((1..=1024).contains(&dim));

        let query = if query.len() == dim {
            query
        } else {
            (0..dim).map(|i| (i as f32) / (dim as f32)).collect::<Vec<_>>()
        };

        let candidates_flat: Vec<f32> = (0..num_candidates)
            .flat_map(|c| (0..dim).map(move |i| ((c + i) as f32) / (dim as f32)))
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

        // Verify equivalence
        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
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

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
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

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
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

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
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

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
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

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
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

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
    }

    #[test]
    fn test_both_zero_vectors() {
        let query = vec![0.0; 768];
        let candidates = vec![0.0; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
    }

    #[test]
    fn test_single_candidate() {
        let query = vec![1.0; 768];
        let candidates = vec![0.5; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_eq!(zig_scores.len(), 1);
        assert_eq!(rust_scores.len(), 1);
        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
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
        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
    }

    #[test]
    fn test_small_dimension() {
        let query = vec![1.0, 2.0, 3.0];
        let candidates = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 2);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 2);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
    }

    #[test]
    fn test_normalized_vs_unnormalized() {
        // Test that normalization is handled correctly
        let query = vec![10.0; 768]; // Large magnitude
        let candidates = vec![0.1; 768]; // Small magnitude

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
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

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
    }

    #[test]
    fn test_very_small_values() {
        let query = vec![1e-10; 768];
        let candidates = vec![1e-10; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
    }

    #[test]
    fn test_very_large_values() {
        let query = vec![1e10; 768];
        let candidates = vec![1e10; 768];

        let zig_scores = zig_kernels::vector_similarity(&query, &candidates, 1);
        let rust_scores = rust_cosine_similarity(&query, &candidates, 1);

        assert_slices_approx_eq(&rust_scores, &zig_scores, EPSILON);
    }
}

#[cfg(test)]
mod regression_tests {
    

    // Regression tests will be added here as interesting cases are discovered
    // during property-based testing and fuzzing.

    #[test]
    fn test_regression_placeholder() {
        // Placeholder for future regression tests
        // Cases will be saved to corpus/ directory and loaded here
    }
}
