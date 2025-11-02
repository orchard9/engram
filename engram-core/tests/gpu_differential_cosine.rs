//! Differential testing for GPU cosine similarity implementation
//!
//! This test suite ensures CPU-GPU numerical equivalence for cosine similarity
//! computation across various edge cases and batch sizes. All tests require
//! CUDA GPU to be available at runtime.
//!
//! # Test Coverage
//!
//! - CPU-GPU divergence <1e-6 for all test cases
//! - Edge cases: zero vectors, identical vectors, orthogonal vectors
//! - Batch sizes: [1, 16, 64, 256, 1024, 10000]
//! - Property-based testing with random vectors
//! - Numerical stability: accumulation error, normalization edge cases
//!
//! # Success Criteria
//!
//! All tests must pass with divergence <1e-6 to meet Task 003 acceptance criteria.

#![cfg(all(test, feature = "gpu", cuda_available))]

use engram_core::compute::cuda::cosine_similarity::GpuCosineSimilarity;
use engram_core::compute::scalar::ScalarVectorOps;
use engram_core::compute::{VectorOps, cuda};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const TOLERANCE: f32 = 1e-6;

/// Generate random 768-dimensional vector with values in [-1, 1]
fn random_vector_768(rng: &mut impl Rng) -> [f32; 768] {
    let mut vec = [0.0f32; 768];
    for val in &mut vec {
        *val = rng.gen_range(-1.0..1.0);
    }
    vec
}

/// Generate batch of random 768-dimensional vectors
fn random_batch_768(count: usize, rng: &mut impl Rng) -> Vec<[f32; 768]> {
    (0..count).map(|_| random_vector_768(rng)).collect()
}

/// Check if GPU is available, skip test if not
fn require_gpu() -> bool {
    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return false;
    }
    true
}

/// Assert CPU and GPU results match within tolerance
fn assert_results_match(cpu: &[f32], gpu: &[f32], context: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{}: Length mismatch", context);

    let mut max_divergence = 0.0f32;
    let mut max_divergence_idx = 0;

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let divergence = (c - g).abs();
        if divergence > max_divergence {
            max_divergence = divergence;
            max_divergence_idx = i;
        }

        assert!(
            divergence < TOLERANCE,
            "{}: Divergence {:.2e} at index {} exceeds tolerance {:.2e} (CPU={}, GPU={})",
            context,
            divergence,
            i,
            TOLERANCE,
            c,
            g
        );
    }

    println!(
        "{}: Max divergence {:.2e} at index {} (within tolerance)",
        context, max_divergence, max_divergence_idx
    );
}

#[test]
fn test_identical_vectors() {
    if !require_gpu() {
        return;
    }

    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let query = [1.0f32; 768];
    let targets = vec![[1.0f32; 768]; 256];

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Identical vectors");

    // All similarities should be 1.0
    for &sim in &gpu_result {
        assert!((sim - 1.0).abs() < TOLERANCE, "Expected 1.0, got {}", sim);
    }
}

#[test]
fn test_opposite_vectors() {
    if !require_gpu() {
        return;
    }

    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let query = [1.0f32; 768];
    let targets = vec![[-1.0f32; 768]; 256];

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Opposite vectors");

    // All similarities should be -1.0
    for &sim in &gpu_result {
        assert!((sim + 1.0).abs() < TOLERANCE, "Expected -1.0, got {}", sim);
    }
}

#[test]
fn test_orthogonal_vectors() {
    if !require_gpu() {
        return;
    }

    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    // Query: unit vector in first dimension
    let mut query = [0.0f32; 768];
    query[0] = 1.0;

    // Targets: unit vectors in different dimensions (orthogonal to query)
    let mut targets = Vec::new();
    for i in 1..257 {
        let mut vec = [0.0f32; 768];
        vec[i % 768] = 1.0;
        if i % 768 != 0 {
            // Skip dimension 0 (parallel to query)
            targets.push(vec);
        }
    }

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Orthogonal vectors");

    // All similarities should be ~0.0
    for &sim in &gpu_result {
        assert!(sim.abs() < TOLERANCE, "Expected ~0.0, got {}", sim);
    }
}

#[test]
fn test_zero_query_vector() {
    if !require_gpu() {
        return;
    }

    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let query = [0.0f32; 768]; // Zero vector
    let targets = vec![[1.0f32; 768]; 256];

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Zero query vector");

    // All similarities should be 0.0
    for &sim in &gpu_result {
        assert_eq!(sim, 0.0, "Expected 0.0, got {}", sim);
    }
}

#[test]
fn test_zero_target_vectors() {
    if !require_gpu() {
        return;
    }

    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let query = [1.0f32; 768];
    let targets = vec![[0.0f32; 768]; 256]; // Zero vectors

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Zero target vectors");

    // All similarities should be 0.0
    for &sim in &gpu_result {
        assert_eq!(sim, 0.0, "Expected 0.0, got {}", sim);
    }
}

#[test]
fn test_mixed_zero_nonzero() {
    if !require_gpu() {
        return;
    }

    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let query = [1.0f32; 768];
    let mut targets = Vec::new();

    // Mix of zero and non-zero vectors
    for i in 0..256 {
        if i % 2 == 0 {
            targets.push([0.0f32; 768]);
        } else {
            targets.push([1.0f32; 768]);
        }
    }

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Mixed zero/non-zero");
}

#[test]
fn test_random_vectors_batch_16() {
    if !require_gpu() {
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let query = random_vector_768(&mut rng);
    let targets = random_batch_768(16, &mut rng);

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Random batch size 16");
}

#[test]
fn test_random_vectors_batch_64() {
    if !require_gpu() {
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let query = random_vector_768(&mut rng);
    let targets = random_batch_768(64, &mut rng);

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Random batch size 64");
}

#[test]
fn test_random_vectors_batch_256() {
    if !require_gpu() {
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(456);
    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let query = random_vector_768(&mut rng);
    let targets = random_batch_768(256, &mut rng);

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Random batch size 256");
}

#[test]
fn test_random_vectors_batch_1024() {
    if !require_gpu() {
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(789);
    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let query = random_vector_768(&mut rng);
    let targets = random_batch_768(1024, &mut rng);

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Random batch size 1024");
}

#[test]
fn test_random_vectors_batch_10000() {
    if !require_gpu() {
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(999);
    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let query = random_vector_768(&mut rng);
    let targets = random_batch_768(10000, &mut rng);

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Random batch size 10000");
}

#[test]
fn test_very_small_values() {
    if !require_gpu() {
        return;
    }

    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    // Test numerical stability with very small values
    let query = [1e-10f32; 768];
    let targets = vec![[1e-10f32; 768]; 256];

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Very small values");
}

#[test]
fn test_very_large_values() {
    if !require_gpu() {
        return;
    }

    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    // Test numerical stability with very large values
    let query = [1e10f32; 768];
    let targets = vec![[1e10f32; 768]; 256];

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Very large values");
}

#[test]
fn test_mixed_magnitudes() {
    if !require_gpu() {
        return;
    }

    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let mut query = [0.0f32; 768];
    let mut targets = Vec::new();

    // Query with mixed magnitudes
    for i in 0..768 {
        query[i] = if i % 2 == 0 { 1e-5 } else { 1e5 };
    }

    // Targets with different magnitude patterns
    for batch_idx in 0..256 {
        let mut vec = [0.0f32; 768];
        for i in 0..768 {
            vec[i] = if (i + batch_idx) % 3 == 0 { 1e-3 } else { 1e3 };
        }
        targets.push(vec);
    }

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Mixed magnitudes");
}

#[test]
fn test_normalized_vectors() {
    if !require_gpu() {
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(321);
    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    // Generate random vectors and normalize them
    let mut query = random_vector_768(&mut rng);
    let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut query {
        *val /= norm;
    }

    let mut targets = random_batch_768(256, &mut rng);
    for target in &mut targets {
        let norm: f32 = target.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in target {
            *val /= norm;
        }
    }

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Normalized vectors");
}

#[test]
fn test_sparse_vectors() {
    if !require_gpu() {
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(654);
    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    // Sparse vectors: only 10% of dimensions are non-zero
    let mut query = [0.0f32; 768];
    for i in (0..768).step_by(10) {
        query[i] = rng.gen_range(-1.0..1.0);
    }

    let mut targets = Vec::new();
    for _ in 0..256 {
        let mut vec = [0.0f32; 768];
        for i in (0..768).step_by(10) {
            vec[i] = rng.gen_range(-1.0..1.0);
        }
        targets.push(vec);
    }

    let cpu_result = cpu_ops.cosine_similarity_batch_768(&query, &targets);
    let gpu_result = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    assert_results_match(&cpu_result, &gpu_result, "Sparse vectors");
}

#[test]
fn test_property_associativity() {
    if !require_gpu() {
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(111);
    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let query = random_vector_768(&mut rng);
    let targets = random_batch_768(256, &mut rng);

    // Test that order doesn't matter
    let result1 = gpu_ops.cosine_similarity_batch_768(&query, &targets);

    // Compute same batch in different order
    let mut shuffled_targets = targets.clone();
    shuffled_targets.reverse();
    let result2 = gpu_ops.cosine_similarity_batch_768(&query, &shuffled_targets);

    // Results should match original in reversed order
    for (i, &val) in result2.iter().enumerate() {
        let original_val = result1[targets.len() - 1 - i];
        assert!(
            (val - original_val).abs() < TOLERANCE,
            "Associativity failed at index {}: {} != {}",
            i,
            val,
            original_val
        );
    }
}

#[test]
fn test_single_vector_consistency() {
    if !require_gpu() {
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(222);
    let cpu_ops = ScalarVectorOps::new();
    let gpu_ops = GpuCosineSimilarity::new();

    let query = random_vector_768(&mut rng);
    let target = random_vector_768(&mut rng);

    // Single vector similarity
    let single_result = cpu_ops.cosine_similarity_768(&query, &target);

    // Batch of one
    let batch_result = gpu_ops.cosine_similarity_batch_768(&query, &[target]);

    assert_eq!(batch_result.len(), 1);
    assert!(
        (single_result - batch_result[0]).abs() < TOLERANCE,
        "Single vector consistency: {} != {}",
        single_result,
        batch_result[0]
    );
}
