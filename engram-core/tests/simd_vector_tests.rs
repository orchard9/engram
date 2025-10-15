//! Integration tests for SIMD vector operations
//!
//! Validates correctness and performance of SIMD implementations
//! against scalar reference implementations.

use engram_core::compute::scalar::ScalarVectorOps;
use engram_core::compute::{self, VectorOps};

#[cfg(target_arch = "x86_64")]
use engram_core::compute::avx2::Avx2VectorOps;

const EPSILON: f32 = 1e-6;
const BENCH_ITERATIONS: u32 = 10_000;

/// Test that all implementations produce the same results
#[test]
fn test_cosine_similarity_consistency() {
    let scalar_ops = ScalarVectorOps::new();
    let simd_ops = compute::create_vector_ops();

    // Test cases
    let test_cases = vec![
        // Identical vectors
        ([1.0f32; 768], [1.0f32; 768], 1.0),
        // Opposite vectors
        ([1.0f32; 768], [-1.0f32; 768], -1.0),
        // Orthogonal vectors
        (
            {
                let mut a = [0.0f32; 768];
                a[0] = 1.0;
                a
            },
            {
                let mut b = [0.0f32; 768];
                b[1] = 1.0;
                b
            },
            0.0,
        ),
        // Different patterns
        (
            {
                let mut v = [0.0f32; 768];
                for (idx, slot) in v.iter_mut().enumerate() {
                    *slot = if idx % 2 == 0 { 1.0 } else { -1.0 };
                }
                v
            },
            {
                let mut v = [0.0f32; 768];
                for (idx, slot) in v.iter_mut().enumerate() {
                    *slot = if idx % 2 == 0 { -1.0 } else { 1.0 };
                }
                v
            },
            -1.0,
        ), // Alternating opposite patterns
    ];

    for (a, b, expected) in test_cases {
        let scalar_result = scalar_ops.cosine_similarity_768(&a, &b);
        let simd_result = simd_ops.cosine_similarity_768(&a, &b);

        assert!(
            (scalar_result - simd_result).abs() < EPSILON,
            "Results differ: scalar={scalar_result}, simd={simd_result}"
        );

        assert!(
            (scalar_result - expected).abs() < 0.01,
            "Unexpected result: got={scalar_result}, expected={expected}"
        );
    }
}

#[test]
fn test_dot_product_consistency() {
    let scalar_ops = ScalarVectorOps::new();
    let simd_ops = compute::create_vector_ops();

    let test_cases = vec![
        ([2.0f32; 768], [3.0f32; 768], 768.0 * 6.0),
        ([1.0f32; 768], [0.0f32; 768], 0.0),
        ([1.0f32; 768], [-1.0f32; 768], -768.0),
    ];

    for (a, b, expected) in test_cases {
        let scalar_result = scalar_ops.dot_product_768(&a, &b);
        let simd_result = simd_ops.dot_product_768(&a, &b);

        assert!(
            (scalar_result - simd_result).abs() < 0.01,
            "Dot product differs: scalar={scalar_result}, simd={simd_result}"
        );

        assert!(
            (scalar_result - expected).abs() < 0.01,
            "Unexpected dot product: got={scalar_result}, expected={expected}"
        );
    }
}

#[test]
fn test_l2_norm_consistency() {
    let scalar_ops = ScalarVectorOps::new();
    let simd_ops = compute::create_vector_ops();

    // Pythagorean triple test
    let mut vector = [0.0f32; 768];
    vector[0] = 3.0;
    vector[1] = 4.0;

    let scalar_norm = scalar_ops.l2_norm_768(&vector);
    let simd_norm = simd_ops.l2_norm_768(&vector);

    assert!(
        (scalar_norm - simd_norm).abs() < EPSILON,
        "L2 norm differs: scalar={scalar_norm}, simd={simd_norm}"
    );

    assert!(
        (scalar_norm - 5.0).abs() < EPSILON,
        "Unexpected L2 norm: got={scalar_norm}, expected=5.0"
    );
}

#[test]
fn test_batch_cosine_similarity() {
    let scalar_ops = ScalarVectorOps::new();
    let simd_ops = compute::create_vector_ops();

    let query = [1.0f32; 768];
    let vectors = vec![[1.0f32; 768], [-1.0f32; 768], [0.5f32; 768], {
        let mut v = [0.0f32; 768];
        v[0] = 1.0;
        v
    }];

    let scalar_results = scalar_ops.cosine_similarity_batch_768(&query, &vectors);
    let simd_results = simd_ops.cosine_similarity_batch_768(&query, &vectors);

    assert_eq!(scalar_results.len(), simd_results.len());

    for (i, (scalar_res, simd_res)) in scalar_results.iter().zip(simd_results.iter()).enumerate() {
        assert!(
            (scalar_res - simd_res).abs() < EPSILON,
            "Batch result {i} differs: scalar={scalar_res}, simd={simd_res}"
        );
    }
}

#[test]
fn test_vector_operations() {
    let scalar_ops = ScalarVectorOps::new();
    let simd_ops = compute::create_vector_ops();

    let a = [1.0f32; 768];
    let b = [2.0f32; 768];

    // Test addition
    let scalar_add = scalar_ops.vector_add_768(&a, &b);
    let simd_add = simd_ops.vector_add_768(&a, &b);

    for (index, (scalar, simd)) in scalar_add.iter().zip(simd_add.iter()).enumerate() {
        assert!(
            (scalar - simd).abs() < EPSILON,
            "Addition differs at index {index}: scalar={scalar}, simd={simd}"
        );
        assert!(
            (scalar - 3.0).abs() < EPSILON,
            "Expected 3.0 at index {index}, got {scalar}"
        );
    }

    // Test scaling
    let scale = 2.5;
    let scalar_scale = scalar_ops.vector_scale_768(&a, scale);
    let simd_scale = simd_ops.vector_scale_768(&a, scale);

    for (index, (scalar, simd)) in scalar_scale.iter().zip(simd_scale.iter()).enumerate() {
        assert!(
            (scalar - simd).abs() < EPSILON,
            "Scaling differs at index {index}: scalar={scalar}, simd={simd}"
        );
        assert!(
            (scalar - scale).abs() < EPSILON,
            "Expected {scale} at index {index}, got {scalar}"
        );
    }
}

#[test]
fn test_weighted_average() {
    let scalar_ops = ScalarVectorOps::new();
    let simd_ops = compute::create_vector_ops();

    let v1 = [1.0f32; 768];
    let v2 = [3.0f32; 768];
    let vectors: Vec<&[f32; 768]> = vec![&v1, &v2];
    let weights = vec![1.0, 1.0];

    let scalar_avg = scalar_ops.weighted_average_768(&vectors, &weights);
    let simd_avg = simd_ops.weighted_average_768(&vectors, &weights);

    for (index, (scalar, simd)) in scalar_avg.iter().zip(simd_avg.iter()).enumerate() {
        assert!(
            (scalar - simd).abs() < EPSILON,
            "Weighted average differs at index {index}: scalar={scalar}, simd={simd}"
        );
        assert!(
            (scalar - 2.0).abs() < EPSILON,
            "Expected 2.0 at index {index}, got {scalar}"
        );
    }
}

#[test]
fn test_edge_cases() {
    let ops = compute::create_vector_ops();

    // Zero vectors
    let zero = [0.0f32; 768];
    let normal = [1.0f32; 768];

    let similarity = ops.cosine_similarity_768(&zero, &normal);
    assert!(
        similarity.abs() < EPSILON,
        "Zero vector should have 0 similarity, got {similarity}"
    );

    // NaN/Inf handling
    let mut invalid = [1.0f32; 768];
    invalid[0] = f32::NAN;

    // This should not panic, though result may be NaN
    let _ = ops.cosine_similarity_768(&invalid, &normal);
}

#[test]
fn test_performance_validation() {
    // Ensure SIMD is actually being used on supported platforms
    let capability = compute::detect_cpu_features();
    println!("Detected CPU capability: {capability:?}");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            assert!(
                matches!(
                    capability,
                    compute::CpuCapability::Avx2
                        | compute::CpuCapability::Avx2Fma
                        | compute::CpuCapability::Avx512F
                ),
                "AVX2 detected but not selected"
            );
        }
    }
}

#[test]
fn test_implementation_validation() {
    assert!(
        compute::validate_implementation(),
        "SIMD implementation failed validation against scalar reference"
    );
}

/// Benchmark to ensure SIMD provides expected speedup
#[test]
#[ignore] // Benchmark: 10k iterations, ~5-10s runtime. Run with: cargo test --test simd_vector_tests bench_simd_speedup -- --ignored --nocapture
fn bench_simd_speedup() {
    use std::time::Instant;

    let scalar_ops = ScalarVectorOps::new();
    let simd_ops = compute::create_vector_ops();

    let a = [0.707f32; 768];
    let b = [0.707f32; 768];

    // Benchmark scalar
    let start = Instant::now();
    for _ in 0..BENCH_ITERATIONS {
        std::hint::black_box(scalar_ops.cosine_similarity_768(&a, &b));
    }
    let scalar_duration = start.elapsed();

    // Benchmark SIMD
    let start = Instant::now();
    for _ in 0..BENCH_ITERATIONS {
        std::hint::black_box(simd_ops.cosine_similarity_768(&a, &b));
    }
    let simd_duration = start.elapsed();

    let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();

    println!("Scalar: {scalar_duration:?} for {BENCH_ITERATIONS} iterations");
    println!("SIMD: {simd_duration:?} for {BENCH_ITERATIONS} iterations");
    println!("Speedup: {speedup:.2}x");

    // On AVX2 hardware, we should see at least 2x speedup
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            assert!(speedup > 1.5, "SIMD speedup too low: {speedup:.2}x");
        }
    }
}
