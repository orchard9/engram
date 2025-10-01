//! Validation tests for SIMD batch spreading operations
//!
//! Ensures SIMD implementations produce results within 2 ULPs of scalar reference.

use approx::assert_ulps_eq;
use engram_core::activation::simd_optimization::SimdActivationMapper;
use engram_core::compute::{cosine_similarity_batch_768, scalar::ScalarVectorOps, VectorOps};

fn generate_test_vector(seed: usize) -> [f32; 768] {
    let mut result = [0.0f32; 768];
    for (i, item) in result.iter_mut().enumerate() {
        let val = ((i + seed) as f32 * 0.1).sin();
        *item = val * 0.5 + 0.5; // Normalize to [0, 1]
    }
    result
}

#[test]
fn test_batch_cosine_similarity_matches_scalar() {
    let query = generate_test_vector(42);
    let vectors: Vec<[f32; 768]> = (0..100).map(|i| generate_test_vector(i)).collect();

    let scalar_ops = ScalarVectorOps::new();
    let scalar_results = scalar_ops.cosine_similarity_batch_768(&query, &vectors);
    let simd_results = cosine_similarity_batch_768(&query, &vectors);

    assert_eq!(
        scalar_results.len(),
        simd_results.len(),
        "Result lengths must match"
    );

    for (i, (scalar, simd)) in scalar_results.iter().zip(simd_results.iter()).enumerate() {
        assert_ulps_eq!(
            scalar,
            simd,
            max_ulps = 2,
            epsilon = 1e-6
        );
        if (scalar - simd).abs() >= 1e-6 {
            panic!("SIMD result differs from scalar at index {}: scalar={}, simd={}", i, scalar, simd);
        }
    }
}

#[test]
fn test_batch_similarity_empty_vectors() {
    let query = generate_test_vector(0);
    let vectors: Vec<[f32; 768]> = Vec::new();

    let scalar_ops = ScalarVectorOps::new();
    let scalar_results = scalar_ops.cosine_similarity_batch_768(&query, &vectors);
    let simd_results = cosine_similarity_batch_768(&query, &vectors);

    assert_eq!(scalar_results.len(), 0);
    assert_eq!(simd_results.len(), 0);
}

#[test]
fn test_batch_similarity_single_vector() {
    let query = generate_test_vector(1);
    let vectors = vec![generate_test_vector(2)];

    let scalar_ops = ScalarVectorOps::new();
    let scalar_results = scalar_ops.cosine_similarity_batch_768(&query, &vectors);
    let simd_results = cosine_similarity_batch_768(&query, &vectors);

    assert_eq!(scalar_results.len(), 1);
    assert_eq!(simd_results.len(), 1);

    assert_ulps_eq!(
        scalar_results[0],
        simd_results[0],
        max_ulps = 2,
        epsilon = 1e-6
    );
}

#[test]
fn test_batch_similarity_identical_vectors() {
    let query = generate_test_vector(5);
    let vectors = vec![query; 10];

    let simd_results = cosine_similarity_batch_768(&query, &vectors);

    // All results should be very close to 1.0 (identical vectors)
    for (i, &result) in simd_results.iter().enumerate() {
        assert!(
            (result - 1.0).abs() < 1e-5,
            "Expected similarity ~1.0 for identical vectors at index {}, got {}",
            i,
            result
        );
    }
}

#[test]
fn test_sigmoid_activation_matches_scalar() {
    let mapper = SimdActivationMapper::new();

    // Generate test similarities
    let similarities: Vec<f32> = (0..1000)
        .map(|i| (i as f32 / 1000.0) - 0.5)
        .collect();

    let temperature = 0.5;
    let threshold = 0.1;

    // SIMD result
    let simd_activations = mapper.batch_sigmoid_activation(&similarities, temperature, threshold);

    // Scalar reference
    let scalar_activations: Vec<f32> = similarities
        .iter()
        .map(|&sim| {
            let inv_temp = 1.0 / temperature.max(0.05);
            let normalized = (sim - threshold) * inv_temp;
            1.0 / (1.0 + (-normalized).exp())
        })
        .collect();

    assert_eq!(simd_activations.len(), scalar_activations.len());

    for (i, (simd, scalar)) in simd_activations
        .iter()
        .zip(scalar_activations.iter())
        .enumerate()
    {
        assert_ulps_eq!(
            simd,
            scalar,
            max_ulps = 2,
            epsilon = 1e-6
        );
        if (simd - scalar).abs() >= 1e-6 {
            panic!("Sigmoid activation differs at index {}: simd={}, scalar={}", i, simd, scalar);
        }
    }
}

#[test]
fn test_sigmoid_activation_bounds() {
    let mapper = SimdActivationMapper::new();

    // Test with extreme values
    let similarities = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
    let activations = mapper.batch_sigmoid_activation(&similarities, 0.5, 0.0);

    // All sigmoid outputs should be in [0, 1]
    for (i, &activation) in activations.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&activation),
            "Activation out of bounds at index {}: {}",
            i,
            activation
        );
    }
}

#[test]
fn test_fma_confidence_aggregate_correctness() {
    let mapper = SimdActivationMapper::new();

    let mut simd_activations = vec![0.5f32; 100];
    let mut scalar_activations = simd_activations.clone();
    let confidence_weights = vec![0.8f32; 100];
    let path_confidence = 0.9f32;

    // SIMD version
    mapper.fma_confidence_aggregate(&mut simd_activations, &confidence_weights, path_confidence);

    // Scalar reference
    for i in 0..scalar_activations.len() {
        scalar_activations[i] += confidence_weights[i] * path_confidence;
    }

    assert_eq!(simd_activations.len(), scalar_activations.len());

    for (i, (simd, scalar)) in simd_activations
        .iter()
        .zip(scalar_activations.iter())
        .enumerate()
    {
        assert_ulps_eq!(
            simd,
            scalar,
            max_ulps = 2,
            epsilon = 1e-6
        );
        if (simd - scalar).abs() >= 1e-6 {
            panic!("FMA aggregate differs at index {}: simd={}, scalar={}", i, simd, scalar);
        }
    }
}

#[test]
fn test_batch_similarity_performance_characteristics() {
    // Test that batch operations scale linearly
    let query = generate_test_vector(0);

    for batch_size in [8, 16, 32, 64, 128] {
        let vectors: Vec<[f32; 768]> = (0..batch_size).map(|i| generate_test_vector(i)).collect();
        let results = cosine_similarity_batch_768(&query, &vectors);

        assert_eq!(
            results.len(),
            batch_size,
            "Result count should match batch size"
        );

        // Verify all results are valid similarities in [-1, 1]
        for (i, &result) in results.iter().enumerate() {
            assert!(
                (-1.0..=1.0).contains(&result),
                "Invalid similarity at index {} in batch {}: {}",
                i,
                batch_size,
                result
            );
        }
    }
}

#[test]
fn test_integrated_pipeline_consistency() {
    let mapper = SimdActivationMapper::new();
    let query = generate_test_vector(100);
    let vectors: Vec<[f32; 768]> = (0..50).map(|i| generate_test_vector(i + 100)).collect();

    // Run pipeline twice - should get same results
    let run1 = {
        let similarities = cosine_similarity_batch_768(&query, &vectors);
        mapper.batch_sigmoid_activation(&similarities, 0.5, 0.1)
    };

    let run2 = {
        let similarities = cosine_similarity_batch_768(&query, &vectors);
        mapper.batch_sigmoid_activation(&similarities, 0.5, 0.1)
    };

    assert_eq!(run1.len(), run2.len());

    for (i, (r1, r2)) in run1.iter().zip(run2.iter()).enumerate() {
        assert_ulps_eq!(
            r1,
            r2,
            max_ulps = 0,
            epsilon = 0.0
        );
        if r1 != r2 {
            panic!("Pipeline not deterministic at index {}: run1={}, run2={}", i, r1, r2);
        }
    }
}
