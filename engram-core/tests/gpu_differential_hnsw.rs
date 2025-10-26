//! Differential Testing: CPU-GPU HNSW Top-K Equivalence
//!
//! Validates that GPU-accelerated HNSW candidate scoring produces
//! identical results to CPU implementation within floating-point tolerance.
//!
//! Test Coverage:
//! - Top-k results match exactly (same indices in same order)
//! - Distances match within 1e-6 tolerance
//! - Edge cases: k=1, k=num_candidates, duplicate distances
//! - Various candidate sizes: [10, 64, 256, 1024, 5000]
//! - Both L2 and cosine distance metrics

#[cfg(cuda_available)]
use engram_core::compute::cuda;

#[cfg(cuda_available)]
use engram_core::compute::cuda::hnsw::{DistanceMetric, cpu_hnsw_top_k, gpu_hnsw_top_k};

// Generate deterministic vector for testing
#[cfg(cuda_available)]
fn generate_vector(seed: usize) -> [f32; 768] {
    let mut vec = [0.0f32; 768];
    for (i, elem) in vec.iter_mut().enumerate() {
        *elem = ((seed * 768 + i) as f32 * 0.001).sin();
    }
    vec
}

// Generate candidate vectors
#[cfg(cuda_available)]
fn generate_candidates(count: usize) -> Vec<[f32; 768]> {
    (0..count).map(generate_vector).collect()
}

#[test]
#[cfg(cuda_available)]
fn test_gpu_cpu_equivalence_cosine_basic() {
    if !cuda::is_available() {
        println!("GPU not available, skipping differential test");
        return;
    }

    let query = generate_vector(0);
    let candidates = generate_candidates(1000);
    let k = 10;

    let cpu_results = cpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine);
    let gpu_results =
        gpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine).expect("GPU top-k failed");

    assert_eq!(
        cpu_results.len(),
        gpu_results.len(),
        "Result count mismatch"
    );

    // Verify top-k indices match exactly
    for i in 0..k {
        assert_eq!(
            cpu_results[i].index, gpu_results[i].index,
            "Top-{} index mismatch: CPU={}, GPU={}",
            i, cpu_results[i].index, gpu_results[i].index
        );

        // Verify distances match within tolerance
        let distance_diff = (cpu_results[i].distance - gpu_results[i].distance).abs();
        assert!(
            distance_diff < 1e-6,
            "Top-{} distance mismatch: CPU={}, GPU={}, diff={}",
            i,
            cpu_results[i].distance,
            gpu_results[i].distance,
            distance_diff
        );
    }

    println!(
        "CPU-GPU cosine equivalence test passed: {} candidates",
        candidates.len()
    );
}

#[test]
#[cfg(cuda_available)]
fn test_gpu_cpu_equivalence_l2_basic() {
    if !cuda::is_available() {
        println!("GPU not available, skipping differential test");
        return;
    }

    let query = generate_vector(0);
    let candidates = generate_candidates(1000);
    let k = 10;

    let cpu_results = cpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::L2);
    let gpu_results =
        gpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::L2).expect("GPU top-k failed");

    assert_eq!(
        cpu_results.len(),
        gpu_results.len(),
        "Result count mismatch"
    );

    // Verify top-k indices match exactly
    for i in 0..k {
        assert_eq!(
            cpu_results[i].index, gpu_results[i].index,
            "Top-{} index mismatch: CPU={}, GPU={}",
            i, cpu_results[i].index, gpu_results[i].index
        );

        // Verify distances match within tolerance
        let distance_diff = (cpu_results[i].distance - gpu_results[i].distance).abs();
        assert!(
            distance_diff < 1e-6,
            "Top-{} distance mismatch: CPU={}, GPU={}, diff={}",
            i,
            cpu_results[i].distance,
            gpu_results[i].distance,
            distance_diff
        );
    }

    println!(
        "CPU-GPU L2 equivalence test passed: {} candidates",
        candidates.len()
    );
}

#[test]
#[cfg(cuda_available)]
fn test_gpu_cpu_edge_case_k1() {
    if !cuda::is_available() {
        println!("GPU not available, skipping differential test");
        return;
    }

    let query = generate_vector(0);
    let candidates = generate_candidates(1000);
    let k = 1; // Edge case: single result

    let cpu_results = cpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine);
    let gpu_results =
        gpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine).expect("GPU top-k failed");

    assert_eq!(cpu_results.len(), 1);
    assert_eq!(gpu_results.len(), 1);
    assert_eq!(
        cpu_results[0].index, gpu_results[0].index,
        "k=1 index mismatch"
    );

    let distance_diff = (cpu_results[0].distance - gpu_results[0].distance).abs();
    assert!(distance_diff < 1e-6, "k=1 distance mismatch");

    println!("k=1 edge case test passed");
}

#[test]
#[cfg(cuda_available)]
fn test_gpu_cpu_edge_case_k_equals_candidates() {
    if !cuda::is_available() {
        println!("GPU not available, skipping differential test");
        return;
    }

    let query = generate_vector(0);
    let candidates = generate_candidates(100);
    let k = 100; // Edge case: k = num_candidates

    let cpu_results = cpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine);
    let gpu_results =
        gpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine).expect("GPU top-k failed");

    assert_eq!(cpu_results.len(), k);
    assert_eq!(gpu_results.len(), k);

    // All candidates should be returned in sorted order
    for i in 0..k {
        assert_eq!(
            cpu_results[i].index, gpu_results[i].index,
            "k=num_candidates index mismatch at position {}",
            i
        );

        let distance_diff = (cpu_results[i].distance - gpu_results[i].distance).abs();
        assert!(
            distance_diff < 1e-6,
            "k=num_candidates distance mismatch at position {}",
            i
        );
    }

    println!("k=num_candidates edge case test passed");
}

#[test]
#[cfg(cuda_available)]
fn test_gpu_cpu_various_candidate_sizes() {
    if !cuda::is_available() {
        println!("GPU not available, skipping differential test");
        return;
    }

    let query = generate_vector(0);
    let k = 10;

    // Test various candidate set sizes
    for &num_candidates in &[10, 64, 256, 1024, 5000] {
        let candidates = generate_candidates(num_candidates);
        let test_k = k.min(num_candidates); // Adjust k if candidates < k

        let cpu_results = cpu_hnsw_top_k(&query, &candidates, test_k, DistanceMetric::Cosine);
        let gpu_results = gpu_hnsw_top_k(&query, &candidates, test_k, DistanceMetric::Cosine)
            .expect("GPU top-k failed");

        assert_eq!(
            cpu_results.len(),
            gpu_results.len(),
            "Size {} result count mismatch",
            num_candidates
        );

        for i in 0..test_k {
            assert_eq!(
                cpu_results[i].index, gpu_results[i].index,
                "Size {} top-{} index mismatch",
                num_candidates, i
            );

            let distance_diff = (cpu_results[i].distance - gpu_results[i].distance).abs();
            assert!(
                distance_diff < 1e-6,
                "Size {} top-{} distance mismatch",
                num_candidates,
                i
            );
        }

        println!("Equivalence test passed for {} candidates", num_candidates);
    }
}

#[test]
#[cfg(cuda_available)]
fn test_gpu_cpu_varying_k() {
    if !cuda::is_available() {
        println!("GPU not available, skipping differential test");
        return;
    }

    let query = generate_vector(0);
    let candidates = generate_candidates(2000);

    // Test various k values
    for &k in &[1, 5, 10, 50, 100, 500, 1000] {
        let cpu_results = cpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine);
        let gpu_results = gpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine)
            .expect("GPU top-k failed");

        assert_eq!(cpu_results.len(), k, "k={} CPU result count mismatch", k);
        assert_eq!(gpu_results.len(), k, "k={} GPU result count mismatch", k);

        for i in 0..k {
            assert_eq!(
                cpu_results[i].index, gpu_results[i].index,
                "k={} top-{} index mismatch",
                k, i
            );

            let distance_diff = (cpu_results[i].distance - gpu_results[i].distance).abs();
            assert!(distance_diff < 1e-6, "k={} top-{} distance mismatch", k, i);
        }

        println!("k={} equivalence test passed", k);
    }
}

#[test]
#[cfg(cuda_available)]
fn test_gpu_cpu_zero_vector_handling() {
    if !cuda::is_available() {
        println!("GPU not available, skipping differential test");
        return;
    }

    // Test with zero query vector (edge case for cosine similarity)
    let query = [0.0f32; 768];
    let mut candidates = generate_candidates(100);

    // Add some zero vectors to candidates
    candidates[0] = [0.0f32; 768];
    candidates[50] = [0.0f32; 768];

    let k = 10;

    let cpu_results = cpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine);
    let gpu_results =
        gpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine).expect("GPU top-k failed");

    assert_eq!(
        cpu_results.len(),
        gpu_results.len(),
        "Zero vector result count mismatch"
    );

    // Verify handling of zero vectors is consistent
    for i in 0..k {
        assert_eq!(
            cpu_results[i].index, gpu_results[i].index,
            "Zero vector top-{} index mismatch",
            i
        );

        let distance_diff = (cpu_results[i].distance - gpu_results[i].distance).abs();
        assert!(
            distance_diff < 1e-6,
            "Zero vector top-{} distance mismatch",
            i
        );
    }

    println!("Zero vector handling test passed");
}

#[test]
#[cfg(cuda_available)]
fn test_gpu_cpu_normalized_vectors() {
    if !cuda::is_available() {
        println!("GPU not available, skipping differential test");
        return;
    }

    // Test with unit-normalized vectors (common in production)
    let mut query = generate_vector(0);
    let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut query {
        *x /= norm;
    }

    let mut candidates = generate_candidates(1000);
    for candidate in &mut candidates {
        let norm: f32 = candidate.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in candidate.iter_mut() {
            *x /= norm;
        }
    }

    let k = 10;

    let cpu_results = cpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine);
    let gpu_results =
        gpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine).expect("GPU top-k failed");

    assert_eq!(
        cpu_results.len(),
        gpu_results.len(),
        "Normalized vector result count mismatch"
    );

    for i in 0..k {
        assert_eq!(
            cpu_results[i].index, gpu_results[i].index,
            "Normalized vector top-{} index mismatch",
            i
        );

        let distance_diff = (cpu_results[i].distance - gpu_results[i].distance).abs();
        assert!(
            distance_diff < 1e-6,
            "Normalized vector top-{} distance mismatch",
            i
        );
    }

    println!("Normalized vector test passed");
}

#[test]
#[cfg(cuda_available)]
fn test_gpu_sorted_order_invariant() {
    if !cuda::is_available() {
        println!("GPU not available, skipping differential test");
        return;
    }

    let query = generate_vector(0);
    let candidates = generate_candidates(2000);
    let k = 50;

    let results =
        gpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine).expect("GPU top-k failed");

    // Verify results are sorted by distance (ascending)
    for i in 1..results.len() {
        assert!(
            results[i - 1].distance <= results[i].distance,
            "GPU results not sorted: position {} distance {} > position {} distance {}",
            i - 1,
            results[i - 1].distance,
            i,
            results[i].distance
        );
    }

    println!("GPU sorted order test passed");
}

#[test]
#[cfg(cuda_available)]
fn test_gpu_no_duplicates() {
    if !cuda::is_available() {
        println!("GPU not available, skipping differential test");
        return;
    }

    let query = generate_vector(0);
    let candidates = generate_candidates(2000);
    let k = 50;

    let results =
        gpu_hnsw_top_k(&query, &candidates, k, DistanceMetric::Cosine).expect("GPU top-k failed");

    // Verify no duplicate indices in top-k
    let mut seen_indices = std::collections::HashSet::new();
    for result in &results {
        assert!(
            seen_indices.insert(result.index),
            "Duplicate index {} in top-k results",
            result.index
        );
    }

    println!("GPU no duplicates test passed");
}
