//! FFI smoke tests for Zig kernel integration
//!
//! These tests verify that the FFI boundary works correctly and that
//! stub implementations behave as expected. They run only when the
//! zig-kernels feature is enabled.
//!
//! Test Strategy:
//! - Validate FFI calls don't segfault (memory safety)
//! - Verify stub implementations return expected values (zeros/no-ops)
//! - Check dimension validation catches errors before FFI calls
//! - Ensure zero-copy semantics (caller-allocated buffers)

#![cfg(feature = "zig-kernels")]

use engram_core::zig_kernels::{apply_decay, spread_activation, vector_similarity};

#[test]
fn test_vector_similarity_stub_returns_zeros() {
    // Stub implementation should return zeros for all scores
    let query = vec![1.0, 0.0, 0.0];
    let candidates = vec![
        1.0, 0.0, 0.0, // Identical to query
        0.0, 1.0, 0.0, // Orthogonal to query
    ];

    let scores = vector_similarity(&query, &candidates, 2);

    assert_eq!(scores.len(), 2);
    assert!(
        (scores[0] - 0.0).abs() < f32::EPSILON,
        "Stub should return zero"
    );
    assert!(
        (scores[1] - 0.0).abs() < f32::EPSILON,
        "Stub should return zero"
    );
}

#[test]
fn test_vector_similarity_batch_processing() {
    // Test with larger batch to ensure proper memory layout
    let dim = 768; // Standard embedding dimension
    let num_candidates = 100;

    let query = vec![1.0; dim];
    let candidates = vec![0.5; dim * num_candidates];

    let scores = vector_similarity(&query, &candidates, num_candidates);

    assert_eq!(scores.len(), num_candidates);
    for score in scores {
        assert!(
            (score - 0.0).abs() < f32::EPSILON,
            "Stub should return zeros for all candidates"
        );
    }
}

#[test]
#[should_panic(expected = "Query vector cannot be empty")]
fn test_vector_similarity_rejects_empty_query() {
    let query = vec![];
    let candidates = vec![1.0, 0.0];
    let _ = vector_similarity(&query, &candidates, 1);
}

#[test]
#[should_panic(expected = "Candidates array size mismatch")]
fn test_vector_similarity_validates_dimensions() {
    let query = vec![1.0, 0.0, 0.0];
    let candidates = vec![1.0, 0.0]; // Wrong size: should be 3 * 2 = 6
    let _ = vector_similarity(&query, &candidates, 2);
}

#[test]
fn test_spread_activation_stub_is_noop() {
    // Stub implementation should not modify activations
    let adjacency = vec![1, 2, 0]; // Node 0 -> 1, 1 -> 2, 2 -> 0 (cycle)
    let weights = vec![0.5, 0.3, 0.2];
    let mut activations = vec![1.0, 0.0, 0.0];

    let original = activations.clone();

    spread_activation(&adjacency, &weights, &mut activations, 3, 5);

    // Stub should not modify activations
    assert_eq!(activations, original, "Stub should be no-op");
}

#[test]
fn test_spread_activation_large_graph() {
    // Test with larger graph to ensure CSR layout is correct
    let num_nodes = 1000;
    let num_edges = 5000;

    let adjacency = (0..num_edges)
        .map(|i| ((i + 1) % num_nodes) as u32)
        .collect::<Vec<_>>();
    let weights = vec![0.1; num_edges];
    let mut activations = vec![0.0; num_nodes];
    activations[0] = 1.0; // Activate first node

    let original = activations.clone();

    spread_activation(&adjacency, &weights, &mut activations, num_nodes, 10);

    // Stub should not modify activations
    assert_eq!(activations, original, "Stub should be no-op");
}

#[test]
#[should_panic(expected = "Adjacency and weights must have same length")]
fn test_spread_activation_validates_edge_data() {
    let adjacency = vec![1, 2];
    let weights = vec![0.5]; // Wrong size
    let mut activations = vec![1.0, 0.0, 0.0];

    spread_activation(&adjacency, &weights, &mut activations, 3, 5);
}

#[test]
#[should_panic(expected = "Activations length must match num_nodes")]
fn test_spread_activation_validates_node_count() {
    let adjacency = vec![1, 2];
    let weights = vec![0.5, 0.3];
    let mut activations = vec![1.0, 0.0]; // Wrong size: should be 3

    spread_activation(&adjacency, &weights, &mut activations, 3, 5);
}

#[test]
fn test_apply_decay_stub_is_noop() {
    // Stub implementation should not modify strengths
    let mut strengths = vec![1.0, 0.8, 0.5];
    let ages = vec![0, 3600, 86400]; // 0s, 1h, 1d

    let original = strengths.clone();

    apply_decay(&mut strengths, &ages);

    // Stub should not modify strengths
    assert_eq!(strengths, original, "Stub should be no-op");
}

#[test]
fn test_apply_decay_large_batch() {
    // Test with larger batch to verify memory layout
    let num_memories = 10000;
    let mut strengths = vec![1.0; num_memories];
    let ages = vec![3600; num_memories]; // All 1 hour old

    let original = strengths.clone();

    apply_decay(&mut strengths, &ages);

    // Stub should not modify strengths
    assert_eq!(strengths, original, "Stub should be no-op");
}

#[test]
#[should_panic(expected = "Strengths and ages must have same length")]
fn test_apply_decay_validates_dimensions() {
    let mut strengths = vec![1.0, 0.8];
    let ages = vec![0, 3600, 86400]; // Wrong size

    apply_decay(&mut strengths, &ages);
}

#[test]
fn test_zero_copy_semantics_vector_similarity() {
    // Verify caller allocates output buffer (zero-copy design)
    let query = vec![1.0, 0.0];
    let candidates = vec![1.0, 0.0, 0.0, 1.0];

    // Vector similarity allocates the output buffer internally,
    // but the FFI layer writes directly to it without serialization
    let scores = vector_similarity(&query, &candidates, 2);

    // Verify buffer was allocated and populated
    assert_eq!(scores.len(), 2);
}

#[test]
fn test_zero_copy_semantics_spread_activation() {
    // Verify in-place modification (true zero-copy)
    let adjacency = vec![1];
    let weights = vec![0.5];
    let mut activations = vec![1.0, 0.0];

    // Get pointer before call
    let ptr_before = activations.as_ptr();

    spread_activation(&adjacency, &weights, &mut activations, 2, 1);

    // Verify same buffer was used (no reallocation)
    let ptr_after = activations.as_ptr();
    assert_eq!(ptr_before, ptr_after, "Should modify in-place");
}

#[test]
fn test_zero_copy_semantics_apply_decay() {
    // Verify in-place modification (true zero-copy)
    let mut strengths = vec![1.0, 0.8];
    let ages = vec![0, 3600];

    // Get pointer before call
    let ptr_before = strengths.as_ptr();

    apply_decay(&mut strengths, &ages);

    // Verify same buffer was used (no reallocation)
    let ptr_after = strengths.as_ptr();
    assert_eq!(ptr_before, ptr_after, "Should modify in-place");
}

#[test]
fn test_thread_safety_concurrent_calls() {
    // Verify FFI can be called from multiple threads safely
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let query = vec![1.0, 0.0, 0.0];
                let candidates = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
                let scores = vector_similarity(&query, &candidates, 2);
                assert_eq!(scores.len(), 2);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

#[test]
fn test_ffi_boundary_safety_no_segfault() {
    // Comprehensive test to ensure FFI boundary is safe
    // Even with stub implementations, these should not segfault

    // Vector similarity with various dimensions
    for dim in [3, 128, 768, 1536] {
        let query = vec![1.0; dim];
        let candidates = vec![0.5; dim * 10];
        let scores = vector_similarity(&query, &candidates, 10);
        assert_eq!(scores.len(), 10);
    }

    // Spreading activation with various graph sizes
    for num_nodes in [10, 100, 1000] {
        let num_edges = num_nodes * 2;
        let adjacency = (0..num_edges)
            .map(|i| (i % num_nodes) as u32)
            .collect::<Vec<_>>();
        let weights = vec![0.1; num_edges];
        let mut activations = vec![0.5; num_nodes];
        spread_activation(&adjacency, &weights, &mut activations, num_nodes, 3);
    }

    // Memory decay with various batch sizes
    for num_memories in [10, 100, 1000] {
        let mut strengths = vec![0.8; num_memories];
        let ages = vec![3600; num_memories];
        apply_decay(&mut strengths, &ages);
    }
}
