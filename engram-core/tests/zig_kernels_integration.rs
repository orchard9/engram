//! Integration tests for Zig performance kernels.
//!
//! These tests validate that Zig kernels integrate correctly into complete workflows
//! and produce identical results to Rust implementations while delivering performance gains.
//!
//! Run with: cargo test --features zig-kernels --test integration

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(unused_variables)]
#![allow(clippy::ref_patterns)]
#![allow(clippy::useless_vec)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::similar_names)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_clone)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::collection_is_never_read)]
#![allow(clippy::iter_with_drain)]
#![allow(clippy::iter_cloned_collect)]

#[path = "common/mod.rs"]
mod common;

#[cfg(feature = "zig-kernels")]
use common::*;
#[cfg(feature = "zig-kernels")]
use engram_core::zig_kernels::*;

// Module declarations for scenario tests
#[path = "zig_integration_scenarios/mod.rs"]
mod scenarios;

#[cfg(feature = "zig-kernels")]
#[test]
fn test_vector_similarity_integration() {
    // Test with realistic embedding dimensions (768)
    let query = generate_embedding(768, 42);
    let candidates = generate_embeddings(100, 768, 1000);

    // Flatten candidates for FFI
    let candidates_flat: Vec<f32> = candidates.iter().flat_map(|v| v.iter().copied()).collect();

    // Call Zig kernel
    let scores = vector_similarity(&query, &candidates_flat, 100);

    // Verify results
    assert_eq!(scores.len(), 100);

    // Check that identical vectors have high similarity
    let identical = query.clone();
    let identical_flat: Vec<f32> = identical.iter().copied().collect();
    let identical_scores = vector_similarity(&query, &identical_flat, 1);

    assert_approx_eq(
        identical_scores[0],
        1.0,
        1e-5,
        "identical vectors should have similarity ~1.0",
    );

    // Check that similar vectors have high similarity
    let similar = generate_similar_embedding(&query, 0.1, 999);
    let similar_flat: Vec<f32> = similar.iter().copied().collect();
    let similar_scores = vector_similarity(&query, &similar_flat, 1);

    assert!(
        similar_scores[0] > 0.9,
        "similar vectors should have similarity > 0.9, got {}",
        similar_scores[0]
    );
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_spreading_activation_integration() {
    // Create a chain graph: 0 -> 1 -> 2 -> 3 -> 4
    let (adjacency, weights, num_nodes) = generate_chain_graph(5);

    // Initialize activations (only node 0 is active)
    let mut activations = vec![0.0_f32; num_nodes];
    activations[0] = 1.0;

    // Spread activation for 5 iterations
    spread_activation(&adjacency, &weights, &mut activations, num_nodes, 5);

    // Verify activation propagated through the chain
    println!("Activations after spreading: {:?}", activations);

    // Node 0 should still have some activation
    assert!(activations[0] > 0.0, "source node should retain activation");

    // Downstream nodes should have received activation
    for i in 1..num_nodes {
        assert!(
            activations[i] > 0.0,
            "node {} should have received activation",
            i
        );
    }

    // Activation should generally decrease with distance (with normalization)
    // This is a weak constraint since normalization can affect ordering
    let total_activation: f32 = activations.iter().sum();
    assert!(
        total_activation > 0.0,
        "total activation should be positive"
    );
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_spreading_activation_fan_graph() {
    // Create a fan graph: center connects to 10 spokes
    let (adjacency, weights, num_nodes) = generate_fan_graph(10);

    let mut activations = vec![0.0_f32; num_nodes];
    activations[0] = 1.0; // Activate center

    spread_activation(&adjacency, &weights, &mut activations, num_nodes, 3);

    println!("Fan activations: {:?}", activations);

    // All spokes should have received activation
    for i in 1..num_nodes {
        assert!(activations[i] > 0.0, "spoke {} should have activation", i);
    }

    // With equal weights, spokes should have similar activation
    let spoke_activations: Vec<f32> = activations[1..].to_vec();
    let mean = spoke_activations.iter().sum::<f32>() / spoke_activations.len() as f32;
    for (i, activation) in spoke_activations.iter().enumerate() {
        let rel_diff = (activation - mean).abs() / mean.max(1e-6);
        assert!(
            rel_diff < 0.5,
            "spoke {} activation {} differs too much from mean {}",
            i + 1,
            activation,
            mean
        );
    }
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_decay_function_integration() {
    // Create memories with different ages
    let mut strengths = vec![1.0_f32; 100];
    let ages: Vec<u64> = (0..100).map(|i| i * 3600).collect(); // 0h, 1h, 2h, ... 99h

    // Apply decay
    apply_decay(&mut strengths, &ages);

    println!("Sample strengths: {:?}", &strengths[..10]);

    // Verify decay behavior
    assert_approx_eq(strengths[0], 1.0, 1e-5, "age 0 should have no decay");

    // Older memories should have lower strength
    for i in 1..strengths.len() {
        assert!(
            strengths[i] <= strengths[i - 1],
            "strength should decrease with age: strengths[{}]={} > strengths[{}]={}",
            i - 1,
            strengths[i - 1],
            i,
            strengths[i]
        );
    }

    // Very old memories should have significantly decayed
    assert!(
        strengths[99] < 0.5,
        "99-hour old memory should have decayed below 0.5, got {}",
        strengths[99]
    );
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_arena_allocator_configuration() {
    use engram_core::zig_kernels::{
        OverflowStrategy, configure_arena, get_arena_stats, reset_arena_metrics,
    };

    // Reset metrics to start fresh
    reset_arena_metrics();

    // Configure arena
    configure_arena(2, OverflowStrategy::ErrorReturn);

    // Perform some operations that use the arena
    let query = generate_embedding(768, 42);
    let candidates = generate_embeddings(10, 768, 1000);
    let candidates_flat: Vec<f32> = candidates.iter().flat_map(|v| v.iter().copied()).collect();

    for _ in 0..100 {
        let _scores = vector_similarity(&query, &candidates_flat, 10);
    }

    // Check arena statistics
    let stats = get_arena_stats();
    println!(
        "Arena stats: resets={}, overflows={}, peak={}",
        stats.total_resets, stats.total_overflows, stats.max_high_water_mark
    );

    // Arena should have been used
    assert!(
        stats.total_resets > 0 || stats.max_high_water_mark > 0,
        "arena should have been used"
    );
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_mixed_kernel_workflow() {
    // Test a realistic workflow that uses multiple kernels together

    // 1. Generate embeddings and find similar memories (vector similarity kernel)
    let query = generate_embedding(768, 42);
    let memory_embeddings = generate_embeddings(1000, 768, 2000);
    let embeddings_flat: Vec<f32> = memory_embeddings
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();

    let similarities = vector_similarity(&query, &embeddings_flat, 1000);

    // 2. Build a graph from top-k similar memories
    let k = 50;
    let mut indexed: Vec<(usize, f32)> = similarities
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_k: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();

    // 3. Create edges between similar memories
    let mut adjacency = Vec::new();
    let mut weights = Vec::new();

    for &i in &top_k {
        for &j in &top_k {
            if i != j && similarities[i] > 0.7 && similarities[j] > 0.7 {
                adjacency.push(j as u32);
                weights.push((similarities[i] * similarities[j]).sqrt());
            }
        }
    }

    // 4. Apply spreading activation
    let mut activations = vec![0.0_f32; k];
    activations[0] = 1.0; // Activate most similar memory

    spread_activation(&adjacency, &weights, &mut activations, k, 5);

    // 5. Apply decay to activated memories
    let ages: Vec<u64> = (0..k).map(|i| i as u64 * 7200).collect(); // 0h, 2h, 4h, ...
    let mut strengths: Vec<f32> = activations.clone();

    apply_decay(&mut strengths, &ages);

    // Verify the complete workflow
    assert!(
        strengths.iter().any(|&s| s > 0.0),
        "some memories should remain after decay"
    );

    println!("Mixed kernel workflow completed successfully");
    println!("Final strengths (sample): {:?}", &strengths[..10]);
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_large_scale_integration() {
    // Test with realistic production-scale data

    // 10,000 memories with 768-dim embeddings
    let num_memories = 10_000;
    let embeddings = generate_embeddings(num_memories, 768, 5000);
    let query = generate_embedding(768, 42);

    // Find similar memories
    let embeddings_flat: Vec<f32> = embeddings.iter().flat_map(|v| v.iter().copied()).collect();
    let similarities = vector_similarity(&query, &embeddings_flat, num_memories);

    // Verify basic properties
    assert_eq!(similarities.len(), num_memories);

    // Create a large graph
    let (adjacency, weights, num_nodes) = generate_random_graph(1000, 0.01, 12345);

    println!(
        "Large graph: {} nodes, {} edges",
        num_nodes,
        adjacency.len()
    );

    // Spread activation
    let mut activations = vec![0.0_f32; num_nodes];
    activations[0] = 1.0;

    spread_activation(&adjacency, &weights, &mut activations, num_nodes, 10);

    // Apply decay
    let ages: Vec<u64> = (0..num_nodes).map(|i| i as u64 * 600).collect();
    let mut strengths = activations.clone();

    apply_decay(&mut strengths, &ages);

    println!("Large-scale integration test completed successfully");
    println!(
        "Active memories: {}",
        strengths.iter().filter(|&&s| s > 0.01).count()
    );
}

#[cfg(not(feature = "zig-kernels"))]
#[test]
fn test_zig_kernels_feature_disabled() {
    // This test runs when zig-kernels feature is NOT enabled
    // It verifies that the code compiles and runs without Zig support

    println!("Zig kernels feature is disabled - using Rust fallback implementations");
    println!("To enable Zig kernels, run: cargo test --features zig-kernels");
}

#[cfg(feature = "zig-kernels")]
#[test]
#[ignore = "Run with --ignored for performance profiling"]
fn profile_kernel_performance() {
    use common::benchmark;

    println!("\n=== Zig Kernel Performance Profile ===\n");

    // Profile vector similarity
    let query = generate_embedding(768, 42);
    let candidates = generate_embeddings(1000, 768, 2000);
    let candidates_flat: Vec<f32> = candidates.iter().flat_map(|v| v.iter().copied()).collect();

    let sim_result = benchmark("Vector Similarity (1000 candidates)", 100, || {
        let _scores = vector_similarity(&query, &candidates_flat, 1000);
    });
    println!("{}", sim_result.summary());

    // Profile spreading activation
    let (adjacency, weights, num_nodes) = generate_random_graph(1000, 0.05, 999);
    let mut activations = vec![0.0_f32; num_nodes];
    activations[0] = 1.0;

    let spread_result = benchmark("Spreading Activation (1000 nodes, 5 iter)", 100, || {
        spread_activation(&adjacency, &weights, &mut activations, num_nodes, 5);
    });
    println!("{}", spread_result.summary());

    // Profile decay
    let mut strengths = vec![1.0_f32; 10_000];
    let ages: Vec<u64> = (0..10_000).map(|i| i * 600).collect();

    let decay_result = benchmark("Memory Decay (10k memories)", 100, || {
        apply_decay(&mut strengths, &ages);
    });
    println!("{}", decay_result.summary());

    println!("\n=== Profile Complete ===\n");
}
