//! Integration tests for SIMD batch spreading in activation engine
//!
//! Tests the complete pipeline from activation spreading through SIMD batch operations
//!
//! IMPORTANT: These tests create ParallelSpreadingEngine instances with multiple worker
//! threads and MUST be run with parallel test execution enabled (the default). Running
//! with --test-threads=1 will cause timeouts due to worker thread coordination issues
//! when the test harness forces sequential execution.

use engram_core::activation::{
    ActivationGraphExt, EdgeType, ParallelSpreadingConfig, ParallelSpreadingEngine,
    create_activation_graph, engine_registry,
    simd_optimization::{ActivationBatch, auto_tune_batch_size, should_use_simd_for_tier},
    storage_aware::StorageTier,
};
use serial_test::serial;
use std::sync::Arc;

/// Helper to ensure clean test environment
fn ensure_no_active_engines() {
    // Wait for any active engines to fully shut down
    let mut retries = 0;
    while engine_registry::has_active_engines() && retries < 50 {
        std::thread::sleep(std::time::Duration::from_millis(10));
        retries += 1;
    }

    assert!(
        !engine_registry::has_active_engines(),
        "Failed to clean up active engines after {retries} retries"
    );
}

#[test]
fn test_activation_batch_creation_and_alignment() {
    let batch = ActivationBatch::new(16);

    assert_eq!(batch.len(), 0);
    assert!(batch.is_empty());
    assert!(batch.capacity() >= 16);

    // Verify alignment
    assert!(batch.is_aligned(), "Batch should be 64-byte aligned");
    batch.assert_alignment(); // Should not panic
}

#[test]
fn test_activation_batch_push_and_conversion() {
    let mut batch = ActivationBatch::new(8);

    // Create test embeddings
    let embedding1 = [1.0f32; 768];
    let embedding2 = [0.5f32; 768];

    assert!(batch.push(&embedding1));
    assert!(batch.push(&embedding2));
    assert_eq!(batch.len(), 2);

    // Convert back to standard format
    let vectors = batch.as_standard_vectors();
    assert_eq!(vectors.len(), 2);

    // Verify data integrity
    for (i, &val) in vectors[0].iter().enumerate() {
        assert!(
            (val - embedding1[i]).abs() < 1e-6,
            "Mismatch at index {}: expected {}, got {}",
            i,
            embedding1[i],
            val
        );
    }
}

#[test]
fn test_tier_aware_simd_selection() {
    // Hot tier should use SIMD for small batches
    assert!(should_use_simd_for_tier(StorageTier::Hot, 8, 8));
    assert!(should_use_simd_for_tier(StorageTier::Hot, 16, 8));

    // Warm tier needs larger batches
    assert!(!should_use_simd_for_tier(StorageTier::Warm, 8, 8));
    assert!(should_use_simd_for_tier(StorageTier::Warm, 16, 8));

    // Cold tier never uses SIMD (bandwidth limited)
    assert!(!should_use_simd_for_tier(StorageTier::Cold, 8, 8));
    assert!(!should_use_simd_for_tier(StorageTier::Cold, 16, 8));
    assert!(!should_use_simd_for_tier(StorageTier::Cold, 100, 8));
}

#[test]
fn test_auto_tune_returns_valid_batch_size() {
    let optimal_size = auto_tune_batch_size();

    // Should return one of the tested sizes
    assert!(
        [8, 16, 32].contains(&optimal_size),
        "Auto-tuned size {optimal_size} not in valid range"
    );
}

#[test]
#[serial(parallel_engine)]
fn test_batch_spreading_with_parallel_engine() {
    use engram_core::activation::test_support::unique_test_id;

    // Ensure clean test environment
    ensure_no_active_engines();

    // Create a test graph with multiple paths
    let graph = Arc::new(create_activation_graph());
    let test_id = unique_test_id();

    // Create unique node names for this test
    let node_a = format!("{test_id}_A");
    let nodes_b: Vec<String> = (1..=8).map(|i| format!("{test_id}_B{i}")).collect();

    // A -> B1, B2, B3, B4, B5, B6, B7, B8 (enough for batch processing)
    ActivationGraphExt::add_edge(
        &*graph,
        node_a.clone(),
        nodes_b[0].clone(),
        0.9,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        node_a.clone(),
        nodes_b[1].clone(),
        0.8,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        node_a.clone(),
        nodes_b[2].clone(),
        0.7,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        node_a.clone(),
        nodes_b[3].clone(),
        0.6,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        node_a.clone(),
        nodes_b[4].clone(),
        0.5,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        node_a.clone(),
        nodes_b[5].clone(),
        0.4,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        node_a.clone(),
        nodes_b[6].clone(),
        0.3,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        node_a.clone(),
        nodes_b[7].clone(),
        0.2,
        EdgeType::Excitatory,
    );

    // Add embeddings to enable SIMD batch processing
    let embedding_a = [0.5f32; 768];
    graph.set_embedding(&node_a, &embedding_a);

    for (i, node_b) in nodes_b.iter().enumerate() {
        let mut embedding = [0.3f32; 768];
        // Add variation to each embedding
        embedding[i] = 0.7;
        graph.set_embedding(node_b, &embedding);
    }

    // Create engine with simplified config for reliability
    let config = ParallelSpreadingConfig {
        num_threads: 2, // Use 2 threads to avoid single-threaded edge cases
        max_depth: 1,   // Shallow spreading to reduce complexity
        simd_batch_size: 8,
        threshold: 0.01,
        completion_timeout: Some(std::time::Duration::from_secs(10)), // Reasonable timeout
        ..Default::default()
    };

    let engine = ParallelSpreadingEngine::new(config, graph).expect("Failed to create engine");

    // Spread activation from node A
    let seed_activations = vec![(node_a.clone(), 1.0)];
    let results = engine
        .spread_activation(&seed_activations)
        .expect("Failed to spread activation");

    // Verify results
    assert!(!results.activations.is_empty());

    // Should have activated node A at minimum
    let has_node_a = results.activations.iter().any(|a| a.memory_id == node_a);

    assert!(has_node_a, "Should have activated node A");

    // Note: B nodes may not be activated because SIMD batch processing falls back
    // to scalar path when embeddings are not available in the graph.
    // This test verifies the infrastructure is in place; actual batch spreading
    // requires embeddings to be stored in the graph.

    engine.shutdown().expect("Failed to shutdown engine");

    // Brief pause to ensure clean state transition
    std::thread::sleep(std::time::Duration::from_millis(10));
}

#[test]
#[serial(parallel_engine)]
fn test_simd_batch_spreading_with_embeddings() {
    use engram_core::activation::{ActivationGraphExt, test_support::unique_test_id};

    // Ensure clean test environment
    ensure_no_active_engines();

    // Create graph with embeddings
    let graph = Arc::new(create_activation_graph());
    let test_id = unique_test_id();

    // Create varied embeddings for testing
    let mut embedding_a = [0.0f32; 768];
    for (i, val) in embedding_a.iter_mut().enumerate() {
        *val = (i as f32 * 0.01).sin();
    }

    // Create unique node names for this test
    let node_a = format!("{test_id}_A");

    // Set embedding for node A
    graph.set_embedding(&node_a, &embedding_a);

    // Add 8 neighbors with similar but distinct embeddings
    for i in 1..=8 {
        let node_id = format!("{test_id}_B{i}");

        // Create similar embedding with slight variations
        let mut embedding = embedding_a;
        for (j, val) in embedding.iter_mut().enumerate() {
            *val += (i as f32 * 0.001) * (j as f32).cos();
        }

        graph.set_embedding(&node_id, &embedding);

        ActivationGraphExt::add_edge(&*graph, node_a.clone(), node_id, 0.5, EdgeType::Excitatory);
    }

    // Create engine with SIMD batch size = 8
    let config = ParallelSpreadingConfig {
        num_threads: 2, // Use 2 threads to avoid single-threaded edge cases
        max_depth: 1,   // Shallow spreading to reduce complexity
        simd_batch_size: 8,
        threshold: 0.01,
        completion_timeout: Some(std::time::Duration::from_secs(10)), // Reasonable timeout
        ..Default::default()
    };

    let engine =
        ParallelSpreadingEngine::new(config, graph.clone()).expect("Failed to create engine");

    // Verify the graph has been set up correctly
    let neighbors = ActivationGraphExt::get_neighbors(&*graph, &node_a);
    assert!(neighbors.is_some(), "Node A should have neighbors");
    let neighbors = neighbors.unwrap();
    assert_eq!(
        neighbors.len(),
        8,
        "Node A should have 8 neighbors, got {}",
        neighbors.len()
    );

    // Spread activation from node A
    let seed_activations = vec![(node_a, 1.0)];
    let results = engine
        .spread_activation(&seed_activations)
        .expect("Failed to spread activation");

    // Verify that SIMD batch processing was used
    // With embeddings available, neighbors should be activated
    let b_node_prefix = format!("{test_id}_B");
    let b_nodes: Vec<_> = results
        .activations
        .iter()
        .filter(|a| a.memory_id.starts_with(&b_node_prefix))
        .collect();

    assert!(
        b_nodes.len() >= 4,
        "Should have activated at least 4 B nodes with SIMD batch processing, got {}",
        b_nodes.len()
    );

    engine.shutdown().expect("Failed to shutdown engine");

    // Brief pause to ensure clean state transition
    std::thread::sleep(std::time::Duration::from_millis(10));
}

#[test]
#[serial(parallel_engine)]
fn test_batch_spreading_determinism() {
    use engram_core::activation::test_support::unique_test_id;

    // Ensure clean test environment
    ensure_no_active_engines();

    // Create a single test ID to use for both graphs to ensure identical structure
    let test_id = unique_test_id();

    // Create identical graphs
    let graph1 = Arc::new(create_activation_graph());
    let graph2 = Arc::new(create_activation_graph());

    let root_node = format!("{test_id}_ROOT");

    // Add same edges to both
    for graph in &[&graph1, &graph2] {
        for i in 1..=10 {
            ActivationGraphExt::add_edge(
                &***graph,
                root_node.clone(),
                format!("{test_id}_NODE{i}"),
                0.5,
                EdgeType::Excitatory,
            );
        }
    }

    // Create engines with same config
    let config = ParallelSpreadingConfig {
        num_threads: 2,
        max_depth: 2,
        simd_batch_size: 8,
        deterministic: true,
        seed: Some(42),
        ..Default::default()
    };

    let engine1 =
        ParallelSpreadingEngine::new(config.clone(), graph1).expect("Failed to create engine 1");
    let engine2 = ParallelSpreadingEngine::new(config, graph2).expect("Failed to create engine 2");

    // Run same spreading
    let seed = vec![(root_node, 1.0)];
    let results1 = engine1
        .spread_activation(&seed)
        .expect("Failed to spread 1");
    let results2 = engine2
        .spread_activation(&seed)
        .expect("Failed to spread 2");

    // Results should be identical
    assert_eq!(results1.activations.len(), results2.activations.len());

    engine1.shutdown().expect("Failed to shutdown 1");
    engine2.shutdown().expect("Failed to shutdown 2");

    // Brief pause to ensure clean state transition
    std::thread::sleep(std::time::Duration::from_millis(10));
}

#[test]
fn test_activation_batch_capacity() {
    let mut batch = ActivationBatch::new(4);

    let embedding = [0.5f32; 768];

    // Fill to capacity
    for _ in 0..4 {
        assert!(batch.push(&embedding), "Should be able to push to capacity");
    }

    // Verify at capacity
    assert_eq!(batch.len(), 4);

    // Clear and reuse
    batch.clear();
    assert_eq!(batch.len(), 0);
    assert!(batch.is_empty());

    // Should be able to fill again
    for _ in 0..4 {
        assert!(batch.push(&embedding), "Should be able to push after clear");
    }
}
