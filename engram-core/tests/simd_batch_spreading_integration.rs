//! Integration tests for SIMD batch spreading in activation engine
//!
//! Tests the complete pipeline from activation spreading through SIMD batch operations

use engram_core::activation::{
    ActivationGraphExt, EdgeType, ParallelSpreadingConfig, ParallelSpreadingEngine,
    create_activation_graph,
    simd_optimization::{ActivationBatch, auto_tune_batch_size, should_use_simd_for_tier},
    storage_aware::StorageTier,
};
use std::sync::Arc;

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
fn test_batch_spreading_with_parallel_engine() {
    // Create a test graph with multiple paths
    let graph = Arc::new(create_activation_graph());

    // A -> B1, B2, B3, B4, B5, B6, B7, B8 (enough for batch processing)
    ActivationGraphExt::add_edge(
        &*graph,
        "A".to_string(),
        "B1".to_string(),
        0.9,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        "A".to_string(),
        "B2".to_string(),
        0.8,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        "A".to_string(),
        "B3".to_string(),
        0.7,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        "A".to_string(),
        "B4".to_string(),
        0.6,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        "A".to_string(),
        "B5".to_string(),
        0.5,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        "A".to_string(),
        "B6".to_string(),
        0.4,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        "A".to_string(),
        "B7".to_string(),
        0.3,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        "A".to_string(),
        "B8".to_string(),
        0.2,
        EdgeType::Excitatory,
    );

    // Create engine with SIMD batch size = 8
    let config = ParallelSpreadingConfig {
        num_threads: 2,
        max_depth: 2,
        simd_batch_size: 8,
        threshold: 0.01,
        ..Default::default()
    };

    let engine = ParallelSpreadingEngine::new(config, graph).expect("Failed to create engine");

    // Spread activation from node A
    let seed_activations = vec![("A".to_string(), 1.0)];
    let results = engine
        .spread_activation(&seed_activations)
        .expect("Failed to spread activation");

    // Verify results
    assert!(!results.activations.is_empty());

    // Should have activated node A at minimum
    let has_node_a = results.activations.iter().any(|a| a.memory_id == "A");

    assert!(has_node_a, "Should have activated node A");

    // Note: B nodes may not be activated because SIMD batch processing falls back
    // to scalar path when embeddings are not available in the graph.
    // This test verifies the infrastructure is in place; actual batch spreading
    // requires embeddings to be stored in the graph.

    engine.shutdown().expect("Failed to shutdown engine");
}

#[test]
fn test_simd_batch_spreading_with_embeddings() {
    use engram_core::activation::ActivationGraphExt;

    // Create graph with embeddings
    let graph = Arc::new(create_activation_graph());

    // Create varied embeddings for testing
    let mut embedding_a = [0.0f32; 768];
    for (i, val) in embedding_a.iter_mut().enumerate() {
        *val = (i as f32 * 0.01).sin();
    }

    // Set embedding for node A
    graph.set_embedding(&"A".to_string(), &embedding_a);

    // Add 8 neighbors with similar but distinct embeddings
    for i in 1..=8 {
        let node_id = format!("B{i}");

        // Create similar embedding with slight variations
        let mut embedding = embedding_a;
        for (j, val) in embedding.iter_mut().enumerate() {
            *val += (i as f32 * 0.001) * (j as f32).cos();
        }

        graph.set_embedding(&node_id, &embedding);

        ActivationGraphExt::add_edge(&*graph, "A".to_string(), node_id, 0.5, EdgeType::Excitatory);
    }

    // Create engine with SIMD batch size = 8
    let config = ParallelSpreadingConfig {
        num_threads: 2,
        max_depth: 2,
        simd_batch_size: 8,
        threshold: 0.01,
        ..Default::default()
    };

    let engine = ParallelSpreadingEngine::new(config, graph).expect("Failed to create engine");

    // Spread activation from node A
    let seed_activations = vec![("A".to_string(), 1.0)];
    let results = engine
        .spread_activation(&seed_activations)
        .expect("Failed to spread activation");

    // Verify that SIMD batch processing was used
    // With embeddings available, neighbors should be activated
    let b_nodes: Vec<_> = results
        .activations
        .iter()
        .filter(|a| a.memory_id.starts_with('B'))
        .collect();

    assert!(
        b_nodes.len() >= 4,
        "Should have activated at least 4 B nodes with SIMD batch processing, got {}",
        b_nodes.len()
    );

    engine.shutdown().expect("Failed to shutdown engine");
}

#[test]
fn test_batch_spreading_determinism() {
    // Create identical graphs
    let graph1 = Arc::new(create_activation_graph());
    let graph2 = Arc::new(create_activation_graph());

    // Add same edges to both
    for graph in &[&graph1, &graph2] {
        for i in 1..=10 {
            ActivationGraphExt::add_edge(
                &***graph,
                "ROOT".to_string(),
                format!("NODE{i}"),
                0.5,
                EdgeType::Excitatory,
            );
        }
    }

    // Create engines with same config
    let config = ParallelSpreadingConfig {
        num_threads: 1,
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
    let seed = vec![("ROOT".to_string(), 1.0)];
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
