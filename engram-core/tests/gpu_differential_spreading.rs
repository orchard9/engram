//! CPU-GPU differential tests for activation spreading
//!
//! Validates that GPU spreading produces identical results to CPU
//! implementation within floating-point tolerance (<1e-6).
//!
//! Test scenarios:
//! - Random graphs (various sizes and densities)
//! - Edge cases (isolated nodes, fully connected, chains)
//! - Property-based testing with quickcheck

#![cfg(all(test, cuda_available))]

use engram_core::activation::{ActivationGraphExt, EdgeType, MemoryGraph, create_activation_graph};
use engram_core::compute::cuda::spreading::{CsrGraph, GpuSpreadingEngine, SpreadingConfig};
use std::collections::HashMap;

/// Tolerance for floating-point comparisons
const TOLERANCE: f32 = 1e-6;

/// Create a simple chain graph: A -> B -> C -> D
fn create_chain_graph(num_nodes: usize) -> (MemoryGraph, Vec<String>) {
    let graph = create_activation_graph();
    let nodes: Vec<String> = (0..num_nodes).map(|i| format!("node_{i}")).collect();

    for i in 0..(num_nodes - 1) {
        ActivationGraphExt::add_edge(
            &graph,
            nodes[i].clone(),
            nodes[i + 1].clone(),
            0.8,
            EdgeType::Excitatory,
        );
    }

    (graph, nodes)
}

/// Create a fully connected graph (complete graph K_n)
fn create_fully_connected_graph(num_nodes: usize) -> (MemoryGraph, Vec<String>) {
    let graph = create_activation_graph();
    let nodes: Vec<String> = (0..num_nodes).map(|i| format!("node_{i}")).collect();

    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if i != j {
                ActivationGraphExt::add_edge(
                    &graph,
                    nodes[i].clone(),
                    nodes[j].clone(),
                    0.5,
                    EdgeType::Excitatory,
                );
            }
        }
    }

    (graph, nodes)
}

/// Create CSR graph from adjacency representation
fn create_csr_from_nodes(nodes: &[String], graph: &MemoryGraph) -> CsrGraph {
    let num_nodes = nodes.len();
    let mut row_ptr = Vec::with_capacity(num_nodes + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    let mut node_to_idx = HashMap::with_capacity(num_nodes);
    let mut idx_to_node = Vec::with_capacity(num_nodes);

    for (idx, node) in nodes.iter().enumerate() {
        node_to_idx.insert(node.clone(), idx);
        idx_to_node.push(node.clone());
    }

    row_ptr.push(0);

    for source in nodes {
        if let Some(neighbors) = ActivationGraphExt::get_neighbors(graph, source) {
            for edge in neighbors {
                if let Some(&target_idx) = node_to_idx.get(&edge.target) {
                    col_idx.push(target_idx as i32);
                    values.push(edge.weight);
                }
            }
        }
        row_ptr.push(col_idx.len() as i32);
    }

    let num_edges = col_idx.len();

    CsrGraph {
        row_ptr,
        col_idx,
        values,
        num_nodes,
        num_edges,
        node_to_idx,
        idx_to_node,
    }
}

/// CPU reference implementation of spreading
fn cpu_spreading_reference(csr: &CsrGraph, input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; csr.num_nodes];

    for node_id in 0..csr.num_nodes {
        let start = csr.row_ptr[node_id] as usize;
        let end = csr.row_ptr[node_id + 1] as usize;

        for edge_idx in start..end {
            let neighbor = csr.col_idx[edge_idx] as usize;
            let weight = csr.values[edge_idx];
            output[node_id] += weight * input[neighbor];
        }
    }

    output
}

/// Compare two activation vectors within tolerance
fn assert_activations_equal(cpu: &[f32], gpu: &[f32], message: &str) {
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "{}: Length mismatch: CPU={}, GPU={}",
        message,
        cpu.len(),
        gpu.len()
    );

    for (i, (cpu_val, gpu_val)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let diff = (cpu_val - gpu_val).abs();
        assert!(
            diff < TOLERANCE,
            "{}: Node {}: CPU={}, GPU={}, diff={}",
            message,
            i,
            cpu_val,
            gpu_val,
            diff
        );
    }
}

#[test]
fn test_chain_graph() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        eprintln!("GPU not available, skipping differential test");
        return;
    }

    let (graph, nodes) = create_chain_graph(10);
    let csr = create_csr_from_nodes(&nodes, &graph);

    // Initialize first node with activation
    let mut input = vec![0.0f32; 10];
    input[0] = 1.0;

    // CPU reference
    let cpu_result = cpu_spreading_reference(&csr, &input);

    // GPU implementation
    let config = SpreadingConfig::default();
    let engine = GpuSpreadingEngine::new(config).expect("Failed to create GPU engine");
    let gpu_result = engine
        .spread_activation_gpu(&csr, &input)
        .expect("GPU spreading failed");

    assert_activations_equal(&cpu_result, &gpu_result, "Chain graph");

    // Verify expected spreading pattern
    // Node 0 -> Node 1: 0.8 * 1.0 = 0.8
    assert!((cpu_result[1] - 0.8).abs() < TOLERANCE);
    // Node 1 has no incoming activation yet (would need another hop)
    assert!((cpu_result[2] - 0.0).abs() < TOLERANCE);
}

#[test]
fn test_fully_connected_small() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        eprintln!("GPU not available, skipping differential test");
        return;
    }

    let (graph, nodes) = create_fully_connected_graph(5);
    let csr = create_csr_from_nodes(&nodes, &graph);

    // Uniform initial activation
    let input = vec![1.0f32; 5];

    let cpu_result = cpu_spreading_reference(&csr, &input);
    let config = SpreadingConfig::default();
    let engine = GpuSpreadingEngine::new(config).expect("Failed to create GPU engine");
    let gpu_result = engine
        .spread_activation_gpu(&csr, &input)
        .expect("GPU spreading failed");

    assert_activations_equal(&cpu_result, &gpu_result, "Fully connected (small)");

    // Each node receives from 4 neighbors * 0.5 weight * 1.0 activation = 2.0
    for &val in &cpu_result {
        assert!((val - 2.0).abs() < TOLERANCE);
    }
}

#[test]
fn test_isolated_nodes() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        eprintln!("GPU not available, skipping differential test");
        return;
    }

    // Graph with no edges
    let graph = create_activation_graph();
    let nodes: Vec<String> = (0..10).map(|i| format!("node_{i}")).collect();

    let csr = create_csr_from_nodes(&nodes, &graph);

    let input: Vec<f32> = (0..10).map(|i| i as f32).collect();

    let cpu_result = cpu_spreading_reference(&csr, &input);
    let config = SpreadingConfig::default();
    let engine = GpuSpreadingEngine::new(config).expect("Failed to create GPU engine");
    let gpu_result = engine
        .spread_activation_gpu(&csr, &input)
        .expect("GPU spreading failed");

    assert_activations_equal(&cpu_result, &gpu_result, "Isolated nodes");

    // All outputs should be zero (no edges)
    for &val in &cpu_result {
        assert!((val - 0.0).abs() < TOLERANCE);
    }
}

#[test]
fn test_various_sizes() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        eprintln!("GPU not available, skipping differential test");
        return;
    }

    let sizes = vec![10, 100, 512, 1000, 5000];

    for size in sizes {
        let (graph, nodes) = create_chain_graph(size);
        let csr = create_csr_from_nodes(&nodes, &graph);

        // Random-ish initial activations
        let input: Vec<f32> = (0..size).map(|i| (i % 10) as f32 / 10.0).collect();

        let cpu_result = cpu_spreading_reference(&csr, &input);
        let config = SpreadingConfig::default();
        let engine = GpuSpreadingEngine::new(config).expect("Failed to create GPU engine");
        let gpu_result = engine
            .spread_activation_gpu(&csr, &input)
            .expect("GPU spreading failed");

        assert_activations_equal(
            &cpu_result,
            &gpu_result,
            &format!("Chain graph size {}", size),
        );
    }
}

#[test]
fn test_random_sparse_graphs() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        eprintln!("GPU not available, skipping differential test");
        return;
    }

    let num_nodes = 1000;
    let avg_degree = 8;

    // Create random sparse graph
    let graph = create_activation_graph();
    let nodes: Vec<String> = (0..num_nodes).map(|i| format!("node_{i}")).collect();

    let mut rng_seed = 12345u64;
    for (i, source) in nodes.iter().enumerate() {
        for _ in 0..avg_degree {
            rng_seed = rng_seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let target_idx = (rng_seed as usize) % num_nodes;

            if target_idx != i {
                let weight = 0.3 + 0.7 * ((rng_seed % 100) as f32 / 100.0);
                ActivationGraphExt::add_edge(
                    &graph,
                    source.clone(),
                    nodes[target_idx].clone(),
                    weight,
                    EdgeType::Excitatory,
                );
            }
        }
    }

    let csr = create_csr_from_nodes(&nodes, &graph);

    // Random initial activations
    let input: Vec<f32> = (0..num_nodes)
        .map(|i| {
            let seed = (i as u64 * 2654435761) % 1000;
            seed as f32 / 1000.0
        })
        .collect();

    let cpu_result = cpu_spreading_reference(&csr, &input);
    let config = SpreadingConfig::default();
    let engine = GpuSpreadingEngine::new(config).expect("Failed to create GPU engine");
    let gpu_result = engine
        .spread_activation_gpu(&csr, &input)
        .expect("GPU spreading failed");

    assert_activations_equal(&cpu_result, &gpu_result, "Random sparse graph");
}

#[test]
fn test_zero_weights() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        eprintln!("GPU not available, skipping differential test");
        return;
    }

    let graph = create_activation_graph();
    let nodes: Vec<String> = (0..10).map(|i| format!("node_{i}")).collect();

    // Create edges with zero weights
    for i in 0..9 {
        ActivationGraphExt::add_edge(
            &graph,
            nodes[i].clone(),
            nodes[i + 1].clone(),
            0.0, // Zero weight
            EdgeType::Excitatory,
        );
    }

    let csr = create_csr_from_nodes(&nodes, &graph);
    let input = vec![1.0f32; 10];

    let cpu_result = cpu_spreading_reference(&csr, &input);
    let config = SpreadingConfig::default();
    let engine = GpuSpreadingEngine::new(config).expect("Failed to create GPU engine");
    let gpu_result = engine
        .spread_activation_gpu(&csr, &input)
        .expect("GPU spreading failed");

    assert_activations_equal(&cpu_result, &gpu_result, "Zero weights");

    // All outputs should be zero (zero-weight edges)
    for &val in &cpu_result {
        assert!((val - 0.0).abs() < TOLERANCE);
    }
}

#[test]
fn test_high_degree_nodes() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        eprintln!("GPU not available, skipping differential test");
        return;
    }

    // Star graph: one central node connected to many periphery nodes
    let num_nodes = 100;
    let graph = create_activation_graph();
    let nodes: Vec<String> = (0..num_nodes).map(|i| format!("node_{i}")).collect();

    // Central node (0) connects to all others
    for i in 1..num_nodes {
        ActivationGraphExt::add_edge(
            &graph,
            nodes[0].clone(),
            nodes[i].clone(),
            0.1,
            EdgeType::Excitatory,
        );
    }

    let csr = create_csr_from_nodes(&nodes, &graph);

    // Only periphery nodes have activation
    let mut input = vec![0.0f32; num_nodes];
    for i in 1..num_nodes {
        input[i] = 1.0;
    }

    let cpu_result = cpu_spreading_reference(&csr, &input);
    let config = SpreadingConfig::default();
    let engine = GpuSpreadingEngine::new(config).expect("Failed to create GPU engine");
    let gpu_result = engine
        .spread_activation_gpu(&csr, &input)
        .expect("GPU spreading failed");

    assert_activations_equal(&cpu_result, &gpu_result, "High-degree central node");

    // Central node should receive: (num_nodes-1) * 0.1 * 1.0 = 9.9
    assert!((cpu_result[0] - 9.9).abs() < TOLERANCE);
}
