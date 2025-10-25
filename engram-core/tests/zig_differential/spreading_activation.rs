//! Differential tests for spreading activation kernels.
//!
//! Validates that Zig cache-optimized BFS implementation produces numerically
//! identical results to the Rust baseline across various graph topologies.
//!
//! # Test Coverage
//!
//! 1. **Property-Based Tests** - 10,000 random graph topologies
//! 2. **Edge Cases** - Isolated nodes, fully connected, cyclic, linear chains
//! 3. **Boundary Conditions** - Single node, empty graph, no edges
//!
//! # Current Status (Task 002)
//!
//! Zig kernels are currently stubs (no-op). These tests will FAIL until Task 006
//! implements the actual cache-optimized spreading activation. This validates the
//! differential testing framework.

use super::{EPSILON, NUM_PROPTEST_CASES, TestGraph, assert_slices_approx_eq};
use proptest::prelude::*;

// Conditional import - tests work with or without zig-kernels feature
#[cfg(feature = "zig-kernels")]
use engram_core::zig_kernels;

// Mock Zig kernels module when feature is disabled (for testing Rust baseline)
#[cfg(not(feature = "zig-kernels"))]
mod zig_kernels {
    /// Mock spread_activation for testing without zig-kernels feature
    pub const fn spread_activation(
        _adjacency: &[u32],
        _weights: &[f32],
        _activations: &mut [f32],
        _num_nodes: usize,
        _iterations: u32,
    ) {
        // Fallback uses stub behavior (no-op) to match Task 002 Zig stubs
    }
}

/// Rust baseline implementation of spreading activation
fn rust_spread_activation(
    adjacency: &[u32],
    weights: &[f32],
    activations: &mut [f32],
    num_nodes: usize,
    iterations: u32,
) {
    assert_eq!(adjacency.len(), weights.len());

    // Build adjacency map for easier lookup
    let mut adj_map: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_nodes];
    for (edge_idx, &target) in adjacency.iter().enumerate() {
        let target_idx = target as usize;
        if target_idx < num_nodes && edge_idx < weights.len() {
            // Find source by counting edges
            let mut source = 0;
            let mut edges_so_far = 0;
            for node in 0..num_nodes {
                let node_edges = adjacency
                    .iter()
                    .enumerate()
                    .skip(edges_so_far)
                    .take_while(|_| {
                        // Simple heuristic: assume edges are grouped by source
                        true
                    })
                    .count();
                if edges_so_far + node_edges > edge_idx {
                    source = node;
                    break;
                }
                edges_so_far += node_edges;
            }

            adj_map[source].push((target_idx, weights[edge_idx]));
        }
    }

    // Perform spreading iterations
    for _ in 0..iterations {
        let current_activations = activations.to_vec();

        for source in 0..num_nodes {
            for &(target, weight) in &adj_map[source] {
                activations[target] += current_activations[source] * weight;
            }
        }

        // Normalize to prevent explosion
        let max_activation = activations.iter().copied().fold(0.0_f32, f32::max);
        if max_activation > 1.0 {
            for act in activations.iter_mut() {
                *act /= max_activation;
            }
        }
    }
}

/// Simplified baseline that matches CSR format expectations
#[allow(dead_code)]
fn rust_spread_activation_csr(graph: &TestGraph, iterations: u32) -> Vec<f32> {
    let mut activations = graph.activations.clone();

    for _ in 0..iterations {
        let current = activations.clone();

        // Simple spreading: just accumulate from edges
        // Note: This is a simplified model - real spreading would need CSR offsets
        for (i, &target) in graph.adjacency.iter().enumerate() {
            let weight = graph.weights[i];
            let target_idx = target as usize;

            if target_idx < graph.num_nodes {
                // Spread from all nodes proportionally
                #[allow(clippy::cast_precision_loss)]
                let factor = graph.num_nodes as f32;
                for &current_val in &current {
                    if current_val > 0.0 {
                        activations[target_idx] += current_val * weight / factor;
                    }
                }
            }
        }

        // Decay to prevent explosion
        for act in &mut activations {
            *act *= 0.95;
        }
    }

    activations
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(NUM_PROPTEST_CASES))]

    /// Property test: Zig and Rust implementations match for random graphs
    #[test]
    fn prop_spreading_activation_random_graphs(
        num_nodes in 10_usize..100,
        edge_prob in 0.05_f64..0.3,
        iterations in 1_u32..10,
        seed in 0_u64..10000
    ) {
        let graph = TestGraph::random(num_nodes, edge_prob, seed);

        // Clone activations for both implementations
        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        // Call Zig kernel
        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            graph.num_nodes,
            iterations,
        );

        // Call Rust baseline
        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            iterations,
        );

        // Verify equivalence
        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
    }

    /// Property test: Linear chains with varying lengths
    #[test]
    fn prop_spreading_linear_chains(
        num_nodes in 2_usize..100,
        iterations in 1_u32..20
    ) {
        let graph = TestGraph::linear_chain(num_nodes);

        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            graph.num_nodes,
            iterations,
        );

        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            iterations,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
    }

    /// Property test: Star graphs (hub-and-spoke)
    #[test]
    fn prop_spreading_star_graphs(
        num_nodes in 3_usize..50,
        iterations in 1_u32..10
    ) {
        let graph = TestGraph::star(num_nodes);

        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            graph.num_nodes,
            iterations,
        );

        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            iterations,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_isolated_nodes() {
        // Graph with no edges - activations should not change
        let graph = TestGraph {
            num_nodes: 5,
            adjacency: vec![],
            weights: vec![],
            activations: vec![1.0, 0.5, 0.0, 0.3, 0.0],
        };

        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            5,
            10,
        );
        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            5,
            10,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
        // With no edges, stub no-op behavior should preserve activations
        // (but real implementation would also preserve them)
    }

    #[test]
    fn test_single_node() {
        let graph = TestGraph {
            num_nodes: 1,
            adjacency: vec![],
            weights: vec![],
            activations: vec![1.0],
        };

        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            1,
            5,
        );
        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            1,
            5,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
    }

    #[test]
    fn test_fully_connected() {
        let graph = TestGraph::fully_connected(10);

        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            graph.num_nodes,
            5,
        );

        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            5,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
    }

    #[test]
    fn test_linear_chain_long() {
        let graph = TestGraph::linear_chain(100);

        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            graph.num_nodes,
            50,
        );

        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            50,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
    }

    #[test]
    fn test_cyclic_graph() {
        // Create a cycle: 0 -> 1 -> 2 -> 0
        let graph = TestGraph {
            num_nodes: 3,
            adjacency: vec![1, 2, 0],
            weights: vec![1.0, 1.0, 1.0],
            activations: vec![1.0, 0.0, 0.0],
        };

        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            graph.num_nodes,
            10,
        );

        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            10,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
    }

    #[test]
    fn test_zero_weights() {
        // Graph with edges but zero weights - no spreading should occur
        let graph = TestGraph {
            num_nodes: 4,
            adjacency: vec![1, 2, 3],
            weights: vec![0.0, 0.0, 0.0],
            activations: vec![1.0, 0.0, 0.0, 0.0],
        };

        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            graph.num_nodes,
            5,
        );

        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            5,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
    }

    #[test]
    fn test_negative_weights() {
        // Graph with negative weights (inhibitory connections)
        let graph = TestGraph {
            num_nodes: 3,
            adjacency: vec![1, 2],
            weights: vec![1.0, -0.5],
            activations: vec![1.0, 0.0, 0.0],
        };

        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            graph.num_nodes,
            5,
        );

        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            5,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
    }

    #[test]
    fn test_all_nodes_activated() {
        let graph = TestGraph {
            num_nodes: 5,
            adjacency: vec![1, 2, 3, 4],
            weights: vec![0.5, 0.5, 0.5, 0.5],
            activations: vec![1.0, 1.0, 1.0, 1.0, 1.0],
        };

        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            graph.num_nodes,
            5,
        );

        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            5,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
    }

    #[test]
    fn test_single_iteration() {
        let graph = TestGraph::linear_chain(10);

        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            graph.num_nodes,
            1,
        );

        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            1,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
    }

    #[test]
    fn test_many_iterations() {
        let graph = TestGraph::star(20);

        let mut zig_activations = graph.activations.clone();
        let mut rust_activations = graph.activations.clone();

        zig_kernels::spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut zig_activations,
            graph.num_nodes,
            100,
        );

        rust_spread_activation(
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            100,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON);
    }
}

#[cfg(test)]
mod regression_tests {
    

    #[test]
    fn test_regression_placeholder() {
        // Regression tests will be added as interesting cases are discovered
    }
}
