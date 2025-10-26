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

use super::{EPSILON_ITERATIVE, NUM_PROPTEST_CASES, TestGraph, assert_slices_approx_eq};
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
///
/// This is the reference implementation that uses explicit edge source tracking
/// to properly compute spreading activation across graph edges.
fn rust_spread_activation(
    edge_sources: &[u32],
    adjacency: &[u32],
    weights: &[f32],
    activations: &mut [f32],
    num_nodes: usize,
    iterations: u32,
) {
    assert_eq!(edge_sources.len(), adjacency.len());
    assert_eq!(adjacency.len(), weights.len());

    // Build adjacency map from explicit edge sources
    let mut adj_map: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_nodes];

    for (edge_idx, (&source, &target)) in edge_sources.iter().zip(adjacency.iter()).enumerate() {
        let source_idx = source as usize;
        let target_idx = target as usize;
        if source_idx < num_nodes && target_idx < num_nodes {
            adj_map[source_idx].push((target_idx, weights[edge_idx]));
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            iterations,
        );

        // Verify equivalence
        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            iterations,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            iterations,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
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
            edge_sources: vec![],
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            5,
            10,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
        // With no edges, stub no-op behavior should preserve activations
        // (but real implementation would also preserve them)
    }

    #[test]
    fn test_single_node() {
        let graph = TestGraph {
            num_nodes: 1,
            edge_sources: vec![],
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            1,
            5,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            5,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            50,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
    }

    #[test]
    fn test_cyclic_graph() {
        // Create a cycle: 0 -> 1 -> 2 -> 0
        let graph = TestGraph {
            num_nodes: 3,
            edge_sources: vec![0, 1, 2],
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            10,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
    }

    #[test]
    fn test_zero_weights() {
        // Graph with edges but zero weights - no spreading should occur
        let graph = TestGraph {
            num_nodes: 4,
            edge_sources: vec![0, 0, 0],
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            5,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
    }

    #[test]
    fn test_negative_weights() {
        // Graph with negative weights (inhibitory connections)
        let graph = TestGraph {
            num_nodes: 3,
            edge_sources: vec![0, 0],
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            5,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
    }

    #[test]
    fn test_all_nodes_activated() {
        let graph = TestGraph {
            num_nodes: 5,
            edge_sources: vec![0, 0, 0, 0],
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            5,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            1,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            100,
        );

        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    /// Regression test for Issue C1: Broken edge source inference
    ///
    /// Previously, the Rust baseline used `take_while(|_| true)` which consumed
    /// ALL edges on the first iteration. This test verifies proper edge tracking.
    #[test]
    fn test_regression_edge_source_tracking() {
        // Simple graph where source tracking matters: 0->1, 1->2
        let graph = TestGraph {
            num_nodes: 3,
            edge_sources: vec![0, 1],
            adjacency: vec![1, 2],
            weights: vec![1.0, 1.0],
            activations: vec![1.0, 0.0, 0.0],
        };

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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            1,
        );

        // After 1 iteration: node 0 should spread to node 1
        // Node 1 should NOT spread yet (activation happens after iteration completes)
        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
    }

    /// Regression test for self-loops (missing edge case from H8)
    #[test]
    fn test_regression_self_loops() {
        // Each node has a self-loop with weight 0.9
        let graph = TestGraph {
            num_nodes: 3,
            edge_sources: vec![0, 1, 2],
            adjacency: vec![0, 1, 2],
            weights: vec![0.9, 0.9, 0.9],
            activations: vec![1.0, 0.5, 0.2],
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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            5,
        );

        // Verify activations don't explode (normalization kicks in)
        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
        for &act in &rust_activations {
            assert!(act <= 1.0, "Activations should be normalized");
        }
    }

    /// Regression test for multi-edges (missing edge case from H8)
    #[test]
    fn test_regression_multi_edges() {
        // Multiple edges from node 0 to node 1
        let graph = TestGraph {
            num_nodes: 2,
            edge_sources: vec![0, 0, 0],
            adjacency: vec![1, 1, 1],
            weights: vec![0.3, 0.3, 0.3],
            activations: vec![1.0, 0.0],
        };

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
            &graph.edge_sources,
            &graph.adjacency,
            &graph.weights,
            &mut rust_activations,
            graph.num_nodes,
            1,
        );

        // Weights should sum (node 1 should receive 0.3 + 0.3 + 0.3 = 0.9)
        assert_slices_approx_eq(&rust_activations, &zig_activations, EPSILON_ITERATIVE);
        assert!(
            (rust_activations[1] - 0.9).abs() < EPSILON_ITERATIVE,
            "Multi-edge weights should sum, got {}",
            rust_activations[1]
        );
    }
}
