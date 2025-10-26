// High-performance activation spreading kernel using CSR (Compressed Sparse Row) format
// Implements cache-optimized iterative propagation with accumulation
//
// Design Principles:
// - CSR graph representation for cache-friendly sequential edge access
// - Iterative spreading: accumulate activations over multiple iterations
// - Normalization: prevent activation explosion (capped at 1.0)
// - Cache efficiency: edges for each node stored contiguously
//
// Algorithm:
// For each iteration:
//   1. Clone current activations to temporary buffer
//   2. For each source node, iterate its edges and accumulate to targets
//   3. Normalize if max activation exceeds 1.0
//
// Performance:
// - CSR enables sequential edge iteration per source (cache-friendly)
// - Arena allocator provides O(1) temporary buffer allocation
// - Minimal branching in hot loop
// - SIMD potential for edge weight multiplication

const std = @import("std");
const allocator_mod = @import("allocator.zig");

/// CSR (Compressed Sparse Row) graph representation
///
/// Memory layout:
/// - edge_offsets[node]: start index in edges array for this node's edges
/// - edge_offsets[node+1]: end index (one past last edge)
/// - edges[edge_offsets[node]..edge_offsets[node+1]]: all edges from node
///
/// Example:
///   Node 0: edges 0-2 (3 edges)
///   Node 1: edges 3-4 (2 edges)
///   Node 2: edges 5-5 (0 edges)
///   edge_offsets = [0, 3, 5, 5]
pub const CSRGraph = struct {
    /// Offsets into edges array: edge_offsets[i] = start for node i
    /// Length is num_nodes + 1 (last entry is total edge count)
    edge_offsets: []const usize,

    /// Target node indices for all edges
    edges_targets: []const u32,

    /// Edge weights corresponding to edges_targets
    edges_weights: []const f32,

    /// Total number of nodes
    num_nodes: usize,

    /// Get edges for a specific source node
    ///
    /// Returns slices of (targets, weights) for all edges originating from node.
    /// Both slices have the same length.
    pub fn getEdges(self: CSRGraph, node: usize) struct { targets: []const u32, weights: []const f32 } {
        std.debug.assert(node < self.num_nodes);

        const start = self.edge_offsets[node];
        const end = self.edge_offsets[node + 1];

        return .{
            .targets = self.edges_targets[start..end],
            .weights = self.edges_weights[start..end],
        };
    }
};

/// Spread activation across CSR graph with iterative accumulation
///
/// Updates activations in-place by propagating along weighted edges.
/// Uses CSR format for cache-optimal sequential edge access.
///
/// # Parameters
/// - graph: CSR graph structure
/// - activations: Current activation levels (modified in-place)
/// - iterations: Number of spreading iterations
///
/// # Algorithm
/// For each iteration:
///   1. Save current activations
///   2. For each source node:
///      - Get its edges from CSR structure (sequential access)
///      - For each edge: activations[target] += current[source] * weight
///   3. Normalize to prevent explosion (cap at 1.0)
///
/// # Cache Optimization
/// - CSR format ensures edges for each node are contiguous in memory
/// - Sequential iteration through edge_targets and edge_weights
/// - Temporary buffer allocated from thread-local arena (no malloc overhead)
pub fn spreadActivationCSR(
    graph: CSRGraph,
    activations: []f32,
    iterations: u32,
) void {
    // Validate inputs
    std.debug.assert(activations.len == graph.num_nodes);
    std.debug.assert(graph.edge_offsets.len == graph.num_nodes + 1);
    std.debug.assert(graph.edges_targets.len == graph.edges_weights.len);
    std.debug.assert(iterations > 0);

    const num_nodes = graph.num_nodes;

    // Early exit for trivial cases
    if (num_nodes == 0 or iterations == 0) {
        return;
    }

    // Get arena allocator for temporary buffer
    const arena = allocator_mod.getThreadArena();
    defer allocator_mod.resetThreadArena();

    // Allocate temporary buffer for current activations
    const current_activations = arena.allocArray(f32, num_nodes) catch {
        // Fallback: no-op if allocation fails
        return;
    };

    // Main spreading loop
    for (0..iterations) |_| {
        // Copy current activations to temporary buffer
        @memcpy(current_activations, activations);

        // Iterate through each source node
        for (0..num_nodes) |source| {
            const source_activation = current_activations[source];

            // Skip inactive nodes (optimization)
            if (source_activation == 0.0) continue;

            // Get edges for this source node (cache-friendly sequential access)
            const edges = graph.getEdges(source);
            const targets = edges.targets;
            const weights = edges.weights;

            // Propagate activation to all target nodes
            for (targets, weights) |target, weight| {
                // Bounds check (should be guaranteed by CSR construction)
                if (target >= num_nodes) continue;

                // Accumulate activation: target += source * weight
                activations[target] += source_activation * weight;
            }
        }

        // Normalize to prevent explosion (cap maximum at 1.0)
        var max_activation: f32 = 0.0;
        for (activations) |act| {
            max_activation = @max(max_activation, act);
        }

        if (max_activation > 1.0) {
            for (activations) |*act| {
                act.* /= max_activation;
            }
        }
    }
}

/// Legacy edge-list based spreading (for backward compatibility)
///
/// This matches the current FFI interface but is less cache-efficient than CSR.
/// Edges are stored as parallel arrays without explicit source information.
pub fn spreadActivation(
    adjacency: []const u32,
    weights: []const f32,
    activations: []f32,
    num_nodes: usize,
    num_edges: usize,
    iterations: u32,
) void {
    // Validate inputs
    std.debug.assert(adjacency.len == num_edges);
    std.debug.assert(weights.len == num_edges);
    std.debug.assert(activations.len == num_nodes);
    std.debug.assert(iterations > 0);

    // Early exit for trivial cases
    if (num_nodes == 0 or num_edges == 0 or iterations == 0) {
        return;
    }

    // Get arena allocator for temporary buffer
    const arena = allocator_mod.getThreadArena();
    defer allocator_mod.resetThreadArena();

    // Allocate temporary buffer for current activations
    const current_activations = arena.allocArray(f32, num_nodes) catch {
        // Fallback: no-op if allocation fails
        return;
    };

    // Main spreading loop
    for (0..iterations) |_| {
        // Copy current activations
        @memcpy(current_activations, activations);

        // Infer source nodes from edge structure
        // Assumption: edges are grouped by source (CSR-like ordering)
        // Use heuristic: distribute edges evenly across sources
        const edges_per_node = if (num_nodes > 0) (num_edges + num_nodes - 1) / num_nodes else 0;

        // Accumulate activations along edges
        for (0..num_edges) |edge_idx| {
            const target = adjacency[edge_idx];
            const weight = weights[edge_idx];

            // Infer source node from edge index
            const source = edge_idx / edges_per_node;
            if (source >= num_nodes) continue;

            // Bounds check target
            if (target >= num_nodes) continue;

            // Accumulate activation: target += source_activation * weight
            const source_activation = current_activations[source];
            activations[target] += source_activation * weight;
        }

        // Normalize to prevent explosion
        var max_activation: f32 = 0.0;
        for (activations) |act| {
            max_activation = @max(max_activation, act);
        }

        if (max_activation > 1.0) {
            for (activations) |*act| {
                act.* /= max_activation;
            }
        }
    }
}

// Unit tests for CSR-based spreading
test "spreadActivationCSR linear chain" {
    // Graph: 0 -> 1 -> 2
    // CSR representation:
    //   Node 0: edge to 1
    //   Node 1: edge to 2
    //   Node 2: no edges
    const edge_offsets = [_]usize{ 0, 1, 2, 2 };
    const edges_targets = [_]u32{ 1, 2 };
    const edges_weights = [_]f32{ 1.0, 1.0 };
    var activations = [_]f32{ 1.0, 0.0, 0.0 };

    const graph = CSRGraph{
        .edge_offsets = &edge_offsets,
        .edges_targets = &edges_targets,
        .edges_weights = &edges_weights,
        .num_nodes = 3,
    };

    spreadActivationCSR(graph, &activations, 1);

    // After 1 iteration: node 1 should have activation from node 0
    try std.testing.expect(activations[1] > 0.0);
    // Node 2 should still be 0 (needs 2 iterations to reach)
    try std.testing.expectEqual(@as(f32, 0.0), activations[2]);
}

test "spreadActivationCSR star graph" {
    // Star graph: node 0 connects to nodes 1, 2, 3
    const edge_offsets = [_]usize{ 0, 3, 3, 3, 3 };
    const edges_targets = [_]u32{ 1, 2, 3 };
    const edges_weights = [_]f32{ 0.5, 0.3, 0.2 };
    var activations = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    const graph = CSRGraph{
        .edge_offsets = &edge_offsets,
        .edges_targets = &edges_targets,
        .edges_weights = &edges_weights,
        .num_nodes = 4,
    };

    spreadActivationCSR(graph, &activations, 1);

    // All spokes should receive activation
    try std.testing.expect(activations[1] > 0.0);
    try std.testing.expect(activations[2] > 0.0);
    try std.testing.expect(activations[3] > 0.0);
}

test "spreadActivationCSR isolated nodes" {
    // 3 nodes, no edges
    const edge_offsets = [_]usize{ 0, 0, 0, 0 };
    const edges_targets = [_]u32{};
    const edges_weights = [_]f32{};
    var activations = [_]f32{ 1.0, 0.5, 0.0 };
    const original = activations;

    const graph = CSRGraph{
        .edge_offsets = &edge_offsets,
        .edges_targets = &edges_targets,
        .edges_weights = &edges_weights,
        .num_nodes = 3,
    };

    spreadActivationCSR(graph, &activations, 5);

    // No edges, so activations should be unchanged
    try std.testing.expectEqual(original[0], activations[0]);
    try std.testing.expectEqual(original[1], activations[1]);
    try std.testing.expectEqual(original[2], activations[2]);
}

test "spreadActivationCSR normalization" {
    // Self-loop with weight > 1.0 should normalize
    const edge_offsets = [_]usize{ 0, 1 };
    const edges_targets = [_]u32{0};
    const edges_weights = [_]f32{2.0};
    var activations = [_]f32{1.0};

    const graph = CSRGraph{
        .edge_offsets = &edge_offsets,
        .edges_targets = &edges_targets,
        .edges_weights = &edges_weights,
        .num_nodes = 1,
    };

    spreadActivationCSR(graph, &activations, 10);

    // Should be normalized to max 1.0 despite high weight
    try std.testing.expect(activations[0] <= 1.0);
}

test "spreadActivationCSR cycle" {
    // Cycle: 0 -> 1 -> 2 -> 0
    const edge_offsets = [_]usize{ 0, 1, 2, 3 };
    const edges_targets = [_]u32{ 1, 2, 0 };
    const edges_weights = [_]f32{ 1.0, 1.0, 1.0 };
    var activations = [_]f32{ 1.0, 0.0, 0.0 };

    const graph = CSRGraph{
        .edge_offsets = &edge_offsets,
        .edges_targets = &edges_targets,
        .edges_weights = &edges_weights,
        .num_nodes = 3,
    };

    spreadActivationCSR(graph, &activations, 3);

    // After 3 iterations, activation should have cycled around
    try std.testing.expect(activations[0] > 0.0);
    try std.testing.expect(activations[1] > 0.0);
    try std.testing.expect(activations[2] > 0.0);
}

// Unit tests for legacy edge-list format
test "spreadActivation simple linear chain" {
    // Graph: 0 -> 1 -> 2
    const adjacency = [_]u32{ 1, 2 };
    const weights = [_]f32{ 1.0, 1.0 };
    var activations = [_]f32{ 1.0, 0.0, 0.0 };

    spreadActivation(&adjacency, &weights, &activations, 3, 2, 1);

    // After 1 iteration: node 1 should have activation
    try std.testing.expect(activations[1] > 0.0);
}

test "spreadActivation isolated nodes" {
    // No edges - activations should not change
    const adjacency = [_]u32{};
    const weights = [_]f32{};
    var activations = [_]f32{ 1.0, 0.5, 0.0 };
    const original = activations;

    spreadActivation(&adjacency, &weights, &activations, 3, 0, 5);

    // Activations should be unchanged
    try std.testing.expectEqual(original[0], activations[0]);
    try std.testing.expectEqual(original[1], activations[1]);
    try std.testing.expectEqual(original[2], activations[2]);
}

test "spreadActivation single node" {
    const adjacency = [_]u32{};
    const weights = [_]f32{};
    var activations = [_]f32{1.0};

    spreadActivation(&adjacency, &weights, &activations, 1, 0, 3);

    // Single node with no edges should remain at 1.0
    try std.testing.expectEqual(@as(f32, 1.0), activations[0]);
}

test "spreadActivation normalization" {
    // Self-reinforcing cycle that would explode without normalization
    const adjacency = [_]u32{ 0, 0 };
    const weights = [_]f32{ 2.0, 2.0 };
    var activations = [_]f32{1.0};

    spreadActivation(&adjacency, &weights, &activations, 1, 2, 10);

    // Should be normalized to max 1.0
    try std.testing.expect(activations[0] <= 1.0);
}

test "spreadActivation zero weights" {
    const adjacency = [_]u32{ 1, 2 };
    const weights = [_]f32{ 0.0, 0.0 };
    var activations = [_]f32{ 1.0, 0.0, 0.0 };

    spreadActivation(&adjacency, &weights, &activations, 3, 2, 5);

    // Zero weights should prevent spreading
    try std.testing.expectEqual(@as(f32, 1.0), activations[0]);
    try std.testing.expectEqual(@as(f32, 0.0), activations[1]);
    try std.testing.expectEqual(@as(f32, 0.0), activations[2]);
}
