# Task 006: Activation Spreading Kernel

**Duration:** 3 days
**Status:** Pending
**Dependencies:** 005 (Vector Similarity Kernel)

## Objectives

Implement cache-optimized activation spreading kernel in Zig to accelerate graph traversal and activation propagation. This is the second-highest impact optimization, accounting for 20-30% of compute time. The implementation must handle arbitrary graph topologies while maintaining numerical consistency with Rust baseline.

1. **BFS-based spreading** - Breadth-first traversal with activation accumulation
2. **Edge batching** - Cache-friendly access patterns for adjacency data
3. **Decay integration** - Apply refractory period decay during propagation
4. **Performance validation** - 20-35% improvement over Rust baseline

## Dependencies

- Task 005 (Vector Similarity Kernel) - SIMD infrastructure and testing patterns established

## Deliverables

### Files to Create

1. `/zig/src/spreading_activation.zig` - Core implementation
   - spreadActivation: BFS with activation accumulation
   - Edge iteration with cache-optimal layout
   - Refractory period decay application

2. `/zig/src/spreading_activation_test.zig` - Unit tests
   - Simple graph topologies (linear, star, cycle)
   - Edge cases (isolated nodes, self-loops)
   - Activation threshold behavior

3. `/benches/spreading_comparison.rs` - Performance benchmarks
   - Rust baseline vs. Zig kernel
   - Various graph sizes (100, 1000, 10000 nodes)
   - Different connectivity patterns (sparse, dense)

### Files to Modify

1. `/zig/src/ffi.zig` - Export spreading function
   - engram_spread_activation implementation
   - Graph data structure marshaling

2. `/src/zig_kernels/mod.rs` - Rust wrapper
   - Safe wrapper for spread_activation
   - Graph serialization to FFI-compatible format
   - Activation map reconstruction

3. `/tests/zig_differential/spreading_activation.rs` - Extend differential tests
   - Property-based graph generation
   - Validate activation distributions

## Acceptance Criteria

1. All differential tests pass (Zig matches Rust within epsilon = 1e-5)
2. Performance improvement: 20-35% faster than Rust baseline
3. Correctly handles various graph topologies (DAGs, cycles, disconnected components)
4. Activation thresholds and decay applied consistently
5. Benchmarks show improvement across sparse and dense graphs

## Implementation Guidance

### Graph Representation for FFI

Use compressed sparse row (CSR) format for cache-friendly access:

```zig
const std = @import("std");

/// Graph in CSR (Compressed Sparse Row) format
pub const CSRGraph = struct {
    /// Adjacency list offsets: edge_offsets[i] = start index in edges for node i
    edge_offsets: []const usize,

    /// Target nodes and weights: edges[edge_offsets[i]..edge_offsets[i+1]]
    edges: []const Edge,

    /// Number of nodes
    num_nodes: usize,

    pub const Edge = struct {
        target: u32,
        weight: f32,
    };

    /// Get edges for a node
    pub fn getEdges(self: CSRGraph, node: usize) []const Edge {
        const start = self.edge_offsets[node];
        const end = if (node + 1 < self.num_nodes)
            self.edge_offsets[node + 1]
        else
            self.edges.len;
        return self.edges[start..end];
    }
};
```

### Activation Spreading Algorithm

Implement BFS with activation accumulation and decay:

```zig
const allocator = @import("allocator.zig");

pub fn spreadActivation(
    graph: CSRGraph,
    source: u32,
    initial_activation: f32,
    iterations: u32,
    threshold: f32,
    decay_rate: f32,
    activations: []f32,
) void {
    const num_nodes = graph.num_nodes;
    std.debug.assert(activations.len == num_nodes);

    // Initialize activations
    @memset(activations, 0.0);
    activations[source] = initial_activation;

    // Get thread-local arena for temporary buffers
    const arena = allocator.getThreadArena();
    defer allocator.resetThreadArena();

    // Allocate frontier queues
    var current_frontier = arena.allocArray(u32, num_nodes) catch return;
    var next_frontier = arena.allocArray(u32, num_nodes) catch return;
    var current_size: usize = 1;
    var next_size: usize = 0;

    current_frontier[0] = source;

    // Activation deltas for next iteration
    var delta_activations = arena.allocArray(f32, num_nodes) catch return;

    for (0..iterations) |_| {
        @memset(delta_activations, 0.0);
        next_size = 0;

        // Process current frontier
        for (current_frontier[0..current_size]) |node| {
            const activation = activations[node];
            if (activation < threshold) continue;

            // Propagate to neighbors
            const edges = graph.getEdges(node);
            for (edges) |edge| {
                const target = edge.target;
                const weight = edge.weight;

                // Accumulate activation
                const delta = activation * weight * decay_rate;
                delta_activations[target] += delta;

                // Add to next frontier if newly activated
                if (activations[target] < threshold and
                    delta_activations[target] >= threshold) {
                    next_frontier[next_size] = target;
                    next_size += 1;
                }
            }
        }

        // Apply deltas
        for (0..num_nodes) |i| {
            activations[i] += delta_activations[i];
        }

        // Swap frontiers
        const temp = current_frontier;
        current_frontier = next_frontier;
        next_frontier = temp;
        current_size = next_size;

        // Early termination if no active nodes
        if (current_size == 0) break;
    }
}
```

### Optimized Edge Iteration

Use SIMD for parallel edge weight processing:

```zig
/// Process edges in batches for cache efficiency
fn propagateEdges(
    edges: []const CSRGraph.Edge,
    source_activation: f32,
    decay_rate: f32,
    delta_activations: []f32,
) void {
    const simd_width = 8;
    const simd_len = (edges.len / simd_width) * simd_width;

    // SIMD broadcast of source activation and decay
    const src_act_vec: @Vector(8, f32) = @splat(source_activation * decay_rate);

    var i: usize = 0;
    while (i < simd_len) : (i += simd_width) {
        // Load edge weights
        var weights: @Vector(8, f32) = undefined;
        inline for (0..8) |j| {
            weights[j] = edges[i + j].weight;
        }

        // Compute deltas
        const deltas = src_act_vec * weights;

        // Scatter to target nodes (no SIMD gather/scatter in Zig yet)
        inline for (0..8) |j| {
            const target = edges[i + j].target;
            delta_activations[target] += deltas[j];
        }
    }

    // Handle remaining edges
    while (i < edges.len) : (i += 1) {
        const target = edges[i].target;
        const weight = edges[i].weight;
        delta_activations[target] += source_activation * decay_rate * weight;
    }
}
```

### FFI Integration

```zig
// ffi.zig
const spreading = @import("spreading_activation.zig");

/// FFI function for activation spreading
export fn engram_spread_activation(
    edge_offsets: [*]const usize,
    edges_targets: [*]const u32,
    edges_weights: [*]const f32,
    num_nodes: usize,
    num_edges: usize,
    source: u32,
    initial_activation: f32,
    iterations: u32,
    threshold: f32,
    decay_rate: f32,
    activations: [*]f32,
) void {
    // Reconstruct CSR graph
    const graph = spreading.CSRGraph{
        .edge_offsets = edge_offsets[0..(num_nodes + 1)],
        .edges = undefined, // Build from separate arrays
        .num_nodes = num_nodes,
    };

    // Pack edges from parallel arrays
    const arena = @import("allocator.zig").getThreadArena();
    defer @import("allocator.zig").resetThreadArena();

    var edges = arena.allocArray(spreading.CSRGraph.Edge, num_edges) catch return;
    for (0..num_edges) |i| {
        edges[i] = .{
            .target = edges_targets[i],
            .weight = edges_weights[i],
        };
    }

    const graph_with_edges = spreading.CSRGraph{
        .edge_offsets = graph.edge_offsets,
        .edges = edges,
        .num_nodes = num_nodes,
    };

    spreading.spreadActivation(
        graph_with_edges,
        source,
        initial_activation,
        iterations,
        threshold,
        decay_rate,
        activations[0..num_nodes],
    );
}
```

### Rust Wrapper

```rust
// src/zig_kernels/mod.rs

#[cfg(feature = "zig-kernels")]
pub fn spread_activation(
    graph: &MemoryGraph,
    source: NodeId,
    initial_activation: f32,
    iterations: u32,
) -> HashMap<NodeId, f32> {
    // Convert graph to CSR format
    let (edge_offsets, edges_targets, edges_weights) = graph.to_csr();
    let num_nodes = graph.node_count();
    let num_edges = graph.edge_count();

    let mut activations = vec![0.0_f32; num_nodes];

    unsafe {
        ffi::engram_spread_activation(
            edge_offsets.as_ptr(),
            edges_targets.as_ptr(),
            edges_weights.as_ptr(),
            num_nodes,
            num_edges,
            source.0,
            initial_activation,
            iterations,
            ACTIVATION_THRESHOLD,
            DECAY_RATE,
            activations.as_mut_ptr(),
        );
    }

    // Convert back to HashMap
    activations
        .into_iter()
        .enumerate()
        .filter(|(_, act)| *act > ACTIVATION_THRESHOLD)
        .map(|(idx, act)| (NodeId(idx as u32), act))
        .collect()
}
```

### Performance Benchmarks

```rust
// benches/spreading_comparison.rs
fn bench_spreading_activation(c: &mut Criterion) {
    let mut group = c.benchmark_group("spreading_activation");

    for num_nodes in [100, 1000, 10_000] {
        for avg_degree in [5, 10, 20] {
            let graph = generate_random_graph(num_nodes, avg_degree);
            let source = graph.random_node();

            // Rust baseline
            group.bench_with_input(
                BenchmarkId::new("rust", format!("{}n_{}d", num_nodes, avg_degree)),
                &(&graph, source),
                |b, (graph, source)| {
                    b.iter(|| {
                        let result = rust_spread_activation(graph, *source, 1.0, 100);
                        black_box(result);
                    });
                },
            );

            // Zig kernel
            #[cfg(feature = "zig-kernels")]
            group.bench_with_input(
                BenchmarkId::new("zig", format!("{}n_{}d", num_nodes, avg_degree)),
                &(&graph, source),
                |b, (graph, source)| {
                    b.iter(|| {
                        let result = zig_spread_activation(graph, *source, 1.0, 100);
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}
```

## Testing Approach

1. **Unit tests**
   - Simple topologies (linear chain, star, complete graph)
   - Verify activation propagation correctness
   - Test threshold cutoff behavior

2. **Differential tests**
   - Property-based graph generation
   - Verify activation distributions match Rust
   - Test with various decay rates and thresholds

3. **Performance validation**
   - Benchmark sparse vs. dense graphs
   - Verify 20-35% improvement
   - Test frontier queue efficiency

## Integration Points

- **Task 003 (Differential Testing)** - Graph generation strategies
- **Task 007 (Decay Function)** - Integrate decay calculations
- **Task 009 (Integration Testing)** - Validate in full memory consolidation pipeline

## Notes

- CSR format provides cache-friendly sequential access to edges
- Consider edge reordering for better cache locality (sort by target)
- Frontier queue optimization is critical for sparse graphs
- Document activation threshold tuning in operational guide
- Profile memory bandwidth (may be bottleneck, not compute)
