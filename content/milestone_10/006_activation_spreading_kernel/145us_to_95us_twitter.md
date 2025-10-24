# Twitter Thread: 145μs → 95μs - Making Graph Traversal 35% Faster

## Tweet 1/10
Activation spreading models associative memory: "coffee" → "morning" → "alarm clock"

Profiling showed it consuming 28% of Engram's runtime.

Rust baseline: 145μs per spreading operation (1000 nodes, 5000 edges)
Zig optimized: 95μs

35% faster. Here's the breakdown.

## Tweet 2/10
The algorithm: BFS with weighted accumulation.

1. Activate source node
2. Propagate to neighbors (activation × weight × decay)
3. Accumulate incoming activation
4. Repeat until convergence

Simple concept. Performance is all about data structure layout.

## Tweet 3/10
Optimization 1: CSR (Compressed Sparse Row) graph representation

Instead of HashMap<Node, Vec<Edge>>, use:
```zig
edge_offsets: []usize  // offsets[i] = start of edges for node i
edges: []Edge          // All edges, contiguous
```

Edges for a node are sequential in memory = cache-friendly = fast iteration.

Saved: ~20μs (cache locality)

## Tweet 4/10
Optimization 2: Flat activation arrays

Replace HashMap<NodeId, f32> with Vec<f32>:

```zig
var activations: [1000]f32;
activations[source] = 1.0;
```

Benefits:
- O(1) index access (no hash computation)
- Sequential iteration (cache-friendly)
- Bulk memset (vectorized)

Saved: ~15μs (no HashMap overhead)

## Tweet 5/10
Optimization 3: Frontier tracking

Don't iterate ALL nodes - only active ones:

```zig
var frontier: []u32 = [source];

for (iterations) {
    for (frontier) |node| {  // Only active nodes
        propagate_to_neighbors(node);
    }
}
```

For sparse graphs: ~50 active nodes instead of 1000 total.
20x less iteration.

Saved: ~10μs (sparse iteration)

## Tweet 6/10
Optimization 4: SIMD edge batching

Process 8 edge weights in parallel:

```zig
const src_act_vec: @Vector(8, f32) = @splat(source_act * decay);

while (i < edges.len) : (i += 8) {
    var weights: @Vector(8, f32) = load_8_weights(edges[i..]);
    const deltas = src_act_vec * weights;  // 8 multiplies in parallel
    scatter_to_targets(deltas, edges[i..]);
}
```

Saved: ~5μs (vectorized weight multiplication)

## Tweet 7/10
Optimization 5: Arena allocation

Thread-local arena for frontier arrays:

```zig
const arena = getThreadArena();
defer resetThreadArena();

var frontier = arena.alloc(u32, num_nodes);
```

malloc: 50-100ns per allocation
Arena: 5ns per allocation

Overhead: negligible

## Tweet 8/10
Performance breakdown:

Rust baseline: 145μs
- CSR representation: -20μs → 125μs
- Flat arrays: -15μs → 110μs
- Frontier tracking: -10μs → 100μs
- SIMD edge batching: -5μs → 95μs

Total improvement: 50μs (35% faster)

## Tweet 9/10
Correctness validation with property-based testing:

5,000 random graphs (10-100 nodes, varying density)
100% match between Zig and Rust (epsilon = 1e-5)

Bugs differential testing caught:
- Frontier queue off-by-one
- Delta accumulation order dependency
- Early termination edge case

## Tweet 10/10
Key lessons:

1. Data layout > algorithm tricks (CSR representation biggest win)
2. Flat arrays beat HashMaps for dense IDs
3. Frontier tracking for sparse graphs
4. SIMD helps even for irregular access patterns
5. Profile, optimize bottlenecks, validate correctness

145μs → 95μs. Microseconds matter at scale.
