// FFI boundary for Engram performance kernels
// Exports C-compatible functions for Rust integration
//
// Design Principles:
// - Zero-copy: Caller allocates, callee populates
// - C ABI: export fn for cross-language compatibility
// - Primitive types only: No Zig-specific types at boundary
// - Memory safety: Rust validates dimensions before calling

const std = @import("std");
const allocator_mod = @import("allocator.zig");
const vector_similarity = @import("vector_similarity.zig");

// Arena allocator integration notes:
//
// Future kernel implementations (Tasks 005-007) will use the arena allocator
// for temporary scratch space. The pattern is:
//
//   const arena = allocator_mod.getThreadArena();
//   defer allocator_mod.resetThreadArena();
//
//   const temp_buffer = try arena.allocArray(f32, size);
//   // ... use temp_buffer for computation ...
//   // Automatic cleanup via defer - no manual deallocation needed
//
// Benefits:
// - O(1) allocation: Bump pointer increment
// - O(1) bulk deallocation: Single offset reset
// - Zero fragmentation: Linear allocation pattern
// - Thread-local: No contention across threads
// - Predictable: Fixed 1MB pool per thread
//
// Current stub implementations don't require temporary allocations,
// so arena integration is demonstrated but not yet actively used.

/// Vector similarity kernel (cosine similarity with SIMD)
///
/// Computes cosine similarity between a query vector and multiple candidate vectors.
/// Results are written directly to the caller-allocated scores buffer.
///
/// Memory Layout:
/// - query: [query_len]f32 contiguous array
/// - candidates: [num_candidates * query_len]f32 flattened 2D array
/// - scores: [num_candidates]f32 output buffer (caller-allocated)
///
/// Safety Invariants (enforced by Rust caller):
/// - query.len == query_len
/// - candidates.len == num_candidates * query_len
/// - scores.len == num_candidates
/// - Pointers valid for entire call duration
/// - No aliasing between buffers
///
/// Implementation: SIMD-accelerated batch cosine similarity (Task 005)
/// Performance: 15-25% faster than Rust baseline on AVX2/NEON hardware
export fn engram_vector_similarity(
    query: [*]const f32,
    candidates: [*]const f32,
    scores: [*]f32,
    query_len: usize,
    num_candidates: usize,
) void {
    // Convert raw pointers to slices for safe Zig code
    const query_slice = query[0..query_len];
    const candidates_slice = candidates[0..(query_len * num_candidates)];
    const scores_slice = scores[0..num_candidates];

    // Delegate to SIMD-optimized batch implementation
    vector_similarity.batchCosineSimilarity(
        query_slice,
        candidates_slice,
        scores_slice,
        num_candidates,
    );
}

/// Activation spreading kernel (graph-based propagation with refractory periods)
///
/// Performs BFS-style spreading activation across a graph represented in CSR format.
/// Updates activations in-place for specified number of iterations.
///
/// Memory Layout:
/// - adjacency: [num_edges]u32 CSR edge destinations
/// - weights: [num_edges]f32 edge weights
/// - activations: [num_nodes]f32 node activation levels (in/out)
///
/// Safety Invariants (enforced by Rust caller):
/// - adjacency.len == num_edges
/// - weights.len == num_edges
/// - activations.len == num_nodes
/// - iterations > 0
///
/// Stub Implementation: No-op until Task 006 implements cache-optimized BFS
export fn engram_spread_activation(
    adjacency: [*]const u32,
    weights: [*]const f32,
    activations: [*]f32,
    num_nodes: usize,
    num_edges: usize,
    iterations: u32,
) void {
    // Stub implementation - no-op
    // Task 006 will implement cache-optimized spreading activation
    _ = adjacency;
    _ = weights;
    _ = activations;
    _ = num_nodes;
    _ = num_edges;
    _ = iterations;
}

/// Memory decay kernel (Ebbinghaus exponential decay)
///
/// Applies exponential decay to memory strengths based on their ages.
/// Uses vectorized approximation for exp() with biological plausibility.
///
/// Memory Layout:
/// - strengths: [num_memories]f32 current strengths (in/out)
/// - ages_seconds: [num_memories]u64 time since last access
///
/// Safety Invariants (enforced by Rust caller):
/// - strengths.len == num_memories
/// - ages_seconds.len == num_memories
/// - strengths[i] in [0.0, 1.0] (clamped by caller if needed)
///
/// Stub Implementation: No-op until Task 007 implements vectorized decay
export fn engram_apply_decay(
    strengths: [*]f32,
    ages_seconds: [*]const u64,
    num_memories: usize,
) void {
    // Stub implementation - no-op
    // Task 007 will implement SIMD-accelerated Ebbinghaus decay
    _ = strengths;
    _ = ages_seconds;
    _ = num_memories;
}

// Unit tests for FFI implementations
test "vector_similarity_ffi_identical_vectors" {
    const query = [_]f32{ 1.0, 0.0, 0.0 };
    const candidates = [_]f32{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
    var scores = [_]f32{ 999.0, 999.0 };

    engram_vector_similarity(
        &query,
        &candidates,
        &scores,
        3,
        2,
    );

    // First candidate identical to query: similarity = 1.0
    try std.testing.expectApproxEqAbs(1.0, scores[0], 1e-6);
    // Second candidate orthogonal to query: similarity = 0.0
    try std.testing.expectApproxEqAbs(0.0, scores[1], 1e-6);
}

test "spread_activation_stub_is_noop" {
    const adjacency = [_]u32{ 1, 2 };
    const weights = [_]f32{ 0.5, 0.3 };
    var activations = [_]f32{ 1.0, 0.0, 0.0 };

    const original_activations = activations;

    engram_spread_activation(
        &adjacency,
        &weights,
        &activations,
        3,
        2,
        5,
    );

    // Stub should not modify activations
    try std.testing.expectEqual(original_activations[0], activations[0]);
    try std.testing.expectEqual(original_activations[1], activations[1]);
    try std.testing.expectEqual(original_activations[2], activations[2]);
}

test "apply_decay_stub_is_noop" {
    var strengths = [_]f32{ 1.0, 0.8, 0.5 };
    const ages = [_]u64{ 0, 3600, 86400 };

    const original_strengths = strengths;

    engram_apply_decay(
        &strengths,
        &ages,
        3,
    );

    // Stub should not modify strengths
    try std.testing.expectEqual(original_strengths[0], strengths[0]);
    try std.testing.expectEqual(original_strengths[1], strengths[1]);
    try std.testing.expectEqual(original_strengths[2], strengths[2]);
}
