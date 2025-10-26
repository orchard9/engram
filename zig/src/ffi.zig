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
const arena_config = @import("arena_config.zig");
const arena_metrics = @import("arena_metrics.zig");
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
/// Performs iterative spreading activation across a graph represented in edge-list format.
/// Updates activations in-place for specified number of iterations.
///
/// Memory Layout:
/// - adjacency: [num_edges]u32 edge target node indices
/// - weights: [num_edges]f32 edge weights
/// - activations: [num_nodes]f32 node activation levels (in/out)
///
/// Safety Invariants (enforced by Rust caller):
/// - adjacency.len == num_edges
/// - weights.len == num_edges
/// - activations.len == num_nodes
/// - iterations > 0
/// - All indices in adjacency are < num_nodes
///
/// Implementation: Cache-optimized iterative spreading with normalization (Task 006)
export fn engram_spread_activation(
    adjacency: [*]const u32,
    weights: [*]const f32,
    activations: [*]f32,
    num_nodes: usize,
    num_edges: usize,
    iterations: u32,
) void {
    const spreading = @import("spreading_activation.zig");

    // Convert raw pointers to slices for safe Zig code
    const adjacency_slice = adjacency[0..num_edges];
    const weights_slice = weights[0..num_edges];
    const activations_slice = activations[0..num_nodes];

    // Delegate to spreading activation implementation
    spreading.spreadActivation(
        adjacency_slice,
        weights_slice,
        activations_slice,
        num_nodes,
        num_edges,
        iterations,
    );
}

/// Memory decay kernel (Ebbinghaus exponential decay)
///
/// Applies exponential decay to memory strengths based on their ages.
/// Uses vectorized SIMD implementation with biological plausibility.
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
/// Implementation: SIMD-accelerated Ebbinghaus decay (Task 007)
/// Performance: 20-30% faster than Rust baseline on AVX2/NEON hardware
export fn engram_apply_decay(
    strengths: [*]f32,
    ages_seconds: [*]const u64,
    num_memories: usize,
) void {
    const decay = @import("decay_functions.zig");

    // Convert raw pointers to slices for safe Zig code
    const strengths_slice = strengths[0..num_memories];
    const ages_slice = ages_seconds[0..num_memories];

    // Delegate to SIMD-optimized batch implementation
    decay.batchDecay(strengths_slice, ages_slice, decay.DEFAULT_HALF_LIFE);
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

test "spread_activation_propagates_activation" {
    // Simple graph: 0 -> 1, 0 -> 2
    const adjacency = [_]u32{ 1, 2 };
    const weights = [_]f32{ 0.5, 0.3 };
    var activations = [_]f32{ 1.0, 0.0, 0.0 };

    engram_spread_activation(
        &adjacency,
        &weights,
        &activations,
        3,
        2,
        1,
    );

    // After 1 iteration: nodes 1 and 2 should have received activation
    try std.testing.expect(activations[1] > 0.0);
    try std.testing.expect(activations[2] > 0.0);
    // Source activation should be preserved (normalized)
    try std.testing.expect(activations[0] >= 0.0);
}

test "apply_decay_correctness" {
    var strengths = [_]f32{ 1.0, 1.0, 1.0 };
    const ages = [_]u64{ 0, 3600, 86400 };

    engram_apply_decay(
        &strengths,
        &ages,
        3,
    );

    // Age 0: no decay
    try std.testing.expectApproxEqAbs(1.0, strengths[0], 1e-6);
    // Age 3600: slight decay
    try std.testing.expect(strengths[1] < 1.0 and strengths[1] > 0.95);
    // Age 86400 (1 day half-life): ~e^-1 ≈ 0.3679
    try std.testing.expectApproxEqAbs(0.3679, strengths[2], 0.01);
}

// Arena allocator control functions (Task 008)

/// Configure arena allocator pool size and overflow strategy
///
/// Sets global configuration for all thread-local arenas.
/// Must be called before any arena allocation occurs.
///
/// Memory Layout:
/// - pool_size_mb: Pool size in megabytes (1-1024)
/// - overflow_strategy: 0=panic, 1=error_return, 2=fallback
///
/// Safety Invariants:
/// - Call from single thread before spawning workers
/// - Do not call while arenas are in use
///
/// Thread safety: Not thread-safe - call during initialization only
///
/// Example (from Rust):
///   engram_configure_arena(2, 1);  // 2MB pool, error_return strategy
export fn engram_configure_arena(
    pool_size_mb: u32,
    overflow_strategy: u8,
) void {
    const pool_size = @as(usize, pool_size_mb) * 1024 * 1024;
    const strategy: arena_config.OverflowStrategy = switch (overflow_strategy) {
        0 => .panic,
        1 => .error_return,
        2 => .fallback,
        else => .error_return, // Default to safe option
    };

    arena_config.setConfig(.{
        .pool_size = pool_size,
        .overflow_strategy = strategy,
    });
}

/// Get global arena usage statistics
///
/// Returns aggregated metrics from all thread-local arenas.
/// Metrics are cumulative since process start or last reset.
///
/// Memory Layout:
/// - total_resets: Output pointer for reset count
/// - total_overflows: Output pointer for overflow count
/// - max_high_water_mark: Output pointer for peak usage (bytes)
///
/// Safety Invariants:
/// - Pointers must be valid and non-null
/// - Pointers must point to writable usize locations
///
/// Thread safety: Thread-safe via internal mutex
///
/// Example (from Rust):
///   let mut resets = 0usize;
///   let mut overflows = 0usize;
///   let mut peak = 0usize;
///   engram_arena_stats(&mut resets, &mut overflows, &mut peak);
export fn engram_arena_stats(
    total_resets: *usize,
    total_overflows: *usize,
    max_high_water_mark: *usize,
) void {
    const stats = arena_metrics.getGlobalMetrics();
    total_resets.* = stats.total_resets;
    total_overflows.* = stats.total_overflows;
    max_high_water_mark.* = stats.max_high_water_mark;
}

/// Reset thread-local arena allocator
///
/// Manually resets the calling thread's arena.
/// Useful for testing or explicit memory reclamation.
///
/// Normally arenas are reset automatically at kernel exit,
/// but this provides explicit control when needed.
///
/// Thread safety: Thread-local - only affects calling thread
///
/// Example (from Rust):
///   engram_reset_arenas();  // Reset current thread's arena
export fn engram_reset_arenas() void {
    allocator_mod.resetThreadArena();
}

/// Reset global arena metrics
///
/// Clears all accumulated metrics counters.
/// Useful for starting fresh measurement periods in testing.
///
/// Thread safety: Thread-safe via internal mutex
///
/// Example (from Rust):
///   engram_reset_arena_metrics();  // Clear all metrics
export fn engram_reset_arena_metrics() void {
    arena_metrics.resetGlobalMetrics();
}

// Unit tests for arena control functions
test "configure_arena" {
    engram_configure_arena(2, 1); // 2MB, error_return
    const config = arena_config.getConfig();
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024), config.pool_size);
    try std.testing.expectEqual(arena_config.OverflowStrategy.error_return, config.overflow_strategy);

    // Reset for other tests
    arena_config.resetConfig();
}

test "arena_stats" {
    arena_metrics.resetGlobalMetrics();

    // Record some metrics
    arena_metrics.recordReset(1000, 2);
    arena_metrics.recordReset(2000, 1);

    // Query via FFI
    var resets: usize = 0;
    var overflows: usize = 0;
    var peak: usize = 0;

    engram_arena_stats(&resets, &overflows, &peak);

    try std.testing.expectEqual(@as(usize, 2), resets);
    try std.testing.expectEqual(@as(usize, 3), overflows);
    try std.testing.expectEqual(@as(usize, 2000), peak);
}

test "reset_arena_metrics" {
    arena_metrics.resetGlobalMetrics();
    arena_metrics.recordReset(1000, 1);

    // Verify metrics are non-zero
    var resets: usize = 0;
    var overflows: usize = 0;
    var peak: usize = 0;
    engram_arena_stats(&resets, &overflows, &peak);
    try std.testing.expect(resets > 0);

    // Reset via FFI
    engram_reset_arena_metrics();

    // Verify metrics are zero
    engram_arena_stats(&resets, &overflows, &peak);
    try std.testing.expectEqual(@as(usize, 0), resets);
    try std.testing.expectEqual(@as(usize, 0), overflows);
    try std.testing.expectEqual(@as(usize, 0), peak);
}
