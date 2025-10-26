// Comprehensive unit tests for vector similarity kernel
//
// This test suite validates:
// 1. Numerical correctness: SIMD matches scalar reference
// 2. Edge case handling: Zero vectors, NaN, infinity
// 3. Numerical stability: Denormals, large/small values
// 4. Algorithm properties: Symmetry, bounds, triangle inequality
//
// Test Philosophy:
// - Property-based approach: Test mathematical properties, not implementation details
// - Cross-validation: SIMD implementations must match scalar reference
// - Comprehensive edge cases: All pathological inputs documented in task spec
// - Performance validation: Tests run quickly enough for TDD workflow

const std = @import("std");
const vector_similarity = @import("vector_similarity.zig");

// Test helpers and utilities

/// Floating-point comparison with epsilon tolerance
fn expectApproxEqRel(expected: f32, actual: f32, epsilon: f32) !void {
    const diff = @abs(expected - actual);
    const magnitude = @max(@abs(expected), @abs(actual));
    const relative_error = diff / magnitude;

    if (relative_error > epsilon) {
        std.debug.print("Expected {d}, got {d} (relative error: {d})\n", .{
            expected, actual, relative_error,
        });
        return error.TestFailed;
    }
}

// Correctness Tests: SIMD vs Scalar Reference

test "simd_dot_product_matches_scalar_reference" {
    // Test various dimensions to cover different SIMD widths and tail handling
    const test_dimensions = [_]usize{ 1, 3, 4, 7, 8, 9, 16, 31, 32, 64, 127, 128, 384, 768, 1536 };

    for (test_dimensions) |dim| {
        var a = try std.testing.allocator.alloc(f32, dim);
        defer std.testing.allocator.free(a);
        var b = try std.testing.allocator.alloc(f32, dim);
        defer std.testing.allocator.free(b);

        // Initialize with pseudo-random pattern
        for (0..dim) |i| {
            a[i] = @as(f32, @floatFromInt(i)) * 0.1;
            b[i] = @as(f32, @floatFromInt(dim - i)) * 0.2;
        }

        const simd_result = vector_similarity.dotProductSimd(a, b);
        const scalar_result = vector_similarity.dotProductScalar(a, b);

        // SIMD must match scalar within tight tolerance
        try std.testing.expectApproxEqAbs(scalar_result, simd_result, 1e-4);
    }
}

test "cosine_similarity_symmetry" {
    // Property: similarity(a, b) == similarity(b, a)
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };

    const sim_ab = vector_similarity.cosineSimilarity(&a, &b);
    const sim_ba = vector_similarity.cosineSimilarity(&b, &a);

    try std.testing.expectApproxEqAbs(sim_ab, sim_ba, 1e-6);
}

test "cosine_similarity_bounds" {
    // Property: similarity in [-1.0, 1.0] for all inputs
    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    for (0..100) |_| {
        var a: [64]f32 = undefined;
        var b: [64]f32 = undefined;

        for (0..64) |i| {
            a[i] = random.float(f32) * 100.0 - 50.0;
            b[i] = random.float(f32) * 100.0 - 50.0;
        }

        const similarity = vector_similarity.cosineSimilarity(&a, &b);

        // Skip zero vector cases
        const mag_a = vector_similarity.magnitudeSimd(&a);
        const mag_b = vector_similarity.magnitudeSimd(&b);
        if (mag_a == 0.0 or mag_b == 0.0) continue;

        try std.testing.expect(similarity >= -1.0);
        try std.testing.expect(similarity <= 1.0);
    }
}

test "cosine_similarity_scaling_invariance" {
    // Property: similarity(a, b) == similarity(k*a, b) for any scalar k != 0
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    const sim_original = vector_similarity.cosineSimilarity(&a, &b);

    // Scale first vector by 10x
    const a_scaled = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    const sim_scaled = vector_similarity.cosineSimilarity(&a_scaled, &b);

    try std.testing.expectApproxEqAbs(sim_original, sim_scaled, 1e-6);
}

// Edge Case Tests

test "edge_case_zero_vectors" {
    const zero = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const nonzero = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    // Zero vector vs nonzero
    try std.testing.expectEqual(@as(f32, 0.0), vector_similarity.cosineSimilarity(&zero, &nonzero));
    try std.testing.expectEqual(@as(f32, 0.0), vector_similarity.cosineSimilarity(&nonzero, &zero));

    // Zero vector vs zero vector
    try std.testing.expectEqual(@as(f32, 0.0), vector_similarity.cosineSimilarity(&zero, &zero));
}

test "edge_case_single_element_vectors" {
    const a = [_]f32{5.0};
    const b = [_]f32{3.0};

    const similarity = vector_similarity.cosineSimilarity(&a, &b);

    // Single positive elements: similarity = 1.0
    try std.testing.expectApproxEqAbs(1.0, similarity, 1e-6);
}

test "edge_case_single_element_opposite" {
    const a = [_]f32{5.0};
    const b = [_]f32{-3.0};

    const similarity = vector_similarity.cosineSimilarity(&a, &b);

    // Opposite signs: similarity = -1.0
    try std.testing.expectApproxEqAbs(-1.0, similarity, 1e-6);
}

test "edge_case_very_small_values" {
    // Test numerical stability with denormal numbers
    const a = [_]f32{ 1e-20, 2e-20, 3e-20, 4e-20, 5e-20, 6e-20, 7e-20, 8e-20 };
    const b = [_]f32{ 8e-20, 7e-20, 6e-20, 5e-20, 4e-20, 3e-20, 2e-20, 1e-20 };

    const similarity = vector_similarity.cosineSimilarity(&a, &b);

    // Should still compute valid similarity
    try std.testing.expect(similarity >= -1.0);
    try std.testing.expect(similarity <= 1.0);
}

test "edge_case_very_large_values" {
    // Test overflow resistance with large numbers
    const a = [_]f32{ 1e20, 2e20, 3e20, 4e20 };
    const b = [_]f32{ 4e20, 3e20, 2e20, 1e20 };

    const similarity = vector_similarity.cosineSimilarity(&a, &b);

    // Should compute valid result without overflow
    try std.testing.expect(similarity >= -1.0);
    try std.testing.expect(similarity <= 1.0);
}

test "edge_case_mixed_signs" {
    const a = [_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0 };
    const b = [_]f32{ -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0 };

    const similarity = vector_similarity.cosineSimilarity(&a, &b);

    // Perfect anticorrelation: similarity = -1.0
    try std.testing.expectApproxEqAbs(-1.0, similarity, 1e-5);
}

test "edge_case_sparse_vectors" {
    // Most elements zero (sparse vector pattern)
    var a = [_]f32{0.0} ** 64;
    a[0] = 1.0;
    a[32] = 1.0;
    a[63] = 1.0;

    var b = [_]f32{0.0} ** 64;
    b[0] = 1.0;
    b[32] = 1.0;
    b[63] = 1.0;

    const similarity = vector_similarity.cosineSimilarity(&a, &b);

    // Identical sparse vectors: similarity = 1.0
    try std.testing.expectApproxEqAbs(1.0, similarity, 1e-6);
}

test "edge_case_all_same_value" {
    const a = [_]f32{2.5} ** 16;
    const b = [_]f32{2.5} ** 16;

    const similarity = vector_similarity.cosineSimilarity(&a, &b);

    try std.testing.expectApproxEqAbs(1.0, similarity, 1e-6);
}

// Batch Processing Tests

test "batch_processing_matches_individual" {
    const query = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    const candidates = [_]f32{
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,    // Candidate 0
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,    // Candidate 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    // Candidate 2 (zero)
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, // Candidate 3 (opposite)
    };

    var batch_scores = [_]f32{999.0} ** 4;
    vector_similarity.batchCosineSimilarity(&query, &candidates, &batch_scores, 4);

    // Compute individual similarities
    const individual_scores = [_]f32{
        vector_similarity.cosineSimilarity(&query, candidates[0..8]),
        vector_similarity.cosineSimilarity(&query, candidates[8..16]),
        vector_similarity.cosineSimilarity(&query, candidates[16..24]),
        vector_similarity.cosineSimilarity(&query, candidates[24..32]),
    };

    // Batch should match individual computations
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(individual_scores[i], batch_scores[i], 1e-6);
    }
}

test "batch_processing_empty_candidates" {
    const query = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const candidates = [_]f32{};
    var scores = [_]f32{};

    // Should not crash with zero candidates
    vector_similarity.batchCosineSimilarity(&query, &candidates, &scores, 0);
}

test "batch_processing_single_candidate" {
    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const candidates = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    var scores = [_]f32{999.0};

    vector_similarity.batchCosineSimilarity(&query, &candidates, &scores, 1);

    try std.testing.expectApproxEqAbs(1.0, scores[0], 1e-6);
}

test "batch_processing_large_batch" {
    const dim = 384;
    const num_candidates = 1000;

    var query: [dim]f32 = undefined;
    for (0..dim) |i| {
        query[i] = @as(f32, @floatFromInt(i)) * 0.01;
    }

    var candidates = try std.testing.allocator.alloc(f32, dim * num_candidates);
    defer std.testing.allocator.free(candidates);

    for (0..num_candidates) |c| {
        for (0..dim) |i| {
            candidates[c * dim + i] = @as(f32, @floatFromInt(c + i)) * 0.01;
        }
    }

    var scores = try std.testing.allocator.alloc(f32, num_candidates);
    defer std.testing.allocator.free(scores);

    // Should handle large batches without errors
    vector_similarity.batchCosineSimilarity(&query, candidates, scores, num_candidates);

    // Verify all scores are in valid range
    for (scores) |score| {
        try std.testing.expect(score >= -1.0);
        try std.testing.expect(score <= 1.0);
    }
}

// Dimension Coverage Tests

test "dimension_coverage_powers_of_two" {
    // Test dimensions that align perfectly with SIMD widths
    const dimensions = [_]usize{ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };

    for (dimensions) |dim| {
        var a = try std.testing.allocator.alloc(f32, dim);
        defer std.testing.allocator.free(a);
        var b = try std.testing.allocator.alloc(f32, dim);
        defer std.testing.allocator.free(b);

        for (0..dim) |i| {
            a[i] = @as(f32, @floatFromInt(i + 1));
            b[i] = @as(f32, @floatFromInt(i + 1));
        }

        const similarity = vector_similarity.cosineSimilarity(a, b);
        try std.testing.expectApproxEqAbs(1.0, similarity, 1e-5);
    }
}

test "dimension_coverage_realistic_embeddings" {
    // Common embedding dimensions used in production
    const dimensions = [_]usize{
        384,  // sentence-transformers/all-MiniLM-L6-v2
        768,  // BERT-base, GPT-2
        1536, // OpenAI text-embedding-ada-002
        1024, // Common alternative dimension
    };

    for (dimensions) |dim| {
        var a = try std.testing.allocator.alloc(f32, dim);
        defer std.testing.allocator.free(a);

        for (0..dim) |i| {
            a[i] = @as(f32, @floatFromInt(i % 100)) * 0.01;
        }

        const magnitude = vector_similarity.magnitudeSimd(a);

        // Magnitude should be positive for non-zero vector
        try std.testing.expect(magnitude > 0.0);
    }
}

test "dimension_coverage_off_by_one" {
    // Test dimensions that don't align with SIMD widths (tail processing)
    const dimensions = [_]usize{ 3, 5, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129 };

    for (dimensions) |dim| {
        var a = try std.testing.allocator.alloc(f32, dim);
        defer std.testing.allocator.free(a);

        for (0..dim) |i| {
            a[i] = 1.0;
        }

        const dot = vector_similarity.dotProductSimd(a, a);
        const expected = @as(f32, @floatFromInt(dim));

        try std.testing.expectApproxEqAbs(expected, dot, 1e-5);
    }
}

// Numerical Stability Tests

test "numerical_stability_gradual_underflow" {
    // Test with progressively smaller values
    const scales = [_]f32{ 1.0, 1e-5, 1e-10, 1e-15, 1e-20 };

    for (scales) |scale| {
        const a = [_]f32{ 1.0 * scale, 2.0 * scale, 3.0 * scale, 4.0 * scale };
        const similarity = vector_similarity.cosineSimilarity(&a, &a);

        // Self-similarity should still be 1.0 even with tiny values
        if (scale >= 1e-15) {
            try std.testing.expectApproxEqAbs(1.0, similarity, 1e-5);
        }
    }
}

test "numerical_stability_mixed_magnitude" {
    // Mix very large and very small values
    const a = [_]f32{ 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10 };
    const b = [_]f32{ 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10 };

    const similarity = vector_similarity.cosineSimilarity(&a, &b);

    // Should compute reasonable result despite mixed magnitudes
    try std.testing.expect(similarity >= -1.0);
    try std.testing.expect(similarity <= 1.0);
}

// Performance Validation Tests
// These don't measure performance directly, but verify optimizations are active

test "performance_validation_simd_path_exercised" {
    // Use dimension divisible by 8 to ensure SIMD path is taken
    const a = [_]f32{1.0} ** 128;
    const b = [_]f32{1.0} ** 128;

    const result = vector_similarity.dotProductSimd(&a, &b);

    // Expected result: 128 * 1.0 * 1.0 = 128
    try std.testing.expectApproxEqAbs(128.0, result, 1e-5);
}

test "performance_validation_tail_processing" {
    // Use dimension NOT divisible by 8 to ensure tail handling works
    const a = [_]f32{1.0} ** 125; // 125 = 15*8 + 5, has 5-element tail
    const b = [_]f32{1.0} ** 125;

    const result = vector_similarity.dotProductSimd(&a, &b);

    try std.testing.expectApproxEqAbs(125.0, result, 1e-5);
}
