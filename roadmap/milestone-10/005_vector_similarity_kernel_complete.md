# Task 005: Vector Similarity Kernel

**Duration:** 3 days
**Status:** Pending
**Dependencies:** 004 (Memory Pool Allocator)

## Objectives

Implement high-performance vector similarity kernel in Zig using SIMD intrinsics to accelerate cosine similarity calculations. This is the highest-impact optimization target, accounting for 15-25% of total compute time in profiling results.

1. **SIMD cosine similarity** - Vectorized dot product and magnitude calculation
2. **Batch processing** - Optimize for 1:N similarity (one query vs. many candidates)
3. **Numerical stability** - Handle edge cases (zero vectors, denormals)
4. **Performance validation** - 15-25% improvement over Rust baseline

## Dependencies

- Task 004 (Memory Pool Allocator) - Arena allocator for temporary buffers

## Deliverables

### Files to Create

1. `/zig/src/vector_similarity.zig` - SIMD implementation
   - cosine_similarity_simd for single pair
   - batch_cosine_similarity for 1:N queries
   - SIMD vector operations (dot product, magnitude)

2. `/zig/src/vector_similarity_test.zig` - Unit tests
   - Correctness tests for SIMD operations
   - Edge case handling (zeros, infinities)
   - Numerical stability tests

3. `/benches/vector_similarity_comparison.rs` - Performance benchmarks
   - Rust baseline vs. Zig kernel
   - Various dimensions (384, 768, 1536)
   - Batch sizes (10, 100, 1000 candidates)

### Files to Modify

1. `/zig/src/ffi.zig` - Export vector similarity function
   - engram_vector_similarity implementation
   - Integrate with arena allocator

2. `/src/zig_kernels/mod.rs` - Rust wrapper
   - Safe wrapper for vector_similarity
   - Dimension validation
   - Score normalization

3. `/tests/zig_differential/vector_similarity.rs` - Extend differential tests
   - Add SIMD-specific edge cases
   - Validate numerical equivalence

## Acceptance Criteria

1. All differential tests pass (Zig matches Rust within epsilon = 1e-6)
2. Performance improvement: 15-25% faster than Rust baseline
3. SIMD code paths validated on both x86_64 (AVX2) and ARM64 (NEON)
4. Benchmarks show consistent performance across varying dimensions
5. Edge cases handled correctly (zero vectors, NaN, Inf)

## Implementation Guidance

### SIMD Cosine Similarity

Implement vectorized cosine similarity using platform-specific intrinsics:

```zig
const std = @import("std");
const builtin = @import("builtin");

/// Compute cosine similarity between two vectors using SIMD
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    const len = a.len;
    if (len == 0) return 0.0;

    // Use SIMD for bulk computation
    const dot = dotProductSimd(a, b);
    const mag_a = magnitudeSimd(a);
    const mag_b = magnitudeSimd(b);

    // Handle zero vectors
    if (mag_a == 0.0 or mag_b == 0.0) {
        return 0.0;
    }

    return dot / (mag_a * mag_b);
}

/// SIMD dot product
fn dotProductSimd(a: []const f32, b: []const f32) f32 {
    const len = a.len;

    if (builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
        return dotProductAvx2(a, b);
    } else if (builtin.cpu.arch == .aarch64) {
        return dotProductNeon(a, b);
    } else {
        return dotProductScalar(a, b);
    }
}

/// AVX2 dot product (process 8 floats at a time)
fn dotProductAvx2(a: []const f32, b: []const f32) f32 {
    const len = a.len;
    const simd_width = 8;
    const simd_len = (len / simd_width) * simd_width;

    var sum: @Vector(8, f32) = @splat(0.0);

    var i: usize = 0;
    while (i < simd_len) : (i += simd_width) {
        const va: @Vector(8, f32) = a[i..][0..simd_width].*;
        const vb: @Vector(8, f32) = b[i..][0..simd_width].*;
        sum += va * vb;
    }

    // Horizontal sum
    var result: f32 = @reduce(.Add, sum);

    // Handle remaining elements
    while (i < len) : (i += 1) {
        result += a[i] * b[i];
    }

    return result;
}

/// NEON dot product (process 4 floats at a time)
fn dotProductNeon(a: []const f32, b: []const f32) f32 {
    const len = a.len;
    const simd_width = 4;
    const simd_len = (len / simd_width) * simd_width;

    var sum: @Vector(4, f32) = @splat(0.0);

    var i: usize = 0;
    while (i < simd_len) : (i += simd_width) {
        const va: @Vector(4, f32) = a[i..][0..simd_width].*;
        const vb: @Vector(4, f32) = b[i..][0..simd_width].*;
        sum += va * vb;
    }

    var result: f32 = @reduce(.Add, sum);

    while (i < len) : (i += 1) {
        result += a[i] * b[i];
    }

    return result;
}

/// Scalar fallback
fn dotProductScalar(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |x, y| {
        sum += x * y;
    }
    return sum;
}

/// SIMD magnitude (L2 norm)
fn magnitudeSimd(v: []const f32) f32 {
    const dot = dotProductSimd(v, v);
    return @sqrt(dot);
}
```

### Batch Processing Optimization

Optimize for 1:N queries with cache-friendly access patterns:

```zig
/// Batch cosine similarity: one query vs. many candidates
pub fn batchCosineSimilarity(
    query: []const f32,
    candidates: []const f32,
    scores: []f32,
    num_candidates: usize,
) void {
    const dim = query.len;
    std.debug.assert(candidates.len == dim * num_candidates);
    std.debug.assert(scores.len == num_candidates);

    // Pre-compute query magnitude (amortize across all candidates)
    const query_mag = magnitudeSimd(query);
    if (query_mag == 0.0) {
        @memset(scores, 0.0);
        return;
    }

    // Process candidates
    for (0..num_candidates) |i| {
        const candidate_start = i * dim;
        const candidate = candidates[candidate_start..][0..dim];

        const dot = dotProductSimd(query, candidate);
        const candidate_mag = magnitudeSimd(candidate);

        if (candidate_mag == 0.0) {
            scores[i] = 0.0;
        } else {
            scores[i] = dot / (query_mag * candidate_mag);
        }
    }
}
```

### FFI Integration

```zig
// ffi.zig
const vector_similarity = @import("vector_similarity.zig");

export fn engram_vector_similarity(
    query: [*]const f32,
    candidates: [*]const f32,
    scores: [*]f32,
    query_len: usize,
    num_candidates: usize,
) void {
    const query_slice = query[0..query_len];
    const candidates_slice = candidates[0..(query_len * num_candidates)];
    const scores_slice = scores[0..num_candidates];

    vector_similarity.batchCosineSimilarity(
        query_slice,
        candidates_slice,
        scores_slice,
        num_candidates,
    );
}
```

### Rust Wrapper

```rust
// src/zig_kernels/mod.rs

#[cfg(feature = "zig-kernels")]
pub fn batch_cosine_similarity(query: &[f32], candidates: &[Vec<f32>]) -> Vec<f32> {
    let dim = query.len();
    let num_candidates = candidates.len();

    // Validate dimensions
    for candidate in candidates {
        assert_eq!(candidate.len(), dim, "All candidates must have same dimension as query");
    }

    // Flatten candidates
    let candidates_flat: Vec<f32> = candidates
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();

    let mut scores = vec![0.0_f32; num_candidates];

    unsafe {
        ffi::engram_vector_similarity(
            query.as_ptr(),
            candidates_flat.as_ptr(),
            scores.as_mut_ptr(),
            dim,
            num_candidates,
        );
    }

    scores
}
```

### Performance Benchmarks

```rust
// benches/vector_similarity_comparison.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_vector_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_similarity");

    for dim in [384, 768, 1536] {
        for num_candidates in [10, 100, 1000] {
            let query: Vec<f32> = (0..dim).map(|i| i as f32).collect();
            let candidates: Vec<Vec<f32>> = (0..num_candidates)
                .map(|_| (0..dim).map(|i| i as f32 * 0.99).collect())
                .collect();

            // Rust baseline
            group.bench_with_input(
                BenchmarkId::new("rust", format!("{}d_{}c", dim, num_candidates)),
                &(&query, &candidates),
                |b, (query, candidates)| {
                    b.iter(|| {
                        let scores = rust_batch_cosine_similarity(query, candidates);
                        black_box(scores);
                    });
                },
            );

            // Zig kernel
            #[cfg(feature = "zig-kernels")]
            group.bench_with_input(
                BenchmarkId::new("zig", format!("{}d_{}c", dim, num_candidates)),
                &(&query, &candidates),
                |b, (query, candidates)| {
                    b.iter(|| {
                        let scores = zig_batch_cosine_similarity(query, candidates);
                        black_box(scores);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_vector_similarity);
criterion_main!(benches);
```

## Testing Approach

1. **Unit tests**
   - SIMD correctness vs. scalar implementation
   - Edge cases: zero vectors, orthogonal, parallel
   - Numerical stability with denormals

2. **Differential tests**
   - Property-based testing with arbitrary inputs
   - Verify Zig matches Rust within epsilon
   - Test various dimensions (not just 768)

3. **Performance validation**
   - Benchmark Zig vs. Rust baseline
   - Verify 15-25% improvement
   - Test across different CPU architectures

## Integration Points

- **Task 003 (Differential Testing)** - Extend test suite for SIMD edge cases
- **Task 009 (Integration Testing)** - Validate in full graph context
- **Task 010 (Performance Regression)** - Add to regression benchmark suite

## Notes

- Use @Vector for portable SIMD (Zig auto-lowers to platform intrinsics)
- Test on both x86_64 (AVX2) and ARM64 (NEON) architectures
- Consider FMA (fused multiply-add) for further optimization
- Document CPU feature requirements (AVX2, NEON) in deployment guide
- Benchmark with realistic embedding dimensions (OpenAI uses 1536 for ada-002)
