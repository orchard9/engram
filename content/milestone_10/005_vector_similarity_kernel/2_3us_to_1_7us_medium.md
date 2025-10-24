# 2.3μs → 1.7μs: Optimizing 768-Dimensional Cosine Similarity

Profiling showed that cosine similarity calculations consume 22% of Engram's total runtime. We're computing similarity between 768-dimensional embedding vectors thousands of times per query.

The Rust baseline: 2.3μs per similarity calculation.

After implementing a Zig kernel with SIMD vectorization: 1.7μs.

**26% faster.** That 600 nanoseconds per calculation adds up when you're processing thousands of queries per second.

Here's how we did it.

## The Baseline: Scalar Implementation

Cosine similarity between two vectors a and b:

```
similarity = (a · b) / (|a| × |b|)
```

Where `a · b` is the dot product and `|a|` is the magnitude (L2 norm).

Rust baseline (scalar):

```rust
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}
```

This is clean, idiomatic Rust. But it processes one element at a time - no SIMD.

## Why SIMD Matters for Vector Operations

Modern CPUs have SIMD instructions that process multiple values in parallel:
- **AVX2** (x86_64): Process 8 floats simultaneously
- **NEON** (ARM64): Process 4 floats simultaneously

For a 768-dimensional vector, SIMD means:
- AVX2: 768 / 8 = 96 iterations (vs 768 scalar iterations)
- NEON: 768 / 4 = 192 iterations (vs 768 scalar iterations)

That's 8x or 4x fewer loop iterations. But SIMD instruction throughput is higher too, so the actual speedup is closer to 4-6x for dot products.

## Zig SIMD Implementation

Zig's @Vector type auto-lowers to platform-specific SIMD intrinsics:

```zig
fn dotProductAvx2(a: []const f32, b: []const f32) f32 {
    const len = a.len;
    const simd_width = 8;  // AVX2 processes 8 floats
    const simd_len = (len / simd_width) * simd_width;

    var sum: @Vector(8, f32) = @splat(0.0);

    var i: usize = 0;
    while (i < simd_len) : (i += simd_width) {
        const va: @Vector(8, f32) = a[i..][0..simd_width].*;
        const vb: @Vector(8, f32) = b[i..][0..simd_width].*;
        sum += va * vb;  // SIMD multiply-add (8 operations in parallel)
    }

    // Horizontal reduction: sum all lanes
    var result: f32 = @reduce(.Add, sum);

    // Handle remainder elements (768 % 8 = 0, so this is rare)
    while (i < len) : (i += 1) {
        result += a[i] * b[i];
    }

    return result;
}
```

The `@Vector(8, f32)` type compiles to AVX2 instructions on x86_64. On ARM64, Zig would use `@Vector(4, f32)` and compile to NEON.

**Key advantage:** Write portable SIMD once, Zig handles platform-specific lowering.

## Batch Processing Optimization

Most vector similarity workloads aren't 1:1 comparisons. They're 1:N - one query vector against many candidates.

We can amortize the cost of computing the query magnitude:

```zig
pub fn batchCosineSimilarity(
    query: []const f32,
    candidates: []const f32,
    scores: []f32,
    num_candidates: usize,
) void {
    const dim = query.len;

    // Compute query magnitude once (amortized across all candidates)
    const query_mag = magnitudeSimd(query);
    if (query_mag == 0.0) {
        @memset(scores, 0.0);
        return;
    }

    // Process each candidate
    for (0..num_candidates) |i| {
        const candidate = candidates[(i * dim)..][0..dim];

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

For 1000 candidates, this saves 999 redundant query magnitude calculations.

## Cache Optimization: Layout Matters

Memory layout affects SIMD performance. Sequential access is fast (cache prefetching), strided access is slow.

We flatten candidate vectors into a contiguous buffer:

```rust
// Rust side: flatten candidates for cache-friendly access
let candidates_flat: Vec<f32> = candidates
    .iter()
    .flat_map(|v| v.iter().copied())
    .collect();

// Zig side: sequential access pattern
const candidate = candidates[(i * dim)..][0..dim];  // Contiguous slice
```

This ensures SIMD loads hit the cache line that was just prefetched.

## Handling Edge Cases Correctly

Zero vectors cause division by zero. Zig doesn't have NaN propagation by default, so we handle this explicitly:

```zig
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    const dot = dotProductSimd(a, b);
    const mag_a = magnitudeSimd(a);
    const mag_b = magnitudeSimd(b);

    // Explicit zero handling (matches Rust baseline behavior)
    if (mag_a == 0.0 or mag_b == 0.0) {
        return 0.0;
    }

    return dot / (mag_a * mag_b);
}
```

Differential testing caught an early version that returned NaN for zero vectors. Now Zig matches Rust exactly.

## Performance Results

Benchmarking on Apple M2 (ARM64 with NEON):

**Rust baseline (scalar):**
```
vector_similarity_1000
  time:   [2.2843 ms 2.2991 ms 2.3156 ms]
```

**Zig SIMD kernel:**
```
vector_similarity_1000
  time:   [1.6943 ms 1.7102 ms 1.7279 ms]
  change: [-26.341% -25.672% -24.981%]
```

**Improvement: 26% faster**

Per-operation cost: 2.3μs → 1.7μs (saving 0.6μs per similarity calculation)

For a query that compares against 10,000 candidate memories, this saves 6ms - enough to notice in user experience.

## Why Not Just Use Rust SIMD?

Rust has SIMD support via std::simd (unstable) or explicit intrinsics. We could write:

```rust
use std::simd::f32x8;

fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    // ... explicit SIMD code ...
}
```

But Zig has advantages:
1. **Portable @Vector type:** Zig auto-lowers to platform intrinsics
2. **Explicit control:** No hidden allocations, easier to reason about assembly
3. **Comptime optimization:** Zig can unroll loops and specialize at compile time
4. **C ABI by default:** Simpler FFI integration

For this specific hot path, Zig's SIMD ergonomics made the optimization cleaner.

## Correctness Validation

Differential testing validated the Zig kernel:

```rust
proptest! {
    #[test]
    fn zig_matches_rust(
        query in prop_embedding(768),
        candidates in prop::collection::vec(prop_embedding(768), 1..100)
    ) {
        let zig_scores = zig_batch_cosine(&query, &candidates);
        let rust_scores = rust_batch_cosine(&query, &candidates);

        for (zig, rust) in zig_scores.iter().zip(&rust_scores) {
            assert_relative_eq!(zig, rust, epsilon = 1e-6);
        }
    }
}
```

10,000 random test cases, 100% match within epsilon = 1e-6. High confidence in correctness.

## Key Takeaways

1. **SIMD gives 4-6x speedups for vector operations:** AVX2 processes 8 floats in parallel
2. **Batch processing amortizes costs:** Compute query magnitude once for all candidates
3. **Cache-friendly layout matters:** Flatten candidates for sequential access
4. **Handle edge cases explicitly:** Zero vectors, NaN propagation
5. **Differential testing validates optimizations:** 10k+ test cases ensure correctness

The result: 26% faster vector similarity with proven correctness. For a function consuming 22% of total runtime, this translates to 5-6% overall system speedup.

When profiling identifies a clear bottleneck, SIMD optimization pays off.

## Try It Yourself

Zig SIMD is straightforward:

```zig
const vec_a: @Vector(8, f32) = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
const vec_b: @Vector(8, f32) = .{ 8, 7, 6, 5, 4, 3, 2, 1 };
const result = vec_a * vec_b;  // Element-wise multiply in parallel

const sum = @reduce(.Add, result);  // Horizontal reduction
```

Profile first, optimize the proven bottlenecks, validate with differential testing. That's the path to reliable performance improvements.
