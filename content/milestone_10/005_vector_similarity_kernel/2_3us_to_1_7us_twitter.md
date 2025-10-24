# Twitter Thread: 2.3μs → 1.7μs - Optimizing 768-Dimensional Cosine Similarity

## Tweet 1/9
Profiling showed cosine similarity consuming 22% of Engram's runtime.

Rust baseline: 2.3μs per calculation
Zig SIMD kernel: 1.7μs

26% faster. That 600ns adds up when processing thousands of queries/sec.

Here's how we did it.

## Tweet 2/9
The baseline (scalar):

```rust
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let mag_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (mag_a * mag_b)
}
```

Clean, idiomatic. But processes one element at a time - no SIMD.

## Tweet 3/9
Why SIMD matters:

Modern CPUs process multiple values in parallel:
- AVX2 (x86_64): 8 floats simultaneously
- NEON (ARM64): 4 floats simultaneously

For 768-dim vector:
- AVX2: 96 iterations vs 768 scalar
- NEON: 192 iterations vs 768 scalar

4-8x fewer loop iterations.

## Tweet 4/9
Zig's @Vector type auto-lowers to platform SIMD:

```zig
fn dotProductAvx2(a: []const f32, b: []const f32) f32 {
    var sum: @Vector(8, f32) = @splat(0.0);

    var i: usize = 0;
    while (i < len) : (i += 8) {
        const va: @Vector(8, f32) = a[i..][0..8].*;
        const vb: @Vector(8, f32) = b[i..][0..8].*;
        sum += va * vb;  // 8 operations in parallel
    }

    return @reduce(.Add, sum);
}
```

Compiles to AVX2 on x86, NEON on ARM. Portable SIMD.

## Tweet 5/9
Batch processing optimization:

Most workloads are 1:N (one query vs many candidates), not 1:1.

Compute query magnitude once, amortize across all candidates:

```zig
const query_mag = magnitudeSimd(query);  // Once
for (candidates) |candidate| {
    const candidate_mag = magnitudeSimd(candidate);
    scores[i] = dot / (query_mag * candidate_mag);
}
```

For 1000 candidates, saves 999 redundant calculations.

## Tweet 6/9
Cache optimization: layout matters.

Flatten candidates into contiguous buffer:

```rust
let candidates_flat: Vec<f32> = candidates
    .iter()
    .flat_map(|v| v.iter().copied())
    .collect();
```

Sequential access = cache-friendly = SIMD loads hit prefetched lines.

## Tweet 7/9
Edge case handling:

Zero vectors cause division by zero. Handle explicitly:

```zig
if (mag_a == 0.0 or mag_b == 0.0) {
    return 0.0;  // Match Rust baseline behavior
}
```

Differential testing caught early version that returned NaN. Now matches Rust exactly.

## Tweet 8/9
Results on Apple M2 (ARM64 NEON):

Rust: 2.3ms per 1000 queries
Zig: 1.7ms per 1000 queries

26% faster (600ns saved per calculation)

For query against 10k candidates: saves 6ms - noticeable in UX.

Validated with 10k random test cases, 100% match (epsilon = 1e-6).

## Tweet 9/9
Why Zig, not Rust SIMD?

Rust has SIMD support, but Zig advantages:
- Portable @Vector (auto-lowers to platform intrinsics)
- Explicit control (easier to reason about assembly)
- Comptime optimization
- C ABI by default (simpler FFI)

For this hot path, Zig's SIMD ergonomics won.

Profile first. Optimize proven bottlenecks. Validate with testing.
