# Proving Fast Code Is Correct Code: Differential Testing for Performance Kernels

Your Zig kernel is 30% faster than the Rust baseline. Benchmarks prove it. But does it produce the **same results**?

This is the optimization paradox: making code faster often makes it more complex, and complexity breeds bugs. You need a way to validate that your performance improvements don't break correctness.

Enter differential testing.

## The Problem with Traditional Unit Tests

Traditional unit testing requires you to anticipate every edge case:

```rust
#[test]
fn test_cosine_similarity() {
    // Test case 1: Identical vectors
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);

    // Test case 2: Orthogonal vectors
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);

    // Test case 3: Opposite vectors
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);

    // Test case 4: ... (50 more cases you thought of)
}
```

The problem: You only test cases you explicitly wrote. What about the edge cases you **didn't** think of?

## Differential Testing: Compare Against Known-Good Baseline

Instead of testing the Zig implementation directly, compare it against the Rust baseline:

```rust
proptest! {
    #[test]
    fn zig_matches_rust(
        query in prop_embedding(768),
        candidates in prop::collection::vec(prop_embedding(768), 1..100)
    ) {
        // Run both implementations
        let zig_scores = zig_vector_similarity(&query, &candidates);
        let rust_scores = rust_vector_similarity(&query, &candidates);

        // Verify they match
        for (zig, rust) in zig_scores.iter().zip(&rust_scores) {
            assert_relative_eq!(zig, rust, epsilon = 1e-6);
        }
    }
}
```

Proptest generates thousands of random test cases automatically. If Zig and Rust disagree on **any** of them, the test fails.

For Engram, this approach caught bugs we never would have found with manual unit tests.

## Real Bug Found by Differential Testing

Here's an actual bug proptest found in Engram's Zig kernel:

**Bug:** Zig kernel returned `NaN` when the query vector was all zeros.

**Rust baseline:** Returned `0.0` (correctly handled division by zero).

**How it was found:** After 2,341 randomly generated test cases, proptest created a zero vector. Zig and Rust disagreed. Test failed.

**The minimal reproducing case proptest gave us:**

```rust
query = [0.0, 0.0, 0.0]
candidates = [[1.0, 0.0, 0.0]]

zig_result = NaN
rust_result = 0.0
```

**The fix:**

```zig
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    const dot = dotProductSimd(a, b);
    const mag_a = magnitudeSimd(a);
    const mag_b = magnitudeSimd(b);

    // Handle zero vectors explicitly
    if (mag_a == 0.0 or mag_b == 0.0) {
        return 0.0;
    }

    return dot / (mag_a * mag_b);
}
```

Without proptest, this bug would have shipped to production. Our manual unit tests never checked zero vectors because we "knew" embeddings would be normalized.

Differential testing **doesn't care what you know.** It tests reality.

## Floating-Point Equivalence: Why epsilon Matters

You can't use exact equality for floating-point comparisons:

```rust
assert_eq!(0.1 + 0.2, 0.3);  // FAILS!
// 0.30000000000000004 != 0.3
```

This isn't a bug - it's how floating-point arithmetic works. Different operation orders produce slightly different results.

For differential testing, we use epsilon-based comparison:

```rust
assert_relative_eq!(zig_result, rust_result, epsilon = 1e-6);
```

This tolerates:
- Rounding differences from operation order
- FMA (fused multiply-add) vs separate multiply and add
- Different SIMD reduction strategies

**Choosing epsilon:**
- Too tight (1e-9): False positives from legitimate rounding differences
- Too loose (1e-3): Masks real bugs
- Sweet spot (1e-6): Catches bugs while tolerating expected variance

For single-precision floats (f32), epsilon = 1e-6 is the Goldilocks zone - about 1 part per million tolerance.

## Property-Based Testing Strategies

Proptest uses "strategies" to generate test inputs:

```rust
// Strategy: Generate embeddings (768-dimensional vectors)
fn prop_embedding(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        -1000.0_f32..1000.0,  // Reasonable range for embeddings
        dim..=dim              // Exact dimension
    )
}

// Strategy: Generate random graphs
fn prop_graph(nodes: impl Into<SizeRange>) -> impl Strategy<Value = TestGraph> {
    nodes.into().prop_flat_map(|n| {
        prop::collection::vec(
            prop::collection::vec((0..n, -1.0_f32..1.0), 0..n),
            n..=n
        ).prop_map(TestGraph::from_adjacency)
    })
}
```

These strategies ensure inputs are realistic (graphs have edges, embeddings have reasonable magnitudes) while still exploring edge cases.

## Shrinking: Minimal Reproducing Cases

When proptest finds a failing case, it **shrinks** the input to the simplest form that still fails:

**Original failing input:**
```rust
query = [0.0032, -0.0041, 0.0, 0.0017, ... (768 dimensions)]
candidates = [[0.981, -0.763, 0.0, ...], [1.234, 0.876, 0.0, ...], ...]
```

**Shrunk to minimal case:**
```rust
query = [0.0, 0.0, 0.0]
candidates = [[1.0, 0.0, 0.0]]
```

This makes debugging trivial. Instead of a massive failing input, you get the simplest case that triggers the bug.

## Building a Regression Corpus

When differential testing finds a bug, save the failing input:

```rust
// tests/zig_differential/corpus/vector_nan_case.json
{
    "query": [0.0, 0.0, 0.0],
    "candidates": [[1.0, 0.0, 0.0]]
}
```

Now you have a deterministic regression test:

```rust
#[test]
fn regression_vector_nan() {
    let input: VectorInput = load_test_case("vector_nan_case");
    let scores = vector_similarity(&input.query, &input.candidates);

    // Bug we fixed: used to return NaN
    assert!(!scores[0].is_nan());
    assert_eq!(scores[0], 0.0);
}
```

Over time, your corpus grows to include every edge case that differential testing discovered. Future changes are validated against this corpus.

## Numerical Stability Tests

Some bugs only appear with specific numerical patterns:

**Test: Large and small values (catastrophic cancellation)**
```rust
let query = vec![1e20, 1.0, -1e20];
let candidate = vec![1e20, -1.0, -1e20];
// Does SIMD preserve precision when mixing magnitudes?
```

**Test: Denormal numbers**
```rust
let query = vec![1e-40; 768];
let candidate = vec![1e-40; 768];
// Does implementation handle extremely small values?
```

**Test: Mixed signs in SIMD lanes**
```rust
let query: Vec<_> = (0..768)
    .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
    .collect();
// Does SIMD correctly handle sign changes?
```

These tests catch subtle SIMD bugs that unit tests miss.

## Results for Engram

Running differential tests on our three Zig kernels:

**Vector similarity:** 10,000 random test cases, 100% match (epsilon = 1e-6)
**Activation spreading:** 5,000 random graphs, 100% match (epsilon = 1e-5)
**Memory decay:** 10,000 random age distributions, 100% match (epsilon = 1e-6)

**Bugs found before production:**
- Zero vector handling (NaN → 0.0)
- SIMD remainder loop off-by-one error
- Denormal number handling in magnitude calculation

**Confidence level:** High. If Zig and Rust match on 25,000+ random inputs spanning edge cases, we trust the optimization is correct.

## Key Takeaways

1. **Differential testing validates optimizations:** Compare fast code against correct baseline
2. **Property-based testing finds unexpected edge cases:** Generates thousands of inputs automatically
3. **Use epsilon for floating-point comparison:** Exact equality fails due to legitimate rounding
4. **Shrinking produces minimal reproducing cases:** Makes debugging trivial
5. **Build a regression corpus:** Save discovered edge cases as deterministic tests

## Try It Yourself

Add proptest to your project:

```toml
[dev-dependencies]
proptest = "1.4"
approx = "0.5"
```

Write a differential test:

```rust
use proptest::prelude::*;
use approx::assert_relative_eq;

proptest! {
    #[test]
    fn optimized_matches_baseline(input in 0..1000_i32) {
        let optimized = your_fast_function(input);
        let baseline = your_correct_function(input);
        assert_eq!(optimized, baseline);
    }
}
```

Run it:

```bash
cargo test
```

Proptest will generate 256 random test cases by default (configurable). If any fail, you get the minimal reproducing input.

Optimization without correctness is just a fast way to get wrong answers. Differential testing gives you both.
