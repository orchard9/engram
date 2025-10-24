# Twitter Thread: Proving Fast Code Is Correct Code

## Tweet 1/10
Your Zig kernel is 30% faster than the Rust baseline. Benchmarks prove it.

But does it produce the SAME RESULTS?

Optimization adds complexity. Complexity breeds bugs.

How do you validate that performance improvements don't break correctness?

Differential testing.

## Tweet 2/10
Traditional unit tests are brittle:

```rust
#[test]
fn test_cosine() {
    assert_eq!(cosine(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
    assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    // ... 50 more cases you thought of
}
```

Problem: You only test cases you explicitly wrote.

What about edge cases you DIDN'T think of?

## Tweet 3/10
Differential testing: Compare optimized code against known-good baseline.

```rust
proptest! {
    #[test]
    fn zig_matches_rust(
        query in prop_vector(768),
        candidates in prop::collection::vec(prop_vector(768), 1..100)
    ) {
        assert_relative_eq!(
            zig_cosine(&query, &candidates),
            rust_cosine(&query, &candidates),
            epsilon = 1e-6
        );
    }
}
```

Generates thousands of random cases automatically.

## Tweet 4/10
Real bug proptest found in Engram:

Zig kernel returned NaN when query = [0.0, 0.0, 0.0]
Rust baseline returned 0.0 (correct)

Found after 2,341 randomly generated test cases.

Our manual unit tests never checked zero vectors because we "knew" embeddings were normalized.

Proptest doesn't care what you know. It tests reality.

## Tweet 5/10
Why not exact equality?

```rust
assert_eq!(0.1 + 0.2, 0.3);  // FAILS!
// 0.30000000000000004 != 0.3
```

Floating-point arithmetic has rounding. Different operation orders produce slightly different results.

Use epsilon-based comparison:
```rust
assert_relative_eq!(result, expected, epsilon = 1e-6);
```

Tolerates legitimate rounding, catches real bugs.

## Tweet 6/10
Choosing epsilon:

- Too tight (1e-9): False positives from legitimate rounding
- Too loose (1e-3): Masks real bugs
- Sweet spot (1e-6): ~1 part per million tolerance

For f32, epsilon = 1e-6 is the Goldilocks zone.

## Tweet 7/10
Property-based testing shrinks failing inputs to minimal reproducing case:

Original failure:
```rust
query = [0.0032, -0.0041, 0.0, ... (768 values)]
candidates = [[0.981, -0.763, ...], ...]
```

Shrunk to:
```rust
query = [0.0, 0.0, 0.0]
candidates = [[1.0, 0.0, 0.0]]
```

Makes debugging trivial.

## Tweet 8/10
Build a regression corpus:

When proptest finds a bug, save the failing input to JSON.

Now you have a deterministic test that prevents regression:

```rust
#[test]
fn regression_vector_nan() {
    let input = load_test_case("vector_nan_case");
    let result = vector_similarity(&input);
    assert!(!result.is_nan());
}
```

Your corpus grows to cover every discovered edge case.

## Tweet 9/10
Numerical stability tests catch SIMD bugs:

Test large/small values:
```rust
vec![1e20, 1.0, -1e20]  // Catastrophic cancellation?
```

Test denormals:
```rust
vec![1e-40; 768]  // Underflow handling?
```

Test mixed signs:
```rust
(0..768).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
```

These reveal bugs unit tests miss.

## Tweet 10/10
Results for Engram's Zig kernels:

- Vector similarity: 10k cases, 100% match
- Activation spreading: 5k graphs, 100% match
- Memory decay: 10k distributions, 100% match

Bugs found BEFORE production:
- Zero vector NaN handling
- SIMD remainder loop off-by-one
- Denormal magnitude calculation

Differential testing: Fast code that's actually correct.
