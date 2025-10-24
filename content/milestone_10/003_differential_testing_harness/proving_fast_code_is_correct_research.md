# Research: Proving Fast Code Is Correct Code

## Core Question

You've written a Zig kernel that's 30% faster than your Rust baseline. But does it produce the same results?

Differential testing answers this question: run both implementations with the same inputs, verify they produce identical outputs. If they match across thousands of random test cases, you have high confidence the optimization is correct.

## Context: The Correctness-Performance Tradeoff

Optimization is risky. When you replace a simple, correct implementation with a complex, fast one, you introduce bugs:
- Off-by-one errors in SIMD loops
- Incorrect edge case handling (NaN, zero vectors)
- Numerical instability (accumulation errors)
- Memory safety violations (buffer overruns)

The traditional approach: write unit tests for the new implementation. But this requires anticipating every edge case. What about the cases you didn't think of?

Differential testing flips the problem: instead of testing the new implementation directly, **compare it against a known-good baseline**. If they match on arbitrary inputs, the optimization is probably correct.

## Research Findings

### Property-Based Testing

Traditional unit tests check specific inputs:

```rust
#[test]
fn test_cosine_similarity() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    assert_eq!(cosine_similarity(&a, &b), 1.0);
}
```

This is brittle - you only test cases you thought of.

Property-based testing generates arbitrary inputs:

```rust
proptest! {
    #[test]
    fn cosine_similarity_matches_baseline(
        a in prop::collection::vec(-1000.0_f32..1000.0, 1..1000),
        b in prop::collection::vec(-1000.0_f32..1000.0, 1..1000)
    ) {
        let zig_result = zig_cosine_similarity(&a, &b);
        let rust_result = rust_cosine_similarity(&a, &b);
        assert_relative_eq!(zig_result, rust_result, epsilon = 1e-6);
    }
}
```

Proptest generates thousands of random test cases. If Zig and Rust disagree on any of them, the test fails and shows you the minimal reproducing input.

### Floating-Point Equivalence

Exact equality (`==`) doesn't work for floating-point:

```rust
assert_eq!(0.1 + 0.2, 0.3);  // FAILS! (0.30000000000000004 != 0.3)
```

You need epsilon-based comparison:

```rust
assert_relative_eq!(0.1 + 0.2, 0.3, epsilon = 1e-10);  // PASSES
```

For Engram's kernels, we use epsilon = 1e-6 (single-precision float has ~7 digits of precision). This tolerates:
- Rounding errors from different operation orders
- FMA (fused multiply-add) differences
- Different SIMD reduction strategies

Tighter epsilon (1e-7 or 1e-8) catches real bugs. Looser epsilon (1e-4) hides bugs.

### Edge Case Discovery

Property-based testing finds edge cases you didn't anticipate:

**Example from Engram:**
- **Bug found:** Zig kernel returned NaN when query vector was all zeros
- **Rust baseline:** Returned 0.0 (handled division by zero gracefully)
- **Discovered by:** Proptest generated a zero vector after 2,341 test cases
- **Fix:** Added explicit zero-vector check in Zig

Without proptest, this bug would have shipped to production. Manual unit tests never tested zero vectors because we "knew" embeddings would be normalized.

### Fuzzing vs Property-Based Testing

Two approaches to finding bugs with random inputs:

**Property-based testing (proptest):**
- Generates inputs from strategies (e.g., "vectors of length 1-1000")
- Shrinks failing inputs to minimal reproducing case
- Integrated with cargo test
- Fast (thousands of cases per second)

**Fuzzing (cargo-fuzz):**
- Generates byte sequences, feeds to parser
- Uses coverage-guided mutation (finds new code paths)
- Runs for hours/days
- Slower but more thorough

For Engram: Proptest for differential testing (fast feedback), fuzzing for memory safety (long-running CI job).

### Regression Test Corpus

When proptest finds a bug, save the failing input:

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
    assert!(!scores[0].is_nan());
}
```

This prevents the bug from reoccurring.

### Numerical Stability Testing

Some bugs only appear with specific numerical patterns:

**Test: Catastrophic cancellation**
```rust
let a = vec![1e20, 1.0, -1e20];  // Large and small values
let b = vec![1e20, -1.0, -1e20];
// Does SIMD implementation preserve precision?
```

**Test: Denormal numbers**
```rust
let a = vec![1e-40; 768];  // Extremely small values
let b = vec![1e-40; 768];
// Does implementation handle denormals correctly?
```

**Test: Mixed signs**
```rust
let a = (0..768).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
// Does SIMD handle sign changes correctly?
```

These tests catch subtle SIMD bugs that unit tests miss.

## Key Insights

1. **Differential testing validates optimizations:** Compare fast implementation against correct baseline
2. **Property-based testing finds unexpected edge cases:** Generates thousands of random inputs automatically
3. **Floating-point equivalence needs epsilon:** Exact equality fails due to rounding differences
4. **Save failing cases as regression tests:** Build corpus of edge cases discovered by fuzzing
5. **Test numerical stability explicitly:** Large/small values, denormals, mixed signs reveal SIMD bugs

## References

- "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs" (Claessen & Hughes, 2000)
- Proptest documentation: https://github.com/proptest-rs/proptest
- "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (Goldberg, 1991)
- Approx crate documentation: https://docs.rs/approx/
- "Effective Testing with RSpec 3" (Chapter on property-based testing)

## Next Steps

With differential testing in place:
1. Generate arbitrary test inputs (vectors, graphs, ages)
2. Run both Rust and Zig implementations
3. Verify results match within epsilon
4. Save any failing cases to regression corpus
5. Validate all kernels before declaring them production-ready
