# Integration Testing - Twitter Thread

**Tweet 1/9:**

All unit tests passed. All differential tests passed. We shipped the Zig kernels.

Then integration tests found bugs anyway.

Here's what unit tests miss and why integration tests matter.

**Tweet 2/9:**

The bug: Precision mismatch across FFI boundaries.

Zig decay: f32 arithmetic
Rust spreading: f64 arithmetic

Unit tests verified each kernel individually. Integration tests verified them composed.

Composition revealed accumulated rounding errors.

**Tweet 3/9:**

Unit test:
"Does this function work correctly?"

Integration test:
"Do these functions work correctly together?"

Bugs live in the gaps between components. That's where integration tests look.

**Tweet 4/9:**

Good integration tests use realistic data:

Bad: uniform random vectors (all orthogonal, similarity ≈ 0)
Good: semantic clusters (realistic 0.3-0.9 similarity)

Edge cases appear with realistic distributions, not synthetic data.

**Tweet 5/9:**

Good integration tests exercise complete workflows:

1. Add memories → similarity search
2. Build associations → spreading activation
3. Apply decay → weakens old memories
4. Query → ranked retrieval

Test the pipeline, not the parts.

**Tweet 6/9:**

Error injection matters:

```rust
// Force arena overflow
configure_arena(1MB);
let huge_query = vec![1.0; 100_000];

// Should fall back gracefully
let result = graph.query(&huge_query);
assert!(result.is_ok());
```

Unit tests rarely test failures. Integration tests must.

**Tweet 7/9:**

Performance validation:

Unit benchmark: "Kernel executes in 1.7μs"
Integration benchmark: "End-to-end query takes 2ms"

FFI overhead, memory copying, cache effects only appear in full workflow.

**Tweet 8/9:**

Four bug classes unit tests missed:

1. Precision mismatches (f32 vs f64)
2. Memory layout assumptions (packing, alignment)
3. Error propagation (swallowed errors)
4. Emergent behavior (compounding rounding)

All caught by integration tests within minutes.

**Tweet 9/9:**

Testing strategy:

Unit: 1000+ tests, fast feedback
Differential: 30,000+ property tests
Integration: 50+ workflow tests
Performance: 10+ end-to-end benchmarks

Each layer catches different bugs. All layers necessary.
