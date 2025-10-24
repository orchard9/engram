# Task 003: Differential Testing Harness

**Duration:** 2 days
**Status:** Pending
**Dependencies:** 002 (Zig Build System)

## Objectives

Establish comprehensive differential testing framework that guarantees Zig kernels produce identical results to Rust baseline implementations. The harness must use property-based testing to explore edge cases and validate correctness across arbitrary inputs.

1. **Property-based testing** - Use proptest to generate arbitrary test inputs
2. **Numerical equivalence** - Define acceptable floating-point error bounds
3. **Fuzzing infrastructure** - Test kernels with pathological inputs
4. **Regression test corpus** - Save interesting inputs for deterministic tests

## Dependencies

- Task 002 (Zig Build System) - FFI bindings available for testing

## Deliverables

### Files to Create

1. `/tests/zig_differential/vector_similarity.rs` - Differential tests for cosine similarity
   - Property: Zig and Rust produce identical scores (within epsilon)
   - Edge cases: zero vectors, orthogonal vectors, parallel vectors
   - Fuzzing: random embeddings with various dimensions

2. `/tests/zig_differential/spreading_activation.rs` - Differential tests for spreading
   - Property: Activation maps match between implementations
   - Edge cases: isolated nodes, fully connected graphs, cyclic graphs
   - Fuzzing: random graph topologies

3. `/tests/zig_differential/decay_functions.rs` - Differential tests for decay
   - Property: Strength updates match (within epsilon)
   - Edge cases: brand new memories, ancient memories, zero age
   - Fuzzing: random age distributions

4. `/tests/zig_differential/mod.rs` - Test harness utilities
   - Floating-point comparison with configurable epsilon
   - Test graph generation helpers
   - Input corpus serialization

5. `/tests/zig_differential/corpus/` - Saved test cases
   - Regression test inputs discovered by fuzzing
   - Pathological cases that revealed bugs

### Files to Modify

1. `/Cargo.toml` - Add testing dependencies
   ```toml
   [dev-dependencies]
   proptest = "1.4"
   approx = "0.5"
   ```

## Acceptance Criteria

1. All differential tests pass with zig-kernels feature enabled
2. Property-based tests execute 10,000 random test cases per kernel
3. Floating-point equivalence holds within epsilon = 1e-6
4. Regression corpus contains at least 20 saved test cases
5. cargo test --features zig-kernels runs all differential tests

## Implementation Guidance

### Property-Based Testing Framework

Use proptest to generate arbitrary inputs and verify equivalence:

```rust
use proptest::prelude::*;
use approx::assert_relative_eq;

proptest! {
    #[test]
    fn vector_similarity_matches_rust(
        query in prop_embedding(768),
        candidates in prop::collection::vec(prop_embedding(768), 1..100)
    ) {
        let query_dim = query.len();
        let num_candidates = candidates.len();

        // Flatten candidates for Zig FFI
        let candidates_flat: Vec<f32> = candidates.iter()
            .flat_map(|v| v.iter().copied())
            .collect();

        // Call Zig kernel
        let zig_scores = crate::zig_kernels::vector_similarity(
            &query,
            &candidates_flat,
            num_candidates,
        );

        // Call Rust baseline
        let rust_scores: Vec<f32> = candidates.iter()
            .map(|candidate| cosine_similarity(&query, candidate))
            .collect();

        // Verify equivalence
        assert_eq!(zig_scores.len(), rust_scores.len());
        for (zig, rust) in zig_scores.iter().zip(rust_scores.iter()) {
            assert_relative_eq!(zig, rust, epsilon = 1e-6);
        }
    }
}

// Proptest strategy for generating embeddings
fn prop_embedding(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop::num::f32::NORMAL, // Normal distribution around 0
        dim..=dim
    )
}
```

### Edge Case Testing

Explicitly test pathological inputs that reveal bugs:

```rust
#[test]
#[cfg(feature = "zig-kernels")]
fn test_vector_similarity_edge_cases() {
    // Zero vector
    {
        let query = vec![0.0; 768];
        let candidates = vec![1.0; 768];
        let scores = vector_similarity(&query, &candidates, 1);
        assert!(scores[0].is_nan() || scores[0] == 0.0);
    }

    // Orthogonal vectors
    {
        let mut query = vec![0.0; 768];
        query[0] = 1.0;

        let mut candidate = vec![0.0; 768];
        candidate[1] = 1.0;

        let scores = vector_similarity(&query, &candidate, 1);
        assert_relative_eq!(scores[0], 0.0, epsilon = 1e-6);
    }

    // Identical vectors
    {
        let query: Vec<f32> = (0..768).map(|i| i as f32).collect();
        let scores = vector_similarity(&query, &query, 1);
        assert_relative_eq!(scores[0], 1.0, epsilon = 1e-6);
    }

    // Opposite vectors
    {
        let query: Vec<f32> = (0..768).map(|i| i as f32).collect();
        let candidate: Vec<f32> = query.iter().map(|x| -x).collect();
        let scores = vector_similarity(&query, &candidate, 1);
        assert_relative_eq!(scores[0], -1.0, epsilon = 1e-6);
    }
}
```

### Activation Spreading Differential Tests

Test graph algorithms with various topologies:

```rust
proptest! {
    #[test]
    fn spreading_activation_matches_rust(
        graph in prop_graph(10..100, 0.1)
    ) {
        let source_id = graph.random_node();
        let iterations = 100;

        // Call Zig kernel
        let zig_activations = crate::zig_kernels::spread_activation(
            &graph,
            source_id,
            iterations,
        );

        // Call Rust baseline
        let rust_activations = graph.spread_activation_rust(
            source_id,
            iterations,
        );

        // Verify equivalence
        assert_eq!(zig_activations.len(), rust_activations.len());
        for (node_id, zig_act) in &zig_activations {
            let rust_act = rust_activations.get(node_id).unwrap();
            assert_relative_eq!(zig_act, rust_act, epsilon = 1e-5);
        }
    }
}

// Strategy for generating random graphs
fn prop_graph(
    nodes: impl Into<SizeRange>,
    edge_prob: f64
) -> impl Strategy<Value = TestGraph> {
    nodes.into().prop_flat_map(move |n| {
        // Generate adjacency list
        prop::collection::vec(
            prop::collection::vec(
                (0..n, prop::num::f32::NORMAL), // (target, weight)
                0..n
            ),
            n..=n
        ).prop_map(|adj| TestGraph::from_adjacency(adj))
    })
}
```

### Decay Function Differential Tests

Test memory decay with various age distributions:

```rust
proptest! {
    #[test]
    fn decay_calculation_matches_rust(
        strengths in prop::collection::vec(0.0_f32..1.0_f32, 100..10_000),
        ages in prop::collection::vec(0_u64..1_000_000_u64, 100..10_000)
    ) {
        let mut zig_strengths = strengths.clone();
        let mut rust_strengths = strengths.clone();

        // Call Zig kernel
        crate::zig_kernels::apply_decay(&mut zig_strengths, &ages);

        // Call Rust baseline
        for (strength, age) in rust_strengths.iter_mut().zip(ages.iter()) {
            *strength = ebbinghaus_decay(*strength, *age);
        }

        // Verify equivalence
        for (zig, rust) in zig_strengths.iter().zip(rust_strengths.iter()) {
            assert_relative_eq!(zig, rust, epsilon = 1e-6);
        }
    }
}

#[test]
#[cfg(feature = "zig-kernels")]
fn test_decay_edge_cases() {
    // Zero age (no decay)
    {
        let mut strengths = vec![0.5; 100];
        let ages = vec![0; 100];
        apply_decay(&mut strengths, &ages);
        for s in strengths {
            assert_relative_eq!(s, 0.5, epsilon = 1e-6);
        }
    }

    // Ancient memories (full decay)
    {
        let mut strengths = vec![1.0; 100];
        let ages = vec![1_000_000_000; 100]; // ~31 years
        apply_decay(&mut strengths, &ages);
        for s in strengths {
            assert!(s < 0.01); // Nearly fully decayed
        }
    }

    // Zero strength (stays zero)
    {
        let mut strengths = vec![0.0; 100];
        let ages = vec![100_000; 100];
        apply_decay(&mut strengths, &ages);
        for s in strengths {
            assert_relative_eq!(s, 0.0, epsilon = 1e-6);
        }
    }
}
```

### Regression Test Corpus

Save interesting test cases discovered by fuzzing:

```rust
// Test harness utilities
pub fn save_test_case<T: Serialize>(name: &str, input: &T) {
    let path = format!("tests/zig_differential/corpus/{}.json", name);
    let json = serde_json::to_string_pretty(input).unwrap();
    std::fs::write(path, json).unwrap();
}

pub fn load_test_case<T: DeserializeOwned>(name: &str) -> T {
    let path = format!("tests/zig_differential/corpus/{}.json", name);
    let json = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&json).unwrap()
}

// Use in tests
#[test]
fn regression_test_vector_similarity_nan() {
    let input: VectorSimilarityInput = load_test_case("vector_nan_case");
    // Test that previously buggy input now works correctly
}
```

## Testing Approach

1. **Property-based testing**
   - Run proptest with 10,000 cases per kernel
   - Save any failing cases to regression corpus
   - Verify all tests pass consistently

2. **Edge case validation**
   - Test all documented edge cases
   - Verify NaN/Inf handling matches Rust
   - Test boundary conditions (empty inputs, max sizes)

3. **Fuzzing**
   - Use cargo fuzz for extended testing (optional)
   - Focus on input validation and memory safety
   - Document any crashes or panics discovered

## Integration Points

- **Task 005-007 (Kernels)** - Each kernel implementation must pass differential tests
- **Task 009 (Integration Testing)** - Differential tests run as part of integration suite

## Notes

- Use approx crate for floating-point comparison with configurable epsilon
- Consider using quickcheck as alternative to proptest for simpler properties
- Save minimal reproducing cases to corpus, not full fuzzer outputs
- Document any cases where Zig and Rust legitimately differ (e.g., NaN handling)
