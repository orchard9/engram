# Task 009: Integration Testing

**Duration:** 2 days
**Status:** Pending
**Dependencies:** 008 (Arena Allocator)

## Objectives

Validate Zig kernels in complete end-to-end workflows, ensuring they integrate correctly with the full Engram system. Integration tests verify that kernels compose correctly and produce expected results in realistic usage scenarios beyond unit and differential testing.

1. **End-to-end scenarios** - Full memory consolidation pipelines with Zig kernels
2. **API compatibility** - Verify drop-in replacement for Rust implementations
3. **Error handling** - Validate graceful degradation under various failure modes
4. **Performance profiling** - Measure real-world performance improvements

## Dependencies

- Task 008 (Arena Allocator) - All kernels and infrastructure complete

## Deliverables

### Files to Create

1. `/tests/integration/zig_kernels.rs` - Integration test suite
   - Full memory consolidation workflow
   - Pattern completion with spreading activation
   - Mixed Rust/Zig execution paths
   - Error injection and recovery

2. `/tests/integration/scenarios/` - Realistic test scenarios
   - scenario_memory_recall.rs - Query with spreading and similarity
   - scenario_consolidation.rs - Long-term memory formation
   - scenario_pattern_completion.rs - Associative recall

3. `/scripts/integration_profile.sh` - Profiling script
   - Run integration tests with profiling enabled
   - Compare Zig-enabled vs. Rust-only performance
   - Generate performance report

### Files to Modify

1. `/src/graph/memory_graph.rs` - Add Zig kernel integration points
   - Feature-gated Zig kernel paths
   - Fallback to Rust on error
   - Kernel selection logging

2. `/tests/common/mod.rs` - Shared test utilities
   - Graph generation helpers
   - Performance comparison utilities
   - Assertion helpers for fuzzy float comparison

## Acceptance Criteria

1. All integration tests pass with zig-kernels feature enabled
2. Tests verify identical behavior between Rust and Zig paths
3. Error injection tests demonstrate graceful fallback
4. Performance profiling shows expected improvements (15-35%)
5. Integration tests complete in <30 seconds

## Implementation Guidance

### Full Memory Consolidation Workflow

```rust
// tests/integration/zig_kernels.rs

#[cfg(feature = "zig-kernels")]
#[test]
fn test_memory_consolidation_with_zig_kernels() {
    // Create memory graph
    let mut graph = MemoryGraph::new();

    // Add episodic memories
    let memory_ids: Vec<_> = (0..1000)
        .map(|i| {
            let embedding = generate_embedding(768);
            let content = format!("Memory {}", i);
            graph.add_memory(content, embedding)
        })
        .collect();

    // Add associations (uses Zig vector similarity internally)
    for i in 0..1000 {
        let similar = graph.find_similar(memory_ids[i], 10);
        for (similar_id, similarity) in similar {
            if similarity > 0.7 {
                graph.add_edge(memory_ids[i], similar_id, similarity);
            }
        }
    }

    // Simulate retrieval with spreading activation
    let query_embedding = generate_embedding(768);
    let initial_matches = graph.find_similar_by_embedding(&query_embedding, 5);

    let mut activations = HashMap::new();
    for (memory_id, similarity) in initial_matches {
        activations.insert(memory_id, similarity);
    }

    // Spread activation (uses Zig spreading kernel)
    let spread_results = graph.spread_activation_multi(activations, 100);

    // Verify results
    assert!(spread_results.len() > 5, "Spreading should activate additional memories");
    assert!(spread_results.len() < 1000, "Spreading should be constrained by threshold");

    // Apply decay (uses Zig decay kernel)
    std::thread::sleep(Duration::from_secs(1));
    graph.apply_decay(Duration::from_secs(1));

    // Verify decay was applied
    for memory_id in &memory_ids[0..10] {
        let memory = graph.get_memory(*memory_id).unwrap();
        assert!(memory.strength < 1.0, "Decay should reduce strength");
    }
}
```

### API Compatibility Testing

```rust
#[cfg(feature = "zig-kernels")]
#[test]
fn test_zig_rust_equivalence() {
    let query = generate_embedding(768);
    let candidates: Vec<_> = (0..100).map(|_| generate_embedding(768)).collect();

    // Call Rust implementation
    let rust_scores = rust_batch_cosine_similarity(&query, &candidates);

    // Call Zig implementation
    let zig_scores = zig_batch_cosine_similarity(&query, &candidates);

    // Verify equivalence
    assert_eq!(rust_scores.len(), zig_scores.len());
    for (rust_score, zig_score) in rust_scores.iter().zip(zig_scores.iter()) {
        assert_relative_eq!(rust_score, zig_score, epsilon = 1e-6);
    }
}
```

### Pattern Completion Scenario

```rust
// tests/integration/scenarios/scenario_pattern_completion.rs

#[cfg(feature = "zig-kernels")]
#[test]
fn test_pattern_completion_with_partial_cue() {
    let mut graph = MemoryGraph::new();

    // Create pattern: A -> B -> C -> D
    let memory_a = graph.add_memory("A", embedding_from_text("alpha"));
    let memory_b = graph.add_memory("B", embedding_from_text("beta"));
    let memory_c = graph.add_memory("C", embedding_from_text("gamma"));
    let memory_d = graph.add_memory("D", embedding_from_text("delta"));

    graph.add_edge(memory_a, memory_b, 0.9);
    graph.add_edge(memory_b, memory_c, 0.9);
    graph.add_edge(memory_c, memory_d, 0.9);

    // Provide partial cue (A) and expect completion to D
    let activations = graph.spread_activation(memory_a, 100);

    // Verify pattern completion
    assert!(activations.contains_key(&memory_b), "Should activate B");
    assert!(activations.contains_key(&memory_c), "Should activate C");
    assert!(activations.contains_key(&memory_d), "Should activate D");

    // Verify activation decay
    assert!(activations[&memory_b] > activations[&memory_c]);
    assert!(activations[&memory_c] > activations[&memory_d]);
}
```

### Error Injection and Recovery

```rust
#[cfg(feature = "zig-kernels")]
#[test]
fn test_graceful_fallback_on_error() {
    // Configure arena to overflow easily
    configure_arena(1, OverflowStrategy::ErrorReturn);

    let mut graph = MemoryGraph::new();

    // Add memories
    for i in 0..100 {
        let embedding = generate_embedding(768);
        graph.add_memory(format!("Memory {}", i), embedding);
    }

    // Force arena overflow with huge query
    let huge_embedding = generate_embedding(100_000); // Too large for 1MB arena

    // Should fall back to Rust implementation gracefully
    let result = std::panic::catch_unwind(|| {
        graph.find_similar_by_embedding(&huge_embedding, 10)
    });

    // Verify system didn't crash
    assert!(result.is_ok(), "System should handle overflow gracefully");

    // Check metrics
    let stats = get_arena_stats();
    // May have overflows recorded
}
```

### Performance Profiling

```rust
// tests/integration/zig_kernels.rs

#[cfg(feature = "zig-kernels")]
#[test]
#[ignore] // Run manually with --ignored
fn profile_zig_vs_rust_performance() {
    use std::time::Instant;

    let mut graph = MemoryGraph::new();

    // Setup
    for i in 0..10_000 {
        let embedding = generate_embedding(768);
        graph.add_memory(format!("Memory {}", i), embedding);
    }

    let query = generate_embedding(768);

    // Benchmark Zig kernel path
    let start = Instant::now();
    for _ in 0..100 {
        let _results = graph.find_similar_by_embedding(&query, 10);
    }
    let zig_duration = start.elapsed();

    println!("Zig kernel: {:?}", zig_duration);

    // Compare with expected improvement
    // (Baseline from profiling would be ~2.3us per query)
    let per_query = zig_duration.as_micros() / 100;
    assert!(per_query < 2000, "Expected <2us per query with Zig kernels");
}
```

### Integration Test Script

```bash
#!/bin/bash
# scripts/integration_profile.sh
set -euo pipefail

echo "Running integration tests with Zig kernels..."

# Run with profiling
RUSTFLAGS="-C force-frame-pointers=yes" \
    cargo test --features zig-kernels --test integration -- --ignored

# Generate performance report
echo ""
echo "Performance Summary:"
echo "==================="

# Extract timing from test output
cargo test --features zig-kernels --test integration -- --ignored --nocapture 2>&1 \
    | grep "kernel:"

echo ""
echo "Integration tests complete. Check tmp/integration_profile.log for details."
```

## Testing Approach

1. **End-to-end workflows**
   - Memory consolidation pipeline
   - Pattern completion scenarios
   - Mixed kernel execution

2. **API compatibility**
   - Drop-in replacement verification
   - Identical behavior testing
   - Feature flag validation

3. **Error handling**
   - Arena overflow recovery
   - Invalid input handling
   - Graceful degradation

4. **Performance profiling**
   - Real-world improvement validation
   - Identify integration overhead
   - Verify expected speedups

## Integration Points

- **Task 010 (Performance Regression)** - Integration tests feed into regression suite
- **Task 012 (Final Validation)** - Integration tests part of UAT

## Notes

- Integration tests should use realistic data distributions
- Consider adding chaos testing (random error injection)
- Profile memory usage in addition to runtime
- Document expected performance characteristics in test comments
- Use test feature flags to isolate Zig-specific behavior
