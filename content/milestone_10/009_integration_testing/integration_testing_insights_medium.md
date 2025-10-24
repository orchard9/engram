# What Integration Tests Catch That Unit Tests Miss

Unit tests told us the Zig kernels were correct. Every differential test passed. Vector similarity matched Rust to 1e-6 precision. Spreading activation produced identical results. Decay calculations were spot-on.

Then we ran integration tests and found bugs anyway.

## The Bug Unit Tests Missed

Here's what happened. We had three Zig kernels:
1. Vector similarity (finds similar memories)
2. Spreading activation (traverses associations)
3. Decay (weakens old memories)

Each kernel had unit tests. Each unit test passed. We were confident.

Then we wrote an integration test for the complete memory retrieval flow:

```rust
#[test]
fn test_memory_retrieval_with_zig_kernels() {
    let mut graph = MemoryGraph::new();

    // Add 1000 memories
    for i in 0..1000 {
        let embedding = generate_realistic_embedding();
        graph.add_memory(format!("Memory {}", i), embedding);
    }

    // Simulate passage of time
    std::thread::sleep(Duration::from_secs(2));

    // Query with spreading activation + decay
    let query = generate_query_embedding();
    let results = graph.query(&query, top_k=10);

    // Verify results match Rust-only path
    let rust_results = graph_rust_only.query(&query, top_k=10);
    assert_eq!(results.len(), rust_results.len());

    for (zig_result, rust_result) in results.iter().zip(rust_results.iter()) {
        assert_eq!(zig_result.memory_id, rust_result.memory_id);
        assert_relative_eq!(zig_result.score, rust_result.score, epsilon=1e-3);
    }
}
```

The test failed. Different memories were retrieved. Scores didn't match.

## The Root Cause

The problem was precision loss across FFI boundaries.

Zig decay kernel:
```zig
pub fn apply_decay(strengths: []f32, ages: []const u64) void {
    for (strengths, ages) |*strength, age| {
        const decay_factor = std.math.exp(-age_f32 / half_life_f32);
        strength.* *= decay_factor;  // f32 precision
    }
}
```

Rust spreading activation (original):
```rust
fn spread_activation(&mut self, initial: HashMap<MemoryId, f64>) {
    // f64 precision throughout
}
```

The mismatch:
1. Decay reduces strengths using f32 arithmetic
2. Spreading activation internally uses f64
3. Conversion f32 → f64 → f32 accumulated rounding errors
4. After 3-4 spreading iterations, errors compounded
5. Memories near retrieval threshold crossed in opposite directions

Unit tests didn't catch this because they tested each kernel in isolation with ideal inputs.

Integration tests caught it because they composed kernels in realistic workflows.

## What Makes a Good Integration Test

Integration tests must exercise complete workflows with realistic data:

### 1. Real Data Distributions

Don't use uniform random vectors. Use realistic embeddings:

```rust
fn generate_realistic_embedding(topic: &str) -> Vec<f32> {
    // Simulate semantic clustering
    let base = semantic_vector_for_topic(topic);
    let noise = gaussian_noise(stddev=0.1);
    base.iter().zip(noise).map(|(b, n)| b + n).collect()
}
```

Why? Uniform random vectors are nearly orthogonal (similarity ≈ 0). Real embeddings cluster around topics (similarity 0.3-0.9). Edge cases appear only with realistic distributions.

### 2. Realistic Graph Topology

Don't create fully-connected graphs. Use power-law distributions:

```rust
fn generate_realistic_graph(num_memories: usize) -> MemoryGraph {
    let mut graph = MemoryGraph::new();

    // Add memories with semantic clustering
    for i in 0..num_memories {
        let topic = select_topic_zipfian();  // Some topics common, others rare
        let embedding = generate_realistic_embedding(topic);
        graph.add_memory(embedding);
    }

    // Edges form based on similarity (creates natural clusters)
    graph.build_similarity_edges(threshold=0.7);

    graph
}
```

Real graphs have hubs (highly connected memories) and sparse regions. Bugs hide in these structures.

### 3. End-to-End Workflows

Test complete user-facing operations:

```rust
#[test]
fn test_pattern_completion_workflow() {
    let mut graph = setup_realistic_graph();

    // User stores memory sequence: A → B → C → D
    let memory_a = graph.add_memory("coffee", embedding_a);
    let memory_b = graph.add_memory("morning", embedding_b);
    let memory_c = graph.add_memory("alarm", embedding_c);
    let memory_d = graph.add_memory("wakeup", embedding_d);

    // System learns associations via co-occurrence
    graph.associate(memory_a, memory_b, strength=0.9);
    graph.associate(memory_b, memory_c, strength=0.9);
    graph.associate(memory_c, memory_d, strength=0.8);

    // Simulate time passing (decay affects weak associations)
    std::thread::sleep(Duration::from_secs(1));
    graph.apply_decay();

    // User provides partial cue ("coffee")
    let activated = graph.spread_activation(memory_a, iterations=5);

    // Pattern completion should recall the sequence
    assert!(activated.contains_key(&memory_b), "Should activate 'morning'");
    assert!(activated.contains_key(&memory_c), "Should activate 'alarm'");
    assert!(activated.contains_key(&memory_d), "Should activate 'wakeup'");

    // Verify decay affected activation strength
    assert!(activated[&memory_b] > activated[&memory_d],
        "Earlier associations should be stronger after decay");
}
```

This test exercises:
- Memory creation
- Edge weight learning
- Decay application
- Spreading activation
- Composition of all kernels

## Error Injection: Testing Failure Modes

Integration tests should validate graceful degradation:

```rust
#[test]
fn test_arena_overflow_fallback() {
    // Force arena exhaustion
    configure_arena(pool_size_mb=1, overflow_strategy=OverflowStrategy::ErrorReturn);

    let mut graph = MemoryGraph::new();

    // Add memories that fit
    for i in 0..100 {
        graph.add_memory(generate_small_embedding());
    }

    // Trigger arena overflow with huge query
    let huge_query = vec![1.0; 100_000];  // Too large for 1MB arena

    // Should fall back to Rust kernels gracefully
    let result = std::panic::catch_unwind(|| {
        graph.query(&huge_query, top_k=10)
    });

    assert!(result.is_ok(), "Should handle overflow without crashing");

    // Verify fallback occurred
    let stats = get_arena_stats();
    assert!(stats.overflow_count > 0, "Overflow should be recorded");
}
```

Unit tests rarely test failure modes. Integration tests must.

## Performance Validation

Integration tests measure end-to-end latency:

```rust
#[test]
#[ignore]  // Run manually: cargo test --ignored
fn integration_performance_benchmark() {
    let graph = setup_realistic_graph(num_memories=10_000);
    let queries: Vec<_> = (0..100)
        .map(|_| generate_realistic_embedding("query"))
        .collect();

    let start = Instant::now();
    for query in &queries {
        let _results = graph.query(query, top_k=10);
    }
    let duration = start.elapsed();

    let avg_latency = duration / 100;
    println!("Average query latency: {:?}", avg_latency);

    // Verify p99 latency target
    assert!(avg_latency < Duration::from_micros(2000),
        "Query latency {} exceeds 2ms target", avg_latency.as_micros());
}
```

This catches:
- FFI overhead
- Memory copying costs
- Cache effects in realistic workloads
- Contention under concurrent load

Unit benchmarks test kernels in isolation. Integration benchmarks test kernels in context.

## Lessons From Integration Test Failures

We found four classes of bugs that unit tests missed:

1. Precision mismatches across FFI (f32 vs f64)
2. Memory layout assumptions (Rust packed vs Zig alignment)
3. Error propagation (kernel errors swallowed by intermediate layers)
4. Emergent behavior (rounding errors compounding over iterations)

None of these bugs appeared in unit tests. All appeared in integration tests within minutes.

## The Right Balance

Unit tests are still essential:
- Fast feedback (milliseconds)
- Pinpoint failures
- Enable refactoring with confidence

Integration tests catch what unit tests miss:
- Interface mismatches
- Composition errors
- Realistic workload behavior

Our testing strategy:
- Unit tests: 1000+ tests, run on every commit
- Differential tests: 30,000+ property-based tests
- Integration tests: 50+ tests, run before merge
- Performance tests: 10+ benchmarks, run nightly

Each layer catches different bugs. All layers are necessary.
