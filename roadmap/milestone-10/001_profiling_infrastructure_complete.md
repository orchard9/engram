# Task 001: Profiling Infrastructure

**Duration:** 2 days
**Status:** Pending
**Dependencies:** None

## Objectives

Establish comprehensive profiling infrastructure to identify compute-intensive hot paths and validate performance improvements from Zig kernels. The profiling system must provide both high-level flamegraphs for hotspot identification and fine-grained micro-benchmarks for regression detection.

1. **Flamegraph profiling** - Integrate cargo-flamegraph for visual hotspot analysis
2. **Criterion benchmarks** - Establish baseline performance for target operations
3. **Profiling harness** - Create reproducible workload for consistent measurements
4. **Hotspot documentation** - Identify and document top 10 compute-intensive functions

## Dependencies

- None (foundational task)

## Deliverables

### Files to Create

1. `/benches/profiling_harness.rs` - Reproducible workload for profiling
   - Memory creation with 10k nodes, 50k edges
   - 1000 spreading activation queries
   - 1000 vector similarity comparisons
   - 1000 decay calculations

2. `/benches/baseline_performance.rs` - Criterion benchmarks for hot paths
   - Vector similarity: query vs. 1000 candidates
   - Spreading activation: 100 iterations with decay
   - Memory decay: 10k memories with varying ages

3. `/scripts/profile_hotspots.sh` - Automated profiling script
   - cargo flamegraph --bench profiling_harness
   - Output to tmp/flamegraph.svg
   - Extract top 10 functions by cumulative time

4. `/docs/internal/profiling_results.md` - Baseline performance documentation
   - Flamegraph analysis with hotspot identification
   - Criterion benchmark results with confidence intervals
   - Prioritized list of optimization candidates

### Files to Modify

1. `/Cargo.toml` - Add profiling dependencies
   ```toml
   [dev-dependencies]
   criterion = { version = "0.5", features = ["html_reports"] }

   [[bench]]
   name = "profiling_harness"
   harness = false

   [[bench]]
   name = "baseline_performance"
   harness = false
   ```

## Acceptance Criteria

1. `./scripts/profile_hotspots.sh` generates flamegraph in tmp/
2. Criterion benchmarks run successfully with `cargo bench`
3. Profiling results document identifies:
   - Vector similarity accounts for 15-25% of compute time
   - Activation spreading accounts for 20-30% of compute time
   - Memory decay accounts for 10-15% of compute time
4. Benchmark variance <5% across 10 consecutive runs
5. All benchmarks complete in <5 minutes total runtime

## Implementation Guidance

### Profiling Harness Design

The profiling harness must create a realistic workload that exercises all target hot paths:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engram::graph::MemoryGraph;
use engram::embedding::Embedding;

fn profiling_workload() {
    // Create graph with realistic structure
    let graph = MemoryGraph::new();

    // Load 10k memories with embeddings
    for i in 0..10_000 {
        let embedding = generate_random_embedding();
        graph.add_memory(format!("memory_{}", i), embedding);
    }

    // Add 50k edges (5:1 edge-to-node ratio)
    for _ in 0..50_000 {
        let source = random_node(&graph);
        let target = random_node(&graph);
        graph.add_edge(source, target, random_weight());
    }

    // Execute 1000 spreading activation queries
    for _ in 0..1000 {
        let source = random_node(&graph);
        let result = graph.spread_activation(source, 100);
        black_box(result);
    }

    // Execute 1000 vector similarity queries
    for _ in 0..1000 {
        let query = generate_random_embedding();
        let results = graph.find_similar(&query, 10);
        black_box(results);
    }

    // Execute decay on all memories
    graph.apply_decay(Duration::from_secs(86400));
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("profiling_workload", |b| {
        b.iter(|| profiling_workload())
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
```

### Flamegraph Analysis

The profiling script must extract actionable insights:

```bash
#!/bin/bash
set -euo pipefail

echo "Running flamegraph profiling..."
cargo flamegraph --bench profiling_harness -- --bench

echo "Extracting hotspots..."
# Parse flamegraph SVG to extract top functions
# (SVG contains embedded performance data)

echo "Flamegraph saved to tmp/flamegraph.svg"
echo "Open in browser to analyze hotspots"
```

### Baseline Benchmarks

Create separate benchmark groups for each optimization target:

```rust
fn vector_similarity_baseline(c: &mut Criterion) {
    let query = generate_embedding(768);
    let candidates: Vec<_> = (0..1000)
        .map(|_| generate_embedding(768))
        .collect();

    c.bench_function("vector_similarity_1000", |b| {
        b.iter(|| {
            for candidate in &candidates {
                let score = cosine_similarity(&query, candidate);
                black_box(score);
            }
        })
    });
}

fn spreading_activation_baseline(c: &mut Criterion) {
    let graph = create_test_graph(1000, 5000);
    let source = graph.random_node();

    c.bench_function("spreading_activation_1000", |b| {
        b.iter(|| {
            let result = graph.spread_activation(source, 100);
            black_box(result);
        })
    });
}

fn decay_calculation_baseline(c: &mut Criterion) {
    let graph = create_test_graph(10_000, 0);
    let delta = Duration::from_secs(86400);

    c.bench_function("decay_10k_memories", |b| {
        b.iter(|| {
            graph.apply_decay(delta);
        })
    });
}
```

## Testing Approach

1. **Profiling harness validation**
   - Verify workload creates expected graph structure
   - Confirm all hot paths are exercised
   - Validate reproducibility across runs

2. **Benchmark stability**
   - Run benchmarks 10 times consecutively
   - Calculate variance and confidence intervals
   - Ensure variance <5% for reliable regression detection

3. **Hotspot identification**
   - Review flamegraph for expected hot functions
   - Validate cumulative time percentages
   - Cross-reference with Criterion micro-benchmarks

## Integration Points

- **Task 002 (Zig Build System)** - Benchmarks will include Zig kernel variants
- **Task 010 (Performance Regression)** - Baseline benchmarks become regression tests

## Notes

- Use release profile with debug symbols for accurate profiling: `cargo flamegraph --release`
- Disable CPU frequency scaling during benchmarking for consistency
- Consider using perf stat for hardware counter analysis (cache misses, branch mispredictions)
- Document system configuration (CPU, RAM, OS) in profiling_results.md
