# Research: Finding the 2% That Costs 98% of Your Time

## Core Question

How do you identify which 2% of your code consumes 98% of your runtime? More specifically, how do we build profiling infrastructure that tells us exactly where to invest optimization effort in Engram's graph engine?

## Context: The Optimization Trap

Developers love to optimize. We see a for loop and immediately think "I could make this faster with SIMD!" But here's the problem: optimizing the wrong thing wastes time and adds complexity for zero user benefit.

The 80/20 rule (or in performance, the 2/98 rule) is brutal: most of your code doesn't matter for performance. A tiny fraction of functions account for almost all runtime. The hard part is finding that fraction.

## What Makes Good Profiling Infrastructure?

### 1. Flamegraphs for Visual Hotspot Discovery

Flamegraphs show function call stacks as horizontal bars, where width represents cumulative time. They answer "where does my program spend time?" at a glance.

For Engram, we expect to see:
- Vector similarity (cosine distance calculations): 15-25% of runtime
- Activation spreading (BFS graph traversal): 20-30% of runtime
- Memory decay (Ebbinghaus function evaluation): 10-15% of runtime

These three operations become our optimization targets because they dominate the profile.

### 2. Criterion Benchmarks for Regression Detection

Flamegraphs identify hotspots, but you need micro-benchmarks to validate improvements. Criterion provides:
- Statistical confidence intervals (± measurement noise)
- Comparison against baseline (is Zig actually faster?)
- Performance regression detection (did my change make things slower?)

### 3. Reproducible Workloads

Profiling is only useful if it's consistent. You need a synthetic workload that:
- Exercises all hot paths (spreading, similarity, decay)
- Uses realistic data sizes (10k memories, 768-dim embeddings)
- Runs deterministically (same input, same profile every time)

## Research Findings

### Flamegraph Analysis Tools

**cargo-flamegraph** (chosen for Engram):
- Built on perf (Linux) or dtrace (macOS)
- Integrates with cargo: `cargo flamegraph --bench workload`
- Outputs SVG with interactive zooming
- Shows both user code and system calls

Alternative: **samply** (Firefox Profiler format):
- More detailed than flamegraphs
- Shows thread interleaving
- Higher overhead, less suitable for CI

### Profiling Target Selection

Based on Engram's architecture, we expect these hotspots:

**Vector Similarity (15-25% of runtime):**
- Function: `cosine_similarity` in `engram-core/src/embedding.rs`
- Why hot: O(d) dot product per query, called thousands of times
- Optimization target: SIMD vectorization (AVX2/NEON)

**Activation Spreading (20-30% of runtime):**
- Function: `spread_activation` in `engram-core/src/graph/spreading.rs`
- Why hot: BFS traversal with weight accumulation, O(V + E) per query
- Optimization target: Edge batching, cache-friendly access patterns

**Memory Decay (10-15% of runtime):**
- Function: `apply_decay` in `engram-core/src/decay.rs`
- Why hot: Exponential calculation for every memory, O(n) per consolidation
- Optimization target: Vectorized exp() approximation

### Benchmark Stability Requirements

Criterion uses statistical analysis to detect performance changes. For reliable results:
- Run on isolated CPU (disable frequency scaling)
- Sufficient sample size (100+ iterations)
- Control for variance (<5% noise acceptable)

Without stability, you get false positives: "Your optimization made things 3% faster!" when it's really just measurement noise.

### Profiling Harness Design

The workload must be realistic but deterministic:

```rust
fn profiling_workload() {
    // Create graph: 10k nodes, 50k edges (5:1 ratio)
    let graph = MemoryGraph::new();
    for i in 0..10_000 {
        graph.add_memory(format!("m_{}", i), random_embedding(768));
    }
    for _ in 0..50_000 {
        graph.add_edge(random_node(), random_node(), random_weight());
    }

    // Exercise hot paths
    for _ in 0..1000 {
        graph.spread_activation(random_node(), 100);  // 20-30% of time
        graph.find_similar(&random_embedding(768), 10);  // 15-25% of time
    }
    graph.apply_decay(Duration::from_secs(86400));  // 10-15% of time
}
```

This workload ensures profiling captures realistic usage patterns.

## Key Insights

1. **Optimize with data, not intuition:** Flamegraphs prevent wasted effort on cold paths
2. **Statistical rigor matters:** Criterion's confidence intervals distinguish real improvements from noise
3. **Reproducibility is non-negotiable:** Deterministic workloads make profiling actionable
4. **Profile before and after:** Always validate that optimizations actually work

## References

- Brendan Gregg's "Flame Graphs": http://www.brendangregg.com/flamegraphs.html
- Criterion.rs documentation: https://github.com/bheisler/criterion.rs
- "Systems Performance" by Brendan Gregg (Chapter 6: CPUs)
- cargo-flamegraph: https://github.com/flamegraph-rs/flamegraph

## Next Steps

With profiling infrastructure in place, we can:
1. Generate baseline flamegraphs showing Rust hot paths
2. Identify optimization targets (vector similarity, spreading)
3. Implement Zig kernels for those specific functions
4. Validate improvements with Criterion benchmarks
