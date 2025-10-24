# Twitter Thread: Finding the 2% That Costs 98% of Your Time

## Tweet 1/9
You're building a graph database. Queries feel slow. A teammate says "I can optimize that for loop!"

Should you let them?

Without profiling data, you're guessing. And guessing wastes engineering time.

Here's how we built profiling infrastructure for Engram to find the real bottlenecks:

## Tweet 2/9
The brutal truth about performance: 2-5% of your functions account for 90%+ of runtime.

The other 95% of your code? Basically free.

Optimizing cold paths is like rearranging deck chairs on the Titanic - technically an improvement, but not where the problem lives.

## Tweet 3/9
Step 1: Flamegraphs

Visual representation of where your program spends time. Each horizontal bar is a function, width = cumulative runtime.

```bash
cargo install flamegraph
cargo flamegraph --bench workload
```

For Engram, three functions consumed 62% of runtime:
- cosine_similarity: 22%
- spread_activation: 28%
- apply_decay: 12%

That's where we optimize. Nowhere else (yet).

## Tweet 4/9
But flamegraphs show WHERE code is slow, not HOW MUCH faster you made it.

Enter Criterion.rs: statistical benchmarking with confidence intervals.

Not "I think it's faster", but "26% faster, ±2%, p < 0.05"

Real improvements vs measurement noise.

## Tweet 5/9
Example benchmark comparing Rust baseline vs Zig SIMD kernel:

```
vector_similarity_zig
  time:   [1.6943 ms 1.7102 ms 1.7279 ms]
  change: [-26.341% -25.672% -24.981%]
  Performance improved.
```

That's a real 25% speedup, not noise. Ship it.

## Tweet 6/9
Key insight from profiling Engram:

We expected graph traversal (DashMap lookups, BFS) to dominate. Wrong.

Vector operations (cosine similarity over 768-dim embeddings) consumed 25% of runtime.

It's not the graph that's slow - it's the vector math.

This completely changed our optimization strategy.

## Tweet 7/9
Optimization is an investment decision. You must justify the cost:

Option A: Optimize vector similarity (25% of runtime, SIMD gives 4-6x speedup)
→ Save 150ms, 15% faster overall

Option B: Optimize graph traversal (15% of runtime, algorithms give 20% improvement)
→ Save 30ms, 3% faster overall

Option A: 5x more ROI for similar effort. That's Amdahl's Law in action.

## Tweet 8/9
Reproducibility matters.

We built a deterministic workload (10k memories, 1000 queries, consistent RNG seed) so profiling results are actionable.

If flamegraph says "22% in cosine_similarity", that's reliable data, not variance from background processes.

## Tweet 9/9
Key lessons:

1. Profile before optimizing (flamegraphs find hotspots)
2. Benchmark with statistics (Criterion detects real improvements)
3. Make workloads reproducible (consistent results are actionable)
4. Calculate ROI (optimize where impact × potential is maximized)

Optimize the 2% that matters. Ignore the 98% that doesn't.

Not intuition. Data.
