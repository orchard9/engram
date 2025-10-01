# Comprehensive Spreading Validation: Proving Cognitive Spreading Works

*Perspective: Verification & Testing*

Spreading activation now powers Engram's recall, but without a rigorous validation suite we cannot trust its behavior. Task 011 builds that safety net by combining deterministic fixtures, cognitive experiments, performance benchmarks, and stress tests. The result is a suite that certifies correctness, performance, and cognitive plausibility before every release.

## Deterministic Fixtures
We start with deterministic graphs: chains, cycles, balanced trees, cliques, and fan-effect networks. Each fixture has an expected activation trace stored as a golden JSON snapshot. Running in deterministic mode ensures snapshots remain stable. When a change alters activation magnitudes, the snapshot diff highlights exactly which nodes changed.

```rust
let snapshot = engine
    .spread_with_config(&fixture, params.clone())
    .await?;
assert_snapshot!("chain_depth_5", snapshot.to_snapshot());
```

## Property-Based Random Graphs
Golden tests catch regressions on known structures; property-based tests explore the rest. Using `proptest`, we generate random graphs with variable degree distributions, run spreading, and assert invariants: termination within hop limits, monotonic activation decay, confidence ∈ [0, 1]. Failures shrink to minimal counterexamples, enabling fast debugging.

## Cognitive Experiment Replication
To confirm cognitive realism, we encode classic experiments:
- **Semantic Priming** (Meyer & Schvaneveldt, 1971): activation from `DOCTOR` boosts `NURSE` more than unrelated words.
- **Fan Effect** (Anderson, 1974): nodes with higher fan distribute activation more thinly.
- **Decay Curves** (Wixted, 1990): activation decreases exponentially with hop count.

Passing these tests demonstrates Engram's spreading behavior aligns with decades of cognitive research.

## Performance Regression Benchmarks
Criterion and Divan benchmarks lock in latency targets. We benchmark single-hop and multi-hop spreads on graphs of increasing size, capturing P50, P95, and throughput metrics. Results export to JSON artifacts and Prometheus, so CI flags regressions. Perf counters (`LLC-load-misses`, `instructions`) record cache efficiency improvements from Task 010.

## Stress and Concurrency Tests
Large-scale tests use 1M-node Barabási–Albert graphs to simulate real-world knowledge networks. We run 100 concurrent spreads to stress lock-free pools, monitoring memory usage before and after to catch leaks. Loom-based unit tests explore small interleavings deterministically, while nightly soak tests ensure long-run stability.

## Running the Suite
Developers execute the full suite with `cargo test -p engram-core --features full_validation`. Benchmarks run via `cargo bench --bench spreading`. Failing tests provide deterministic reproduction steps, snapshot diffs, and log bundles for rapid analysis.

## References
- Meyer, D. E., & Schvaneveldt, R. W. "Facilitation in recognizing pairs of words." *Journal of Experimental Psychology* (1971).
- Anderson, J. R. "Retrieval of propositional information from long-term memory." *Cognitive Psychology* (1974).
- Wixted, J. T. "Analyzing the empirical course of forgetting." *Journal of Experimental Psychology: Learning, Memory, and Cognition* (1990).
