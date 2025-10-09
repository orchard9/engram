# Task 011: Comprehensive Spreading Validation

## Objective
Ship a full validation suite that confirms correctness, performance, cognitive plausibility, and reliability of the spreading engine.

## Priority
P0 (Critical Path)

## Effort Estimate
1.5 days

## Dependencies
- Task 010: Spreading Performance Optimization

## Technical Approach

### Validation Layers
1. **Deterministic Fixtures**
   - Store golden activation traces under `engram-core/tests/data/spreading_snapshots/`. Serialize `ActivationResultData` (includes the deterministic trace from Task 006) and compare using `insta::assert_yaml_snapshot!`.
   - Build fixtures with `graph_builders.rs` utility (new helper) creating chains, cycles, binary trees, and fan-effect graphs.

2. **Property-Based Graph Testing**
   - Use `proptest` to generate random graphs (Erdős–Rényi, Barabási–Albert). Assert invariants: termination within `max_hops`, monotonic decay, confidence bounds.
   - Run in deterministic mode to make failures reproducible.

3. **Performance Regression**
   - Extend `engram-core/benches/spreading_benchmarks.rs` with Criterion benchmarks for single-hop (1 k nodes) and multi-hop (10 k nodes) workloads.
   - Emit JSON artifacts consumed by CI and Task 012 dashboards.

4. **Cognitive Experiment Replication**
   - Add `tests/cognitive_correctness.rs` replicating semantic priming (Meyer & Schvaneveldt, 1971), fan effect (Anderson, 1974), and exponential decay (Wixted, 1990).
   - Use the recall pipeline once Task 008 lands so tests exercise the full stack.

5. **Stress & Concurrency**
   - `tests/spreading_stress.rs` builds 1 M-node scale-free graphs via streaming generator and asserts completion under budget.
   - Run 100 concurrent spreads using Tokio to stress lock-free pools; monitor memory via `ActivationMemoryPool::stats`.

### Implementation Details
- Create `engram-core/tests/spreading_validation.rs` orchestrating fixtures and property tests.
- Expose helpers in `activation/test_support.rs` for constructing `ParallelSpreadingConfig` with deterministic + metrics instrumentation.
- Capture metrics snapshots (`ActivationMetrics::export`) and assert on latency, cycle detection, cache hit rate.

### Preparation Notes
- `engram-core/tests/spreading_validation_prep.rs` seeds deterministic configs and triangle graph fixtures for upcoming snapshot generation.
- `engram-core/tests/data/spreading_snapshots/` now holds documentation for storing golden traces; populate YAML fixtures once Task 010 completes.

### Acceptance Criteria
- [ ] Golden snapshots captured for canonical graphs and stored in repo
- [ ] Property tests cover ≥1 000 random graphs per run with invariants enforced
- [ ] Criterion benchmarks produce <10 ms P95 latency on warm-tier dataset; results archived
- [ ] Cognitive experiments assert priming, fan effect, and decay patterns
- [ ] Stress tests validate 1 M-node graph spreading within configured budget and bounded memory growth (<100 MB delta)
- [ ] CI pipeline runs validation suite nightly and fails on regressions

### Testing Approach
- Use deterministic mode (`ParallelSpreadingConfig::deterministic`) to keep snapshots stable
- Run soak tests with sanitizers (`cargo +nightly miri test`, `ASAN_OPTIONS`) to catch leaks/data races
- Hook benchmark outputs into Prometheus for cross-check against production metrics

## Risk Mitigation
- **Snapshot churn** → guard updates with `cargo insta review` and require reviewer sign-off
- **Long-running stress tests** → mark as `#[ignore]` in default CI but execute nightly
- **High variance benchmarks** → pin CPU frequency during benchmarking and warm caches before measurement

## Notes
Relevant modules:
- Parallel engine tests (`engram-core/src/activation/parallel.rs::test_deterministic_spreading`)
- SIMD mapper tests (`engram-core/src/activation/simd_optimization.rs`)
- Latency budget manager (`engram-core/src/activation/latency_budget.rs`)
