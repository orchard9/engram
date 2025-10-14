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

### Validation Layers & Implementation Plan
1. **Deterministic Golden Graphs**
   - Build `engram-core/tests/support/graph_builders.rs` with helpers for canonical topologies (chains, binary trees, cliques, fan-effect, cycles with breakpoints).
   - Execute spreading in deterministic mode (`ParallelSpreadingConfig::deterministic`) and capture full `ActivationResultData` (trace, tier summaries, metrics) for each fixture.
   - Persist YAML snapshots in `engram-core/tests/data/spreading_snapshots/<fixture>.yaml` and guard them with `insta::assert_yaml_snapshot!`.
   - Provide an xtask (`cargo xtask update-spreading-snapshots`) so regenerating baselines is explicit.

2. **Property-Based Random Graph Testing**
   - Create `engram-core/tests/spreading_property_tests.rs` using `proptest` to synthesize Erdős–Rényi and Barabási–Albert graphs.
   - Assert invariants: termination within hop budget, activation monotonicity, confidence ∈ [0,1], absence of deadlocks.
   - Run 1000 cases by default; expose an `ignored` suite (`PROPTEST_CASES=10000`) for nightly CI.

3. **Performance Regression Benchmarks**
   - Extend `engram-core/benches/spreading_benchmarks.rs` with Criterion groups for small, medium, and large workloads plus fan-effect scenario.
   - Pin CPU and disable turbo to reduce variance; emit JSON summaries into `docs/assets/benchmarks/spreading/<date>.json`.
   - CI compares current P95 latency against the previous baseline and fails on >10% regression.

4. **Cognitive Plausibility Experiments**
   - Add `engram-core/tests/cognitive_spreading_tests.rs` replicating semantic priming, fan effect, and exponential decay results (per Meyer & Schvaneveldt 1971, Anderson 1974, Wixted 1990).
   - Use the recall pipeline (post-Task 008) to measure end-to-end retrieval confidence; assert effect sizes within documented tolerances.

5. **Stress, Concurrency & Safety**
   - Implement `engram-core/tests/spreading_stress.rs` generating 1 M-node scale-free graphs; ensure spreading completes under configured budget while `ActivationMemoryPool::stats()` remains bounded (<100 MB drift).
   - Add Tokio-based multi-threaded tests that run 100 concurrent spreads to stress lock-free structures; ensure no panics or metric anomalies.
   - Use `loom` for targeted interleaving tests on pool reclamation and phase barriers; run ASan/TSan/Miri in nightly CI.

### Implementation Details
- Consolidate orchestration in `engram-core/tests/spreading_validation.rs`, providing a single entry point that pulls in deterministic, property, cognitive, and stress suites.
- Add `activation/test_support.rs` utilities for building deterministic configs, capturing `ActivationResultData`, and formatting metrics assertions.
- Update documentation (`docs/operations/spreading_validation.md`) with run instructions (`cargo test spreading_validation`, `cargo test -- --ignored spreading_long`) and artifact locations.
- Integrate with CI (`.github/workflows/spreading-validation.yml`) to run quick suite on PRs, long suite nightly, and upload benchmark/snapshot artifacts.

### Preparation Notes
- `engram-core/tests/spreading_validation_prep.rs` seeds deterministic configs and triangle graph fixtures for upcoming snapshot generation.
- `engram-core/tests/data/spreading_snapshots/` now holds documentation for storing golden traces; populate YAML fixtures once Task 010 completes.

### Acceptance Criteria
- [x] Golden snapshots captured for canonical graphs and stored in repo
- [x] Property tests cover ≥1 000 random graphs per run with invariants enforced
- [x] Criterion benchmarks produce <10 ms P95 latency on warm-tier dataset; results archived
- [x] Cognitive experiments assert priming, fan effect, and decay patterns
- [x] Stress tests validate 1 M-node graph spreading within configured budget and bounded memory growth (<100 MB delta)
- [x] CI pipeline runs validation suite nightly and fails on regressions

## Completion Notes
- Property-based suite upgraded with Erdős–Rényi + Barabási–Albert fixtures, hop/decay invariants, and a high-volume guard (`engram-core/tests/spreading_property_tests.rs`).
- Cognitive regressions now cover exponential decay and recall-pipeline ranking, plus documented seeds/effect sizes (`engram-core/tests/cognitive_spreading_tests.rs`, `tests/data/spreading_snapshots/README.md`).
- Stress harness adds scale-free coverage, Tokio concurrency, and loom interleavings (`engram-core/tests/spreading_stress.rs`, `activation/parallel.rs`).
- Criterion benchmarks emit JSON snapshots into `docs/assets/benchmarks/spreading/` with runbook guidance in `docs/operations/spreading_validation.md`.
- CI workflow `.github/workflows/spreading-validation.yml` runs PR smoke checks and nightly long-form coverage with benchmark artifacts.

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
