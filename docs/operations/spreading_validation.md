# Spreading Validation Runbook

This runbook captures the full validation surface for the spreading activation engine: deterministic snapshots, property fuzzing, cognitive plausibility checks, stress/concurrency harnesses, and performance baselines.

## Golden Snapshots

- Regenerate fixtures with `cargo xtask update-spreading-snapshots` (runs in deterministic mode with per-fixture overrides from `tests/support/graph_builders.rs`).

- Review changes interactively via `cargo insta review` before committing.

- Reference seeds/effect sizes in `tests/data/spreading_snapshots/README.md` when interpreting diffs.

## Property-Based Coverage

- Default fast suite (`1_000` cases, ER + BA graphs):

  ```bash
  cargo test --package engram-core spreading_activation_invariants_hold
  ```

- Nightly volume run (`10_000` cases) guarded behind `#[ignore]`:

  ```bash
  PROPTEST_CASES=10000 cargo test --package engram-core \
    spreading_activation_invariants_high_volume -- --ignored
  ```

## Stress & Concurrency

- Scale-free regression guard (5k/20k nodes) runs by default:

  ```bash
  cargo test --package engram-core scale_free_graphs_complete_within_budget
  ```

- Million-node soak (≈ minutes, gated):

  ```bash
  cargo test --package engram-core million_node_scale_free_soak -- --ignored
  ```

- Tokio concurrency flood (requires default `memory_mapped_persistence` feature):

  ```bash
  cargo test --package engram-core concurrent_spreading_runs_share_pools_safely
  ```

## Loom & Sanitizers

- Activation pool + barrier loom models (run with nightly `loom` builds):

  ```bash
  RUSTFLAGS="--cfg loom" cargo test --package engram-core loom_models::activation_pool_reclaims_records_across_interleavings
  RUSTFLAGS="--cfg loom" cargo test --package engram-core loom_phase_barrier_resets_without_deadlock
  ```

- Sanitizer sweeps (nightly toolchain):

  ```bash
  # AddressSanitizer
  RUSTFLAGS="-Zsanitizer=address" cargo +nightly test --package engram-core
  # ThreadSanitizer
  RUSTFLAGS="-Zsanitizer=thread" cargo +nightly test --package engram-core
  # Miri (slow; enable only on targeted modules)
  MIRIFLAGS="-Zmiri-tag-raw-pointers" cargo +nightly miri test --package engram-core
  ```

  Set `ASAN_OPTIONS=detect_leaks=1` when running ASan locally to surface pool leaks.

## Performance Benchmarks

- Run deterministic Criterion suite:

  ```bash
  cargo bench --bench spreading_benchmarks -- --baseline main
  ```

  Results land in `docs/assets/benchmarks/spreading/` with Criterion’s JSON output.

- Compare against prior baseline via `cargo bench ... --baseline <previous-tag>`.

- Treat >10% movement in median/P95 as a regression candidate; attach deltas to review notes.

- Use `cargo xtask check-spreading-benchmarks` to compare the latest run against `docs/assets/benchmarks/spreading/baseline.json`; CI fails when drift exceeds the configured tolerance.

## Quick Review Checklist

- [ ] Snapshots updated (or explicitly unchanged) with reviewer approval.

- [ ] Property tests (fast + nightly) exercised, seeds recorded for new failures.

- [ ] Stress/concurrency suite green (include soak evidence if modified).

- [ ] Loom + sanitizer sweeps issue-free, or follow-up ticket filed.

- [ ] Latest benchmark JSON archived and compared to baseline.
