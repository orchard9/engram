# Spreading Performance Validation Guide

> **Status**: Outline – fill in raw metrics and interpretations once validation runs complete.

## Prefetch Validation (x86_64)
- Command template: `perf stat -e LLC-load-misses,cycles,instructions cargo bench -p engram-core --bench recall_performance -- --profile spreading`
- Capture baseline (prefetch disabled via `ENGRAM_PREFETCH_DISTANCE=0`) and optimized runs (default config).
- Record target metric: L2 miss rate < 5%, IPC delta, cycles reduction.
- Table placeholders:
  | Scenario | LLC Miss Rate | IPC | Notes |
  |----------|---------------|-----|-------|
  | Baseline | _TODO_ | _TODO_ | Disable prefetch |
  | Optimized | _TODO_ | _TODO_ | Prefetch distance N |
- Archive raw `perf` output under `engram-core/benches/artifacts/2025-10-xx-prefetch/`.

## ARM/NEON Fallback Checks
- Build scalar variant: `cargo test -p engram-core --lib --target aarch64-apple-darwin --no-default-features --features portable` (adjust target/toolchain per CI host).
- Ensure cfg-gated regression exercises `prefetch::maybe_prefetch` no-ops without panicking.
- Record qualitative notes + test output snippets (TODO) in this README once executed.

## Adaptive Batch Benchmarks (Post-RFC)
- Criterion suite: `cargo bench -p engram-core --bench comprehensive -- --measurement-time 30`
- Data set: `engram-data/warm_tier.dat` via streaming/mmap (avoid unit test inclusion).
- Metrics to capture: P95 latency (<10 ms target), throughput (requests/s), pool utilization plateau, adaptive convergence iterations.
- Store CSV/JSON results in `engram-core/benches/artifacts/2025-10-xx-adaptive/` with accompanying interpretation markdown.

## Hardware Scheduling
- Primary rig: `perf-rig-02` (32c/64t, 256 GB RAM) – request 3-hour window (TODO: submit ops ticket).
- Fallback: `perf-rig-arm01` for ARM validation (2-hour slot, ensure toolchain installed).
- Pre-run checklist: update to latest `main`, confirm `cargo clean`, pin CPU governor to performance, disable background services.

## Reporting Checklist
- Update `tmp/010—rewiring-todo.md` Section 7 with perf results summary.
- Attach raw artifacts + annotated tables in this README.
- Cross-link to Task 010 RFC + operator docs once numbers verified.
