---
title: Spreading API Reference
outline: deep
---

# Spreading API Reference

This reference mirrors the Rustdoc tables embedded directly into the codebase. Use it alongside the generated docs for a quick lookup.

> **Beta flag**: Spreading APIs are controlled by the `spreading_api_beta` feature flag. The CLI stores the flag in `~/.config/engram/config.toml`; use `engram config get feature_flags.spreading_api_beta` to inspect the current value.

## `ParallelSpreadingConfig`

```rust
use engram_core::activation::ParallelSpreadingConfig;

```

`ParallelSpreadingConfig` controls concurrency, hop limits, decay, determinism, and GPU thresholds. The Rustdoc table is sourced from `engram-core/src/activation/doc/parallel_spreading_config.md`.

Key highlights:

- `num_threads` — worker count; set to `num_cpus::get()` for throughput, lower to constrain CPU usage.

- `max_depth` — hop limit; aligns with semantic distance.

- `threshold` — activation floor; increases latency when too low.

- `deterministic` + `seed` — reproducible spreads for debugging.

- `enable_gpu` / `gpu_threshold` — offload to GPU batch kernels when ready.

## `SpreadingConfig` (HNSW)

```rust
use engram_core::activation::HnswSpreadingConfig;

```

Only available when `hnsw_index` is enabled. The doc include lives in `engram-core/src/activation/doc/hnsw_spreading_config.md`.

Fields:

- `similarity_threshold` — minimum cosine similarity to propagate activation.

- `distance_decay` — exponential decay per hop.

- `max_hops` — HNSW layer traversal depth.

- `use_hierarchical` — toggles multi-layer navigation.

- `confidence_threshold` — minimum confidence for neighbours.

## `GPUSpreadingInterface`

```rust
use engram_core::activation::GPUSpreadingInterface;

```

The GPU trait documents supported capabilities in `engram-core/src/activation/doc/gpu_interface.md`. When compiled without GPU backends, Engram falls back to CPU processing.

Methods:

- `capabilities()` — inspect device limits (`GpuCapabilities`).

- `is_available()` — runtime check for GPU readiness.

- `launch()` — submit activation batches (returns `GpuLaunchFuture`).

- `warm_up()` / `cleanup()` — manage device lifecycle hooks.

## CLI Hooks

In the CLI, spreading is surfaced through `RecallMode` and `RecallConfig` (see `engram-core/src/activation/recall.rs`). The `spreading_api_beta` flag toggles the API state in configuration files, and the new `spreading_visualizer` example converts traces into DOT/PNG files.

## HTTP Recall Parameters

`GET /api/v1/memories/recall` accepts additional query parameters to control spreading at runtime:

| Parameter | Values | Behaviour |
| --- | --- | --- |
| `mode` | `similarity` (default), `spreading`, `hybrid` | Overrides the server’s configured `RecallMode`. Spreading/Hybrid require `hnsw_index` and `spreading_api_beta=true`; otherwise the API returns HTTP 400. |
| `max_results` | integer | Caps ranked memories. Combined with `mode=spreading` to constrain hop fan-out. |
| `trace_activation` | `true` / `false` | Reserved for future activation trace exports. Currently no-op but preserved for compatibility. |

Changing the feature flag writes to `~/.config/engram/config.toml`; restart the server after toggling it so the override applies to the in-memory store.

## Monitoring Endpoints

- `GET /api/v1/monitoring/events` — SSE stream for activation summaries (`event_types=activation,spreading`).

- `GET /api/v1/monitoring/activations` — Focused activation traces (`include_spreading=true`).

- `GET /metrics` — JSON snapshot that exposes `engram_spreading_*` counters/gauges for Prometheus scraping.

## Additional Resources

- [Spreading Activation Tutorial](../tutorials/spreading_getting_started.md)

- [Spreading Performance Guide](../howto/spreading_performance.md)

- [Spreading Monitoring Runbook](../howto/spreading_monitoring.md)

- [Cognitive Spreading Explanation](../explanation/cognitive_spreading.md)
