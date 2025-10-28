---
title: Spreading Debugging Playbook
outline: deep
---

# Spreading Debugging Playbook

Diagnose spreading activation issues by combining API overrides, deterministic traces, and Task 012 metrics. This guide assumes you have already walked through the [Spreading Tutorial](../tutorials/spreading_getting_started.md).

## Prerequisites

- Engram built with the default feature set (`cargo build --workspace --features full`)

- CLI compiled with `hnsw_index` enabled

- `spreading_api_beta` set to `true` in `~/.config/engram/config.toml`
  - Run `engram config set feature_flags.spreading_api_beta true`
  - Restart the server (`engram stop && engram start`) so the flag is applied

## 1. Confirm the Recall Mode

Verify that spreading is available and that your request is using the expected strategy:

```bash
# Inspect the current flag value
engram config get feature_flags.spreading_api_beta

# Issue a spreading recall (falls back to similarity if the flag is off)
curl "http://localhost:7432/api/v1/memories/recall?query=dr+harmon&max_results=5&mode=spreading"

```

If you receive a 400 response mentioning the feature flag, restart the daemon after enabling it or switch to `mode=hybrid` while you investigate.

## 2. Capture Live Activation Events

Use the monitoring SSE endpoint to observe how activation propagates. Each event is JSON and can be inspected with `jq`:

```bash
curl "http://localhost:7432/api/v1/monitoring/events?event_types=activation,spreading&include_causality=true&max_frequency=5" \
  --no-buffer | jq '.activation // .spreading'

```

Look for:

- Missing activation entries for expected nodes (indicates sparse edges or low thresholds)

- High `latency_ms` or repeated `caused_by` IDs (suggests cycle pressure)

- Causality chains referencing old sequences (potentially stale recall results)

## 3. Generate Deterministic Traces

The CLI ships with seeded samples so you can compare faulty traces against a known-good baseline:

```bash
cargo run -p engram-cli --example spreading_visualizer --features hnsw_index \
  -- --input-trace docs/assets/spreading/trace_samples/seed_42_trace.json \
     --output target/debug-spreading.dot \
     --render-png target/debug-spreading.png

```

Open the generated PNG to verify:

- Activation fades with depth (`distance_decay` working)

- Confidence narrows along intended paths

- No isolated nodes remain bright after the spread completes

To diagnose a live issue, replace `--input-trace` with a trace captured from SSE and diff the resulting DOT against the baseline.

## 4. Inspect Metrics

Correlate visual findings with Prometheus metrics:

| Metric | Interpretation |
| --- | --- |
| `engram_spreading_latency_hot_seconds_bucket` (and warm/cold) | Latency distribution per tier; spikes indicate storage pressure |
| `engram_spreading_pool_utilization` | Pool saturation; values >0.8 imply aggressive spreading or memory leaks |
| `engram_spreading_breaker_state` | Circuit breaker position (1 = open) |
| `engram_spreading_autotune_changes_total` | Auto-tuner churn; sudden increases suggest unstable presets |
| `engram_spreading_gpu_fallback_total` | GPU disabled or overwhelmed, falling back to CPU |

Fetch a quick snapshot:

```bash
curl http://localhost:7432/metrics | rg "engram_spreading_"

```

## 5. Apply Fixes or Roll Back

| Symptom | Checks | Mitigation |
| --- | --- | --- |
| High latency across tiers | Latency histograms, SSE `latency_ms` | Reduce `max_hops`, lower `max_results`, or switch to `mode=hybrid` |
| Empty or low-confidence results | SSE payload shows low `activation_level` | Increase cue strength (provide embeddings) or raise `RecallConfig::min_confidence` |
| Circuit breaker stuck open | `engram_spreading_breaker_state == 1` | Drop to `RecallMode::Similarity`, investigate recent auto-tuner changes, then re-enable |

Document any parameter changes in `docs/changelog.md` and notify operators when reverting to spreading mode.

## Next Steps

- [Spreading Monitoring Runbook](spreading_monitoring.md)

- [Performance Tuning Guide](spreading_performance.md)

- [API Reference](../reference/spreading_api.md)
