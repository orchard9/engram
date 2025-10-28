---
title: Spreading Activation Tutorial
outline: deep
---

# Spreading Activation Tutorial

Learn how to enable cognitive spreading activation, run recalls with the Engram CLI, and capture traces you can feed into the visualizer.

## Prerequisites

- Engram built with the default feature set (`cargo build --workspace`)

- Spreading enabled through the `hnsw_index` feature (default) and the `spreading_api_beta` flag

- A running Engram server (`engram start`)

## 1. Start Engram With Spreading Enabled

```bash
cargo run -p engram-cli -- start --port 7432 --grpc-port 50051

```

On startup the CLI logs whether the `spreading_api_beta` flag is enabled. Flip it on or off at runtime:

```bash
# view config
engram config list

# toggle spreading
engram config set feature_flags.spreading_api_beta true

```

> The server reads feature flags at startup. After toggling `spreading_api_beta`, restart the Engram daemon (`engram stop` followed by `engram start`) so the spreading endpoints become available.

## 2. Seed Memories

Use the CLI memory helpers to seed a minimal semantic net and an episodic fragment:

```bash
engram memory create "Dr. Harmon sees patient Lucy" --confidence 0.82
engram memory create "Nurse Lucy schedules follow-up" --confidence 0.78
engram memory create "Doctor consults cardiology" --confidence 0.74

```

These short phrases become nodes in the spreading graph after embeddings are generated.

## 3. Run Spreading Recall

Spreading-driven recalls flow through the `recall` API with `recall_mode=spreading`.

```bash
curl "http://localhost:7432/api/v1/memories/recall?query=doctor&max_results=5&mode=spreading"

```

Expect results that include activation heat (`activation`), aggregated confidence (`confidence.raw`), and semantic similarity when available.

## 4. Capture Deterministic Traces

For reproducible docs and debugging, use the bundled deterministic trace with the visualizer:

```bash
cargo run -p engram-cli --example spreading_visualizer --features hnsw_index \
  -- --input-trace docs/assets/spreading/trace_samples/seed_42_trace.json \
     --output target/spreading.dot \
     --render-png target/spreading.png

```

To capture live activation events from the running server, tail the server-sent events endpoint:

```bash
curl "http://localhost:7432/api/v1/monitoring/events?event_types=activation,spreading&include_causality=true" \
  --no-buffer | jq '.spreading | select(. != null)'

```

Save the JSON payload to a file—`examples/spread_trace.json`—for use with the visualizer example described later in this guide.

## 5. Verify Monitoring Metrics

Spreading produces Prometheus metrics when Task 012 is complete. Spot-check them locally:

```bash
curl http://localhost:7432/metrics | rg "engram_spreading_"

```

Key series to watch during the tutorial:

- `engram_spreading_latency_hot_seconds_bucket`

- `engram_spreading_latency_warm_seconds_bucket`

- `engram_spreading_latency_cold_seconds_bucket`

- `engram_spreading_pool_utilization`

- `engram_spreading_breaker_state`

## 6. Next Steps

- Step through the [performance tuning guide](../howto/spreading_performance.md)

- Dive into the theory behind spreading in [Cognitive Spreading](../explanation/cognitive_spreading.md)

- Explore the API surface in [Spreading Reference](../reference/spreading_api.md)

- Render the saved trace with `cargo run -p engram-cli --example spreading_visualizer`

With these pieces in place, teams can spin up spreading activation in less than ten minutes and validate it end-to-end.
