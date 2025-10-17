# RFC 010: Adaptive Batcher for Parallel Spreading

> **Status**: Draft – structure prepared, awaiting parameter tuning data and reviewer feedback.

## Summary

- Introduce an adaptive batching component that tunes spreading batch sizes based on CPU topology and recent spread latencies.
- Ensure convergence within three iterations via EWMA smoothing with configurable bounds.
- Surface live batch size, convergence confidence, and guardrail activations through streaming metrics and logs.

## Goals

- Reduce per-hop latency variance across heterogeneous hosts.
- Maintain deterministic trace compatibility when deterministic mode is enabled.
- Expose telemetry that enables Task 012 GPU dispatcher to consume adaptive signals without additional wiring.

## Non-Goals

- GPU-specific scheduling (handled in Task 012).
- Alternative adaptive algorithms (keep scope to EWMA + bounds for now).
- Dynamic NUMA pinning (evaluate separately under systems-architecture-optimizer guidance).

## Background & Current State

- Recap existing static `ParallelSpreadingConfig::batch_size` heuristic (TODO: link to source + numbers).
- Summarize pool/cache improvements landed in phase 2 and highlight remaining contention points.
- Note outstanding flaky test (`test_deterministic_across_thread_counts`) and mitigation plan.

## Design Overview

1. **Topology Probe**
   - Collect `available_parallelism`, logical vs physical core counts, NUMA socket map (optional `numa` crate fallback).
   - Cache topology snapshot in `AdaptiveBatcher::TopologyState` with change detection.
2. **Feedback Loop**
   - Maintain rolling history per workload class: `(batch_size, hop_count, tier_mix, observed_latency)`.
   - Calculate recommended batch size via EWMA: `next = clamp(alpha * observed + (1 - alpha) * current)`.
   - Guardrails: min/max batch size, oscillation dampening threshold, stall detection.
3. **Integration Points**
   - Update `ParallelSpreadingConfig` at the start of each spread; persist final batch size into `SpreadingMetrics`.
   - Emit telemetry event `adaptive_batcher_update` in logs with rationale and guardrail flags.
   - Provide `AdaptiveBatcherHandle` to `AdaptiveSpreadingEngine` for Task 012 reuse.

## Data Structures

- `AdaptiveBatcher { topology: TopologyState, ewma: EwmaController, history: RingBuffer<Observation>, guardrails: GuardrailSet }`
- `Observation { batch_size: usize, latency_ns: u64, hop_count: usize, tier_mix: TierMix }`
- `GuardrailSet { min_batch, max_batch, oscillation_threshold, stall_timeout }`
- `TelemetrySnapshot { batch_size, ewma_state, guardrail_triggered, confidence }`

## Metrics & Telemetry

| Signal | Source | Description | Reset Semantics |
|--------|--------|-------------|-----------------|
| `adaptive_batch_size_current` | Streaming gauge | Latest batch size applied | Zeroed on `SpreadingMetrics::reset` |
| `adaptive_batch_update_count` | Counter | Number of adjustments performed | Monotonic |
| `adaptive_batch_guardrail_hits` | Counter | Guardrail activations per type | Monotonic |
| `adaptive_batch_latency_ewma_ns` | Gauge | Smoothed latency output | Zeroed on reset |
| `adaptive_batch_confidence` | Gauge | 0.0–1.0 convergence confidence | Zeroed on reset |

Additional structured log payload: `{ batch_size, ewma, guardrail, reason, topology_hash }` (hash = TODO for stable identifier).

## Parameter Defaults

- **EWMA alpha**: 0.45 (balance responsiveness vs stability on current benches).
- **Batch bounds**: min = 4, max = 128 (respect SIMD lanes + queue depth limits).
- **Cooldown interval**: 50 spreads between adjustments to prevent thrashing.
- **Oscillation threshold**: ignore adjustments if |Δbatch| < 2 for two consecutive iterations.
- **Topology hash**: stable tuple `(logical_cores, physical_cores, numa_domains)` hashed with FNV; reset EWMA if value changes.
- **Confidence calculation**: `confidence = 1.0 - exp(-updates / convergence_window)` with window = 5 observations.

## Failure Modes & Mitigations

- **Oscillation**: clamp delta updates, require cooldown interval before next adjustment.
- **Topology Drift**: re-run probe on start and periodically (configurable), reset EWMA if topology hash changes.
- **Cold Start**: seed EWMA with config default and mark confidence low until history length ≥ 3.
- **Budget Overruns**: integrate with `LatencyBudgetManager` to request fallback; log guardrail event + increment counter.

## API Changes

- Extend `ParallelSpreadingConfig` with `adaptive: Option<AdaptiveBatcherConfig>` containing alpha, bounds, cooldowns.
- Add `AdaptiveBatcherHandle` to `AdaptiveSpreadingEngine` interface (no functional GPU work yet).
- Update gRPC/HTTP telemetry surfaces to include adaptive batch fields.

## Testing Strategy

- Unit tests for convergence across low/high core counts (simulate with fake topology + synthetic observations).
- Property tests for oscillation damping (random workloads, ensure guardrail hits ≤ threshold).
- Integration test hooking `ParallelSpreadingEngine` to mock topology to verify deterministic mode unaffected.
- Snapshot refresh for `/metrics` and gRPC payloads once adaptive fields land.

## Rollout Plan

1. Land adaptive batcher behind feature flag (`adaptive_batching`), disabled by default.
2. Enable in staging with additional logging; monitor `adaptive_batch_guardrail_hits` and latency distributions.
3. If stable, default to adaptive mode on multi-core hosts; retain config escape hatch.
4. Document fallback procedure (toggle flag + restart engine) in operations runbook.

## Open Questions

- How should EWMA alpha vary with hop count diversity? (Need benchmark data.)
- Do we require per-tier batch sizing or global value is sufficient?
- Should NUMA topology changes trigger full history reset or partial decay?
- Hardware counter integration timing relative to adaptive rollout (tie-in with Task 012?).

## Reviewer Checklist

- Systems architecture (rust-graph-engine-architect) – concurrency & topology validation.
- Verification/testing lead – coverage adequacy for oscillation/failure modes.
- Product/planning – rollout UX & config surface alignment.

## Reviewers & Stakeholders (Draft)

- `@systems-architecture-optimizer` (Aditi) – topology + scheduling review.
- `@verification-testing-lead` (Rowan) – test plan validation.
- `@technical-communication-lead` (Mira) – telemetry documentation alignment.
- `@systems-product-planner` (Noah) – roadmap + feature flag coordination.

See `docs/rfcs/rfc-010-adaptive-batcher-summary.md` for the review meeting agenda and invite checklist.

### Circulation Plan

- Prepare 2-page summary deck with RFC highlights + parameter table for synchronous walkthrough.
- Schedule reviewer meeting (target 2025-10-16) once alpha/bounds validated on warm-tier dataset.
- Publish RFC PR linking to reviewer checklist and attach latest `docs/assets/metrics/` samples for telemetry context.

## Dependencies & Blocking Work

- Metrics reset regression harness (landed 2025-10-12).
- Resolve `test_deterministic_across_thread_counts` timeout (prerequisite for confidence in adaptive loops).
- Confirm perf hardware availability window (contact ops to reserve `perf-rig-02`, TBD).
- Replace synthetic soak snapshots (`docs/assets/metrics/2025-10-12-longrun/`) with live CLI run ahead of rollout to validate pool telemetry under sustained load.

## Timeline (Tentative)

- Draft review circulation: **2025-10-15**.
- RFC sign-off: **2025-10-20** (pending test stability + hardware confirmation).
- Implementation window: **2025-10-21 – 2025-10-24**.

## Action Items

- [ ] Fill in topology probe details (link to helpers, error handling).
- [x] Produce EWMA parameter recommendations from existing benchmarks (seed defaults documented 2025-10-12; validate against upcoming perf runs).
- [ ] Document guardrail math and thresholds in detail.
- [ ] Confirm reviewer availability and schedule live walkthrough.
- [ ] Log hardware reservation ticket (ops-123) once window approved.
