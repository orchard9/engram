# Task 013: Production-Grade Monitoring for Dual Memory Operations

**Status**: Pending
**Estimated Duration**: 3 days
**Dependencies**: Tasks 001-012
**Owner**: TBD

## Objective

Instrument the dual-memory system (concept formation, binding dynamics, fan effect, recall) with low-overhead metrics, publish Grafana dashboards + Prometheus alerts, and validate the monitoring stack in a realistic cluster. Metrics must capture cognitive health (coherence, confidence calibration) in addition to traditional latency/throughput to give operators actionable signals before failures surface.

## Current Implementation Snapshot

- `engram-core/src/metrics` provides general-purpose counters/histograms/gauges and Prometheus export but lacks dual-memory-specific hooks.
- Grafana dashboards focus on storage/tier operations; no concept-level visibility.

## Technical Specification

### Metrics Module

Create `engram-core/src/metrics/dual_memory.rs` (feature-gated under `dual_memory_types` + `monitoring`) with instrumented structs:
- `ConceptFormationResult` (concept count, duration, coherence distribution).
- `BindingOperationResult` (bindings created/strengthened/weakened/gc’d, strength histograms, binding age gauges).
- `FanEffectObservation` (node degree, penalty magnitude, high-fan detection).
- `RecallMetrics` (episodic/semantic/blended latencies, accuracy/precision).

Expose helpers via `metrics/mod.rs` so consolidation/activation code can call `record_metrics()`.

### Instrumentation Points

- `consolidation/concept_formation.rs`: emit `ConceptFormationResult` after each run.
- `memory_graph/binding_index.rs`: emit `BindingOperationResult` when operations complete.
- `activation/parallel.rs` & `activation/blended_recall.rs`: record fan-effect and recall metrics.
- Ensure instrumentation respects feature flags (`cfg(feature = "monitoring")`).

### Prometheus Export & Dashboards

- Verify `/metrics` endpoint includes new series (e.g., `engram_concepts_formed_total`). Add integration test in `engram-cli/tests/monitoring_tests.rs` checking for these names when flags enabled.
- Update Grafana dashboards (`deployments/grafana/dashboards/`) with panels for concept lifecycle, binding dynamics, fan-effect penalties, recall splits. Document in `deployments/grafana/dashboards/README.md`.
- Add alert rules (Prometheus YAML) for key thresholds: e.g., concept quality violations > 0 over 10m, fan-effect penalty > threshold, recall latency exceeding SLO.

### Validation

- Unit tests for the metrics module (e.g., calling `record_metrics` increments counters/histograms).
- Manual/local validation: run `cargo run -p engram-cli -- start` with sample workloads, verify Grafana shows data.
- Document configuration knobs (sampling rates, thresholds) and update `docs/reference/configuration.md`.

## Acceptance Criteria

1. Dual-memory metrics module exists, exported, and covered by tests.
2. Consolidation/activation/binding pathways publish metrics when the monitoring feature is enabled.
3. `/metrics` exposes the new series, and Grafana dashboards render them.
4. Alert rules exist with justified thresholds tied to cognitive health signals.
5. Documentation explains each metric and how to interpret dashboard panels.
