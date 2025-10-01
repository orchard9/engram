# Task 012: Production Integration and Monitoring

## Objective
Instrument spreading with production-grade monitoring, health checks, circuit breakers, and auto-tuning so operators can keep cognitive recall healthy.

## Priority
P1 (Production Readiness)

## Effort Estimate
1 day

## Dependencies
- Task 011: Comprehensive Spreading Validation

## Technical Approach

### Metrics
- Extend `engram-core/src/metrics/prometheus.rs` with spreading counters/histograms:
  - `engram_spreading_latency_seconds` (histogram per tier)
  - `engram_spreading_cycles_total`
  - `engram_spreading_pool_utilization`
  - `engram_spreading_gpu_launch_total` / `engram_spreading_gpu_fallback_total`
- Register metrics in `ActivationMetrics` (`activation/mod.rs`) and expose `fn record_spreading_latency(&self, tier, duration)` helpers.

### Health Checks
- Implement `SpreadingHealthChecker` in `engram-core/src/activation/health_checks.rs` using the existing `HealthChecker` trait (`metrics/health.rs`).
- Health check runs `ParallelSpreadingEngine::spread` on a 5-node cycle graph and asserts non-zero activation plus latency <50 ms. Expose via HTTP endpoint (`engram-cli` already has `/health` wiring).

### Circuit Breaker
- Add `SpreadingCircuitBreaker` to `activation/circuit_breaker.rs`, leveraging `Crossbeam` atomics. States: Closed/Open/HalfOpen. Failure criteria: failure rate >5 % over last 50 spreads or latency > time budget ×1.5.
- Wrap `CognitiveRecall::recall` calls: on breaker open, fall back to `recall_similarity` and increment metric.

### Auto-Tuning
- Implement `SpreadingAutoTuner` (`activation/auto_tuning.rs`) consuming metrics (latency, hop count, tier mix) and using EWMA to recommend new `SpreadingConfig` parameters (batch size, hop limit, decay rate). Guard with `predicted_improvement > 10 %` before applying.
- Log changes via `tracing::info!(old=?, new=?)` and expose to Prometheus as `engram_spreading_autotune_changes_total`.

### Dashboards & Alerts
- Provide Grafana JSON snippet under `docs/operations/spreading_dashboard.json` with panels for latency P95, throughput, pool utilization, circuit breaker state.
- Define Prometheus alert rules (`deploy/prometheus/spreading.rules.yml`): latency, failure rate, circuit breaker open.

## Acceptance Criteria
- [ ] Metrics compiled into Prometheus export and verified via `curl /metrics`
- [ ] Health checker integrated with CLI readiness endpoint; fails within 30 s of spreading outage
- [ ] Circuit breaker protects recall; tests verify fallback triggers after threshold and recovers correctly
- [ ] Auto-tuner adjusts parameters and logs decisions with audit trail
- [ ] Grafana dashboard and Prometheus rules committed with documentation
- [ ] Chaos test: inject failures (simulate high latency) and confirm alerts + circuit breaker behavior

## Testing Approach
- Unit tests for metrics registration (assert counters increment)
- Integration test toggling health checker success/failure
- Simulate latency spikes using mocked engine to ensure breaker trips and auto-tuner backs off batch size
- Chaos engineering script (`scripts/fuzz_spreading_latency.rs`) to validate alert pipeline

## Risk Mitigation
- **Metrics overhead** → batch histogram updates and sample high-frequency metrics at 1/10 rate when throughput high
- **Circuit breaker flapping** → use hysteresis (half-open limited to 5 probes, 30 s cool-down)
- **Auto-tuner instability** → enforce guardrails (batch size within [8, 128], hop limit within [2, 6]) and allow operator override

## Notes
Relevant modules:
- Prometheus exporter (`engram-core/src/metrics/prometheus.rs`)
- Health infrastructure (`engram-core/src/metrics/health.rs`)
- Activation metrics (`engram-core/src/activation/mod.rs`)
- CLI health routes (`engram-cli/src/server/health.rs`)
