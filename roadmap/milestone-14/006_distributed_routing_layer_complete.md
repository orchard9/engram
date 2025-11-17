# Task 006: Distributed Routing Layer — Complete

**Status**: Complete (code, config, docs, and tests landed)

## What Shipped

- Rebuilt `engram-cli/src/router.rs` into a full routing engine with configurable deadlines, connection pooling, exponential backoff + jitter, circuit breakers, replica-aware routing (`PrimaryOnly`, `NearestReplica`, `ScatterGather`), and replica-write fallbacks that respect Task 005 lag metadata. New metrics (`engram_router_requests_total`, `…_retries_total`, `…_circuit_breakers_open`, `…_replica_fallback_total`) feed the existing registry, and router errors map cleanly onto API/gRPC responses.
- HTTP API handlers (`remember_memory`, `remember_episode`, `recall_memories`) now lean on the router for both write and read operations. Requests automatically proxy to primaries/replicas, remote responses reuse the cognitive payloads, and structured errors expose partition/circuit-breaker state. Read routing supports nearest-replica fan-out with remote recall results merged into the existing `RecallResponse` format.
- gRPC service methods (`remember`, `recall`) gained the same behavior: writes proxy through `router.proxy_write`, reads call `router.proxy_read` using the configured strategy, and router failures convert to the correct `tonic::Status` codes. `ApiState::route_for_write` defers to the router, so the old SWIM-stat checks are no longer duplicated.
- Configuration grew a `[router]` section (`engram-cli/config/default.toml` + `cluster.toml`) that exposes retry counts, timeouts, circuit-breaker thresholds, and replica-fallback knobs. `CliConfig` now carries this block, and `start_server` wires it into the router.
- `/cluster/health` and `/api/v1/system/health` now surface router health snapshots (request counters, retries, replica fallbacks, breaker states), and `engram status --json` prints the same information so operators can diagnose routing pressure. The CLI docs/config samples describe the new `[router]` knobs and health output.
- New helpers in `api.rs` (`RecallCueSpec`, proto conversion utilities, remote recall response builder) keep HTTP recall logic unified between local and remote paths. Circuit-breaker and routing topology unit tests live inside `router.rs` to cover the state machine (including scatter/gather planning and breaker reporting).

## Validation

- `cargo fmt --all`
- `cargo test -p engram-cli router::tests::circuit_breaker_opens_after_threshold -- --exact`
- `cargo test -p engram-cli router::tests::circuit_breaker_closes_after_success -- --exact`
- `cargo test -p engram-cli router::tests::route_read_scatter_gather_targets_multiple_nodes -- --exact`
- `cargo test -p engram-cli router::tests::health_snapshot_reports_open_breakers -- --exact`
- `cargo test -p engram-cli cluster_health_reports_membership_breakdown -- --exact`
- `cargo test -p engram-cli recall_memories -- --nocapture` (spot-checks HTTP recall path)
- `cargo test -p engram-cli grpc::tests::memory_service_remember_routes_remote -- --exact` (updated gRPC routing behavior)

## Follow-Ups

- Add a lightweight multi-node harness (mock Engram peers) so future tests can drive actual remote recalls/writes without spinning up full servers, enabling regression tests for `proxy_write`/`proxy_read` behavior under failure.
- Feed router breaker/lag state into the monitoring dashboards (Grafana) once the metrics exporter wiring is ready, so operators can visualize routing pressure alongside SWIM membership.
