# 006: Per-Space Metrics & Observability — _pending_

## Goal
Extend monitoring and diagnostics so operators can inspect health, capacity, and activation metrics for each memory space independently. Observability must make it obvious which tenant is causing pressure and integrate with existing CLI and diagnostics workflows.

## Deliverables
- Metrics registry updates that attach `memory_space` labels to counters, gauges, and histograms (store counts, WAL lag, tier usage, consolidation health).
- HTTP status and health endpoints returning per-space JSON structures (counts, pressure, consolidation cadence).
- CLI status output showing per-space metrics (tabular format) and an optional `--space` filter aligning with Task 004.
- SSE/monitoring streams enriched with `memory_space_id` fields and filtered results.
- Diagnostics script (`scripts/engram_diagnostics.sh`) capturing per-space metrics and appending to `tmp/engram_diagnostics.log` without breaking legacy format.

## Implementation Plan

1. **Metrics Registry Enhancements**
   - Update `engram-core/src/metrics/mod.rs` to accept `MemorySpaceId` when recording metrics (e.g., `increment_counter_with_labels`).
   - Introduce helper `fn with_space(label: &MemorySpaceId) -> Vec<(&'static str, String)>` for consistent label creation.
   - Modify store operations (`store.rs`) to pass space label for counters/histograms.

2. **MemoryEvent Updates**
   - Ensure `MemoryEvent` (already augmented in Task 002) carries `memory_space_id` and that event emission records space.
   - Update SSE serialization in `engram-cli/src/api.rs` to include space field.

3. **Health & Status Endpoints**
   - Extend `/api/v1/system/health` handler to aggregate metrics per space (counts, pressure ratios, WAL backlog) using registry APIs.
   - Adjust response schema documented in `engram-cli/src/api.rs` (utoipa docs) to reflect new structure and add examples.

4. **CLI Status**
   - Modify `engram-cli/src/cli/status.rs` to fetch extended health payload, print per-space table (columns: space, memories, pressure, wal_lag_ms, consolidation_rate).
   - Add `--space` filter flag to limit output.

5. **Diagnostics Script**
   - Update `scripts/engram_diagnostics.sh` to call new health endpoint and append per-space metrics with clear delimiters.
   - Ensure script remains compatible with existing log parser (prepend timestamp, maintain headings).

6. **Metrics Exporters**
   - If Prometheus/JSON exporters exist, ensure label cardinality is acceptable; document label keys in code comments.
   - Update `docs/metrics-schema-changelog.md` with new series names.

7. **Tracing Integration**
   - Add `memory_space` field to tracing spans in key paths (store, recall, persistence) to enable log filtering.

## Integration Points
- `engram-core/src/metrics/*`, `engram-core/src/store.rs`, `engram-core/src/storage/wal.rs` (for lag metrics).
- `engram-cli/src/api.rs` (health + SSE), `engram-cli/src/cli/status.rs`, `engram-cli/src/docs.rs`.
- `scripts/engram_diagnostics.sh`.
- Documentation files: `docs/metrics-schema-changelog.md`, `docs/operations/operations.md`.

## Acceptance Criteria
1. Metrics exporters expose `memory_space` label on existing series; unit test asserts label presence.
2. `/api/v1/system/health` returns JSON keyed by space; CLI renders table and filter flag works.
3. SSE subscribers receive events with `memory_space_id` and filtering logic verified via tests.
4. Diagnostics script logs per-space sections without breaking historic format (manual verification + automated diff in tests).
5. Metrics documentation updated with new series and labels.

## Testing Strategy
- Unit tests for metrics helper ensuring label generation consistent and sanitized.
- Integration tests hitting health endpoint with multiple spaces populated (use test server) and asserting JSON schema via serde_json.
- SSE integration test subscribing to two spaces concurrently verifying isolation.
- Script test (bash `bats` or Rust integration) running diagnostics against test server and checking output formatting.
- Regression test verifying single-space deployment still produces metrics without label explosion.

## Review Agent
- `systems-architecture-optimizer` to validate metrics performance and labeling conventions.
