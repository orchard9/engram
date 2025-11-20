# 006c: Per-Space Metrics & Observability — _90_percent_complete_

## Current Status: 90% Complete

**What's Implemented**:
- ✅ `with_space()` helper for consistent label creation (engram-core/src/metrics/mod.rs:180)
- ✅ `record_counter_with_space()` and `record_histogram_with_space()` methods
- ✅ `memory_space` label attached to metrics throughout codebase
- ✅ MemoryEvent variants include `memory_space_id` field (from Task 002)
- ✅ Metrics infrastructure ready for per-space tracking
- ✅ Tracing context includes space ID in spans

**Minor Gaps Remaining** (10%):
- ❌ CLI status command doesn't display per-space metrics breakdown
- ❌ Health endpoint aggregation by space not tested/validated
- ❌ Diagnostics script not updated to capture per-space metrics
- ❌ SSE stream filtering by space not validated in tests

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

1. ✅ **COMPLETE**: Metrics exporters expose `memory_space` label on existing series
   - Implementation: `with_space()` helper provides consistent labeling
   - Methods: `record_counter_with_space()`, `record_histogram_with_space()`
   - ⚠️ Missing: Unit test explicitly asserting label presence

2. ⚠️ **PARTIAL**: `/api/v1/system/health` returns JSON keyed by space
   - Infrastructure: Ready to aggregate by space
   - ❌ Missing: Health endpoint implementation aggregating per-space metrics
   - ❌ Missing: CLI rendering of per-space table

3. ⚠️ **PARTIAL**: SSE subscribers receive events with `memory_space_id`
   - ✅ Complete: MemoryEvent includes space ID
   - ❌ Missing: Filtering validation tests
   - ❌ Missing: Confirmation streams are actually filtered by space

4. ❌ **NOT STARTED**: Diagnostics script logs per-space sections
   - Script: `scripts/engram_diagnostics.sh` not yet updated
   - Format: Need to maintain backward compatibility with existing log parser
   - Requirement: Clear delimiters for per-space metric sections

5. ❌ **NOT STARTED**: Metrics documentation updated
   - File: `docs/metrics-schema-changelog.md` not updated
   - Need: Document new `memory_space` label and series

## Remaining Work

1. **Health Endpoint Per-Space Aggregation** (3 hours)
   - File: `engram-cli/src/api.rs` health handler
   - Action: Aggregate metrics by space using registry
   - Response: JSON structure with per-space counts, pressure, WAL lag
   - Schema: Update utoipa docs

2. **CLI Status Per-Space Display** (2 hours)
   - File: `engram-cli/src/cli/status.rs`
   - Action: Fetch extended health payload, render table
   - Columns: space, memories, pressure, wal_lag_ms, consolidation_rate
   - Flag: Add --space filter option

3. **Diagnostics Script Update** (2 hours)
   - File: `scripts/engram_diagnostics.sh`
   - Action: Call health endpoint, parse per-space metrics
   - Format: Maintain compatibility with existing log parser
   - Append: Per-space sections to `tmp/engram_diagnostics.log`

4. **Metrics Documentation** (1 hour)
   - File: `docs/metrics-schema-changelog.md`
   - Content: Document `memory_space` label on all series
   - Examples: Show sample Prometheus queries with space filter

5. **SSE Filtering Tests** (1 hour)
   - File: `engram-cli/tests/streaming_tests.rs`
   - Test: Subscribe to multiple spaces, verify isolation
   - Validate: Events include space ID and no cross-space leakage

## Testing Strategy
- Unit tests for metrics helper ensuring label generation consistent and sanitized.
- Integration tests hitting health endpoint with multiple spaces populated (use test server) and asserting JSON schema via serde_json.
- SSE integration test subscribing to two spaces concurrently verifying isolation.
- Script test (bash `bats` or Rust integration) running diagnostics against test server and checking output formatting.
- Regression test verifying single-space deployment still produces metrics without label explosion.

## Review Agent
- `systems-architecture-optimizer` to validate metrics performance and labeling conventions.
