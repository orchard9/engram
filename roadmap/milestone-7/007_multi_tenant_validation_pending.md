# 007: Multi-Tenant Validation & Fuzzing — _pending_

## Goal
Build a validation suite that exercises memory spaces under concurrent load, ensures isolation across store/recall/persistence paths, and catches regressions before release. Coverage must span HTTP, gRPC, persistence recovery, and activation workflows.

## Deliverables
- Shared integration harness that can spin up the CLI/server with configurable spaces and execute REST/gRPC scenarios.
- End-to-end tests that create multiple spaces, perform concurrent operations, and assert strict isolation (no cross-space recall, persistence, or SSE leakage).
- Fuzz/property tests targeting registry/store routing to detect race conditions.
- Crash recovery scenario validating WAL replay and consolidation per space.
- Benchmark scripts measuring latency overhead introduced by multi-space lookups.
- CI wiring to run the new suite (or nightly job) with documented runtime budget.

## Implementation Plan

1. **Test Infrastructure**
   - Create `tests/common/mod.rs` with helpers for launching Engram server in temp directory specifying space bootstrap list, returning base URLs and cleanup handles.
   - Provide wrappers for REST (reqwest) and gRPC (tonic) clients that accept `MemorySpaceId` and set headers/fields accordingly.

2. **Integration Tests**
   - Add `tests/multi_space_http.rs` covering:
     - Create spaces `alpha`, `beta`.
     - Store memories in each via REST; assert recall only returns matching space data.
     - Attempt cross-space recall, expect 404/empty response.
   - Add `tests/multi_space_grpc.rs` verifying remember/recall/stream flows with explicit and default spaces, including fallback warning check.
   - Add SSE test ensuring event streams for `alpha` do not include `beta` data (subscribe concurrently).

3. **Persistence & Recovery**
   - Integration test `tests/multi_space_persistence.rs` to:
     - Start server, write data in two spaces.
     - Stop server abruptly (kill process or drop handle) and rerun with recovery; assert persisted data stays isolated.
     - Validate directory layout using helper from Task 003.

4. **Fuzzing / Property Tests**
   - Use `engram-core/fuzz` to add scenario `multi_space_routing` that randomly issues store/recall operations across spaces, ensuring returned IDs match requested space.
   - Add property-based tests using `proptest` under `engram-core/tests` to stress registry concurrency (create/get space from multiple threads).

5. **Benchmarks**
   - Extend existing Criterion benchmarks (e.g., `engram-core/benches/`) with variant measuring recall latency with 1, 5, 10 spaces.
   - Document baseline vs multi-space overhead for milestone report.

6. **CI Integration**
   - Update GitHub Actions or local `Makefile` to run new tests (perhaps behind `cargo test --features multi-space-tests`).
   - Ensure runtime under target (documented in README of tests folder).

7. **Reporting**
   - Produce summary artifact (JSON/log) with isolation assertions to integrate into release checklist.

## Integration Points
- `tests/` in root and crate-level (HTTP/gRPC). Use existing test harness patterns for CLI launch.
- `engram-core/fuzz` for fuzz harness updates.
- `engram-core/benches` for performance measurement.
- CI configuration files (`.github/workflows/*` or `Makefile`).

## Acceptance Criteria
1. Integration tests fail when cross-space leak occurs (validated by deliberate regression during review).
2. Fuzz/property tests execute in CI and catch simulated race conditions (document seeds/regressions).
3. Persistence recovery test confirms isolated WAL replay and directory layout checks.
4. Benchmark results recorded and shared with milestone summary (≥1 data point).
5. CI includes new tests or documents nightly job with clear instructions.

## Testing Strategy
- Graph acceptance scenarios via `graph-systems-acceptance-tester` to verify cognitive operations across spaces.
- Differential testing comparing store/recall outputs between spaces using recorded fixtures.
- Fuzz harness integrated into `cargo fuzz` or `cargo test` with feature flag to keep runtime manageable.

## Review Agent
- `verification-testing-lead` (primary) and `graph-systems-acceptance-tester` (secondary) before marking complete.
