# 007: Multi-Tenant Validation & Fuzzing — _75_percent_complete_

## Current Status: 75% Complete

**What's Implemented**:
- ✅ Comprehensive HTTP integration tests (engram-core/tests/multi_space_isolation.rs, 434 lines)
  - Same-ID isolation across spaces
  - Runtime guard verification
  - Concurrent operations across 5 spaces
  - Default space backward compatibility
  - Space handle metadata tracking
- ✅ Directory creation and isolation tests
- ✅ Test infrastructure for multi-space scenarios
- ✅ Concurrent stress testing (5 simultaneous spaces)

**Missing** (25%):
- ❌ gRPC multi-space integration tests
- ❌ Property-based fuzzing for registry concurrency
- ❌ Crash recovery scenario tests
- ❌ Performance benchmarks measuring multi-space overhead
- ❌ CI integration/documentation

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

1. ✅ **COMPLETE**: Integration tests fail when cross-space leak occurs
   - Implementation: 5 isolation tests in multi_space_isolation.rs
   - Coverage: Same-ID isolation, runtime guards, concurrent operations
   - Validation: Tests would catch deliberate cross-space data leakage
   - Status: All 5 tests passing

2. ❌ **NOT STARTED**: Fuzz/property tests execute in CI
   - Missing: Property-based tests for registry concurrency using proptest
   - Missing: Fuzz harness under engram-core/fuzz for multi_space_routing
   - Need: Document seeds and regression cases

3. ❌ **NOT STARTED**: Persistence recovery test confirms isolated WAL replay
   - Missing: Crash simulation test (kill server mid-transaction)
   - Missing: Directory layout validation after recovery
   - Missing: Per-space WAL replay isolation verification

4. ❌ **NOT STARTED**: Benchmark results recorded
   - Missing: Criterion benchmarks comparing 1/5/10 space overhead
   - Missing: Baseline vs multi-space latency measurements
   - Missing: Milestone summary data points

5. ⚠️ **PARTIAL**: CI includes new tests
   - ✅ Tests run via cargo test --workspace
   - ❌ Missing: Explicit CI job or feature flag documentation
   - ❌ Missing: Runtime budget validation

## Remaining Work

1. **gRPC Multi-Space Integration Tests** (4 hours)
   - File: Create `tests/multi_space_grpc.rs`
   - Scenarios:
     - remember/recall with explicit space field
     - Default space fallback behavior
     - Stream isolation (concurrent subscriptions)
     - Unknown space error handling
   - Infrastructure: tonic test client setup

2. **Property-Based Concurrency Tests** (3 hours)
   - File: Add to `engram-core/tests/property_tests.rs`
   - Framework: Use proptest
   - Strategy: Generate random sequences of create_or_get operations
   - Validate: Registry handles concurrent access correctly
   - Assertions: No panics, consistent handle returns

3. **Crash Recovery Tests** (4 hours)
   - File: Add to `engram-core/tests/multi_space_persistence.rs`
   - Scenario:
     ```rust
     // 1. Start server, write to alpha and beta
     // 2. Abruptly terminate (drop runtime without flush)
     // 3. Restart with recovery enabled
     // 4. Verify alpha data recovered, beta data recovered
     // 5. Verify no cross-space contamination
     ```
   - Validation: WAL replay isolation

4. **Performance Benchmarks** (3 hours)
   - File: `engram-core/benches/multi_space_overhead.rs`
   - Framework: Criterion
   - Scenarios:
     - Recall latency: 1 space baseline
     - Recall latency: 5 spaces
     - Recall latency: 10 spaces
   - Metrics: P50, P99 latency, throughput
   - Document: Overhead percentage in milestone summary

5. **Fuzzing Harness** (3 hours)
   - Directory: `engram-core/fuzz/`
   - Target: multi_space_routing.rs
   - Strategy: Random store/recall operations across spaces
   - Validation: Returned IDs match requested space
   - Integration: cargo fuzz or cargo test with feature flag

6. **CI Documentation** (1 hour)
   - File: Update root README or `.github/workflows/` comments
   - Content: Document multi-space test execution
   - Runtime: Validate tests complete within budget (<60s)
   - Nightly: Consider moving fuzz tests to nightly job

## Testing Strategy
- Graph acceptance scenarios via `graph-systems-acceptance-tester` to verify cognitive operations across spaces.
- Differential testing comparing store/recall outputs between spaces using recorded fixtures.
- Fuzz harness integrated into `cargo fuzz` or `cargo test` with feature flag to keep runtime manageable.

## Review Agent
- `verification-testing-lead` (primary) and `graph-systems-acceptance-tester` (secondary) before marking complete.
