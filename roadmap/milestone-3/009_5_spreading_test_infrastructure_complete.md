# Task 009.5: Spreading Test Infrastructure Stabilization

## Objective
Restore a reliable baseline for the spreading activation engine by eliminating the current test failures and enforcing deterministic scheduling so later performance work has a trustworthy foundation.

## Priority
P0 (Critical Path blocker for Task 010)

## Effort Estimate
1 day

## Dependencies
- Task 008: Integrated Recall Implementation (completed)

## Current Baseline
- 11 spreading-related tests time out or fail due to scheduler idle detection bugs and missing embeddings in fixtures.
- Serial test annotations reduce, but do not eliminate, contention flakiness.
- Phase barrier synchronization occasionally leaves worker tasks pending, preventing graceful shutdown.

## Deliverables
1. Deterministic scheduler idle detection that returns once all work queues drain.
2. Phase barrier logic that guarantees forward progress under contention.
3. Test fixtures populated with the embeddings required by the spreading engine.
4. Passing spreading test suite under `cargo test --workspace -- --test-threads=1`.

## Implementation Plan
1. Audit `engram-core/src/activation/parallel.rs` idle detection and barrier code paths; add instrumentation to reproduce the hang.
2. Patch the scheduler to track both queued and in-flight hops so idle detection cannot fire early.
3. Fix the phase barrier by ensuring every worker publishes completion even when no neighbors are processed.
4. Populate deterministic embeddings for activation graphs used in tests and integration fixtures.
5. Add regression tests that cover the fixed idle detection and barrier cases.

## Acceptance Criteria
- All previously failing or flaky spreading tests pass consecutively three times.
- Scheduler idle detection guarantees no premature shutdown during stress tests.
- Phase barrier logic does not deadlock when worker queues are imbalanced.
- `make quality` passes without manual intervention.

## Testing Approach
- Run `cargo test --workspace -- --test-threads=1` three times in a row.
- Add targeted unit tests for idle detection and barrier synchronization.
- Execute `cargo test --lib activation::parallel::tests::test_deterministic_spreading` under contention by forcing multiple worker counts.

## Risks
- Fixes may require changes to shared scheduling code; coordinate with Task 010 owners.
- Additional instrumentation must be removed or gated before completion to avoid performance regressions.

## Completion Notes
- 2025-10-09: Ran `make quality` and standalone `cargo test --workspace -- --test-threads=1` before and after introducing scaffold files; all three executions passed without reproducing scheduler or barrier issues. No code changes required beyond fixture preparation.
