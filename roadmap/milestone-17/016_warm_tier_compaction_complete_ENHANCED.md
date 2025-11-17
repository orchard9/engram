# Task 016: Warm Tier Content Storage Compaction – ENHANCED

**Status:** Pending
**Priority:** Critical
**Dependencies:** Task 005 (Binding Formation)

## Objective

Implement a safe compaction mechanism for warm-tier content storage that avoids data corruption, limits memory overhead, and keeps read latency acceptable. The current design has unresolved race conditions; this task revises the approach and documents what needs to be built/tested before declaring compaction production-ready.

## Current Issues

Margo Seltzer’s architecture review identified the following blockers:
1. Offset updates aren’t atomic; readers may observe inconsistent data.
2. Compaction doubles memory usage (old + new buffers) and risks OOM.
3. Compaction pauses the world for ~2 s, blocking warm-tier reads.
4. No rollback/resume strategy if compaction fails mid-way.
5. Lock ordering unspecified, risking deadlocks.
6. Startup compaction/fragmentation not addressed.

## Required Changes

- Replace the stop-the-world approach with a versioned/double-buffered compaction that keeps one buffer readable while another is rebuilt, then atomically swaps and cleans up.
- Introduce a transactional update for offsets (e.g., staged remap table, swap only when all offsets updated).
- Limit memory spikes (chunked compaction or memory-mapped temp files) and enforce thresholds (abort if insufficient memory).
- Add error handling, checkpoints, and retries; compaction must be all-or-nothing.
- Document and enforce lock ordering to avoid deadlocks.
- Define when to trigger compaction (runtime + startup).
- Update metrics, tests, and runbooks accordingly.

## Deliverables

1. Revised design (double-buffering or chunked compaction) implemented in `MappedWarmStorage`.
2. Automated tests covering concurrent reads during compaction, failure cases, and startup compaction.
3. Documentation (code comments + ops runbook) describing the compaction process, triggers, and monitoring.
4. Metrics for compaction duration, bytes reclaimed, failures, and pause time.

## Acceptance Criteria

- Compaction can run without blocking readers for more than 100 ms.
- Memory overhead stays within configurable limits (no uncontrolled 2× spikes).
- Failure mid-compaction leaves data intact and compaction can resume/retry.
- Tests demonstrate data integrity and concurrency safety.
