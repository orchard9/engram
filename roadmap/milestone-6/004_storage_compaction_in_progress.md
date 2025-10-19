# Task 004: Storage Compaction

## Status
IN_PROGRESS

## Priority
P0 (Critical Path)

## Effort Estimate
2 days

## Dependencies
- Task 003

## Objective
See MILESTONE_5_6_ROADMAP.md for complete specification.

## Technical Approach
Complete technical specification available in MILESTONE_5_6_ROADMAP.md (Milestone 6, Task 004).
Refer to main roadmap document for detailed implementation plan, code examples, integration points, and acceptance criteria.

## Acceptance Criteria
See MILESTONE_5_6_ROADMAP.md for detailed acceptance criteria.

## Testing Approach
See MILESTONE_5_6_ROADMAP.md for comprehensive testing strategy.

## Current Progress
- **NEW (2025-10-19)**: Storage compaction module implemented in `engram-core/src/consolidation/compaction.rs`
  - `StorageCompactor` struct with configurable compaction behavior
  - `CompactionConfig` with min_episode_age (7 days default), preserve_threshold (0.95), and reconstruction_threshold (0.8)
  - `verify_reconstruction()` validates semantic memory can reconstruct episodes with ≥0.8 similarity before deletion
  - `is_episode_eligible()` checks both age and confidence criteria to protect high-value memories
  - `compute_storage_reduction()` calculates storage savings from replacing episodes with semantic patterns
  - `CompactionResult` tracks episodes_removed, storage_reduction_bytes, and average_similarity
- **NEW (2025-10-19)**: Comprehensive test coverage with 8 unit tests
  - Embedding similarity computation (identical and orthogonal cases)
  - Reconstruction verification (success and failure paths)
  - Episode eligibility filtering (by age and confidence)
  - Storage reduction computation and ratio calculation
  - All tests passing with zero clippy warnings

## Next Checkpoints
- Implement full 5-phase `compact_storage()` method (currently only Phase 1 verification is implemented)
- Add MemoryStore integration methods: `store_semantic()`, `mark_consolidated()`, `remove_consolidated_episodes()`
- Create integration tests in `engram-core/tests/storage_compaction_tests.rs` for end-to-end compaction pipeline
- Add compaction metrics tracking (success rate, storage savings, rollback frequency)
- Coordinate with Task 006 to emit compaction events as observability signals

## Notes
- Design should integrate with `ConsolidationService` so compaction operates on cached snapshots rather than rebuilding from scratch.
- Use soak baseline (`docs/assets/consolidation/baseline/`) to validate compaction does not regress belief metrics.
- Coordinate with Task 006 to ensure compaction events emit observability signals (metrics + belief-update logs).
- Currently using existing `SemanticPattern` type from `completion/consolidation.rs` as the semantic memory representation.
