# Task 001: Query Executor Core

## Status
PENDING

## Priority
P0 (Critical Path)

## Effort Estimate
3 days

## Dependencies
- Milestone 3: ConfidenceAggregator
- Milestone 1: ProbabilisticQueryResult
- Milestone 0: Confidence type

## Objective
Implement `ProbabilisticQueryExecutor` that transforms recall results into probabilistic query results with evidence tracking and uncertainty propagation.

## Technical Approach
See MILESTONE_5_6_ROADMAP.md lines 20-33 for complete specification.

### Key Files
- Create: `engram-core/src/query/executor.rs`
- Modify: `engram-core/src/query/mod.rs`, `engram-core/src/store.rs`
- API: `engram-cli/src/api.rs` (add `/api/v1/query/probabilistic` endpoint)

### Integration Points
- Uses `ConfidenceAggregator` from `activation/confidence_aggregation.rs`
- Extends `ProbabilisticQueryResult` from `query/mod.rs`
- Integrates with `SpreadingMetrics` for evidence extraction

## Acceptance Criteria
- [ ] Query executor compiles and passes unit tests
- [ ] Evidence chain extraction captures all sources
- [ ] Confidence aggregation uses existing aggregator
- [ ] HTTP API endpoint functional
- [ ] Performance: <1ms P95 latency for 10-result queries
- [ ] Integration tests validate end-to-end flow
- [ ] Documentation complete with Rustdoc examples

## Testing Approach
Create `engram-core/tests/query_executor_tests.rs` with unit and integration tests.

## Notes
Complete technical specification in MILESTONE_5_6_ROADMAP.md
