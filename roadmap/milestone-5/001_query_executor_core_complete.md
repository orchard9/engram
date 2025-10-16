# Task 001: Query Executor Core

## Status
COMPLETE ✅

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
- [x] Query executor compiles and passes unit tests
- [x] Evidence chain extraction captures all sources
- [x] Confidence aggregation uses existing aggregator
- [x] HTTP API endpoint functional (GET /api/v1/query/probabilistic)
- [x] Performance: <1ms P95 latency for 10-result queries
- [x] Integration tests validate end-to-end flow (query_executor_tests.rs)
- [x] Documentation complete with Rustdoc examples

## Testing Approach
Create `engram-core/tests/query_executor_tests.rs` with unit and integration tests.

## Implementation Summary

### Files Created/Modified
1. **engram-core/src/query/executor.rs** ✅
   - `ProbabilisticQueryExecutor` with configuration options
   - Evidence chain tracking from spreading activation
   - Uncertainty source quantification
   - Integration with CalibrationTracker and UncertaintyTracker

2. **engram-core/src/query/integration.rs** ✅
   - `ProbabilisticRecall` trait implementation for `MemoryStore`
   - Full integration with evidence aggregator and uncertainty tracker
   - Confidence interval computation with variance estimation

3. **engram-cli/src/api.rs** ✅
   - `GET /api/v1/query/probabilistic` endpoint (lines 1071-1329)
   - `ProbabilisticQueryRequest` and `ProbabilisticQueryResponse` types
   - Evidence chain and uncertainty sources optional inclusion
   - Detailed evidence descriptions for all source types
   - Route registered at line 2745

4. **engram-core/tests/query_executor_tests.rs** ✅
   - 15 integration tests covering all functionality
   - End-to-end query execution validation
   - Evidence chain and uncertainty tracking tests
   - Configuration option tests

### API Endpoint Details

**Endpoint**: `GET /api/v1/query/probabilistic`

**Query Parameters**:
- `query` (optional): Text query string
- `embedding` (optional): 768-dim embedding vector as JSON array
- `max_results` (optional, default 10): Maximum number of results
- `include_evidence` (optional, default false): Include evidence chain
- `include_uncertainty` (optional, default false): Include uncertainty sources

**Response**:
```json
{
  "memories": [...],
  "confidence_interval": {
    "lower": 0.65,
    "upper": 0.85,
    "point": 0.75,
    "width": 0.20
  },
  "evidence_chain": [...],
  "uncertainty_sources": [...],
  "system_message": "Probabilistic query completed in 12ms..."
}
```

### Integration with Other Frameworks
- **Evidence Aggregator** (Task 002): Dependency-aware Bayesian aggregation
- **Uncertainty Tracker** (Task 003): System pressure and noise quantification
- **Confidence Calibration** (Task 004): Empirical confidence adjustment

### Quality Checks
- ✅ Zero clippy warnings with `-D warnings`
- ✅ All unit tests pass (15 tests in query_executor_tests.rs)
- ✅ Integration tests validate end-to-end flow
- ✅ Proper error handling and type safety
- ✅ OpenAPI documentation with utoipa

## Notes
Complete technical specification in MILESTONE_5_6_ROADMAP.md

**Production Status**: The `/api/v1/query/probabilistic` endpoint is now fully implemented and ready for production use. All Milestone 5 frameworks (Tasks 001-004) are now accessible via the HTTP API.
