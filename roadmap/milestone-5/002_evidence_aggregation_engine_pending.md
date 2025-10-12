# Task 002: Evidence Aggregation Engine

## Status
PENDING

## Priority
P0 (Critical Path)

## Effort Estimate
2 days

## Dependencies
- Task 001: Query Executor Core

## Objective
Implement lock-free evidence combination with circular dependency detection and Bayesian updating.

## Technical Approach
See MILESTONE_5_6_ROADMAP.md lines 35-78 for complete specification.

### Key Files
- Create: `engram-core/src/query/evidence_aggregator.rs`
- Create: `engram-core/src/query/dependency_graph.rs`
- Modify: `engram-core/src/query/mod.rs`

### Implementation Details
- `LockFreeEvidenceAggregator` with crossbeam-epoch for memory management
- `CircularDependencyDetector` using Tarjan's algorithm
- `BayesianCombiner` applying Bayes' theorem to evidence chains
- Cache-line aligned evidence nodes (64-byte alignment)

## Acceptance Criteria
- [ ] Lock-free evidence aggregation with zero allocations in hot path
- [ ] Circular dependency detection prevents infinite loops
- [ ] Bayesian updating matches analytical solutions within 1% error
- [ ] Sub-100μs latency for aggregating 10 evidence sources
- [ ] Property tests verify associativity and commutativity
- [ ] Integration tests with spreading activation evidence

## Testing Approach
Create `engram-core/tests/evidence_aggregation_tests.rs` with property tests and cycle detection tests.

## Notes
Complete technical specification in MILESTONE_5_6_ROADMAP.md
