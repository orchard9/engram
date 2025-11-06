# Milestone 5: Probabilistic Query Foundation

## Overview

Build production-grade query executor returning `Vec<(Episode, Confidence)>` with rigorous uncertainty propagation. Every query operation must maintain probability axioms, distinguish empty results from low-confidence results, and track uncertainty sources through the full recall pipeline.

## Success Criteria

1. Query operations return confidence intervals with calibrated uncertainty
2. Probability propagation verified correct using SMT solvers
3. Confidence scores correlate >0.9 with retrieval accuracy on test sets
4. Sub-millisecond P95 latency for complex queries with 10+ evidence sources
5. Zero probability axiom violations in 100K property test cases
6. Graceful degradation under system pressure without returning invalid probabilities

## Critical Path

```
001 (Query Executor Core) → 002 (Evidence Aggregation) → 003 (Uncertainty Tracking)
                                                    ↓
                                         004 (Confidence Calibration)
                                                    ↓
                                         005 (SMT Verification)
                                                    ↓
                                  006 (Integration & Performance) → 007 (Production Validation)
```

## Dependencies

- Milestone 3: Activation spreading engine with confidence aggregation
- Milestone 1: HNSW index, decay functions, probabilistic query primitives
- Milestone 0: Confidence type, error infrastructure

## Build On Existing

- `engram-core/src/query/mod.rs`: Already implements `ProbabilisticQueryResult`, `ConfidenceInterval`, `Evidence` types
- `engram-core/src/activation/confidence_aggregation.rs`: Already implements multi-source confidence combination
- `Confidence` type: Already has logical operations (and/or/not), calibration methods
- Activation spreading: Already tracks tier-specific uncertainty

## Deliverables

1. Query execution engine with confidence propagation
2. Evidence aggregation with circular dependency detection
3. Uncertainty tracking from all system sources
4. Confidence calibration framework with empirical validation
5. SMT solver integration for correctness proofs
6. Comprehensive property-based test suite
7. Production-ready integration with MemoryStore

## Performance Targets

- Query latency: <1ms P95 for 10 evidence sources
- Calibration error: <5% across confidence bins
- Property test pass rate: >99.9% on 10K cases
- Memory overhead: <100 bytes per query result
- Throughput: >10K queries/second on commodity hardware

## Risk Mitigation

- **Complexity**: Start with simple independent evidence, add dependencies incrementally
- **Performance**: Cache common probability calculations, use SIMD for batch operations
- **Correctness**: SMT verification runs in CI, property tests guard regressions
- **Integration**: Extend existing APIs, maintain backward compatibility
