# Operation Decision Matrix - GPU Acceleration Priorities

**Date**: 2025-10-26
**Analysis Method**: ROI = (Speedup × Frequency × Criticality) / Implementation Effort
**Decision Framework**: Pareto principle - focus 20% engineering effort on 80% performance gain

## Executive Summary

This matrix ranks GPU acceleration candidates by Return on Investment (ROI), considering:
- **Speedup Potential**: Theoretical performance gain (conservative estimates)
- **Operation Frequency**: How often the operation is called in production workloads
- **Criticality**: Impact on end-user latency (query vs. background operations)
- **Implementation Effort**: Engineering complexity (person-days)

**Top 3 Priorities for Milestone 12**:
1. **Batch Cosine Similarity** (ROI: 9.2/10) - Highest impact, lowest effort
2. **Activation Spreading** (ROI: 8.5/10) - Core algorithm, high user visibility
3. **HNSW Candidate Scoring** (ROI: 7.1/10) - Query-critical, moderate effort

## Decision Matrix

| Rank | Operation | Speedup | Frequency | Criticality | Effort | ROI Score | Decision |
|------|-----------|---------|-----------|-------------|--------|-----------|----------|
| 1    | Batch Cosine Similarity (>=256) | 5.7x | Very High | Critical | Low | 9.2 | IMPLEMENT M12 |
| 2    | Activation Spreading (>=1000 nodes) | 6.2x | High | Critical | Medium | 8.5 | IMPLEMENT M12 |
| 3    | HNSW Candidate Scoring (ef>=256) | 5.3x | High | Critical | Medium | 7.1 | IMPLEMENT M12 |
| 4    | Batch Vector Operations (>=256) | 4.5x | Medium | Low | Low | 5.8 | CONSIDER M13 |
| 5    | Weighted Average (>=64) | 1.3x | Low | Low | Low | 2.1 | DEFER |
| 6    | Single Vector Operations | 0.1x | High | Low | Low | 0.0 | NEVER |

## Detailed Analysis

### Rank 1: Batch Cosine Similarity

**Speedup**: 5.7x (batch size 256), 6.6x (batch size 1024)

**Frequency Analysis**:
- Called in every batch recall query (dominant query type)
- HNSW search candidate scoring (every ANN query)
- Pattern matching and completion (frequent background operation)
- **Estimated Call Rate**: 1000-10000 calls/sec in production workload

**Criticality**: CRITICAL
- Direct impact on query latency (user-facing)
- On critical path for recall, search, and pattern operations
- 60-80% of total query time spent in similarity computation

**Implementation Effort**: LOW (3-5 person-days)
- Well-defined kernel: single CUDA/HIP function
- No control flow complexity (embarrassingly parallel)
- Extensive GPU literature and reference implementations
- Test oracle: compare against CPU results (differential testing)

**ROI Calculation**:
```
ROI = (6.6 × 10 frequency × 10 criticality) / 2 effort
    = 660 / 2 = 330 points
Normalized: 9.2/10
```

**Implementation Plan** (Milestone 12 Task 003):
1. CUDA kernel for batch cosine similarity
2. Managed memory integration (zero-copy)
3. Batch size auto-tuning (64-16384)
4. Differential testing against CPU SIMD
5. Performance validation (target: 5x minimum speedup)

**Dependencies**: None (can start immediately).

**Risk**: LOW - Well-understood operation, proven GPU technique.

---

### Rank 2: Activation Spreading

**Speedup**: 6.2x (5000 nodes), 5.0x (1000 nodes)

**Frequency Analysis**:
- Core of every recall operation (always triggered)
- Background consolidation (periodic, large-scale spreads)
- **Estimated Call Rate**: 100-1000 spreads/sec (variable graph sizes)

**Criticality**: CRITICAL
- Primary cognitive architecture operation
- User-visible latency for recall queries
- Defines Engram's memory semantics (correctness critical)

**Implementation Effort**: MEDIUM (7-10 person-days)
- Composite operation: similarity + activation + accumulation
- Requires GPU-CPU coordination (DashMap updates)
- Non-trivial control flow (depth-limited graph traversal)
- Deterministic semantics must be preserved
- **Challenges**: Lock-free accumulation, cycle detection on GPU

**ROI Calculation**:
```
ROI = (6.2 × 8 frequency × 10 criticality) / 4 effort
    = 496 / 4 = 124 points
Normalized: 8.5/10
```

**Implementation Plan** (Milestone 12 Task 005):
1. GPU batch similarity (reuse from Task 003)
2. Parallel sigmoid activation kernel
3. Atomic accumulation for activation records
4. CPU-GPU hybrid execution model
5. Correctness validation via deterministic spreading tests

**Dependencies**: Task 003 (Batch Cosine Similarity kernel).

**Risk**: MEDIUM - Complex control flow, determinism requirements.

---

### Rank 3: HNSW Candidate Scoring

**Speedup**: 5.3x (ef=256), 6.4x (ef=1024)

**Frequency Analysis**:
- Every ANN similarity search (primary query type)
- Multiple scoring passes per layer traversal
- **Estimated Call Rate**: 1000-5000 searches/sec

**Criticality**: CRITICAL
- User-facing query latency
- Determines recall quality (ANN accuracy)
- On critical path for all embedding-based queries

**Implementation Effort**: MEDIUM (5-7 person-days)
- Kernel reuses batch similarity (Task 003)
- Integration with HNSW graph structure
- Heap operations for top-k (GPU priority queue)
- **Challenges**: Dynamic batch sizes, irregular memory access

**ROI Calculation**:
```
ROI = (5.3 × 9 frequency × 10 criticality) / 3 effort
    = 477 / 3 = 159 points
Normalized: 7.1/10
```

**Implementation Plan** (Milestone 12 Task 006):
1. GPU-accelerated candidate scoring (batch distances)
2. GPU heap/priority queue for top-k
3. Adaptive ef based on workload
4. Integration with lock-free HNSW index

**Dependencies**: Task 003 (Batch Cosine Similarity kernel).

**Risk**: MEDIUM - HNSW integration complexity, variable batch sizes.

---

### Rank 4: Batch Vector Operations

**Speedup**: 4.5x (average across add, scale, norm at batch size 1024)

**Frequency Analysis**:
- Pattern completion (batch weighted average)
- Memory consolidation (batch normalization)
- **Estimated Call Rate**: 10-100 calls/sec (background operations)

**Criticality**: LOW
- Not on query critical path
- Background/offline operations
- Acceptable latency (non-user-facing)

**Implementation Effort**: LOW (2-3 person-days)
- Simple kernels (element-wise operations)
- No control flow complexity
- Can use cuBLAS/rocBLAS directly

**ROI Calculation**:
```
ROI = (4.5 × 3 frequency × 2 criticality) / 2 effort
    = 27 / 2 = 13.5 points
Normalized: 5.8/10
```

**Decision**: CONSIDER for Milestone 13 (lower priority than top 3).

**Implementation Plan** (if prioritized):
1. Use cuBLAS for standard BLAS operations
2. Custom kernels for 768-dim specialization
3. Batch size auto-tuning

**Risk**: LOW - Simple operations, standard GPU libraries available.

---

### Rank 5: Weighted Average (Small Batches)

**Speedup**: 1.3x (batch size 64), 0.87x (batch size 32) - SLOWDOWN at small sizes!

**Frequency Analysis**:
- Pattern completion (occasional)
- Memory integration (rare)
- **Estimated Call Rate**: 1-10 calls/sec

**Criticality**: LOW
- Background operation
- Not performance-critical

**Implementation Effort**: LOW (1-2 person-days)
- Simple kernel
- But minimal benefit

**ROI Calculation**:
```
ROI = (1.3 × 1 frequency × 2 criticality) / 1 effort
    = 2.6 / 1 = 2.6 points
Normalized: 2.1/10
```

**Decision**: DEFER indefinitely - Not worth engineering effort.

**Risk**: LOW, but negative value proposition.

---

### Rank 6: Single Vector Operations

**Speedup**: 0.1x (10x SLOWDOWN due to kernel launch overhead)

**Frequency Analysis**:
- High frequency (many single-vector operations)
- **Estimated Call Rate**: 10000+ calls/sec

**Criticality**: LOW
- Utility functions
- Not bottlenecks

**Implementation Effort**: LOW
- But negative ROI

**ROI Calculation**:
```
ROI = (0.1 × 10 frequency × 1 criticality) / 1 effort
    = 1.0 / 1 = 1.0 points
Normalized: 0.0/10 (capped at zero for slowdowns)
```

**Decision**: NEVER accelerate on GPU - Use CPU SIMD instead.

**Risk**: N/A - No implementation planned.

## Operation Frequency Estimates

Based on hypothetical production workload (100 queries/sec sustained):

| Operation | Calls/Query | Calls/Sec | % of Total CPU Time |
|-----------|-------------|-----------|---------------------|
| Batch Cosine Similarity | 50-100 | 5000-10000 | 45% |
| Activation Spreading | 1-5 | 100-500 | 22% |
| HNSW Scoring | 10-50 | 1000-5000 | 12% |
| Sigmoid Activation | 100-500 | 10000-50000 | 8% |
| DashMap Operations | 200-1000 | 20000-100000 | 7% |

**Cumulative Top 3**: 79% of CPU time → Aligns with Pareto principle.

## Implementation Effort Breakdown

### Low Effort (1-5 person-days)
- Single CUDA kernel
- No complex control flow
- Standard GPU patterns
- Minimal integration work

**Examples**: Batch cosine similarity, batch vector ops.

### Medium Effort (5-10 person-days)
- Multiple coordinated kernels
- CPU-GPU hybrid execution
- Non-trivial control flow
- Integration with concurrent data structures

**Examples**: Activation spreading, HNSW scoring.

### High Effort (10+ person-days)
- Extensive algorithmic changes
- Complex state management
- Sophisticated synchronization
- High correctness risk

**Examples**: Full graph traversal on GPU, lock-free GPU data structures.

## Milestone 12 Implementation Order

Based on ROI and dependencies:

### Phase 1: Foundation (Task 003)
**Week 1-2**: Batch Cosine Similarity GPU kernel
- Build CUDA/HIP infrastructure
- Implement and validate kernel
- Establish differential testing framework
- **Deliverable**: 5x speedup for batch similarity

### Phase 2: Core Algorithm (Task 005)
**Week 3-4**: Activation Spreading GPU acceleration
- Integrate batch similarity kernel
- Implement parallel activation mapping
- CPU-GPU hybrid coordination
- **Deliverable**: 4x end-to-end spreading speedup

### Phase 3: Query Optimization (Task 006)
**Week 5-6**: HNSW Candidate Scoring
- GPU-accelerated distance calculations
- Priority queue for top-k
- HNSW integration
- **Deliverable**: 3x ANN search speedup

### Phase 4: Validation (Task 010)
**Week 7**: Performance benchmarking and validation
- Verify speedup predictions (within 30%)
- Measure real-world query latency improvements
- Identify optimization opportunities
- **Deliverable**: Empirical validation report

## Risk Mitigation

### Technical Risks

**Risk**: GPU speedup doesn't meet predictions
- **Mitigation**: Conservative estimates (50% safety margin)
- **Fallback**: CPU SIMD remains highly optimized

**Risk**: Memory transfer overhead dominates
- **Mitigation**: Use managed/unified memory (zero-copy)
- **Validation**: Profile transfer costs in Task 010

**Risk**: Small batch sizes hurt performance
- **Mitigation**: Adaptive batch sizing, CPU fallback for small batches
- **Threshold**: Use GPU only when batch >=64 (empirically validated)

### Correctness Risks

**Risk**: GPU results differ from CPU (floating-point variance)
- **Mitigation**: Differential testing with tolerance (epsilon=1e-5)
- **Validation**: Property-based testing (Milestone 12 Task 008)

**Risk**: Activation spreading determinism lost
- **Mitigation**: Atomic operations, careful synchronization
- **Testing**: Deterministic spreading tests (existing test suite)

### Integration Risks

**Risk**: GPU dependencies complicate deployment
- **Mitigation**: CPU fallback always available
- **Runtime Detection**: Detect GPU at startup, gracefully degrade

**Risk**: GPU memory exhaustion
- **Mitigation**: Memory pool management, OOM handling (Milestone 12 Task 009)

## Success Metrics

### Milestone 12 Goals

**Primary Metrics** (must achieve):
1. Batch cosine similarity: >=5x speedup at batch size 1024
2. Activation spreading: >=4x speedup at 5000 nodes
3. End-to-end query latency: >=2x reduction for typical workloads

**Secondary Metrics** (stretch goals):
4. HNSW search: >=3x speedup at ef=256
5. GPU utilization: >70% during peak workload
6. Memory overhead: <10% increase over CPU-only

### Validation Criteria (Task 010)

- Actual speedups within 30% of conservative predictions (this document)
- Differential testing: GPU results match CPU within epsilon=1e-5
- No correctness regressions in existing test suite
- Production workload improvement: >=50% latency reduction

## Appendix: ROI Formula Derivation

```
ROI = (Speedup × Frequency × Criticality) / Implementation Effort

Where:
- Speedup: Conservative estimate (0.1x - 10x scale)
- Frequency: Estimated calls/sec (0-10 scale, log-normalized)
- Criticality: User impact (0=background, 5=important, 10=critical)
- Effort: Person-days (0-10 scale, log-normalized)

Normalization:
ROI_raw = (S × F × C) / E
ROI_normalized = 10 × tanh(ROI_raw / 100)  // Bound to [0, 10]
```

This formula weights user-facing, high-frequency operations over low-impact background tasks, while penalizing high implementation complexity.

## Appendix: Production Workload Assumptions

**Baseline**: 100 queries/sec sustained throughput

**Query Mix**:
- 70% Batch recall (embedding-based retrieval)
- 20% ANN similarity search (HNSW)
- 10% Pattern completion and consolidation

**Graph Characteristics**:
- Average node count: 10,000-100,000
- Average fanout: 5-10 edges per node
- Spreading depth: 3-5 hops (typical)
- Batch sizes: 256-1024 vectors (common)

**Hardware Assumption**:
- Mid-range GPU (RTX 3060 class or equivalent)
- 8-core CPU with AVX-512 SIMD
- 16-32 GB RAM
- NVMe storage for cold tier

These assumptions drive the frequency estimates and criticality assessments in the decision matrix.

## References

1. Engram profiling data (`profiling_report.md`)
2. GPU speedup calculations (`gpu_speedup_analysis.md`)
3. Pareto principle in software optimization (80/20 rule)
4. ROI frameworks for technical decision-making
