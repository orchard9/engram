# Task 005: Activation Spreading Matrix Multiply Kernel

**Status**: Pending
**Estimated Duration**: 3 days
**Priority**: High (second hottest operation)
**Owner**: Graph Algorithm Engineer

## Objective

GPU-accelerate activation propagation through graph edges using sparse matrix multiplication. This is the second most CPU-intensive operation after cosine similarity.

## Deliverables

1. Sparse matrix multiply kernel (CSR format)
2. Warp-level reduction for node neighborhoods
3. Integration with ParallelSpreadingEngine
4. Performance benchmarks vs CPU parallel spreading

## Technical Specification

See MILESTONE_12_IMPLEMENTATION_SPEC.md "Kernel 2: Activation Spreading Matrix Multiply" for:
- CSR sparse matrix format layout
- Thread configuration and warp optimization
- Integration with existing spreading engine

## Acceptance Criteria

- [ ] Achieves >5x speedup over CPU for graphs >512 nodes
- [ ] Correctly handles sparse graphs (average degree <10)
- [ ] Maintains confidence score precision (<1e-6 divergence)
- [ ] Graceful fallback to CPU for small graphs

## Dependencies

- Task 004 (unified memory for graph data) - BLOCKING
