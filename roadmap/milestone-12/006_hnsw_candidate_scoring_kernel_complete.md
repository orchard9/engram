# Task 006: HNSW Candidate Scoring Kernel

**Status**: Pending
**Estimated Duration**: 2 days
**Priority**: Medium (accelerates vector index operations)
**Owner**: Index Optimization Engineer

## Objective

Accelerate HNSW candidate evaluation during vector similarity search using GPU batch distance computation and warp-level top-k selection.

## Deliverables

1. Batch distance computation for HNSW candidates
2. Warp-level top-k selection (replaces CPU sort)
3. Integration with HnswIndex
4. Performance benchmarks vs CPU HNSW search

## Technical Specification

See MILESTONE_12_IMPLEMENTATION_SPEC.md "Kernel 3: HNSW Candidate Scoring" for:
- Thread configuration and shared memory tiling
- Top-k selection using warp primitives
- Integration with existing HNSW index

## Acceptance Criteria

- [ ] Achieves >4x speedup for candidate sets >1024 vectors
- [ ] Top-k results identical to CPU implementation
- [ ] Maintains HNSW recall accuracy (no degradation)
- [ ] Works with L2 distance and cosine similarity

## Dependencies

- Task 004 (unified memory for embeddings) - BLOCKING
