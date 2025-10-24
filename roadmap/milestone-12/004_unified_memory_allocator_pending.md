# Task 004: Unified Memory Allocator

**Status**: Pending
**Estimated Duration**: 3 days
**Priority**: Critical (enables zero-copy GPU operations)
**Owner**: Memory Systems Engineer

## Objective

Implement zero-copy memory management using CUDA Unified Memory, with automatic prefetching and graceful fallback to pinned memory for older GPU architectures. This eliminates explicit CPU-GPU memory transfers from hot paths.

## Deliverables

1. Unified memory allocation pool with RAII wrappers
2. Memory advise hints for CPU/GPU locality optimization
3. Prefetch automation based on access patterns
4. Fallback to pinned memory for non-unified systems

## Technical Specification

See MILESTONE_12_IMPLEMENTATION_SPEC.md "Unified Memory Strategy" section for:
- Allocation model and memory pool design
- Memory advise hints and prefetch strategy
- Fallback mechanisms for older GPUs
- OOM prevention techniques

## Acceptance Criteria

- [ ] Zero explicit cudaMemcpy calls in hot paths
- [ ] Automatic prefetching hides 80% of transfer latency
- [ ] Works on Pascal+ (unified) and Maxwell (pinned fallback)
- [ ] OOM prevention via batch size adaptation

## Dependencies

- Task 003 (first kernel operational) - BLOCKING
