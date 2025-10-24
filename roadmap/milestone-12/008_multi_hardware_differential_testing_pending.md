# Task 008: Multi-Hardware Differential Testing

**Status**: Pending
**Estimated Duration**: 2 days
**Priority**: Critical (validates correctness across GPUs)
**Owner**: Verification Engineer

## Objective

Validate GPU kernel correctness and numerical stability across diverse GPU architectures: Maxwell, Pascal, Ampere, and Hopper generations.

## Deliverables

1. Test suite running on multiple GPU architectures
2. Numerical stability validation across generations
3. Performance regression tests per GPU type
4. CI integration for GPU testing

## Technical Specification

Test matrix:
- Maxwell (GTX 1060): No Tensor Cores, no Unified Memory
- Pascal (GTX 1080): Unified Memory, no Tensor Cores
- Ampere (RTX 3060): Tensor Cores, FP32/FP16 mix
- Hopper (H100): Advanced Tensor Cores

## Acceptance Criteria

- [ ] All tests pass on all GPU generations
- [ ] CPU-GPU divergence <1e-6 on all architectures
- [ ] Performance increases with newer generations
- [ ] Older GPUs gracefully degrade (FP32 instead of FP16)

## Dependencies

- Task 007 (hybrid executor complete) - BLOCKING
