# Task 003: Batch Cosine Similarity CUDA Kernel

**Status**: Pending
**Estimated Duration**: 3 days
**Priority**: Critical (first production GPU kernel)
**Owner**: GPU Performance Engineer

## Objective

Implement and optimize the first production CUDA kernel for batch cosine similarity computation between a query vector and multiple target vectors. This kernel must achieve >3x speedup over CPU AVX-512 implementation while maintaining numerical equivalence.

## Background

Cosine similarity batch computation is the hottest operation in Engram's recall path, consuming ~60% of CPU time in production workloads (per Task 001 profiling). This is the ideal first GPU kernel: purely compute-bound, embarrassingly parallel, and well-understood.

## Research Foundation

Cosine similarity is embarrassingly parallel - each similarity(Q, Ti) computation is independent. This is archetypal memory-bound: 768-dim vectors = 3KB per vector (768 floats × 4 bytes), computation is 768 multiplies + 767 adds + 1 division + 2 square roots, arithmetic intensity only 0.13 FLOPS/byte.

**Warp-level optimization rationale:**
CUDA thread hierarchy: Thread (1 CUDA core) → Warp (32 threads, SIMT execution) → Block (128-1024 threads) → Grid (unlimited blocks). The warp is the fundamental unit - all 32 threads execute the same instruction simultaneously.

Naive approach: each thread computes entire dot product sequentially (no parallelism within dot product). Warp-optimized: distribute 768 dimensions across 32 threads (each computes 768/32 = 24 multiplications), then cooperate via warp shuffle to sum results - NO shared memory needed, threads exchange data directly.

**Warp shuffle instructions** (`__shfl_down_sync`) are faster than shared memory (no memory access latency). Traditional reduction requires shared memory writes/reads; shuffle uses register-to-register transfers within warp.

**Numerical stability:**
- GPU reduction order must match CPU to ensure confidence score equivalence
- Force IEEE 754 rounding (no fast-math) to prevent divergence
- Use Kahan summation for dot products >1024 dimensions (prevent accumulation drift)

**Performance model:**
- CPU AVX-512: 2.1 μs/vector
- GPU target: 0.3 μs/vector (7x speedup)
- Break-even: 64 vectors (accounting for 10μs kernel launch overhead)
- Block size: 256 threads (8 warps for memory coalescing)
- Memory access: query cached in constant memory, targets coalesced 128-byte aligned reads

## Deliverables

1. CUDA kernel implementation with warp-level optimization
2. Rust FFI wrapper with CPU fallback
3. Performance benchmarks vs CPU SIMD (AVX-512, AVX2, NEON)
4. Differential testing ensuring <1e-6 divergence from CPU

## Technical Specification

See MILESTONE_12_IMPLEMENTATION_SPEC.md "Kernel 1: Batch Cosine Similarity" for complete specification including:
- Memory layout and thread configuration
- Warp-level reduction optimization
- Tensor Core utilization for Ampere+
- Shared memory usage patterns

## Acceptance Criteria

- [ ] Achieves >3x speedup over AVX-512 for batches >64 vectors
- [ ] CPU-GPU result divergence <1e-6 for all test vectors
- [ ] Gracefully falls back to CPU if GPU unavailable
- [ ] Handles batch sizes from 1 to 100,000 correctly

## Files to Create

- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/cuda/kernels/cosine_similarity.cu`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/compute/cuda/cosine_similarity.rs`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/benches/gpu_cosine_similarity.rs`

## Dependencies

- Task 002 (CUDA environment setup) - BLOCKING

## Testing Approach

- Differential testing against scalar CPU implementation
- Property-based testing with random vectors
- Benchmark against cuBLAS for validation
- Test on GTX 1060, RTX 3060, A100
