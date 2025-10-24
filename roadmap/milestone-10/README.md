# Milestone 10: Zig Performance Kernels

**Duration:** 23 days
**Status:** Pending
**Dependencies:** Milestone 9 (Consolidation & Pattern Completion)

## Overview

This milestone introduces Zig as a performance acceleration layer for compute-intensive operations in Engram. Rather than replacing Rust's safe graph engine, Zig provides targeted kernels for hot-path operations that demand maximum performance: vector similarity calculations, activation spreading, and memory decay functions.

The implementation follows a rigorous differential testing approach, ensuring that Zig kernels produce identical results to their Rust counterparts while delivering measurable performance improvements. A comprehensive profiling and regression testing framework validates that optimizations remain stable across future changes.

## Architectural Principles

1. **Rust remains the primary language** - All graph data structures, concurrency primitives, and API surfaces stay in Rust
2. **Zig kernels for compute-bound operations** - Vector operations, activation spreading, decay calculations
3. **Zero-copy FFI boundaries** - Pass pointers to pre-allocated buffers, avoid serialization overhead
4. **Differential testing guarantees correctness** - Every Zig kernel must produce identical output to Rust baseline
5. **Performance regression detection** - Automated benchmarks fail the build if kernels regress >5%
6. **Graceful fallback** - Runtime detection falls back to Rust if Zig kernels unavailable

## Success Criteria

1. Zig build system integrated with cargo build workflow
2. Three production kernels (vector similarity, spreading, decay) with differential tests
3. Performance improvements: 15-25% on vector similarity, 20-35% on spreading activation
4. Zero correctness regressions - all differential tests pass
5. Performance regression framework integrated into CI
6. Complete documentation of Zig integration and rollback procedures

## Task Breakdown

### Phase 1: Foundation (5 days)
- **001_profiling_infrastructure_pending.md** (2 days) - Flamegraph profiling, hotspot identification
- **002_zig_build_system_pending.md** (1 day) - build.zig, cargo integration, FFI bindings
- **003_differential_testing_harness_pending.md** (2 days) - Property-based testing framework

### Phase 2: Core Kernels (10 days)
- **004_memory_pool_allocator_pending.md** (2 days) - Arena-style allocator for kernel scratch space
- **005_vector_similarity_kernel_pending.md** (3 days) - Cosine similarity with SIMD
- **006_activation_spreading_kernel_pending.md** (3 days) - BFS activation with edge batching
- **007_decay_function_kernel_pending.md** (2 days) - Optimized Ebbinghaus decay

### Phase 3: Integration & Validation (8 days)
- **008_arena_allocator_pending.md** (2 days) - Thread-local memory pools
- **009_integration_testing_pending.md** (2 days) - End-to-end system tests
- **010_performance_regression_framework_pending.md** (2 days) - Automated benchmarking in CI
- **011_documentation_and_rollback_pending.md** (1 day) - Deployment guides, rollback procedures
- **012_final_validation_pending.md** (1 day) - UAT, performance verification, sign-off

## Technical Specifications

### FFI Boundary Design

All Zig kernels use C-compatible ABI with explicit memory ownership:

```zig
// Vector similarity kernel signature
export fn engram_vector_similarity(
    query: [*]const f32,
    candidates: [*]const f32,
    scores: [*]f32,
    query_len: usize,
    num_candidates: usize,
) void;
```

Rust caller manages allocation:
```rust
let mut scores = vec![0.0_f32; candidates.len()];
unsafe {
    engram_vector_similarity(
        query.as_ptr(),
        candidates.as_ptr(),
        scores.as_mut_ptr(),
        query.len(),
        candidates.len(),
    );
}
```

### Performance Targets

| Operation | Baseline (Rust) | Target (Zig) | Improvement |
|-----------|----------------|--------------|-------------|
| Vector Similarity (768-dim) | 2.3 us | 1.7 us | 25% |
| Spreading Activation (1000 nodes) | 145 us | 95 us | 35% |
| Decay Calculation (10k memories) | 89 us | 65 us | 27% |

### Memory Consistency Model

Zig kernels operate on immutable input and mutable output buffers with the following guarantees:

1. **No aliasing** - Input and output pointers never overlap
2. **Synchronous execution** - Kernels return only when computation complete
3. **Thread-safety** - Kernels are pure functions with no shared state
4. **Allocation separation** - Kernels use caller-provided buffers or internal arena allocators

## Dependencies

- **Milestone 9 complete** - Consolidation and pattern completion algorithms finalized
- **Zig 0.13.0** - Stable release with C ABI guarantees
- **cargo-criterion** - Benchmarking framework for regression detection
- **proptest** - Property-based testing for differential validation

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Zig compiler instability | High | Pin to Zig 0.13.0 stable, avoid nightly features |
| FFI boundary bugs | Critical | Extensive fuzzing with arbitrary inputs, memory sanitizers |
| Performance regressions in Rust | Medium | Maintain Rust fast paths, runtime feature detection |
| Build complexity | Low | Document build.zig thoroughly, provide fallback to Rust-only builds |

## Out of Scope

- Rewriting graph engine core in Zig (stays in Rust)
- GPU acceleration (deferred to Milestone 14)
- Cross-compilation for embedded targets (future work)
- WASM bindings for Zig kernels (deferred to Milestone 16)

## Deliverables

1. `/zig/src/kernels/` - Vector similarity, spreading, decay implementations
2. `/zig/build.zig` - Build system with cargo integration
3. `/tests/zig_differential/` - Differential test suite with 95%+ coverage
4. `/benches/zig_regression/` - Performance regression benchmarks
5. `/docs/operations/zig_performance_kernels.md` - Deployment and rollback guide
6. Performance report documenting improvements and regression tests

## Acceptance Criteria

1. All differential tests pass with zero correctness regressions
2. Performance benchmarks show 15-35% improvement on target operations
3. CI pipeline includes Zig build and regression tests
4. Documentation covers deployment, monitoring, and rollback procedures
5. make quality passes with zero clippy warnings
6. UAT validates production workload performance improvements
