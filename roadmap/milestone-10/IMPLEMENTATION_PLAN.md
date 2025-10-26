# Milestone 10 - Implementation Plan

**Milestone**: Zig Performance Kernels
**Duration**: 23 days
**Status**: Planning
**Date**: 2025-10-25

## Executive Summary

Milestone 10 integrates Zig performance kernels into Engram to accelerate compute-intensive operations by 15-35%. The implementation follows a 3-phase approach: Foundation (profiling and build system), Core Kernels (SIMD implementations), and Integration & Validation (testing and deployment readiness).

## Phase Overview

### Phase 1: Foundation (5 days)
- Task 001: Profiling Infrastructure
- Task 002: Zig Build System
- Task 003: Differential Testing Harness
- Task 004: Memory Pool Allocator

### Phase 2: Core Kernels (7 days)
- Task 005: Vector Similarity Kernel
- Task 006: Activation Spreading Kernel
- Task 007: Decay Function Kernel
- Task 008: Arena Allocator Enhancement

### Phase 3: Integration & Validation (6 days)
- Task 009: Integration Testing
- Task 010: Performance Regression Framework
- Task 011: Documentation and Rollback
- Task 012: Final Validation

## Critical Path

```
001 → 002 → 003 → 004 → 005 → 006 → 007 → 008 → 009 → 010 → 011 → 012
```

All tasks are sequential dependencies with no parallel work opportunities due to architectural dependencies.

## Agent Assignments

### Task 001: Profiling Infrastructure (1 day)
**Primary Agent**: verification-testing-lead
**Rationale**: Specialized in performance testing, benchmarking, and profiling

**Responsibilities**:
- Set up cargo-flamegraph for profiling
- Create Criterion baseline benchmarks
- Design profiling harness (10k nodes, 50k edges, 1000 queries)
- Identify hotspots and establish performance baselines

**Deliverables**:
- /benches/baseline_performance.rs
- /scripts/profile_hotspots.sh
- /docs/internal/profiling_results.md

### Task 002: Zig Build System (1 day)
**Primary Agent**: rust-graph-engine-architect
**Supporting Agent**: gpu-acceleration-architect

**Rationale**:
- rust-graph-engine-architect: Expert in Rust build systems and FFI
- gpu-acceleration-architect: Knowledge of low-level systems and SIMD infrastructure

**Responsibilities**:
- Create build.zig for static library compilation
- Integrate Zig build into cargo workflow via build.rs
- Implement zero-copy FFI boundary design
- Create feature flag system (zig-kernels)

**Deliverables**:
- /zig/build.zig
- /zig/src/ffi.zig
- /build.rs
- /src/zig_kernels/mod.rs

### Task 003: Differential Testing Harness (2 days)
**Primary Agent**: verification-testing-lead
**Rationale**: Specialized in differential testing between Rust and Zig implementations

**Responsibilities**:
- Design property-based testing framework with proptest
- Implement floating-point equivalence testing (epsilon = 1e-6)
- Create 10,000 test case corpus for each kernel
- Build regression test infrastructure

**Deliverables**:
- /tests/zig_differential/mod.rs
- /tests/zig_differential/vector_similarity.rs
- /tests/zig_differential/spreading_activation.rs
- /tests/zig_differential/decay_functions.rs

### Task 004: Memory Pool Allocator (1 day)
**Primary Agent**: systems-architecture-optimizer
**Rationale**: Specialized in lock-free data structures, memory management, and arena allocators

**Responsibilities**:
- Design arena/bump-pointer allocator
- Implement thread-local memory pools (1MB per thread)
- Create zero-copy allocation patterns
- Ensure thread-safe isolation

**Deliverables**:
- /zig/src/allocator.zig
- /zig/src/allocator_test.zig

### Task 005: Vector Similarity Kernel (3 days)
**Primary Agent**: gpu-acceleration-architect
**Rationale**: Expert in SIMD vectorization (AVX2, NEON) and parallel computing

**Responsibilities**:
- Implement SIMD cosine similarity (AVX2 for x86_64, NEON for ARM64)
- Design batch processing for 1:N queries
- Optimize memory access patterns
- Achieve 15-25% performance improvement

**Deliverables**:
- /zig/src/vector_similarity.zig
- /zig/src/vector_similarity_test.zig
- /benches/similarity_comparison.rs

### Task 006: Activation Spreading Kernel (3 days)
**Primary Agent**: gpu-acceleration-architect
**Supporting Agent**: rust-graph-engine-architect

**Rationale**:
- gpu-acceleration-architect: Cache optimization and SIMD expertise
- rust-graph-engine-architect: Graph algorithm knowledge

**Responsibilities**:
- Implement BFS-based spreading with CSR graph format
- Design cache-optimal edge batching
- Integrate refractory period decay
- Achieve 20-35% performance improvement

**Deliverables**:
- /zig/src/spreading_activation.zig
- /zig/src/spreading_activation_test.zig
- /benches/spreading_comparison.rs

### Task 007: Decay Function Kernel (2 days)
**Primary Agent**: gpu-acceleration-architect
**Supporting Agent**: memory-systems-researcher

**Rationale**:
- gpu-acceleration-architect: SIMD vectorization expertise
- memory-systems-researcher: Ebbinghaus decay curve biological grounding

**Responsibilities**:
- Implement vectorized exponential approximation
- Design batch decay processing
- Ensure biological plausibility
- Achieve 20-30% performance improvement

**Deliverables**:
- /zig/src/decay_functions.zig
- /zig/src/decay_functions_test.zig
- /benches/decay_comparison.rs

### Task 008: Arena Allocator Enhancement (2 days)
**Primary Agent**: systems-architecture-optimizer
**Rationale**: Expert in memory management, configuration systems, and monitoring

**Responsibilities**:
- Add configurable pool sizes (1MB to 100MB)
- Implement overflow strategies (panic, error, fallback)
- Create usage metrics tracking
- Support multi-threaded stress testing (32+ threads)

**Deliverables**:
- /zig/src/arena_config.zig
- /zig/src/arena_metrics.zig
- /tests/arena_stress.rs

### Task 009: Integration Testing (2 days)
**Primary Agent**: verification-testing-lead
**Rationale**: Specialized in end-to-end testing, API compatibility, and error handling

**Responsibilities**:
- Create full memory consolidation workflow tests
- Validate pattern completion with spreading activation
- Test mixed Rust/Zig execution paths
- Implement error injection and recovery testing

**Deliverables**:
- /tests/integration/zig_kernels.rs
- /tests/integration/scenarios/scenario_memory_recall.rs
- /tests/integration/scenarios/scenario_consolidation.rs
- /tests/integration/scenarios/scenario_pattern_completion.rs

### Task 010: Performance Regression Framework (2 days)
**Primary Agent**: verification-testing-lead
**Rationale**: Expert in automated benchmarking, regression detection, and CI integration

**Responsibilities**:
- Establish baseline performance for all kernels
- Create automated CI benchmark pipeline
- Implement >5% regression detection
- Build performance history tracking

**Deliverables**:
- /benches/regression/mod.rs
- /benches/regression/baselines.json
- /scripts/benchmark_regression.sh
- /.github/workflows/performance.yml (NOTE: Per CLAUDE.md, we should NOT create github workflows)

### Task 011: Documentation and Rollback (1 day)
**Primary Agent**: technical-communication-lead
**Rationale**: Specialized in creating developer-friendly documentation and operational guides

**Responsibilities**:
- Write deployment guide with step-by-step procedures
- Document performance tuning strategies
- Create monitoring integration guide
- Design rollback procedures and testing

**Deliverables**:
- /docs/operations/zig_performance_kernels.md
- /docs/operations/zig_rollback_procedures.md
- /docs/internal/zig_architecture.md
- Update CHANGELOG.md and README.md

### Task 012: Final Validation (1 day)
**Primary Agent**: graph-systems-acceptance-tester
**Rationale**: Specialized in validating graph database functionality for production readiness

**Responsibilities**:
- Execute comprehensive UAT test suite
- Validate performance targets achieved
- Review documentation completeness
- Complete production readiness checklist
- Obtain milestone sign-off

**Deliverables**:
- /docs/internal/milestone_10_uat.md
- /docs/internal/milestone_10_performance_report.md
- /tmp/milestone_10_validation_log.txt

## Success Metrics

### Performance Targets
- Vector Similarity: 15-25% improvement over Rust baseline
- Activation Spreading: 20-35% improvement
- Memory Decay: 20-30% improvement

### Quality Targets
- 100% differential test pass rate (30,000 test cases)
- 100% integration test pass rate
- Zero clippy warnings (make quality passes)
- Epsilon = 1e-6 numerical equivalence

### Operational Targets
- Thread-safe arena allocator (32+ threads)
- Zero arena overflows in testing
- <5% performance regression tolerance
- Complete rollback procedure validation

## Risk Assessment

### Technical Risks

**Risk 1: FFI Boundary Complexity**
- **Probability**: Medium
- **Impact**: High
- **Mitigation**: Extensive differential testing, zero-copy design validation
- **Owner**: rust-graph-engine-architect

**Risk 2: SIMD Portability (x86_64 vs ARM64)**
- **Probability**: Medium
- **Impact**: Medium
- **Mitigation**: Platform-specific baselines, fallback scalar paths
- **Owner**: gpu-acceleration-architect

**Risk 3: Arena Overflow in Production**
- **Probability**: Low
- **Impact**: High
- **Mitigation**: Configurable sizing, overflow strategies, stress testing
- **Owner**: systems-architecture-optimizer

**Risk 4: Numerical Divergence**
- **Probability**: Low
- **Impact**: Critical
- **Mitigation**: 30,000 property-based tests, epsilon validation
- **Owner**: verification-testing-lead

**Risk 5: CI/CD Pipeline Complexity**
- **Probability**: Medium
- **Impact**: Low
- **Mitigation**: Manual regression testing, documented build procedures
- **Owner**: verification-testing-lead
- **Note**: Per CLAUDE.md, we avoid GitHub workflows

### Schedule Risks

**Risk**: Dependencies are fully sequential - any delay cascades
- **Mitigation**: Buffer days built into estimates, daily status checks
- **Owner**: systems-product-planner

**Risk**: Zig compiler version pinning (0.13.0 stability)
- **Mitigation**: Document Zig version exactly, test upgrade path
- **Owner**: rust-graph-engine-architect

## Implementation Sequence

### Week 1: Foundation (5 days)
**Days 1-5**: Tasks 001-004

**Agent Allocation**:
- verification-testing-lead: Task 001 (1 day)
- rust-graph-engine-architect + gpu-acceleration-architect: Task 002 (1 day)
- verification-testing-lead: Task 003 (2 days)
- systems-architecture-optimizer: Task 004 (1 day)

**Milestone**: Build system functional, differential testing ready

### Week 2: Core Kernels (7 days)
**Days 6-12**: Tasks 005-008

**Agent Allocation**:
- gpu-acceleration-architect: Task 005 (3 days)
- gpu-acceleration-architect + rust-graph-engine-architect: Task 006 (3 days)
- gpu-acceleration-architect + memory-systems-researcher: Task 007 (2 days)
- systems-architecture-optimizer: Task 008 (2 days)

**Note**: Task 007 overlaps with Task 008 (different agents)

**Milestone**: All three kernels implemented and validated

### Week 3: Integration & Validation (6 days)
**Days 13-18**: Tasks 009-012

**Agent Allocation**:
- verification-testing-lead: Task 009 (2 days)
- verification-testing-lead: Task 010 (2 days)
- technical-communication-lead: Task 011 (1 day) - can run in parallel with Task 010
- graph-systems-acceptance-tester: Task 012 (1 day)

**Milestone**: Production-ready, documented, signed off

## Special Considerations

### Per CLAUDE.md Guidelines

1. **No GitHub Workflows**: Task 010 deliverable .github/workflows/performance.yml should be SKIPPED
   - Use manual regression testing scripts instead
   - Document CI-like procedures in scripts/benchmark_regression.sh

2. **Zero Emojis**: All documentation must avoid emojis entirely

3. **Diagnostics After Tests**: After each test run, execute ./scripts/engram_diagnostics.sh and prepend results to tmp/engram_diagnostics.log

4. **No Git Shortcuts**: Never use git restore, stash, or reset to work around test failures

5. **Clippy Zero Tolerance**: make quality must pass with ZERO warnings before task completion

### Build System Notes

- Pin Zig to exactly 0.13.0 stable
- Use -Doptimize=ReleaseFast for production kernels
- Static linking (libengram_kernels.a) for deployment simplicity
- Feature flag: zig-kernels (optional compilation)

### Testing Strategy

**Unit Tests**: Zig-side tests for kernel correctness
**Differential Tests**: 10,000 property-based cases per kernel (Rust vs Zig)
**Integration Tests**: End-to-end workflows with real graph data
**Regression Tests**: Baseline comparisons to prevent performance degradation

### Rollback Strategy

**Emergency Rollback**: Rebuild without --features zig-kernels (5-10 minute RTO)
**Gradual Rollback**: Feature flags or canary instances
**Validation**: Functional and performance checks post-rollback

## Handoff Requirements

### To Milestone 11 (Future)

- Documented Zig kernel architecture
- Performance baselines established
- Operational procedures validated
- Lessons learned documented

### Knowledge Transfer

- How to update Zig kernels
- How to update performance baselines
- How to tune arena allocator
- How to debug FFI issues

## Approval

**Plan Reviewed By**: [To be completed]
**Date**: [To be completed]
**Approved By**: [To be completed]
**Date**: [To be completed]

## Appendices

### Appendix A: Agent Expertise Mapping

| Agent | Tasks | Expertise Applied |
|-------|-------|-------------------|
| verification-testing-lead | 001, 003, 009, 010 | Profiling, differential testing, benchmarking, regression detection |
| rust-graph-engine-architect | 002, 006 | Rust FFI, build systems, graph algorithms |
| gpu-acceleration-architect | 002, 005, 006, 007 | SIMD vectorization, cache optimization, low-level systems |
| systems-architecture-optimizer | 004, 008 | Arena allocators, lock-free structures, memory management |
| memory-systems-researcher | 007 | Ebbinghaus decay curves, biological plausibility |
| technical-communication-lead | 011 | Developer documentation, operational guides |
| graph-systems-acceptance-tester | 012 | Production validation, UAT execution |

### Appendix B: File Creation Summary

**Zig Codebase** (12 files):
- /zig/build.zig
- /zig/src/ffi.zig
- /zig/src/allocator.zig
- /zig/src/vector_similarity.zig
- /zig/src/spreading_activation.zig
- /zig/src/decay_functions.zig
- /zig/src/arena_config.zig
- /zig/src/arena_metrics.zig
- Plus corresponding test files

**Rust Integration** (8 files):
- /build.rs
- /src/zig_kernels/mod.rs
- /tests/zig_differential/ (4 files)
- /tests/integration/zig_kernels.rs
- /tests/integration/scenarios/ (3 files)
- /tests/arena_stress.rs

**Benchmarks** (5 files):
- /benches/baseline_performance.rs
- /benches/similarity_comparison.rs
- /benches/spreading_comparison.rs
- /benches/decay_comparison.rs
- /benches/regression/mod.rs
- /benches/regression/baselines.json

**Scripts** (4 files):
- /scripts/profile_hotspots.sh
- /scripts/build_with_zig.sh
- /scripts/benchmark_regression.sh
- /scripts/update_baselines.sh

**Documentation** (6 files):
- /docs/internal/profiling_results.md
- /docs/internal/zig_architecture.md
- /docs/internal/milestone_10_uat.md
- /docs/internal/milestone_10_performance_report.md
- /docs/operations/zig_performance_kernels.md
- /docs/operations/zig_rollback_procedures.md

**Total**: ~50 files created/modified

### Appendix C: Dependencies Graph

```
001 (Profiling)
  ↓
002 (Build System)
  ↓
003 (Differential Testing)
  ↓
004 (Memory Pool Allocator)
  ↓
005 (Vector Similarity Kernel)
  ↓
006 (Activation Spreading Kernel)
  ↓
007 (Decay Function Kernel)
  ↓
008 (Arena Allocator Enhancement)
  ↓
009 (Integration Testing)
  ↓
010 (Performance Regression Framework)
  ↓
011 (Documentation and Rollback)
  ↓
012 (Final Validation)
```

### Appendix D: Key Performance Indicators

**Week 1 (Foundation)**:
- Build system compiles successfully with zig-kernels feature
- Differential testing framework validates 10,000 test cases
- Arena allocator passes stress tests (32+ threads)

**Week 2 (Core Kernels)**:
- Vector similarity: 15-25% faster than baseline
- Activation spreading: 20-35% faster than baseline
- Memory decay: 20-30% faster than baseline
- All differential tests pass within epsilon = 1e-6

**Week 3 (Integration)**:
- Integration tests: 100% pass rate
- Regression benchmarks: All pass
- Documentation: Complete and reviewed
- UAT: Signed off

### Appendix E: Communication Plan

**Daily Standups**: Review progress, blockers, dependencies
**Weekly Demos**: Show working kernels, performance improvements
**Milestone Reviews**: After each phase (Foundation, Core Kernels, Integration)
**Sign-off Meeting**: Final validation and approval

### Appendix F: Tools and Infrastructure

**Required Tools**:
- Zig 0.13.0 compiler
- Rust 1.75+ toolchain
- cargo-flamegraph for profiling
- Criterion for benchmarking
- proptest for property-based testing

**Infrastructure**:
- Development machines with AVX2 or NEON support
- Staging environment for rollback testing
- Performance baseline storage
