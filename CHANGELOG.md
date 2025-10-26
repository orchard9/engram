# Changelog

All notable changes to Engram will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Milestone 10: Zig Performance Kernels

#### Performance Kernels

- **Vector Similarity Kernel**: SIMD-accelerated cosine similarity calculations for embedding search
  - 15-25% performance improvement over Rust baseline
  - AVX2 support for x86_64 (8 floats per instruction)
  - NEON support for ARM64 (4 floats per instruction)
  - Automatic scalar fallback when SIMD unavailable
  - Location: `zig/src/vector_similarity.zig`

- **Spreading Activation Kernel**: Cache-optimized graph traversal for associative memory retrieval
  - 20-35% performance improvement over Rust baseline
  - Breadth-first search with edge batching
  - Thread-local arena allocations for BFS queues
  - Location: `zig/src/spreading_activation.zig`

- **Memory Decay Kernel**: Vectorized Ebbinghaus decay calculations for temporal dynamics
  - 20-30% performance improvement over Rust baseline
  - SIMD exponential function approximations
  - Batch processing of memory age calculations
  - Location: `zig/src/decay_functions.zig`

#### Memory Management

- **Arena Allocator**: Thread-local memory pools for kernel scratch space
  - O(1) bump-pointer allocation with zero fragmentation
  - Configurable pool sizes via `ENGRAM_ARENA_SIZE` environment variable
  - Overflow detection with configurable strategies (panic, error, fallback)
  - High-water mark tracking for capacity planning
  - Location: `zig/src/allocator.zig`

- **Arena Configuration**: Runtime configuration for production workloads
  - Environment variable support: `ENGRAM_ARENA_SIZE`, `ENGRAM_ARENA_OVERFLOW`
  - Programmatic API for dynamic configuration
  - Per-thread isolation (zero contention)
  - Location: `zig/src/arena_config.zig`

- **Arena Metrics**: Usage tracking and monitoring integration
  - Total allocations, overflows, and resets
  - High-water mark per thread
  - Thread-safe global metrics aggregation
  - Location: `zig/src/arena_metrics.zig`

#### Build System

- **Zig Build Integration**: Seamless integration with Cargo build workflow
  - `zig-kernels` feature flag for opt-in compilation
  - Static library linking (libengram_kernels.a)
  - Automatic Zig compiler detection
  - Build script: `scripts/build_with_zig.sh`
  - Location: `zig/build.zig`, `build.rs`

- **FFI Bindings**: C-compatible interface between Rust and Zig
  - Zero-copy data passing (pointer-based)
  - Caller-allocated buffers (Rust), callee-computed results (Zig)
  - Thread-safe kernel invocations
  - Location: `zig/src/ffi.zig`, `src/zig_kernels/mod.rs`

#### Testing and Validation

- **Differential Testing**: Property-based testing ensures correctness
  - Automatic comparison of Zig kernels vs. Rust baseline
  - Epsilon-based floating-point validation (1e-6 tolerance)
  - Proptest integration for random input generation
  - Location: `tests/zig_differential.rs`

- **Performance Regression Framework**: Automated benchmarking prevents degradation
  - Baseline performance storage (`benches/regression/baselines.json`)
  - Automated CI benchmarks on every commit
  - Build failure if regression exceeds 5%
  - Benchmark script: `scripts/benchmark_regression.sh`
  - Location: `benches/regression/mod.rs`

- **Integration Testing**: End-to-end validation with production workloads
  - Multi-threaded stress testing (32+ threads)
  - Arena exhaustion scenarios
  - Numerical accuracy validation
  - Location: `tests/arena_stress.rs`, `engram-core/tests/zig_integration_tests.rs`

#### Documentation

- **Operations Guide**: Complete deployment and troubleshooting documentation
  - Installation instructions (macOS, Linux, ARM64)
  - Configuration reference (arena sizing, overflow strategies)
  - Performance tuning guidelines
  - Monitoring integration with Prometheus/Grafana
  - Troubleshooting common issues
  - Location: `docs/operations/zig_performance_kernels.md`

- **Rollback Procedures**: Emergency and gradual rollback strategies
  - Emergency rollback (5-10 minute RTO)
  - Gradual rollback (canary, traffic shifting)
  - Common rollback scenarios with root cause analysis
  - Rollback testing procedures
  - Location: `docs/operations/zig_rollback_procedures.md`

- **Architecture Documentation**: Internal design for maintainers
  - FFI boundary design and memory ownership model
  - Arena allocator architecture
  - SIMD implementation details (AVX2, NEON)
  - Performance characteristics and bottleneck analysis
  - Location: `docs/internal/zig_architecture.md`

- **Performance Regression Guide**: Benchmarking framework usage
  - Baseline establishment and update procedures
  - Regression detection configuration
  - CI integration
  - Location: `docs/internal/performance_regression_guide.md`

- **Profiling Results**: Hotspot analysis and kernel selection rationale
  - Flamegraph analysis identifying compute-bound operations
  - Kernel candidate selection criteria
  - Performance improvement targets
  - Location: `docs/internal/profiling_results.md`

### Changed

- **Feature Flag System**: Added `zig-kernels` feature for opt-in compilation
  - Build without Zig: `cargo build` (Rust-only, no Zig dependency)
  - Build with Zig: `cargo build --features zig-kernels`
  - Graceful fallback when Zig compiler unavailable

- **Build Process**: Extended build system to support multi-language compilation
  - `build.rs` invokes Zig compiler when `zig-kernels` feature enabled
  - Static library linking during Rust compilation
  - Platform-specific library paths (macOS, Linux)

### Dependencies

- **Zig 0.13.0**: Required for building with `zig-kernels` feature
  - C ABI compatibility for FFI
  - SIMD intrinsics (AVX2, NEON)
  - Thread-local storage support

### Breaking Changes

None. Zig kernels are entirely opt-in via feature flag.

### Migration Guide

#### For Operators

To enable Zig performance kernels in production:

1. Install Zig 0.13.0 on all deployment nodes
2. Rebuild with `--features zig-kernels`
3. Configure arena size for workload: `export ENGRAM_ARENA_SIZE=2097152`
4. Monitor arena metrics and performance improvements
5. Follow rollback procedures if issues arise

See [Operations Guide](docs/operations/zig_performance_kernels.md) for full deployment instructions.

#### For Developers

No API changes. Zig kernels are drop-in replacements for Rust implementations:

```rust
// No code changes needed - feature flag controls implementation
use engram::vector_similarity;

let scores = vector_similarity(&query, &candidates);
// Uses Zig kernel if zig-kernels feature enabled
// Uses Rust baseline otherwise
```

### Performance Impact

Expected performance improvements with `zig-kernels` feature enabled:

| Operation | Baseline (Rust) | With Zig Kernels | Improvement |
|-----------|----------------|------------------|-------------|
| Vector Similarity (768-dim) | 2.3 us | 1.7 us | 25% faster |
| Spreading Activation (1000 nodes) | 145 us | 95 us | 35% faster |
| Memory Decay (10k memories) | 89 us | 65 us | 27% faster |

### Known Issues

- Zig compiler version 0.13.0 required (ABI compatibility)
- AVX2 required for optimal performance on x86_64 (graceful fallback to scalar)
- Arena overflow warnings indicate insufficient `ENGRAM_ARENA_SIZE` configuration

### Acknowledgments

Milestone 10 implemented by the Engram core team following the architectural principles outlined in `vision.md` and `milestones.md`. Special thanks to the Zig community for language design that enabled zero-cost FFI integration.

---

## [Previous Versions]

Previous milestones (1-9) predated this CHANGELOG. For historical changes, see:

- `roadmap/milestone-9/` - Query Language & Pattern Completion
- `roadmap/milestone-8/` - Memory Consolidation
- `roadmap/milestone-7/` - Probabilistic Queries
- Earlier milestones in `roadmap/`

---

## Format Guidelines

### Categories

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

### Version Format

- **[Unreleased]**: Changes in development
- **[1.0.0] - YYYY-MM-DD**: Released versions with date

### Milestone Integration

Major milestones (e.g., Milestone 10) are documented under **[Unreleased]** until official release, then moved to a versioned section.
