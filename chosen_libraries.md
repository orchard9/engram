# Chosen Libraries

## Core Dependencies

### Graph Foundation
- `petgraph 0.6`: Base graph algorithms. Stable, well-tested. Extend, don't wrap.

### Concurrency
- `tokio 1.x`: Async runtime. Single-threaded executors for region actors.
- `parking_lot 0.12`: Faster mutexes with smaller memory footprint.
- `crossbeam 0.8`: Lock-free data structures and channels.
- `rayon 1.10`: Data parallelism for batch operations.

### Memory Management
- `mimalloc 0.1`: Replaces system allocator. Better performance for small allocations.
- `bumpalo 3.16`: Arena allocation for temporary graph operations.

### Vector Operations
- `nalgebra 0.33`: Linear algebra without GPU dependencies.
- `simdeez 2.0`: Portable SIMD operations with runtime detection.
- `wide 0.7`: Fixed-size SIMD vectors for embedding operations.

### Serialization
- `rkyv 0.7`: Zero-copy deserialization for memory-mapped files.
- `bincode 1.3`: Fast binary serialization for network protocol.

### Storage
- `memmap2 0.9`: Memory-mapped files for cold storage tier.
- `zstd 0.13`: Compression for append-only logs.

### Error Handling
- `thiserror 1.0`: Derive macro for error types.
- `color-eyre 0.6`: Development error reporting only. Disabled in release.

### Testing
- `proptest 1.5`: Property-based testing for probabilistic operations.
- `criterion 0.5`: Statistical benchmarking framework.
- `divan 0.1`: Allocation-tracking benchmarks.

### Profiling
- `pprof 0.13`: CPU profiling with flamegraph generation.
- `memory-stats 1.2`: Heap usage tracking.

## GPU Libraries

### CUDA
- Build from Zig with manual bindings. No Rust wrapper overhead.

### Compute
- `wgpu 22.0`: Fallback portable GPU computation.

## Development Dependencies

### Fuzzing
- `arbitrary 1.3`: Structured fuzzing input generation.
- `cargo-fuzz`: AFL-based fuzzing harness.

### Linting
- `clippy`: Pedantic mode enabled.
- Custom lints for project patterns.

## Explicitly Rejected

### Not Using
- `diesel`/`sqlx`: SQL abstractions inappropriate for graph operations
- `tantivy`: Full-text search overkill for embedding similarity
- `sled`: Embedded database with wrong consistency model
- `rocksdb`: LSM tree structure mismatched for activation patterns
- `ndarray`: Scientific computing focus, not SIMD-optimized
- `candle`/`burn`: ML frameworks too heavyweight for embedding ops
- `async-trait`: Not needed with Edition 2024 native async traits
- `once_cell`: Use `std::sync::OnceLock` instead
- `serde`: Reflection overhead, use `rkyv` for performance

## Version Policy

- Lock minor versions in `Cargo.toml`
- Update quarterly unless security issue
- Benchmark before and after updates
- Document breaking changes in CHANGELOG

## Dependency Audit

Run weekly:
```bash
cargo audit
cargo outdated
cargo tree --duplicates
```

Maximum transitive dependencies: 100
Maximum total binary size: 50MB

## FFI Libraries

### Zig Interop
Direct linking, no C ABI overhead:
- Shared memory pools
- Direct function calls
- No serialization between Rust/Zig

### Platform Specific

Linux:
- `io-uring 0.6`: Async I/O for storage tier

macOS:
- `metal 0.29`: GPU computation fallback

Windows:
- Not supported initially

## Benchmarking Baselines

Compare against:
- `neo4j-rust-driver`: Traditional graph operations
- `qdrant-client`: Vector similarity search
- `moka`: Cache eviction strategies

Performance must exceed all three for respective operations.

## Security Requirements

- No dependencies with RUSTSEC advisories
- Supply chain verification via `cargo-crev`
- Reproducible builds with locked dependencies
- No dependencies pulling from git repositories
- Corporate CLA required for contributions
