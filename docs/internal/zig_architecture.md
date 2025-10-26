# Zig Kernels - Architecture Documentation

## Overview

This document describes the internal architecture of Zig performance kernels for Engram maintainers and contributors. It covers FFI boundary design, memory management, SIMD implementations, and performance characteristics.

**Audience**: Engram core developers, performance engineers, contributors

**For operators**: See [Operations Guide](../operations/zig_performance_kernels.md)

## Architectural Principles

1. **Rust owns the graph**: All graph data structures, concurrency primitives, and API surfaces remain in Rust
2. **Zig computes intensively**: Vector operations, activation spreading, decay calculations delegated to Zig
3. **Zero-copy FFI**: Pass pointers to pre-allocated buffers, avoid serialization overhead
4. **Caller allocates, callee computes**: Rust allocates all output buffers, Zig writes results
5. **Thread-local arenas**: Each thread has isolated memory pool for kernel scratch space
6. **Graceful degradation**: Runtime detection falls back to Rust if Zig unavailable

## FFI Boundary Design

### Memory Ownership Model

The FFI boundary follows strict ownership rules to prevent memory safety violations:

```
┌─────────────────────────────────────────────────────────────┐
│                    Rust (Caller)                            │
│                                                             │
│  1. Allocate input buffers (query, candidates, graph)      │
│  2. Allocate output buffers (scores, activations)          │
│  3. Call Zig kernel via FFI                                │
│  4. Read results from output buffers                       │
│  5. Deallocate all buffers                                 │
│                                                             │
└──────────────────┬──────────────────────────────────────────┘
                   │ FFI Boundary (C ABI)
                   │ Ownership: Rust retains ownership
                   │ Lifetime: Guaranteed for call duration
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                    Zig (Callee)                             │
│                                                             │
│  1. Receive pointers to Rust-allocated buffers             │
│  2. Compute results (SIMD operations)                      │
│  3. Write to output buffers                                │
│  4. Return control to Rust                                 │
│  5. NEVER free Rust-allocated memory                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Invariants**:

- **No ownership transfer**: Zig never takes ownership of Rust-allocated memory
- **No cross-language freeing**: Rust frees what Rust allocates, Zig never calls free on Rust pointers
- **Synchronous execution**: Kernels return only when computation complete (no async callbacks)
- **No aliasing**: Input and output pointers never overlap

### Function Signatures

All FFI functions follow this pattern:

```zig
// zig/src/ffi.zig
export fn engram_kernel_name(
    inputs: [*]const InputType,   // Read-only inputs
    outputs: [*]OutputType,       // Write-only outputs
    count: usize,                 // Element counts for validation
) void;  // Never return errors (handle gracefully)
```

**Design rationale**:

- `export fn`: Generates C-compatible ABI (no name mangling)
- `[*]const`: Many-item pointer, read-only (Rust: `*const T`)
- `[*]`: Many-item pointer, mutable (Rust: `*mut T`)
- `void` return: Errors handled internally via overflow strategies, never panic across FFI

### Example: Vector Similarity Kernel

#### Zig Side

```zig
// zig/src/ffi.zig
export fn engram_vector_similarity(
    query: [*]const f32,
    candidates: [*]const f32,
    scores: [*]f32,
    query_len: usize,
    num_candidates: usize,
) void {
    const impl = @import("vector_similarity.zig");

    // Convert to Zig slices for safety
    const query_slice = query[0..query_len];
    const candidate_slice = candidates[0..(query_len * num_candidates)];
    const scores_slice = scores[0..num_candidates];

    // Delegate to implementation
    impl.batchCosineSimilarity(
        query_slice,
        candidate_slice,
        scores_slice,
        num_candidates,
    );
}
```

#### Rust Side

```rust
// src/zig_kernels/mod.rs
#[cfg(feature = "zig-kernels")]
mod ffi {
    extern "C" {
        pub fn engram_vector_similarity(
            query: *const f32,
            candidates: *const f32,
            scores: *mut f32,
            query_len: usize,
            num_candidates: usize,
        );
    }
}

pub fn vector_similarity(query: &[f32], candidates: &[f32]) -> Vec<f32> {
    // Validate inputs
    assert!(!query.is_empty(), "Query cannot be empty");
    assert_eq!(
        candidates.len() % query.len(),
        0,
        "Candidates must be multiple of query length"
    );

    let num_candidates = candidates.len() / query.len();
    let mut scores = vec![0.0_f32; num_candidates];

    // Call Zig kernel
    unsafe {
        ffi::engram_vector_similarity(
            query.as_ptr(),
            candidates.as_ptr(),
            scores.as_mut_ptr(),
            query.len(),
            num_candidates,
        );
    }

    scores
}
```

**Safety invariants upheld**:

1. Pointers valid for call duration (query, candidates, scores live until kernel returns)
2. Slices have correct lengths (validated with assertions)
3. No aliasing (query != candidates != scores)
4. Output buffer pre-allocated with correct size

## Memory Management

### Arena Allocator Design

Zig kernels use thread-local arena allocators for scratch space:

```zig
// zig/src/allocator.zig
pub const ArenaAllocator = struct {
    buffer: []u8,          // Fixed-size memory pool
    offset: usize,         // Current allocation offset (bump pointer)
    high_water_mark: usize,// Maximum offset reached
    overflow_count: usize, // Number of overflow events

    pub fn alloc(self: *ArenaAllocator, size: usize, alignment: usize) ![]u8 {
        // Align offset to required boundary
        const aligned_offset = std.mem.alignForward(usize, self.offset, alignment);

        // Check capacity
        if (aligned_offset + size > self.buffer.len) {
            self.overflow_count += 1;
            return error.OutOfMemory;
        }

        // Bump pointer allocation (O(1))
        const ptr = self.buffer[aligned_offset..][0..size];
        self.offset = aligned_offset + size;
        self.high_water_mark = @max(self.high_water_mark, self.offset);

        return ptr;
    }

    pub fn reset(self: *ArenaAllocator) void {
        // Bulk deallocation (O(1))
        self.offset = 0;
        // Keep high_water_mark for diagnostics
    }
};
```

**Performance characteristics**:

- Allocation: O(1) - pointer increment with alignment padding
- Deallocation: O(1) - bulk reset
- Space overhead: 0 bytes per allocation (no metadata)
- Fragmentation: None (linear allocation pattern)

### Thread-Local Storage

Each thread maintains an independent arena:

```zig
// zig/src/allocator.zig
threadlocal var kernel_arena_buffer: ?[]u8 = null;
threadlocal var kernel_arena: ?ArenaAllocator = null;

pub fn getThreadArena() *ArenaAllocator {
    if (kernel_arena == null) {
        // Lazy initialization on first access
        const config = @import("arena_config.zig").getConfig();
        const buffer = std.heap.page_allocator.alloc(u8, config.pool_size)
            catch @panic("Failed to allocate thread arena");

        kernel_arena_buffer = buffer;
        kernel_arena = ArenaAllocator.init(buffer);
    }

    return &kernel_arena.?;
}

pub fn resetThreadArena() void {
    if (kernel_arena) |*arena| {
        arena.reset();
    }
}
```

**Design rationale**:

- **Zero contention**: Each thread has isolated arena, no locking needed
- **NUMA-aware**: Thread-local allocations happen on local memory node
- **Lazy initialization**: Arena created on first kernel invocation per thread
- **Bounded memory**: 1MB per thread (configurable), easy to reason about

### Overflow Handling

When arena capacity is exhausted:

```zig
// zig/src/arena_config.zig
pub const OverflowStrategy = enum {
    panic,          // Abort process (development/testing)
    error_return,   // Return error, log warning (production)
    fallback,       // Attempt system allocator (experimental)
};

// In allocator
if (aligned_offset + size > self.buffer.len) {
    self.overflow_count += 1;

    const strategy = config.getConfig().overflow_strategy;
    switch (strategy) {
        .panic => @panic("Arena allocator overflow"),
        .error_return => return error.OutOfMemory,
        .fallback => {
            std.log.warn("Arena overflow, consider increasing ENGRAM_ARENA_SIZE", .{});
            return error.OutOfMemory;
        },
    }
}
```

**Production recommendation**: Use `error_return` strategy to prevent crashes.

## SIMD Implementation

### Platform Detection

Zig kernels detect CPU features at compile time:

```zig
// zig/src/vector_similarity.zig
const builtin = @import("builtin");
const std = @import("std");

pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    if (comptime builtin.cpu.arch == .x86_64) {
        // Check for AVX2 support at compile time
        if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) {
            return dotProductAVX2(a, b);
        }
    } else if (comptime builtin.cpu.arch == .aarch64) {
        // NEON is standard on ARMv8+
        return dotProductNEON(a, b);
    }

    // Scalar fallback
    return dotProductScalar(a, b);
}
```

### AVX2 Implementation (x86_64)

```zig
fn dotProductAVX2(a: []const f32, b: []const f32) f32 {
    const vec_size = 8;  // AVX2 processes 8 floats per instruction
    const num_vecs = a.len / vec_size;

    var sum: f32 = 0.0;
    var i: usize = 0;

    // Vectorized loop
    while (i < num_vecs) : (i += 1) {
        const offset = i * vec_size;

        // Load 8 floats into AVX2 registers
        const a_vec = @as(@Vector(8, f32), a[offset..][0..vec_size].*);
        const b_vec = @as(@Vector(8, f32), b[offset..][0..vec_size].*);

        // Multiply and accumulate
        const product = a_vec * b_vec;
        sum += @reduce(.Add, product);
    }

    // Handle remainder with scalar code
    const remainder_start = num_vecs * vec_size;
    for (a[remainder_start..], b[remainder_start..]) |a_val, b_val| {
        sum += a_val * b_val;
    }

    return sum;
}
```

**Performance**: Processes 8 floats per instruction vs. 1 in scalar code.

### NEON Implementation (ARM64)

```zig
fn dotProductNEON(a: []const f32, b: []const f32) f32 {
    const vec_size = 4;  // NEON processes 4 floats per instruction
    const num_vecs = a.len / vec_size;

    var sum: f32 = 0.0;
    var i: usize = 0;

    // Vectorized loop
    while (i < num_vecs) : (i += 1) {
        const offset = i * vec_size;

        // Load 4 floats into NEON registers
        const a_vec = @as(@Vector(4, f32), a[offset..][0..vec_size].*);
        const b_vec = @as(@Vector(4, f32), b[offset..][0..vec_size].*);

        // Multiply and accumulate
        const product = a_vec * b_vec;
        sum += @reduce(.Add, product);
    }

    // Handle remainder
    const remainder_start = num_vecs * vec_size;
    for (a[remainder_start..], b[remainder_start..]) |a_val, b_val| {
        sum += a_val * b_val;
    }

    return sum;
}
```

**Performance**: Processes 4 floats per instruction vs. 1 in scalar code.

### Scalar Fallback

```zig
fn dotProductScalar(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;

    for (a, b) |a_val, b_val| {
        sum += a_val * b_val;
    }

    return sum;
}
```

**Note**: Used when SIMD unavailable or for remainder elements.

## Performance Characteristics

### Expected Improvements

Based on profiling and benchmarking:

| Operation | Rust Baseline | Zig Kernel | Improvement | Primary Optimization |
|-----------|--------------|------------|-------------|---------------------|
| Vector Similarity (768d) | 2.3 us | 1.7 us | 25% | SIMD dot product |
| Spreading Activation (1000n) | 145 us | 95 us | 35% | Cache-optimized BFS |
| Decay Calculation (10k) | 89 us | 65 us | 27% | SIMD exponential |

### Bottleneck Analysis

#### Vector Similarity

**Primary bottleneck**: Memory bandwidth

- Loading 768-dimensional embeddings from RAM dominates computation
- SIMD accelerates compute, but memory bandwidth is the limiting factor
- **Optimization**: Use smaller embeddings (384d) or prefetching

#### Spreading Activation

**Primary bottleneck**: Cache locality

- Random graph access patterns cause cache misses
- Edge traversal jumps across non-contiguous memory
- **Optimization**: Graph edge reordering for sequential access patterns

#### Decay Calculation

**Primary bottleneck**: Exponential function

- `exp()` is computationally expensive even with SIMD
- Called once per memory (10k+ invocations)
- **Optimization**: Lookup table for discrete time intervals

### Future Optimizations

Potential improvements for future milestones:

1. **FMA (Fused Multiply-Add)**: Use `vfmadd` instruction for dot products (x86_64)
2. **Exponential LUT**: Replace `exp()` with lookup table for decay (10-20% faster)
3. **Graph Edge Reordering**: Sort edges by access pattern for cache locality
4. **GPU Offload**: Port kernels to CUDA/ROCm for massive parallelism (Milestone 14)
5. **Prefetching**: Manual prefetch instructions for embedding loads

## Testing Strategy

### Differential Testing

Every Zig kernel must produce identical results to Rust baseline:

```rust
// tests/zig_differential.rs
use proptest::prelude::*;

proptest! {
    #[test]
    fn zig_vector_similarity_matches_rust(
        query in vec_f32_embedding(768),
        candidates in vec_vec_f32_embeddings(100, 768),
    ) {
        let zig_scores = zig_kernels::vector_similarity(&query, &candidates);
        let rust_scores = rust_baseline::vector_similarity(&query, &candidates);

        for (zig, rust) in zig_scores.iter().zip(rust_scores.iter()) {
            // Allow small epsilon for floating-point precision
            assert!((zig - rust).abs() < 1e-6, "Divergence detected");
        }
    }
}
```

**Property-based testing**:

- Generates random inputs (embeddings, graphs, ages)
- Compares Zig kernel output to Rust baseline
- Fails if any divergence exceeds epsilon (1e-6)

### Performance Testing

Regression benchmarks prevent performance degradation:

```rust
// benches/regression/mod.rs
fn regression_benchmark(c: &mut Criterion) {
    let baselines = Baselines::load();

    let query = generate_embedding(768);
    let candidates = generate_embeddings(1000, 768);

    let mut group = c.benchmark_group("regression");
    group.bench_function("vector_similarity_768d_1000c", |b| {
        b.iter(|| {
            let scores = vector_similarity(&query, &candidates);
            criterion::black_box(scores);
        });
    });

    // Check against baseline
    let result = extract_benchmark_result("vector_similarity_768d_1000c");
    baselines.check_regression("vector_similarity_768d_1000c", result.mean_ns)?;
}
```

**Regression detection**:

- Compares current performance to stored baselines
- Fails build if regression exceeds 5%
- Runs automatically in CI on every commit

## Build System Integration

### Zig Build Configuration

```zig
// zig/build.zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Static library for FFI
    const lib = b.addStaticLibrary(.{
        .name = "engram_kernels",
        .root_source_file = .{ .path = "src/ffi.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Enable C ABI exports
    lib.bundle_compiler_rt = true;

    b.installArtifact(lib);
}
```

### Cargo Integration

```rust
// build.rs
#[cfg(feature = "zig-kernels")]
fn build_zig_library() {
    // Check Zig availability
    let zig_available = Command::new("zig")
        .arg("version")
        .output()
        .is_ok();

    if !zig_available {
        panic!("Zig compiler not found");
    }

    // Build Zig library
    let status = Command::new("zig")
        .args(&["build", "-Doptimize=ReleaseFast"])
        .current_dir("zig")
        .status()
        .expect("Failed to build Zig library");

    assert!(status.success(), "Zig build failed");

    // Link static library
    println!("cargo:rustc-link-search=native=zig/zig-out/lib");
    println!("cargo:rustc-link-lib=static=engram_kernels");
}
```

## Error Handling

### FFI Boundary Error Handling

Errors cannot cross FFI boundary safely, so kernels handle errors internally:

```zig
export fn engram_vector_similarity(...) void {
    const arena = allocator.getThreadArena();
    defer allocator.resetThreadArena();

    // Allocate scratch space
    const normalized = arena.alloc(f32, query_len) catch |err| {
        // Handle allocation failure gracefully
        std.log.warn("Arena allocation failed: {}", .{err});
        // Write zeros to output (caller will detect anomaly)
        for (scores[0..num_candidates]) |*score| {
            score.* = 0.0;
        }
        return;  // Exit without panicking
    };

    // Continue with computation...
}
```

**Design rationale**:

- FFI boundary cannot propagate Zig errors to Rust
- Kernels write sentinel values (zeros) on error
- Caller can detect anomalies and handle appropriately

### Rust-Side Validation

```rust
pub fn vector_similarity(query: &[f32], candidates: &[f32]) -> Vec<f32> {
    // Validate inputs before FFI call
    assert!(!query.is_empty(), "Query cannot be empty");
    assert_eq!(
        candidates.len() % query.len(),
        0,
        "Candidates must be multiple of query length"
    );

    let scores = /* call Zig kernel */;

    // Validate outputs after FFI call
    for (i, &score) in scores.iter().enumerate() {
        if score.is_nan() || score.is_infinite() {
            panic!("Kernel returned invalid score at index {}: {}", i, score);
        }
    }

    scores
}
```

## Debugging and Profiling

### Enabling Debug Symbols

```bash
# Build Zig library with debug info
cd zig
zig build -Doptimize=Debug

# Build Rust with debug info
cargo build --features zig-kernels
```

### Profiling Tools

#### Linux (perf)

```bash
# Profile with perf
perf record -g ./target/release/engram-cli benchmark

# View flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

#### macOS (Instruments)

```bash
# Profile with Instruments
xcrun xctrace record --template 'Time Profiler' \
  --launch ./target/release/engram-cli benchmark

# Open in Instruments GUI
```

### Arena Utilization Tracking

```zig
pub fn getThreadArenaUtilization() f64 {
    if (kernel_arena) |*arena| {
        const capacity = @as(f64, @floatFromInt(arena.buffer.len));
        const used = @as(f64, @floatFromInt(arena.high_water_mark));
        return used / capacity;
    }
    return 0.0;
}
```

## See Also

- [Operations Guide](../operations/zig_performance_kernels.md) - Deployment and configuration
- [Rollback Procedures](../operations/zig_rollback_procedures.md) - Emergency and gradual rollback
- [Performance Regression Guide](./performance_regression_guide.md) - Benchmarking framework
- [Profiling Results](./profiling_results.md) - Hotspot analysis
