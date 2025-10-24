# Research: build.zig Meets cargo build - FFI Without the Pain

## Core Question

How do you integrate Zig performance kernels into a Rust codebase without turning your build system into a nightmare? We need seamless cargo build integration, zero-copy FFI boundaries, and graceful fallback when Zig isn't available.

## Context: The FFI Problem

Foreign Function Interface (FFI) lets different languages talk to each other. It's powerful but painful:
- **Build complexity:** Two compilers, two build systems, dependency management across languages
- **ABI compatibility:** Rust and Zig must agree on memory layout, calling conventions, name mangling
- **Type safety:** Crossing the FFI boundary means leaving Rust's safety guarantees
- **Performance overhead:** Bad FFI design adds serialization, copying, or allocation costs

The goal: Make calling Zig from Rust feel natural, safe, and zero-cost.

## Research Findings

### Why Zig for Performance Kernels?

Zig isn't "better Rust" - it's a different tool for different problems:

**Zig advantages:**
- **Explicit control:** No hidden allocations, no hidden control flow
- **Comptime metaprogramming:** Code generation at compile time, zero runtime cost
- **C ABI by default:** Export functions with `export fn`, no wrapper needed
- **Simpler optimization:** Easier to reason about generated assembly
- **SIMD primitives:** @Vector type auto-lowers to platform intrinsics

**Rust advantages:**
- **Memory safety:** Borrow checker prevents use-after-free, data races
- **Ecosystem:** Rich crate ecosystem, mature tooling
- **High-level abstractions:** Zero-cost iterators, async/await
- **Safe concurrency:** Type system enforces thread safety

For Engram: Rust owns the graph engine (safety matters), Zig owns hot-path compute kernels (explicit control matters).

### C ABI as Common Ground

Both Rust and Zig can export C-compatible functions:

**Zig side:**
```zig
export fn engram_vector_similarity(
    query: [*]const f32,
    candidates: [*]const f32,
    scores: [*]f32,
    query_len: usize,
    num_candidates: usize,
) void {
    // Implementation
}
```

**Rust side:**
```rust
extern "C" {
    fn engram_vector_similarity(
        query: *const f32,
        candidates: *const f32,
        scores: *mut f32,
        query_len: usize,
        num_candidates: usize,
    );
}
```

Key principle: **C ABI is the lingua franca.** No custom serialization, no wrapper generation - just pointers and primitive types.

### Zero-Copy FFI Design

Bad FFI design copies data at the boundary:

```rust
// BAD: Serialize to JSON, parse in Zig, return JSON
let json_input = serde_json::to_string(&data)?;
let json_output = unsafe { zig_function(json_input.as_ptr()) };
let result: Vec<f32> = serde_json::from_str(json_output)?;
```

This adds latency and defeats the purpose of optimization.

Good FFI design passes pointers to existing buffers:

```rust
// GOOD: Pass pointers, Zig writes directly to Rust's buffer
let mut scores = vec![0.0_f32; num_candidates];
unsafe {
    engram_vector_similarity(
        query.as_ptr(),
        candidates.as_ptr(),
        scores.as_mut_ptr(),
        query.len(),
        num_candidates,
    );
}
// scores now contains results, no copy needed
```

Caller allocates, callee populates. Zero-copy by design.

### Build System Integration

Three approaches researched:

**Option A: Manual build scripts**
- Call `zig build` from `build.rs`
- Link resulting `libengram_kernels.a`
- Pro: Full control
- Con: Fragile, platform-specific

**Option B: cc crate compilation**
- Treat Zig files as C sources
- Let cc crate handle compilation
- Pro: Leverages existing Rust tooling
- Con: Doesn't use Zig's build system features

**Option C: Hybrid approach (chosen)**
- Use build.rs to invoke `zig build`
- Zig's build.zig produces static library
- Feature flag enables/disables Zig kernels
- Pro: Best of both worlds (Zig features + Rust integration)
- Con: Requires Zig installation (mitigated with feature flag)

### Feature Flags for Graceful Fallback

Critical design decision: Zig kernels must be **optional**.

```toml
[features]
default = []
zig-kernels = []
```

Without `--features zig-kernels`, Engram builds using pure Rust implementations. With the feature, it links Zig kernels for performance.

This allows:
- **Development without Zig:** Contributors without Zig installed can still build
- **Platform-specific optimization:** Enable Zig on x86_64, fall back on other architectures
- **Debugging:** Disable Zig kernels to isolate performance vs correctness issues

### Static vs Dynamic Linking

Static linking (chosen):
- Pro: Single binary, no runtime dependencies
- Pro: Link-time optimization across language boundary
- Con: Larger binary size

Dynamic linking:
- Pro: Smaller binaries, shared library reuse
- Con: Runtime dependency management, versioning hell

For Engram: Static linking keeps deployment simple. Binary size is acceptable for server deployments.

### Memory Safety at FFI Boundary

FFI is inherently unsafe - you're promising to uphold invariants the compiler can't check:

**Unsafe invariants we must maintain:**
1. Pointers are valid for entire call duration
2. Slices have correct length (no buffer overruns)
3. No aliasing between input and output pointers
4. Returned pointers (if any) have defined lifetime

**Safety strategy:**
```rust
pub fn vector_similarity(query: &[f32], candidates: &[Vec<f32>]) -> Vec<f32> {
    // Validate dimensions (safe Rust layer)
    for candidate in candidates {
        assert_eq!(candidate.len(), query.len());
    }

    // Flatten candidates
    let candidates_flat: Vec<f32> = candidates
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();

    let mut scores = vec![0.0_f32; candidates.len()];

    // Unsafe block clearly delineated
    unsafe {
        ffi::engram_vector_similarity(
            query.as_ptr(),
            candidates_flat.as_ptr(),
            scores.as_mut_ptr(),
            query.len(),
            candidates.len(),
        );
    }

    scores
}
```

Safe wrapper does validation, unsafe block is minimized.

## Key Insights

1. **C ABI as common ground:** Both Rust and Zig speak C, no custom serialization
2. **Zero-copy design:** Pass pointers, avoid serialization overhead
3. **Feature flags for fallback:** Zig kernels optional, not required
4. **Static linking for simplicity:** Single binary, no runtime dependencies
5. **Safe wrappers minimize unsafe:** Validation in Rust, raw FFI in small blocks

## References

- Zig Build System documentation: https://ziglang.org/documentation/master/#Build-System
- Rust FFI Guide: https://doc.rust-lang.org/nomicon/ffi.html
- "Linking Rust and Zig" by Andrew Kelley: https://ziglang.org/learn/overview/
- build.rs documentation: https://doc.rust-lang.org/cargo/reference/build-scripts.html

## Next Steps

With build system integrated:
1. Stub implementations verify FFI works (return zeros)
2. Differential testing harness validates correctness
3. Actual kernel implementations replace stubs
4. Performance benchmarks validate optimizations
