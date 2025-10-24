# build.zig Meets cargo build: FFI Without the Pain

You've identified your performance bottlenecks. Profiling showed that three functions consume 60% of your runtime. You know SIMD vectorization could make them 4x faster.

But your codebase is Rust. And writing SIMD intrinsics in Rust is... not fun.

What if you could write those three hot functions in Zig, get automatic SIMD optimization, and seamlessly call them from Rust with zero overhead?

That's exactly what we built for Engram.

## Why Not Just Use Rust?

Rust is fantastic for building safe, concurrent systems. But for low-level performance kernels, Zig has advantages:

**Explicit control:** No hidden allocations, no hidden control flow. If you write a loop, that's exactly what compiles.

**Comptime metaprogramming:** Code generation at compile time with no runtime cost. Think C++ templates, but sane.

**SIMD primitives:** `@Vector(8, f32)` auto-lowers to AVX2 on x86_64, NEON on ARM64. No manual intrinsics.

**C ABI by default:** `export fn` just works. No `#[no_mangle]` dance, no wrapper generation.

For Engram's vector similarity kernel (calculating cosine distance over 768-dimensional embeddings), these advantages matter. We're doing millions of floating-point operations - explicit control over SIMD makes a huge difference.

## The C ABI Common Ground

The key insight: Both Rust and Zig speak C fluently. We don't need custom serialization or FFI wrapper generators. Just export C-compatible functions.

Zig side:

```zig
export fn engram_vector_similarity(
    query: [*]const f32,
    candidates: [*]const f32,
    scores: [*]f32,
    query_len: usize,
    num_candidates: usize,
) void {
    // Implementation with SIMD
    for (0..num_candidates) |i| {
        const candidate_start = i * query_len;
        scores[i] = cosineSimilarity(
            query[0..query_len],
            candidates[candidate_start..][0..query_len],
        );
    }
}
```

Rust side:

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

No serialization. No wrapper code. Just pointers and sizes.

## Zero-Copy Design

The worst FFI mistake: copying data at the boundary.

Bad approach (adds 100μs+ of overhead):

```rust
// Serialize to JSON, call Zig, deserialize result
let json = serde_json::to_string(&data)?;
let result_json = unsafe { zig_function(json.as_ptr()) };
let result = serde_json::from_str(result_json)?;
```

Good approach (zero overhead):

```rust
// Allocate output buffer, let Zig write to it directly
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
// scores now contains results, no copy
```

Caller allocates, callee populates. This is how C libraries work - battle-tested for decades.

## Build System Integration

Here's the tricky part: making `cargo build` work seamlessly.

Our `build.rs` script:

```rust
fn main() {
    #[cfg(feature = "zig-kernels")]
    {
        // Check if Zig is available
        let zig_available = Command::new("zig")
            .arg("version")
            .output()
            .is_ok();

        if !zig_available {
            panic!("Zig compiler not found. Install Zig 0.13.0 or disable zig-kernels feature.");
        }

        // Build Zig library
        let status = Command::new("zig")
            .args(&["build", "-Doptimize=ReleaseFast"])
            .current_dir("zig")
            .status()
            .expect("Failed to build Zig library");

        if !status.success() {
            panic!("Zig build failed");
        }

        // Link the static library
        println!("cargo:rustc-link-search=native=zig/zig-out/lib");
        println!("cargo:rustc-link-lib=static=engram_kernels");
    }

    println!("cargo:rerun-if-changed=zig/");
}
```

When you run `cargo build --features zig-kernels`, this script:
1. Checks if Zig is installed
2. Runs `zig build` to produce `libengram_kernels.a`
3. Links the static library into your Rust binary

Without the feature flag, Rust implementations are used instead. Zero Zig dependency required.

## Feature Flags for Graceful Fallback

This is critical: Zig kernels must be optional.

```toml
[features]
default = []
zig-kernels = []
```

Our implementation:

```rust
#[cfg(feature = "zig-kernels")]
pub fn vector_similarity(query: &[f32], candidates: &[Vec<f32>]) -> Vec<f32> {
    // Call Zig kernel (fast path)
    let candidates_flat = flatten_candidates(candidates);
    let mut scores = vec![0.0; candidates.len()];
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

#[cfg(not(feature = "zig-kernels"))]
pub fn vector_similarity(query: &[f32], candidates: &[Vec<f32>]) -> Vec<f32> {
    // Pure Rust fallback
    candidates
        .iter()
        .map(|c| cosine_similarity_rust(query, c))
        .collect()
}
```

Same API, two implementations. The calling code doesn't care which one is used.

This enables:
- **Development without Zig:** Contributors can build with pure Rust
- **Platform-specific optimization:** Enable Zig on x86_64, fall back on WebAssembly
- **Debugging:** Disable Zig to isolate performance vs correctness bugs

## Safe Wrappers Around Unsafe FFI

FFI is inherently unsafe. You're promising to uphold invariants the compiler can't verify:
- Pointers are valid
- Slice lengths are correct
- No aliasing between inputs and outputs

Our strategy: Keep unsafe blocks small, do validation in safe Rust.

```rust
pub fn vector_similarity(query: &[f32], candidates: &[Vec<f32>]) -> Vec<f32> {
    // Validation in safe Rust
    for candidate in candidates {
        assert_eq!(
            candidate.len(),
            query.len(),
            "All candidates must have same dimension as query"
        );
    }

    let candidates_flat: Vec<f32> = candidates
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();

    let mut scores = vec![0.0_f32; candidates.len()];

    // Unsafe block minimized
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

The unsafe block is 10 lines. The safe wrapper is 30 lines. This ratio matters.

## Zig's Build System

Zig's `build.zig` produces the static library that Rust links against:

```zig
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

    b.installArtifact(lib);
}
```

Running `zig build -Doptimize=ReleaseFast` produces `libengram_kernels.a` in `zig-out/lib/`. Rust's build.rs script links this library.

Static linking keeps deployment simple: one binary, no runtime dependencies.

## What We Achieved

With this integration:
- `cargo build` works without Zig installed (falls back to Rust)
- `cargo build --features zig-kernels` enables optimized paths
- FFI has zero serialization overhead (pass pointers directly)
- Zig kernels are drop-in replacements (same API as Rust implementations)

The result: 25% faster vector similarity, 35% faster activation spreading, with clean boundaries between languages.

## Key Takeaways

1. **C ABI as common ground:** Both Rust and Zig speak C natively, no custom serialization needed
2. **Zero-copy design:** Pass pointers to pre-allocated buffers, avoid copying at FFI boundary
3. **Feature flags for flexibility:** Make optimizations optional, not required
4. **Safe wrappers minimize unsafe:** Validation in Rust, raw FFI in small blocks
5. **Static linking for simplicity:** Single binary, no runtime dependency management

## Try It Yourself

To integrate Zig kernels in your Rust project:

1. Add feature flag to Cargo.toml:
```toml
[features]
zig-kernels = []

[build-dependencies]
cc = "1.0"
```

2. Create build.rs:
```rust
#[cfg(feature = "zig-kernels")]
fn main() {
    Command::new("zig")
        .args(&["build", "-Doptimize=ReleaseFast"])
        .current_dir("zig")
        .status()
        .expect("Zig build failed");

    println!("cargo:rustc-link-search=native=zig/zig-out/lib");
    println!("cargo:rustc-link-lib=static=your_lib_name");
}
```

3. Export C-compatible functions from Zig
4. Declare extern "C" bindings in Rust
5. Wrap unsafe FFI in safe Rust functions

Start with stub implementations (return zeros) to verify the build system works. Then replace stubs with real optimized kernels.

The build complexity is worth it when you need those last 20-40% of performance that Rust can't easily give you.
