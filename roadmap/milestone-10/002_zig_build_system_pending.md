# Task 002: Zig Build System

**Duration:** 1 day
**Status:** Pending
**Dependencies:** 001 (Profiling Infrastructure)

## Objectives

Establish Zig build system integrated with cargo workflow, providing C-compatible FFI bindings for performance kernels. The build system must support both development (with Zig kernels) and fallback (Rust-only) modes.

1. **build.zig configuration** - Static library compilation with C ABI exports
2. **Cargo integration** - Build Zig library during cargo build process
3. **FFI bindings** - Rust declarations for Zig kernel functions
4. **Feature flags** - Conditional compilation for Zig vs. Rust implementations

## Dependencies

- Task 001 (Profiling Infrastructure) - Hotspots identified for kernel candidates

## Deliverables

### Files to Create

1. `/zig/build.zig` - Zig build configuration
   - Static library target: libengram_kernels.a
   - C ABI export for kernel functions
   - Debug and release optimization modes

2. `/zig/src/ffi.zig` - FFI function declarations
   - Export signatures for vector similarity, spreading, decay
   - C-compatible types (no Zig-specific features)

3. `/build.rs` - Cargo build script
   - Invoke zig build during cargo build
   - Link libengram_kernels.a with Rust binary
   - Detect Zig compiler availability

4. `/src/zig_kernels/mod.rs` - Rust FFI module
   - Extern declarations for Zig functions
   - Safe Rust wrappers with bounds checking
   - Feature-gated compilation

5. `/scripts/build_with_zig.sh` - Development build script
   - Set ZIG_AVAILABLE=1 environment variable
   - cargo build --features zig-kernels

### Files to Modify

1. `/Cargo.toml` - Add Zig feature flag
   ```toml
   [features]
   default = []
   zig-kernels = []

   [build-dependencies]
   cc = "1.0"
   ```

2. `/.gitignore` - Ignore Zig build artifacts
   ```
   /zig/zig-cache/
   /zig/zig-out/
   ```

## Acceptance Criteria

1. `cargo build --features zig-kernels` successfully builds with Zig library
2. `cargo build` (without feature) compiles without Zig dependency
3. `./scripts/build_with_zig.sh` produces binary with Zig kernels linked
4. FFI smoke test calls Zig function from Rust and verifies return value
5. Build fails gracefully with clear error if Zig not installed

## Implementation Guidance

### build.zig Structure

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

    // Enable C ABI exports
    lib.bundle_compiler_rt = true;

    b.installArtifact(lib);

    // Tests
    const tests = b.addTest(.{
        .root_source_file = .{ .path = "src/ffi.zig" },
        .target = target,
        .optimize = optimize,
    });

    const test_step = b.step("test", "Run Zig tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
```

### FFI Function Signatures

All exported functions use C-compatible types and explicit memory ownership:

```zig
// ffi.zig
const std = @import("std");

// Vector similarity kernel (stub for now)
export fn engram_vector_similarity(
    query: [*]const f32,
    candidates: [*]const f32,
    scores: [*]f32,
    query_len: usize,
    num_candidates: usize,
) void {
    // Stub implementation - returns zeros
    for (0..num_candidates) |i| {
        scores[i] = 0.0;
    }
}

// Activation spreading kernel (stub)
export fn engram_spread_activation(
    adjacency: [*]const u32,
    weights: [*]const f32,
    activations: [*]f32,
    num_nodes: usize,
    num_edges: usize,
    iterations: u32,
) void {
    // Stub implementation
    _ = adjacency;
    _ = weights;
    _ = activations;
    _ = num_nodes;
    _ = num_edges;
    _ = iterations;
}

// Decay function kernel (stub)
export fn engram_apply_decay(
    strengths: [*]f32,
    ages_seconds: [*]const u64,
    num_memories: usize,
) void {
    // Stub implementation
    _ = strengths;
    _ = ages_seconds;
    _ = num_memories;
}
```

### Cargo Build Script

```rust
// build.rs
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only build Zig library if feature enabled
    #[cfg(feature = "zig-kernels")]
    {
        build_zig_library();
    }

    println!("cargo:rerun-if-changed=zig/");
}

#[cfg(feature = "zig-kernels")]
fn build_zig_library() {
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

    // Link the library
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let zig_lib = out_dir.join("../../../zig/zig-out/lib");

    println!("cargo:rustc-link-search=native={}", zig_lib.display());
    println!("cargo:rustc-link-lib=static=engram_kernels");
}
```

### Rust FFI Wrapper

```rust
// src/zig_kernels/mod.rs

#[cfg(feature = "zig-kernels")]
mod ffi {
    use std::os::raw::c_void;

    extern "C" {
        pub fn engram_vector_similarity(
            query: *const f32,
            candidates: *const f32,
            scores: *mut f32,
            query_len: usize,
            num_candidates: usize,
        );

        pub fn engram_spread_activation(
            adjacency: *const u32,
            weights: *const f32,
            activations: *mut f32,
            num_nodes: usize,
            num_edges: usize,
            iterations: u32,
        );

        pub fn engram_apply_decay(
            strengths: *mut f32,
            ages_seconds: *const u64,
            num_memories: usize,
        );
    }
}

#[cfg(feature = "zig-kernels")]
pub fn vector_similarity(query: &[f32], candidates: &[f32], num_candidates: usize) -> Vec<f32> {
    assert_eq!(candidates.len() % num_candidates, 0);
    let dim = query.len();

    let mut scores = vec![0.0_f32; num_candidates];

    unsafe {
        ffi::engram_vector_similarity(
            query.as_ptr(),
            candidates.as_ptr(),
            scores.as_mut_ptr(),
            dim,
            num_candidates,
        );
    }

    scores
}

#[cfg(not(feature = "zig-kernels"))]
pub fn vector_similarity(query: &[f32], candidates: &[f32], num_candidates: usize) -> Vec<f32> {
    // Fallback to Rust implementation
    crate::embedding::batch_cosine_similarity(query, candidates, num_candidates)
}
```

## Testing Approach

1. **Build system validation**
   - cargo build --features zig-kernels succeeds
   - cargo build (without feature) succeeds
   - Verify libengram_kernels.a is created in zig/zig-out/lib/

2. **FFI smoke test**
   ```rust
   #[test]
   #[cfg(feature = "zig-kernels")]
   fn test_zig_ffi_smoke() {
       let query = vec![1.0, 0.0, 0.0];
       let candidates = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
       let scores = vector_similarity(&query, &candidates, 2);
       assert_eq!(scores.len(), 2);
   }
   ```

3. **Feature flag verification**
   - With zig-kernels: cfg!(feature = "zig-kernels") is true
   - Without: Rust fallback paths are used

## Integration Points

- **Task 003 (Differential Testing)** - Uses FFI wrappers to compare Zig vs. Rust
- **Task 005-007 (Kernels)** - Implement actual kernel logic in ffi.zig

## Notes

- Pin Zig to 0.13.0 stable to avoid compiler instability
- Use -Doptimize=ReleaseFast for production kernels
- Consider adding zig fmt check to CI for code style
- Document Zig installation in README.md
