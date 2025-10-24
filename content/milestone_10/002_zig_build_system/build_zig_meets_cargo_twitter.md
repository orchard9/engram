# Twitter Thread: build.zig Meets cargo build - FFI Without the Pain

## Tweet 1/8
You profiled your Rust codebase. Three functions consume 60% of runtime. SIMD could make them 4x faster.

But writing SIMD in Rust is painful.

What if you could write those functions in Zig, get automatic SIMD, and call them from Rust with zero overhead?

Here's how we integrated Zig kernels into Engram:

## Tweet 2/8
Why Zig for performance kernels?

- Explicit control (no hidden allocations)
- @Vector type auto-lowers to AVX2/NEON
- C ABI by default (export fn just works)
- Comptime metaprogramming (zero runtime cost)

Not "better than Rust" - different tool, different tradeoffs.

## Tweet 3/8
The key insight: Both Rust and Zig speak C fluently.

Zig:
```zig
export fn engram_vector_similarity(
    query: [*]const f32,
    scores: [*]f32,
    len: usize,
) void { }
```

Rust:
```rust
extern "C" {
    fn engram_vector_similarity(
        query: *const f32,
        scores: *mut f32,
        len: usize,
    );
}
```

No serialization. Just pointers.

## Tweet 4/8
Zero-copy FFI design:

BAD: Serialize → JSON → parse → process → serialize → JSON → parse
(adds 100μs+ overhead)

GOOD: Pass pointers → Zig writes to Rust's buffer
(zero overhead)

Caller allocates, callee populates. Like C libraries - battle-tested for decades.

## Tweet 5/8
Build system integration via build.rs:

```rust
#[cfg(feature = "zig-kernels")]
{
    Command::new("zig")
        .args(&["build", "-Doptimize=ReleaseFast"])
        .status()
        .expect("Zig build failed");

    println!("cargo:rustc-link-lib=static=engram_kernels");
}
```

`cargo build --features zig-kernels` → builds Zig → links statically → single binary.

## Tweet 6/8
Critical design: Zig kernels are OPTIONAL.

```rust
#[cfg(feature = "zig-kernels")]
fn vector_similarity() {
    // Fast Zig path
}

#[cfg(not(feature = "zig-kernels"))]
fn vector_similarity() {
    // Rust fallback
}
```

Same API. Two implementations. Calling code doesn't care.

Without feature flag, builds without Zig dependency.

## Tweet 7/8
Safety strategy: Keep unsafe blocks small, validate in Rust.

```rust
// Validation in safe Rust
assert_eq!(query.len(), candidate.len());

// Unsafe block minimized
unsafe {
    ffi::engram_vector_similarity(
        query.as_ptr(),
        scores.as_mut_ptr(),
        query.len(),
    );
}
```

10 lines unsafe, 30 lines safe wrapper. This ratio matters.

## Tweet 8/8
Results:
- cargo build works without Zig (Rust fallback)
- cargo build --features zig-kernels enables optimizations
- Zero serialization overhead (pass pointers directly)
- 25% faster vector similarity, 35% faster activation spreading

Clean boundaries. Optional optimization. Zero-copy FFI.

C ABI: still the lingua franca.
