# Perspectives: CUDA Build Environment Integration

## GPU-Acceleration-Architect Perspective

Build systems are where GPU projects die. I've seen teams spend weeks fighting NVCC-Cargo integration, only to give up and maintain separate build pipelines. This is unacceptable for production systems.

The critical insight: CUDA must be optional. Every line of GPU code should have a CPU fallback. This means the build system needs conditional compilation that actually works - not just in theory, but in practice across different systems and CUDA versions.

Fat binary versus PTX is an easy decision. Yes, fat binaries are 4-5x larger. Who cares? We're adding 20MB to a multi-hundred-MB application. But JIT compilation from PTX adds 100-500ms to first kernel launch. That's user-visible latency. Not acceptable.

Multi-architecture support is non-negotiable. We can't tell users "sorry, your GTX 1660 isn't supported, buy an RTX 3060." Compile for sm_60 (Pascal), sm_75 (Turing), sm_86 (Ampere), and sm_90 (Hopper). Runtime selects the right code. This is why fat binaries exist.

The build script must fail fast with clear errors. Don't let someone waste 30 minutes compiling only to get a cryptic linker error. Detect CUDA version up front. Require CUDA 11.0+. Abort immediately if not found.

## Systems-Architecture-Optimizer Perspective

Build system performance matters. CUDA compilation is slow - 10-30 seconds for a moderately complex kernel. Multiply that by the number of kernels and iterations during development, and you're looking at hours of compile time.

This is where Cargo's incremental compilation and build caching shine. The `cargo:rerun-if-changed` directive tells Cargo to only rebuild CUDA code when `.cu` files change. Not on every Rust change. This is the difference between 2-second and 20-second builds during development.

Link-time optimization adds another 10-20 seconds to build time but produces 10-15% faster kernels through cross-unit inlining and register allocation. For production builds, this trade-off is worth it. For development builds, skip it.

The linker path configuration is fragile. CUDA installs to different locations on different systems: `/usr/local/cuda` on Linux, `/Developer/NVIDIA/CUDA-11.x` on macOS (when supported), `C:\Program Files\NVIDIA GPU Computing Toolkit` on Windows. The build script must search common locations and fail gracefully.

What's interesting about Unified Memory is it eliminates explicit transfer overhead in the build's FFI layer. No `cudaMemcpy` calls means simpler generated bindings. But runtime performance is unpredictable - page faults on first access, automatic prefetching with variable latency. We'll need profiling to validate this approach.

## Rust-Graph-Engine-Architect Perspective

FFI safety is paramount. Manual extern declarations are too error-prone - one wrong type and you have undefined behavior. Bindgen auto-generates correct bindings from C headers, catching type mismatches at compile time.

The architecture question is where the GPU abstraction lives. Do we expose CUDA types in public APIs, or hide them behind traits? Clear answer: hide them. Public API should be `impl VectorOps`, not `CudaVectorOps`. GPU is an implementation detail.

This means the build system must support conditional compilation at the module level:
```rust
#[cfg(cuda_enabled)]
mod cuda;

#[cfg(not(cuda_enabled))]
mod cpu_only;
```

Both modules implement the same trait. User code never knows which backend is used.

Error handling across the FFI boundary needs discipline. CUDA functions return error codes; Rust uses `Result`. The bridge layer converts codes to typed errors:
```rust
enum CudaError {
    OutOfMemory,
    InvalidValue,
    DeviceUnavailable,
}
```

This enables proper error propagation without panics.

## Verification-Testing-Lead Perspective

Build system correctness is testable. We need CI matrix testing:
- Ubuntu 20.04 with CUDA 11.0
- Ubuntu 22.04 with CUDA 11.8
- Ubuntu 22.04 with CUDA 12.0
- Ubuntu 22.04 without CUDA (CPU-only build)

Each configuration must build successfully. CPU-only build is critical - it validates our conditional compilation.

The fat binary correctness requires runtime testing. CI must run on actual hardware with different GPU generations:
- Pascal (GTX 1060): validates sm_60 code path
- Turing (RTX 2060): validates sm_75 code path
- Ampere (RTX 3060): validates sm_86 code path

If runtime selects the wrong architecture code, performance degrades or crashes. This happened to me once - built for sm_75 but runtime tried to load on sm_60. Silent performance degradation until we profiled.

The build script itself needs unit testing. Mock CUDA detection to test error paths:
- CUDA not installed: should gracefully fall back
- CUDA installed but old version: should error with clear message
- NVCC in PATH but CUDA libraries missing: should detect and error

Integration testing validates the full pipeline: compile with CUDA enabled, run tests, verify GPU kernels executed (not CPU fallback).
