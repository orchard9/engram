# Making CUDA and Cargo Play Nice: A Build System Love Story

Here's a problem: Rust's Cargo build system expects `.rs` files compiled by `rustc`. CUDA requires `.cu` files compiled by `nvcc`. These are fundamentally different compilation pipelines.

Oh, and your system must build successfully on machines without CUDA installed. Because GPU acceleration is optional, not mandatory.

Welcome to the CUDA-Cargo integration challenge.

## The Problem: Two Worlds Colliding

Cargo is designed for Rust. It knows about Rust source files, Rust dependencies, and `rustc` compilation. It doesn't know about CUDA.

NVCC (NVIDIA's CUDA compiler) is designed for CUDA C++. It knows about GPU architectures, kernel launch syntax, and device memory. It doesn't know about Rust.

The bridge between these worlds is `build.rs` - Cargo's custom build script system.

## Build.rs: The Integration Layer

Cargo runs `build.rs` before compilation, allowing custom build logic. This is where we invoke `nvcc`, generate FFI bindings, and set linker flags.

Here's the architecture:

```rust
// build.rs
fn main() {
    // 1. Detect CUDA availability
    let cuda_available = detect_cuda_toolkit();

    if !cuda_available {
        println!("cargo:rustc-cfg=cpu_only");
        return; // Build succeeds, CPU-only mode
    }

    // 2. Compile CUDA kernels with nvcc
    compile_cuda_kernels();

    // 3. Generate Rust FFI bindings
    generate_ffi_bindings();

    // 4. Link CUDA runtime library
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-cfg=cuda_enabled");
}
```

This script runs before `rustc` sees any Rust code. It compiles CUDA kernels to a shared library, generates Rust bindings for the CUDA API, and tells Cargo to link against the CUDA runtime.

## The Fat Binary Problem: Supporting Multiple GPU Generations

Different GPUs have different instruction sets. An RTX 3060 (Ampere architecture) can't run code compiled for an H100 (Hopper architecture). And vice versa.

The solution: fat binaries. Compile kernels for multiple architectures in one binary:

```bash
nvcc -gencode arch=compute_60,code=sm_60 \
     -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_90,code=sm_90 \
     kernel.cu -o kernel.o
```

Each `-gencode` directive produces SASS (machine code) for a specific architecture:
- `sm_60`: Pascal (GTX 1000 series)
- `sm_75`: Turing (RTX 2000 series)
- `sm_86`: Ampere (RTX 3000 series, A100)
- `sm_90`: Hopper (H100)

The CUDA runtime automatically selects the correct version at runtime based on the actual GPU.

Cost: binary size increases 4-5x. For Engram, that's 20MB additional. Negligible compared to total application size.

Benefit: works on all GPUs from GTX 1060 to H100. No user complaints about "unsupported GPU."

## The Alternative: PTX JIT Compilation

The alternative to fat binaries is including PTX (intermediate representation) and JIT-compiling at runtime:

```bash
nvcc -gencode arch=compute_86,code=compute_86 kernel.cu -o kernel.o
```

The `code=compute_86` directive includes PTX instead of SASS. The CUDA runtime compiles PTX to SASS on first use.

Pros:
- Smaller binary size
- Forward compatibility with future GPUs

Cons:
- 100-500ms first-run compilation delay
- JIT cache complexity
- Potential runtime compilation failures

For Engram, we choose fat binaries. User experience beats binary size. The 100-500ms JIT delay is user-visible, especially during startup when multiple kernels compile simultaneously.

## FFI Binding Generation: Type Safety Across Languages

CUDA kernels expose C API functions. Rust calls them via FFI (Foreign Function Interface).

Manual bindings are error-prone:

```rust
extern "C" {
    fn cuda_cosine_similarity(
        query: *const f32,
        targets: *const f32,
        results: *mut f32,
        count: usize,
    ) -> i32;
}
```

One type mismatch (say, `i32` instead of `usize`) and you have undefined behavior. Silent corruption or crashes.

The solution: Bindgen auto-generates bindings from C headers:

```rust
// build.rs
bindgen::Builder::default()
    .header("cuda/api.h")
    .parse_callbacks(Box::new(bindgen::CargoCallbacks))
    .generate()
    .expect("Unable to generate bindings")
    .write_to_file("src/cuda/ffi.rs")
    .expect("Couldn't write bindings");
```

Bindgen parses the C header, extracts function signatures, and generates correct Rust FFI declarations. Type mismatches become compile errors instead of runtime UB.

## Conditional Compilation: Making GPU Optional

Engram must build on systems without CUDA. The build script detects CUDA availability and sets conditional compilation flags:

```rust
// build.rs
if cuda_available {
    println!("cargo:rustc-cfg=cuda_enabled");
} else {
    println!("cargo:rustc-cfg=cpu_only");
}
```

Rust code uses these flags for conditional compilation:

```rust
#[cfg(cuda_enabled)]
mod cuda_impl {
    pub fn cosine_similarity(...) {
        // GPU implementation
    }
}

#[cfg(cpu_only)]
mod cpu_impl {
    pub fn cosine_similarity(...) {
        // CPU SIMD implementation
    }
}

pub use {
    #[cfg(cuda_enabled)]
    cuda_impl::cosine_similarity,

    #[cfg(cpu_only)]
    cpu_impl::cosine_similarity,
};
```

User code calls `cosine_similarity()` without knowing which backend is active. GPU is an implementation detail.

## CUDA Detection: The Devil in the Details

Detecting CUDA sounds simple. Check if `nvcc` is in PATH. Done, right?

Wrong. Here's what can go wrong:

1. **NVCC installed but incompatible driver**: CUDA 11.8 requires driver 520+. If driver is 510, compilation succeeds but runtime fails.

2. **Multiple CUDA versions installed**: System has CUDA 10.2 and 11.8. Which does `nvcc` use? Depends on PATH order.

3. **CUDA installed but libraries missing**: NVCC found but `libcudart.so` not in linker search path.

4. **Cross-compilation**: Building on x86_64 for aarch64. CUDA for x86_64 won't work on aarch64.

Robust detection requires checking:
- NVCC version >= 11.0
- Driver version compatible with CUDA version
- CUDA libraries present and linkable
- Architecture compatibility for cross-compilation

And if any check fails, provide clear error messages:

```
error: CUDA 11.0+ required, found 10.2
       install CUDA 11.0+ from https://developer.nvidia.com/cuda-downloads
       or set ENGRAM_FORCE_CPU=1 to build without GPU support
```

## Build Caching: Making Iteration Fast

CUDA compilation is slow. A moderately complex kernel takes 10-30 seconds to compile. During development, this becomes painful.

Cargo's build cache helps:

```rust
// build.rs
println!("cargo:rerun-if-changed=cuda/kernel.cu");
```

This tells Cargo to only rebuild CUDA code when `kernel.cu` changes. Rust changes don't trigger CUDA recompilation.

The result: 2-second builds when iterating on Rust code, 20-second builds when changing CUDA kernels. This is the difference between flow state and context-switching hell.

## Error Handling: CUDA's Sticky Error Model

CUDA has a peculiar error model: errors are sticky. If one API call fails, subsequent calls may also fail until you call `cudaGetLastError()` to clear the error state.

This is dangerous for Rust's error handling:

```rust
// BAD: error not cleared
fn operation_a() -> Result<(), CudaError> {
    let result = unsafe { cuda_function_a(...) };
    if result != cudaSuccess {
        return Err(CudaError::from(result));
        // Error state still set!
    }
    Ok(())
}

fn operation_b() -> Result<(), CudaError> {
    let result = unsafe { cuda_function_b(...) };
    // This might fail because of operation_a's error
    if result != cudaSuccess {
        return Err(CudaError::from(result));
    }
    Ok(())
}
```

The fix: always check and clear:

```rust
fn cuda_check(result: cudaError_t) -> Result<(), CudaError> {
    if result != cudaSuccess {
        let error = unsafe { cudaGetLastError() }; // Clear error
        Err(CudaError::from(error))
    } else {
        Ok(())
    }
}
```

Every CUDA API call goes through `cuda_check()`. Errors are caught and cleared immediately.

## The Build Matrix: Testing All Configurations

Build system correctness requires testing multiple configurations:

| Configuration | CUDA Version | Driver | Expected |
|--------------|--------------|---------|----------|
| Ubuntu 20.04 | None | None | CPU-only build succeeds |
| Ubuntu 22.04 | 11.0 | 520+ | GPU build succeeds |
| Ubuntu 22.04 | 11.8 | 520+ | GPU build succeeds |
| Ubuntu 22.04 | 12.0 | 525+ | GPU build succeeds |
| Ubuntu 20.04 | 10.2 | 510 | Error: CUDA too old |

Each configuration runs in CI. If any fails, the build system is broken.

## Real-World CUDA Detection Code

Here's production-quality CUDA detection:

```rust
// build.rs
fn detect_cuda() -> Option<CudaInfo> {
    // 1. Find nvcc
    let nvcc = which::which("nvcc").ok()?;

    // 2. Get CUDA version
    let version_output = Command::new(&nvcc)
        .arg("--version")
        .output()
        .ok()?;
    let version_str = String::from_utf8_lossy(&version_output.stdout);
    let version = parse_cuda_version(&version_str)?;

    if version < (11, 0) {
        eprintln!("CUDA 11.0+ required, found {}.{}", version.0, version.1);
        return None;
    }

    // 3. Check driver compatibility
    let driver_version = get_driver_version()?;
    if !is_driver_compatible(version, driver_version) {
        eprintln!("CUDA {} requires driver {}+, found {}",
                  version.0, min_driver(version), driver_version);
        return None;
    }

    // 4. Find CUDA libraries
    let lib_path = find_cuda_libs()?;

    Some(CudaInfo { version, nvcc, lib_path })
}
```

This checks version compatibility, driver compatibility, and library availability before proceeding.

## Conclusion: Build Systems Are Infrastructure

Build system integration isn't glamorous. But it's foundational. A broken build system means nobody can compile your code. A slow build system means developer productivity tanks.

For Engram's GPU acceleration, the build system must:
1. Work on systems without CUDA (CPU-only mode)
2. Support multiple GPU architectures (fat binaries)
3. Generate type-safe FFI bindings (Bindgen)
4. Cache compilation results (incremental builds)
5. Fail fast with clear errors (CUDA detection)

Task 002's build system enables all subsequent GPU work in Milestone 12. It's the foundation everything else builds on.

When the build system works, developers don't think about it. It just works. That's the goal.
