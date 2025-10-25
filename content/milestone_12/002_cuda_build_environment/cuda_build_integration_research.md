# Research: CUDA Build Environment Integration with Rust

## Overview

This research examines the technical challenges and best practices for integrating CUDA compilation into Rust's Cargo build system, with focus on maintaining zero-dependency CPU-only builds while enabling optional GPU acceleration.

## Background: The Cargo-CUDA Impedance Mismatch

Rust's Cargo build system expects source files in `.rs` format compiled by `rustc`. CUDA requires `.cu` files compiled by `nvcc` (NVIDIA's CUDA compiler). These are fundamentally different compilation pipelines with different toolchains.

The challenge: integrate CUDA compilation into Cargo without breaking CPU-only builds on systems without CUDA toolkit installed.

## Build.rs: Cargo's Custom Build Script System

Cargo supports custom build logic via `build.rs` scripts that run before compilation. This is where CUDA integration happens.

### Build Script Capabilities

Build scripts can:
- Invoke external compilers (like `nvcc`)
- Set linker flags via `cargo:rustc-link-lib`
- Define conditional compilation via `cargo:rustc-cfg`
- Generate source code (FFI bindings)
- Download dependencies

### Build Script Limitations

Build scripts cannot:
- Access the compiled crate's source
- Modify the Cargo dependency tree
- Run arbitrary code in the build environment (sandboxed)
- Depend on the crate being built (circular dependency)

## CUDA Compilation Pipeline

The NVCC compilation process has multiple stages:

### Stage 1: CUDA C++ to PTX
```bash
nvcc -ptx -O3 --gpu-architecture=sm_86 kernel.cu -o kernel.ptx
```

PTX (Parallel Thread Execution) is NVIDIA's intermediate representation, analogous to LLVM IR.

### Stage 2: PTX to SASS
```bash
nvcc -cubin kernel.ptx -o kernel.cubin
```

SASS (Shader Assembly) is the architecture-specific machine code.

### Stage 3: Linking
```bash
nvcc -shared kernel.o -o libkernel.so
```

For Rust FFI, we produce a shared library that Rust can link against.

## Architecture Detection and Multi-GPU Support

Different GPU architectures require different SASS code:

| Architecture | Compute Capability | Examples |
|--------------|-------------------|----------|
| Maxwell | sm_50, sm_52 | GTX 900 series |
| Pascal | sm_60, sm_61 | GTX 1000 series |
| Volta | sm_70 | Titan V |
| Turing | sm_75 | RTX 2000 series |
| Ampere | sm_80, sm_86 | RTX 3000, A100 |
| Hopper | sm_90 | H100 |

### Fat Binary Approach

Compile kernels for multiple architectures in a single binary:

```bash
nvcc -gencode arch=compute_60,code=sm_60 \
     -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_90,code=sm_90 \
     kernel.cu -o kernel.o
```

This produces a "fat binary" with SASS for each architecture. Runtime selects the correct version.

Trade-off: larger binary size (4-5x) but works on all GPUs.

### JIT Compilation from PTX

Include PTX in the binary and JIT-compile at runtime:

```bash
nvcc -gencode arch=compute_86,code=compute_86 kernel.cu -o kernel.o
```

The `code=compute_XX` directive includes PTX. CUDA runtime JIT-compiles to SASS on first use.

Pros: smaller binary, forward compatibility
Cons: first-run compilation delay (100-500ms), JIT cache complexity

## Conditional Compilation Strategy

Engram must compile on systems without CUDA toolkit. Three approaches:

### Approach 1: Feature Flag
```toml
[features]
cuda = ["cc", "bindgen"]
```

Users enable with `cargo build --features cuda`. Clean but requires explicit opt-in.

### Approach 2: Auto-Detection
```rust
// build.rs
fn main() {
    if cuda_toolkit_available() {
        compile_cuda();
    }
}
```

Automatically uses CUDA if available. Risk: build failures if detection logic is wrong.

### Approach 3: Separate Crate
```
engram-core/        # CPU-only
engram-cuda/        # GPU acceleration
```

Clean separation but more complex dependency management.

**Engram Choice**: Approach 2 (auto-detection) with explicit `ENGRAM_FORCE_CPU` override.

## FFI Binding Generation

CUDA kernels expose C API functions that Rust calls via FFI.

### Manual Bindings
```rust
extern "C" {
    fn cuda_cosine_similarity_batch(
        query: *const f32,
        targets: *const f32,
        results: *mut f32,
        num_targets: usize,
    ) -> i32;
}
```

Simple but error-prone. Type mismatches cause UB.

### Bindgen Auto-Generation
```rust
// build.rs
bindgen::Builder::default()
    .header("cuda/api.h")
    .generate()
    .unwrap()
    .write_to_file("src/cuda/ffi.rs")
    .unwrap();
```

Generates Rust bindings from C headers. Safer but requires `libclang`.

**Engram Choice**: Bindgen for type safety.

## CUDA Unified Memory and Zero-Copy

Traditional GPU programming requires explicit memory transfers:
```c
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
kernel<<<blocks, threads>>>(d_data);
cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

CUDA Unified Memory eliminates this:
```c
cudaMallocManaged(&data, size);
kernel<<<blocks, threads>>>(data); // Automatic migration
cudaFree(data);
```

Pros: simpler code, automatic prefetching
Cons: performance unpredictability, requires Pascal or newer

**Engram Choice**: Unified Memory for Task 004, with fallback to explicit transfers.

## Link-Time Optimization and Kernel Inlining

NVCC supports device-side link-time optimization:

```bash
nvcc -dc kernel.cu -o kernel.o              # Separate compilation
nvcc -dlink kernel.o other.o -o device.o    # Device link
nvcc kernel.o other.o device.o -o lib.so    # Host link
```

Enables:
- Inlining across compilation units
- Dead code elimination
- Register allocation optimization

Trade-off: longer compile times for better runtime performance.

## Error Handling Strategy

CUDA API functions return error codes. Rust must check these:

```rust
fn cuda_check(code: cudaError_t) -> Result<(), CudaError> {
    if code == cudaSuccess {
        Ok(())
    } else {
        Err(CudaError::from(code))
    }
}
```

Critical: CUDA errors are sticky. If one kernel fails, subsequent calls may fail until `cudaGetLastError()` clears the error.

## Build Caching and Incremental Compilation

CUDA compilation is slow (5-30 seconds). Cargo's build cache helps:

```rust
// build.rs
fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_path = Path::new("cuda/kernel.cu");

    if !needs_recompile(kernel_path, &out_dir) {
        println!("cargo:rerun-if-changed=cuda/kernel.cu");
        return; // Use cached version
    }

    compile_cuda(kernel_path, &out_dir);
}
```

`cargo:rerun-if-changed` tells Cargo to only rebuild if CUDA source changed.

## Runtime GPU Capability Detection

At runtime, detect GPU capabilities before loading kernels:

```c
int device;
cudaGetDevice(&device);

cudaDeviceProp props;
cudaGetDeviceProperties(&props, device);

if (props.major < 6) {
    // Pascal or newer required
    return ERROR_UNSUPPORTED_GPU;
}

if (props.totalGlobalMem < 4ULL * 1024 * 1024 * 1024) {
    // Minimum 4GB VRAM
    return ERROR_INSUFFICIENT_VRAM;
}
```

Graceful degradation: log warning and fall back to CPU.

## Academic References

1. Nickolls, J., Buck, I., Garland, M., & Skadron, K. (2008). "Scalable parallel programming with CUDA." Queue, 6(2), 40-53.
   - Foundational CUDA programming model
   - Compilation pipeline details

2. Mei, X., & Chu, X. (2017). "Dissecting GPU memory hierarchy through microbenchmarking." IEEE Transactions on Parallel and Distributed Systems, 28(1), 72-86.
   - Memory system architecture details
   - Unified Memory performance characteristics

3. Fatica, M. (2009). "CUDA Fortran for scientists and engineers." NVIDIA Technical Report.
   - Cross-language CUDA integration
   - Build system best practices

## Industry Practices

### PyTorch CUDA Integration
PyTorch's build system detects CUDA at compile time:
```python
# setup.py
if torch.cuda.is_available():
    extra_compile_args = ['-DWITH_CUDA']
    nvcc_compile_args = ['-gencode', 'arch=compute_70,code=sm_70']
```

Falls back to CPU-only if CUDA unavailable.

### TensorFlow CUDA Build
TensorFlow uses Bazel build system with CUDA rules:
```python
# BUILD
cuda_library(
    name = "kernel",
    srcs = ["kernel.cu.cc"],
    hdrs = ["kernel.h"],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)
```

Declarative dependency management with automatic GPU architecture detection.

## Practical Insights for Engram

### Insight 1: Fat Binary vs PTX Trade-Off
For Engram, fat binary approach is correct. Binary size increase (10-20MB) is negligible compared to total application size, but JIT compilation delay (100-500ms) hurts user experience.

### Insight 2: Auto-Detection Fragility
CUDA toolkit detection logic must handle edge cases:
- CUDA installed but incompatible driver version
- Multiple CUDA versions installed
- Cross-compilation for different architectures

Robust detection with clear error messages is critical.

### Insight 3: FFI Type Safety
Bindgen-generated bindings prevent UB from type mismatches. Manual FFI is too risky for production systems.

### Insight 4: Error Handling Discipline
Every CUDA API call must be checked. Unchecked errors lead to mysterious crashes hours later when the sticky error finally manifests.

## Build Script Implementation Pattern

```rust
// build.rs
fn main() {
    // 1. Detect CUDA availability
    let cuda_available = detect_cuda();

    if !cuda_available || env::var("ENGRAM_FORCE_CPU").is_ok() {
        println!("cargo:rustc-cfg=cpu_only");
        return;
    }

    // 2. Set up CUDA compiler
    let nvcc = find_nvcc().expect("nvcc not found in PATH");
    let cuda_version = get_cuda_version(&nvcc);

    if cuda_version < (11, 0) {
        panic!("CUDA 11.0+ required, found {}.{}", cuda_version.0, cuda_version.1);
    }

    // 3. Compile kernels
    compile_kernels(&nvcc);

    // 4. Generate FFI bindings
    generate_bindings();

    // 5. Link CUDA runtime
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    // 6. Set feature flags
    println!("cargo:rustc-cfg=cuda_enabled");
}
```

## Testing Build System Correctness

Build system testing is often neglected but critical:

### Test 1: CPU-Only Build
```bash
ENGRAM_FORCE_CPU=1 cargo build
# Should succeed without CUDA toolkit
```

### Test 2: GPU Build
```bash
cargo build --release
# Should detect CUDA and compile kernels
```

### Test 3: Cross-Compilation
```bash
cargo build --target x86_64-unknown-linux-gnu
# Should handle architecture mismatch gracefully
```

## Conclusion

Integrating CUDA into Cargo requires careful build script design to handle:
- CUDA toolkit detection with graceful fallback
- Multi-GPU architecture support via fat binaries
- FFI binding generation for type safety
- Error handling discipline for CUDA API calls
- Build caching for incremental compilation

The key insight: make GPU acceleration optional. CPU-only builds must work seamlessly on systems without CUDA toolkit. Auto-detection with explicit override (`ENGRAM_FORCE_CPU`) provides flexibility for development and production.

Task 002's build system will enable all subsequent GPU work in Milestone 12 while maintaining Engram's zero-dependency CPU-only builds.
