# Task 002: CUDA Kernel Development Environment Setup

**Status**: Pending
**Estimated Duration**: 2 days
**Priority**: Critical (enables all GPU development)
**Owner**: Systems Engineer

## Objective

Establish build system for CUDA code with Rust FFI integration, enabling seamless compilation of `.cu` files alongside Rust code. Ensure builds work on systems without CUDA toolkit (graceful no-op fallback).

## Background

CUDA kernel development requires coordinating multiple toolchains: `nvcc` for GPU code, `rustc` for host code, and linker integration for FFI. This task creates the infrastructure that all subsequent GPU work depends on.

Critical requirement: systems without CUDA toolkit must still compile successfully, with GPU features disabled at runtime (not compile-time).

## Deliverables

1. **CUDA Compilation in Cargo Build**
   - `build.rs` script that detects CUDA toolkit
   - Compiles `.cu` files if CUDA available
   - Generates no-op stubs if CUDA unavailable
   - Links CUDA runtime library correctly

2. **FFI Bindings for CUDA Runtime**
   - Safe Rust wrappers for CUDA API
   - Error handling that converts `cudaError_t` to `Result`
   - Device query and capability detection
   - Memory allocation primitives

3. **Minimal Kernel Validation**
   - Trivial "hello world" kernel that can launch
   - Device query test (enumerates GPUs)
   - Memory allocation/deallocation test
   - Proves toolchain integration works

4. **CI Integration**
   - Build succeeds on systems without CUDA
   - Tests skip GPU tests when CUDA unavailable
   - Optional CI runner with GPU for validation

## Technical Specification

### Build Script Architecture

```rust
// engram-core/build.rs

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/");

    // Detect CUDA toolkit
    let cuda_available = detect_cuda_toolkit();

    if cuda_available {
        compile_cuda_kernels();
        link_cuda_runtime();
    } else {
        println!("cargo:warning=CUDA toolkit not found, GPU acceleration disabled");
        generate_fallback_stubs();
    }
}

fn detect_cuda_toolkit() -> bool {
    // Check for nvcc in PATH
    let nvcc_result = Command::new("nvcc")
        .arg("--version")
        .output();

    if nvcc_result.is_err() {
        return false;
    }

    // Check for CUDA_PATH environment variable
    if env::var("CUDA_PATH").is_ok() {
        return true;
    }

    // Check standard installation paths
    let standard_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
    ];

    for path in &standard_paths {
        if PathBuf::from(path).exists() {
            return true;
        }
    }

    false
}

fn compile_cuda_kernels() {
    let cuda_files = [
        "cuda/kernels/cosine_similarity.cu",
        "cuda/kernels/spreading_matmul.cu",
        "cuda/kernels/hnsw_scoring.cu",
    ];

    let out_dir = env::var("OUT_DIR").unwrap();

    for cu_file in &cuda_files {
        println!("cargo:rerun-if-changed={}", cu_file);

        let output = Command::new("nvcc")
            .arg("-O3")
            .arg("-arch=sm_60") // Maxwell+, covers GTX 1060+
            .arg("-std=c++14")
            .arg("--compiler-options=-fPIC")
            .arg("-c")
            .arg(cu_file)
            .arg("-o")
            .arg(format!("{}/{}.o", out_dir, cu_file.replace("/", "_")))
            .output()
            .expect("Failed to compile CUDA kernel");

        if !output.status.success() {
            panic!("CUDA compilation failed:\n{}",
                   String::from_utf8_lossy(&output.stderr));
        }
    }

    // Link all object files into static library
    let ar_output = Command::new("ar")
        .arg("rcs")
        .arg(format!("{}/libengram_cuda.a", out_dir))
        .args(&cuda_files.iter()
            .map(|f| format!("{}/{}.o", out_dir, f.replace("/", "_")))
            .collect::<Vec<_>>())
        .output()
        .expect("Failed to create static library");

    if !ar_output.status.success() {
        panic!("Static library creation failed");
    }

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=engram_cuda");
}

fn link_cuda_runtime() {
    // Link CUDA runtime library
    let cuda_path = env::var("CUDA_PATH")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
}

fn generate_fallback_stubs() {
    // Generate no-op implementations for GPU functions
    let stub_code = r#"
    // This file is auto-generated when CUDA is unavailable
    // All GPU operations return errors indicating unavailability

    #[no_mangle]
    pub extern "C" fn cuda_initialize() -> i32 { -1 }

    #[no_mangle]
    pub extern "C" fn cuda_batch_cosine_similarity(
        query: *const f32,
        targets: *const f32,
        results: *mut f32,
        count: usize,
    ) -> i32 { -1 }
    "#;

    let out_dir = env::var("OUT_DIR").unwrap();
    std::fs::write(
        format!("{}/cuda_stubs.rs", out_dir),
        stub_code
    ).expect("Failed to write stub file");
}
```

### CUDA FFI Bindings

```rust
// engram-core/src/compute/cuda/ffi.rs

use std::ffi::c_void;

/// CUDA error codes
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaError {
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    NoDevice = 100,
    // ... other error codes
}

impl CudaError {
    pub fn from_raw(code: i32) -> Self {
        match code {
            0 => CudaError::Success,
            1 => CudaError::InvalidValue,
            2 => CudaError::OutOfMemory,
            3 => CudaError::NotInitialized,
            100 => CudaError::NoDevice,
            _ => CudaError::InvalidValue, // Unknown error
        }
    }

    pub fn to_result(self) -> Result<(), CudaError> {
        match self {
            CudaError::Success => Ok(()),
            err => Err(err),
        }
    }
}

/// CUDA device properties
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CudaDeviceProperties {
    pub name: [u8; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_threads_dim: [i32; 3],
    pub max_grid_size: [i32; 3],
    pub clock_rate: i32,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub managed_memory: i32, // boolean: 1 if supported
}

extern "C" {
    // Device management
    pub fn cudaGetDeviceCount(count: *mut i32) -> i32;
    pub fn cudaGetDeviceProperties(
        prop: *mut CudaDeviceProperties,
        device: i32,
    ) -> i32;
    pub fn cudaSetDevice(device: i32) -> i32;

    // Memory management
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(devPtr: *mut c_void) -> i32;
    pub fn cudaMallocManaged(devPtr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
    ) -> i32;

    // Synchronization
    pub fn cudaDeviceSynchronize() -> i32;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;

    // Error handling
    pub fn cudaGetLastError() -> i32;
    pub fn cudaGetErrorString(error: i32) -> *const i8;
}

/// Safe wrapper for CUDA device count query
pub fn get_device_count() -> Result<i32, CudaError> {
    let mut count: i32 = 0;
    unsafe {
        let result = cudaGetDeviceCount(&mut count);
        CudaError::from_raw(result).to_result()?;
    }
    Ok(count)
}

/// Safe wrapper for CUDA device properties query
pub fn get_device_properties(device_id: i32) -> Result<CudaDeviceProperties, CudaError> {
    let mut props = unsafe { std::mem::zeroed() };
    unsafe {
        let result = cudaGetDeviceProperties(&mut props, device_id);
        CudaError::from_raw(result).to_result()?;
    }
    Ok(props)
}
```

### Minimal Validation Kernel

```cuda
// engram-core/cuda/kernels/validation.cu

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_kernel(int* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = idx * 2; // Simple computation
}

extern "C" {

int cuda_validate_environment() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        return -1;
    }
    if (device_count == 0) {
        return -2;
    }

    // Try allocating and freeing memory
    void* test_ptr;
    err = cudaMalloc(&test_ptr, 1024);
    if (err != cudaSuccess) {
        return -3;
    }
    cudaFree(test_ptr);

    return 0; // Success
}

int cuda_test_kernel_launch(int* h_output, int size) {
    int* d_output;
    cudaError_t err = cudaMalloc((void**)&d_output, size * sizeof(int));
    if (err != cudaSuccess) {
        return -1;
    }

    // Launch kernel
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    hello_kernel<<<grid_size, block_size>>>(d_output);

    // Copy results back
    err = cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);

    return (err == cudaSuccess) ? 0 : -2;
}

} // extern "C"
```

## Acceptance Criteria

1. **Build System**
   - [ ] `cargo build` succeeds on system with CUDA toolkit
   - [ ] `cargo build` succeeds on system without CUDA toolkit
   - [ ] GPU features automatically disabled when CUDA unavailable
   - [ ] No warnings or errors in build output

2. **FFI Integration**
   - [ ] Can query CUDA device count from Rust
   - [ ] Can retrieve device properties from Rust
   - [ ] CUDA errors convert to Rust `Result` types
   - [ ] Memory allocation/deallocation works via FFI

3. **Kernel Validation**
   - [ ] Trivial kernel launches and executes
   - [ ] Results copied back to host correctly
   - [ ] Device synchronization works
   - [ ] Error handling catches CUDA errors

4. **CI Integration**
   - [ ] GitHub Actions builds pass without CUDA
   - [ ] Optional GPU runner validates CUDA path (if available)
   - [ ] Tests skip GPU tests when CUDA unavailable

## Integration Points

### Existing Code to Modify

1. **`engram-core/Cargo.toml`**
   - Add build script: `build = "build.rs"`
   - Add dependencies: `cc = "1.0"` (for build script)
   - Add feature flag: `gpu = ["cuda"]`

2. **`engram-core/src/compute/mod.rs`**
   - Add `#[cfg(feature = "gpu")]` conditional compilation
   - Add `pub mod cuda;` module declaration
   - Extend `CpuCapability` enum to include GPU

3. **`.github/workflows/ci.yml`**
   - Add matrix build: with and without CUDA
   - Add GPU runner configuration (optional)

## Testing Approach

### Test 1: Build Without CUDA

```bash
# Remove CUDA from PATH
export PATH=$(echo $PATH | sed 's|/usr/local/cuda[^:]*:||g')
unset CUDA_PATH

# Build should succeed with warning
cargo build 2>&1 | grep "CUDA toolkit not found"
```

### Test 2: Device Query

```rust
#[test]
#[cfg(feature = "gpu")]
fn test_cuda_device_query() {
    use crate::compute::cuda::ffi;

    match ffi::get_device_count() {
        Ok(count) => {
            println!("Found {} CUDA devices", count);
            assert!(count > 0);

            // Query first device properties
            let props = ffi::get_device_properties(0).unwrap();
            println!("Device 0: {:?}", props);
        }
        Err(e) => {
            // GPU not available in this test environment
            println!("CUDA not available: {:?}", e);
        }
    }
}
```

### Test 3: Kernel Launch

```rust
#[test]
#[cfg(feature = "gpu")]
fn test_trivial_kernel_launch() {
    let size = 1024;
    let mut output = vec![0i32; size];

    let result = unsafe {
        cuda_test_kernel_launch(output.as_mut_ptr(), size as i32)
    };

    if result == 0 {
        // Validate kernel computed correctly
        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, (i * 2) as i32);
        }
    } else {
        println!("Kernel launch failed (GPU unavailable?)");
    }
}
```

## Files to Create/Modify

### New Files
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/build.rs`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/compute/cuda/mod.rs`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/compute/cuda/ffi.rs`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/cuda/common.cuh`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/cuda/kernels/validation.cu`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/cuda/Makefile`

### Modified Files
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/Cargo.toml`
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/compute/mod.rs`
- `/Users/jordanwashburn/Workspace/orchard9/engram/.github/workflows/ci.yml`

## Dependencies

**Blocking**: None (parallel with Task 001)

**Blocked By This Task**:
- Task 003 (Cosine Similarity Kernel)
- Task 004 (Unified Memory Allocator)
- All subsequent GPU tasks

## Risk Assessment

### Risk: CUDA Toolkit Version Incompatibility

**Probability**: MEDIUM
**Mitigation**: Target CUDA 11.0+ (widely available), test with multiple versions

### Risk: Build Complexity Breaks CI

**Probability**: LOW
**Mitigation**: Extensive testing on clean systems, clear error messages

### Risk: FFI Binding Errors

**Probability**: LOW
**Mitigation**: Use well-tested CUDA APIs, validate with trivial kernels

## Success Metrics

1. **Build Success Rate**: 100% on both CUDA and non-CUDA systems
2. **FFI Correctness**: All CUDA API calls return expected values
3. **Kernel Launch Success**: Trivial kernel executes and produces correct results
4. **CI Stability**: No flaky test failures related to GPU detection

## Notes

This task establishes the foundation for all GPU work. Emphasis on robustness: builds must never fail due to missing CUDA toolkit, only disable GPU features gracefully.

The trivial validation kernel is intentionally simple - its purpose is to prove the toolchain works, not to demonstrate GPU performance.

All subsequent GPU tasks assume this infrastructure is in place and working correctly.
