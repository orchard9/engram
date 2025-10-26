//! Cargo build script for Zig and CUDA kernel integration
//!
//! This build script conditionally compiles performance kernels:
//! - Zig kernels: when "zig-kernels" feature is enabled
//! - CUDA kernels: when "gpu" feature is enabled and CUDA toolkit is available
//!
//! Build Process (Zig):
//! 1. Check if zig-kernels feature is enabled
//! 2. Verify Zig compiler is available (version 0.13.0 required)
//! 3. Invoke `zig build -Doptimize=ReleaseFast` to build libengram_kernels.a
//! 4. Link the static library with the Rust binary
//!
//! Build Process (CUDA):
//! 1. Check if gpu feature is enabled
//! 2. Detect CUDA toolkit (nvcc, CUDA_PATH, standard paths)
//! 3. Compile .cu files with nvcc targeting sm_60+ (Maxwell)
//! 4. Create static library from CUDA kernels
//! 5. Link CUDA runtime library
//! 6. Generate fallback stubs if CUDA unavailable
//!
//! Graceful Degradation:
//! - Without features: No external dependencies, pure Rust
//! - With feature but no compiler: Clear error message (Zig) or warning (CUDA)
//! - Build failure: Propagate error to cargo

#[cfg(any(feature = "zig-kernels", feature = "gpu"))]
use std::env;
#[cfg(any(feature = "zig-kernels", feature = "gpu"))]
use std::path::PathBuf;
#[cfg(any(feature = "zig-kernels", feature = "gpu"))]
use std::process::Command;

fn main() {
    // Register custom cfg for CUDA availability
    println!("cargo::rustc-check-cfg=cfg(cuda_available)");

    // Build Zig library if feature enabled
    #[cfg(feature = "zig-kernels")]
    {
        build_zig_library();
    }

    // Build CUDA kernels if feature enabled
    #[cfg(feature = "gpu")]
    {
        build_cuda_kernels();
    }

    // Granular rebuild triggers - only rebuild when sources actually change
    // Note: These are relative to CARGO_MANIFEST_DIR (engram-core directory)

    // Zig sources are in parent directory's zig/ subdirectory
    println!("cargo:rerun-if-changed=../zig/build.zig");
    println!("cargo:rerun-if-changed=../zig/src/ffi.zig");
    println!("cargo:rerun-if-changed=../zig/src/vector_similarity.zig");
    println!("cargo:rerun-if-changed=../zig/src/spreading_activation.zig");
    println!("cargo:rerun-if-changed=../zig/src/decay_functions.zig");
    println!("cargo:rerun-if-changed=../zig/src/arena_allocator.zig");

    // CUDA sources are in engram-core/cuda/ subdirectory
    println!("cargo:rerun-if-changed=cuda/");
    println!("cargo:rerun-if-changed=cuda/common.cuh");
    println!("cargo:rerun-if-changed=cuda/kernels/validation.cu");
    println!("cargo:rerun-if-changed=cuda/kernels/cosine_similarity.cu");
    println!("cargo:rerun-if-changed=cuda/kernels/hnsw_scoring.cu");

    // Rebuild if build script itself changes
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "zig-kernels")]
fn build_zig_library() {
    // Check if Zig compiler is available
    let zig_version_output = Command::new("zig").arg("version").output();

    let zig_available = match zig_version_output {
        Ok(output) => {
            let version = String::from_utf8_lossy(&output.stdout);
            let version_str = version.trim();
            println!("cargo:warning=Found Zig version: {version_str}");

            // Verify Zig version is 0.13.0 or later
            // Our FFI uses C ABI (export fn) which is stable across Zig versions,
            // but we require 0.13.0+ for build system compatibility
            let version_parts: Vec<&str> = version_str.split('.').collect();
            let major = version_parts.first().and_then(|s| s.parse::<u32>().ok());
            let minor = version_parts.get(1).and_then(|s| s.parse::<u32>().ok());

            match (major, minor) {
                (Some(0), Some(minor_ver)) if minor_ver >= 13 => {
                    if minor_ver > 13 {
                        println!(
                            "cargo:warning=Using Zig 0.{minor_ver}.x (tested with 0.13.x). \
                             FFI uses C ABI which should be stable, but please report any issues."
                        );
                    }
                }
                _ => {
                    panic!(
                        "Zig 0.13.0 or later is required, found: {version_str}\n\
                         \n\
                         Our FFI uses C ABI (via export fn) which is stable,\n\
                         but older Zig versions may have incompatible build systems.\n\
                         \n\
                         To fix this:\n\
                         1. Install Zig 0.13.0+ from: https://ziglang.org/download/\n\
                         2. Or disable zig-kernels feature: cargo build --no-default-features"
                    );
                }
            }

            output.status.success()
        }
        Err(e) => {
            eprintln!("cargo:warning=Zig compiler not found: {e}");
            false
        }
    };

    assert!(
        zig_available,
        "Zig compiler not found but zig-kernels feature is enabled.\n\
         \n\
         To fix this:\n\
         1. Install Zig 0.13.0+ from https://ziglang.org/download/\n\
         2. Or disable zig-kernels feature: cargo build --no-default-features\n\
         \n\
         Note: Our FFI uses C ABI which is stable across Zig versions >= 0.13.0"
    );

    // Determine optimization level based on Cargo profile
    let optimize_flag = if env::var("PROFILE").unwrap_or_default() == "release" {
        "-Doptimize=ReleaseFast"
    } else {
        "-Doptimize=Debug"
    };

    // Build Zig static library
    println!("cargo:warning=Building Zig kernels with: zig build {optimize_flag}");

    // Zig directory is in workspace root (parent of engram-core)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let zig_dir = manifest_dir.parent().unwrap().join("zig");

    let status = Command::new("zig")
        .args(["build", optimize_flag])
        .current_dir(&zig_dir)
        .status()
        .expect("Failed to execute zig build");

    assert!(
        status.success(),
        "Zig build failed. Check zig/build.zig for errors."
    );

    // Find the Zig output directory
    // CARGO_MANIFEST_DIR points to engram-core directory
    // Zig sources are in workspace root's zig/ subdirectory (parent of engram-core)
    // Zig places output in zig-out/lib/ relative to build.zig location
    // This gives us an absolute path, avoiding issues with different cargo invocation directories
    let zig_lib_dir = zig_dir.join("zig-out").join("lib");

    assert!(
        zig_lib_dir.exists(),
        "Zig build succeeded but library not found at: {}\n\
         This is a bug in the build system.",
        zig_lib_dir.display()
    );

    // Tell cargo where to find the library and link it
    println!("cargo:rustc-link-search=native={}", zig_lib_dir.display());
    println!("cargo:rustc-link-lib=static=engram_kernels");

    // Additional system libraries may be needed depending on platform
    // Zig's bundle_compiler_rt should handle most dependencies

    println!(
        "cargo:warning=Zig kernels linked successfully from {}",
        zig_lib_dir.display()
    );
}

#[cfg(feature = "gpu")]
fn build_cuda_kernels() {
    // Detect CUDA toolkit availability
    let cuda_available = detect_cuda_toolkit();

    if cuda_available {
        println!("cargo:rustc-cfg=cuda_available");
        compile_cuda_kernels();
        link_cuda_runtime();
    } else {
        println!("cargo:warning=CUDA toolkit not found - GPU acceleration disabled");
        println!(
            "cargo:warning=To enable GPU support, install CUDA toolkit 11.0+ and ensure nvcc is in PATH"
        );
        generate_fallback_stubs();
    }
}

#[cfg(feature = "gpu")]
fn detect_cuda_toolkit() -> bool {
    // Determine nvcc binary name based on platform (Windows needs .exe extension)
    let nvcc_name = if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    };

    // Check for nvcc in PATH
    if let Ok(output) = Command::new(nvcc_name).arg("--version").output()
        && output.status.success()
    {
        let version_str = String::from_utf8_lossy(&output.stdout);
        println!(
            "cargo:warning=Found CUDA compiler: {}",
            version_str.lines().next().unwrap_or("")
        );
        return true;
    }

    // Check for CUDA_PATH environment variable
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc_path = PathBuf::from(&cuda_path).join("bin").join(nvcc_name);
        if nvcc_path.exists() {
            println!("cargo:warning=Found CUDA at CUDA_PATH: {cuda_path}");
            return true;
        }
    }

    // Check standard installation paths (platform-specific)
    let standard_paths = if cfg!(target_os = "windows") {
        // Windows-specific CUDA installation paths
        // Check multiple versions as CUDA toolkit installs in versioned directories
        vec![
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0",
        ]
    } else {
        // Unix-like systems (Linux, macOS)
        vec![
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/lib/cuda",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-11.8",
        ]
    };

    for path in &standard_paths {
        let cuda_root = PathBuf::from(path);
        let nvcc_path = cuda_root.join("bin").join(nvcc_name);
        if nvcc_path.exists() {
            println!("cargo:warning=Found CUDA at standard path: {path}");
            // SAFETY: This is safe in build.rs context as there are no concurrent threads
            // and we're setting an environment variable that will be used by subsequent
            // build script operations (link_cuda_runtime).
            #[allow(unsafe_code)]
            unsafe {
                env::set_var("CUDA_PATH", path);
            }
            return true;
        }
    }

    false
}

#[cfg(feature = "gpu")]
fn compile_cuda_kernels() {
    let cuda_files = [
        "cuda/kernels/validation.cu",
        "cuda/kernels/cosine_similarity.cu",
        "cuda/kernels/hnsw_scoring.cu",
        "cuda/kernels/spreading_matmul.cu",
    ];

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(&out_dir);

    // Determine nvcc binary name based on platform
    let nvcc_name = if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    };

    // Determine compute capability target
    // sm_60 = Maxwell (GTX 1060+), widely compatible
    // sm_70 = Volta (V100), sm_75 = Turing (RTX 2000), sm_80 = Ampere (RTX 3000)
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_60".to_string());

    let mut object_files = Vec::new();

    for cu_file in &cuda_files {
        let cu_path = PathBuf::from(cu_file);
        if !cu_path.exists() {
            println!("cargo:warning=CUDA kernel not found: {cu_file} - skipping");
            continue;
        }

        let obj_name = cu_file.replace('/', "_").replace(".cu", ".o");
        let obj_path = out_path.join(&obj_name);

        println!(
            "cargo:warning=Compiling CUDA kernel: {cu_file} -> {}",
            obj_path.display()
        );

        let mut nvcc = Command::new(nvcc_name);
        nvcc.arg("-O3")
            .arg(format!("-arch={arch}"))
            .arg("-std=c++14");

        // Platform-specific compiler options
        if !cfg!(target_os = "windows") {
            // Unix-like systems need -fPIC for shared libraries
            nvcc.arg("--compiler-options").arg("-fPIC");
        }

        nvcc.arg("-c").arg(cu_file).arg("-o").arg(&obj_path);

        // Add include path for common headers
        nvcc.arg("-I").arg("cuda");

        let output = nvcc.output().expect("Failed to execute nvcc");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("CUDA compilation failed for {cu_file}:\n{stderr}");
        }

        object_files.push(obj_path);
    }

    if object_files.is_empty() {
        println!("cargo:warning=No CUDA kernels compiled");
        return;
    }

    // Create static library from object files
    let lib_path = out_path.join("libengram_cuda.a");

    println!(
        "cargo:warning=Creating CUDA static library: {}",
        lib_path.display()
    );

    let mut ar = Command::new("ar");
    ar.arg("rcs").arg(&lib_path);

    for obj in &object_files {
        ar.arg(obj);
    }

    let ar_output = ar.output().expect("Failed to execute ar");

    if !ar_output.status.success() {
        let stderr = String::from_utf8_lossy(&ar_output.stderr);
        panic!("Static library creation failed:\n{stderr}");
    }

    // Link the static library
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=engram_cuda");
}

#[cfg(feature = "gpu")]
fn link_cuda_runtime() {
    // Determine CUDA library path
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // Try different library directory names (lib64 on Linux, lib on macOS, lib/x64 on Windows)
    let lib_dirs = ["lib64", "lib", "lib/x64"];

    for lib_dir in &lib_dirs {
        let lib_path = PathBuf::from(&cuda_path).join(lib_dir);
        if lib_path.exists() {
            println!("cargo:rustc-link-search=native={}", lib_path.display());
            break;
        }
    }

    // Link CUDA runtime library
    println!("cargo:rustc-link-lib=cudart");

    // On Linux, we may also need to link libstdc++ for C++ runtime
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

#[cfg(feature = "gpu")]
fn generate_fallback_stubs() {
    // When CUDA is unavailable, we generate stub implementations
    // that return error codes indicating GPU is not available.
    // This allows the codebase to compile on systems without CUDA.
    //
    // All stub functions return -1 to indicate "CUDA not available" error.
    // This matches the error handling conventions in the actual CUDA kernels.

    let stub_code = r#"
// Auto-generated fallback stubs for CUDA functions
// These stubs are used when CUDA toolkit is not available
//
// All functions return -1 to indicate CUDA is not available.
// This enables graceful degradation on systems without GPU support.

use std::ffi::c_int;

// Validation kernel stubs (validation.cu)
#[no_mangle]
pub extern "C" fn cuda_validate_environment() -> c_int {
    -1 // CUDA not available
}

#[no_mangle]
pub extern "C" fn cuda_test_kernel_launch(_output: *mut i32, _size: i32) -> c_int {
    -1 // CUDA not available
}

#[no_mangle]
pub extern "C" fn cuda_test_vector_kernel(
    _input: *const f32,
    _output: *mut f32,
    _count: i32,
) -> c_int {
    -1 // CUDA not available
}

// Cosine similarity kernel stubs (cosine_similarity.cu)
#[no_mangle]
pub extern "C" fn cuda_cosine_similarity_batch(
    _d_targets: *const f32,
    _query_norm_sq: f32,
    _d_results: *mut f32,
    _batch_size: i32,
) -> c_int {
    -1 // CUDA not available
}

#[no_mangle]
pub extern "C" fn cuda_cosine_set_query(_h_query: *const f32) -> c_int {
    -1 // CUDA not available
}

#[no_mangle]
pub extern "C" fn cuda_cosine_similarity_batch_managed(
    _h_query: *const f32,
    _h_targets: *const f32,
    _query_norm_sq: f32,
    _h_results: *mut f32,
    _batch_size: i32,
) -> c_int {
    -1 // CUDA not available
}

// Spreading activation kernel stubs (spreading_matmul.cu)
#[no_mangle]
pub extern "C" fn cuda_sparse_spreading(
    _d_row_ptr: *const c_int,
    _d_col_idx: *const c_int,
    _d_weights: *const f32,
    _d_input_activation: *const f32,
    _d_output_activation: *mut f32,
    _num_nodes: c_int,
    _num_edges: c_int,
) -> c_int {
    -1 // CUDA not available
}

#[no_mangle]
pub extern "C" fn cuda_sparse_spreading_managed(
    _h_row_ptr: *const c_int,
    _h_col_idx: *const c_int,
    _h_weights: *const f32,
    _h_input_activation: *const f32,
    _h_output_activation: *mut f32,
    _num_nodes: c_int,
    _num_edges: c_int,
) -> c_int {
    -1 // CUDA not available
}

// HNSW scoring kernel stubs (hnsw_scoring.cu)
#[no_mangle]
pub extern "C" fn cuda_hnsw_top_k(
    _h_query: *const f32,
    _h_candidates: *const f32,
    _h_top_k_distances: *mut f32,
    _h_top_k_indices: *mut c_int,
    _num_candidates: c_int,
    _k: c_int,
    _distance_metric: c_int,
    _query_norm_sq: f32,
) -> c_int {
    -1 // CUDA not available
}
"#;

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let stub_path = PathBuf::from(&out_dir).join("cuda_stubs.rs");

    std::fs::write(&stub_path, stub_code).expect("Failed to write CUDA stub file");

    println!(
        "cargo:warning=Generated CUDA fallback stubs at {}",
        stub_path.display()
    );
}
