//! Cargo build script for Zig kernel integration
//!
//! This build script conditionally compiles Zig performance kernels when
//! the "zig-kernels" feature is enabled. Without the feature, Engram falls
//! back to pure Rust implementations.
//!
//! Build Process:
//! 1. Check if zig-kernels feature is enabled
//! 2. Verify Zig compiler is available (version 0.13.0 required)
//! 3. Invoke `zig build -Doptimize=ReleaseFast` to build libengram_kernels.a
//! 4. Link the static library with the Rust binary
//!
//! Graceful Degradation:
//! - Without feature: No Zig dependency, pure Rust
//! - With feature but no Zig: Clear error message (not segfault)
//! - Zig build failure: Propagate error to cargo

#[cfg(feature = "zig-kernels")]
use std::env;
#[cfg(feature = "zig-kernels")]
use std::path::PathBuf;
#[cfg(feature = "zig-kernels")]
use std::process::Command;

fn main() {
    // Only build Zig library if feature enabled
    #[cfg(feature = "zig-kernels")]
    {
        build_zig_library();
    }

    // Granular rebuild triggers - only rebuild when Zig sources actually change
    // Note: These are relative to CARGO_MANIFEST_DIR (engram-core directory)
    // Zig sources are in parent directory's zig/ subdirectory
    println!("cargo:rerun-if-changed=../zig/build.zig");
    println!("cargo:rerun-if-changed=../zig/src/ffi.zig");
    println!("cargo:rerun-if-changed=../zig/src/vector_similarity.zig");
    println!("cargo:rerun-if-changed=../zig/src/spreading_activation.zig");
    println!("cargo:rerun-if-changed=../zig/src/decay_functions.zig");
    println!("cargo:rerun-if-changed=../zig/src/arena_allocator.zig");

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
