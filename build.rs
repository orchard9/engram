// Cargo build script for Zig kernel integration
//
// This build script conditionally compiles Zig performance kernels when
// the "zig-kernels" feature is enabled. Without the feature, Engram falls
// back to pure Rust implementations.
//
// Build Process:
// 1. Check if zig-kernels feature is enabled
// 2. Verify Zig compiler is available (version 0.13.0 required)
// 3. Invoke `zig build -Doptimize=ReleaseFast` to build libengram_kernels.a
// 4. Link the static library with the Rust binary
//
// Graceful Degradation:
// - Without feature: No Zig dependency, pure Rust
// - With feature but no Zig: Clear error message (not segfault)
// - Zig build failure: Propagate error to cargo

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only build Zig library if feature enabled
    #[cfg(feature = "zig-kernels")]
    {
        build_zig_library();
    }

    // Rebuild if Zig source changes
    println!("cargo:rerun-if-changed=zig/");
    println!("cargo:rerun-if-changed=zig/build.zig");
    println!("cargo:rerun-if-changed=zig/src/");
}

#[cfg(feature = "zig-kernels")]
fn build_zig_library() {
    // Check if Zig compiler is available
    let zig_version_output = Command::new("zig")
        .arg("version")
        .output();

    let zig_available = match zig_version_output {
        Ok(output) => {
            let version = String::from_utf8_lossy(&output.stdout);
            println!("cargo:warning=Found Zig version: {}", version.trim());
            output.status.success()
        }
        Err(e) => {
            eprintln!("cargo:warning=Zig compiler not found: {}", e);
            false
        }
    };

    if !zig_available {
        panic!(
            "Zig compiler not found but zig-kernels feature is enabled.\n\
             \n\
             To fix this:\n\
             1. Install Zig 0.13.0 from https://ziglang.org/download/\n\
             2. Or disable zig-kernels feature: cargo build --no-default-features\n\
             \n\
             Note: Zig 0.13.0 is required for ABI compatibility."
        );
    }

    // Determine optimization level based on Cargo profile
    let optimize_flag = if env::var("PROFILE").unwrap_or_default() == "release" {
        "-Doptimize=ReleaseFast"
    } else {
        "-Doptimize=Debug"
    };

    // Build Zig static library
    println!("cargo:warning=Building Zig kernels with: zig build {}", optimize_flag);

    let status = Command::new("zig")
        .args(&["build", optimize_flag])
        .current_dir("zig")
        .status()
        .expect("Failed to execute zig build");

    if !status.success() {
        panic!("Zig build failed. Check zig/build.zig for errors.");
    }

    // Find the Zig output directory
    // Zig places output in zig-out/lib/ relative to build.zig
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let zig_lib_dir = manifest_dir.join("zig").join("zig-out").join("lib");

    if !zig_lib_dir.exists() {
        panic!(
            "Zig build succeeded but library not found at: {}\n\
             This is a bug in the build system.",
            zig_lib_dir.display()
        );
    }

    // Tell cargo where to find the library and link it
    println!("cargo:rustc-link-search=native={}", zig_lib_dir.display());
    println!("cargo:rustc-link-lib=static=engram_kernels");

    // Additional system libraries may be needed depending on platform
    // Zig's bundle_compiler_rt should handle most dependencies

    println!("cargo:warning=Zig kernels linked successfully from {}", zig_lib_dir.display());
}
