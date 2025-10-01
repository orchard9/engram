//! Simplified startup benchmark module
//!
//! This module provides a straightforward benchmark implementation
//! that measures time from clone to operational cluster.

use anyhow::{Context, Result};
use std::fs;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Run the startup benchmark
///
/// # Errors
///
/// Returns error if git clone, build, or server start fails
///
/// # Panics
///
/// Panics if temporary directory path cannot be converted to string
pub fn run_benchmark(repo_url: String, use_release: bool, verbose: bool) -> Result<bool> {
    println!("🚀 Engram Startup Benchmark");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Target: Git clone to operational in <60 seconds");
    println!();

    let start_time = Instant::now();
    let temp_dir = std::env::temp_dir().join(format!("engram-bench-{}", std::process::id()));

    // Clean up any existing directory
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)?;
    }
    fs::create_dir_all(&temp_dir)?;

    // Phase 1: Clone
    print_phase("📦 Phase 1: Clone Repository");
    let clone_start = Instant::now();

    let clone_result = Command::new("git")
        .args([
            "clone",
            "--depth=1",
            &repo_url,
            temp_dir
                .to_str()
                .expect("temp dir path should be valid UTF-8"),
        ])
        .stdout(if verbose {
            Stdio::inherit()
        } else {
            Stdio::null()
        })
        .stderr(if verbose {
            Stdio::inherit()
        } else {
            Stdio::null()
        })
        .status()
        .context("Failed to execute git clone")?;

    if !clone_result.success() {
        anyhow::bail!("Git clone failed");
    }

    let clone_duration = clone_start.elapsed();
    println!("✅ Clone completed in {:.2}s", clone_duration.as_secs_f64());

    // Phase 2: Build
    print_phase("🔨 Phase 2: Build");
    let build_start = Instant::now();

    let build_args = if use_release {
        vec!["build", "--release"]
    } else {
        vec!["build"]
    };

    let build_result = Command::new("cargo")
        .args(&build_args)
        .current_dir(&temp_dir)
        .env("RUSTFLAGS", "-C target-cpu=native")
        .stdout(if verbose {
            Stdio::inherit()
        } else {
            Stdio::null()
        })
        .stderr(if verbose {
            Stdio::inherit()
        } else {
            Stdio::null()
        })
        .status()
        .context("Failed to execute cargo build")?;

    if !build_result.success() {
        anyhow::bail!("Cargo build failed");
    }

    let build_duration = build_start.elapsed();
    println!("✅ Build completed in {:.2}s", build_duration.as_secs_f64());

    // Phase 3: Start Server
    print_phase("🌟 Phase 3: Start Server");
    let start_server = Instant::now();

    let binary_path = temp_dir.join(if use_release {
        "target/release/engram"
    } else {
        "target/debug/engram"
    });

    // Start server in background
    let mut server = Command::new(&binary_path)
        .args(["start", "--port", "7432", "--grpc-port", "50051"])
        .current_dir(&temp_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to start Engram server")?;

    // Wait for server to be ready
    let mut server_ready = false;
    for _ in 0..60 {
        std::thread::sleep(Duration::from_millis(500));

        // Try to connect to health endpoint
        let health_check = Command::new("curl")
            .args(["-s", "http://localhost:7432/health"])
            .output();

        if let Ok(output) = health_check {
            if output.status.success() {
                server_ready = true;
                break;
            }
        }
    }

    if !server_ready {
        let _ = server.kill();
        anyhow::bail!("Server failed to start within 30 seconds");
    }

    let start_duration = start_server.elapsed();
    println!("✅ Server started in {:.2}s", start_duration.as_secs_f64());

    // Phase 4: First Query
    print_phase("🔍 Phase 4: First Query");
    let query_start = Instant::now();

    let query_result = Command::new("curl")
        .args(["-s", "http://localhost:7432/health"])
        .output()
        .context("Failed to execute health check")?;

    if !query_result.status.success() {
        let _ = server.kill();
        anyhow::bail!("Health check failed");
    }

    let query_duration = query_start.elapsed();
    println!(
        "✅ First query completed in {:.2}s",
        query_duration.as_secs_f64()
    );

    // Stop server
    let _ = server.kill();

    // Calculate total time
    let total_duration = start_time.elapsed();

    // Print results
    print_phase("⏱️  Performance Breakdown");
    println!();
    println!("Phase Timings:");
    println!("─────────────────────────────────────────────────────────────");
    print_timing("Clone Repository", clone_duration, total_duration);
    print_timing("Build Project", build_duration, total_duration);
    print_timing("Start Server", start_duration, total_duration);
    print_timing("First Query", query_duration, total_duration);
    println!("─────────────────────────────────────────────────────────────");
    println!("Total Time: {:.2}s", total_duration.as_secs_f64());
    println!();

    let target_met = total_duration.as_secs() < 60;

    if target_met {
        println!("✅ PASS: Startup completed in under 60 seconds!");
    } else {
        println!("❌ FAIL: Startup exceeded 60 second target");
    }

    // Print recommendations if slow
    if !target_met {
        print_phase("💡 Optimization Recommendations");

        if build_duration.as_secs() > 30 {
            println!(
                "⚠️  Build phase is slow ({:.2}s)",
                build_duration.as_secs_f64()
            );
            println!("   Recommendations:");
            println!("   • Enable sccache for build caching");
            println!("   • Use 'cargo build --release' with lto='thin'");
            println!("   • Consider pre-built binaries for CI");
        }

        if clone_duration.as_secs() > 10 {
            println!(
                "⚠️  Clone phase is slow ({:.2}s)",
                clone_duration.as_secs_f64()
            );
            println!("   Recommendations:");
            println!("   • Use shallow clone with --depth=1");
            println!("   • Consider Git LFS for large files");
            println!("   • Use CDN-backed mirrors");
        }
    }

    // Clean up
    let _ = fs::remove_dir_all(&temp_dir);

    Ok(target_met)
}

/// Run benchmark with hyperfine for statistical analysis
///
/// # Errors
///
/// Returns error if Hyperfine is not installed or benchmark fails
///
/// # Panics
///
/// Panics if script path cannot be converted to string
pub fn run_with_hyperfine(
    repo_url: String,
    warmup_runs: u32,
    benchmark_runs: u32,
    use_release: bool,
) -> Result<()> {
    println!("🔬 Running statistical benchmark with hyperfine");

    // Check if hyperfine is installed
    let hyperfine_check = Command::new("hyperfine").arg("--version").output();

    if hyperfine_check.is_err() {
        println!("⚠️  hyperfine not found. Please install it:");
        println!("   cargo install hyperfine");
        anyhow::bail!("hyperfine not installed");
    }

    // Create benchmark script
    let script_path = std::env::temp_dir().join("engram-bench.sh");
    let build_cmd = if use_release {
        "cargo build --release"
    } else {
        "cargo build"
    };

    let binary = if use_release {
        "./target/release/engram"
    } else {
        "./target/debug/engram"
    };

    let script_content = format!(
        r"#!/bin/bash
set -e
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR
git clone --depth=1 {repo_url} engram
cd engram
{build_cmd}
{binary} start --port 7432 &
SERVER_PID=$!
sleep 5
curl -s http://localhost:7432/health > /dev/null
kill $SERVER_PID 2>/dev/null || true
cd /
rm -rf $TEMP_DIR
"
    );

    fs::write(&script_path, script_content)?;

    // Make script executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&script_path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&script_path, perms)?;
    }

    // Run hyperfine
    let output = Command::new("hyperfine")
        .args([
            "--warmup",
            &warmup_runs.to_string(),
            "--runs",
            &benchmark_runs.to_string(),
            "--time-unit",
            "second",
            script_path
                .to_str()
                .expect("script path should be valid UTF-8"),
        ])
        .output()
        .context("Failed to run hyperfine")?;

    println!("{}", String::from_utf8_lossy(&output.stdout));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("hyperfine failed: {}", stderr);
    }

    // Clean up
    let _ = fs::remove_file(&script_path);

    Ok(())
}

fn print_phase(title: &str) {
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("{title}");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

fn print_timing(phase: &str, duration: Duration, total: Duration) {
    let percentage = (duration.as_secs_f64() / total.as_secs_f64()) * 100.0;
    let bar_length = (percentage / 2.0).max(0.0).min(50.0) as usize;
    let bar = "█".repeat(bar_length.min(50));

    println!(
        "{:<25} {:>7.2}s {:>6.1}% {}",
        phase,
        duration.as_secs_f64(),
        percentage,
        bar
    );
}
