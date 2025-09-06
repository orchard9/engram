//! Startup benchmark module for measuring time from clone to operational cluster
//!
//! This module provides cognitive-friendly benchmarking with real-time progress,
//! phase breakdown, and optimization recommendations.

use anyhow::{Context, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Benchmark phase representing a discrete step in the startup process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkPhase {
    /// Name of the phase
    pub name: String,
    /// Duration of the phase
    pub duration: Duration,
    /// Whether the phase succeeded
    pub success: bool,
    /// Optional error message if phase failed
    pub error: Option<String>,
    /// Cognitive complexity of this phase (for user understanding)
    pub complexity: PhaseComplexity,
}

/// Cognitive complexity levels for benchmark phases
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PhaseComplexity {
    /// Simple, well-understood operation
    Simple,
    /// Moderate complexity, may have variations
    Moderate,
    /// Complex operation with many variables
    Complex,
}

/// Complete benchmark result with all phases and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Individual phase timings
    pub phases: Vec<BenchmarkPhase>,
    /// Total duration from start to finish
    pub total_duration: Duration,
    /// Whether the benchmark met the <60s target
    pub target_met: bool,
    /// Platform information
    pub platform: PlatformInfo,
    /// Performance bottlenecks identified
    pub bottlenecks: Vec<Bottleneck>,
    /// Optimization recommendations
    pub recommendations: Vec<Recommendation>,
    /// Timestamp of benchmark run
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Platform information for benchmark context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    /// Operating system
    pub os: String,
    /// CPU architecture
    pub arch: String,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Available memory in bytes
    pub memory_bytes: u64,
    /// Rust version
    pub rust_version: String,
}

/// Identified performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Phase where bottleneck occurred
    pub phase: String,
    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,
    /// Impact on total time (percentage)
    pub impact_percentage: f64,
    /// Detailed description
    pub description: String,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BottleneckType {
    /// Network-related delays
    Network,
    /// CPU-bound compilation
    Compilation,
    /// Disk I/O bottleneck
    DiskIO,
    /// Memory constraints
    Memory,
    /// Configuration overhead
    Configuration,
}

/// Optimization recommendation with cognitive-friendly explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Priority level (1 = highest)
    pub priority: u8,
    /// Short title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Expected time savings
    pub expected_savings: Duration,
    /// Implementation difficulty
    pub difficulty: Difficulty,
    /// Specific commands or configurations
    pub implementation: Vec<String>,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Difficulty {
    /// Can be done immediately
    Trivial,
    /// Requires some configuration
    Easy,
    /// Requires moderate effort
    Medium,
    /// Requires significant changes
    Hard,
}

/// Main benchmark runner
pub struct StartupBenchmark {
    /// Multi-progress bar for parallel operations
    multi_progress: MultiProgress,
    /// Benchmark configuration
    config: BenchmarkConfig,
    /// Phase results
    phases: Vec<BenchmarkPhase>,
    /// Start time
    start_time: Instant,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Repository URL to clone
    pub repo_url: String,
    /// Target directory for benchmark
    pub target_dir: PathBuf,
    /// Whether to use release build
    pub release_build: bool,
    /// Number of warmup runs
    pub warmup_runs: u32,
    /// Number of benchmark runs
    pub benchmark_runs: u32,
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            repo_url: "https://github.com/orchard9/engram.git".to_string(),
            target_dir: std::env::temp_dir().join("engram-bench"),
            release_build: true,
            warmup_runs: 1,
            benchmark_runs: 3,
            verbose: false,
        }
    }
}

impl StartupBenchmark {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            multi_progress: MultiProgress::new(),
            config,
            phases: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Run the complete benchmark
    pub async fn run(&mut self) -> Result<BenchmarkResult> {
        println!("🚀 Engram Startup Benchmark");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Target: Git clone to operational in <60 seconds");
        
        let platform = detect_platform()?;
        println!("Platform: {} {}", platform.os, platform.arch);
        println!("CPU Cores: {}", platform.cpu_cores);
        println!("Rust: {}", platform.rust_version);
        println!();

        // Ensure clean target directory
        prepare_target_directory(&self.config.target_dir)?;

        // Run benchmark phases
        let clone_phase = run_phase(&self.multi_progress, "Clone Repository", PhaseComplexity::Simple, || {
            clone_repository(&self.config)
        })?;
        self.phases.push(clone_phase);

        let build_phase = run_phase(&self.multi_progress, "Build Project", PhaseComplexity::Complex, || {
            build_project(&self.config)
        })?;
        self.phases.push(build_phase);

        let start_phase = run_phase(&self.multi_progress, "Start Server", PhaseComplexity::Moderate, || {
            start_server(&self.config)
        })?;
        self.phases.push(start_phase);

        let query_phase = run_phase(&self.multi_progress, "First Query", PhaseComplexity::Simple, || {
            execute_first_query(&self.config)
        })?;
        self.phases.push(query_phase);

        // Calculate results
        let total_duration = self.start_time.elapsed();
        let target_met = total_duration.as_secs() < 60;

        // Analyze performance
        let bottlenecks = self.identify_bottlenecks();
        let recommendations = self.generate_recommendations(&bottlenecks);

        // Print results
        self.print_results(total_duration, target_met);

        Ok(BenchmarkResult {
            phases: self.phases.clone(),
            total_duration,
            target_met,
            platform,
            bottlenecks,
            recommendations,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Run a single benchmark phase with progress reporting
    fn run_phase<F>(&mut self, name: &str, complexity: PhaseComplexity, f: F) -> Result<()>
    where
        F: FnOnce(&ProgressBar) -> Result<()>,
    {
        let pb = self.multi_progress.add(ProgressBar::new(100));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {elapsed_precise}")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message(format!("⏳ {}", name));

        let phase_start = Instant::now();
        let result = f(&pb);
        let duration = phase_start.elapsed();

        let (success, error) = match result {
            Ok(()) => {
                pb.finish_with_message(format!("✅ {} ({:.2}s)", name, duration.as_secs_f64()));
                (true, None)
            }
            Err(e) => {
                pb.finish_with_message(format!("❌ {} failed", name));
                (false, Some(e.to_string()))
            }
        };

        self.phases.push(BenchmarkPhase {
            name: name.to_string(),
            duration,
            success,
            error,
            complexity,
        });

        result
    }

    /// Prepare the target directory
    fn prepare_target_directory(&self) -> Result<()> {
        if self.config.target_dir.exists() {
            fs::remove_dir_all(&self.config.target_dir)
                .context("Failed to clean target directory")?;
        }
        fs::create_dir_all(&self.config.target_dir)
            .context("Failed to create target directory")?;
        Ok(())
    }

    /// Clone the repository
    fn clone_repository(&self, pb: &ProgressBar) -> Result<()> {
        pb.set_position(10);
        
        let output = Command::new("git")
            .args(&[
                "clone",
                "--depth=1",
                &self.config.repo_url,
                self.config.target_dir.to_str().unwrap(),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("Failed to execute git clone")?;

        pb.set_position(100);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Git clone failed: {}", stderr);
        }

        Ok(())
    }

    /// Build the project
    fn build_project(&self, pb: &ProgressBar) -> Result<()> {
        let build_cmd = if self.config.release_build {
            vec!["build", "--release"]
        } else {
            vec!["build"]
        };

        pb.set_position(20);

        let output = Command::new("cargo")
            .args(&build_cmd)
            .current_dir(&self.config.target_dir)
            .env("RUSTFLAGS", "-C target-cpu=native")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("Failed to execute cargo build")?;

        pb.set_position(100);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Cargo build failed: {}", stderr);
        }

        Ok(())
    }

    /// Start the Engram server
    fn start_server(&self, pb: &ProgressBar) -> Result<()> {
        pb.set_position(30);

        let binary_path = self.config.target_dir.join(
            if self.config.release_build {
                "target/release/engram"
            } else {
                "target/debug/engram"
            }
        );

        // Start server in background
        let mut child = Command::new(&binary_path)
            .args(&["start", "--port", "7432", "--grpc-port", "50051"])
            .current_dir(&self.config.target_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to start Engram server")?;

        pb.set_position(60);

        // Wait for server to be ready (check health endpoint)
        let start_wait = Instant::now();
        let max_wait = Duration::from_secs(30);

        while start_wait.elapsed() < max_wait {
            if let Ok(response) = reqwest::blocking::get("http://localhost:7432/health") {
                if response.status().is_success() {
                    pb.set_position(100);
                    // Keep server handle to stop it later
                    std::mem::forget(child); // Let it run for now
                    return Ok(());
                }
            }
            std::thread::sleep(Duration::from_millis(500));
        }

        // Kill the server if it didn't start
        let _ = child.kill();
        anyhow::bail!("Server failed to start within 30 seconds")
    }

    /// Execute the first query
    fn execute_first_query(&self, pb: &ProgressBar) -> Result<()> {
        pb.set_position(50);

        let response = reqwest::blocking::get("http://localhost:7432/health")
            .context("Failed to execute health check")?;

        pb.set_position(100);

        if !response.status().is_success() {
            anyhow::bail!("Health check failed with status: {}", response.status());
        }

        // Stop the server
        let _ = Command::new("pkill")
            .arg("-f")
            .arg("engram start")
            .output();

        Ok(())
    }

    /// Detect platform information
    fn detect_platform(&self) -> Result<PlatformInfo> {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        let cpu_cores = num_cpus::get();

        let rust_version = Command::new("rustc")
            .arg("--version")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let memory_bytes = sys_info::mem_info()
            .map(|m| m.total * 1024)
            .unwrap_or(0);

        Ok(PlatformInfo {
            os,
            arch,
            cpu_cores,
            memory_bytes,
            rust_version,
        })
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(&self) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();
        let total_time = self.phases.iter().map(|p| p.duration).sum::<Duration>();

        for phase in &self.phases {
            let impact = (phase.duration.as_secs_f64() / total_time.as_secs_f64()) * 100.0;

            if impact > 40.0 {
                let bottleneck_type = match phase.name.as_str() {
                    "Clone Repository" => BottleneckType::Network,
                    "Build Project" => BottleneckType::Compilation,
                    "Start Server" => BottleneckType::Configuration,
                    _ => BottleneckType::DiskIO,
                };

                bottlenecks.push(Bottleneck {
                    phase: phase.name.clone(),
                    bottleneck_type,
                    impact_percentage: impact,
                    description: format!(
                        "{} is taking {:.1}% of total time ({:.2}s)",
                        phase.name,
                        impact,
                        phase.duration.as_secs_f64()
                    ),
                });
            }
        }

        bottlenecks
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self, bottlenecks: &[Bottleneck]) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        for bottleneck in bottlenecks {
            match bottleneck.bottleneck_type {
                BottleneckType::Compilation => {
                    recommendations.push(Recommendation {
                        priority: 1,
                        title: "Enable sccache for build caching".to_string(),
                        description: "sccache can dramatically reduce build times by caching compilation artifacts".to_string(),
                        expected_savings: Duration::from_secs(20),
                        difficulty: Difficulty::Easy,
                        implementation: vec![
                            "cargo install sccache".to_string(),
                            "export RUSTC_WRAPPER=sccache".to_string(),
                        ],
                    });

                    recommendations.push(Recommendation {
                        priority: 2,
                        title: "Use thin LTO for faster linking".to_string(),
                        description: "Thin LTO provides most optimization benefits with faster link times".to_string(),
                        expected_savings: Duration::from_secs(10),
                        difficulty: Difficulty::Trivial,
                        implementation: vec![
                            "Add to Cargo.toml: [profile.release] lto = \"thin\"".to_string(),
                        ],
                    });
                }
                BottleneckType::Network => {
                    recommendations.push(Recommendation {
                        priority: 1,
                        title: "Use shallow clone".to_string(),
                        description: "Already using --depth=1, consider local mirror for CI".to_string(),
                        expected_savings: Duration::from_secs(5),
                        difficulty: Difficulty::Medium,
                        implementation: vec![
                            "Set up local Git mirror".to_string(),
                            "Use git clone --reference for faster clones".to_string(),
                        ],
                    });
                }
                _ => {}
            }
        }

        recommendations.sort_by_key(|r| r.priority);
        recommendations
    }

    /// Print benchmark results with cognitive-friendly formatting
    fn print_results(&self, total_duration: Duration, target_met: bool) {
        println!("\n⏱️  Performance Breakdown");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        for phase in &self.phases {
            let percentage = (phase.duration.as_secs_f64() / total_duration.as_secs_f64()) * 100.0;
            let bar_length = (percentage / 2.0) as usize;
            let bar = "█".repeat(bar_length);
            
            println!(
                "{:<25} {:>7.2}s {:>6.1}% {}",
                phase.name,
                phase.duration.as_secs_f64(),
                percentage,
                bar
            );
        }
        
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Total Time: {:.2}s", total_duration.as_secs_f64());
        
        if target_met {
            println!("\n✅ PASS: Startup completed in under 60 seconds!");
        } else {
            println!("\n❌ FAIL: Startup exceeded 60 second target");
        }
    }
}

/// Run benchmark with hyperfine for statistical analysis
pub async fn run_with_hyperfine(config: BenchmarkConfig) -> Result<()> {
    println!("🔬 Running statistical benchmark with hyperfine");
    
    // Check if hyperfine is installed
    let hyperfine_check = Command::new("hyperfine")
        .arg("--version")
        .output();
    
    if hyperfine_check.is_err() {
        println!("⚠️  hyperfine not found. Installing...");
        // Attempt to install hyperfine
        let install_result = Command::new("cargo")
            .args(&["install", "hyperfine"])
            .output()
            .context("Failed to install hyperfine")?;
        
        if !install_result.status.success() {
            anyhow::bail!("Failed to install hyperfine. Please install manually.");
        }
    }
    
    // Create benchmark script for hyperfine
    let script_path = config.target_dir.join("benchmark.sh");
    let script_content = format!(
        r#"#!/bin/bash
set -e
rm -rf /tmp/engram-bench-hyperfine
git clone --depth=1 {} /tmp/engram-bench-hyperfine
cd /tmp/engram-bench-hyperfine
cargo build --release
./target/release/engram start --port 7432 &
SERVER_PID=$!
sleep 2
curl -s http://localhost:7432/health > /dev/null
kill $SERVER_PID 2>/dev/null || true
"#,
        config.repo_url
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
        .args(&[
            "--warmup", &config.warmup_runs.to_string(),
            "--runs", &config.benchmark_runs.to_string(),
            "--time-unit", "second",
            "--export-json", config.target_dir.join("results.json").to_str().unwrap(),
            "--export-markdown", config.target_dir.join("results.md").to_str().unwrap(),
            script_path.to_str().unwrap(),
        ])
        .output()
        .context("Failed to run hyperfine")?;
    
    println!("{}", String::from_utf8_lossy(&output.stdout));
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("hyperfine failed: {}", stderr);
    }
    
    // Parse and display results
    let results_json = fs::read_to_string(config.target_dir.join("results.json"))?;
    let results: serde_json::Value = serde_json::from_str(&results_json)?;
    
    if let Some(mean) = results["results"][0]["mean"].as_f64() {
        println!("\n📊 Statistical Results:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Mean time: {:.2}s", mean);
        
        if let Some(stddev) = results["results"][0]["stddev"].as_f64() {
            println!("Std deviation: {:.2}s", stddev);
        }
        
        if mean < 60.0 {
            println!("\n✅ PASS: Mean time under 60 seconds!");
        } else {
            println!("\n❌ FAIL: Mean time exceeds 60 second target");
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let config = BenchmarkConfig::default();
        let mut benchmark = StartupBenchmark::new(config);
        let platform = benchmark.detect_platform().unwrap();
        
        assert!(!platform.os.is_empty());
        assert!(!platform.arch.is_empty());
        assert!(platform.cpu_cores > 0);
    }

    #[test]
    fn test_bottleneck_identification() {
        let mut benchmark = StartupBenchmark::new(BenchmarkConfig::default());
        
        // Add mock phases
        benchmark.phases.push(BenchmarkPhase {
            name: "Clone Repository".to_string(),
            duration: Duration::from_secs(5),
            success: true,
            error: None,
            complexity: PhaseComplexity::Simple,
        });
        
        benchmark.phases.push(BenchmarkPhase {
            name: "Build Project".to_string(),
            duration: Duration::from_secs(45), // This is a bottleneck
            success: true,
            error: None,
            complexity: PhaseComplexity::Complex,
        });
        
        let bottlenecks = benchmark.identify_bottlenecks();
        assert!(!bottlenecks.is_empty());
        assert_eq!(bottlenecks[0].phase, "Build Project");
        assert!(bottlenecks[0].impact_percentage > 40.0);
    }
}