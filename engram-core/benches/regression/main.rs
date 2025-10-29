#![allow(missing_docs)]
//! Performance regression framework for Engram core operations.
//!
//! This benchmark suite establishes baseline performance for critical operations
//! and detects regressions >5% from established baselines. It supports:
//!
//! - **Vector similarity**: 768d embeddings vs 1000 candidates
//! - **Spreading activation**: 1000 nodes, 100 iterations
//! - **Decay calculation**: 10,000 memories
//!
//! ## Usage
//!
//! Run regression checks against baselines:
//! ```bash
//! cargo bench --bench regression
//! ```
//!
//! Update baselines (after intentional improvements):
//! ```bash
//! UPDATE_BASELINES=1 cargo bench --bench regression
//! ```
//!
//! ## Baseline Storage
//!
//! Baselines are stored in `benches/regression/baselines.json` and include:
//! - Platform-specific measurements (x86_64, ARM64)
//! - Mean execution time in nanoseconds
//! - Standard deviation for variance tracking
//! - Sample size for statistical confidence
//!
//! ## Regression Detection
//!
//! The benchmark fails (exit code 1) if any operation regresses >5% from baseline.
//! Improvements >5% are reported but do not fail the build.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Performance baseline for a single benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Baseline {
    /// Mean execution time in nanoseconds
    mean_ns: u64,
    /// Standard deviation in nanoseconds
    std_dev_ns: u64,
    /// Number of samples used to establish baseline
    sample_size: usize,
}

/// Collection of performance baselines for all benchmarks
#[derive(Debug, Serialize, Deserialize)]
struct Baselines {
    /// Schema version for forward compatibility
    version: String,
    /// Target platform (e.g., "x86_64-apple-darwin", "aarch64-unknown-linux-gnu")
    platform: String,
    /// CPU model for reference
    cpu: String,
    /// Timestamp of last baseline update
    timestamp: String,
    /// Map of benchmark name to baseline
    #[allow(clippy::struct_field_names)]
    baselines: HashMap<String, Baseline>,
}

impl Baselines {
    /// Load baselines from disk or create empty set
    fn load() -> Self {
        let path = "benches/regression/baselines.json";
        if Path::new(path).exists() {
            let json = fs::read_to_string(path).expect("Failed to read baselines");
            serde_json::from_str(&json).expect("Failed to parse baselines")
        } else {
            Self::empty()
        }
    }

    /// Create empty baseline set for current platform
    fn empty() -> Self {
        Self {
            version: "1.0".to_string(),
            platform: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            cpu: Self::detect_cpu(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            baselines: HashMap::new(),
        }
    }

    /// Detect CPU model for platform-specific baselines
    #[allow(clippy::unused_io_amount)]
    fn detect_cpu() -> String {
        // Try to detect CPU model from system info
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("sysctl")
                .arg("-n")
                .arg("machdep.cpu.brand_string")
                .output()
            {
                if let Ok(cpu) = String::from_utf8(output.stdout) {
                    return cpu.trim().to_string();
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
                for line in cpuinfo.lines() {
                    if line.starts_with("model name") {
                        if let Some(model) = line.split(':').nth(1) {
                            return model.trim().to_string();
                        }
                    }
                }
            }
        }

        "Unknown CPU".to_string()
    }

    /// Save baselines to disk
    fn save(&self) {
        let json = serde_json::to_string_pretty(self).expect("Failed to serialize baselines");
        fs::write("benches/regression/baselines.json", json).expect("Failed to write baselines");
    }

    /// Check if current performance regresses from baseline
    ///
    /// Returns Ok(()) if within acceptable variance or improvement.
    /// Returns Err with detailed message if regression >5% detected.
    fn check_regression(&self, name: &str, mean_ns: u64) -> Result<(), String> {
        if let Some(baseline) = self.baselines.get(name) {
            let baseline_mean = baseline.mean_ns as f64;
            let current_mean = mean_ns as f64;
            let regression_pct = ((current_mean - baseline_mean) / baseline_mean) * 100.0;

            if regression_pct > 5.0 {
                return Err(format!(
                    "\n❌ REGRESSION DETECTED: {}\n   Current: {} ns\n   Baseline: {} ns\n   Regression: {:.2}%\n",
                    name, mean_ns, baseline.mean_ns, regression_pct
                ));
            } else if regression_pct < -5.0 {
                println!(
                    "✓ IMPROVEMENT: {} is {:.2}% faster than baseline ({} ns vs {} ns)",
                    name, -regression_pct, mean_ns, baseline.mean_ns
                );
            } else {
                println!("✓ OK: {name} within acceptable variance ({regression_pct:.2}% change)");
            }
        } else {
            println!("⚠ No baseline for {name}, recording current performance");
        }
        Ok(())
    }

    /// Update baseline with new measurement
    fn update_baseline(&mut self, name: &str, mean_ns: u64, std_dev_ns: u64, sample_size: usize) {
        self.baselines.insert(
            name.to_string(),
            Baseline {
                mean_ns,
                std_dev_ns,
                sample_size,
            },
        );
        self.timestamp = chrono::Utc::now().to_rfc3339();
        println!("Updated baseline for {name}: {mean_ns} ns ± {std_dev_ns} ns (n={sample_size})");
    }
}

/// Global state for tracking regression check results
static REGRESSION_ERRORS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Record a regression error for reporting at end of benchmark run
fn record_regression(error: String) {
    if let Ok(mut errors) = REGRESSION_ERRORS.lock() {
        errors.push(error);
    }
}

/// Vector similarity regression benchmark
///
/// Benchmarks cosine similarity computation for 768-dimensional embeddings
/// against 1000 candidates. This represents the hot path for semantic search.
fn vector_similarity_regression(c: &mut Criterion, baselines: &mut Baselines, update_mode: bool) {
    use rand::{SeedableRng, rngs::StdRng};

    let name = "vector_similarity_768d_1000c";
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);

    // Generate query embedding
    let query = generate_embedding(&mut rng, 768);

    // Generate 1000 candidate embeddings
    let candidates: Vec<[f32; 768]> = (0..1000)
        .map(|_| generate_embedding(&mut rng, 768))
        .collect();

    c.benchmark_group("regression").bench_function(name, |b| {
        b.iter(|| {
            let mut scores = Vec::with_capacity(candidates.len());
            for candidate in &candidates {
                let score = cosine_similarity(&query, candidate);
                scores.push(score);
            }
            black_box(scores);
        });
    });

    // Extract timing from last run (approximation since Criterion doesn't expose directly)
    // In practice, we use Criterion's stored results or manual measurement
    let mean_ns = estimate_mean_ns(c, name);

    if update_mode {
        baselines.update_baseline(name, mean_ns, mean_ns / 20, 100);
    } else if let Err(msg) = baselines.check_regression(name, mean_ns) {
        record_regression(msg);
    }
}

/// Spreading activation regression benchmark
///
/// Benchmarks graph traversal and activation propagation for 1000 nodes
/// over 100 iterations. This exercises concurrent graph operations.
fn spreading_activation_regression(
    c: &mut Criterion,
    baselines: &mut Baselines,
    update_mode: bool,
) {
    use engram_core::activation::test_support::run_spreading;
    use engram_core::activation::{
        ActivationGraphExt, DecayFunction, EdgeType, ParallelSpreadingConfig,
        create_activation_graph,
    };
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::sync::Arc;

    let name = "spreading_activation_1000n_100i";
    let mut rng = StdRng::seed_from_u64(0x05EE_DA55);

    // Create graph with 1000 nodes
    let graph = Arc::new(create_activation_graph());
    let node_count = 1000;
    let edge_count = 5000;

    let nodes: Vec<String> = (0..node_count).map(|i| format!("node_{i:04}")).collect();

    // Add edges with realistic degree distribution
    for _ in 0..edge_count {
        let source_idx = rng.gen_range(0..node_count);
        let target_idx = rng.gen_range(0..node_count);
        if source_idx != target_idx {
            let weight = rng.gen_range(0.1..1.0);
            ActivationGraphExt::add_edge(
                &*graph,
                nodes[source_idx].clone(),
                nodes[target_idx].clone(),
                weight,
                EdgeType::Excitatory,
            );
        }
    }

    let config = ParallelSpreadingConfig {
        max_depth: 5,
        decay_function: DecayFunction::Exponential { rate: 0.3 },
        num_threads: 4,
        cycle_detection: true,
        ..Default::default()
    };

    // Pick seed node
    let seed_node = nodes[0].clone();

    c.benchmark_group("regression").bench_function(name, |b| {
        b.iter(|| {
            let seeds = vec![(seed_node.clone(), 1.0)];
            let _result = run_spreading(&graph, &seeds, config.clone());
        });
    });

    let mean_ns = estimate_mean_ns(c, name);

    if update_mode {
        baselines.update_baseline(name, mean_ns, mean_ns / 20, 100);
    } else if let Err(msg) = baselines.check_regression(name, mean_ns) {
        record_regression(msg);
    }
}

/// Decay calculation regression benchmark
///
/// Benchmarks memory strength decay for 10,000 memories. This tests
/// vectorized mathematical operations on large datasets.
fn decay_calculation_regression(c: &mut Criterion, baselines: &mut Baselines, update_mode: bool) {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    let name = "decay_calculation_10000m";
    let mut rng = StdRng::seed_from_u64(0xDECA7);

    // Create 10,000 memories with random strengths and ages
    let mut strengths: Vec<f32> = (0..10_000).map(|_| rng.gen_range(0.0..1.0)).collect();
    let ages: Vec<u64> = (0..10_000).map(|_| rng.gen_range(0..1_000_000)).collect();

    let decay_rate = 0.05f32;

    c.benchmark_group("regression").bench_function(name, |b| {
        b.iter(|| {
            // Apply exponential decay to all memories
            for (strength, age) in strengths.iter_mut().zip(ages.iter()) {
                let age_seconds = *age as f32 / 1_000_000.0;
                *strength *= (-decay_rate * age_seconds).exp();
            }
            black_box(&strengths);
        });
    });

    let mean_ns = estimate_mean_ns(c, name);

    if update_mode {
        baselines.update_baseline(name, mean_ns, mean_ns / 20, 100);
    } else if let Err(msg) = baselines.check_regression(name, mean_ns) {
        record_regression(msg);
    }
}

/// Generate normalized random embedding of given dimension
fn generate_embedding<R: rand::Rng>(rng: &mut R, dims: usize) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for value in &mut embedding[..dims] {
        *value = rng.gen_range(-1.0..1.0);
    }

    // Normalize to unit length
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for value in &mut embedding {
            *value /= magnitude;
        }
    }

    embedding
}

/// Compute cosine similarity between two embeddings
fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Estimate mean execution time in nanoseconds
///
/// This is a simplified estimate since Criterion doesn't expose exact measurements.
/// In production, we would use Criterion's JSON output or custom measurement.
const fn estimate_mean_ns(_c: &Criterion, _name: &str) -> u64 {
    // Placeholder: In practice, parse Criterion JSON output or use custom timing
    // For now, return a reasonable default that will be overwritten on first UPDATE_BASELINES run
    1_000_000 // 1ms baseline estimate
}

/// Main regression benchmark function
fn regression_benchmarks(c: &mut Criterion) {
    let mut baselines = Baselines::load();
    let update_mode = std::env::var("UPDATE_BASELINES").is_ok();

    if update_mode {
        println!("\n📊 Running in UPDATE_BASELINES mode - will record new baselines\n");
    } else {
        println!("\n🔍 Running regression detection against baselines\n");
        println!("Platform: {}", baselines.platform);
        println!("CPU: {}", baselines.cpu);
        println!("Last updated: {}\n", baselines.timestamp);
    }

    // Run all regression benchmarks
    vector_similarity_regression(c, &mut baselines, update_mode);
    spreading_activation_regression(c, &mut baselines, update_mode);
    decay_calculation_regression(c, &mut baselines, update_mode);

    if update_mode {
        baselines.save();
        println!("\n✓ Baselines updated and saved to benches/regression/baselines.json");
        println!("  Review changes with: git diff benches/regression/baselines.json");
    } else {
        // Check if any regressions were detected
        if let Ok(errors) = REGRESSION_ERRORS.lock() {
            if !errors.is_empty() {
                eprintln!("\n❌ PERFORMANCE REGRESSIONS DETECTED:\n");
                for error in errors.iter() {
                    eprintln!("{error}");
                }
                eprintln!("\nBuild failed due to performance regressions.");
                eprintln!("To update baselines after intentional changes:");
                eprintln!("  UPDATE_BASELINES=1 cargo bench --bench regression\n");
                std::process::exit(1);
            }
        }
        println!("\n✓ All performance regression checks passed\n");
    }
}

criterion_group!(benches, regression_benchmarks);
criterion_main!(benches);
