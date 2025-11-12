//! Load testing CLI tool for Engram
//!
//! Generates realistic workload patterns for capacity testing and stress testing.
//! All workloads are deterministic when using --seed for reproducibility.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod comparative;
mod distribution;
mod hypothesis_testing;
mod metrics_collector;
mod replay;
mod report;
mod workload_generator;

use metrics_collector::MetricsCollector;
use report::ReportGenerator;
use workload_generator::{WorkloadConfig, WorkloadGenerator};

#[derive(Parser)]
#[command(name = "loadtest")]
#[command(about = "Load testing tool for Engram", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a load test using a predefined or custom scenario
    Run {
        /// Path to scenario TOML configuration
        #[arg(short, long)]
        scenario: PathBuf,

        /// Target operations per second (overrides scenario default)
        #[arg(short = 'r', long)]
        target_rate: Option<u64>,

        /// Test duration in seconds (overrides scenario default)
        #[arg(short, long)]
        duration: Option<u64>,

        /// Seed for deterministic workload generation
        #[arg(short, long)]
        seed: Option<u64>,

        /// Output file for results (JSON format)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Engram endpoint URL
        #[arg(long, default_value = "http://localhost:7432")]
        endpoint: String,
    },

    /// Replay a recorded traffic trace
    Replay {
        /// Path to trace file (JSON format)
        #[arg(short, long)]
        trace: PathBuf,

        /// Rate multiplier (1.0 = original speed, 2.0 = 2x speed)
        #[arg(long, default_value = "1.0")]
        rate_multiplier: f64,

        /// Output file for results (JSON format)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Engram endpoint URL
        #[arg(long, default_value = "http://localhost:7432")]
        endpoint: String,
    },

    /// List available predefined scenarios
    List,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            scenario,
            target_rate,
            duration,
            seed,
            output,
            endpoint,
        } => {
            run_load_test(scenario, target_rate, duration, seed, output, endpoint).await?;
        }
        Commands::Replay {
            trace,
            rate_multiplier,
            output,
            endpoint,
        } => {
            replay_trace(trace, rate_multiplier, output, endpoint).await?;
        }
        Commands::List => {
            list_scenarios()?;
        }
    }

    Ok(())
}

async fn run_load_test(
    scenario_path: PathBuf,
    target_rate: Option<u64>,
    duration: Option<u64>,
    seed: Option<u64>,
    output: Option<PathBuf>,
    endpoint: String,
) -> Result<()> {
    // Load scenario configuration
    let scenario_content = std::fs::read_to_string(&scenario_path)
        .with_context(|| format!("Failed to read scenario file: {}", scenario_path.display()))?;

    let mut config: WorkloadConfig = toml::from_str(&scenario_content)
        .with_context(|| format!("Failed to parse scenario TOML: {}", scenario_path.display()))?;

    // Apply overrides
    if let Some(rate) = target_rate {
        config.set_target_rate(rate);
    }
    if let Some(dur) = duration {
        config.set_duration(dur);
    }

    // Use provided seed or generate one
    let workload_seed = seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });

    tracing::info!(
        "Starting load test: {} (seed: {})",
        config.name(),
        workload_seed
    );
    tracing::info!("Target rate: {} ops/sec", config.target_rate());
    tracing::info!("Duration: {} seconds", config.duration());
    tracing::info!("Endpoint: {}", endpoint);

    // Create workload generator
    let generator = WorkloadGenerator::new(workload_seed, config.clone())?;

    // Create metrics collector
    let metrics = MetricsCollector::new();

    // Run the load test
    let start_time = std::time::Instant::now();
    let results = run_workload(generator, metrics, &endpoint, &config).await?;
    let elapsed = start_time.elapsed();

    tracing::info!("Load test completed in {:.2}s", elapsed.as_secs_f64());

    // Generate report
    let report_gen = ReportGenerator::new(results);
    let report = report_gen.generate(&config)?;

    // Display summary
    println!("\n{}", report.summary());

    // Save detailed results if output specified
    if let Some(output_path) = output {
        report.save_json(&output_path)?;
        tracing::info!("Detailed results saved to: {}", output_path.display());
    }

    // Check validation criteria
    if !report.meets_validation_criteria(&config) {
        tracing::warn!("Load test did not meet validation criteria!");
        std::process::exit(1);
    }

    Ok(())
}

async fn run_workload(
    mut generator: WorkloadGenerator,
    mut metrics: MetricsCollector,
    endpoint: &str,
    config: &WorkloadConfig,
) -> Result<MetricsCollector> {
    use indicatif::{ProgressBar, ProgressStyle};
    use std::time::{Duration, Instant};

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    let duration = Duration::from_secs(config.duration());
    let target_rate = config.target_rate();
    let interval = Duration::from_secs_f64(1.0 / target_rate as f64);

    let pb = ProgressBar::new(duration.as_secs());
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len}s {msg}")?
            .progress_chars("=>-"),
    );

    let test_start = Instant::now();
    let mut next_op_time = Instant::now();
    let mut operations_sent = 0u64;

    while test_start.elapsed() < duration {
        let now = Instant::now();

        if now >= next_op_time {
            // Generate next operation
            let operation = generator.next_operation();
            let op_start = Instant::now();

            // Execute operation
            let result = execute_operation(&client, endpoint, &operation).await;
            let op_elapsed = op_start.elapsed();

            // Record metrics
            metrics.record_operation(operation.op_type(), op_elapsed, result.is_ok());

            if let Err(e) = result {
                tracing::warn!(
                    operation = ?operation.op_type(),
                    error = %e,
                    "Operation failed"
                );
            }

            operations_sent += 1;
            next_op_time += interval;

            // Update progress bar every second
            let elapsed_secs = test_start.elapsed().as_secs();
            if pb.position() < elapsed_secs {
                pb.set_position(elapsed_secs);
                let current_rate = metrics.current_throughput();
                pb.set_message(format!(
                    "Rate: {:.0} ops/sec | P99: {:.1}ms | Errors: {:.2}%",
                    current_rate,
                    metrics.p99_latency_ms(),
                    metrics.error_rate() * 100.0
                ));
            }
        } else {
            // Sleep for a short interval
            tokio::time::sleep(Duration::from_micros(100)).await;
        }
    }

    pb.finish_with_message(format!("Completed {} operations", operations_sent));

    Ok(metrics)
}

async fn execute_operation(
    client: &reqwest::Client,
    endpoint: &str,
    operation: &workload_generator::Operation,
) -> Result<()> {
    use workload_generator::Operation;

    match operation {
        Operation::Store { memory } => {
            let url = format!("{}/api/v1/memories", endpoint);
            // Engram's REST API requires 'content' field for memory creation
            // For load testing, we use synthetic content with the embedding
            let body = serde_json::json!({
                "content": format!("Load test memory {}", uuid::Uuid::new_v4()),
                "confidence": memory.confidence,
                "embedding": memory.embedding,
            });
            let response = client
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("Store: network error")?;

            let status = response.status();
            if !status.is_success() {
                let error_body = response.text().await.unwrap_or_default();
                anyhow::bail!("Store failed with status {}: {}", status, error_body);
            }
        }
        Operation::Recall { cue } => {
            // Engram uses GET /api/v1/memories/recall with query params
            let embedding_json = serde_json::to_string(&cue.embedding)?;
            let url = format!(
                "{}/api/v1/memories/recall?embedding={}&threshold={}&max_results={}&space={}",
                endpoint,
                urlencoding::encode(&embedding_json),
                cue.threshold,
                cue.max_depth,
                urlencoding::encode(&cue.memory_space)
            );
            let response = client
                .get(&url)
                .send()
                .await
                .context("Recall: network error")?;

            let status = response.status();
            if !status.is_success() {
                let error_body = response.text().await.unwrap_or_default();
                anyhow::bail!("Recall failed with status {}: {}", status, error_body);
            }
        }
        Operation::EmbeddingSearch { query: _, k } => {
            // Engram uses GET /api/v1/memories/search with text query parameter
            // Generate synthetic query text for load testing
            let synthetic_query = format!("test query {}", uuid::Uuid::new_v4());
            let url = format!(
                "{}/api/v1/memories/search?query={}&limit={}",
                endpoint,
                urlencoding::encode(&synthetic_query),
                k
            );
            let response = client
                .get(&url)
                .send()
                .await
                .context("Search: network error")?;

            let status = response.status();
            if !status.is_success() {
                let error_body = response.text().await.unwrap_or_default();
                anyhow::bail!("Search failed with status {}: {}", status, error_body);
            }
        }
        Operation::PatternCompletion { partial } => {
            let url = format!("{}/api/v1/complete", endpoint);
            // Engram requires exactly 768 dimensions in partial_embedding
            // The generator provides embedding_dim/2 values (e.g., 384)
            // Create a 768-element array with Some() for known values and None for masked positions
            const EMBEDDING_DIM: usize = 768;
            let mut partial_embedding: Vec<Option<f32>> = vec![None; EMBEDDING_DIM];

            // Fill in known values at alternating positions to simulate partial pattern
            // This creates a realistic pattern completion scenario where ~half the dimensions are known
            for (i, &value) in partial.iter().enumerate() {
                if i * 2 < EMBEDDING_DIM {
                    partial_embedding[i * 2] = Some(value);
                }
            }

            // Pattern completion configuration for cold-start scenarios
            //
            // BIOLOGICAL JUSTIFICATION:
            // Pattern completion relies on hippocampal CA3 autoassociative networks that
            // require learned attractor basins. In cold-start scenarios (no pre-trained
            // patterns), CA3 convergence is slow, leading to low completion confidence.
            //
            // Default ca1_threshold (0.7) assumes warm-start with learned patterns.
            // For cold-start benchmarking, we use lower thresholds appropriate for
            // unprimed memory systems.
            //
            // CONFIGURATION RATIONALE:
            // - ca1_threshold: 0.3 (vs default 0.7)
            //   Allows completion when CA3 takes 3-5 iterations to converge
            //   Still blocks truly random noise (>6 iterations)
            //
            // - max_iterations: 10 (vs default 7)
            //   More convergence time without learned attractors
            //   Biological constraint (theta rhythm) less critical for synthetic benchmarks
            //
            // - num_hypotheses: 3 (default)
            //   Standard System 2 reasoning capacity
            //
            // See tmp/finish-task-006-with-no-loose-ends.md for full investigation.
            //
            // TODO: Move to scenario-level configuration (scenarios/*/hybrid_*.toml)
            //       to support both cold-start and warm-start pattern completion tests.
            let body = serde_json::json!({
                "partial_episode": {
                    "known_fields": {
                        "what": format!("test pattern {}", uuid::Uuid::new_v4())
                    },
                    "partial_embedding": partial_embedding,
                    "cue_strength": 0.7
                },
                "config": {
                    "ca1_threshold": 0.3,
                    "num_hypotheses": 3,
                    "max_iterations": 10
                }
            });
            let response = client
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("PatternCompletion: network error")?;

            let status = response.status();
            if !status.is_success() {
                let error_body = response.text().await.unwrap_or_default();
                anyhow::bail!(
                    "PatternCompletion failed with status {}: {}",
                    status,
                    error_body
                );
            }
        }
    }

    Ok(())
}

async fn replay_trace(
    trace_path: PathBuf,
    rate_multiplier: f64,
    output: Option<PathBuf>,
    endpoint: String,
) -> Result<()> {
    tracing::info!("Replaying trace: {}", trace_path.display());
    tracing::info!("Rate multiplier: {}x", rate_multiplier);
    tracing::info!("Endpoint: {}", endpoint);

    let trace = replay::load_trace(&trace_path)?;
    let metrics = replay::replay_trace(trace, rate_multiplier, &endpoint).await?;

    // Generate report
    let config = WorkloadConfig::from_trace_metadata()?;
    let report_gen = ReportGenerator::new(metrics);
    let report = report_gen.generate(&config)?;

    println!("\n{}", report.summary());

    if let Some(output_path) = output {
        report.save_json(&output_path)?;
        tracing::info!("Results saved to: {}", output_path.display());
    }

    Ok(())
}

fn list_scenarios() -> Result<()> {
    println!("Available predefined scenarios:\n");

    let scenarios = [
        (
            "write_heavy.toml",
            "80% store, 20% recall - initial ingestion",
        ),
        (
            "read_heavy.toml",
            "20% store, 80% recall - query-heavy workload",
        ),
        (
            "mixed_balanced.toml",
            "50/50 read/write - balanced operations",
        ),
        (
            "burst_traffic.toml",
            "Periodic load spikes - realistic traffic",
        ),
        ("embeddings_search.toml", "Similarity search focused"),
        ("consolidation.toml", "Background consolidation during load"),
        ("multi_tenant.toml", "Multiple memory spaces concurrently"),
    ];

    for (name, description) in &scenarios {
        println!("  {} - {}", name, description);
    }

    println!("\nUsage:");
    println!("  loadtest run --scenario scenarios/write_heavy.toml --duration 3600");

    Ok(())
}
