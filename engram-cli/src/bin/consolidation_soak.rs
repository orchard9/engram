//! Consolidation soak harness for capturing long-form metrics and belief deltas.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use chrono::Utc;
use clap::Parser;
use engram_core::completion::{CompletionConfig, ConsolidationScheduler, SchedulerConfig};
use engram_core::memory::EpisodeBuilder;
use engram_core::metrics;
use engram_core::{Confidence, MemoryStore};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tokio::time::{Instant, MissedTickBehavior};
use tracing::{Level, info};

#[derive(Parser, Debug)]
#[command(
    name = "consolidation-soak",
    about = "Run a long-form consolidation soak and capture metrics."
)]
struct Args {
    /// Duration of the soak test in seconds (default: 3600)
    #[arg(long, default_value = "3600")]
    duration_secs: u64,

    /// Scheduler interval in seconds
    #[arg(long, default_value = "60")]
    scheduler_interval_secs: u64,

    /// Minimum episodes required before scheduler runs
    #[arg(long, default_value = "64")]
    min_episodes_threshold: usize,

    /// Maximum episodes processed per scheduler run
    #[arg(long, default_value = "256")]
    max_episodes_per_run: usize,

    /// Interval at which snapshots/metrics are sampled (seconds)
    #[arg(long, default_value = "60")]
    sample_interval_secs: u64,

    /// Number of episodes to inject on each sampling tick
    #[arg(long, default_value = "32")]
    episodes_per_tick: usize,

    /// Maximum episodes aggregated when forcing on-demand snapshots (0 => all)
    #[arg(long, default_value = "0")]
    max_episodes_per_snapshot: usize,

    /// Capacity for the in-memory store
    #[arg(long, default_value = "4096")]
    store_capacity: usize,

    /// Initial episodes to seed before scheduler starts
    #[arg(long, default_value = "512")]
    initial_episodes: usize,

    /// Random seed used for episode generation
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Output directory for metrics and snapshots
    #[arg(long, default_value = "docs/assets/consolidation/baseline")]
    output_dir: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    let args = Args::parse();
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create output dir {}", args.output_dir.display()))?;

    let _metrics_registry = metrics::init();
    let store = Arc::new(MemoryStore::new(args.store_capacity));
    store.set_consolidation_alert_log_path(args.output_dir.join("belief_updates.jsonl"));

    let mut rng = StdRng::seed_from_u64(args.seed);
    seed_store(&store, args.initial_episodes, &mut rng);

    let scheduler_config = SchedulerConfig {
        consolidation_interval_secs: args.scheduler_interval_secs,
        min_episodes_threshold: args.min_episodes_threshold,
        max_episodes_per_run: args.max_episodes_per_run,
        enabled: true,
    };

    let scheduler = Arc::new(ConsolidationScheduler::new(
        CompletionConfig::default(),
        scheduler_config,
    ));
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let scheduler_handle = scheduler.clone().spawn(Arc::clone(&store), shutdown_rx);

    let mut snapshot_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(args.output_dir.join("snapshots.jsonl"))
        .context("failed to open snapshots output")?;
    let mut metrics_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(args.output_dir.join("metrics.jsonl"))
        .context("failed to open metrics output")?;

    let mut ticker = tokio::time::interval(Duration::from_secs(args.sample_interval_secs));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

    let start = Instant::now();
    info!(
        duration_secs = args.duration_secs,
        sample_interval_secs = args.sample_interval_secs,
        "Starting consolidation soak"
    );

    while start.elapsed() < Duration::from_secs(args.duration_secs) {
        ticker.tick().await;

        seed_store(&store, args.episodes_per_tick, &mut rng);

        let snapshot = store.consolidation_snapshot(args.max_episodes_per_snapshot);
        let snapshot_summary = serde_json::json!({
            "captured_at": Utc::now(),
            "generated_at": snapshot.generated_at,
            "pattern_count": snapshot.patterns.len(),
            "stats": {
                "total_replays": snapshot.stats.total_replays,
                "successful_consolidations": snapshot.stats.successful_consolidations,
                "failed_consolidations": snapshot.stats.failed_consolidations,
                "average_replay_speed": snapshot.stats.average_replay_speed,
            }
        });
        snapshot_file
            .write_all(serde_json::to_string(&snapshot_summary)?.as_bytes())
            .context("failed to write snapshot summary")?;
        snapshot_file.write_all(b"\n")?;
        snapshot_file.flush()?;

        if let Some(registry) = metrics::metrics() {
            let metrics_snapshot = registry.streaming_snapshot();
            metrics_file
                .write_all(serde_json::to_string(&metrics_snapshot)?.as_bytes())
                .context("failed to write metrics snapshot")?;
            metrics_file.write_all(b"\n")?;
            metrics_file.flush()?;
        }
    }

    shutdown_tx.send(true).ok();
    if let Err(error) = scheduler_handle.await {
        tracing::warn!(?error, "scheduler join returned error");
    }

    info!("Consolidation soak complete");
    Ok(())
}

fn seed_store(store: &MemoryStore, count: usize, rng: &mut StdRng) {
    for idx in 0..count {
        let id = format!("soak_ep_{}_{}", Utc::now().timestamp(), idx);
        let embedding = random_embedding(rng, idx as f32);
        let episode = EpisodeBuilder::new()
            .id(id)
            .when(Utc::now())
            .what(format!("seeded episode {idx}"))
            .embedding(embedding)
            .confidence(Confidence::exact(rng.gen_range(0.6..0.98)))
            .build();
        store.store(episode);
    }
}

fn random_embedding(rng: &mut StdRng, base: f32) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for (idx, slot) in embedding.iter_mut().enumerate() {
        let phase = base + idx as f32 * 0.017;
        let noise = rng.gen_range(-0.05..0.05);
        *slot = (phase.sin() + noise).clamp(-1.0, 1.0);
    }

    let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }
    embedding
}
