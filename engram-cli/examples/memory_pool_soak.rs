//! Simulate a long-running pool metrics soak and capture start/mid/end snapshots.

use std::{
    env, fs,
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

use anyhow::{Context, Result};
use engram_core::activation::{ActivationRecordPoolStats, SpreadingMetrics};
use engram_core::metrics;
use serde_json::json;

type Stage = (&'static str, usize);

const STAGES: [Stage; 3] = [("start", 0), ("mid", 300), ("end", 599)];
const ITERATIONS: usize = 600;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .without_time()
        .try_init()
        .ok();

    let output_dir = env::args().nth(1).map_or_else(
        || PathBuf::from("docs/assets/metrics/2025-10-12-longrun"),
        PathBuf::from,
    );
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("unable to create output directory {}", output_dir.display()))?;

    let registry = metrics::init();
    // Drain stale updates from previous runs.
    let _ = registry.streaming_snapshot();

    let spreading_metrics = SpreadingMetrics::default();

    for iteration in 0..ITERATIONS {
        let stats = synthesize_stats(iteration);
        spreading_metrics.record_pool_snapshot(&stats);

        if let Some((label, _)) = STAGES.iter().find(|stage| stage.1 == iteration) {
            write_snapshot(&registry, &output_dir, label)
                .with_context(|| format!("failed to write snapshot for stage {label}"))?;
            registry.log_streaming_snapshot(label);
        }

        thread::sleep(Duration::from_millis(10));
    }

    Ok(())
}

fn synthesize_stats(iteration: usize) -> ActivationRecordPoolStats {
    let base_available = 40 + (iteration % 15);
    let in_flight = 4 + (iteration % 6) as u64;
    let high_water = 80 + (iteration % 10);
    let total_created = 120 + (iteration * 2) as u64;
    let total_reused = total_created.saturating_sub(20);
    let misses = (iteration / 20) as u64 + 5;
    let utilization = ((base_available as f32) / (high_water as f32)).clamp(0.0, 1.0);
    let hit_rate = 1.0 - (misses as f32 / (total_created.max(1) as f32));

    ActivationRecordPoolStats {
        available: base_available,
        in_flight,
        high_water_mark: high_water,
        total_created,
        total_reused,
        misses,
        hit_rate,
        utilization,
        release_failures: (iteration / 100) as u64,
    }
}

fn write_snapshot(registry: &metrics::MetricsRegistry, dir: &Path, label: &str) -> Result<()> {
    let snapshot = registry.streaming_snapshot();
    let export = registry.streaming_stats();
    let payload = json!({
        "snapshot": snapshot,
        "export": export,
    });

    let path = dir.join(format!("{label}_snapshot.json"));
    fs::write(&path, serde_json::to_string_pretty(&payload)?)
        .with_context(|| format!("failed writing {}", path.display()))
}
