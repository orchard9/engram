//! Emit a representative streaming metrics snapshot and log entry for docs.

use anyhow::Result;
use engram_core::activation::{
    ActivationRecordPoolStats, AdaptiveBatcher, AdaptiveBatcherConfig, AdaptiveMode, Observation,
    SpreadingMetrics, storage_aware::StorageTier,
};
use engram_core::metrics;
use serde_json::json;

fn record_pool_snapshot(metrics: &SpreadingMetrics, stats: &ActivationRecordPoolStats) {
    metrics.record_pool_snapshot(stats);
}

const fn sample_stats() -> ActivationRecordPoolStats {
    ActivationRecordPoolStats {
        available: 12,
        in_flight: 4,
        high_water_mark: 32,
        total_created: 48,
        total_reused: 36,
        misses: 12,
        hit_rate: 0.75,
        utilization: 0.5,
        release_failures: 1,
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .without_time()
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .try_init()
        .ok();

    let registry = metrics::init();
    let spreading_metrics = SpreadingMetrics::default();
    let stats = sample_stats();
    let adaptive_config = AdaptiveBatcherConfig::default();
    let adaptive_batcher = AdaptiveBatcher::new(adaptive_config.clone(), AdaptiveMode::default());

    // Emit a log snapshot example.
    record_pool_snapshot(&spreading_metrics, &stats);
    adaptive_batcher.record_observation(Observation {
        batch_size: adaptive_config.max_batch_size,
        latency_ns: 1_500_000,
        hop_count: 2,
        tier: StorageTier::Hot,
    });
    adaptive_batcher.process_observations();
    spreading_metrics.record_adaptive_batcher_snapshot(&adaptive_batcher.snapshot());
    registry.log_streaming_snapshot("docs-example");

    // Refresh metrics so the HTTP snapshot has data as well.
    record_pool_snapshot(&spreading_metrics, &stats);
    spreading_metrics.record_adaptive_batcher_snapshot(&adaptive_batcher.snapshot());
    let snapshot = registry.streaming_snapshot();
    let export = registry.streaming_stats();

    let payload = json!({
        "snapshot": snapshot,
        "export": export,
    });

    println!("{}", serde_json::to_string_pretty(&payload)?);

    Ok(())
}
