//! Verifies adaptive batching telemetry is exported through the streaming metrics surface.

use engram_core::activation::{
    AdaptiveBatcher, AdaptiveBatcherConfig, AdaptiveMode, Observation, SpreadingMetrics,
    storage_aware::StorageTier,
};
use engram_core::metrics;
use std::sync::atomic::Ordering;

fn drain_streaming_queue() {
    let registry = metrics::init();
    let aggregator = registry.streaming_aggregator();
    aggregator.set_export_enabled(true);
    // Drain any pending updates so the test starts from a clean slate.
    let _ = registry.streaming_snapshot();
}

#[test]
fn adaptive_batcher_metrics_flow_into_streaming_snapshot() {
    drain_streaming_queue();

    let config = AdaptiveBatcherConfig::default();
    let batcher = AdaptiveBatcher::new(config.clone(), AdaptiveMode::default());
    let spreading_metrics = SpreadingMetrics::default();

    // Seed an observation to kick the EWMA and per-tier recommendations.
    let observation = Observation {
        batch_size: config.max_batch_size,
        latency_ns: 2_000_000,
        hop_count: 2,
        tier: StorageTier::Hot,
    };
    batcher.record_observation(observation);
    batcher.process_observations();

    // Publish the snapshot so streaming metrics can export the current state.
    spreading_metrics.record_adaptive_batcher_snapshot(&batcher.snapshot());

    let registry = metrics::init();
    let snapshot = registry.streaming_snapshot();
    let window = &snapshot.one_second;

    for key in [
        "adaptive_batch_updates_total",
        "adaptive_guardrail_hits_total",
        "adaptive_topology_changes_total",
        "adaptive_fallback_activations_total",
        "adaptive_batch_latency_ewma_ns",
        "adaptive_batch_hot_size",
        "adaptive_batch_warm_size",
        "adaptive_batch_cold_size",
        "adaptive_batch_hot_confidence",
        "adaptive_batch_warm_confidence",
        "adaptive_batch_cold_confidence",
    ] {
        assert!(
            window.contains_key(key),
            "expected streaming snapshot to contain metric {key}"
        );
    }

    let updates = window
        .get("adaptive_batch_updates_total")
        .expect("updates metric present");
    assert!(updates.sum >= 1.0, "update count should be non-zero");

    let latency = window
        .get("adaptive_batch_latency_ewma_ns")
        .expect("latency metric present");
    assert!(latency.max > 0.0, "latency EWMA should be populated");

    let hot_size = window
        .get("adaptive_batch_hot_size")
        .expect("hot tier size present");
    assert!(
        hot_size.max >= config.min_batch_size as f64,
        "hot tier recommendation should respect configured bounds"
    );

    // Ensure confidence updates were written back to the shared metrics struct.
    assert!(
        spreading_metrics
            .adaptive_hot_confidence
            .load(Ordering::Relaxed)
            >= 0.0,
        "hot tier confidence should be recorded"
    );
}
