//! Regression harness for streaming metrics reset behavior.
//!
//! Ensures that core spreading metrics reset clears the pool counters while the
//! streaming snapshot retains the last-seen values so operators keep a stable
//! view until fresh updates are published.

use std::sync::atomic::Ordering;

use engram_core::activation::{ActivationRecordPoolStats, SpreadingMetrics};
use engram_core::metrics;

fn drain_streaming_queue() {
    let registry = metrics::init();
    let aggregator = registry.streaming_aggregator();

    // Ensure stale payloads from other tests don't influence this run.
    aggregator.set_export_enabled(true);
    // Drain any queued updates so we have a clean baseline.
    let _ = registry.streaming_snapshot();
}

#[test]
fn spreading_metrics_reset_clears_pool_counters_and_streaming_snapshot() {
    drain_streaming_queue();

    let metrics = SpreadingMetrics::default();

    // Seed counters with non-zero values so the reset has work to do.
    metrics.total_activations.store(42, Ordering::Relaxed);
    metrics.cache_hits.store(21, Ordering::Relaxed);
    metrics.cache_misses.store(7, Ordering::Relaxed);

    let stats = ActivationRecordPoolStats {
        available: 12,
        in_flight: 8,
        high_water_mark: 24,
        total_created: 48,
        total_reused: 32,
        misses: 16,
        hit_rate: 0.667,
        utilization: 0.5,
        release_failures: 3,
    };

    // Publishing a snapshot should enqueue gauge updates for the streaming
    // export so the HTTP layer observes non-zero data before the reset.
    metrics.record_pool_snapshot(&stats);

    let registry = metrics::init();
    let pre_reset_snapshot = registry.streaming_snapshot();
    assert!(
        !pre_reset_snapshot.one_second.is_empty(),
        "expected pre-reset snapshot to contain queued gauge values"
    );

    // Act: reset metrics and validate counters are cleared.
    metrics.reset();

    assert_eq!(metrics.total_activations.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.cache_hits.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.cache_misses.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.pool_available.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.pool_in_flight.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.pool_high_water_mark.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.pool_total_created.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.pool_total_reused.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.pool_miss_count.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.pool_release_failures.load(Ordering::Relaxed), 0);
    assert!(metrics.pool_hit_rate().abs() < f32::EPSILON);
    assert!(metrics.pool_utilization().abs() < f32::EPSILON);

    let post_reset_snapshot = registry.streaming_snapshot();
    assert_eq!(
        post_reset_snapshot.schema_version,
        pre_reset_snapshot.schema_version
    );
    assert_eq!(
        post_reset_snapshot.one_second,
        pre_reset_snapshot.one_second
    );
    assert_eq!(
        post_reset_snapshot.ten_seconds,
        pre_reset_snapshot.ten_seconds
    );
    assert_eq!(
        post_reset_snapshot.one_minute,
        pre_reset_snapshot.one_minute
    );
    assert_eq!(
        post_reset_snapshot.five_minutes,
        pre_reset_snapshot.five_minutes
    );
    assert_eq!(post_reset_snapshot.spreading, pre_reset_snapshot.spreading);
}
