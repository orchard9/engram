//! Prometheus text format exporter for streaming metrics
//!
//! Converts AggregatedMetrics snapshots to Prometheus exposition format.
//! This enables compatibility with Prometheus scrapers while preserving the
//! streaming-first architecture and JSON endpoint for SSE consumers.

use super::streaming::{AggregatedMetrics, MetricAggregate, WindowSnapshot};
use std::fmt::Write as FmtWrite;

/// Convert aggregated metrics snapshot to Prometheus text format
///
/// This exporter translates the internal JSON-based streaming metrics into
/// Prometheus exposition format for scraping. It uses the 5-minute window
/// aggregates to provide stable metric values that align with Prometheus
/// scrape intervals (typically 10-15s).
///
/// # Metric Type Mapping
/// - Counters: Use `sum` from aggregation window
/// - Gauges: Use `mean` from aggregation window
/// - Histograms: Expose as summary with p50, p90, p99 quantiles
///
/// # Example Output
/// ```text
/// # HELP engram_spreading_activations_total Total activation operations
/// # TYPE engram_spreading_activations_total counter
/// engram_spreading_activations_total 12450
///
/// # HELP engram_consolidation_freshness_seconds Age of consolidation snapshot
/// # TYPE engram_consolidation_freshness_seconds gauge
/// engram_consolidation_freshness_seconds 145.2
/// ```
#[must_use]
pub fn to_prometheus_text(metrics: &AggregatedMetrics) -> String {
    let mut output = String::with_capacity(8192);

    // Use 5-minute window for stable metrics aligned with scrape intervals
    let snapshot = &metrics.five_minutes;

    // Export spreading activation metrics
    export_counter(
        &mut output,
        "engram_spreading_activations_total",
        "Total activation operations",
        snapshot,
        "engram_spreading_activations_total",
    );

    export_summary(
        &mut output,
        "engram_spreading_latency_hot_seconds",
        "Hot tier activation latency",
        snapshot,
        "engram_spreading_latency_hot_seconds",
    );

    export_summary(
        &mut output,
        "engram_spreading_latency_warm_seconds",
        "Warm tier activation latency",
        snapshot,
        "engram_spreading_latency_warm_seconds",
    );

    export_summary(
        &mut output,
        "engram_spreading_latency_cold_seconds",
        "Cold tier activation latency",
        snapshot,
        "engram_spreading_latency_cold_seconds",
    );

    export_counter(
        &mut output,
        "engram_spreading_latency_budget_violations_total",
        "Activations exceeding latency SLO",
        snapshot,
        "engram_spreading_latency_budget_violations_total",
    );

    export_counter(
        &mut output,
        "engram_spreading_fallback_total",
        "GPU to CPU fallback count",
        snapshot,
        "engram_spreading_fallback_total",
    );

    export_counter(
        &mut output,
        "engram_spreading_failures_total",
        "Failed activation operations",
        snapshot,
        "engram_spreading_failures_total",
    );

    export_gauge(
        &mut output,
        "engram_spreading_breaker_state",
        "Circuit breaker state (0=closed, 1=open, 2=half-open)",
        snapshot,
        "engram_spreading_breaker_state",
    );

    export_counter(
        &mut output,
        "engram_spreading_breaker_transitions_total",
        "Circuit breaker state changes",
        snapshot,
        "engram_spreading_breaker_transitions_total",
    );

    export_counter(
        &mut output,
        "engram_spreading_gpu_launch_total",
        "GPU kernel launches",
        snapshot,
        "engram_spreading_gpu_launch_total",
    );

    export_counter(
        &mut output,
        "engram_spreading_gpu_fallback_total",
        "GPU fallback to CPU",
        snapshot,
        "engram_spreading_gpu_fallback_total",
    );

    export_gauge(
        &mut output,
        "engram_spreading_pool_utilization",
        "Activation pool utilization (0.0-1.0)",
        snapshot,
        "engram_spreading_pool_utilization",
    );

    export_gauge(
        &mut output,
        "engram_spreading_pool_hit_rate",
        "Pool cache hit rate (0.0-1.0)",
        snapshot,
        "engram_spreading_pool_hit_rate",
    );

    // Export consolidation metrics
    export_counter(
        &mut output,
        "engram_consolidation_runs_total",
        "Successful consolidation runs",
        snapshot,
        "engram_consolidation_runs_total",
    );

    export_counter(
        &mut output,
        "engram_consolidation_failures_total",
        "Failed consolidation runs",
        snapshot,
        "engram_consolidation_failures_total",
    );

    export_gauge(
        &mut output,
        "engram_consolidation_novelty_gauge",
        "Latest novelty delta from scheduler",
        snapshot,
        "engram_consolidation_novelty_gauge",
    );

    export_gauge(
        &mut output,
        "engram_consolidation_novelty_variance",
        "Novelty variance across patterns",
        snapshot,
        "engram_consolidation_novelty_variance",
    );

    export_gauge(
        &mut output,
        "engram_consolidation_citation_churn",
        "Citation change rate (0.0-1.0)",
        snapshot,
        "engram_consolidation_citation_churn",
    );

    export_gauge(
        &mut output,
        "engram_consolidation_freshness_seconds",
        "Snapshot age in seconds",
        snapshot,
        "engram_consolidation_freshness_seconds",
    );

    export_gauge(
        &mut output,
        "engram_consolidation_citations_current",
        "Total citations in snapshot",
        snapshot,
        "engram_consolidation_citations_current",
    );

    // Export storage/compaction metrics
    export_counter(
        &mut output,
        "engram_compaction_attempts_total",
        "Compaction initiation count",
        snapshot,
        "engram_compaction_attempts_total",
    );

    export_counter(
        &mut output,
        "engram_compaction_success_total",
        "Successful compactions",
        snapshot,
        "engram_compaction_success_total",
    );

    export_counter(
        &mut output,
        "engram_compaction_rollback_total",
        "Rolled-back compactions",
        snapshot,
        "engram_compaction_rollback_total",
    );

    export_counter(
        &mut output,
        "engram_compaction_episodes_removed",
        "Episodes removed via compaction",
        snapshot,
        "engram_compaction_episodes_removed",
    );

    export_counter(
        &mut output,
        "engram_compaction_storage_saved_bytes",
        "Bytes reclaimed",
        snapshot,
        "engram_compaction_storage_saved_bytes",
    );

    export_counter(
        &mut output,
        "engram_wal_recovery_successes_total",
        "WAL recovery successes",
        snapshot,
        "engram_wal_recovery_successes_total",
    );

    export_counter(
        &mut output,
        "engram_wal_recovery_failures_total",
        "WAL recovery failures",
        snapshot,
        "engram_wal_recovery_failures_total",
    );

    export_summary(
        &mut output,
        "engram_wal_recovery_duration_seconds",
        "WAL recovery latency",
        snapshot,
        "engram_wal_recovery_duration_seconds",
    );

    export_counter(
        &mut output,
        "engram_wal_compaction_runs_total",
        "WAL compaction operations",
        snapshot,
        "engram_wal_compaction_runs_total",
    );

    export_counter(
        &mut output,
        "engram_wal_compaction_bytes_reclaimed",
        "Bytes reclaimed from WAL",
        snapshot,
        "engram_wal_compaction_bytes_reclaimed",
    );

    // Export activation pool metrics
    export_gauge(
        &mut output,
        "activation_pool_available_records",
        "Available pool slots",
        snapshot,
        "activation_pool_available_records",
    );

    export_gauge(
        &mut output,
        "activation_pool_in_flight_records",
        "In-use pool slots",
        snapshot,
        "activation_pool_in_flight_records",
    );

    export_gauge(
        &mut output,
        "activation_pool_high_water_mark",
        "Peak pool usage",
        snapshot,
        "activation_pool_high_water_mark",
    );

    export_gauge(
        &mut output,
        "activation_pool_total_created",
        "Total records created",
        snapshot,
        "activation_pool_total_created",
    );

    export_gauge(
        &mut output,
        "activation_pool_total_reused",
        "Total reused records",
        snapshot,
        "activation_pool_total_reused",
    );

    export_gauge(
        &mut output,
        "activation_pool_miss_count",
        "Cache misses",
        snapshot,
        "activation_pool_miss_count",
    );

    export_gauge(
        &mut output,
        "activation_pool_release_failures",
        "Failed releases",
        snapshot,
        "activation_pool_release_failures",
    );

    export_gauge(
        &mut output,
        "activation_pool_hit_rate",
        "Cache hit rate (0.0-1.0)",
        snapshot,
        "activation_pool_hit_rate",
    );

    export_gauge(
        &mut output,
        "activation_pool_utilization",
        "Pool utilization ratio (0.0-1.0)",
        snapshot,
        "activation_pool_utilization",
    );

    // Export adaptive batching metrics
    export_counter(
        &mut output,
        "adaptive_batch_updates_total",
        "Adaptive controller updates",
        snapshot,
        "adaptive_batch_updates_total",
    );

    export_counter(
        &mut output,
        "adaptive_guardrail_hits_total",
        "Guardrail constraint activations",
        snapshot,
        "adaptive_guardrail_hits_total",
    );

    export_counter(
        &mut output,
        "adaptive_topology_changes_total",
        "Topology change detections",
        snapshot,
        "adaptive_topology_changes_total",
    );

    export_counter(
        &mut output,
        "adaptive_fallback_activations_total",
        "Adaptive fallback activations",
        snapshot,
        "adaptive_fallback_activations_total",
    );

    export_gauge(
        &mut output,
        "adaptive_batch_latency_ewma_ns",
        "Smoothed latency EWMA",
        snapshot,
        "adaptive_batch_latency_ewma_ns",
    );

    export_gauge(
        &mut output,
        "adaptive_batch_hot_size",
        "Hot tier batch size",
        snapshot,
        "adaptive_batch_hot_size",
    );

    export_gauge(
        &mut output,
        "adaptive_batch_warm_size",
        "Warm tier batch size",
        snapshot,
        "adaptive_batch_warm_size",
    );

    export_gauge(
        &mut output,
        "adaptive_batch_cold_size",
        "Cold tier batch size",
        snapshot,
        "adaptive_batch_cold_size",
    );

    export_gauge(
        &mut output,
        "adaptive_batch_hot_confidence",
        "Hot tier convergence confidence (0.0-1.0)",
        snapshot,
        "adaptive_batch_hot_confidence",
    );

    export_gauge(
        &mut output,
        "adaptive_batch_warm_confidence",
        "Warm tier convergence confidence (0.0-1.0)",
        snapshot,
        "adaptive_batch_warm_confidence",
    );

    export_gauge(
        &mut output,
        "adaptive_batch_cold_confidence",
        "Cold tier convergence confidence (0.0-1.0)",
        snapshot,
        "adaptive_batch_cold_confidence",
    );

    output
}

/// Export a counter metric
fn export_counter(
    output: &mut String,
    name: &str,
    help: &str,
    snapshot: &WindowSnapshot,
    metric_key: &str,
) {
    if let Some(aggregate) = snapshot.get(metric_key) {
        let _ = writeln!(output, "# HELP {name} {help}");
        let _ = writeln!(output, "# TYPE {name} counter");
        let sum = aggregate.sum as u64;
        let _ = writeln!(output, "{name} {sum}");
        let _ = writeln!(output);
    }
}

/// Export a gauge metric
fn export_gauge(
    output: &mut String,
    name: &str,
    help: &str,
    snapshot: &WindowSnapshot,
    metric_key: &str,
) {
    if let Some(aggregate) = snapshot.get(metric_key) {
        let _ = writeln!(output, "# HELP {name} {help}");
        let _ = writeln!(output, "# TYPE {name} gauge");
        let mean = aggregate.mean;
        let _ = writeln!(output, "{name} {mean}");
        let _ = writeln!(output);
    }
}

/// Export a summary metric with quantiles
fn export_summary(
    output: &mut String,
    name: &str,
    help: &str,
    snapshot: &WindowSnapshot,
    metric_key: &str,
) {
    if let Some(aggregate) = snapshot.get(metric_key) {
        let _ = writeln!(output, "# HELP {name} {help}");
        let _ = writeln!(output, "# TYPE {name} summary");
        let p50 = aggregate.p50;
        let p90 = aggregate.p90;
        let p99 = aggregate.p99;
        let sum = aggregate.sum;
        let count = aggregate.count;
        let _ = writeln!(output, "{name}{{quantile=\"0.5\"}} {p50}");
        let _ = writeln!(output, "{name}{{quantile=\"0.9\"}} {p90}");
        let _ = writeln!(output, "{name}{{quantile=\"0.99\"}} {p99}");
        let _ = writeln!(output, "{name}_sum {sum}");
        let _ = writeln!(output, "{name}_count {count}");
        let _ = writeln!(output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn test_prometheus_export_format() {
        let mut snapshot = BTreeMap::new();
        snapshot.insert(
            "engram_consolidation_freshness_seconds",
            MetricAggregate {
                count: 100,
                sum: 14500.0,
                mean: 145.0,
                min: 120.0,
                max: 180.0,
                p50: 145.0,
                p90: 165.0,
                p99: 175.0,
            },
        );

        let metrics = AggregatedMetrics {
            schema_version: Some("1.0.0".to_string()),
            one_second: BTreeMap::new(),
            ten_seconds: BTreeMap::new(),
            one_minute: BTreeMap::new(),
            five_minutes: snapshot,
            spreading: None,
        };

        let output = to_prometheus_text(&metrics);

        assert!(output.contains("# HELP engram_consolidation_freshness_seconds"));
        assert!(output.contains("# TYPE engram_consolidation_freshness_seconds gauge"));
        assert!(output.contains("engram_consolidation_freshness_seconds 145"));
    }

    #[test]
    fn test_summary_export_with_quantiles() {
        let mut snapshot = BTreeMap::new();
        snapshot.insert(
            "engram_spreading_latency_hot_seconds",
            MetricAggregate {
                count: 1000,
                sum: 45.5,
                mean: 0.0455,
                min: 0.001,
                max: 0.150,
                p50: 0.040,
                p90: 0.080,
                p99: 0.120,
            },
        );

        let metrics = AggregatedMetrics {
            schema_version: Some("1.0.0".to_string()),
            one_second: BTreeMap::new(),
            ten_seconds: BTreeMap::new(),
            one_minute: BTreeMap::new(),
            five_minutes: snapshot,
            spreading: None,
        };

        let output = to_prometheus_text(&metrics);

        assert!(output.contains("# TYPE engram_spreading_latency_hot_seconds summary"));
        assert!(output.contains("engram_spreading_latency_hot_seconds{quantile=\"0.5\"} 0.04"));
        assert!(output.contains("engram_spreading_latency_hot_seconds{quantile=\"0.9\"} 0.08"));
        assert!(output.contains("engram_spreading_latency_hot_seconds{quantile=\"0.99\"} 0.12"));
        assert!(output.contains("engram_spreading_latency_hot_seconds_sum 45.5"));
        assert!(output.contains("engram_spreading_latency_hot_seconds_count 1000"));
    }

    #[test]
    fn test_counter_export() {
        let mut snapshot = BTreeMap::new();
        snapshot.insert(
            "engram_consolidation_runs_total",
            MetricAggregate {
                count: 50,
                sum: 250.0,
                mean: 5.0,
                min: 5.0,
                max: 5.0,
                p50: 5.0,
                p90: 5.0,
                p99: 5.0,
            },
        );

        let metrics = AggregatedMetrics {
            schema_version: Some("1.0.0".to_string()),
            one_second: BTreeMap::new(),
            ten_seconds: BTreeMap::new(),
            one_minute: BTreeMap::new(),
            five_minutes: snapshot,
            spreading: None,
        };

        let output = to_prometheus_text(&metrics);

        assert!(output.contains("# TYPE engram_consolidation_runs_total counter"));
        assert!(output.contains("engram_consolidation_runs_total 250"));
    }
}
