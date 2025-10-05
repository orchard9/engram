//! Prometheus metrics export format

use super::MetricsRegistry;
use std::fmt::Write;

/// Export all metrics in Prometheus format
#[must_use]
pub fn export_all(registry: &MetricsRegistry) -> String {
    let mut output = String::with_capacity(8192);

    // Add header
    writeln!(&mut output, "# HELP engram_info Engram system information").unwrap();
    writeln!(&mut output, "# TYPE engram_info gauge").unwrap();
    writeln!(
        &mut output,
        r#"engram_info{{version="0.1.0",arch="{}"}} 1"#,
        std::env::consts::ARCH
    )
    .unwrap();
    writeln!(&mut output).unwrap();

    // Export counters
    export_counters(&mut output, registry);

    // Export histograms
    export_histograms(&mut output, registry);

    // Export cognitive metrics
    export_cognitive(&mut output, registry);

    // Export hardware metrics
    export_hardware(&mut output, registry);

    // Export health status
    export_health(&mut output, registry);

    output
}

/// Export counter metrics
fn export_counters(output: &mut String, registry: &MetricsRegistry) {
    // Common counters
    let counters = [
        ("operations_total", "Total number of operations"),
        ("memories_created_total", "Total memories created"),
        ("episodes_stored_total", "Total episodes stored"),
        ("queries_executed_total", "Total queries executed"),
        ("errors_total", "Total errors encountered"),
        ("spreading_gpu_launch_total", "Total GPU spreading launches"),
        (
            "spreading_gpu_fallback_total",
            "Total GPU spreading fallbacks to CPU",
        ),
    ];

    for (name, help) in counters {
        let value = registry.counters.get(name);
        if value > 0 {
            writeln!(output, "# HELP engram_{name} {help}").unwrap();
            writeln!(output, "# TYPE engram_{name} counter").unwrap();
            writeln!(output, "engram_{name} {value}").unwrap();
            writeln!(output).unwrap();
        }
    }
}

/// Export histogram metrics
fn export_histograms(output: &mut String, registry: &MetricsRegistry) {
    // Common histograms
    let histograms = [
        ("query_duration_seconds", "Query duration in seconds"),
        (
            "consolidation_duration_seconds",
            "Consolidation duration in seconds",
        ),
        ("activation_spreading_depth", "Activation spreading depth"),
        (
            "pattern_completion_iterations",
            "Pattern completion iterations",
        ),
        (
            "gpu_transfer_latency_seconds",
            "GPU transfer latency in seconds",
        ),
    ];

    for (name, help) in histograms {
        let quantiles = registry.histograms.quantiles(name, &[0.5, 0.9, 0.99]);

        // Only export if we have data
        if quantiles.iter().any(|&v| v > 0.0) {
            writeln!(output, "# HELP engram_{name} {help}").unwrap();
            writeln!(output, "# TYPE engram_{name} summary").unwrap();

            writeln!(
                output,
                r#"engram_{}{{quantile="0.5"}} {}"#,
                name, quantiles[0]
            )
            .unwrap();
            writeln!(
                output,
                r#"engram_{}{{quantile="0.9"}} {}"#,
                name, quantiles[1]
            )
            .unwrap();
            writeln!(
                output,
                r#"engram_{}{{quantile="0.99"}} {}"#,
                name, quantiles[2]
            )
            .unwrap();
            writeln!(output).unwrap();
        }
    }
}

/// Export cognitive architecture metrics
fn export_cognitive(output: &mut String, registry: &MetricsRegistry) {
    // CLS balance
    let (hippo, neo) = registry.cognitive.get_cls_balance();
    writeln!(
        output,
        "# HELP engram_cls_hippocampal_weight Hippocampal contribution weight"
    )
    .unwrap();
    writeln!(output, "# TYPE engram_cls_hippocampal_weight gauge").unwrap();
    writeln!(output, "engram_cls_hippocampal_weight {hippo}").unwrap();
    writeln!(output).unwrap();

    writeln!(
        output,
        "# HELP engram_cls_neocortical_weight Neocortical contribution weight"
    )
    .unwrap();
    writeln!(output, "# TYPE engram_cls_neocortical_weight gauge").unwrap();
    writeln!(output, "engram_cls_neocortical_weight {neo}").unwrap();
    writeln!(output).unwrap();

    // Pattern completion stats
    let pc_stats = registry.cognitive.get_pattern_completion_stats();
    writeln!(
        output,
        "# HELP engram_pattern_completion_plausibility Pattern completion plausibility score"
    )
    .unwrap();
    writeln!(
        output,
        "# TYPE engram_pattern_completion_plausibility gauge"
    )
    .unwrap();
    writeln!(
        output,
        "engram_pattern_completion_plausibility {}",
        pc_stats.plausibility
    )
    .unwrap();
    writeln!(output).unwrap();

    writeln!(
        output,
        "# HELP engram_false_memory_rate Rate of false memory generation"
    )
    .unwrap();
    writeln!(output, "# TYPE engram_false_memory_rate gauge").unwrap();
    writeln!(
        output,
        "engram_false_memory_rate {}",
        pc_stats.false_memory_rate
    )
    .unwrap();
    writeln!(output).unwrap();

    // Activation depth
    let avg_depth = registry.cognitive.get_average_activation_depth();
    writeln!(
        output,
        "# HELP engram_activation_depth_average Average spreading activation depth"
    )
    .unwrap();
    writeln!(output, "# TYPE engram_activation_depth_average gauge").unwrap();
    writeln!(output, "engram_activation_depth_average {avg_depth}").unwrap();
    writeln!(output).unwrap();

    // Calibration stats
    let cal_stats = registry.cognitive.get_calibration_stats();
    writeln!(
        output,
        "# HELP engram_overconfidence_corrections_total Total overconfidence corrections"
    )
    .unwrap();
    writeln!(
        output,
        "# TYPE engram_overconfidence_corrections_total counter"
    )
    .unwrap();
    writeln!(
        output,
        "engram_overconfidence_corrections_total {}",
        cal_stats.overconfidence_corrections
    )
    .unwrap();
    writeln!(output).unwrap();
}

/// Export hardware performance metrics
fn export_hardware(output: &mut String, registry: &MetricsRegistry) {
    use super::hardware::CacheLevel;

    // SIMD utilization
    let simd_util = registry.hardware.simd_utilization();
    writeln!(
        output,
        "# HELP engram_simd_utilization_percent SIMD instruction utilization percentage"
    )
    .unwrap();
    writeln!(output, "# TYPE engram_simd_utilization_percent gauge").unwrap();
    writeln!(output, "engram_simd_utilization_percent {simd_util}").unwrap();
    writeln!(output).unwrap();

    // Cache hit ratios
    for level in [CacheLevel::L1, CacheLevel::L2, CacheLevel::L3] {
        let hit_ratio = registry.hardware.cache_hit_ratio(level);
        let level_str = format!("{level:?}").to_lowercase();

        writeln!(
            output,
            "# HELP engram_cache_{}_hit_ratio Cache {} hit ratio",
            level_str,
            level_str.to_uppercase()
        )
        .unwrap();
        writeln!(output, "# TYPE engram_cache_{level_str}_hit_ratio gauge").unwrap();
        writeln!(output, "engram_cache_{level_str}_hit_ratio {hit_ratio}").unwrap();
        writeln!(output).unwrap();
    }

    // Branch prediction accuracy
    let branch_accuracy = registry.hardware.branch_prediction_accuracy();
    writeln!(
        output,
        "# HELP engram_branch_prediction_accuracy Branch prediction accuracy"
    )
    .unwrap();
    writeln!(output, "# TYPE engram_branch_prediction_accuracy gauge").unwrap();
    writeln!(
        output,
        "engram_branch_prediction_accuracy {branch_accuracy}"
    )
    .unwrap();
    writeln!(output).unwrap();
}

/// Export health status
fn export_health(output: &mut String, registry: &MetricsRegistry) {
    let status = registry.health.current_status();

    let health_value = match status {
        super::health::HealthStatus::Healthy => 1.0,
        super::health::HealthStatus::Degraded => 0.5,
        super::health::HealthStatus::Unhealthy => 0.0,
    };

    writeln!(
        output,
        "# HELP engram_health_status System health status (1=healthy, 0.5=degraded, 0=unhealthy)"
    )
    .unwrap();
    writeln!(output, "# TYPE engram_health_status gauge").unwrap();
    writeln!(output, "engram_health_status {health_value}").unwrap();
    writeln!(output).unwrap();

    // Export individual health checks
    let report = registry.health.health_report();
    for check in report.checks {
        let check_value = match check.status {
            super::health::HealthStatus::Healthy => 1.0,
            super::health::HealthStatus::Degraded => 0.5,
            super::health::HealthStatus::Unhealthy => 0.0,
        };

        writeln!(
            output,
            r#"engram_health_check{{name="{}"}} {}"#,
            check.name, check_value
        )
        .unwrap();
    }
    writeln!(output).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prometheus_export() {
        let registry = MetricsRegistry::new();

        // Add some test data
        registry.increment_counter("operations_total", 100);
        registry.observe_histogram("query_duration_seconds", 0.05);

        // Export metrics
        let output = export_all(&registry);

        // Check output format
        assert!(output.contains("# HELP"));
        assert!(output.contains("# TYPE"));
        assert!(output.contains("engram_info"));
    }

    #[test]
    fn test_health_export() {
        let registry = MetricsRegistry::new();
        registry.health.check_all();

        let mut output = String::new();
        export_health(&mut output, &registry);

        assert!(output.contains("engram_health_status"));
        assert!(output.contains("engram_health_check"));
    }
}
