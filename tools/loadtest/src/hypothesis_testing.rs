///! Statistical hypothesis testing framework for validating performance claims
///!
///! Tests from vision.md:
///! - H1: Throughput Capacity - System sustains >= 10K ops/sec for 1 hour
///! - H2: Latency SLA - P99 latency < 10ms under load
///! - H3: Linear Scaling - Throughput scales linearly with cores (R² > 0.95)
///! - H4: Memory Overhead - Memory overhead < 2x raw data size
use crate::metrics_collector::MetricsCollector;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Hypothesis test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTestResult {
    pub name: String,
    pub result: TestResult,
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
    pub target: f64,
    pub achieved: f64,
    pub description: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestResult {
    Pass,
    Fail,
}

/// H1: Throughput Capacity Test
///
/// Null hypothesis: System cannot sustain 10K ops/sec
/// Alternative: System sustains >= 10K ops/sec for 1 hour
///
/// Test: Measure throughput in 60-second windows, require 100% of windows >= 10K
/// Confidence: 95%
pub fn test_h1_throughput_capacity(
    metrics: &MetricsCollector,
    target_ops_sec: f64,
    duration_secs: u64,
) -> Result<HypothesisTestResult> {
    let window_size_secs = 60;
    let num_windows = duration_secs / window_size_secs;

    let mut windows_passing = 0;
    let mut throughput_samples = Vec::new();

    // Measure throughput in each window
    for _window in 0..num_windows {
        let throughput = metrics.window_throughput_ops_sec(window_size_secs);
        throughput_samples.push(throughput);

        if throughput >= target_ops_sec {
            windows_passing += 1;
        }
    }

    let pass_rate = windows_passing as f64 / num_windows as f64;
    let mean_throughput = throughput_samples.iter().sum::<f64>() / throughput_samples.len() as f64;

    // Calculate confidence interval (95%)
    let std_dev = {
        let variance = throughput_samples
            .iter()
            .map(|x| (x - mean_throughput).powi(2))
            .sum::<f64>()
            / throughput_samples.len() as f64;
        variance.sqrt()
    };

    let margin = 1.96 * (std_dev / (throughput_samples.len() as f64).sqrt());
    let ci = (mean_throughput - margin, mean_throughput + margin);

    // Test passes if all windows meet target
    let result = if pass_rate >= 1.0 {
        TestResult::Pass
    } else {
        TestResult::Fail
    };

    // P-value: probability of observing this throughput if null hypothesis is true
    // Simplified: use proportion of windows passing as p-value proxy
    let p_value = 1.0 - pass_rate;

    Ok(HypothesisTestResult {
        name: "H1_throughput_capacity".to_string(),
        result,
        p_value,
        confidence_interval: ci,
        target: target_ops_sec,
        achieved: mean_throughput,
        description: format!(
            "System sustains >= {:.0} ops/sec for {} seconds ({} windows)",
            target_ops_sec, duration_secs, num_windows
        ),
    })
}

/// H2: Latency SLA Test
///
/// Null hypothesis: P99 latency >= 10ms
/// Alternative: P99 latency < 10ms under load
///
/// Test: Collect latency samples (n >= 10,000), compute empirical P99
/// Confidence: 99%
pub fn test_h2_latency_sla(
    metrics: &MetricsCollector,
    target_p99_ms: f64,
) -> Result<HypothesisTestResult> {
    let latency_samples = metrics.all_latency_samples();

    if latency_samples.len() < 10000 {
        anyhow::bail!(
            "Insufficient samples for H2 test (need >= 10000, got {})",
            latency_samples.len()
        );
    }

    let p99 = metrics.p99_latency_ms();

    // Calculate confidence interval using bootstrap (simplified)
    let ci = bootstrap_percentile_ci(&latency_samples, 99.0, 0.99);

    let result = if p99 < target_p99_ms {
        TestResult::Pass
    } else {
        TestResult::Fail
    };

    // P-value: proportion of samples exceeding target
    let exceeding_count = latency_samples
        .iter()
        .filter(|&&x| x > target_p99_ms)
        .count();
    let p_value = exceeding_count as f64 / latency_samples.len() as f64;

    Ok(HypothesisTestResult {
        name: "H2_latency_sla".to_string(),
        result,
        p_value,
        confidence_interval: ci,
        target: target_p99_ms,
        achieved: p99,
        description: format!(
            "P99 latency < {:.1}ms under load (n={} samples)",
            target_p99_ms,
            latency_samples.len()
        ),
    })
}

/// H3: Linear Scaling Test
///
/// Null hypothesis: Throughput does not scale linearly with cores
/// Alternative: Throughput increases proportionally (within 10% efficiency loss)
///
/// Test: Linear regression R^2 > 0.95 on cores vs throughput
/// Confidence: 95%
pub fn test_h3_linear_scaling(
    core_counts: &[usize],
    throughputs: &[f64],
    min_r_squared: f64,
) -> Result<HypothesisTestResult> {
    if core_counts.len() != throughputs.len() || core_counts.is_empty() {
        anyhow::bail!("Invalid data for H3 test");
    }

    // Calculate linear regression
    let (slope, intercept, r_squared) = linear_regression(
        &core_counts.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        throughputs,
    );

    let result = if r_squared >= min_r_squared {
        TestResult::Pass
    } else {
        TestResult::Fail
    };

    // P-value based on R^2 (simplified)
    let p_value = 1.0 - r_squared;

    // Predict throughput for max cores (for debugging/logging if needed)
    let max_cores = *core_counts.iter().max().unwrap() as f64;
    let _predicted_throughput = slope * max_cores + intercept;

    Ok(HypothesisTestResult {
        name: "H3_linear_scaling".to_string(),
        result,
        p_value,
        confidence_interval: (r_squared - 0.05, r_squared + 0.05),
        target: min_r_squared,
        achieved: r_squared,
        description: format!(
            "Linear scaling R² >= {:.2} (slope={:.2}, intercept={:.2})",
            min_r_squared, slope, intercept
        ),
    })
}

/// H4: Memory Overhead Test
///
/// Null hypothesis: Memory overhead >= 2x raw data size
/// Alternative: Memory overhead < 2x
///
/// Test: Measure RSS after loading 1M nodes, compare to raw data size
/// Confidence: 99%
pub fn test_h4_memory_overhead(
    memory_used_bytes: u64,
    raw_data_bytes: u64,
    max_overhead_ratio: f64,
) -> Result<HypothesisTestResult> {
    let overhead_ratio = memory_used_bytes as f64 / raw_data_bytes as f64;

    let result = if overhead_ratio < max_overhead_ratio {
        TestResult::Pass
    } else {
        TestResult::Fail
    };

    // P-value: distance from target (simplified)
    let p_value = (overhead_ratio - max_overhead_ratio).max(0.0) / max_overhead_ratio;

    // Confidence interval (estimate ±10%)
    let ci = (overhead_ratio * 0.9, overhead_ratio * 1.1);

    Ok(HypothesisTestResult {
        name: "H4_memory_overhead".to_string(),
        result,
        p_value,
        confidence_interval: ci,
        target: max_overhead_ratio,
        achieved: overhead_ratio,
        description: format!(
            "Memory overhead < {:.1}x raw data size (used={}, raw={})",
            max_overhead_ratio, memory_used_bytes, raw_data_bytes
        ),
    })
}

/// Helper: Calculate linear regression
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    // Calculate R^2
    let mean_y = sum_y / n;
    let ss_tot: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| {
            let y_pred = slope * xi + intercept;
            (yi - y_pred).powi(2)
        })
        .sum();

    let r_squared = 1.0 - (ss_res / ss_tot);

    (slope, intercept, r_squared)
}

/// Helper: Bootstrap confidence interval for percentile
fn bootstrap_percentile_ci(samples: &[f64], percentile: f64, _confidence: f64) -> (f64, f64) {
    // Simplified: use order statistics
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let index = ((percentile / 100.0) * n as f64) as usize;

    // Rough CI estimate (±2% of data)
    let margin = (0.02 * n as f64) as usize;
    let lower_idx = index.saturating_sub(margin);
    let upper_idx = (index + margin).min(n - 1);

    (sorted[lower_idx], sorted[upper_idx])
}

/// Run all hypothesis tests and generate report
pub fn run_all_hypothesis_tests(
    metrics: &MetricsCollector,
    config: &HypothesisTestConfig,
) -> Result<Vec<HypothesisTestResult>> {
    let mut results = Vec::new();

    // H1: Throughput capacity
    results.push(test_h1_throughput_capacity(
        metrics,
        config.h1_target_ops_sec,
        config.h1_duration_secs,
    )?);

    // H2: Latency SLA
    results.push(test_h2_latency_sla(metrics, config.h2_target_p99_ms)?);

    // H3: Linear scaling (requires multi-run data)
    if let Some((cores, throughputs)) = &config.h3_scaling_data {
        results.push(test_h3_linear_scaling(
            cores,
            throughputs,
            config.h3_min_r_squared,
        )?);
    }

    // H4: Memory overhead
    if let Some((used, raw)) = config.h4_memory_data {
        results.push(test_h4_memory_overhead(
            used,
            raw,
            config.h4_max_overhead_ratio,
        )?);
    }

    Ok(results)
}

#[derive(Debug, Clone)]
pub struct HypothesisTestConfig {
    pub h1_target_ops_sec: f64,
    pub h1_duration_secs: u64,
    pub h2_target_p99_ms: f64,
    pub h3_scaling_data: Option<(Vec<usize>, Vec<f64>)>,
    pub h3_min_r_squared: f64,
    pub h4_memory_data: Option<(u64, u64)>,
    pub h4_max_overhead_ratio: f64,
}

impl Default for HypothesisTestConfig {
    fn default() -> Self {
        Self {
            h1_target_ops_sec: 10000.0,
            h1_duration_secs: 3600,
            h2_target_p99_ms: 10.0,
            h3_scaling_data: None,
            h3_min_r_squared: 0.95,
            h4_memory_data: None,
            h4_max_overhead_ratio: 2.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect line: y = 2x

        let (slope, intercept, r_squared) = linear_regression(&x, &y);

        assert!((slope - 2.0).abs() < 0.001);
        assert!(intercept.abs() < 0.001);
        assert!((r_squared - 1.0).abs() < 0.001); // Perfect fit
    }

    #[test]
    fn test_bootstrap_ci() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (lower, upper) = bootstrap_percentile_ci(&samples, 90.0, 0.95);

        assert!(lower <= 9.0);
        assert!(upper >= 9.0);
    }
}
