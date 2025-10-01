use statrs::distribution::{ContinuousCDF, Normal};
use std::cmp::Ordering;
use std::collections::HashMap;

const fn usize_to_f64(value: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    {
        value as f64
    }
}

fn ceil_to_usize(value: f64) -> usize {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        value.max(0.0).ceil() as usize
    }
}

fn percentile_index(fraction: f64, len: usize) -> usize {
    if len == 0 {
        return 0;
    }

    let capped_fraction = fraction.clamp(0.0, 1.0);
    let max_index = usize_to_f64(len.saturating_sub(1));

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        (capped_fraction * max_index).round() as usize
    }
}

fn standard_normal() -> Normal {
    Normal::new(0.0, 1.0).unwrap_or_else(|_| unreachable!("unit variance should be valid"))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().sum::<f64>() / usize_to_f64(values.len())
}

fn unbiased_variance(values: &[f64], mean_value: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let numerator = values.iter().map(|v| (v - mean_value).powi(2)).sum::<f64>();

    numerator / usize_to_f64(values.len() - 1)
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StatisticalBenchmarkFramework {
    power_calculator: PowerAnalysisCalculator,
    fdr_controller: BenjaminiHochbergController,
    bootstrap_sampler: BiasCorrectectedBootstrapSampler,
    effect_size_calculator: ComprehensiveEffectSizeCalculator,
    performance_database: HistoricalPerformanceDB,
    bayesian_analyzer: BayesianHypothesisTestEngine,
    time_series_analyzer: ARIMAPerformanceAnalyzer,
    distribution_comparator: KolmogorovSmirnovComparator,
}

impl StatisticalBenchmarkFramework {
    pub fn new() -> Self {
        Self {
            power_calculator: PowerAnalysisCalculator::new(),
            fdr_controller: BenjaminiHochbergController::new(),
            bootstrap_sampler: BiasCorrectectedBootstrapSampler::new(),
            effect_size_calculator: ComprehensiveEffectSizeCalculator::new(),
            performance_database: HistoricalPerformanceDB::new(),
            bayesian_analyzer: BayesianHypothesisTestEngine::new(),
            time_series_analyzer: ARIMAPerformanceAnalyzer::new(),
            distribution_comparator: KolmogorovSmirnovComparator::new(),
        }
    }

    pub fn detect_regression(
        &self,
        current_samples: &[f64],
        historical_samples: &[f64],
        metric_name: &str,
    ) -> RegressionAnalysis {
        // Power analysis to ensure sufficient samples
        let required_n = self.power_calculator.required_sample_size(
            0.05,  // 5% regression threshold
            0.001, // 0.1% Type I error rate
            0.005, // 0.5% Type II error rate (99.5% power)
        );

        if current_samples.len() < required_n {
            return RegressionAnalysis::InsufficientData {
                required: required_n,
            };
        }

        // Non-parametric test for distribution difference
        let mann_whitney_result = Self::mann_whitney_u_test(current_samples, historical_samples);

        // Bootstrap confidence intervals for effect size
        let effect_size_ci = self.bootstrap_sampler.bootstrap_effect_size(
            current_samples,
            historical_samples,
            10_000,
        );

        // Practical significance check
        let cohens_d = self
            .effect_size_calculator
            .cohens_d(current_samples, historical_samples);

        let recommendation = self.generate_recommendation(&mann_whitney_result, cohens_d);

        RegressionAnalysis::Detected {
            metric_name: metric_name.to_string(),
            statistical_significance: mann_whitney_result,
            practical_significance: cohens_d.abs() > 0.2, // Small effect size threshold
            effect_size_ci,
            recommendation,
        }
    }

    #[allow(clippy::many_single_char_names)]
    fn mann_whitney_u_test(x: &[f64], y: &[f64]) -> TestResult {
        let mut combined: Vec<(f64, bool)> = x
            .iter()
            .map(|&val| (val, true))
            .chain(y.iter().map(|&val| (val, false)))
            .collect();
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let mut ranks = vec![0.0; combined.len()];
        let mut i = 0;
        while i < combined.len() {
            let mut j = i;
            while j + 1 < combined.len() && (combined[j + 1].0 - combined[i].0).abs() < f64::EPSILON
            {
                j += 1;
            }

            let avg_rank = (usize_to_f64(i + 1) + usize_to_f64(j + 1)) * 0.5;
            for rank in &mut ranks[i..=j] {
                *rank = avg_rank;
            }

            i = j + 1;
        }

        let r1: f64 = combined
            .iter()
            .zip(ranks.iter())
            .filter(|((_, is_x), _)| *is_x)
            .map(|(_, rank)| *rank)
            .sum();

        let n1 = usize_to_f64(x.len());
        let n2 = usize_to_f64(y.len());
        let u1 = (n1 * (n1 + 1.0)).mul_add(-0.5, r1);
        let u2 = n1.mul_add(n2, -u1);
        let u = u1.min(u2);

        let mean_u = n1 * n2 * 0.5;
        let std_u = (n1 * n2 * (n1 + n2 + 1.0) / 12.0).sqrt();
        let z = (u - mean_u) / std_u.max(f64::EPSILON);

        let normal = standard_normal();
        let p_value = 2.0 * normal.cdf(z.abs());

        TestResult {
            statistic: u,
            p_value,
            significant: p_value < 0.001,
        }
    }

    fn generate_recommendation(&self, test_result: &TestResult, effect_size: f64) -> String {
        if !test_result.significant {
            return "No significant regression detected".to_string();
        }

        let benchmark_count = self.performance_database.total_entries();

        match effect_size.abs() {
            x if x < 0.2 => "Small regression detected - monitor but no immediate action required",
            x if x < 0.5 => "Medium regression detected - investigate root cause",
            x if x < 0.8 => "Large regression detected - immediate investigation required",
            _ if benchmark_count > 10 => {
                "Very large regression detected - critical performance issue"
            }
            _ => "Very large regression detected - gather more benchmark evidence",
        }
        .to_string()
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum RegressionAnalysis {
    InsufficientData {
        required: usize,
    },
    Detected {
        metric_name: String,
        statistical_significance: TestResult,
        practical_significance: bool,
        effect_size_ci: ConfidenceInterval,
        recommendation: String,
    },
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub significant: bool,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
}

#[derive(Debug, Clone)]
pub struct PowerAnalysisCalculator {
    minimum_power: f64,
}

impl PowerAnalysisCalculator {
    pub const fn new() -> Self {
        Self { minimum_power: 0.8 }
    }

    pub fn required_sample_size(&self, effect_size: f64, alpha: f64, beta: f64) -> usize {
        let target_beta = beta.max(1.0 - self.minimum_power);

        let normal = standard_normal();
        let z_alpha = normal.inverse_cdf(1.0 - alpha / 2.0);
        let z_beta = normal.inverse_cdf(1.0 - target_beta);

        let numerator = 2.0 * (z_alpha + z_beta).powi(2);
        let denominator = effect_size.powi(2);

        ceil_to_usize(numerator / denominator)
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BenjaminiHochbergController {
    max_fdr: f64,
}

impl BenjaminiHochbergController {
    pub const fn new() -> Self {
        Self { max_fdr: 0.1 }
    }

    #[allow(dead_code)]
    pub fn apply_correction(&self, p_values: &[f64], alpha: f64) -> Vec<bool> {
        let n = p_values.len();
        if n == 0 {
            return Vec::new();
        }

        let mut indexed: Vec<(usize, f64)> =
            p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let mut rejected = vec![false; n];
        let mut max_k = 0;
        let effective_alpha = alpha.min(self.max_fdr);
        let denominator = usize_to_f64(n);

        for (k, &(_idx, p)) in indexed.iter().enumerate() {
            let numerator = usize_to_f64(k + 1);
            let threshold = effective_alpha * numerator / denominator;
            if p <= threshold {
                max_k = k + 1;
            }
        }

        for &(idx, _) in indexed.iter().take(max_k) {
            rejected[idx] = true;
        }

        rejected
    }
}

#[derive(Debug, Clone)]
pub struct BiasCorrectectedBootstrapSampler {
    confidence_level: f64,
}

impl BiasCorrectectedBootstrapSampler {
    pub const fn new() -> Self {
        Self {
            confidence_level: 0.95,
        }
    }

    pub fn bootstrap_effect_size(
        &self,
        x: &[f64],
        y: &[f64],
        n_bootstrap: usize,
    ) -> ConfidenceInterval {
        use rand::prelude::*;
        let mut rng = thread_rng();

        let mut bootstrap_effects = Vec::with_capacity(n_bootstrap);

        for _ in 0..n_bootstrap {
            let x_sample: Vec<f64> = (0..x.len()).map(|_| x[rng.gen_range(0..x.len())]).collect();
            let y_sample: Vec<f64> = (0..y.len()).map(|_| y[rng.gen_range(0..y.len())]).collect();

            let effect = Self::calculate_cohens_d(&x_sample, &y_sample);
            bootstrap_effects.push(effect);
        }

        bootstrap_effects.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        // BCa (bias-corrected and accelerated) bootstrap
        let original_effect = Self::calculate_cohens_d(x, y);
        let z0 = Self::calculate_bias_correction(&bootstrap_effects, original_effect);
        let acceleration = self.calculate_acceleration(x, y);

        let alpha = 1.0 - self.confidence_level;
        let normal = standard_normal();
        let z_alpha_lower = normal.inverse_cdf(alpha / 2.0);
        let z_alpha_upper = normal.inverse_cdf(1.0 - alpha / 2.0);

        let alpha_lower = normal
            .cdf(z0 + (z0 + z_alpha_lower) / acceleration.mul_add(-(z0 + z_alpha_lower), 1.0));
        let alpha_upper = normal
            .cdf(z0 + (z0 + z_alpha_upper) / acceleration.mul_add(-(z0 + z_alpha_upper), 1.0));

        let lower_idx = percentile_index(alpha_lower, bootstrap_effects.len());
        let upper_idx = percentile_index(alpha_upper, bootstrap_effects.len());

        ConfidenceInterval {
            lower: bootstrap_effects[lower_idx],
            upper: bootstrap_effects[upper_idx.min(bootstrap_effects.len() - 1)],
            confidence_level: self.confidence_level,
        }
    }

    fn calculate_cohens_d(x: &[f64], y: &[f64]) -> f64 {
        let mean_x = mean(x);
        let mean_y = mean(y);

        let var_x = unbiased_variance(x, mean_x);
        let var_y = unbiased_variance(y, mean_y);

        let pooled_std = f64::midpoint(var_x, var_y).sqrt();

        (mean_x - mean_y) / pooled_std
    }

    fn calculate_bias_correction(bootstrap_effects: &[f64], original_effect: f64) -> f64 {
        let prop_less = bootstrap_effects
            .iter()
            .filter(|&&x| x < original_effect)
            .count();

        let fraction = if bootstrap_effects.is_empty() {
            0.5
        } else {
            usize_to_f64(prop_less) / usize_to_f64(bootstrap_effects.len())
        };

        standard_normal().inverse_cdf(fraction)
    }

    fn calculate_acceleration(&self, _x: &[f64], _y: &[f64]) -> f64 {
        1.0 - self.confidence_level
    }
}

#[derive(Debug, Clone)]
pub struct ComprehensiveEffectSizeCalculator {
    minimum_effect: f64,
}

impl ComprehensiveEffectSizeCalculator {
    pub const fn new() -> Self {
        Self {
            minimum_effect: 0.1,
        }
    }

    pub fn cohens_d(&self, x: &[f64], y: &[f64]) -> f64 {
        let mean_x = mean(x);
        let mean_y = mean(y);

        let var_x = unbiased_variance(x, mean_x);
        let var_y = unbiased_variance(y, mean_y);

        let pooled_std = f64::midpoint(var_x, var_y).sqrt();
        let effect = (mean_x - mean_y) / pooled_std.max(f64::EPSILON);

        if effect.abs() < self.minimum_effect {
            0.0
        } else {
            effect
        }
    }

    #[allow(dead_code)]
    pub fn hedges_g(&self, x: &[f64], y: &[f64]) -> f64 {
        let d = self.cohens_d(x, y);
        let n = usize_to_f64(x.len() + y.len());

        let correction = 1.0 - 3.0 / 4.0f64.mul_add(n, -9.0);
        d * correction
    }

    #[allow(dead_code)]
    pub fn glass_delta(x: &[f64], y: &[f64]) -> f64 {
        let mean_x = mean(x);
        let mean_y = mean(y);

        let var_y = unbiased_variance(y, mean_y);
        let std_y = var_y.sqrt();

        (mean_x - mean_y) / std_y.max(f64::EPSILON)
    }
}

#[derive(Debug, Clone)]
pub struct HistoricalPerformanceDB {
    data: HashMap<String, Vec<f64>>,
}

impl HistoricalPerformanceDB {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    #[allow(dead_code)]
    pub fn get_historical_metrics(&self, metric_name: &str) -> Vec<f64> {
        self.data.get(metric_name).cloned().unwrap_or_default()
    }

    #[allow(dead_code)]
    pub fn store_metrics(&mut self, metric_name: &str, values: Vec<f64>) {
        self.data.insert(metric_name.to_string(), values);
    }

    pub fn total_entries(&self) -> usize {
        self.data.values().map(Vec::len).sum()
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BayesianHypothesisTestEngine {
    scale_parameter: f64,
}

impl BayesianHypothesisTestEngine {
    pub const fn new() -> Self {
        Self {
            scale_parameter: 0.707,
        }
    }

    #[allow(dead_code)]
    pub fn bayes_factor(&self, x: &[f64], y: &[f64]) -> f64 {
        let t_stat = Self::calculate_t_statistic(x, y);
        let df = usize_to_f64(x.len() + y.len() - 2);

        let numerator = (1.0 + t_stat.powi(2) / df).powf(-(df + 1.0) / 2.0);
        let denominator =
            (1.0 + t_stat.powi(2) / (df * self.scale_parameter.powi(2))).powf(-(df + 1.0) / 2.0);

        numerator / denominator
    }

    fn calculate_t_statistic(x: &[f64], y: &[f64]) -> f64 {
        let mean_x = mean(x);
        let mean_y = mean(y);

        let var_x = unbiased_variance(x, mean_x);
        let var_y = unbiased_variance(y, mean_y);

        let weight_x = usize_to_f64(x.len().saturating_sub(1));
        let weight_y = usize_to_f64(y.len().saturating_sub(1));
        let denominator = usize_to_f64(x.len() + y.len() - 2);

        let pooled_var = weight_x.mul_add(var_x, weight_y * var_y) / denominator.max(f64::EPSILON);
        let se = (pooled_var * (1.0 / usize_to_f64(x.len()) + 1.0 / usize_to_f64(y.len()))).sqrt();

        (mean_x - mean_y) / se.max(f64::EPSILON)
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ARIMAPerformanceAnalyzer {
    detrend_window: usize,
}

impl ARIMAPerformanceAnalyzer {
    pub const fn new() -> Self {
        Self { detrend_window: 5 }
    }

    #[allow(dead_code)]
    pub fn analyze_trend(&self, time_series: &[f64]) -> TrendAnalysis {
        if time_series.is_empty() {
            return TrendAnalysis {
                slope: 0.0,
                intercept: 0.0,
                trend_direction: "stable".to_string(),
            };
        }

        let x: Vec<f64> = (0..time_series.len()).map(usize_to_f64).collect();

        let mean_x = mean(&x);
        let mean_y = mean(time_series);

        let detrended: Vec<f64> = time_series
            .windows(self.detrend_window.max(1))
            .map(mean)
            .collect();

        let effective_series = if detrended.is_empty() {
            time_series.to_vec()
        } else {
            detrended
        };

        let cov_xy: f64 = x
            .iter()
            .zip(effective_series.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / usize_to_f64(time_series.len());

        let var_x =
            x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / usize_to_f64(time_series.len());

        let slope = cov_xy / var_x.max(f64::EPSILON);
        let intercept = slope.mul_add(-mean_x, mean_y);

        TrendAnalysis {
            slope,
            intercept,
            trend_direction: if slope > 0.0 {
                "increasing"
            } else if slope < 0.0 {
                "decreasing"
            } else {
                "stable"
            }
            .to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TrendAnalysis {
    pub slope: f64,
    pub intercept: f64,
    pub trend_direction: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct KolmogorovSmirnovComparator {
    significance_level: f64,
}

impl KolmogorovSmirnovComparator {
    pub const fn new() -> Self {
        Self {
            significance_level: 0.001,
        }
    }

    #[allow(dead_code)]
    pub fn ks_test(&self, x: &[f64], y: &[f64]) -> TestResult {
        let mut x_sorted = x.to_vec();
        let mut y_sorted = y.to_vec();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let mut max_diff = 0.0;
        let n_x = usize_to_f64(x_sorted.len());
        let n_y = usize_to_f64(y_sorted.len());

        let mut i = 0;
        let mut j = 0;

        while i < x_sorted.len() || j < y_sorted.len() {
            let f_x = usize_to_f64(i) / n_x.max(f64::EPSILON);
            let f_y = usize_to_f64(j) / n_y.max(f64::EPSILON);

            let diff = (f_x - f_y).abs();
            if diff > max_diff {
                max_diff = diff;
            }

            if i < x_sorted.len() && (j >= y_sorted.len() || x_sorted[i] <= y_sorted[j]) {
                i += 1;
            } else {
                j += 1;
            }
        }

        // Calculate p-value using asymptotic approximation
        let effective_n = (n_x * n_y / (n_x + n_y)).sqrt();
        let lambda = max_diff * effective_n;

        // Kolmogorov distribution approximation
        let p_value = 2.0 * (-2.0 * lambda.powi(2)).exp();

        TestResult {
            statistic: max_diff,
            p_value,
            significant: p_value < self.significance_level,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StatisticalBenchmarkResults {
    task_results: HashMap<String, TaskBenchmarkResult>,
}

impl StatisticalBenchmarkResults {
    pub fn new() -> Self {
        Self {
            task_results: HashMap::new(),
        }
    }

    pub fn add_task_results(&mut self, task_name: &str, results: TaskBenchmarkResult) {
        self.task_results.insert(task_name.to_string(), results);
    }

    pub const fn task_results(&self) -> &HashMap<String, TaskBenchmarkResult> {
        &self.task_results
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TaskBenchmarkResult {
    pub mean_latency: f64,
    pub p95_latency: f64,
    pub p99_latency: f64,
    pub throughput: f64,
    pub samples: Vec<f64>,
}
