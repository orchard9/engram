use std::collections::HashMap;
use statrs::distribution::{Normal, ContinuousCDF, InverseCDF};
use statrs::statistics::{Statistics, OrderStatistics, Data};

#[derive(Debug, Clone)]
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
            0.05,    // 5% regression threshold
            0.001,   // 0.1% Type I error rate
            0.005,   // 0.5% Type II error rate (99.5% power)
        );
        
        if current_samples.len() < required_n {
            return RegressionAnalysis::InsufficientData { required: required_n };
        }
        
        // Non-parametric test for distribution difference
        let mann_whitney_result = self.mann_whitney_u_test(
            current_samples, 
            historical_samples
        );
        
        // Bootstrap confidence intervals for effect size
        let effect_size_ci = self.bootstrap_sampler.bootstrap_effect_size(
            current_samples,
            historical_samples,
            10_000,
        );
        
        // Practical significance check
        let cohens_d = self.effect_size_calculator.cohens_d(
            current_samples,
            historical_samples,
        );
        
        RegressionAnalysis::Detected {
            metric_name: metric_name.to_string(),
            statistical_significance: mann_whitney_result,
            practical_significance: cohens_d.abs() > 0.2, // Small effect size threshold
            effect_size_ci,
            recommendation: self.generate_recommendation(mann_whitney_result, cohens_d),
        }
    }

    fn mann_whitney_u_test(&self, x: &[f64], y: &[f64]) -> TestResult {
        // Combine and rank all observations
        let mut combined: Vec<(f64, bool)> = Vec::new();
        for &val in x {
            combined.push((val, true));
        }
        for &val in y {
            combined.push((val, false));
        }
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Assign ranks
        let mut ranks = vec![0.0; combined.len()];
        let mut i = 0;
        while i < combined.len() {
            let j = (i..combined.len())
                .take_while(|&k| (combined[k].0 - combined[i].0).abs() < 1e-10)
                .last()
                .unwrap();
            let avg_rank = ((i + 1)..=(j + 1)).map(|r| r as f64).sum::<f64>() / ((j - i + 1) as f64);
            for k in i..=j {
                ranks[k] = avg_rank;
            }
            i = j + 1;
        }
        
        // Calculate U statistic
        let r1: f64 = combined.iter().zip(ranks.iter())
            .filter(|((_, is_x), _)| *is_x)
            .map(|(_, rank)| rank)
            .sum();
        
        let n1 = x.len() as f64;
        let n2 = y.len() as f64;
        let u1 = r1 - n1 * (n1 + 1.0) / 2.0;
        let u2 = n1 * n2 - u1;
        let u = u1.min(u2);
        
        // Normal approximation for large samples
        let mean_u = n1 * n2 / 2.0;
        let std_u = ((n1 * n2 * (n1 + n2 + 1.0)) / 12.0).sqrt();
        let z = (u - mean_u) / std_u;
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * normal.cdf(z.abs());
        
        TestResult {
            statistic: u,
            p_value,
            significant: p_value < 0.001,
        }
    }

    fn generate_recommendation(&self, test_result: TestResult, effect_size: f64) -> String {
        if !test_result.significant {
            return "No significant regression detected".to_string();
        }
        
        match effect_size.abs() {
            x if x < 0.2 => "Small regression detected - monitor but no immediate action required",
            x if x < 0.5 => "Medium regression detected - investigate root cause",
            x if x < 0.8 => "Large regression detected - immediate investigation required",
            _ => "Very large regression detected - critical performance issue",
        }.to_string()
    }
}

#[derive(Debug, Clone)]
pub enum RegressionAnalysis {
    InsufficientData { required: usize },
    Detected {
        metric_name: String,
        statistical_significance: TestResult,
        practical_significance: bool,
        effect_size_ci: ConfidenceInterval,
        recommendation: String,
    },
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub significant: bool,
}

#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
}

#[derive(Debug, Clone)]
pub struct PowerAnalysisCalculator;

impl PowerAnalysisCalculator {
    pub fn new() -> Self {
        Self
    }

    pub fn required_sample_size(&self, effect_size: f64, alpha: f64, beta: f64) -> usize {
        // Using Cohen's formulation for two-sample t-test approximation
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_alpha = normal.inverse_cdf(1.0 - alpha / 2.0);
        let z_beta = normal.inverse_cdf(1.0 - beta);
        
        let numerator = 2.0 * (z_alpha + z_beta).powi(2);
        let denominator = effect_size.powi(2);
        
        (numerator / denominator).ceil() as usize
    }
}

#[derive(Debug, Clone)]
pub struct BenjaminiHochbergController;

impl BenjaminiHochbergController {
    pub fn new() -> Self {
        Self
    }

    pub fn apply_correction(&self, p_values: &[f64], alpha: f64) -> Vec<bool> {
        let n = p_values.len();
        let mut indexed: Vec<(usize, f64)> = p_values.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let mut rejected = vec![false; n];
        let mut max_k = 0;
        
        for (k, &(original_idx, p)) in indexed.iter().enumerate() {
            let threshold = alpha * ((k + 1) as f64) / (n as f64);
            if p <= threshold {
                max_k = k + 1;
            }
        }
        
        for k in 0..max_k {
            rejected[indexed[k].0] = true;
        }
        
        rejected
    }
}

#[derive(Debug, Clone)]
pub struct BiasCorrectectedBootstrapSampler;

impl BiasCorrectectedBootstrapSampler {
    pub fn new() -> Self {
        Self
    }

    pub fn bootstrap_effect_size(
        &self,
        x: &[f64],
        y: &[f64],
        n_bootstrap: usize,
    ) -> ConfidenceInterval {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        let mut bootstrap_effects = Vec::new();
        
        for _ in 0..n_bootstrap {
            let x_sample: Vec<f64> = (0..x.len())
                .map(|_| x[rng.gen_range(0..x.len())])
                .collect();
            let y_sample: Vec<f64> = (0..y.len())
                .map(|_| y[rng.gen_range(0..y.len())])
                .collect();
            
            let effect = self.calculate_cohens_d(&x_sample, &y_sample);
            bootstrap_effects.push(effect);
        }
        
        bootstrap_effects.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // BCa (bias-corrected and accelerated) bootstrap
        let original_effect = self.calculate_cohens_d(x, y);
        let z0 = self.calculate_bias_correction(&bootstrap_effects, original_effect);
        let acceleration = self.calculate_acceleration(x, y);
        
        let alpha = 0.05;
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_alpha_lower = normal.inverse_cdf(alpha / 2.0);
        let z_alpha_upper = normal.inverse_cdf(1.0 - alpha / 2.0);
        
        let alpha_lower = normal.cdf(z0 + (z0 + z_alpha_lower) / (1.0 - acceleration * (z0 + z_alpha_lower)));
        let alpha_upper = normal.cdf(z0 + (z0 + z_alpha_upper) / (1.0 - acceleration * (z0 + z_alpha_upper)));
        
        let lower_idx = (alpha_lower * n_bootstrap as f64) as usize;
        let upper_idx = (alpha_upper * n_bootstrap as f64) as usize;
        
        ConfidenceInterval {
            lower: bootstrap_effects[lower_idx],
            upper: bootstrap_effects[upper_idx.min(bootstrap_effects.len() - 1)],
            confidence_level: 0.95,
        }
    }

    fn calculate_cohens_d(&self, x: &[f64], y: &[f64]) -> f64 {
        let mean_x: f64 = x.iter().sum::<f64>() / x.len() as f64;
        let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;
        
        let var_x: f64 = x.iter().map(|v| (v - mean_x).powi(2)).sum::<f64>() / (x.len() - 1) as f64;
        let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / (y.len() - 1) as f64;
        
        let pooled_std = ((var_x + var_y) / 2.0).sqrt();
        
        (mean_x - mean_y) / pooled_std
    }

    fn calculate_bias_correction(&self, bootstrap_effects: &[f64], original_effect: f64) -> f64 {
        let prop_less = bootstrap_effects.iter()
            .filter(|&&x| x < original_effect)
            .count() as f64 / bootstrap_effects.len() as f64;
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.inverse_cdf(prop_less)
    }

    fn calculate_acceleration(&self, _x: &[f64], _y: &[f64]) -> f64 {
        // Simplified acceleration calculation
        0.0
    }
}

#[derive(Debug, Clone)]
pub struct ComprehensiveEffectSizeCalculator;

impl ComprehensiveEffectSizeCalculator {
    pub fn new() -> Self {
        Self
    }

    pub fn cohens_d(&self, x: &[f64], y: &[f64]) -> f64 {
        let mean_x: f64 = x.iter().sum::<f64>() / x.len() as f64;
        let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;
        
        let var_x: f64 = x.iter().map(|v| (v - mean_x).powi(2)).sum::<f64>() / (x.len() - 1) as f64;
        let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / (y.len() - 1) as f64;
        
        let pooled_std = ((var_x + var_y) / 2.0).sqrt();
        
        (mean_x - mean_y) / pooled_std
    }

    pub fn hedges_g(&self, x: &[f64], y: &[f64]) -> f64 {
        let d = self.cohens_d(x, y);
        let n = (x.len() + y.len()) as f64;
        
        // Hedges' correction factor
        let correction = 1.0 - 3.0 / (4.0 * n - 9.0);
        
        d * correction
    }

    pub fn glass_delta(&self, x: &[f64], y: &[f64]) -> f64 {
        let mean_x: f64 = x.iter().sum::<f64>() / x.len() as f64;
        let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;
        
        // Use control group (y) standard deviation
        let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / (y.len() - 1) as f64;
        let std_y = var_y.sqrt();
        
        (mean_x - mean_y) / std_y
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

    pub fn get_historical_metrics(&self, metric_name: &str) -> Vec<f64> {
        self.data.get(metric_name).cloned().unwrap_or_default()
    }

    pub fn store_metrics(&mut self, metric_name: &str, values: Vec<f64>) {
        self.data.insert(metric_name.to_string(), values);
    }
}

#[derive(Debug, Clone)]
pub struct BayesianHypothesisTestEngine;

impl BayesianHypothesisTestEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn bayes_factor(&self, x: &[f64], y: &[f64]) -> f64 {
        // Simplified Bayes factor calculation
        // In practice, would use more sophisticated methods
        let t_stat = self.calculate_t_statistic(x, y);
        let df = (x.len() + y.len() - 2) as f64;
        
        // JZS Bayes factor approximation
        let r = 0.707; // scale parameter
        let bf = ((1.0 + t_stat.powi(2) / df).powf(-(df + 1.0) / 2.0)) /
                 ((1.0 + t_stat.powi(2) / (df * r.powi(2))).powf(-(df + 1.0) / 2.0));
        
        bf
    }

    fn calculate_t_statistic(&self, x: &[f64], y: &[f64]) -> f64 {
        let mean_x: f64 = x.iter().sum::<f64>() / x.len() as f64;
        let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;
        
        let var_x: f64 = x.iter().map(|v| (v - mean_x).powi(2)).sum::<f64>() / (x.len() - 1) as f64;
        let var_y: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / (y.len() - 1) as f64;
        
        let pooled_var = ((x.len() - 1) as f64 * var_x + (y.len() - 1) as f64 * var_y) /
                          ((x.len() + y.len() - 2) as f64);
        
        let se = (pooled_var * (1.0 / x.len() as f64 + 1.0 / y.len() as f64)).sqrt();
        
        (mean_x - mean_y) / se
    }
}

#[derive(Debug, Clone)]
pub struct ARIMAPerformanceAnalyzer;

impl ARIMAPerformanceAnalyzer {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze_trend(&self, time_series: &[f64]) -> TrendAnalysis {
        // Simplified trend analysis
        let n = time_series.len() as f64;
        let x: Vec<f64> = (0..time_series.len()).map(|i| i as f64).collect();
        
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = time_series.iter().sum::<f64>() / n;
        
        let cov_xy: f64 = x.iter().zip(time_series.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>() / n;
        
        let var_x: f64 = x.iter()
            .map(|xi| (xi - mean_x).powi(2))
            .sum::<f64>() / n;
        
        let slope = cov_xy / var_x;
        let intercept = mean_y - slope * mean_x;
        
        TrendAnalysis {
            slope,
            intercept,
            trend_direction: if slope > 0.0 { "increasing" } 
                           else if slope < 0.0 { "decreasing" } 
                           else { "stable" }.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub slope: f64,
    pub intercept: f64,
    pub trend_direction: String,
}

#[derive(Debug, Clone)]
pub struct KolmogorovSmirnovComparator;

impl KolmogorovSmirnovComparator {
    pub fn new() -> Self {
        Self
    }

    pub fn ks_test(&self, x: &[f64], y: &[f64]) -> TestResult {
        let mut x_sorted = x.to_vec();
        let mut y_sorted = y.to_vec();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut max_diff = 0.0;
        let n_x = x_sorted.len() as f64;
        let n_y = y_sorted.len() as f64;
        
        let mut i = 0;
        let mut j = 0;
        
        while i < x_sorted.len() || j < y_sorted.len() {
            let f_x = (i as f64) / n_x;
            let f_y = (j as f64) / n_y;
            
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
            significant: p_value < 0.001,
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

    pub fn task_results(&self) -> &HashMap<String, TaskBenchmarkResult> {
        &self.task_results
    }
}

#[derive(Debug, Clone)]
pub struct TaskBenchmarkResult {
    pub mean_latency: f64,
    pub p95_latency: f64,
    pub p99_latency: f64,
    pub throughput: f64,
    pub samples: Vec<f64>,
}