use statrs::distribution::{ContinuousCDF, StudentsT};

/// Result of Welchs t-test
#[derive(Debug, Clone, Copy)]
pub struct WelchTTestResult {
    pub t_stat: f32,
    pub degrees_of_freedom: f32,
    pub p_value: f32,
}

/// Compute arithmetic mean
pub fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// Sample variance (n-1 denominator)
pub fn variance(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = mean(values);
    let sum_sq = values
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f32>();
    sum_sq / (values.len() - 1) as f32
}

/// Sample standard deviation
#[allow(dead_code)]
pub fn standard_deviation(values: &[f32]) -> f32 {
    variance(values).sqrt()
}

/// Pearson correlation coefficient between paired samples
pub fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let mean_x = mean(x) as f64;
    let mean_y = mean(y) as f64;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = *xi as f64 - mean_x;
        let dy = *yi as f64 - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    if sum_sq_x == 0.0 || sum_sq_y == 0.0 {
        return 0.0;
    }

    (numerator / (sum_sq_x.sqrt() * sum_sq_y.sqrt())).clamp(-1.0, 1.0) as f32
}

/// Linear regression slope (beta) for y = alpha + beta * x
pub fn linear_regression_slope(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().map(|v| *v as f64).sum();
    let sum_y: f64 = y.iter().map(|v| *v as f64).sum();
    let sum_xy: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let sum_x2: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();

    let denominator = n * sum_x2 - sum_x.powi(2);
    if denominator.abs() < f64::EPSILON {
        return 0.0;
    }

    ((n * sum_xy - sum_x * sum_y) / denominator) as f32
}

/// Compute Cohen's d effect size from raw samples
pub fn cohens_d(sample_a: &[f32], sample_b: &[f32]) -> f32 {
    if sample_a.len() < 2 || sample_b.len() < 2 {
        return 0.0;
    }

    let mean_a = mean(sample_a);
    let mean_b = mean(sample_b);
    let var_a = variance(sample_a);
    let var_b = variance(sample_b);

    let pooled = ((sample_a.len() - 1) as f32 * var_a + (sample_b.len() - 1) as f32 * var_b)
        / ((sample_a.len() + sample_b.len() - 2) as f32);

    if pooled <= 0.0 {
        return 0.0;
    }

    (mean_a - mean_b) / pooled.sqrt()
}

/// Welch's t-test for independent samples
pub fn welch_t_test(sample_a: &[f32], sample_b: &[f32]) -> WelchTTestResult {
    if sample_a.len() < 2 || sample_b.len() < 2 {
        return WelchTTestResult {
            t_stat: 0.0,
            degrees_of_freedom: 0.0,
            p_value: 1.0,
        };
    }

    let mean_a = mean(sample_a) as f64;
    let mean_b = mean(sample_b) as f64;
    let var_a = variance(sample_a) as f64;
    let var_b = variance(sample_b) as f64;

    let n_a = sample_a.len() as f64;
    let n_b = sample_b.len() as f64;

    let standard_error = (var_a / n_a + var_b / n_b).sqrt();
    if standard_error <= f64::EPSILON {
        return WelchTTestResult {
            t_stat: 0.0,
            degrees_of_freedom: 0.0,
            p_value: 1.0,
        };
    }

    let t_stat = (mean_a - mean_b) / standard_error;
    let numerator = (var_a / n_a + var_b / n_b).powi(2);
    let denominator =
        (var_a.powi(2) / (n_a * n_a * (n_a - 1.0))) + (var_b.powi(2) / (n_b * n_b * (n_b - 1.0)));
    let df = if denominator <= f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    };

    let distribution = StudentsT::new(0.0, 1.0, df).expect("valid t distribution");
    let cdf = distribution.cdf(t_stat.abs());
    let p_value = (1.0 - cdf) * 2.0;

    WelchTTestResult {
        t_stat: t_stat as f32,
        degrees_of_freedom: df as f32,
        p_value: p_value as f32,
    }
}

/// Eta-squared (η²) effect size for one-way ANOVA designs
pub fn eta_squared(groups: &[&[f32]]) -> f32 {
    if groups.is_empty() {
        return 0.0;
    }

    let total_count: usize = groups.iter().map(|g| g.len()).sum();
    if total_count == 0 {
        return 0.0;
    }

    let grand_mean = groups
        .iter()
        .flat_map(|group| group.iter())
        .copied()
        .sum::<f32>()
        / total_count as f32;

    let mut ss_between = 0.0_f32;
    let mut ss_within = 0.0_f32;

    for group in groups {
        if group.is_empty() {
            continue;
        }
        let group_mean = mean(group);
        let diff = group_mean - grand_mean;
        ss_between += diff * diff * group.len() as f32;
        ss_within += group
            .iter()
            .map(|value| {
                let delta = *value - group_mean;
                delta * delta
            })
            .sum::<f32>();
    }

    let ss_total = ss_between + ss_within;
    if ss_total <= f32::EPSILON {
        0.0
    } else {
        ss_between / ss_total
    }
}
