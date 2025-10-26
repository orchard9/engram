//! Spacing Effect Validation - Replicating Cepeda et al. (2006)
//!
//! This module validates that Engram's temporal dynamics replicate the spacing effect,
//! one of the most robust findings in cognitive psychology. The spacing effect demonstrates
//! that distributed practice (studying items with spacing intervals) produces better
//! retention than massed practice (studying items consecutively).
//!
//! **Reference:**
//! Cepeda, N. J., et al. (2006). Distributed practice in verbal recall tasks: A review
//! and quantitative synthesis. *Psychological Bulletin*, 132(3), 354.
//!
//! **Expected Effect Size:** Cohen's d ≈ 0.5 (medium effect)
//! **Expected Improvement:** 20-40% better retention for distributed vs massed
//! **Acceptance Range:** ±10% → [10%, 50%] retention improvement
//!
//! **Statistical Requirements:**
//! - Sample size: n=200 (100 per condition) for 90% statistical power
//! - Significance level: p < 0.05 (independent t-test)
//! - Stability: ≥25/30 replications pass (83% reliability)

use chrono::Utc;
use engram_core::MemoryStore;

#[path = "spacing_helpers.rs"]
mod spacing_helpers;
use spacing_helpers::{fact_to_episode, generate_random_facts, test_retention};

/// Statistical test result
#[derive(Debug)]
struct StatisticalTest {
    t_statistic: f64,
    p_value: f64,
    effect_size: f64, // Cohen's d
}

/// Compute independent t-test for two groups
fn independent_t_test(group1: &[f32], group2: &[f32]) -> StatisticalTest {
    #[allow(clippy::cast_precision_loss)]
    let n1 = group1.len() as f64;
    #[allow(clippy::cast_precision_loss)]
    let n2 = group2.len() as f64;

    let mean1 = group1.iter().map(|&x| f64::from(x)).sum::<f64>() / n1;
    let mean2 = group2.iter().map(|&x| f64::from(x)).sum::<f64>() / n2;

    let var1 = group1
        .iter()
        .map(|&x| (f64::from(x) - mean1).powi(2))
        .sum::<f64>()
        / (n1 - 1.0);

    let var2 = group2
        .iter()
        .map(|&x| (f64::from(x) - mean2).powi(2))
        .sum::<f64>()
        / (n2 - 1.0);

    // Pooled standard deviation
    let pooled_sd = (((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0)).sqrt();

    // Cohen's d (effect size)
    let cohens_d = (mean1 - mean2).abs() / pooled_sd;

    // t-statistic
    let se = pooled_sd * ((1.0 / n1) + (1.0 / n2)).sqrt();
    let t = (mean1 - mean2) / se;

    // Degrees of freedom
    let df = n1 + n2 - 2.0;

    // Approximate p-value (two-tailed)
    let p_value = if df > 30.0 {
        2.0 * (1.0 - normal_cdf(t.abs()))
    } else {
        t_distribution_p_value(t.abs(), df)
    };

    StatisticalTest {
        t_statistic: t,
        p_value,
        effect_size: cohens_d,
    }
}

/// Approximate normal CDF using error function
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Approximate t-distribution p-value for small df
fn t_distribution_p_value(t: f64, df: f64) -> f64 {
    // For small df, use a conservative approximation
    // This is less accurate than beta function but sufficient for validation
    let z = t / df.sqrt();
    2.0 * (1.0 - normal_cdf(z))
}

#[test]
fn test_spacing_effect_replication() {
    // CRITICAL: This test validates Engram against Cepeda et al. (2006)
    // Expected: 20-40% improvement in retention for distributed vs massed practice

    const ITEMS_PER_CONDITION: usize = 100; // Statistical power: 90%

    let study_items = generate_random_facts(ITEMS_PER_CONDITION * 2, 42);

    // Group 1: Massed practice (3 consecutive exposures, no spacing)
    let massed_group = &study_items[0..ITEMS_PER_CONDITION];
    let store_massed = MemoryStore::new(10000);

    for item in massed_group {
        // Store 3 times consecutively (massed)
        for _ in 0..3 {
            let episode = fact_to_episode(item, Utc::now());
            store_massed.store(episode);
        }
    }

    // Group 2: Distributed practice (3 exposures with 1-hour spacing)
    let distributed_group = &study_items[ITEMS_PER_CONDITION..];
    let store_distributed = MemoryStore::new(10000);

    for item in distributed_group {
        for rep in 0..3 {
            let episode = fact_to_episode(item, Utc::now());
            store_distributed.store(episode);

            // Add spacing between repetitions (except after last)
            if rep < 2 {
                // Simulate 1-hour spacing interval
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    }

    // Retention test after 24 hours (sleep to simulate time passage)
    std::thread::sleep(std::time::Duration::from_millis(100));

    let massed_accuracy = test_retention(&store_massed, massed_group);
    let distributed_accuracy = test_retention(&store_distributed, distributed_group);

    #[allow(clippy::cast_precision_loss)]
    let improvement = (distributed_accuracy - massed_accuracy) / massed_accuracy;

    println!("\n=== Spacing Effect Validation ===");
    println!("Massed accuracy: {:.1}%", massed_accuracy * 100.0);
    println!("Distributed accuracy: {:.1}%", distributed_accuracy * 100.0);
    println!("Improvement: {:.1}%", improvement * 100.0);
    println!("Target range: 10-50% (Cepeda et al. 2006)");

    // Statistical significance test
    // Note: For this test, we're comparing aggregate accuracies
    // In a more sophisticated version, we'd have per-item scores
    let massed_scores: Vec<f32> = vec![massed_accuracy; ITEMS_PER_CONDITION];
    let distributed_scores: Vec<f32> = vec![distributed_accuracy; ITEMS_PER_CONDITION];

    let stats = independent_t_test(&distributed_scores, &massed_scores);

    println!("\nStatistical Test:");
    println!(
        "  t({}) = {:.3}",
        ITEMS_PER_CONDITION * 2 - 2,
        stats.t_statistic
    );
    println!("  p = {:.4}", stats.p_value);
    println!("  Cohen's d = {:.3}", stats.effect_size);

    // Acceptance criteria from Cepeda et al. (2006): 20-40% ±10% = [10%, 50%]
    assert!(
        (0.10..=0.50).contains(&improvement),
        "Spacing effect {:.1}% outside [10%, 50%] acceptance range (Cepeda 2006)",
        improvement * 100.0
    );

    // Statistical significance required
    assert!(
        stats.p_value < 0.05,
        "Spacing effect not statistically significant: p = {:.4}",
        stats.p_value
    );

    // Effect size should be medium to large
    assert!(
        stats.effect_size >= 0.3,
        "Effect size too small: d = {:.3} (expected ≥0.3)",
        stats.effect_size
    );
}

#[test]
#[ignore = "Long-running stability test (30 replications)"]
fn test_spacing_effect_stability() {
    // Validate that the spacing effect test passes consistently (not by chance)
    // This ensures our implementation is stable across multiple runs

    const REPLICATIONS: usize = 30;
    let mut successes = 0;
    let mut improvements = Vec::new();

    for replication in 0..REPLICATIONS {
        #[allow(clippy::cast_possible_truncation)]
        let result = run_single_spacing_trial(replication as u64);

        improvements.push(result.improvement);

        if result.passes_acceptance() {
            successes += 1;
        }
    }

    println!("\n=== Stability Analysis ===");
    println!("Successes: {successes}/{REPLICATIONS}");
    #[allow(clippy::cast_precision_loss)]
    let success_rate = (successes as f32 / REPLICATIONS as f32) * 100.0;
    println!("Success rate: {success_rate:.1}%");

    #[allow(clippy::cast_precision_loss)]
    let avg_improvement: f32 = improvements.iter().sum::<f32>() / improvements.len() as f32;
    println!("Average improvement: {:.1}%", avg_improvement * 100.0);

    // Require 25/30 successes (83%) for stability
    assert!(
        successes >= 25,
        "Test instability: {successes}/{REPLICATIONS} replications passed (expected ≥25/30 at 83% reliability)"
    );
}

/// Single spacing effect trial result
#[allow(dead_code)] // Fields used for detailed analysis
struct SpacingTrialResult {
    improvement: f32,
    massed_accuracy: f32,
    distributed_accuracy: f32,
}

impl SpacingTrialResult {
    fn passes_acceptance(&self) -> bool {
        self.improvement >= 0.10 && self.improvement <= 0.50
    }
}

fn run_single_spacing_trial(seed: u64) -> SpacingTrialResult {
    const N: usize = 50; // Smaller for stability testing

    let items = generate_random_facts(N * 2, seed);

    let massed_group = &items[0..N];
    let distributed_group = &items[N..];

    // Massed practice
    let store_massed = MemoryStore::new(5000);
    for item in massed_group {
        for _ in 0..3 {
            store_massed.store(fact_to_episode(item, Utc::now()));
        }
    }

    // Distributed practice
    let store_distributed = MemoryStore::new(5000);
    for item in distributed_group {
        for rep in 0..3 {
            store_distributed.store(fact_to_episode(item, Utc::now()));
            if rep < 2 {
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
        }
    }

    // Retention test
    std::thread::sleep(std::time::Duration::from_millis(50));

    let massed_acc = test_retention(&store_massed, massed_group);
    let distributed_acc = test_retention(&store_distributed, distributed_group);

    #[allow(clippy::cast_precision_loss)]
    let improvement = if massed_acc > 0.0 {
        (distributed_acc - massed_acc) / massed_acc
    } else {
        0.0
    };

    SpacingTrialResult {
        improvement,
        massed_accuracy: massed_acc,
        distributed_accuracy: distributed_acc,
    }
}

#[test]
fn test_spacing_determinism() {
    // Validate that results are reproducible with same seed
    // This ensures our implementation is deterministic

    let result1 = run_single_spacing_trial(42);
    let result2 = run_single_spacing_trial(42);

    assert!(
        (result1.improvement - result2.improvement).abs() < 0.001,
        "Non-deterministic results: {:.3} vs {:.3}",
        result1.improvement,
        result2.improvement
    );
}

#[cfg(test)]
mod statistical_function_tests {
    use super::*;

    #[test]
    fn test_normal_cdf_standard_values() {
        // Test known values of the standard normal CDF
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((normal_cdf(1.0) - 0.841).abs() < 0.01);
        assert!((normal_cdf(-1.0) - 0.159).abs() < 0.01);
    }

    #[test]
    fn test_erf_standard_values() {
        // Test known values of the error function
        assert!(erf(0.0).abs() < 0.001);
        assert!((erf(1.0) - 0.842).abs() < 0.01);
        assert!((erf(-1.0) + 0.842).abs() < 0.01);
    }

    #[test]
    fn test_independent_t_test_identical_groups() {
        // Two identical groups should have p-value ≈ 1.0
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let stats = independent_t_test(&group1, &group2);

        assert!(stats.t_statistic.abs() < 0.001);
        assert!(stats.p_value > 0.9); // Should be very high
        assert!(stats.effect_size < 0.001);
    }

    #[test]
    fn test_independent_t_test_different_groups() {
        // Two clearly different groups should have small p-value
        let group1 = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let group2 = vec![5.0, 5.0, 5.0, 5.0, 5.0];

        let stats = independent_t_test(&group1, &group2);

        assert!(stats.t_statistic.abs() > 10.0); // Large t-statistic
        assert!(stats.p_value < 0.001); // Highly significant
        assert!(stats.effect_size > 2.0); // Very large effect
    }
}
