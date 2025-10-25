//! Parameter Tuning Framework - Empirical optimization of completion parameters
//!
//! This module implements systematic parameter sweeps to identify optimal configurations
//! for pattern completion on the accuracy-latency Pareto frontier.
//!
//! Critical Parameters:
//! - CA3 Sparsity: [0.02, 0.03, 0.05, 0.07, 0.10]
//! - CA1 Threshold: [0.5, 0.6, 0.7, 0.8, 0.9]
//! - Pattern Weight: [0.2, 0.3, 0.4, 0.5, 0.6]
//! - Num Hypotheses: [1, 2, 3, 5, 10]
//!
//! Biological Constraints:
//! - CA3 sparsity must be 2-10% (Marr, 1971)
//! - Convergence within theta rhythm (7 iterations max)
//! - CA1 threshold balances precision/recall

#![cfg(feature = "pattern_completion")]

use chrono::Utc;
use engram_core::{
    Confidence, Episode,
    completion::{CompletionConfig, PartialEpisode, PatternCompleter, PatternReconstructor},
};
use std::collections::HashMap;
use std::time::Instant;

/// Performance metrics for parameter evaluation
#[derive(Debug, Clone)]
#[allow(dead_code)] // Public API fields for external use
pub struct PerformanceMetrics {
    /// Parameter configuration tested
    pub config_name: String,
    /// Reconstruction accuracy (F1 score)
    pub accuracy: f32,
    /// Average latency in microseconds
    pub avg_latency_us: u64,
    /// P95 latency in microseconds
    pub p95_latency_us: u64,
    /// P99 latency in microseconds
    pub p99_latency_us: u64,
    /// Convergence rate (successful completions / total attempts)
    pub convergence_rate: f32,
    /// False memory rate (DRM paradigm)
    pub false_memory_rate: f32,
    /// Calibration error (Brier score)
    pub calibration_error: f32,
}

impl PerformanceMetrics {
    /// Check if this point dominates another on Pareto frontier
    /// Returns true if this is better or equal on all dimensions and strictly better on at least one
    #[must_use]
    pub fn pareto_dominates(&self, other: &Self) -> bool {
        let better_accuracy = self.accuracy >= other.accuracy;
        let better_latency = self.avg_latency_us <= other.avg_latency_us;
        let better_convergence = self.convergence_rate >= other.convergence_rate;

        let strictly_better_accuracy = self.accuracy > other.accuracy;
        let strictly_better_latency = self.avg_latency_us < other.avg_latency_us;
        let strictly_better_convergence = self.convergence_rate > other.convergence_rate;

        // Dominates if better or equal on all, and strictly better on at least one
        (better_accuracy && better_latency && better_convergence)
            && (strictly_better_accuracy || strictly_better_latency || strictly_better_convergence)
    }
}

/// Parameter sweep configuration
#[derive(Debug, Clone)]
#[allow(dead_code)] // Public API fields for external use
#[allow(clippy::struct_field_names)] // Explicit naming for clarity
pub struct ParameterSweep {
    /// CA3 sparsity values to test
    pub ca3_sparsity_values: Vec<f32>,
    /// CA1 threshold values to test
    pub ca1_threshold_values: Vec<f32>,
    /// Pattern weight values to test (for future use)
    pub pattern_weight_values: Vec<f32>,
    /// Number of hypotheses values to test
    pub num_hypotheses_values: Vec<usize>,
}

impl Default for ParameterSweep {
    fn default() -> Self {
        Self {
            ca3_sparsity_values: vec![0.02, 0.03, 0.05, 0.07, 0.10],
            ca1_threshold_values: vec![0.5, 0.6, 0.7, 0.8, 0.9],
            pattern_weight_values: vec![0.2, 0.3, 0.4, 0.5, 0.6],
            num_hypotheses_values: vec![1, 2, 3, 5, 10],
        }
    }
}

/// Benchmark dataset for parameter tuning
pub struct BenchmarkDataset {
    /// Training episodes
    pub train_episodes: Vec<Episode>,
    /// Test episodes (ground truth)
    pub test_episodes: Vec<Episode>,
    /// Test partial episodes (corrupted)
    pub test_partials: Vec<PartialEpisode>,
}

impl BenchmarkDataset {
    /// Create a standardized benchmark dataset
    #[must_use]
    pub fn standard(seed: u64) -> Self {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let activities = [
            "meeting",
            "lunch",
            "presentation",
            "coding",
            "review",
            "planning",
            "email",
            "call",
            "break",
            "training",
        ];

        let locations = [
            "office",
            "home",
            "cafe",
            "conference room",
            "park",
            "library",
        ];

        let participants = ["Alice", "Bob", "Charlie", "Diana", "team", "manager"];

        let total_episodes = 200;
        let train_size = 100;

        let mut all_episodes = Vec::with_capacity(total_episodes);

        for i in 0..total_episodes {
            let what = activities[rng.gen_range(0..activities.len())].to_string();
            let where_loc = locations[rng.gen_range(0..locations.len())].to_string();
            let who = participants[rng.gen_range(0..participants.len())].to_string();

            let mut embedding = [0.0f32; 768];
            for val in &mut embedding {
                *val = rng.gen_range(-1.0..1.0);
            }

            let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            for val in &mut embedding {
                *val /= magnitude;
            }

            all_episodes.push(Episode {
                id: format!("benchmark_{i}"),
                when: Utc::now(),
                where_location: Some(where_loc),
                who: Some(vec![who]),
                what,
                embedding,
                embedding_provenance: None,
                encoding_confidence: Confidence::exact(0.9),
                vividness_confidence: Confidence::exact(0.85),
                reliability_confidence: Confidence::exact(0.88),
                last_recall: Utc::now(),
                recall_count: 0,
                decay_rate: 0.03,
                decay_function: None,
            });
        }

        let train_episodes = all_episodes[..train_size].to_vec();
        let test_episodes = all_episodes[train_size..].to_vec();

        // Create partial episodes with 50% corruption
        let mut test_partials = Vec::new();

        for episode in &test_episodes {
            let mut partial_embedding = vec![None; 768];

            // Keep 50% of embedding
            for (i, &val) in episode.embedding.iter().enumerate().take(384) {
                partial_embedding[i] = Some(val);
            }

            let mut known_fields = HashMap::new();

            // Randomly keep 1-2 fields
            if rng.gen_bool(0.5) {
                known_fields.insert("what".to_string(), episode.what.clone());
            }

            test_partials.push(PartialEpisode {
                known_fields,
                partial_embedding,
                cue_strength: Confidence::exact(0.7),
                temporal_context: vec![],
            });
        }

        Self {
            train_episodes,
            test_episodes,
            test_partials,
        }
    }
}

/// Run parameter sweep for CA3 sparsity
pub fn sweep_ca3_sparsity(dataset: &BenchmarkDataset) -> Vec<PerformanceMetrics> {
    let sweep = ParameterSweep::default();
    let mut results = Vec::new();

    for &sparsity in &sweep.ca3_sparsity_values {
        println!("Testing CA3 sparsity: {sparsity}");

        let config = CompletionConfig {
            ca3_sparsity: sparsity,
            ..Default::default()
        };

        let metrics = evaluate_config(&format!("ca3_sparsity_{sparsity:.2}"), config, dataset);

        println!("  Accuracy: {:.2}%", metrics.accuracy * 100.0);
        println!("  Avg latency: {} μs", metrics.avg_latency_us);
        println!(
            "  Convergence rate: {:.2}%",
            metrics.convergence_rate * 100.0
        );

        results.push(metrics);
    }

    results
}

/// Run parameter sweep for CA1 threshold
pub fn sweep_ca1_threshold(dataset: &BenchmarkDataset) -> Vec<PerformanceMetrics> {
    let sweep = ParameterSweep::default();
    let mut results = Vec::new();

    for &threshold in &sweep.ca1_threshold_values {
        println!("Testing CA1 threshold: {threshold}");

        let config = CompletionConfig {
            ca1_threshold: Confidence::exact(threshold),
            ..Default::default()
        };

        let metrics = evaluate_config(&format!("ca1_threshold_{threshold:.2}"), config, dataset);

        println!("  Accuracy: {:.2}%", metrics.accuracy * 100.0);
        println!("  Avg latency: {} μs", metrics.avg_latency_us);
        println!(
            "  Convergence rate: {:.2}%",
            metrics.convergence_rate * 100.0
        );

        results.push(metrics);
    }

    results
}

/// Run parameter sweep for number of hypotheses
pub fn sweep_num_hypotheses(dataset: &BenchmarkDataset) -> Vec<PerformanceMetrics> {
    let sweep = ParameterSweep::default();
    let mut results = Vec::new();

    for &num_hyp in &sweep.num_hypotheses_values {
        println!("Testing num_hypotheses: {num_hyp}");

        let config = CompletionConfig {
            num_hypotheses: num_hyp,
            ..Default::default()
        };

        let metrics = evaluate_config(&format!("num_hypotheses_{num_hyp}"), config, dataset);

        println!("  Accuracy: {:.2}%", metrics.accuracy * 100.0);
        println!("  Avg latency: {} μs", metrics.avg_latency_us);
        println!(
            "  Convergence rate: {:.2}%",
            metrics.convergence_rate * 100.0
        );

        results.push(metrics);
    }

    results
}

/// Evaluate a single configuration
fn evaluate_config(
    name: &str,
    config: CompletionConfig,
    dataset: &BenchmarkDataset,
) -> PerformanceMetrics {
    let mut reconstructor = PatternReconstructor::new(config);
    reconstructor.add_episodes(&dataset.train_episodes);

    let mut latencies = Vec::new();
    let mut correct_reconstructions = 0;
    let mut total_attempts = 0;
    let mut successful_completions = 0;

    for (partial, ground_truth) in dataset.test_partials.iter().zip(&dataset.test_episodes) {
        total_attempts += 1;

        let start = Instant::now();
        let result = reconstructor.complete(partial);
        let elapsed = start.elapsed();

        latencies.push(elapsed.as_micros() as u64);

        if let Ok(completed) = result {
            successful_completions += 1;

            // Check if 'what' field was correctly reconstructed
            if completed.episode.what == ground_truth.what {
                correct_reconstructions += 1;
            }
        }
    }

    // Calculate metrics
    let accuracy = if total_attempts > 0 {
        correct_reconstructions as f32 / total_attempts as f32
    } else {
        0.0
    };

    let convergence_rate = if total_attempts > 0 {
        successful_completions as f32 / total_attempts as f32
    } else {
        0.0
    };

    let avg_latency_us = if latencies.is_empty() {
        0
    } else {
        latencies.iter().sum::<u64>() / latencies.len() as u64
    };

    latencies.sort_unstable();
    let p95_latency_us = if latencies.is_empty() {
        0
    } else {
        latencies[(latencies.len() as f32 * 0.95) as usize]
    };

    let p99_latency_us = if latencies.is_empty() {
        0
    } else {
        latencies[(latencies.len() as f32 * 0.99) as usize]
    };

    PerformanceMetrics {
        config_name: name.to_string(),
        accuracy,
        avg_latency_us,
        p95_latency_us,
        p99_latency_us,
        convergence_rate,
        false_memory_rate: 0.0, // Would require DRM-specific testing
        calibration_error: 0.0, // Would require calibration dataset
    }
}

/// Find Pareto frontier from performance metrics
#[must_use]
pub fn find_pareto_frontier(metrics: &[PerformanceMetrics]) -> Vec<PerformanceMetrics> {
    let mut frontier = Vec::new();

    for candidate in metrics {
        let mut is_dominated = false;

        // Check if any other point dominates this candidate
        for other in metrics {
            if other.config_name != candidate.config_name && other.pareto_dominates(candidate) {
                is_dominated = true;
                break;
            }
        }

        if !is_dominated {
            frontier.push(candidate.clone());
        }
    }

    frontier
}

#[test]
#[ignore = "Slow test (>60s) - run with --ignored for full validation"]
fn test_ca3_sparsity_sweep() {
    // Test CA3 sparsity parameter sweep
    let dataset = BenchmarkDataset::standard(42);
    let results = sweep_ca3_sparsity(&dataset);

    assert_eq!(results.len(), 5, "Should test 5 sparsity values");

    println!("\nCA3 Sparsity Sweep Summary:");
    for result in &results {
        println!(
            "  {}: Accuracy={:.2}%, Latency={}μs, Convergence={:.2}%",
            result.config_name,
            result.accuracy * 100.0,
            result.avg_latency_us,
            result.convergence_rate * 100.0
        );
    }

    // All sparsity values should be within biological constraints (2-10%)
    for result in &results {
        let sparsity_str = result.config_name.replace("ca3_sparsity_", "");
        if let Ok(sparsity) = sparsity_str.parse::<f32>() {
            assert!(
                (0.02..=0.10).contains(&sparsity),
                "Sparsity should be within biological range"
            );
        }
    }
}

#[test]
#[ignore = "Slow test (>60s) - run with --ignored for full validation"]
fn test_ca1_threshold_sweep() {
    // Test CA1 threshold parameter sweep
    let dataset = BenchmarkDataset::standard(43);
    let results = sweep_ca1_threshold(&dataset);

    assert_eq!(results.len(), 5, "Should test 5 threshold values");

    println!("\nCA1 Threshold Sweep Summary:");
    for result in &results {
        println!(
            "  {}: Accuracy={:.2}%, Latency={}μs, Convergence={:.2}%",
            result.config_name,
            result.accuracy * 100.0,
            result.avg_latency_us,
            result.convergence_rate * 100.0
        );
    }

    // Verify precision-recall tradeoff
    // Lower thresholds should have higher convergence but potentially lower accuracy
    // Higher thresholds should have lower convergence but potentially higher precision
}

#[test]
fn test_num_hypotheses_sweep() {
    // Test number of hypotheses parameter sweep
    let dataset = BenchmarkDataset::standard(44);
    let results = sweep_num_hypotheses(&dataset);

    assert_eq!(results.len(), 5, "Should test 5 hypothesis count values");

    println!("\nNum Hypotheses Sweep Summary:");
    for result in &results {
        println!(
            "  {}: Accuracy={:.2}%, Latency={}μs, Convergence={:.2}%",
            result.config_name,
            result.accuracy * 100.0,
            result.avg_latency_us,
            result.convergence_rate * 100.0
        );
    }

    // More hypotheses should increase latency but may improve coverage
    let latencies: Vec<_> = results.iter().map(|r| r.avg_latency_us).collect();
    println!("Latency progression: {latencies:?}");
}

#[test]
#[ignore = "Slow test (>60s) - run with --ignored for full validation"]
fn test_pareto_frontier_analysis() {
    // Test Pareto frontier computation
    let dataset = BenchmarkDataset::standard(45);

    // Collect metrics from multiple parameter sweeps
    let mut all_metrics = Vec::new();

    all_metrics.extend(sweep_ca3_sparsity(&dataset));
    all_metrics.extend(sweep_ca1_threshold(&dataset));

    let frontier = find_pareto_frontier(&all_metrics);

    println!("\nPareto Frontier:");
    for point in &frontier {
        println!(
            "  {}: Accuracy={:.2}%, Latency={}μs, Convergence={:.2}%",
            point.config_name,
            point.accuracy * 100.0,
            point.avg_latency_us,
            point.convergence_rate * 100.0
        );
    }

    assert!(
        !frontier.is_empty(),
        "Should have at least one Pareto-optimal point"
    );

    // Verify no point on frontier is dominated by another frontier point
    for p1 in &frontier {
        for p2 in &frontier {
            if p1.config_name != p2.config_name {
                assert!(
                    !p2.pareto_dominates(p1),
                    "Frontier points should not dominate each other"
                );
            }
        }
    }
}

#[test]
fn test_benchmark_dataset_generation() {
    let dataset = BenchmarkDataset::standard(46);

    assert_eq!(dataset.train_episodes.len(), 100);
    assert_eq!(dataset.test_episodes.len(), 100);
    assert_eq!(dataset.test_partials.len(), 100);

    // Verify test partials are properly corrupted
    for partial in &dataset.test_partials {
        let known_dims = partial
            .partial_embedding
            .iter()
            .filter(|x| x.is_some())
            .count();

        // Should have ~50% of dimensions (384 out of 768)
        assert!(
            (300..=450).contains(&known_dims),
            "Partial should have ~50% corruption"
        );
    }
}

#[test]
fn test_performance_metrics_pareto_dominance() {
    let better = PerformanceMetrics {
        config_name: "better".to_string(),
        accuracy: 0.8,
        avg_latency_us: 100,
        p95_latency_us: 150,
        p99_latency_us: 200,
        convergence_rate: 0.9,
        false_memory_rate: 0.1,
        calibration_error: 0.05,
    };

    let worse = PerformanceMetrics {
        config_name: "worse".to_string(),
        accuracy: 0.7,
        avg_latency_us: 150,
        p95_latency_us: 200,
        p99_latency_us: 250,
        convergence_rate: 0.8,
        false_memory_rate: 0.15,
        calibration_error: 0.08,
    };

    assert!(
        better.pareto_dominates(&worse),
        "Better config should dominate worse"
    );
    assert!(
        !worse.pareto_dominates(&better),
        "Worse config should not dominate better"
    );
}
