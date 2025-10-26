//! Shared test utilities for integration testing with Zig kernels.
//!
//! This module provides helpers for:
//! - Generating test data (embeddings, graphs)
//! - Performance comparison utilities
//! - Fuzzy float assertions for numerical stability
//! - Benchmark helpers for profiling

#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

/// Generate a random f32 embedding vector with controlled distribution
///
/// # Arguments
///
/// * `dim` - Dimension of the embedding
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// A normalized vector of length `dim`
pub fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Normalize to unit vector for cosine similarity
    let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 1e-6 {
        for val in &mut vec {
            *val /= magnitude;
        }
    }

    vec
}

/// Generate a batch of random embeddings
///
/// # Arguments
///
/// * `count` - Number of embeddings to generate
/// * `dim` - Dimension of each embedding
/// * `seed` - Base random seed (each embedding gets seed + i)
pub fn generate_embeddings(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| generate_embedding(dim, seed + i as u64))
        .collect()
}

/// Generate a similar embedding by adding small noise
///
/// # Arguments
///
/// * `base` - Base embedding to derive from
/// * `noise_scale` - Scale of Gaussian noise (0.0-1.0)
/// * `seed` - Random seed for reproducibility
pub fn generate_similar_embedding(base: &[f32], noise_scale: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vec: Vec<f32> = base
        .iter()
        .map(|&x| x + rng.gen_range(-noise_scale..noise_scale))
        .collect();

    // Re-normalize to unit vector
    let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 1e-6 {
        for val in &mut vec {
            *val /= magnitude;
        }
    }

    vec
}

/// Assert that two floats are approximately equal with relative tolerance
///
/// Uses both absolute and relative epsilon for numerical stability.
///
/// # Panics
///
/// Panics if values differ by more than epsilon
pub fn assert_approx_eq(a: f32, b: f32, epsilon: f32, label: &str) {
    let abs_diff = (a - b).abs();
    let rel_diff = if b.abs() > 1e-6 {
        abs_diff / b.abs()
    } else {
        abs_diff
    };

    assert!(
        !(abs_diff > epsilon && rel_diff > epsilon),
        "{label}: values differ: {a} vs {b} (abs_diff={abs_diff}, rel_diff={rel_diff})"
    );
}

/// Assert that two float slices are approximately equal element-wise
pub fn assert_slice_approx_eq(a: &[f32], b: &[f32], epsilon: f32, label: &str) {
    let a_len = a.len();
    let b_len = b.len();
    assert_eq!(
        a_len, b_len,
        "{label}: slice lengths differ: {a_len} vs {b_len}"
    );

    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert_approx_eq(*av, *bv, epsilon, &format!("{label}[{i}]"));
    }
}

/// Performance measurement result
#[derive(Debug, Clone)]
pub struct PerfResult {
    pub label: String,
    pub duration: Duration,
    pub operations: usize,
    pub ops_per_sec: f64,
}

impl PerfResult {
    /// Create a new performance result
    pub fn new(label: String, duration: Duration, operations: usize) -> Self {
        let ops_per_sec = if duration.as_secs_f64() > 0.0 {
            operations as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        Self {
            label,
            duration,
            operations,
            ops_per_sec,
        }
    }

    /// Calculate speedup ratio compared to baseline
    pub fn speedup(&self, baseline: &Self) -> f64 {
        if self.duration.as_secs_f64() > 0.0 {
            baseline.duration.as_secs_f64() / self.duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Format as human-readable summary
    pub fn summary(&self) -> String {
        let label = &self.label;
        let duration_ms = self.duration.as_secs_f64() * 1000.0;
        let operations = self.operations;
        let ops_per_sec = self.ops_per_sec;
        format!("{label}: {duration_ms:.2}ms ({operations} ops, {ops_per_sec:.0} ops/sec)")
    }
}

/// Benchmark a closure and return performance results
///
/// # Arguments
///
/// * `label` - Label for the benchmark
/// * `iterations` - Number of times to run the operation
/// * `f` - Closure to benchmark
pub fn benchmark<F>(label: &str, iterations: usize, mut f: F) -> PerfResult
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..10.min(iterations / 10) {
        f();
    }

    // Actual measurement
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let duration = start.elapsed();

    PerfResult::new(label.to_string(), duration, iterations)
}

/// Compare performance of two implementations
///
/// Returns (baseline_result, optimized_result, speedup)
pub fn compare_performance<F1, F2>(
    baseline_label: &str,
    optimized_label: &str,
    iterations: usize,
    mut baseline: F1,
    mut optimized: F2,
) -> (PerfResult, PerfResult, f64)
where
    F1: FnMut(),
    F2: FnMut(),
{
    let baseline_result = benchmark(baseline_label, iterations, &mut baseline);
    let optimized_result = benchmark(optimized_label, iterations, &mut optimized);
    let speedup = optimized_result.speedup(&baseline_result);

    println!("\n=== Performance Comparison ===");
    println!("Baseline:  {}", baseline_result.summary());
    println!("Optimized: {}", optimized_result.summary());
    println!("Speedup:   {speedup:.2}x");
    println!();

    (baseline_result, optimized_result, speedup)
}

/// Generate a simple CSR graph representation for testing spreading activation
///
/// Creates a linear chain: 0 -> 1 -> 2 -> ... -> (n-1)
///
/// Returns (adjacency, weights, num_nodes)
pub fn generate_chain_graph(length: usize) -> (Vec<u32>, Vec<f32>, usize) {
    let mut adjacency = Vec::new();
    let mut weights = Vec::new();

    for i in 0..(length - 1) {
        adjacency.push((i + 1) as u32);
        weights.push(0.9);
    }

    (adjacency, weights, length)
}

/// Generate a fan-out graph: central node connects to n spokes
///
/// Node 0 is the center, nodes 1..=n are spokes
///
/// Returns (adjacency, weights, num_nodes)
pub fn generate_fan_graph(num_spokes: usize) -> (Vec<u32>, Vec<f32>, usize) {
    let mut adjacency = Vec::new();
    let mut weights = Vec::new();
    let weight = 1.0 / num_spokes as f32;

    for i in 1..=num_spokes {
        adjacency.push(i as u32);
        weights.push(weight);
    }

    (adjacency, weights, num_spokes + 1)
}

/// Generate a random graph with given density
///
/// # Arguments
///
/// * `num_nodes` - Number of nodes
/// * `edge_probability` - Probability of edge between any two nodes (0.0-1.0)
/// * `seed` - Random seed
pub fn generate_random_graph(
    num_nodes: usize,
    edge_probability: f32,
    seed: u64,
) -> (Vec<u32>, Vec<f32>, usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut adjacency = Vec::new();
    let mut weights = Vec::new();

    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if i != j && rng.gen_range(0.0..1.0) < edge_probability {
                adjacency.push(j as u32);
                weights.push(rng.gen_range(0.5..1.0));
            }
        }
    }

    (adjacency, weights, num_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_embedding() {
        let emb = generate_embedding(768, 42);
        assert_eq!(emb.len(), 768);

        // Verify normalization
        let magnitude: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_approx_eq(magnitude, 1.0, 1e-5, "embedding magnitude");
    }

    #[test]
    fn test_generate_similar_embedding() {
        let base = generate_embedding(768, 42);
        let similar = generate_similar_embedding(&base, 0.02, 43); // Very small noise

        // Verify they're similar but not identical
        assert_ne!(base, similar);

        // Compute cosine similarity (should be high with small noise)
        let dot: f32 = base.iter().zip(similar.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot > 0.9,
            "similarity should be high with small noise: {dot}"
        );
    }

    #[test]
    fn test_assert_approx_eq() {
        assert_approx_eq(1.0, 1.0, 1e-6, "exact match");
        assert_approx_eq(1.0, 1.000_001, 1e-5, "close match");
    }

    #[test]
    #[should_panic(expected = "values differ")]
    fn test_assert_approx_eq_fails() {
        assert_approx_eq(1.0, 2.0, 1e-6, "should fail");
    }

    #[test]
    fn test_benchmark() {
        let result = benchmark("test_op", 100, || {
            let _x = (0..1000).sum::<i32>();
        });

        assert_eq!(result.operations, 100);
        assert!(result.duration.as_nanos() > 0);
        assert!(result.ops_per_sec > 0.0);
    }

    #[test]
    fn test_generate_chain_graph() {
        let (adj, weights, num_nodes) = generate_chain_graph(5);

        assert_eq!(num_nodes, 5);
        assert_eq!(adj.len(), 4); // 4 edges in chain of 5
        assert_eq!(weights.len(), 4);

        // Verify chain structure
        assert_eq!(adj, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_generate_fan_graph() {
        let (adj, weights, num_nodes) = generate_fan_graph(10);

        assert_eq!(num_nodes, 11); // center + 10 spokes
        assert_eq!(adj.len(), 10);
        assert_eq!(weights.len(), 10);

        // All weights should be equal
        for w in &weights {
            assert_approx_eq(*w, 0.1, 1e-6, "fan weight");
        }
    }
}
