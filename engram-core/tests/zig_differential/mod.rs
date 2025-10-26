//! Differential testing harness for Zig kernels vs. Rust baseline implementations.
//!
//! This module provides comprehensive property-based testing to ensure that Zig-implemented
//! performance kernels produce numerically identical results to the reference Rust implementations.
//!
//! # Test Strategy
//!
//! 1. **Property-Based Testing** - Use proptest to generate 10,000 random test cases per kernel
//! 2. **Floating-Point Equivalence** - Verify outputs match within epsilon = 1e-6
//! 3. **Edge Case Coverage** - Explicitly test pathological inputs (zeros, NaN, overflow)
//! 4. **Regression Corpus** - Save interesting cases discovered by fuzzing
//!
//! # Test Organization
//!
//! - `vector_similarity.rs` - Differential tests for cosine similarity kernels
//! - `spreading_activation.rs` - Differential tests for graph spreading algorithms
//! - `decay_functions.rs` - Differential tests for memory decay functions
//! - `corpus/` - Saved regression test cases
//!
//! # Implementation Note
//!
//! Current Zig kernels are stubs (Task 002) that return zeros or no-ops. These tests
//! will initially detect the divergence between stubs and real implementations, which
//! is expected. Once Tasks 005-007 implement actual kernels, these tests will validate
//! correctness.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

// Sub-modules for specific kernel tests
pub mod decay_functions;
pub mod spreading_activation;
pub mod vector_similarity;

/// Epsilon values for floating-point comparisons (operation-specific tolerances)
///
/// Different operations accumulate different amounts of floating-point error:
/// - Exact operations (dot product, addition): 1e-6 is appropriate
/// - Vector operations with sqrt/division: 1e-5 accounts for transcendental error
/// - Transcendental functions (exp, log): 1e-4 accounts for ULP accumulation
/// - Iterative algorithms (spreading): 1e-3 accounts for error compounding
pub const EPSILON_EXACT: f32 = 1e-6;
pub const EPSILON_VECTOR_OPS: f32 = 1e-5;
pub const EPSILON_TRANSCENDENTAL: f32 = 1e-4;
pub const EPSILON_ITERATIVE: f32 = 1e-3;

/// Legacy epsilon for backwards compatibility
pub const EPSILON: f32 = EPSILON_EXACT;

/// Number of property-based test cases per kernel (10,000 as specified)
pub const NUM_PROPTEST_CASES: u32 = 10_000;

/// Floating-point equivalence with configurable epsilon.
///
/// Wraps two f32 values and implements ApproxEq for convenient assertion.
#[derive(Debug, Clone, Copy)]
pub struct FloatEq {
    pub expected: f32,
    pub actual: f32,
}

impl FloatEq {
    pub const fn new(expected: f32, actual: f32) -> Self {
        Self { expected, actual }
    }

    /// Assert approximate equality with default epsilon
    #[track_caller]
    #[allow(dead_code)]
    pub fn assert_eq(&self) {
        self.assert_eq_epsilon(EPSILON);
    }

    /// Assert approximate equality with custom epsilon
    #[track_caller]
    pub fn assert_eq_epsilon(&self, epsilon: f32) {
        assert!(
            (self.expected - self.actual).abs() <= epsilon,
            "Float values not equal within epsilon {}: expected {}, got {} (delta: {})",
            epsilon,
            self.expected,
            self.actual,
            (self.expected - self.actual).abs()
        );
    }

    /// Check if values are approximately equal
    #[must_use]
    pub fn is_approx_eq(&self) -> bool {
        self.is_approx_eq_epsilon(EPSILON)
    }

    /// Check if values are approximately equal with custom epsilon
    #[must_use]
    pub fn is_approx_eq_epsilon(&self, epsilon: f32) -> bool {
        (self.expected - self.actual).abs() <= epsilon
    }
}

/// Assert that two f32 slices are element-wise equal within epsilon
#[track_caller]
pub fn assert_slices_approx_eq(expected: &[f32], actual: &[f32], epsilon: f32) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "Slice lengths differ: expected {}, got {}",
        expected.len(),
        actual.len()
    );

    for (i, (exp, act)) in expected.iter().zip(actual.iter()).enumerate() {
        FloatEq::new(*exp, *act).assert_eq_epsilon(epsilon);
        assert!(
            FloatEq::new(*exp, *act).is_approx_eq_epsilon(epsilon),
            "Slice elements differ at index {i}: expected {exp}, got {act} (epsilon: {epsilon})"
        );
    }
}

/// Test corpus for regression testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase<T> {
    /// Name of the test case
    pub name: String,
    /// Description of why this case is interesting
    pub description: String,
    /// Input data for the test
    pub input: T,
    /// Expected output (from Rust baseline)
    pub expected_output: Vec<f32>,
}

/// Save a test case to the regression corpus
pub fn save_test_case<T: Serialize>(test_case: &TestCase<T>) -> Result<(), std::io::Error> {
    let corpus_dir = get_corpus_dir();
    fs::create_dir_all(&corpus_dir)?;

    let file_path = corpus_dir.join(format!("{}.json", test_case.name));
    let json = serde_json::to_string_pretty(test_case)?;
    fs::write(file_path, json)?;

    Ok(())
}

/// Load a test case from the regression corpus
pub fn load_test_case<T: for<'de> Deserialize<'de>>(
    name: &str,
) -> Result<TestCase<T>, std::io::Error> {
    let corpus_dir = get_corpus_dir();
    let file_path = corpus_dir.join(format!("{name}.json"));
    let json = fs::read_to_string(file_path)?;
    let test_case = serde_json::from_str(&json)?;
    Ok(test_case)
}

/// Get the path to the test corpus directory
fn get_corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("zig_differential")
        .join("corpus")
}

/// Test graph structure for spreading activation tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestGraph {
    /// Number of nodes in the graph
    pub num_nodes: usize,
    /// Edge source nodes (for proper differential testing)
    pub edge_sources: Vec<u32>,
    /// CSR edge destinations (adjacency list)
    pub adjacency: Vec<u32>,
    /// Edge weights corresponding to adjacency
    pub weights: Vec<f32>,
    /// Initial activation values
    pub activations: Vec<f32>,
}

impl TestGraph {
    /// Create a test graph with random structure
    #[must_use]
    pub fn random(num_nodes: usize, edge_probability: f64, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed);
        let mut edge_sources = Vec::new();
        let mut adjacency = Vec::new();
        let mut weights = Vec::new();

        // Generate edges with given probability
        for source in 0..num_nodes {
            for target in 0..num_nodes {
                if source != target && rng.gen_bool(edge_probability) {
                    edge_sources.push(source as u32);
                    adjacency.push(target as u32);
                    weights.push(rng.r#gen::<f32>());
                }
            }
        }

        // Random initial activations
        let activations = (0..num_nodes).map(|_| rng.r#gen::<f32>()).collect();

        Self {
            num_nodes,
            edge_sources,
            adjacency,
            weights,
            activations,
        }
    }

    /// Create a simple linear chain graph: 0 -> 1 -> 2 -> ... -> n
    #[must_use]
    pub fn linear_chain(num_nodes: usize) -> Self {
        let mut edge_sources = Vec::new();
        let mut adjacency = Vec::new();
        let mut weights = Vec::new();

        for i in 0..num_nodes - 1 {
            edge_sources.push(i as u32);
            adjacency.push((i + 1) as u32);
            weights.push(1.0);
        }

        let mut activations = vec![0.0; num_nodes];
        activations[0] = 1.0; // Activate first node

        Self {
            num_nodes,
            edge_sources,
            adjacency,
            weights,
            activations,
        }
    }

    /// Create a fully connected graph
    #[must_use]
    pub fn fully_connected(num_nodes: usize) -> Self {
        let mut edge_sources = Vec::new();
        let mut adjacency = Vec::new();
        let mut weights = Vec::new();

        for source in 0..num_nodes {
            for target in 0..num_nodes {
                if source != target {
                    edge_sources.push(source as u32);
                    adjacency.push(target as u32);
                    weights.push(1.0 / num_nodes as f32);
                }
            }
        }

        let mut activations = vec![0.0; num_nodes];
        activations[0] = 1.0;

        Self {
            num_nodes,
            edge_sources,
            adjacency,
            weights,
            activations,
        }
    }

    /// Create a star graph (one central node connected to all others)
    #[must_use]
    pub fn star(num_nodes: usize) -> Self {
        let mut edge_sources = Vec::new();
        let mut adjacency = Vec::new();
        let mut weights = Vec::new();

        // Center (node 0) connects to all others
        for target in 1..num_nodes {
            edge_sources.push(0);
            adjacency.push(target as u32);
            weights.push(1.0);
        }

        // All others connect back to center
        for source in 1..num_nodes {
            edge_sources.push(source as u32);
            adjacency.push(0);
            weights.push(1.0);
        }

        let mut activations = vec![0.0; num_nodes];
        activations[0] = 1.0; // Activate center

        Self {
            num_nodes,
            edge_sources,
            adjacency,
            weights,
            activations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_eq() {
        let eq = FloatEq::new(1.0, 1.0 + EPSILON / 2.0);
        assert!(eq.is_approx_eq());

        let not_eq = FloatEq::new(1.0, 1.0 + EPSILON * 2.0);
        assert!(!not_eq.is_approx_eq());
    }

    #[test]
    fn test_slices_approx_eq() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0 + EPSILON / 2.0, 2.0 - EPSILON / 2.0, 3.0];
        assert_slices_approx_eq(&a, &b, EPSILON);
    }

    #[test]
    #[should_panic(expected = "Slice lengths differ")]
    fn test_slices_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_slices_approx_eq(&a, &b, EPSILON);
    }

    #[test]
    fn test_graph_linear_chain() {
        let graph = TestGraph::linear_chain(5);
        assert_eq!(graph.num_nodes, 5);
        assert_eq!(graph.adjacency.len(), 4); // 4 edges in chain of 5
        assert_eq!(graph.activations[0], 1.0);
        assert_eq!(graph.activations[1], 0.0);
    }

    #[test]
    fn test_graph_fully_connected() {
        let graph = TestGraph::fully_connected(4);
        assert_eq!(graph.num_nodes, 4);
        assert_eq!(graph.adjacency.len(), 12); // 4*3 edges (no self-loops)
    }

    #[test]
    fn test_graph_star() {
        let graph = TestGraph::star(5);
        assert_eq!(graph.num_nodes, 5);
        assert_eq!(graph.adjacency.len(), 8); // 4 from center + 4 back to center
        assert_eq!(graph.activations[0], 1.0);
    }

    #[test]
    fn test_corpus_save_load() {
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct TestInput {
            values: Vec<f32>,
        }

        let test_case = TestCase {
            name: "test_save_load".to_string(),
            description: "Test corpus save/load functionality".to_string(),
            input: TestInput {
                values: vec![1.0, 2.0, 3.0],
            },
            expected_output: vec![4.0, 5.0, 6.0],
        };

        // Save
        save_test_case(&test_case).expect("Failed to save test case");

        // Load
        let loaded: TestCase<TestInput> =
            load_test_case("test_save_load").expect("Failed to load test case");

        assert_eq!(test_case.name, loaded.name);
        assert_eq!(test_case.description, loaded.description);
        assert_eq!(test_case.input, loaded.input);
        assert_eq!(test_case.expected_output, loaded.expected_output);

        // Cleanup
        let corpus_dir = get_corpus_dir();
        let _ = fs::remove_file(corpus_dir.join("test_save_load.json"));
    }
}
