//! Differential testing framework for cross-implementation behavioral equivalence
//!
//! This framework ensures that different implementations (Rust, future Zig, etc.)
//! produce identical results for all operations. It provides cognitive-friendly
//! error reporting that helps developers understand implementation divergences.

use engram_core::differential::reporting;
use engram_core::differential::*;
use engram_core::{Confidence, Episode, Memory, MemoryGraph};
use quickcheck::{Arbitrary, Gen, QuickCheck};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

/// Represents different implementation variants for differential testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Implementation {
    /// Primary Rust implementation (current reference)
    RustReference,
    /// Future Rust optimized implementation  
    RustOptimized,
    /// Future Zig implementation
    Zig,
    /// Future C++ implementation
    Cpp,
}

impl fmt::Display for Implementation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Implementation::RustReference => write!(f, "Rust (Reference)"),
            Implementation::RustOptimized => write!(f, "Rust (Optimized)"),
            Implementation::Zig => write!(f, "Zig"),
            Implementation::Cpp => write!(f, "C++"),
        }
    }
}

/// A test operation that can be executed across different implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifferentialOperation {
    /// Test confidence operations
    ConfidenceAnd {
        conf_a: f32,
        conf_b: f32,
    },
    ConfidenceOr {
        conf_a: f32,
        conf_b: f32,
    },
    ConfidenceNot {
        conf: f32,
    },
    ConfidenceCombineWeighted {
        conf_a: f32,
        conf_b: f32,
        weight_a: f32,
        weight_b: f32,
    },
    ConfidenceCalibration {
        conf: f32,
    },
    ConfidenceFromSuccesses {
        successes: u32,
        total: u32,
    },
    ConfidenceFromPercent {
        percent: u8,
    },
    /// Future: memory operations
    MemoryStore {
        id: String,
        content: Vec<u8>,
        confidence: f32,
    },
    MemoryRecall {
        id: String,
    },
    /// Future: graph operations
    GraphSpreadingActivation {
        start_node: String,
        activation_energy: f64,
    },
}

impl Arbitrary for DifferentialOperation {
    fn arbitrary(g: &mut Gen) -> Self {
        let operation_type: u8 = Arbitrary::arbitrary(g);
        match operation_type % 8 {
            0 => DifferentialOperation::ConfidenceAnd {
                conf_a: Arbitrary::arbitrary(g),
                conf_b: Arbitrary::arbitrary(g),
            },
            1 => DifferentialOperation::ConfidenceOr {
                conf_a: Arbitrary::arbitrary(g),
                conf_b: Arbitrary::arbitrary(g),
            },
            2 => DifferentialOperation::ConfidenceNot {
                conf: Arbitrary::arbitrary(g),
            },
            3 => DifferentialOperation::ConfidenceCombineWeighted {
                conf_a: Arbitrary::arbitrary(g),
                conf_b: Arbitrary::arbitrary(g),
                weight_a: Arbitrary::arbitrary(g),
                weight_b: Arbitrary::arbitrary(g),
            },
            4 => DifferentialOperation::ConfidenceCalibration {
                conf: Arbitrary::arbitrary(g),
            },
            5 => DifferentialOperation::ConfidenceFromSuccesses {
                successes: Arbitrary::arbitrary(g),
                total: Arbitrary::arbitrary(g),
            },
            6 => DifferentialOperation::ConfidenceFromPercent {
                percent: Arbitrary::arbitrary(g),
            },
            _ => DifferentialOperation::MemoryStore {
                id: format!("memory_{}", u32::arbitrary(g)),
                content: (0..10).map(|_| u8::arbitrary(g)).collect(),
                confidence: Arbitrary::arbitrary(g),
            },
        }
    }
}

/// Result of executing a differential operation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationResult {
    /// The implementation that produced this result
    pub implementation: Implementation,
    /// Success or failure of the operation
    pub success: bool,
    /// Serialized result data (for success cases)
    pub result_data: Option<Vec<u8>>,
    /// Error message (for failure cases)
    pub error_message: Option<String>,
    /// Execution time in nanoseconds
    pub execution_time_nanos: u64,
    /// Peak memory usage during operation (if measurable)
    pub peak_memory_bytes: Option<u64>,
}

/// Local comparison result for internal use
#[derive(Debug, Clone)]
struct LocalComparisonResult {
    /// Are the results behaviorally equivalent?
    pub equivalent: bool,
    /// Detailed explanation of differences (cognitive-friendly)
    pub difference_explanation: Option<String>,
    /// Performance comparison data
    pub performance_comparison: PerformanceComparison,
    /// Confidence in the comparison (for probabilistic operations)
    pub comparison_confidence: Confidence,
}

/// Performance comparison between implementations
#[derive(Debug, Clone)]
pub struct PerformanceComparison {
    /// Speed ratio (reference/other) - higher means other is faster
    pub speed_ratio: f64,
    /// Memory ratio (reference/other) - higher means other uses less memory  
    pub memory_ratio: Option<f64>,
    /// Human-readable performance summary
    pub summary: String,
}

/// Result of differential testing session
#[derive(Debug)]
pub struct DifferentialTestResult {
    /// Total number of operations tested
    pub operations_tested: u32,
    /// Number of equivalent results
    pub equivalent_results: u32,
    /// Number of divergent results
    pub divergent_results: u32,
    /// List of all divergences found
    pub divergences: Vec<Divergence>,
    /// Overall performance comparison
    pub performance_summary: HashMap<Implementation, PerformanceStats>,
    /// Test duration
    pub test_duration: Duration,
}

/// A specific divergence between implementations
#[derive(Debug, Clone)]
pub struct Divergence {
    /// The operation that caused divergence
    pub operation: DifferentialOperation,
    /// Results from each implementation
    pub results: HashMap<Implementation, OperationResult>,
    /// Cognitive-friendly explanation of the divergence
    pub explanation: String,
    /// Severity of the divergence
    pub severity: DivergenceSeverity,
    /// Suggested investigation steps
    pub investigation_steps: Vec<String>,
}

/// Severity levels for implementation divergences
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DivergenceSeverity {
    /// Minor performance difference (functionality identical)
    Performance,
    /// Floating point precision difference (within epsilon)
    Precision,
    /// Different error handling (same logical outcome)
    ErrorHandling,
    /// Different behavior (user-visible difference)
    Behavioral,
    /// Correctness issue (one implementation is wrong)
    Correctness,
}

impl fmt::Display for DivergenceSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DivergenceSeverity::Performance => write!(f, "Performance"),
            DivergenceSeverity::Precision => write!(f, "Precision"),
            DivergenceSeverity::ErrorHandling => write!(f, "Error Handling"),
            DivergenceSeverity::Behavioral => write!(f, "Behavioral"),
            DivergenceSeverity::Correctness => write!(f, "Correctness"),
        }
    }
}

/// Performance statistics for an implementation
#[derive(Debug, Default)]
pub struct PerformanceStats {
    /// Total operations executed
    pub operations_count: u32,
    /// Mean execution time in nanoseconds
    pub mean_execution_time_nanos: f64,
    /// Standard deviation of execution time
    pub std_dev_execution_time_nanos: f64,
    /// Peak memory usage observed
    pub peak_memory_bytes: Option<u64>,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
}

/// Main differential testing harness
pub struct DifferentialTester {
    /// Available implementations to test
    pub implementations: Vec<Implementation>,
    /// Reference implementation (used as baseline)
    pub reference_implementation: Implementation,
    /// Test configuration
    pub config: DifferentialTestConfig,
}

/// Configuration for differential testing
#[derive(Debug, Clone)]
pub struct DifferentialTestConfig {
    /// Maximum number of operations to test
    pub max_operations: u32,
    /// Timeout for individual operations
    pub operation_timeout: Duration,
    /// Epsilon for floating-point comparison
    pub epsilon: f64,
    /// Whether to collect performance metrics
    pub collect_performance: bool,
    /// Whether to save interesting test cases
    pub save_test_cases: bool,
    /// Random seed for reproducible testing
    pub random_seed: Option<u64>,
}

impl Default for DifferentialTestConfig {
    fn default() -> Self {
        Self {
            max_operations: 10_000,
            operation_timeout: Duration::from_millis(1000),
            epsilon: f32::EPSILON as f64,
            collect_performance: true,
            save_test_cases: true,
            random_seed: None,
        }
    }
}

impl DifferentialTester {
    /// Create a new differential tester
    pub fn new() -> Self {
        Self {
            implementations: vec![Implementation::RustReference],
            reference_implementation: Implementation::RustReference,
            config: DifferentialTestConfig::default(),
        }
    }

    /// Add an implementation to test
    pub fn add_implementation(&mut self, implementation: Implementation) {
        if !self.implementations.contains(&implementation) {
            self.implementations.push(implementation);
        }
    }

    /// Set the reference implementation
    pub fn set_reference(&mut self, reference: Implementation) {
        self.reference_implementation = reference;
        if !self.implementations.contains(&reference) {
            self.implementations.push(reference);
        }
    }

    /// Configure the tester
    pub fn with_config(mut self, config: DifferentialTestConfig) -> Self {
        self.config = config;
        self
    }

    /// Generate a set of test cases using QuickCheck
    pub fn generate_test_cases(&self, count: usize) -> Vec<DifferentialOperation> {
        use quickcheck::Gen;
        use rand::SeedableRng;

        let mut rng = if let Some(seed) = self.config.random_seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let mut generator = Gen::new(count);
        let mut operations = Vec::with_capacity(count);

        for _ in 0..count {
            let operation: DifferentialOperation = Arbitrary::arbitrary(&mut generator);
            operations.push(operation);
        }

        operations
    }

    /// Execute a differential test session
    pub fn run_test_session(&self) -> DifferentialTestResult {
        let start_time = Instant::now();
        let mut operations_tested = 0;
        let mut equivalent_results = 0;
        let mut divergent_results = 0;
        let mut divergences = Vec::new();
        let mut performance_stats: HashMap<Implementation, Vec<OperationResult>> = HashMap::new();

        println!("🔬 Starting differential testing session...");
        println!("   Implementations: {:?}", self.implementations);
        println!("   Reference: {}", self.reference_implementation);
        println!("   Max operations: {}", self.config.max_operations);

        // Initialize performance tracking
        for impl_type in &self.implementations {
            performance_stats.insert(*impl_type, Vec::new());
        }

        // Generate and execute test operations
        let mut qc = QuickCheck::new();
        if let Some(seed) = self.config.random_seed {
            qc = qc.generator(Gen::new(seed as usize));
        }

        // Generate and execute test operations manually since we need mutable references
        let operations = self.generate_test_cases(self.config.max_operations as usize);

        for operation in operations {
            operations_tested += 1;

            let results = self.execute_operation(&operation);

            // Track performance stats
            for (impl_type, result) in &results {
                if let Some(stats) = performance_stats.get_mut(impl_type) {
                    stats.push(result.clone());
                }
            }

            let comparison = self.compare_results(&results);

            if comparison.equivalent {
                equivalent_results += 1;
            } else {
                divergent_results += 1;

                // Create divergence record
                let divergence = self.create_divergence_record(operation, results, comparison);
                divergences.push(divergence);
            }
        }

        let test_duration = start_time.elapsed();

        // Compile performance statistics
        let performance_summary = self.compile_performance_stats(performance_stats);

        println!("✅ Differential testing completed:");
        println!("   Operations tested: {}", operations_tested);
        println!("   Equivalent results: {}", equivalent_results);
        println!("   Divergent results: {}", divergent_results);
        println!("   Test duration: {:?}", test_duration);

        DifferentialTestResult {
            operations_tested,
            equivalent_results,
            divergent_results,
            divergences,
            performance_summary,
            test_duration,
        }
    }

    /// Execute a single operation across all implementations
    pub fn execute_operation(
        &self,
        operation: &DifferentialOperation,
    ) -> HashMap<Implementation, OperationResult> {
        let mut results = HashMap::new();

        for impl_type in &self.implementations {
            let result = self.execute_operation_on_implementation(operation, *impl_type);
            results.insert(*impl_type, result);
        }

        results
    }

    /// Compare results from different implementations
    pub fn compare_results(
        &self,
        results: &HashMap<Implementation, OperationResult>,
    ) -> crate::differential::ComparisonResult {
        let reference_result = results
            .get(&self.reference_implementation)
            .expect("Reference implementation result not found");

        let mut all_equivalent = true;
        let mut explanations = Vec::new();

        for (impl_type, result) in results {
            if *impl_type == self.reference_implementation {
                continue;
            }

            let comparison = self.compare_two_results(reference_result, result);
            if !comparison.equivalent {
                all_equivalent = false;
                if let Some(explanation) = comparison.difference_explanation {
                    explanations.push(format!("{}: {}", impl_type, explanation));
                }
            }
        }

        let difference_explanation = if explanations.is_empty() {
            None
        } else {
            Some(explanations.join("; "))
        };

        crate::differential::ComparisonResult {
            equivalent: all_equivalent,
            difference_explanation,
            comparison_confidence: Confidence::HIGH, // TODO: Calculate based on operation type
        }
    }

    /// Execute an operation on a specific implementation
    fn execute_operation_on_implementation(
        &self,
        operation: &DifferentialOperation,
        implementation: Implementation,
    ) -> OperationResult {
        let start_time = Instant::now();

        // Currently we only have Rust implementation, so we execute everything on Rust
        // In the future, this will dispatch to different implementation backends
        let result = match implementation {
            Implementation::RustReference => self.execute_rust_operation(operation),
            Implementation::RustOptimized => {
                // Future: optimized Rust implementation
                self.execute_rust_operation(operation) // Placeholder
            }
            Implementation::Zig => {
                // Future: Zig implementation via FFI
                OperationResult {
                    implementation,
                    success: false,
                    result_data: None,
                    error_message: Some("Zig implementation not yet available".to_string()),
                    execution_time_nanos: 0,
                    peak_memory_bytes: None,
                }
            }
            Implementation::Cpp => {
                // Future: C++ implementation via FFI
                OperationResult {
                    implementation,
                    success: false,
                    result_data: None,
                    error_message: Some("C++ implementation not yet available".to_string()),
                    execution_time_nanos: 0,
                    peak_memory_bytes: None,
                }
            }
        };

        let execution_time = start_time.elapsed();

        OperationResult {
            implementation,
            execution_time_nanos: execution_time.as_nanos() as u64,
            ..result
        }
    }

    /// Execute operation using Rust implementation
    fn execute_rust_operation(&self, operation: &DifferentialOperation) -> OperationResult {
        match operation {
            DifferentialOperation::ConfidenceAnd { conf_a, conf_b } => {
                let confidence_a = Confidence::exact(*conf_a);
                let confidence_b = Confidence::exact(*conf_b);
                let result = confidence_a.and(confidence_b);

                OperationResult {
                    implementation: Implementation::RustReference,
                    success: true,
                    result_data: Some(bincode::serialize(&result.raw()).unwrap()),
                    error_message: None,
                    execution_time_nanos: 0, // Will be filled by caller
                    peak_memory_bytes: None,
                }
            }
            DifferentialOperation::ConfidenceOr { conf_a, conf_b } => {
                let confidence_a = Confidence::exact(*conf_a);
                let confidence_b = Confidence::exact(*conf_b);
                let result = confidence_a.or(confidence_b);

                OperationResult {
                    implementation: Implementation::RustReference,
                    success: true,
                    result_data: Some(bincode::serialize(&result.raw()).unwrap()),
                    error_message: None,
                    execution_time_nanos: 0,
                    peak_memory_bytes: None,
                }
            }
            DifferentialOperation::ConfidenceNot { conf } => {
                let confidence = Confidence::exact(*conf);
                let result = confidence.not();

                OperationResult {
                    implementation: Implementation::RustReference,
                    success: true,
                    result_data: Some(bincode::serialize(&result.raw()).unwrap()),
                    error_message: None,
                    execution_time_nanos: 0,
                    peak_memory_bytes: None,
                }
            }
            DifferentialOperation::ConfidenceCombineWeighted {
                conf_a,
                conf_b,
                weight_a,
                weight_b,
            } => {
                let confidence_a = Confidence::exact(*conf_a);
                let confidence_b = Confidence::exact(*conf_b);
                let result = confidence_a.combine_weighted(confidence_b, *weight_a, *weight_b);

                OperationResult {
                    implementation: Implementation::RustReference,
                    success: true,
                    result_data: Some(bincode::serialize(&result.raw()).unwrap()),
                    error_message: None,
                    execution_time_nanos: 0,
                    peak_memory_bytes: None,
                }
            }
            DifferentialOperation::ConfidenceCalibration { conf } => {
                let confidence = Confidence::exact(*conf);
                let result = confidence.calibrate_overconfidence();

                OperationResult {
                    implementation: Implementation::RustReference,
                    success: true,
                    result_data: Some(bincode::serialize(&result.raw()).unwrap()),
                    error_message: None,
                    execution_time_nanos: 0,
                    peak_memory_bytes: None,
                }
            }
            DifferentialOperation::ConfidenceFromSuccesses { successes, total } => {
                let result = Confidence::from_successes(*successes, *total);

                OperationResult {
                    implementation: Implementation::RustReference,
                    success: true,
                    result_data: Some(bincode::serialize(&result.raw()).unwrap()),
                    error_message: None,
                    execution_time_nanos: 0,
                    peak_memory_bytes: None,
                }
            }
            DifferentialOperation::ConfidenceFromPercent { percent } => {
                let result = Confidence::from_percent(*percent);

                OperationResult {
                    implementation: Implementation::RustReference,
                    success: true,
                    result_data: Some(bincode::serialize(&result.raw()).unwrap()),
                    error_message: None,
                    execution_time_nanos: 0,
                    peak_memory_bytes: None,
                }
            }
            DifferentialOperation::MemoryStore {
                id,
                content,
                confidence,
            } => {
                // Future: memory operations
                OperationResult {
                    implementation: Implementation::RustReference,
                    success: false,
                    result_data: None,
                    error_message: Some("Memory operations not yet implemented".to_string()),
                    execution_time_nanos: 0,
                    peak_memory_bytes: None,
                }
            }
            DifferentialOperation::MemoryRecall { id } => {
                // Future: memory operations
                OperationResult {
                    implementation: Implementation::RustReference,
                    success: false,
                    result_data: None,
                    error_message: Some("Memory operations not yet implemented".to_string()),
                    execution_time_nanos: 0,
                    peak_memory_bytes: None,
                }
            }
            DifferentialOperation::GraphSpreadingActivation {
                start_node,
                activation_energy,
            } => {
                // Future: graph operations
                OperationResult {
                    implementation: Implementation::RustReference,
                    success: false,
                    result_data: None,
                    error_message: Some("Graph operations not yet implemented".to_string()),
                    execution_time_nanos: 0,
                    peak_memory_bytes: None,
                }
            }
        }
    }

    /// Compare two operation results
    fn compare_two_results(
        &self,
        reference: &OperationResult,
        other: &OperationResult,
    ) -> LocalComparisonResult {
        // Check success/failure match
        if reference.success != other.success {
            return LocalComparisonResult {
                equivalent: false,
                difference_explanation: Some(format!(
                    "Success mismatch: {} succeeded: {}, {} succeeded: {}",
                    reference.implementation,
                    reference.success,
                    other.implementation,
                    other.success
                )),
                performance_comparison: self.calculate_performance_comparison(reference, other),
                comparison_confidence: Confidence::HIGH,
            };
        }

        // If both failed, compare error messages
        if !reference.success {
            let error_match = match (&reference.error_message, &other.error_message) {
                (Some(ref_err), Some(other_err)) => ref_err == other_err,
                (None, None) => true,
                _ => false,
            };

            return LocalComparisonResult {
                equivalent: error_match,
                difference_explanation: if !error_match {
                    Some(format!(
                        "Different error messages: {} vs {}",
                        reference.error_message.as_deref().unwrap_or("None"),
                        other.error_message.as_deref().unwrap_or("None")
                    ))
                } else {
                    None
                },
                performance_comparison: self.calculate_performance_comparison(reference, other),
                comparison_confidence: Confidence::HIGH,
            };
        }

        // Compare result data for successful operations
        let data_equivalent = match (&reference.result_data, &other.result_data) {
            (Some(ref_data), Some(other_data)) => {
                // For floating point results, use epsilon comparison
                if ref_data.len() == other_data.len() && ref_data.len() == 4 {
                    // Assume f32 result
                    if let (Ok(ref_f32), Ok(other_f32)) = (
                        bincode::deserialize::<f32>(ref_data),
                        bincode::deserialize::<f32>(other_data),
                    ) {
                        (ref_f32 - other_f32).abs() < self.config.epsilon as f32
                    } else {
                        ref_data == other_data
                    }
                } else {
                    ref_data == other_data
                }
            }
            (None, None) => true,
            _ => false,
        };

        LocalComparisonResult {
            equivalent: data_equivalent,
            difference_explanation: if !data_equivalent {
                Some("Result data mismatch".to_string())
            } else {
                None
            },
            performance_comparison: self.calculate_performance_comparison(reference, other),
            comparison_confidence: Confidence::HIGH,
        }
    }

    /// Calculate performance comparison between two results
    fn calculate_performance_comparison(
        &self,
        reference: &OperationResult,
        other: &OperationResult,
    ) -> PerformanceComparison {
        let speed_ratio = if other.execution_time_nanos == 0 {
            1.0
        } else {
            reference.execution_time_nanos as f64 / other.execution_time_nanos as f64
        };

        let memory_ratio = match (&reference.peak_memory_bytes, &other.peak_memory_bytes) {
            (Some(ref_mem), Some(other_mem)) => {
                if *other_mem == 0 {
                    None
                } else {
                    Some(*ref_mem as f64 / *other_mem as f64)
                }
            }
            _ => None,
        };

        let summary = if speed_ratio > 1.1 {
            format!("{} is {:.1}x faster", other.implementation, speed_ratio)
        } else if speed_ratio < 0.9 {
            format!(
                "{} is {:.1}x slower",
                other.implementation,
                1.0 / speed_ratio
            )
        } else {
            format!("Similar performance ({:.1}x)", speed_ratio)
        };

        PerformanceComparison {
            speed_ratio,
            memory_ratio,
            summary,
        }
    }

    /// Create a divergence record
    fn create_divergence_record(
        &self,
        operation: DifferentialOperation,
        results: HashMap<Implementation, OperationResult>,
        comparison: crate::differential::ComparisonResult,
    ) -> Divergence {
        let explanation = comparison
            .difference_explanation
            .clone()
            .unwrap_or_else(|| "Unknown divergence".to_string());

        let severity = self.classify_divergence_severity(&results, &comparison);

        let investigation_steps = self.suggest_investigation_steps(&operation, &results, severity);

        Divergence {
            operation,
            results,
            explanation,
            severity,
            investigation_steps,
        }
    }

    /// Classify the severity of a divergence
    fn classify_divergence_severity(
        &self,
        results: &HashMap<Implementation, OperationResult>,
        comparison: &crate::differential::ComparisonResult,
    ) -> DivergenceSeverity {
        // Check if it's just a performance difference
        if comparison.equivalent {
            return DivergenceSeverity::Performance;
        }

        // Check if all implementations succeeded but with different results
        let all_succeeded = results.values().all(|r| r.success);
        if all_succeeded {
            // Could be precision or behavioral difference
            // For now, assume precision - would need more sophisticated analysis
            DivergenceSeverity::Precision
        } else {
            // Some succeeded, some failed - likely behavioral or correctness issue
            DivergenceSeverity::Behavioral
        }
    }

    /// Suggest investigation steps for a divergence
    fn suggest_investigation_steps(
        &self,
        operation: &DifferentialOperation,
        results: &HashMap<Implementation, OperationResult>,
        severity: DivergenceSeverity,
    ) -> Vec<String> {
        let mut steps = Vec::new();

        steps.push("1. Reproduce the divergence with the exact same inputs".to_string());

        match severity {
            DivergenceSeverity::Performance => {
                steps.push(
                    "2. Profile both implementations to identify performance bottlenecks"
                        .to_string(),
                );
                steps.push("3. Check for different algorithmic approaches".to_string());
            }
            DivergenceSeverity::Precision => {
                steps.push(
                    "2. Check floating-point precision settings and rounding modes".to_string(),
                );
                steps.push(
                    "3. Verify mathematical operations are implemented identically".to_string(),
                );
            }
            DivergenceSeverity::Behavioral | DivergenceSeverity::Correctness => {
                steps.push(
                    "2. Review the mathematical specification for this operation".to_string(),
                );
                steps.push("3. Check boundary condition handling".to_string());
                steps.push("4. Verify input validation and error handling".to_string());
                steps.push(
                    "5. Test with similar but simpler inputs to narrow down the issue".to_string(),
                );
            }
            DivergenceSeverity::ErrorHandling => {
                steps.push("2. Compare error handling logic between implementations".to_string());
                steps.push("3. Verify error messages provide equivalent information".to_string());
            }
        }

        steps.push("Final: Document the root cause and ensure consistency".to_string());
        steps
    }

    /// Aggregate multiple performance comparisons
    fn aggregate_performance_comparisons(
        &self,
        comparisons: Vec<PerformanceComparison>,
    ) -> PerformanceComparison {
        if comparisons.is_empty() {
            return PerformanceComparison {
                speed_ratio: 1.0,
                memory_ratio: None,
                summary: "No performance data".to_string(),
            };
        }

        let mean_speed_ratio: f64 =
            comparisons.iter().map(|c| c.speed_ratio).sum::<f64>() / comparisons.len() as f64;

        let mean_memory_ratio = {
            let memory_ratios: Vec<f64> =
                comparisons.iter().filter_map(|c| c.memory_ratio).collect();
            if memory_ratios.is_empty() {
                None
            } else {
                Some(memory_ratios.iter().sum::<f64>() / memory_ratios.len() as f64)
            }
        };

        PerformanceComparison {
            speed_ratio: mean_speed_ratio,
            memory_ratio: mean_memory_ratio,
            summary: format!("Average performance ratio: {:.2}x", mean_speed_ratio),
        }
    }

    /// Compile performance statistics from collected results
    fn compile_performance_stats(
        &self,
        results: HashMap<Implementation, Vec<OperationResult>>,
    ) -> HashMap<Implementation, PerformanceStats> {
        let mut stats = HashMap::new();

        for (impl_type, operation_results) in results {
            let successful_results: Vec<_> =
                operation_results.iter().filter(|r| r.success).collect();

            let operations_count = operation_results.len() as u32;
            let success_rate = successful_results.len() as f64 / operations_count as f64;

            let execution_times: Vec<f64> = successful_results
                .iter()
                .map(|r| r.execution_time_nanos as f64)
                .collect();

            let mean_execution_time_nanos = if execution_times.is_empty() {
                0.0
            } else {
                execution_times.iter().sum::<f64>() / execution_times.len() as f64
            };

            let std_dev_execution_time_nanos = if execution_times.len() < 2 {
                0.0
            } else {
                let variance = execution_times
                    .iter()
                    .map(|t| (t - mean_execution_time_nanos).powi(2))
                    .sum::<f64>()
                    / (execution_times.len() - 1) as f64;
                variance.sqrt()
            };

            let peak_memory_bytes = successful_results
                .iter()
                .filter_map(|r| r.peak_memory_bytes)
                .max();

            stats.insert(
                impl_type,
                PerformanceStats {
                    operations_count,
                    mean_execution_time_nanos,
                    std_dev_execution_time_nanos,
                    peak_memory_bytes,
                    success_rate,
                },
            );
        }

        stats
    }
}

impl Default for DifferentialTester {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentialExecutor for DifferentialTester {
    fn execute_operation(
        &self,
        operation: &DifferentialOperation,
    ) -> HashMap<Implementation, OperationResult> {
        DifferentialTester::execute_operation(self, operation)
    }

    fn compare_results(
        &self,
        results: &HashMap<Implementation, OperationResult>,
    ) -> crate::differential::ComparisonResult {
        DifferentialTester::compare_results(self, results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differential_tester_creation() {
        let tester = DifferentialTester::new();
        assert_eq!(tester.implementations, vec![Implementation::RustReference]);
        assert_eq!(
            tester.reference_implementation,
            Implementation::RustReference
        );
    }

    #[test]
    fn test_confidence_and_operation() {
        let tester = DifferentialTester::new();
        let operation = DifferentialOperation::ConfidenceAnd {
            conf_a: 0.8,
            conf_b: 0.6,
        };

        let results = tester.execute_operation(&operation);
        assert_eq!(results.len(), 1);

        let rust_result = results.get(&Implementation::RustReference).unwrap();
        assert!(rust_result.success);
        assert!(rust_result.result_data.is_some());
    }

    #[test]
    fn test_operation_comparison() {
        let tester = DifferentialTester::new();
        let operation = DifferentialOperation::ConfidenceNot { conf: 0.3 };

        let results = tester.execute_operation(&operation);
        let comparison = tester.compare_results(&results);

        // With only one implementation, should always be equivalent to itself
        assert!(comparison.equivalent);
        assert!(comparison.difference_explanation.is_none());
    }

    #[test]
    fn test_differential_test_session() {
        let tester = DifferentialTester::new().with_config(DifferentialTestConfig {
            max_operations: 100, // Small number for test
            ..Default::default()
        });

        let result = tester.run_test_session();

        assert!(result.operations_tested > 0);
        assert_eq!(result.equivalent_results, result.operations_tested); // All should be equivalent with single impl
        assert_eq!(result.divergent_results, 0);
        assert!(result.divergences.is_empty());
    }

    #[test]
    fn test_generate_test_cases() {
        let tester = DifferentialTester::new();
        let operations = tester.generate_test_cases(1000);

        assert_eq!(operations.len(), 1000);

        // Verify we get different types of operations
        let mut confidence_and_count = 0;
        let mut confidence_or_count = 0;
        let mut confidence_not_count = 0;

        for operation in &operations {
            match operation {
                DifferentialOperation::ConfidenceAnd { .. } => confidence_and_count += 1,
                DifferentialOperation::ConfidenceOr { .. } => confidence_or_count += 1,
                DifferentialOperation::ConfidenceNot { .. } => confidence_not_count += 1,
                _ => {}
            }
        }

        // Should have some variety (at least 10% of each type)
        assert!(confidence_and_count >= 50);
        assert!(confidence_or_count >= 50);
        assert!(confidence_not_count >= 50);
    }

    #[test]
    #[ignore] // Extended test for nightly CI runs
    fn extended_differential_testing_suite() {
        let mut tester = DifferentialTester::new();

        // Generate 10,000 operations for comprehensive testing
        let operations = tester.generate_test_cases(10_000);
        println!(
            "Generated {} test operations for extended differential testing",
            operations.len()
        );

        let mut total_divergences = 0;
        let mut performance_stats = std::collections::HashMap::new();

        for (i, operation) in operations.iter().enumerate() {
            if i % 1000 == 0 {
                println!("Progress: {}/{} operations", i, operations.len());
            }

            let results = tester.execute_operation(operation);
            let comparison = tester.compare_results(&results);

            if !comparison.equivalent {
                total_divergences += 1;
                println!(
                    "Divergence #{} found in operation: {}",
                    total_divergences,
                    operation.description()
                );
            }

            // Collect performance statistics
            for (impl_type, result) in &results {
                let stats = performance_stats
                    .entry(*impl_type)
                    .or_insert_with(PerformanceStats::default);
                stats.operations_count += 1;
                stats.mean_execution_time_nanos = (stats.mean_execution_time_nanos
                    * (stats.operations_count - 1) as f64
                    + result.execution_time_nanos as f64)
                    / stats.operations_count as f64;
                if result.success {
                    stats.success_rate = (stats.success_rate * (stats.operations_count - 1) as f64
                        + 1.0)
                        / stats.operations_count as f64;
                } else {
                    stats.success_rate = (stats.success_rate * (stats.operations_count - 1) as f64)
                        / stats.operations_count as f64;
                }
            }
        }

        println!("Extended differential testing complete:");
        println!("  Total operations: {}", operations.len());
        println!("  Total divergences: {}", total_divergences);
        println!(
            "  Divergence rate: {:.4}%",
            (total_divergences as f64 / operations.len() as f64) * 100.0
        );

        for (impl_type, stats) in &performance_stats {
            println!(
                "  {}: {:.2}ns avg, {:.2}% success rate",
                impl_type,
                stats.mean_execution_time_nanos,
                stats.success_rate * 100.0
            );
        }

        // Fail if divergence rate is too high
        let divergence_rate = total_divergences as f64 / operations.len() as f64;
        assert!(
            divergence_rate < 0.001,
            "Divergence rate {:.4}% exceeds 0.1% threshold",
            divergence_rate * 100.0
        );
    }

    #[test]
    fn generate_report() {
        let mut tester = DifferentialTester::new();

        // Generate some test data
        let operations = tester.generate_test_cases(100);
        let mut all_results = Vec::new();

        for operation in &operations[..10] {
            // Test with first 10 operations
            let results = tester.execute_operation(operation);
            let comparison = tester.compare_results(&results);
            all_results.push((operation.clone(), results, comparison));
        }

        // Generate reports
        let html_report = reporting::generate_html_report(&all_results);
        let text_report = reporting::generate_text_report(&all_results);

        // Write reports to target directory
        use std::fs;
        use std::path::Path;

        let target_dir = Path::new("target");
        fs::create_dir_all(target_dir).ok();

        fs::write(
            target_dir.join("differential_testing_report.html"),
            html_report,
        )
        .unwrap();
        fs::write(
            target_dir.join("differential_testing_report.txt"),
            text_report,
        )
        .unwrap();

        println!("Differential testing reports generated in target/");
    }
}
