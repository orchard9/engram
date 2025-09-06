//! Differential testing infrastructure for cross-implementation behavioral equivalence
//!
//! This module provides tools for ensuring different implementations (Rust, future Zig, etc.)
//! produce identical results while providing cognitive-friendly analysis of any divergences.

pub mod bisection;
pub mod reporting;

use crate::Confidence;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Trait for executing differential operations across implementations
pub trait DifferentialExecutor {
    /// Execute a single operation across all implementations
    fn execute_operation(
        &self,
        operation: &DifferentialOperation,
    ) -> HashMap<Implementation, OperationResult>;

    /// Compare results from different implementations
    fn compare_results(
        &self,
        results: &HashMap<Implementation, OperationResult>,
    ) -> ComparisonResult;
}

/// Result comparison for differential testing
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Are the results behaviorally equivalent?
    pub equivalent: bool,
    /// Detailed explanation of differences (cognitive-friendly)
    pub difference_explanation: Option<String>,
    /// Confidence in the comparison (for probabilistic operations)
    pub comparison_confidence: Confidence,
}

impl ComparisonResult {
    /// Check if this comparison shows divergences
    pub fn has_divergences(&self) -> bool {
        !self.equivalent
    }
}

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

/// A test operation that can be executed across different implementations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

impl DifferentialOperation {
    /// Get a human-readable description of this operation
    pub fn description(&self) -> String {
        match self {
            DifferentialOperation::ConfidenceAnd { conf_a, conf_b } => {
                format!("Confidence AND: {:.3} ∧ {:.3}", conf_a, conf_b)
            }
            DifferentialOperation::ConfidenceOr { conf_a, conf_b } => {
                format!("Confidence OR: {:.3} ∨ {:.3}", conf_a, conf_b)
            }
            DifferentialOperation::ConfidenceNot { conf } => {
                format!("Confidence NOT: ¬{:.3}", conf)
            }
            DifferentialOperation::ConfidenceCombineWeighted {
                conf_a,
                conf_b,
                weight_a,
                weight_b,
            } => {
                format!(
                    "Weighted combination: {:.3}×{:.2} + {:.3}×{:.2}",
                    conf_a, weight_a, conf_b, weight_b
                )
            }
            DifferentialOperation::ConfidenceCalibration { conf } => {
                format!("Overconfidence calibration: {:.3}", conf)
            }
            DifferentialOperation::ConfidenceFromSuccesses { successes, total } => {
                format!("Frequency-based confidence: {} / {}", successes, total)
            }
            DifferentialOperation::ConfidenceFromPercent { percent } => {
                format!("Percentage confidence: {}%", percent)
            }
            DifferentialOperation::MemoryStore {
                id,
                content,
                confidence,
            } => {
                format!(
                    "Memory store: {} ({} bytes, conf={:.3})",
                    id,
                    content.len(),
                    confidence
                )
            }
            DifferentialOperation::MemoryRecall { id } => {
                format!("Memory recall: {}", id)
            }
            DifferentialOperation::GraphSpreadingActivation {
                start_node,
                activation_energy,
            } => {
                format!(
                    "Spreading activation: {} with energy {:.3}",
                    start_node, activation_energy
                )
            }
        }
    }

    /// Get the expected cognitive complexity of this operation
    pub fn cognitive_complexity(&self) -> CognitiveComplexity {
        match self {
            DifferentialOperation::ConfidenceNot { .. } => CognitiveComplexity::Simple,
            DifferentialOperation::ConfidenceFromPercent { .. } => CognitiveComplexity::Simple,
            DifferentialOperation::ConfidenceAnd { .. }
            | DifferentialOperation::ConfidenceOr { .. } => CognitiveComplexity::Moderate,
            DifferentialOperation::ConfidenceFromSuccesses { .. } => CognitiveComplexity::Moderate,
            DifferentialOperation::ConfidenceCalibration { .. } => CognitiveComplexity::Complex,
            DifferentialOperation::ConfidenceCombineWeighted { .. } => CognitiveComplexity::Complex,
            DifferentialOperation::MemoryStore { .. }
            | DifferentialOperation::MemoryRecall { .. } => CognitiveComplexity::Moderate,
            DifferentialOperation::GraphSpreadingActivation { .. } => CognitiveComplexity::Complex,
        }
    }

    /// Simplify the operation for bisection (reduce input complexity)
    pub fn simplify(&self) -> Option<Self> {
        match self {
            DifferentialOperation::ConfidenceCombineWeighted { conf_a, conf_b, .. } => {
                // Simplify to equal weights
                Some(DifferentialOperation::ConfidenceCombineWeighted {
                    conf_a: *conf_a,
                    conf_b: *conf_b,
                    weight_a: 1.0,
                    weight_b: 1.0,
                })
            }
            DifferentialOperation::ConfidenceFromSuccesses { successes, total } => {
                if *total > 10 {
                    // Simplify to smaller numbers
                    let ratio = *successes as f64 / *total as f64;
                    let new_total = 10u32;
                    let new_successes = (ratio * new_total as f64).round() as u32;
                    Some(DifferentialOperation::ConfidenceFromSuccesses {
                        successes: new_successes,
                        total: new_total,
                    })
                } else {
                    None
                }
            }
            _ => None, // Cannot be further simplified
        }
    }
}

/// Cognitive complexity levels for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CognitiveComplexity {
    /// Simple operations developers can reason about easily
    Simple,
    /// Moderate complexity requiring some mental effort
    Moderate,
    /// Complex operations requiring significant cognitive resources
    Complex,
}

impl fmt::Display for CognitiveComplexity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CognitiveComplexity::Simple => write!(f, "Simple"),
            CognitiveComplexity::Moderate => write!(f, "Moderate"),
            CognitiveComplexity::Complex => write!(f, "Complex"),
        }
    }
}
