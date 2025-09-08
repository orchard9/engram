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
    #[must_use]
    pub const fn has_divergences(&self) -> bool {
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
            Self::RustReference => write!(f, "Rust (Reference)"),
            Self::RustOptimized => write!(f, "Rust (Optimized)"),
            Self::Zig => write!(f, "Zig"),
            Self::Cpp => write!(f, "C++"),
        }
    }
}

/// Result of executing a differential operation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
            Self::Performance => write!(f, "Performance"),
            Self::Precision => write!(f, "Precision"),
            Self::ErrorHandling => write!(f, "Error Handling"),
            Self::Behavioral => write!(f, "Behavioral"),
            Self::Correctness => write!(f, "Correctness"),
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
    /// Test confidence AND operations
    ConfidenceAnd {
        /// First confidence value
        conf_a: f32,
        /// Second confidence value
        conf_b: f32,
    },
    /// Test confidence OR operations
    ConfidenceOr {
        /// First confidence value
        conf_a: f32,
        /// Second confidence value
        conf_b: f32,
    },
    /// Test confidence NOT operations
    ConfidenceNot {
        /// Confidence value to negate
        conf: f32,
    },
    /// Test weighted confidence combination
    ConfidenceCombineWeighted {
        /// First confidence value
        conf_a: f32,
        /// Second confidence value
        conf_b: f32,
        /// Weight for first confidence
        weight_a: f32,
        /// Weight for second confidence
        weight_b: f32,
    },
    /// Test confidence calibration
    ConfidenceCalibration {
        /// Confidence value to calibrate
        conf: f32,
    },
    /// Test confidence from success rate
    ConfidenceFromSuccesses {
        /// Number of successes
        successes: u32,
        /// Total number of trials
        total: u32,
    },
    /// Test confidence from percentage
    ConfidenceFromPercent {
        /// Percentage value (0-100)
        percent: u8,
    },
    /// Future: memory operations
    MemoryStore {
        /// Memory identifier
        id: String,
        /// Memory content data
        content: Vec<u8>,
        /// Storage confidence level
        confidence: f32,
    },
    /// Test memory recall operations
    MemoryRecall {
        /// Memory identifier to recall
        id: String,
    },
    /// Future: graph operations
    GraphSpreadingActivation {
        /// Starting node for activation spread
        start_node: String,
        /// Initial activation energy
        activation_energy: f64,
    },
}

impl DifferentialOperation {
    /// Get a human-readable description of this operation
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::ConfidenceAnd { conf_a, conf_b } => {
                format!("Confidence AND: {conf_a:.3} ∧ {conf_b:.3}")
            }
            Self::ConfidenceOr { conf_a, conf_b } => {
                format!("Confidence OR: {conf_a:.3} ∨ {conf_b:.3}")
            }
            Self::ConfidenceNot { conf } => {
                format!("Confidence NOT: ¬{conf:.3}")
            }
            Self::ConfidenceCombineWeighted {
                conf_a,
                conf_b,
                weight_a,
                weight_b,
            } => {
                format!(
                    "Weighted combination: {conf_a:.3}×{weight_a:.2} + {conf_b:.3}×{weight_b:.2}"
                )
            }
            Self::ConfidenceCalibration { conf } => {
                format!("Overconfidence calibration: {conf:.3}")
            }
            Self::ConfidenceFromSuccesses { successes, total } => {
                format!("Frequency-based confidence: {successes} / {total}")
            }
            Self::ConfidenceFromPercent { percent } => {
                format!("Percentage confidence: {percent}%")
            }
            Self::MemoryStore {
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
            Self::MemoryRecall { id } => {
                format!("Memory recall: {id}")
            }
            Self::GraphSpreadingActivation {
                start_node,
                activation_energy,
            } => {
                format!("Spreading activation: {start_node} with energy {activation_energy:.3}")
            }
        }
    }

    /// Get the expected cognitive complexity of this operation
    #[must_use]
    pub const fn cognitive_complexity(&self) -> CognitiveComplexity {
        match self {
            Self::ConfidenceNot { .. } => CognitiveComplexity::Simple,
            Self::ConfidenceFromPercent { .. } => CognitiveComplexity::Simple,
            Self::ConfidenceAnd { .. } | Self::ConfidenceOr { .. } => CognitiveComplexity::Moderate,
            Self::ConfidenceFromSuccesses { .. } => CognitiveComplexity::Moderate,
            Self::ConfidenceCalibration { .. } => CognitiveComplexity::Complex,
            Self::ConfidenceCombineWeighted { .. } => CognitiveComplexity::Complex,
            Self::MemoryStore { .. } | Self::MemoryRecall { .. } => CognitiveComplexity::Moderate,
            Self::GraphSpreadingActivation { .. } => CognitiveComplexity::Complex,
        }
    }

    /// Simplify the operation for bisection (reduce input complexity)
    #[must_use]
    pub fn simplify(&self) -> Option<Self> {
        match self {
            Self::ConfidenceCombineWeighted { conf_a, conf_b, .. } => {
                // Simplify to equal weights
                Some(Self::ConfidenceCombineWeighted {
                    conf_a: *conf_a,
                    conf_b: *conf_b,
                    weight_a: 1.0,
                    weight_b: 1.0,
                })
            }
            Self::ConfidenceFromSuccesses { successes, total } => {
                if *total > 10 {
                    // Simplify to smaller numbers
                    let ratio = f64::from(*successes) / f64::from(*total);
                    let new_total = 10u32;
                    let new_successes = (ratio * f64::from(new_total)).round() as u32;
                    Some(Self::ConfidenceFromSuccesses {
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
            Self::Simple => write!(f, "Simple"),
            Self::Moderate => write!(f, "Moderate"),
            Self::Complex => write!(f, "Complex"),
        }
    }
}
