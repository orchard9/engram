//! Core error types for engram-core
//!
//! Follows cognitive guidance principles: every error includes context, suggestion, and example
//! to help developers resolve issues quickly, even when tired or under stress.

use crate::{cognitive_error, error::CognitiveError};
use crate::Confidence;
use thiserror::Error;

/// Core error types for the engram system (legacy compatibility)
///
/// This enum provides backward compatibility while internally using `CognitiveError`.
/// New code should use `CognitiveError` directly for richer error context.
#[derive(Error, Debug)]
pub enum CoreError {
    #[error(
        "Memory node '{id}' not found in graph\n  Expected: Valid node ID from current graph\n  Suggestion: Use graph.nodes() to list available nodes{similar}\n  Example: let node = graph.get_node(\"node_id\").or_insert_default();"
    )]
    /// Memory node not found in graph
    NodeNotFound {
        /// Node ID that was not found
        id: String,
        /// "Did you mean?" suggestions, formatted as ", or did you mean 'x', 'y'?"
        similar: String,
    },

    #[error(
        "Invalid activation level: {value} (must be in range [0.0, 1.0])\n  Expected: Activation between 0.0 (inactive) and 1.0 (fully active)\n  Suggestion: Use activation.clamp(0.0, 1.0) to ensure valid range\n  Example: node.activate(energy.clamp(0.0, 1.0));"
    )]
    /// Invalid activation level (must be in range [0.0, 1.0])
    InvalidActivation {
        /// The invalid activation value
        value: f64
    },

    #[error(
        "Reconstruction failed: activation level {level:.3} below threshold {threshold:.3}\n  Expected: Sufficient activation for pattern completion\n  Suggestion: Increase activation through more recall attempts or lower threshold\n  Example: graph.activate_neighbors(node_id, boost=0.3).reconstruct_with_threshold(0.5)"
    )]
    /// Insufficient activation for reconstruction
    InsufficientActivation {
        /// Current activation level
        level: f64,
        /// Required activation threshold
        threshold: f64
    },

    #[error(
        "Invalid confidence interval: mean={mean:.3} not in range [{lower:.3}, {upper:.3}]\n  Expected: mean ∈ [lower, upper] and all values ∈ [0, 1]\n  Suggestion: Use Confidence::new_clamped() to automatically fix bounds\n  Example: let conf = Confidence::new_clamped(mean, lower, upper);"
    )]
    /// Invalid confidence interval
    InvalidConfidence {
        /// Mean confidence value
        mean: f64,
        /// Lower bound of confidence interval
        lower: f64,
        /// Upper bound of confidence interval
        upper: f64
    },

    #[error(
        "Serialization failed: {context}\n  Expected: Valid JSON-serializable memory structure\n  Suggestion: Check for NaN/Infinity values in embeddings or confidence scores\n  Example: node.validate_serializable()?.to_json()"
    )]
    /// Serialization failed
    SerializationError {
        /// Context where serialization failed
        context: String,
        #[source]
        /// The underlying serialization error
        source: serde_json::Error,
    },

    #[error(
        "Validation failed: {reason}\n  Expected: {expected}\n  Suggestion: {suggestion}\n  Example: {example}"
    )]
    /// Validation failed
    ValidationError {
        /// Reason for validation failure
        reason: String,
        /// What was expected
        expected: String,
        /// Suggested fix
        suggestion: String,
        /// Example usage
        example: String,
    },

    #[error(
        "IO operation failed: {context}\n  Expected: {expected}\n  Suggestion: {suggestion}\n  Example: {example}"
    )]
    /// IO operation failed
    IoError {
        /// Context of the IO operation
        context: String,
        /// What was expected to happen
        expected: String,
        /// Suggested fix
        suggestion: String,
        /// Example usage
        example: String,
        #[source]
        /// The underlying IO error
        source: std::io::Error,
    },
}

impl CoreError {
    /// Convert to a `CognitiveError` with full context
    pub fn to_cognitive(&self) -> CognitiveError {
        match self {
            Self::NodeNotFound { id, similar } => {
                let similar_vec = if similar.is_empty() {
                    vec![]
                } else {
                    similar
                        .trim_start_matches(", or did you mean: ")
                        .trim_end_matches('?')
                        .split(", ")
                        .map(String::from)
                        .collect()
                };

                cognitive_error!(
                    summary: format!("Memory node '{}' not found", id),
                    context: expected = "Valid node ID from current graph",
                             actual = id.clone(),
                    suggestion: "Use graph.nodes() to list available nodes",
                    example: r#"let node = graph.get_node("node_id").or_insert_default();"#,
                    confidence: Confidence::HIGH,
                    similar: similar_vec
                )
            }
            Self::InvalidActivation { value } => {
                cognitive_error!(
                    summary: format!("Invalid activation level: {}", value),
                    context: expected = "Activation between 0.0 and 1.0",
                             actual = format!("{}", value),
                    suggestion: "Use activation.clamp(0.0, 1.0) to ensure valid range",
                    example: "node.activate(energy.clamp(0.0, 1.0));",
                    confidence: Confidence::exact(1.0)
                )
            }
            Self::InsufficientActivation { level, threshold } => {
                cognitive_error!(
                    summary: "Reconstruction failed due to insufficient activation",
                    context: expected = format!("Activation >= {:.3}", threshold),
                             actual = format!("Activation = {:.3}", level),
                    suggestion: "Increase activation through more recall attempts or lower threshold",
                    example: "graph.activate_neighbors(node_id, boost=0.3).reconstruct_with_threshold(0.5)",
                    confidence: Confidence::HIGH
                )
            }
            Self::InvalidConfidence { mean, lower, upper } => {
                cognitive_error!(
                    summary: "Invalid confidence interval",
                    context: expected = "mean ∈ [lower, upper] and all values ∈ [0, 1]",
                             actual = format!("mean={:.3}, range=[{:.3}, {:.3}]", mean, lower, upper),
                    suggestion: "Use Confidence::new_clamped() to automatically fix bounds",
                    example: "let conf = Confidence::new_clamped(mean, lower, upper);",
                    confidence: Confidence::exact(1.0)
                )
            }
            Self::SerializationError { context, .. } => {
                cognitive_error!(
                    summary: "Serialization failed",
                    context: expected = "Valid JSON-serializable memory structure",
                             actual = context.clone(),
                    suggestion: "Check for NaN/Infinity values in embeddings or confidence scores",
                    example: "node.validate_serializable()?.to_json()",
                    confidence: Confidence::HIGH
                )
            }
            Self::ValidationError {
                reason,
                expected,
                suggestion,
                example,
            } => {
                cognitive_error!(
                    summary: format!("Validation failed: {}", reason),
                    context: expected = expected.clone(),
                             actual = reason.clone(),
                    suggestion: suggestion.clone(),
                    example: example.clone(),
                    confidence: Confidence::HIGH
                )
            }
            Self::IoError {
                context,
                expected,
                suggestion,
                example,
                ..
            } => {
                cognitive_error!(
                    summary: "IO operation failed",
                    context: expected = expected.clone(),
                             actual = context.clone(),
                    suggestion: suggestion.clone(),
                    example: example.clone(),
                    confidence: Confidence::HIGH
                )
            }
        }
    }

    /// Helper to create `NodeNotFound` with similar suggestions
    pub fn node_not_found(id: impl Into<String>, similar: Vec<String>) -> Self {
        let id = id.into();
        let similar = if similar.is_empty() {
            String::new()
        } else {
            format!(", or did you mean: {}?", similar.join(", "))
        };
        Self::NodeNotFound { id, similar }
    }

    /// Helper to create `IoError` with full context
    pub fn io_error(
        source: std::io::Error,
        context: impl Into<String>,
        expected: impl Into<String>,
        suggestion: impl Into<String>,
        example: impl Into<String>,
    ) -> Self {
        Self::IoError {
            context: context.into(),
            expected: expected.into(),
            suggestion: suggestion.into(),
            example: example.into(),
            source,
        }
    }
}

/// Result type alias for core operations
pub type Result<T> = std::result::Result<T, CoreError>;

/// Convert `CoreError` to `CognitiveError` for richer error context
impl From<CoreError> for CognitiveError {
    fn from(err: CoreError) -> Self {
        err.to_cognitive()
    }
}

/// Convert `CognitiveError` to `CoreError` for backward compatibility
impl From<CognitiveError> for CoreError {
    fn from(err: CognitiveError) -> Self {
        // Create a generic validation error that preserves the cognitive error information
        Self::ValidationError {
            reason: err.summary.clone(),
            expected: err.context.expected.clone(),
            suggestion: err.suggestion.clone(),
            example: err.example,
        }
    }
}
