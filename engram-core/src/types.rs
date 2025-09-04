//! Core error types for engram-core
//!
//! Follows cognitive guidance principles: every error includes context, suggestion, and example
//! to help developers resolve issues quickly, even when tired or under stress.

use thiserror::Error;

/// Core error types for the engram system
///
/// Each error follows the Context-Suggestion-Example pattern to support
/// System 1 (pattern recognition) thinking when debugging.
#[derive(Error, Debug)]
pub enum CoreError {
    #[error(
        "Memory node '{id}' not found in graph\n  Expected: Valid node ID from current graph\n  Suggestion: Use graph.nodes() to list available nodes{similar}\n  Example: let node = graph.get_node(\"node_id\").or_insert_default();"
    )]
    NodeNotFound {
        id: String,
        /// "Did you mean?" suggestions, formatted as ", or did you mean 'x', 'y'?"
        similar: String,
    },

    #[error(
        "Invalid activation level: {value} (must be in range [0.0, 1.0])\n  Expected: Activation between 0.0 (inactive) and 1.0 (fully active)\n  Suggestion: Use activation.clamp(0.0, 1.0) to ensure valid range\n  Example: node.activate(energy.clamp(0.0, 1.0));"
    )]
    InvalidActivation { value: f64 },

    #[error(
        "Reconstruction failed: activation level {level:.3} below threshold {threshold:.3}\n  Expected: Sufficient activation for pattern completion\n  Suggestion: Increase activation through more recall attempts or lower threshold\n  Example: graph.activate_neighbors(node_id, boost=0.3).reconstruct_with_threshold(0.5)"
    )]
    InsufficientActivation { level: f64, threshold: f64 },

    #[error(
        "Invalid confidence interval: mean={mean:.3} not in range [{lower:.3}, {upper:.3}]\n  Expected: mean ∈ [lower, upper] and all values ∈ [0, 1]\n  Suggestion: Use Confidence::new_clamped() to automatically fix bounds\n  Example: let conf = Confidence::new_clamped(mean, lower, upper);"
    )]
    InvalidConfidence { mean: f64, lower: f64, upper: f64 },

    #[error(
        "Serialization failed: {context}\n  Expected: Valid JSON-serializable memory structure\n  Suggestion: Check for NaN/Infinity values in embeddings or confidence scores\n  Example: node.validate_serializable()?.to_json()"
    )]
    SerializationError {
        context: String,
        #[source]
        source: serde_json::Error,
    },

    #[error(
        "Validation failed: {reason}\n  Expected: {expected}\n  Suggestion: {suggestion}\n  Example: {example}"
    )]
    ValidationError {
        reason: String,
        expected: String,
        suggestion: String,
        example: String,
    },

    #[error(
        "IO operation failed: {context}\n  Expected: {expected}\n  Suggestion: {suggestion}\n  Example: {example}"
    )]
    IoError {
        context: String,
        expected: String,
        suggestion: String,
        example: String,
        #[source]
        source: std::io::Error,
    },
}

impl CoreError {
    /// Helper to create NodeNotFound with similar suggestions
    pub fn node_not_found(id: impl Into<String>, similar: Vec<String>) -> Self {
        let id = id.into();
        let similar = if similar.is_empty() {
            String::new()
        } else {
            format!(", or did you mean: {}?", similar.join(", "))
        };
        Self::NodeNotFound { id, similar }
    }

    /// Helper to create IoError with full context
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
