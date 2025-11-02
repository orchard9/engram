//! Core error types for engram-core
//!
//! Follows cognitive guidance principles: every error includes context, suggestion, and example
//! to help developers resolve issues quickly, even when tired or under stress.

use crate::Confidence;
use crate::{cognitive_error, error::CognitiveError};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{self, Display};
use std::result::Result as StdResult;
use std::str::FromStr;
use std::sync::Arc;
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
        value: f64,
    },

    #[error(
        "Reconstruction failed: activation level {level:.3} below threshold {threshold:.3}\n  Expected: Sufficient activation for pattern completion\n  Suggestion: Increase activation through more recall attempts or lower threshold\n  Example: graph.activate_neighbors(node_id, boost=0.3).reconstruct_with_threshold(0.5)"
    )]
    /// Insufficient activation for reconstruction
    InsufficientActivation {
        /// Current activation level
        level: f64,
        /// Required activation threshold
        threshold: f64,
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
        upper: f64,
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

/// Memory space identifier parsing errors.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MemorySpaceIdError {
    /// Provided identifier was empty or whitespace.
    #[error(
        "Memory space identifier is empty\n  Expected: 3-64 characters using [a-z0-9_-]\n  Suggestion: Use lowercase tenant slugs like 'tenant_alpha'\n  Example: MemorySpaceId::try_from(\"tenant_alpha\")"
    )]
    Empty,

    /// Identifier length outside allowed bounds.
    #[error(
        "Memory space identifier '{value}' has invalid length {length}\n  Expected: length between 3 and 64 characters\n  Suggestion: Shorten or extend to meet bounds\n  Example: MemorySpaceId::try_from(\"alpha_team\")"
    )]
    InvalidLength {
        /// Provided value.
        value: String,
        /// Observed length.
        length: usize,
    },

    /// Identifier contained unsupported characters.
    #[error(
        "Memory space identifier '{value}' contains unsupported characters\n  Expected: Only lowercase letters, digits, '-', '_'\n  Suggestion: Convert to lowercase slug (e.g., 'tenant-42')\n  Example: MemorySpaceId::try_from(\"tenant_42\")"
    )]
    InvalidCharacters {
        /// Provided value.
        value: String,
    },
}

const MEMORY_SPACE_ID_MIN_LEN: usize = 3;
const MEMORY_SPACE_ID_MAX_LEN: usize = 64;

/// Unique identifier for a memory space/tenant.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MemorySpaceId(Arc<str>);

impl MemorySpaceId {
    /// Canonical default memory space identifier used for migrated single-tenant deployments.
    pub const DEFAULT_STR: &'static str = "default";

    /// Construct a [`MemorySpaceId`], validating the provided string.
    pub fn new<S: AsRef<str>>(value: S) -> StdResult<Self, MemorySpaceIdError> {
        Self::validate_and_intern(value.as_ref())
    }

    fn validate_and_intern(raw: &str) -> StdResult<Self, MemorySpaceIdError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(MemorySpaceIdError::Empty);
        }

        let len = trimmed.len();
        if !(MEMORY_SPACE_ID_MIN_LEN..=MEMORY_SPACE_ID_MAX_LEN).contains(&len) {
            return Err(MemorySpaceIdError::InvalidLength {
                value: trimmed.to_owned(),
                length: len,
            });
        }

        if !trimmed
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '-' | '_'))
        {
            return Err(MemorySpaceIdError::InvalidCharacters {
                value: trimmed.to_owned(),
            });
        }

        Ok(Self(Arc::from(trimmed)))
    }

    /// Borrow the identifier as `str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for MemorySpaceId {
    fn default() -> Self {
        // Safe unwrap: DEFAULT_STR is guaranteed valid.
        Self::validate_and_intern(Self::DEFAULT_STR).unwrap_or_else(|_| {
            // SAFETY: DEFAULT_STR ("default") is hardcoded and always valid
            unreachable!("DEFAULT_STR must be valid")
        })
    }
}

impl AsRef<str> for MemorySpaceId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Display for MemorySpaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self.as_str(), f)
    }
}

impl FromStr for MemorySpaceId {
    type Err = MemorySpaceIdError;

    fn from_str(s: &str) -> StdResult<Self, Self::Err> {
        Self::new(s)
    }
}

impl TryFrom<&str> for MemorySpaceId {
    type Error = MemorySpaceIdError;

    fn try_from(value: &str) -> StdResult<Self, Self::Error> {
        Self::new(value)
    }
}

impl Serialize for MemorySpaceId {
    fn serialize<S>(&self, serializer: S) -> StdResult<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for MemorySpaceId {
    fn deserialize<D>(deserializer: D) -> StdResult<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
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
                    summary: format!("Invalid activation level: {value}"),
                    context: expected = "Activation between 0.0 and 1.0",
                             actual = format!("{value}"),
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
                    summary: format!("Validation failed: {reason}"),
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
    pub fn node_not_found(
        id: impl Into<String>,
        similar: impl IntoIterator<Item = String>,
    ) -> Self {
        let id = id.into();
        let suggestions: Vec<String> = similar.into_iter().collect();
        let similar = if suggestions.is_empty() {
            String::new()
        } else {
            format!(", or did you mean: {}?", suggestions.join(", "))
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod memory_space_id_tests {
    use super::{MemorySpaceId, MemorySpaceIdError};

    #[test]
    fn accepts_valid_identifier() {
        let id = MemorySpaceId::try_from("tenant_alpha").unwrap();
        assert_eq!(id.as_str(), "tenant_alpha");
    }

    #[test]
    fn rejects_empty_identifier() {
        let err = MemorySpaceId::try_from("").unwrap_err();
        assert!(matches!(err, MemorySpaceIdError::Empty));
    }

    #[test]
    fn rejects_uppercase_characters() {
        let err = MemorySpaceId::try_from("Tenant").unwrap_err();
        assert!(matches!(err, MemorySpaceIdError::InvalidCharacters { .. }));
    }

    #[test]
    fn rejects_short_identifier() {
        let err = MemorySpaceId::try_from("ab").unwrap_err();
        assert!(matches!(
            err,
            MemorySpaceIdError::InvalidLength { length: 2, .. }
        ));
    }

    #[test]
    fn serde_round_trips() {
        let original = MemorySpaceId::try_from("tenant_123").unwrap();
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: MemorySpaceId = serde_json::from_str(&serialized).unwrap();
        assert_eq!(original, deserialized);
    }
}
