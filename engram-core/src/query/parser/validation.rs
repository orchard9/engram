//! Parser validation functions for query language semantic constraints.
//!
//! This module provides validation functions that catch invalid query parameters
//! BEFORE constructing AST nodes. This ensures that values like confidence scores,
//! decay rates, thresholds, and hop counts are within valid ranges.
//!
//! ## Design Principle
//!
//! The parser should reject invalid queries at parse time, not execution time.
//! These validation functions catch semantic errors that would otherwise be
//! silently clamped by `Confidence::exact()` or cause runtime failures.
//!
//! ## Validation Rules
//!
//! - **Confidence values**: Must be in [0.0, 1.0] (no silent clamping)
//! - **Decay rates**: Must be in [0.0, 1.0] (exponential decay formula)
//! - **Thresholds**: Must be in [0.0, 1.0] (activation/similarity cutoffs)
//! - **Max hops**: Must be in [1, 100] (reasonable graph traversal limits)
//! - **Embedding dimension**: Must be exactly 768 (system configuration)
//! - **Identifier length**: Must be < 1000 chars (prevent DoS attacks)
//!
//! ## Integration with Parser
//!
//! Validation functions are called BEFORE constructing Confidence objects or
//! AST nodes. This ensures that `Confidence::exact()` only receives valid inputs.

// ParseError is intentionally large (~128 bytes) to provide rich error messages.
// This is acceptable because error paths are cold (not performance critical).
// See parser.rs for detailed justification.
#![allow(clippy::result_large_err)]

use super::error::ParseError;
use super::token::Position;

// ============================================================================
// Confidence Value Validation
// ============================================================================

/// Validate confidence value is in valid range [0.0, 1.0].
///
/// This validation MUST be called before passing values to `Confidence::exact()`,
/// which silently clamps out-of-range values. We want to reject invalid queries
/// at parse time with actionable error messages.
///
/// # Errors
///
/// Returns `ParseError::ValidationError` if value is not in [0.0, 1.0].
///
/// # Examples
///
/// ```rust
/// # use engram_core::query::parser::validation::validate_confidence_value;
/// # use engram_core::query::parser::token::Position;
/// assert!(validate_confidence_value(0.7, Position::start()).is_ok());
/// assert!(validate_confidence_value(1.5, Position::start()).is_err()); // Too high
/// assert!(validate_confidence_value(-0.5, Position::start()).is_err()); // Negative
/// ```
pub fn validate_confidence_value(value: f32, position: Position) -> Result<(), ParseError> {
    if !value.is_finite() {
        return Err(ParseError::validation_error(
            format!("Confidence value {value} is not a valid number (NaN or infinity)"),
            position,
            "Use a finite number between 0.0 and 1.0",
            "WHERE confidence > 0.7",
        ));
    }

    if value < 0.0 {
        return Err(ParseError::validation_error(
            format!("Confidence value {value} is negative (must be >= 0.0)"),
            position,
            "Use confidence value between 0.0 and 1.0",
            "WHERE confidence > 0.7",
        ));
    }

    if value > 1.0 {
        return Err(ParseError::validation_error(
            format!("Confidence value {value} exceeds maximum (must be <= 1.0)"),
            position,
            "Use confidence value between 0.0 and 1.0",
            "WHERE confidence > 0.7",
        ));
    }

    Ok(())
}

// ============================================================================
// Decay Rate Validation
// ============================================================================

/// Validate decay rate is in valid range [0.0, 1.0].
///
/// Decay rates are used in exponential decay formulas: `activation * exp(-decay_rate * hops)`.
/// Values outside [0, 1] produce non-physical or numerically unstable results.
///
/// # Errors
///
/// Returns `ParseError::ValidationError` if value is not in [0.0, 1.0].
///
/// # Examples
///
/// ```rust
/// # use engram_core::query::parser::validation::validate_decay_rate;
/// # use engram_core::query::parser::token::Position;
/// assert!(validate_decay_rate(0.15, Position::start()).is_ok());
/// assert!(validate_decay_rate(1.5, Position::start()).is_err()); // Too high
/// assert!(validate_decay_rate(-0.5, Position::start()).is_err()); // Negative
/// ```
pub fn validate_decay_rate(value: f32, position: Position) -> Result<(), ParseError> {
    if !value.is_finite() {
        return Err(ParseError::validation_error(
            format!("Decay rate {value} is not a valid number (NaN or infinity)"),
            position,
            "Use a finite number between 0.0 and 1.0",
            "SPREAD FROM node DECAY 0.15",
        ));
    }

    if value < 0.0 {
        return Err(ParseError::validation_error(
            format!("Decay rate {value} is negative (must be >= 0.0)"),
            position,
            "Use decay rate between 0.0 and 1.0",
            "SPREAD FROM node DECAY 0.15",
        ));
    }

    if value > 1.0 {
        return Err(ParseError::validation_error(
            format!("Decay rate {value} exceeds maximum (must be <= 1.0)"),
            position,
            "Use decay rate between 0.0 and 1.0",
            "SPREAD FROM node DECAY 0.15",
        ));
    }

    Ok(())
}

// ============================================================================
// Threshold Validation
// ============================================================================

/// Validate threshold is in valid range [0.0, 1.0].
///
/// Thresholds are used for activation cutoffs, similarity matching, and
/// filtering. Values outside [0, 1] don't make sense for probabilistic systems.
///
/// # Errors
///
/// Returns `ParseError::ValidationError` if value is not in [0.0, 1.0].
///
/// # Examples
///
/// ```rust
/// # use engram_core::query::parser::validation::validate_threshold;
/// # use engram_core::query::parser::token::Position;
/// assert!(validate_threshold(0.8, Position::start()).is_ok());
/// assert!(validate_threshold(1.5, Position::start()).is_err()); // Too high
/// assert!(validate_threshold(-0.1, Position::start()).is_err()); // Negative
/// ```
pub fn validate_threshold(value: f32, position: Position) -> Result<(), ParseError> {
    if !value.is_finite() {
        return Err(ParseError::validation_error(
            format!("Threshold {value} is not a valid number (NaN or infinity)"),
            position,
            "Use a finite number between 0.0 and 1.0",
            "SPREAD FROM node THRESHOLD 0.1",
        ));
    }

    if value < 0.0 {
        return Err(ParseError::validation_error(
            format!("Threshold {value} is negative (must be >= 0.0)"),
            position,
            "Use threshold between 0.0 and 1.0",
            "SPREAD FROM node THRESHOLD 0.1",
        ));
    }

    if value > 1.0 {
        return Err(ParseError::validation_error(
            format!("Threshold {value} exceeds maximum (must be <= 1.0)"),
            position,
            "Use threshold between 0.0 and 1.0",
            "SPREAD FROM node THRESHOLD 0.1",
        ));
    }

    Ok(())
}

// ============================================================================
// Novelty Validation
// ============================================================================

/// Validate novelty level is in valid range [0.0, 1.0].
///
/// Novelty controls the balance between familiar patterns (0.0) and novel
/// combinations (1.0) in IMAGINE queries. Values outside [0, 1] are meaningless.
///
/// # Errors
///
/// Returns `ParseError::ValidationError` if value is not in [0.0, 1.0].
///
/// # Examples
///
/// ```rust
/// # use engram_core::query::parser::validation::validate_novelty;
/// # use engram_core::query::parser::token::Position;
/// assert!(validate_novelty(0.3, Position::start()).is_ok());
/// assert!(validate_novelty(1.5, Position::start()).is_err()); // Too high
/// assert!(validate_novelty(-0.3, Position::start()).is_err()); // Negative
/// ```
pub fn validate_novelty(value: f32, position: Position) -> Result<(), ParseError> {
    if !value.is_finite() {
        return Err(ParseError::validation_error(
            format!("Novelty level {value} is not a valid number (NaN or infinity)"),
            position,
            "Use a finite number between 0.0 and 1.0",
            "IMAGINE episode NOVELTY 0.3",
        ));
    }

    if value < 0.0 {
        return Err(ParseError::validation_error(
            format!("Novelty level {value} is negative (must be >= 0.0)"),
            position,
            "Use novelty between 0.0 and 1.0",
            "IMAGINE episode NOVELTY 0.3",
        ));
    }

    if value > 1.0 {
        return Err(ParseError::validation_error(
            format!("Novelty level {value} exceeds maximum (must be <= 1.0)"),
            position,
            "Use novelty between 0.0 and 1.0",
            "IMAGINE episode NOVELTY 0.3",
        ));
    }

    Ok(())
}

// ============================================================================
// Max Hops Validation
// ============================================================================

/// Validate max_hops is in valid range [1, 100].
///
/// Max hops controls graph traversal depth. Zero hops is meaningless (no traversal),
/// and values > 100 risk exponential explosion in large graphs.
///
/// # Errors
///
/// Returns `ParseError::ValidationError` if value is not in [1, 100].
///
/// # Examples
///
/// ```rust
/// # use engram_core::query::parser::validation::validate_max_hops;
/// # use engram_core::query::parser::token::Position;
/// assert!(validate_max_hops(5, Position::start()).is_ok());
/// assert!(validate_max_hops(0, Position::start()).is_err()); // Zero
/// assert!(validate_max_hops(1000, Position::start()).is_err()); // Too high
/// ```
pub fn validate_max_hops(value: u16, position: Position) -> Result<(), ParseError> {
    if value == 0 {
        return Err(ParseError::validation_error(
            "MAX_HOPS value 0 is invalid (must be at least 1)",
            position,
            "Use MAX_HOPS value between 1 and 100",
            "SPREAD FROM node MAX_HOPS 5",
        ));
    }

    if value > 100 {
        return Err(ParseError::validation_error(
            format!("MAX_HOPS value {value} exceeds maximum (must be <= 100)"),
            position,
            "Use MAX_HOPS value between 1 and 100",
            "SPREAD FROM node MAX_HOPS 10",
        ));
    }

    Ok(())
}

// ============================================================================
// Embedding Dimension Validation
// ============================================================================

/// Validate embedding dimension matches system configuration (768).
///
/// Engram uses 768-dimensional embeddings (matching text-embedding-ada-002).
/// All embeddings must match this dimension for consistency.
///
/// # Errors
///
/// Returns `ParseError::ValidationError` if dimension doesn't match EMBEDDING_DIM.
///
/// # Examples
///
/// ```rust
/// # use engram_core::query::parser::validation::validate_embedding_dimension;
/// # use engram_core::query::parser::token::Position;
/// assert!(validate_embedding_dimension(768, Position::start()).is_ok());
/// assert!(validate_embedding_dimension(3, Position::start()).is_err()); // Wrong dimension
/// ```
pub fn validate_embedding_dimension(actual: usize, position: Position) -> Result<(), ParseError> {
    const EXPECTED: usize = crate::EMBEDDING_DIM;

    if actual != EXPECTED {
        return Err(ParseError::validation_error(
            format!("Invalid embedding dimension: expected {EXPECTED}, got {actual}"),
            position,
            format!("Use {EXPECTED}-dimensional embedding vector (system configuration)"),
            format!("RECALL [{:.1}; {}] THRESHOLD 0.8", 0.1, EXPECTED),
        ));
    }

    Ok(())
}

// ============================================================================
// Identifier Length Validation
// ============================================================================

/// Validate identifier length is reasonable (< 1000 chars).
///
/// Very long identifiers can cause DoS attacks or indicate malformed queries.
/// This limit is generous but prevents abuse.
///
/// # Errors
///
/// Returns `ParseError::ValidationError` if identifier exceeds 1000 characters.
///
/// # Examples
///
/// ```rust
/// # use engram_core::query::parser::validation::validate_identifier_length;
/// # use engram_core::query::parser::token::Position;
/// assert!(validate_identifier_length("episode_123", Position::start()).is_ok());
/// let long_id = "a".repeat(10000);
/// assert!(validate_identifier_length(&long_id, Position::start()).is_err());
/// ```
pub fn validate_identifier_length(identifier: &str, position: Position) -> Result<(), ParseError> {
    const MAX_LENGTH: usize = 1000;

    if identifier.is_empty() {
        return Err(ParseError::validation_error(
            "Empty identifier (must be non-empty)",
            position,
            "Provide a non-empty node identifier",
            "RECALL episode_123",
        ));
    }

    if identifier.len() > MAX_LENGTH {
        return Err(ParseError::validation_error(
            format!(
                "Identifier too long: {} characters (max {MAX_LENGTH})",
                identifier.len()
            ),
            position,
            format!("Use shorter identifier (max {MAX_LENGTH} characters)"),
            "SPREAD FROM node_123",
        ));
    }

    Ok(())
}

// ============================================================================
// Confidence Interval Validation
// ============================================================================

/// Validate confidence interval bounds are ordered correctly.
///
/// For intervals [lower, upper], we require lower <= upper.
///
/// # Errors
///
/// Returns `ParseError::ValidationError` if lower > upper.
///
/// # Examples
///
/// ```rust
/// # use engram_core::query::parser::validation::validate_confidence_interval;
/// # use engram_core::query::parser::token::Position;
/// assert!(validate_confidence_interval(0.5, 0.9, Position::start()).is_ok());
/// assert!(validate_confidence_interval(0.9, 0.5, Position::start()).is_err()); // Reversed
/// ```
pub fn validate_confidence_interval(
    lower: f32,
    upper: f32,
    position: Position,
) -> Result<(), ParseError> {
    if lower > upper {
        return Err(ParseError::validation_error(
            format!("Invalid confidence interval: lower={lower} > upper={upper}"),
            position,
            "Ensure lower bound <= upper bound",
            "CONFIDENCE BETWEEN 0.5 AND 0.9",
        ));
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)] // Tests are allowed to use unwrap
mod tests {
    use super::*;

    #[test]
    fn test_validate_confidence_value() {
        let pos = Position::start();

        // Valid values
        assert!(validate_confidence_value(0.0, pos).is_ok());
        assert!(validate_confidence_value(0.5, pos).is_ok());
        assert!(validate_confidence_value(1.0, pos).is_ok());

        // Invalid values
        assert!(validate_confidence_value(-0.1, pos).is_err());
        assert!(validate_confidence_value(1.1, pos).is_err());
        assert!(validate_confidence_value(-0.5, pos).is_err());
        assert!(validate_confidence_value(1.5, pos).is_err());
        assert!(validate_confidence_value(f32::NAN, pos).is_err());
        assert!(validate_confidence_value(f32::INFINITY, pos).is_err());
    }

    #[test]
    fn test_validate_decay_rate() {
        let pos = Position::start();

        // Valid values
        assert!(validate_decay_rate(0.0, pos).is_ok());
        assert!(validate_decay_rate(0.15, pos).is_ok());
        assert!(validate_decay_rate(1.0, pos).is_ok());

        // Invalid values
        assert!(validate_decay_rate(-0.5, pos).is_err());
        assert!(validate_decay_rate(1.5, pos).is_err());
        assert!(validate_decay_rate(f32::NAN, pos).is_err());
    }

    #[test]
    fn test_validate_threshold() {
        let pos = Position::start();

        // Valid values
        assert!(validate_threshold(0.0, pos).is_ok());
        assert!(validate_threshold(0.8, pos).is_ok());
        assert!(validate_threshold(1.0, pos).is_ok());

        // Invalid values
        assert!(validate_threshold(-0.1, pos).is_err());
        assert!(validate_threshold(1.5, pos).is_err());
    }

    #[test]
    fn test_validate_novelty() {
        let pos = Position::start();

        // Valid values
        assert!(validate_novelty(0.0, pos).is_ok());
        assert!(validate_novelty(0.3, pos).is_ok());
        assert!(validate_novelty(1.0, pos).is_ok());

        // Invalid values
        assert!(validate_novelty(-0.3, pos).is_err());
        assert!(validate_novelty(1.5, pos).is_err());
    }

    #[test]
    fn test_validate_max_hops() {
        let pos = Position::start();

        // Valid values
        assert!(validate_max_hops(1, pos).is_ok());
        assert!(validate_max_hops(5, pos).is_ok());
        assert!(validate_max_hops(100, pos).is_ok());

        // Invalid values
        assert!(validate_max_hops(0, pos).is_err());
        assert!(validate_max_hops(101, pos).is_err());
        assert!(validate_max_hops(1000, pos).is_err());
    }

    #[test]
    fn test_validate_embedding_dimension() {
        let pos = Position::start();

        // Valid dimension
        assert!(validate_embedding_dimension(768, pos).is_ok());

        // Invalid dimensions
        assert!(validate_embedding_dimension(3, pos).is_err());
        assert!(validate_embedding_dimension(512, pos).is_err());
        assert!(validate_embedding_dimension(1024, pos).is_err());
    }

    #[test]
    fn test_validate_identifier_length() {
        let pos = Position::start();

        // Valid identifiers
        assert!(validate_identifier_length("node", pos).is_ok());
        assert!(validate_identifier_length("episode_123", pos).is_ok());
        assert!(validate_identifier_length(&"a".repeat(999), pos).is_ok());

        // Invalid identifiers
        assert!(validate_identifier_length("", pos).is_err());
        assert!(validate_identifier_length(&"a".repeat(1001), pos).is_err());
        assert!(validate_identifier_length(&"a".repeat(10000), pos).is_err());
    }

    #[test]
    fn test_validate_confidence_interval() {
        let pos = Position::start();

        // Valid intervals
        assert!(validate_confidence_interval(0.5, 0.9, pos).is_ok());
        assert!(validate_confidence_interval(0.0, 1.0, pos).is_ok());
        assert!(validate_confidence_interval(0.5, 0.5, pos).is_ok()); // Equal bounds OK

        // Invalid intervals
        assert!(validate_confidence_interval(0.9, 0.5, pos).is_err());
        assert!(validate_confidence_interval(0.8, 0.6, pos).is_err());
    }

    #[test]
    fn test_error_messages_contain_suggestions() {
        let pos = Position::start();

        let err = validate_confidence_value(1.5, pos).unwrap_err();
        assert!(!err.suggestion.is_empty());
        assert!(!err.example.is_empty());
        assert!(err.suggestion.contains("0.0") || err.suggestion.contains("1.0"));

        let err = validate_max_hops(0, pos).unwrap_err();
        assert!(!err.suggestion.is_empty());
        assert!(err.suggestion.contains('1'));

        let err = validate_decay_rate(-0.5, pos).unwrap_err();
        assert!(err.suggestion.contains("0.0") || err.suggestion.contains("1.0"));
    }
}
