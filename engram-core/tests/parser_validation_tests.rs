//! Integration tests for parser validation.
//!
//! This test suite verifies that the parser properly validates query parameters
//! BEFORE constructing AST nodes. These tests ensure that invalid values are
//! rejected at parse time with actionable error messages.

#![allow(clippy::uninlined_format_args)]

use engram_core::query::parser::Parser;
use engram_core::query::parser::error::ErrorKind;

// ============================================================================
// Confidence Value Validation Tests
// ============================================================================

#[test]
fn test_confidence_out_of_range_high() {
    let query = "RECALL episode WHERE confidence > 1.5";
    let result = Parser::parse(query);

    assert!(result.is_err(), "Query should fail validation");
    let err = result.unwrap_err();
    assert!(
        matches!(err.kind, ErrorKind::ValidationError { .. }),
        "Expected ValidationError, got {:?}",
        err.kind
    );
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("confidence"),
        "Error should mention confidence"
    );
    assert!(
        msg.contains("1.0") || msg.contains("maximum"),
        "Error should mention max value"
    );
}

#[test]
fn test_confidence_out_of_range_low() {
    let query = "RECALL episode WHERE confidence > -0.5";
    let result = Parser::parse(query);

    assert!(result.is_err(), "Query should fail validation");
    let err = result.unwrap_err();
    // Negative numbers may be caught during tokenization OR validation
    // Both are acceptable as long as the error is caught
    let is_validation_or_syntax = matches!(
        err.kind,
        ErrorKind::ValidationError { .. } | ErrorKind::InvalidSyntax { .. }
    );
    assert!(
        is_validation_or_syntax,
        "Expected ValidationError or InvalidSyntax, got {:?}",
        err.kind
    );
}

// ============================================================================
// Decay Rate Validation Tests
// ============================================================================

#[test]
fn test_decay_rate_negative() {
    let query = "SPREAD FROM node DECAY -0.5";
    let result = Parser::parse(query);

    assert!(result.is_err(), "Query should fail validation");
    let err = result.unwrap_err();
    // Negative numbers may be caught during tokenization OR validation
    let is_validation_or_syntax = matches!(
        err.kind,
        ErrorKind::ValidationError { .. } | ErrorKind::InvalidSyntax { .. }
    );
    assert!(
        is_validation_or_syntax,
        "Expected ValidationError or InvalidSyntax, got {:?}",
        err.kind
    );
}

#[test]
fn test_decay_rate_too_high() {
    let query = "SPREAD FROM node DECAY 1.5";
    let result = Parser::parse(query);

    assert!(result.is_err(), "Query should fail validation");
    let err = result.unwrap_err();
    assert!(
        matches!(err.kind, ErrorKind::ValidationError { .. }),
        "Expected ValidationError"
    );
    let msg = err.to_string().to_lowercase();
    assert!(msg.contains("decay"), "Error should mention decay");
}

// ============================================================================
// Threshold Validation Tests
// ============================================================================

#[test]
fn test_threshold_negative() {
    let query = "SPREAD FROM node THRESHOLD -0.1";
    let result = Parser::parse(query);

    assert!(result.is_err(), "Query should fail validation");
    let err = result.unwrap_err();
    // Negative numbers may be caught during tokenization OR validation
    let is_validation_or_syntax = matches!(
        err.kind,
        ErrorKind::ValidationError { .. } | ErrorKind::InvalidSyntax { .. }
    );
    assert!(
        is_validation_or_syntax,
        "Expected ValidationError or InvalidSyntax, got {:?}",
        err.kind
    );
}

#[test]
fn test_threshold_too_high() {
    let query = "SPREAD FROM node THRESHOLD 1.5";
    let result = Parser::parse(query);

    assert!(result.is_err(), "Query should fail validation");
    let err = result.unwrap_err();
    assert!(
        matches!(err.kind, ErrorKind::ValidationError { .. }),
        "Expected ValidationError"
    );
    let msg = err.to_string().to_lowercase();
    assert!(msg.contains("threshold"), "Error should mention threshold");
}

// ============================================================================
// Embedding Threshold Validation Tests
// ============================================================================

#[test]
fn test_embedding_threshold_too_high() {
    // Create 768-dimensional embedding
    let mut values = Vec::new();
    for _i in 0..768 {
        values.push("0.1".to_string());
    }
    let query = format!("RECALL [{}] THRESHOLD 1.5", values.join(", "));
    let result = Parser::parse(&query);

    assert!(result.is_err(), "Query should fail validation");
    let err = result.unwrap_err();
    assert!(
        matches!(err.kind, ErrorKind::ValidationError { .. }),
        "Expected ValidationError"
    );
    let msg = err.to_string().to_lowercase();
    assert!(msg.contains("threshold"), "Error should mention threshold");
}

// ============================================================================
// Max Hops Validation Tests
// ============================================================================

#[test]
fn test_max_hops_zero() {
    let query = "SPREAD FROM node MAX_HOPS 0";
    let result = Parser::parse(query);

    assert!(result.is_err(), "Query should fail validation");
    let err = result.unwrap_err();
    assert!(
        matches!(err.kind, ErrorKind::ValidationError { .. }),
        "Expected ValidationError"
    );
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("max_hops") || msg.contains("hop"),
        "Error should mention MAX_HOPS"
    );
}

#[test]
fn test_max_hops_too_large() {
    let query = "SPREAD FROM node MAX_HOPS 1000";
    let result = Parser::parse(query);

    assert!(result.is_err(), "Query should fail validation");
    let err = result.unwrap_err();
    assert!(
        matches!(err.kind, ErrorKind::ValidationError { .. }),
        "Expected ValidationError"
    );
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("max_hops") || msg.contains("hop"),
        "Error should mention MAX_HOPS"
    );
}

// ============================================================================
// Novelty Validation Tests
// ============================================================================

#[test]
fn test_novelty_negative() {
    let query = "IMAGINE episode BASED ON seed NOVELTY -0.3";
    let result = Parser::parse(query);

    assert!(result.is_err(), "Query should fail validation");
    let err = result.unwrap_err();
    // Negative numbers may be caught during tokenization OR validation
    let is_validation_or_syntax = matches!(
        err.kind,
        ErrorKind::ValidationError { .. } | ErrorKind::InvalidSyntax { .. }
    );
    assert!(
        is_validation_or_syntax,
        "Expected ValidationError or InvalidSyntax, got {:?}",
        err.kind
    );
}

#[test]
fn test_novelty_too_high() {
    let query = "IMAGINE episode BASED ON seed NOVELTY 1.5";
    let result = Parser::parse(query);

    assert!(result.is_err(), "Query should fail validation");
    let err = result.unwrap_err();
    assert!(
        matches!(err.kind, ErrorKind::ValidationError { .. }),
        "Expected ValidationError"
    );
    let msg = err.to_string().to_lowercase();
    assert!(msg.contains("novelty"), "Error should mention novelty");
}

// ============================================================================
// Identifier Length Validation Tests
// ============================================================================

#[test]
fn test_identifier_too_long() {
    let long_id = "a".repeat(10000);
    let query = format!("SPREAD FROM {}", long_id);
    let result = Parser::parse(&query);

    assert!(result.is_err(), "Query should fail validation");
    let err = result.unwrap_err();
    assert!(
        matches!(err.kind, ErrorKind::ValidationError { .. }),
        "Expected ValidationError"
    );
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("identifier") || msg.contains("long"),
        "Error should mention identifier length"
    );
}

// ============================================================================
// Valid Boundary Cases (Should Pass)
// ============================================================================

#[test]
fn test_confidence_valid_boundaries() {
    let queries = vec![
        "RECALL episode WHERE confidence > 0.0",
        "RECALL episode WHERE confidence > 1.0",
        "RECALL episode WHERE confidence > 0.5",
    ];

    for query in queries {
        let result = Parser::parse(query);
        assert!(result.is_ok(), "Query should parse successfully: {}", query);
    }
}

#[test]
fn test_decay_valid_boundaries() {
    let queries = vec![
        "SPREAD FROM node DECAY 0.0",
        "SPREAD FROM node DECAY 1.0",
        "SPREAD FROM node DECAY 0.15",
    ];

    for query in queries {
        let result = Parser::parse(query);
        assert!(result.is_ok(), "Query should parse successfully: {}", query);
    }
}

#[test]
fn test_max_hops_valid_boundaries() {
    let queries = vec![
        "SPREAD FROM node MAX_HOPS 1",
        "SPREAD FROM node MAX_HOPS 100",
        "SPREAD FROM node MAX_HOPS 50",
    ];

    for query in queries {
        let result = Parser::parse(query);
        assert!(result.is_ok(), "Query should parse successfully: {}", query);
    }
}

// ============================================================================
// Error Message Quality Tests
// ============================================================================

#[test]
fn test_validation_errors_have_suggestions() {
    let invalid_queries = vec![
        "RECALL episode WHERE confidence > 1.5",
        "SPREAD FROM node DECAY -0.5",
        "SPREAD FROM node MAX_HOPS 0",
        "IMAGINE episode NOVELTY 1.5",
    ];

    for query in invalid_queries {
        let result = Parser::parse(query);
        assert!(result.is_err(), "Query should fail: {}", query);

        let err = result.unwrap_err();
        assert!(
            !err.suggestion.is_empty(),
            "Error should have suggestion for: {}",
            query
        );
        assert!(
            !err.example.is_empty(),
            "Error should have example for: {}",
            query
        );
    }
}

#[test]
fn test_validation_error_contains_value() {
    let query = "RECALL episode WHERE confidence > 1.5";
    let result = Parser::parse(query);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();

    // Error message should mention the invalid value
    assert!(
        msg.contains("1.5") || msg.contains("confidence"),
        "Error should mention the invalid value or field"
    );
}
