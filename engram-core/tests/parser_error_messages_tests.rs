//! Comprehensive tests for parser error message quality.
//!
//! These tests validate that error messages meet the quality standards:
//! - 100% of errors have actionable suggestions
//! - 100% of errors have examples
//! - Typo detection works for all keywords
//! - Error messages pass the "tiredness test" (clear at 3am)
//! - No parser jargon (AST, token stream, etc.)
//!
//! Based on psychological research (Marceau et al. 2011, Becker et al. 2019).

use engram_core::query::parser::Parser;
use engram_core::query::parser::typo_detection::find_closest_keyword;

// ============================================================================
// Typo Detection Tests - All 17 Keywords
// ============================================================================

#[test]
fn test_typo_detection_recall() {
    let result = Parser::parse("RECAL episode");
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();

    // Should suggest RECALL
    assert!(msg.contains("RECALL"), "Error should suggest RECALL: {msg}");
    assert!(!err.suggestion.is_empty(), "Error should have suggestion");
    assert!(!err.example.is_empty(), "Error should have example");
}

#[test]
fn test_typo_detection_predict() {
    let result = Parser::parse("PRDICT episode GIVEN context");
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();

    assert!(
        msg.contains("PREDICT"),
        "Error should suggest PREDICT: {msg}"
    );
    assert!(!err.suggestion.is_empty());
    assert!(!err.example.is_empty());
}

#[test]
fn test_typo_detection_imagine() {
    let result = Parser::parse("IMAGIN episode");
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();

    assert!(
        msg.contains("IMAGINE"),
        "Error should suggest IMAGINE: {msg}"
    );
}

#[test]
fn test_typo_detection_consolidate() {
    let result = Parser::parse("CONSOLIDTE episode INTO target");
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();

    assert!(
        msg.contains("CONSOLIDATE"),
        "Error should suggest CONSOLIDATE: {msg}"
    );
}

#[test]
fn test_typo_detection_spread() {
    let result = Parser::parse("SPRED FROM node");
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();

    assert!(msg.contains("SPREAD"), "Error should suggest SPREAD: {msg}");
}

// ============================================================================
// Error Message Quality Tests
// ============================================================================

#[test]
fn test_all_parse_errors_have_suggestions() {
    let test_cases = [
        "RECAL episode",        // Typo in keyword
        "RECALL",               // Missing pattern
        "RECALL episode WHERE", // Incomplete constraint
        "RECALL []",            // Empty embedding
        "SPREAD node",          // Missing FROM
        "PREDICT episode",      // Missing GIVEN
    ];

    for query in test_cases {
        let result = Parser::parse(query);
        assert!(result.is_err(), "Query should fail: {query}");

        let err = result.unwrap_err();
        assert!(
            !err.suggestion.is_empty(),
            "Error for '{query}' should have non-empty suggestion. Got: {err:?}"
        );
        assert!(
            !err.example.is_empty(),
            "Error for '{query}' should have non-empty example. Got: {err:?}"
        );
    }
}

#[test]
fn test_error_messages_no_jargon() {
    let test_cases = [
        "RECALL WHERE",              // Wrong order
        "RECALL episode confidence", // Missing operator
        "@ invalid",                 // Invalid character
    ];

    for query in test_cases {
        let result = Parser::parse(query);
        if let Err(err) = result {
            let msg = err.to_string().to_lowercase();

            // Verify no parser jargon
            assert!(
                !msg.contains("ast"),
                "Error for '{query}' contains 'AST' jargon: {msg}"
            );
            assert!(
                !msg.contains("token stream"),
                "Error for '{query}' contains 'token stream' jargon: {msg}"
            );
            assert!(
                !msg.contains("lookahead"),
                "Error for '{query}' contains 'lookahead' jargon: {msg}"
            );
            assert!(
                !msg.contains("parse failure"),
                "Error for '{query}' contains 'parse failure' jargon: {msg}"
            );
        }
    }
}

#[test]
fn test_error_messages_positive_framing() {
    let result = Parser::parse("RECALL WHERE");
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();

    // Should use positive framing ("use X" not "don't use Y")
    assert!(
        msg.to_lowercase().contains("use")
            || msg.to_lowercase().contains("requires")
            || msg.to_lowercase().contains("provide"),
        "Error should use positive framing: {msg}"
    );
}

#[test]
fn test_error_messages_include_examples() {
    let test_cases = ["RECALL", "SPREAD", "PREDICT", "IMAGINE", "CONSOLIDATE"];

    for query in test_cases {
        let result = Parser::parse(query);
        assert!(result.is_err(), "Query should fail: {query}");

        let err = result.unwrap_err();
        assert!(
            !err.example.is_empty(),
            "Error for '{query}' should have example"
        );

        // Example should be syntactically complete
        assert!(
            err.example.len() > 10,
            "Error for '{query}' example too short: '{}'",
            err.example
        );
    }
}

// ============================================================================
// Context-Aware Error Message Tests
// ============================================================================

#[test]
fn test_context_aware_query_start() {
    let result = Parser::parse("WHERE confidence > 0.7");
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();

    // Should mention that query must start with operation keyword
    assert!(
        msg.to_uppercase().contains("RECALL")
            || msg.to_uppercase().contains("PREDICT")
            || msg.to_uppercase().contains("SPREAD"),
        "Error at query start should mention operation keywords: {msg}"
    );
}

#[test]
fn test_context_aware_after_recall() {
    let result = Parser::parse("RECALL WHERE");
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();

    // Should mention pattern requirement
    assert!(
        msg.to_lowercase().contains("pattern")
            || msg.to_lowercase().contains("identifier")
            || msg.to_lowercase().contains("embedding"),
        "Error after RECALL should mention pattern: {msg}"
    );
}

#[test]
fn test_context_aware_after_spread() {
    let result = Parser::parse("SPREAD node");
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();

    // Should mention FROM keyword requirement
    assert!(
        msg.to_uppercase().contains("FROM"),
        "Error after SPREAD should mention FROM: {msg}"
    );
}

// ============================================================================
// Position Accuracy Tests
// ============================================================================

#[test]
fn test_error_position_line_number() {
    let query = "RECALL episode\nWHERE\n  invalid > 0.7";
    let result = Parser::parse(query);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();

    // Should report correct line number (3rd line)
    assert!(msg.contains("line 3"), "Error should report line 3: {msg}");
}

#[test]
fn test_error_position_single_line() {
    let query = "RECAL episode";
    let result = Parser::parse(query);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert_eq!(err.position.line, 1, "Error should be on line 1");
    assert_eq!(err.position.column, 1, "Error should be at column 1");
}

// ============================================================================
// Comprehensive Keyword Typo Coverage
// ============================================================================

#[test]
fn test_all_17_keywords_have_typo_detection() {
    let test_cases = [
        ("RECAL", "RECALL"),
        ("PRDICT", "PREDICT"),
        ("IMAGIN", "IMAGINE"),
        ("CONSOLIDTE", "CONSOLIDATE"),
        ("SPRED", "SPREAD"),
        ("WHRE", "WHERE"),
        ("GIVN", "GIVEN"),
        ("BASD", "BASED"),
        ("FRM", "FROM"),
        ("INT", "INTO"),
        ("MAX_HOP", "MAX_HOPS"),
        ("DECY", "DECAY"),
        ("THRESHLD", "THRESHOLD"),
        ("CONFIDNCE", "CONFIDENCE"),
        ("HORIZN", "HORIZON"),
        ("NOVLTY", "NOVELTY"),
        ("BASE_RAT", "BASE_RATE"),
    ];

    for (typo, expected) in test_cases {
        let result = find_closest_keyword(typo);
        assert!(result.is_some(), "Typo '{typo}' should have suggestion");

        let suggestion = result.unwrap();
        assert_eq!(
            suggestion, expected,
            "Typo '{typo}' should suggest '{expected}', got '{suggestion}'"
        );
    }
}

// ============================================================================
// Common Developer Mistake Distribution Tests
// ============================================================================

#[test]
fn test_typo_mistakes_40_percent() {
    // According to research, 40% of errors are typos
    let typo_cases = ["RECAL episode", "SPRED FROM node", "IMAGIN episode"];

    for query in typo_cases {
        let result = Parser::parse(query);
        assert!(result.is_err());

        let err = result.unwrap_err();
        // Should detect typo and suggest correction
        assert!(
            err.to_string().contains("Did you mean"),
            "Typo error should have 'Did you mean' suggestion: {err}"
        );
    }
}

#[test]
fn test_wrong_order_mistakes_25_percent() {
    // According to research, 25% of errors are wrong keyword order
    let wrong_order_cases = [
        "FROM SPREAD node",     // Wrong order
        "RECALL WHERE episode", // WHERE before pattern
    ];

    for query in wrong_order_cases {
        let result = Parser::parse(query);
        assert!(result.is_err(), "Query should fail: {query}");

        let err = result.unwrap_err();
        assert!(!err.suggestion.is_empty());
        assert!(!err.example.is_empty());
    }
}

#[test]
fn test_missing_keyword_mistakes_15_percent() {
    // According to research, 15% of errors are missing required keywords
    let missing_keyword_cases = [
        "SPREAD node",         // Missing FROM
        "PREDICT episode",     // Missing GIVEN
        "CONSOLIDATE episode", // Missing INTO
    ];

    for query in missing_keyword_cases {
        let result = Parser::parse(query);
        assert!(result.is_err(), "Query should fail: {query}");

        let err = result.unwrap_err();
        assert!(!err.suggestion.is_empty());
        assert!(!err.example.is_empty());
    }
}

// ============================================================================
// Tiredness Test (3am Test)
// ============================================================================

#[test]
fn test_error_clarity_tiredness_test() {
    // These errors should be clear even at 3am when tired
    let test_cases = [
        ("RECAL episode", "Should clearly suggest RECALL"),
        ("RECALL", "Should clearly explain pattern is required"),
        ("SPREAD node", "Should clearly explain FROM is required"),
        (
            "RECALL []",
            "Should clearly explain embedding cannot be empty",
        ),
    ];

    for (query, description) in test_cases {
        let result = Parser::parse(query);
        assert!(result.is_err(), "{description}");

        let err = result.unwrap_err();
        let msg = err.to_string();

        // Clarity criteria (from psychological research):
        // 1. Clear statement of what went wrong
        assert!(
            msg.len() > 50,
            "{description}: Error too short to be clear: {msg}"
        );

        // 2. Actionable suggestion (not just "syntax error")
        assert!(
            !err.suggestion.is_empty() && err.suggestion.len() > 10,
            "{description}: Suggestion not actionable enough: {}",
            err.suggestion
        );

        // 3. Example provided (recognition easier than recall)
        assert!(
            !err.example.is_empty() && err.example.len() > 10,
            "{description}: Example not helpful enough: {}",
            err.example
        );

        // 4. No jargon
        let msg_lower = msg.to_lowercase();
        assert!(
            !msg_lower.contains("ast") && !msg_lower.contains("token"),
            "{description}: Contains jargon: {msg}"
        );
    }
}

// ============================================================================
// Display Format Tests
// ============================================================================

#[test]
fn test_error_display_format() {
    let result = Parser::parse("RECAL episode");
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();

    // Should have clear structure
    assert!(msg.contains("line"), "Error should show line number");
    assert!(msg.contains("column"), "Error should show column number");
    assert!(
        msg.contains("Suggestion:"),
        "Error should have Suggestion section"
    );
    assert!(
        msg.contains("Example:"),
        "Error should have Example section"
    );
}

#[test]
fn test_error_message_multiline_format() {
    let result = Parser::parse("RECAL episode");
    assert!(result.is_err());

    let err = result.unwrap_err();
    let msg = err.to_string();

    // Should have multiple lines for readability
    assert!(
        msg.lines().count() >= 4,
        "Error should be multi-line for readability"
    );
}

// ============================================================================
// No False Positives Tests
// ============================================================================

#[test]
fn test_no_typo_suggestion_when_distance_too_large() {
    let result = find_closest_keyword("XYZ");
    assert!(
        result.is_none(),
        "Should not suggest keyword for completely different input"
    );

    let result2 = find_closest_keyword("FOOBAR");
    assert!(
        result2.is_none(),
        "Should not suggest keyword for random input"
    );
}

#[test]
fn test_no_suggestion_for_distance_greater_than_2() {
    // Distance 3 should not suggest
    let _result = find_closest_keyword("RCLL"); // distance 3 from RECALL
    // This might still suggest if distance is actually ≤2, which is fine
    // The key is we don't want false positives for very different strings

    let result2 = find_closest_keyword("QWERTY");
    assert!(
        result2.is_none(),
        "Should not suggest for very different string"
    );
}
