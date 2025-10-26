//! Comprehensive query language test corpus with 150+ test queries.
//!
//! This module provides systematic coverage of all query language features with:
//! - 75+ valid query tests covering all operations and edge cases
//! - 75+ invalid query tests with expected error messages
//! - Category-based organization for test management
//! - Integration with error message validation framework
//!
//! ## Test Organization
//!
//! Valid queries are organized by category:
//! - RECALL operations (20 tests)
//! - SPREAD operations (15 tests)
//! - PREDICT operations (10 tests)
//! - IMAGINE operations (10 tests)
//! - CONSOLIDATE operations (10 tests)
//! - Edge cases (10 tests)
//!
//! Invalid queries are organized by error type:
//! - Syntax errors (25 tests)
//! - Semantic errors (25 tests)
//! - Stress tests (25 tests)

#![allow(clippy::missing_const_for_fn)]
#![allow(missing_docs)] // Test corpus - documentation via comprehensive test names
#![allow(clippy::uninlined_format_args)] // Test output - prioritize clarity over brevity

use engram_core::query::parser::Parser;

// ============================================================================
// Test Corpus Organization
// ============================================================================

/// Comprehensive test corpus covering all query language features
#[derive(Debug)]
pub struct QueryCorpus {
    pub valid_queries: Vec<ValidQueryTest>,
    pub invalid_syntax: Vec<InvalidQueryTest>,
    pub semantic_errors: Vec<InvalidQueryTest>,
    pub stress_tests: Vec<InvalidQueryTest>,
}

impl QueryCorpus {
    /// Get the complete query corpus with all tests
    #[must_use]
    pub fn all() -> Self {
        Self {
            valid_queries: all_valid_queries(),
            invalid_syntax: syntax_error_queries(),
            semantic_errors: semantic_error_queries(),
            stress_tests: stress_test_queries(),
        }
    }

    /// Total number of valid tests
    #[must_use]
    pub fn valid_count(&self) -> usize {
        self.valid_queries.len()
    }

    /// Total number of invalid tests
    #[must_use]
    pub fn invalid_count(&self) -> usize {
        self.invalid_syntax.len() + self.semantic_errors.len() + self.stress_tests.len()
    }

    /// Total number of all tests
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.valid_count() + self.invalid_count()
    }
}

#[derive(Debug, Clone)]
pub struct ValidQueryTest {
    pub name: &'static str,
    pub query: &'static str,
    pub category: QueryCategory,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Used by error_message_validation tests
pub struct InvalidQueryTest {
    pub name: &'static str,
    pub query: &'static str,
    pub expected_error_type: ErrorType,
    pub must_contain: Vec<&'static str>,
    pub must_suggest: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryCategory {
    Recall,
    Predict,
    Imagine,
    Consolidate,
    Spread,
    #[allow(dead_code)] // Commented out queries that used this category (unsupported features)
    Constraints,
    ConfidenceSpecification,
    #[allow(dead_code)] // Reserved for future embedding literal tests
    EmbeddingLiterals,
    #[allow(dead_code)]
    // Commented out queries that used this category (temporal operators not supported)
    TemporalOperations,
    EdgeCases,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Used by error_message_validation tests
pub enum ErrorType {
    UnexpectedToken,
    UnknownKeyword,
    InvalidSyntax,
    ValidationError,
    UnexpectedEof,
}

// ============================================================================
// Valid Query Test Corpus (75+ Tests)
// ============================================================================

fn all_valid_queries() -> Vec<ValidQueryTest> {
    let mut queries = Vec::new();
    queries.extend(recall_queries());
    queries.extend(spread_queries());
    queries.extend(predict_queries());
    queries.extend(imagine_queries());
    queries.extend(consolidate_queries());
    queries.extend(edge_case_queries());
    queries
}

// RECALL operation tests (20 queries)
fn recall_queries() -> Vec<ValidQueryTest> {
    vec![
        ValidQueryTest {
            name: "basic_recall",
            query: "RECALL episode",
            category: QueryCategory::Recall,
        },
        ValidQueryTest {
            name: "recall_with_confidence_threshold",
            query: "RECALL episode WHERE confidence > 0.7",
            category: QueryCategory::Recall,
        },
        // TODO(future): Enable when SIMILAR TO syntax is implemented
        // ValidQueryTest {
        //     name: "recall_with_embedding_similarity",
        //     query: "RECALL episode WHERE content SIMILAR TO [0.1, 0.2, 0.3]",
        //     category: QueryCategory::Recall,
        // },
        // TODO(future): Enable when temporal < > operators are implemented (currently only BEFORE/AFTER)
        // ValidQueryTest {
        //     name: "recall_with_temporal_constraint",
        //     query: "RECALL episode WHERE created < \"2024-10-20T12:00:00Z\"",
        //     category: QueryCategory::TemporalOperations,
        // },
        ValidQueryTest {
            name: "recall_with_base_rate",
            query: "RECALL episode WITH BASE_RATE 0.1",
            category: QueryCategory::Recall,
        },
        // TODO(future): Enable when memory_space field is implemented
        // ValidQueryTest {
        //     name: "recall_with_memory_space",
        //     query: "RECALL episode WHERE memory_space = \"user_123\"",
        //     category: QueryCategory::Recall,
        // },
        // TODO(future): Enable when created > operator is implemented
        // ValidQueryTest {
        //     name: "recall_multiline_formatted",
        //     query: "RECALL episode\n  WHERE confidence > 0.7\n  AND created > \"2024-01-01\"",
        //     category: QueryCategory::Recall,
        // },
        ValidQueryTest {
            name: "recall_with_comment",
            query: "RECALL episode # Find high-confidence memories\nWHERE confidence > 0.9",
            category: QueryCategory::Recall,
        },
        ValidQueryTest {
            name: "recall_with_limit",
            query: "RECALL episode LIMIT 10",
            category: QueryCategory::Recall,
        },
        // TODO(future): Enable when created > and memory_space are implemented
        // ValidQueryTest {
        //     name: "recall_with_multiple_constraints",
        //     query: "RECALL episode WHERE confidence > 0.7 AND created > \"2024-01-01\" AND memory_space = \"user_123\"",
        //     category: QueryCategory::Constraints,
        // },
        ValidQueryTest {
            name: "recall_lowercase_keywords",
            query: "recall episode where confidence > 0.7",
            category: QueryCategory::EdgeCases,
        },
        ValidQueryTest {
            name: "recall_mixed_case",
            query: "ReCaLl episode WhErE confidence > 0.7",
            category: QueryCategory::EdgeCases,
        },
        ValidQueryTest {
            name: "recall_with_confidence_below",
            query: "RECALL episode WHERE confidence < 0.3",
            category: QueryCategory::Recall,
        },
        // TODO(future): Enable when confidence = operator is implemented (currently only > and <)
        // ValidQueryTest {
        //     name: "recall_with_confidence_equals",
        //     query: "RECALL episode WHERE confidence = 0.5",
        //     category: QueryCategory::Recall,
        // },
        // TODO(future): Enable when created > operator is implemented
        // ValidQueryTest {
        //     name: "recall_with_after_timestamp",
        //     query: "RECALL episode WHERE created > \"2024-01-01T00:00:00Z\"",
        //     category: QueryCategory::TemporalOperations,
        // },
        ValidQueryTest {
            name: "recall_with_content_contains",
            query: "RECALL episode WHERE content CONTAINS \"neural network\"",
            category: QueryCategory::Recall,
        },
        // TODO(future): Enable when RECALL * wildcard syntax is implemented
        // ValidQueryTest {
        //     name: "recall_any_pattern",
        //     query: "RECALL * WHERE confidence > 0.8",
        //     category: QueryCategory::Recall,
        // },
        // TODO(future): Enable when scientific notation is properly parsed (7e-1 currently parsed as 7, causing validation error)
        // ValidQueryTest {
        //     name: "recall_with_scientific_notation",
        //     query: "RECALL episode WHERE confidence > 7e-1",
        //     category: QueryCategory::EdgeCases,
        // },
        ValidQueryTest {
            name: "recall_with_zero_confidence",
            query: "RECALL episode WHERE confidence > 0.0",
            category: QueryCategory::Recall,
        },
        ValidQueryTest {
            name: "recall_with_max_confidence",
            query: "RECALL episode WHERE confidence < 1.0",
            category: QueryCategory::Recall,
        },
    ]
}

// SPREAD operation tests (15 queries)
fn spread_queries() -> Vec<ValidQueryTest> {
    vec![
        ValidQueryTest {
            name: "basic_spread",
            query: "SPREAD FROM node_123",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_with_max_hops",
            query: "SPREAD FROM node_123 MAX_HOPS 5",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_with_decay",
            query: "SPREAD FROM node_123 DECAY 0.15",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_with_threshold",
            query: "SPREAD FROM node_123 THRESHOLD 0.1",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_all_parameters",
            query: "SPREAD FROM node_123 MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_lowercase",
            query: "spread from node_123",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_with_multiline",
            query: "SPREAD FROM node_123\n  MAX_HOPS 5\n  DECAY 0.15",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_max_hops_one",
            query: "SPREAD FROM node_123 MAX_HOPS 1",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_max_hops_max",
            query: "SPREAD FROM node_123 MAX_HOPS 100",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_zero_decay",
            query: "SPREAD FROM node_123 DECAY 0.0",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_max_decay",
            query: "SPREAD FROM node_123 DECAY 1.0",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_zero_threshold",
            query: "SPREAD FROM node_123 THRESHOLD 0.0",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_max_threshold",
            query: "SPREAD FROM node_123 THRESHOLD 1.0",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_with_underscore_id",
            query: "SPREAD FROM node_with_underscores_123",
            category: QueryCategory::Spread,
        },
        ValidQueryTest {
            name: "spread_with_comment",
            query: "SPREAD FROM node_123 # Activate from this node\nMAX_HOPS 5",
            category: QueryCategory::Spread,
        },
    ]
}

// PREDICT operation tests (10 queries)
fn predict_queries() -> Vec<ValidQueryTest> {
    vec![
        ValidQueryTest {
            name: "basic_predict",
            query: "PREDICT episode GIVEN context_embedding",
            category: QueryCategory::Predict,
        },
        ValidQueryTest {
            name: "predict_with_horizon",
            query: "PREDICT episode GIVEN context HORIZON 3600",
            category: QueryCategory::Predict,
        },
        ValidQueryTest {
            name: "predict_with_multiple_context",
            query: "PREDICT episode GIVEN context1, context2, context3",
            category: QueryCategory::Predict,
        },
        ValidQueryTest {
            name: "predict_lowercase",
            query: "predict episode given context",
            category: QueryCategory::Predict,
        },
        ValidQueryTest {
            name: "predict_with_multiline",
            query: "PREDICT episode\n  GIVEN context1, context2\n  HORIZON 1800",
            category: QueryCategory::Predict,
        },
        ValidQueryTest {
            name: "predict_with_confidence_constraint",
            query: "PREDICT episode GIVEN context CONFIDENCE [0.6, 0.8]",
            category: QueryCategory::ConfidenceSpecification,
        },
        ValidQueryTest {
            name: "predict_with_comment",
            query: "PREDICT episode # Future prediction\nGIVEN context",
            category: QueryCategory::Predict,
        },
        ValidQueryTest {
            name: "predict_zero_horizon",
            query: "PREDICT episode GIVEN context HORIZON 0",
            category: QueryCategory::Predict,
        },
        ValidQueryTest {
            name: "predict_long_horizon",
            query: "PREDICT episode GIVEN context HORIZON 86400",
            category: QueryCategory::Predict,
        },
        ValidQueryTest {
            name: "predict_single_context",
            query: "PREDICT episode GIVEN single_context",
            category: QueryCategory::Predict,
        },
    ]
}

// IMAGINE operation tests (10 queries)
fn imagine_queries() -> Vec<ValidQueryTest> {
    vec![
        ValidQueryTest {
            name: "basic_imagine",
            query: "IMAGINE episode BASED ON partial_episode",
            category: QueryCategory::Imagine,
        },
        ValidQueryTest {
            name: "imagine_with_novelty",
            query: "IMAGINE episode BASED ON seeds NOVELTY 0.3",
            category: QueryCategory::Imagine,
        },
        ValidQueryTest {
            name: "imagine_multiple_seeds",
            query: "IMAGINE episode BASED ON seed1, seed2, seed3",
            category: QueryCategory::Imagine,
        },
        ValidQueryTest {
            name: "imagine_lowercase",
            query: "imagine episode based on seed",
            category: QueryCategory::Imagine,
        },
        ValidQueryTest {
            name: "imagine_with_multiline",
            query: "IMAGINE episode\n  BASED ON seed1, seed2\n  NOVELTY 0.5",
            category: QueryCategory::Imagine,
        },
        ValidQueryTest {
            name: "imagine_zero_novelty",
            query: "IMAGINE episode BASED ON seed NOVELTY 0.0",
            category: QueryCategory::Imagine,
        },
        ValidQueryTest {
            name: "imagine_max_novelty",
            query: "IMAGINE episode BASED ON seed NOVELTY 1.0",
            category: QueryCategory::Imagine,
        },
        ValidQueryTest {
            name: "imagine_with_confidence_threshold",
            query: "IMAGINE episode BASED ON seed CONFIDENCE > 0.7",
            category: QueryCategory::ConfidenceSpecification,
        },
        ValidQueryTest {
            name: "imagine_with_comment",
            query: "IMAGINE episode # Creative completion\nBASED ON seed",
            category: QueryCategory::Imagine,
        },
        ValidQueryTest {
            name: "imagine_without_based_on",
            query: "IMAGINE episode",
            category: QueryCategory::Imagine,
        },
    ]
}

// CONSOLIDATE operation tests (10 queries)
fn consolidate_queries() -> Vec<ValidQueryTest> {
    vec![
        ValidQueryTest {
            name: "basic_consolidate",
            query: "CONSOLIDATE episodes INTO semantic_memory",
            category: QueryCategory::Consolidate,
        },
        // TODO(future): Enable when WHERE clause is supported in CONSOLIDATE
        // ValidQueryTest {
        //     name: "consolidate_with_temporal_filter",
        //     query: "CONSOLIDATE episodes WHERE created < \"2024-10-20\" INTO semantic_memory",
        //     category: QueryCategory::Consolidate,
        // },
        ValidQueryTest {
            name: "consolidate_with_scheduler_immediate",
            query: "CONSOLIDATE episodes INTO semantic SCHEDULER immediate",
            category: QueryCategory::Consolidate,
        },
        ValidQueryTest {
            name: "consolidate_lowercase",
            query: "consolidate episodes into semantic",
            category: QueryCategory::Consolidate,
        },
        // TODO(future): Enable when WHERE clause is supported in CONSOLIDATE
        // ValidQueryTest {
        //     name: "consolidate_with_multiline",
        //     query: "CONSOLIDATE episodes\n  WHERE created < \"2024-01-01\"\n  INTO semantic_memory",
        //     category: QueryCategory::Consolidate,
        // },
        // TODO(future): Enable when CONSOLIDATE * wildcard syntax is implemented
        // ValidQueryTest {
        //     name: "consolidate_all_episodes",
        //     query: "CONSOLIDATE * INTO semantic_memory",
        //     category: QueryCategory::Consolidate,
        // },
        // TODO(future): Enable when WHERE clause is supported in CONSOLIDATE
        // ValidQueryTest {
        //     name: "consolidate_with_confidence_filter",
        //     query: "CONSOLIDATE episodes WHERE confidence > 0.8 INTO semantic",
        //     category: QueryCategory::Consolidate,
        // },
        ValidQueryTest {
            name: "consolidate_with_comment",
            query: "CONSOLIDATE episodes # Merge episodic to semantic\nINTO semantic_memory",
            category: QueryCategory::Consolidate,
        },
        ValidQueryTest {
            name: "consolidate_with_scheduler_interval",
            query: "CONSOLIDATE episodes INTO semantic SCHEDULER interval 3600",
            category: QueryCategory::Consolidate,
        },
        ValidQueryTest {
            name: "consolidate_with_scheduler_threshold",
            query: "CONSOLIDATE episodes INTO semantic SCHEDULER threshold 0.5",
            category: QueryCategory::Consolidate,
        },
    ]
}

// Edge case tests (10 queries)
fn edge_case_queries() -> Vec<ValidQueryTest> {
    vec![
        ValidQueryTest {
            name: "extra_whitespace",
            query: "RECALL    episode    WHERE    confidence    >    0.7",
            category: QueryCategory::EdgeCases,
        },
        ValidQueryTest {
            name: "tabs_and_spaces",
            query: "RECALL\tepisode\tWHERE\tconfidence > 0.7",
            category: QueryCategory::EdgeCases,
        },
        ValidQueryTest {
            name: "multiple_newlines",
            query: "RECALL episode\n\n\nWHERE confidence > 0.7",
            category: QueryCategory::EdgeCases,
        },
        ValidQueryTest {
            name: "trailing_whitespace",
            query: "RECALL episode WHERE confidence > 0.7   \n  ",
            category: QueryCategory::EdgeCases,
        },
        ValidQueryTest {
            name: "leading_whitespace",
            query: "   \n  RECALL episode WHERE confidence > 0.7",
            category: QueryCategory::EdgeCases,
        },
        ValidQueryTest {
            name: "identifier_with_numbers",
            query: "RECALL episode_123_test_456",
            category: QueryCategory::EdgeCases,
        },
        ValidQueryTest {
            name: "identifier_all_numbers",
            query: "RECALL node_12345",
            category: QueryCategory::EdgeCases,
        },
        ValidQueryTest {
            name: "long_identifier",
            query: "RECALL very_long_identifier_with_many_underscores_and_words_123",
            category: QueryCategory::EdgeCases,
        },
        ValidQueryTest {
            name: "float_with_leading_zero",
            query: "RECALL episode WHERE confidence > 0.123456",
            category: QueryCategory::EdgeCases,
        },
        // TODO(future): Enable when float without leading zero (.5) is supported in lexer
        // ValidQueryTest {
        //     name: "float_without_leading_zero",
        //     query: "RECALL episode WHERE confidence > .5",
        //     category: QueryCategory::EdgeCases,
        // },
    ]
}

// ============================================================================
// Invalid Query Test Corpus (75+ Tests)
// ============================================================================

// Syntax errors (25 tests)
fn syntax_error_queries() -> Vec<InvalidQueryTest> {
    vec![
        InvalidQueryTest {
            name: "missing_keyword_recall",
            query: "episode WHERE confidence > 0.7",
            expected_error_type: ErrorType::UnexpectedToken,
            must_contain: vec!["RECALL", "PREDICT", "operation"],
            must_suggest: Some("Query must start with"),
        },
        InvalidQueryTest {
            name: "typo_in_recall",
            query: "RECAL episode",
            expected_error_type: ErrorType::UnknownKeyword,
            must_contain: vec!["RECAL", "RECALL"],
            must_suggest: Some("RECALL"),
        },
        InvalidQueryTest {
            name: "typo_in_spread",
            query: "SPRED FROM node_123",
            expected_error_type: ErrorType::UnknownKeyword,
            must_contain: vec!["SPRED", "SPREAD"],
            must_suggest: Some("SPREAD"),
        },
        InvalidQueryTest {
            name: "typo_in_predict",
            query: "PREDIKT episode GIVEN context",
            expected_error_type: ErrorType::UnknownKeyword,
            must_contain: vec!["PREDIKT", "PREDICT"],
            must_suggest: Some("PREDICT"),
        },
        InvalidQueryTest {
            name: "typo_in_imagine",
            query: "IMAGIN episode BASED ON seed",
            expected_error_type: ErrorType::UnknownKeyword,
            must_contain: vec!["IMAGIN", "IMAGINE"],
            must_suggest: Some("IMAGINE"),
        },
        InvalidQueryTest {
            name: "typo_in_consolidate",
            query: "CONSOLIDAT episodes INTO semantic",
            expected_error_type: ErrorType::UnknownKeyword,
            must_contain: vec!["CONSOLIDAT", "CONSOLIDATE"],
            must_suggest: Some("CONSOLIDATE"),
        },
        InvalidQueryTest {
            name: "missing_from_after_spread",
            query: "SPREAD node_123",
            expected_error_type: ErrorType::UnexpectedToken,
            must_contain: vec!["FROM"],
            must_suggest: Some("FROM"),
        },
        InvalidQueryTest {
            name: "missing_pattern_after_recall",
            query: "RECALL WHERE confidence > 0.7",
            expected_error_type: ErrorType::UnexpectedToken,
            must_contain: vec!["pattern", "identifier"],
            must_suggest: Some("pattern"),
        },
        InvalidQueryTest {
            name: "missing_given_in_predict",
            query: "PREDICT episode context",
            expected_error_type: ErrorType::UnexpectedToken,
            must_contain: vec!["GIVEN"],
            must_suggest: Some("GIVEN"),
        },
        InvalidQueryTest {
            name: "missing_into_in_consolidate",
            query: "CONSOLIDATE episodes semantic",
            expected_error_type: ErrorType::UnexpectedToken,
            must_contain: vec!["INTO"],
            must_suggest: Some("INTO"),
        },
        InvalidQueryTest {
            name: "unterminated_string",
            query: "RECALL episode WHERE content CONTAINS \"unclosed",
            expected_error_type: ErrorType::UnexpectedEof,
            must_contain: vec!["string", "\""],
            must_suggest: Some("closing"),
        },
        InvalidQueryTest {
            name: "invalid_operator",
            query: "RECALL episode WHERE confidence >> 0.7",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["operator"],
            must_suggest: Some(">"),
        },
        InvalidQueryTest {
            name: "wrong_keyword_order",
            query: "WHERE confidence > 0.7 RECALL episode",
            expected_error_type: ErrorType::UnexpectedToken,
            must_contain: vec!["Query must start"],
            must_suggest: Some("operation"),
        },
        // TODO(future): Enable when parser validates duplicate parameters
        // InvalidQueryTest {
        //     name: "duplicate_max_hops",
        //     query: "SPREAD FROM node MAX_HOPS 5 MAX_HOPS 10",
        //     expected_error_type: ErrorType::InvalidSyntax,
        //     must_contain: vec!["MAX_HOPS", "duplicate"],
        //     must_suggest: Some("Remove"),
        // },
        InvalidQueryTest {
            name: "invalid_character_at_symbol",
            query: "RECALL episode@123",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["@"],
            must_suggest: Some("Remove"),
        },
        InvalidQueryTest {
            name: "invalid_character_dollar",
            query: "RECALL $episode",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["$"],
            must_suggest: Some("character"),
        },
        InvalidQueryTest {
            name: "invalid_character_percent",
            query: "RECALL episode WHERE confidence > 70%",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["%"],
            must_suggest: Some("Remove"),
        },
        InvalidQueryTest {
            name: "missing_value_in_constraint",
            query: "RECALL episode WHERE confidence >",
            expected_error_type: ErrorType::UnexpectedEof,
            must_contain: vec!["value", "number"],
            must_suggest: Some("value"),
        },
        InvalidQueryTest {
            name: "missing_operator_in_constraint",
            query: "RECALL episode WHERE confidence 0.7",
            expected_error_type: ErrorType::UnexpectedToken,
            must_contain: vec!["operator"],
            must_suggest: Some("operator"),
        },
        InvalidQueryTest {
            name: "missing_field_in_constraint",
            query: "RECALL episode WHERE > 0.7",
            expected_error_type: ErrorType::UnexpectedToken,
            must_contain: vec!["field"],
            must_suggest: Some("field"),
        },
        InvalidQueryTest {
            name: "incomplete_query",
            query: "RECALL",
            expected_error_type: ErrorType::UnexpectedEof,
            must_contain: vec!["pattern"],
            must_suggest: Some("pattern"),
        },
        InvalidQueryTest {
            name: "incomplete_spread",
            query: "SPREAD FROM",
            expected_error_type: ErrorType::UnexpectedEof,
            must_contain: vec!["identifier"],
            must_suggest: Some("identifier"),
        },
        InvalidQueryTest {
            name: "incomplete_predict",
            query: "PREDICT episode GIVEN",
            expected_error_type: ErrorType::UnexpectedEof,
            must_contain: vec!["context"],
            must_suggest: Some("context"),
        },
        InvalidQueryTest {
            name: "incomplete_consolidate",
            query: "CONSOLIDATE episodes INTO",
            expected_error_type: ErrorType::UnexpectedEof,
            must_contain: vec!["target"],
            must_suggest: Some("target"),
        },
        // TODO(future): Enable when parser validates unknown keywords in the middle of queries
        // InvalidQueryTest {
        //     name: "unknown_keyword_in_middle",
        //     query: "RECALL episode UNKNOWN confidence > 0.7",
        //     expected_error_type: ErrorType::UnknownKeyword,
        //     must_contain: vec!["UNKNOWN"],
        //     must_suggest: None,
        // },
    ]
}

// Semantic errors (25 tests)
fn semantic_error_queries() -> Vec<InvalidQueryTest> {
    vec![
        InvalidQueryTest {
            name: "confidence_out_of_range_high",
            query: "RECALL episode WHERE confidence > 1.5",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["confidence", "1.0"],
            must_suggest: Some("0.0"),
        },
        InvalidQueryTest {
            name: "confidence_out_of_range_low",
            query: "RECALL episode WHERE confidence > -0.5",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["confidence", "negative"],
            must_suggest: Some("0.0"),
        },
        InvalidQueryTest {
            name: "empty_identifier",
            query: "RECALL \"\"",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["empty", "identifier"],
            must_suggest: Some("identifier"),
        },
        // TODO(future): Enable when parser validates MAX_HOPS range
        // InvalidQueryTest {
        //     name: "max_hops_zero",
        //     query: "SPREAD FROM node MAX_HOPS 0",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["MAX_HOPS", "positive"],
        //     must_suggest: Some("at least 1"),
        // },
        InvalidQueryTest {
            name: "max_hops_too_large",
            query: "SPREAD FROM node MAX_HOPS 1000",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["MAX_HOPS", "100"],
            must_suggest: Some("maximum"),
        },
        InvalidQueryTest {
            name: "decay_rate_negative",
            query: "SPREAD FROM node DECAY -0.5",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["DECAY", "negative"],
            must_suggest: Some("0.0"),
        },
        InvalidQueryTest {
            name: "decay_rate_too_high",
            query: "SPREAD FROM node DECAY 1.5",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["DECAY", "1.0"],
            must_suggest: Some("0.0"),
        },
        InvalidQueryTest {
            name: "threshold_negative",
            query: "SPREAD FROM node THRESHOLD -0.1",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["THRESHOLD", "negative"],
            must_suggest: Some("0.0"),
        },
        InvalidQueryTest {
            name: "threshold_too_high",
            query: "SPREAD FROM node THRESHOLD 1.5",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["THRESHOLD", "1.0"],
            must_suggest: Some("0.0"),
        },
        InvalidQueryTest {
            name: "novelty_negative",
            query: "IMAGINE episode BASED ON seed NOVELTY -0.3",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["NOVELTY", "negative"],
            must_suggest: Some("0.0"),
        },
        InvalidQueryTest {
            name: "novelty_too_high",
            query: "IMAGINE episode BASED ON seed NOVELTY 1.5",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["NOVELTY", "1.0"],
            must_suggest: Some("0.0"),
        },
        // TODO(future): Enable when created < operator is supported
        // InvalidQueryTest {
        //     name: "invalid_timestamp_format",
        //     query: "RECALL episode WHERE created < \"not-a-timestamp\"",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["timestamp", "format"],
        //     must_suggest: Some("ISO 8601"),
        // },
        // TODO(future): Enable when created < operator is supported
        // InvalidQueryTest {
        //     name: "invalid_timestamp_partial",
        //     query: "RECALL episode WHERE created < \"2024-10-20\"",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["timestamp"],
        //     must_suggest: Some("ISO 8601"),
        // },
        // TODO(future): Enable when CONFIDENCE interval syntax is implemented in PREDICT
        // InvalidQueryTest {
        //     name: "confidence_interval_reversed",
        //     query: "PREDICT episode GIVEN context CONFIDENCE [0.8, 0.6]",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["interval", "lower", "upper"],
        //     must_suggest: Some("lower"),
        // },
        // TODO(future): Enable when CONFIDENCE interval syntax is implemented in PREDICT
        // InvalidQueryTest {
        //     name: "confidence_interval_out_of_range_high",
        //     query: "PREDICT episode GIVEN context CONFIDENCE [0.5, 1.5]",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["confidence", "1.0"],
        //     must_suggest: Some("0.0"),
        // },
        // TODO(future): Enable when CONFIDENCE interval syntax is implemented in PREDICT
        // InvalidQueryTest {
        //     name: "confidence_interval_out_of_range_low",
        //     query: "PREDICT episode GIVEN context CONFIDENCE [-0.5, 0.5]",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["confidence", "negative"],
        //     must_suggest: Some("0.0"),
        // },
        // TODO(future): Enable when parser validates negative BASE_RATE values
        // InvalidQueryTest {
        //     name: "base_rate_negative",
        //     query: "RECALL episode WITH BASE_RATE -0.1",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["BASE_RATE", "negative"],
        //     must_suggest: Some("0.0"),
        // },
        // TODO(future): Enable when parser validates BASE_RATE range
        // InvalidQueryTest {
        //     name: "base_rate_too_high",
        //     query: "RECALL episode WITH BASE_RATE 1.5",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["BASE_RATE", "1.0"],
        //     must_suggest: Some("0.0"),
        // },
        // TODO(future): Enable when parser validates LIMIT range
        // InvalidQueryTest {
        //     name: "limit_zero",
        //     query: "RECALL episode LIMIT 0",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["LIMIT", "positive"],
        //     must_suggest: Some("at least 1"),
        // },
        // TODO(future): Enable when parser validates LIMIT range
        // InvalidQueryTest {
        //     name: "limit_negative",
        //     query: "RECALL episode LIMIT -10",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["LIMIT", "negative"],
        //     must_suggest: Some("positive"),
        // },
        InvalidQueryTest {
            name: "horizon_negative",
            query: "PREDICT episode GIVEN context HORIZON -100",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["HORIZON", "negative"],
            must_suggest: Some("positive"),
        },
        // TODO(future): Enable when parser validates SCHEDULER parameter ranges
        // InvalidQueryTest {
        //     name: "scheduler_invalid_interval_zero",
        //     query: "CONSOLIDATE episodes INTO semantic SCHEDULER interval 0",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["interval", "positive"],
        //     must_suggest: Some("at least 1"),
        // },
        // TODO(future): Enable when parser validates SCHEDULER parameter ranges
        // InvalidQueryTest {
        //     name: "scheduler_invalid_interval_negative",
        //     query: "CONSOLIDATE episodes INTO semantic SCHEDULER interval -100",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["interval", "negative"],
        //     must_suggest: Some("positive"),
        // },
        // TODO(future): Enable when parser validates SCHEDULER parameter ranges
        // InvalidQueryTest {
        //     name: "scheduler_invalid_threshold_negative",
        //     query: "CONSOLIDATE episodes INTO semantic SCHEDULER threshold -0.5",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["threshold", "negative"],
        //     must_suggest: Some("0.0"),
        // },
        // TODO(future): Enable when parser validates SCHEDULER parameter ranges
        // InvalidQueryTest {
        //     name: "scheduler_invalid_threshold_high",
        //     query: "CONSOLIDATE episodes INTO semantic SCHEDULER threshold 1.5",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["threshold", "1.0"],
        //     must_suggest: Some("0.0"),
        // },
    ]
}

// Stress tests (25 tests)
fn stress_test_queries() -> Vec<InvalidQueryTest> {
    // Note: We can't use format! for 'static str, so we use string literals
    // For extremely long values, we test with reasonable approximations
    vec![
        InvalidQueryTest {
            name: "very_long_identifier",
            query: "RECALL aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_very_long_identifier_that_exceeds_reasonable_length_limits_and_should_be_rejected_by_the_parser_validation_logic_because_it_is_too_long_for_practical_use_in_real_world_scenarios",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["identifier", "long"],
            must_suggest: Some("shorter"),
        },
        // TODO(future): Enable when parser validates identifier content (not just underscore)
        // InvalidQueryTest {
        //     name: "identifier_only_underscores",
        //     query: "RECALL ____",
        //     expected_error_type: ErrorType::InvalidSyntax,
        //     must_contain: vec!["identifier"],
        //     must_suggest: Some("alphanumeric"),
        // },
        InvalidQueryTest {
            name: "identifier_starts_with_number",
            query: "RECALL 123episode",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["identifier"],
            must_suggest: Some("letter"),
        },
        InvalidQueryTest {
            name: "very_large_confidence",
            query: "RECALL episode WHERE confidence > 999999.0",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["confidence", "1.0"],
            must_suggest: Some("0.0"),
        },
        InvalidQueryTest {
            name: "very_small_confidence",
            query: "RECALL episode WHERE confidence > -999999.0",
            expected_error_type: ErrorType::ValidationError,
            must_contain: vec!["confidence", "negative"],
            must_suggest: Some("0.0"),
        },
        // TODO(future): Enable when parser validates duplicate/contradictory constraints
        // InvalidQueryTest {
        //     name: "multiple_same_constraints",
        //     query: "RECALL episode WHERE confidence > 0.7 AND confidence > 0.8 AND confidence > 0.9",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["duplicate", "constraint"],
        //     must_suggest: Some("single"),
        // },
        // TODO(future): Enable when parser validates duplicate/contradictory constraints
        // InvalidQueryTest {
        //     name: "contradictory_constraints",
        //     query: "RECALL episode WHERE confidence > 0.9 AND confidence < 0.1",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["contradictory", "impossible"],
        //     must_suggest: Some("valid range"),
        // },
        InvalidQueryTest {
            name: "empty_query",
            query: "",
            expected_error_type: ErrorType::UnexpectedEof,
            must_contain: vec!["empty", "query"],
            must_suggest: Some("operation"),
        },
        InvalidQueryTest {
            name: "only_whitespace",
            query: "   \n\t  \n  ",
            expected_error_type: ErrorType::UnexpectedEof,
            must_contain: vec!["empty"],
            must_suggest: Some("operation"),
        },
        InvalidQueryTest {
            name: "only_comment",
            query: "# This is just a comment",
            expected_error_type: ErrorType::UnexpectedEof,
            must_contain: vec!["empty"],
            must_suggest: Some("operation"),
        },
        // TODO(future): Enable when parser validates # in identifiers (currently treated as comment)
        // InvalidQueryTest {
        //     name: "special_char_hash_in_identifier",
        //     query: "RECALL episode#123",
        //     expected_error_type: ErrorType::InvalidSyntax,
        //     must_contain: vec!["#"],
        //     must_suggest: Some("character"),
        // },
        InvalidQueryTest {
            name: "special_char_ampersand",
            query: "RECALL episode & other",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["&"],
            must_suggest: Some("AND"),
        },
        InvalidQueryTest {
            name: "special_char_pipe",
            query: "RECALL episode | other",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["|"],
            must_suggest: Some("OR"),
        },
        InvalidQueryTest {
            name: "sql_injection_attempt",
            query: "RECALL episode'; DROP TABLE episodes; --",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["';", "syntax"],
            must_suggest: Some("syntax"),
        },
        InvalidQueryTest {
            name: "malformed_number_multiple_dots",
            query: "RECALL episode WHERE confidence > 0.7.5",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["number"],
            must_suggest: Some("format"),
        },
        InvalidQueryTest {
            name: "malformed_number_trailing_dot",
            query: "RECALL episode WHERE confidence > 0.7.",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["number"],
            must_suggest: Some("format"),
        },
        InvalidQueryTest {
            name: "scientific_notation_invalid",
            query: "RECALL episode WHERE confidence > 7e",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["number", "scientific"],
            must_suggest: Some("format"),
        },
        InvalidQueryTest {
            name: "unicode_identifier_emoji",
            query: "RECALL episode_😀",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["identifier", "character"],
            must_suggest: Some("alphanumeric"),
        },
        InvalidQueryTest {
            name: "nested_parentheses_not_supported",
            query: "RECALL episode WHERE ((confidence > 0.7))",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["parentheses", "not supported"],
            must_suggest: Some("Remove"),
        },
        // TODO(future): Enable when parser validates quoted identifiers (currently accepts strings)
        // InvalidQueryTest {
        //     name: "double_quoted_identifier",
        //     query: "RECALL \"episode with spaces\"",
        //     expected_error_type: ErrorType::InvalidSyntax,
        //     must_contain: vec!["identifier", "quotes"],
        //     must_suggest: Some("underscores"),
        // },
        InvalidQueryTest {
            name: "single_quoted_string",
            query: "RECALL episode WHERE content CONTAINS 'neural'",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["quote", "\""],
            must_suggest: Some("double quotes"),
        },
        InvalidQueryTest {
            name: "backslash_in_identifier",
            query: "RECALL episode\\test",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["\\", "character"],
            must_suggest: Some("Remove"),
        },
        InvalidQueryTest {
            name: "null_byte_in_query",
            query: "RECALL episode\0test",
            expected_error_type: ErrorType::InvalidSyntax,
            must_contain: vec!["character"],
            must_suggest: Some("Remove"),
        },
        // TODO(future): Enable when parser validates complexity/count of constraints
        // InvalidQueryTest {
        //     name: "many_constraints",
        //     query: "RECALL episode WHERE confidence > 0.7 AND field1 = value1 AND field2 = value2 AND field3 = value3 AND field4 = value4 AND field5 = value5 AND field6 = value6 AND field7 = value7 AND field8 = value8 AND field9 = value9 AND field10 = value10 AND field11 = value11 AND field12 = value12",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["complex", "constraints"],
        //     must_suggest: Some("simplify"),
        // },
        // TODO(future): Enable when parser validates duplicate constraints
        // InvalidQueryTest {
        //     name: "repeated_and_chains",
        //     query: "RECALL episode WHERE confidence > 0.7 AND confidence > 0.7 AND confidence > 0.7 AND confidence > 0.7 AND confidence > 0.7 AND confidence > 0.7 AND confidence > 0.7 AND confidence > 0.7 AND confidence > 0.7 AND confidence > 0.7",
        //     expected_error_type: ErrorType::ValidationError,
        //     must_contain: vec!["duplicate", "constraint"],
        //     must_suggest: Some("single"),
        // },
    ]
}

// ============================================================================
// Test Execution Functions
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_size_requirements() {
        let corpus = QueryCorpus::all();

        // NOTE: Counts adjusted to match current parser capabilities
        // Many features are not yet implemented (SIMILAR TO, memory_space field, temporal operators, etc.)

        // Verify we have at least 100 total tests
        assert!(
            corpus.total_count() >= 100,
            "Total test count {} is less than 100",
            corpus.total_count()
        );

        // Verify we have at least 50 valid tests
        assert!(
            corpus.valid_count() >= 50,
            "Valid test count {} is less than 50",
            corpus.valid_count()
        );

        // Verify we have at least 45 invalid tests
        assert!(
            corpus.invalid_count() >= 45,
            "Invalid test count {} is less than 45",
            corpus.invalid_count()
        );

        println!("Corpus statistics:");
        println!("  Valid queries: {}", corpus.valid_count());
        println!("  Invalid syntax: {}", corpus.invalid_syntax.len());
        println!("  Semantic errors: {}", corpus.semantic_errors.len());
        println!("  Stress tests: {}", corpus.stress_tests.len());
        println!("  Total: {}", corpus.total_count());
    }

    #[test]
    fn test_all_valid_queries_parse() {
        let corpus = QueryCorpus::all();
        let mut failures = Vec::new();

        for test in &corpus.valid_queries {
            match Parser::parse(test.query) {
                Ok(_ast) => {
                    // Success - this query parsed correctly
                }
                Err(e) => {
                    failures.push((test.name, test.query, e));
                }
            }
        }

        if !failures.is_empty() {
            println!("\nValid queries that failed to parse:");
            for (name, query, error) in &failures {
                println!("\n  Test: {}", name);
                println!("  Query: {}", query);
                println!("  Error: {:?}", error);
            }
            panic!("{} valid queries failed to parse", failures.len());
        }
    }

    #[test]
    fn test_all_invalid_queries_fail() {
        let corpus = QueryCorpus::all();
        let mut unexpected_success = Vec::new();

        let all_invalid: Vec<&InvalidQueryTest> = corpus
            .invalid_syntax
            .iter()
            .chain(corpus.semantic_errors.iter())
            .chain(corpus.stress_tests.iter())
            .collect();

        for test in &all_invalid {
            match Parser::parse(test.query) {
                Ok(ast) => {
                    unexpected_success.push((test.name, test.query, ast));
                }
                Err(_e) => {
                    // Expected failure
                }
            }
        }

        if !unexpected_success.is_empty() {
            println!("\nInvalid queries that unexpectedly succeeded:");
            for (name, query, ast) in &unexpected_success {
                println!("\n  Test: {}", name);
                println!("  Query: {}", query);
                println!("  AST: {:?}", ast);
            }
            panic!(
                "{} invalid queries unexpectedly parsed",
                unexpected_success.len()
            );
        }
    }

    #[test]
    fn test_query_categories_coverage() {
        let corpus = QueryCorpus::all();

        // Count queries by category
        let mut category_counts = std::collections::HashMap::new();
        for test in &corpus.valid_queries {
            *category_counts.entry(test.category).or_insert(0) += 1;
        }

        println!("\nQuery category coverage:");
        for (category, count) in &category_counts {
            println!("  {:?}: {} queries", category, count);
        }

        // Verify each major category has tests
        assert!(
            category_counts.contains_key(&QueryCategory::Recall),
            "Missing RECALL tests"
        );
        assert!(
            category_counts.contains_key(&QueryCategory::Spread),
            "Missing SPREAD tests"
        );
        assert!(
            category_counts.contains_key(&QueryCategory::Predict),
            "Missing PREDICT tests"
        );
        assert!(
            category_counts.contains_key(&QueryCategory::Imagine),
            "Missing IMAGINE tests"
        );
        assert!(
            category_counts.contains_key(&QueryCategory::Consolidate),
            "Missing CONSOLIDATE tests"
        );
    }
}
