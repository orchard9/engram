# Task 008: Query Language Validation Suite

**Status**: Pending
**Duration**: 2 days (extended from 1 day for comprehensive test corpus)
**Dependencies**: Task 004 (Error Recovery), Task 006 (RECALL), Task 007 (SPREAD)
**Owner**: TBD

---

## Objective

Comprehensive validation suite with 150+ test queries, property-based testing of parser invariants, differential testing against reference implementations, and fuzzing infrastructure for robustness. Verify 100% of errors have actionable messages and all valid queries parse in <100μs.

---

## Technical Specification

### 1. Test Corpus Organization

```rust
// File: engram-core/tests/query_language_corpus.rs

/// Comprehensive test corpus covering all query language features
pub struct QueryCorpus {
    pub valid_queries: Vec<ValidQueryTest>,
    pub invalid_syntax: Vec<InvalidQueryTest>,
    pub error_messages: Vec<ErrorMessageTest>,
    pub edge_cases: Vec<EdgeCaseTest>,
    pub performance: Vec<PerformanceTest>,
}

#[derive(Debug, Clone)]
pub struct ValidQueryTest {
    pub name: &'static str,
    pub query: &'static str,
    pub expected_ast: Box<dyn Fn(&Query) -> bool>,
    pub category: QueryCategory,
}

#[derive(Debug, Clone)]
pub struct InvalidQueryTest {
    pub name: &'static str,
    pub query: &'static str,
    pub expected_error_kind: ErrorKind,
    pub must_contain: Vec<&'static str>,  // Error message must contain these
    pub must_suggest: Option<&'static str>,  // Expected suggestion
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryCategory {
    Recall,
    Predict,
    Imagine,
    Consolidate,
    Spread,
    Constraints,
    ConfidenceSpecification,
    EmbeddingLiterals,
    TemporalOperations,
}
```

### 2. Valid Query Test Corpus (75+ Tests)

```rust
// RECALL operation tests (20 queries)
const RECALL_QUERIES: &[ValidQueryTest] = &[
    ValidQueryTest {
        name: "basic_recall",
        query: "RECALL episode",
        expected_ast: |q| matches!(q, Query::Recall(_)),
        category: QueryCategory::Recall,
    },
    ValidQueryTest {
        name: "recall_with_confidence_threshold",
        query: "RECALL episode WHERE confidence > 0.7",
        expected_ast: |q| {
            matches!(q, Query::Recall(r) if r.confidence_threshold.is_some())
        },
        category: QueryCategory::Recall,
    },
    ValidQueryTest {
        name: "recall_with_embedding_similarity",
        query: "RECALL episode WHERE content SIMILAR TO [0.1, 0.2, 0.3] THRESHOLD 0.8",
        expected_ast: |q| {
            matches!(q, Query::Recall(r) if !r.constraints.is_empty())
        },
        category: QueryCategory::Recall,
    },
    ValidQueryTest {
        name: "recall_with_large_embedding",
        query: "RECALL episode WHERE content SIMILAR TO [0.1; 768]",  // Compact notation
        expected_ast: |q| matches!(q, Query::Recall(_)),
        category: QueryCategory::EmbeddingLiterals,
    },
    ValidQueryTest {
        name: "recall_with_temporal_constraint",
        query: "RECALL episode WHERE created < \"2024-10-20T12:00:00Z\"",
        expected_ast: |q| {
            matches!(q, Query::Recall(r) if r.constraints.iter().any(|c| matches!(c, Constraint::CreatedBefore(_))))
        },
        category: QueryCategory::TemporalOperations,
    },
    ValidQueryTest {
        name: "recall_with_confidence_interval",
        query: "RECALL episode CONFIDENCE [0.6, 0.8]",
        expected_ast: |q| {
            matches!(q, Query::Recall(r) if r.confidence_threshold.is_some())
        },
        category: QueryCategory::ConfidenceSpecification,
    },
    ValidQueryTest {
        name: "recall_with_base_rate",
        query: "RECALL episode WITH BASE_RATE 0.1",
        expected_ast: |q| {
            matches!(q, Query::Recall(r) if r.base_rate.is_some())
        },
        category: QueryCategory::Recall,
    },
    ValidQueryTest {
        name: "recall_with_memory_space",
        query: "RECALL episode WHERE memory_space = \"user_123\"",
        expected_ast: |q| {
            matches!(q, Query::Recall(r) if r.constraints.iter().any(|c| matches!(c, Constraint::MemorySpace(_))))
        },
        category: QueryCategory::Recall,
    },
    ValidQueryTest {
        name: "recall_multiline_formatted",
        query: "RECALL episode\n  WHERE confidence > 0.7\n  AND created > \"2024-01-01\"",
        expected_ast: |q| {
            matches!(q, Query::Recall(r) if r.constraints.len() >= 2)
        },
        category: QueryCategory::Recall,
    },
    ValidQueryTest {
        name: "recall_with_comment",
        query: "RECALL episode # Find high-confidence memories\n  WHERE confidence > 0.9",
        expected_ast: |q| matches!(q, Query::Recall(_)),
        category: QueryCategory::Recall,
    },
    // ... 10 more RECALL variations
];

// SPREAD operation tests (15 queries)
const SPREAD_QUERIES: &[ValidQueryTest] = &[
    ValidQueryTest {
        name: "basic_spread",
        query: "SPREAD FROM node_123",
        expected_ast: |q| matches!(q, Query::Spread(_)),
        category: QueryCategory::Spread,
    },
    ValidQueryTest {
        name: "spread_with_max_hops",
        query: "SPREAD FROM node_123 MAX_HOPS 5",
        expected_ast: |q| {
            matches!(q, Query::Spread(s) if s.max_hops == Some(5))
        },
        category: QueryCategory::Spread,
    },
    ValidQueryTest {
        name: "spread_with_decay",
        query: "SPREAD FROM node_123 DECAY 0.15",
        expected_ast: |q| {
            matches!(q, Query::Spread(s) if s.decay_rate.is_some())
        },
        category: QueryCategory::Spread,
    },
    ValidQueryTest {
        name: "spread_with_threshold",
        query: "SPREAD FROM node_123 THRESHOLD 0.1",
        expected_ast: |q| {
            matches!(q, Query::Spread(s) if s.activation_threshold.is_some())
        },
        category: QueryCategory::Spread,
    },
    ValidQueryTest {
        name: "spread_all_parameters",
        query: "SPREAD FROM node_123 MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1",
        expected_ast: |q| {
            matches!(q, Query::Spread(s)
                if s.max_hops.is_some()
                && s.decay_rate.is_some()
                && s.activation_threshold.is_some())
        },
        category: QueryCategory::Spread,
    },
    ValidQueryTest {
        name: "spread_from_embedding",
        query: "SPREAD FROM [0.1, 0.2, 0.3] MAX_HOPS 3",
        expected_ast: |q| matches!(q, Query::Spread(_)),
        category: QueryCategory::Spread,
    },
    // ... 9 more SPREAD variations
];

// PREDICT operation tests (10 queries)
const PREDICT_QUERIES: &[ValidQueryTest] = &[
    ValidQueryTest {
        name: "basic_predict",
        query: "PREDICT episode GIVEN context_embedding",
        expected_ast: |q| matches!(q, Query::Predict(_)),
        category: QueryCategory::Predict,
    },
    ValidQueryTest {
        name: "predict_with_horizon",
        query: "PREDICT episode GIVEN context HORIZON 3600",
        expected_ast: |q| {
            matches!(q, Query::Predict(p) if p.horizon.is_some())
        },
        category: QueryCategory::Predict,
    },
    ValidQueryTest {
        name: "predict_with_confidence_range",
        query: "PREDICT episode GIVEN context CONFIDENCE [0.6, 0.8]",
        expected_ast: |q| matches!(q, Query::Predict(_)),
        category: QueryCategory::Predict,
    },
    // ... 7 more PREDICT variations
];

// IMAGINE operation tests (10 queries)
const IMAGINE_QUERIES: &[ValidQueryTest] = &[
    ValidQueryTest {
        name: "basic_imagine",
        query: "IMAGINE episode BASED ON partial_episode",
        expected_ast: |q| matches!(q, Query::Imagine(_)),
        category: QueryCategory::Imagine,
    },
    ValidQueryTest {
        name: "imagine_with_novelty",
        query: "IMAGINE episode BASED ON seeds NOVELTY 0.3",
        expected_ast: |q| {
            matches!(q, Query::Imagine(i) if i.novelty_level.is_some())
        },
        category: QueryCategory::Imagine,
    },
    ValidQueryTest {
        name: "imagine_multiple_seeds",
        query: "IMAGINE episode BASED ON [seed1, seed2, seed3] NOVELTY 0.5",
        expected_ast: |q| matches!(q, Query::Imagine(_)),
        category: QueryCategory::Imagine,
    },
    // ... 7 more IMAGINE variations
];

// CONSOLIDATE operation tests (10 queries)
const CONSOLIDATE_QUERIES: &[ValidQueryTest] = &[
    ValidQueryTest {
        name: "basic_consolidate",
        query: "CONSOLIDATE episodes INTO semantic_memory",
        expected_ast: |q| matches!(q, Query::Consolidate(_)),
        category: QueryCategory::Consolidate,
    },
    ValidQueryTest {
        name: "consolidate_with_temporal_filter",
        query: "CONSOLIDATE episodes WHERE created < \"2024-10-20\" INTO semantic_memory",
        expected_ast: |q| {
            matches!(q, Query::Consolidate(c) if !c.constraints.is_empty())
        },
        category: QueryCategory::Consolidate,
    },
    ValidQueryTest {
        name: "consolidate_with_scheduler",
        query: "CONSOLIDATE episodes INTO semantic SCHEDULER immediate",
        expected_ast: |q| {
            matches!(q, Query::Consolidate(c) if c.scheduler_policy.is_some())
        },
        category: QueryCategory::Consolidate,
    },
    // ... 7 more CONSOLIDATE variations
];

// Edge case tests (10 queries)
const EDGE_CASE_QUERIES: &[ValidQueryTest] = &[
    ValidQueryTest {
        name: "case_insensitive_keywords",
        query: "recall episode where confidence > 0.7",
        expected_ast: |q| matches!(q, Query::Recall(_)),
        category: QueryCategory::Recall,
    },
    ValidQueryTest {
        name: "unicode_identifiers",
        query: "RECALL эпизод_123",  // Cyrillic characters
        expected_ast: |q| matches!(q, Query::Recall(_)),
        category: QueryCategory::EdgeCases,
    },
    ValidQueryTest {
        name: "very_long_embedding",
        query: &format!("RECALL episode WHERE content SIMILAR TO [{}]",
            (0..1536).map(|i| format!("{:.6}", i as f32 / 1536.0)).collect::<Vec<_>>().join(", ")),
        expected_ast: |q| matches!(q, Query::Recall(_)),
        category: QueryCategory::EmbeddingLiterals,
    },
    ValidQueryTest {
        name: "scientific_notation",
        query: "RECALL episode WHERE confidence > 7e-1",
        expected_ast: |q| matches!(q, Query::Recall(_)),
        category: QueryCategory::EdgeCases,
    },
    ValidQueryTest {
        name: "extreme_confidence_values",
        query: "RECALL episode WHERE confidence > 0.0 AND confidence < 1.0",
        expected_ast: |q| matches!(q, Query::Recall(_)),
        category: QueryCategory::Constraints,
    },
    // ... 5 more edge cases
];
```

### 3. Invalid Query Test Corpus (75+ Tests)

```rust
// Syntax errors (25 tests)
const SYNTAX_ERROR_QUERIES: &[InvalidQueryTest] = &[
    InvalidQueryTest {
        name: "missing_keyword_recall",
        query: "episode WHERE confidence > 0.7",
        expected_error_kind: ErrorKind::UnexpectedToken,
        must_contain: vec!["Expected:", "RECALL", "PREDICT"],
        must_suggest: Some("Query must start with a cognitive operation keyword"),
    },
    InvalidQueryTest {
        name: "typo_in_recall",
        query: "RECAL episode",
        expected_error_kind: ErrorKind::UnknownKeyword,
        must_contain: vec!["RECAL", "RECALL"],
        must_suggest: Some("Did you mean: 'RECALL'?"),
    },
    InvalidQueryTest {
        name: "typo_in_spread",
        query: "SPRED FROM node_123",
        expected_error_kind: ErrorKind::UnknownKeyword,
        must_contain: vec!["SPRED", "SPREAD"],
        must_suggest: Some("Did you mean: 'SPREAD'?"),
    },
    InvalidQueryTest {
        name: "missing_from_after_spread",
        query: "SPREAD node_123",
        expected_error_kind: ErrorKind::UnexpectedToken,
        must_contain: vec!["FROM", "Expected"],
        must_suggest: Some("SPREAD requires FROM keyword"),
    },
    InvalidQueryTest {
        name: "missing_pattern_after_recall",
        query: "RECALL WHERE confidence > 0.7",
        expected_error_kind: ErrorKind::UnexpectedToken,
        must_contain: vec!["pattern", "identifier"],
        must_suggest: Some("RECALL requires a pattern"),
    },
    InvalidQueryTest {
        name: "unterminated_embedding",
        query: "RECALL episode WHERE content SIMILAR TO [0.1, 0.2",
        expected_error_kind: ErrorKind::UnexpectedEof,
        must_contain: vec!["]", "Expected"],
        must_suggest: Some("Unterminated embedding literal"),
    },
    InvalidQueryTest {
        name: "invalid_operator_in_constraint",
        query: "RECALL episode WHERE confidence >> 0.7",
        expected_error_kind: ErrorKind::InvalidSyntax,
        must_contain: vec!["operator", ">", "<", "="],
        must_suggest: Some("Use >, <, >=, <=, or ="),
    },
    InvalidQueryTest {
        name: "wrong_keyword_order",
        query: "WHERE confidence > 0.7 RECALL episode",
        expected_error_kind: ErrorKind::UnexpectedToken,
        must_contain: vec!["Query must start"],
        must_suggest: None,
    },
    InvalidQueryTest {
        name: "duplicate_max_hops",
        query: "SPREAD FROM node MAX_HOPS 5 MAX_HOPS 10",
        expected_error_kind: ErrorKind::InvalidSyntax,
        must_contain: vec!["MAX_HOPS", "already specified"],
        must_suggest: Some("Remove duplicate parameter"),
    },
    // ... 16 more syntax errors
];

// Semantic errors (25 tests)
const SEMANTIC_ERROR_QUERIES: &[InvalidQueryTest] = &[
    InvalidQueryTest {
        name: "confidence_out_of_range_high",
        query: "RECALL episode WHERE confidence > 1.5",
        expected_error_kind: ErrorKind::ValidationError,
        must_contain: vec!["confidence", "0.0", "1.0"],
        must_suggest: Some("Confidence must be between 0.0 and 1.0"),
    },
    InvalidQueryTest {
        name: "confidence_out_of_range_low",
        query: "RECALL episode WHERE confidence > -0.5",
        expected_error_kind: ErrorKind::ValidationError,
        must_contain: vec!["confidence", "negative"],
        must_suggest: Some("Confidence must be non-negative"),
    },
    InvalidQueryTest {
        name: "empty_embedding",
        query: "RECALL episode WHERE content SIMILAR TO []",
        expected_error_kind: ErrorKind::ValidationError,
        must_contain: vec!["embedding", "empty"],
        must_suggest: Some("Embedding must contain at least one value"),
    },
    InvalidQueryTest {
        name: "max_hops_zero",
        query: "SPREAD FROM node MAX_HOPS 0",
        expected_error_kind: ErrorKind::ValidationError,
        must_contain: vec!["MAX_HOPS", "positive"],
        must_suggest: Some("MAX_HOPS must be at least 1"),
    },
    InvalidQueryTest {
        name: "max_hops_too_large",
        query: "SPREAD FROM node MAX_HOPS 1000",
        expected_error_kind: ErrorKind::ValidationError,
        must_contain: vec!["MAX_HOPS", "maximum"],
        must_suggest: Some("MAX_HOPS cannot exceed 100"),
    },
    InvalidQueryTest {
        name: "decay_rate_negative",
        query: "SPREAD FROM node DECAY -0.5",
        expected_error_kind: ErrorKind::ValidationError,
        must_contain: vec!["DECAY", "negative"],
        must_suggest: Some("Decay rate must be between 0.0 and 1.0"),
    },
    InvalidQueryTest {
        name: "invalid_timestamp_format",
        query: "RECALL episode WHERE created < \"not-a-timestamp\"",
        expected_error_kind: ErrorKind::ValidationError,
        must_contain: vec!["timestamp", "format"],
        must_suggest: Some("Use ISO 8601 format"),
    },
    InvalidQueryTest {
        name: "confidence_interval_reversed",
        query: "RECALL episode CONFIDENCE [0.8, 0.6]",
        expected_error_kind: ErrorKind::ValidationError,
        must_contain: vec!["interval", "lower", "upper"],
        must_suggest: Some("Lower bound must be less than upper bound"),
    },
    // ... 17 more semantic errors
];

// Parser stress tests (25 tests)
const STRESS_TEST_QUERIES: &[InvalidQueryTest] = &[
    InvalidQueryTest {
        name: "extremely_long_identifier",
        query: &format!("RECALL {}", "a".repeat(10000)),
        expected_error_kind: ErrorKind::ValidationError,
        must_contain: vec!["identifier", "too long"],
        must_suggest: Some("Identifier must be shorter than 1000 characters"),
    },
    InvalidQueryTest {
        name: "deeply_nested_constraints",
        query: "RECALL episode WHERE (((((confidence > 0.7)))))",  // Not supported
        expected_error_kind: ErrorKind::InvalidSyntax,
        must_contain: vec!["constraint", "syntax"],
        must_suggest: None,
    },
    InvalidQueryTest {
        name: "special_characters_in_identifier",
        query: "RECALL episode@#$%",
        expected_error_kind: ErrorKind::InvalidSyntax,
        must_contain: vec!["identifier", "invalid character"],
        must_suggest: Some("Identifiers can only contain alphanumeric characters"),
    },
    // ... 22 more stress tests
];
```

### 4. Property-Based Testing Specifications

```rust
// File: engram-core/tests/query_parser_property_tests.rs

use proptest::prelude::*;
use proptest::test_runner::Config as ProptestConfig;

/// Property 1: Parse-unparse-parse round-trip preserves semantics
/// For all valid queries Q: parse(unparse(parse(Q))) == parse(Q)
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        ..ProptestConfig::default()
    })]

    #[test]
    fn parse_unparse_roundtrip_preserves_semantics(
        query in valid_query_generator()
    ) {
        let ast1 = Parser::parse(&query).expect("first parse");
        let unparsed = ast1.to_query_string();
        let ast2 = Parser::parse(&unparsed).expect("second parse");

        prop_assert_eq!(ast1, ast2,
            "Round-trip failed:\nOriginal: {}\nUnparsed: {}\n",
            query, unparsed);
    }
}

/// Property 2: All invalid queries produce actionable errors
/// For all invalid queries Q: parse(Q).is_err() && error has suggestion
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        ..ProptestConfig::default()
    })]

    #[test]
    fn invalid_queries_always_have_suggestions(
        invalid_query in invalid_query_generator()
    ) {
        let result = Parser::parse(&invalid_query);

        prop_assert!(result.is_err(), "Expected parse failure");

        let error = result.unwrap_err();
        prop_assert!(!error.suggestion.is_empty(),
            "Error must have suggestion: {:?}", error);
        prop_assert!(!error.example.is_empty(),
            "Error must have example: {:?}", error);
        prop_assert!(error.position.line > 0,
            "Error must have valid position");
    }
}

/// Property 3: Parser is deterministic
/// For all queries Q: parse(Q) == parse(Q) (same result every time)
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        ..ProptestConfig::default()
    })]

    #[test]
    fn parser_is_deterministic(query in any_query_generator()) {
        let result1 = Parser::parse(&query);
        let result2 = Parser::parse(&query);

        prop_assert_eq!(result1, result2,
            "Parser must be deterministic for: {}", query);
    }
}

/// Property 4: Position tracking is accurate
/// For all queries Q: error.position points to actual error location
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 300,
        ..ProptestConfig::default()
    })]

    #[test]
    fn position_tracking_is_accurate(
        (query, error_offset) in query_with_injected_error()
    ) {
        let result = Parser::parse(&query);
        prop_assert!(result.is_err());

        let error = result.unwrap_err();
        let error_position = error.position.offset;

        // Error position should be within ±5 chars of injected error
        prop_assert!((error_position as isize - error_offset as isize).abs() <= 5,
            "Position tracking inaccurate: expected {}, got {}",
            error_offset, error_position);
    }
}

/// Property 5: Parser never panics
/// For all strings S: parse(S) returns Ok or Err (never panics)
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 10000,
        max_shrink_iters: 100000,
        ..ProptestConfig::default()
    })]

    #[test]
    fn parser_never_panics(arbitrary_input in "\\PC*") {
        // This should never panic, only return Ok or Err
        let _ = std::panic::catch_unwind(|| {
            let _ = Parser::parse(&arbitrary_input);
        }).expect("Parser panicked on input");
    }
}

/// Property 6: Keywords are case-insensitive
/// For all valid queries Q: parse(Q.lowercase()) == parse(Q.uppercase())
proptest! {
    #[test]
    fn keywords_are_case_insensitive(query in valid_query_generator()) {
        let lowercase = query.to_lowercase();
        let uppercase = query.to_uppercase();

        let ast_lower = Parser::parse(&lowercase);
        let ast_upper = Parser::parse(&uppercase);

        prop_assert_eq!(ast_lower, ast_upper,
            "Case sensitivity differs for: {}", query);
    }
}

/// Property 7: Whitespace normalization
/// For all valid queries Q: parse(Q) == parse(normalize_whitespace(Q))
proptest! {
    #[test]
    fn whitespace_is_normalized(query in valid_query_generator()) {
        let normalized = query.split_whitespace().collect::<Vec<_>>().join(" ");

        let ast1 = Parser::parse(&query);
        let ast2 = Parser::parse(&normalized);

        prop_assert_eq!(ast1, ast2,
            "Whitespace handling differs");
    }
}

// Generator functions
fn valid_query_generator() -> impl Strategy<Value = String> {
    prop_oneof![
        recall_query_generator(),
        spread_query_generator(),
        predict_query_generator(),
        imagine_query_generator(),
        consolidate_query_generator(),
    ]
}

fn recall_query_generator() -> impl Strategy<Value = String> {
    (
        identifier_generator(),
        option_constraints_generator(),
        option_confidence_generator(),
    ).prop_map(|(pattern, constraints, confidence)| {
        let mut query = format!("RECALL {}", pattern);
        if let Some(c) = constraints {
            query.push_str(&format!(" WHERE {}", c));
        }
        if let Some(conf) = confidence {
            query.push_str(&format!(" CONFIDENCE > {}", conf));
        }
        query
    })
}

fn spread_query_generator() -> impl Strategy<Value = String> {
    (
        identifier_generator(),
        option::of(1u16..=100u16),  // max_hops
        option::of(0.0f32..=1.0f32),  // decay
        option::of(0.0f32..=1.0f32),  // threshold
    ).prop_map(|(source, hops, decay, threshold)| {
        let mut query = format!("SPREAD FROM {}", source);
        if let Some(h) = hops {
            query.push_str(&format!(" MAX_HOPS {}", h));
        }
        if let Some(d) = decay {
            query.push_str(&format!(" DECAY {:.2}", d));
        }
        if let Some(t) = threshold {
            query.push_str(&format!(" THRESHOLD {:.2}", t));
        }
        query
    })
}

fn invalid_query_generator() -> impl Strategy<Value = String> {
    prop_oneof![
        // Typos in keywords
        Just("RECAL episode".to_string()),
        Just("SPRED FROM node".to_string()),

        // Missing required keywords
        Just("episode WHERE confidence > 0.7".to_string()),
        Just("SPREAD node_123".to_string()),

        // Invalid syntax
        Just("RECALL WHERE".to_string()),
        Just("SPREAD FROM".to_string()),

        // Out of range values
        (0.0f32..=2.0f32).prop_map(|v| format!("RECALL episode WHERE confidence > {}", v)),
    ]
}

fn query_with_injected_error() -> impl Strategy<Value = (String, usize)> {
    valid_query_generator().prop_flat_map(|query| {
        let error_offset = (0..query.len()).prop_map(move |offset| {
            let mut corrupted = query.clone();
            corrupted.insert(offset, '@');  // Inject invalid character
            (corrupted, offset)
        });
        error_offset
    })
}

fn identifier_generator() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_]{0,50}"
}
```

### 5. Fuzzing Infrastructure

```rust
// File: engram-core/fuzz/fuzz_targets/query_parser.rs

#![no_main]
use libfuzzer_sys::fuzz_target;
use engram_core::query::parser::Parser;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        // Parser should never panic on any input
        let _ = Parser::parse(s);
    }
});

// File: engram-core/fuzz/fuzz_targets/query_parser_structured.rs

#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct FuzzQuery {
    operation: FuzzOperation,
    pattern: String,
    constraints: Vec<FuzzConstraint>,
}

#[derive(Arbitrary, Debug)]
enum FuzzOperation {
    Recall,
    Spread,
    Predict,
    Imagine,
    Consolidate,
}

#[derive(Arbitrary, Debug)]
struct FuzzConstraint {
    field: String,
    operator: FuzzOperator,
    value: FuzzValue,
}

#[derive(Arbitrary, Debug)]
enum FuzzOperator {
    GreaterThan,
    LessThan,
    Equal,
}

#[derive(Arbitrary, Debug)]
enum FuzzValue {
    Float(f32),
    String(String),
    Embedding(Vec<f32>),
}

fuzz_target!(|fuzz_query: FuzzQuery| {
    let query_string = fuzz_query.to_query_string();
    let _ = Parser::parse(&query_string);
});
```

### 6. Performance Regression Tests

```rust
// File: engram-core/benches/query_parser_performance.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use engram_core::query::parser::Parser;

fn benchmark_parse_times(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser_performance");

    // Set strict performance targets
    group.significance_level(0.01)
        .sample_size(1000)
        .measurement_time(Duration::from_secs(10));

    let queries = vec![
        ("simple_recall", "RECALL episode"),
        ("recall_with_constraints", "RECALL episode WHERE confidence > 0.7"),
        ("complex_spread", "SPREAD FROM node MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1"),
        ("large_embedding", &format!("RECALL episode WHERE content SIMILAR TO [{}]",
            (0..768).map(|i| format!("{:.6}", i as f32 / 768.0)).collect::<Vec<_>>().join(", "))),
        ("multiline", "RECALL episode\n  WHERE confidence > 0.7\n  AND created > \"2024-01-01\""),
    ];

    for (name, query) in queries {
        group.bench_with_input(BenchmarkId::new("parse", name), &query, |b, q| {
            b.iter(|| Parser::parse(black_box(q)))
        });
    }

    group.finish();
}

fn benchmark_parse_regression(c: &mut Criterion) {
    // Regression test: fail CI if parse time exceeds baseline by >10%
    let baseline = Duration::from_micros(100);
    let query = "RECALL episode WHERE confidence > 0.7";

    c.bench_function("parse_regression_guard", |b| {
        b.iter(|| {
            let start = Instant::now();
            let _ = Parser::parse(black_box(query));
            let elapsed = start.elapsed();

            assert!(elapsed < baseline * 110 / 100,
                "Parse time regression: {:?} exceeds baseline {:?} by >10%",
                elapsed, baseline);
        });
    });
}

criterion_group!(benches, benchmark_parse_times, benchmark_parse_regression);
criterion_main!(benches);
```

### 7. Error Message Validation Framework

```rust
// File: engram-core/tests/error_message_validation.rs

/// Validates that every error message meets quality standards
#[test]
fn all_errors_have_actionable_messages() {
    for test in SYNTAX_ERROR_QUERIES.iter()
        .chain(SEMANTIC_ERROR_QUERIES.iter())
        .chain(STRESS_TEST_QUERIES.iter())
    {
        let result = Parser::parse(test.query);
        assert!(result.is_err(), "Expected error for: {}", test.name);

        let error = result.unwrap_err();
        let error_msg = error.to_string();

        // Verify required content
        for required in &test.must_contain {
            assert!(error_msg.contains(required),
                "Error for '{}' must contain '{}'\nGot: {}",
                test.name, required, error_msg);
        }

        // Verify suggestion if specified
        if let Some(suggestion) = test.must_suggest {
            assert!(error.suggestion.contains(suggestion) || error_msg.contains(suggestion),
                "Error for '{}' must suggest '{}'\nGot: {}",
                test.name, suggestion, error_msg);
        }

        // Verify position information
        assert!(error.position.line > 0, "Must have line number");
        assert!(error.position.column > 0, "Must have column number");

        // Verify example is provided
        assert!(!error.example.is_empty(),
            "Error for '{}' must provide example", test.name);
    }
}

/// Validates error message consistency across similar errors
#[test]
fn error_messages_are_consistent() {
    let typo_tests = vec![
        ("RECAL episode", "RECALL"),
        ("SPRED FROM node", "SPREAD"),
        ("IMAGIN episode", "IMAGINE"),
    ];

    for (query, expected) in typo_tests {
        let error = Parser::parse(query).unwrap_err();
        let msg = error.to_string();

        assert!(msg.contains("Did you mean"));
        assert!(msg.contains(expected));
        assert!(msg.contains("Example:"));
    }
}

/// Validates that error positions are accurate
#[test]
fn error_positions_are_accurate() {
    let tests = vec![
        ("RECALL\n  episode\n  INVALID", 3, "INVALID"),  // Line 3
        ("SPREAD FROM node @invalid", 1, "@"),           // Column with @
    ];

    for (query, expected_line, error_token) in tests {
        let error = Parser::parse(query).unwrap_err();

        assert_eq!(error.position.line, expected_line,
            "Wrong line number for error near '{}'", error_token);
    }
}
```

---

## Files to Create/Modify

### New Files
1. **Create**: `engram-core/tests/query_language_corpus.rs`
   - Main test corpus with 150+ queries
   - Valid and invalid query test cases
   - Error message validation tests

2. **Create**: `engram-core/tests/query_parser_property_tests.rs`
   - Property-based tests using proptest
   - 7 core properties with 1000+ cases each

3. **Create**: `engram-core/tests/error_message_validation.rs`
   - Systematic error message quality validation
   - Position accuracy tests
   - Consistency checks

4. **Create**: `engram-core/benches/query_parser_performance.rs`
   - Performance benchmarks with regression guards
   - Parse time tracking for CI

5. **Create**: `engram-core/fuzz/fuzz_targets/query_parser.rs`
   - Basic fuzzer for arbitrary input
   - Structured fuzzer with grammar-aware generation

6. **Create**: `engram-core/fuzz/Cargo.toml`
   - Fuzzing configuration

### Modified Files
1. **Modify**: `engram-core/Cargo.toml`
   - Add fuzzing dependencies
   - Add arbitrary for structured fuzzing

---

## Testing Strategy

### Unit Tests (50+ tests)
- Tokenizer correctness (20 tests)
- AST construction (15 tests)
- Parser edge cases (15 tests)

### Integration Tests (150+ tests)
- Valid query corpus (75 tests)
- Invalid query corpus (75 tests)

### Property-Based Tests (7 properties × 1000 cases = 7000 tests)
1. Round-trip preservation
2. Actionable error messages
3. Determinism
4. Position accuracy
5. No panics
6. Case insensitivity
7. Whitespace normalization

### Fuzzing (Continuous)
- Run 1M iterations minimum
- Coverage-guided fuzzing
- Structured fuzzing for deeper testing

### Performance Tests
- Benchmark all query types
- Regression guard: fail CI if >10% slower
- Target: <100μs P90, <200μs P99

---

## Acceptance Criteria

- [ ] 150+ test queries (75 valid, 75 invalid) with full coverage
- [ ] 100% of invalid queries produce actionable error messages
  - Every error has: line, column, suggestion, example
  - Typo detection works (Levenshtein distance ≤2)
  - Context-aware expected tokens
- [ ] All valid queries parse in <100μs (P90)
- [ ] Property-based tests pass 1000+ cases per property
- [ ] Fuzzer runs 1M iterations without finding crashes
- [ ] Performance benchmarks in CI fail on >10% regression
- [ ] Error message validation framework passes 100% of tests
- [ ] Zero clippy warnings
- [ ] Test coverage >95% for parser module

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Parse time (simple) | <50μs | Criterion |
| Parse time (complex) | <100μs P90 | Criterion |
| Parse time (large embedding) | <200μs P99 | Criterion |
| Fuzzer throughput | >10k exec/sec | cargo fuzz |
| Property test time | <5min for 7k cases | proptest |
| Memory per parse | <1KB | Manual profiling |

---

## Integration Points

- **Task 001-003**: Uses Parser implementation
- **Task 004**: Validates error messages and recovery
- **Task 006-007**: Tests query execution integration
- **Task 010**: Performance optimization baseline
- **Task 012**: End-to-end integration tests

---

## Differential Testing Strategy

If multiple parser implementations exist (e.g., experimental vs. production):

```rust
#[test]
fn differential_testing_against_reference() {
    for query in &VALID_QUERIES {
        let ast_main = Parser::parse(query.query).unwrap();
        let ast_reference = ReferenceParser::parse(query.query).unwrap();

        assert_eq!(ast_main, ast_reference,
            "Parser divergence on: {}", query.name);
    }
}
```

---

## Fuzzing Execution Plan

```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# Run basic fuzzer (1M iterations minimum)
cargo fuzz run query_parser -- -runs=1000000

# Run structured fuzzer (generates valid-looking queries)
cargo fuzz run query_parser_structured -- -runs=1000000

# Coverage-guided fuzzing (overnight run)
cargo fuzz run query_parser -- -max_total_time=28800  # 8 hours

# Minimize any crashes found
cargo fuzz cmin query_parser
```

---

## Continuous Integration

Add to `.github/workflows/ci.yml`:

```yaml
- name: Run query parser tests
  run: |
    cargo test --test query_language_corpus
    cargo test --test query_parser_property_tests
    cargo test --test error_message_validation

- name: Run parser benchmarks (regression check)
  run: |
    cargo bench --bench query_parser_performance -- --save-baseline current

- name: Run fuzzer (smoke test)
  run: |
    cargo install cargo-fuzz
    cargo fuzz run query_parser -- -runs=10000 -max_total_time=60
```

---

## References

- Property-based testing: https://hypothesis.works/articles/what-is-property-based-testing/
- Fuzzing best practices: https://rust-fuzz.github.io/book/
- Parser testing patterns: https://matklad.github.io/2018/06/06/modern-parser-generator.html
- Error message design: https://elm-lang.org/news/compiler-errors-for-humans
- Levenshtein distance: https://en.wikipedia.org/wiki/Levenshtein_distance
