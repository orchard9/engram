//! Integration tests for the cognitive query parser.
//!
//! These tests validate end-to-end parsing of all query types and edge cases.

use engram_core::query::parser::{ConfidenceThreshold, Parser, Pattern, Query};

#[test]
fn test_parse_all_example_queries() {
    let queries = vec![
        "RECALL episode",
        "RECALL episode CONFIDENCE > 0.7",
        "SPREAD FROM cue_node MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1",
        "PREDICT episode GIVEN context1, context2 HORIZON 3600",
        "IMAGINE episode BASED ON seed1, seed2 NOVELTY 0.3",
        "CONSOLIDATE episode INTO semantic_memory",
    ];

    for query in queries {
        let result = Parser::parse(query);
        assert!(result.is_ok(), "Failed to parse: {query}");
    }
}

#[test]
fn test_recall_with_multiple_constraints() {
    let query = "RECALL episode CONFIDENCE > 0.7";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Recall(recall) => {
            assert!(matches!(recall.pattern, Pattern::NodeId(_)));
            assert!(matches!(
                recall.confidence_threshold,
                Some(ConfidenceThreshold::Above(_))
            ));
        }
        _ => panic!("Expected Recall query"),
    }
}

#[test]
fn test_recall_with_base_rate() {
    let query = "RECALL episode BASE_RATE 0.5";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Recall(recall) => {
            assert!(recall.base_rate.is_some());
            assert!((recall.base_rate.unwrap().raw() - 0.5).abs() < 1e-6);
        }
        _ => panic!("Expected Recall query"),
    }
}

#[test]
fn test_recall_with_embedding_768_dimensions() {
    // Simulate a 768-dimensional embedding (truncated for test readability)
    let mut parts = vec!["RECALL [".to_string()];
    for i in 0..768 {
        if i > 0 {
            parts.push(format!(", {:.3}", i as f32 / 768.0));
        } else {
            parts.push(format!("{:.3}", i as f32 / 768.0));
        }
    }
    parts.push("] THRESHOLD 0.85".to_string());
    let embedding_str = parts.concat();

    let ast = Parser::parse(&embedding_str).unwrap();

    match ast {
        Query::Recall(recall) => {
            if let Pattern::Embedding { vector, threshold } = recall.pattern {
                assert_eq!(vector.len(), 768);
                assert!((threshold - 0.85).abs() < 1e-6);
            } else {
                panic!("Expected embedding pattern");
            }
        }
        _ => panic!("Expected Recall query"),
    }
}

#[test]
fn test_spread_minimal() {
    let query = "SPREAD FROM node_123";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Spread(spread) => {
            assert_eq!(spread.source.as_str(), "node_123");
            assert!(spread.max_hops.is_none());
            assert!(spread.decay_rate.is_none());
            assert!(spread.activation_threshold.is_none());
        }
        _ => panic!("Expected Spread query"),
    }
}

#[test]
fn test_spread_full_options() {
    let query = "SPREAD FROM source_node MAX_HOPS 10 DECAY 0.2 THRESHOLD 0.05";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Spread(spread) => {
            assert_eq!(spread.source.as_str(), "source_node");
            assert_eq!(spread.max_hops, Some(10));
            assert!((spread.decay_rate.unwrap() - 0.2).abs() < 1e-6);
            assert!((spread.activation_threshold.unwrap() - 0.05).abs() < 1e-6);
        }
        _ => panic!("Expected Spread query"),
    }
}

#[test]
fn test_predict_single_context() {
    let query = "PREDICT outcome GIVEN context1";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Predict(predict) => {
            assert_eq!(predict.context.len(), 1);
            assert_eq!(predict.context[0].as_str(), "context1");
        }
        _ => panic!("Expected Predict query"),
    }
}

#[test]
fn test_predict_multiple_contexts() {
    let query = "PREDICT outcome GIVEN ctx1, ctx2, ctx3, ctx4";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Predict(predict) => {
            assert_eq!(predict.context.len(), 4);
        }
        _ => panic!("Expected Predict query"),
    }
}

#[test]
fn test_imagine_without_seeds() {
    let query = "IMAGINE novel_concept NOVELTY 0.8";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Imagine(imagine) => {
            assert!(imagine.seeds.is_empty());
            assert!((imagine.novelty.unwrap() - 0.8).abs() < 1e-6);
        }
        _ => panic!("Expected Imagine query"),
    }
}

#[test]
fn test_imagine_with_confidence() {
    let query = "IMAGINE concept BASED ON seed1 NOVELTY 0.5 CONFIDENCE > 0.6";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Imagine(imagine) => {
            assert_eq!(imagine.seeds.len(), 1);
            assert!((imagine.novelty.unwrap() - 0.5).abs() < 1e-6);
            assert!(matches!(
                imagine.confidence_threshold,
                Some(ConfidenceThreshold::Above(_))
            ));
        }
        _ => panic!("Expected Imagine query"),
    }
}

#[test]
fn test_consolidate_with_pattern() {
    let query = "CONSOLIDATE episode INTO semantic_node";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Consolidate(consolidate) => {
            assert_eq!(consolidate.target.as_str(), "semantic_node");
        }
        _ => panic!("Expected Consolidate query"),
    }
}

#[test]
fn test_pattern_node_id() {
    let query = "RECALL episode_12345";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Recall(recall) => {
            if let Pattern::NodeId(id) = recall.pattern {
                assert_eq!(id.as_str(), "episode_12345");
            } else {
                panic!("Expected NodeId pattern");
            }
        }
        _ => panic!("Expected Recall query"),
    }
}

#[test]
fn test_pattern_string_match() {
    let query = r#"RECALL "neural networks and deep learning""#;
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Recall(recall) => {
            assert!(matches!(recall.pattern, Pattern::ContentMatch(_)));
        }
        _ => panic!("Expected Recall query"),
    }
}

#[test]
fn test_embedding_with_default_threshold() {
    // Small embeddings should now fail validation
    let query = "RECALL [0.1, 0.2, 0.3]";
    let result = Parser::parse(query);
    assert!(result.is_err(), "Small embedding should fail validation");

    // Valid 768-dimensional embedding with default threshold
    let mut values = Vec::new();
    for i in 0..768 {
        values.push(format!("{:.3}", i as f32 / 768.0));
    }
    let query = format!("RECALL [{}]", values.join(", "));
    let ast = Parser::parse(&query).unwrap();

    match ast {
        Query::Recall(recall) => {
            if let Pattern::Embedding { vector, threshold } = recall.pattern {
                assert_eq!(vector.len(), 768);
                assert!((threshold - 0.8).abs() < 1e-6); // Default threshold
            } else {
                panic!("Expected embedding pattern");
            }
        }
        _ => panic!("Expected Recall query"),
    }
}

#[test]
fn test_embedding_with_explicit_threshold() {
    // Create 768-dimensional embedding
    let mut values = Vec::new();
    for _ in 0..768 {
        values.push("0.5".to_string());
    }
    let query = format!("RECALL [{}] THRESHOLD 0.95", values.join(", "));
    let ast = Parser::parse(&query).unwrap();

    match ast {
        Query::Recall(recall) => {
            if let Pattern::Embedding { vector, threshold } = recall.pattern {
                assert_eq!(vector.len(), 768);
                assert!((threshold - 0.95).abs() < 1e-6);
            } else {
                panic!("Expected embedding pattern");
            }
        }
        _ => panic!("Expected Recall query"),
    }
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_error_missing_pattern() {
    let query = "RECALL";
    let result = Parser::parse(query);
    assert!(result.is_err());
}

#[test]
fn test_error_invalid_operator() {
    let query = "RECALL episode WHERE confidence * 0.5";
    let result = Parser::parse(query);
    // Should fail because '*' is not a valid operator
    assert!(result.is_err());
}

#[test]
fn test_error_empty_embedding() {
    let query = "RECALL []";
    let result = Parser::parse(query);
    assert!(result.is_err());
}

#[test]
fn test_error_unterminated_embedding() {
    let query = "RECALL [0.1, 0.2";
    let result = Parser::parse(query);
    assert!(result.is_err());
}

#[test]
fn test_error_missing_from_in_spread() {
    let query = "SPREAD node_123";
    let result = Parser::parse(query);
    assert!(result.is_err());
}

#[test]
fn test_error_missing_given_in_predict() {
    let query = "PREDICT outcome";
    let result = Parser::parse(query);
    assert!(result.is_err());
}

#[test]
fn test_error_missing_into_in_consolidate() {
    let query = "CONSOLIDATE episodes";
    let result = Parser::parse(query);
    assert!(result.is_err());
}

#[test]
fn test_error_unexpected_keyword() {
    let query = "RETRIEVE episode"; // Invalid keyword
    let result = Parser::parse(query);
    assert!(result.is_err());
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_single_element_embedding() {
    // Single-element embedding should fail validation (expected 768)
    let query = "RECALL [1.0]";
    let result = Parser::parse(query);
    assert!(
        result.is_err(),
        "Single-element embedding should fail validation"
    );
}

#[test]
fn test_large_embedding() {
    // Test with exact system dimension (768)
    let mut parts = vec!["RECALL [".to_string()];
    for i in 0..768 {
        if i > 0 {
            parts.push(format!(", {:.3}", i as f32 / 768.0));
        } else {
            parts.push(format!("{:.3}", i as f32 / 768.0));
        }
    }
    parts.push("]".to_string());
    let embedding_str = parts.concat();

    let ast = Parser::parse(&embedding_str).unwrap();

    match ast {
        Query::Recall(recall) => {
            if let Pattern::Embedding { vector, .. } = recall.pattern {
                assert_eq!(vector.len(), 768);
            } else {
                panic!("Expected embedding pattern");
            }
        }
        _ => panic!("Expected Recall query"),
    }
}

#[test]
fn test_integer_confidence_value() {
    let query = "RECALL episode CONFIDENCE > 1";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Recall(recall) => {
            assert!(matches!(
                recall.confidence_threshold,
                Some(ConfidenceThreshold::Above(_))
            ));
        }
        _ => panic!("Expected Recall query"),
    }
}

#[test]
fn test_zero_confidence() {
    let query = "RECALL episode CONFIDENCE > 0";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Recall(recall) => {
            if let Some(ConfidenceThreshold::Above(conf)) = recall.confidence_threshold {
                assert!((conf.raw() - 0.0).abs() < 1e-6);
            } else {
                panic!("Expected confidence threshold");
            }
        }
        _ => panic!("Expected Recall query"),
    }
}

#[test]
fn test_node_id_with_underscores() {
    let query = "RECALL my_node_id_123_abc";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Recall(recall) => {
            if let Pattern::NodeId(id) = recall.pattern {
                assert_eq!(id.as_str(), "my_node_id_123_abc");
            } else {
                panic!("Expected NodeId pattern");
            }
        }
        _ => panic!("Expected Recall query"),
    }
}

#[test]
fn test_spread_max_hops_boundary() {
    // Test valid boundary (MAX_HOPS <= 100)
    let query = "SPREAD FROM node MAX_HOPS 100";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Spread(spread) => {
            assert_eq!(spread.max_hops, Some(100));
        }
        _ => panic!("Expected Spread query"),
    }
}

#[test]
fn test_spread_max_hops_overflow() {
    let query = "SPREAD FROM node MAX_HOPS 65536"; // Overflow u16
    let result = Parser::parse(query);
    assert!(result.is_err()); // Should fail due to u16 overflow
}

#[test]
fn test_predict_with_horizon_zero() {
    let query = "PREDICT outcome GIVEN context HORIZON 0";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Predict(predict) => {
            assert_eq!(predict.horizon, Some(std::time::Duration::from_secs(0)));
        }
        _ => panic!("Expected Predict query"),
    }
}

#[test]
fn test_case_insensitive_keywords() {
    let queries = vec![
        "recall episode",
        "RECALL episode",
        "Recall episode",
        "ReCaLl episode",
    ];

    for query in queries {
        let result = Parser::parse(query);
        assert!(result.is_ok(), "Failed to parse case-insensitive: {query}");
    }
}

#[test]
fn test_whitespace_handling() {
    let queries = vec![
        "RECALL episode",
        "RECALL  episode",    // Double space
        "RECALL\tepisode",    // Tab
        "RECALL\nepisode",    // Newline
        "  RECALL episode  ", // Leading/trailing whitespace
    ];

    for query in queries {
        let result = Parser::parse(query);
        assert!(result.is_ok(), "Failed to parse with whitespace: {query:?}");
    }
}

#[test]
fn test_string_with_escapes() {
    let query = r#"RECALL "line one\nline two\ttab""#;
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Recall(recall) => {
            if let Pattern::ContentMatch(content) = recall.pattern {
                assert!(content.contains('\n'));
                assert!(content.contains('\t'));
            } else {
                panic!("Expected ContentMatch pattern");
            }
        }
        _ => panic!("Expected Recall query"),
    }
}

#[test]
fn test_multiline_query() {
    let query = "SPREAD\n  FROM\n    node_123\n  MAX_HOPS\n    5";
    let ast = Parser::parse(query).unwrap();

    match ast {
        Query::Spread(spread) => {
            assert_eq!(spread.source.as_str(), "node_123");
            assert_eq!(spread.max_hops, Some(5));
        }
        _ => panic!("Expected Spread query"),
    }
}

#[test]
fn test_query_categories() {
    let queries = vec![
        ("RECALL episode", "Recall"),
        ("SPREAD FROM node", "Spread"),
        ("PREDICT outcome GIVEN ctx", "Predict"),
        ("IMAGINE concept", "Imagine"),
        ("CONSOLIDATE ep INTO sem", "Consolidate"),
    ];

    for (query_str, expected_category) in queries {
        let query = Parser::parse(query_str).unwrap();
        let category = format!("{:?}", query.category());
        assert_eq!(category, expected_category);
    }
}

#[test]
fn test_read_only_queries() {
    let read_only = vec![
        "RECALL episode",
        "SPREAD FROM node",
        "PREDICT outcome GIVEN ctx",
    ];

    for query_str in read_only {
        let query = Parser::parse(query_str).unwrap();
        assert!(query.is_read_only(), "{query_str} should be read-only");
    }

    let mutating = vec!["IMAGINE concept", "CONSOLIDATE ep INTO sem"];

    for query_str in mutating {
        let query = Parser::parse(query_str).unwrap();
        assert!(!query.is_read_only(), "{query_str} should be mutating");
    }
}
