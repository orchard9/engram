//! Comprehensive tests for AST types and builders.

#![allow(clippy::unwrap_used)] // Tests are allowed to use unwrap
#![allow(clippy::panic)] // Tests are allowed to panic
#![allow(clippy::float_cmp)] // Tests need exact float comparisons
#![allow(clippy::redundant_clone)] // Tests sometimes clone for clarity

use super::*;
use crate::Confidence;
use std::borrow::Cow;
use std::time::{Duration, SystemTime};

// ============================================================================
// Unit Tests
// ============================================================================

#[test]
fn test_recall_query_construction() {
    let query = RecallQuery {
        pattern: Pattern::NodeId(NodeIdentifier::from("episode_123")),
        constraints: vec![Constraint::ConfidenceAbove(Confidence::from_raw(0.7))],
        confidence_threshold: Some(ConfidenceThreshold::Above(Confidence::from_raw(0.8))),
        base_rate: None,
        limit: Some(10),
    };

    assert_eq!(
        query.pattern,
        Pattern::NodeId(NodeIdentifier::from("episode_123"))
    );
    assert_eq!(query.constraints.len(), 1);
    assert!(query.limit == Some(10));
}

#[test]
fn test_pattern_validation() {
    // Valid embedding with correct dimensions
    let pattern = Pattern::Embedding {
        vector: vec![0.1; 768],
        threshold: 0.8,
    };
    assert!(pattern.validate().is_ok());

    // Invalid: empty embedding
    let pattern = Pattern::Embedding {
        vector: vec![],
        threshold: 0.8,
    };
    assert!(matches!(
        pattern.validate(),
        Err(ValidationError::EmptyEmbedding)
    ));

    // Invalid: threshold out of range
    let pattern = Pattern::Embedding {
        vector: vec![0.1; 768],
        threshold: 1.5,
    };
    assert!(matches!(
        pattern.validate(),
        Err(ValidationError::InvalidThreshold(_))
    ));

    // Invalid: wrong dimension
    let pattern = Pattern::Embedding {
        vector: vec![0.1, 0.2, 0.3],
        threshold: 0.8,
    };
    assert!(matches!(
        pattern.validate(),
        Err(ValidationError::InvalidEmbeddingDimension { .. })
    ));

    // Valid: ContentMatch
    let pattern = Pattern::ContentMatch(Cow::Borrowed("test"));
    assert!(pattern.validate().is_ok());

    // Invalid: Empty ContentMatch
    let pattern = Pattern::ContentMatch(Cow::Borrowed(""));
    assert!(matches!(
        pattern.validate(),
        Err(ValidationError::EmptyContentMatch)
    ));

    // Valid: Any pattern
    let pattern = Pattern::<'static>::Any;
    assert!(pattern.validate().is_ok());
}

#[test]
fn test_confidence_threshold_matching() {
    let threshold = ConfidenceThreshold::Above(Confidence::from_raw(0.7));
    assert!(threshold.matches(Confidence::from_raw(0.8)));
    assert!(!threshold.matches(Confidence::from_raw(0.6)));

    let threshold = ConfidenceThreshold::Below(Confidence::from_raw(0.3));
    assert!(threshold.matches(Confidence::from_raw(0.2)));
    assert!(!threshold.matches(Confidence::from_raw(0.4)));

    let threshold = ConfidenceThreshold::Between {
        lower: Confidence::from_raw(0.5),
        upper: Confidence::from_raw(0.9),
    };
    assert!(threshold.matches(Confidence::from_raw(0.7)));
    assert!(!threshold.matches(Confidence::from_raw(0.3)));
    assert!(!threshold.matches(Confidence::from_raw(0.95)));
}

#[test]
fn test_confidence_interval_validation() {
    // Valid interval
    let interval = ConfidenceInterval::new(Confidence::from_raw(0.5), Confidence::from_raw(0.9));
    assert!(interval.is_ok());

    // Invalid: lower > upper
    let interval = ConfidenceInterval::new(Confidence::from_raw(0.9), Confidence::from_raw(0.5));
    assert!(matches!(
        interval,
        Err(ValidationError::InvalidInterval { .. })
    ));
}

#[test]
fn test_confidence_interval_contains() {
    let interval =
        ConfidenceInterval::new(Confidence::from_raw(0.5), Confidence::from_raw(0.9)).unwrap();

    assert!(interval.contains(Confidence::from_raw(0.7)));
    assert!(interval.contains(Confidence::from_raw(0.5)));
    assert!(interval.contains(Confidence::from_raw(0.9)));
    assert!(!interval.contains(Confidence::from_raw(0.3)));
    assert!(!interval.contains(Confidence::from_raw(0.95)));
}

#[test]
fn test_node_identifier_validation() {
    let id = NodeIdentifier::new("valid_id");
    assert!(id.validate().is_ok());

    let id = NodeIdentifier::new("");
    assert!(matches!(id.validate(), Err(ValidationError::EmptyNodeId)));

    // Too long
    let long_id = "a".repeat(300);
    let id = NodeIdentifier::new(long_id);
    assert!(matches!(
        id.validate(),
        Err(ValidationError::NodeIdTooLong { .. })
    ));
}

#[test]
fn test_node_identifier_zero_copy() {
    let source = "test_node";
    let id = NodeIdentifier::borrowed(source);
    assert_eq!(id.as_str(), source);

    // Verify it's actually borrowed
    assert!(matches!(id.0, Cow::Borrowed(_)));
}

#[test]
fn test_node_identifier_owned() {
    let owned = String::from("owned_node");
    let id = NodeIdentifier::owned(owned.clone());
    assert_eq!(id.as_str(), "owned_node");

    // Verify it's actually owned
    assert!(matches!(id.0, Cow::Owned(_)));
}

#[test]
fn test_node_identifier_into_owned() {
    let source = "borrowed";
    let borrowed_id = NodeIdentifier::borrowed(source);
    let owned_id = borrowed_id.into_owned();

    assert_eq!(owned_id.as_str(), "borrowed");
    assert!(matches!(owned_id.0, Cow::Owned(_)));
}

#[test]
fn test_estimated_cost() {
    let recall = Query::Recall(RecallQuery {
        pattern: Pattern::Any,
        constraints: vec![],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    });
    assert_eq!(recall.estimated_cost(), 1000);

    // Recall with constraints
    let recall_with_constraints = Query::Recall(RecallQuery {
        pattern: Pattern::Any,
        constraints: vec![
            Constraint::ConfidenceAbove(Confidence::from_raw(0.5)),
            Constraint::ConfidenceAbove(Confidence::from_raw(0.7)),
        ],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    });
    assert_eq!(recall_with_constraints.estimated_cost(), 1200); // 1000 + 2*100

    let spread = Query::Spread(SpreadQuery {
        source: NodeIdentifier::new("node"),
        max_hops: Some(3),
        decay_rate: None,
        activation_threshold: None,
        refractory_period: None,
    });
    assert!(spread.estimated_cost() > recall.estimated_cost());
}

#[test]
fn test_spread_query_defaults() {
    let query = SpreadQuery {
        source: NodeIdentifier::new("test"),
        max_hops: None,
        decay_rate: None,
        activation_threshold: None,
        refractory_period: None,
    };

    assert_eq!(
        query.effective_decay_rate(),
        SpreadQuery::DEFAULT_DECAY_RATE
    );
    assert_eq!(query.effective_threshold(), SpreadQuery::DEFAULT_THRESHOLD);
}

#[test]
fn test_spread_query_validation() {
    let query = SpreadQuery {
        source: NodeIdentifier::new("test"),
        max_hops: Some(3),
        decay_rate: Some(0.5),
        activation_threshold: Some(0.1),
        refractory_period: None,
    };
    assert!(query.validate().is_ok());

    // Invalid decay rate
    let query = SpreadQuery {
        source: NodeIdentifier::new("test"),
        max_hops: Some(3),
        decay_rate: Some(1.5),
        activation_threshold: Some(0.1),
        refractory_period: None,
    };
    assert!(matches!(
        query.validate(),
        Err(ValidationError::InvalidDecayRate(_))
    ));

    // Invalid activation threshold
    let query = SpreadQuery {
        source: NodeIdentifier::new("test"),
        max_hops: Some(3),
        decay_rate: Some(0.5),
        activation_threshold: Some(-0.1),
        refractory_period: None,
    };
    assert!(matches!(
        query.validate(),
        Err(ValidationError::InvalidActivationThreshold(_))
    ));
}

#[test]
fn test_query_category() {
    let recall = Query::Recall(RecallQuery {
        pattern: Pattern::Any,
        constraints: vec![],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    });
    assert_eq!(recall.category(), QueryCategory::Recall);

    let spread = Query::Spread(SpreadQuery {
        source: NodeIdentifier::new("node"),
        max_hops: None,
        decay_rate: None,
        activation_threshold: None,
        refractory_period: None,
    });
    assert_eq!(spread.category(), QueryCategory::Spread);
}

#[test]
fn test_query_is_read_only() {
    let recall = Query::Recall(RecallQuery {
        pattern: Pattern::Any,
        constraints: vec![],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    });
    assert!(recall.is_read_only());

    let consolidate = Query::Consolidate(ConsolidateQuery {
        episodes: EpisodeSelector::All,
        target: NodeIdentifier::new("target"),
        scheduler_policy: None,
    });
    assert!(!consolidate.is_read_only());
}

#[test]
fn test_constraint_validation() {
    // Valid SimilarTo
    let constraint = Constraint::SimilarTo {
        embedding: vec![0.1, 0.2, 0.3],
        threshold: 0.8,
    };
    assert!(constraint.validate().is_ok());

    // Invalid: empty embedding
    let constraint = Constraint::SimilarTo {
        embedding: vec![],
        threshold: 0.8,
    };
    assert!(matches!(
        constraint.validate(),
        Err(ValidationError::EmptyEmbedding)
    ));

    // Invalid: threshold out of range
    let constraint = Constraint::SimilarTo {
        embedding: vec![0.1, 0.2],
        threshold: 2.0,
    };
    assert!(matches!(
        constraint.validate(),
        Err(ValidationError::InvalidThreshold(_))
    ));

    // Valid: ContentContains
    let constraint = Constraint::ContentContains(Cow::Borrowed("test"));
    assert!(constraint.validate().is_ok());

    // Invalid: empty ContentContains
    let constraint = Constraint::ContentContains(Cow::Borrowed(""));
    assert!(matches!(
        constraint.validate(),
        Err(ValidationError::EmptyContentMatch)
    ));
}

#[test]
fn test_imagine_query_validation() {
    let query = ImagineQuery {
        pattern: Pattern::Any,
        seeds: vec![NodeIdentifier::new("seed1")],
        novelty: Some(0.5),
        confidence_threshold: None,
    };
    assert!(query.validate().is_ok());

    // Invalid novelty
    let query = ImagineQuery {
        pattern: Pattern::Any,
        seeds: vec![],
        novelty: Some(1.5),
        confidence_threshold: None,
    };
    assert!(matches!(
        query.validate(),
        Err(ValidationError::InvalidNovelty(_))
    ));
}

#[test]
fn test_episode_selector_validation() {
    let selector = EpisodeSelector::<'static>::All;
    assert!(selector.validate().is_ok());

    let selector = EpisodeSelector::Pattern(Pattern::Any);
    assert!(selector.validate().is_ok());

    let selector =
        EpisodeSelector::Where(vec![Constraint::ConfidenceAbove(Confidence::from_raw(0.5))]);
    assert!(selector.validate().is_ok());
}

// ============================================================================
// Builder Pattern Tests
// ============================================================================

#[test]
fn test_recall_query_builder() {
    let query = RecallQueryBuilder::new()
        .pattern(Pattern::NodeId(NodeIdentifier::from("episode_123")))
        .constraint(Constraint::ConfidenceAbove(Confidence::from_raw(0.7)))
        .limit(10)
        .build()
        .unwrap();

    assert_eq!(
        query.pattern,
        Pattern::NodeId(NodeIdentifier::from("episode_123"))
    );
    assert_eq!(query.constraints.len(), 1);
    assert_eq!(query.limit, Some(10));
}

#[test]
fn test_recall_query_builder_multiple_constraints() {
    let query = RecallQueryBuilder::new()
        .pattern(Pattern::Any)
        .constraint(Constraint::ConfidenceAbove(Confidence::from_raw(0.5)))
        .constraint(Constraint::ContentContains(Cow::Owned("test".to_string())))
        .build()
        .unwrap();

    assert_eq!(query.constraints.len(), 2);
}

#[test]
fn test_recall_query_builder_with_all_options() {
    let query = RecallQueryBuilder::new()
        .pattern(Pattern::NodeId(NodeIdentifier::from("test")))
        .constraint(Constraint::ConfidenceAbove(Confidence::from_raw(0.7)))
        .confidence_threshold(ConfidenceThreshold::Above(Confidence::from_raw(0.8)))
        .base_rate(Confidence::from_raw(0.5))
        .limit(100)
        .build()
        .unwrap();

    assert!(query.confidence_threshold.is_some());
    assert!(query.base_rate.is_some());
    assert_eq!(query.limit, Some(100));
}

#[test]
fn test_recall_query_builder_validation_failure() {
    // Invalid pattern should fail validation
    let result = RecallQueryBuilder::new()
        .pattern(Pattern::Embedding {
            vector: vec![],
            threshold: 0.8,
        })
        .build();

    assert!(result.is_err());
}

// ============================================================================
// Into Owned Tests
// ============================================================================

#[test]
fn test_pattern_into_owned() {
    let source = "test_node";
    let borrowed_pattern = Pattern::NodeId(NodeIdentifier::borrowed(source));
    let owned_pattern = borrowed_pattern.into_owned();

    match owned_pattern {
        Pattern::NodeId(id) => {
            assert_eq!(id.as_str(), "test_node");
            assert!(matches!(id.0, Cow::Owned(_)));
        }
        _ => panic!("Expected NodeId pattern"),
    }
}

#[test]
fn test_constraint_into_owned() {
    let source = "test content";
    let borrowed_constraint = Constraint::ContentContains(Cow::Borrowed(source));
    let owned_constraint = borrowed_constraint.into_owned();

    match owned_constraint {
        Constraint::ContentContains(s) => {
            assert_eq!(s.as_ref(), "test content");
            assert!(matches!(s, Cow::Owned(_)));
        }
        _ => panic!("Expected ContentContains constraint"),
    }
}

#[test]
fn test_query_into_owned() {
    let source = "test_node";
    let borrowed_query = Query::Spread(SpreadQuery {
        source: NodeIdentifier::borrowed(source),
        max_hops: Some(3),
        decay_rate: None,
        activation_threshold: None,
        refractory_period: None,
    });

    let owned_query = borrowed_query.into_owned();

    match owned_query {
        Query::Spread(q) => {
            assert_eq!(q.source.as_str(), "test_node");
            assert!(matches!(q.source.0, Cow::Owned(_)));
        }
        _ => panic!("Expected Spread query"),
    }
}

#[test]
fn test_episode_selector_into_owned() {
    let source = "test";
    let selector = EpisodeSelector::Pattern(Pattern::NodeId(NodeIdentifier::borrowed(source)));
    let owned = selector.into_owned();

    match owned {
        EpisodeSelector::Pattern(Pattern::NodeId(id)) => {
            assert_eq!(id.as_str(), "test");
            assert!(matches!(id.0, Cow::Owned(_)));
        }
        _ => panic!("Expected Pattern selector with NodeId"),
    }
}

// ============================================================================
// Serialization Tests
// ============================================================================

#[test]
fn test_serde_roundtrip_recall_query() {
    let query = Query::Recall(RecallQuery {
        pattern: Pattern::Embedding {
            vector: vec![0.1; 768],
            threshold: 0.8,
        },
        constraints: vec![Constraint::ConfidenceAbove(Confidence::from_raw(0.7))],
        confidence_threshold: None,
        base_rate: None,
        limit: Some(10),
    });

    // Serialize to JSON
    let json = serde_json::to_string(&query).unwrap();

    // Deserialize back
    let deserialized: Query = serde_json::from_str(&json).unwrap();

    assert_eq!(query, deserialized);
}

#[test]
fn test_serde_roundtrip_spread_query() {
    let query = Query::Spread(SpreadQuery {
        source: NodeIdentifier::owned("test_node".to_string()),
        max_hops: Some(5),
        decay_rate: Some(0.2),
        activation_threshold: Some(0.05),
        refractory_period: Some(Duration::from_millis(100)),
    });

    let json = serde_json::to_string(&query).unwrap();
    let deserialized: Query = serde_json::from_str(&json).unwrap();

    assert_eq!(query, deserialized);
}

#[test]
fn test_serde_roundtrip_pattern() {
    let pattern = Pattern::NodeId(NodeIdentifier::owned("test".to_string()));
    let json = serde_json::to_string(&pattern).unwrap();
    let deserialized: Pattern = serde_json::from_str(&json).unwrap();
    assert_eq!(pattern, deserialized);

    let pattern = Pattern::Embedding {
        vector: vec![0.1, 0.2, 0.3],
        threshold: 0.8,
    };
    let json = serde_json::to_string(&pattern).unwrap();
    let deserialized: Pattern = serde_json::from_str(&json).unwrap();
    assert_eq!(pattern, deserialized);
}

#[test]
fn test_serde_roundtrip_constraint() {
    let constraint = Constraint::ContentContains(Cow::Owned("test".to_string()));
    let json = serde_json::to_string(&constraint).unwrap();
    let deserialized: Constraint = serde_json::from_str(&json).unwrap();
    assert_eq!(constraint, deserialized);

    let now = SystemTime::now();
    let constraint = Constraint::CreatedBefore(now);
    let json = serde_json::to_string(&constraint).unwrap();
    let deserialized: Constraint = serde_json::from_str(&json).unwrap();
    assert_eq!(constraint, deserialized);
}

#[test]
fn test_serde_roundtrip_confidence_threshold() {
    let threshold = ConfidenceThreshold::Above(Confidence::from_raw(0.7));
    let json = serde_json::to_string(&threshold).unwrap();
    let deserialized: ConfidenceThreshold = serde_json::from_str(&json).unwrap();
    assert_eq!(threshold, deserialized);

    let threshold = ConfidenceThreshold::Between {
        lower: Confidence::from_raw(0.5),
        upper: Confidence::from_raw(0.9),
    };
    let json = serde_json::to_string(&threshold).unwrap();
    let deserialized: ConfidenceThreshold = serde_json::from_str(&json).unwrap();
    assert_eq!(threshold, deserialized);
}

// ============================================================================
// Display and Debug Tests
// ============================================================================

#[test]
fn test_node_identifier_display() {
    let id = NodeIdentifier::new("test_node");
    assert_eq!(format!("{id}"), "test_node");
}

#[test]
fn test_node_identifier_as_ref() {
    let id = NodeIdentifier::new("test_node");
    let s: &str = id.as_ref();
    assert_eq!(s, "test_node");
}

// ============================================================================
// Memory Layout Tests
// ============================================================================

#[test]
fn test_query_size() {
    use std::mem::size_of;

    // Query enum should be reasonable size (3-4 cache lines max)
    let size = size_of::<Query>();
    assert!(size < 256, "Query too large: {size} bytes (target: <256)");

    // Print actual size for optimization tracking
    eprintln!("Query size: {size} bytes");
}

#[test]
fn test_pattern_size() {
    use std::mem::size_of;

    let size = size_of::<Pattern>();
    assert!(size < 128, "Pattern too large: {size} bytes (target: <128)");
    eprintln!("Pattern size: {size} bytes");
}

#[test]
fn test_constraint_size() {
    use std::mem::size_of;

    let size = size_of::<Constraint>();
    assert!(
        size < 128,
        "Constraint too large: {size} bytes (target: <128)"
    );
    eprintln!("Constraint size: {size} bytes");
}

#[test]
fn test_node_identifier_size() {
    use std::mem::size_of;

    let size = size_of::<NodeIdentifier>();
    // NodeIdentifier should be same size as Cow<str> which is 32 bytes on 64-bit
    eprintln!("NodeIdentifier size: {size} bytes");
    assert!(
        size <= 32,
        "NodeIdentifier unexpectedly large: {size} bytes"
    );
}

#[test]
fn test_confidence_threshold_size() {
    use std::mem::size_of;

    let size = size_of::<ConfidenceThreshold>();
    // Should be small (discriminant + Confidence which is f32)
    assert!(size <= 16, "ConfidenceThreshold too large: {size} bytes");
    eprintln!("ConfidenceThreshold size: {size} bytes");
}

#[test]
fn test_query_category_size() {
    use std::mem::size_of;

    let size = size_of::<QueryCategory>();
    // Enum with 5 variants = 1 byte discriminant
    assert_eq!(size, 1, "QueryCategory should be 1 byte");
}

#[test]
fn test_cow_size() {
    use std::borrow::Cow;
    use std::mem::size_of;

    // Verify zero-cost abstraction: Cow<'a, str> has same size as String
    assert_eq!(
        size_of::<Cow<'_, str>>(),
        size_of::<String>(),
        "Cow should have same size as String"
    );
}

// ============================================================================
// Alignment Tests
// ============================================================================

#[test]
fn test_cache_line_alignment() {
    use std::mem::align_of;

    // Query variants should be well-aligned
    assert_eq!(align_of::<Query>() % 8, 0, "Query should be 8-byte aligned");

    let alignment = align_of::<Query>();
    eprintln!("Query alignment: {alignment} bytes");
}

#[test]
fn test_type_state_builder_compile_time_safety() {
    // This test verifies that the type-state builder works correctly.
    // The key safety feature is that the following code should NOT compile:
    //
    // let query = RecallQueryBuilder::new()
    //     .limit(10)
    //     .build();  // ERROR: no method `build` on RecallQueryBuilder<NoPattern>
    //
    // Since we can't test non-compiling code in tests, we verify that the
    // correct usage DOES compile:

    let _query = RecallQueryBuilder::new()
        .pattern(Pattern::Any)
        .build()
        .unwrap();

    // If this compiles, the type-state pattern is working
}
