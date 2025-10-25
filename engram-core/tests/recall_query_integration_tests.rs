//! Integration tests for RECALL query execution
//!
//! Tests the complete RECALL operation from AST query through constraint
//! application to probabilistic result generation.

use chrono::Utc;
use engram_core::query::executor::{QueryExecutorConfig, RecallExecutor};
use engram_core::query::parser::ast::{
    ConfidenceThreshold, Constraint, NodeIdentifier, Pattern, RecallQuery,
};
use engram_core::{Confidence, Cue, Episode, MemoryStore};

/// Create a test memory store with sample episodes
fn create_test_store() -> MemoryStore {
    let store = MemoryStore::new(16);

    // Create episodes with varied confidence and content
    let episodes = vec![
        (
            "ep1",
            Confidence::HIGH,
            "The cat sat on the mat",
            [0.9f32; 768],
        ),
        (
            "ep2",
            Confidence::MEDIUM,
            "The dog barked loudly",
            [0.6f32; 768],
        ),
        (
            "ep3",
            Confidence::LOW,
            "A cat and dog played together",
            [0.3f32; 768],
        ),
        (
            "ep4",
            Confidence::HIGH,
            "The mat was soft and warm",
            [0.8f32; 768],
        ),
    ];

    for (id, confidence, content, embedding) in episodes {
        let episode = Episode::new(
            id.to_string(),
            Utc::now(),
            content.to_string(),
            embedding,
            confidence,
        );
        let _ = store.store_episode(episode);
    }

    store
}

#[test]
fn test_recall_with_content_pattern() {
    let store = create_test_store();
    let executor = RecallExecutor::default();

    let query = RecallQuery {
        pattern: Pattern::ContentMatch(std::borrow::Cow::Borrowed("cat")),
        constraints: vec![],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    };

    let result = executor.execute(query, &store).unwrap();

    // Should retrieve episodes containing "cat"
    assert!(!result.is_empty());
    assert!(result.len() >= 2); // At least ep1 and ep3

    // Check that returned episodes contain "cat"
    for (episode, _) in &result.episodes {
        assert!(
            episode.what.to_lowercase().contains("cat"),
            "Episode {} should contain 'cat'",
            episode.id
        );
    }
}

#[test]
fn test_recall_with_confidence_constraint() {
    let store = create_test_store();
    let executor = RecallExecutor::default();

    let query = RecallQuery {
        pattern: Pattern::Any,
        constraints: vec![Constraint::ConfidenceAbove(Confidence::MEDIUM)],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    };

    let result = executor.execute(query, &store).unwrap();

    // Should only return HIGH confidence episodes
    assert!(!result.is_empty());
    for (_, confidence) in &result.episodes {
        assert!(
            confidence.raw() > Confidence::MEDIUM.raw(),
            "Confidence should be above MEDIUM"
        );
    }
}

#[test]
fn test_recall_with_multiple_constraints() {
    let store = create_test_store();
    let executor = RecallExecutor::default();

    let query = RecallQuery {
        pattern: Pattern::Any,
        constraints: vec![
            Constraint::ConfidenceAbove(Confidence::MEDIUM),
            Constraint::ContentContains(std::borrow::Cow::Borrowed("mat")),
        ],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    };

    let result = executor.execute(query, &store).unwrap();

    // Should return only HIGH confidence episodes containing "mat"
    assert!(!result.is_empty());
    for (episode, confidence) in &result.episodes {
        assert!(
            confidence.raw() > Confidence::MEDIUM.raw(),
            "Confidence should be above MEDIUM"
        );
        assert!(
            episode.what.to_lowercase().contains("mat"),
            "Episode should contain 'mat'"
        );
    }
}

#[test]
fn test_recall_with_limit() {
    let store = create_test_store();
    let executor = RecallExecutor::default();

    let query = RecallQuery {
        pattern: Pattern::Any,
        constraints: vec![],
        confidence_threshold: None,
        base_rate: None,
        limit: Some(2),
    };

    let result = executor.execute(query, &store).unwrap();

    // Should return at most 2 results
    assert!(result.len() <= 2, "Should respect limit of 2");
}

#[test]
fn test_recall_with_confidence_threshold() {
    let store = create_test_store();
    let executor = RecallExecutor::default();

    let query = RecallQuery {
        pattern: Pattern::Any,
        constraints: vec![],
        confidence_threshold: Some(ConfidenceThreshold::Above(Confidence::MEDIUM)),
        base_rate: None,
        limit: None,
    };

    let result = executor.execute(query, &store).unwrap();

    // All results should have confidence above MEDIUM
    for (_, confidence) in &result.episodes {
        assert!(
            confidence.raw() > Confidence::MEDIUM.raw(),
            "Confidence should be above threshold"
        );
    }
}

#[test]
fn test_recall_with_embedding_pattern() {
    let store = create_test_store();
    let executor = RecallExecutor::default();

    // Create an embedding pattern that matches high confidence episodes
    let embedding_vec = vec![0.85f32; 768];

    let query = RecallQuery {
        pattern: Pattern::Embedding {
            vector: embedding_vec,
            threshold: 0.7,
        },
        constraints: vec![],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    };

    let result = executor.execute(query, &store).unwrap();

    // Should retrieve episodes with similar embeddings
    assert!(!result.is_empty());
}

#[test]
fn test_recall_empty_results() {
    let store = create_test_store();
    let executor = RecallExecutor::default();

    // Query for non-existent content
    let query = RecallQuery {
        pattern: Pattern::ContentMatch(std::borrow::Cow::Borrowed("nonexistent content xyz")),
        constraints: vec![],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    };

    let result = executor.execute(query, &store).unwrap();

    // Should return empty result, not an error
    assert!(
        result.is_empty(),
        "Should return empty result for no matches"
    );
    assert_eq!(
        result.confidence_interval.point,
        Confidence::NONE,
        "Confidence should be NONE for empty results"
    );
}

#[test]
fn test_recall_probabilistic_properties() {
    let store = create_test_store();
    let executor = RecallExecutor::default();

    let query = RecallQuery {
        pattern: Pattern::ContentMatch(std::borrow::Cow::Borrowed("cat")),
        constraints: vec![],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    };

    let result = executor.execute(query, &store).unwrap();

    // Verify probabilistic result properties
    assert!(result.confidence_interval.lower.raw() >= 0.0);
    assert!(result.confidence_interval.upper.raw() <= 1.0);
    assert!(result.confidence_interval.lower.raw() <= result.confidence_interval.point.raw());
    assert!(result.confidence_interval.point.raw() <= result.confidence_interval.upper.raw());
}

#[test]
fn test_recall_with_temporal_constraints() {
    use chrono::Duration;

    let store = MemoryStore::new(16);

    let now = Utc::now();
    let past = now - Duration::hours(2);
    let recent = now - Duration::hours(1);

    // Store episodes with different timestamps
    let mut ep1 = Episode::new(
        "ep1".to_string(),
        past,
        "Old episode".to_string(),
        [0.5f32; 768],
        Confidence::HIGH,
    );
    ep1.when = past;

    let mut ep2 = Episode::new(
        "ep2".to_string(),
        recent,
        "Recent episode".to_string(),
        [0.5f32; 768],
        Confidence::HIGH,
    );
    ep2.when = recent;

    let _ = store.store_episode(ep1);
    let _ = store.store_episode(ep2);

    let executor = RecallExecutor::default();

    // Query for episodes before now - 90 minutes (should only get ep1)
    let cutoff = now - Duration::minutes(90);
    let query = RecallQuery {
        pattern: Pattern::Any,
        constraints: vec![Constraint::CreatedBefore(cutoff)],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    };

    let result = executor.execute(query, &store).unwrap();

    assert!(!result.is_empty());
    // Should only return the older episode
    assert_eq!(result.episodes.len(), 1);
    assert_eq!(result.episodes[0].0.id, "ep1");
}

#[test]
fn test_recall_with_embedding_similarity_constraint() {
    let store = create_test_store();
    let executor = RecallExecutor::default();

    // Create a query embedding similar to ep1's embedding (0.9)
    let query_embedding = vec![0.88f32; 768];

    let query = RecallQuery {
        pattern: Pattern::Any,
        constraints: vec![Constraint::SimilarTo {
            embedding: query_embedding,
            threshold: 0.95, // High similarity threshold
        }],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    };

    let result = executor.execute(query, &store).unwrap();

    // Should filter to only similar episodes
    // Note: With our cosine similarity mapping to [0,1], we expect matches
    assert!(!result.is_empty());
}

#[test]
fn test_recall_node_id_pattern() {
    let store = create_test_store();
    let executor = RecallExecutor::default();

    let query = RecallQuery {
        pattern: Pattern::NodeId(NodeIdentifier::from("ep1")),
        constraints: vec![],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    };

    let result = executor.execute(query, &store).unwrap();

    // Should find the episode (via semantic matching on ID)
    assert!(!result.is_empty());
}

#[test]
fn test_recall_confidence_interval_width() {
    let store = create_test_store();
    let executor = RecallExecutor::default();

    // Query with diverse results should have wider confidence interval
    let query1 = RecallQuery {
        pattern: Pattern::Any,
        constraints: vec![],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    };

    let result1 = executor.execute(query1, &store).unwrap();

    // Query with specific constraint should have narrower confidence interval
    let query2 = RecallQuery {
        pattern: Pattern::ContentMatch(std::borrow::Cow::Borrowed("cat")),
        constraints: vec![Constraint::ConfidenceAbove(Confidence::HIGH)],
        confidence_threshold: None,
        base_rate: None,
        limit: None,
    };

    let result2 = executor.execute(query2, &store).unwrap();

    // Verify both have valid confidence intervals
    assert!(result1.confidence_interval.width >= 0.0);
    assert!(result2.confidence_interval.width >= 0.0);
}
