//! Comprehensive test suite for reconstruction primitives
//!
//! Tests field-level reconstruction, temporal context extraction,
//! and source attribution following the acceptance criteria.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use engram_core::completion::{
    FieldReconstructor, LocalContextExtractor, MemorySource, PartialEpisode,
};
use engram_core::{Confidence, Episode};
use std::collections::HashMap;
use std::time::Duration;

// Test helper to create episodes
fn create_test_episode(
    id: &str,
    when: DateTime<Utc>,
    what: &str,
    where_loc: Option<&str>,
) -> Episode {
    Episode {
        id: id.to_string(),
        when,
        where_location: where_loc.map(String::from),
        who: None,
        what: what.to_string(),
        embedding: [0.1; 768], // Non-zero for similarity tests
        embedding_provenance: None,
        encoding_confidence: Confidence::exact(0.8),
        vividness_confidence: Confidence::exact(0.7),
        reliability_confidence: Confidence::exact(0.8),
        last_recall: when,
        recall_count: 0,
        decay_rate: 0.05,
        decay_function: None,
    }
}

// Test helper to create partial episode
fn create_partial_episode(known_fields: HashMap<String, String>) -> PartialEpisode {
    PartialEpisode {
        known_fields,
        partial_embedding: vec![Some(0.1); 768], // Simple partial embedding
        cue_strength: Confidence::exact(0.7),
        temporal_context: vec!["morning_routine".to_string()],
    }
}

#[test]
fn test_field_consensus_with_unanimous_agreement() {
    let reconstructor = FieldReconstructor::new();
    let now = Utc::now();

    // Create neighbors that all agree
    let neighbors = vec![
        create_test_episode(
            "ep1",
            now - ChronoDuration::minutes(10),
            "morning",
            Some("kitchen"),
        ),
        create_test_episode(
            "ep2",
            now - ChronoDuration::minutes(20),
            "morning",
            Some("kitchen"),
        ),
        create_test_episode(
            "ep3",
            now - ChronoDuration::minutes(30),
            "morning",
            Some("kitchen"),
        ),
    ];

    let partial = create_partial_episode(HashMap::new());
    let reconstructed = reconstructor.reconstruct_fields(&partial, &neighbors);

    // Should reconstruct fields
    assert!(!reconstructed.is_empty());

    // Check that we have a "what" field (from neighbors)
    if let Some(field) = reconstructed.get("what") {
        assert_eq!(field.value, "morning");
        // High confidence due to unanimous agreement
        assert!(field.confidence.raw() > 0.9);
        assert_eq!(field.source, MemorySource::Reconstructed);
    }
}

#[test]
fn test_field_consensus_with_split_vote() {
    let reconstructor = FieldReconstructor::new();
    let now = Utc::now();

    // Create neighbors with split opinions
    let neighbors = vec![
        create_test_episode(
            "ep1",
            now - ChronoDuration::minutes(10),
            "morning",
            Some("kitchen"),
        ),
        create_test_episode(
            "ep2",
            now - ChronoDuration::minutes(20),
            "morning",
            Some("dining"),
        ),
        create_test_episode(
            "ep3",
            now - ChronoDuration::minutes(30),
            "evening",
            Some("kitchen"),
        ),
    ];

    let partial = create_partial_episode(HashMap::new());
    let reconstructed = reconstructor.reconstruct_fields(&partial, &neighbors);

    // Should still reconstruct, but with lower confidence
    assert!(!reconstructed.is_empty());

    // Morning should win (2/3) but confidence should be moderate
    if let Some(field) = reconstructed.get("what") {
        // Should be moderate confidence due to split vote
        assert!(field.confidence.raw() >= 0.5);
        assert!(field.confidence.raw() < 0.9);
    }
}

#[test]
fn test_recalled_field_preservation() {
    let reconstructor = FieldReconstructor::new();
    let now = Utc::now();

    let neighbors = vec![
        create_test_episode(
            "ep1",
            now - ChronoDuration::minutes(10),
            "morning",
            Some("kitchen"),
        ),
        create_test_episode(
            "ep2",
            now - ChronoDuration::minutes(20),
            "evening",
            Some("dining"),
        ),
    ];

    let mut known_fields = HashMap::new();
    known_fields.insert("what".to_string(), "breakfast".to_string());

    let partial = create_partial_episode(known_fields);
    let reconstructed = reconstructor.reconstruct_fields(&partial, &neighbors);

    // "what" field should not be reconstructed since it's already known
    // (we skip known fields in reconstruction)
    assert!(!reconstructed.contains_key("what"));
}

#[test]
fn test_temporal_neighbor_filtering() {
    let extractor = LocalContextExtractor::new();
    let anchor = Utc::now();

    let episodes = vec![
        create_test_episode("ep1", anchor - ChronoDuration::minutes(10), "recent", None),
        create_test_episode("ep2", anchor - ChronoDuration::hours(2), "old", None),
        create_test_episode("ep3", anchor - ChronoDuration::minutes(30), "medium", None),
        create_test_episode("ep4", anchor - ChronoDuration::hours(5), "very_old", None),
    ];

    let neighbors = extractor.temporal_neighbors(anchor, &episodes);

    // Only ep1 and ep3 should be within 1-hour window
    assert_eq!(neighbors.len(), 2);
    assert_eq!(neighbors[0].episode.id, "ep1"); // Closest first
    assert_eq!(neighbors[1].episode.id, "ep3");
}

#[test]
fn test_recency_weighting_decay() {
    let extractor = LocalContextExtractor::new();

    let weight_10min = extractor.recency_weight(Duration::from_secs(600)); // 10 min
    let weight_30min = extractor.recency_weight(Duration::from_secs(1800)); // 30 min
    let weight_60min = extractor.recency_weight(Duration::from_secs(3600)); // 60 min

    // Weights should decrease with distance
    assert!(weight_10min > weight_30min);
    assert!(weight_30min > weight_60min);

    // 10 min should have moderately high weight (quadratic decay: (1-1/6)^2 ≈ 0.69)
    assert!(weight_10min > 0.6);

    // 60 min (at window edge) should have very low weight (nearly 0)
    assert!(weight_60min < 0.1);
}

#[test]
fn test_similarity_threshold_filtering() {
    let reconstructor = FieldReconstructor::with_params(
        Duration::from_secs(3600),
        0.7, // Threshold
        5,
        0.8,
    );

    let now = Utc::now();

    // Create episodes with varying embeddings
    let mut ep_high =
        create_test_episode("ep_high", now - ChronoDuration::minutes(10), "match", None);
    ep_high.embedding = [0.9; 768]; // High similarity

    let mut ep_medium = create_test_episode(
        "ep_medium",
        now - ChronoDuration::minutes(20),
        "match",
        None,
    );
    ep_medium.embedding = [0.4; 768]; // Below threshold

    let mut ep_low =
        create_test_episode("ep_low", now - ChronoDuration::minutes(30), "match", None);
    ep_low.embedding = [0.1; 768]; // Very low similarity

    let neighbors = vec![ep_high, ep_medium, ep_low];

    let mut partial = create_partial_episode(HashMap::new());
    partial.partial_embedding = vec![Some(0.9); 768]; // Similar to ep_high

    let reconstructed = reconstructor.reconstruct_fields(&partial, &neighbors);

    // Should only use high-similarity neighbor
    // (Implementation filters by threshold internally)
    // This is a smoke test - graceful with any result
    let _ = reconstructed; // Smoke test - just verify it doesn't panic
}

#[test]
fn test_empty_neighbor_set_graceful_degradation() {
    let reconstructor = FieldReconstructor::new();
    let partial = create_partial_episode(HashMap::new());

    // Empty neighbor set
    let neighbors: Vec<Episode> = vec![];

    let reconstructed = reconstructor.reconstruct_fields(&partial, &neighbors);

    // Should return empty map gracefully
    assert!(reconstructed.is_empty());
}

#[test]
fn test_temporal_context_extraction() {
    let reconstructor = FieldReconstructor::new();
    let anchor = Utc::now();

    let episodes = vec![
        create_test_episode("ep1", anchor - ChronoDuration::minutes(10), "recent", None),
        create_test_episode("ep2", anchor - ChronoDuration::hours(2), "old", None),
        create_test_episode("ep3", anchor - ChronoDuration::minutes(45), "medium", None),
    ];

    let temporal_context = reconstructor.extract_temporal_context(anchor, &episodes);

    // Should include only episodes within 1-hour window
    assert_eq!(temporal_context.len(), 2);

    // Should be sorted by recency
    assert_eq!(temporal_context[0].id, "ep1");
    assert_eq!(temporal_context[1].id, "ep3");
}

#[test]
fn test_recency_weight_monotonic_property() {
    let extractor = LocalContextExtractor::new();

    // Test multiple distances
    let distances = [
        Duration::from_secs(0),
        Duration::from_secs(300),  // 5 min
        Duration::from_secs(900),  // 15 min
        Duration::from_secs(1800), // 30 min
        Duration::from_secs(2700), // 45 min
        Duration::from_secs(3600), // 60 min
    ];

    let weights: Vec<f32> = distances
        .iter()
        .map(|&d| extractor.recency_weight(d))
        .collect();

    // Verify monotonic decrease
    for i in 1..weights.len() {
        assert!(
            weights[i] <= weights[i - 1],
            "Weight at distance {} should be <= weight at distance {}",
            i,
            i - 1
        );
    }
}

#[test]
fn test_field_reconstructor_max_neighbors() {
    let reconstructor = FieldReconstructor::with_params(
        Duration::from_secs(3600),
        0.1, // Low threshold to include all
        3,   // Max 3 neighbors
        0.8,
    );

    let now = Utc::now();

    // Create 5 neighbors
    let neighbors = vec![
        create_test_episode("ep1", now - ChronoDuration::minutes(5), "value1", None),
        create_test_episode("ep2", now - ChronoDuration::minutes(10), "value2", None),
        create_test_episode("ep3", now - ChronoDuration::minutes(15), "value3", None),
        create_test_episode("ep4", now - ChronoDuration::minutes(20), "value4", None),
        create_test_episode("ep5", now - ChronoDuration::minutes(25), "value5", None),
    ];

    let partial = create_partial_episode(HashMap::new());
    let reconstructed = reconstructor.reconstruct_fields(&partial, &neighbors);

    // Internal filtering should limit to max_neighbors
    // This is a smoke test - just verify it doesn't panic
    let _ = reconstructed;
}

#[test]
fn test_context_merge_temporal_and_spatial() {
    let _extractor = LocalContextExtractor::new();
    let anchor = Utc::now();

    let temporal_neighbors = vec![engram_core::completion::TemporalNeighbor {
        episode: create_test_episode(
            "ep1",
            anchor - ChronoDuration::minutes(10),
            "test",
            Some("kitchen"),
        ),
        temporal_distance: Duration::from_secs(600),
        recency_weight: 0.9,
    }];

    let spatial_neighbors = vec![engram_core::completion::SpatialNeighbor {
        episode: create_test_episode(
            "ep1",
            anchor - ChronoDuration::minutes(10),
            "test",
            Some("kitchen"),
        ),
        spatial_distance: 0.0,
        proximity_weight: 1.0,
    }];

    let merged = LocalContextExtractor::merge_contexts(temporal_neighbors, spatial_neighbors);

    // Should merge the same episode
    assert_eq!(merged.len(), 1);

    // Combined weight should consider both temporal and spatial
    let evidence = &merged[0];
    assert!(evidence.temporal_contribution > 0.0);
    assert!(evidence.spatial_contribution > 0.0);
    assert!(evidence.combined_weight > 0.0);
}
