//! Comprehensive integration tests for source attribution system
//!
//! Tests source monitoring accuracy, alternative hypothesis generation,
//! and prevention of false memory formation.

use chrono::Utc;
use engram_core::completion::{
    AlternativeHypothesisGenerator, IntegratedField, MemorySource, PartialEpisode, SourceMonitor,
};
use engram_core::{Confidence, Episode};
use std::collections::HashMap;

// Test helper to create episode
fn create_test_episode(id: &str, embedding: &[f32; 768], what: &str) -> Episode {
    Episode {
        id: id.to_string(),
        when: Utc::now(),
        where_location: None,
        who: None,
        what: what.to_string(),
        embedding: *embedding,
        embedding_provenance: None,
        encoding_confidence: Confidence::exact(0.9),
        vividness_confidence: Confidence::exact(0.8),
        reliability_confidence: Confidence::exact(0.85),
        last_recall: Utc::now(),
        recall_count: 0,
        decay_rate: 0.05,
        decay_function: None,
        metadata: std::collections::HashMap::new(),
    }
}

#[test]
fn test_source_attribution_recalled_fields() {
    let monitor = SourceMonitor::new();

    // Create partial with known field
    let mut known_fields = HashMap::new();
    known_fields.insert("what".to_string(), "coffee".to_string());

    let partial = PartialEpisode {
        known_fields,
        partial_embedding: vec![None; 768],
        cue_strength: Confidence::exact(0.7),
        temporal_context: vec![],
    };

    // Create integrated field for "what"
    let mut integrated_fields = HashMap::new();
    integrated_fields.insert(
        "what".to_string(),
        IntegratedField::new(
            "what".to_string(),
            "coffee".to_string(),
            Confidence::exact(0.9),
            1.0,
            0.0,
            false,
        ),
    );

    let source_map = monitor.attribute_sources(&partial, &integrated_fields);

    // Should be classified as Recalled
    assert_eq!(
        source_map.field_sources.get("what"),
        Some(&MemorySource::Recalled)
    );

    // Should have high source confidence
    let conf = source_map.source_confidence.get("what").unwrap();
    assert!(conf.raw() >= 0.8);
}

#[test]
fn test_source_attribution_reconstructed_fields() {
    let monitor = SourceMonitor::new();

    // Partial with no known fields
    let partial = PartialEpisode {
        known_fields: HashMap::new(),
        partial_embedding: vec![None; 768],
        cue_strength: Confidence::exact(0.7),
        temporal_context: vec![],
    };

    // Field dominated by local context
    let mut integrated_fields = HashMap::new();
    integrated_fields.insert(
        "who".to_string(),
        IntegratedField::new(
            "who".to_string(),
            "Alice".to_string(),
            Confidence::exact(0.6),
            0.8, // Local dominated
            0.2,
            false,
        ),
    );

    let source_map = monitor.attribute_sources(&partial, &integrated_fields);

    // Should be Reconstructed
    assert_eq!(
        source_map.field_sources.get("who"),
        Some(&MemorySource::Reconstructed)
    );
}

#[test]
fn test_source_attribution_consolidated_fields() {
    let monitor = SourceMonitor::new();

    // Partial with no known fields
    let partial = PartialEpisode {
        known_fields: HashMap::new(),
        partial_embedding: vec![None; 768],
        cue_strength: Confidence::exact(0.7),
        temporal_context: vec![],
    };

    // Field dominated by global patterns
    let mut integrated_fields = HashMap::new();
    integrated_fields.insert(
        "where".to_string(),
        IntegratedField::new(
            "where".to_string(),
            "cafe".to_string(),
            Confidence::exact(0.75),
            0.2,
            0.8, // Global dominated
            true,
        ),
    );

    let source_map = monitor.attribute_sources(&partial, &integrated_fields);

    // Should be Consolidated
    assert_eq!(
        source_map.field_sources.get("where"),
        Some(&MemorySource::Consolidated)
    );
}

#[test]
fn test_source_attribution_imagined_fields() {
    let monitor = SourceMonitor::new();

    // Partial with no known fields
    let partial = PartialEpisode {
        known_fields: HashMap::new(),
        partial_embedding: vec![None; 768],
        cue_strength: Confidence::exact(0.7),
        temporal_context: vec![],
    };

    // Field with low confidence (speculative)
    let mut integrated_fields = HashMap::new();
    integrated_fields.insert(
        "who".to_string(),
        IntegratedField::new(
            "who".to_string(),
            "Bob".to_string(),
            Confidence::exact(0.3), // Below reconstructed threshold
            0.5,
            0.5,
            false,
        ),
    );

    let source_map = monitor.attribute_sources(&partial, &integrated_fields);

    // Should be Imagined
    assert_eq!(
        source_map.field_sources.get("who"),
        Some(&MemorySource::Imagined)
    );

    // Low confidence
    let conf = source_map.source_confidence.get("who").unwrap();
    assert!(conf.raw() < 0.5);
}

#[test]
fn test_source_confidence_independent_from_field_confidence() {
    let monitor = SourceMonitor::new();

    // Source confidence for Recalled is always high (regardless of field confidence)
    let conf = monitor.compute_source_confidence(
        MemorySource::Recalled,
        Confidence::exact(0.5), // Low field confidence
        1.0,
    );

    assert!((conf.raw() - 0.85).abs() < 0.01); // recalled_threshold

    // Source confidence for Imagined is penalized
    let conf_imagined = monitor.compute_source_confidence(
        MemorySource::Imagined,
        Confidence::exact(0.8), // High field confidence
        1.0,
    );

    assert!(conf_imagined.raw() < 0.7); // Penalty applied
}

#[test]
fn test_reliable_source_filtering() {
    let monitor = SourceMonitor::new();

    // Recalled is always reliable
    assert!(monitor.is_reliable_source(MemorySource::Recalled, Confidence::exact(0.5)));

    // Consolidated is always reliable
    assert!(monitor.is_reliable_source(MemorySource::Consolidated, Confidence::exact(0.6)));

    // Reconstructed requires threshold
    assert!(monitor.is_reliable_source(MemorySource::Reconstructed, Confidence::exact(0.6)));
    assert!(!monitor.is_reliable_source(MemorySource::Reconstructed, Confidence::exact(0.4)));

    // Imagined is never reliable
    assert!(!monitor.is_reliable_source(MemorySource::Imagined, Confidence::exact(0.9)));
}

#[test]
fn test_alternative_hypothesis_diversity() {
    let generator = AlternativeHypothesisGenerator::new();

    // Create actually diverse embeddings
    let mut emb1 = [0.0; 768];
    let mut emb2 = [0.0; 768];
    let mut emb3 = [0.0; 768];
    for i in 0..768 {
        emb1[i] = (i as f32).sin();
        emb2[i] = (i as f32 + 0.1).sin(); // Very similar to emb1
        emb3[i] = -(i as f32).sin(); // Opposite of emb1 (diverse)
    }

    let episode1 = create_test_episode("1", &emb1, "coffee");
    let episode2 = create_test_episode("2", &emb2, "coffee"); // Too similar
    let episode3 = create_test_episode("3", &emb3, "tea"); // Diverse

    let hypotheses = vec![
        (episode1, Confidence::exact(0.9)),
        (episode2, Confidence::exact(0.85)),
        (episode3, Confidence::exact(0.8)),
    ];

    let diverse = generator.ensure_diversity(hypotheses);

    // Should filter out episode2 (too similar to episode1)
    assert_eq!(diverse.len(), 2);
    assert_eq!(diverse[0].0.id, "1");
    assert_eq!(diverse[1].0.id, "3");
}

#[test]
fn test_alternative_hypothesis_coverage() {
    // Create embeddings with controlled similarity
    let mut truth_emb = [0.0; 768];
    let mut close_emb = [0.0; 768];
    let mut far_emb = [0.0; 768];
    for i in 0..768 {
        truth_emb[i] = (i as f32).sin();
        close_emb[i] = (i as f32 + 0.2).sin(); // Close but not identical (similarity ~0.85)
        far_emb[i] = (i as f32 + std::f32::consts::PI).sin(); // Opposite (similarity ~-1.0)
    }

    let ground_truth = create_test_episode("truth", &truth_emb, "coffee");
    let alt1 = create_test_episode("alt1", &close_emb, "coffee"); // Close match
    let alt2 = create_test_episode("alt2", &far_emb, "tea");

    let alternatives = vec![
        (alt1, Confidence::exact(0.9)),
        (alt2, Confidence::exact(0.8)),
    ];

    // Should contain ground truth (alt1 is close)
    assert!(AlternativeHypothesisGenerator::contains_ground_truth(
        &ground_truth,
        &alternatives,
        0.8
    ));

    // With very high threshold, should not match
    assert!(!AlternativeHypothesisGenerator::contains_ground_truth(
        &ground_truth,
        &alternatives,
        0.99
    ));
}

#[test]
fn test_mixed_source_attribution_scenario() {
    let monitor = SourceMonitor::new();

    // Partial with one known field
    let mut known_fields = HashMap::new();
    known_fields.insert("what".to_string(), "coffee".to_string());

    let partial = PartialEpisode {
        known_fields,
        partial_embedding: vec![None; 768],
        cue_strength: Confidence::exact(0.7),
        temporal_context: vec![],
    };

    // Multiple fields with different sources
    let mut integrated_fields = HashMap::new();

    // Recalled
    integrated_fields.insert(
        "what".to_string(),
        IntegratedField::new(
            "what".to_string(),
            "coffee".to_string(),
            Confidence::exact(0.9),
            1.0,
            0.0,
            false,
        ),
    );

    // Reconstructed
    integrated_fields.insert(
        "who".to_string(),
        IntegratedField::new(
            "who".to_string(),
            "Alice".to_string(),
            Confidence::exact(0.6),
            0.8,
            0.2,
            false,
        ),
    );

    // Consolidated
    integrated_fields.insert(
        "where".to_string(),
        IntegratedField::new(
            "where".to_string(),
            "cafe".to_string(),
            Confidence::exact(0.75),
            0.2,
            0.8,
            true,
        ),
    );

    // Imagined
    integrated_fields.insert(
        "when_detail".to_string(),
        IntegratedField::new(
            "when_detail".to_string(),
            "morning".to_string(),
            Confidence::exact(0.25),
            0.5,
            0.5,
            false,
        ),
    );

    let source_map = monitor.attribute_sources(&partial, &integrated_fields);

    // Verify each source
    assert_eq!(
        source_map.field_sources.get("what"),
        Some(&MemorySource::Recalled)
    );
    assert_eq!(
        source_map.field_sources.get("who"),
        Some(&MemorySource::Reconstructed)
    );
    assert_eq!(
        source_map.field_sources.get("where"),
        Some(&MemorySource::Consolidated)
    );
    assert_eq!(
        source_map.field_sources.get("when_detail"),
        Some(&MemorySource::Imagined)
    );

    // All should have confidence scores
    assert!(source_map.source_confidence.contains_key("what"));
    assert!(source_map.source_confidence.contains_key("who"));
    assert!(source_map.source_confidence.contains_key("where"));
    assert!(source_map.source_confidence.contains_key("when_detail"));
}

#[test]
fn test_custom_threshold_effects() {
    // Very strict thresholds
    let monitor = SourceMonitor::with_thresholds(0.95, 0.85, 0.75);

    let partial = PartialEpisode {
        known_fields: HashMap::new(),
        partial_embedding: vec![None; 768],
        cue_strength: Confidence::exact(0.7),
        temporal_context: vec![],
    };

    // Moderate confidence field that would normally be Reconstructed
    let mut integrated_fields = HashMap::new();
    integrated_fields.insert(
        "who".to_string(),
        IntegratedField::new(
            "who".to_string(),
            "Alice".to_string(),
            Confidence::exact(0.6), // Below strict threshold
            0.8,
            0.2,
            false,
        ),
    );

    let source_map = monitor.attribute_sources(&partial, &integrated_fields);

    // Should be Imagined due to strict threshold
    assert_eq!(
        source_map.field_sources.get("who"),
        Some(&MemorySource::Imagined)
    );
}

#[test]
fn test_alternative_hypothesis_includes_primary() {
    let generator = AlternativeHypothesisGenerator::new();

    let primary = create_test_episode("primary", &[1.0; 768], "coffee");
    let partial = PartialEpisode {
        known_fields: HashMap::new(),
        partial_embedding: vec![None; 768],
        cue_strength: Confidence::exact(0.7),
        temporal_context: vec![],
    };

    let alternatives = generator.generate_alternatives(&partial, &primary, &[]);

    // Should always include primary completion
    assert!(!alternatives.is_empty());
    assert_eq!(alternatives[0].0.id, "primary");
}

#[test]
fn test_source_confidence_calibration() {
    let monitor = SourceMonitor::new();

    // Test that source confidence correlates with evidence consensus

    // High consensus → High source confidence
    let conf_high = monitor.compute_source_confidence(
        MemorySource::Reconstructed,
        Confidence::exact(0.7),
        0.9, // High consensus
    );

    // Low consensus → Lower source confidence
    let conf_low = monitor.compute_source_confidence(
        MemorySource::Reconstructed,
        Confidence::exact(0.7),
        0.4, // Low consensus
    );

    assert!(conf_high.raw() > conf_low.raw());
}

#[test]
fn test_weight_variation_reproducibility() {
    let generator = AlternativeHypothesisGenerator::new();

    let base_weights = vec![0.5, 0.6, 0.7, 0.8];

    let varied1 = generator.vary_pattern_weights(&base_weights, 0);
    let varied2 = generator.vary_pattern_weights(&base_weights, 0);

    // Same variation index should produce same result
    assert_eq!(varied1, varied2);
}

#[test]
fn test_source_attribution_precision_target() {
    let monitor = SourceMonitor::new();

    // Simulate 100 source classifications
    let mut correct = 0;
    let total = 100;

    for i in 0..total {
        // Create different scenarios
        let in_partial = i < 30; // 30% recalled
        let global_dom = (30..60).contains(&i); // 30% consolidated
        let local_dom = (60..90).contains(&i); // 30% reconstructed
        // 10% imagined

        let (local_contrib, global_contrib, confidence) = if in_partial {
            (1.0, 0.0, 0.9)
        } else if global_dom {
            (0.2, 0.8, 0.75)
        } else if local_dom {
            (0.8, 0.2, 0.6)
        } else {
            (0.5, 0.5, 0.3)
        };

        let integrated = IntegratedField::new(
            "test".to_string(),
            "value".to_string(),
            Confidence::exact(confidence),
            local_contrib,
            global_contrib,
            false,
        );

        let (source, _) = monitor.classify_source("test", &integrated, in_partial);

        // Check if classification matches expected source
        let expected = if in_partial {
            MemorySource::Recalled
        } else if global_dom {
            MemorySource::Consolidated
        } else if local_dom {
            MemorySource::Reconstructed
        } else {
            MemorySource::Imagined
        };

        if source == expected {
            correct += 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let precision = correct as f32 / total as f32;

    // Acceptance criterion: >90% precision
    assert!(
        precision >= 0.9,
        "Precision {precision} below 90% threshold"
    );
}
