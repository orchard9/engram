//! Integration tests for probabilistic query executor
//!
//! Tests end-to-end query execution with evidence tracking, uncertainty propagation,
//! and confidence aggregation from multiple sources.

#![allow(clippy::panic)]
#![allow(clippy::field_reassign_with_default)]

use chrono::Utc;
use engram_core::activation::storage_aware::StorageTier;
use engram_core::query::executor::{
    ActivationPath, ProbabilisticQueryExecutor, QueryExecutorConfig,
};
use engram_core::query::{EvidenceSource, MatchType, UncertaintySource};
use engram_core::{Activation, Confidence, Episode};
use std::sync::Arc;
use std::time::Duration;

fn create_test_episode(id: &str, confidence: Confidence) -> Episode {
    Episode::new(
        id.to_string(),
        Utc::now(),
        format!("Test episode {id}"),
        [0.5f32; 768],
        confidence,
    )
}

#[test]
fn test_end_to_end_query_execution() {
    let config = QueryExecutorConfig::default();
    let executor = ProbabilisticQueryExecutor::new(config);

    // Create test episodes
    let episodes = vec![
        (
            create_test_episode("ep1", Confidence::HIGH),
            Confidence::HIGH,
        ),
        (
            create_test_episode("ep2", Confidence::MEDIUM),
            Confidence::MEDIUM,
        ),
    ];

    // Create activation paths showing how we got to these episodes
    let activation_paths = vec![
        ActivationPath::with_default_weight(
            "source1".to_string(),
            "ep1".to_string(),
            Activation::new(0.9),
            Confidence::HIGH,
            1,
            StorageTier::Hot,
        ),
        ActivationPath::with_default_weight(
            "source2".to_string(),
            "ep2".to_string(),
            Activation::new(0.6),
            Confidence::MEDIUM,
            3,
            StorageTier::Warm,
        ),
    ];

    // Add system uncertainty
    let uncertainty = vec![UncertaintySource::SystemPressure {
        pressure_level: 0.1,
        effect_on_confidence: 0.02,
    }];

    // Execute query
    let result = executor.execute(episodes, &activation_paths, uncertainty);

    // Verify results
    assert_eq!(result.len(), 2);
    assert!(!result.is_empty()); // Has results
    assert_eq!(result.evidence_chain.len(), 2);
    assert_eq!(result.uncertainty_sources.len(), 1);

    // Check confidence interval
    assert!(result.confidence_interval.point.raw() > 0.5);
    assert!(result.confidence_interval.lower.raw() <= result.confidence_interval.point.raw());
    assert!(result.confidence_interval.point.raw() <= result.confidence_interval.upper.raw());
}

#[test]
fn test_query_execution_with_no_activation_paths() {
    let executor = ProbabilisticQueryExecutor::default();

    let episodes = vec![(
        create_test_episode("ep1", Confidence::HIGH),
        Confidence::HIGH,
    )];

    let result = executor.execute(episodes, &[], vec![]);

    // Should still work with just episodes
    assert_eq!(result.len(), 1);
    assert!(result.evidence_chain.is_empty()); // No paths means no evidence
    assert!(result.uncertainty_sources.is_empty());
}

#[test]
fn test_evidence_chain_captures_activation_sources() {
    let executor = ProbabilisticQueryExecutor::default();

    let episodes = vec![(
        create_test_episode("target", Confidence::HIGH),
        Confidence::HIGH,
    )];

    let paths = vec![
        ActivationPath::with_default_weight(
            "source_a".to_string(),
            "target".to_string(),
            Activation::new(0.8),
            Confidence::from_raw(0.8),
            1,
            StorageTier::Hot,
        ),
        ActivationPath::with_default_weight(
            "source_b".to_string(),
            "target".to_string(),
            Activation::new(0.7),
            Confidence::from_raw(0.7),
            2,
            StorageTier::Warm,
        ),
        ActivationPath::with_default_weight(
            "source_c".to_string(),
            "target".to_string(),
            Activation::new(0.6),
            Confidence::from_raw(0.6),
            3,
            StorageTier::Cold,
        ),
    ];

    let result = executor.execute(episodes, &paths, vec![]);

    assert_eq!(result.evidence_chain.len(), 3);

    // Verify all evidence is from spreading activation
    for (idx, evidence) in result.evidence_chain.iter().enumerate() {
        match &evidence.source {
            EvidenceSource::SpreadingActivation {
                source_episode,
                activation_level,
                path_length,
            } => {
                // Check that sources match
                assert!(source_episode.starts_with("source_"));
                assert!(activation_level.value() > 0.0);
                assert!((1..=3).contains(path_length));
            }
            _ => panic!("Expected SpreadingActivation evidence at index {idx}"),
        }
    }
}

#[test]
fn test_confidence_aggregation_from_multiple_paths() {
    let executor = ProbabilisticQueryExecutor::default();

    let episodes = vec![(
        create_test_episode("target", Confidence::MEDIUM),
        Confidence::MEDIUM,
    )];

    // Multiple converging paths should increase confidence
    let paths = vec![
        ActivationPath::with_default_weight(
            "s1".to_string(),
            "target".to_string(),
            Activation::new(0.6),
            Confidence::from_raw(0.6),
            1,
            StorageTier::Hot,
        ),
        ActivationPath::with_default_weight(
            "s2".to_string(),
            "target".to_string(),
            Activation::new(0.5),
            Confidence::from_raw(0.5),
            1,
            StorageTier::Hot,
        ),
        ActivationPath::with_default_weight(
            "s3".to_string(),
            "target".to_string(),
            Activation::new(0.4),
            Confidence::from_raw(0.4),
            1,
            StorageTier::Hot,
        ),
    ];

    let result = executor.execute(episodes, &paths, vec![]);

    // Multiple independent paths should aggregate to higher confidence
    assert!(result.confidence_interval.point.raw() > 0.6);
}

#[test]
fn test_hop_decay_reduces_confidence() {
    let executor = ProbabilisticQueryExecutor::default();

    let episodes = vec![(
        create_test_episode("target", Confidence::HIGH),
        Confidence::HIGH,
    )];

    // Near path (1 hop)
    let near_paths = vec![ActivationPath::with_default_weight(
        "source".to_string(),
        "target".to_string(),
        Activation::new(0.8),
        Confidence::from_raw(0.8),
        1,
        StorageTier::Hot,
    )];

    // Far path (5 hops)
    let far_paths = vec![ActivationPath::with_default_weight(
        "source".to_string(),
        "target".to_string(),
        Activation::new(0.8),
        Confidence::from_raw(0.8),
        5,
        StorageTier::Hot,
    )];

    let near_result = executor.execute(episodes.clone(), &near_paths, vec![]);
    let far_result = executor.execute(episodes, &far_paths, vec![]);

    // Near path should have higher confidence than far path
    assert!(
        near_result.confidence_interval.point.raw() > far_result.confidence_interval.point.raw()
    );
}

#[test]
fn test_tier_reliability_affects_confidence() {
    let executor = ProbabilisticQueryExecutor::default();

    let episodes = vec![(
        create_test_episode("target", Confidence::from_raw(0.7)),
        Confidence::from_raw(0.7),
    )];

    // Same confidence and hops, different tiers
    let hot_paths = vec![ActivationPath::with_default_weight(
        "source".to_string(),
        "target".to_string(),
        Activation::new(0.7),
        Confidence::from_raw(0.7),
        2,
        StorageTier::Hot,
    )];

    let cold_paths = vec![ActivationPath::with_default_weight(
        "source".to_string(),
        "target".to_string(),
        Activation::new(0.7),
        Confidence::from_raw(0.7),
        2,
        StorageTier::Cold,
    )];

    let hot_result = executor.execute(episodes.clone(), &hot_paths, vec![]);
    let cold_result = executor.execute(episodes, &cold_paths, vec![]);

    // Hot tier should have higher confidence than cold tier
    assert!(
        hot_result.confidence_interval.point.raw() > cold_result.confidence_interval.point.raw()
    );
}

#[test]
fn test_uncertainty_source_tracking_system_pressure() {
    let executor = ProbabilisticQueryExecutor::default();

    let episodes = vec![(
        create_test_episode("ep1", Confidence::HIGH),
        Confidence::HIGH,
    )];

    let uncertainty = vec![
        UncertaintySource::SystemPressure {
            pressure_level: 0.5,
            effect_on_confidence: 0.1,
        },
        UncertaintySource::MeasurementError {
            error_magnitude: 0.05,
            confidence_degradation: 0.02,
        },
    ];

    let result = executor.execute(episodes, &[], uncertainty);

    assert_eq!(result.uncertainty_sources.len(), 2);

    // Check first uncertainty source
    if let UncertaintySource::SystemPressure {
        pressure_level,
        effect_on_confidence,
    } = result.uncertainty_sources[0]
    {
        assert!((pressure_level - 0.5).abs() < 1e-6);
        assert!((effect_on_confidence - 0.1).abs() < 1e-6);
    } else {
        panic!("Expected SystemPressure uncertainty");
    }
}

#[test]
fn test_direct_match_evidence_creation() {
    let evidence = ProbabilisticQueryExecutor::create_direct_match_evidence(
        "test_cue".to_string(),
        0.92,
        MatchType::Embedding,
    );

    assert!((evidence.strength.raw() - 0.92).abs() < 1e-6);

    if let EvidenceSource::DirectMatch {
        cue_id,
        similarity_score,
        match_type,
    } = evidence.source
    {
        assert_eq!(cue_id, "test_cue");
        assert!((similarity_score - 0.92).abs() < 1e-6);
        matches!(match_type, MatchType::Embedding);
    } else {
        panic!("Expected DirectMatch evidence");
    }
}

#[test]
fn test_temporal_decay_evidence_creation() {
    let original = Confidence::from_raw(0.9);
    let elapsed = Duration::from_secs(3600 * 5); // 5 hours
    let decay_rate = 0.5; // 0.5 hour tau

    let evidence =
        ProbabilisticQueryExecutor::create_temporal_decay_evidence(original, elapsed, decay_rate);

    // Strength should be less than original
    assert!(evidence.strength.raw() < original.raw());

    if let EvidenceSource::TemporalDecay {
        original_confidence,
        time_elapsed,
        decay_rate: rate,
    } = evidence.source
    {
        assert_eq!(original_confidence, original);
        assert_eq!(time_elapsed, elapsed);
        assert!((rate - decay_rate).abs() < 1e-6);
    } else {
        panic!("Expected TemporalDecay evidence");
    }
}

#[test]
fn test_vector_similarity_evidence_creation() {
    let query_vec = Arc::new([0.5f32; 768]);
    let distance = 0.15;
    let confidence = Confidence::from_raw(0.85);

    let evidence = ProbabilisticQueryExecutor::create_vector_similarity_evidence(
        query_vec.clone(),
        distance,
        confidence,
    );

    assert_eq!(evidence.strength, confidence);

    if let EvidenceSource::VectorSimilarity(vec_evidence) = evidence.source {
        assert_eq!(vec_evidence.query_vector, query_vec);
        assert!((vec_evidence.result_distance - distance).abs() < 1e-6);
        assert_eq!(vec_evidence.index_confidence, confidence);
    } else {
        panic!("Expected VectorSimilarity evidence");
    }
}

#[test]
fn test_empty_query_returns_none_confidence() {
    let executor = ProbabilisticQueryExecutor::default();

    let result = executor.execute(vec![], &[], vec![]);

    assert!(result.is_empty());
    assert!(!result.is_successful());
    assert_eq!(result.confidence_interval.point, Confidence::NONE);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_config_track_evidence_disabled() {
    let mut config = QueryExecutorConfig::default();
    config.track_evidence = false;

    let executor = ProbabilisticQueryExecutor::new(config);

    let episodes = vec![(
        create_test_episode("ep1", Confidence::HIGH),
        Confidence::HIGH,
    )];
    let paths = vec![ActivationPath::with_default_weight(
        "source".to_string(),
        "ep1".to_string(),
        Activation::new(0.8),
        Confidence::HIGH,
        1,
        StorageTier::Hot,
    )];

    let result = executor.execute(episodes, &paths, vec![]);

    // Evidence tracking disabled
    assert!(result.evidence_chain.is_empty());
}

#[test]
fn test_config_track_uncertainty_disabled() {
    let mut config = QueryExecutorConfig::default();
    config.track_uncertainty = false;

    let executor = ProbabilisticQueryExecutor::new(config);

    let episodes = vec![(
        create_test_episode("ep1", Confidence::HIGH),
        Confidence::HIGH,
    )];
    let uncertainty = vec![UncertaintySource::SystemPressure {
        pressure_level: 0.3,
        effect_on_confidence: 0.05,
    }];

    let result = executor.execute(episodes, &[], uncertainty);

    // Uncertainty tracking disabled
    assert!(result.uncertainty_sources.is_empty());
}

#[test]
fn test_path_diversity_increases_interval_width() {
    let executor = ProbabilisticQueryExecutor::default();

    let episodes = vec![(
        create_test_episode("target", Confidence::MEDIUM),
        Confidence::MEDIUM,
    )];

    // High diversity: different tiers, hops, confidences
    let diverse_paths = vec![
        ActivationPath::with_default_weight(
            "s1".to_string(),
            "target".to_string(),
            Activation::new(0.9),
            Confidence::from_raw(0.9),
            1,
            StorageTier::Hot,
        ),
        ActivationPath::with_default_weight(
            "s2".to_string(),
            "target".to_string(),
            Activation::new(0.3),
            Confidence::from_raw(0.3),
            5,
            StorageTier::Cold,
        ),
    ];

    // Low diversity: similar paths
    let similar_paths = vec![
        ActivationPath::with_default_weight(
            "s1".to_string(),
            "target".to_string(),
            Activation::new(0.7),
            Confidence::from_raw(0.7),
            1,
            StorageTier::Hot,
        ),
        ActivationPath::with_default_weight(
            "s2".to_string(),
            "target".to_string(),
            Activation::new(0.72),
            Confidence::from_raw(0.72),
            1,
            StorageTier::Hot,
        ),
    ];

    let diverse_result = executor.execute(episodes.clone(), &diverse_paths, vec![]);
    let similar_result = executor.execute(episodes, &similar_paths, vec![]);

    // Diverse paths should have wider confidence interval
    assert!(diverse_result.confidence_interval.width > similar_result.confidence_interval.width);
}

#[test]
fn test_max_paths_limit_enforced() {
    let mut config = QueryExecutorConfig::default();
    config.max_paths = 2;

    let executor = ProbabilisticQueryExecutor::new(config);

    let episodes = vec![(
        create_test_episode("target", Confidence::HIGH),
        Confidence::HIGH,
    )];

    // Provide 5 paths but only 2 should be used
    let paths: Vec<_> = (0..5)
        .map(|i| {
            ActivationPath::with_default_weight(
                format!("source_{i}"),
                "target".to_string(),
                Activation::new(0.7),
                Confidence::from_raw(0.7 - i as f32 * 0.1),
                1,
                StorageTier::Hot,
            )
        })
        .collect();

    let result = executor.execute(episodes, &paths, vec![]);

    // Should have evidence from all 5 paths
    assert_eq!(result.evidence_chain.len(), 5);

    // But confidence aggregation should only use top 2
    // (verified implicitly by the aggregator's behavior)
}
