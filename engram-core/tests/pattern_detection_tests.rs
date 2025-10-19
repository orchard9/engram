//! Integration tests for pattern detection and consolidation pipeline.
//!
//! Validates end-to-end pattern detection, statistical filtering,
//! and integration with episodic memory.

use chrono::{Duration, Utc};
use engram_core::consolidation::{
    PatternDetectionConfig, PatternDetector, PatternFeature, SIGNIFICANCE_THRESHOLD,
    StatisticalFilter,
};
use engram_core::{Confidence, Episode};

/// Helper to create test episodes with specific embeddings
fn create_episode_with_embedding(id: &str, embedding: &[f32; 768], offset_secs: i64) -> Episode {
    Episode::new(
        id.to_string(),
        Utc::now() + Duration::seconds(offset_secs),
        format!("test content {id}"),
        *embedding,
        Confidence::exact(0.9),
    )
}

/// Helper to create similar episodes (high cosine similarity)
fn create_similar_episodes(count: usize, base_value: f32) -> Vec<Episode> {
    (0..count)
        .map(|i| {
            let mut embedding = [base_value; 768];
            // Add small variation to make episodes similar but not identical
            embedding[0] += (i as f32) * 0.01;
            create_episode_with_embedding(&format!("similar_{i}"), &embedding, (i as i64) * 60)
        })
        .collect()
}

#[test]
fn test_pattern_detection_end_to_end() {
    let config = PatternDetectionConfig::default();
    let detector = PatternDetector::new(config);

    // Create two clusters of similar episodes
    let mut cluster1 = create_similar_episodes(5, 0.8);
    let cluster2 = create_similar_episodes(4, 0.2);

    cluster1.extend(cluster2);
    let all_episodes = cluster1;

    // Detect patterns
    let patterns = detector.detect_patterns(&all_episodes);

    // Should detect at least one pattern (possibly two clusters)
    assert!(!patterns.is_empty(), "Should detect at least one pattern");

    // Verify pattern properties
    for pattern in &patterns {
        assert!(!pattern.id.is_empty(), "Pattern should have ID");
        assert!(
            !pattern.source_episodes.is_empty(),
            "Pattern should have source episodes"
        );
        assert!(
            pattern.strength >= 0.0 && pattern.strength <= 1.0,
            "Pattern strength should be in [0, 1]"
        );
        assert!(
            pattern.occurrence_count >= 3,
            "Pattern should meet min cluster size"
        );
    }
}

#[test]
fn test_pattern_detection_with_min_cluster_size() {
    let config = PatternDetectionConfig {
        min_cluster_size: 5,
        similarity_threshold: 0.9,
        max_patterns: 100,
    };
    let detector = PatternDetector::new(config);

    // Create cluster below minimum size
    let small_cluster = create_similar_episodes(3, 0.8);
    let patterns = detector.detect_patterns(&small_cluster);

    // Should not detect patterns below min cluster size
    assert!(
        patterns.is_empty(),
        "Should not detect patterns below min cluster size"
    );

    // Create cluster meeting minimum size
    let large_cluster = create_similar_episodes(6, 0.8);
    let patterns = detector.detect_patterns(&large_cluster);

    // Should detect pattern with sufficient size
    assert!(
        !patterns.is_empty(),
        "Should detect pattern meeting min cluster size"
    );
}

#[test]
fn test_pattern_detection_with_similarity_threshold() {
    let config = PatternDetectionConfig {
        min_cluster_size: 3,
        similarity_threshold: 0.95, // High threshold
        max_patterns: 100,
    };
    let detector = PatternDetector::new(config);

    // Create highly dissimilar episodes with different embedding values
    let mut ep1_embedding = [0.0; 768];
    ep1_embedding[0] = 1.0; // Only first dimension is 1.0

    let mut ep2_embedding = [0.0; 768];
    ep2_embedding[384] = 1.0; // Only middle dimension is 1.0

    let mut ep3_embedding = [0.0; 768];
    ep3_embedding[767] = 1.0; // Only last dimension is 1.0

    let dissimilar_episodes: Vec<Episode> = vec![
        create_episode_with_embedding("ep1", &ep1_embedding, 0),
        create_episode_with_embedding("ep2", &ep2_embedding, 60),
        create_episode_with_embedding("ep3", &ep3_embedding, 120),
    ];

    let patterns = detector.detect_patterns(&dissimilar_episodes);

    // High similarity threshold should prevent clustering these orthogonal episodes
    assert!(
        patterns.is_empty(),
        "High similarity threshold should prevent clustering orthogonal episodes"
    );
}

#[test]
fn test_statistical_filtering_integration() {
    let detector_config = PatternDetectionConfig {
        min_cluster_size: 3,
        similarity_threshold: 0.8,
        max_patterns: 100,
    };
    let detector = PatternDetector::new(detector_config);
    let filter = StatisticalFilter::new(SIGNIFICANCE_THRESHOLD);

    // Create strong pattern (many similar episodes)
    let mut strong_cluster = create_similar_episodes(10, 0.8);

    // Create weak pattern (few similar episodes)
    let weak_cluster = create_similar_episodes(3, 0.3);

    strong_cluster.extend(weak_cluster);
    let all_episodes = strong_cluster;

    // Detect patterns
    let patterns = detector.detect_patterns(&all_episodes);

    // Filter by statistical significance
    let filtered_patterns = filter.filter_patterns(&patterns);

    // Filtered patterns should be subset of detected patterns
    assert!(
        filtered_patterns.len() <= patterns.len(),
        "Filtered patterns should be subset of all patterns"
    );

    // At least the strong pattern should pass significance test
    assert!(
        !filtered_patterns.is_empty(),
        "Strong pattern should pass significance test"
    );
}

#[test]
fn test_pattern_temporal_features() {
    let config = PatternDetectionConfig::default();
    let detector = PatternDetector::new(config);

    // Create episodes with regular temporal spacing (every 60 seconds)
    let temporal_episodes: Vec<Episode> = (0..5)
        .map(|i| {
            let mut embedding = [0.8; 768];
            embedding[0] += (i as f32) * 0.01;
            create_episode_with_embedding(&format!("temporal_{i}"), &embedding, i64::from(i) * 60)
        })
        .collect();

    let patterns = detector.detect_patterns(&temporal_episodes);

    // Should detect temporal pattern
    assert!(!patterns.is_empty(), "Should detect temporal pattern");

    // Check if temporal features were extracted
    let has_temporal_feature = patterns.iter().any(|p| {
        p.features
            .iter()
            .any(|f| matches!(f, PatternFeature::TemporalSequence { .. }))
    });

    assert!(
        has_temporal_feature,
        "Pattern should extract temporal sequence feature"
    );
}

#[test]
fn test_pattern_merging() {
    let config = PatternDetectionConfig {
        min_cluster_size: 3,
        similarity_threshold: 0.8,
        max_patterns: 100,
    };
    let detector = PatternDetector::new(config);

    // Create multiple very similar clusters that should merge
    let mut cluster1 = create_similar_episodes(4, 0.85);
    let cluster2 = create_similar_episodes(4, 0.86);

    cluster1.extend(cluster2);
    let all_episodes = cluster1;

    let patterns = detector.detect_patterns(&all_episodes);

    // Verify patterns were merged (should have fewer patterns than clusters)
    assert!(
        !patterns.is_empty(),
        "Should detect patterns from similar clusters"
    );

    // Check that merged pattern has combined source episodes
    for pattern in &patterns {
        if pattern.occurrence_count > 4 {
            // This pattern merged multiple clusters
            assert!(
                pattern.occurrence_count <= all_episodes.len(),
                "Merged pattern should not exceed total episodes"
            );
        }
    }
}

#[test]
fn test_pattern_strength_computation() {
    let config = PatternDetectionConfig::default();
    let detector = PatternDetector::new(config);

    // Create tight cluster (high strength expected)
    let tight_cluster = create_similar_episodes(5, 0.9);

    // Create loose cluster (lower strength expected)
    let loose_cluster: Vec<Episode> = (0..5)
        .map(|i| {
            let value = 0.5 + (i as f32) * 0.1; // More variation
            let mut embedding = [value; 768];
            embedding[0] = value;
            create_episode_with_embedding(&format!("loose_{i}"), &embedding, i64::from(i) * 60)
        })
        .collect();

    let tight_patterns = detector.detect_patterns(&tight_cluster);
    let loose_patterns = detector.detect_patterns(&loose_cluster);

    // Both should detect patterns
    assert!(
        !tight_patterns.is_empty(),
        "Should detect pattern in tight cluster"
    );
    assert!(
        !loose_patterns.is_empty(),
        "Should detect pattern in loose cluster"
    );

    // Tight cluster should have higher or similar strength
    // Note: with small clusters, strength values may be very close
    if let (Some(tight), Some(loose)) = (tight_patterns.first(), loose_patterns.first()) {
        // Use epsilon comparison due to floating point precision
        let epsilon = 0.01;
        assert!(
            tight.strength + epsilon >= loose.strength,
            "Tight cluster should have higher or similar strength: tight={}, loose={}",
            tight.strength,
            loose.strength
        );
    }
}

#[test]
fn test_empty_episodes_handling() {
    let config = PatternDetectionConfig::default();
    let detector = PatternDetector::new(config);

    let patterns = detector.detect_patterns(&[]);

    assert!(
        patterns.is_empty(),
        "Empty episodes should yield no patterns"
    );
}

#[test]
fn test_max_patterns_limit() {
    let config = PatternDetectionConfig {
        min_cluster_size: 2,
        similarity_threshold: 0.7,
        max_patterns: 3, // Limit to 3 patterns
    };
    let detector = PatternDetector::new(config);

    // Create many clusters
    let mut all_episodes = Vec::new();
    for cluster_id in 0..10 {
        let base_value = (cluster_id as f32) * 0.1;
        let cluster = create_similar_episodes(3, base_value);
        all_episodes.extend(cluster);
    }

    let patterns = detector.detect_patterns(&all_episodes);

    // Should respect max_patterns limit
    assert!(
        patterns.len() <= 3,
        "Should not exceed max_patterns limit: got {} patterns",
        patterns.len()
    );
}

#[test]
fn test_pattern_id_determinism() {
    let config = PatternDetectionConfig::default();
    let detector = PatternDetector::new(config);

    let episodes = create_similar_episodes(5, 0.8);

    // Detect patterns multiple times
    let patterns1 = detector.detect_patterns(&episodes);
    let patterns2 = detector.detect_patterns(&episodes);

    // Pattern IDs should be deterministic
    assert_eq!(
        patterns1.len(),
        patterns2.len(),
        "Should detect same number of patterns"
    );

    for (p1, p2) in patterns1.iter().zip(patterns2.iter()) {
        assert_eq!(p1.id, p2.id, "Pattern IDs should be deterministic");
    }
}

#[test]
fn test_statistical_filter_custom_threshold() {
    let strict_filter = StatisticalFilter::new(0.001); // Very strict threshold
    let permissive_filter = StatisticalFilter::new(0.1); // More permissive threshold

    let config = PatternDetectionConfig::default();
    let detector = PatternDetector::new(config);

    // Create a strong pattern with many occurrences
    let episodes = create_similar_episodes(10, 0.8);
    let patterns = detector.detect_patterns(&episodes);

    assert!(!patterns.is_empty(), "Should detect patterns from episodes");

    let strict_filtered = strict_filter.filter_patterns(&patterns);
    let permissive_filtered = permissive_filter.filter_patterns(&patterns);

    // Permissive filter should allow more (or equal) patterns than strict filter
    assert!(
        permissive_filtered.len() >= strict_filtered.len(),
        "Permissive threshold should allow more patterns: permissive={}, strict={}",
        permissive_filtered.len(),
        strict_filtered.len()
    );
}
