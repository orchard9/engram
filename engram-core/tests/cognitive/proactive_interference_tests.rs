//! Comprehensive test suite for proactive interference detection
//!
//! Tests validate implementation against Underwood (1957) empirical findings
//! and ensure all boundary conditions are correctly enforced.

use chrono::{Duration, Utc};
use engram_core::cognitive::{ProactiveInterferenceDetector, ProactiveInterferenceResult};
use engram_core::memory::Episode;
use engram_core::Confidence;

// ============================================================================
// Test Helpers
// ============================================================================

fn create_episode_with_embedding(
    id: &str,
    embedding: [f32; 768],
    when: chrono::DateTime<Utc>,
) -> Episode {
    Episode {
        id: id.to_string(),
        when,
        where_location: None,
        who: None,
        what: "test episode".to_string(),
        embedding,
        embedding_provenance: None,
        encoding_confidence: Confidence::HIGH,
        vividness_confidence: Confidence::HIGH,
        reliability_confidence: Confidence::HIGH,
        last_recall: Utc::now(),
        recall_count: 0,
        decay_rate: 0.05,
        decay_function: None,
        metadata: std::collections::HashMap::new(),
        metadata: std::collections::HashMap::new(),
    }
}

fn create_similar_embedding(base_value: f32) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for i in 0..768 {
        #[allow(clippy::cast_precision_loss)]
        let val = i as f32;
        embedding[i] = base_value + (val * 0.001);
    }
    // Normalize to unit length
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }
    embedding
}

fn create_dissimilar_embedding(base_value: f32) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for i in 0..768 {
        #[allow(clippy::cast_precision_loss)]
        let val = i as f32;
        // Use negative pattern for dissimilarity
        embedding[i] = base_value - (val * 0.001);
    }
    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }
    embedding
}

// ============================================================================
// Test 1: Underwood (1957) Replication
// ============================================================================

#[test]
fn test_underwood_1957_replication() {
    let detector = ProactiveInterferenceDetector::default();
    let now = Utc::now();

    // Create new list to be learned
    let new_list: Vec<Episode> = (0..10)
        .map(|i| {
            create_episode_with_embedding(
                &format!("new_item_{i}"),
                create_similar_embedding(1.0),
                now,
            )
        })
        .collect();

    // Create 5 prior similar lists (within 6-hour window)
    let mut prior_episodes = Vec::new();
    for list_idx in 0..5 {
        for item_idx in 0..10 {
            let timestamp = now - Duration::hours(3) + Duration::minutes(list_idx * 10);
            let episode = create_episode_with_embedding(
                &format!("prior_list_{list_idx}_item_{item_idx}"),
                create_similar_embedding(1.0), // Similar to new list
                timestamp,
            );
            prior_episodes.push(episode);
        }
    }

    // Test interference on new list items
    let mut total_magnitude = 0.0;
    let mut interference_count = 0;

    for new_item in &new_list {
        let result = detector.detect_interference(new_item, &prior_episodes);
        total_magnitude += result.magnitude;
        if result.is_significant() {
            interference_count += 1;
        }
    }

    let avg_magnitude = total_magnitude / new_list.len() as f32;

    // Underwood (1957): 20-30% accuracy reduction with 5 prior lists
    // Expected: 5 lists × 10 items = 50 similar prior items
    // Interference = min(50 × 0.05, 0.30) = min(2.5, 0.30) = 0.30 (capped)
    assert!(
        (avg_magnitude - 0.30).abs() < 0.01,
        "Expected 30% interference with 5 prior lists, got {:.2}%",
        avg_magnitude * 100.0
    );

    assert_eq!(
        interference_count, 10,
        "All new items should experience significant interference"
    );
}

// ============================================================================
// Test 2: Linear Accumulation
// ============================================================================

#[test]
fn test_linear_accumulation_with_prior_items() {
    let detector = ProactiveInterferenceDetector::default();
    let now = Utc::now();

    let new_episode = create_episode_with_embedding(
        "new",
        create_similar_embedding(1.0),
        now,
    );

    // Test with varying numbers of prior episodes
    for num_prior in 0..=10 {
        let prior_episodes: Vec<Episode> = (0..num_prior)
            .map(|i| {
                create_episode_with_embedding(
                    &format!("prior_{i}"),
                    create_similar_embedding(1.0),
                    now - Duration::hours(3),
                )
            })
            .collect();

        let result = detector.detect_interference(&new_episode, &prior_episodes);

        #[allow(clippy::cast_precision_loss)]
        let expected = (num_prior as f32 * 0.05).min(0.30);

        assert_eq!(
            result.magnitude, expected,
            "Interference should scale linearly: {num_prior} prior → {:.2}% interference",
            expected * 100.0
        );

        assert_eq!(
            result.count, num_prior,
            "Count should match number of interfering episodes"
        );
    }
}

// ============================================================================
// Test 3: Similarity Threshold Enforcement
// ============================================================================

#[test]
fn test_similarity_threshold_enforcement() {
    let detector = ProactiveInterferenceDetector::default();
    let now = Utc::now();

    let new_episode = create_episode_with_embedding(
        "new",
        create_similar_embedding(1.0),
        now,
    );

    // Similar prior episode (same embedding pattern)
    let similar_prior = create_episode_with_embedding(
        "similar",
        create_similar_embedding(1.0),
        now - Duration::hours(1),
    );

    // Dissimilar prior episode (different embedding pattern)
    let dissimilar_prior = create_episode_with_embedding(
        "dissimilar",
        create_dissimilar_embedding(-1.0),
        now - Duration::hours(1),
    );

    let result_similar = detector.detect_interference(&new_episode, &[similar_prior]);
    assert!(
        result_similar.magnitude > 0.0,
        "Similar episodes should interfere"
    );
    assert_eq!(result_similar.count, 1);

    let result_dissimilar = detector.detect_interference(&new_episode, &[dissimilar_prior]);
    assert_eq!(
        result_dissimilar.magnitude, 0.0,
        "Dissimilar episodes should NOT interfere"
    );
    assert_eq!(result_dissimilar.count, 0);
}

// ============================================================================
// Test 4: Temporal Window (6 hours - CORRECTED)
// ============================================================================

#[test]
fn test_temporal_window_6_hours() {
    let detector = ProactiveInterferenceDetector::default();
    let now = Utc::now();

    let new_episode = create_episode_with_embedding(
        "new",
        create_similar_embedding(1.0),
        now,
    );

    // Within window (3 hours ago)
    let recent_prior = create_episode_with_embedding(
        "recent",
        create_similar_embedding(1.0),
        now - Duration::hours(3),
    );

    // Outside window (8 hours ago)
    let old_prior = create_episode_with_embedding(
        "old",
        create_similar_embedding(1.0),
        now - Duration::hours(8),
    );

    let result_recent = detector.detect_interference(&new_episode, &[recent_prior]);
    assert!(
        result_recent.magnitude > 0.0,
        "Should interfere within 6-hour window"
    );

    let result_old = detector.detect_interference(&new_episode, &[old_prior]);
    assert_eq!(
        result_old.magnitude, 0.0,
        "Should NOT interfere outside 6-hour window"
    );
}

// ============================================================================
// Test 5: Consolidation Boundary Effect
// ============================================================================

#[test]
fn test_consolidation_boundary_reduces_interference() {
    let detector = ProactiveInterferenceDetector::default();
    let now = Utc::now();

    let new_episode = create_episode_with_embedding(
        "new",
        create_similar_embedding(1.0),
        now,
    );

    // Unconsolidated memory (<6 hours ago) - should interfere
    let unconsolidated_prior = create_episode_with_embedding(
        "unconsolidated",
        create_similar_embedding(1.0),
        now - Duration::hours(2),
    );

    // Consolidated memory (>6 hours ago) - should NOT interfere
    let consolidated_prior = create_episode_with_embedding(
        "consolidated",
        create_similar_embedding(1.0),
        now - Duration::hours(7),
    );

    let result_unconsolidated =
        detector.detect_interference(&new_episode, &[unconsolidated_prior]);
    assert!(
        result_unconsolidated.magnitude > 0.0,
        "Unconsolidated memory should interfere"
    );

    let result_consolidated = detector.detect_interference(&new_episode, &[consolidated_prior]);
    assert_eq!(
        result_consolidated.magnitude, 0.0,
        "Consolidated memory should NOT interfere (outside 6h window)"
    );
}

// ============================================================================
// Test 6: Exact Temporal Boundary
// ============================================================================

#[test]
fn test_temporal_window_enforced_exactly() {
    let detector = ProactiveInterferenceDetector::default();
    let now = Utc::now();

    let new_episode = create_episode_with_embedding(
        "new",
        create_similar_embedding(1.0),
        now,
    );

    // Just inside window (5h 59m)
    let just_inside = create_episode_with_embedding(
        "just_inside",
        create_similar_embedding(1.0),
        now - Duration::hours(5) - Duration::minutes(59),
    );

    // Just outside window (6h 1m)
    let just_outside = create_episode_with_embedding(
        "just_outside",
        create_similar_embedding(1.0),
        now - Duration::hours(6) - Duration::minutes(1),
    );

    let result_inside = detector.detect_interference(&new_episode, &[just_inside]);
    assert!(
        result_inside.magnitude > 0.0,
        "Should interfere at 5h 59m (inside window)"
    );

    let result_outside = detector.detect_interference(&new_episode, &[just_outside]);
    assert_eq!(
        result_outside.magnitude, 0.0,
        "Should NOT interfere at 6h 01m (outside window)"
    );
}

// ============================================================================
// Test 7: Temporal Direction Enforcement
// ============================================================================

#[test]
fn test_temporal_direction_only_old_to_new() {
    let detector = ProactiveInterferenceDetector::default();
    let now = Utc::now();

    let earlier_episode = create_episode_with_embedding(
        "earlier",
        create_similar_embedding(1.0),
        now - Duration::hours(2),
    );

    let later_episode = create_episode_with_embedding(
        "later",
        create_similar_embedding(1.0),
        now,
    );

    // Earlier should interfere with later (proactive)
    let result_forward = detector.detect_interference(&later_episode, &[earlier_episode.clone()]);
    assert!(
        result_forward.magnitude > 0.0,
        "Earlier memory should interfere with later (proactive)"
    );

    // Later should NOT interfere with earlier (that would be retroactive)
    let result_backward = detector.detect_interference(&earlier_episode, &[later_episode]);
    assert_eq!(
        result_backward.magnitude, 0.0,
        "Later memory should NOT interfere with earlier (wrong direction)"
    );
}

// ============================================================================
// Test 8: Apply Interference to Confidence
// ============================================================================

#[test]
fn test_apply_interference_to_confidence() {
    let base_confidence = Confidence::exact(0.9);

    // 25% interference
    let interference = ProactiveInterferenceResult {
        magnitude: 0.25,
        interfering_episodes: vec!["prior1".to_string(), "prior2".to_string()],
        count: 5,
    };

    let adjusted = ProactiveInterferenceDetector::apply_interference(base_confidence, &interference);

    // Expected: 0.9 * (1 - 0.25) = 0.9 * 0.75 = 0.675
    assert!(
        (adjusted.raw() - 0.675).abs() < 0.001,
        "Confidence should be reduced by interference magnitude"
    );
}

// ============================================================================
// Test 9: Interference Result Helpers
// ============================================================================

#[test]
fn test_interference_result_helpers() {
    let significant = ProactiveInterferenceResult {
        magnitude: 0.25,
        interfering_episodes: vec!["ep1".to_string(), "ep2".to_string()],
        count: 5,
    };

    assert!(significant.is_significant());
    assert_eq!(significant.accuracy_reduction_percent(), 25.0);

    let insignificant = ProactiveInterferenceResult {
        magnitude: 0.05,
        interfering_episodes: vec![],
        count: 1,
    };

    assert!(!insignificant.is_significant());
    assert_eq!(insignificant.accuracy_reduction_percent(), 5.0);
}

// ============================================================================
// Test 10: Maximum Interference Cap
// ============================================================================

#[test]
fn test_maximum_interference_cap() {
    let detector = ProactiveInterferenceDetector::default();
    let now = Utc::now();

    let new_episode = create_episode_with_embedding(
        "new",
        create_similar_embedding(1.0),
        now,
    );

    // Create many prior episodes (more than cap allows)
    let prior_episodes: Vec<Episode> = (0..20)
        .map(|i| {
            create_episode_with_embedding(
                &format!("prior_{i}"),
                create_similar_embedding(1.0),
                now - Duration::hours(3),
            )
        })
        .collect();

    let result = detector.detect_interference(&new_episode, &prior_episodes);

    // Should be capped at 30%, not 20 * 5% = 100%
    assert_eq!(
        result.magnitude, 0.30,
        "Interference should be capped at 30% maximum"
    );
}

// ============================================================================
// Test 11: No Interference Without Similar Prior Episodes
// ============================================================================

#[test]
fn test_no_interference_without_prior_episodes() {
    let detector = ProactiveInterferenceDetector::default();
    let now = Utc::now();

    let new_episode = create_episode_with_embedding(
        "new",
        create_similar_embedding(1.0),
        now,
    );

    let result = detector.detect_interference(&new_episode, &[]);

    assert_eq!(result.magnitude, 0.0);
    assert_eq!(result.count, 0);
    assert!(result.interfering_episodes.is_empty());
}

// ============================================================================
// Test 12: Custom Detector Parameters
// ============================================================================

#[test]
fn test_custom_detector_parameters() {
    let detector = ProactiveInterferenceDetector::new(
        0.8,                   // Higher similarity threshold
        Duration::hours(12),   // Longer temporal window
        0.10,                  // 10% per item
        0.50,                  // 50% max
    );

    assert_eq!(detector.similarity_threshold(), 0.8);
    assert_eq!(detector.prior_memory_window(), Duration::hours(12));
    assert_eq!(detector.interference_per_item(), 0.10);
    assert_eq!(detector.max_interference(), 0.50);

    let now = Utc::now();
    let new_episode = create_episode_with_embedding(
        "new",
        create_similar_embedding(1.0),
        now,
    );

    // Episode 10 hours ago (within custom 12h window)
    let prior = create_episode_with_embedding(
        "prior",
        create_similar_embedding(1.0),
        now - Duration::hours(10),
    );

    let result = detector.detect_interference(&new_episode, &[prior]);

    // Should interfere with custom 12h window
    assert!(result.magnitude > 0.0);
}

// ============================================================================
// Test 13: Interfering Episodes List Accuracy
// ============================================================================

#[test]
fn test_interfering_episodes_list_accuracy() {
    let detector = ProactiveInterferenceDetector::default();
    let now = Utc::now();

    let new_episode = create_episode_with_embedding(
        "new",
        create_similar_embedding(1.0),
        now,
    );

    // Mix of interfering and non-interfering episodes
    let interfering1 = create_episode_with_embedding(
        "interfering1",
        create_similar_embedding(1.0),
        now - Duration::hours(2),
    );

    let interfering2 = create_episode_with_embedding(
        "interfering2",
        create_similar_embedding(1.0),
        now - Duration::hours(4),
    );

    let too_old = create_episode_with_embedding(
        "too_old",
        create_similar_embedding(1.0),
        now - Duration::hours(8),
    );

    let dissimilar = create_episode_with_embedding(
        "dissimilar",
        create_dissimilar_embedding(-1.0),
        now - Duration::hours(1),
    );

    let prior_episodes = vec![
        interfering1.clone(),
        too_old,
        interfering2.clone(),
        dissimilar,
    ];

    let result = detector.detect_interference(&new_episode, &prior_episodes);

    assert_eq!(result.count, 2);
    assert_eq!(result.interfering_episodes.len(), 2);
    assert!(result.interfering_episodes.contains(&"interfering1".to_string()));
    assert!(result.interfering_episodes.contains(&"interfering2".to_string()));
}
