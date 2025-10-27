//! Comprehensive tests for memory reconsolidation engine
//!
//! Validates exact boundary conditions from Nader et al. (2000) and inverted-U
//! plasticity dynamics from Nader & Einarsson (2010).

use chrono::{Duration, Utc};
use engram_core::cognitive::reconsolidation::{
    EpisodeModifications, ModificationType, ReconsolidationEngine,
};
use engram_core::{Confidence, Episode};

/// Helper to create test episode with specific age
fn create_test_episode(age_hours: i64) -> Episode {
    let when = Utc::now() - Duration::hours(age_hours);
    let embedding = [0.1f32; 768];

    Episode::new(
        format!("episode_{age_hours}"),
        when,
        "test memory content".to_string(),
        embedding,
        Confidence::HIGH,
    )
}

/// Helper to create minimal update modifications
const fn minimal_update() -> EpisodeModifications {
    EpisodeModifications {
        what: None,
        where_location: None,
        who: None,
        modification_extent: 0.1,
        modification_type: ModificationType::Update,
    }
}

// ============================================================================
// Boundary Condition Tests (EXACT per Nader et al. 2000)
// ============================================================================

#[test]
fn test_window_start_boundary_exact_1_hour() {
    // Boundary: Reconsolidation starts EXACTLY at 1 hour post-recall
    let engine = ReconsolidationEngine::new();
    let episode = create_test_episode(48); // 48h old = consolidated

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    // At 59 minutes: should be rejected (too soon)
    let just_before = recall_time + Duration::minutes(59);
    let result = engine.attempt_reconsolidation(&episode.id, &minimal_update(), just_before);
    assert!(
        result.is_none(),
        "Should reject at 59 minutes (before window start)"
    );

    // At exactly 1 hour: should be accepted
    let exactly_1h = recall_time + Duration::hours(1);
    let result = engine.attempt_reconsolidation(&episode.id, &minimal_update(), exactly_1h);
    assert!(
        result.is_some(),
        "Should accept at exactly 1 hour (window start)"
    );
}

#[test]
fn test_window_end_boundary_exact_6_hours() {
    // Boundary: Reconsolidation ends EXACTLY at 6 hours post-recall
    let engine = ReconsolidationEngine::new();
    let episode = create_test_episode(48);

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    // At exactly 6 hours: should be accepted
    let exactly_6h = recall_time + Duration::hours(6);
    let result = engine.attempt_reconsolidation(&episode.id, &minimal_update(), exactly_6h);
    assert!(
        result.is_some(),
        "Should accept at exactly 6 hours (window end)"
    );

    // At 6 hours 1 minute: should be rejected (too late)
    let just_after = recall_time + Duration::hours(6) + Duration::minutes(1);
    let result = engine.attempt_reconsolidation(&episode.id, &minimal_update(), just_after);
    assert!(
        result.is_none(),
        "Should reject at 6h 1m (after window end)"
    );
}

#[test]
fn test_min_age_boundary_exact_24_hours() {
    // Boundary: Memory must be >24 hours old (consolidated)
    let engine = ReconsolidationEngine::new();

    // Episode at 23 hours: too young (not consolidated)
    let too_young = create_test_episode(23);
    let recall_time = Utc::now();
    engine.record_recall(&too_young, recall_time, true);

    let recon_time = recall_time + Duration::hours(2); // Within window
    let result = engine.attempt_reconsolidation(&too_young.id, &minimal_update(), recon_time);
    assert!(
        result.is_none(),
        "Should reject memory at 23h (not consolidated)"
    );

    // Episode at exactly 24 hours + 1 minute: should be accepted
    // We add a small buffer to account for test execution time
    let exactly_24h = create_test_episode(24);
    // Wait slightly before recall to ensure memory is definitely >= 24h old
    let recall_time_24h = recall_time + Duration::milliseconds(100);
    engine.record_recall(&exactly_24h, recall_time_24h, true);

    let recon_time_24h = recall_time_24h + Duration::hours(2); // Within window
    let result = engine.attempt_reconsolidation(&exactly_24h.id, &minimal_update(), recon_time_24h);
    assert!(
        result.is_some(),
        "Should accept memory at 24h+ (consolidated)"
    );

    // Episode at 25 hours: definitely consolidated
    let consolidated = create_test_episode(25);
    engine.record_recall(&consolidated, recall_time, true);

    let result = engine.attempt_reconsolidation(&consolidated.id, &minimal_update(), recon_time);
    assert!(
        result.is_some(),
        "Should accept memory at 25h (consolidated)"
    );
}

#[test]
fn test_max_age_boundary_365_days() {
    // Boundary: Memory must be <365 days old (not too remote)
    let engine = ReconsolidationEngine::new();

    // Memory at 364 days: should be accepted
    let days_364 = create_test_episode(364 * 24);
    let recall_time = Utc::now();
    engine.record_recall(&days_364, recall_time, true);

    let recon_time = recall_time + Duration::hours(3); // Peak window
    let result = engine.attempt_reconsolidation(&days_364.id, &minimal_update(), recon_time);
    assert!(
        result.is_some(),
        "Should accept memory at 364 days (within boundary)"
    );

    // Memory at 366 days: too remote
    let days_366 = create_test_episode(366 * 24);
    engine.record_recall(&days_366, recall_time, true);

    let result = engine.attempt_reconsolidation(&days_366.id, &minimal_update(), recon_time);
    assert!(
        result.is_none(),
        "Should reject memory at 366 days (too remote)"
    );
}

#[test]
fn test_active_recall_required() {
    // Boundary: Must be active recall, not passive re-exposure
    let engine = ReconsolidationEngine::new();
    let episode = create_test_episode(48);

    let recall_time = Utc::now();

    // Passive recall (is_active = false): should be rejected
    engine.record_recall(&episode, recall_time, false);

    let recon_time = recall_time + Duration::hours(3);
    let result = engine.attempt_reconsolidation(&episode.id, &minimal_update(), recon_time);
    assert!(
        result.is_none(),
        "Should reject passive recall (not active)"
    );

    // Active recall (is_active = true): should be accepted
    engine.record_recall(&episode, recall_time, true);

    let result = engine.attempt_reconsolidation(&episode.id, &minimal_update(), recon_time);
    assert!(result.is_some(), "Should accept active recall");
}

// ============================================================================
// Plasticity Dynamics Tests (Inverted-U per Nader & Einarsson 2010)
// ============================================================================

#[test]
fn test_plasticity_inverted_u_shape() {
    // Validates that plasticity follows inverted-U curve with peak at midpoint
    let engine = ReconsolidationEngine::new();

    // Sample plasticity at different window positions
    let early = engine.compute_plasticity(0.0); // 1h post-recall
    let mid_early = engine.compute_plasticity(0.25); // ~2.25h
    let peak = engine.compute_plasticity(0.5); // 3.5h (peak)
    let mid_late = engine.compute_plasticity(0.75); // ~4.75h
    let late = engine.compute_plasticity(1.0); // 6h

    // Plasticity should increase from early to peak
    assert!(
        mid_early > early,
        "Plasticity should increase: {mid_early} > {early}"
    );
    assert!(
        peak > mid_early,
        "Plasticity should reach peak: {peak} > {mid_early}"
    );

    // Plasticity should decrease from peak to late
    assert!(
        mid_late < peak,
        "Plasticity should decrease: {mid_late} < {peak}"
    );
    assert!(
        late < mid_late,
        "Plasticity should continue decreasing: {late} < {mid_late}"
    );

    // Peak should be at window_position = 0.5
    assert!(
        peak > early && peak > late,
        "Peak plasticity ({peak}) should exceed early ({early}) and late ({late})"
    );

    // Verify symmetry (inverted-U is symmetric)
    let symmetry_tolerance = 0.01;
    assert!(
        (early - late).abs() < symmetry_tolerance,
        "Inverted-U should be symmetric: early {early} ≈ late {late}"
    );
}

#[test]
fn test_plasticity_peak_at_3_to_4_hours() {
    // Peak plasticity should occur at 3-4 hours post-recall (middle of window)
    let engine = ReconsolidationEngine::new();
    let episode = create_test_episode(48);

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    // Test at 3 hours (should be near peak)
    let at_3h = recall_time + Duration::hours(3);
    let result_3h = engine
        .attempt_reconsolidation(&episode.id, &minimal_update(), at_3h)
        .expect("Should succeed at 3h");

    // Test at 3.5 hours (should be at peak)
    let at_3_5h = recall_time + Duration::hours(3) + Duration::minutes(30);
    let result_3_5h = engine
        .attempt_reconsolidation(&episode.id, &minimal_update(), at_3_5h)
        .expect("Should succeed at 3.5h");

    // Test at 4 hours (should still be near peak)
    let at_4h = recall_time + Duration::hours(4);
    let result_4h = engine
        .attempt_reconsolidation(&episode.id, &minimal_update(), at_4h)
        .expect("Should succeed at 4h");

    // All should have high plasticity
    assert!(
        result_3h.plasticity_factor > 0.4,
        "Plasticity at 3h should be high: {}",
        result_3h.plasticity_factor
    );
    assert!(
        result_3_5h.plasticity_factor > 0.4,
        "Plasticity at 3.5h should be high: {}",
        result_3_5h.plasticity_factor
    );
    assert!(
        result_4h.plasticity_factor > 0.4,
        "Plasticity at 4h should be high: {}",
        result_4h.plasticity_factor
    );

    // 3.5h should be highest (peak)
    assert!(
        result_3_5h.plasticity_factor >= result_3h.plasticity_factor,
        "3.5h ({}) should be >= 3h ({})",
        result_3_5h.plasticity_factor,
        result_3h.plasticity_factor
    );
    assert!(
        result_3_5h.plasticity_factor >= result_4h.plasticity_factor,
        "3.5h ({}) should be >= 4h ({})",
        result_3_5h.plasticity_factor,
        result_4h.plasticity_factor
    );
}

#[test]
fn test_plasticity_lower_at_window_edges() {
    // Plasticity should be lower at 1h and 6h (window edges) than at peak
    let engine = ReconsolidationEngine::new();
    let episode = create_test_episode(48);

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    // At 1h (window start)
    let at_1h = recall_time + Duration::hours(1);
    let result_1h = engine
        .attempt_reconsolidation(&episode.id, &minimal_update(), at_1h)
        .expect("Should succeed at 1h");

    // At peak (3.5h)
    let at_peak = recall_time + Duration::hours(3) + Duration::minutes(30);
    let result_peak = engine
        .attempt_reconsolidation(&episode.id, &minimal_update(), at_peak)
        .expect("Should succeed at peak");

    // At 6h (window end)
    let at_6h = recall_time + Duration::hours(6);
    let result_6h = engine
        .attempt_reconsolidation(&episode.id, &minimal_update(), at_6h)
        .expect("Should succeed at 6h");

    // Peak should be significantly higher than edges
    assert!(
        result_peak.plasticity_factor > result_1h.plasticity_factor * 1.5,
        "Peak plasticity ({}) should be >1.5x edge ({})",
        result_peak.plasticity_factor,
        result_1h.plasticity_factor
    );
    assert!(
        result_peak.plasticity_factor > result_6h.plasticity_factor * 1.5,
        "Peak plasticity ({}) should be >1.5x edge ({})",
        result_peak.plasticity_factor,
        result_6h.plasticity_factor
    );
}

// ============================================================================
// Modification Type Tests (Update/Corruption/Replacement)
// ============================================================================

#[test]
fn test_update_modification_strengthens_memory() {
    // Update type should maintain or increase confidence (retrieval-induced strengthening)
    let engine = ReconsolidationEngine::new();
    let episode = create_test_episode(48);

    let original_confidence = episode.reliability_confidence;

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    let modifications = EpisodeModifications {
        what: Some("enhanced memory content".to_string()),
        where_location: None,
        who: None,
        modification_extent: 0.3,
        modification_type: ModificationType::Update,
    };

    let recon_time = recall_time + Duration::hours(3); // Peak plasticity
    let result = engine
        .attempt_reconsolidation(&episode.id, &modifications, recon_time)
        .expect("Should succeed");

    // Confidence should increase or stay same
    assert!(
        result.modified_episode.reliability_confidence.raw() >= original_confidence.raw(),
        "Update should not decrease confidence: {} -> {}",
        original_confidence.raw(),
        result.modified_episode.reliability_confidence.raw()
    );

    // Vividness should also increase or stay same (retrieval strengthening)
    assert!(
        result.modified_episode.vividness_confidence.raw() >= episode.vividness_confidence.raw(),
        "Update should not decrease vividness: {} -> {}",
        episode.vividness_confidence.raw(),
        result.modified_episode.vividness_confidence.raw()
    );
}

#[test]
fn test_corruption_modification_reduces_confidence() {
    // Corruption type should reduce confidence (conflicting information)
    let engine = ReconsolidationEngine::new();
    let episode = create_test_episode(48);

    let original_confidence = episode.reliability_confidence;

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    let modifications = EpisodeModifications {
        what: Some("conflicting memory content".to_string()),
        where_location: None,
        who: None,
        modification_extent: 0.5, // Moderate corruption
        modification_type: ModificationType::Corruption,
    };

    let recon_time = recall_time + Duration::hours(3); // Peak plasticity
    let result = engine
        .attempt_reconsolidation(&episode.id, &modifications, recon_time)
        .expect("Should succeed");

    // Confidence should decrease
    assert!(
        result.modified_episode.reliability_confidence.raw() < original_confidence.raw(),
        "Corruption should decrease confidence: {} -> {}",
        original_confidence.raw(),
        result.modified_episode.reliability_confidence.raw()
    );

    // Should not reduce below minimum threshold (0.1)
    assert!(
        result.modified_episode.reliability_confidence.raw() >= 0.1,
        "Confidence should not drop below 0.1: {}",
        result.modified_episode.reliability_confidence.raw()
    );
}

#[test]
fn test_replacement_modification_resets_confidence() {
    // Replacement type should reset confidence to moderate level
    let engine = ReconsolidationEngine::new();
    let episode = create_test_episode(48);

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    let modifications = EpisodeModifications {
        what: Some("completely new memory content".to_string()),
        where_location: Some("new location".to_string()),
        who: Some(vec!["new person".to_string()]),
        modification_extent: 0.7, // High quality replacement
        modification_type: ModificationType::Replacement,
    };

    let recon_time = recall_time + Duration::hours(3);
    let result = engine
        .attempt_reconsolidation(&episode.id, &modifications, recon_time)
        .expect("Should succeed");

    // Confidence should be reset to moderate range [0.3, 0.8]
    let new_confidence = result.modified_episode.reliability_confidence.raw();
    assert!(
        (0.3..=0.8).contains(&new_confidence),
        "Replacement confidence {new_confidence} should be in [0.3, 0.8]"
    );

    // Should not be at original high confidence
    assert!(
        (new_confidence - episode.reliability_confidence.raw()).abs() > 0.1,
        "Replacement should change confidence significantly: {} -> {}",
        episode.reliability_confidence.raw(),
        new_confidence
    );
}

#[test]
fn test_modification_type_affects_vividness_differently() {
    // Different modification types should affect vividness differently
    let engine = ReconsolidationEngine::new();
    let base_episode = create_test_episode(48);

    let recall_time = Utc::now();
    let recon_time = recall_time + Duration::hours(3);

    // Test Update (should increase vividness)
    let episode_update = base_episode.clone();
    engine.record_recall(&episode_update, recall_time, true);

    let update_mods = EpisodeModifications {
        what: None,
        where_location: None,
        who: None,
        modification_extent: 0.3,
        modification_type: ModificationType::Update,
    };

    let result_update = engine
        .attempt_reconsolidation(&episode_update.id, &update_mods, recon_time)
        .expect("Update should succeed");

    // Test Corruption (should decrease vividness)
    let mut episode_corrupt = create_test_episode(48);
    episode_corrupt
        .id
        .clone_from(&format!("{}_corrupt", base_episode.id));
    engine.record_recall(&episode_corrupt, recall_time, true);

    let corrupt_mods = EpisodeModifications {
        what: None,
        where_location: None,
        who: None,
        modification_extent: 0.3,
        modification_type: ModificationType::Corruption,
    };

    let result_corrupt = engine
        .attempt_reconsolidation(&episode_corrupt.id, &corrupt_mods, recon_time)
        .expect("Corruption should succeed");

    // Update should increase vividness
    assert!(
        result_update.modified_episode.vividness_confidence.raw()
            >= base_episode.vividness_confidence.raw(),
        "Update should maintain/increase vividness"
    );

    // Corruption should decrease vividness
    assert!(
        result_corrupt.modified_episode.vividness_confidence.raw()
            < episode_corrupt.vividness_confidence.raw(),
        "Corruption should decrease vividness"
    );
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_no_recall_recorded_returns_none() {
    // Without recording recall, reconsolidation should fail
    let engine = ReconsolidationEngine::new();
    let episode = create_test_episode(48);

    // Don't record recall
    let recon_time = Utc::now() + Duration::hours(3);
    let result = engine.attempt_reconsolidation(&episode.id, &minimal_update(), recon_time);

    assert!(
        result.is_none(),
        "Should return None when recall not recorded"
    );
}

#[test]
fn test_original_episode_preserved() {
    // Original episode should be preserved in result for auditing
    let engine = ReconsolidationEngine::new();
    let episode = create_test_episode(48);
    let original_what = episode.what.clone();

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    let modifications = EpisodeModifications {
        what: Some("modified content".to_string()),
        where_location: None,
        who: None,
        modification_extent: 0.5,
        modification_type: ModificationType::Update,
    };

    let recon_time = recall_time + Duration::hours(3);
    let result = engine
        .attempt_reconsolidation(&episode.id, &modifications, recon_time)
        .expect("Should succeed");

    // Original should be preserved
    assert_eq!(
        result.original_episode.what, original_what,
        "Original episode should be preserved"
    );

    // Modified should be different
    assert_eq!(
        result.modified_episode.what, "modified content",
        "Modified episode should have new content"
    );
}

#[test]
fn test_window_position_reported_correctly() {
    // Window position should be calculated correctly
    let engine = ReconsolidationEngine::new();
    let episode = create_test_episode(48);

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    // At 1h (window start): position should be ~0.0
    let at_start = recall_time + Duration::hours(1);
    let result_start = engine
        .attempt_reconsolidation(&episode.id, &minimal_update(), at_start)
        .expect("Should succeed at start");
    assert!(
        result_start.window_position < 0.1,
        "Position at start should be ~0.0: {}",
        result_start.window_position
    );

    // At 3.5h (middle): position should be ~0.5
    let at_mid = recall_time + Duration::hours(3) + Duration::minutes(30);
    let result_mid = engine
        .attempt_reconsolidation(&episode.id, &minimal_update(), at_mid)
        .expect("Should succeed at mid");
    assert!(
        (result_mid.window_position - 0.5).abs() < 0.1,
        "Position at mid should be ~0.5: {}",
        result_mid.window_position
    );

    // At 6h (window end): position should be ~1.0
    let at_end = recall_time + Duration::hours(6);
    let result_end = engine
        .attempt_reconsolidation(&episode.id, &minimal_update(), at_end)
        .expect("Should succeed at end");
    assert!(
        result_end.window_position > 0.9,
        "Position at end should be ~1.0: {}",
        result_end.window_position
    );
}

#[test]
fn test_modification_confidence_correlates_with_plasticity() {
    // Modification confidence should be higher at peak plasticity
    let engine = ReconsolidationEngine::new();
    let episode = create_test_episode(48);

    let recall_time = Utc::now();
    engine.record_recall(&episode, recall_time, true);

    // At window edges (low plasticity)
    let at_1h = recall_time + Duration::hours(1);
    let result_1h = engine
        .attempt_reconsolidation(&episode.id, &minimal_update(), at_1h)
        .expect("Should succeed");

    // At peak (high plasticity)
    let at_peak = recall_time + Duration::hours(3) + Duration::minutes(30);
    let result_peak = engine
        .attempt_reconsolidation(&episode.id, &minimal_update(), at_peak)
        .expect("Should succeed");

    // Modification confidence should be higher at peak
    assert!(
        result_peak.modification_confidence.raw() > result_1h.modification_confidence.raw(),
        "Modification confidence at peak ({}) should exceed edges ({})",
        result_peak.modification_confidence.raw(),
        result_1h.modification_confidence.raw()
    );
}

#[test]
fn test_multiple_recalls_prune_old_events() {
    // Old recall events should be pruned automatically
    let engine = ReconsolidationEngine::new();

    // Record many recalls
    for i in 0..100 {
        let mut episode = create_test_episode(48);
        episode.id.clone_from(&format!("episode_{i}"));
        engine.record_recall(&episode, Utc::now() - Duration::hours(10), true);
    }

    // Record a new recall that triggers pruning
    let recent = create_test_episode(48);
    engine.record_recall(&recent, Utc::now(), true);

    // Old recalls (>6h ago) should be pruned
    // We can't directly check internal state, but verify the engine doesn't grow unbounded
    // This is a smoke test to ensure pruning happens
}
