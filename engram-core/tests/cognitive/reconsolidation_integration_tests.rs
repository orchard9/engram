//! Integration tests for reconsolidation-consolidation bridge
//!
//! Tests the integration between reconsolidation (Task 006) and consolidation
//! system (Milestone 6), verifying that reconsolidated memories correctly
//! re-enter the consolidation pipeline.
//!
//! # Biological Validation
//!
//! These tests verify the principle from Nader et al. (2000) that reconsolidation
//! re-uses the same consolidation machinery as initial memory formation.

use chrono::{Duration, Utc};
use engram_core::cognitive::{
    EpisodeModifications, ModificationType, ReconsolidationConsolidationBridge,
    ReconsolidationEngine,
};
use engram_core::decay::consolidation::ConsolidationProcessor;
use engram_core::{Confidence, Episode};
use std::sync::Arc;

/// Create a test episode that is old enough to be consolidated (>24h)
fn create_consolidated_episode(id: &str) -> Episode {
    Episode::new(
        id.to_string(),
        Utc::now() - Duration::hours(48), // 2 days old (consolidated)
        format!("Test memory: {id}"),
        [0.1; 768],
        Confidence::exact(0.8),
    )
}

/// Create minimal episode modifications for testing
fn create_test_modifications() -> EpisodeModifications {
    EpisodeModifications {
        what: Some("Updated memory content".to_string()),
        where_location: Some("Updated location".to_string()),
        who: None,
        modification_extent: 0.3,
        modification_type: ModificationType::Update,
    }
}

#[test]
fn test_reconsolidated_memory_re_enters_pipeline() {
    // Setup: Create reconsolidation and consolidation systems
    let reconsolidation = Arc::new(ReconsolidationEngine::new());
    let consolidation = Arc::new(ConsolidationProcessor::new());
    let bridge = ReconsolidationConsolidationBridge::new(
        Arc::clone(&reconsolidation),
        Arc::clone(&consolidation),
    );

    // 1. Create consolidated episode (>24h old)
    let episode = create_consolidated_episode("test_episode");
    let episode_id = episode.id.clone();

    // 2. Record recall event (simulates memory retrieval)
    let recall_time = Utc::now();
    reconsolidation.record_recall(&episode, recall_time, true);

    // 3. Attempt reconsolidation during window (3 hours after recall - peak plasticity)
    let modification_time = recall_time + Duration::hours(3);
    let modifications = create_test_modifications();

    let recon_result = reconsolidation
        .attempt_reconsolidation(&episode_id, &modifications, modification_time)
        .expect("Reconsolidation should succeed within window");

    // 4. Process reconsolidated memory - should re-enter consolidation pipeline
    let tagged_episode = bridge
        .process_reconsolidated_memory(recon_result.modified_episode.clone(), &recon_result)
        .expect("Should successfully process reconsolidated memory");

    // Verify: Episode has reconsolidation metadata
    assert!(
        tagged_episode.has_metadata("reconsolidation_event"),
        "Episode should be tagged with reconsolidation event"
    );
    assert!(
        tagged_episode.has_metadata("reconsolidation_cycles"),
        "Episode should track reconsolidation cycles"
    );
    assert!(
        tagged_episode.has_metadata("window_position"),
        "Episode should track window position"
    );

    // Verify: Cycle count incremented
    assert_eq!(
        bridge.get_cycle_count(&episode_id),
        1,
        "Reconsolidation cycle count should increment"
    );

    // Verify: Plasticity factor recorded
    let plasticity_str = tagged_episode
        .get_metadata("reconsolidation_event")
        .expect("Should have reconsolidation_event metadata");
    let plasticity: f32 = plasticity_str
        .parse()
        .expect("Plasticity should be valid f32");
    assert!(
        plasticity > 0.0 && plasticity <= 0.5,
        "Plasticity factor should be in valid range"
    );
}

#[test]
fn test_reconsolidation_preserves_episode_id() {
    let reconsolidation = Arc::new(ReconsolidationEngine::new());
    let consolidation = Arc::new(ConsolidationProcessor::new());
    let bridge = ReconsolidationConsolidationBridge::new(
        Arc::clone(&reconsolidation),
        Arc::clone(&consolidation),
    );

    let episode = create_consolidated_episode("stable_episode");
    let original_id = episode.id.clone();

    // Reconsolidate multiple times
    for cycle in 1..=5 {
        let recall_time = Utc::now();
        reconsolidation.record_recall(&episode, recall_time, true);

        let modification_time = recall_time + Duration::hours(3);
        let modifications = create_test_modifications();

        let recon_result = reconsolidation
            .attempt_reconsolidation(&original_id, &modifications, modification_time)
            .expect("Reconsolidation should succeed");

        let tagged = bridge
            .process_reconsolidated_memory(recon_result.modified_episode.clone(), &recon_result)
            .expect("Should process successfully");

        // Verify: ID remains unchanged
        assert_eq!(
            tagged.id, original_id,
            "Episode ID should remain unchanged after reconsolidation cycle {cycle}"
        );

        // Verify: Cycle counter increments correctly
        assert_eq!(
            bridge.get_cycle_count(&original_id),
            cycle,
            "Cycle count should match iteration number"
        );
    }

    // Final verification: 5 cycles recorded
    assert_eq!(
        bridge.get_cycle_count(&original_id),
        5,
        "Should have exactly 5 reconsolidation cycles"
    );
}

#[test]
fn test_repeated_reconsolidation_tracked_and_bounded() {
    let reconsolidation = Arc::new(ReconsolidationEngine::new());
    let consolidation = Arc::new(ConsolidationProcessor::new());

    // Create bridge with low threshold (5) to test warning behavior
    let bridge =
        ReconsolidationConsolidationBridge::with_threshold(reconsolidation.clone(), consolidation, 5);

    let episode = create_consolidated_episode("unstable_episode");
    let episode_id = episode.id.clone();

    // Reconsolidate 12 times (exceeds threshold but under hard limit)
    for cycle in 1..=12 {
        let recall_time = Utc::now();
        reconsolidation.record_recall(&episode, recall_time, true);

        let modification_time = recall_time + Duration::hours(3);
        let modifications = create_test_modifications();

        let recon_result = reconsolidation
            .attempt_reconsolidation(&episode_id, &modifications, modification_time)
            .expect("Reconsolidation should succeed");

        let result = bridge.process_reconsolidated_memory(
            recon_result.modified_episode.clone(),
            &recon_result,
        );

        // Should succeed but emit warnings after threshold (5)
        if cycle <= 10 {
            // Under hard limit (threshold * 2 = 10)
            assert!(
                result.is_ok(),
                "Cycle {cycle} should succeed (under hard limit)"
            );
        } else {
            // Exceeds hard limit
            assert!(
                result.is_err(),
                "Cycle {cycle} should fail (exceeds hard limit)"
            );
        }
    }

    // Verify final cycle count (should be 10, stopped at hard limit)
    assert_eq!(
        bridge.get_cycle_count(&episode_id),
        10,
        "Should stop at hard limit (threshold * 2)"
    );
}

#[test]
fn test_metadata_tracks_plasticity_window() {
    let reconsolidation = Arc::new(ReconsolidationEngine::new());
    let consolidation = Arc::new(ConsolidationProcessor::new());
    let bridge = ReconsolidationConsolidationBridge::new(reconsolidation.clone(), consolidation);

    let episode = create_consolidated_episode("test_episode");

    // Test at different window positions
    let test_cases = vec![
        (Duration::hours(1), "early"),   // Start of window (low plasticity)
        (Duration::hours(3), "peak"),    // Peak plasticity
        (Duration::hours(5), "late"),    // End of window (declining plasticity)
    ];

    for (offset, phase) in test_cases {
        let recall_time = Utc::now();
        reconsolidation.record_recall(&episode, recall_time, true);

        let modification_time = recall_time + offset;
        let modifications = create_test_modifications();

        let recon_result = reconsolidation
            .attempt_reconsolidation(&episode.id, &modifications, modification_time)
            .expect("Reconsolidation should succeed");

        let tagged = bridge
            .process_reconsolidated_memory(recon_result.modified_episode.clone(), &recon_result)
            .expect("Should process successfully");

        // Verify window position metadata
        let window_pos_str = tagged
            .get_metadata("window_position")
            .expect("Should have window_position metadata");
        let window_pos: f32 = window_pos_str
            .parse()
            .expect("Window position should be valid f32");

        assert!(
            (0.0..=1.0).contains(&window_pos),
            "Window position should be in [0, 1] for phase: {phase}"
        );

        // Peak should have higher window position (~0.5)
        if phase == "peak" {
            assert!(
                (0.3..=0.7).contains(&window_pos),
                "Peak phase should have window position near 0.5"
            );
        }
    }
}

#[test]
fn test_cycle_counter_reset() {
    let reconsolidation = Arc::new(ReconsolidationEngine::new());
    let consolidation = Arc::new(ConsolidationProcessor::new());
    let bridge = ReconsolidationConsolidationBridge::new(reconsolidation.clone(), consolidation);

    let episode = create_consolidated_episode("test_episode");
    let episode_id = episode.id.clone();

    // Reconsolidate 3 times
    for _ in 0..3 {
        let recall_time = Utc::now();
        reconsolidation.record_recall(&episode, recall_time, true);

        let modification_time = recall_time + Duration::hours(3);
        let modifications = create_test_modifications();

        let recon_result = reconsolidation
            .attempt_reconsolidation(&episode_id, &modifications, modification_time)
            .expect("Reconsolidation should succeed");

        let _ = bridge
            .process_reconsolidated_memory(recon_result.modified_episode.clone(), &recon_result)
            .expect("Should process successfully");
    }

    assert_eq!(bridge.get_cycle_count(&episode_id), 3);

    // Reset counter (simulates long-term stabilization)
    bridge.reset_cycle_count(&episode_id);

    assert_eq!(
        bridge.get_cycle_count(&episode_id),
        0,
        "Cycle count should reset to 0"
    );
}

#[test]
fn test_modification_confidence_tracked() {
    let reconsolidation = Arc::new(ReconsolidationEngine::new());
    let consolidation = Arc::new(ConsolidationProcessor::new());
    let bridge = ReconsolidationConsolidationBridge::new(reconsolidation.clone(), consolidation);

    let episode = create_consolidated_episode("test_episode");

    let recall_time = Utc::now();
    reconsolidation.record_recall(&episode, recall_time, true);

    let modification_time = recall_time + Duration::hours(3);
    let modifications = create_test_modifications();

    let recon_result = reconsolidation
        .attempt_reconsolidation(&episode.id, &modifications, modification_time)
        .expect("Reconsolidation should succeed");

    let tagged = bridge
        .process_reconsolidated_memory(recon_result.modified_episode.clone(), &recon_result)
        .expect("Should process successfully");

    // Verify modification confidence metadata
    let conf_str = tagged
        .get_metadata("modification_confidence")
        .expect("Should have modification_confidence metadata");
    let confidence: f32 = conf_str.parse().expect("Confidence should be valid f32");

    assert!(
        (0.0..=1.0).contains(&confidence),
        "Modification confidence should be in [0, 1]"
    );
}

#[test]
fn test_concurrent_reconsolidation_safe() {
    use std::thread;

    let reconsolidation = Arc::new(ReconsolidationEngine::new());
    let consolidation = Arc::new(ConsolidationProcessor::new());
    let bridge = Arc::new(ReconsolidationConsolidationBridge::new(
        reconsolidation.clone(),
        consolidation,
    ));

    // Create multiple episodes
    let episodes: Vec<_> = (0..5)
        .map(|i| create_consolidated_episode(&format!("episode_{i}")))
        .collect();

    // Spawn threads to reconsolidate concurrently
    let handles: Vec<_> = episodes
        .into_iter()
        .map(|episode| {
            let recon = Arc::clone(&reconsolidation);
            let bridge_clone = Arc::clone(&bridge);

            thread::spawn(move || {
                let recall_time = Utc::now();
                recon.record_recall(&episode, recall_time, true);

                let modification_time = recall_time + Duration::hours(3);
                let modifications = create_test_modifications();

                let recon_result = recon
                    .attempt_reconsolidation(&episode.id, &modifications, modification_time)
                    .expect("Reconsolidation should succeed");

                bridge_clone
                    .process_reconsolidated_memory(
                        recon_result.modified_episode.clone(),
                        &recon_result,
                    )
                    .expect("Should process successfully")
            })
        })
        .collect();

    // Wait for all threads
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Verify all episodes processed successfully
    assert_eq!(results.len(), 5, "All episodes should process successfully");

    // Verify each has exactly 1 reconsolidation cycle
    for i in 0..5 {
        assert_eq!(
            bridge.get_cycle_count(&format!("episode_{i}")),
            1,
            "Episode {i} should have 1 cycle"
        );
    }
}

#[test]
fn test_outside_window_skips_integration() {
    let reconsolidation = Arc::new(ReconsolidationEngine::new());
    let consolidation = Arc::new(ConsolidationProcessor::new());
    let bridge = ReconsolidationConsolidationBridge::new(reconsolidation.clone(), consolidation);

    let episode = create_consolidated_episode("test_episode");

    let recall_time = Utc::now();
    reconsolidation.record_recall(&episode, recall_time, true);

    // Try to modify outside window (7 hours after recall - window closes at 6h)
    let modification_time = recall_time + Duration::hours(7);
    let modifications = create_test_modifications();

    let recon_result = reconsolidation.attempt_reconsolidation(
        &episode.id,
        &modifications,
        modification_time,
    );

    // Should return None (outside window)
    assert!(
        recon_result.is_none(),
        "Reconsolidation should fail outside window"
    );

    // Cycle count should remain 0
    assert_eq!(
        bridge.get_cycle_count(&episode.id),
        0,
        "Cycle count should not increment for failed reconsolidation"
    );
}

#[test]
fn test_bridge_accessors() {
    let reconsolidation = Arc::new(ReconsolidationEngine::new());
    let consolidation = Arc::new(ConsolidationProcessor::new());
    let bridge = ReconsolidationConsolidationBridge::new(
        Arc::clone(&reconsolidation),
        Arc::clone(&consolidation),
    );

    // Test accessor methods
    let recon_ref = bridge.reconsolidation_engine();
    let consol_ref = bridge.consolidation_processor();

    assert!(Arc::ptr_eq(recon_ref, &reconsolidation));
    assert!(Arc::ptr_eq(consol_ref, &consolidation));
}
