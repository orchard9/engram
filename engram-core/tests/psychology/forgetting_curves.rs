//! Forgetting Curve Validation (Milestone 4)
//!
//! Validates Engram's temporal decay functions against published psychology data.
//! Tests exponential and power-law decay curves against Ebbinghaus (1885) and
//! Wixted & Ebbesen (1991) empirical forgetting curves.
//!
//! These tests require longer execution times to simulate multi-day decay periods.
//! Run with: cargo test --test psychology --ignored

#![allow(clippy::uninlined_format_args)]

use engram_core::{Confidence, Episode, EpisodeBuilder, MemoryStore};
use std::time::Duration;

/// Ebbinghaus (1885) forgetting curve shows ~60% retention at 20 minutes,
/// ~45% at 1 hour, ~35% at 1 day, ~25% at 6 days
const EBBINGHAUS_RETENTION: [(Duration, f32); 4] = [
    (Duration::from_secs(20 * 60), 0.60),      // 20 minutes
    (Duration::from_secs(60 * 60), 0.45),      // 1 hour
    (Duration::from_secs(24 * 60 * 60), 0.35), // 1 day
    (Duration::from_secs(6 * 24 * 60 * 60), 0.25), // 6 days
];

const TOLERANCE: f32 = 0.05; // 5% error tolerance (from Milestone 4 requirement)

#[tokio::test]
#[cfg_attr(not(feature = "long_running_tests"), ignore)]
async fn test_exponential_decay_matches_ebbinghaus() {
    let store = MemoryStore::new(1000);

    // Store 100 test memories with uniform initial confidence
    let memory_ids: Vec<String> = (0..100)
        .map(|i| {
            let episode = EpisodeBuilder::new()
                .id(format!("memory_{}", i))
                .what(format!("Test memory {}", i))
                .when(chrono::Utc::now())
                .confidence(Confidence::from_raw(0.95))
                .embedding(vec![0.1; 768])
                .build()
                .unwrap();

            let id = episode.id.clone();
            store.store(episode);
            id
        })
        .collect();

    // Test decay at Ebbinghaus time intervals
    for (elapsed, expected_retention) in EBBINGHAUS_RETENTION {
        // Simulate time passage (in production, use actual sleep or time manipulation)
        tokio::time::sleep(elapsed).await;

        // Measure actual retention by querying all memories
        let retained_count = memory_ids
            .iter()
            .filter_map(|id| {
                let results = store.recall_by_id(id);
                results.first().map(|(_, conf)| conf.raw())
            })
            .filter(|&conf| conf >= 0.3) // Retention threshold
            .count();

        let actual_retention = retained_count as f32 / memory_ids.len() as f32;

        // Validate against empirical data within tolerance
        let error = (actual_retention - expected_retention).abs();
        assert!(
            error <= TOLERANCE,
            "Decay at {:?} should match Ebbinghaus curve. Expected {:.2}±{}, got {:.2} (error: {:.2})",
            elapsed,
            expected_retention,
            TOLERANCE,
            actual_retention,
            error
        );
    }
}

#[tokio::test]
#[cfg_attr(not(feature = "long_running_tests"), ignore)]
async fn test_power_law_decay_matches_wixted() {
    // TODO: Implement power-law decay validation against Wixted & Ebbesen (1991)
    // Power law: R(t) = a * t^(-b) where a ≈ 0.95, b ≈ 0.15
    todo!("Implement when power-law decay function is available")
}

#[tokio::test]
#[cfg_attr(not(feature = "long_running_tests"), ignore)]
async fn test_spacing_effect_validation() {
    // TODO: Validate spaced repetition improves retention
    // Bjork & Allen (1970) - distributed practice yields better long-term retention
    todo!("Implement when spaced repetition scheduler is available")
}

/// Quick smoke test - validates basic decay behavior without long wait times
#[tokio::test]
async fn test_decay_direction() {
    let store = MemoryStore::new(100);

    let episode = EpisodeBuilder::new()
        .id("test_decay")
        .what("Test memory")
        .when(chrono::Utc::now())
        .confidence(Confidence::from_raw(0.95))
        .embedding(vec![0.1; 768])
        .build()
        .unwrap();

    store.store(episode);

    // Initial confidence should be high
    let initial_results = store.recall_by_id("test_decay");
    let initial_confidence = initial_results[0].1.raw();
    assert!(initial_confidence >= 0.90);

    // After some time, confidence should decrease (or stay same if no decay)
    tokio::time::sleep(Duration::from_secs(1)).await;
    let later_results = store.recall_by_id("test_decay");
    let later_confidence = later_results[0].1.raw();

    // Decay should be monotonically decreasing (never increase spontaneously)
    assert!(
        later_confidence <= initial_confidence,
        "Confidence should not increase over time without reinforcement"
    );
}
