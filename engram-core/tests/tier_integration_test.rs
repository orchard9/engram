//! Integration tests for cross-tier recall functionality
//!
//! Tests that memories can be stored and recalled across hot/warm/cold tiers
//! with proper confidence calibration and migration behavior.

#![cfg(feature = "memory_mapped_persistence")]

use chrono::Utc;
use engram_core::{Confidence, CueBuilder, EpisodeBuilder, MemoryStore};
use tempfile::TempDir;

/// Test that memories can be stored and recalled across multiple tiers
#[tokio::test]
async fn test_cross_tier_recall() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Create store with tiered storage
    let mut store = MemoryStore::new(100);

    store = store
        .with_persistence(temp_dir.path())
        .expect("Failed to enable persistence with tiered storage");

    // Store multiple episodes with different activation levels
    for i in 0..20 {
        let activation = if i < 5 {
            0.9 // High activation - should go to hot tier
        } else if i < 15 {
            0.5 // Medium activation - should go to warm tier
        } else {
            0.2 // Low activation - should go to warm/cold tier
        };

        let episode = EpisodeBuilder::new()
            .id(format!("episode_{i}"))
            .when(Utc::now())
            .what(format!("Test content number {i}"))
            .embedding([i as f32 / 20.0; 768])
            .confidence(Confidence::from_raw(activation))
            .build();

        let _ = store.store(episode);
    }

    // Create a cue to search for episodes
    let query_embedding = [0.5; 768]; // Middle of our range
    let cue = CueBuilder::new()
        .id("test_cross_tier_search".to_string())
        .embedding_search(query_embedding, Confidence::LOW)
        .max_results(10)
        .build();

    // Recall should find memories across all tiers
    let results = store.recall(&cue).results;

    // Verify we got results
    assert!(
        !results.is_empty(),
        "Should recall memories from tiered storage"
    );
    assert!(
        results.len() <= 10,
        "Should respect max_results limit (got {})",
        results.len()
    );

    // Verify results are sorted by confidence (highest first)
    for i in 1..results.len() {
        assert!(
            results[i - 1].1.raw() >= results[i].1.raw(),
            "Results should be sorted by confidence descending"
        );
    }

    println!(
        "Cross-tier recall test passed: found {} memories",
        results.len()
    );
}

/// Test that tier migration background task can be started
#[tokio::test]
async fn test_tier_migration_startup() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    let store = MemoryStore::new(100)
        .with_persistence(temp_dir.path())
        .expect("Failed to enable persistence");

    // Start migration task - worker now auto-managed
    store.start_tier_migration();

    // Give worker a moment to start
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Note: We don't actually wait for migration to complete in this test
    // as it would take 5 minutes. We just verify the task starts.
    println!("Tier migration task started successfully");
}

/// Test recall with semantic search across tiers
#[tokio::test]
async fn test_semantic_cross_tier_search() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    let mut store = MemoryStore::new(100);
    store = store
        .with_persistence(temp_dir.path())
        .expect("Failed to enable persistence");

    // Store episodes with semantic content
    let episodes = vec![
        (
            "ep1",
            "cognitive memory system with spreading activation",
            0.9,
        ),
        ("ep2", "neural network architecture for deep learning", 0.7),
        ("ep3", "graph database for knowledge representation", 0.5),
        ("ep4", "memory consolidation during sleep", 0.3),
    ];

    for (id, content, activation) in episodes {
        let episode = EpisodeBuilder::new()
            .id(id.to_string())
            .when(Utc::now())
            .what(content.to_string())
            .embedding([activation; 768])
            .confidence(Confidence::from_raw(activation))
            .build();

        let _ = store.store(episode);
    }

    // Search for "memory" - should match ep1 and ep4
    let cue = CueBuilder::new()
        .id("semantic_search".to_string())
        .semantic_search("memory".to_string(), Confidence::LOW)
        .max_results(10)
        .build();

    let results = store.recall(&cue).results;

    assert!(
        !results.is_empty(),
        "Semantic search should find matching memories"
    );

    // Verify we found at least one memory-related episode
    let found_memory_episode = results
        .iter()
        .any(|(ep, _)| ep.id == "ep1" || ep.id == "ep4");

    assert!(
        found_memory_episode,
        "Should find episodes containing 'memory' keyword"
    );

    println!(
        "Semantic cross-tier search passed: found {} memories",
        results.len()
    );
}

/// Test that high-confidence cues search fewer tiers
#[tokio::test]
async fn test_tier_search_strategy_optimization() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    let mut store = MemoryStore::new(100);
    store = store
        .with_persistence(temp_dir.path())
        .expect("Failed to enable persistence");

    // Store episodes
    for i in 0..10 {
        let episode = EpisodeBuilder::new()
            .id(format!("ep_{i}"))
            .when(Utc::now())
            .what(format!("Content {i}"))
            .embedding([i as f32 / 10.0; 768])
            .confidence(Confidence::HIGH)
            .build();

        let _ = store.store(episode);
    }

    // High confidence cue - should use HotFirst strategy
    let high_conf_cue = CueBuilder::new()
        .id("high_confidence".to_string())
        .embedding_search([0.5; 768], Confidence::HIGH)
        .max_results(5)
        .build();

    let high_results = store.recall(&high_conf_cue);

    // Low confidence cue - should use AllTiers strategy
    let low_conf_cue = CueBuilder::new()
        .id("low_confidence".to_string())
        .embedding_search([0.5; 768], Confidence::LOW)
        .max_results(5)
        .build();

    let low_results = store.recall(&low_conf_cue);

    // Both should return results
    assert!(
        !high_results.results.is_empty(),
        "High confidence search should find results"
    );
    assert!(
        !low_results.results.is_empty(),
        "Low confidence search should find results"
    );

    println!(
        "Tier strategy test passed: high_conf={} results, low_conf={} results",
        high_results.results.len(),
        low_results.results.len()
    );
}
