//! Integration tests for end-to-end consolidation pipeline.
//!
//! Validates the complete flow from episodes to semantic memories:
//! Episodes → ConsolidationEngine → SemanticPatterns → Semantic Memories

use chrono::{Duration, Utc};
use engram_core::completion::{CompletionConfig, ConsolidationEngine};
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
fn test_end_to_end_consolidation_pipeline() {
    let config = CompletionConfig::default();
    let mut engine = ConsolidationEngine::new(config);

    // Create two clusters of similar episodes
    let mut cluster1 = create_similar_episodes(5, 0.8);
    let cluster2 = create_similar_episodes(4, 0.2);

    cluster1.extend(cluster2);
    let all_episodes = cluster1;

    // Perform consolidation
    engine.ripple_replay(&all_episodes);

    // Verify semantic patterns were extracted
    let patterns = engine.patterns();
    assert!(
        !patterns.is_empty(),
        "Should extract semantic patterns from episodes"
    );

    // Verify pattern properties
    for pattern in &patterns {
        assert!(!pattern.id.is_empty(), "Pattern should have ID");
        assert!(
            !pattern.source_episodes.is_empty(),
            "Pattern should reference source episodes"
        );
        assert!(
            pattern.strength >= 0.0 && pattern.strength <= 1.0,
            "Pattern strength should be in [0, 1]"
        );
        assert!(
            pattern.schema_confidence.raw() >= 0.0 && pattern.schema_confidence.raw() <= 1.0,
            "Schema confidence should be in [0, 1]"
        );
    }

    // Verify consolidation stats were updated
    let stats = engine.stats();
    assert_eq!(stats.total_replays, 1, "Should record one replay");
    assert!(
        stats.total_patterns_extracted > 0,
        "Should extract patterns"
    );
    assert!(
        stats.successful_consolidations > 0,
        "Should have successful consolidations"
    );
}

#[test]
fn test_episodic_to_semantic_transformation() {
    let config = CompletionConfig::default();
    let mut engine = ConsolidationEngine::new(config);

    // Create episodes
    let episodes = create_similar_episodes(6, 0.8);

    // Transform episodic to semantic memories
    let semantic_memories = engine.episodic_to_semantic(&episodes);

    // Verify semantic memories were created
    assert!(
        !semantic_memories.is_empty(),
        "Should create semantic memories from episodes"
    );

    // Verify memory properties
    for memory in &semantic_memories {
        assert!(!memory.id.is_empty(), "Memory should have ID");
        assert!(
            memory.confidence.raw() >= 0.0 && memory.confidence.raw() <= 1.0,
            "Memory confidence should be in [0, 1]"
        );
    }

    // Verify consolidation occurred
    let stats = engine.stats();
    assert!(
        stats.total_replays > 0,
        "Should have performed replay during consolidation"
    );
}

#[test]
fn test_consolidation_snapshot() {
    let config = CompletionConfig::default();
    let mut engine = ConsolidationEngine::new(config);

    // Create and consolidate episodes
    let episodes = create_similar_episodes(5, 0.9);
    engine.ripple_replay(&episodes);

    // Get snapshot
    let snapshot = engine.snapshot();

    // Verify snapshot properties
    assert!(
        !snapshot.patterns.is_empty(),
        "Snapshot should contain patterns"
    );
    assert_eq!(
        snapshot.stats.total_replays,
        engine.stats().total_replays,
        "Snapshot stats should match engine stats"
    );

    // Verify snapshot timestamp is recent
    let now = Utc::now();
    let age = now - snapshot.generated_at;
    assert!(
        age.num_seconds() < 2,
        "Snapshot should be generated recently"
    );
}

#[test]
fn test_consolidation_progress_tracking() {
    let config = CompletionConfig::default();
    let mut engine = ConsolidationEngine::new(config);

    // Create episode
    let episode = create_episode_with_embedding("test_ep", &[0.8; 768], 0);

    // Check initial progress (should be 0)
    let initial_progress = engine.get_consolidation_progress(&episode);
    assert!(
        initial_progress.abs() < f32::EPSILON,
        "Initial consolidation progress should be 0"
    );

    // Consolidate with this episode
    engine.ripple_replay(&vec![
        episode.clone(),
        create_episode_with_embedding("test_ep2", &[0.81; 768], 60),
        create_episode_with_embedding("test_ep3", &[0.82; 768], 120),
    ]);

    // Check progress after consolidation
    let progress = engine.get_consolidation_progress(&episode);
    // Progress may be 0 if no pattern was found, or >0 if pattern was created
    assert!(
        (0.0..=1.0).contains(&progress),
        "Progress should be in [0, 1]"
    );
}

#[test]
fn test_multiple_replay_cycles() {
    let config = CompletionConfig::default();
    let mut engine = ConsolidationEngine::new(config);

    // First replay cycle
    let episodes1 = create_similar_episodes(4, 0.8);
    engine.ripple_replay(&episodes1);

    let stats_after_first = engine.stats();
    assert_eq!(stats_after_first.total_replays, 1);

    // Second replay cycle
    let episodes2 = create_similar_episodes(5, 0.3);
    engine.ripple_replay(&episodes2);

    let stats_after_second = engine.stats();
    assert_eq!(stats_after_second.total_replays, 2);

    // Verify patterns accumulated
    let patterns = engine.patterns();
    assert!(
        !patterns.is_empty(),
        "Should have patterns from multiple replays"
    );
}

#[test]
fn test_pattern_retrieval_by_id() {
    let config = CompletionConfig::default();
    let mut engine = ConsolidationEngine::new(config);

    // Create and consolidate episodes
    let episodes = create_similar_episodes(5, 0.9);
    engine.ripple_replay(&episodes);

    let patterns = engine.patterns();
    if let Some(first_pattern) = patterns.first() {
        // Retrieve pattern by ID
        let retrieved = engine.pattern_by_id(&first_pattern.id);
        assert!(retrieved.is_some(), "Should retrieve pattern by ID");

        let retrieved_pattern = retrieved.unwrap();
        assert_eq!(
            retrieved_pattern.id, first_pattern.id,
            "Retrieved pattern should match"
        );
        assert_eq!(
            retrieved_pattern.source_episodes.len(),
            first_pattern.source_episodes.len(),
            "Source episodes should match"
        );
    }
}

#[test]
fn test_consolidation_stats_accuracy() {
    let config = CompletionConfig::default();
    let mut engine = ConsolidationEngine::new(config);

    // Verify initial stats
    let initial_stats = engine.stats();
    assert_eq!(initial_stats.total_replays, 0);
    assert_eq!(initial_stats.successful_consolidations, 0);
    assert_eq!(initial_stats.failed_consolidations, 0);
    assert_eq!(initial_stats.total_patterns_extracted, 0);

    // Perform consolidation with strong pattern
    let strong_cluster = create_similar_episodes(6, 0.9);
    engine.ripple_replay(&strong_cluster);

    let stats_after_strong = engine.stats();
    assert_eq!(stats_after_strong.total_replays, 1);
    assert!(
        stats_after_strong.total_patterns_extracted > 0,
        "Should extract patterns from strong cluster"
    );
}

#[test]
fn test_semantic_pattern_source_tracking() {
    let config = CompletionConfig::default();
    let mut engine = ConsolidationEngine::new(config);

    // Create episodes with known IDs
    let episodes: Vec<Episode> = (0..5)
        .map(|i| {
            let mut embedding = [0.85; 768];
            embedding[0] += (i as f32) * 0.01;
            create_episode_with_embedding(&format!("known_ep_{i}"), &embedding, i64::from(i) * 60)
        })
        .collect();

    let episode_ids: Vec<String> = episodes.iter().map(|ep| ep.id.clone()).collect();

    // Consolidate
    engine.ripple_replay(&episodes);

    // Verify patterns track their source episodes
    let patterns = engine.patterns();
    for pattern in &patterns {
        // Check that source episodes are from our known set
        for source_id in &pattern.source_episodes {
            assert!(
                episode_ids.contains(source_id),
                "Pattern source episodes should come from input episodes"
            );
        }
    }
}

#[test]
fn test_consolidation_with_empty_episodes() {
    let config = CompletionConfig::default();
    let mut engine = ConsolidationEngine::new(config);

    // Consolidate with empty episode list
    engine.ripple_replay(&[]);

    // Should handle gracefully without crashing
    let stats = engine.stats();
    assert_eq!(stats.total_replays, 0, "Should not count empty replay");

    let patterns = engine.patterns();
    assert!(
        patterns.is_empty(),
        "Should not create patterns from empty list"
    );
}

#[test]
fn test_systems_consolidation_time_delay() {
    use chrono::Duration;

    // Create old and new episodes
    let old_episode = Episode::new(
        "old_ep".to_string(),
        Utc::now() - Duration::hours(48),
        "old content".to_string(),
        [0.8; 768],
        Confidence::exact(0.9),
    );

    let new_episode = Episode::new(
        "new_ep".to_string(),
        Utc::now() - Duration::hours(1),
        "new content".to_string(),
        [0.8; 768],
        Confidence::exact(0.9),
    );

    let episodes = vec![old_episode, new_episode];

    // Systems consolidation with 24h delay
    let time_delay = Duration::hours(24);
    let semantic_memories = ConsolidationEngine::systems_consolidation(episodes, time_delay);

    // Only the old episode should be consolidated
    assert_eq!(
        semantic_memories.len(),
        1,
        "Only episodes older than time_delay should consolidate"
    );

    // Verify it's the old episode
    assert!(
        semantic_memories[0].id.contains("old_ep"),
        "Should consolidate the old episode"
    );
}
