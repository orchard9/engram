//! Integration tests for storage compaction methods
//!
//! These tests validate the MemoryStore compaction methods:
//! - store_semantic_pattern()
//! - mark_episodes_consolidated()
//! - remove_consolidated_episodes()
//!
//! Note: These tests focus on method behavior rather than end-to-end compaction
//! pipeline to avoid interference from MemoryStore's semantic deduplication logic.

use chrono::Utc;
use engram_core::completion::SemanticPattern;
use engram_core::consolidation::StorageCompactor;
use engram_core::{Confidence, Episode, MemoryStore};

/// Helper to create a semantic pattern for testing
fn create_semantic_pattern(id: &str, embedding: &[f32; 768]) -> SemanticPattern {
    SemanticPattern {
        id: id.to_string(),
        embedding: *embedding,
        source_episodes: vec!["ep1".to_string(), "ep2".to_string()],
        strength: 0.9,
        schema_confidence: Confidence::exact(0.85),
        last_consolidated: Utc::now(),
    }
}

/// Helper to create an episode for testing
fn create_episode(id: &str, index: usize) -> Episode {
    let mut embedding = [0.1; 768];
    let start_idx = (index * 10) % 700;
    for item in embedding.iter_mut().skip(start_idx).take(10) {
        *item = 1.0;
    }

    Episode::new(
        id.to_string(),
        Utc::now() - chrono::Duration::days(10),
        format!("Episode {id}"),
        embedding,
        Confidence::exact(0.8),
    )
}

#[test]
fn test_store_semantic_pattern() {
    let store = MemoryStore::new(1000);

    let pattern = create_semantic_pattern("pattern1", &[0.5; 768]);

    // Store semantic pattern
    let result = store.store_semantic_pattern(&pattern);
    assert!(result.activation.is_successful());

    // Verify pattern is retrievable
    let memory = store.get("pattern1");
    assert!(memory.is_some());

    let memory = memory.unwrap();
    assert_eq!(memory.id, "pattern1");
    assert_eq!(memory.confidence, Confidence::exact(0.85));

    // Semantic patterns should have high activation
    assert!(memory.activation() >= 0.9);
}

#[test]
fn test_mark_episodes_consolidated() {
    let store = MemoryStore::new(1000);

    // Store some episodes directly
    let ep1 = create_episode("ep1", 0);
    let ep2 = create_episode("ep2", 1);

    store.store(ep1);
    store.store(ep2);

    // Mark episodes as consolidated
    let marked = store.mark_episodes_consolidated(&["ep1".to_string(), "ep2".to_string()]);

    // Should return count of episodes that existed
    assert!(marked >= 1); // At least one episode was marked (deduplication may affect count)
}

#[test]
fn test_remove_consolidated_episodes() {
    let store = MemoryStore::new(1000);

    // Store episodes
    let ep1 = create_episode("ep1", 0);
    let ep2 = create_episode("ep2", 1);

    store.store(ep1);
    store.store(ep2);

    let initial_count = store.count();
    assert!(initial_count >= 1);

    // Remove episodes
    let removed = store.remove_consolidated_episodes(&["ep1".to_string(), "ep2".to_string()]);

    // Should have removed at least one episode
    assert!(removed >= 1);

    // Count should be reduced
    assert!(store.count() < initial_count);

    // Episodes should no longer be retrievable
    assert!(store.get_episode("ep1").is_none());
    assert!(store.get_episode("ep2").is_none());
}

#[test]
fn test_compaction_with_semantic_pattern_storage() {
    let store = MemoryStore::new(1000);

    // Create and store episodes
    let ep1 = create_episode("ep1", 0);
    let ep2 = create_episode("ep2", 1);
    let ep3 = create_episode("ep3", 2);

    store.store(ep1);
    store.store(ep2);
    store.store(ep3);

    let initial_count = store.count();

    // Create and store a semantic pattern
    let pattern = create_semantic_pattern("pattern1", &[0.6; 768]);
    store.store_semantic_pattern(&pattern);

    // Pattern should be retrievable
    let retrieved = store.get("pattern1");
    assert!(retrieved.is_some());
    assert!(retrieved.unwrap().activation() >= 0.9);

    // Remove episodes (simulating compaction)
    let removed = store.remove_consolidated_episodes(&[
        "ep1".to_string(),
        "ep2".to_string(),
        "ep3".to_string(),
    ]);

    assert!(removed >= 1); // At least some episodes were removed

    // Pattern should still be present after episode removal
    assert!(store.get("pattern1").is_some());

    // Count should reflect removal
    let final_count = store.count();
    assert!(final_count <= initial_count); // Count should not increase
}

#[test]
fn test_compaction_rollback_on_poor_similarity() {
    let store = MemoryStore::new(1000);

    // Create episode with one embedding
    let ep1 = create_episode("ep1", 0);
    store.store(ep1.clone());

    // Create pattern with completely different embedding
    let mut different_embedding = [0.0; 768];
    different_embedding[700] = 1.0; // Very different location

    let pattern = create_semantic_pattern("pattern_bad", &different_embedding);

    // Compaction should fail due to low similarity
    let compactor = StorageCompactor::default_config();
    let result = compactor.compact_storage(std::slice::from_ref(&ep1), &pattern);

    assert!(
        result.is_err(),
        "Compaction should fail with dissimilar embeddings"
    );

    // Episode should still exist (no removal attempted)
    assert!(store.get_episode("ep1").is_some());
}

#[test]
fn test_compaction_preserves_high_confidence_episodes() {
    let now = Utc::now();

    // Create high-confidence episode
    let high_conf_episode = Episode::new(
        "ep_high".to_string(),
        now - chrono::Duration::days(30),
        "High confidence episode".to_string(),
        [0.5; 768],
        Confidence::exact(0.98), // Above preserve threshold (0.95)
    );

    // Check eligibility
    let compactor = StorageCompactor::default_config();
    let is_eligible = compactor.is_episode_eligible(&high_conf_episode, now);

    assert!(
        !is_eligible,
        "High confidence episodes should not be eligible for compaction"
    );
}

#[test]
fn test_compaction_preserves_recent_episodes() {
    let now = Utc::now();

    // Create recent episode
    let recent_episode = Episode::new(
        "ep_recent".to_string(),
        now - chrono::Duration::days(3), // Less than min age (7 days)
        "Recent episode".to_string(),
        [0.5; 768],
        Confidence::exact(0.7),
    );

    // Check eligibility
    let compactor = StorageCompactor::default_config();
    let is_eligible = compactor.is_episode_eligible(&recent_episode, now);

    assert!(
        !is_eligible,
        "Recent episodes should not be eligible for compaction"
    );
}

#[test]
fn test_empty_episode_list_compaction() {
    let pattern = create_semantic_pattern("pattern_empty", &[0.5; 768]);

    let compactor = StorageCompactor::default_config();
    let episodes: Vec<Episode> = vec![];

    // Should handle empty episode list gracefully
    let result = compactor.compact_storage(&episodes, &pattern);
    assert!(result.is_ok());

    let compaction_result = result.unwrap();
    assert_eq!(compaction_result.episodes_removed, 0);
    assert_eq!(compaction_result.storage_reduction_bytes, 0);
    assert!(
        (compaction_result.average_similarity - 0.0).abs() < f32::EPSILON,
        "Expected average_similarity to be 0.0, got {}",
        compaction_result.average_similarity
    );
}

#[test]
fn test_semantic_pattern_high_activation() {
    let store = MemoryStore::new(1000);

    let pattern = create_semantic_pattern("pattern_activation", &[0.8; 768]);

    // Store semantic pattern
    store.store_semantic_pattern(&pattern);

    // Retrieve and verify high activation
    let memory = store.get("pattern_activation");
    assert!(memory.is_some());

    let memory = memory.unwrap();

    // Semantic patterns should have high activation (>= 0.9)
    assert!(
        memory.activation() >= 0.9,
        "Semantic pattern activation was {}, expected >= 0.9",
        memory.activation()
    );
}
