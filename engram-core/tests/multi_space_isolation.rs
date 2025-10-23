//! Integration tests proving memory space isolation guarantees.
//!
//! These tests validate that the registry-based isolation pattern prevents
//! cross-space memory leakage under various scenarios.

use engram_core::{Confidence, Cue, Episode, MemorySpaceId, MemorySpaceRegistry, MemoryStore};
use std::sync::Arc;
use tempfile::tempdir;

/// Test helper: Create a registry with custom store factory
fn test_registry(temp_path: &std::path::Path) -> MemorySpaceRegistry {
    MemorySpaceRegistry::new(temp_path, |space_id, _directories| {
        Ok(Arc::new(MemoryStore::for_space(
            space_id.clone(),
            1000,
        )))
    })
    .expect("registry creation")
}

#[tokio::test]
async fn spaces_are_isolated_different_episodes_same_id() {
    // CRITICAL TEST: Two spaces can have episodes with same ID without collision
    let temp_dir = tempdir().expect("temp dir");
    let registry = test_registry(temp_dir.path());

    let alpha = MemorySpaceId::try_from("alpha").unwrap();
    let beta = MemorySpaceId::try_from("beta").unwrap();

    let alpha_handle = registry.create_or_get(&alpha).await.unwrap();
    let beta_handle = registry.create_or_get(&beta).await.unwrap();

    let alpha_store = alpha_handle.store();
    let beta_store = beta_handle.store();

    // Same episode ID, different content
    let episode_alpha = Episode::new(
        "ep_001".to_string(),
        chrono::Utc::now(),
        "Alpha's secret data".to_string(),
        [0.1; 768],
        Confidence::exact(0.9),
    );

    let episode_beta = Episode::new(
        "ep_001".to_string(),
        chrono::Utc::now(),
        "Beta's secret data".to_string(),
        [0.9; 768],
        Confidence::exact(0.9),
    );

    // Store in separate spaces
    alpha_store.store(episode_alpha);
    beta_store.store(episode_beta);

    // Recall from alpha using embedding similarity - should only see alpha's data
    let cue_alpha = Cue::embedding(
        "alpha_query".to_string(),
        [0.1; 768], // Matches alpha's embedding
        Confidence::exact(0.1),
    );
    let alpha_results = alpha_store.recall(&cue_alpha);

    assert!(
        !alpha_results.results.is_empty(),
        "Alpha space should have at least one result"
    );
    assert_eq!(alpha_results.results[0].0.id, "ep_001");
    assert_eq!(alpha_results.results[0].0.what, "Alpha's secret data");

    // Recall from beta using its embedding - should only see beta's data
    let cue_beta = Cue::embedding(
        "beta_query".to_string(),
        [0.9; 768], // Matches beta's embedding
        Confidence::exact(0.1),
    );
    let beta_results = beta_store.recall(&cue_beta);

    assert!(
        !beta_results.results.is_empty(),
        "Beta space should have at least one result"
    );
    assert_eq!(beta_results.results[0].0.id, "ep_001");
    assert_eq!(beta_results.results[0].0.what, "Beta's secret data");

    // ISOLATION PROOF: Same ID, different content, no leakage
    // Each space only sees its own data even with same episode ID
}

#[tokio::test]
async fn verify_space_catches_wrong_store_usage() {
    let temp_dir = tempdir().expect("temp dir");
    let registry = test_registry(temp_dir.path());

    let alpha = MemorySpaceId::try_from("alpha").unwrap();
    let beta = MemorySpaceId::try_from("beta").unwrap();

    let alpha_handle = registry.create_or_get(&alpha).await.unwrap();
    let alpha_store = alpha_handle.store();

    // Runtime guard should catch mismatch
    let result = alpha_store.verify_space(&beta);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("Memory space mismatch"));

    // Correct usage should succeed
    let result = alpha_store.verify_space(&alpha);
    assert!(result.is_ok());
}

#[tokio::test]
async fn space_handle_returns_correct_space_id() {
    let temp_dir = tempdir().expect("temp dir");
    let registry = test_registry(temp_dir.path());

    let tenant = MemorySpaceId::try_from("tenant_xyz").unwrap();
    let handle = registry.create_or_get(&tenant).await.unwrap();

    assert_eq!(handle.id().as_str(), "tenant_xyz");
    assert_eq!(handle.store().space_id().as_str(), "tenant_xyz");
}

#[tokio::test]
async fn concurrent_operations_across_spaces_dont_interfere() {
    // Stress test: Multiple spaces with concurrent writes
    let temp_dir = tempdir().expect("temp dir");
    let registry = Arc::new(test_registry(temp_dir.path()));

    let spaces: Vec<MemorySpaceId> = (0..5)
        .map(|i| MemorySpaceId::try_from(format!("space_{i}").as_str()).unwrap())
        .collect();

    // Spawn concurrent tasks writing to different spaces
    let handles: Vec<_> = spaces
        .iter()
        .enumerate()
        .map(|(i, space_id)| {
            let registry = Arc::clone(&registry);
            let space = space_id.clone();
            tokio::spawn(async move {
                let handle = registry.create_or_get(&space).await.unwrap();
                let store = handle.store();

                // Write 10 episodes
                for j in 0..10 {
                    let episode = Episode::new(
                        format!("ep_{i}_{j}"),
                        chrono::Utc::now(),
                        format!("Data from space {i}, episode {j}"),
                        [i as f32; 768],
                        Confidence::exact(0.9),
                    );
                    store.store(episode);
                }

                // Verify all 10 are in this space
                let cue = Cue::semantic(
                    "test".to_string(),
                    format!("space {i}"),
                    Confidence::exact(0.3),
                );
                let results = store.recall(&cue);
                (space, results.results.len())
            })
        })
        .collect();

    // Wait for all concurrent operations
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // Verify each space has correct count
    for (space, count) in results {
        // Should recall at least some of the 10 episodes
        assert!(
            count > 0,
            "Space {space} should have recalled some episodes"
        );
    }
}

#[tokio::test]
async fn default_space_backward_compatibility() {
    let temp_dir = tempdir().expect("temp dir");
    let registry = test_registry(temp_dir.path());

    // Default space should work
    let default = MemorySpaceId::default();
    let handle = registry.create_or_get(&default).await.unwrap();

    assert_eq!(handle.id().as_str(), "default");

    // Can store and recall
    let episode = Episode::new(
        "test_default".to_string(),
        chrono::Utc::now(),
        "Default space data".to_string(),
        [0.5; 768],
        Confidence::exact(0.8),
    );

    let store = handle.store();
    store.store(episode);

    let cue = Cue::semantic("test".to_string(), "default".to_string(), Confidence::exact(0.5));
    let results = store.recall(&cue);

    assert_eq!(results.results.len(), 1);
}
