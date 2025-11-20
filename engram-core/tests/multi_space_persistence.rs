//! Multi-space persistence integration tests for Milestone 7 Task 003
//!
//! Tests verify that persistence is properly isolated across memory spaces:
//! - Separate WAL directories per space
//! - Isolated tier storage (hot/warm/cold)
//! - Concurrent writes to different spaces
//! - Recovery integrity per space

#![cfg(feature = "memory_mapped_persistence")]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::too_many_lines)]

use chrono::Utc;
use std::path::Path;
use std::sync::Arc;
use tempfile::tempdir;

use engram_core::{
    Confidence, Episode, MemorySpaceId, MemorySpaceRegistry, MemoryStore,
    storage::{FsyncMode, PersistenceConfig},
};

/// Create a test registry with custom persistence configuration
fn create_test_registry(data_root: &Path, config: PersistenceConfig) -> Arc<MemorySpaceRegistry> {
    Arc::new(
        MemorySpaceRegistry::with_persistence_config(
            data_root,
            |space_id, _directories| {
                let store = MemoryStore::for_space(space_id.clone(), 100);
                Ok(Arc::new(store))
            },
            config,
        )
        .expect("registry creation"),
    )
}

#[tokio::test]
async fn test_separate_wal_directories_per_space() {
    let temp_dir = tempdir().expect("temp dir");
    let config = PersistenceConfig {
        hot_capacity: 1000,
        warm_capacity: 10000,
        cold_capacity: 100_000,
        fsync_mode: FsyncMode::PerBatch,
    };

    let registry = create_test_registry(temp_dir.path(), config);

    // Create two distinct memory spaces
    let space_alpha = MemorySpaceId::new("alpha").expect("valid space id");
    let space_beta = MemorySpaceId::new("beta").expect("valid space id");

    let alpha_handle = registry.create_or_get(&space_alpha).await.expect("alpha");
    let beta_handle = registry.create_or_get(&space_beta).await.expect("beta");

    // Verify directory structure: <root>/<space_id>/{wal,hot,warm,cold}
    let alpha_dirs = alpha_handle.directories();
    let beta_dirs = beta_handle.directories();

    // Check root directories are separate
    assert!(alpha_dirs.root.exists(), "Alpha root should exist");
    assert!(beta_dirs.root.exists(), "Beta root should exist");
    assert_ne!(alpha_dirs.root, beta_dirs.root, "Roots must be separate");

    // Check all tier directories exist
    for (space, dirs) in [("alpha", alpha_dirs), ("beta", beta_dirs)] {
        assert!(dirs.wal.exists(), "{space} WAL directory should exist");
        assert!(dirs.hot.exists(), "{space} hot tier directory should exist");
        assert!(
            dirs.warm.exists(),
            "{space} warm tier directory should exist"
        );
        assert!(
            dirs.cold.exists(),
            "{space} cold tier directory should exist"
        );
    }

    // Verify WAL directories are isolated
    assert_ne!(
        alpha_dirs.wal, beta_dirs.wal,
        "WAL directories must be separate"
    );
}

#[tokio::test]
async fn test_concurrent_writes_to_different_spaces() {
    let temp_dir = tempdir().expect("temp dir");
    let config = PersistenceConfig {
        hot_capacity: 1000,
        warm_capacity: 10000,
        cold_capacity: 100_000,
        fsync_mode: FsyncMode::PerWrite,
    };

    let registry = create_test_registry(temp_dir.path(), config);

    // Create 5 spaces and write concurrently
    let spaces: Vec<_> = (0..5)
        .map(|i| MemorySpaceId::new(format!("concurrent_{i}")).expect("valid"))
        .collect();

    let mut handles = vec![];
    for (idx, space_id) in spaces.iter().enumerate() {
        let registry = Arc::clone(&registry);
        let space_id = space_id.clone();

        let handle = tokio::spawn(async move {
            let h = registry.create_or_get(&space_id).await.expect("space");
            let store = h.store();

            for i in 0..10 {
                // Create distinct embeddings for each episode
                // Use modulo patterns based on episode index to ensure low similarity
                let mut emb = [0.0f32; 768];
                for (dim, val) in emb.iter_mut().enumerate() {
                    // Different episodes activate completely different dimensions
                    // This ensures cosine similarity < 0.95 between episodes
                    let activated_range = i * 76; // Each episode activates a different 76-dim block
                    if dim >= activated_range && dim < activated_range + 76 {
                        *val = 1.0 + (idx as f32 * 0.1);
                    } else {
                        *val = 0.0;
                    }
                }

                let ep = Episode::new(
                    format!("s{idx}_e{i}"),
                    Utc::now(),
                    format!("Space {idx} content {i}"),
                    emb,
                    Confidence::exact(0.9),
                );
                let _ = store.store(ep);
            }

            (space_id, store.count())
        });

        handles.push(handle);
    }

    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.expect("task"))
        .collect();

    for (space_id, count) in results {
        assert_eq!(count, 10, "Space {space_id} should have 10 episodes");
    }
}

#[tokio::test]
async fn test_space_directories_are_isolated() {
    let temp_dir = tempdir().expect("temp dir");
    let config = PersistenceConfig::default();
    let registry = create_test_registry(temp_dir.path(), config);

    let spaces = vec![
        MemorySpaceId::new("alpha").expect("valid"),
        MemorySpaceId::new("beta").expect("valid"),
        MemorySpaceId::new("gamma").expect("valid"),
    ];

    let mut all_dirs = vec![];
    for space_id in &spaces {
        let handle = registry.create_or_get(space_id).await.expect("create");
        all_dirs.push(handle.directories().clone());
    }

    // Verify all directories are unique
    for i in 0..all_dirs.len() {
        for j in (i + 1)..all_dirs.len() {
            assert_ne!(all_dirs[i].root, all_dirs[j].root);
            assert_ne!(all_dirs[i].wal, all_dirs[j].wal);
            assert_ne!(all_dirs[i].hot, all_dirs[j].hot);
        }
    }
}
