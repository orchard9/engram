//! Integration tests for the memory space registry lifecycle.

use std::sync::Arc;

use engram_core::{MemorySpaceId, MemorySpaceRegistry, MemoryStore};
use tempfile::tempdir;

fn test_registry(temp_path: &std::path::Path) -> MemorySpaceRegistry {
    MemorySpaceRegistry::new(temp_path, |_, _| Ok(Arc::new(MemoryStore::new(128))))
        .expect("registry")
}

#[tokio::test]
async fn create_or_get_is_idempotent_under_concurrency() {
    let temp_dir = tempdir().expect("temp dir");
    let registry = test_registry(temp_dir.path());
    let space = MemorySpaceId::try_from("alpha_space").unwrap();

    let (first, second) = tokio::join!(
        registry.create_or_get(&space),
        registry.create_or_get(&space)
    );

    let first = first.expect("first handle");
    let second = second.expect("second handle");

    assert!(Arc::ptr_eq(&first.store(), &second.store()));
    assert_eq!(first.id(), second.id());
}

#[tokio::test]
async fn directories_are_created_per_space() {
    let temp_dir = tempdir().expect("temp dir");
    let registry = test_registry(temp_dir.path());
    let id = MemorySpaceId::try_from("tenant_beta").unwrap();

    let handle = registry.create_or_get(&id).await.expect("space handle");
    let dirs = handle.directories();

    assert!(dirs.root.exists(), "root directory missing");
    assert!(dirs.wal.exists(), "wal directory missing");
    assert!(dirs.hot.exists(), "hot directory missing");
    assert!(dirs.warm.exists(), "warm directory missing");
    assert!(dirs.cold.exists(), "cold directory missing");
}

#[tokio::test]
async fn list_returns_sorted_space_summaries() {
    let temp_dir = tempdir().expect("temp dir");
    let registry = test_registry(temp_dir.path());

    let alpha = MemorySpaceId::try_from("alpha").unwrap();
    let beta = MemorySpaceId::try_from("beta").unwrap();
    let gamma = MemorySpaceId::try_from("gamma").unwrap();

    registry
        .ensure_spaces(vec![beta.clone(), gamma.clone(), alpha.clone()])
        .await
        .unwrap();

    let summaries = registry.list();
    let collected: Vec<_> = summaries.into_iter().map(|s| s.id).collect();
    assert_eq!(collected, vec![alpha, beta, gamma]);
}
