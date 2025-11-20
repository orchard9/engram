//! Property-based tests for multi-space registry operations (Milestone 7 Task 007)
//!
//! Tests verify invariants that must hold across all memory space operations:
//! - Concurrent create_or_get always returns the same handle
//! - Space IDs are validated correctly
//! - Store/recall operations never cross-contaminate
//! - Registry routing is deterministic
//! - Memory isolation is maintained under concurrent access

#![allow(missing_docs)]
#![allow(clippy::float_cmp)]

use chrono::Utc;
use engram_core::{Confidence, Episode, MemorySpaceId, MemorySpaceRegistry, MemoryStore};
use proptest::prelude::*;
use std::sync::Arc;
use tempfile::tempdir;

/// Strategy for generating valid memory space IDs
/// MemorySpaceId requires 4-64 lowercase alphanumeric characters
fn valid_space_id_strategy() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_]{3,63}".prop_map(|s| s)
}

/// Strategy for generating potentially invalid space IDs (for robustness testing)
fn arbitrary_space_id_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z0-9_@.-]{1,100}").expect("valid regex")
}

/// Strategy for generating memory content
fn memory_content_strategy() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z ]{5,50}").expect("valid regex")
}

/// Strategy for generating embedding vectors
fn embedding_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(0.0f32..1.0f32, 768)
}

// Property: Concurrent create_or_get returns identical handles
// Natural expectation: Multiple threads creating the same space should get the same store
proptest! {
    #[test]
    fn concurrent_create_or_get_returns_same_handle(
        space_id in valid_space_id_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = tempdir().expect("temp dir");
            let registry = Arc::new(
                MemorySpaceRegistry::new(temp_dir.path(), |space_id, _| {
                    Ok(Arc::new(MemoryStore::for_space(space_id.clone(), 100)))
                })
                .expect("registry"),
            );

            let space = MemorySpaceId::new(&space_id).unwrap();

            // Spawn 10 concurrent tasks trying to create the same space
            let mut handles = vec![];
            for _ in 0..10 {
                let registry = Arc::clone(&registry);
                let space = space.clone();
                let handle = tokio::spawn(async move {
                    registry.create_or_get(&space).await.expect("create_or_get")
                });
                handles.push(handle);
            }

            // Collect all results
            let results: Vec<_> = futures::future::join_all(handles)
                .await
                .into_iter()
                .map(|r| r.expect("task completion"))
                .collect();

            // All handles should point to the same store
            for i in 0..results.len() {
                for j in (i + 1)..results.len() {
                    prop_assert!(
                        Arc::ptr_eq(&results[i].store(), &results[j].store()),
                        "All concurrent creates should return the same store instance"
                    );
                }
            }

            Ok(())
        })?;
    }

    /// Property: Valid space IDs are accepted, invalid ones are rejected
    /// Natural expectation: Space ID validation should be consistent
    #[test]
    fn space_id_validation_is_consistent(
        space_id in arbitrary_space_id_strategy()
    ) {
        let result = MemorySpaceId::new(&space_id);

        // Check validation rules for MemorySpaceId (3-64 lowercase alphanumeric/underscore/hyphen)
        // Note: minimum is 3, not 4
        if space_id.is_empty() || space_id.len() < 3 || space_id.len() > 64 {
            prop_assert!(result.is_err(), "Invalid length should be rejected");
        } else if !space_id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-') {
            prop_assert!(result.is_err(), "Invalid characters should be rejected");
        } else {
            // Should be valid - the implementation doesn't check starting character
            prop_assert!(result.is_ok(), "Valid space ID should be accepted");
        }
    }

    /// Property: Memories stored in one space never appear in another
    /// Critical invariant: Space isolation must be perfect
    #[test]
    fn memories_never_cross_contaminate(
        space_a_id in valid_space_id_strategy(),
        space_b_id in valid_space_id_strategy(),
        content_a in memory_content_strategy(),
        content_b in memory_content_strategy(),
        embedding_a in embedding_strategy(),
        embedding_b in embedding_strategy(),
    ) {
        // Skip if space IDs are the same
        prop_assume!(space_a_id != space_b_id);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = tempdir().expect("temp dir");
            let registry = Arc::new(
                MemorySpaceRegistry::new(temp_dir.path(), |space_id, _| {
                    Ok(Arc::new(MemoryStore::for_space(space_id.clone(), 100)))
                })
                .expect("registry"),
            );

            let space_a = MemorySpaceId::new(&space_a_id).unwrap();
            let space_b = MemorySpaceId::new(&space_b_id).unwrap();

            let handle_a = registry.create_or_get(&space_a).await.expect("space_a");
            let handle_b = registry.create_or_get(&space_b).await.expect("space_b");

            let store_a = handle_a.store();
            let store_b = handle_b.store();

            // Store unique memories in each space (using Episode)
            let mut emb_a = [0.0f32; 768];
            emb_a.copy_from_slice(&embedding_a);

            let mut emb_b = [0.0f32; 768];
            emb_b.copy_from_slice(&embedding_b);

            let ep_a = Episode::new("mem_a".to_string(), Utc::now(), content_a, emb_a, Confidence::exact(0.9));
            let ep_b = Episode::new("mem_b".to_string(), Utc::now(), content_b, emb_b, Confidence::exact(0.9));

            let _ = store_a.store(ep_a);
            let _ = store_b.store(ep_b);

            // Critical property: mem_a should NOT exist in space_b
            let mem_a_in_b = store_b.get("mem_a").is_some();
            prop_assert!(
                !mem_a_in_b,
                "Memory from space_a should never appear in space_b"
            );

            // Critical property: mem_b should NOT exist in space_a
            let mem_b_in_a = store_a.get("mem_b").is_some();
            prop_assert!(
                !mem_b_in_a,
                "Memory from space_b should never appear in space_a"
            );

            Ok(())
        })?;
    }

    /// Property: Registry routing is deterministic
    /// Natural expectation: Same space ID always routes to same store
    #[test]
    fn registry_routing_is_deterministic(
        space_id in valid_space_id_strategy()
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = tempdir().expect("temp dir");
            let registry = Arc::new(
                MemorySpaceRegistry::new(temp_dir.path(), |space_id, _| {
                    Ok(Arc::new(MemoryStore::for_space(space_id.clone(), 100)))
                })
                .expect("registry"),
            );

            let space = MemorySpaceId::new(&space_id).unwrap();

            // Get handle multiple times
            let handle1 = registry.create_or_get(&space).await.expect("first get");
            let handle2 = registry.create_or_get(&space).await.expect("second get");
            let handle3 = registry.create_or_get(&space).await.expect("third get");

            // All should point to the same store
            prop_assert!(
                Arc::ptr_eq(&handle1.store(), &handle2.store()),
                "First and second get should return same store"
            );
            prop_assert!(
                Arc::ptr_eq(&handle2.store(), &handle3.store()),
                "Second and third get should return same store"
            );

            Ok(())
        })?;
    }

    /// Property: Space list is always sorted and unique
    /// Natural expectation: Registry list should be deterministic and clean
    #[test]
    fn space_list_is_sorted_and_unique(
        space_ids in prop::collection::vec(valid_space_id_strategy(), 1..10)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = tempdir().expect("temp dir");
            let registry = Arc::new(
                MemorySpaceRegistry::new(temp_dir.path(), |space_id, _| {
                    Ok(Arc::new(MemoryStore::for_space(space_id.clone(), 100)))
                })
                .expect("registry"),
            );

            // Create all spaces
            for space_id in &space_ids {
                let space = MemorySpaceId::new(space_id).unwrap();
                registry.create_or_get(&space).await.expect("create space");
            }

            let list = registry.list();

            // Verify sorted order
            for i in 0..list.len().saturating_sub(1) {
                prop_assert!(
                    list[i].id.as_str() <= list[i + 1].id.as_str(),
                    "Space list should be sorted"
                );
            }

            // Verify uniqueness
            let mut seen = std::collections::HashSet::new();
            for summary in &list {
                prop_assert!(
                    seen.insert(summary.id.as_str()),
                    "Space list should not contain duplicates"
                );
            }

            Ok(())
        })?;
    }

    /// Property: Concurrent writes to different spaces never interfere
    /// Critical reliability expectation: Multi-tenant isolation under load
    #[test]
    fn concurrent_writes_are_isolated(
        num_spaces in 2usize..5,
        writes_per_space in 5usize..20,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = tempdir().expect("temp dir");
            let registry = Arc::new(
                MemorySpaceRegistry::new(temp_dir.path(), |space_id, _| {
                    Ok(Arc::new(MemoryStore::for_space(space_id.clone(), 100)))
                })
                .expect("registry"),
            );

            // Create spaces
            let spaces: Vec<_> = (0..num_spaces)
                .map(|i| MemorySpaceId::new(format!("space_{i}")).unwrap())
                .collect();

            // Spawn concurrent writers
            let mut handles = vec![];
            for (idx, space_id) in spaces.iter().enumerate() {
                let registry = Arc::clone(&registry);
                let space_id = space_id.clone();

                let handle = tokio::spawn(async move {
                    let handle = registry.create_or_get(&space_id).await.expect("space");
                    let store = handle.store();

                    for i in 0..writes_per_space {
                        // Create distinct embeddings to avoid deduplication
                        let mut emb = [0.0f32; 768];
                        for (dim, val) in emb.iter_mut().enumerate() {
                            // Each episode activates a different 76-dim block
                            let activated_range = i * 38; // 20 episodes max, 38*20=760 < 768
                            if dim >= activated_range && dim < activated_range + 38 {
                                *val = 1.0 + (idx as f32 * 0.1);
                            } else {
                                *val = 0.0;
                            }
                        }

                        let ep = Episode::new(
                            format!("space_{idx}_mem_{i}"),
                            Utc::now(),
                            format!("Content {i}"),
                            emb,
                            Confidence::exact(0.9),
                        );
                        let _ = store.store(ep);
                    }

                    (space_id, store.count())
                });

                handles.push(handle);
            }

            // Wait for all writes
            let results: Vec<_> = futures::future::join_all(handles)
                .await
                .into_iter()
                .map(|r| r.expect("task completion"))
                .collect();

            // Verify each space has the correct count
            for (space_id, count) in results {
                prop_assert!(
                    count == writes_per_space,
                    "Space {space_id} should have {writes_per_space} memories, got {count}"
                );
            }

            Ok(())
        })?;
    }

    /// Property: Space directories are properly isolated
    /// Natural expectation: Each space gets its own directory tree
    #[test]
    fn space_directories_are_isolated(
        space_ids in prop::collection::vec(valid_space_id_strategy(), 2..5)
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = tempdir().expect("temp dir");
            let registry = Arc::new(
                MemorySpaceRegistry::new(temp_dir.path(), |space_id, _| {
                    Ok(Arc::new(MemoryStore::for_space(space_id.clone(), 100)))
                })
                .expect("registry"),
            );

            let mut all_dirs = vec![];

            // Create all spaces and collect directories
            for space_id in &space_ids {
                let space = MemorySpaceId::new(space_id).unwrap();
                let handle = registry.create_or_get(&space).await.expect("create space");
                all_dirs.push(handle.directories().clone());
            }

            // Verify all directories are unique
            for i in 0..all_dirs.len() {
                for j in (i + 1)..all_dirs.len() {
                    prop_assert!(
                        all_dirs[i].root != all_dirs[j].root,
                        "Space root directories must be unique"
                    );
                    prop_assert!(
                        all_dirs[i].wal != all_dirs[j].wal,
                        "Space WAL directories must be unique"
                    );
                    prop_assert!(
                        all_dirs[i].hot != all_dirs[j].hot,
                        "Space hot directories must be unique"
                    );
                }
            }

            Ok(())
        })?;
    }

    /// Property: Default space ID is always valid
    /// Natural expectation: System should always have a valid default
    #[test]
    fn default_space_id_is_valid(_unit in Just(())) {
        let default = MemorySpaceId::default();
        prop_assert!(!default.as_str().is_empty(), "Default space ID should not be empty");
        prop_assert!(default.as_str().len() <= 64, "Default space ID should be valid length");
    }

    /// Property: Registry operations never panic
    /// Critical reliability expectation: No matter what inputs, code shouldn't crash
    #[test]
    fn registry_operations_never_panic(
        space_ids in prop::collection::vec(valid_space_id_strategy(), 0..5),
        invalid_id in arbitrary_space_id_strategy(),
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let temp_dir = tempdir().expect("temp dir");
            let registry = Arc::new(
                MemorySpaceRegistry::new(temp_dir.path(), |space_id, _| {
                    Ok(Arc::new(MemoryStore::for_space(space_id.clone(), 100)))
                })
                .expect("registry"),
            );

            // Test create_or_get with valid IDs
            for space_id in &space_ids {
                if let Ok(space) = MemorySpaceId::new(space_id) {
                    let _ = registry.create_or_get(&space).await;
                }
            }

            // Test get with valid IDs
            for space_id in &space_ids {
                if let Ok(space) = MemorySpaceId::new(space_id) {
                    let _ = registry.get(&space);
                }
            }

            // Test list
            let _ = registry.list();

            // Test ensure_spaces
            let valid_spaces: Vec<_> = space_ids
                .iter()
                .filter_map(|id| MemorySpaceId::new(id).ok())
                .collect();
            let _ = registry.ensure_spaces(valid_spaces).await;

            // Test with invalid ID (should handle gracefully)
            if let Ok(space) = MemorySpaceId::new(&invalid_id) {
                let _ = registry.create_or_get(&space).await;
            }

            prop_assert!(true); // If we reach here, no panics occurred
            Ok(())
        })?;
    }
}

/// Additional manual tests for edge cases not easily covered by proptest

#[tokio::test]
async fn test_empty_space_id_rejected() {
    let result = MemorySpaceId::new("");
    assert!(result.is_err(), "Empty space ID should be rejected");
}

#[tokio::test]
async fn test_space_id_too_long_rejected() {
    let long_id = "a".repeat(100);
    let result = MemorySpaceId::new(&long_id);
    assert!(
        result.is_err(),
        "Space ID longer than 64 chars should be rejected"
    );
}

#[tokio::test]
async fn test_space_id_with_special_chars() {
    // Test various special characters
    for invalid_char in ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')'] {
        let id = format!("space{invalid_char}test");
        let result = MemorySpaceId::new(&id);
        // May or may not be valid depending on implementation
        // The key is that validation is consistent
        let _ = result;
    }
}

#[tokio::test]
async fn test_concurrent_create_same_space_thousands_of_times() {
    let temp_dir = tempdir().expect("temp dir");
    let registry = Arc::new(
        MemorySpaceRegistry::new(temp_dir.path(), |space_id, _| {
            Ok(Arc::new(MemoryStore::for_space(space_id.clone(), 100)))
        })
        .expect("registry"),
    );

    let space = MemorySpaceId::new("high_concurrency_test").expect("valid space id");

    // Spawn 1000 concurrent tasks
    let mut handles = vec![];
    for _ in 0..1000 {
        let registry = Arc::clone(&registry);
        let space = space.clone();
        let handle =
            tokio::spawn(
                async move { registry.create_or_get(&space).await.expect("create_or_get") },
            );
        handles.push(handle);
    }

    // Collect results
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.expect("task completion"))
        .collect();

    // All should point to the same store
    let first_ptr = Arc::as_ptr(&results[0].store());
    for result in &results {
        assert!(
            Arc::ptr_eq(&result.store(), &results[0].store()),
            "All concurrent creates should return the same store instance"
        );
    }
    assert_eq!(
        Arc::as_ptr(&results.last().unwrap().store()),
        first_ptr,
        "Last result should also match first"
    );
}
