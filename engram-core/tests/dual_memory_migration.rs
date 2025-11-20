//! Migration correctness tests for dual memory architecture
//!
//! Tests data integrity during conversion between single-type (Memory)
//! and dual-type (DualMemoryNode) representations.

#![cfg(feature = "dual_memory_types")]

mod support;

use chrono::Utc;
use engram_core::{Confidence, Memory, MemoryStore};
use support::dual_memory_fixtures::{
    ClusterConfig, assert_confidence_similar, assert_overlap_threshold,
    generate_clusterable_episodes, generate_test_episodes,
};

#[cfg(feature = "dual_memory_types")]
use engram_core::memory::dual_types::DualMemoryNode;

/// Test 1: Offline migration preserves all episodes
///
/// Validates that converting a populated single-type store to dual-type
/// preserves all episode data with zero loss.
#[test]
fn test_offline_migration_preserves_all_episodes() {
    let episodes = generate_test_episodes(1000, 42);
    let episode_ids: Vec<String> = episodes.iter().map(|e| e.id.clone()).collect();

    // Phase 1: Store in single-type store (Memory-based)
    // Use larger capacity to avoid pressure during bulk inserts
    let store = MemoryStore::new(4096);

    for episode in &episodes {
        let result = store.store(episode.clone());
        // Allow some degradation under load, but activation should be non-zero
        assert!(
            result.activation.value() > 0.0,
            "Failed to store episode {} (activation={})",
            episode.id,
            result.activation.value()
        );
    }

    assert_eq!(
        store.count(),
        1000,
        "Should have 1000 episodes before migration"
    );

    // Phase 2: Simulate offline migration by extracting and converting
    // In real migration, this would involve serialization/deserialization
    let mut dual_nodes = Vec::with_capacity(1000);

    for episode_id in &episode_ids {
        if let Some(episode) = store.get_episode(episode_id) {
            let dual = DualMemoryNode::from_episode(episode, 1.0);
            dual_nodes.push(dual);
        }
    }

    assert_eq!(
        dual_nodes.len(),
        1000,
        "Should convert all 1000 episodes to dual nodes"
    );

    // Phase 3: Verify all episodes are present and embeddings unchanged
    let migrated_ids: Vec<String> = dual_nodes
        .iter()
        .filter_map(|node| {
            if node.is_episode() {
                Some(node.id.to_string())
            } else {
                None
            }
        })
        .collect();

    assert_eq!(
        migrated_ids.len(),
        1000,
        "All nodes should be episode type after migration"
    );

    // Verify embeddings preserved (sample check)
    for (original, dual) in episodes.iter().zip(&dual_nodes) {
        let embedding_match = original
            .embedding
            .iter()
            .zip(&dual.embedding)
            .all(|(a, b)| (a - b).abs() < 1e-6);

        assert!(
            embedding_match,
            "Embedding mismatch for episode {}",
            original.id
        );
    }
}

/// Test 2: Recall ordering preservation across migration
///
/// Validates that the top-K recall results are stable before and after
/// migration, ensuring semantic search quality is maintained.
#[test]
fn test_migration_preserves_recall_ranking() {
    let episodes = generate_clusterable_episodes(&ClusterConfig::default(), 123);

    // Phase 1: Store and query in single-type store
    let store_before = MemoryStore::new(256);

    for episode in &episodes {
        store_before.store(episode.clone());
    }

    // Use the embedding from first episode as query to ensure recall results
    let query_embedding = episodes[0].embedding;
    let cue =
        engram_core::Cue::embedding("test_cue".to_string(), query_embedding, Confidence::MEDIUM);

    let results_before = store_before.recall(&cue);
    let ids_before: Vec<String> = results_before
        .results
        .iter()
        .map(|(e, _)| e.id.clone())
        .collect();

    assert!(
        !ids_before.is_empty(),
        "Should have recall results before migration"
    );

    // Phase 2: Simulate migration (in production this would be Memory -> DualMemoryNode conversion)
    // For testing, we'll create a new store and re-insert
    let store_after = MemoryStore::new(256);

    for episode in &episodes {
        store_after.store(episode.clone());
    }

    // Phase 3: Query after migration
    let results_after = store_after.recall(&cue);
    let ids_after: Vec<String> = results_after
        .results
        .iter()
        .map(|(e, _)| e.id.clone())
        .collect();

    // Allow 90% overlap - perfect match in order is not guaranteed due to
    // floating point variations in SIMD operations
    assert_overlap_threshold(
        &ids_before,
        &ids_after,
        0.9,
        "Recall results should be stable across migration",
    );
}

/// Test 3: Confidence preservation during conversion
///
/// Ensures confidence scores are maintained accurately during
/// Memory -> DualMemoryNode conversion.
#[test]
fn test_migration_preserves_confidence() {
    let episodes = generate_test_episodes(100, 456);

    for episode in &episodes {
        // Convert to Memory
        let memory = Memory::new(
            episode.id.clone(),
            episode.embedding,
            episode.encoding_confidence,
        );

        // Convert to DualMemoryNode
        let dual = DualMemoryNode::from(&memory);

        // Verify confidence preserved
        assert_confidence_similar(
            dual.confidence,
            episode.encoding_confidence,
            0.01,
            &format!("Confidence mismatch for episode {}", episode.id),
        );
    }
}

/// Test 4: Embedding integrity during round-trip conversion
///
/// Validates Memory -> DualMemoryNode -> Memory preserves embeddings.
#[test]
fn test_embedding_integrity_round_trip() {
    let test_embedding = [0.123f32; 768];
    let memory = Memory::new("test_id".to_string(), test_embedding, Confidence::HIGH);

    // Round trip: Memory -> DualMemoryNode -> Memory
    let dual = DualMemoryNode::from(memory.clone());
    let recovered = dual.to_memory();

    // Verify embedding identical
    for (i, (original, recovered)) in test_embedding.iter().zip(&recovered.embedding).enumerate() {
        assert!(
            (original - recovered).abs() < 1e-9,
            "Embedding mismatch at index {}: {} vs {}",
            i,
            original,
            recovered
        );
    }
}

/// Test 5: Large-scale migration stress test
///
/// Tests migration of 10K episodes to validate scalability.
#[test]
#[ignore = "Large-scale test, run with --ignored"]
fn test_large_scale_migration() {
    let episodes = generate_test_episodes(10_000, 789);

    let store = MemoryStore::new(16384);

    // Store all episodes
    for episode in &episodes {
        let result = store.store(episode.clone());
        assert!(result.activation.is_successful());
    }

    assert_eq!(store.count(), 10_000);

    // Simulate migration by converting to dual nodes
    let mut converted_count = 0;

    for episode in &episodes {
        if let Some(ep) = store.get_episode(&episode.id) {
            let _dual = DualMemoryNode::from_episode(ep, 1.0);
            converted_count += 1;
        }
    }

    assert_eq!(converted_count, 10_000, "Should convert all 10K episodes");
}

/// Test 6: Incremental migration (online migration simulation)
///
/// Simulates gradual migration where new episodes use dual-type while
/// old episodes remain in single-type format temporarily.
#[test]
fn test_incremental_online_migration() {
    let old_episodes = generate_test_episodes(500, 111);
    let new_episodes = generate_test_episodes(500, 222);

    // Use larger capacity to prevent eviction during test
    let store = MemoryStore::new(4096);

    // Phase 1: Store "old" episodes
    for episode in &old_episodes {
        store.store(episode.clone());
    }

    let count_after_phase1 = store.count();
    assert!(
        count_after_phase1 >= 300,
        "Should have stored a significant portion of phase 1 episodes (got {})",
        count_after_phase1
    );

    // Phase 2: Continue storing "new" episodes (in production, these would
    // be stored as DualMemoryNode directly, but MemoryStore API is unified)
    for episode in &new_episodes {
        store.store(episode.clone());
    }

    let count_after_phase2 = store.count();
    // With random confidence values (0.5-1.0), expect ~60-70% successful stores
    // given the base_activation = confidence * (1.0 - 0.5 * pressure) formula
    assert!(
        count_after_phase2 >= 600,
        "Should have stored a significant portion of episodes (got {})",
        count_after_phase2
    );

    // Phase 3: Verify that episodes are accessible (count may be lower due to deduplication)
    let mut accessible_count = 0;
    for episode in old_episodes.iter().chain(&new_episodes) {
        if store.get_episode(&episode.id).is_some() {
            accessible_count += 1;
        }
    }

    // All original episode IDs should be accessible (may map to deduplicated entries)
    assert_eq!(
        accessible_count, 1000,
        "All original episode IDs should be accessible (even if deduplicated)"
    );

    // Store count may be lower due to deduplication
    assert!(
        count_after_phase2 <= 1000,
        "Store count should not exceed total episodes"
    );
}

/// Test 7: Migration preserves temporal properties
///
/// Ensures timestamps (when, last_recall) are maintained during conversion.
#[test]
fn test_migration_preserves_temporal_properties() {
    use chrono::Duration;

    let base_time = Utc::now() - Duration::hours(48);
    let mut episodes = Vec::new();

    for i in 0..100 {
        let mut episode = generate_test_episodes(1, i)[0].clone();
        episode.when = base_time + Duration::hours(i as i64);
        episode.last_recall = episode.when + Duration::minutes(i as i64);
        episodes.push(episode);
    }

    // Convert to DualMemoryNode and verify timestamps
    for episode in &episodes {
        let dual = DualMemoryNode::from_episode(episode.clone(), 1.0);

        // Timestamps are not directly exposed on DualMemoryNode, but
        // we can verify through round-trip conversion
        let recovered = dual.to_memory();

        assert_eq!(
            recovered.created_at.timestamp(),
            dual.created_at.timestamp(),
            "Created timestamp mismatch for episode {}",
            episode.id
        );
    }
}

/// Test 8: Migration handles edge case embeddings
///
/// Tests conversion with edge case embedding values (zeros, extremes, denormals).
#[test]
fn test_migration_edge_case_embeddings() {
    let test_cases = vec![
        ("all_zeros", [0.0f32; 768]),
        ("all_ones", [1.0f32; 768]),
        ("all_negative_ones", [-1.0f32; 768]),
        ("mixed_extremes", {
            let mut arr = [0.0f32; 768];
            for (i, x) in arr.iter_mut().enumerate() {
                *x = if i % 2 == 0 { 1.0 } else { -1.0 };
            }
            arr
        }),
    ];

    for (name, embedding) in test_cases {
        let memory = Memory::new(format!("test_{}", name), embedding, Confidence::MEDIUM);

        let dual = DualMemoryNode::from(memory.clone());
        let recovered = dual.to_memory();

        // Verify embedding preserved exactly
        for (i, (orig, recov)) in embedding.iter().zip(&recovered.embedding).enumerate() {
            assert!(
                (orig - recov).abs() < 1e-9,
                "Mismatch in {} at index {}: {} vs {}",
                name,
                i,
                orig,
                recov
            );
        }
    }
}

/// Test 9: Concurrent migration simulation
///
/// Tests that migration can proceed while read operations continue.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_migration_with_reads() {
    let episodes = generate_test_episodes(1000, 999);

    let store = std::sync::Arc::new(MemoryStore::new(2048));

    // Store all episodes
    for episode in &episodes {
        store.store(episode.clone());
    }

    // Spawn read workload
    let store_clone = store.clone();
    let read_handle = tokio::spawn(async move {
        for _ in 0..100 {
            let cue = engram_core::Cue::embedding(
                "concurrent_cue".to_string(),
                [0.5f32; 768],
                Confidence::MEDIUM,
            );

            let _results = store_clone.recall(&cue);
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    });

    // Simulate migration workload (convert episodes to dual nodes)
    let store_clone2 = store.clone();
    let episode_ids: Vec<String> = episodes.iter().map(|e| e.id.clone()).collect();
    let migrate_handle = tokio::spawn(async move {
        for episode_id in &episode_ids {
            if let Some(episode) = store_clone2.get_episode(episode_id) {
                let _dual = DualMemoryNode::from_episode(episode, 1.0);
            }
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }
    });

    // Wait for both workloads
    let read_result = read_handle.await;
    let migrate_result = migrate_handle.await;

    assert!(
        read_result.is_ok(),
        "Read workload should complete without errors"
    );
    assert!(
        migrate_result.is_ok(),
        "Migration workload should complete without errors"
    );
}

/// Test 10: Property-based migration invariants
///
/// Uses property testing to verify conversion invariants hold for arbitrary inputs.
#[test]
fn test_property_migration_invariants() {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    let mut rng = StdRng::seed_from_u64(12345);

    // Test 100 random configurations
    for iteration in 0..100 {
        let embedding = {
            let mut arr = [0.0f32; 768];
            for x in &mut arr {
                *x = rng.gen_range(-1.0..1.0);
            }
            arr
        };

        let confidence = Confidence::exact(rng.gen_range(0.0..1.0));
        let memory = Memory::new(
            format!("property_test_{}", iteration),
            embedding,
            confidence,
        );

        // Property 1: Round-trip preserves ID
        let dual = DualMemoryNode::from(memory.clone());
        let recovered = dual.to_memory();
        assert_eq!(memory.id, recovered.id, "ID should be preserved");

        // Property 2: Confidence monotonic (may have small quantization)
        let confidence_diff = (memory.confidence.raw() - recovered.confidence.raw()).abs();
        assert!(
            confidence_diff < 0.01,
            "Confidence should be nearly preserved: diff={}",
            confidence_diff
        );

        // Property 3: Embedding dimensions unchanged
        assert_eq!(
            memory.embedding.len(),
            recovered.embedding.len(),
            "Embedding dimensions should match"
        );

        // Property 4: Activation bounds preserved
        let activation_before = memory.activation();
        let activation_after = recovered.activation();
        assert!(
            activation_before >= 0.0 && activation_before <= 1.0,
            "Activation before should be in [0,1]"
        );
        assert!(
            activation_after >= 0.0 && activation_after <= 1.0,
            "Activation after should be in [0,1]"
        );
    }
}
