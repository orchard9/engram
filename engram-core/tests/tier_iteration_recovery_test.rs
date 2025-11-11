//! Test tier iteration behavior during WAL recovery

#[cfg(feature = "memory_mapped_persistence")]
mod recovery_tests {
    use chrono::Utc;
    use engram_core::{Confidence, Episode, EpisodeBuilder, MemoryStore};
    use tempfile::TempDir;

    fn create_test_embedding(seed: f32) -> [f32; 768] {
        let mut embedding = [0.0f32; 768];
        for (i, val) in embedding.iter_mut().enumerate() {
            *val = (i as f32 * seed).sin();
        }
        embedding
    }

    #[test]
    fn test_tier_counts_after_recovery() {
        let temp_dir = TempDir::new().unwrap();

        // Create store with persistence
        let store = MemoryStore::new(10)
            .with_persistence(temp_dir.path())
            .expect("Failed to create store with persistence");

        // Store episodes
        for i in 0..5 {
            let ep = EpisodeBuilder::new()
                .id(format!("ep{i}"))
                .when(Utc::now())
                .what(format!("episode {i}"))
                .embedding(create_test_embedding(i as f32 * 0.1))
                .confidence(Confidence::HIGH)
                .build();

            store.store(ep);
        }

        // Check counts before recovery
        let counts_before = store.get_tier_counts();
        assert_eq!(counts_before.hot, 5);

        // Simulate recovery
        let recovered = store.recover_from_wal().expect("Recovery failed");
        println!("Recovered {recovered} episodes");

        // After recovery, counts should still be correct
        let counts_after = store.get_tier_counts();
        println!("Before: {}, After: {}", counts_before.hot, counts_after.hot);

        // Iteration should match count
        assert_eq!(
            store.iter_hot_memories().count(),
            counts_after.hot,
            "Iteration count doesn't match tier count after recovery"
        );
    }

    #[test]
    fn test_no_duplicates_after_recovery() {
        let temp_dir = TempDir::new().unwrap();

        // Create store with persistence
        let store = MemoryStore::new(20)
            .with_persistence(temp_dir.path())
            .expect("Failed to create store with persistence");

        // Store episodes
        for i in 0..10 {
            let ep = EpisodeBuilder::new()
                .id(format!("ep{i}"))
                .when(Utc::now())
                .what(format!("episode {i}"))
                .embedding(create_test_embedding(i as f32 * 0.1))
                .confidence(Confidence::HIGH)
                .build();

            store.store(ep);
        }

        // Recover (should skip already present episodes)
        let _recovered = store.recover_from_wal().expect("Recovery failed");

        // Check for duplicates
        let hot_episodes: Vec<(String, Episode)> = store.iter_hot_memories().collect();
        let ids: Vec<String> = hot_episodes.iter().map(|(id, _)| id.clone()).collect();

        let mut unique_ids = ids.clone();
        unique_ids.sort();
        unique_ids.dedup();

        assert_eq!(
            ids.len(),
            unique_ids.len(),
            "Found duplicate IDs after recovery - wal_buffer and hot_memories may be out of sync"
        );
    }
}
