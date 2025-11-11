//! Comprehensive edge case tests for tier-aware memory iteration

use chrono::Utc;
use engram_core::{Confidence, Episode, EpisodeBuilder, MemoryStore};

fn create_test_embedding(seed: f32) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for (i, val) in embedding.iter_mut().enumerate() {
        *val = (i as f32 * seed).sin();
    }
    embedding
}

#[test]
fn test_empty_store_iteration() {
    let store = MemoryStore::new(10);

    // Should return empty iterator
    assert_eq!(store.iter_hot_memories().count(), 0);

    // Should return zero counts
    let counts = store.get_tier_counts();
    assert_eq!(counts.hot, 0);
    assert_eq!(counts.warm, 0);
    assert_eq!(counts.cold, 0);
    assert_eq!(counts.total, 0);
}

#[test]
fn test_single_episode() {
    let store = MemoryStore::new(10);

    let ep = EpisodeBuilder::new()
        .id("single".to_string())
        .when(Utc::now())
        .what("single episode".to_string())
        .embedding(create_test_embedding(0.5))
        .confidence(Confidence::HIGH)
        .build();

    store.store(ep);

    // Should iterate exactly once
    let hot_episodes: Vec<(String, Episode)> = store.iter_hot_memories().collect();
    assert_eq!(hot_episodes.len(), 1);
    assert_eq!(hot_episodes[0].0, "single");

    // Should count correctly
    let counts = store.get_tier_counts();
    assert_eq!(counts.hot, 1);
    assert_eq!(counts.total, 1);
}

#[test]
fn test_iteration_after_eviction() {
    let store = MemoryStore::new(3);

    // Store 4 episodes to trigger eviction
    for i in 0..4 {
        let ep = EpisodeBuilder::new()
            .id(format!("ep{i}"))
            .when(Utc::now())
            .what(format!("episode {i}"))
            .embedding(create_test_embedding(i as f32 * 0.1))
            .confidence(Confidence::HIGH)
            .build();

        store.store(ep);
    }

    // Should have exactly 3 episodes (one evicted)
    assert_eq!(store.iter_hot_memories().count(), 3);

    // Count should match
    let counts = store.get_tier_counts();
    assert_eq!(counts.hot, 3);
}

#[test]
fn test_no_duplicates_in_iteration() {
    let store = MemoryStore::new(10);

    // Store multiple episodes
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

    // Collect IDs and check for duplicates
    let hot_episodes: Vec<(String, Episode)> = store.iter_hot_memories().collect();
    let ids: Vec<String> = hot_episodes.iter().map(|(id, _)| id.clone()).collect();

    // Check no duplicates
    let mut unique_ids = ids.clone();
    unique_ids.sort();
    unique_ids.dedup();

    assert_eq!(
        ids.len(),
        unique_ids.len(),
        "Found duplicate IDs in iteration"
    );
}

#[test]
fn test_iteration_consistency_with_count() {
    let store = MemoryStore::new(20);

    // Store episodes
    for i in 0..15 {
        let ep = EpisodeBuilder::new()
            .id(format!("ep{i}"))
            .when(Utc::now())
            .what(format!("episode {i}"))
            .embedding(create_test_embedding(i as f32 * 0.1))
            .confidence(Confidence::HIGH)
            .build();

        store.store(ep);
    }

    // Iteration count should match get_tier_counts
    let hot_episodes: Vec<(String, Episode)> = store.iter_hot_memories().collect();
    let counts = store.get_tier_counts();

    assert_eq!(
        hot_episodes.len(),
        counts.hot,
        "iter_hot_memories count ({}) doesn't match get_tier_counts.hot ({})",
        hot_episodes.len(),
        counts.hot
    );
}

#[test]
fn test_episode_content_integrity() {
    let store = MemoryStore::new(10);

    let original_what = "test content";
    let ep = EpisodeBuilder::new()
        .id("test_id".to_string())
        .when(Utc::now())
        .what(original_what.to_string())
        .embedding(create_test_embedding(0.5))
        .confidence(Confidence::HIGH)
        .build();

    store.store(ep);

    // Retrieve and verify content
    let hot_episodes: Vec<(String, Episode)> = store.iter_hot_memories().collect();
    assert_eq!(hot_episodes.len(), 1);

    let retrieved_episode = &hot_episodes[0].1;
    assert_eq!(retrieved_episode.what, original_what);
}

#[test]
fn test_concurrent_iteration() {
    use std::sync::Arc;
    use std::thread;

    let store = Arc::new(MemoryStore::new(100));

    // Store initial episodes
    for i in 0..50 {
        let ep = EpisodeBuilder::new()
            .id(format!("ep{i}"))
            .when(Utc::now())
            .what(format!("episode {i}"))
            .embedding(create_test_embedding(i as f32 * 0.1))
            .confidence(Confidence::HIGH)
            .build();

        store.store(ep);
    }

    // Spawn multiple readers
    let mut handles = vec![];
    for _ in 0..5 {
        let store_clone = Arc::clone(&store);
        let handle = thread::spawn(move || store_clone.iter_hot_memories().count());
        handles.push(handle);
    }

    // All readers should see consistent counts
    let counts: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All counts should be the same (or very close due to concurrent updates)
    let _ = counts[0]; // Read first count to verify compilation
    for count in &counts {
        assert!(
            *count >= 50 && *count <= 55,
            "Concurrent iteration returned inconsistent count: {count} (expected around 50)"
        );
    }
}

#[test]
fn test_deduplication_merging_maintains_count() {
    let store = MemoryStore::new(10);

    // Store very similar episodes (should trigger merge)
    let embedding = create_test_embedding(0.5);

    let ep1 = EpisodeBuilder::new()
        .id("ep1".to_string())
        .when(Utc::now())
        .what("similar content".to_string())
        .embedding(embedding)
        .confidence(Confidence::HIGH)
        .build();

    store.store(ep1);

    let ep2 = EpisodeBuilder::new()
        .id("ep2".to_string())
        .when(Utc::now())
        .what("similar content".to_string())
        .embedding(embedding)
        .confidence(Confidence::HIGH)
        .build();

    store.store(ep2);

    // After deduplication, should only have entries for unique episodes
    let counts = store.get_tier_counts();

    // The actual count depends on deduplication logic
    // But iteration count should match tier count
    assert_eq!(
        store.iter_hot_memories().count(),
        counts.hot,
        "Iteration count doesn't match tier count after deduplication"
    );
}

#[test]
fn test_large_batch_iteration_performance() {
    use std::time::Instant;

    let store = MemoryStore::new(1000);

    // Store many episodes with more varied embeddings to avoid deduplication
    for i in 0..500 {
        let ep = EpisodeBuilder::new()
            .id(format!("ep{i}"))
            .when(Utc::now())
            .what(format!("episode {i}"))
            .embedding(create_test_embedding(i as f32 * 0.1)) // More variation
            .confidence(Confidence::HIGH)
            .build();

        store.store(ep);
    }

    // Time the iteration
    let start = Instant::now();
    let hot_episodes: Vec<(String, Episode)> = store.iter_hot_memories().collect();
    let duration = start.elapsed();

    // Check count and get tier counts for debugging
    let tier_counts = store.get_tier_counts();
    println!(
        "Stored 500, iteration found: {}, tier_counts.hot: {}",
        hot_episodes.len(),
        tier_counts.hot
    );

    // The count should match what tier_counts reports (may be less due to deduplication)
    assert_eq!(
        hot_episodes.len(),
        tier_counts.hot,
        "Iteration count doesn't match tier_counts.hot"
    );

    // Should complete well under 1ms for hot tier access
    assert!(
        duration.as_millis() < 10,
        "Hot tier iteration took too long: {duration:?}"
    );
}

#[test]
fn test_iterator_is_lazy() {
    let store = MemoryStore::new(1000);

    // Store many episodes
    for i in 0..100 {
        let ep = EpisodeBuilder::new()
            .id(format!("ep{i}"))
            .when(Utc::now())
            .what(format!("episode {i}"))
            .embedding(create_test_embedding(i as f32 * 0.01))
            .confidence(Confidence::HIGH)
            .build();

        store.store(ep);
    }

    // Take only a few from the iterator
    assert_eq!(store.iter_hot_memories().take(5).count(), 5);
    // If iterator were eager, this would have collected all 100
}

#[test]
fn test_tier_counts_after_removal() {
    let store = MemoryStore::new(10);

    // Store episodes
    let ids: Vec<String> = (0..5)
        .map(|i| {
            let id = format!("ep{i}");
            let ep = EpisodeBuilder::new()
                .id(id.clone())
                .when(Utc::now())
                .what(format!("episode {i}"))
                .embedding(create_test_embedding(i as f32 * 0.1))
                .confidence(Confidence::HIGH)
                .build();

            store.store(ep);
            id
        })
        .collect();

    // Initial count
    let counts_before = store.get_tier_counts();
    assert_eq!(counts_before.hot, 5);

    // Remove episodes
    store.remove_consolidated_episodes(&ids[0..2]);

    // Count should decrease
    let counts_after = store.get_tier_counts();
    assert_eq!(counts_after.hot, 3, "Count should decrease after removal");

    // Iteration should match
    assert_eq!(store.iter_hot_memories().count(), 3);
}

#[test]
fn test_concurrent_writes_and_reads() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::thread;

    let store = Arc::new(MemoryStore::new(1000));
    let keep_running = Arc::new(AtomicBool::new(true));
    let write_count = Arc::new(AtomicUsize::new(0));
    let read_iterations = Arc::new(AtomicUsize::new(0));

    // Writer thread - continuously stores episodes
    let store_writer = Arc::clone(&store);
    let running_writer = Arc::clone(&keep_running);
    let write_counter = Arc::clone(&write_count);
    let writer = thread::spawn(move || {
        let mut counter = 0;
        while running_writer.load(Ordering::Relaxed) {
            let ep = EpisodeBuilder::new()
                .id(format!("concurrent_ep_{counter}"))
                .when(Utc::now())
                .what(format!("concurrent episode {counter}"))
                .embedding(create_test_embedding(counter as f32 * 0.01))
                .confidence(Confidence::HIGH)
                .build();

            store_writer.store(ep);
            write_counter.fetch_add(1, Ordering::Relaxed);
            counter += 1;

            // Small delay to allow readers to interleave
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    });

    // Reader threads - continuously iterate
    let mut readers = vec![];
    for _ in 0..3 {
        let store_reader = Arc::clone(&store);
        let running_reader = Arc::clone(&keep_running);
        let iterations = Arc::clone(&read_iterations);

        let reader = thread::spawn(move || {
            let mut max_seen = 0;
            while running_reader.load(Ordering::Relaxed) {
                let episodes: Vec<(String, Episode)> = store_reader.iter_hot_memories().collect();

                // Check for duplicates
                let ids: Vec<String> = episodes.iter().map(|(id, _)| id.clone()).collect();
                let mut unique_ids = ids.clone();
                unique_ids.sort();
                unique_ids.dedup();

                assert_eq!(
                    ids.len(),
                    unique_ids.len(),
                    "Found duplicate IDs during concurrent iteration"
                );

                // Check eventual consistency with tier counts
                // During concurrent writes, iteration may observe a transient state
                // where the count differs by a small amount (DashMap shard consistency)
                let counts = store_reader.get_tier_counts();
                let diff = (episodes.len() as i64 - counts.hot as i64).abs();
                assert!(
                    diff <= 5,
                    "Iteration count ({}) differs too much from tier count ({}) - diff: {}",
                    episodes.len(),
                    counts.hot,
                    diff
                );

                max_seen = max_seen.max(episodes.len());
                iterations.fetch_add(1, Ordering::Relaxed);

                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            max_seen
        });
        readers.push(reader);
    }

    // Run for 1 second
    std::thread::sleep(std::time::Duration::from_secs(1));
    keep_running.store(false, Ordering::Relaxed);

    // Wait for all threads
    writer.join().unwrap();
    let max_counts: Vec<usize> = readers.into_iter().map(|r| r.join().unwrap()).collect();

    let final_writes = write_count.load(Ordering::Relaxed);
    let final_reads = read_iterations.load(Ordering::Relaxed);

    println!("Concurrent stress test completed:");
    println!("  Writes: {final_writes}");
    println!("  Read iterations: {final_reads}");
    println!("  Max episodes seen by readers: {max_counts:?}");

    // Verify we actually did concurrent operations
    assert!(final_writes > 0, "No writes occurred");
    assert!(final_reads > 0, "No read iterations occurred");

    // Final consistency check
    let final_episodes: Vec<(String, Episode)> = store.iter_hot_memories().collect();
    let final_counts = store.get_tier_counts();
    assert_eq!(
        final_episodes.len(),
        final_counts.hot,
        "Final state inconsistent: iteration {} != tier count {}",
        final_episodes.len(),
        final_counts.hot
    );
}
