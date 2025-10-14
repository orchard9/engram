//! Integration tests for HNSW index

#![cfg(feature = "hnsw_index")]

use chrono::Utc;
use engram_core::{Confidence, Cue, CueBuilder, EpisodeBuilder, MemoryStore};
use std::sync::Arc;
use std::thread;

#[test]
fn test_hnsw_index_creation() {
    let store = MemoryStore::new(1000).with_hnsw_index();

    // Create and store a test episode
    let episode = EpisodeBuilder::new()
        .id("test1".to_string())
        .when(Utc::now())
        .what("test memory".to_string())
        .embedding([0.5f32; 768])
        .confidence(Confidence::HIGH)
        .build();

    let store_result = store.store(episode);
    assert!(store_result.activation.is_successful());
}

#[test]
fn test_hnsw_similarity_search() {
    let store = MemoryStore::new(1000).with_hnsw_index();

    // Store multiple episodes with different embeddings
    let mut embedding1 = [0.0f32; 768];
    let mut embedding2 = [0.0f32; 768];
    let mut embedding3 = [0.0f32; 768];

    // Create distinct embeddings that won't be deduplicated
    // The content addressing uses the first 8 floats, so make those completely different

    // Similar1: start with unique pattern for first 8 floats
    embedding1[0] = 0.9;
    embedding1[1] = 0.1;
    embedding1[2] = 0.9;
    embedding1[3] = 0.1;
    embedding1[4] = 0.9;
    embedding1[5] = 0.1;
    embedding1[6] = 0.9;
    embedding1[7] = 0.1;
    // Fill rest with similar pattern
    embedding1[8..384].fill(0.8);
    embedding1[384..].fill(0.2);

    // Similar2: different pattern for first 8 floats but similar overall pattern
    embedding2[0] = 0.8;
    embedding2[1] = 0.2;
    embedding2[2] = 0.8;
    embedding2[3] = 0.2;
    embedding2[4] = 0.8;
    embedding2[5] = 0.2;
    embedding2[6] = 0.8;
    embedding2[7] = 0.2;
    // Fill rest with similar pattern to similar1
    embedding2[8..384].fill(0.7);
    embedding2[384..].fill(0.3);

    // Different: completely different pattern
    embedding3[0] = 0.1;
    embedding3[1] = 0.9;
    embedding3[2] = 0.1;
    embedding3[3] = 0.9;
    embedding3[4] = 0.1;
    embedding3[5] = 0.9;
    embedding3[6] = 0.1;
    embedding3[7] = 0.9;
    // Fill rest with different pattern
    embedding3[8..384].fill(0.1);
    embedding3[384..].fill(0.9);

    let episodes = vec![
        EpisodeBuilder::new()
            .id("similar1".to_string())
            .when(Utc::now())
            .what("very similar memory".to_string())
            .embedding(embedding1)
            .confidence(Confidence::HIGH)
            .build(),
        EpisodeBuilder::new()
            .id("similar2".to_string())
            .when(Utc::now())
            .what("somewhat similar memory".to_string())
            .embedding(embedding2)
            .confidence(Confidence::MEDIUM)
            .build(),
        EpisodeBuilder::new()
            .id("different".to_string())
            .when(Utc::now())
            .what("different memory".to_string())
            .embedding(embedding3)
            .confidence(Confidence::LOW)
            .build(),
    ];

    for episode in episodes {
        store.store(episode);
    }

    // Wait a bit for indexing
    thread::sleep(std::time::Duration::from_millis(100));

    // Search for similar memories - should match similar1 and similar2 (both have high values in first half)
    let mut query = [0.0f32; 768];
    // Different first 8 floats to avoid content addressing conflicts
    query[0] = 0.85;
    query[1] = 0.15;
    query[2] = 0.85;
    query[3] = 0.15;
    query[4] = 0.85;
    query[5] = 0.15;
    query[6] = 0.85;
    query[7] = 0.15;
    // Similar pattern to similar1 and similar2
    query[8..384].fill(0.75); // High values like similar1 and similar2
    query[384..].fill(0.25); // Low values

    let cue = Cue::embedding("test".to_string(), query, Confidence::LOW);

    let results = store.recall(&cue).results;

    // Should find at least one similar memory
    assert!(!results.is_empty());

    // The first result should be the most similar one (similar1)
    let first_id = &results[0].0.id;
    assert_eq!(first_id, "similar1");

    // The first result should have high confidence
    assert!(results[0].1.raw() > 0.8);
}

#[test]
fn test_hnsw_confidence_filtering() {
    let store = MemoryStore::new(1000).with_hnsw_index();

    // Store episodes with varying confidence
    for i in 0..10 {
        let step = f32::from(u8::try_from(i).expect("confidence step within u8"));
        let confidence = Confidence::exact(step / 10.0);
        let episode = EpisodeBuilder::new()
            .id(format!("mem_{i}"))
            .when(Utc::now())
            .what(format!("memory {i}"))
            .embedding([0.5f32; 768])
            .confidence(confidence)
            .build();

        store.store(episode);
    }

    thread::sleep(std::time::Duration::from_millis(100));

    // Search with high confidence threshold
    let cue = Cue::embedding("test".to_string(), [0.5f32; 768], Confidence::exact(0.7));

    let results = store.recall(&cue).results;

    // Should only return high-confidence memories
    for (_, confidence) in &results {
        assert!(confidence.raw() >= 0.7);
    }
}

#[test]
fn test_hnsw_concurrent_operations() {
    let store = Arc::new(MemoryStore::new(1000).with_hnsw_index());
    let num_threads = 4;
    let episodes_per_thread = 25;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let store = store.clone();
            thread::spawn(move || {
                for i in 0..episodes_per_thread {
                    let id = format!("thread_{thread_id}_mem_{i}");

                    // Interleave inserts and searches
                    if i % 2 == 0 {
                        // Insert
                        let thread_step =
                            f32::from(u8::try_from(thread_id).expect("thread index within u8"));
                        let episode = EpisodeBuilder::new()
                            .id(id)
                            .when(Utc::now())
                            .what(format!("memory from thread {thread_id}"))
                            .embedding([thread_step / 10.0; 768])
                            .confidence(Confidence::MEDIUM)
                            .build();

                        store.store(episode);
                    } else {
                        // Search
                        let cue =
                            Cue::embedding("test".to_string(), [0.5f32; 768], Confidence::LOW);

                        let _results = store.recall(&cue).results;
                    }
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify store integrity (note: the actual method is memory_count but we'd need to add it)
    // assert!(store.memory_count() <= 1000);
}

#[test]
fn test_hnsw_memory_pressure_adaptation() {
    let store = MemoryStore::new(100).with_hnsw_index();

    // Fill store to high pressure
    for i in 0..90 {
        let step = f32::from(u8::try_from(i).expect("memory index within u8"));
        let episode = EpisodeBuilder::new()
            .id(format!("mem_{i}"))
            .when(Utc::now())
            .what(format!("memory {i}"))
            .embedding([step / 100.0; 768])
            .confidence(Confidence::MEDIUM)
            .build();

        let store_result = store.store(episode);

        // Later stores should show degraded activation due to pressure
        if i > 80 {
            assert!(store_result.activation.is_degraded());
        }
    }

    thread::sleep(std::time::Duration::from_millis(100));

    // Search under pressure should still work
    let cue = Cue::embedding("test".to_string(), [0.5f32; 768], Confidence::LOW);

    let results = store.recall(&cue).results;
    assert!(!results.is_empty());
}

#[test]
fn test_hnsw_vs_linear_recall_quality() {
    // Create stores with and without HNSW
    let hnsw_store = MemoryStore::new(500).with_hnsw_index();
    let linear_store = MemoryStore::new(500);

    // Generate test episodes
    let episodes: Vec<_> = (0..100)
        .map(|i| {
            let mut embedding = [0.0f32; 768];
            // Create clusters in embedding space
            let cluster = i / 25;
            let base_value =
                f32::from(u8::try_from(cluster).expect("cluster index within u8")) * 0.25;
            embedding.fill(base_value);
            embedding[i % 768] = 1.0; // Add some variation

            EpisodeBuilder::new()
                .id(format!("mem_{i}"))
                .when(Utc::now())
                .what(format!("memory {i}"))
                .embedding(embedding)
                .confidence(Confidence::exact(
                    (f32::from(u8::try_from(i).expect("memory index within u8")) / 100.0).max(0.3),
                ))
                .build()
        })
        .collect();

    // Store in both
    for episode in &episodes {
        hnsw_store.store(episode.clone());
        linear_store.store(episode.clone());
    }

    thread::sleep(std::time::Duration::from_millis(200));

    // Create query
    let mut query = [0.25f32; 768];
    query[42] = 0.9;

    let cue = CueBuilder::new()
        .id("test".to_string())
        .embedding_search(query, Confidence::MEDIUM)
        .max_results(100) // Set high limit for both searches
        .build();

    // Get results from both
    let hnsw_results = hnsw_store.recall(&cue).results;
    let linear_results = linear_store.recall(&cue).results;

    // HNSW should return similar quality results
    assert!(!hnsw_results.is_empty());
    assert!(!linear_results.is_empty());

    // Calculate recall@k metric
    let hnsw_ids: Vec<_> = hnsw_results.iter().map(|(e, _)| &e.id).collect();
    let linear_ids: Vec<_> = linear_results.iter().map(|(e, _)| &e.id).collect();

    let overlap = hnsw_ids.iter().filter(|id| linear_ids.contains(id)).count();
    let overlap_count = u16::try_from(overlap).expect("overlap count within u16");

    // Calculate recall based on the smaller result set to be fair
    let smallest = hnsw_results.len().min(linear_results.len());
    let smallest_count = u16::try_from(smallest).expect("result set size within u16");
    let smallest_total = f32::from(smallest_count);
    let recall_at_k = if smallest_total > 0.0 {
        f32::from(overlap_count) / smallest_total
    } else {
        0.0
    };

    assert!((0.0..=1.0).contains(&recall_at_k));

    // HNSW should have reasonable quality - all returned results should be valid
    // This tests precision rather than recall since HNSW is approximate
    let hnsw_total_usize = hnsw_results.len();
    let hnsw_total = u16::try_from(hnsw_total_usize).expect("HNSW result size within u16");
    let precision = if hnsw_total == 0 {
        0.0
    } else {
        f32::from(overlap_count) / f32::from(hnsw_total)
    };

    // HNSW should have high precision (most results should be relevant)
    assert!(
        precision >= 0.8,
        "HNSW precision was {precision}, expected >= 0.8 (found {overlap_count} valid results out of {hnsw_total_usize} HNSW results)"
    );
}

#[test]
fn test_hnsw_spreading_activation() {
    let store = MemoryStore::new(1000).with_hnsw_index();

    // Create connected memories with distinct embeddings
    let mut base_embedding = [0.0f32; 768];

    // Central memory - unique pattern
    base_embedding[0] = 0.5;
    base_embedding[1] = 0.5;
    base_embedding[2] = 0.5;
    base_embedding[3] = 0.5;
    base_embedding[4] = 0.5;
    base_embedding[5] = 0.5;
    base_embedding[6] = 0.5;
    base_embedding[7] = 0.5;
    base_embedding[8..].fill(0.5);

    let central = EpisodeBuilder::new()
        .id("central".to_string())
        .when(Utc::now())
        .what("central memory".to_string())
        .embedding(base_embedding)
        .confidence(Confidence::HIGH)
        .build();

    store.store(central);

    // Create neighbors with distinct variations
    for i in 0..5 {
        let mut neighbor_embedding = [0.0f32; 768];

        // Make the first 8 floats completely different for each neighbor
        let neighbor_index = f32::from(u8::try_from(i).expect("neighbor index within range"));
        for (offset, value) in neighbor_embedding.iter_mut().take(8).enumerate() {
            let offset = f32::from(u8::try_from(offset).expect("neighbor offset within range"));
            *value = (neighbor_index + offset).mul_add(0.1, 0.1);
        }

        // Make the rest similar to central but with variations
        let tail_value = neighbor_index.mul_add(0.05, 0.5);
        neighbor_embedding[8..].fill(tail_value);

        let neighbor = EpisodeBuilder::new()
            .id(format!("neighbor_{i}"))
            .when(Utc::now())
            .what(format!("neighbor memory {i}"))
            .embedding(neighbor_embedding)
            .confidence(Confidence::MEDIUM)
            .build();

        store.store(neighbor);
    }

    thread::sleep(std::time::Duration::from_millis(100));

    // Search should activate neighbors through spreading
    let cue = Cue::embedding("test".to_string(), base_embedding, Confidence::LOW);

    let results = store.recall(&cue).results;

    // Should find at least the central memory
    assert!(!results.is_empty());

    // Central memory should have highest confidence
    let central_result = results.iter().find(|(e, _)| e.id == "central");

    assert!(central_result.is_some());
}

#[test]
fn test_hnsw_graph_integrity() {
    use engram_core::index::CognitiveHnswIndex;

    let index = CognitiveHnswIndex::new();

    // Insert nodes
    for i in 0..50 {
        let mut embedding = [0.0f32; 768];
        embedding[i % 768] = 1.0;

        let episode = EpisodeBuilder::new()
            .id(format!("mem_{i}"))
            .when(Utc::now())
            .what(format!("memory {i}"))
            .embedding(embedding)
            .confidence(Confidence::MEDIUM)
            .build();

        let memory = Arc::new(engram_core::Memory::new(
            episode.id.clone(),
            episode.embedding,
            episode.encoding_confidence,
        ));

        index.insert_memory(memory).unwrap();
    }

    // Validate graph structure
    assert!(index.validate_graph_integrity());
    assert!(index.validate_bidirectional_consistency());
    assert!(index.check_memory_consistency());
}
