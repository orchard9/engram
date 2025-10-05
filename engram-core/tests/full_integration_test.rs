//! Full integration test for Milestone 1 components
//!
//! Tests the complete pipeline:
//! Store → HNSW Index → Activation Spreading → Query Engine → Pattern Completion

use chrono::Utc;
use engram_core::{
    Confidence, Memory, MemoryStore,
    memory::{CueBuilder, EpisodeBuilder},
};
use std::{convert::TryFrom, sync::Arc};

#[cfg(feature = "pattern_completion")]
use engram_core::completion::{
    CompletionConfig, MemorySource, PartialEpisode, PatternCompleter, PatternReconstructor,
};

#[cfg(feature = "probabilistic_queries")]
use engram_core::query::ProbabilisticRecall;

/// Test the full memory lifecycle from storage to recall
#[test]
fn test_full_memory_lifecycle() {
    // Create a memory store
    let store = MemoryStore::new(1000);

    // Create test episodes with embeddings
    let embedding1 = create_test_embedding(0.1);
    let embedding2 = create_test_embedding(0.2);
    let embedding3 = create_test_embedding(0.3);

    let episode1 = EpisodeBuilder::new()
        .id("episode_1".to_string())
        .when(Utc::now())
        .what("First test memory".to_string())
        .embedding(embedding1)
        .confidence(Confidence::HIGH)
        .where_location("Test Lab".to_string())
        .build();

    let episode2 = EpisodeBuilder::new()
        .id("episode_2".to_string())
        .when(Utc::now())
        .what("Second test memory".to_string())
        .embedding(embedding2)
        .confidence(Confidence::MEDIUM)
        .build();

    let episode3 = EpisodeBuilder::new()
        .id("episode_3".to_string())
        .when(Utc::now())
        .what("Third test memory".to_string())
        .embedding(embedding3)
        .confidence(Confidence::LOW)
        .build();

    // Store episodes
    store.store(episode1);
    store.store(episode2);
    store.store(episode3);

    // Test recall with embedding-based cue
    let query_embedding = create_test_embedding(0.15); // Similar to embedding1
    let cue = CueBuilder::new()
        .id("test_cue".to_string())
        .embedding_search(query_embedding, Confidence::LOW)
        .max_results(2)
        .build();

    let results = store.recall(&cue);

    // Verify results
    assert!(!results.is_empty(), "Should have recall results");
    assert!(results.len() <= 2, "Should respect max_results");

    // The first result should be most similar
    if let Some((_first_episode, confidence)) = results.first() {
        assert!(confidence.raw() > 0.5, "Should have reasonable confidence");
        // Could be episode_1 or episode_2 depending on similarity calculation
    }
}

/// Test SIMD vector operations
#[test]
fn test_simd_vector_operations() {
    use engram_core::compute::{cosine_similarity_768, cosine_similarity_batch_768};

    let vec1 = create_test_embedding(1.0);
    let vec2 = create_test_embedding(1.0);
    let vec3 = create_test_embedding(-1.0);

    // Test single similarity
    let similarity_same = cosine_similarity_768(&vec1, &vec2);
    assert!(
        (similarity_same - 1.0).abs() < 0.001,
        "Same vectors should have similarity ~1.0"
    );

    let similarity_opposite = cosine_similarity_768(&vec1, &vec3);
    assert!(
        similarity_opposite < -0.9,
        "Opposite vectors should have negative similarity"
    );

    // Test batch similarity
    let batch = vec![vec2, vec3];
    let batch_results = cosine_similarity_batch_768(&vec1, &batch);
    assert_eq!(batch_results.len(), 2);
    assert!((batch_results[0] - 1.0).abs() < 0.001);
    assert!(batch_results[1] < -0.9);
}

/// Test activation spreading through memory network
#[cfg(feature = "hnsw_index")]
#[test]
fn test_activation_spreading() {
    let store = MemoryStore::new(1000).with_hnsw_index();

    for i in 0..5 {
        let embedding = create_test_embedding(f32_from_usize(i) * 0.05);
        let episode = EpisodeBuilder::new()
            .id(format!("episode_{i}"))
            .when(Utc::now())
            .what(format!("Memory {i}"))
            .embedding(embedding)
            .confidence(Confidence::MEDIUM)
            .build();

        store.store(episode);
    }

    let query_embedding = create_test_embedding(0.05);
    let cue = CueBuilder::new()
        .id("activation_test".to_string())
        .embedding_search(query_embedding, Confidence::LOW)
        .max_results(5)
        .build();

    let results = store.recall(&cue);
    assert!(
        !results.is_empty(),
        "expected recall results when querying inserted embeddings"
    );
    assert!(
        results.iter().any(|(episode, _)| episode.id == "episode_0"),
        "recall results should contain the closest episode"
    );

    let confidences: Vec<f32> = results
        .iter()
        .map(|(_, confidence)| confidence.raw())
        .collect();
    assert!(
        confidences
            .windows(2)
            .all(|window| window[0] + f32::EPSILON >= window[1]),
        "recall confidences must be sorted in descending order"
    );
}

/// Test pattern completion for partial episodes
#[cfg(feature = "pattern_completion")]
#[test]
fn test_pattern_completion() {
    use std::collections::HashMap;

    let mut reconstructor = PatternReconstructor::new(CompletionConfig::default());

    let base_episode = EpisodeBuilder::new()
        .id("episode_base".to_string())
        .when(Utc::now())
        .what("Lunch with team".to_string())
        .embedding(create_test_embedding(0.2))
        .confidence(Confidence::HIGH)
        .where_location("Cafeteria".to_string())
        .build();

    let alt_episode = EpisodeBuilder::new()
        .id("episode_alt".to_string())
        .when(Utc::now())
        .what("Lunch with client".to_string())
        .embedding(create_test_embedding(0.25))
        .confidence(Confidence::MEDIUM)
        .where_location("Cafeteria".to_string())
        .build();

    reconstructor.add_episodes(&[base_episode.clone(), alt_episode.clone()]);
    reconstructor.add_memories(vec![
        Memory::from_episode(base_episode.clone(), 0.9),
        Memory::from_episode(alt_episode.clone(), 0.8),
    ]);

    let partial = PartialEpisode {
        known_fields: HashMap::from([("what".to_string(), "Lunch with team".to_string())]),
        partial_embedding: base_episode
            .embedding
            .iter()
            .map(|value| Some(*value))
            .collect(),
        cue_strength: Confidence::exact(0.7),
        temporal_context: vec![base_episode.id.clone()],
    };

    let completed = reconstructor
        .complete(&partial)
        .expect("pattern completion should succeed");

    assert_eq!(
        completed.episode.where_location.as_deref(),
        Some("Cafeteria")
    );
    let where_source = completed
        .source_attribution
        .field_sources
        .get("where")
        .copied()
        .expect("reconstructed field should include attribution");
    assert_eq!(where_source, MemorySource::Reconstructed);
    assert!(
        completed.completion_confidence.raw() > 0.0,
        "completion confidence should be positive"
    );
}

/// Test probabilistic query engine with confidence propagation
#[cfg(feature = "probabilistic_queries")]
#[test]
fn test_probabilistic_queries() {
    let store = MemoryStore::new(1000);

    for offset in [0.0_f32, 0.02, 0.5] {
        let embedding = create_test_embedding(0.3 + offset);
        let episode = EpisodeBuilder::new()
            .id(format!("prob_ep_{offset}"))
            .when(Utc::now())
            .what(format!("Probabilistic episode {offset}"))
            .embedding(embedding)
            .confidence(Confidence::exact(0.6))
            .build();
        store.store(episode);
    }

    let cue = CueBuilder::new()
        .id("probabilistic_cue".to_string())
        .embedding_search(create_test_embedding(0.31), Confidence::exact(0.3))
        .max_results(3)
        .build();

    let result = store.recall_probabilistic(cue);

    assert!(
        !result.episodes.is_empty(),
        "probabilistic recall should return underlying recall results"
    );
    assert!(
        result.confidence_interval.width >= 0.0 && result.confidence_interval.width <= 1.0,
        "confidence interval width must be within probability bounds"
    );
    assert!(
        result
            .confidence_interval
            .contains(result.confidence_interval.point.raw()),
        "interval should contain the point estimate"
    );
    assert!(
        !result.evidence_chain.is_empty(),
        "probabilistic recall should provide supporting evidence"
    );
}

/// Test batch operations for high throughput
#[test]
fn test_batch_operations() {
    // Simple batch test using individual stores
    let store = Arc::new(MemoryStore::new(1000));

    // Store episodes individually (simulating batch)
    for i in 0..10 {
        let episode = EpisodeBuilder::new()
            .id(format!("batch_episode_{i}"))
            .when(Utc::now())
            .what(format!("Batch memory {i}"))
            .embedding(create_test_embedding(f32_from_usize(i) * 0.01))
            .confidence(Confidence::MEDIUM)
            .build();

        store.store(episode);
    }

    // Verify memories were stored
    let query_embedding = create_test_embedding(0.05);
    let cue = CueBuilder::new()
        .id("batch_test_cue".to_string())
        .embedding_search(query_embedding, Confidence::LOW)
        .max_results(5)
        .build();

    let recall_results = store.recall(&cue);
    assert!(!recall_results.is_empty(), "Should recall stored memories");
}

/// Test memory decay functions
#[test]
fn test_psychological_decay() {
    let mut memory = Memory::new(
        "decay_mem".to_string(),
        create_test_embedding(0.4),
        Confidence::HIGH,
    );
    let initial_confidence = memory.confidence;
    memory.apply_forgetting_decay(24.0);

    assert!(
        memory.confidence.raw() < initial_confidence.raw(),
        "confidence should decay over time"
    );
    assert!(memory.confidence.raw() >= 0.0);
}

/// Helper function to create test embeddings
fn create_test_embedding(value: f32) -> [f32; 768] {
    let mut embedding = [0.0; 768];
    for (index, element) in embedding.iter_mut().enumerate() {
        let index = f32::from(u16::try_from(index).expect("index within u16 range"));
        let scale = index.mul_add(0.001, 1.0);
        *element = value * scale;
    }
    // Normalize
    let norm: f32 = embedding
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for component in &mut embedding {
            *component /= norm;
        }
    }
    embedding
}

fn f32_from_usize(value: usize) -> f32 {
    let narrowed = u16::try_from(value).expect("value fits within u16 range");
    f32::from(narrowed)
}

/// Test that monitoring has minimal overhead
#[cfg(feature = "monitoring")]
#[test]
fn test_monitoring_overhead() {
    // TODO: Implement when monitoring feature is ready
    // This test is currently stubbed out as the LockFreeMetricsRegistry is not yet implemented
    println!("Monitoring overhead test skipped - feature not implemented");
}
