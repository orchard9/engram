//! Full integration test for Milestone 1 components
//!
//! Tests the complete pipeline:
//! Store → HNSW Index → Activation Spreading → Query Engine → Pattern Completion

use engram_core::{
    Confidence, Episode, Memory, MemoryStore, Cue,
    memory::{EpisodeBuilder, CueBuilder, MemoryBuilder},
};
use chrono::Utc;
use std::sync::Arc;

/// Test the full memory lifecycle from storage to recall
#[test]
fn test_full_memory_lifecycle() {
    // Create a memory store
    let store = MemoryStore::new();
    
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
    store.store(episode1.clone()).expect("Failed to store episode 1");
    store.store(episode2.clone()).expect("Failed to store episode 2");
    store.store(episode3.clone()).expect("Failed to store episode 3");
    
    // Test recall with embedding-based cue
    let query_embedding = create_test_embedding(0.15); // Similar to embedding1
    let cue = CueBuilder::new()
        .id("test_cue".to_string())
        .embedding_search(query_embedding, Confidence::LOW)
        .max_results(2)
        .build();
    
    let results = store.recall(&cue).expect("Failed to recall memories");
    
    // Verify results
    assert!(!results.is_empty(), "Should have recall results");
    assert!(results.len() <= 2, "Should respect max_results");
    
    // The first result should be most similar
    if let Some((first_episode, confidence)) = results.first() {
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
    assert!((similarity_same - 1.0).abs() < 0.001, "Same vectors should have similarity ~1.0");
    
    let similarity_opposite = cosine_similarity_768(&vec1, &vec3);
    assert!(similarity_opposite < -0.9, "Opposite vectors should have negative similarity");
    
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
    use engram_core::activation::simple_parallel::SimpleParallelSpreading;
    use engram_core::Activation;
    
    let store = MemoryStore::new();
    
    // Create a network of related memories
    for i in 0..5 {
        let embedding = create_test_embedding(i as f32 * 0.1);
        let episode = EpisodeBuilder::new()
            .id(format!("episode_{}", i))
            .when(Utc::now())
            .what(format!("Memory {}", i))
            .embedding(embedding)
            .confidence(Confidence::MEDIUM)
            .build();
        
        store.store(episode).expect("Failed to store episode");
    }
    
    // Create activation spreading engine
    let spreading = SimpleParallelSpreading::new(4);
    
    // Spread activation from first memory
    let initial_activation = Activation::new("episode_0".to_string(), 1.0);
    let spread_results = spreading.spread_simple(
        vec![initial_activation],
        3, // max hops
        0.5, // min activation
        |_node_id| {
            // Return mock neighbors for testing
            vec![
                ("episode_1".to_string(), 0.9),
                ("episode_2".to_string(), 0.7),
            ]
        }
    );
    
    // Verify activation spread
    assert!(!spread_results.is_empty(), "Should have spread activation");
    
    // Check that activation decreases with distance
    let activations: std::collections::HashMap<_, _> = spread_results
        .into_iter()
        .map(|a| (a.node_id, a.activation))
        .collect();
    
    if let Some(&activation_1) = activations.get("episode_1") {
        assert!(activation_1 > 0.0 && activation_1 < 1.0, "Should have partial activation");
    }
}

/// Test pattern completion for partial episodes
#[cfg(feature = "pattern_completion")]
#[test]
fn test_pattern_completion() {
    use engram_core::completion::{PatternCompletionEngine, PartialEpisode};
    
    let store = MemoryStore::new();
    
    // Store complete episodes
    let complete_episode = EpisodeBuilder::new()
        .id("complete_1".to_string())
        .when(Utc::now())
        .what("Complete memory with all details".to_string())
        .embedding(create_test_embedding(0.5))
        .confidence(Confidence::HIGH)
        .where_location("Office".to_string())
        .who(vec!["Alice".to_string(), "Bob".to_string()])
        .build();
    
    store.store(complete_episode.clone()).expect("Failed to store complete episode");
    
    // Create partial episode missing some fields
    let mut partial = PartialEpisode {
        known_fields: std::collections::HashMap::new(),
        partial_embedding: vec![Some(0.5); 768],
        cue_strength: Confidence::MEDIUM,
        temporal_context: vec![],
    };
    partial.known_fields.insert("what".to_string(), "memory with all details".to_string());
    
    // Complete the pattern
    let engine = PatternCompletionEngine::new(Arc::new(store));
    let completed = engine.complete(&partial).expect("Failed to complete pattern");
    
    // Verify completion
    assert!(!completed.is_empty(), "Should have completion hypotheses");
    
    let best_completion = &completed[0];
    assert!(best_completion.confidence.raw() > 0.5, "Should have reasonable confidence");
    
    // Check if location was reconstructed
    if let Some(location) = best_completion.reconstructed_fields.get("where") {
        // Might match "Office" if pattern completion works well
        assert!(!location.is_empty(), "Should have reconstructed location");
    }
}

/// Test probabilistic query engine with confidence propagation
#[cfg(feature = "probabilistic_queries")]
#[test]
fn test_probabilistic_queries() {
    use engram_core::query::{ProbabilisticQueryResult, ConfidenceInterval};
    
    let store = MemoryStore::new();
    
    // Store episodes with varying confidence
    for i in 0..3 {
        let confidence = match i {
            0 => Confidence::HIGH,
            1 => Confidence::MEDIUM,
            _ => Confidence::LOW,
        };
        
        let episode = EpisodeBuilder::new()
            .id(format!("prob_episode_{}", i))
            .when(Utc::now())
            .what(format!("Probabilistic memory {}", i))
            .embedding(create_test_embedding(i as f32 * 0.3))
            .confidence(confidence)
            .build();
        
        store.store(episode).expect("Failed to store episode");
    }
    
    // Query with confidence tracking
    let query_embedding = create_test_embedding(0.4);
    let cue = CueBuilder::new()
        .id("prob_cue".to_string())
        .embedding_search(query_embedding, Confidence::LOW)
        .result_threshold(Confidence::LOW)
        .build();
    
    let results = store.recall(&cue).expect("Failed to query");
    
    // Verify confidence propagation
    for (episode, confidence) in results {
        // Confidence should be propagated correctly
        assert!(confidence.raw() > 0.0 && confidence.raw() <= 1.0);
        
        // Higher confidence episodes should generally rank higher
        // (though similarity also matters)
    }
}

/// Test batch operations for high throughput
#[test]
fn test_batch_operations() {
    use engram_core::batch::{BatchOperation, BatchEngine};
    use engram_core::batch::operations::StoreOp;
    
    let store = Arc::new(MemoryStore::new());
    let engine = BatchEngine::new(store.clone(), 4);
    
    // Create batch of episodes
    let mut batch_ops = Vec::new();
    for i in 0..100 {
        let episode = EpisodeBuilder::new()
            .id(format!("batch_episode_{}", i))
            .when(Utc::now())
            .what(format!("Batch memory {}", i))
            .embedding(create_test_embedding(i as f32 * 0.01))
            .confidence(Confidence::MEDIUM)
            .build();
        
        batch_ops.push(BatchOperation::Store(StoreOp { episode }));
    }
    
    // Execute batch
    let results = engine.execute_batch(batch_ops).expect("Failed to execute batch");
    
    // Verify all operations succeeded
    assert_eq!(results.len(), 100);
    for result in results {
        assert!(result.is_ok(), "Batch operation should succeed");
    }
    
    // Verify memories were stored
    let query_embedding = create_test_embedding(0.5);
    let cue = CueBuilder::new()
        .id("batch_test_cue".to_string())
        .embedding_search(query_embedding, Confidence::LOW)
        .max_results(10)
        .build();
    
    let recall_results = store.recall(&cue).expect("Failed to recall");
    assert!(!recall_results.is_empty(), "Should recall batch-stored memories");
}

/// Test memory decay functions
#[test]
fn test_psychological_decay() {
    use engram_core::decay::{DecayModel, EbbinghausDecay};
    
    let mut memory = MemoryBuilder::new()
        .id("decay_test".to_string())
        .embedding(create_test_embedding(0.5))
        .confidence(Confidence::HIGH)
        .decay_rate(0.1)
        .build();
    
    let initial_confidence = memory.confidence.raw();
    
    // Apply Ebbinghaus forgetting curve
    let decay_model = EbbinghausDecay::new(0.1);
    let retention = decay_model.calculate_retention(24.0); // 24 hours
    
    memory.apply_forgetting_decay(24.0);
    
    // Confidence should have decayed
    assert!(memory.confidence.raw() < initial_confidence);
    assert!(memory.confidence.raw() > 0.0); // But not to zero
    
    // Should roughly match theoretical retention
    let expected = initial_confidence * retention;
    assert!((memory.confidence.raw() - expected).abs() < 0.1);
}

/// Helper function to create test embeddings
fn create_test_embedding(value: f32) -> [f32; 768] {
    let mut embedding = [0.0; 768];
    for i in 0..768 {
        embedding[i] = value * (1.0 + (i as f32 * 0.001));
    }
    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }
    embedding
}

/// Test that monitoring has minimal overhead
#[cfg(feature = "monitoring")]
#[test]
fn test_monitoring_overhead() {
    use engram_core::metrics::LockFreeMetricsRegistry;
    use std::time::Instant;
    
    let metrics = LockFreeMetricsRegistry::new();
    
    // Measure overhead of metric recording
    let iterations = 100_000;
    
    // Baseline: loop without metrics
    let start = Instant::now();
    for i in 0..iterations {
        // Simulate some work
        std::hint::black_box(i);
    }
    let baseline_duration = start.elapsed();
    
    // With metrics
    let start = Instant::now();
    for i in 0..iterations {
        metrics.record_counter("test_counter", 1);
        std::hint::black_box(i);
    }
    let metrics_duration = start.elapsed();
    
    // Calculate overhead
    let overhead = (metrics_duration.as_nanos() as f64 - baseline_duration.as_nanos() as f64) 
        / baseline_duration.as_nanos() as f64;
    
    // Should be less than 1% overhead
    assert!(overhead < 0.01, "Monitoring overhead should be <1%, got {}%", overhead * 100.0);
}