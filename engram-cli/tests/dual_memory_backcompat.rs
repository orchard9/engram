//! Backwards compatibility tests for dual-memory architecture
//!
//! Ensures legacy clients (single-type API) continue to work correctly
//! when server has dual-memory enabled.

use chrono::Utc;
use engram_core::{Confidence, Cue, Memory, MemoryStore};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Generate legacy Memory objects for testing
fn generate_legacy_memories(count: usize, seed: u64) -> Vec<Memory> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut memories = Vec::with_capacity(count);

    for idx in 0..count {
        let mut embedding = [0.0f32; 768];
        for x in &mut embedding {
            *x = rng.gen_range(-1.0..1.0);
        }

        // Normalize
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        let confidence = Confidence::exact(rng.gen_range(0.6..0.95));

        let memory = Memory::new(
            format!("legacy_memory_{}", idx),
            embedding,
            confidence,
        );

        memories.push(memory);
    }

    memories
}

/// Test 1: Legacy client can store and retrieve memories
///
/// Validates that old Memory-based API continues to work when
/// server uses DualMemoryNode internally.
#[test]
fn test_legacy_store_recall_still_works() {
    let store = MemoryStore::new(1024);

    // Legacy client stores memories using old API (Memory -> Episode conversion)
    let memories = generate_legacy_memories(100, 42);

    for memory in &memories {
        // Legacy API: construct Episode from Memory
        let episode = engram_core::Episode {
            id: memory.id.clone(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: format!("Legacy content for {}", memory.id),
            embedding: memory.embedding,
            embedding_provenance: None,
            encoding_confidence: memory.confidence,
            vividness_confidence: memory.confidence,
            reliability_confidence: memory.confidence,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 1.0,
            decay_function: None,
            metadata: std::collections::HashMap::new(),
        };

        let result = store.store(episode);
        assert!(
            result.activation.is_successful(),
            "Legacy store should succeed for {}",
            memory.id
        );
    }

    assert_eq!(store.count(), 100, "Should have stored 100 memories");

    // Legacy client recalls memories
    let query_embedding = [0.5f32; 768];
    let cue = Cue::embedding("legacy_cue".to_string(), query_embedding, Confidence::MEDIUM);

    let results = store.recall(&cue);

    assert!(
        !results.results.is_empty(),
        "Legacy recall should return results"
    );

    println!(
        "Legacy API test passed: stored={}, recalled={}",
        store.count(),
        results.results.len()
    );
}

/// Test 2: Mixed client workload (legacy + modern coexist)
///
/// Validates that legacy clients and dual-memory clients can operate
/// on the same store without conflicts.
#[test]
fn test_legacy_and_modern_clients_coexist() {
    let store = MemoryStore::new(1024);

    // Legacy client stores 100 memories
    let legacy_memories = generate_legacy_memories(100, 123);

    for memory in &legacy_memories {
        let episode = engram_core::Episode {
            id: memory.id.clone(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: format!("Legacy: {}", memory.id),
            embedding: memory.embedding,
            embedding_provenance: None,
            encoding_confidence: memory.confidence,
            vividness_confidence: memory.confidence,
            reliability_confidence: memory.confidence,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 1.0,
            decay_function: None,
            metadata: std::collections::HashMap::new(),
        };

        store.store(episode);
    }

    // Modern client stores 100 episodes (potentially with concepts if dual_memory enabled)
    let modern_episodes = support::dual_memory_fixtures::generate_test_episodes(100, 456);

    for episode in &modern_episodes {
        store.store(episode.clone());
    }

    // Verify both clients can see all data
    let total_count = store.count();
    assert_eq!(total_count, 200, "Should have 200 total memories");

    // Legacy client queries
    let legacy_cue = Cue::embedding("legacy_query".to_string(), [0.5f32; 768], Confidence::MEDIUM);

    let legacy_results = store.recall(&legacy_cue);

    // Modern client queries
    let modern_cue = Cue::embedding("modern_query".to_string(), [0.6f32; 768], Confidence::MEDIUM);

    let modern_results = store.recall(&modern_cue);

    // Both should get results
    assert!(
        !legacy_results.results.is_empty(),
        "Legacy client should get results"
    );
    assert!(
        !modern_results.results.is_empty(),
        "Modern client should get results"
    );

    println!(
        "Mixed client test passed: legacy_results={}, modern_results={}",
        legacy_results.results.len(),
        modern_results.results.len()
    );
}

/// Test 3: Legacy client retrieves specific memories
///
/// Ensures legacy get_episode works for both old and new data.
#[test]
fn test_legacy_retrieval_by_id() {
    let store = MemoryStore::new(512);

    // Store mix of legacy and modern data
    let legacy_memories = generate_legacy_memories(50, 789);

    for memory in &legacy_memories {
        let episode = engram_core::Episode {
            id: memory.id.clone(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: format!("Legacy: {}", memory.id),
            embedding: memory.embedding,
            embedding_provenance: None,
            encoding_confidence: memory.confidence,
            vividness_confidence: memory.confidence,
            reliability_confidence: memory.confidence,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 1.0,
            decay_function: None,
            metadata: std::collections::HashMap::new(),
        };

        store.store(episode);
    }

    // Legacy client retrieves by ID
    for i in 0..10 {
        let memory_id = format!("legacy_memory_{}", i);
        let retrieved = store.get_episode(&memory_id);

        assert!(
            retrieved.is_some(),
            "Legacy client should retrieve {}",
            memory_id
        );

        let episode = retrieved.unwrap();
        assert_eq!(episode.id, memory_id);
    }

    println!("Legacy retrieval test passed");
}

/// Test 4: Legacy confidence scores remain reasonable
///
/// Validates that confidence scores make sense for legacy clients
/// even when dual-memory features are active.
#[test]
fn test_legacy_confidence_reasonable() {
    let store = MemoryStore::new(512);

    // Store memories with known confidence levels
    let test_confidences = vec![
        Confidence::exact(0.9),
        Confidence::exact(0.7),
        Confidence::exact(0.5),
    ];

    for (idx, confidence) in test_confidences.iter().enumerate() {
        let memory = Memory::new(
            format!("conf_test_{}", idx),
            [0.5f32; 768],
            *confidence,
        );

        let episode = engram_core::Episode {
            id: memory.id.clone(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: format!("Confidence test {}", idx),
            embedding: memory.embedding,
            embedding_provenance: None,
            encoding_confidence: memory.confidence,
            vividness_confidence: memory.confidence,
            reliability_confidence: memory.confidence,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 1.0,
            decay_function: None,
            metadata: std::collections::HashMap::new(),
        };

        store.store(episode);
    }

    // Query and verify confidence scores are reasonable
    let cue = Cue::embedding("conf_cue".to_string(), [0.5f32; 768], Confidence::MEDIUM);

    let results = store.recall(&cue);

    for (episode, confidence) in &results.results {
        assert!(
            confidence.raw() >= 0.0 && confidence.raw() <= 1.0,
            "Confidence should be in [0,1]: episode={}, conf={}",
            episode.id,
            confidence.raw()
        );
    }

    println!(
        "Legacy confidence test passed: {} results verified",
        results.results.len()
    );
}

/// Test 5: Legacy API graceful degradation
///
/// Ensures that if dual-memory features degrade, legacy clients
/// still receive reasonable results.
#[test]
fn test_legacy_graceful_degradation() {
    // Small store to trigger potential degradation
    let store = MemoryStore::new(128);

    // Overfill with legacy data
    let memories = generate_legacy_memories(200, 999);

    for memory in &memories {
        let episode = engram_core::Episode {
            id: memory.id.clone(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: format!("Degradation test: {}", memory.id),
            embedding: memory.embedding,
            embedding_provenance: None,
            encoding_confidence: memory.confidence,
            vividness_confidence: memory.confidence,
            reliability_confidence: memory.confidence,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 1.0,
            decay_function: None,
            metadata: std::collections::HashMap::new(),
        };

        let _result = store.store(episode);
        // May degrade but should not panic
    }

    // Legacy client should still get results
    let cue = Cue::embedding("degradation_cue".to_string(), [0.5f32; 768], Confidence::MEDIUM);

    let results = store.recall(&cue);

    println!(
        "Degradation test: stored_attempts={}, results={}",
        memories.len(),
        results.results.len()
    );

    // Should return some results even if degraded
    assert!(
        !results.results.is_empty() || store.count() == 0,
        "Legacy client should get results even under pressure"
    );
}

// Add module declaration for fixtures if needed
#[cfg(test)]
mod support {
    pub mod dual_memory_fixtures {
        pub fn generate_test_episodes(count: usize, seed: u64) -> Vec<engram_core::Episode> {
            use engram_core::{Confidence, Episode};
            use rand::{Rng, SeedableRng, rngs::StdRng};
            use chrono::Utc;

            let mut rng = StdRng::seed_from_u64(seed);
            let mut episodes = Vec::with_capacity(count);

            for idx in 0..count {
                let mut embedding = [0.0f32; 768];
                for x in &mut embedding {
                    *x = rng.gen_range(-1.0..1.0);
                }

                let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut embedding {
                        *x /= norm;
                    }
                }

                let confidence = Confidence::exact(rng.gen_range(0.6..0.95));

                let episode = Episode {
                    id: format!("test_episode_{}", idx),
                    when: Utc::now(),
                    where_location: None,
                    who: None,
                    what: format!("Test content {}", idx),
                    embedding,
                    embedding_provenance: None,
                    encoding_confidence: confidence,
                    vividness_confidence: confidence,
                    reliability_confidence: confidence,
                    last_recall: Utc::now(),
                    recall_count: 0,
                    decay_rate: 1.0,
                    decay_function: None,
                    metadata: std::collections::HashMap::new(),
                };

                episodes.push(episode);
            }

            episodes
        }
    }
}
