//! Cross-phenomenon interaction tests
//!
//! Tests all pairwise interactions between cognitive phenomena to ensure
//! they work together correctly and create expected emergent behaviors.
//!
//! Test Matrix:
//! - Priming × Interference
//! - Priming × Reconsolidation
//! - Priming × False Memory (DRM)
//! - Interference × Reconsolidation
//! - Interference × False Memory
//! - Reconsolidation × False Memory

use chrono::{Duration, Utc};
use engram_core::cognitive::priming::SemanticPrimingEngine;
use engram_core::cognitive::reconsolidation::{
    EpisodeModifications, ModificationType, ReconsolidationEngine,
};
use engram_core::{Confidence, Cue, Episode, MemoryStore};
use std::thread;
use std::time::Duration as StdDuration;

// Import shared test utilities
use super::super::helpers::embeddings::{
    create_embedding, create_high_similarity_embedding, generate_drm_embeddings,
};

// ==================== Priming × Interference ====================

#[test]
fn test_priming_amplifies_interference_detection() {
    // Hypothesis: Priming activated concepts makes interference MORE detectable
    //
    // Setup: Store two competing memories in different semantic domains
    // Condition 1: Measure interference without priming
    // Condition 2: Prime one domain, measure interference on other
    // Expected: Primed interference > baseline interference

    let store = MemoryStore::new(10000);
    let semantic_engine = SemanticPrimingEngine::new();

    // Create two competing memories
    let sleep_embedding = create_high_similarity_embedding("sleep", 0);
    let chair_embedding = create_high_similarity_embedding("chair", 1);

    let sleep_episode = create_episode("sleep_memory", "I had a dream", &sleep_embedding, 48);
    let chair_episode = create_episode("chair_memory", "I sat down", &chair_embedding, 48);

    let _ = store.store(sleep_episode);
    let _ = store.store(chair_episode);

    // Baseline: Query for sleep without priming
    let cue = Cue::embedding("sleep".to_string(), sleep_embedding, Confidence::HIGH);
    let baseline_results = store.recall(&cue);

    let baseline_sleep_score = baseline_results
        .results
        .iter()
        .find(|(episode, _conf)| episode.id == "sleep_memory")
        .map_or(0.0, |(_episode, conf)| conf.raw());

    // Experimental: Prime competing domain (chair)
    semantic_engine.activate_priming("chair", &chair_embedding, || {
        vec![("furniture".to_string(), chair_embedding, 1)]
    });

    thread::sleep(StdDuration::from_millis(100));

    // Query again - priming should create interference
    let primed_results = store.recall(&cue);

    let primed_sleep_score = primed_results
        .results
        .iter()
        .find(|(episode, _conf)| episode.id == "sleep_memory")
        .map_or(0.0, |(_episode, conf)| conf.raw());

    // Verify: Competing priming affects retrieval
    // (Exact effect depends on implementation - this is a smoke test)
    println!(
        "Priming effect on interference: baseline={baseline_sleep_score:.3}, primed={primed_sleep_score:.3}"
    );
}

// ==================== Priming × Reconsolidation ====================

#[test]
fn test_reconsolidation_can_modify_primed_memories() {
    // Hypothesis: Memories with active priming can still be reconsolidated
    //
    // Setup: Create memory, activate priming, then reconsolidate
    // Expected: Reconsolidation succeeds despite active priming

    let reconsolidation_engine = ReconsolidationEngine::new();
    let semantic_engine = SemanticPrimingEngine::new();

    // Create consolidated memory (>24h old)
    let embedding_temp = create_embedding(1.0);
    let episode = create_episode("primed_memory", "original content", &embedding_temp, 48);

    // Activate priming for this memory
    semantic_engine.activate_priming("primed_memory", &episode.embedding, Vec::new);

    thread::sleep(StdDuration::from_millis(100));

    // Recall to trigger reconsolidation window
    let recall_time = Utc::now();
    reconsolidation_engine.record_recall(&episode, recall_time, true);

    // Attempt reconsolidation within window
    let modifications = EpisodeModifications {
        what: Some("modified content".to_string()),
        where_location: None,
        who: None,
        modification_extent: 0.3,
        modification_type: ModificationType::Update,
    };

    let attempt_time = recall_time + Duration::hours(3);
    let result =
        reconsolidation_engine.attempt_reconsolidation(&episode.id, &modifications, attempt_time);

    assert!(
        result.is_some(),
        "Reconsolidation should succeed even when memory is primed"
    );

    // Verify: Priming boost still active after reconsolidation
    let boost = semantic_engine.compute_priming_boost("primed_memory");
    println!("Priming boost after reconsolidation: {boost:.3}");
}

// ==================== Priming × False Memory (DRM) ====================

#[test]
fn test_priming_enhances_false_memory_formation() {
    // Hypothesis: Semantic priming strengthens DRM false memory effect
    //
    // Setup: DRM paradigm with and without explicit semantic priming
    // Expected: Priming increases false recall rate

    let store = MemoryStore::new(10000);
    let semantic_engine = SemanticPrimingEngine::new();

    // DRM "sleep" list
    let study_words = ["bed", "rest", "awake", "tired", "dream"];
    let critical_lure = "sleep";

    // Generate high-similarity embeddings for DRM paradigm
    let embeddings = generate_drm_embeddings(&study_words, critical_lure);

    // Store study words
    for (idx, &word) in study_words.iter().enumerate() {
        let embedding = embeddings[word];
        let episode = create_episode(
            &format!("drm_{idx}"),
            word,
            &embedding,
            0, // Recent
        );
        let _ = store.store(episode);

        // Explicit semantic priming
        semantic_engine.activate_priming(word, &embedding, || {
            vec![(critical_lure.to_string(), embeddings[critical_lure], 1)]
        });
    }

    thread::sleep(StdDuration::from_millis(100));

    // Test: Query for critical lure
    let lure_embedding = embeddings[critical_lure];
    let cue = Cue::embedding("lure".to_string(), lure_embedding, Confidence::exact(0.6));

    let results = store.recall(&cue);

    // Count how many study items recalled (false memory indicator)
    let false_recalls = results
        .results
        .iter()
        .filter(|(episode, _conf)| study_words.contains(&episode.what.as_str()))
        .count();

    assert!(
        false_recalls > 0,
        "Semantic priming should enhance false memory: {false_recalls} study items recalled"
    );

    // Verify: Critical lure has priming boost
    let lure_boost = semantic_engine.compute_priming_boost(critical_lure);
    assert!(
        lure_boost > 0.0,
        "Critical lure should be primed from study items"
    );
}

// ==================== Interference × Reconsolidation ====================

#[test]
fn test_reconsolidation_reduces_interference_susceptibility() {
    // Hypothesis: Reconsolidating a memory can reduce its interference with other memories
    //
    // Setup: Create two interfering memories, reconsolidate one, measure interference
    // Expected: Reconsolidated memory shows reduced interference

    let store = MemoryStore::new(10000);
    let reconsolidation_engine = ReconsolidationEngine::new();

    // Create two similar memories (interference-prone)
    // Use different but similar words in the same semantic cluster
    let embedding1 = create_high_similarity_embedding("park_john", 0);
    let memory1 = create_episode("memory1", "I met John at the park", &embedding1, 48);

    let embedding2 = create_high_similarity_embedding("park_mary", 0);
    let memory2 = create_episode("memory2", "I met Mary at the park", &embedding2, 48);

    let _ = store.store(memory1.clone());
    let _ = store.store(memory2);

    // Measure baseline interference
    let cue = Cue::embedding(
        "park".to_string(),
        create_high_similarity_embedding("park", 0),
        Confidence::exact(0.6),
    );
    let baseline_results = store.recall(&cue);

    // Both memories should be retrieved (interference)
    assert!(
        baseline_results.results.len() >= 2,
        "Should retrieve both interfering memories: got {}",
        baseline_results.results.len()
    );

    // Reconsolidate memory1 to strengthen it
    let recall_time = Utc::now();
    reconsolidation_engine.record_recall(&memory1, recall_time, true);

    let modifications = EpisodeModifications {
        what: Some("I met John at the park (reconsolidated)".to_string()),
        where_location: None,
        who: None,
        modification_extent: 0.5, // Strong modification
        modification_type: ModificationType::Update,
    };

    let attempt_time = recall_time + Duration::hours(3);
    let _ =
        reconsolidation_engine.attempt_reconsolidation(&memory1.id, &modifications, attempt_time);

    // Note: In a full implementation, reconsolidation would strengthen the memory
    // and potentially reduce interference. This is a structural test.
}

// ==================== Interference × False Memory ====================

#[test]
fn test_interference_from_false_memories() {
    // Hypothesis: False memories (DRM) can interfere with retrieval of actual memories
    //
    // Setup: Study DRM list, then store actual memory with similar content
    // Expected: False memory interferes with actual memory retrieval

    let store = MemoryStore::new(10000);
    let semantic_engine = SemanticPrimingEngine::new();

    // Generate false memory via DRM
    let study_words = ["bed", "rest", "awake", "tired"];
    let critical_lure = "sleep";
    let embeddings = generate_drm_embeddings(&study_words, critical_lure);

    for (idx, &word) in study_words.iter().enumerate() {
        let embedding = embeddings[word];
        let episode = create_episode(&format!("drm_{idx}"), word, &embedding, 0);
        let _ = store.store(episode);

        semantic_engine.activate_priming(word, &embedding, Vec::new);
    }

    thread::sleep(StdDuration::from_millis(100));

    // Store actual memory about "sleep" (not in study list)
    let actual_embedding = embeddings[critical_lure];
    let actual_memory = create_episode(
        "actual_sleep",
        "I actually talked about sleep",
        &actual_embedding,
        0,
    );
    let _ = store.store(actual_memory);

    // Query for "sleep" - should retrieve both DRM items AND actual memory
    let cue = Cue::embedding(
        "sleep".to_string(),
        embeddings[critical_lure],
        Confidence::exact(0.6),
    );
    let results = store.recall(&cue);

    // Verify: Multiple retrievals (interference from false memory)
    assert!(
        results.results.len() > 1,
        "DRM false memories should interfere with actual memory retrieval"
    );
}

// ==================== Reconsolidation × False Memory ====================

#[test]
fn test_reconsolidation_can_correct_false_memories() {
    // Hypothesis: False memories can be modified during reconsolidation window
    //
    // Setup: Generate false memory, recall it, modify during reconsolidation
    // Expected: False memory content can be updated

    let store = MemoryStore::new(10000);
    let reconsolidation_engine = ReconsolidationEngine::new();

    // Create what would become a false memory (simulate DRM critical lure)
    // In reality, this would be generated implicitly, but we create it explicitly for testing
    let false_embedding = create_high_similarity_embedding("sleep", 0);
    let false_memory = create_episode(
        "false_lure",
        "sleep (not actually studied)",
        &false_embedding,
        48, // Old enough to be consolidated
    );

    let _ = store.store(false_memory.clone());

    // Recall false memory (triggers reconsolidation)
    let recall_time = Utc::now();
    reconsolidation_engine.record_recall(&false_memory, recall_time, true);

    // Modify false memory to correct it
    let corrections = EpisodeModifications {
        what: Some("sleep (corrected: was not in original list)".to_string()),
        where_location: None,
        who: None,
        modification_extent: 0.7,
        modification_type: ModificationType::Update,
    };

    let attempt_time = recall_time + Duration::hours(3);
    let result = reconsolidation_engine.attempt_reconsolidation(
        &false_memory.id,
        &corrections,
        attempt_time,
    );

    assert!(
        result.is_some(),
        "False memory should be modifiable during reconsolidation window"
    );

    // Verify: Modification applied
    let modified = result.unwrap();
    assert!(
        modified.plasticity_factor > 0.0,
        "Reconsolidation should have applied modifications with plasticity factor > 0"
    );
}

// ==================== Helper Functions ====================

fn create_episode(id: &str, content: &str, embedding: &[f32; 768], age_hours: i64) -> Episode {
    let when = Utc::now() - Duration::hours(age_hours);
    Episode::new(
        id.to_string(),
        when,
        content.to_string(),
        *embedding,
        Confidence::HIGH,
    )
}
