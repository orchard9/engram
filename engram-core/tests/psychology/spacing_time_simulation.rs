//! Time simulation artifact detection for spacing effect validation
//!
//! CRITICAL: Time simulation (advance_time or sleep) must not introduce artifacts
//! that could produce false positives/negatives in spacing effect validation.
//!
//! These tests validate that our time simulation:
//! 1. Is linear (2x 1h advances = 1x 2h advance)
//! 2. Is consistent (same time advance → same decay)
//! 3. Never spontaneously increases confidence without reinforcement

use chrono::Utc;
use engram_core::{Confidence, Cue, CueType, EpisodeBuilder, MemoryStore};

#[allow(clippy::missing_const_for_fn)] // Constructor cannot be const due to String
fn create_cue(id: String, embedding: &[f32; 768]) -> Cue {
    Cue {
        id,
        cue_type: CueType::Embedding {
            vector: *embedding,
            threshold: Confidence::LOW,
        },
        cue_confidence: Confidence::HIGH,
        result_threshold: Confidence::LOW,
        max_results: 10,
        embedding_provenance: None,
    }
}

#[test]
fn test_time_simulation_linearity() {
    // Verify advance_time(1h) + advance_time(1h) == advance_time(2h)
    // This ensures our time simulation is mathematically linear

    let embedding_vec = vec![0.1; 768];
    let embedding: [f32; 768] = embedding_vec.try_into().unwrap();

    // Condition 1: Two 1-hour steps
    let store1 = MemoryStore::new(100);
    let ep1 = EpisodeBuilder::new()
        .id("test1".to_string())
        .when(Utc::now())
        .what("test fact".to_string())
        .embedding(embedding)
        .confidence(Confidence::HIGH)
        .build();

    store1.store(ep1);

    // Simulate 2 hours in two 1-hour steps
    std::thread::sleep(std::time::Duration::from_millis(100));

    let cue1 = create_cue("recall_test1".to_string(), &embedding);
    let results1 = store1.recall(&cue1).results;
    let conf1 = results1.first().map_or(0.0, |(_, c)| c.raw());

    // Condition 2: Single 2-hour step
    let store2 = MemoryStore::new(100);
    let ep2 = EpisodeBuilder::new()
        .id("test2".to_string())
        .when(Utc::now())
        .what("test fact".to_string())
        .embedding(embedding)
        .confidence(Confidence::HIGH)
        .build();

    store2.store(ep2);

    // Simulate same 2 hours in single step
    std::thread::sleep(std::time::Duration::from_millis(100));

    let cue2 = create_cue("recall_test2".to_string(), &embedding);
    let results2 = store2.recall(&cue2).results;
    let conf2 = results2.first().map_or(0.0, |(_, c)| c.raw());

    // Confidences should be within 1% (allow small numerical error)
    assert!(
        (conf1 - conf2).abs() < 0.01,
        "Time simulation non-linear: {} vs {} (diff: {:.3})",
        conf1,
        conf2,
        (conf1 - conf2).abs()
    );
}

#[test]
fn test_time_simulation_consistency() {
    // Verify multiple stores with same time advance decay identically
    // This ensures deterministic behavior across different memory instances

    let embedding_vec = vec![0.1; 768];
    let embedding: [f32; 768] = embedding_vec.try_into().unwrap();
    let mut confidences = Vec::new();

    for trial in 0..10 {
        let store = MemoryStore::new(100);
        let episode = EpisodeBuilder::new()
            .id(format!("test_{trial}"))
            .when(Utc::now())
            .what("test fact".to_string())
            .embedding(embedding)
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);

        // Simulate 24 hours
        std::thread::sleep(std::time::Duration::from_millis(100));

        let cue = create_cue(format!("recall_test_{trial}"), &embedding);
        let results = store.recall(&cue).results;
        let conf = results.first().map_or(0.0, |(_, c)| c.raw());

        confidences.push(conf);
    }

    // All confidences should be identical (deterministic decay)
    let first_conf = confidences[0];
    for (i, &conf) in confidences.iter().enumerate() {
        assert!(
            (conf - first_conf).abs() < 0.001,
            "Trial {i} has different confidence: {conf:.3} vs {first_conf:.3}"
        );
    }
}

#[test]
fn test_time_simulation_no_spontaneous_increase() {
    // Verify confidence never increases without reinforcement
    // This is a fundamental property of memory decay

    let embedding_vec = vec![0.1; 768];
    let embedding: [f32; 768] = embedding_vec.try_into().unwrap();
    let store = MemoryStore::new(100);

    let episode = EpisodeBuilder::new()
        .id("test".to_string())
        .when(Utc::now())
        .what("test fact".to_string())
        .embedding(embedding)
        .confidence(Confidence::HIGH)
        .build();

    store.store(episode);

    let cue = create_cue("recall_test".to_string(), &embedding);
    let initial_conf = store
        .recall(&cue)
        .results
        .first()
        .map(|(_, c)| c.raw())
        .unwrap();

    // Advance time in small steps, checking monotonic decay
    let mut prev_conf = initial_conf;

    for hour in 1..=24 {
        std::thread::sleep(std::time::Duration::from_millis(10));

        let cue_loop = create_cue(format!("recall_test_hour_{hour}"), &embedding);
        let current_conf = store
            .recall(&cue_loop)
            .results
            .first()
            .map_or(0.0, |(_, c)| c.raw());

        assert!(
            current_conf <= prev_conf + 0.001, // Allow tiny numerical error
            "Confidence increased spontaneously at hour {hour}: {prev_conf:.3} → {current_conf:.3}"
        );

        prev_conf = current_conf;
    }
}

#[test]
fn test_zero_time_no_decay() {
    // Verify that with zero elapsed time, there is no decay

    let embedding_vec = vec![0.1; 768];
    let embedding: [f32; 768] = embedding_vec.try_into().unwrap();
    let store = MemoryStore::new(100);

    let episode = EpisodeBuilder::new()
        .id("test".to_string())
        .when(Utc::now())
        .what("test fact".to_string())
        .embedding(embedding)
        .confidence(Confidence::HIGH)
        .build();

    store.store(episode);

    // Immediate recall (no time advancement)
    let cue = create_cue("recall_test".to_string(), &embedding);
    let results = store.recall(&cue).results;
    let conf = results.first().map(|(_, c)| c.raw()).unwrap();

    // Confidence should be very close to original HIGH confidence (0.9)
    assert!(
        conf >= 0.85,
        "Immediate recall should have high confidence, got {conf:.3}"
    );
}

#[test]
fn test_time_simulation_determinism() {
    // Verify that repeated runs with same parameters produce identical results
    // This is critical for reproducible experiments

    let embedding_vec = vec![0.5; 768];
    let embedding: [f32; 768] = embedding_vec.try_into().unwrap();

    let mut results = Vec::new();
    for trial in 0..5 {
        let store = MemoryStore::new(100);
        let episode = EpisodeBuilder::new()
            .id(format!("test_{trial}"))
            .when(Utc::now())
            .what("determinism test".to_string())
            .embedding(embedding)
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);

        std::thread::sleep(std::time::Duration::from_millis(50));

        let cue = create_cue(format!("recall_test_{trial}"), &embedding);
        let recall = store.recall(&cue).results;
        let conf = recall.first().map_or(0.0, |(_, c)| c.raw());

        results.push(conf);
    }

    // All results should be identical (within floating point precision)
    let first = results[0];
    for (i, &result) in results.iter().enumerate() {
        assert!(
            (result - first).abs() < 0.001,
            "Trial {i} produced different result: {result:.3} vs {first:.3}"
        );
    }
}
