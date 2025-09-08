//! Tests for protobuf types and conversions

use chrono::Utc;
use engram_proto::*;

#[test]
fn test_confidence_creation() {
    // Test basic confidence creation
    let conf = Confidence::new(0.75);
    assert!((conf.value - 0.75).abs() < f32::EPSILON);
    assert_eq!(conf.category, ConfidenceCategory::High as i32);
    assert_eq!(conf.reasoning, "");

    // Test confidence with reasoning
    let conf_with_reason = Confidence::new(0.5).with_reasoning("Based on similarity metrics");
    assert!((conf_with_reason.value - 0.5).abs() < f32::EPSILON);
    assert_eq!(conf_with_reason.category, ConfidenceCategory::Medium as i32);
    assert_eq!(conf_with_reason.reasoning, "Based on similarity metrics");

    // Test boundary values
    assert_eq!(
        Confidence::new(0.0).category,
        ConfidenceCategory::None as i32
    );
    assert_eq!(
        Confidence::new(0.1).category,
        ConfidenceCategory::Low as i32
    );
    assert_eq!(
        Confidence::new(0.95).category,
        ConfidenceCategory::High as i32
    );
    assert_eq!(
        Confidence::new(1.0).category,
        ConfidenceCategory::Certain as i32
    );

    // Test clamping
    assert!(Confidence::new(-0.5).value.abs() < f32::EPSILON);
    assert!((Confidence::new(1.5).value - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_confidence_methods() {
    let conf = Confidence::new(0.8);

    // Test threshold checking
    assert!(conf.is_above(0.7));
    assert!(conf.is_above(0.8));
    assert!(!conf.is_above(0.9));

    // Test semantic category
    assert_eq!(conf.semantic_category(), ConfidenceCategory::High);
}

#[test]
fn test_memory_builder() {
    let embedding = vec![0.1; 768];
    let memory = Memory::new("test_memory", embedding.clone())
        .with_content("This is a test memory")
        .add_tag("test")
        .add_tag("example")
        .with_type(MemoryType::Semantic)
        .with_confidence(Confidence::new(0.9));

    assert_eq!(memory.id, "test_memory");
    assert_eq!(memory.embedding, embedding);
    assert_eq!(memory.content, "This is a test memory");
    assert_eq!(memory.tags, vec!["test", "example"]);
    assert_eq!(memory.memory_type, MemoryType::Semantic as i32);
    assert!((memory.confidence.as_ref().unwrap().value - 0.9).abs() < f32::EPSILON);
    assert!(memory.activation.abs() < f32::EPSILON);
    assert!((memory.decay_rate - 0.1).abs() < f32::EPSILON);
}

#[test]
fn test_episode_builder() {
    let now = Utc::now();
    let embedding = vec![0.2; 768];
    let episode = Episode::new(
        "episode_001",
        now,
        "Attended conference presentation",
        embedding.clone(),
    )
    .at_location("Convention Center Room A")
    .with_people(vec!["Alice".to_string(), "Bob".to_string()])
    .with_emotion(0.7)
    .with_importance(0.9);

    assert_eq!(episode.id, "episode_001");
    assert_eq!(episode.what, "Attended conference presentation");
    assert_eq!(episode.where_location, "Convention Center Room A");
    assert_eq!(episode.who, vec!["Alice", "Bob"]);
    assert!((episode.emotional_valence - 0.7).abs() < f32::EPSILON);
    assert!((episode.importance - 0.9).abs() < f32::EPSILON);
    assert_eq!(episode.embedding, embedding);
    assert_eq!(
        episode.consolidation_state,
        ConsolidationState::Recent as i32
    );

    // Check timestamp conversion
    let ts = episode.when.as_ref().unwrap();
    let converted_time = timestamp_to_datetime(ts);
    assert!((converted_time.timestamp() - now.timestamp()).abs() <= 1);
}

#[test]
fn test_episode_emotional_clamping() {
    let episode = Episode::new("test", Utc::now(), "Test", vec![0.0; 768])
        .with_emotion(2.0) // Should be clamped to 1.0
        .with_importance(-0.5); // Should be clamped to 0.0

    assert!((episode.emotional_valence - 1.0).abs() < f32::EPSILON);
    assert!(episode.importance.abs() < f32::EPSILON);
}

#[test]
fn test_cue_embedding() {
    let embedding = vec![0.3; 768];
    let cue = Cue::from_embedding(embedding.clone(), 0.8);

    assert!(!cue.id.is_empty()); // Should have generated UUID
    assert_eq!(cue.max_results, 10);
    assert!(cue.spread_activation);
    assert!((cue.activation_decay - 0.8).abs() < f32::EPSILON);
    assert!((cue.threshold.as_ref().unwrap().value - 0.8).abs() < f32::EPSILON);

    // Check the cue type
    match cue.cue_type {
        Some(cue::CueType::Embedding(ref emb_cue)) => {
            assert_eq!(emb_cue.vector, embedding);
            assert!((emb_cue.similarity_threshold - 0.8).abs() < f32::EPSILON);
        }
        _ => panic!("Expected embedding cue"),
    }
}

#[test]
fn test_cue_semantic() {
    let cue = Cue::from_query("Find memories about rust programming");

    assert!(!cue.id.is_empty());
    assert_eq!(cue.max_results, 10);
    assert!(cue.spread_activation);
    assert!((cue.threshold.as_ref().unwrap().value - 0.5).abs() < f32::EPSILON);

    match cue.cue_type {
        Some(cue::CueType::Semantic(ref sem_cue)) => {
            assert_eq!(sem_cue.query, "Find memories about rust programming");
            assert!((sem_cue.fuzzy_threshold - 0.7).abs() < f32::EPSILON);
            assert!(sem_cue.required_tags.is_empty());
            assert!(sem_cue.excluded_tags.is_empty());
        }
        _ => panic!("Expected semantic cue"),
    }
}

#[test]
fn test_cue_context() {
    let start = Utc::now() - chrono::Duration::hours(2);
    let end = Utc::now();
    let cue = Cue::from_context(Some(start), Some(end), Some("Meeting Room".to_string()));

    assert!(!cue.id.is_empty());

    match cue.cue_type {
        Some(cue::CueType::Context(ref ctx_cue)) => {
            assert!(ctx_cue.time_start.is_some());
            assert!(ctx_cue.time_end.is_some());
            assert_eq!(ctx_cue.location, "Meeting Room");
            assert!(ctx_cue.participants.is_empty());
        }
        _ => panic!("Expected context cue"),
    }
}

#[test]
fn test_timestamp_conversion() {
    let now = Utc::now();
    let ts = datetime_to_timestamp(now);
    let converted = timestamp_to_datetime(&ts);

    // Should be within a microsecond (accounting for nanosecond precision loss)
    assert!((converted.timestamp_micros() - now.timestamp_micros()).abs() <= 1);
}

#[test]
fn test_memory_defaults() {
    let memory = Memory::new("test", vec![0.0; 768]);

    // Check default values
    assert!(memory.activation.abs() < f32::EPSILON);
    assert!((memory.decay_rate - 0.1).abs() < f32::EPSILON);
    assert_eq!(memory.content, "");
    assert!(memory.metadata.is_empty());
    assert!(memory.tags.is_empty());
    assert_eq!(memory.memory_type, MemoryType::Unspecified as i32);
    assert!(memory.confidence.is_some());
    assert!(memory.last_access.is_some());
    assert!(memory.created_at.is_some());
}

#[test]
fn test_episode_defaults() {
    let episode = Episode::new("test", Utc::now(), "Event", vec![0.0; 768]);

    // Check default values
    assert_eq!(episode.where_location, "");
    assert!(episode.who.is_empty());
    assert_eq!(episode.why, "");
    assert_eq!(episode.how, "");
    assert!((episode.decay_rate - 0.1).abs() < f32::EPSILON);
    assert!(episode.emotional_valence.abs() < f32::EPSILON);
    assert!((episode.importance - 0.5).abs() < f32::EPSILON);
    assert_eq!(
        episode.consolidation_state,
        ConsolidationState::Recent as i32
    );
    assert!(episode.last_replay.is_none());
}

#[test]
fn test_confidence_categories() {
    // Test all confidence category boundaries
    let test_cases = vec![
        (0.0, ConfidenceCategory::None),
        (0.05, ConfidenceCategory::Low),
        (0.2, ConfidenceCategory::Low),
        (0.21, ConfidenceCategory::Medium),
        (0.5, ConfidenceCategory::Medium),
        (0.7, ConfidenceCategory::Medium),
        (0.71, ConfidenceCategory::High),
        (0.95, ConfidenceCategory::High),
        (0.96, ConfidenceCategory::Certain),
        (1.0, ConfidenceCategory::Certain),
    ];

    for (value, expected_category) in test_cases {
        let conf = Confidence::new(value);
        assert_eq!(
            conf.semantic_category(),
            expected_category,
            "Failed for value {}",
            value
        );
    }
}

#[test]
fn test_memory_metadata() {
    let mut memory = Memory::new("test", vec![0.0; 768]);

    // Add metadata
    memory
        .metadata
        .insert("source".to_string(), "api".to_string());
    memory
        .metadata
        .insert("version".to_string(), "1.0".to_string());

    assert_eq!(memory.metadata.get("source"), Some(&"api".to_string()));
    assert_eq!(memory.metadata.get("version"), Some(&"1.0".to_string()));
    assert_eq!(memory.metadata.len(), 2);
}

#[test]
fn test_uuid_generation() {
    // Ensure unique IDs are generated for cues
    let cue1 = Cue::from_query("test");
    let cue2 = Cue::from_query("test");
    let cue3 = Cue::from_embedding(vec![0.0; 768], 0.5);

    assert_ne!(cue1.id, cue2.id);
    assert_ne!(cue1.id, cue3.id);
    assert_ne!(cue2.id, cue3.id);

    // All should be valid UUIDs
    for cue in [cue1, cue2, cue3] {
        assert!(uuid::Uuid::parse_str(&cue.id).is_ok());
    }
}
