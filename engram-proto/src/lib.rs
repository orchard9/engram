//! Protocol buffer definitions for Engram cognitive graph database.
//!
//! This module provides type-safe gRPC communication with cognitive-friendly
//! message names and rich vocabulary that builds mental models.

// Include the generated protobuf code
tonic::include_proto!("engram.v1");

use chrono::{DateTime, Utc};
use prost_types::Timestamp;

/// Convert chrono DateTime to protobuf Timestamp
pub fn datetime_to_timestamp(dt: DateTime<Utc>) -> Timestamp {
    Timestamp {
        seconds: dt.timestamp(),
        nanos: dt.timestamp_subsec_nanos() as i32,
    }
}

/// Convert protobuf Timestamp to chrono DateTime
pub fn timestamp_to_datetime(ts: &Timestamp) -> DateTime<Utc> {
    DateTime::from_timestamp(ts.seconds, ts.nanos as u32).unwrap_or_else(Utc::now)
}

impl Confidence {
    /// Create a new confidence with automatic category assignment
    pub fn new(value: f32) -> Self {
        let value = value.clamp(0.0, 1.0);
        let category = match value {
            v if v == 0.0 => ConfidenceCategory::None as i32,
            v if v <= 0.2 => ConfidenceCategory::Low as i32,
            v if v <= 0.7 => ConfidenceCategory::Medium as i32,
            v if v <= 0.95 => ConfidenceCategory::High as i32,
            _ => ConfidenceCategory::Certain as i32,
        };

        Self {
            value,
            category,
            reasoning: String::new(),
        }
    }

    /// Create confidence with reasoning explanation
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = reasoning.into();
        self
    }

    /// Check if confidence is above a threshold
    pub fn is_above(&self, threshold: f32) -> bool {
        self.value >= threshold
    }

    /// Get semantic category as enum
    pub fn semantic_category(&self) -> ConfidenceCategory {
        ConfidenceCategory::try_from(self.category).unwrap_or(ConfidenceCategory::Unspecified)
    }
}

impl Memory {
    /// Create a new memory with required fields
    pub fn new(id: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            embedding,
            activation: 0.0,
            confidence: Some(Confidence::new(0.5)), // Default medium confidence
            last_access: Some(datetime_to_timestamp(Utc::now())),
            created_at: Some(datetime_to_timestamp(Utc::now())),
            decay_rate: 0.1,
            content: String::new(),
            metadata: Default::default(),
            tags: Vec::new(),
            memory_type: MemoryType::Unspecified as i32,
        }
    }

    /// Set the confidence for this memory
    pub fn with_confidence(mut self, confidence: Confidence) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Set human-readable content
    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = content.into();
        self
    }

    /// Add a tag to this memory
    pub fn add_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set memory type
    pub fn with_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = memory_type as i32;
        self
    }
}

impl Episode {
    /// Create a new episode with required fields
    pub fn new(
        id: impl Into<String>,
        when: DateTime<Utc>,
        what: impl Into<String>,
        embedding: Vec<f32>,
    ) -> Self {
        Self {
            id: id.into(),
            when: Some(datetime_to_timestamp(when)),
            what: what.into(),
            embedding,
            encoding_confidence: Some(Confidence::new(0.8)), // Default high confidence
            where_location: String::new(),
            who: Vec::new(),
            why: String::new(),
            how: String::new(),
            decay_rate: 0.1,
            emotional_valence: 0.0,
            importance: 0.5,
            consolidation_state: ConsolidationState::Recent as i32,
            last_replay: None,
        }
    }

    /// Set location context
    pub fn at_location(mut self, location: impl Into<String>) -> Self {
        self.where_location = location.into();
        self
    }

    /// Set participants
    pub fn with_people(mut self, people: Vec<String>) -> Self {
        self.who = people;
        self
    }

    /// Set emotional valence (-1 to 1)
    pub fn with_emotion(mut self, valence: f32) -> Self {
        self.emotional_valence = valence.clamp(-1.0, 1.0);
        self
    }

    /// Set importance (0 to 1)
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }
}

impl Cue {
    /// Create an embedding-based cue
    pub fn from_embedding(vector: Vec<f32>, threshold: f32) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            cue_type: Some(cue::CueType::Embedding(EmbeddingCue {
                vector,
                similarity_threshold: threshold,
            })),
            threshold: Some(Confidence::new(threshold)),
            max_results: 10,
            spread_activation: true,
            activation_decay: 0.8,
        }
    }

    /// Create a semantic search cue
    pub fn from_query(query: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            cue_type: Some(cue::CueType::Semantic(SemanticCue {
                query: query.into(),
                fuzzy_threshold: 0.7,
                required_tags: Vec::new(),
                excluded_tags: Vec::new(),
            })),
            threshold: Some(Confidence::new(0.5)),
            max_results: 10,
            spread_activation: true,
            activation_decay: 0.8,
        }
    }

    /// Create a context-based cue
    pub fn from_context(
        time_start: Option<DateTime<Utc>>,
        time_end: Option<DateTime<Utc>>,
        location: Option<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            cue_type: Some(cue::CueType::Context(ContextCue {
                time_start: time_start.map(datetime_to_timestamp),
                time_end: time_end.map(datetime_to_timestamp),
                location: location.unwrap_or_default(),
                participants: Vec::new(),
            })),
            threshold: Some(Confidence::new(0.5)),
            max_results: 10,
            spread_activation: true,
            activation_decay: 0.8,
        }
    }
}

// Add uuid for generating IDs
use uuid;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_categories() {
        assert_eq!(
            Confidence::new(0.0).semantic_category(),
            ConfidenceCategory::None
        );
        assert_eq!(
            Confidence::new(0.1).semantic_category(),
            ConfidenceCategory::Low
        );
        assert_eq!(
            Confidence::new(0.5).semantic_category(),
            ConfidenceCategory::Medium
        );
        assert_eq!(
            Confidence::new(0.9).semantic_category(),
            ConfidenceCategory::High
        );
        assert_eq!(
            Confidence::new(1.0).semantic_category(),
            ConfidenceCategory::Certain
        );
    }

    #[test]
    fn test_memory_builder() {
        let memory = Memory::new("test_id", vec![0.1; 768])
            .with_content("Test memory")
            .add_tag("test")
            .with_type(MemoryType::Semantic);

        assert_eq!(memory.id, "test_id");
        assert_eq!(memory.content, "Test memory");
        assert_eq!(memory.tags, vec!["test"]);
        assert_eq!(memory.memory_type, MemoryType::Semantic as i32);
    }

    #[test]
    fn test_episode_builder() {
        let episode = Episode::new(
            "episode_1",
            Utc::now(),
            "Something happened",
            vec![0.2; 768],
        )
        .at_location("Home")
        .with_emotion(0.7)
        .with_importance(0.9);

        assert_eq!(episode.where_location, "Home");
        assert_eq!(episode.emotional_valence, 0.7);
        assert_eq!(episode.importance, 0.9);
    }
}
