//! Neo4j graph to Engram memory transformation

use engram_core::{Confidence, Episode};
use migration_common::{MemoryTransformer, MigrationResult, SourceRecord};
use std::collections::HashMap;

/// Transforms Neo4j nodes and relationships to Engram memories
pub struct Neo4jTransformer {
    memory_space_prefix: String,
    label_to_space: HashMap<String, String>,
    default_confidence: Confidence,
}

impl Neo4jTransformer {
    /// Create a new Neo4j transformer
    #[must_use]
    pub fn new(memory_space_prefix: String, label_to_space: HashMap<String, String>) -> Self {
        Self {
            memory_space_prefix,
            label_to_space,
            default_confidence: Confidence::MEDIUM,
        }
    }

    /// Extract primary label from record properties
    fn extract_primary_label(&self, record: &SourceRecord) -> Option<String> {
        record
            .properties
            .get("labels")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .map(ToString::to_string)
    }
}

impl MemoryTransformer for Neo4jTransformer {
    fn transform(&self, record: &SourceRecord, embedding: &[f32; 768]) -> MigrationResult<Episode> {
        // Create episode ID with neo4j prefix
        let id = format!("neo4j_node_{}", record.id);

        // Create episode with embedding
        let episode = Episode {
            id,
            what: record.text_content.clone(),
            embedding: *embedding,
            confidence: self.default_confidence,
            when: record.created_at,
            provenance: Some(format!("neo4j:{}", record.id)),
        };

        Ok(episode)
    }

    fn memory_space_id(&self, record: &SourceRecord) -> String {
        // Try to map label to specific space
        if let Some(label) = self.extract_primary_label(record) {
            if let Some(space) = self.label_to_space.get(&label) {
                return space.clone();
            }
            // Otherwise use label as space name
            return format!("{}_{}", self.memory_space_prefix, label.to_lowercase());
        }

        // Default to prefix
        self.memory_space_prefix.clone()
    }

    fn initial_confidence(&self) -> Confidence {
        self.default_confidence
    }
}
