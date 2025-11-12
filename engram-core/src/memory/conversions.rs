//! Conversion utilities between Memory and DualMemoryNode
//!
//! These conversions support gradual migration from pure episodic to dual memory.
//! Zero-copy semantics are used where possible via Arc and reference counting.

use super::dual_types::DualMemoryNode;
use super::types::{Episode, Memory};
use uuid::Uuid;

#[cfg(test)]
use super::dual_types::MemoryNodeType;

impl From<Memory> for DualMemoryNode {
    /// Convert Memory to DualMemoryNode as an Episode.
    ///
    /// This is a zero-alloc conversion - the embedding is moved, not copied.
    /// Used during migration when loading existing pure-episodic graphs.
    fn from(memory: Memory) -> Self {
        let id = Uuid::parse_str(&memory.id).unwrap_or_else(|_| Uuid::new_v4());

        Self::new_episode(
            id,
            memory.id.clone(),
            memory.embedding,
            memory.confidence,
            memory.activation(),
        )
    }
}

impl From<&Memory> for DualMemoryNode {
    /// Convert &Memory to DualMemoryNode (requires embedding copy).
    ///
    /// Use this when the original Memory must remain valid.
    fn from(memory: &Memory) -> Self {
        let id = Uuid::parse_str(&memory.id).unwrap_or_else(|_| Uuid::new_v4());

        Self::new_episode(
            id,
            memory.id.clone(),
            memory.embedding,
            memory.confidence,
            memory.activation(),
        )
    }
}

impl From<DualMemoryNode> for Memory {
    /// Convert DualMemoryNode to Memory (for backwards compatibility).
    ///
    /// Concept nodes are represented as Memory with centroid embedding.
    /// This enables gradual rollout without breaking existing APIs.
    fn from(dual: DualMemoryNode) -> Self {
        dual.to_memory()
    }
}

impl DualMemoryNode {
    /// Convert back to Memory (for backwards compatibility).
    ///
    /// Concept nodes are represented as Memory with centroid embedding.
    /// This enables gradual rollout without breaking existing APIs.
    #[must_use]
    pub fn to_memory(&self) -> Memory {
        let mut memory = Memory::new(self.id.to_string(), self.embedding, self.confidence);

        memory.set_activation(self.activation());
        memory.last_access = self.last_access;
        memory.created_at = self.created_at;

        memory
    }

    /// Create from Episode struct (existing Engram type).
    ///
    /// This is the primary ingestion path for new memories.
    /// Episode is passed by value to take ownership and avoid cloning the embedding.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_episode(episode: Episode, strength: f32) -> Self {
        let id = Uuid::parse_str(&episode.id).unwrap_or_else(|_| Uuid::new_v4());

        Self::new_episode(
            id,
            episode.id.clone(),
            episode.embedding,
            episode.encoding_confidence,
            strength,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Confidence;

    #[test]
    fn test_memory_to_dual_conversion() {
        let memory = Memory::new("test-123".to_string(), [0.5f32; 768], Confidence::HIGH);
        memory.set_activation(0.8);

        let dual = DualMemoryNode::from(memory);
        assert!(dual.is_episode());
        assert!((dual.activation() - 0.8).abs() < 0.001);
        assert_eq!(dual.confidence, Confidence::HIGH);
    }

    #[test]
    fn test_dual_to_memory_conversion() {
        let dual = DualMemoryNode::new_episode(
            Uuid::new_v4(),
            "episode-456".to_string(),
            [0.3f32; 768],
            Confidence::MEDIUM,
            0.6,
        );

        let memory = dual.to_memory();
        assert!((memory.activation() - 0.6).abs() < 0.001);
        assert_eq!(memory.confidence, Confidence::MEDIUM);
    }

    #[test]
    fn test_consolidation_score_atomic_update() {
        let dual = DualMemoryNode::new_episode(
            Uuid::new_v4(),
            "episode-789".to_string(),
            [0.1f32; 768],
            Confidence::LOW,
            0.5,
        );

        assert_eq!(dual.node_type.consolidation_score(), Some(0.0));

        dual.node_type.update_consolidation_score(0.75);
        assert_eq!(dual.node_type.consolidation_score(), Some(0.75));
    }

    #[test]
    fn test_concept_instance_count_atomic() {
        let dual =
            DualMemoryNode::new_concept(Uuid::new_v4(), [0.2f32; 768], 0.85, 5, Confidence::HIGH);

        assert_eq!(dual.node_type.instance_count(), Some(5));

        dual.node_type.increment_instances();
        assert_eq!(dual.node_type.instance_count(), Some(6));
    }

    #[test]
    fn test_memory_reference_conversion() {
        let memory = Memory::new("test-ref".to_string(), [0.7f32; 768], Confidence::MEDIUM);
        memory.set_activation(0.5);

        let dual = DualMemoryNode::from(&memory);
        assert!(dual.is_episode());
        assert!((dual.activation() - 0.5).abs() < 0.001);

        // Original memory should still be valid
        assert!((memory.activation() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_episode_to_dual_conversion() {
        use chrono::Utc;

        let episode = Episode::new(
            "episode-test".to_string(),
            Utc::now(),
            "Test episode content".to_string(),
            [0.4f32; 768],
            Confidence::HIGH,
        );

        let dual = DualMemoryNode::from_episode(episode, 0.9);
        assert!(dual.is_episode());
        assert!((dual.activation() - 0.9).abs() < 0.001);
        assert_eq!(dual.confidence, Confidence::HIGH);
    }

    #[test]
    fn test_concept_node_creation() {
        let centroid = [0.5f32; 768];
        let dual =
            DualMemoryNode::new_concept(Uuid::new_v4(), centroid, 0.92, 10, Confidence::HIGH);

        assert!(dual.is_concept());
        assert!(!dual.is_episode());
        assert_eq!(dual.node_type.instance_count(), Some(10));

        // Embedding should match centroid
        for (a, b) in dual.embedding.iter().zip(centroid.iter()) {
            assert!((a - b).abs() < 0.0001);
        }
    }

    #[test]
    fn test_roundtrip_memory_conversion() {
        let original = Memory::new("roundtrip".to_string(), [0.6f32; 768], Confidence::MEDIUM);
        original.set_activation(0.75);

        let dual = DualMemoryNode::from(&original);
        let back_to_memory = dual.to_memory();

        assert_eq!(back_to_memory.confidence, original.confidence);
        assert!((back_to_memory.activation() - original.activation()).abs() < 0.001);
    }

    #[test]
    fn test_consolidation_score_clamping() {
        let dual = DualMemoryNode::new_episode(
            Uuid::new_v4(),
            "clamp-test".to_string(),
            [0.1f32; 768],
            Confidence::LOW,
            0.5,
        );

        // Test upper bound clamping
        dual.node_type.update_consolidation_score(1.5);
        assert_eq!(dual.node_type.consolidation_score(), Some(1.0));

        // Test lower bound clamping
        dual.node_type.update_consolidation_score(-0.5);
        assert_eq!(dual.node_type.consolidation_score(), Some(0.0));
    }

    #[test]
    fn test_activation_clamping() {
        let dual = DualMemoryNode::new_episode(
            Uuid::new_v4(),
            "activation-clamp".to_string(),
            [0.2f32; 768],
            Confidence::MEDIUM,
            0.5,
        );

        // Test upper bound
        dual.set_activation(1.5);
        assert!((dual.activation() - 1.0).abs() < f32::EPSILON);

        // Test lower bound
        dual.set_activation(-0.5);
        assert!((dual.activation() - 0.0).abs() < f32::EPSILON);

        // Test add_activation clamping
        dual.set_activation(0.8);
        dual.add_activation(0.5);
        assert!((dual.activation() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serialization_preparation() {
        let mut dual = DualMemoryNode::new_episode(
            Uuid::new_v4(),
            "serialize-test".to_string(),
            [0.3f32; 768],
            Confidence::HIGH,
            0.6,
        );

        // Update atomic field
        dual.node_type.update_consolidation_score(0.42);

        // Prepare for serialization
        dual.node_type.prepare_serialization();

        // Check that value field was synced
        if let MemoryNodeType::Episode {
            consolidation_score_value,
            ..
        } = &dual.node_type
        {
            assert!((consolidation_score_value - 0.42).abs() < 0.001);
        } else {
            unreachable!("Test created an Episode node, should match");
        }
    }

    #[test]
    fn test_atomic_restore_after_deserialization() {
        let mut dual = DualMemoryNode::new_concept(
            Uuid::new_v4(),
            [0.4f32; 768],
            0.88,
            15,
            Confidence::MEDIUM,
        );

        // Simulate deserialization scenario
        if let MemoryNodeType::Concept {
            instance_count_value,
            ..
        } = &mut dual.node_type
        {
            *instance_count_value = 25; // Simulate deserialized value
        }

        // Restore atomics
        dual.node_type.restore_atomics();

        // Check that atomic field was updated
        assert_eq!(dual.node_type.instance_count(), Some(25));
    }
}
