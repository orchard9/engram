//! Engram core graph engine with probabilistic operations.

pub mod types;

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::time::SystemTime;

/// Confidence interval for probabilistic operations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Confidence {
    pub mean: f64,
    pub lower: f64,
    pub upper: f64,
}

impl Confidence {
    /// Creates a new confidence interval
    pub fn new(mean: f64, lower: f64, upper: f64) -> Self {
        Self { mean, lower, upper }
    }

    /// Creates a new confidence interval with automatic clamping to [0, 1]
    pub fn new_clamped(mean: f64, lower: f64, upper: f64) -> Self {
        let lower = lower.clamp(0.0, 1.0);
        let upper = upper.clamp(lower, 1.0);
        let mean = mean.clamp(lower, upper);
        Self { mean, lower, upper }
    }

    /// Creates a point estimate with no uncertainty
    pub fn exact(value: f64) -> Self {
        let value = value.clamp(0.0, 1.0);
        Self {
            mean: value,
            lower: value,
            upper: value,
        }
    }

    /// High confidence (0.9)
    pub fn high() -> Self {
        Self::exact(0.9)
    }

    /// Medium confidence (0.5)
    pub fn medium() -> Self {
        Self::exact(0.5)
    }

    /// Low confidence (0.1)
    pub fn low() -> Self {
        Self::exact(0.1)
    }
}

// Type-state pattern: Memory node states
/// Unvalidated memory state - raw input
#[derive(Debug, Clone, Copy)]
pub struct Unvalidated;

/// Validated memory state - checked for consistency
#[derive(Debug, Clone, Copy)]
pub struct Validated;

/// Active memory state - ready for operations
#[derive(Debug, Clone, Copy)]
pub struct Active;

/// Consolidated memory state - compressed and optimized
#[derive(Debug, Clone, Copy)]
pub struct Consolidated;

/// Core memory node structure with type-state pattern
///
/// The state parameter ensures invalid state transitions are caught at compile time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode<State = Active> {
    pub id: String,
    pub content: Vec<u8>,
    pub embedding: Option<Vec<f32>>,
    pub activation: f64,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub confidence: Confidence,
    #[serde(skip)]
    _state: PhantomData<State>,
}

impl MemoryNode<Unvalidated> {
    /// Creates a new unvalidated memory node
    pub fn new_unvalidated(id: String, content: Vec<u8>) -> Self {
        Self {
            id,
            content,
            embedding: None,
            activation: 0.0,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            confidence: Confidence::low(),
            _state: PhantomData,
        }
    }

    /// Validates the memory node, transitioning to Validated state
    pub fn validate(self) -> Result<MemoryNode<Validated>, types::CoreError> {
        // Validation logic
        if self.content.is_empty() {
            return Err(types::CoreError::ValidationError {
                reason: "Memory content is empty".into(),
                expected: "Non-empty content vector with memory data".into(),
                suggestion: "Provide actual memory content before validation".into(),
                example:
                    "let node = MemoryNode::new_unvalidated(\"id\".into(), b\"content\".to_vec());"
                        .into(),
            });
        }

        Ok(MemoryNode {
            id: self.id,
            content: self.content,
            embedding: self.embedding,
            activation: self.activation,
            created_at: self.created_at,
            last_accessed: self.last_accessed,
            confidence: Confidence::medium(), // Increase confidence after validation
            _state: PhantomData,
        })
    }
}

impl MemoryNode<Validated> {
    /// Activates the memory node, transitioning to Active state
    pub fn activate(self) -> MemoryNode<Active> {
        MemoryNode {
            id: self.id,
            content: self.content,
            embedding: self.embedding,
            activation: 0.5, // Set initial activation
            created_at: self.created_at,
            last_accessed: SystemTime::now(),
            confidence: Confidence::high(), // High confidence for active memories
            _state: PhantomData,
        }
    }
}

impl MemoryNode<Active> {
    /// Consolidates the memory node, transitioning to Consolidated state
    pub fn consolidate(self) -> MemoryNode<Consolidated> {
        MemoryNode {
            id: self.id,
            content: self.content, // In real implementation, would compress
            embedding: self.embedding,
            activation: self.activation * 0.8, // Slight decay during consolidation
            created_at: self.created_at,
            last_accessed: self.last_accessed,
            confidence: self.confidence,
            _state: PhantomData,
        }
    }
}

// For backwards compatibility, default to Active state
impl MemoryNode {
    /// Creates a new active memory node (backwards compatible)
    pub fn new(id: String, content: Vec<u8>) -> Self {
        Self {
            id,
            content,
            embedding: None,
            activation: 0.5,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            confidence: Confidence::high(),
            _state: PhantomData,
        }
    }
}

/// Memory edge representing relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub source: String,
    pub target: String,
    pub weight: f64,
    pub edge_type: EdgeType,
    pub confidence: Confidence,
}

/// Types of edges in the memory graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    Semantic,
    Temporal,
    Causal,
    Associative,
}

/// Trait for activatable memory elements
pub trait Activatable {
    fn activate(&mut self, energy: f64);
    fn decay(&mut self, rate: f64);
    fn activation_level(&self) -> f64;
}

/// Trait for decayable memory elements
pub trait Decayable {
    fn apply_decay(&mut self, delta_time: f64);
    fn decay_rate(&self) -> f64;
}

/// Result of reconstruction operations with confidence scores
#[derive(Debug, Clone)]
pub struct ReconstructionResult {
    /// The reconstructed data (may be partial)
    pub data: Vec<u8>,
    /// Confidence in the reconstruction
    pub confidence: Confidence,
    /// Parts that couldn't be reconstructed with high confidence
    pub uncertain_regions: Vec<(usize, usize)>, // (start, end) byte ranges
}

/// Trait for reconstructable patterns
///
/// Following cognitive guidance: always returns something, even if low confidence.
/// This mirrors hippocampal pattern completion which provides partial reconstructions
/// rather than failing completely.
pub trait Reconstructable {
    /// Check if reconstruction would exceed threshold confidence
    fn can_reconstruct(&self, threshold: f64) -> bool;

    /// Reconstruct pattern, always returning partial results with confidence
    ///
    /// Never fails - returns best effort reconstruction with confidence score.
    /// Low confidence indicates uncertain or missing data.
    fn reconstruct(&self) -> ReconstructionResult;

    /// Reconstruct with a specific confidence threshold
    ///
    /// Returns partial data for regions above threshold, placeholder for below.
    fn reconstruct_with_threshold(&self, threshold: f64) -> ReconstructionResult;
}

impl<State> Activatable for MemoryNode<State> {
    fn activate(&mut self, energy: f64) {
        let clamped_energy = energy.clamp(0.0, 1.0);
        self.activation = (self.activation + clamped_energy).min(1.0);
        self.last_accessed = SystemTime::now();
    }

    fn decay(&mut self, rate: f64) {
        let clamped_rate = rate.clamp(0.0, 1.0);
        self.activation = (self.activation * (1.0 - clamped_rate)).max(0.0);
    }

    fn activation_level(&self) -> f64 {
        self.activation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_exact() {
        let conf = Confidence::exact(0.5);
        assert_eq!(conf.mean, 0.5);
        assert_eq!(conf.lower, 0.5);
        assert_eq!(conf.upper, 0.5);
    }

    #[test]
    fn test_memory_node_activation() {
        let mut node = MemoryNode::new("test".to_string(), vec![1, 2, 3]);
        node.activation = 0.3; // Set initial activation for test

        node.activate(0.4);
        assert_eq!(node.activation_level(), 0.7);

        node.activate(0.5);
        assert_eq!(node.activation_level(), 1.0); // Capped at 1.0

        node.decay(0.2);
        assert_eq!(node.activation_level(), 0.8);
    }

    #[test]
    fn test_confidence_clamping() {
        let conf = Confidence::new_clamped(1.5, -0.5, 2.0);
        assert_eq!(conf.lower, 0.0);
        assert_eq!(conf.upper, 1.0);
        assert_eq!(conf.mean, 1.0);
    }

    #[test]
    fn test_type_state_transitions() {
        let unvalidated = MemoryNode::new_unvalidated("test".to_string(), vec![1, 2, 3]);
        let validated = unvalidated.validate().unwrap();
        let active = validated.activate();
        let consolidated = active.consolidate();

        // Can't call validate on already validated node (won't compile)
        // consolidated.validate(); // This would be a compile error

        assert_eq!(consolidated.id, "test");
    }

    #[test]
    fn test_error_messages() {
        use crate::types::CoreError;

        let err = CoreError::node_not_found("user_123", vec!["user_124".into(), "user_125".into()]);
        let msg = err.to_string();

        // Check that error contains all required parts
        assert!(msg.contains("user_123"));
        assert!(msg.contains("Expected:"));
        assert!(msg.contains("Suggestion:"));
        assert!(msg.contains("Example:"));
        assert!(msg.contains("did you mean"));
        assert!(msg.contains("user_124"));
    }
}
