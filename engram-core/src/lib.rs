//! Engram core graph engine with probabilistic operations.

#![cfg_attr(docsrs, feature(doc_cfg))]
// Safety-focused Clippy lints to prevent unsafe error handling regression
#![warn(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented,
    clippy::todo
)]
#![deny(clippy::unwrap_in_result, clippy::panic_in_result_fn)]

// Allow in tests only
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
// mod test_helpers; // TODO: Create test_helpers module

// Use mimalloc as global allocator for better performance
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod activation;
#[cfg(feature = "security")]
pub mod auth;
pub mod batch;
pub mod cognitive;
#[cfg(feature = "pattern_completion")]
pub mod completion;
pub mod compute;
pub mod consolidation;
pub mod cue;
#[cfg(feature = "psychological_decay")]
pub mod decay;
pub mod differential;
pub mod embedding;
pub mod error;
pub mod error_review;
pub mod error_testing;
pub mod features;
pub mod graph;
#[cfg(feature = "hnsw_index")]
pub mod index;
pub mod memory;
pub mod memory_graph;
#[cfg(feature = "monitoring")]
pub mod metrics;
mod numeric;
pub mod query;
pub mod registry;
#[cfg(feature = "security")]
pub mod security;
#[cfg(feature = "memory_mapped_persistence")]
pub mod storage;
pub mod store;
pub mod streaming;
pub mod streaming_health;
#[cfg(any(feature = "cognitive_tracing", not(feature = "cognitive_tracing")))]
pub mod tracing;
pub mod types;
#[cfg(feature = "zig-kernels")]
pub mod zig_kernels;

pub use registry::{
    MemorySpaceError, MemorySpaceRegistry, SpaceDirectories, SpaceHandle, SpaceSummary,
};
pub use types::{MemorySpaceId, MemorySpaceIdError};

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::SystemTime;

// ============================================================================
// System Constants
// ============================================================================

/// Standard embedding dimension for all memory vectors in the system.
///
/// This matches the output dimension of text-embedding-ada-002 and similar
/// embedding models. All embeddings stored in memory, patterns, and queries
/// must use this dimension.
///
/// Changing this value requires rebuilding the entire memory graph.
pub const EMBEDDING_DIM: usize = 768;

/// Cognitive confidence type with human-centered design for probabilistic reasoning.
///
/// This newtype wrapper provides cognitive ergonomics that align with human intuition,
/// prevent systematic biases, and support natural mental models. Values are always
/// constrained to the 0..=1 range with zero-cost abstraction in release builds.
///
/// Based on research by Gigerenzer & Hoffrage (1995), Kahneman & Tversky (conjunction fallacy),
/// and dual-process theory from Kahneman (2011).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Confidence(#[serde(deserialize_with = "validate_confidence")] f32);

impl Confidence {
    // Qualitative categories for natural reasoning
    /// High confidence (0.9) - matches "I'm quite sure" intuition
    pub const HIGH: Self = Self(0.9);
    /// Medium confidence (0.5) - matches "maybe" or "unsure" intuition  
    pub const MEDIUM: Self = Self(0.5);
    /// Low confidence (0.1) - matches "probably not" or "unlikely" intuition
    pub const LOW: Self = Self(0.1);
    /// Maximum confidence (1.0) - complete certainty
    pub const CERTAIN: Self = Self(1.0);
    /// Minimum confidence (0.0) - complete uncertainty/impossibility
    pub const NONE: Self = Self(0.0);

    /// Creates exact confidence value with automatic clamping to 0..=1
    ///
    /// This is the primary constructor that ensures range invariants.
    #[must_use]
    pub const fn exact(value: f32) -> Self {
        // Const-compatible clamping
        let clamped = if value < 0.0 {
            0.0
        } else if value > 1.0 {
            1.0
        } else {
            value
        };
        Self(clamped)
    }

    /// Frequency-based constructor matching human intuition: "3 out of 10 times"
    ///
    /// Humans understand frequencies better than decimals (Gigerenzer & Hoffrage 1995).
    /// This prevents common probability estimation errors.
    #[must_use]
    pub const fn from_successes(successes: u32, total: u32) -> Self {
        if total == 0 {
            return Self::NONE;
        }
        #[allow(clippy::cast_precision_loss)]
        let successes_f32 = successes as f32;
        #[allow(clippy::cast_precision_loss)]
        let total_f32 = total as f32;
        let ratio = (successes_f32 / total_f32).clamp(0.0, 1.0);
        Self::exact(ratio)
    }

    /// Creates confidence from percentage (0-100) with natural language feel
    #[must_use]
    pub const fn from_percent(percent: u8) -> Self {
        let decimal = (percent as f32).min(100.0) / 100.0;
        Self::exact(decimal)
    }

    /// Extract raw f32 value - use sparingly, prefer cognitive methods
    #[must_use]
    pub const fn raw(self) -> f32 {
        self.0
    }

    /// Create from raw value (will be clamped to 0..=1)
    #[must_use]
    pub const fn from_raw(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Create from a probability value (alias for `from_raw`)
    #[must_use]
    pub const fn from_probability(value: f32) -> Self {
        Self::from_raw(value)
    }

    // System 1-friendly operations that feel automatic and natural

    /// Complementary probability (1 - confidence) with safe clamping.
    #[must_use]
    pub const fn complement(self) -> Self {
        Self::from_raw(1.0 - self.0)
    }

    /// Apply exponential decay based on hop count to model activation attenuation.
    #[must_use]
    pub fn decayed(self, decay_rate: f32, hop_count: u16) -> Self {
        let decay_factor = (-decay_rate * f32::from(hop_count)).exp();
        Self::from_raw(self.0 * decay_factor)
    }

    /// Scale the confidence by the provided factor, preserving probabilistic bounds.
    #[must_use]
    pub fn scaled(self, factor: f32) -> Self {
        Self::from_raw(self.0 * factor.clamp(0.0, 1.0))
    }

    /// Fast cognitive check: "Does this seem high confidence?"
    /// Matches natural "seems legitimate" thinking pattern
    #[must_use]
    pub const fn is_high(self) -> bool {
        self.0 >= 0.7
    }

    /// Fast cognitive check: "Does this seem low confidence?"
    #[must_use]
    pub const fn is_low(self) -> bool {
        self.0 <= 0.3
    }

    /// Fast cognitive check: "Does this seem like medium confidence?"
    #[must_use]
    pub const fn is_medium(self) -> bool {
        self.0 > 0.3 && self.0 < 0.7
    }

    /// Natural language: "seems legitimate" - matches cognitive pattern recognition
    #[must_use]
    pub const fn seems_legitimate(self) -> bool {
        self.0 >= 0.6
    }

    /// Natural language: "seems questionable" - matches skepticism threshold  
    #[must_use]
    pub const fn seems_questionable(self) -> bool {
        self.0 <= 0.4
    }

    // Logical operations with bias prevention

    /// Logical AND with conjunction fallacy prevention
    ///
    /// Ensures P(A ∧ B) ≤ min(P(A), P(B)) to prevent the conjunction fallacy
    /// where people incorrectly estimate P(A ∧ B) > P(A).
    #[must_use]
    pub const fn and(self, other: Self) -> Self {
        let result = self.0 * other.0;
        Self(result)
    }

    /// Logical OR with proper probability combination
    ///
    /// Uses P(A ∨ B) = P(A) + P(B) - P(A ∧ B) to prevent overconfidence
    #[must_use]
    pub const fn or(self, other: Self) -> Self {
        let combined = self.0 + other.0 - (self.0 * other.0);
        Self::exact(combined)
    }

    /// Negation: 1 - p, for "not confident" reasoning
    #[must_use]
    pub const fn not(self) -> Self {
        Self(1.0 - self.0)
    }

    /// Weighted combination with explicit reasoning about source reliability
    ///
    /// Helps prevent base rate neglect by making weights explicit
    #[must_use]
    pub const fn combine_weighted(self, other: Self, self_weight: f32, other_weight: f32) -> Self {
        let total_weight = self_weight + other_weight;
        if total_weight == 0.0 {
            return Self::MEDIUM;
        }
        let weighted_avg = (self.0 * self_weight + other.0 * other_weight) / total_weight;
        Self::exact(weighted_avg)
    }

    // Cognitive calibration and bias correction

    /// Applies overconfidence correction based on historical accuracy
    ///
    /// Research shows people are systematically overconfident. This applies
    /// empirically-derived correction factors.
    #[must_use]
    pub const fn calibrate_overconfidence(self) -> Self {
        // Conservative correction based on psychological research
        let corrected = if self.0 > 0.8 {
            self.0 * 0.85 // High confidence gets reduced more
        } else if self.0 > 0.6 {
            self.0 * 0.9 // Medium-high confidence gets modest reduction  
        } else {
            self.0 // Low confidence often accurate
        };
        Self::exact(corrected)
    }

    /// Updates confidence with base rate information to prevent base rate neglect
    ///
    /// Uses Bayesian updating: P(A|B) ∝ P(B|A) * P(A)
    #[must_use]
    pub const fn update_with_base_rate(self, base_rate: Self) -> Self {
        // Simplified Bayesian update assuming this confidence is P(evidence|hypothesis)
        let posterior = (self.0 * base_rate.0)
            / ((self.0 * base_rate.0) + ((1.0 - self.0) * (1.0 - base_rate.0)));
        Self::exact(posterior)
    }
}

/// Confidence budget for tracking expansion costs across async operations.
///
/// Used in query expansion to ensure embedding computation and variant generation
/// never exceed latency budgets. Thread-safe with atomic operations.
///
/// ## Design Principles
///
/// - **Strict Enforcement**: Budget exhaustion returns error, never silently truncates
/// - **Async-Safe**: Uses atomics for tracking consumption across async boundaries
/// - **Observable**: Provides remaining() method for observability
///
/// ## Usage
///
/// ```rust
/// use engram_core::ConfidenceBudget;
///
/// let budget = ConfidenceBudget::new(1.0);
/// assert!(budget.consume(0.3)); // succeeds
/// assert!(!budget.consume(0.8)); // fails, would exceed budget
/// assert!((budget.remaining() - 0.7).abs() < 0.01);
/// ```
#[derive(Debug, Clone)]
pub struct ConfidenceBudget {
    /// Initial budget allocation
    initial: f32,

    /// Consumed amount tracked atomically for async safety
    consumed: Arc<std::sync::atomic::AtomicU32>,
}

impl ConfidenceBudget {
    /// Create a new confidence budget with initial allocation.
    ///
    /// # Arguments
    ///
    /// * `initial` - Initial budget (typically 0.0-1.0)
    ///
    /// # Example
    ///
    /// ```rust
    /// use engram_core::ConfidenceBudget;
    ///
    /// let budget = ConfidenceBudget::new(1.0);
    /// ```
    #[must_use]
    pub fn new(initial: f32) -> Self {
        Self {
            initial: initial.clamp(0.0, f32::MAX),
            consumed: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    /// Attempt to consume budget amount.
    ///
    /// Returns `true` if consumption succeeded (budget sufficient), `false` otherwise.
    /// Uses atomic compare-and-swap for thread-safety.
    ///
    /// # Arguments
    ///
    /// * `amount` - Amount to consume (typically 0.0-1.0)
    ///
    /// # Returns
    ///
    /// `true` if budget was sufficient and consumption succeeded, `false` if insufficient.
    ///
    /// # Example
    ///
    /// ```rust
    /// use engram_core::ConfidenceBudget;
    ///
    /// let budget = ConfidenceBudget::new(1.0);
    /// assert!(budget.consume(0.3)); // succeeds
    /// assert!(budget.consume(0.5)); // succeeds
    /// assert!(!budget.consume(0.3)); // fails, would exceed
    /// ```
    #[must_use]
    pub fn consume(&self, amount: f32) -> bool {
        if amount < 0.0 {
            return false;
        }

        // Convert to u32 for atomic operations (store as fixed-point * 1000)
        let amount_fixed = (amount * 1000.0) as u32;
        let initial_fixed = (self.initial * 1000.0) as u32;

        loop {
            let current = self.consumed.load(std::sync::atomic::Ordering::Acquire);
            let new_value = current.saturating_add(amount_fixed);

            if new_value > initial_fixed {
                return false; // Would exceed budget
            }

            // Try to update atomically
            if self
                .consumed
                .compare_exchange(
                    current,
                    new_value,
                    std::sync::atomic::Ordering::Release,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_ok()
            {
                return true;
            }
            // CAS failed, retry
        }
    }

    /// Get remaining budget.
    ///
    /// # Returns
    ///
    /// Amount of budget remaining (never negative).
    ///
    /// # Example
    ///
    /// ```rust
    /// use engram_core::ConfidenceBudget;
    ///
    /// let budget = ConfidenceBudget::new(1.0);
    /// budget.consume(0.3);
    /// assert!((budget.remaining() - 0.7).abs() < 0.01);
    /// ```
    #[must_use]
    pub fn remaining(&self) -> f32 {
        let consumed_fixed = self.consumed.load(std::sync::atomic::Ordering::Acquire);
        let consumed = (consumed_fixed as f32) / 1000.0;
        (self.initial - consumed).max(0.0)
    }

    /// Get initial budget allocation.
    #[must_use]
    pub const fn initial(&self) -> f32 {
        self.initial
    }

    /// Get consumed amount.
    #[must_use]
    pub fn consumed(&self) -> f32 {
        let consumed_fixed = self.consumed.load(std::sync::atomic::Ordering::Acquire);
        (consumed_fixed as f32) / 1000.0
    }

    /// Check if budget is exhausted.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.remaining() < 0.001 // Within epsilon of zero
    }

    /// Reset the budget to initial value.
    ///
    /// Useful for reusing the same budget tracker across multiple operations.
    pub fn reset(&self) {
        self.consumed.store(0, std::sync::atomic::Ordering::Release);
    }
}

// Custom serde validation to maintain [0,1] invariant
fn validate_confidence<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = f32::deserialize(deserializer)?;
    if !(0.0..=1.0).contains(&value) {
        return Err(serde::de::Error::custom(format!(
            "Confidence value {value} is outside valid range [0,1]"
        )));
    }
    Ok(value)
}

// Public exports
pub use batch::{
    BackpressureStrategy, BatchConfig, BatchEngine, BatchError, BatchOperation,
    BatchOperationResult, BatchOperations, BatchRecallResult, BatchResult, BatchSimilarityResult,
    BatchStoreResult,
};
pub use cue::{
    ContextCueHandler, CueContext, CueDispatcher, CueHandler, EmbeddingCueHandler,
    SemanticCueHandler, TemporalCueHandler,
};
pub use embedding::{
    EmbeddingError, EmbeddingProvenance, EmbeddingProvider, EmbeddingWithProvenance, ModelVersion,
    SentenceTokenizer, TokenizationResult,
};

#[cfg(feature = "multilingual_embeddings")]
pub use embedding::multilingual::MultilingualEncoder;
pub use memory::{
    Cue, CueBuilder, CueType, Episode, EpisodeBuilder, Memory, MemoryBuilder, TemporalPattern,
};

#[cfg(feature = "dual_memory_types")]
pub use memory::{DualMemoryNode, EpisodeId, MemoryNodeType};

pub use query::{
    ConfidenceInterval, Evidence, EvidenceSource, MatchType, ProbabilisticError,
    ProbabilisticQueryResult, ProbabilisticRecall, ProbabilisticResult, UncertaintySource,
    analogy::{AnalogyEngine, AnalogyError, AnalogyPattern, AnalogyRelation},
    expansion::{
        ExpandedQuery, ExpansionError, ExpansionMetadata, QueryExpander, QueryExpanderBuilder,
        QueryVariant, VariantType,
    },
    figurative::{FigurativeInterpreter, IdiomLexicon, InterpretationError},
    lexicon::{AbbreviationLexicon, CompositeLexicon, Lexicon, SynonymLexicon},
};
pub use store::{Activation, MemoryStore, RecallResult, StoreResult};
pub use streaming_health::{StreamingHealthMetrics, StreamingHealthStatus, StreamingHealthTracker};

#[cfg(feature = "psychological_decay")]
pub use decay::{
    BiologicalDecaySystem, DecayError, DecayIntegration, DecayResult, HippocampalDecayFunction,
    IndividualDifferenceProfile, NeocorticalDecayFunction, RemergeProcessor, TwoComponentModel,
};

#[cfg(feature = "pattern_completion")]
pub use completion::{
    ActivationPathway, ActivationTrace, CompletedEpisode, CompletionConfig, CompletionError,
    CompletionResult, ConsolidationEngine, ConsolidationScheduler, EntorhinalContext, GridModule,
    HippocampalCompletion, Hypothesis, MemorySource, MetacognitiveConfidence, PartialEpisode,
    PatternCompleter, PatternReconstructor, SchedulerConfig, SchedulerState, SchedulerStats,
    SourceMap, System2Reasoner,
};

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
    /// Unique identifier for the memory node
    pub id: String,
    /// Content payload of the memory
    pub content: Vec<u8>,
    /// Optional embedding vector for similarity comparisons
    pub embedding: Option<Vec<f32>>,
    /// Current activation level (0.0-1.0)
    pub activation: f64,
    /// Timestamp when the memory was created
    pub created_at: SystemTime,
    /// Timestamp of last access
    pub last_accessed: SystemTime,
    /// Confidence score for this memory
    pub confidence: Confidence,
    #[serde(skip)]
    _state: PhantomData<State>,
}

impl MemoryNode<Unvalidated> {
    /// Creates a new unvalidated memory node
    #[must_use]
    pub fn new_unvalidated(id: String, content: Vec<u8>) -> Self {
        Self {
            id,
            content,
            embedding: None,
            activation: 0.0,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            confidence: Confidence::LOW,
            _state: PhantomData,
        }
    }

    /// Validates the memory node, transitioning to the `Validated` state.
    ///
    /// # Errors
    /// - Returns [`types::CoreError::ValidationError`] when `content` is empty so callers can
    ///   attach meaningful memory data prior to validation.
    ///
    /// # Panics
    /// - Never panics.
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
            confidence: Confidence::MEDIUM, // Increase confidence after validation
            _state: PhantomData,
        })
    }
}

impl MemoryNode<Validated> {
    /// Activates the memory node, transitioning to Active state
    #[must_use]
    pub fn activate(self) -> MemoryNode<Active> {
        MemoryNode {
            id: self.id,
            content: self.content,
            embedding: self.embedding,
            activation: 0.5, // Set initial activation
            created_at: self.created_at,
            last_accessed: SystemTime::now(),
            confidence: Confidence::HIGH, // High confidence for active memories
            _state: PhantomData,
        }
    }
}

impl MemoryNode<Active> {
    /// Consolidates the memory node, transitioning to Consolidated state
    #[must_use]
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
    #[must_use]
    pub fn new(id: String, content: Vec<u8>) -> Self {
        Self {
            id,
            content,
            embedding: None,
            activation: 0.5,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            confidence: Confidence::HIGH,
            _state: PhantomData,
        }
    }
}

/// Memory edge representing relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge weight/strength (0.0-1.0)
    pub weight: f64,
    /// Type of relationship
    pub edge_type: EdgeType,
    /// Confidence in this edge
    pub confidence: Confidence,
}

/// Types of edges in the memory graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Semantic similarity relationship
    Semantic,
    /// Temporal co-occurrence relationship
    Temporal,
    /// Causal relationship
    Causal,
    /// General associative relationship
    Associative,
}

/// Trait for activatable memory elements
pub trait Activatable {
    /// Activate the element with given energy
    fn activate(&mut self, energy: f64);
    /// Apply decay at given rate
    fn decay(&mut self, rate: f64);
    /// Get current activation level
    fn activation_level(&self) -> f64;
}

/// Trait for decayable memory elements
pub trait Decayable {
    /// Apply time-based decay
    fn apply_decay(&mut self, delta_time: f64);
    /// Get current decay rate
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
    use anyhow::{Context, Result, ensure};

    const FLOAT_TOLERANCE: f32 = 1e-6;
    const FLOAT_TOLERANCE_F64: f64 = 1e-9;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < FLOAT_TOLERANCE,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_close_f64(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < FLOAT_TOLERANCE_F64,
            "expected {expected}, got {actual}"
        );
    }

    fn ensure_close(actual: f32, expected: f32) -> Result<()> {
        ensure!(
            (actual - expected).abs() < FLOAT_TOLERANCE,
            "expected {expected}, got {actual}"
        );
        Ok(())
    }

    // Cognitive Confidence Type Tests

    #[test]
    fn test_confidence_range_enforcement() {
        // Values should be automatically clamped to [0,1]
        assert_close(Confidence::exact(-0.5).raw(), 0.0);
        assert_close(Confidence::exact(1.5).raw(), 1.0);
        assert_close(Confidence::exact(0.5).raw(), 0.5);
    }

    #[test]
    fn test_qualitative_categories() {
        // Test that qualitative categories match expected values
        assert_close(Confidence::HIGH.raw(), 0.9);
        assert_close(Confidence::MEDIUM.raw(), 0.5);
        assert_close(Confidence::LOW.raw(), 0.1);
        assert_close(Confidence::CERTAIN.raw(), 1.0);
        assert_close(Confidence::NONE.raw(), 0.0);
    }

    #[test]
    fn test_frequency_based_constructors() {
        // Test frequency-based reasoning: "3 out of 10 times"
        let conf = Confidence::from_successes(3, 10);
        assert_close(conf.raw(), 0.3);

        // Edge cases
        assert_close(Confidence::from_successes(0, 10).raw(), 0.0);
        assert_close(Confidence::from_successes(10, 10).raw(), 1.0);
        assert_close(Confidence::from_successes(5, 0).raw(), 0.0); // Division by zero protection
    }

    #[test]
    fn test_percentage_constructor() {
        assert_close(Confidence::from_percent(0).raw(), 0.0);
        assert_close(Confidence::from_percent(50).raw(), 0.5);
        assert_close(Confidence::from_percent(100).raw(), 1.0);
        assert_close(Confidence::from_percent(150).raw(), 1.0); // Should clamp to 100%
    }

    #[test]
    fn test_system1_cognitive_checks() {
        let high_conf = Confidence::exact(0.8);
        let medium_conf = Confidence::exact(0.5);
        let low_conf = Confidence::exact(0.2);

        // Test cognitive categorization
        assert!(high_conf.is_high());
        assert!(!high_conf.is_low());
        assert!(!high_conf.is_medium());

        assert!(medium_conf.is_medium());
        assert!(!medium_conf.is_high());
        assert!(!medium_conf.is_low());

        assert!(low_conf.is_low());
        assert!(!low_conf.is_high());
        assert!(!low_conf.is_medium());

        // Test natural language patterns
        assert!(high_conf.seems_legitimate());
        assert!(!low_conf.seems_legitimate());
        assert!(low_conf.seems_questionable());
        assert!(!high_conf.seems_questionable());
    }

    #[test]
    fn test_logical_operations_bias_prevention() {
        let conf_a = Confidence::exact(0.8);
        let conf_b = Confidence::exact(0.6);

        // Test conjunction fallacy prevention: P(A ∧ B) ≤ min(P(A), P(B))
        let and_result = conf_a.and(conf_b);
        assert!(and_result.raw() <= conf_a.raw().min(conf_b.raw()));
        assert_close(and_result.raw(), 0.8 * 0.6); // Should be 0.48

        // Test OR operation
        let or_result = conf_a.or(conf_b);
        let expected_or = 0.8f32.mul_add(-0.6, 0.8 + 0.6); // P(A) + P(B) - P(A ∧ B)
        assert!((or_result.raw() - expected_or).abs() < 0.001);

        // Test negation
        let not_a = conf_a.not();
        assert_close(not_a.raw(), 1.0 - 0.8);
    }

    #[test]
    fn test_weighted_combination() {
        let conf_a = Confidence::exact(0.9);
        let conf_b = Confidence::exact(0.3);

        // Equal weights should give average
        let combined = conf_a.combine_weighted(conf_b, 1.0, 1.0);
        assert_close(combined.raw(), 0.6); // (0.9 + 0.3) / 2

        // Higher weight to first confidence
        let weighted = conf_a.combine_weighted(conf_b, 3.0, 1.0);
        let expected = 0.9f32.mul_add(3.0, 0.3 * 1.0) / 4.0; // 0.75
        assert!((weighted.raw() - expected).abs() < 0.001);

        // Zero weights should return MEDIUM
        let zero_weights = conf_a.combine_weighted(conf_b, 0.0, 0.0);
        assert_close(zero_weights.raw(), Confidence::MEDIUM.raw());
    }

    #[test]
    fn test_overconfidence_calibration() {
        // High confidence should be reduced more
        let high_conf = Confidence::exact(0.9);
        let calibrated_high = high_conf.calibrate_overconfidence();
        assert!(calibrated_high.raw() < high_conf.raw());
        assert_close(calibrated_high.raw(), 0.9 * 0.85);

        // Medium confidence should be reduced modestly
        let medium_high_conf = Confidence::exact(0.7);
        let calibrated_medium = medium_high_conf.calibrate_overconfidence();
        assert!(calibrated_medium.raw() < medium_high_conf.raw());
        assert_close(calibrated_medium.raw(), 0.7 * 0.9);

        // Low confidence should remain unchanged
        let low_conf = Confidence::exact(0.4);
        let calibrated_low = low_conf.calibrate_overconfidence();
        assert_close(calibrated_low.raw(), low_conf.raw());
    }

    #[test]
    fn test_base_rate_updating() {
        let evidence_conf = Confidence::exact(0.8); // P(evidence|hypothesis)
        let base_rate = Confidence::exact(0.1); // P(hypothesis) - low base rate

        let updated = evidence_conf.update_with_base_rate(base_rate);

        // Should be significantly lower than evidence due to low base rate
        assert!(updated.raw() < evidence_conf.raw());
        assert!(updated.raw() > base_rate.raw()); // But higher than base rate

        // Manual calculation for verification
        let expected = (0.8 * 0.1) / 0.8f32.mul_add(0.1, (1.0 - 0.8) * (1.0 - 0.1));
        assert!((updated.raw() - expected).abs() < 0.001);
    }

    #[test]
    fn test_zero_cost_abstraction() {
        // This test verifies the type compiles to efficient operations
        let conf = Confidence::exact(0.7);
        let raw_value = conf.raw();

        // Basic operations should compile to simple f32 operations
        let doubled = Confidence::exact(raw_value * 2.0); // Should clamp to 1.0
        assert_close(doubled.raw(), 1.0);

        // Constants should be inlined
        assert_close(Confidence::HIGH.raw(), 0.9);
    }

    #[test]
    fn test_serde_validation() -> Result<()> {
        use serde_json;

        // Valid confidence should serialize/deserialize correctly
        let conf = Confidence::exact(0.7);
        let serialized = serde_json::to_string(&conf).context("serialization failed")?;
        let deserialized: Confidence =
            serde_json::from_str(&serialized).context("deserialization failed")?;
        ensure_close(conf.raw(), deserialized.raw())?;

        // Invalid values should fail deserialization
        let invalid_json = "1.5"; // Outside [0,1] range
        let result: Result<Confidence, _> = serde_json::from_str(invalid_json);
        ensure!(result.is_err(), "invalid confidence should not deserialize");

        let negative_json = "-0.1"; // Negative value
        let result: Result<Confidence, _> = serde_json::from_str(negative_json);
        ensure!(
            result.is_err(),
            "negative confidence should not deserialize"
        );

        Ok(())
    }

    #[test]
    fn test_ordering_and_comparison() {
        let low = Confidence::LOW;
        let medium = Confidence::MEDIUM;
        let high = Confidence::HIGH;

        // Test PartialOrd implementation
        assert!(low < medium);
        assert!(medium < high);
        assert!(high > low);

        // Test PartialEq
        assert_eq!(Confidence::exact(0.5), Confidence::MEDIUM);
        assert_ne!(low, high);
    }

    #[test]
    fn test_memory_node_activation() {
        let mut node = MemoryNode::new("test".to_string(), vec![1, 2, 3]);
        node.activation = 0.3; // Set initial activation for test

        node.activate(0.4);
        assert_close_f64(node.activation_level(), 0.7);

        node.activate(0.5);
        assert_close_f64(node.activation_level(), 1.0); // Capped at 1.0

        node.decay(0.2);
        assert_close_f64(node.activation_level(), 0.8);
    }

    #[test]
    fn test_type_state_transitions() -> Result<()> {
        let unvalidated = MemoryNode::new_unvalidated("test".to_string(), vec![1, 2, 3]);
        let validated = unvalidated
            .validate()
            .context("validation should succeed for non-empty node")?;
        let active = validated.activate();
        let consolidated = active.consolidate();

        // Can't call validate on already validated node (won't compile)
        // consolidated.validate(); // This would be a compile error

        ensure!(
            consolidated.id == "test",
            "expected consolidated id to match"
        );
        Ok(())
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

    #[test]
    fn test_cognitive_error_conversion() {
        use crate::error::CognitiveError;
        use crate::types::CoreError;

        let core_err = CoreError::node_not_found("test_node", vec!["node1".into(), "node2".into()]);
        let cognitive_err: CognitiveError = core_err.into();

        assert!(cognitive_err.summary.contains("test_node"));
        assert_eq!(cognitive_err.similar, vec!["node1", "node2"]);
        assert!(cognitive_err.confidence.raw() >= 0.8);
    }

    #[test]
    fn test_validation_error_with_cognitive_context() -> Result<(), String> {
        let node = MemoryNode::new_unvalidated("test".to_string(), vec![]);
        let result = node.validate();

        let Err(err) = result else {
            return Err("validation should fail for empty node".to_string());
        };
        let msg = err.to_string();

        if !msg.contains("empty") {
            return Err("validation error message should mention empty".to_string());
        }
        if !msg.contains("Non-empty content") {
            return Err("validation message should suggest non-empty content".to_string());
        }
        if !msg.contains("Provide actual memory content") {
            return Err("validation message should guide content provision".to_string());
        }

        Ok(())
    }
}
