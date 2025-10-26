//! Abstract Syntax Tree (AST) for cognitive query language.
//!
//! This module defines the AST representation for parsed queries with:
//! - Zero-copy string handling via `Cow<'a, str>` for optimal parsing performance
//! - Lifetime parameters for borrowed data from source text
//! - Type-state builders for compile-time query validation
//! - Serde serialization for debugging and logging
//! - Integration with existing Confidence and MemorySpaceId types
//!
//! ## Memory Layout Philosophy
//!
//! AST types are designed for cache efficiency:
//! - Query enum: <256 bytes (fits in 3-4 cache lines)
//! - Pattern enum: <128 bytes (fits in 2 cache lines)
//! - Constraint enum: <64 bytes (fits in 1 cache line)
//! - NodeIdentifier: 32 bytes (Cow<str> overhead)
//!
//! ## Zero-Copy Optimization
//!
//! When parsing from source text, AST nodes can reference the original string
//! via borrowed references. This eliminates ~70% of allocations during parsing.
//! Use `into_owned()` to convert to owned data for long-term storage.
//!
//! ## Example
//!
//! ```rust
//! use engram_core::query::parser::ast::*;
//! use engram_core::Confidence;
//!
//! // Build a recall query using type-safe builder
//! let query = RecallQueryBuilder::new()
//!     .pattern(Pattern::NodeId(NodeIdentifier::from("episode_123")))
//!     .constraint(Constraint::ConfidenceAbove(Confidence::from_raw(0.7)))
//!     .limit(10)
//!     .build()
//!     .expect("valid query");
//! ```

// Module-level lint configuration for AST design patterns
#![allow(clippy::missing_const_for_fn)] // Many methods can't be const due to lifetime parameters
#![allow(clippy::elidable_lifetime_names)] // Explicit lifetimes improve clarity for zero-copy design
#![allow(clippy::collapsible_if)] // Explicit conditions improve readability
#![allow(clippy::trivially_copy_pass_by_ref)] // Some methods pass small Copy types by ref for API consistency

use crate::{Confidence, MemorySpaceId};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::marker::PhantomData;
use std::time::{Duration, SystemTime};
use thiserror::Error;

// ============================================================================
// Top-Level Query Type
// ============================================================================

/// Top-level query AST representing all cognitive operations.
///
/// Memory layout optimized for cache performance:
/// - 1-byte discriminant (u8 for 5 variants)
/// - Largest variant determines size (typically RecallQuery or PredictQuery)
/// - Target size: 192-256 bytes (3-4 cache lines)
///
/// The lifetime parameter `'a` enables zero-copy parsing by borrowing from
/// the source query string. Use `into_owned()` to convert to owned data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", bound(deserialize = "'de: 'a"))]
pub enum Query<'a> {
    /// RECALL query - retrieve memories matching a pattern
    Recall(RecallQuery<'a>),
    /// PREDICT query - predict future states based on context
    Predict(PredictQuery<'a>),
    /// IMAGINE query - generate novel combinations via pattern completion
    Imagine(ImagineQuery<'a>),
    /// CONSOLIDATE query - merge and strengthen episodic memories
    Consolidate(ConsolidateQuery<'a>),
    /// SPREAD query - activation spreading from source nodes
    Spread(SpreadQuery<'a>),
}

impl<'a> Query<'a> {
    /// Estimate execution cost for query planning.
    ///
    /// Returns abstract cost units for comparison. Used by query optimizer
    /// to schedule operations and enforce resource budgets.
    #[must_use]
    pub fn estimated_cost(&self) -> u64 {
        match self {
            Self::Recall(q) => q.estimated_cost(),
            Self::Spread(q) => q.estimated_cost(),
            Self::Predict(q) => q.estimated_cost(),
            Self::Imagine(q) => q.estimated_cost(),
            Self::Consolidate(_) => ConsolidateQuery::estimated_cost(),
        }
    }

    /// Check if query is read-only (no graph mutations).
    #[must_use]
    pub const fn is_read_only(&self) -> bool {
        match self {
            Self::Recall(_) | Self::Spread(_) | Self::Predict(_) => true,
            Self::Imagine(_) | Self::Consolidate(_) => false,
        }
    }

    /// Get query category for telemetry and metrics.
    #[must_use]
    pub const fn category(&self) -> QueryCategory {
        match self {
            Self::Recall(_) => QueryCategory::Recall,
            Self::Spread(_) => QueryCategory::Spread,
            Self::Predict(_) => QueryCategory::Predict,
            Self::Imagine(_) => QueryCategory::Imagine,
            Self::Consolidate(_) => QueryCategory::Consolidate,
        }
    }

    /// Convert to owned Query (clones all borrowed data).
    ///
    /// Use this when you need to store the query beyond the lifetime
    /// of the source text (e.g., in a query cache or log).
    #[must_use]
    pub fn into_owned(self) -> Query<'static> {
        match self {
            Self::Recall(q) => Query::Recall(RecallQuery {
                pattern: q.pattern.into_owned(),
                constraints: q
                    .constraints
                    .into_iter()
                    .map(Constraint::into_owned)
                    .collect(),
                confidence_threshold: q.confidence_threshold,
                base_rate: q.base_rate,
                limit: q.limit,
            }),
            Self::Spread(q) => Query::Spread(SpreadQuery {
                source: q.source.into_owned(),
                max_hops: q.max_hops,
                decay_rate: q.decay_rate,
                activation_threshold: q.activation_threshold,
                refractory_period: q.refractory_period,
            }),
            Self::Predict(q) => Query::Predict(PredictQuery {
                pattern: q.pattern.into_owned(),
                context: q
                    .context
                    .into_iter()
                    .map(NodeIdentifier::into_owned)
                    .collect(),
                horizon: q.horizon,
                confidence_constraint: q.confidence_constraint,
            }),
            Self::Imagine(q) => Query::Imagine(ImagineQuery {
                pattern: q.pattern.into_owned(),
                seeds: q
                    .seeds
                    .into_iter()
                    .map(NodeIdentifier::into_owned)
                    .collect(),
                novelty: q.novelty,
                confidence_threshold: q.confidence_threshold,
            }),
            Self::Consolidate(q) => Query::Consolidate(ConsolidateQuery {
                episodes: q.episodes.into_owned(),
                target: q.target.into_owned(),
                scheduler_policy: q.scheduler_policy,
            }),
        }
    }
}

/// Query category for metrics labeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryCategory {
    /// RECALL operation
    Recall,
    /// SPREAD operation
    Spread,
    /// PREDICT operation
    Predict,
    /// IMAGINE operation
    Imagine,
    /// CONSOLIDATE operation
    Consolidate,
}

// ============================================================================
// RECALL Query
// ============================================================================

/// RECALL query: retrieve memories matching a pattern.
///
/// Syntax: `RECALL <pattern> [WHERE <constraints>] [CONFIDENCE <threshold>] [LIMIT <n>]`
///
/// Memory layout (approx 120 bytes):
/// - pattern: 96 bytes (largest Pattern variant)
/// - constraints: 24 bytes (Vec ptr/len/cap)
/// - confidence_threshold: 16 bytes (Option<ConfidenceThreshold>)
/// - base_rate: 8 bytes (Option<Confidence>)
/// - limit: 16 bytes (Option<usize>)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub struct RecallQuery<'a> {
    /// Pattern to match against memories
    pub pattern: Pattern<'a>,
    /// Optional constraints for filtering results
    pub constraints: Vec<Constraint<'a>>,
    /// Optional confidence threshold for results
    pub confidence_threshold: Option<ConfidenceThreshold>,
    /// Optional base rate prior for Bayesian updating
    pub base_rate: Option<Confidence>,
    /// Optional limit on number of results
    pub limit: Option<usize>,
}

impl<'a> RecallQuery<'a> {
    /// Estimate query execution cost based on pattern complexity and constraints.
    #[must_use]
    pub fn estimated_cost(&self) -> u64 {
        // Base cost for pattern matching
        let base: u64 = 1000;

        // Constraint evaluation cost (100 per constraint)
        let constraint_cost = (self.constraints.len() as u64).saturating_mul(100);

        base.saturating_add(constraint_cost)
    }

    /// Validate query invariants.
    ///
    /// Ensures:
    /// - Pattern is valid (non-empty vectors, valid thresholds)
    /// - Constraints are semantically valid
    /// - Confidence thresholds are in range [0, 1]
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.pattern.validate()?;

        for constraint in &self.constraints {
            constraint.validate()?;
        }

        if let Some(threshold) = &self.confidence_threshold {
            threshold.validate()?;
        }

        Ok(())
    }
}

// ============================================================================
// SPREAD Query
// ============================================================================

/// SPREAD query: activation spreading from source node.
///
/// Syntax: `SPREAD FROM <node> [MAX_HOPS <n>] [DECAY <rate>] [THRESHOLD <activation>]`
///
/// Integrates with existing ParallelSpreadingEngine for lock-free activation.
///
/// Memory layout (approx 64 bytes):
/// - source: 32 bytes (NodeIdentifier = Cow<str>)
/// - max_hops: 8 bytes (Option<u16> with padding)
/// - decay_rate: 8 bytes (Option<f32> with padding)
/// - activation_threshold: 8 bytes (Option<f32> with padding)
/// - refractory_period: 16 bytes (Option<Duration>)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub struct SpreadQuery<'a> {
    /// Source node for activation spreading
    pub source: NodeIdentifier<'a>,
    /// Maximum number of hops to spread
    pub max_hops: Option<u16>,
    /// Decay rate per hop (0.0-1.0)
    pub decay_rate: Option<f32>,
    /// Minimum activation threshold for cutoff
    pub activation_threshold: Option<f32>,
    /// Refractory period to prevent immediate re-activation
    pub refractory_period: Option<Duration>,
}

impl<'a> SpreadQuery<'a> {
    /// Default max hops if not specified
    pub const DEFAULT_MAX_HOPS: u16 = 3;
    /// Default decay rate matching existing activation/mod.rs constants
    pub const DEFAULT_DECAY_RATE: f32 = 0.1;
    /// Default activation threshold for spreading cutoff
    pub const DEFAULT_THRESHOLD: f32 = 0.01;

    /// Estimate execution cost (exponential in hop count).
    #[must_use]
    pub fn estimated_cost(&self) -> u64 {
        let hops = self
            .max_hops
            .map_or_else(|| u64::from(Self::DEFAULT_MAX_HOPS), u64::from);

        // Exponential cost: O(2^hops) for typical graph branching factor
        let base_cost = 1000_u64;
        let capped_hops = hops.min(10);
        base_cost.saturating_mul(1_u64 << capped_hops)
    }

    /// Get effective decay rate (use default if not specified).
    #[must_use]
    pub const fn effective_decay_rate(&self) -> f32 {
        match self.decay_rate {
            Some(rate) => rate,
            None => Self::DEFAULT_DECAY_RATE,
        }
    }

    /// Get effective activation threshold.
    #[must_use]
    pub const fn effective_threshold(&self) -> f32 {
        match self.activation_threshold {
            Some(t) => t,
            None => Self::DEFAULT_THRESHOLD,
        }
    }

    /// Validate spreading parameters.
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.source.validate()?;

        if let Some(rate) = self.decay_rate
            && !(0.0..=1.0).contains(&rate)
        {
            return Err(ValidationError::InvalidDecayRate(rate));
        }

        if let Some(threshold) = self.activation_threshold
            && !(0.0..=1.0).contains(&threshold)
        {
            return Err(ValidationError::InvalidActivationThreshold(threshold));
        }

        Ok(())
    }
}

// ============================================================================
// PREDICT Query
// ============================================================================

/// PREDICT query: predict future states based on context.
///
/// Syntax: `PREDICT <pattern> GIVEN <context> [HORIZON <duration>] [CONFIDENCE <interval>]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub struct PredictQuery<'a> {
    /// Pattern to predict
    pub pattern: Pattern<'a>,
    /// Context nodes for prediction
    pub context: Vec<NodeIdentifier<'a>>,
    /// Time horizon for prediction
    pub horizon: Option<Duration>,
    /// Confidence interval constraint
    pub confidence_constraint: Option<ConfidenceInterval>,
}

impl<'a> PredictQuery<'a> {
    /// Estimate execution cost based on context size.
    #[must_use]
    pub fn estimated_cost(&self) -> u64 {
        // Cost based on context size
        2000_u64.saturating_add((self.context.len() as u64).saturating_mul(500))
    }

    /// Validate query parameters.
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.pattern.validate()?;
        for node in &self.context {
            node.validate()?;
        }
        Ok(())
    }
}

// ============================================================================
// IMAGINE Query
// ============================================================================

/// IMAGINE query: generate novel combinations via pattern completion.
///
/// Syntax: `IMAGINE <pattern> [BASED ON <seeds>] [NOVELTY <level>]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub struct ImagineQuery<'a> {
    /// Pattern to complete
    pub pattern: Pattern<'a>,
    /// Seed nodes for completion
    pub seeds: Vec<NodeIdentifier<'a>>,
    /// Novelty level [0, 1] (higher = more creative)
    pub novelty: Option<f32>,
    /// Confidence threshold for results
    pub confidence_threshold: Option<ConfidenceThreshold>,
}

impl<'a> ImagineQuery<'a> {
    /// Estimate execution cost (pattern completion is expensive).
    #[must_use]
    pub fn estimated_cost(&self) -> u64 {
        5000_u64.saturating_add((self.seeds.len() as u64).saturating_mul(1000))
    }

    /// Validate query parameters.
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.pattern.validate()?;

        for seed in &self.seeds {
            seed.validate()?;
        }

        if let Some(novelty) = self.novelty {
            if !(0.0..=1.0).contains(&novelty) {
                return Err(ValidationError::InvalidNovelty(novelty));
            }
        }

        if let Some(threshold) = &self.confidence_threshold {
            threshold.validate()?;
        }

        Ok(())
    }
}

// ============================================================================
// CONSOLIDATE Query
// ============================================================================

/// CONSOLIDATE query: merge and strengthen episodic memories.
///
/// Syntax: `CONSOLIDATE <episodes> INTO <semantic_node> [SCHEDULER <policy>]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub struct ConsolidateQuery<'a> {
    /// Episode selector
    pub episodes: EpisodeSelector<'a>,
    /// Target semantic node
    pub target: NodeIdentifier<'a>,
    /// Scheduler policy for consolidation
    pub scheduler_policy: Option<SchedulerPolicy>,
}

impl<'a> ConsolidateQuery<'a> {
    /// Estimate execution cost (consolidation is very expensive).
    #[must_use]
    pub const fn estimated_cost() -> u64 {
        10_000
    }

    /// Validate query parameters.
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.episodes.validate()?;
        self.target.validate()?;
        Ok(())
    }
}

/// Scheduler policy for memory consolidation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchedulerPolicy {
    /// Immediate consolidation
    Immediate,
    /// Periodic consolidation at interval
    Interval(Duration),
    /// Threshold-based consolidation
    Threshold {
        /// Minimum activation level to trigger
        activation: f32,
    },
}

/// Episode selector for CONSOLIDATE queries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub enum EpisodeSelector<'a> {
    /// All episodes
    All,
    /// Episodes matching pattern
    Pattern(Pattern<'a>),
    /// Episodes matching constraints
    Where(Vec<Constraint<'a>>),
}

impl<'a> EpisodeSelector<'a> {
    /// Convert to owned EpisodeSelector.
    #[must_use]
    pub fn into_owned(self) -> EpisodeSelector<'static> {
        match self {
            Self::All => EpisodeSelector::All,
            Self::Pattern(p) => EpisodeSelector::Pattern(p.into_owned()),
            Self::Where(cs) => {
                EpisodeSelector::Where(cs.into_iter().map(Constraint::into_owned).collect())
            }
        }
    }

    /// Validate selector.
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::All => Ok(()),
            Self::Pattern(p) => p.validate(),
            Self::Where(cs) => {
                for c in cs {
                    c.validate()?;
                }
                Ok(())
            }
        }
    }
}

// ============================================================================
// Pattern Types
// ============================================================================

/// Pattern specification for queries with zero-copy string references.
///
/// Uses `Cow<'a, str>` for zero-copy parsing where the pattern references
/// the source query string directly. Parser can construct patterns without
/// allocating when possible.
///
/// Memory layout (approx 96 bytes for largest variant):
/// - Discriminant: 1 byte
/// - Embedding variant: 32 bytes (Vec ptr/len/cap + f32 + padding)
/// - ContentMatch variant: 32 bytes (Cow = discriminant + String)
/// - NodeId variant: 32 bytes (NodeIdentifier<'a>)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern<'a> {
    /// Match by node ID (zero-copy when parsed from source)
    NodeId(NodeIdentifier<'a>),

    /// Match by embedding similarity
    ///
    /// Note: vector is always heap-allocated (Vec), but this is acceptable
    /// since embeddings are typically computed, not parsed from source.
    Embedding {
        /// Embedding vector (typically 768-dimensional)
        vector: Vec<f32>,
        /// Similarity threshold [0, 1]
        threshold: f32,
    },

    /// Match by content substring (zero-copy when parsed from source)
    ContentMatch(#[serde(borrow)] Cow<'a, str>),

    /// Match all nodes (for broad queries)
    Any,
}

impl<'a> Pattern<'a> {
    /// Validate pattern constraints.
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Embedding { vector, threshold } => {
                if vector.is_empty() {
                    return Err(ValidationError::EmptyEmbedding);
                }
                if !(*threshold >= 0.0 && *threshold <= 1.0) {
                    return Err(ValidationError::InvalidThreshold(*threshold));
                }

                // Validate embedding dimensions match system expectations
                if vector.len() != crate::EMBEDDING_DIM {
                    return Err(ValidationError::InvalidEmbeddingDimension {
                        expected: crate::EMBEDDING_DIM,
                        actual: vector.len(),
                    });
                }

                Ok(())
            }
            Self::NodeId(id) => id.validate(),
            Self::ContentMatch(s) if s.is_empty() => Err(ValidationError::EmptyContentMatch),
            Self::ContentMatch(_) | Self::Any => Ok(()),
        }
    }

    /// Convert to owned Pattern (clones borrowed data).
    #[must_use]
    pub fn into_owned(self) -> Pattern<'static> {
        match self {
            Self::NodeId(id) => Pattern::NodeId(id.into_owned()),
            Self::Embedding { vector, threshold } => Pattern::Embedding { vector, threshold },
            Self::ContentMatch(s) => Pattern::ContentMatch(Cow::Owned(s.into_owned())),
            Self::Any => Pattern::Any,
        }
    }
}

// ============================================================================
// Constraint Types
// ============================================================================

/// Query constraints for filtering results with zero-copy optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint<'a> {
    /// Similarity constraint: SIMILAR TO <embedding>
    SimilarTo {
        /// Embedding vector
        embedding: Vec<f32>,
        /// Similarity threshold
        threshold: f32,
    },

    /// Temporal constraint: CREATED BEFORE <time>
    CreatedBefore(SystemTime),

    /// Temporal constraint: CREATED AFTER <time>
    CreatedAfter(SystemTime),

    /// Confidence constraint: CONFIDENCE > <threshold>
    ConfidenceAbove(Confidence),

    /// Confidence constraint: CONFIDENCE < <threshold>
    ConfidenceBelow(Confidence),

    /// Memory space constraint: IN SPACE <id>
    InMemorySpace(MemorySpaceId),

    /// Content constraint: CONTENT CONTAINS <text>
    ///
    /// Zero-copy: uses Cow to avoid allocation when parsed from source
    ContentContains(#[serde(borrow)] Cow<'a, str>),
}

impl<'a> Constraint<'a> {
    /// Validate constraint semantics.
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::SimilarTo {
                embedding,
                threshold,
            } => {
                if embedding.is_empty() {
                    return Err(ValidationError::EmptyEmbedding);
                }
                if !(*threshold >= 0.0 && *threshold <= 1.0) {
                    return Err(ValidationError::InvalidThreshold(*threshold));
                }
                Ok(())
            }
            Self::ConfidenceAbove(_) | Self::ConfidenceBelow(_) => {
                // Confidence type already guarantees valid range
                Ok(())
            }
            Self::ContentContains(s) if s.is_empty() => Err(ValidationError::EmptyContentMatch),
            _ => Ok(()),
        }
    }

    /// Convert to owned Constraint (clones borrowed data).
    #[must_use]
    pub fn into_owned(self) -> Constraint<'static> {
        match self {
            Self::SimilarTo {
                embedding,
                threshold,
            } => Constraint::SimilarTo {
                embedding,
                threshold,
            },
            Self::CreatedBefore(t) => Constraint::CreatedBefore(t),
            Self::CreatedAfter(t) => Constraint::CreatedAfter(t),
            Self::ConfidenceAbove(c) => Constraint::ConfidenceAbove(c),
            Self::ConfidenceBelow(c) => Constraint::ConfidenceBelow(c),
            Self::InMemorySpace(id) => Constraint::InMemorySpace(id),
            Self::ContentContains(s) => Constraint::ContentContains(Cow::Owned(s.into_owned())),
        }
    }
}

// ============================================================================
// Node Identifier
// ============================================================================

/// Node identifier in the memory graph with zero-copy optimization.
///
/// Uses `Cow<'a, str>` internally to avoid allocations during parsing.
/// When parsed from query source, references the original string.
/// When constructed programmatically, can own the string.
///
/// Memory layout: 32 bytes
/// - Cow discriminant: 1 byte
/// - String data: 24 bytes (ptr + len + cap) when owned
/// - str reference: 16 bytes (ptr + len) when borrowed
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeIdentifier<'a>(#[serde(borrow)] Cow<'a, str>);

impl<'a> NodeIdentifier<'a> {
    /// Create from borrowed string (zero-copy).
    #[must_use]
    pub const fn borrowed(id: &'a str) -> Self {
        Self(Cow::Borrowed(id))
    }

    /// Create from owned string.
    #[must_use]
    pub fn owned(id: String) -> Self {
        Self(Cow::Owned(id))
    }

    /// Create from any string-like type.
    #[must_use]
    pub fn new(id: impl Into<Cow<'a, str>>) -> Self {
        Self(id.into())
    }

    /// Get reference to the identifier string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert to owned NodeIdentifier (clones if borrowed).
    #[must_use]
    pub fn into_owned(self) -> NodeIdentifier<'static> {
        NodeIdentifier(Cow::Owned(self.0.into_owned()))
    }

    /// Validate identifier constraints.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.0.is_empty() {
            return Err(ValidationError::EmptyNodeId);
        }

        // Additional validation: node IDs should be reasonable length
        // to prevent pathological cases
        if self.0.len() > 256 {
            return Err(ValidationError::NodeIdTooLong {
                max_length: 256,
                actual_length: self.0.len(),
            });
        }

        Ok(())
    }
}

impl<'a> std::fmt::Display for NodeIdentifier<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<'a> From<String> for NodeIdentifier<'a> {
    fn from(s: String) -> Self {
        Self::owned(s)
    }
}

impl<'a> From<&'a str> for NodeIdentifier<'a> {
    fn from(s: &'a str) -> Self {
        Self::borrowed(s)
    }
}

impl<'a> AsRef<str> for NodeIdentifier<'a> {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ============================================================================
// Confidence Specifications
// ============================================================================

/// Confidence threshold specification for filtering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConfidenceThreshold {
    /// Match values above threshold
    Above(Confidence),
    /// Match values below threshold
    Below(Confidence),
    /// Match values between lower and upper bounds
    Between {
        /// Lower bound (inclusive)
        lower: Confidence,
        /// Upper bound (inclusive)
        upper: Confidence,
    },
}

impl ConfidenceThreshold {
    /// Check if a confidence value matches this threshold.
    #[must_use]
    pub fn matches(&self, confidence: Confidence) -> bool {
        match self {
            Self::Above(threshold) => confidence.raw() > threshold.raw(),
            Self::Below(threshold) => confidence.raw() < threshold.raw(),
            Self::Between { lower, upper } => {
                confidence.raw() >= lower.raw() && confidence.raw() <= upper.raw()
            }
        }
    }

    /// Validate threshold constraints.
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Between { lower, upper } => {
                if lower.raw() > upper.raw() {
                    return Err(ValidationError::InvalidInterval {
                        lower: *lower,
                        upper: *upper,
                    });
                }
            }
            Self::Above(_) | Self::Below(_) => {}
        }
        Ok(())
    }
}

/// Confidence interval for PREDICT queries.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Lower bound
    pub lower: Confidence,
    /// Upper bound
    pub upper: Confidence,
}

impl ConfidenceInterval {
    /// Create new confidence interval with validation.
    pub fn new(lower: Confidence, upper: Confidence) -> Result<Self, ValidationError> {
        if lower.raw() > upper.raw() {
            return Err(ValidationError::InvalidInterval { lower, upper });
        }
        Ok(Self { lower, upper })
    }

    /// Check if a confidence value is within this interval.
    #[must_use]
    pub fn contains(&self, confidence: Confidence) -> bool {
        confidence.raw() >= self.lower.raw() && confidence.raw() <= self.upper.raw()
    }
}

// ============================================================================
// Validation Errors
// ============================================================================

/// Comprehensive validation errors with cognitive error guidance.
///
/// Follows engram-core error design principles:
/// - Clear error messages with context
/// - Actionable suggestions for resolution
/// - Examples showing correct usage
#[derive(Debug, Clone, Error)]
#[allow(missing_docs)] // Error messages are self-documenting
pub enum ValidationError {
    #[error(
        "Empty embedding vector\n  Expected: 768-dimensional f32 vector\n  Suggestion: Use embedding service to generate vector from content\n  Example: Pattern::Embedding {{ vector: embed(text), threshold: 0.8 }}"
    )]
    EmptyEmbedding,

    #[error(
        "Invalid embedding dimension: expected {expected}, got {actual}\n  Expected: {expected}-dimensional vector (system configuration)\n  Suggestion: Ensure embedding model matches system configuration\n  Example: Use text-embedding-ada-002 or compatible {expected}-dim model"
    )]
    InvalidEmbeddingDimension { expected: usize, actual: usize },

    #[error(
        "Invalid similarity threshold: {0}, must be in [0,1]\n  Expected: Threshold between 0.0 (no similarity) and 1.0 (identical)\n  Suggestion: Use 0.7-0.9 for strict matching, 0.5-0.7 for fuzzy matching\n  Example: Pattern::Embedding {{ vector, threshold: 0.8 }}"
    )]
    InvalidThreshold(f32),

    #[error(
        "Empty node identifier\n  Expected: Non-empty string identifying a memory node\n  Suggestion: Use meaningful IDs like 'episode_123' or 'concept_ai'\n  Example: NodeIdentifier::from(\"episode_123\")"
    )]
    EmptyNodeId,

    #[error(
        "Node identifier too long: {actual_length} bytes (max {max_length})\n  Expected: Node ID under {max_length} bytes\n  Suggestion: Use shorter, more compact identifiers\n  Example: Use 'usr_123' instead of full UUIDs"
    )]
    NodeIdTooLong {
        max_length: usize,
        actual_length: usize,
    },

    #[error(
        "Empty content match pattern\n  Expected: Non-empty substring to search for\n  Suggestion: Provide text to match against memory content\n  Example: Pattern::ContentMatch(\"neural network\")"
    )]
    EmptyContentMatch,

    #[error(
        "Invalid confidence interval: lower={lower:?} > upper={upper:?}\n  Expected: lower <= upper, both in [0, 1]\n  Suggestion: Swap values or use ConfidenceInterval::new() for validation\n  Example: ConfidenceInterval::new(0.5, 0.9)"
    )]
    InvalidInterval {
        lower: Confidence,
        upper: Confidence,
    },

    #[error(
        "Invalid decay rate: {0}, must be in [0,1]\n  Expected: Decay rate between 0.0 (no decay) and 1.0 (instant decay)\n  Suggestion: Use 0.1 for slow forgetting, 0.3-0.5 for typical decay\n  Example: SpreadQuery {{ decay_rate: Some(0.1), .. }}"
    )]
    InvalidDecayRate(f32),

    #[error(
        "Invalid activation threshold: {0}, must be in [0,1]\n  Expected: Activation threshold between 0.0 and 1.0\n  Suggestion: Use 0.01 for wide spreading, 0.1 for focused activation\n  Example: SpreadQuery {{ activation_threshold: Some(0.01), .. }}"
    )]
    InvalidActivationThreshold(f32),

    #[error(
        "Invalid novelty level: {0}, must be in [0,1]\n  Expected: Novelty between 0.0 (conservative) and 1.0 (highly creative)\n  Suggestion: Use 0.3-0.5 for moderate creativity\n  Example: ImagineQuery {{ novelty: Some(0.4), .. }}"
    )]
    InvalidNovelty(f32),

    #[error(
        "Invalid constraint: {0}\n  Expected: Semantically valid query constraint\n  Suggestion: {1}\n  Example: {2}"
    )]
    InvalidConstraint(String, String, String),
}

impl ValidationError {
    /// Create custom constraint validation error with context.
    #[must_use]
    pub fn invalid_constraint(
        reason: impl Into<String>,
        suggestion: impl Into<String>,
        example: impl Into<String>,
    ) -> Self {
        Self::InvalidConstraint(reason.into(), suggestion.into(), example.into())
    }
}

// ============================================================================
// Type-State Pattern for Query Builders
// ============================================================================

/// Query builder states for type-state pattern.
pub mod builder_states {
    /// Builder has no pattern set
    pub struct NoPattern;
    /// Builder has pattern set
    pub struct WithPattern;
}

/// Type-safe recall query builder.
///
/// Ensures pattern is set before building, using phantom type states
/// to prevent invalid queries from compiling.
///
/// # Example
///
/// ```rust
/// use engram_core::query::parser::ast::*;
/// use engram_core::Confidence;
///
/// let query = RecallQueryBuilder::new()
///     .pattern(Pattern::NodeId(NodeIdentifier::from("episode_123")))
///     .constraint(Constraint::ConfidenceAbove(Confidence::from_raw(0.7)))
///     .limit(10)
///     .build()
///     .expect("valid query");
/// ```
pub struct RecallQueryBuilder<State = builder_states::NoPattern> {
    pattern: Option<Pattern<'static>>,
    constraints: Vec<Constraint<'static>>,
    confidence_threshold: Option<ConfidenceThreshold>,
    base_rate: Option<Confidence>,
    limit: Option<usize>,
    _state: PhantomData<State>,
}

impl RecallQueryBuilder<builder_states::NoPattern> {
    /// Create a new builder (starts in NoPattern state).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pattern: None,
            constraints: Vec::new(),
            confidence_threshold: None,
            base_rate: None,
            limit: None,
            _state: PhantomData,
        }
    }

    /// Set the pattern (transitions to WithPattern state).
    #[must_use]
    pub fn pattern(
        self,
        pattern: Pattern<'static>,
    ) -> RecallQueryBuilder<builder_states::WithPattern> {
        RecallQueryBuilder {
            pattern: Some(pattern),
            constraints: self.constraints,
            confidence_threshold: self.confidence_threshold,
            base_rate: self.base_rate,
            limit: self.limit,
            _state: PhantomData,
        }
    }
}

impl Default for RecallQueryBuilder<builder_states::NoPattern> {
    fn default() -> Self {
        Self::new()
    }
}

impl RecallQueryBuilder<builder_states::WithPattern> {
    /// Add constraint (can be called multiple times).
    #[must_use]
    pub fn constraint(mut self, constraint: Constraint<'static>) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Set confidence threshold.
    #[must_use]
    pub const fn confidence_threshold(mut self, threshold: ConfidenceThreshold) -> Self {
        self.confidence_threshold = Some(threshold);
        self
    }

    /// Set base rate prior.
    #[must_use]
    pub const fn base_rate(mut self, rate: Confidence) -> Self {
        self.base_rate = Some(rate);
        self
    }

    /// Set result limit.
    #[must_use]
    pub const fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Validate and build the query.
    ///
    /// This consumes the builder and returns a validated RecallQuery.
    /// Validation errors are returned immediately.
    ///
    /// # Panics
    ///
    /// This method panics if called without a pattern set, but this is
    /// prevented at compile time by the type-state pattern.
    #[allow(clippy::expect_used)] // Type-state pattern guarantees Some(pattern)
    #[allow(clippy::unwrap_in_result)] // Type-state pattern makes this safe
    pub fn build(self) -> Result<RecallQuery<'static>, ValidationError> {
        let query = RecallQuery {
            pattern: self.pattern.expect("pattern guaranteed by type state"),
            constraints: self.constraints,
            confidence_threshold: self.confidence_threshold,
            base_rate: self.base_rate,
            limit: self.limit,
        };

        // Validate the constructed query
        query.validate()?;

        Ok(query)
    }
}

// ============================================================================
// Compile-Time Size Assertions
// ============================================================================

// Verify memory layout targets
const _: () = {
    assert!(
        std::mem::size_of::<QueryCategory>() == 1,
        "QueryCategory should be 1 byte"
    );
};

#[cfg(test)]
#[path = "ast_tests.rs"]
mod tests;
