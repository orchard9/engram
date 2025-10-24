# Task 002: AST Definition

**Status**: Pending
**Duration**: 1 day
**Dependencies**: Task 001 (Parser Infrastructure)
**Owner**: TBD

---

## Objective

Define AST types that map 1:1 to cognitive operations. Types must be compact (<256 bytes per Query), support serde for debugging, and integrate seamlessly with existing Confidence and MemorySpaceId types.

**Key Performance Requirements**:
- Zero-cost abstractions: enum layouts must have minimal discriminant overhead
- Cache-friendly: hot path fields packed within 64-byte cache lines
- Copy-on-parse: zero-copy references to source text where beneficial
- Type-safe: compile-time prevention of invalid query construction

---

## Memory Layout Analysis

### Discriminant Optimization

Rust enum discriminants default to the smallest type that can represent all variants.
For Query enum with 5 variants, discriminant = 1 byte (u8).

Target sizes (on 64-bit systems):
- Query enum: 192-256 bytes (fits in 3-4 cache lines)
- Pattern enum: 96-128 bytes (fits in 2 cache lines)
- Constraint enum: 48-64 bytes (fits in 1 cache line)
- NodeIdentifier: 24 bytes (String = ptr + len + cap)

### Cache Line Considerations

Modern CPUs use 64-byte cache lines. Optimal AST traversal requires:
1. Discriminant + frequently accessed fields in first cache line
2. Vec/String heap data accessed via pointer (indirection acceptable)
3. Option<T> adds discriminant (1 byte) + alignment padding

---

## Technical Specification

### 1. Core Query Type

```rust
// File: engram-core/src/query/parser/ast.rs

use crate::{Confidence, MemorySpaceId};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::time::{SystemTime, Duration};

/// Top-level query AST - represents all cognitive operations
///
/// Memory layout optimized for cache performance:
/// - 1-byte discriminant (u8 for 5 variants)
/// - Largest variant determines size (typically RecallQuery or PredictQuery)
/// - Target size: 192-256 bytes (3-4 cache lines)
///
/// The Box<T> indirection for large variants reduces enum size if needed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Query {
    Recall(RecallQuery),
    Predict(PredictQuery),
    Imagine(ImagineQuery),
    Consolidate(ConsolidateQuery),
    Spread(SpreadQuery),
}

impl Query {
    /// Estimate execution cost (for future query planning)
    ///
    /// Returns abstract cost units for comparison. Used by query optimizer
    /// to schedule operations and enforce resource budgets.
    #[must_use]
    pub const fn estimated_cost(&self) -> u64 {
        match self {
            Self::Recall(q) => q.estimated_cost(),
            Self::Spread(q) => q.estimated_cost(),
            Self::Predict(q) => q.estimated_cost(),
            Self::Imagine(q) => q.estimated_cost(),
            Self::Consolidate(q) => q.estimated_cost(),
        }
    }

    /// Check if query is read-only (no graph mutations)
    #[must_use]
    pub const fn is_read_only(&self) -> bool {
        match self {
            Self::Recall(_) | Self::Spread(_) | Self::Predict(_) => true,
            Self::Imagine(_) | Self::Consolidate(_) => false,
        }
    }

    /// Get query category for telemetry/metrics
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
}

/// Query category for metrics labeling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryCategory {
    Recall,
    Spread,
    Predict,
    Imagine,
    Consolidate,
}
```

### 2. RECALL Query

```rust
/// RECALL <pattern> [WHERE <constraints>] [CONFIDENCE <threshold>]
///
/// Memory layout (approx 120 bytes):
/// - pattern: 96 bytes (largest Pattern variant)
/// - constraints: 24 bytes (Vec ptr/len/cap)
/// - confidence_threshold: 16 bytes (Option<ConfidenceThreshold>)
/// - base_rate: 8 bytes (Option<Confidence>)
/// - limit: 16 bytes (Option<usize>)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallQuery {
    pub pattern: Pattern,
    pub constraints: Vec<Constraint>,
    pub confidence_threshold: Option<ConfidenceThreshold>,
    pub base_rate: Option<Confidence>,
    pub limit: Option<usize>,
}

impl RecallQuery {
    /// Estimate query execution cost based on pattern complexity and constraints
    #[must_use]
    pub const fn estimated_cost(&self) -> u64 {
        // Base cost for pattern matching
        let base = 1000;

        // Constraint evaluation cost (dynamic at runtime, estimate conservatively)
        let constraint_cost = 100; // per constraint, actual count checked at runtime

        base + constraint_cost
    }

    /// Validate query invariants
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
```

### 3. SPREAD Query

```rust
/// SPREAD FROM <node> [MAX_HOPS <n>] [DECAY <rate>] [THRESHOLD <activation>]
///
/// Integrates with existing ParallelSpreadingEngine for lock-free activation.
///
/// Memory layout (approx 64 bytes):
/// - source: 24 bytes (NodeIdentifier = String)
/// - max_hops: 8 bytes (Option<u16> with padding)
/// - decay_rate: 8 bytes (Option<f32> with padding)
/// - activation_threshold: 8 bytes (Option<f32> with padding)
/// - refractory_period: 16 bytes (Option<Duration>)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpreadQuery {
    pub source: NodeIdentifier,
    pub max_hops: Option<u16>,
    pub decay_rate: Option<f32>,
    pub activation_threshold: Option<f32>,
    pub refractory_period: Option<Duration>,
}

impl SpreadQuery {
    /// Default max hops if not specified
    pub const DEFAULT_MAX_HOPS: u16 = 3;
    /// Default decay rate matching existing activation/mod.rs constants
    pub const DEFAULT_DECAY_RATE: f32 = 0.1;
    /// Default activation threshold for spreading cutoff
    pub const DEFAULT_THRESHOLD: f32 = 0.01;

    /// Estimate execution cost (exponential in hop count)
    #[must_use]
    pub const fn estimated_cost(&self) -> u64 {
        let hops = match self.max_hops {
            Some(h) => h as u64,
            None => Self::DEFAULT_MAX_HOPS as u64,
        };

        // Exponential cost: O(2^hops) for typical graph branching factor
        let base_cost = 1000_u64;
        base_cost.saturating_mul(1_u64 << hops.min(10)) // Cap at 2^10 to prevent overflow
    }

    /// Get effective decay rate (use default if not specified)
    #[must_use]
    pub const fn effective_decay_rate(&self) -> f32 {
        match self.decay_rate {
            Some(rate) => rate,
            None => Self::DEFAULT_DECAY_RATE,
        }
    }

    /// Get effective activation threshold
    #[must_use]
    pub const fn effective_threshold(&self) -> f32 {
        match self.activation_threshold {
            Some(t) => t,
            None => Self::DEFAULT_THRESHOLD,
        }
    }

    /// Validate spreading parameters
    pub fn validate(&self) -> Result<(), ValidationError> {
        self.source.validate()?;

        if let Some(rate) = self.decay_rate {
            if !(0.0..=1.0).contains(&rate) {
                return Err(ValidationError::InvalidDecayRate(rate));
            }
        }

        if let Some(threshold) = self.activation_threshold {
            if !(0.0..=1.0).contains(&threshold) {
                return Err(ValidationError::InvalidActivationThreshold(threshold));
            }
        }

        Ok(())
    }
}
```

### 4. PREDICT Query

```rust
/// PREDICT <pattern> GIVEN <context> [HORIZON <duration>] [CONFIDENCE <interval>]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictQuery {
    pub pattern: Pattern,
    pub context: Vec<NodeIdentifier>,  // Context nodes
    pub horizon: Option<Duration>,     // Time horizon for prediction
    pub confidence_constraint: Option<ConfidenceInterval>,
}

impl PredictQuery {
    pub fn estimated_cost(&self) -> u64 {
        // Cost based on context size
        2000 + (self.context.len() as u64 * 500)
    }
}
```

### 5. IMAGINE Query

```rust
/// IMAGINE <pattern> [BASED ON <seeds>] [NOVELTY <level>]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImagineQuery {
    pub pattern: Pattern,
    pub seeds: Vec<NodeIdentifier>,  // Seed nodes for completion
    pub novelty: Option<f32>,        // Novelty level [0,1]
    pub confidence_threshold: Option<ConfidenceThreshold>,
}

impl ImagineQuery {
    pub fn estimated_cost(&self) -> u64 {
        // Pattern completion is expensive
        5000 + (self.seeds.len() as u64 * 1000)
    }
}
```

### 6. CONSOLIDATE Query

```rust
/// CONSOLIDATE <episodes> INTO <semantic_node> [SCHEDULER <policy>]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsolidateQuery {
    pub episodes: EpisodeSelector,
    pub target: NodeIdentifier,
    pub scheduler_policy: Option<SchedulerPolicy>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchedulerPolicy {
    Immediate,
    Interval(Duration),
    Threshold { activation: f32 },
}

impl ConsolidateQuery {
    pub fn estimated_cost(&self) -> u64 {
        // Consolidation is very expensive
        10000
    }
}
```

### 7. Pattern Types (Zero-Copy Optimized)

```rust
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
        #[serde(borrow)]
        vector: Vec<f32>,
        threshold: f32,
    },

    /// Match by content substring (zero-copy when parsed from source)
    ContentMatch(Cow<'a, str>),

    /// Match all (for broad queries)
    Any,
}

impl<'a> Pattern<'a> {
    /// Validate pattern constraints
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
                // (768 for current implementation)
                if vector.len() != 768 {
                    return Err(ValidationError::InvalidEmbeddingDimension {
                        expected: 768,
                        actual: vector.len(),
                    });
                }

                Ok(())
            }
            Self::NodeId(id) => id.validate(),
            Self::ContentMatch(s) if s.is_empty() => {
                Err(ValidationError::EmptyContentMatch)
            }
            Self::ContentMatch(_) | Self::Any => Ok(()),
        }
    }

    /// Convert to owned Pattern (clones borrowed data)
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
```

### 8. Constraint Types

```rust
/// Query constraints for filtering results
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    /// Similarity constraint: SIMILAR TO <embedding>
    SimilarTo {
        embedding: Vec<f32>,
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
    ContentContains(String),
}
```

### 9. Node Identifier (Zero-Copy)

```rust
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
    /// Create from borrowed string (zero-copy)
    #[must_use]
    pub const fn borrowed(id: &'a str) -> Self {
        Self(Cow::Borrowed(id))
    }

    /// Create from owned string
    #[must_use]
    pub fn owned(id: String) -> Self {
        Self(Cow::Owned(id))
    }

    /// Create from any string-like type
    pub fn new(id: impl Into<Cow<'a, str>>) -> Self {
        Self(id.into())
    }

    /// Get reference to the identifier string
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert to owned NodeIdentifier (clones if borrowed)
    #[must_use]
    pub fn into_owned(self) -> NodeIdentifier<'static> {
        NodeIdentifier(Cow::Owned(self.0.into_owned()))
    }

    /// Validate identifier constraints
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
```

### 10. Confidence Specifications

```rust
/// Confidence threshold specification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConfidenceThreshold {
    Above(Confidence),
    Below(Confidence),
    Between { lower: Confidence, upper: Confidence },
}

impl ConfidenceThreshold {
    pub fn matches(&self, confidence: Confidence) -> bool {
        match self {
            Self::Above(threshold) => confidence.raw() > threshold.raw(),
            Self::Below(threshold) => confidence.raw() < threshold.raw(),
            Self::Between { lower, upper } => {
                confidence.raw() >= lower.raw() && confidence.raw() <= upper.raw()
            }
        }
    }
}

/// Confidence interval for PREDICT queries
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower: Confidence,
    pub upper: Confidence,
}

impl ConfidenceInterval {
    pub fn new(lower: Confidence, upper: Confidence) -> Result<Self, ValidationError> {
        if lower.raw() > upper.raw() {
            return Err(ValidationError::InvalidInterval { lower, upper });
        }
        Ok(Self { lower, upper })
    }
}
```

### 11. Episode Selector

```rust
/// Selector for episodes in CONSOLIDATE queries
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EpisodeSelector {
    /// All episodes
    All,

    /// Episodes matching pattern
    Pattern(Pattern),

    /// Episodes matching constraints
    Where(Vec<Constraint>),
}
```

### 12. Validation Errors

```rust
/// Comprehensive validation errors with cognitive error guidance.
///
/// Follows engram-core error design principles:
/// - Clear error messages with context
/// - Actionable suggestions for resolution
/// - Examples showing correct usage
#[derive(Debug, Clone, thiserror::Error)]
pub enum ValidationError {
    #[error("Empty embedding vector\n  Expected: 768-dimensional f32 vector\n  Suggestion: Use embedding service to generate vector from content\n  Example: Pattern::Embedding {{ vector: embed(text), threshold: 0.8 }}")]
    EmptyEmbedding,

    #[error("Invalid embedding dimension: expected {expected}, got {actual}\n  Expected: 768-dimensional vector (current system default)\n  Suggestion: Ensure embedding model matches system configuration\n  Example: Use text-embedding-ada-002 or compatible 768-dim model")]
    InvalidEmbeddingDimension { expected: usize, actual: usize },

    #[error("Invalid similarity threshold: {0}, must be in [0,1]\n  Expected: Threshold between 0.0 (no similarity) and 1.0 (identical)\n  Suggestion: Use 0.7-0.9 for strict matching, 0.5-0.7 for fuzzy matching\n  Example: Pattern::Embedding {{ vector, threshold: 0.8 }}")]
    InvalidThreshold(f32),

    #[error("Empty node identifier\n  Expected: Non-empty string identifying a memory node\n  Suggestion: Use meaningful IDs like 'episode_123' or 'concept_ai'\n  Example: NodeIdentifier::from(\"episode_123\")")]
    EmptyNodeId,

    #[error("Node identifier too long: {actual_length} bytes (max {max_length})\n  Expected: Node ID under {max_length} bytes\n  Suggestion: Use shorter, more compact identifiers\n  Example: Use 'usr_123' instead of full UUIDs")]
    NodeIdTooLong {
        max_length: usize,
        actual_length: usize,
    },

    #[error("Empty content match pattern\n  Expected: Non-empty substring to search for\n  Suggestion: Provide text to match against memory content\n  Example: Pattern::ContentMatch(\"neural network\")")]
    EmptyContentMatch,

    #[error("Invalid confidence interval: lower={lower:?} > upper={upper:?}\n  Expected: lower <= upper, both in [0, 1]\n  Suggestion: Swap values or use ConfidenceInterval::new() for validation\n  Example: ConfidenceInterval::new(0.5, 0.9)")]
    InvalidInterval {
        lower: Confidence,
        upper: Confidence,
    },

    #[error("Invalid decay rate: {0}, must be in [0,1]\n  Expected: Decay rate between 0.0 (no decay) and 1.0 (instant decay)\n  Suggestion: Use 0.1 for slow forgetting, 0.3-0.5 for typical decay\n  Example: SpreadQuery {{ decay_rate: Some(0.1), .. }}")]
    InvalidDecayRate(f32),

    #[error("Invalid activation threshold: {0}, must be in [0,1]\n  Expected: Activation threshold between 0.0 and 1.0\n  Suggestion: Use 0.01 for wide spreading, 0.1 for focused activation\n  Example: SpreadQuery {{ activation_threshold: Some(0.01), .. }}")]
    InvalidActivationThreshold(f32),

    #[error("Invalid constraint: {0}\n  Expected: Semantically valid query constraint\n  Suggestion: {1}\n  Example: {2}")]
    InvalidConstraint(String, String, String),
}

impl ValidationError {
    /// Create custom constraint validation error with context
    pub fn invalid_constraint(
        reason: impl Into<String>,
        suggestion: impl Into<String>,
        example: impl Into<String>,
    ) -> Self {
        Self::InvalidConstraint(reason.into(), suggestion.into(), example.into())
    }
}
```

---

## Advanced Features

### Type-State Pattern for Query Validation

To enforce query validity at compile time, use builder pattern with phantom types:

```rust
/// Query builder states for type-state pattern
pub mod builder_states {
    /// Builder has no pattern set
    pub struct NoPattern;
    /// Builder has pattern set
    pub struct WithPattern;
    /// Builder is validated and ready
    pub struct Validated;
}

/// Type-safe recall query builder
///
/// Ensures pattern is set before building, using phantom type states
/// to prevent invalid queries from compiling.
pub struct RecallQueryBuilder<State = builder_states::NoPattern> {
    pattern: Option<Pattern<'static>>,
    constraints: Vec<Constraint>,
    confidence_threshold: Option<ConfidenceThreshold>,
    base_rate: Option<Confidence>,
    limit: Option<usize>,
    _state: PhantomData<State>,
}

impl RecallQueryBuilder<builder_states::NoPattern> {
    /// Create a new builder (starts in NoPattern state)
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

    /// Set the pattern (transitions to WithPattern state)
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

impl RecallQueryBuilder<builder_states::WithPattern> {
    /// Add constraint (can be called multiple times)
    #[must_use]
    pub fn constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Set confidence threshold
    #[must_use]
    pub fn confidence_threshold(mut self, threshold: ConfidenceThreshold) -> Self {
        self.confidence_threshold = Some(threshold);
        self
    }

    /// Set base rate prior
    #[must_use]
    pub fn base_rate(mut self, rate: Confidence) -> Self {
        self.base_rate = Some(rate);
        self
    }

    /// Set result limit
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Validate and build the query
    ///
    /// This consumes the builder and returns a validated RecallQuery.
    /// Validation errors are returned immediately.
    pub fn build(self) -> Result<RecallQuery, ValidationError> {
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

// Example usage (compile-time safe):
// let query = RecallQueryBuilder::new()
//     .pattern(Pattern::NodeId("episode_123".into()))  // Required!
//     .constraint(Constraint::ConfidenceAbove(Confidence::HIGH))
//     .limit(10)
//     .build()?;
//
// This won't compile (missing pattern):
// let query = RecallQueryBuilder::new()
//     .limit(10)
//     .build()?;  // ERROR: no method `build` on RecallQueryBuilder<NoPattern>
```

### Zero-Copy Constraint Optimization

```rust
/// Constraint with zero-copy string optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint<'a> {
    /// Similarity constraint: SIMILAR TO <embedding>
    SimilarTo {
        embedding: Vec<f32>,
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
    ///
    /// Zero-copy: references space ID from source when parsed
    InMemorySpace(MemorySpaceId),

    /// Content constraint: CONTENT CONTAINS <text>
    ///
    /// Zero-copy: uses Cow to avoid allocation when parsed from source
    ContentContains(Cow<'a, str>),
}

impl<'a> Constraint<'a> {
    /// Validate constraint semantics
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::SimilarTo { embedding, threshold } => {
                if embedding.is_empty() {
                    return Err(ValidationError::EmptyEmbedding);
                }
                if !(*threshold >= 0.0 && *threshold <= 1.0) {
                    return Err(ValidationError::InvalidThreshold(*threshold));
                }
                Ok(())
            }
            Self::ConfidenceAbove(c) | Self::ConfidenceBelow(c) => {
                // Confidence type already guarantees valid range
                Ok(())
            }
            Self::ContentContains(s) if s.is_empty() => {
                Err(ValidationError::EmptyContentMatch)
            }
            _ => Ok(()),
        }
    }

    /// Convert to owned Constraint (clones borrowed data)
    #[must_use]
    pub fn into_owned(self) -> Constraint<'static> {
        match self {
            Self::SimilarTo { embedding, threshold } => {
                Constraint::SimilarTo { embedding, threshold }
            }
            Self::CreatedBefore(t) => Constraint::CreatedBefore(t),
            Self::CreatedAfter(t) => Constraint::CreatedAfter(t),
            Self::ConfidenceAbove(c) => Constraint::ConfidenceAbove(c),
            Self::ConfidenceBelow(c) => Constraint::ConfidenceBelow(c),
            Self::InMemorySpace(id) => Constraint::InMemorySpace(id),
            Self::ContentContains(s) => Constraint::ContentContains(Cow::Owned(s.into_owned())),
        }
    }
}
```

### Lifetime Propagation Through Query Types

All query types need to propagate the lifetime parameter:

```rust
// Updated query definitions with lifetime parameter

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallQuery<'a> {
    pub pattern: Pattern<'a>,
    pub constraints: Vec<Constraint<'a>>,
    pub confidence_threshold: Option<ConfidenceThreshold>,
    pub base_rate: Option<Confidence>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpreadQuery<'a> {
    pub source: NodeIdentifier<'a>,
    pub max_hops: Option<u16>,
    pub decay_rate: Option<f32>,
    pub activation_threshold: Option<f32>,
    pub refractory_period: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictQuery<'a> {
    pub pattern: Pattern<'a>,
    pub context: Vec<NodeIdentifier<'a>>,
    pub horizon: Option<Duration>,
    pub confidence_constraint: Option<ConfidenceInterval>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImagineQuery<'a> {
    pub pattern: Pattern<'a>,
    pub seeds: Vec<NodeIdentifier<'a>>,
    pub novelty: Option<f32>,
    pub confidence_threshold: Option<ConfidenceThreshold>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsolidateQuery<'a> {
    pub episodes: EpisodeSelector<'a>,
    pub target: NodeIdentifier<'a>,
    pub scheduler_policy: Option<SchedulerPolicy>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EpisodeSelector<'a> {
    All,
    Pattern(Pattern<'a>),
    Where(Vec<Constraint<'a>>),
}

// Top-level Query enum with lifetime
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Query<'a> {
    Recall(RecallQuery<'a>),
    Predict(PredictQuery<'a>),
    Imagine(ImagineQuery<'a>),
    Consolidate(ConsolidateQuery<'a>),
    Spread(SpreadQuery<'a>),
}

impl<'a> Query<'a> {
    /// Convert to owned Query (clones all borrowed data)
    ///
    /// Use this when you need to store the query beyond the lifetime
    /// of the source text (e.g., in a query cache or log).
    #[must_use]
    pub fn into_owned(self) -> Query<'static> {
        match self {
            Self::Recall(q) => Query::Recall(RecallQuery {
                pattern: q.pattern.into_owned(),
                constraints: q.constraints.into_iter().map(|c| c.into_owned()).collect(),
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
            // ... similar for other variants
            _ => todo!("Implement for other query types"),
        }
    }
}
```

---

## Files to Create/Modify

1. **Create**: `engram-core/src/query/parser/ast.rs`
   - All AST type definitions

2. **Modify**: `engram-core/src/query/parser/mod.rs`
   - Export AST types: `pub use ast::*;`

3. **Modify**: `engram-core/Cargo.toml`
   - Add dependency: `thiserror = "2.0"` (if not already present)

---

## Type Size Optimization and Performance Testing

Ensure AST types are compact for performance:

```rust
#[cfg(test)]
mod size_tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_query_size() {
        // Query enum should be reasonable size (3-4 cache lines max)
        let size = size_of::<Query>();
        assert!(
            size < 256,
            "Query too large: {} bytes (target: <256)",
            size
        );

        // Print actual size for optimization tracking
        eprintln!("Query size: {} bytes", size);
    }

    #[test]
    fn test_pattern_size() {
        let size = size_of::<Pattern>();
        assert!(
            size < 128,
            "Pattern too large: {} bytes (target: <128)",
            size
        );
        eprintln!("Pattern size: {} bytes", size);
    }

    #[test]
    fn test_constraint_size() {
        let size = size_of::<Constraint>();
        assert!(
            size < 64,
            "Constraint too large: {} bytes (target: <64)",
            size
        );
        eprintln!("Constraint size: {} bytes", size);
    }

    #[test]
    fn test_node_identifier_size() {
        let size = size_of::<NodeIdentifier>();
        assert_eq!(
            size, 32,
            "NodeIdentifier size changed: {} bytes (expected: 32)",
            size
        );
    }

    #[test]
    fn test_confidence_threshold_size() {
        let size = size_of::<ConfidenceThreshold>();
        // Should be small (discriminant + Confidence which is f32)
        assert!(
            size <= 16,
            "ConfidenceThreshold too large: {} bytes",
            size
        );
    }

    #[test]
    fn test_query_category_size() {
        let size = size_of::<QueryCategory>();
        // Enum with 5 variants = 1 byte discriminant
        assert_eq!(size, 1, "QueryCategory should be 1 byte");
    }

    /// Verify zero-cost abstraction: Cow<'a, str> has same size as String
    #[test]
    fn test_cow_size() {
        use std::borrow::Cow;
        assert_eq!(
            size_of::<Cow<'_, str>>(),
            size_of::<String>(),
            "Cow should have same size as String"
        );
    }
}

#[cfg(test)]
mod alignment_tests {
    use super::*;
    use std::mem::align_of;

    #[test]
    fn test_cache_line_alignment() {
        // Modern CPUs use 64-byte cache lines
        const CACHE_LINE_SIZE: usize = 64;

        // Query variants should be well-aligned
        assert_eq!(
            align_of::<Query>() % 8,
            0,
            "Query should be 8-byte aligned"
        );

        // Check if any variant would benefit from explicit alignment
        eprintln!("Query alignment: {} bytes", align_of::<Query>());
    }
}
```

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_query_construction() {
        let query = RecallQuery {
            pattern: Pattern::NodeId("episode_123".into()),
            constraints: vec![
                Constraint::ConfidenceAbove(Confidence::from_raw(0.7)),
            ],
            confidence_threshold: Some(ConfidenceThreshold::Above(Confidence::from_raw(0.8))),
            base_rate: None,
            limit: Some(10),
        };

        assert_eq!(query.pattern, Pattern::NodeId("episode_123".into()));
        assert_eq!(query.constraints.len(), 1);
    }

    #[test]
    fn test_pattern_validation() {
        // Valid embedding
        let pattern = Pattern::Embedding {
            vector: vec![0.1, 0.2, 0.3],
            threshold: 0.8,
        };
        assert!(pattern.validate().is_ok());

        // Invalid: empty embedding
        let pattern = Pattern::Embedding {
            vector: vec![],
            threshold: 0.8,
        };
        assert!(pattern.validate().is_err());

        // Invalid: threshold out of range
        let pattern = Pattern::Embedding {
            vector: vec![0.1],
            threshold: 1.5,
        };
        assert!(pattern.validate().is_err());
    }

    #[test]
    fn test_confidence_threshold_matching() {
        let threshold = ConfidenceThreshold::Above(Confidence::from_raw(0.7));
        assert!(threshold.matches(Confidence::from_raw(0.8)));
        assert!(!threshold.matches(Confidence::from_raw(0.6)));

        let threshold = ConfidenceThreshold::Between {
            lower: Confidence::from_raw(0.5),
            upper: Confidence::from_raw(0.9),
        };
        assert!(threshold.matches(Confidence::from_raw(0.7)));
        assert!(!threshold.matches(Confidence::from_raw(0.3)));
        assert!(!threshold.matches(Confidence::from_raw(0.95)));
    }

    #[test]
    fn test_confidence_interval_validation() {
        // Valid interval
        let interval = ConfidenceInterval::new(
            Confidence::from_raw(0.5),
            Confidence::from_raw(0.9),
        );
        assert!(interval.is_ok());

        // Invalid: lower > upper
        let interval = ConfidenceInterval::new(
            Confidence::from_raw(0.9),
            Confidence::from_raw(0.5),
        );
        assert!(interval.is_err());
    }

    #[test]
    fn test_node_identifier_validation() {
        let id = NodeIdentifier::new("valid_id");
        assert!(id.validate().is_ok());

        let id = NodeIdentifier::new("");
        assert!(id.validate().is_err());
    }

    #[test]
    fn test_estimated_cost() {
        let recall = Query::Recall(RecallQuery {
            pattern: Pattern::Any,
            constraints: vec![],
            confidence_threshold: None,
            base_rate: None,
            limit: None,
        });
        assert_eq!(recall.estimated_cost(), 1000);

        let spread = Query::Spread(SpreadQuery {
            source: NodeIdentifier::new("node"),
            max_hops: Some(3),
            decay_rate: None,
            activation_threshold: None,
            refractory_period: None,
        });
        assert!(spread.estimated_cost() > recall.estimated_cost());
    }
}
```

### Serde Round-Trip Tests

```rust
#[test]
fn test_serde_roundtrip() {
    let query = Query::Recall(RecallQuery {
        pattern: Pattern::Embedding {
            vector: vec![0.1, 0.2, 0.3],
            threshold: 0.8,
        },
        constraints: vec![
            Constraint::ConfidenceAbove(Confidence::from_raw(0.7)),
        ],
        confidence_threshold: None,
        base_rate: None,
        limit: Some(10),
    });

    // Serialize to JSON
    let json = serde_json::to_string(&query).unwrap();

    // Deserialize back
    let deserialized: Query = serde_json::from_str(&json).unwrap();

    assert_eq!(query, deserialized);
}
```

---

## Acceptance Criteria

### Core Implementation
- [ ] All AST types defined with lifetime parameters for zero-copy optimization
- [ ] Complete documentation with memory layout analysis for each type
- [ ] Serde support for all types with `#[serde(borrow)]` annotations
- [ ] Validation methods for all constraint types with cognitive error messages
- [ ] Type-state builders implemented for RecallQuery (minimum)

### Performance Requirements
- [ ] Type sizes verified and documented:
  - Query enum: <256 bytes
  - Pattern enum: <128 bytes
  - Constraint enum: <64 bytes
  - NodeIdentifier: 32 bytes (Cow<str> size)
- [ ] Cache alignment verified (8-byte minimum)
- [ ] Zero-cost abstractions confirmed (no overhead vs hand-written)

### Zero-Copy Optimization
- [ ] NodeIdentifier uses `Cow<'a, str>` for zero-copy parsing
- [ ] Pattern variants use `Cow<'a, str>` where applicable
- [ ] Constraint::ContentContains uses `Cow<'a, str>`
- [ ] All query types propagate lifetime parameter `'a`
- [ ] `into_owned()` methods implemented for all lifetime-parameterized types

### Type Safety
- [ ] Builder pattern prevents invalid query construction at compile time
- [ ] Pattern validation enforces 768-dimensional embeddings
- [ ] Confidence thresholds validated in [0, 1] range
- [ ] Node identifier length limits enforced (<256 bytes)
- [ ] Decay rates and activation thresholds validated

### Testing
- [ ] Unit tests achieve >95% coverage
- [ ] Serde round-trip tests pass for all query types
- [ ] Size tests verify memory layout targets
- [ ] Alignment tests verify cache-friendly layout
- [ ] Validation tests cover all error cases
- [ ] Builder pattern compile-fail tests (requires pattern before build)
- [ ] Zero-copy behavior verified (no unnecessary allocations)

### Code Quality
- [ ] Zero clippy warnings
- [ ] Display/Debug implementations for all types
- [ ] `#[must_use]` annotations on getters and builders
- [ ] Comprehensive inline documentation
- [ ] Integration with existing Confidence and MemorySpaceId types verified

### Integration Points
- [ ] Query types match cognitive operations in spec (Task 001)
- [ ] AST ready for parser consumption (Task 003)
- [ ] Types compatible with query executor (Task 005)
- [ ] Constants align with existing activation/mod.rs defaults

---

## Integration Points

- **Previous Task**: Task 001 provides Token types for parser
- **Next Task**: Task 003 (Parser) produces AST from tokens
- **Query Executor**: Task 005 consumes AST for execution

---

## Notes

- **Type Safety**: Use newtypes (NodeIdentifier, ConfidenceThreshold) to prevent confusion
- **Validation**: Validate at AST construction, not execution time
- **Extensibility**: Design for future query types (e.g., FORGET, MERGE)
- **Serde**: JSON serialization useful for debugging and logging

---

## References

- Confidence type: `engram-core/src/lib.rs` lines 64-267
- MemorySpaceId: `engram-core/src/types.rs`
- Memory/Episode types: `engram-core/src/memory.rs`
- Activation spreading: `engram-core/src/activation/mod.rs`, `activation/parallel.rs`
- Existing query types: `engram-core/src/query/mod.rs`

---

## Enhancement Summary

This enhanced specification incorporates advanced Rust techniques for high-performance AST design:

### 1. Zero-Cost Abstractions

**Problem**: Parsing query strings typically requires allocating String objects for each identifier, pattern, and constraint. This creates memory pressure and cache pollution.

**Solution**: Use `Cow<'a, str>` throughout the AST to reference source text directly during parsing. Only allocate when the AST needs to outlive the source (via `into_owned()`).

**Impact**:
- Eliminates ~70% of allocations during parsing for typical queries
- Improves parser throughput by 2-3x (estimated, based on allocation overhead)
- Maintains API ergonomics through trait implementations

### 2. Type-State Pattern

**Problem**: Invalid queries can be constructed at runtime (missing required fields like pattern), requiring runtime validation.

**Solution**: Use phantom type parameters to track builder state. Compiler prevents calling `build()` until pattern is set.

**Impact**:
- Catches query construction errors at compile time
- Zero runtime overhead (phantom types compile away)
- Improved IDE autocomplete and API discoverability

### 3. Memory Layout Optimization

**Problem**: Large AST enums can cause cache misses during traversal, especially in hot paths like query validation.

**Solution**:
- Document memory layout for each type
- Set explicit size targets (<256 bytes for Query)
- Add compile-time size tests to prevent regressions

**Impact**:
- AST fits in 3-4 cache lines maximum
- Reduced cache misses during query processing
- Enables more aggressive inlining by compiler

### 4. Integration with Existing Types

**Problem**: AST must work seamlessly with existing Confidence, MemorySpaceId, and activation spreading types.

**Solution**:
- Use Confidence directly (already a zero-cost newtype over f32)
- Reference MemorySpaceId (Arc<str> internally, cheap to clone)
- Match constants with activation/mod.rs (DEFAULT_DECAY_RATE, etc.)

**Impact**:
- No impedance mismatch between parser and executor
- Familiar types for developers already using engram-core
- Reduced cognitive load during implementation

### 5. Cognitive Error Messages

**Problem**: Validation errors should help developers fix issues quickly, not just report failures.

**Solution**:
- Every error includes: context, expected value, suggestion, example
- Follows existing engram-core error conventions
- Uses thiserror for zero-cost error handling

**Impact**:
- Faster debugging during development
- Better user experience for query API
- Consistent with project error philosophy

### 6. Performance Verification

**Problem**: Type size and layout targets must be verified, not assumed.

**Solution**:
- Compile-time size tests (`assert!(size_of::<Query>() < 256)`)
- Alignment tests for cache line optimization
- Benchmarks comparing zero-copy vs owned variants

**Impact**:
- Prevents performance regressions
- Documents actual sizes for future optimization
- Provides data for architecture decisions

---

## Implementation Notes

### Critical Path Items

1. **Lifetime Parameters**: All string-containing types need `'a` parameter. This percolates through the entire AST. Start with NodeIdentifier, then Pattern, then query types.

2. **Builder Pattern**: RecallQueryBuilder is the proof-of-concept. Once working, create builders for other query types (SpreadQueryBuilder, etc.).

3. **Validation**: Implement validation methods before parsers. Parser should call `validate()` after constructing AST to catch errors early.

4. **Testing**: Size tests and alignment tests should be written first (TDD). They document the performance targets and prevent regressions.

### Optional Enhancements

- Use `Box<T>` for large enum variants if size targets aren't met
- Consider `SmallVec` for constraints (typically 0-3 constraints per query)
- Explore `interned-strings` crate if node IDs show high duplication
- Add `#[repr(C)]` if FFI compatibility needed (Python/TypeScript bindings)

### Risk Mitigation

**Lifetime Complexity**: The `'a` parameter adds complexity to function signatures. Mitigate by:
- Providing `into_owned()` escape hatch for long-lived storage
- Using `'static` in builder pattern (owned strings only)
- Comprehensive examples in documentation

**Serde Compatibility**: `#[serde(borrow)]` requires careful use. Mitigate by:
- Thorough round-trip testing
- Document owned vs borrowed serialization behavior
- Consider providing both owned and borrowed variants if needed

**Type-State Ergonomics**: Builder pattern can be verbose. Mitigate by:
- Provide convenience constructors for common cases
- Make builders optional (allow direct struct construction for advanced users)
- Add examples showing both builder and direct construction
