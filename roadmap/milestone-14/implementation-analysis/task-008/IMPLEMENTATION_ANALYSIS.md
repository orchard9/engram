# Task 008 (Conflict Resolution) - Codebase Implementation Analysis Report

**Analysis Date**: 2025-11-01
**Scope**: Very thorough examination of Engram codebase for Task 008 (Conflict Resolution for Divergent Consolidations)
**Task File**: /Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/008_conflict_resolution_divergent_consolidations_pending.md

---

## Executive Summary

The Engram codebase has well-established foundations for implementing conflict resolution for divergent consolidations. Key cognitive types (Confidence, Episode, SemanticPattern) are fully defined with serde support and mathematical operations. The codebase already contains:

1. **Cosine similarity implementations** for embedding comparison
2. **Confidence types** with weighted averaging support
3. **Embedding averaging** with Kahan summation for determinism
4. **EpisodicPattern structures** with metadata for tracking
5. **Gossip protocol foundations** in Task 007 specifications

No Vector Clock implementation currently exists in the codebase - this must be created new. This report provides exact line numbers, struct definitions, and integration points for Task 008 implementation.

---

## 1. SEMANTIC PATTERN STRUCTURES

### 1.1 SemanticPattern Definition

**Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/consolidation.rs` (Lines 45-65)

```rust
#[derive(Debug, Clone)]
pub struct SemanticPattern {
    /// Pattern identifier
    pub id: String,

    /// Semantic embedding (averaged across episodes)
    pub embedding: [f32; 768],

    /// Contributing episodes
    pub source_episodes: Vec<String>,

    /// Pattern strength
    pub strength: f32,

    /// Schema confidence
    pub schema_confidence: Confidence,

    /// Last consolidation time
    pub last_consolidated: DateTime<Utc>,
}
```

**Key Characteristics**:
- **ID**: String identifier for unique pattern reference
- **Embedding**: Fixed 768-dimensional array (matches OpenAI's text-embedding-ada-002)
- **Source Episodes**: Vector of episode IDs contributing to pattern
- **Strength**: f32 coherence measure (0.0-1.0)
- **Schema Confidence**: Confidence type (wraps f32 with validation)
- **Timestamp**: DateTime<Utc> for causality tracking

**Integration Point for Task 008**: This structure must be extended with:
- `vector_clock: VectorClock` for distributed causality
- `origin_node_id: String` for tracking source
- `generation: u32` for reconsolidation tracking
- `episode_ids: HashSet<String>` instead of Vec (for efficient overlap)

### 1.2 EpisodicPattern Definition

**Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/pattern_detector.rs` (Lines 54-73)

```rust
#[derive(Debug, Clone)]
pub struct EpisodicPattern {
    /// Pattern identifier (deterministic based on source episodes)
    pub id: String,
    
    /// Average embedding across clustered episodes
    pub embedding: [f32; 768],
    
    /// IDs of source episodes contributing to this pattern
    pub source_episodes: Vec<String>,
    
    /// Pattern strength (coherence of cluster, 0.0-1.0)
    pub strength: f32,
    
    /// Extracted common features
    pub features: Vec<PatternFeature>,
    
    /// First occurrence timestamp
    pub first_occurrence: DateTime<Utc>,
    
    /// Last occurrence timestamp
    pub last_occurrence: DateTime<Utc>,
    
    /// Number of episodes in this pattern
    pub occurrence_count: usize,
}
```

**Relationship to Task 008**: EpisodicPattern is pre-consolidation (local pattern detection). SemanticPattern is post-consolidation (system-wide semantic knowledge). Conflict resolution applies at the SemanticPattern level.

---

## 2. EPISODE STRUCTURES

### 2.1 Episode Definition

**Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/memory.rs` (Lines 206-259)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique identifier
    pub id: String,

    /// When the episode occurred
    pub when: DateTime<Utc>,

    /// Where the episode occurred (optional location)
    pub where_location: Option<String>,

    /// Who was involved (optional participants)
    pub who: Option<Vec<String>>,

    /// What happened (semantic content)
    pub what: String,

    /// Embedding representation of the episode
    #[serde(with = "embedding_serde")]
    pub embedding: [f32; 768],

    /// Optional provenance metadata for the embedding
    #[serde(default)]
    pub embedding_provenance: Option<EmbeddingProvenance>,

    /// Confidence in episode encoding quality
    pub encoding_confidence: Confidence,

    /// Confidence in episode vividness/detail richness
    pub vividness_confidence: Confidence,

    /// Overall episode reliability confidence
    pub reliability_confidence: Confidence,

    /// When this episode was last recalled
    pub last_recall: DateTime<Utc>,

    /// Number of times this episode has been recalled
    pub recall_count: u32,

    /// Decay rate for this specific episode
    pub decay_rate: f32,

    /// Optional decay function override for this episode
    pub decay_function: Option<DecayFunction>,

    /// Arbitrary metadata for tracking events
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}
```

**Critical for Task 008**:
- Episodes are the atomic units that get consolidated into patterns
- Multiple confidence values (encoding, vividness, reliability) provide multi-dimensional confidence
- Episode metadata (HashMap) can store distributed system annotations
- Recall count and timestamps enable frequency-based confidence weighting

### 2.2 Episode-to-Pattern Relationships

**Pattern Source Tracking**:
```
Episode {id: "e1", embedding: [0.2, 0.8, ...], encoding_confidence: 0.85}
Episode {id: "e2", embedding: [0.25, 0.75, ...], encoding_confidence: 0.80}
  ↓
SemanticPattern {
    id: "pattern_morning_routine",
    embedding: [0.225, 0.775, ...],  // averaged
    source_episodes: ["e1", "e2"],
    schema_confidence: 0.82  // confidence-weighted average
}
```

**Task 008 Challenge**: When nodes A and B consolidate overlapping episode sets differently:
- Node A: {e1, e2, e3} → Pattern_A with confidence 0.85
- Node B: {e1, e2, e4} → Pattern_B with confidence 0.78

The resolution must handle the union {e1, e2, e3, e4} and compute merged confidence.

---

## 3. CONFIDENCE CALCULATION

### 3.1 Confidence Type Definition

**Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/lib.rs` (Lines 87-290)

```rust
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Confidence(#[serde(deserialize_with = "validate_confidence")] f32);

// Predefined levels:
impl Confidence {
    pub const HIGH: Self = Self(0.9);
    pub const MEDIUM: Self = Self(0.5);
    pub const LOW: Self = Self(0.1);
    pub const CERTAIN: Self = Self(1.0);
    pub const NONE: Self = Self(0.0);
}
```

### 3.2 Confidence Constructors and Methods

**Primary Constructors**:
```rust
// Exact value with clamping
#[must_use]
pub const fn exact(value: f32) -> Self {
    let clamped = if value < 0.0 { 0.0 } 
                  else if value > 1.0 { 1.0 } 
                  else { value };
    Self(clamped)
}

// Frequency-based (human-friendly)
#[must_use]
pub const fn from_successes(successes: u32, total: u32) -> Self {
    if total == 0 { return Self::NONE; }
    let ratio = (successes as f32 / total as f32).clamp(0.0, 1.0);
    Self::exact(ratio)
}

// From raw f32 (auto-clamped)
#[must_use]
pub const fn from_raw(value: f32) -> Self {
    Self(value.clamp(0.0, 1.0))
}
```

### 3.3 Confidence Arithmetic Operations

**Critical for Task 008 Merging**:

```rust
// Weighted combination (for merging patterns)
#[must_use]
pub const fn combine_weighted(
    self, 
    other: Self, 
    self_weight: f32, 
    other_weight: f32
) -> Self {
    let total_weight = self_weight + other_weight;
    if total_weight == 0.0 { return Self::MEDIUM; }
    let weighted_avg = (self.0 * self_weight + other.0 * other_weight) / total_weight;
    Self::exact(weighted_avg)
}

// Logical operations with bias prevention
#[must_use]
pub const fn and(self, other: Self) -> Self {
    let result = self.0 * other.0;
    Self(result)  // Prevents conjunction fallacy
}

// Scaling (for decay/attenuation)
#[must_use]
pub fn scaled(self, factor: f32) -> Self {
    Self::from_raw(self.0 * factor.clamp(0.0, 1.0))
}

// Decay with hop count
#[must_use]
pub fn decayed(self, decay_rate: f32, hop_count: u16) -> Self {
    let decay_factor = (-decay_rate * f32::from(hop_count)).exp();
    Self::from_raw(self.0 * decay_factor)
}
```

### 3.4 Confidence Bounds and Range

**Range**: [0.0, 1.0] inclusive, guaranteed by Confidence type
**Clamping**: Automatic on all constructors - no invalid states possible
**Serialization**: Custom deserializer validates bounds during deserialization
**Comparison**: PartialOrd implemented for threshold-based decisions

**For Task 008 Confidence Adjustment**:
- After merge with similarity 0.9: penalty = (1.0 - 0.9) × 0.3 = 0.03
- Base confidence = (c1 + c2) / 2
- Adjusted confidence = base × (1.0 - penalty)
- Final = max(adjusted, MIN_CONFIDENCE = 0.1)

---

## 4. EMBEDDING OPERATIONS

### 4.1 Embedding Format

**Constant Definition**:
```
Location: /Users/jordan/Workspace/orchard9/engram/engram-core/src/lib.rs (Line 85)
pub const EMBEDDING_DIM: usize = 768;
```

**Storage**: Fixed-size array `[f32; 768]` (not Vec<f32>)
- **Advantage**: Stack allocation, no heap fragmentation, zero-copy operations
- **Size**: 768 × 4 bytes = 3,072 bytes per embedding
- **Alignment**: 4-byte f32 alignment

**Serialization**: Custom serde module
```rust
mod embedding_serde {
    pub fn serialize<S>(embedding: &[f32; 768], serializer: S) -> Result<S::Ok, S::Error> {
        embedding.as_slice().serialize(serializer)
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<[f32; 768], D::Error> {
        let vec = Vec::<f32>::deserialize(deserializer)?;
        if vec.len() != 768 {
            return Err(serde::de::Error::custom(format!(
                "Expected array of length 768, got {}",
                vec.len()
            )));
        }
        let mut array = [0.0f32; 768];
        array.copy_from_slice(&vec);
        Ok(array)
    }
}
```

### 4.2 Cosine Similarity Implementation

**Location 1**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/cognitive/priming/semantic.rs`

```rust
fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for i in 0..768 {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}
```

**Location 2**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/cognitive/interference/proactive.rs`

```rust
fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    let mut dot_product = 0.0f32;
    let mut norm_a_sq = 0.0f32;
    let mut norm_b_sq = 0.0f32;
    
    for i in 0..768 {
        dot_product += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }
    
    let norm_product = (norm_a_sq * norm_b_sq).sqrt();
    if norm_product == 0.0 { return 0.0; }
    
    (dot_product / norm_product).clamp(-1.0, 1.0)
}
```

**For Task 008**: Use the second implementation (with zero-check and clamp) for robustness.

### 4.3 Embedding Averaging (Deterministic)

**Location**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/pattern_detector.rs` (Lines 302-345)

```rust
fn average_embeddings(episodes: &[Episode]) -> [f32; 768] {
    let mut avg = [0.0f32; 768];
    let count = episodes.len() as f32;

    if count == 0.0 {
        return avg;
    }

    // Use Kahan summation for each dimension for deterministic FP arithmetic
    for (i, avg_val) in avg.iter_mut().enumerate() {
        let values = episodes.iter().map(|ep| ep.embedding[i]);
        let (sum, _compensation) = Self::kahan_sum(values);
        *avg_val = sum / count;
    }

    avg
}

fn kahan_sum(values: impl Iterator<Item = f32>) -> (f32, f32) {
    let mut sum = 0.0f32;
    let mut compensation = 0.0f32;

    for value in values {
        let y = value - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    (sum, compensation)
}
```

**Why Kahan Summation?**
- Floating-point arithmetic is not associative: (a + b) + c ≠ a + (b + c)
- Kahan tracks rounding error and compensates on next iteration
- Ensures deterministic results regardless of episode order
- Critical for Task 008 where multiple nodes must converge to identical patterns

### 4.4 Embedding Normalization

**Pattern Strength Computation**:
```rust
fn compute_pattern_strength(episodes: &[Episode], centroid: &[f32; 768]) -> f32 {
    if episodes.is_empty() { return 0.0; }
    
    // Similarity of each episode to centroid
    let similarities: Vec<f32> = episodes
        .iter()
        .map(|ep| Self::embedding_similarity(&ep.embedding, centroid))
        .collect();
    
    // Average similarity = pattern strength
    let sum: f32 = similarities.iter().sum();
    sum / (similarities.len() as f32)
}
```

**Normalization Note**: The codebase does NOT normalize embeddings after averaging. Task 008 should normalize merged embeddings to maintain unit length for cosine similarity operations.

**Recommended Normalization for Merging**:
```rust
pub fn normalize_embedding(embedding: &mut [f32; 768]) {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in embedding {
            *val /= norm;
        }
    }
}
```

---

## 5. EXISTING SIMILARITY AND DISTANCE FUNCTIONS

### 5.1 Embedding Similarity (Cosine)

**Available in codebase**:
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/cognitive/priming/semantic.rs`
- `/Users/jordan/Workspace/orchard9/engram/engram-core/src/cognitive/interference/proactive.rs`

**Threshold Values Used**:
- `0.6`: Minimum for semantic relation (semantic priming)
- `0.8`: Similarity threshold for pattern clustering (pattern_detector.rs:19)
- `0.85`: Pattern merging threshold (recommended in Task 008 spec)

### 5.2 Episode Overlap Calculation

**Not currently in codebase** - must implement for Task 008:

```rust
// Jaccard index (to add to conflict/mod.rs)
fn compute_jaccard_overlap(episodes1: &HashSet<String>, episodes2: &HashSet<String>) -> f32 {
    let intersection_size = episodes1.intersection(episodes2).count();
    let union_size = episodes1.union(episodes2).count();
    
    if union_size == 0 { return 0.0; }
    intersection_size as f32 / union_size as f32
}
```

### 5.3 Pattern Strength as Confidence Proxy

**Pattern Strength Interpretation**:
- `strength: f32` in EpisodicPattern (Lines 64, pattern_detector.rs)
- Computed as average embedding similarity to cluster centroid
- Range: [0.0, 1.0]
- Can be used to initialize SemanticPattern.schema_confidence

**Conversion to Confidence**:
```rust
let schema_confidence = Confidence::exact(episodic.strength);
```

---

## 6. CONSOLIDATION AND MERGING LOGIC

### 6.1 Current Consolidation Architecture

**Files**:
- `engram-core/src/consolidation/mod.rs` (exports)
- `engram-core/src/consolidation/pattern_detector.rs` (core detection)
- `engram-core/src/completion/consolidation.rs` (semantic patterns)
- `engram-core/src/consolidation/dream.rs` (sleep consolidation)

### 6.2 Pattern Merging (Pre-existing)

**Location**: `engram-core/src/consolidation/pattern_detector.rs` (Lines 139-141)

```rust
// Phase 3: Merge similar patterns
Self::merge_similar_patterns(patterns)
```

**Function Signature** (inferred from usage):
```rust
fn merge_similar_patterns(patterns: Vec<EpisodicPattern>) -> Vec<EpisodicPattern> {
    // Implementation not shown in limit, but follows hierarchical clustering
}
```

**Integration Note**: This merges EpisodicPatterns locally BEFORE SemanticPattern creation. Task 008 handles conflicts AFTER SemanticPatterns are created on different nodes.

### 6.3 Episodic to Semantic Conversion

**Location**: `engram-core/src/completion/consolidation.rs` (Lines 67-81)

```rust
impl From<EpisodicPattern> for SemanticPattern {
    fn from(episodic: EpisodicPattern) -> Self {
        Self {
            id: episodic.id,
            embedding: episodic.embedding,
            source_episodes: episodic.source_episodes,
            strength: episodic.strength,
            schema_confidence: Confidence::exact(episodic.strength),
            last_consolidated: episodic.last_occurrence,
        }
    }
}
```

**Task 008 Extension Point**: When converting, also set:
- `vector_clock: VectorClock::new()` (initialized on this node)
- `origin_node_id: node_id.clone()` (from environment/config)
- `generation: 0` (first generation)

---

## 7. VECTOR CLOCK AND CAUSALITY

### 7.1 Current State: NOT IMPLEMENTED

**Search Results**:
- No existing VectorClock implementation in codebase
- No causality tracking in current patterns
- Task 008 is responsible for implementation from specification

### 7.2 Task 008 Vector Clock Specification

**Source**: `roadmap/milestone-14/008_conflict_resolution_divergent_consolidations_pending.md` (Lines 112-171)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VectorClock {
    clocks: HashMap<String, u64>,
}

impl VectorClock {
    pub fn new() -> Self {
        Self { clocks: HashMap::new() }
    }

    pub fn increment(&mut self, node_id: &str) {
        *self.clocks.entry(node_id.to_string()).or_insert(0) += 1;
    }

    pub fn merge(&mut self, other: &VectorClock) {
        for (node_id, timestamp) in &other.clocks {
            self.clocks
                .entry(node_id.clone())
                .and_modify(|t| *t = (*t).max(*timestamp))
                .or_insert(*timestamp);
        }
    }

    pub fn compare(&self, other: &VectorClock) -> CausalOrder {
        let mut less = false;
        let mut greater = false;

        let mut all_nodes: HashSet<&String> = self.clocks.keys().collect();
        all_nodes.extend(other.clocks.keys());

        for node_id in all_nodes {
            let self_ts = self.clocks.get(node_id).copied().unwrap_or(0);
            let other_ts = other.clocks.get(node_id).copied().unwrap_or(0);

            if self_ts < other_ts { less = true; }
            else if self_ts > other_ts { greater = true; }
        }

        match (less, greater) {
            (true, false) => CausalOrder::Before,
            (false, true) => CausalOrder::After,
            (false, false) => CausalOrder::Equal,
            (true, true) => CausalOrder::Concurrent,
        }
    }
}
```

### 7.3 Causality Relationship Enum

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalOrder {
    Before,      // self < other (happens-before)
    After,       // self > other (happens-after)
    Equal,       // self = other (identical)
    Concurrent,  // Neither before nor after (concurrent events)
}
```

---

## 8. INTEGRATION POINTS WITH EXISTING CODE

### 8.1 SemanticPattern Extension

**File to Modify**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/completion/consolidation.rs`

```rust
// CURRENT (Lines 45-65):
pub struct SemanticPattern {
    pub id: String,
    pub embedding: [f32; 768],
    pub source_episodes: Vec<String>,
    pub strength: f32,
    pub schema_confidence: Confidence,
    pub last_consolidated: DateTime<Utc>,
}

// EXTENDED FOR TASK 008:
pub struct SemanticPattern {
    pub id: String,
    pub embedding: [f32; 768],
    pub source_episodes: Vec<String>,  // Change to HashSet<String> for efficiency
    pub strength: f32,
    pub schema_confidence: Confidence,
    pub last_consolidated: DateTime<Utc>,
    
    // NEW FIELDS:
    pub vector_clock: VectorClock,
    pub origin_node_id: String,
    pub generation: u32,
}
```

### 8.2 Consolidation Service Integration

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/service.rs`

**Integration Point**: After creating SemanticPattern, initialize:
```rust
let mut pattern = SemanticPattern::from(episodic);
pattern.vector_clock.increment(&node_id);
pattern.origin_node_id = node_id.clone();
pattern.generation = 0;
```

### 8.3 Gossip Protocol Integration (Task 007)

**File**: `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/007_gossip_consolidation_state_pending.md` (Lines 78-98)

**State to Synchronize**:
```rust
struct ConsolidationState {
    patterns: HashMap<PatternId, SemanticPattern>,
    last_consolidation: Timestamp,
    consolidation_generation: u64,
}
```

**Task 008 Hook Points**:
1. When receiving remote SemanticPattern during gossip:
   - Call `ConflictResolver::resolve(local, remote)`
   - Apply returned `ResolutionResult` to local state

2. Update vector clock on both patterns:
   - Merge remote vector clock into local
   - Increment local node's clock

3. Track generation numbers for reconsolidation events

### 8.4 Metrics Integration Point

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/metrics/cluster_consolidation.rs`

**Metrics to Add**:
```rust
pub struct ConflictResolutionMetrics {
    pub total_conflicts: Counter,
    pub conflicts_by_type: HashMap<ConflictType, Counter>,
    pub resolutions_by_strategy: HashMap<ResolutionStrategy, Counter>,
    pub avg_information_loss: Gauge,
    pub resolution_latency: Histogram,
    pub convergence_rounds: Histogram,
}
```

---

## 9. CONFIDENCE CALCULATION SPECIFICS

### 9.1 Initial Pattern Confidence

**Source**: EpisodicPattern.strength → SemanticPattern.schema_confidence

```rust
// From pattern_detector.rs:268
let strength = Self::compute_pattern_strength(episodes, &avg_embedding);

// Conversion:
let schema_confidence = Confidence::exact(strength);
```

**Strength Calculation**:
- Average of all episodes' cosine similarities to cluster centroid
- Range: [0.0, 1.0] naturally from averaging similarities
- Higher = more coherent cluster

### 9.2 Merged Pattern Confidence Calculation

**Formula (from Task 008 spec, Lines 537-564)**:

```rust
pub fn merge_patterns(
    &self,
    p1: &DistributedPattern,
    p2: &DistributedPattern,
    similarity: f32,
) -> ResolutionResult {
    let c1 = p1.confidence.raw();
    let c2 = p2.confidence.raw();
    let total_conf = c1 + c2;

    // Confidence-weighted embedding merge
    let merged_embedding = self.merge_embeddings(&p1.embedding, &p2.embedding, c1, c2);

    // Union of episode sets
    let mut merged_episodes = p1.episode_ids.clone();
    merged_episodes.extend(p2.episode_ids.iter().cloned());

    // Compute merged confidence with penalty for uncertainty
    let similarity_penalty = (1.0 - similarity) * self.max_confidence_penalty;
    let base_confidence = total_conf / 2.0;
    let adjusted_confidence = base_confidence * (1.0 - similarity_penalty);
    let final_confidence = Confidence::from_raw(
        adjusted_confidence.max(self.min_confidence.raw())
    );

    // Information loss = semantic divergence
    let information_loss = (1.0 - similarity) * 0.5;
}
```

**Key Variables**:
- `self.max_confidence_penalty: f32` = 0.3 (30% max penalty)
- `self.min_confidence: Confidence` = 0.1 (10% minimum)
- `similarity: f32` = cosine similarity (0.0 - 1.0)

**Example Calculation**:
```
p1.confidence = 0.85, p2.confidence = 0.78, similarity = 0.9
base_confidence = (0.85 + 0.78) / 2 = 0.815
similarity_penalty = (1.0 - 0.9) × 0.3 = 0.03
adjusted = 0.815 × (1.0 - 0.03) = 0.79055
final = max(0.79055, 0.1) = 0.79055
```

### 9.3 Conservative Dual Strategy Confidence

**When similarity is low (<0.60)**, keep both patterns with reduced confidence:

```rust
pub fn apply_conservative_dual(
    &self,
    p1: &DistributedPattern,
    p2: &DistributedPattern,
    similarity: f32,
) -> ResolutionResult {
    let uncertainty_penalty = (1.0 - similarity) * self.max_confidence_penalty;

    let mut adjusted_p1 = p1.clone();
    let mut adjusted_p2 = p2.clone();

    adjusted_p1.confidence = Confidence::from_raw(
        (p1.confidence.raw() * (1.0 - uncertainty_penalty))
            .max(self.min_confidence.raw())
    );

    adjusted_p2.confidence = Confidence::from_raw(
        (p2.confidence.raw() * (1.0 - uncertainty_penalty))
            .max(self.min_confidence.raw())
    );

    ResolutionResult {
        resolved_patterns: vec![adjusted_p1, adjusted_p2],
        strategy: ResolutionStrategy::ConservativeDual,
        confidence_adjustment: -uncertainty_penalty,
        information_loss: 0.0,
        is_deterministic: true,
    }
}
```

**Example**:
```
p1.confidence = 0.85, p2.confidence = 0.78, similarity = 0.4
uncertainty_penalty = (1.0 - 0.4) × 0.3 = 0.18
adjusted_p1 = 0.85 × (1.0 - 0.18) = 0.697
adjusted_p2 = 0.78 × (1.0 - 0.18) = 0.639
Both patterns kept with reduced confidence
```

### 9.4 Confidence Voting Strategy

**When confidence difference is large (>0.15)**, higher confidence wins:

```rust
pub fn apply_confidence_voting(
    &self,
    p1: &DistributedPattern,
    p2: &DistributedPattern,
) -> ResolutionResult {
    let (winner, loser) = if p1.confidence > p2.confidence {
        (p1, p2)
    } else {
        (p2, p1)
    };

    let information_loss = loser.confidence.raw();

    ResolutionResult {
        resolved_patterns: vec![winner.clone()],
        strategy: ResolutionStrategy::ConfidenceVoting,
        confidence_adjustment: 0.0,
        information_loss,  // Loser's confidence = discarded information
        is_deterministic: true,
    }
}
```

---

## 10. CONFLICT DETECTION THRESHOLDS

### 10.1 Current Thresholds in Codebase

**Pattern Clustering** (pattern_detector.rs:19):
```rust
pub struct PatternDetectionConfig {
    pub similarity_threshold: f32,  // Default: 0.8
}
```

**Semantic Priming** (semantic.rs:64):
```rust
semantic_similarity_threshold: f32,  // Default: 0.6
```

### 10.2 Task 008 Recommended Thresholds

**From Task 008 spec** (Lines 304-312):

```rust
pub struct ConflictDetector {
    merge_similarity_threshold: f32,        // Default: 0.85
    overlap_threshold: f32,                  // Default: 0.5
    confidence_voting_threshold: f32,        // Default: 0.15
}
```

**Threshold Justification**:
- **0.85 merge**: Similar to biological CA3 pattern completion threshold
- **0.50 overlap**: Half of episodes shared = significant commonality
- **0.15 confidence**: 15% difference = meaningful divergence

### 10.3 Conflict Classification Rules

**From Task 008 spec (Lines 420-448)**:

```rust
fn classify_conflict(
    &self,
    local: &DistributedPattern,
    remote: &DistributedPattern,
    similarity: f32,
    episode_overlap: f32,
) -> ConflictType {
    // Same pattern ID = concurrent update
    if local.pattern_id == remote.pattern_id {
        return ConflictType::ConcurrentUpdate;
    }

    // High overlap + high similarity = divergent consolidation
    if episode_overlap > self.overlap_threshold 
        && similarity > self.merge_similarity_threshold {
        return ConflictType::DivergentConsolidation;
    }

    // High similarity + low overlap = concurrent creation
    if similarity > self.merge_similarity_threshold && episode_overlap < 0.3 {
        return ConflictType::ConcurrentCreation;
    }

    // Some overlap + low similarity = episode ownership conflict
    if episode_overlap > 0.0 && similarity < 0.6 {
        return ConflictType::EpisodeOwnership;
    }

    ConflictType::NoConflict
}
```

---

## 11. FILES TO CREATE AND MODIFY

### 11.1 Files to CREATE (6 new files)

1. **`engram-core/src/cluster/conflict/mod.rs`**
   - Exports all conflict resolution types
   - VectorClock, DistributedPattern, ConflictDetection, ResolutionResult

2. **`engram-core/src/cluster/conflict/vector_clock.rs`** (or inline in mod.rs)
   - VectorClock implementation
   - CausalOrder enum
   - ~100 lines

3. **`engram-core/src/cluster/conflict/detection.rs`**
   - ConflictDetector struct
   - Similarity/overlap computation
   - ~200 lines

4. **`engram-core/src/cluster/conflict/merger.rs`**
   - PatternMerger struct
   - All 6 merging strategies
   - ~250 lines

5. **`engram-core/src/cluster/conflict/resolver.rs`**
   - ConflictResolver orchestrator
   - Metrics integration hooks
   - ~150 lines

6. **`engram-core/tests/conflict_resolution_tests.rs`**
   - Unit tests (vector clock, detection, merge)
   - Property-based tests (proptest)
   - Integration tests
   - ~500 lines

### 11.2 Files to MODIFY (4 files)

1. **`engram-core/src/completion/consolidation.rs`**
   - **Line 47**: Add `vector_clock: VectorClock` field
   - **Line 49**: Add `origin_node_id: String` field
   - **Line 50**: Add `generation: u32` field
   - **Line 55**: Change `source_episodes: Vec<String>` to `HashSet<String>`
   - Update `From<EpisodicPattern>` impl (Line 67-81)

2. **`engram-core/src/consolidation/service.rs`**
   - Hook point to initialize VectorClock on pattern creation
   - Set origin_node_id from service context
   - Initialize generation to 0

3. **`engram-core/src/metrics/cluster_consolidation.rs`**
   - Add ConflictResolutionMetrics struct
   - Add recording methods for resolution events
   - Track by conflict type and resolution strategy

4. **`engram-core/Cargo.toml`**
   - Add `proptest = "1.4"` to [dev-dependencies] if not present
   - Verify serde, chrono, dashmap are available

### 11.3 Directory Structure

```
engram-core/src/
├── cluster/                  (NEW)
│   └── conflict/             (NEW)
│       ├── mod.rs            (NEW)
│       ├── vector_clock.rs   (NEW)
│       ├── detection.rs      (NEW)
│       ├── merger.rs         (NEW)
│       └── resolver.rs       (NEW)
├── completion/
│   └── consolidation.rs      (MODIFY)
├── consolidation/
│   └── service.rs            (MODIFY)
├── metrics/
│   └── cluster_consolidation.rs (MODIFY)
└── lib.rs                    (exports cluster module)
```

---

## 12. DEPENDENCY ANALYSIS

### 12.1 Already Available (no new crates needed)

**Crates used by existing code**:
- `serde` / `serde_json` - Serialization
- `chrono` - DateTime operations
- `dashmap` - Concurrent HashMaps
- `tokio` - Async runtime (if gossip is async)
- `sha2` - Hashing (used in Task 007 Merkle trees)
- `atomic_float` - Atomic f32 operations

### 12.2 New Dependency

**For property-based testing**:
```toml
[dev-dependencies]
proptest = "1.4"
```

### 12.3 Standard Library Usage

- `std::collections::{HashMap, HashSet}` - Data structures
- `std::sync::atomic::Ordering` - Atomic operations
- `std::hash::{Hash, Hasher}` - Hashing for consistency

---

## 13. INTEGRATION WITH TASK 007 (GOSSIP)

### 13.1 State Transfer Protocol

**From Task 007 spec** (Lines 82-98):

```rust
struct ConsolidationState {
    patterns: HashMap<PatternId, SemanticPattern>,
    last_consolidation: Timestamp,
    consolidation_generation: u64,
}

struct SemanticPattern {
    id: PatternId,
    embedding: [f32; 768],
    confidence: f32,
    citation_count: u32,
    citations: Vec<EpisodeId>,
    created_at: Timestamp,
    updated_at: Timestamp,
    version: VectorClock,  // TASK 008 provides this
}
```

### 13.2 Gossip Round Integration

**Sequence**:
1. Node A exchanges Merkle roots with Node B (Task 007)
2. Roots differ → perform delta sync
3. Node A sends: `remote_pattern: SemanticPattern`
4. Node B receives and calls:
   ```rust
   let resolution = resolver.resolve(&local_pattern, &remote_pattern);
   local_pattern = resolution.resolved_patterns[0].clone();
   ```
5. Update vector clock for next round
6. Continue gossip

### 13.3 Convergence Guarantee

**Mathematical**:
- Each gossip round: fraction with update ≥ 1 - (1-1/N)^k
- For N=100, k=10: >99.9% probability all nodes have update
- Deterministic merging ensures all nodes converge to same state
- Property-based tests verify commutativity and associativity

---

## 14. TESTING STRATEGY BASED ON CODEBASE PATTERNS

### 14.1 Unit Test Pattern (from existing code)

**Location**: Tests typically inline in same file with `#[cfg(test)]` modules

**Example Pattern** (from semantic.rs):
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let similarity = SemanticPrimingEngine::cosine_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 0.001);
    }
}
```

### 14.2 Determinism Tests (Kahan Summation Pattern)

**From pattern_detector.rs**:
The codebase uses Kahan summation specifically to ensure deterministic FP arithmetic. Task 008 tests should verify:

```rust
#[test]
fn test_merge_patterns_deterministic() {
    let merger = PatternMerger::new();
    let p1 = create_test_pattern(...);
    let p2 = create_test_pattern(...);

    let result1 = merger.merge_patterns(&p1, &p2, 0.9);
    let result2 = merger.merge_patterns(&p2, &p1, 0.9);  // Swap order

    // Results should be very similar (allow floating-point tolerance)
    assert!((result1.resolved_patterns[0].confidence.raw() 
             - result2.resolved_patterns[0].confidence.raw()).abs() < 0.001);
}
```

### 14.3 Property-Based Testing Pattern

**From codebase**: Not currently used, but Task 008 introduces:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_resolution_deterministic(
        conf1 in 0.1f32..1.0,
        conf2 in 0.1f32..1.0,
    ) {
        let resolver = ConflictResolver::new();

        let p1 = create_test_pattern("p1", vec!["e1"], conf1, ...);
        let p2 = create_test_pattern("p2", vec!["e2"], conf2, ...);

        let result1 = resolver.resolve(&p1, &p2);
        let result2 = resolver.resolve(&p2, &p1);

        prop_assert_eq!(
            result1.resolved_patterns.len(),
            result2.resolved_patterns.len()
        );
        prop_assert!(result1.is_deterministic);
        prop_assert!(result2.is_deterministic);
    }
}
```

---

## 15. PERFORMANCE CONSIDERATIONS

### 15.1 Embedding Operations Performance

**Current Implementation**:
- Cosine similarity: 768 multiplications + 2 sqrt = ~800 FLOPs
- Embedding average (Kahan): 768×3 iterations = ~2,304 FLOPs
- Typical latency: <0.1ms on modern CPUs

**Task 008 Target**: <3ms for pattern merge, <1ms for similarity

### 15.2 Large Types by Reference

**From CLAUDE.md coding guidelines**:
> Pass large types by reference. For types larger than 64 bytes (like [f32; 768] embeddings), use &T parameters instead of T to avoid expensive copies.

**Application to Task 008**:
```rust
// BAD:
fn compute_similarity(emb1: [f32; 768], emb2: [f32; 768]) -> f32 { ... }

// GOOD:
fn compute_similarity(emb1: &[f32; 768], emb2: &[f32; 768]) -> f32 { ... }
```

### 15.3 Vector Clock Memory Overhead

**From Task 008 spec** (Line 1255):
> Memory overhead: <1KB per vector clock

**Typical Implementation**:
- HashMap<String, u64> for N nodes
- Per node: 24 bytes (String overhead) + 8 bytes (u64) = 32 bytes
- For N=100 nodes: ~3.2 KB (acceptable)

---

## 16. DIAGNOSTIC AND QUALITY CHECKS

### 16.1 Make Quality Requirements

**From CLAUDE.md**:
> CRITICAL: Run `make quality` and fix ALL clippy warnings before proceeding - zero warnings allowed

**Applicable warnings for Task 008**:
- `clippy::unwrap_used` - Avoid unwrap(), use `?` operator
- `clippy::expect_used` - Avoid expect(), return error
- `clippy::float_cmp` - Use `(a - b).abs() < epsilon` for floats
- `clippy::panic` - Don't panic in library code
- `clippy::large_types_passed_by_value` - Pass [f32; 768] by reference

### 16.2 Engram Diagnostics

**From CLAUDE.md**:
> After each test run or engram execution, check diagnostics with `./scripts/engram_diagnostics.sh` and track results in `tmp/engram_diagnostics.log` (prepended)

**Task 008 diagnostics to verify**:
- All tests pass
- No clippy warnings
- Memory usage reasonable
- Performance within targets

---

## SUMMARY TABLE: Key Structures and Locations

| Component | Location | Lines | Status |
|-----------|----------|-------|--------|
| **Confidence Type** | lib.rs | 87-290 | IMPLEMENTED |
| **Episode** | memory.rs | 206-259 | IMPLEMENTED |
| **SemanticPattern** | completion/consolidation.rs | 45-65 | EXISTS (needs fields added) |
| **EpisodicPattern** | consolidation/pattern_detector.rs | 54-73 | IMPLEMENTED |
| **Cosine Similarity** | cognitive/priming/semantic.rs | (method) | IMPLEMENTED |
| **Embedding Average** | consolidation/pattern_detector.rs | 302-324 | IMPLEMENTED |
| **Pattern Strength** | consolidation/pattern_detector.rs | 348+ | IMPLEMENTED |
| **VectorClock** | cluster/conflict/mod.rs | (NEW) | TO CREATE |
| **ConflictDetector** | cluster/conflict/detection.rs | (NEW) | TO CREATE |
| **PatternMerger** | cluster/conflict/merger.rs | (NEW) | TO CREATE |
| **ConflictResolver** | cluster/conflict/resolver.rs | (NEW) | TO CREATE |

---

## NEXT STEPS FOR IMPLEMENTATION

1. **Create Vector Clock** (`cluster/conflict/vector_clock.rs`)
   - Implement VectorClock struct with increment/merge/compare
   - Add CausalOrder enum
   - Add serde support

2. **Extend SemanticPattern**
   - Add vector_clock, origin_node_id, generation fields
   - Update From<EpisodicPattern> impl
   - Change source_episodes to HashSet

3. **Implement Conflict Detection**
   - Create ConflictDetector with similarity/overlap calculations
   - Implement conflict classification logic
   - Add strategy recommendation

4. **Implement Pattern Merging**
   - Create PatternMerger with 6 strategies
   - Add confidence adjustment logic
   - Add embedding normalization

5. **Create Orchestrator**
   - Implement ConflictResolver
   - Integrate with metrics
   - Add determinism validation

6. **Comprehensive Testing**
   - Unit tests for each component
   - Property-based tests for determinism/commutativity
   - Integration tests with Task 007

7. **Performance & Quality**
   - Run `make quality` - fix all clippy warnings
   - Benchmark conflict resolution latency
   - Verify gossip convergence

---

**Report Generated**: 2025-11-01
**Analyst**: Claude Code AI
**Status**: Ready for Implementation
