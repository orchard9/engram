# Task 008: Conflict Resolution for Divergent Consolidations

**Status**: Pending
**Estimated Duration**: 4-5 days
**Dependencies**: Task 007 (Gossip Protocol for Consolidation State)
**Owner**: TBD

## Objective

Implement deterministic, biologically-plausible conflict resolution mechanisms for distributed memory consolidation. When nodes independently consolidate episodic memories into semantic patterns, conflicts arise from timing differences, local context variations, and concurrent operations. Resolution must preserve memory integrity, maintain confidence calibration, and mirror hippocampal-neocortical conflict resolution strategies.

## Research Foundation

### Neuroscience Basis: Hippocampal Conflict Resolution

**Pattern Separation vs Pattern Completion (Leutgeb et al. 2007, Yassa & Stark 2011):**
The hippocampus faces a fundamental conflict: separate similar inputs during encoding (pattern separation via dentate gyrus) versus complete partial inputs during retrieval (pattern completion via CA3 recurrent collaterals). CA3 is the key conflict resolution site - its dynamics exhibit pattern completion for small input changes but allow pattern separation for larger divergences.

**Key insight for Engram:** Use similarity metrics to determine conflict resolution strategy. High similarity (>0.85) triggers pattern completion/merging. Low similarity (<0.60) triggers pattern separation/retention of both patterns.

**Competing Memory Traces (Nader & Hardt 2009, Lee 2009):**
When multiple memory traces compete, the brain resolves conflicts through:
1. **Reconsolidation interference**: Reactivated memories become labile and can be modified by new information
2. **Selective inhibition**: Inhibitory neurons suppress competing engrams during retrieval, enhancing recall stability
3. **Confidence-weighted integration**: More confident (frequently reactivated) memories dominate less confident traces

**Critical boundary:** Reconsolidation creates a 1-6 hour window where competing information can update existing memories. Outside this window, memories resist modification and conflicts persist as separate traces.

**Schema Consistency Effects (Tse et al. 2007, Gilboa & Marlatte 2017):**
The neocortex resolves conflicts by favoring schema-consistent information. When new consolidations conflict with existing semantic knowledge, the brain:
1. Integrates consistent information rapidly (fast neocortical learning)
2. Quarantines inconsistent information for slower integration
3. Adjusts confidence based on consistency with existing schemas

**Biological implementation:** Engram mirrors this with confidence adjustments based on pattern similarity and overlap with existing semantic clusters.

### Distributed Systems Basis: Deterministic Conflict Resolution

**Vector Clocks (Fidge 1988, Mattern 1989):**
Establishes causality in distributed systems without global time. Each node maintains a vector of logical timestamps tracking causal dependencies. For events A and B:
- A happens-before B: vector_clock(A) < vector_clock(B)
- A concurrent with B: vector_clock(A) || vector_clock(B)
- Only concurrent events require semantic merging

**CRDTs (Conflict-Free Replicated Data Types, Shapiro et al. 2011):**
Mathematically proven to converge deterministically. Key properties:
- **Commutativity**: merge(A, B) = merge(B, A)
- **Associativity**: merge(merge(A, B), C) = merge(A, merge(B, C))
- **Idempotence**: merge(A, A) = A

Engram patterns must satisfy these properties for convergence guarantees.

**Operational Transformation (Ellis & Gibbs 1989, Sun & Ellis 1998):**
Resolves concurrent edits by transforming operations based on causality. Applied to Engram: transform consolidation operations when they conflict based on episode overlap and embedding similarity.

**Last-Write-Wins vs Multi-Value vs Semantic Merge:**
- LWW: Simple but loses information (unacceptable for cognitive system)
- Multi-value: Preserves all versions, correct but complex for clients
- Semantic merge: Domain-aware merging using confidence, similarity, overlap (optimal for Engram)

## Technical Specification

### Core Conflict Types

Engram faces four distinct conflict types, each requiring specialized resolution:

**Type 1: Divergent Episode Consolidation**
```
Node A: Episodes {E1, E2, E3} → Pattern P_A (confidence: 0.85, embedding: [0.2, 0.8, ...])
Node B: Episodes {E1, E2, E4} → Pattern P_B (confidence: 0.78, embedding: [0.25, 0.75, ...])

Conflict: Same core episodes but different supporting episodes and embeddings
Resolution: Merge patterns using confidence-weighted embedding averaging, union episode sets
```

**Type 2: Concurrent Pattern Creation**
```
Node A: Creates "morning_routine" pattern from episodes {E1, E2, E3}
Node B: Creates "morning_routine" pattern from episodes {E5, E6, E7}

Conflict: Semantically similar patterns created independently
Resolution: Merge if similarity > threshold, else reduce confidence and keep both
```

**Type 3: Concurrent Updates**
```
Node A: Updates pattern P confidence: 0.7 → 0.9 (vector_clock: [A:5, B:3])
Node B: Updates pattern P confidence: 0.7 → 0.8 (vector_clock: [A:4, B:6])

Conflict: Both updates concurrent (neither happens-before the other)
Resolution: Confidence-weighted averaging with vector clock merge
```

**Type 4: Episode Ownership Conflicts**
```
Node A: Episode E1 consolidated into Pattern P_A
Node B: Episode E1 consolidated into Pattern P_B

Conflict: Same episode claimed by different semantic patterns
Resolution: Multi-membership with confidence split (mirrors overlapping engrams in biology)
```

### Core Data Structures

```rust
// engram-core/src/cluster/conflict/mod.rs

use std::collections::{HashMap, HashSet};
use crate::{Confidence, Embedding};
use chrono::{DateTime, Utc};

/// Vector clock for causality tracking
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VectorClock {
    /// Node ID -> logical timestamp mapping
    clocks: HashMap<String, u64>,
}

impl VectorClock {
    /// Create new vector clock
    #[must_use]
    pub fn new() -> Self {
        Self {
            clocks: HashMap::new(),
        }
    }

    /// Increment this node's clock
    pub fn increment(&mut self, node_id: &str) {
        *self.clocks.entry(node_id.to_string()).or_insert(0) += 1;
    }

    /// Merge with another vector clock (take maximum of each component)
    pub fn merge(&mut self, other: &VectorClock) {
        for (node_id, timestamp) in &other.clocks {
            self.clocks
                .entry(node_id.clone())
                .and_modify(|t| *t = (*t).max(*timestamp))
                .or_insert(*timestamp);
        }
    }

    /// Compare causality relationship
    #[must_use]
    pub fn compare(&self, other: &VectorClock) -> CausalOrder {
        let mut less = false;
        let mut greater = false;

        // Collect all node IDs
        let mut all_nodes: HashSet<&String> = self.clocks.keys().collect();
        all_nodes.extend(other.clocks.keys());

        for node_id in all_nodes {
            let self_ts = self.clocks.get(node_id).copied().unwrap_or(0);
            let other_ts = other.clocks.get(node_id).copied().unwrap_or(0);

            if self_ts < other_ts {
                less = true;
            } else if self_ts > other_ts {
                greater = true;
            }
        }

        match (less, greater) {
            (true, false) => CausalOrder::Before,      // self < other
            (false, true) => CausalOrder::After,       // self > other
            (false, false) => CausalOrder::Equal,      // self = other
            (true, true) => CausalOrder::Concurrent,   // concurrent
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalOrder {
    /// This event happened before the other
    Before,
    /// This event happened after the other
    After,
    /// Events are identical
    Equal,
    /// Events are concurrent (neither before nor after)
    Concurrent,
}

/// Consolidated pattern with distributed metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DistributedPattern {
    /// Unique pattern identifier
    pub pattern_id: String,

    /// Semantic embedding (768-dimensional)
    pub embedding: Embedding,

    /// Confidence score [0.0, 1.0]
    pub confidence: Confidence,

    /// Episodes consolidated into this pattern
    pub episode_ids: HashSet<String>,

    /// Vector clock for causality tracking
    pub vector_clock: VectorClock,

    /// Node that created this pattern
    pub origin_node_id: String,

    /// Timestamp of last update
    pub last_updated: DateTime<Utc>,

    /// Consolidation generation (increments on each reconsolidation)
    pub generation: u32,
}

/// Conflict detection result
#[derive(Debug, Clone)]
pub struct ConflictDetection {
    /// Type of conflict detected
    pub conflict_type: ConflictType,

    /// Patterns involved in conflict
    pub conflicting_patterns: Vec<DistributedPattern>,

    /// Causal relationship between patterns
    pub causal_order: CausalOrder,

    /// Semantic similarity (cosine similarity of embeddings)
    pub similarity: f32,

    /// Episode overlap (Jaccard index)
    pub episode_overlap: f32,

    /// Recommended resolution strategy
    pub recommended_strategy: ResolutionStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    /// Same episodes consolidated differently
    DivergentConsolidation,

    /// Concurrent pattern creation with similar semantics
    ConcurrentCreation,

    /// Concurrent updates to same pattern
    ConcurrentUpdate,

    /// Episode claimed by multiple patterns
    EpisodeOwnership,

    /// No conflict detected
    NoConflict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionStrategy {
    /// Merge patterns using confidence-weighted averaging
    ConfidenceWeightedMerge,

    /// Keep both patterns with reduced confidence
    ConservativeDual,

    /// Select pattern with higher confidence, discard other
    ConfidenceVoting,

    /// Apply vector clock ordering (newer wins)
    VectorClockOrdering,

    /// Allow multi-membership (episode belongs to multiple patterns)
    MultiMembership,

    /// No action needed
    NoAction,
}

/// Result of conflict resolution
#[derive(Debug, Clone)]
pub struct ResolutionResult {
    /// Resolved pattern(s)
    pub resolved_patterns: Vec<DistributedPattern>,

    /// Strategy applied
    pub strategy: ResolutionStrategy,

    /// Confidence adjustment applied
    pub confidence_adjustment: f32,

    /// Information loss metric [0.0, 1.0] (0 = no loss)
    pub information_loss: f32,

    /// Whether resolution was deterministic
    pub is_deterministic: bool,
}
```

### Core Operations

#### 1. Conflict Detection

```rust
// engram-core/src/cluster/conflict/detection.rs

use super::*;

pub struct ConflictDetector {
    /// Similarity threshold for pattern merging (default: 0.85)
    merge_similarity_threshold: f32,

    /// Episode overlap threshold for ownership conflicts (default: 0.5)
    overlap_threshold: f32,

    /// Confidence difference threshold for voting (default: 0.15)
    confidence_voting_threshold: f32,
}

impl ConflictDetector {
    #[must_use]
    pub fn new() -> Self {
        Self {
            merge_similarity_threshold: 0.85,
            overlap_threshold: 0.5,
            confidence_voting_threshold: 0.15,
        }
    }

    /// Detect conflicts between two patterns
    #[must_use]
    pub fn detect_conflict(
        &self,
        local: &DistributedPattern,
        remote: &DistributedPattern,
    ) -> ConflictDetection {
        // Check vector clock causality
        let causal_order = local.vector_clock.compare(&remote.vector_clock);

        // No conflict if one clearly happened before the other
        if matches!(causal_order, CausalOrder::Before | CausalOrder::After) {
            return ConflictDetection {
                conflict_type: ConflictType::NoConflict,
                conflicting_patterns: vec![],
                causal_order,
                similarity: 0.0,
                episode_overlap: 0.0,
                recommended_strategy: ResolutionStrategy::NoAction,
            };
        }

        // Compute semantic similarity (cosine similarity)
        let similarity = self.compute_similarity(&local.embedding, &remote.embedding);

        // Compute episode overlap (Jaccard index)
        let episode_overlap = self.compute_jaccard_overlap(
            &local.episode_ids,
            &remote.episode_ids,
        );

        // Classify conflict type based on patterns
        let conflict_type = self.classify_conflict(
            local,
            remote,
            similarity,
            episode_overlap,
        );

        // Recommend resolution strategy
        let recommended_strategy = self.recommend_strategy(
            conflict_type,
            causal_order,
            similarity,
            episode_overlap,
            local.confidence,
            remote.confidence,
        );

        ConflictDetection {
            conflict_type,
            conflicting_patterns: vec![local.clone(), remote.clone()],
            causal_order,
            similarity,
            episode_overlap,
            recommended_strategy,
        }
    }

    /// Compute cosine similarity between embeddings
    fn compute_similarity(&self, emb1: &Embedding, emb2: &Embedding) -> f32 {
        let dot_product: f32 = emb1.iter()
            .zip(emb2.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        (dot_product / (norm1 * norm2)).clamp(-1.0, 1.0)
    }

    /// Compute Jaccard overlap between episode sets
    fn compute_jaccard_overlap(
        &self,
        episodes1: &HashSet<String>,
        episodes2: &HashSet<String>,
    ) -> f32 {
        let intersection_size = episodes1.intersection(episodes2).count();
        let union_size = episodes1.union(episodes2).count();

        if union_size == 0 {
            return 0.0;
        }

        #[allow(clippy::cast_precision_loss)]
        {
            intersection_size as f32 / union_size as f32
        }
    }

    /// Classify conflict type
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

        // High episode overlap + high similarity = divergent consolidation
        if episode_overlap > self.overlap_threshold && similarity > self.merge_similarity_threshold {
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

    /// Recommend resolution strategy based on conflict characteristics
    fn recommend_strategy(
        &self,
        conflict_type: ConflictType,
        causal_order: CausalOrder,
        similarity: f32,
        episode_overlap: f32,
        local_conf: Confidence,
        remote_conf: Confidence,
    ) -> ResolutionStrategy {
        match conflict_type {
            ConflictType::NoConflict => ResolutionStrategy::NoAction,

            ConflictType::DivergentConsolidation => {
                // High similarity = merge patterns
                if similarity > self.merge_similarity_threshold {
                    ResolutionStrategy::ConfidenceWeightedMerge
                } else {
                    ResolutionStrategy::ConservativeDual
                }
            }

            ConflictType::ConcurrentCreation => {
                // Very high similarity = likely same concept, merge
                if similarity > 0.9 {
                    ResolutionStrategy::ConfidenceWeightedMerge
                } else {
                    // Keep both but reduce confidence to reflect uncertainty
                    ResolutionStrategy::ConservativeDual
                }
            }

            ConflictType::ConcurrentUpdate => {
                let conf_diff = (local_conf.raw() - remote_conf.raw()).abs();

                // Large confidence difference = use voting
                if conf_diff > self.confidence_voting_threshold {
                    ResolutionStrategy::ConfidenceVoting
                } else {
                    // Similar confidence = average
                    ResolutionStrategy::ConfidenceWeightedMerge
                }
            }

            ConflictType::EpisodeOwnership => {
                // Episodes can belong to multiple patterns (biological plausibility)
                ResolutionStrategy::MultiMembership
            }
        }
    }
}
```

#### 2. Pattern Merging (Confidence-Weighted)

```rust
// engram-core/src/cluster/conflict/merger.rs

use super::*;

pub struct PatternMerger {
    /// Minimum confidence after merge (default: 0.1)
    min_confidence: Confidence,

    /// Maximum confidence penalty for uncertainty (default: 0.3)
    max_confidence_penalty: f32,
}

impl PatternMerger {
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_confidence: Confidence::from_raw(0.1),
            max_confidence_penalty: 0.3,
        }
    }

    /// Merge two patterns using confidence-weighted averaging
    ///
    /// Mirrors hippocampal pattern completion: weighted integration
    /// of similar memory traces with confidence reflecting integration quality.
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
        let merged_embedding = self.merge_embeddings(
            &p1.embedding,
            &p2.embedding,
            c1,
            c2,
        );

        // Union of episode sets
        let mut merged_episodes = p1.episode_ids.clone();
        merged_episodes.extend(p2.episode_ids.iter().cloned());

        // Merge vector clocks (take max of each component)
        let mut merged_clock = p1.vector_clock.clone();
        merged_clock.merge(&p2.vector_clock);

        // Compute merged confidence with penalty for uncertainty
        // High similarity = low penalty, low similarity = high penalty
        let similarity_penalty = (1.0 - similarity) * self.max_confidence_penalty;
        let base_confidence = total_conf / 2.0; // Average
        let adjusted_confidence = base_confidence * (1.0 - similarity_penalty);
        let final_confidence = Confidence::from_raw(
            adjusted_confidence.max(self.min_confidence.raw())
        );

        // Compute information loss (1 - similarity captures semantic divergence)
        let information_loss = (1.0 - similarity) * 0.5; // Scale to [0, 0.5]

        // Take newer generation + 1 (merged pattern is new generation)
        let merged_generation = p1.generation.max(p2.generation) + 1;

        let merged_pattern = DistributedPattern {
            pattern_id: p1.pattern_id.clone(), // Keep first pattern's ID
            embedding: merged_embedding,
            confidence: final_confidence,
            episode_ids: merged_episodes,
            vector_clock: merged_clock,
            origin_node_id: p1.origin_node_id.clone(),
            last_updated: Utc::now(),
            generation: merged_generation,
        };

        ResolutionResult {
            resolved_patterns: vec![merged_pattern],
            strategy: ResolutionStrategy::ConfidenceWeightedMerge,
            confidence_adjustment: final_confidence.raw() - c1,
            information_loss,
            is_deterministic: true, // Merge is commutative and deterministic
        }
    }

    /// Merge embeddings using confidence-weighted averaging
    fn merge_embeddings(
        &self,
        emb1: &Embedding,
        emb2: &Embedding,
        conf1: f32,
        conf2: f32,
    ) -> Embedding {
        let total_conf = conf1 + conf2;
        if total_conf == 0.0 {
            return emb1.clone();
        }

        let weight1 = conf1 / total_conf;
        let weight2 = conf2 / total_conf;

        let mut merged = Embedding::new(emb1.len());
        for (i, (e1, e2)) in emb1.iter().zip(emb2.iter()).enumerate() {
            merged[i] = e1 * weight1 + e2 * weight2;
        }

        // Normalize to unit vector (preserves semantic meaning)
        let norm: f32 = merged.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut merged {
                *val /= norm;
            }
        }

        merged
    }

    /// Apply conservative dual strategy: keep both patterns with reduced confidence
    pub fn apply_conservative_dual(
        &self,
        p1: &DistributedPattern,
        p2: &DistributedPattern,
        similarity: f32,
    ) -> ResolutionResult {
        // Reduce confidence based on uncertainty from conflict
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
            information_loss: 0.0, // No information lost
            is_deterministic: true,
        }
    }

    /// Apply confidence voting: higher confidence wins
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

        // Information loss = loser's confidence (represents discarded information)
        let information_loss = loser.confidence.raw();

        ResolutionResult {
            resolved_patterns: vec![winner.clone()],
            strategy: ResolutionStrategy::ConfidenceVoting,
            confidence_adjustment: 0.0,
            information_loss,
            is_deterministic: true, // Deterministic based on confidence ordering
        }
    }

    /// Apply multi-membership: episode belongs to multiple patterns
    pub fn apply_multi_membership(
        &self,
        p1: &DistributedPattern,
        p2: &DistributedPattern,
        episode_overlap: f32,
    ) -> ResolutionResult {
        // Split confidence for overlapping episodes
        let overlap_penalty = episode_overlap * 0.5;

        let mut adjusted_p1 = p1.clone();
        let mut adjusted_p2 = p2.clone();

        adjusted_p1.confidence = Confidence::from_raw(
            (p1.confidence.raw() * (1.0 - overlap_penalty))
                .max(self.min_confidence.raw())
        );

        adjusted_p2.confidence = Confidence::from_raw(
            (p2.confidence.raw() * (1.0 - overlap_penalty))
                .max(self.min_confidence.raw())
        );

        ResolutionResult {
            resolved_patterns: vec![adjusted_p1, adjusted_p2],
            strategy: ResolutionStrategy::MultiMembership,
            confidence_adjustment: -overlap_penalty,
            information_loss: 0.0, // No information lost, reflects biological overlap
            is_deterministic: true,
        }
    }
}
```

#### 3. Conflict Resolution Orchestrator

```rust
// engram-core/src/cluster/conflict/resolver.rs

use super::*;

pub struct ConflictResolver {
    detector: ConflictDetector,
    merger: PatternMerger,

    /// Metrics tracking
    resolution_counter: std::sync::atomic::AtomicU64,
}

impl ConflictResolver {
    #[must_use]
    pub fn new() -> Self {
        Self {
            detector: ConflictDetector::new(),
            merger: PatternMerger::new(),
            resolution_counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Resolve conflict between local and remote pattern
    ///
    /// Guarantees:
    /// 1. Deterministic: same inputs always produce same output
    /// 2. Commutative: resolve(A, B) = resolve(B, A) (after ordering)
    /// 3. Associative: resolve(resolve(A, B), C) = resolve(A, resolve(B, C))
    /// 4. No data loss for high-confidence patterns
    pub fn resolve(
        &self,
        local: &DistributedPattern,
        remote: &DistributedPattern,
    ) -> ResolutionResult {
        // Detect conflict type and recommended strategy
        let detection = self.detector.detect_conflict(local, remote);

        // Track resolution attempt
        self.resolution_counter.fetch_add(
            1,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Apply resolution strategy
        let result = match detection.recommended_strategy {
            ResolutionStrategy::NoAction => {
                // No conflict, keep both (if different IDs) or newer (if same ID)
                self.resolve_no_conflict(local, remote, detection.causal_order)
            }

            ResolutionStrategy::ConfidenceWeightedMerge => {
                self.merger.merge_patterns(local, remote, detection.similarity)
            }

            ResolutionStrategy::ConservativeDual => {
                self.merger.apply_conservative_dual(local, remote, detection.similarity)
            }

            ResolutionStrategy::ConfidenceVoting => {
                self.merger.apply_confidence_voting(local, remote)
            }

            ResolutionStrategy::MultiMembership => {
                self.merger.apply_multi_membership(local, remote, detection.episode_overlap)
            }

            ResolutionStrategy::VectorClockOrdering => {
                self.resolve_by_vector_clock(local, remote, detection.causal_order)
            }
        };

        // Record metrics
        #[cfg(feature = "monitoring")]
        {
            self.record_resolution_metrics(&detection, &result);
        }

        result
    }

    /// Resolve no-conflict case using vector clock ordering
    fn resolve_no_conflict(
        &self,
        local: &DistributedPattern,
        remote: &DistributedPattern,
        causal_order: CausalOrder,
    ) -> ResolutionResult {
        match causal_order {
            CausalOrder::Before => {
                // Local happened before remote, use remote
                ResolutionResult {
                    resolved_patterns: vec![remote.clone()],
                    strategy: ResolutionStrategy::VectorClockOrdering,
                    confidence_adjustment: 0.0,
                    information_loss: 0.0,
                    is_deterministic: true,
                }
            }
            CausalOrder::After => {
                // Local happened after remote, use local
                ResolutionResult {
                    resolved_patterns: vec![local.clone()],
                    strategy: ResolutionStrategy::VectorClockOrdering,
                    confidence_adjustment: 0.0,
                    information_loss: 0.0,
                    is_deterministic: true,
                }
            }
            CausalOrder::Equal => {
                // Identical, keep one
                ResolutionResult {
                    resolved_patterns: vec![local.clone()],
                    strategy: ResolutionStrategy::NoAction,
                    confidence_adjustment: 0.0,
                    information_loss: 0.0,
                    is_deterministic: true,
                }
            }
            CausalOrder::Concurrent => {
                // Should not happen in no-conflict case
                // Fall back to conservative dual
                self.merger.apply_conservative_dual(local, remote, 0.5)
            }
        }
    }

    /// Resolve using vector clock ordering
    fn resolve_by_vector_clock(
        &self,
        local: &DistributedPattern,
        remote: &DistributedPattern,
        causal_order: CausalOrder,
    ) -> ResolutionResult {
        match causal_order {
            CausalOrder::Before => ResolutionResult {
                resolved_patterns: vec![remote.clone()],
                strategy: ResolutionStrategy::VectorClockOrdering,
                confidence_adjustment: 0.0,
                information_loss: local.confidence.raw(), // Discard local
                is_deterministic: true,
            },
            CausalOrder::After => ResolutionResult {
                resolved_patterns: vec![local.clone()],
                strategy: ResolutionStrategy::VectorClockOrdering,
                confidence_adjustment: 0.0,
                information_loss: remote.confidence.raw(), // Discard remote
                is_deterministic: true,
            },
            CausalOrder::Equal | CausalOrder::Concurrent => {
                // Fall back to confidence voting for ties
                self.merger.apply_confidence_voting(local, remote)
            }
        }
    }

    #[cfg(feature = "monitoring")]
    fn record_resolution_metrics(
        &self,
        detection: &ConflictDetection,
        result: &ResolutionResult,
    ) {
        crate::metrics::cluster_consolidation()
            .record_conflict_resolution(
                detection.conflict_type,
                result.strategy,
                result.information_loss,
            );
    }

    /// Get total number of resolutions performed
    #[must_use]
    pub fn resolution_count(&self) -> u64 {
        self.resolution_counter.load(std::sync::atomic::Ordering::Relaxed)
    }
}
```

## Files to Create

1. `engram-core/src/cluster/conflict/mod.rs` - Module exports and core types
2. `engram-core/src/cluster/conflict/detection.rs` - Conflict detection logic
3. `engram-core/src/cluster/conflict/merger.rs` - Pattern merging algorithms
4. `engram-core/src/cluster/conflict/resolver.rs` - Resolution orchestration
5. `engram-core/src/cluster/conflict/vector_clock.rs` - Vector clock implementation (if not in mod.rs)
6. `engram-core/tests/conflict_resolution_tests.rs` - Comprehensive test suite

## Files to Modify

1. `engram-core/src/cluster/gossip/consolidation.rs` - Integrate conflict resolver
2. `engram-core/src/decay/consolidation.rs` - Add vector clock to patterns
3. `engram-core/src/metrics/cluster_consolidation.rs` - Add conflict metrics
4. `engram-core/Cargo.toml` - Add dependencies if needed

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock_causality() {
        let mut vc1 = VectorClock::new();
        let mut vc2 = VectorClock::new();

        vc1.increment("node_a");
        vc1.increment("node_a");
        // vc1: {node_a: 2}

        vc2.increment("node_b");
        // vc2: {node_b: 1}

        // Concurrent events
        assert_eq!(vc1.compare(&vc2), CausalOrder::Concurrent);

        vc2.merge(&vc1);
        // vc2: {node_a: 2, node_b: 1}

        vc2.increment("node_b");
        // vc2: {node_a: 2, node_b: 2}

        // vc2 happened after vc1
        assert_eq!(vc1.compare(&vc2), CausalOrder::Before);
        assert_eq!(vc2.compare(&vc1), CausalOrder::After);
    }

    #[test]
    fn test_conflict_detection_divergent_consolidation() {
        let detector = ConflictDetector::new();

        let p1 = create_test_pattern(
            "pattern_1",
            vec!["e1", "e2", "e3"],
            0.85,
            vec![0.5, 0.5, 0.0], // embedding
        );

        let p2 = create_test_pattern(
            "pattern_2",
            vec!["e1", "e2", "e4"],
            0.78,
            vec![0.5, 0.5, 0.1], // similar embedding
        );

        let detection = detector.detect_conflict(&p1, &p2);

        assert_eq!(detection.conflict_type, ConflictType::DivergentConsolidation);
        assert!(detection.similarity > 0.8);
        assert!(detection.episode_overlap > 0.5);
        assert_eq!(
            detection.recommended_strategy,
            ResolutionStrategy::ConfidenceWeightedMerge
        );
    }

    #[test]
    fn test_merge_patterns_deterministic() {
        let merger = PatternMerger::new();

        let p1 = create_test_pattern(
            "pattern_1",
            vec!["e1", "e2"],
            0.8,
            vec![1.0, 0.0],
        );

        let p2 = create_test_pattern(
            "pattern_2",
            vec!["e3", "e4"],
            0.6,
            vec![0.0, 1.0],
        );

        let result1 = merger.merge_patterns(&p1, &p2, 0.9);
        let result2 = merger.merge_patterns(&p2, &p1, 0.9);

        // Merged embedding should be similar (order shouldn't matter much)
        let emb1 = &result1.resolved_patterns[0].embedding;
        let emb2 = &result2.resolved_patterns[0].embedding;

        let similarity = cosine_similarity(emb1, emb2);
        assert!(similarity > 0.99, "Merge should be nearly commutative");

        // Both should merge to 4 episodes
        assert_eq!(result1.resolved_patterns[0].episode_ids.len(), 4);
        assert_eq!(result2.resolved_patterns[0].episode_ids.len(), 4);
    }

    #[test]
    fn test_confidence_adjustment_on_merge() {
        let merger = PatternMerger::new();

        let p1 = create_test_pattern("p1", vec!["e1"], 0.9, vec![1.0, 0.0]);
        let p2 = create_test_pattern("p2", vec!["e2"], 0.7, vec![1.0, 0.0]);

        // High similarity = low penalty
        let result_high_sim = merger.merge_patterns(&p1, &p2, 0.95);
        let conf_high = result_high_sim.resolved_patterns[0].confidence.raw();

        // Low similarity = high penalty
        let result_low_sim = merger.merge_patterns(&p1, &p2, 0.5);
        let conf_low = result_low_sim.resolved_patterns[0].confidence.raw();

        // Lower similarity should result in lower confidence
        assert!(conf_high > conf_low);

        // Average should be around 0.8, but with penalty
        assert!(conf_high < 0.8);
        assert!(conf_low < 0.6);
    }

    #[test]
    fn test_conservative_dual_preserves_information() {
        let merger = PatternMerger::new();

        let p1 = create_test_pattern("p1", vec!["e1"], 0.85, vec![1.0, 0.0]);
        let p2 = create_test_pattern("p2", vec!["e2"], 0.80, vec![0.0, 1.0]);

        let result = merger.apply_conservative_dual(&p1, &p2, 0.3);

        // Should keep both patterns
        assert_eq!(result.resolved_patterns.len(), 2);

        // No information loss
        assert_eq!(result.information_loss, 0.0);

        // Confidence should be reduced due to uncertainty
        assert!(result.resolved_patterns[0].confidence.raw() < 0.85);
        assert!(result.resolved_patterns[1].confidence.raw() < 0.80);
    }
}
```

### Property-Based Tests (Determinism Proof)

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_resolution_deterministic(
        conf1 in 0.1f32..1.0,
        conf2 in 0.1f32..1.0,
        emb1_x in -1.0f32..1.0,
        emb2_x in -1.0f32..1.0,
    ) {
        let resolver = ConflictResolver::new();

        let p1 = create_test_pattern(
            "p1",
            vec!["e1"],
            conf1,
            vec![emb1_x, 0.0],
        );

        let p2 = create_test_pattern(
            "p2",
            vec!["e2"],
            conf2,
            vec![emb2_x, 0.0],
        );

        // Resolve in both orders
        let result1 = resolver.resolve(&p1, &p2);
        let result2 = resolver.resolve(&p2, &p1);

        // Same number of output patterns
        prop_assert_eq!(
            result1.resolved_patterns.len(),
            result2.resolved_patterns.len()
        );

        // Determinism flag should always be true
        prop_assert!(result1.is_deterministic);
        prop_assert!(result2.is_deterministic);
    }

    #[test]
    fn test_merge_associativity(
        conf1 in 0.1f32..1.0,
        conf2 in 0.1f32..1.0,
        conf3 in 0.1f32..1.0,
    ) {
        let merger = PatternMerger::new();

        let p1 = create_test_pattern("p1", vec!["e1"], conf1, vec![1.0, 0.0]);
        let p2 = create_test_pattern("p2", vec!["e2"], conf2, vec![0.0, 1.0]);
        let p3 = create_test_pattern("p3", vec!["e3"], conf3, vec![0.5, 0.5]);

        // (p1 ⊕ p2) ⊕ p3
        let r12 = merger.merge_patterns(&p1, &p2, 0.9);
        let merged12 = &r12.resolved_patterns[0];
        let r123_left = merger.merge_patterns(merged12, &p3, 0.9);

        // p1 ⊕ (p2 ⊕ p3)
        let r23 = merger.merge_patterns(&p2, &p3, 0.9);
        let merged23 = &r23.resolved_patterns[0];
        let r123_right = merger.merge_patterns(&p1, merged23, 0.9);

        // Should produce similar results (within floating point tolerance)
        let conf_left = r123_left.resolved_patterns[0].confidence.raw();
        let conf_right = r123_right.resolved_patterns[0].confidence.raw();

        prop_assert!((conf_left - conf_right).abs() < 0.05);
    }

    #[test]
    fn test_no_confidence_inflation(
        conf1 in 0.1f32..1.0,
        conf2 in 0.1f32..1.0,
    ) {
        let resolver = ConflictResolver::new();

        let p1 = create_test_pattern("p1", vec!["e1"], conf1, vec![1.0, 0.0]);
        let p2 = create_test_pattern("p2", vec!["e2"], conf2, vec![1.0, 0.0]);

        let result = resolver.resolve(&p1, &p2);

        // Merged confidence should never exceed max of inputs
        for resolved in &result.resolved_patterns {
            prop_assert!(resolved.confidence.raw() <= conf1.max(conf2));
        }
    }
}
```

### Integration Tests

```rust
// engram-core/tests/conflict_resolution_integration.rs

#[tokio::test]
async fn test_gossip_conflict_resolution_integration() {
    // Simulate two nodes consolidating independently
    let node1 = TestNode::new("node1").await;
    let node2 = TestNode::new("node2").await;

    // Both nodes receive same episodes
    let episodes = vec![
        create_episode("e1", "woke up"),
        create_episode("e2", "brushed teeth"),
        create_episode("e3", "made coffee"),
    ];

    for episode in &episodes {
        node1.store_episode(episode.clone()).await;
        node2.store_episode(episode.clone()).await;
    }

    // Trigger consolidation on both nodes
    node1.trigger_consolidation().await;
    node2.trigger_consolidation().await;

    // Wait for consolidation to complete
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Get consolidated patterns (may differ due to timing)
    let pattern1 = node1.get_patterns().await;
    let pattern2 = node2.get_patterns().await;

    // Simulate gossip exchange
    let resolver = ConflictResolver::new();
    let result = resolver.resolve(&pattern1[0], &pattern2[0]);

    // Should produce converged pattern
    assert!(result.is_deterministic);
    assert!(result.information_loss < 0.2); // Low information loss

    // Apply resolution on both nodes
    node1.apply_resolution(&result).await;
    node2.apply_resolution(&result).await;

    // Both nodes should now have identical patterns
    let final_pattern1 = node1.get_patterns().await;
    let final_pattern2 = node2.get_patterns().await;

    assert_eq!(final_pattern1[0].pattern_id, final_pattern2[0].pattern_id);
    assert_eq!(final_pattern1[0].episode_ids, final_pattern2[0].episode_ids);
    assert!((final_pattern1[0].confidence.raw() - final_pattern2[0].confidence.raw()).abs() < 0.001);
}
```

## Dependencies

Add to `engram-core/Cargo.toml`:

```toml
[dependencies]
# Already have: chrono, dashmap, serde, tokio

# For property-based testing
[dev-dependencies]
proptest = "1.4"
```

## Acceptance Criteria

1. **Determinism**: Property tests prove resolution is deterministic for all conflict types
2. **Commutativity**: resolve(A, B) produces equivalent result to resolve(B, A)
3. **Associativity**: resolve(resolve(A, B), C) ≈ resolve(A, resolve(B, C)) (within tolerance)
4. **No Data Loss**: Conservative strategies (dual, multi-membership) preserve all information
5. **Confidence Calibration**: Merged confidence accurately reflects uncertainty (validated via simulation)
6. **Convergence**: 100 nodes with divergent consolidations converge within 10 gossip rounds
7. **Biological Plausibility**: Resolution strategies mirror hippocampal pattern completion/separation
8. **Performance**: Conflict resolution completes in <5ms per conflict (single-threaded)

## Biological Plausibility Validation

### Hippocampal Pattern Separation/Completion Analogy

**DG Pattern Separation → ConflictType::ConcurrentCreation (low similarity)**
- Biology: Dentate gyrus orthogonalizes similar inputs to prevent interference
- Engram: Low similarity (<0.6) triggers ConservativeDual (keep both patterns)
- Validation: Distinct patterns with low overlap should not merge

**CA3 Pattern Completion → ConflictType::DivergentConsolidation (high similarity)**
- Biology: CA3 recurrent collaterals complete partial cues into full memory
- Engram: High similarity (>0.85) triggers ConfidenceWeightedMerge
- Validation: Similar patterns should merge into unified semantic representation

**Reconsolidation Interference → ConcurrentUpdate**
- Biology: Reactivated memories become labile and update with new information
- Engram: Confidence voting or averaging based on temporal proximity
- Validation: More recent/confident updates should dominate

**Overlapping Engrams → EpisodeOwnership MultiMembership**
- Biology: Single episode can participate in multiple memory traces
- Engram: Episodes allowed to belong to multiple patterns with confidence split
- Validation: Episode membership should not be exclusive

## Performance Targets

- Conflict detection: <1ms per pattern pair
- Pattern merge: <3ms for 768-dimensional embeddings
- Full resolution: <5ms per conflict
- Gossip convergence: 10 rounds for 100-node cluster (60 seconds at 6s interval)
- Memory overhead: <1KB per vector clock

## Metrics to Track

```rust
// engram-core/src/metrics/cluster_consolidation.rs

pub struct ClusterConsolidationMetrics {
    /// Total conflicts detected
    pub total_conflicts: Counter,

    /// Conflicts by type
    pub conflicts_by_type: HashMap<ConflictType, Counter>,

    /// Resolutions by strategy
    pub resolutions_by_strategy: HashMap<ResolutionStrategy, Counter>,

    /// Average information loss per resolution
    pub avg_information_loss: Gauge,

    /// Resolution latency histogram
    pub resolution_latency: Histogram,

    /// Convergence time (gossip rounds to convergence)
    pub convergence_rounds: Histogram,
}
```

## Next Steps

After completing this task:
- **Task 009 (Distributed Query Execution)**: Use resolved patterns for distributed recall
- **Task 010 (Network Partition Testing)**: Validate conflict resolution under partitions
- **Task 011 (Jepsen Testing)**: Formally verify convergence properties

## References

1. Fidge, C. J. (1988). Timestamps in message-passing systems that preserve the partial ordering.
2. Shapiro, M., Preguiça, N., Baquero, C., & Zawirski, M. (2011). Conflict-free replicated data types.
3. Nader, K., & Hardt, O. (2009). A single standard for memory: the case for reconsolidation.
4. Lee, J. L. (2009). Reconsolidation: maintaining memory relevance.
5. Leutgeb, J. K., et al. (2007). Pattern separation in the dentate gyrus and CA3 of the hippocampus.
6. Yassa, M. A., & Stark, C. E. (2011). Pattern separation in the hippocampus.
7. Tse, D., et al. (2007). Schemas and memory consolidation.
8. Gilboa, A., & Marlatte, H. (2017). Neurobiology of schemas and schema-mediated memory.
