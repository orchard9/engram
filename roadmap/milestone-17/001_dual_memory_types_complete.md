# Task 001: Dual Memory Type Definitions

## Objective
Define the core type system for dual memory architecture including MemoryNodeType enum, DualMemoryNode struct, and type conversion utilities with enterprise-grade concurrency, cache optimization, and zero-copy semantics.

## Background
Currently Engram only supports episodic memories. We need to introduce a type system that distinguishes between episodes (specific events) and concepts (generalized patterns) while maintaining backwards compatibility.

The existing codebase has:
- Memory struct at `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/memory.rs` (768-dim embeddings, AtomicF32 activation, Confidence type)
- Episode struct with rich temporal/spatial metadata and DecayFunction support
- UnifiedMemoryGraph at `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/memory_graph/graph.rs`
- Backend traits (MemoryBackend, GraphBackend) at `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/memory_graph/traits.rs`
- DashMapBackend using lock-free DashMap + AtomicF32 for concurrent activation updates
- Cache-aligned HnswNode (#[repr(align(64))]) for false sharing prevention

## Requirements
1. Define `MemoryNodeType` enum with Episode and Concept variants
2. Create `DualMemoryNode` wrapper struct that includes node type
3. Implement conversion traits between Memory and DualMemoryNode
4. Add feature flag `dual_memory_types` to gate new functionality
5. Ensure zero-copy conversions where possible
6. Design cache-optimal memory layout for SIMD operations
7. Support lock-free atomic updates for consolidation_score and instance_count

## Technical Specification

### Files to Create

#### `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/memory/dual_types.rs`
Core type definitions with cache-line alignment and atomic operations:

```rust
//! Dual memory type system for episode-concept architecture
//!
//! This module introduces MemoryNodeType to distinguish between episodic
//! and semantic memory representations with lock-free concurrent access.

use crate::{Confidence, EMBEDDING_DIM};
use atomic_float::AtomicF32;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};
use uuid::Uuid;

/// Unique identifier for episodes (distinct from node UUID)
pub type EpisodeId = String;

/// Memory node type discriminator with type-specific metadata.
///
/// Layout is optimized for cache locality - frequently accessed fields
/// (discriminant, atomic counters) are placed first to share cache lines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryNodeType {
    /// Episodic memory: specific event with temporal/spatial context
    Episode {
        /// Reference to original Episode struct for detailed metadata
        episode_id: EpisodeId,

        /// Current strength/activation (0.0 = fully decayed, 1.0 = maximally active)
        strength: f32,

        /// Consolidation progress (0.0 = pure episode, 1.0 = ready for concept formation)
        /// Updated atomically during clustering analysis
        #[serde(skip)]
        consolidation_score: AtomicF32,

        /// Serialization field for consolidation score
        #[serde(rename = "consolidation_score")]
        consolidation_score_value: f32,
    },

    /// Semantic concept: generalized pattern extracted from episodic clusters
    Concept {
        /// Centroid embedding computed from clustered episodes
        /// Stored inline for cache efficiency during similarity search
        #[serde(with = "crate::memory::embedding_serde")]
        centroid: [f32; EMBEDDING_DIM],

        /// Coherence score: tightness of episode cluster (0.0 = diffuse, 1.0 = tight)
        /// Higher coherence = more reliable generalization
        coherence: f32,

        /// Number of episodes contributing to this concept
        /// Updated atomically as new episodes bind to concept
        #[serde(skip)]
        instance_count: AtomicU32,

        /// Serialization field for instance count
        #[serde(rename = "instance_count")]
        instance_count_value: u32,

        /// When this concept was first formed (for decay calculations)
        formation_time: DateTime<Utc>,
    }
}

impl MemoryNodeType {
    /// Check if this is an episode node
    #[must_use]
    pub const fn is_episode(&self) -> bool {
        matches!(self, Self::Episode { .. })
    }

    /// Check if this is a concept node
    #[must_use]
    pub const fn is_concept(&self) -> bool {
        matches!(self, Self::Concept { .. })
    }

    /// Get consolidation score if this is an episode (thread-safe)
    #[must_use]
    pub fn consolidation_score(&self) -> Option<f32> {
        match self {
            Self::Episode { consolidation_score, .. } => {
                Some(consolidation_score.load(Ordering::Relaxed))
            }
            Self::Concept { .. } => None,
        }
    }

    /// Update consolidation score atomically (only for episodes)
    pub fn update_consolidation_score(&self, new_score: f32) -> bool {
        match self {
            Self::Episode { consolidation_score, .. } => {
                consolidation_score.store(new_score.clamp(0.0, 1.0), Ordering::Release);
                true
            }
            Self::Concept { .. } => false,
        }
    }

    /// Get instance count if this is a concept (thread-safe)
    #[must_use]
    pub fn instance_count(&self) -> Option<u32> {
        match self {
            Self::Concept { instance_count, .. } => {
                Some(instance_count.load(Ordering::Relaxed))
            }
            Self::Episode { .. } => None,
        }
    }

    /// Increment instance count atomically (only for concepts)
    pub fn increment_instances(&self) -> bool {
        match self {
            Self::Concept { instance_count, .. } => {
                instance_count.fetch_add(1, Ordering::AcqRel);
                true
            }
            Self::Episode { .. } => false,
        }
    }

    /// Prepare for serialization by syncing atomic fields to their value counterparts
    pub fn prepare_serialization(&mut self) {
        match self {
            Self::Episode { consolidation_score, consolidation_score_value, .. } => {
                *consolidation_score_value = consolidation_score.load(Ordering::Relaxed);
            }
            Self::Concept { instance_count, instance_count_value, .. } => {
                *instance_count_value = instance_count.load(Ordering::Relaxed);
            }
        }
    }

    /// Restore atomic fields after deserialization
    pub fn restore_atomics(&mut self) {
        match self {
            Self::Episode { consolidation_score, consolidation_score_value, .. } => {
                consolidation_score.store(*consolidation_score_value, Ordering::Relaxed);
            }
            Self::Concept { instance_count, instance_count_value, .. } => {
                instance_count.store(*instance_count_value, Ordering::Relaxed);
            }
        }
    }
}

/// Dual memory node wrapping existing Memory with type discrimination.
///
/// Memory layout is cache-optimized:
/// - 64-byte alignment prevents false sharing on multi-core systems
/// - Hot fields (id, node_type discriminant, embedding ptr) fit in single cache line
/// - Atomic fields use Relaxed ordering for hot path, Release for visibility
#[repr(C)]
#[repr(align(64))]
#[derive(Debug)]
pub struct DualMemoryNode {
    /// Unique node identifier (UUID for graph operations)
    pub id: Uuid,

    /// Memory type discriminator with type-specific metadata
    pub node_type: MemoryNodeType,

    /// Dense 768-dimensional embedding vector
    /// For episodes: original event embedding
    /// For concepts: centroid of clustered episodes
    #[allow(clippy::large_types_passed_by_value)]
    pub embedding: [f32; EMBEDDING_DIM],

    /// Current activation level (thread-safe, lock-free updates)
    #[allow(dead_code)]
    activation: AtomicF32,

    /// Cognitive confidence in this memory's reliability
    pub confidence: Confidence,

    /// Last access timestamp for decay calculations
    pub last_access: DateTime<Utc>,

    /// Node creation timestamp
    pub created_at: DateTime<Utc>,
}

impl DualMemoryNode {
    /// Create a new dual memory node from episode
    #[must_use]
    #[allow(clippy::large_types_passed_by_value)]
    pub fn new_episode(
        id: Uuid,
        episode_id: EpisodeId,
        embedding: [f32; EMBEDDING_DIM],
        confidence: Confidence,
        strength: f32,
    ) -> Self {
        let now = Utc::now();
        Self {
            id,
            node_type: MemoryNodeType::Episode {
                episode_id,
                strength,
                consolidation_score: AtomicF32::new(0.0),
                consolidation_score_value: 0.0,
            },
            embedding,
            activation: AtomicF32::new(strength),
            confidence,
            last_access: now,
            created_at: now,
        }
    }

    /// Create a new dual memory node from concept
    #[must_use]
    #[allow(clippy::large_types_passed_by_value)]
    pub fn new_concept(
        id: Uuid,
        centroid: [f32; EMBEDDING_DIM],
        coherence: f32,
        initial_instance_count: u32,
        confidence: Confidence,
    ) -> Self {
        let now = Utc::now();
        Self {
            id,
            node_type: MemoryNodeType::Concept {
                centroid,
                coherence,
                instance_count: AtomicU32::new(initial_instance_count),
                instance_count_value: initial_instance_count,
                formation_time: now,
            },
            embedding: centroid,
            activation: AtomicF32::new(0.0),
            confidence,
            last_access: now,
            created_at: now,
        }
    }

    /// Get current activation level (thread-safe)
    #[must_use]
    pub fn activation(&self) -> f32 {
        self.activation.load(Ordering::Relaxed)
    }

    /// Set activation level (thread-safe)
    pub fn set_activation(&self, value: f32) {
        self.activation.store(value.clamp(0.0, 1.0), Ordering::Release);
    }

    /// Add to current activation (thread-safe)
    pub fn add_activation(&self, delta: f32) {
        let current = self.activation();
        let new_value = (current + delta).clamp(0.0, 1.0);
        self.set_activation(new_value);
    }

    /// Check if this is an episode node
    #[must_use]
    pub fn is_episode(&self) -> bool {
        self.node_type.is_episode()
    }

    /// Check if this is a concept node
    #[must_use]
    pub fn is_concept(&self) -> bool {
        self.node_type.is_concept()
    }
}
```

#### `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/memory/conversions.rs`
Zero-copy conversion implementations:

```rust
//! Conversion utilities between Memory and DualMemoryNode
//!
//! These conversions support gradual migration from pure episodic to dual memory.
//! Zero-copy semantics are used where possible via Arc and reference counting.

use crate::memory::{Memory, Episode};
use super::dual_types::{DualMemoryNode, MemoryNodeType, EpisodeId};
use std::sync::Arc;
use uuid::Uuid;

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

impl DualMemoryNode {
    /// Convert back to Memory (for backwards compatibility).
    ///
    /// Concept nodes are represented as Memory with centroid embedding.
    /// This enables gradual rollout without breaking existing APIs.
    #[must_use]
    pub fn to_memory(&self) -> Memory {
        let mut memory = Memory::new(
            self.id.to_string(),
            self.embedding,
            self.confidence,
        );

        memory.set_activation(self.activation());
        memory.last_access = self.last_access;
        memory.created_at = self.created_at;

        memory
    }

    /// Create from Episode struct (existing Engram type).
    ///
    /// This is the primary ingestion path for new memories.
    #[must_use]
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
        let memory = Memory::new(
            "test-123".to_string(),
            [0.5f32; 768],
            Confidence::HIGH,
        );
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
        let dual = DualMemoryNode::new_concept(
            Uuid::new_v4(),
            [0.2f32; 768],
            0.85,
            5,
            Confidence::HIGH,
        );

        assert_eq!(dual.node_type.instance_count(), Some(5));

        dual.node_type.increment_instances();
        assert_eq!(dual.node_type.instance_count(), Some(6));
    }
}
```

### Files to Modify

#### `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/memory/mod.rs`
Add module exports:
```rust
// Add after existing module declarations:

#[cfg(feature = "dual_memory_types")]
pub mod dual_types;
#[cfg(feature = "dual_memory_types")]
pub mod conversions;

// Add to public exports:
#[cfg(feature = "dual_memory_types")]
pub use dual_types::{DualMemoryNode, MemoryNodeType, EpisodeId};
```

#### `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/lib.rs`
Add feature flag exports:
```rust
// Add after line 67 (after existing pub use statements):

#[cfg(feature = "dual_memory_types")]
pub use memory::{DualMemoryNode, MemoryNodeType, EpisodeId};
```

#### `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/Cargo.toml`
Add feature flag definition (after line 299, in features section):
```toml
# Dual memory architecture (Milestone 17)
dual_memory_types = []
```

### Integration Points with Existing Code

#### UnifiedMemoryGraph Extension
The existing UnifiedMemoryGraph at `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/memory_graph/graph.rs` will be extended in future tasks:

```rust
#[cfg(feature = "dual_memory_types")]
impl<B: MemoryBackend> UnifiedMemoryGraph<B> {
    /// Store a dual memory node (future task 002)
    pub fn store_dual_node(&self, node: DualMemoryNode) -> Result<Uuid, MemoryError> {
        // Convert to Memory for backwards compatibility with existing backends
        let memory = node.to_memory();
        self.store_memory(memory)
    }
}
```

#### Backend Trait Extensions
Future tasks will extend MemoryBackend trait at `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/memory_graph/traits.rs`:

```rust
#[cfg(feature = "dual_memory_types")]
pub trait DualMemoryBackend: MemoryBackend {
    /// Store a dual memory node with type awareness
    fn store_dual(&self, id: Uuid, node: DualMemoryNode) -> Result<(), MemoryError>;

    /// Retrieve with type information preserved
    fn retrieve_dual(&self, id: &Uuid) -> Result<Option<Arc<DualMemoryNode>>, MemoryError>;

    /// Search specifically for episodes or concepts
    fn search_by_type(
        &self,
        embedding: &[f32],
        k: usize,
        node_type: Option<MemoryNodeType>
    ) -> Result<Vec<(Uuid, f32)>, MemoryError>;
}
```

## Implementation Notes

### Cache-Conscious Design
- `#[repr(align(64))]` on DualMemoryNode prevents false sharing across CPU cores
- Hot fields (id, node_type discriminant) share same cache line
- Embedding ptr stored inline for single-deref access during SIMD similarity
- AtomicF32/AtomicU32 use Relaxed ordering for reads (no memory fence)
- Release ordering on writes ensures visibility to consolidation workers

### Lock-Free Atomic Operations
Following existing patterns from DashMapBackend:
- `consolidation_score.store(value, Ordering::Release)` for worker updates
- `instance_count.fetch_add(1, Ordering::AcqRel)` for concurrent binding
- No locks, no mutexes - pure atomic operations only
- Memory ordering semantics match HnswNode at line 20-24 of hnsw_node.rs

### Zero-Copy Semantics
- `From<Memory>` moves embedding array (no copy)
- `to_memory()` copies embedding (required for API compatibility)
- Future Arc<DualMemoryNode> sharing eliminates copies in hot path

### SIMD Considerations
- Embeddings aligned to 64-byte boundary for AVX-512
- Centroid storage inline enables vectorized cosine similarity
- Compatible with existing wide crate SIMD abstractions

### Serialization Strategy
Following existing Memory pattern at memory.rs:38-42:
- Atomic fields skipped via `#[serde(skip)]`
- Separate value fields for persistence (consolidation_score_value, instance_count_value)
- `prepare_serialization()` syncs atomics before write
- `restore_atomics()` reconstructs atomics after read
- Bincode used for WAL (storage/wal.rs:335), serde_json for debugging

## Testing Approach

### Unit Tests (in conversions.rs)
```rust
#[cfg(test)]
mod tests {
    // 1. Type construction and field access
    #[test] fn test_episode_node_creation()
    #[test] fn test_concept_node_creation()

    // 2. Conversion round-trips
    #[test] fn test_memory_to_dual_to_memory_roundtrip()
    #[test] fn test_episode_to_dual_conversion()

    // 3. Atomic operations
    #[test] fn test_consolidation_score_atomic_update()
    #[test] fn test_instance_count_concurrent_increment()

    // 4. Serialization
    #[test] fn test_episode_serialization_roundtrip()
    #[test] fn test_concept_serialization_roundtrip()
}
```

### Property Tests
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_consolidation_score_clamped(score in 0.0f32..2.0f32) {
        let dual = DualMemoryNode::new_episode(...);
        dual.node_type.update_consolidation_score(score);
        let actual = dual.node_type.consolidation_score().unwrap();
        assert!(actual >= 0.0 && actual <= 1.0);
    }

    #[test]
    fn test_coherence_valid_range(coherence in 0.0f32..1.0f32) {
        let dual = DualMemoryNode::new_concept(..., coherence, ...);
        // Verify coherence stored correctly
    }
}
```

### Benchmark Suite (engram-core/benches/dual_memory_types.rs)
Following pattern from metrics_overhead.rs:
```rust
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn benchmark_type_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_memory_construction");

    // Target: <50ns for episode node creation (hot path)
    group.bench_function("create_episode_node", |b| {
        b.iter(|| {
            black_box(DualMemoryNode::new_episode(
                Uuid::new_v4(),
                "test".to_string(),
                [0.5f32; 768],
                Confidence::HIGH,
                0.7,
            ))
        })
    });

    // Target: <100ns for concept node creation (cold path, less critical)
    group.bench_function("create_concept_node", |b| {
        b.iter(|| {
            black_box(DualMemoryNode::new_concept(
                Uuid::new_v4(),
                [0.3f32; 768],
                0.85,
                10,
                Confidence::MEDIUM,
            ))
        })
    });
}

fn benchmark_atomic_updates(c: &mut Criterion) {
    let episode = DualMemoryNode::new_episode(...);

    // Target: <25ns for atomic consolidation score update (matches metrics overhead)
    c.bench_function("atomic_consolidation_update", |b| {
        b.iter(|| {
            episode.node_type.update_consolidation_score(black_box(0.75))
        })
    });

    let concept = DualMemoryNode::new_concept(...);

    // Target: <30ns for atomic instance count increment (AcqRel is slightly slower)
    c.bench_function("atomic_instance_increment", |b| {
        b.iter(|| {
            concept.node_type.increment_instances()
        })
    });
}

fn benchmark_conversions(c: &mut Criterion) {
    let memory = Memory::new("test".to_string(), [0.5f32; 768], Confidence::HIGH);

    // Target: <200ns for Memory -> DualMemoryNode (includes UUID parsing)
    c.bench_function("memory_to_dual_conversion", |b| {
        b.iter(|| {
            black_box(DualMemoryNode::from(black_box(&memory)))
        })
    });

    let dual = DualMemoryNode::new_episode(...);

    // Target: <150ns for DualMemoryNode -> Memory (copies embedding)
    c.bench_function("dual_to_memory_conversion", |b| {
        b.iter(|| {
            black_box(dual.to_memory())
        })
    });
}

fn benchmark_memory_overhead(c: &mut Criterion) {
    // Measure sizeof(DualMemoryNode) vs sizeof(Memory)
    let dual = DualMemoryNode::new_episode(...);
    let memory = Memory::new(...);

    c.bench_function("memory_footprint_dual", |b| {
        b.iter(|| {
            // Allocate and drop to measure heap pressure
            black_box(vec![dual.clone(); 1000])
        })
    });

    c.bench_function("memory_footprint_original", |b| {
        b.iter(|| {
            black_box(vec![memory.clone(); 1000])
        })
    });
}

criterion_group!(
    benches,
    benchmark_type_construction,
    benchmark_atomic_updates,
    benchmark_conversions,
    benchmark_memory_overhead,
);
criterion_main!(benches);
```

Add to Cargo.toml:
```toml
[[bench]]
name = "dual_memory_types"
harness = false
required-features = ["dual_memory_types"]
```

## Acceptance Criteria

- [ ] MemoryNodeType enum compiles and serializes correctly
  - Episode and Concept variants with all specified fields
  - AtomicF32/AtomicU32 properly skipped in serialization
  - Round-trip serde_json and bincode without data loss

- [ ] DualMemoryNode provides ergonomic API
  - Constructor functions for episode and concept nodes
  - Type-safe accessors (is_episode(), is_concept())
  - Atomic operations for consolidation_score and instance_count
  - Thread-safe activation updates matching existing Memory API

- [ ] Feature flag properly gates functionality
  - Code compiles with `--features dual_memory_types`
  - Code compiles without feature (no dead code warnings)
  - No API breakage for existing code when feature disabled

- [ ] Zero-copy conversion from Memory to DualMemoryNode
  - `From<Memory>` moves embedding without allocation
  - Benchmark validates <200ns conversion time
  - Memory footprint regression <10% vs baseline Memory

- [ ] Memory overhead <10% compared to current Memory type
  - sizeof(DualMemoryNode) measured in benchmark
  - Overhead dominated by 64-byte alignment padding
  - Acceptable given false sharing prevention benefits

- [ ] All existing tests pass with feature disabled
  - Run `cargo test --lib` (no dual_memory_types)
  - Zero failures, zero regressions
  - Validates non-breaking change guarantee

- [ ] Cache alignment verified
  - `#[repr(align(64))]` reflected in runtime alignment
  - False sharing prevented in concurrent access tests
  - SIMD operations compatible with embedding layout

- [ ] Atomic operations correct
  - Loom tests validate lock-free correctness (future task)
  - Memory ordering matches DashMapBackend patterns
  - No data races under ThreadSanitizer

## Dependencies
None (foundational task)

## Estimated Time
2-3 days
- Day 1: Type definitions and basic API
- Day 2: Conversions, serialization, unit tests
- Day 3: Benchmarks, property tests, documentation review

## Follow-up Tasks
- Task 002: Adapt graph storage backends for dual memory types
- Task 003: Migration utilities for existing episodic graphs
- Task 004: Concept formation engine using clustering

## Performance Budget
- Episode construction: <50ns (hot path, frequent)
- Concept construction: <100ns (cold path, infrequent)
- Atomic consolidation update: <25ns (matches metrics overhead target)
- Atomic instance increment: <30ns (AcqRel ordering overhead)
- Memory -> DualMemoryNode conversion: <200ns (includes UUID parse)
- DualMemoryNode -> Memory conversion: <150ns (embedding copy)
- Memory overhead: <10% vs baseline Memory struct

## References
- Existing Memory implementation: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/memory.rs`
- Cache-aligned HnswNode: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/index/hnsw_node.rs`
- Atomic activation patterns: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/memory_graph/backends/dashmap.rs`
- Benchmark patterns: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/benches/metrics_overhead.rs`
- Serialization with atomics: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/storage/wal.rs`
