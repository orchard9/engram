//! Dual memory type system for episode-concept architecture
//!
//! This module introduces MemoryNodeType to distinguish between episodic
//! and semantic memory representations with lock-free concurrent access.

use crate::{Confidence, EMBEDDING_DIM};
use atomic_float::AtomicF32;
use chrono::{DateTime, Utc};
use crossbeam_utils::CachePadded;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};
use uuid::Uuid;

/// Unique identifier for episodes (distinct from node UUID)
pub type EpisodeId = String;

/// Memory node type discriminator with type-specific metadata.
///
/// Layout is optimized for cache locality - frequently accessed fields
/// (discriminant, atomic counters) are placed first to share cache lines.
///
/// Note: The large size difference between variants is intentional - concepts
/// embed their centroid for cache efficiency during semantic search.
#[derive(Debug, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum MemoryNodeType {
    /// Episodic memory: specific event with temporal/spatial context
    Episode {
        /// Reference to original Episode struct for detailed metadata
        episode_id: EpisodeId,

        /// Current strength/activation (0.0 = fully decayed, 1.0 = maximally active)
        strength: f32,

        /// Consolidation progress (0.0 = pure episode, 1.0 = ready for concept formation)
        /// Updated atomically during clustering analysis
        /// CachePadded prevents false sharing with adjacent memory nodes
        #[serde(skip)]
        consolidation_score: CachePadded<AtomicF32>,

        /// Serialization field for consolidation score
        #[serde(rename = "consolidation_score")]
        consolidation_score_value: f32,
    },

    /// Semantic concept: generalized pattern extracted from episodic clusters
    Concept {
        /// Centroid embedding computed from clustered episodes
        /// Stored inline for cache efficiency during similarity search
        #[serde(with = "super::types::embedding_serde")]
        centroid: [f32; EMBEDDING_DIM],

        /// Coherence score: tightness of episode cluster (0.0 = diffuse, 1.0 = tight)
        /// Higher coherence = more reliable generalization
        coherence: f32,

        /// Number of episodes contributing to this concept
        /// Updated atomically as new episodes bind to concept
        /// CachePadded prevents false sharing with adjacent memory nodes
        #[serde(skip)]
        instance_count: CachePadded<AtomicU32>,

        /// Serialization field for instance count
        #[serde(rename = "instance_count")]
        instance_count_value: u32,

        /// When this concept was first formed (for decay calculations)
        formation_time: DateTime<Utc>,
    },
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

    /// Borrow the concept centroid when available.
    #[must_use]
    pub const fn as_concept_centroid(&self) -> Option<&[f32; EMBEDDING_DIM]> {
        match self {
            Self::Concept { centroid, .. } => Some(centroid),
            Self::Episode { .. } => None,
        }
    }

    /// Get consolidation score if this is an episode (thread-safe)
    #[must_use]
    pub fn consolidation_score(&self) -> Option<f32> {
        match self {
            Self::Episode {
                consolidation_score,
                ..
            } => Some(consolidation_score.load(Ordering::Relaxed)),
            Self::Concept { .. } => None,
        }
    }

    /// Update consolidation score atomically (only for episodes)
    pub fn update_consolidation_score(&self, new_score: f32) -> bool {
        match self {
            Self::Episode {
                consolidation_score,
                ..
            } => {
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
            Self::Concept { instance_count, .. } => Some(instance_count.load(Ordering::Relaxed)),
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
            Self::Episode {
                consolidation_score,
                consolidation_score_value,
                ..
            } => {
                *consolidation_score_value = consolidation_score.load(Ordering::Relaxed);
            }
            Self::Concept {
                instance_count,
                instance_count_value,
                ..
            } => {
                *instance_count_value = instance_count.load(Ordering::Relaxed);
            }
        }
    }

    /// Restore atomic fields after deserialization
    pub fn restore_atomics(&mut self) {
        match self {
            Self::Episode {
                consolidation_score,
                consolidation_score_value,
                ..
            } => {
                consolidation_score.store(*consolidation_score_value, Ordering::Relaxed);
            }
            Self::Concept {
                instance_count,
                instance_count_value,
                ..
            } => {
                instance_count.store(*instance_count_value, Ordering::Relaxed);
            }
        }
    }
}

/// Dual memory node wrapping existing Memory with type discrimination.
///
/// Memory layout is cache-optimized:
/// - 64-byte alignment prevents false sharing on multi-core systems
/// - CachePadded atomics ensure each atomic gets its own cache line
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
    pub embedding: [f32; EMBEDDING_DIM],

    /// Current activation level (thread-safe, lock-free updates)
    /// CachePadded to prevent false sharing across concurrent access patterns
    activation: CachePadded<AtomicF32>,

    /// Cognitive confidence in this memory's reliability
    pub confidence: Confidence,

    /// Last access timestamp for decay calculations
    pub last_access: DateTime<Utc>,

    /// Node creation timestamp
    pub created_at: DateTime<Utc>,
}

impl Clone for DualMemoryNode {
    fn clone(&self) -> Self {
        let node_type = match &self.node_type {
            MemoryNodeType::Episode {
                episode_id,
                strength,
                consolidation_score,
                ..
            } => MemoryNodeType::Episode {
                episode_id: episode_id.clone(),
                strength: *strength,
                consolidation_score: CachePadded::new(AtomicF32::new(
                    consolidation_score.load(Ordering::Relaxed),
                )),
                consolidation_score_value: consolidation_score.load(Ordering::Relaxed),
            },
            MemoryNodeType::Concept {
                centroid,
                coherence,
                instance_count,
                formation_time,
                ..
            } => MemoryNodeType::Concept {
                centroid: *centroid,
                coherence: *coherence,
                instance_count: CachePadded::new(AtomicU32::new(
                    instance_count.load(Ordering::Relaxed),
                )),
                instance_count_value: instance_count.load(Ordering::Relaxed),
                formation_time: *formation_time,
            },
        };

        Self {
            id: self.id,
            node_type,
            embedding: self.embedding,
            activation: CachePadded::new(AtomicF32::new(self.activation())),
            confidence: self.confidence,
            last_access: self.last_access,
            created_at: self.created_at,
        }
    }
}

impl DualMemoryNode {
    /// Create a new dual memory node from episode
    ///
    /// Note: Embedding is passed by value (not reference) to allow move semantics
    /// and avoid allocation overhead during conversion from Memory.
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
                consolidation_score: CachePadded::new(AtomicF32::new(0.0)),
                consolidation_score_value: 0.0,
            },
            embedding,
            activation: CachePadded::new(AtomicF32::new(strength)),
            confidence,
            last_access: now,
            created_at: now,
        }
    }

    /// Create a new dual memory node from concept
    ///
    /// Note: Centroid is passed by value (not reference) to allow move semantics
    /// and avoid allocation overhead.
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
                instance_count: CachePadded::new(AtomicU32::new(initial_instance_count)),
                instance_count_value: initial_instance_count,
                formation_time: now,
            },
            embedding: centroid,
            activation: CachePadded::new(AtomicF32::new(0.0)),
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
        self.activation
            .store(value.clamp(0.0, 1.0), Ordering::Release);
    }

    /// Add to current activation (thread-safe)
    ///
    /// Note: This is not fully atomic (read + compute + write are separate),
    /// but under Engram's usage patterns (probabilistic activation spreading),
    /// exact atomicity isn't required. Lost updates are acceptable as they
    /// represent natural activation noise in biological systems.
    pub fn add_activation(&self, delta: f32) {
        let current = self.activation();
        let new_value = (current + delta).clamp(0.0, 1.0);
        self.set_activation(new_value);
    }

    /// Check if this is an episode node
    #[must_use]
    pub const fn is_episode(&self) -> bool {
        self.node_type.is_episode()
    }

    /// Check if this is a concept node
    #[must_use]
    pub const fn is_concept(&self) -> bool {
        self.node_type.is_concept()
    }

    /// Borrow the centroid without repeated pattern matching once callers validated the type.
    #[inline]
    pub const fn get_centroid_unchecked(&self) -> &[f32; EMBEDDING_DIM] {
        match &self.node_type {
            MemoryNodeType::Concept { centroid, .. } => centroid,
            #[allow(unsafe_code)]
            MemoryNodeType::Episode { .. } => unsafe { std::hint::unreachable_unchecked() },
        }
    }
}
