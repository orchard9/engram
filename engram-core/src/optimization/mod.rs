//! Hot-path optimization utilities for the dual memory architecture.
//!
//! The modules in this namespace expose cache-friendly data structures,
//! SIMD helpers, and NUMA scaffolding that can be shared across the
//! episodic/concept integration points.

#[cfg(feature = "dual_memory_cache")]
/// Dual-memory caching primitives.
pub mod dual_memory_cache;

#[cfg(feature = "simd_concepts")]
/// SIMD helpers for concept math.
pub mod simd_concepts;

#[cfg(feature = "numa_aware")]
/// NUMA-aware placement scaffolding.
pub mod numa_aware;

#[cfg(feature = "dual_memory_cache")]
pub use dual_memory_cache::{CacheStatistics, ConceptMetadata, DualMemoryCache};

#[cfg(feature = "simd_concepts")]
pub use simd_concepts::{
    batch_binding_decay_avx2, batch_concept_similarity_avx512, batch_fan_effect_division_avx2,
};

#[cfg(feature = "numa_aware")]
pub use numa_aware::{NumaStrategy, bind_concept_storage_to_numa_node};
