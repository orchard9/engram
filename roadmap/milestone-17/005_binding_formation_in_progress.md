# Task 005: Lock-Free Binding Index with Cache-Optimal Access Patterns

## Objective
Design and implement a high-performance, lock-free binding index that creates bidirectional edges between episodes and concepts with atomic strength updates, cache-line-aligned data structures, and SIMD batch operations.

## Background
Bindings connect specific episodes to their generalized concepts, enabling both bottom-up (episode to concept) and top-down (concept to episode) processing. The binding index must handle two distinct access patterns efficiently:

1. **Episode → Concepts (bottom-up)**: Low fan-out (1-5 concepts per episode), high locality
2. **Concept → Episodes (top-down)**: High fan-out (10-1000 episodes per concept), scatter pattern

Performance requirements demand:
- Lock-free concurrent updates during spreading activation
- Cache-optimal layout for sequential traversal
- Atomic strength updates without contention
- Lazy garbage collection to maintain memory overhead <20% of node storage

## Requirements
1. Design cache-line-aligned ConceptBinding struct
2. Implement bidirectional index using DashMap with sharded access
3. Support atomic strength updates without locks
4. Enable efficient traversal in both directions
5. Implement SIMD batch operations for strength calculations
6. Add lazy garbage collection for weak bindings (<0.1 strength)
7. Integrate with existing graph traversal and spreading activation
8. Maintain memory overhead budget <20% of node storage

## Technical Specification

### Files to Create
- `engram-core/src/memory/bindings.rs` - Core binding types with cache alignment
- `engram-core/src/memory_graph/binding_index.rs` - Lock-free bidirectional index
- `engram-core/src/memory/binding_ops.rs` - SIMD batch operations
- `engram-core/src/memory/binding_gc.rs` - Lazy garbage collection

### Files to Modify
- `engram-core/src/memory/mod.rs` - Add binding module exports
- `engram-core/src/memory_graph/mod.rs` - Integrate binding index
- `engram-core/src/memory_graph/traits.rs` - Add binding operations to GraphBackend
- `engram-core/src/memory_graph/backends/dashmap.rs` - Implement binding support

### Cache-Aligned Binding Structure

```rust
use atomic_float::AtomicF32;
use chrono::{DateTime, Utc};
use crossbeam_utils::CachePadded;
use std::sync::Arc;
use uuid::Uuid;

/// Cache-line-aligned binding between episode and concept
///
/// Layout optimized for sequential traversal and cache locality.
/// Total size: 64 bytes (one cache line) when aligned.
#[repr(align(64))]
#[derive(Debug)]
pub struct ConceptBinding {
    /// Episode node ID (16 bytes)
    pub episode_id: Uuid,

    /// Concept node ID (16 bytes)
    pub concept_id: Uuid,

    /// Binding strength (atomic for lock-free updates)
    /// Range: 0.0 (weak) to 1.0 (strong)
    /// Updated during spreading activation and consolidation
    pub strength: AtomicF32,

    /// Episode's contribution to concept formation (0.0-1.0)
    /// Set once during concept formation, read-only afterward
    pub contribution: f32,

    /// Binding creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last activation timestamp (atomic for concurrent access)
    /// Used for garbage collection and temporal decay
    pub last_activated: crossbeam_utils::atomic::AtomicCell<DateTime<Utc>>,

    /// Reserved padding to fill cache line
    _padding: [u8; 8],
}

impl ConceptBinding {
    /// Create a new binding with initial strength
    #[must_use]
    pub fn new(
        episode_id: Uuid,
        concept_id: Uuid,
        initial_strength: f32,
        contribution: f32,
    ) -> Self {
        Self {
            episode_id,
            concept_id,
            strength: AtomicF32::new(initial_strength.clamp(0.0, 1.0)),
            contribution: contribution.clamp(0.0, 1.0),
            created_at: Utc::now(),
            last_activated: crossbeam_utils::atomic::AtomicCell::new(Utc::now()),
            _padding: [0; 8],
        }
    }

    /// Get current strength (relaxed ordering for performance)
    #[inline]
    pub fn get_strength(&self) -> f32 {
        self.strength.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Update strength atomically with compare-and-swap
    /// Returns true if update succeeded
    #[inline]
    pub fn update_strength<F>(&self, f: F) -> bool
    where
        F: FnOnce(f32) -> f32,
    {
        loop {
            let current = self.strength.load(std::sync::atomic::Ordering::Relaxed);
            let new_value = f(current).clamp(0.0, 1.0);

            match self.strength.compare_exchange_weak(
                current,
                new_value,
                std::sync::atomic::Ordering::Release,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.last_activated.store(Utc::now());
                    return true;
                }
                Err(_) => continue,
            }
        }
    }

    /// Add activation to strength (saturating add)
    #[inline]
    pub fn add_activation(&self, delta: f32) {
        self.update_strength(|current| (current + delta).min(1.0));
    }

    /// Apply decay to strength
    #[inline]
    pub fn apply_decay(&self, decay_factor: f32) {
        self.update_strength(|current| current * decay_factor);
    }

    /// Check if binding is eligible for garbage collection
    #[must_use]
    pub fn is_weak(&self, threshold: f32) -> bool {
        self.get_strength() < threshold
    }

    /// Get age since last activation
    #[must_use]
    pub fn age_since_activation(&self) -> chrono::Duration {
        Utc::now().signed_duration_since(self.last_activated.load())
    }
}

/// Compact binding reference for storage efficiency
/// Size: 36 bytes (no alignment padding)
#[derive(Debug, Clone)]
pub struct BindingRef {
    pub target_id: Uuid,
    pub strength_ptr: Arc<AtomicF32>,
    pub contribution: f32,
}

impl BindingRef {
    #[inline]
    pub fn get_strength(&self) -> f32 {
        self.strength_ptr.load(std::sync::atomic::Ordering::Relaxed)
    }
}
```

### Lock-Free Bidirectional Index

```rust
use dashmap::DashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Lock-free bidirectional binding index with sharded access
///
/// Optimized for two distinct access patterns:
/// - Episode → Concepts: low fan-out, high locality (Vec storage)
/// - Concept → Episodes: high fan-out, scatter pattern (Arc<Vec> for sharing)
pub struct BindingIndex {
    /// Episode → Concepts mapping (low fan-out)
    /// DashMap provides lock-free concurrent access with internal sharding
    episode_to_concepts: DashMap<Uuid, Vec<Arc<ConceptBinding>>>,

    /// Concept → Episodes mapping (high fan-out)
    /// Arc<Vec> allows cheap cloning for readers without blocking writers
    concept_to_episodes: DashMap<Uuid, Arc<Vec<Arc<ConceptBinding>>>>,

    /// Weak binding threshold for garbage collection
    gc_threshold: f32,

    /// Total binding count (atomic)
    binding_count: std::sync::atomic::AtomicUsize,
}

impl BindingIndex {
    /// Create a new binding index with specified GC threshold
    #[must_use]
    pub fn new(gc_threshold: f32) -> Self {
        Self {
            episode_to_concepts: DashMap::new(),
            concept_to_episodes: DashMap::new(),
            gc_threshold: gc_threshold.clamp(0.0, 1.0),
            binding_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Create with pre-allocated capacity
    #[must_use]
    pub fn with_capacity(episode_capacity: usize, concept_capacity: usize, gc_threshold: f32) -> Self {
        Self {
            episode_to_concepts: DashMap::with_capacity(episode_capacity),
            concept_to_episodes: DashMap::with_capacity(concept_capacity),
            gc_threshold: gc_threshold.clamp(0.0, 1.0),
            binding_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Add a binding between episode and concept
    ///
    /// This is the only write path for bindings. Both indices share
    /// the same Arc<ConceptBinding> for memory efficiency.
    pub fn add_binding(&self, binding: ConceptBinding) {
        let binding = Arc::new(binding);
        let episode_id = binding.episode_id;
        let concept_id = binding.concept_id;

        // Add to episode → concepts (low fan-out)
        self.episode_to_concepts
            .entry(episode_id)
            .or_default()
            .push(Arc::clone(&binding));

        // Add to concept → episodes (high fan-out)
        // Use copy-on-write pattern for minimal contention
        self.concept_to_episodes
            .entry(concept_id)
            .and_modify(|episodes| {
                let mut new_vec = (**episodes).clone();
                new_vec.push(Arc::clone(&binding));
                *episodes = Arc::new(new_vec);
            })
            .or_insert_with(|| Arc::new(vec![binding]));

        self.binding_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get concepts for an episode (bottom-up access)
    #[must_use]
    pub fn get_concepts_for_episode(&self, episode_id: &Uuid) -> Vec<Arc<ConceptBinding>> {
        self.episode_to_concepts
            .get(episode_id)
            .map(|entry| entry.value().clone())
            .unwrap_or_default()
    }

    /// Get episodes for a concept (top-down access)
    #[must_use]
    pub fn get_episodes_for_concept(&self, concept_id: &Uuid) -> Arc<Vec<Arc<ConceptBinding>>> {
        self.concept_to_episodes
            .get(concept_id)
            .map(|entry| Arc::clone(entry.value()))
            .unwrap_or_else(|| Arc::new(Vec::new()))
    }

    /// Get binding strength between episode and concept
    #[must_use]
    pub fn get_binding_strength(&self, episode_id: &Uuid, concept_id: &Uuid) -> Option<f32> {
        self.episode_to_concepts.get(episode_id).and_then(|entry| {
            entry
                .value()
                .iter()
                .find(|b| b.concept_id == *concept_id)
                .map(|b| b.get_strength())
        })
    }

    /// Update binding strength atomically
    pub fn update_binding_strength<F>(
        &self,
        episode_id: &Uuid,
        concept_id: &Uuid,
        update_fn: F,
    ) -> bool
    where
        F: FnOnce(f32) -> f32,
    {
        if let Some(entry) = self.episode_to_concepts.get(episode_id) {
            if let Some(binding) = entry.value().iter().find(|b| b.concept_id == *concept_id) {
                return binding.update_strength(update_fn);
            }
        }
        false
    }

    /// Remove all bindings for an episode
    pub fn remove_episode_bindings(&self, episode_id: &Uuid) -> usize {
        let mut removed_count = 0;

        if let Some((_, bindings)) = self.episode_to_concepts.remove(episode_id) {
            removed_count = bindings.len();

            // Remove from concept → episodes index
            for binding in bindings {
                self.concept_to_episodes.entry(binding.concept_id).and_modify(|episodes| {
                    let new_vec: Vec<_> = episodes
                        .iter()
                        .filter(|b| b.episode_id != *episode_id)
                        .cloned()
                        .collect();
                    *episodes = Arc::new(new_vec);
                });
            }
        }

        self.binding_count.fetch_sub(removed_count, std::sync::atomic::Ordering::Relaxed);
        removed_count
    }

    /// Remove all bindings for a concept
    pub fn remove_concept_bindings(&self, concept_id: &Uuid) -> usize {
        let mut removed_count = 0;

        if let Some((_, episodes_arc)) = self.concept_to_episodes.remove(concept_id) {
            removed_count = episodes_arc.len();

            // Remove from episode → concepts index
            for binding in episodes_arc.iter() {
                self.episode_to_concepts.entry(binding.episode_id).and_modify(|concepts| {
                    concepts.retain(|b| b.concept_id != *concept_id);
                });
            }
        }

        self.binding_count.fetch_sub(removed_count, std::sync::atomic::Ordering::Relaxed);
        removed_count
    }

    /// Lazy garbage collection: remove weak bindings
    ///
    /// Returns number of bindings removed.
    /// Should be called periodically (e.g., during consolidation).
    pub fn garbage_collect(&self) -> usize {
        let mut removed = 0;
        let threshold = self.gc_threshold;

        // Collect weak bindings
        let mut weak_bindings = Vec::new();

        for entry in self.episode_to_concepts.iter() {
            for binding in entry.value() {
                if binding.is_weak(threshold) {
                    weak_bindings.push((binding.episode_id, binding.concept_id));
                }
            }
        }

        // Remove weak bindings
        for (episode_id, concept_id) in weak_bindings {
            // Remove from episode → concepts
            if let Some(mut entry) = self.episode_to_concepts.get_mut(&episode_id) {
                let initial_len = entry.len();
                entry.retain(|b| !(b.concept_id == concept_id && b.is_weak(threshold)));
                removed += initial_len - entry.len();
            }

            // Remove from concept → episodes
            self.concept_to_episodes.entry(concept_id).and_modify(|episodes| {
                let new_vec: Vec<_> = episodes
                    .iter()
                    .filter(|b| !(b.episode_id == episode_id && b.is_weak(threshold)))
                    .cloned()
                    .collect();
                *episodes = Arc::new(new_vec);
            });
        }

        // Clean up empty entries
        self.episode_to_concepts.retain(|_, v| !v.is_empty());
        self.concept_to_episodes.retain(|_, v| !v.is_empty());

        self.binding_count.fetch_sub(removed, std::sync::atomic::Ordering::Relaxed);
        removed
    }

    /// Get total binding count
    #[must_use]
    pub fn count(&self) -> usize {
        self.binding_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get memory usage statistics
    #[must_use]
    pub fn memory_stats(&self) -> BindingMemoryStats {
        let binding_size = std::mem::size_of::<ConceptBinding>();
        let arc_overhead = std::mem::size_of::<Arc<ConceptBinding>>() - std::mem::size_of::<*const ConceptBinding>();

        let total_bindings = self.count();
        let binding_memory = total_bindings * binding_size;
        let arc_memory = total_bindings * 2 * arc_overhead; // Stored in both indices
        let index_overhead = self.episode_to_concepts.len() * 32 + self.concept_to_episodes.len() * 32; // Rough estimate

        BindingMemoryStats {
            total_bindings,
            binding_memory_bytes: binding_memory,
            arc_overhead_bytes: arc_memory,
            index_overhead_bytes: index_overhead,
            total_bytes: binding_memory + arc_memory + index_overhead,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BindingMemoryStats {
    pub total_bindings: usize,
    pub binding_memory_bytes: usize,
    pub arc_overhead_bytes: usize,
    pub index_overhead_bytes: usize,
    pub total_bytes: usize,
}

impl BindingMemoryStats {
    /// Calculate overhead percentage relative to node storage
    ///
    /// Assumes node size of ~1KB (embedding + metadata)
    #[must_use]
    pub fn overhead_percentage(&self, node_count: usize) -> f32 {
        let node_memory = node_count * 1024;
        (self.total_bytes as f32 / node_memory as f32) * 100.0
    }
}
```

### SIMD Batch Operations

```rust
use std::sync::Arc;

/// SIMD-accelerated batch operations for binding strength calculations
pub struct BindingBatchOps;

impl BindingBatchOps {
    /// Batch update binding strengths with SIMD acceleration
    ///
    /// Uses SIMD to process 8 bindings at a time (f32x8).
    /// Falls back to scalar for remainder.
    #[cfg(target_feature = "avx2")]
    pub fn batch_add_activation(bindings: &[Arc<ConceptBinding>], delta: f32) {
        use std::arch::x86_64::*;

        let delta_vec = unsafe { _mm256_set1_ps(delta) };
        let one_vec = unsafe { _mm256_set1_ps(1.0) };
        let zero_vec = unsafe { _mm256_setzero_ps() };

        let chunks = bindings.chunks_exact(8);
        let remainder = chunks.remainder();

        // Process 8 bindings at a time with SIMD
        for chunk in chunks {
            let mut strengths = [0.0f32; 8];
            for (i, binding) in chunk.iter().enumerate() {
                strengths[i] = binding.get_strength();
            }

            unsafe {
                let current = _mm256_loadu_ps(strengths.as_ptr());
                let updated = _mm256_add_ps(current, delta_vec);
                let clamped = _mm256_min_ps(_mm256_max_ps(updated, zero_vec), one_vec);
                _mm256_storeu_ps(strengths.as_mut_ptr(), clamped);
            }

            for (i, binding) in chunk.iter().enumerate() {
                binding.strength.store(strengths[i], std::sync::atomic::Ordering::Relaxed);
                binding.last_activated.store(Utc::now());
            }
        }

        // Process remainder with scalar operations
        for binding in remainder {
            binding.add_activation(delta);
        }
    }

    /// Batch decay binding strengths
    #[cfg(target_feature = "avx2")]
    pub fn batch_apply_decay(bindings: &[Arc<ConceptBinding>], decay_factor: f32) {
        use std::arch::x86_64::*;

        let decay_vec = unsafe { _mm256_set1_ps(decay_factor) };
        let zero_vec = unsafe { _mm256_setzero_ps() };

        let chunks = bindings.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut strengths = [0.0f32; 8];
            for (i, binding) in chunk.iter().enumerate() {
                strengths[i] = binding.get_strength();
            }

            unsafe {
                let current = _mm256_loadu_ps(strengths.as_ptr());
                let decayed = _mm256_mul_ps(current, decay_vec);
                let clamped = _mm256_max_ps(decayed, zero_vec);
                _mm256_storeu_ps(strengths.as_mut_ptr(), clamped);
            }

            for (i, binding) in chunk.iter().enumerate() {
                binding.strength.store(strengths[i], std::sync::atomic::Ordering::Relaxed);
            }
        }

        for binding in remainder {
            binding.apply_decay(decay_factor);
        }
    }

    /// Count bindings above threshold using SIMD
    #[cfg(target_feature = "avx2")]
    pub fn count_above_threshold(bindings: &[Arc<ConceptBinding>], threshold: f32) -> usize {
        use std::arch::x86_64::*;

        let threshold_vec = unsafe { _mm256_set1_ps(threshold) };
        let mut count = 0;

        let chunks = bindings.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let mut strengths = [0.0f32; 8];
            for (i, binding) in chunk.iter().enumerate() {
                strengths[i] = binding.get_strength();
            }

            unsafe {
                let current = _mm256_loadu_ps(strengths.as_ptr());
                let cmp = _mm256_cmp_ps(current, threshold_vec, _CMP_GT_OQ);
                let mask = _mm256_movemask_ps(cmp);
                count += mask.count_ones() as usize;
            }
        }

        count += remainder.iter().filter(|b| b.get_strength() > threshold).count();
        count
    }

    /// Scalar fallback for non-AVX2 systems
    #[cfg(not(target_feature = "avx2"))]
    pub fn batch_add_activation(bindings: &[Arc<ConceptBinding>], delta: f32) {
        for binding in bindings {
            binding.add_activation(delta);
        }
    }

    #[cfg(not(target_feature = "avx2"))]
    pub fn batch_apply_decay(bindings: &[Arc<ConceptBinding>], decay_factor: f32) {
        for binding in bindings {
            binding.apply_decay(decay_factor);
        }
    }

    #[cfg(not(target_feature = "avx2"))]
    pub fn count_above_threshold(bindings: &[Arc<ConceptBinding>], threshold: f32) -> usize {
        bindings.iter().filter(|b| b.get_strength() > threshold).count()
    }
}
```

### Integration with Graph Traversal

```rust
// In engram-core/src/memory_graph/traits.rs

pub trait GraphBackend: MemoryBackend {
    // ... existing methods ...

    /// Add a concept binding
    fn add_concept_binding(
        &self,
        episode_id: Uuid,
        concept_id: Uuid,
        strength: f32,
        contribution: f32,
    ) -> Result<(), MemoryError>;

    /// Get concepts for an episode
    fn get_episode_concepts(&self, episode_id: &Uuid) -> Result<Vec<BindingRef>, MemoryError>;

    /// Get episodes for a concept
    fn get_concept_episodes(&self, concept_id: &Uuid) -> Result<Vec<BindingRef>, MemoryError>;

    /// Update binding strength
    fn update_binding_strength(
        &self,
        episode_id: &Uuid,
        concept_id: &Uuid,
        new_strength: f32,
    ) -> Result<(), MemoryError>;

    /// Spread activation through bindings
    fn spread_through_bindings(
        &self,
        source_id: &Uuid,
        is_episode: bool,
        decay: f32,
    ) -> Result<(), MemoryError>;
}

// In engram-core/src/memory_graph/backends/dashmap.rs

impl GraphBackend for DashMapBackend {
    fn add_concept_binding(
        &self,
        episode_id: Uuid,
        concept_id: Uuid,
        strength: f32,
        contribution: f32,
    ) -> Result<(), MemoryError> {
        let binding = ConceptBinding::new(episode_id, concept_id, strength, contribution);
        self.binding_index.add_binding(binding);
        Ok(())
    }

    fn get_episode_concepts(&self, episode_id: &Uuid) -> Result<Vec<BindingRef>, MemoryError> {
        let bindings = self.binding_index.get_concepts_for_episode(episode_id);
        Ok(bindings
            .iter()
            .map(|b| BindingRef {
                target_id: b.concept_id,
                strength_ptr: Arc::new(AtomicF32::new(b.get_strength())),
                contribution: b.contribution,
            })
            .collect())
    }

    fn get_concept_episodes(&self, concept_id: &Uuid) -> Result<Vec<BindingRef>, MemoryError> {
        let bindings_arc = self.binding_index.get_episodes_for_concept(concept_id);
        Ok(bindings_arc
            .iter()
            .map(|b| BindingRef {
                target_id: b.episode_id,
                strength_ptr: Arc::new(AtomicF32::new(b.get_strength())),
                contribution: b.contribution,
            })
            .collect())
    }

    fn spread_through_bindings(
        &self,
        source_id: &Uuid,
        is_episode: bool,
        decay: f32,
    ) -> Result<(), MemoryError> {
        let source_activation = self
            .activation_cache
            .get(source_id)
            .map_or(0.5, |a| a.load(Ordering::Relaxed));

        if is_episode {
            // Episode → Concepts (bottom-up)
            let bindings = self.binding_index.get_concepts_for_episode(source_id);
            for binding in &bindings {
                let contribution = source_activation * binding.get_strength() * decay;
                if let Some(target_activation) = self.activation_cache.get(&binding.concept_id) {
                    loop {
                        let current = target_activation.load(Ordering::Relaxed);
                        let new_value = (current + contribution).min(1.0);
                        if target_activation
                            .compare_exchange_weak(
                                current,
                                new_value,
                                Ordering::Release,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                        {
                            break;
                        }
                    }
                }
            }
        } else {
            // Concept → Episodes (top-down)
            let bindings_arc = self.binding_index.get_episodes_for_concept(source_id);

            // Use SIMD batch operations for high fan-out
            let activation_delta = source_activation * decay;
            BindingBatchOps::batch_add_activation(&bindings_arc, activation_delta);

            // Update activation cache
            for binding in bindings_arc.iter() {
                let weighted_contribution = activation_delta * binding.get_strength();
                if let Some(target_activation) = self.activation_cache.get(&binding.episode_id) {
                    loop {
                        let current = target_activation.load(Ordering::Relaxed);
                        let new_value = (current + weighted_contribution).min(1.0);
                        if target_activation
                            .compare_exchange_weak(
                                current,
                                new_value,
                                Ordering::Release,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                        {
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
```

### Memory Overhead Budget

Target: <20% of node storage

**Assumptions:**
- Average node size: 1024 bytes (768 embedding floats + metadata)
- Cache-aligned binding: 64 bytes
- Arc overhead: 16 bytes per reference
- Index overhead: 32 bytes per DashMap entry

**Calculation for 10,000 nodes:**
- Node storage: 10,000 * 1024 = 10.24 MB
- Average bindings per node: 3 (conservative)
- Total bindings: 10,000 * 3 = 30,000
- Binding storage: 30,000 * 64 = 1.92 MB
- Arc overhead: 30,000 * 2 * 16 = 0.96 MB
- Index overhead: 20,000 * 32 = 0.64 MB
- **Total binding overhead: 3.52 MB (34% of node storage)**

**Optimization strategies to achieve <20%:**
1. Use BindingRef (36 bytes) instead of Arc<ConceptBinding> for high fan-out concepts
2. Implement copy-on-write sharing for concept → episodes vectors
3. Aggressive garbage collection of weak bindings
4. Delta encoding for timestamp storage

**Revised calculation with optimizations:**
- Binding storage (50% as BindingRef): 15,000 * 64 + 15,000 * 36 = 1.5 MB
- Arc overhead (only for episode → concepts): 15,000 * 16 = 0.24 MB
- Index overhead: 0.64 MB
- **Total optimized overhead: 2.38 MB (23% of node storage)**

With aggressive GC (removing <0.15 strength): **~18% overhead**

## Implementation Notes

### Cache Optimization
1. **Cache Line Alignment**: ConceptBinding is exactly 64 bytes (one cache line) to minimize false sharing
2. **Sequential Access**: Vec storage for episode → concepts enables prefetching
3. **Hot/Cold Split**: Frequently accessed fields (IDs, strength) at start of struct
4. **Padding**: Explicit padding to prevent cache line boundary crossings

### Lock-Free Concurrency
1. **AtomicF32**: Strength updates use CAS loop for contention-free concurrent modification
2. **DashMap Sharding**: Internal sharding reduces contention across 16+ shards
3. **Copy-on-Write**: Concept → episodes uses Arc<Vec> for lock-free reads
4. **Relaxed Ordering**: Strength loads use Relaxed for maximum performance (Release/Acquire only on updates)

### SIMD Optimization
1. **AVX2 Vectorization**: Process 8 bindings at a time using 256-bit registers
2. **Cache-Friendly Layout**: Sequential strength access maximizes SIMD efficiency
3. **Scalar Fallback**: Automatic fallback for non-AVX2 systems
4. **Alignment**: Ensure binding arrays are 32-byte aligned for SIMD loads

### Garbage Collection Strategy
1. **Lazy Collection**: Run during consolidation cycles (low-priority background task)
2. **Two-Phase**: Identify weak bindings, then batch remove
3. **Threshold Tuning**: Default 0.1, configurable per deployment
4. **Age-Based**: Consider both strength and time since last activation

## Testing Approach

### Unit Tests
1. **Binding Creation and Access**
   - Test cache alignment with `std::mem::align_of`
   - Verify atomic strength updates are linearizable
   - Test bidirectional consistency

2. **Concurrent Updates**
   - Spawn 100 threads updating same binding
   - Verify final strength is consistent
   - Test lock-free property (no deadlocks)

3. **SIMD Correctness**
   - Compare SIMD vs scalar results (epsilon tolerance)
   - Test alignment requirements
   - Verify remainder handling

4. **Garbage Collection**
   - Test weak binding removal
   - Verify bidirectional consistency after GC
   - Test concurrent GC and updates

### Performance Benchmarks
1. **Binding Traversal**
   - Episode → Concepts: <100ns per lookup
   - Concept → Episodes: <500ns for 100 episodes
   - Compare vs HashMap baseline

2. **Concurrent Updates**
   - 1M strength updates with 16 threads: <50ms
   - Measure contention with perf counters
   - Verify linear scaling to 32 threads

3. **SIMD Batch Operations**
   - 10,000 binding decay: <10µs with SIMD
   - Compare vs scalar (expect 4-8x speedup)
   - Test different batch sizes

4. **Memory Overhead**
   - Measure actual overhead at 1K, 10K, 100K nodes
   - Verify <20% target
   - Track overhead growth rate

### Integration Tests
1. **Spreading Activation**
   - Test activation flows through bindings
   - Verify concept → episode fan-out performance
   - Test with realistic graph (1000 nodes, 3000 bindings)

2. **Consolidation Integration**
   - Test binding creation during concept formation
   - Verify GC runs during consolidation
   - Test binding strength updates over time

3. **Graph Traversal**
   - Test BFS including binding edges
   - Verify correct activation propagation
   - Test hierarchical spreading (episodes ↔ concepts)

## Acceptance Criteria
- [ ] ConceptBinding is cache-line aligned (64 bytes)
- [ ] Bidirectional traversal in O(1) average case
- [ ] Atomic strength updates without locks or deadlocks
- [ ] Memory overhead <20% of node storage (measured at 10K nodes)
- [ ] Binding creation <100µs per binding
- [ ] Episode → concepts lookup <100ns
- [ ] Concept → episodes lookup <500ns for 100 episodes
- [ ] Concurrent strength updates scale linearly to 16 threads
- [ ] SIMD batch operations 4-8x faster than scalar
- [ ] Garbage collection removes weak bindings without blocking writes
- [ ] Integration with spreading activation maintains correctness
- [ ] Zero memory leaks under valgrind/miri
- [ ] All operations are Send + Sync

## Dependencies
- Task 001 (Dual Memory Types)
- Task 002 (Graph Storage Adaptation)
- Task 004 (Concept Formation Engine)
- Existing DashMapBackend implementation

## Performance Targets

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Add binding | <100µs | Includes both index updates |
| Get episode concepts | <100ns | Vec lookup, 1-5 results |
| Get concept episodes | <500ns | Arc clone, 10-1000 results |
| Update strength (uncontended) | <10ns | Single atomic CAS |
| Update strength (contended) | <100ns | CAS retry loop |
| Batch decay (1000 bindings) | <10µs | SIMD AVX2 |
| Garbage collect (10K bindings) | <1ms | Background task |
| Memory overhead | <20% | Of total node storage |

## Estimated Time
3 days

### Day 1: Core Data Structures
- Implement ConceptBinding with cache alignment
- Create BindingIndex with DashMap
- Add bidirectional access methods
- Unit tests for basic operations

### Day 2: Concurrency and SIMD
- Implement atomic strength updates with CAS
- Add SIMD batch operations
- Concurrent update tests
- SIMD correctness and performance tests

### Day 3: Integration and Optimization
- Integrate with GraphBackend trait
- Implement garbage collection
- Add spreading activation through bindings
- Memory overhead measurement and tuning
- Integration tests with full graph
