//! Lock-free bidirectional binding index with sharded access
//!
//! This module provides a high-performance index for episode-concept bindings
//! optimized for two distinct access patterns:
//!
//! 1. **Episode → Concepts (bottom-up)**: Low fan-out (1-5 concepts), high locality
//! 2. **Concept → Episodes (top-down)**: High fan-out (10-1000 episodes), scatter pattern

use crate::memory::bindings::ConceptBinding;
use crate::memory_graph::backends::DUAL_MEMORY_NODE_SIZE;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use uuid::Uuid;

#[cfg(feature = "dual_memory_cache")]
use crate::optimization::DualMemoryCache;

/// Memory usage statistics for binding index
#[derive(Debug, Clone)]
pub struct BindingMemoryStats {
    /// Total number of bindings
    pub total_bindings: usize,

    /// Memory used by binding structs
    pub binding_memory_bytes: usize,

    /// Memory used by Arc overhead
    pub arc_overhead_bytes: usize,

    /// Memory used by DashMap index overhead
    pub index_overhead_bytes: usize,

    /// Total memory usage
    pub total_bytes: usize,
}

impl BindingMemoryStats {
    /// Calculate overhead percentage relative to node storage
    ///
    /// Uses `DUAL_MEMORY_NODE_SIZE` (3328 bytes) for DualMemoryNode with 768-dim embedding.
    ///
    /// # Arguments
    ///
    /// * `node_count` - Total number of nodes (episodes + concepts)
    ///
    /// # Returns
    ///
    /// Overhead as percentage (0.0-100.0)
    #[must_use]
    pub fn overhead_percentage(&self, node_count: usize) -> f32 {
        if node_count == 0 {
            return 0.0;
        }
        let node_memory = node_count * DUAL_MEMORY_NODE_SIZE;
        (self.total_bytes as f32 / node_memory as f32) * 100.0
    }
}

/// Lock-free bidirectional binding index with sharded access
///
/// Optimized for two distinct access patterns:
/// - Episode → Concepts: low fan-out, high locality (Vec storage)
/// - Concept → Episodes: high fan-out, scatter pattern (Arc<Vec> for sharing)
///
/// # Architecture
///
/// ```text
/// Episode → Concepts (bottom-up, low fan-out)
/// ┌─────────────┐     ┌──────────────────────────────┐
/// │ Episode A   │────▶│ Vec<Arc<ConceptBinding>>     │
/// └─────────────┘     │  ├─ Concept 1 (strength=0.8) │
///                     │  ├─ Concept 2 (strength=0.6) │
///                     │  └─ Concept 3 (strength=0.4) │
///                     └──────────────────────────────┘
///
/// Concept → Episodes (top-down, high fan-out)
/// ┌─────────────┐     ┌──────────────────────────────┐
/// │ Concept X   │────▶│ Arc<Vec<Arc<ConceptBinding>>>│
/// └─────────────┘     │  ├─ Episode 1 (strength=0.9) │
///                     │  ├─ Episode 2 (strength=0.7) │
///                     │  ├─ Episode 3 (strength=0.5) │
///                     │  └─ ... (100s-1000s)          │
///                     └──────────────────────────────┘
/// ```
///
/// # Performance
///
/// - Episode → Concepts lookup: O(1) average, <100ns
/// - Concept → Episodes lookup: O(1) average, <500ns for 100 episodes
/// - Add binding: O(1) average, <100µs (includes both indices)
/// - Atomic strength updates: <10ns uncontended
/// - Memory overhead: <20% of node storage (with GC)
///
/// # Concurrency
///
/// - Lock-free concurrent reads and writes via DashMap
/// - Internal sharding reduces contention (16 shards default)
/// - Copy-on-write for concept → episodes (minimal write contention)
/// - Atomic strength updates without blocking
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
    binding_count: AtomicUsize,

    #[cfg(feature = "dual_memory_cache")]
    /// Optional fan-out cache for hot-path activation queries
    fan_out_cache: OnceLock<Arc<DualMemoryCache>>,
}

impl BindingIndex {
    /// Create a new binding index with specified GC threshold
    ///
    /// # Arguments
    ///
    /// * `gc_threshold` - Minimum strength to retain (typically 0.1)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let index = BindingIndex::new(0.1);
    /// ```
    #[must_use]
    pub fn new(gc_threshold: f32) -> Self {
        Self {
            episode_to_concepts: DashMap::new(),
            concept_to_episodes: DashMap::new(),
            gc_threshold: gc_threshold.clamp(0.0, 1.0),
            binding_count: AtomicUsize::new(0),
            #[cfg(feature = "dual_memory_cache")]
            fan_out_cache: OnceLock::new(),
        }
    }

    /// Create with pre-allocated capacity
    ///
    /// Pre-allocates capacity to reduce reallocation overhead during
    /// initial binding creation (e.g., during concept formation).
    ///
    /// # Arguments
    ///
    /// * `episode_capacity` - Expected number of episodes with bindings
    /// * `concept_capacity` - Expected number of concepts with bindings
    /// * `gc_threshold` - Minimum strength to retain
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Expect 10K episodes, 1K concepts, 0.1 GC threshold
    /// let index = BindingIndex::with_capacity(10_000, 1_000, 0.1);
    /// ```
    #[must_use]
    pub fn with_capacity(
        episode_capacity: usize,
        concept_capacity: usize,
        gc_threshold: f32,
    ) -> Self {
        Self {
            episode_to_concepts: DashMap::with_capacity(episode_capacity),
            concept_to_episodes: DashMap::with_capacity(concept_capacity),
            gc_threshold: gc_threshold.clamp(0.0, 1.0),
            binding_count: AtomicUsize::new(0),
            #[cfg(feature = "dual_memory_cache")]
            fan_out_cache: OnceLock::new(),
        }
    }

    #[cfg(feature = "dual_memory_cache")]
    fn fan_out_cache(&self) -> Option<&Arc<DualMemoryCache>> {
        self.fan_out_cache.get()
    }

    #[cfg(feature = "dual_memory_cache")]
    /// Attach a shared fan-out cache so activation hot paths can reuse counts.
    pub fn attach_fan_out_cache(&self, cache: &Arc<DualMemoryCache>) {
        if self.fan_out_cache.set(Arc::clone(cache)).is_ok() {
            for entry in &self.concept_to_episodes {
                cache.cache_fan_out(entry.key(), entry.value().len() as u32);
            }
        }
    }

    /// Total number of associations for the provided node identifier.
    #[must_use]
    pub fn association_count(&self, node_id: &Uuid) -> usize {
        if let Some(entry) = self.episode_to_concepts.get(node_id) {
            return entry.value().len();
        }
        self.concept_to_episodes
            .get(node_id)
            .map_or(0, |entry| entry.value().len())
    }

    /// Returns true if the identifier belongs to an episode in the index.
    #[must_use]
    pub fn is_episode_id(&self, node_id: &Uuid) -> bool {
        self.episode_to_concepts.contains_key(node_id)
    }

    /// Returns true if the identifier belongs to a concept in the index.
    #[must_use]
    pub fn is_concept_id(&self, node_id: &Uuid) -> bool {
        self.concept_to_episodes.contains_key(node_id)
    }

    /// Add a binding between episode and concept
    ///
    /// This is the only write path for bindings. Both indices share
    /// the same `Arc<ConceptBinding>` for memory efficiency.
    ///
    /// # Arguments
    ///
    /// * `binding` - Binding to add
    ///
    /// # Performance
    ///
    /// - Typical latency: <100µs (includes both index updates)
    /// - Lock-free, no blocking on contention
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let binding = ConceptBinding::new(
    ///     episode_id,
    ///     concept_id,
    ///     0.8,
    ///     0.6,
    /// );
    /// index.add_binding(binding);
    /// ```
    pub fn add_binding(&self, binding: ConceptBinding) {
        let binding_strength = binding.get_strength();
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

        #[cfg(feature = "dual_memory_cache")]
        if let Some(cache) = self.fan_out_cache() {
            cache.increment_fan_out(&concept_id);
        }

        self.binding_count.fetch_add(1, Ordering::Relaxed);

        // Record binding creation metrics
        #[cfg(feature = "dual_memory_types")]
        if let Some(metrics) = crate::metrics::metrics() {
            metrics.increment_counter("engram_bindings_created_total", 1);
            metrics.record_gauge("engram_binding_avg_strength", f64::from(binding_strength));
        }
    }

    /// Get concepts for an episode (bottom-up access)
    ///
    /// Returns all concept bindings for the given episode.
    ///
    /// # Arguments
    ///
    /// * `episode_id` - Episode UUID
    ///
    /// # Returns
    ///
    /// Vector of concept bindings (typically 1-5 elements)
    ///
    /// # Performance
    ///
    /// - Typical latency: <100ns
    /// - Zero allocations if no bindings exist
    #[must_use]
    pub fn get_concepts_for_episode(&self, episode_id: &Uuid) -> Vec<Arc<ConceptBinding>> {
        self.episode_to_concepts
            .get(episode_id)
            .map(|entry| entry.value().clone())
            .unwrap_or_default()
    }

    /// Get episodes for a concept (top-down access)
    ///
    /// Returns all episode bindings for the given concept.
    /// Uses Arc cloning to avoid copying the entire vector.
    ///
    /// # Arguments
    ///
    /// * `concept_id` - Concept UUID
    ///
    /// # Returns
    ///
    /// Arc-wrapped vector of episode bindings (typically 10-1000 elements)
    ///
    /// # Performance
    ///
    /// - Typical latency: <500ns for 100 episodes
    /// - Cheap Arc clone (no deep copy)
    #[must_use]
    pub fn get_episodes_for_concept(&self, concept_id: &Uuid) -> Arc<Vec<Arc<ConceptBinding>>> {
        self.concept_to_episodes
            .get(concept_id)
            .map_or_else(|| Arc::new(Vec::new()), |entry| Arc::clone(entry.value()))
    }

    /// Get binding strength between episode and concept
    ///
    /// # Arguments
    ///
    /// * `episode_id` - Episode UUID
    /// * `concept_id` - Concept UUID
    ///
    /// # Returns
    ///
    /// Some(strength) if binding exists, None otherwise
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
    ///
    /// Applies the provided function to the current strength and stores
    /// the result atomically.
    ///
    /// # Arguments
    ///
    /// * `episode_id` - Episode UUID
    /// * `concept_id` - Concept UUID
    /// * `update_fn` - Function mapping current strength to new strength
    ///
    /// # Returns
    ///
    /// `true` if binding exists and was updated, `false` otherwise
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Add 0.1 to binding strength
    /// index.update_binding_strength(&ep_id, &con_id, |s| s + 0.1);
    /// ```
    pub fn update_binding_strength<F>(
        &self,
        episode_id: &Uuid,
        concept_id: &Uuid,
        update_fn: F,
    ) -> bool
    where
        F: Fn(f32) -> f32,
    {
        if let Some(entry) = self.episode_to_concepts.get(episode_id)
            && let Some(binding) = entry.value().iter().find(|b| b.concept_id == *concept_id)
        {
            let old_strength = binding.get_strength();
            let updated = binding.update_strength(update_fn);

            #[cfg(feature = "dual_memory_types")]
            if updated {
                let new_strength = binding.get_strength();
                if let Some(metrics) = crate::metrics::metrics() {
                    if new_strength > old_strength {
                        metrics.increment_counter("engram_bindings_strengthened_total", 1);
                    } else if new_strength < old_strength {
                        metrics.increment_counter("engram_bindings_weakened_total", 1);
                    }
                    metrics.record_gauge("engram_binding_avg_strength", f64::from(new_strength));
                }
            }

            return updated;
        }
        false
    }

    /// Remove all bindings for an episode
    ///
    /// Removes the episode from both indices in a single operation.
    ///
    /// # Arguments
    ///
    /// * `episode_id` - Episode UUID
    ///
    /// # Returns
    ///
    /// Number of bindings removed
    pub fn remove_episode_bindings(&self, episode_id: &Uuid) -> usize {
        let mut removed_count = 0;
        #[cfg(feature = "dual_memory_cache")]
        let mut fan_out_deltas: HashMap<Uuid, u32> = HashMap::new();

        if let Some((_, bindings)) = self.episode_to_concepts.remove(episode_id) {
            removed_count = bindings.len();

            // Remove from concept → episodes index
            for binding in bindings {
                #[cfg(feature = "dual_memory_cache")]
                {
                    *fan_out_deltas.entry(binding.concept_id).or_default() += 1;
                }
                self.concept_to_episodes
                    .entry(binding.concept_id)
                    .and_modify(|episodes| {
                        let new_vec: Vec<_> = episodes
                            .iter()
                            .filter(|b| b.episode_id != *episode_id)
                            .cloned()
                            .collect();
                        *episodes = Arc::new(new_vec);
                    });
            }
        }

        self.binding_count
            .fetch_sub(removed_count, Ordering::Relaxed);

        #[cfg(feature = "dual_memory_cache")]
        if removed_count > 0
            && let Some(cache) = self.fan_out_cache()
        {
            for (concept_id, count) in fan_out_deltas {
                cache.decrement_fan_out_by(&concept_id, count);
            }
        }
        removed_count
    }

    /// Remove all bindings for a concept
    ///
    /// Removes the concept from both indices in a single operation.
    ///
    /// # Arguments
    ///
    /// * `concept_id` - Concept UUID
    ///
    /// # Returns
    ///
    /// Number of bindings removed
    pub fn remove_concept_bindings(&self, concept_id: &Uuid) -> usize {
        let mut removed_count = 0;

        if let Some((_, episodes_arc)) = self.concept_to_episodes.remove(concept_id) {
            removed_count = episodes_arc.len();

            #[cfg(feature = "dual_memory_cache")]
            if removed_count > 0
                && let Some(cache) = self.fan_out_cache()
            {
                cache.cache_fan_out(concept_id, 0);
            }

            // Remove from episode → concepts index
            for binding in episodes_arc.iter() {
                self.episode_to_concepts
                    .entry(binding.episode_id)
                    .and_modify(|concepts| {
                        concepts.retain(|b| b.concept_id != *concept_id);
                    });
            }
        }

        self.binding_count
            .fetch_sub(removed_count, Ordering::Relaxed);
        removed_count
    }

    /// Lazy garbage collection: remove weak bindings
    ///
    /// Returns number of bindings removed.
    /// Should be called periodically (e.g., during consolidation).
    ///
    /// # Performance
    ///
    /// - Typical latency: <1ms for 10K bindings
    /// - Two-phase: identify weak bindings, then batch remove
    /// - Does not block concurrent reads/writes
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Run GC during consolidation
    /// let removed = index.garbage_collect();
    /// println!("Removed {} weak bindings", removed);
    /// ```
    pub fn garbage_collect(&self) -> usize {
        let mut removed = 0;
        let threshold = self.gc_threshold;

        // Collect weak bindings
        let mut weak_bindings = Vec::new();
        #[cfg(feature = "dual_memory_cache")]
        let mut fan_out_deltas: HashMap<Uuid, u32> = HashMap::new();

        for entry in &self.episode_to_concepts {
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
                #[cfg(feature = "dual_memory_cache")]
                if initial_len != entry.len() {
                    let delta = (initial_len - entry.len()) as u32;
                    if delta > 0 {
                        *fan_out_deltas.entry(concept_id).or_default() += delta;
                    }
                }
            }

            // Remove from concept → episodes
            self.concept_to_episodes
                .entry(concept_id)
                .and_modify(|episodes| {
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

        self.binding_count.fetch_sub(removed, Ordering::Relaxed);

        #[cfg(feature = "dual_memory_cache")]
        if removed > 0
            && let Some(cache) = self.fan_out_cache()
        {
            for (concept_id, delta) in fan_out_deltas {
                cache.decrement_fan_out_by(&concept_id, delta);
            }
        }

        // Record binding pruning metrics
        #[cfg(feature = "dual_memory_types")]
        if removed > 0 {
            if let Some(metrics) = crate::metrics::metrics() {
                metrics.increment_counter("engram_bindings_pruned_total", removed as u64);
            }
        }

        removed
    }

    /// Get total binding count
    #[must_use]
    pub fn count(&self) -> usize {
        self.binding_count.load(Ordering::Relaxed)
    }

    /// Get memory usage statistics
    ///
    /// Returns detailed breakdown of memory usage for capacity planning
    /// and overhead monitoring.
    #[must_use]
    pub fn memory_stats(&self) -> BindingMemoryStats {
        let binding_size = std::mem::size_of::<ConceptBinding>();
        let arc_overhead =
            std::mem::size_of::<Arc<ConceptBinding>>() - std::mem::size_of::<*const ()>();

        let total_bindings = self.count();
        let binding_memory = total_bindings * binding_size;
        let arc_memory = total_bindings * 2 * arc_overhead; // Stored in both indices
        let index_overhead =
            self.episode_to_concepts.len() * 32 + self.concept_to_episodes.len() * 32; // Rough estimate

        BindingMemoryStats {
            total_bindings,
            binding_memory_bytes: binding_memory,
            arc_overhead_bytes: arc_memory,
            index_overhead_bytes: index_overhead,
            total_bytes: binding_memory + arc_memory + index_overhead,
        }
    }

    /// Find concepts by embedding similarity
    ///
    /// This method is currently stubbed as it requires access to concept embeddings.
    /// The BindingIndex only stores episode-concept connections, not concept embeddings.
    /// Semantic pathway implementation should use the backend's concept iteration directly.
    ///
    /// # Arguments
    ///
    /// * `embedding` - Query embedding vector
    ///
    /// # Returns
    ///
    /// Empty vector (stubbed - requires backend concept access)
    ///
    /// # Implementation Note
    ///
    /// This is intentionally a stub. The semantic pathway should:
    /// 1. Access concepts directly from `DualDashMapBackend::iter_concepts()`
    /// 2. Compute similarity scores using cosine similarity
    /// 3. Use `get_bindings_from_concept()` to map concepts → episodes
    #[must_use]
    pub const fn find_concepts_by_embedding(_embedding: &[f32]) -> Vec<(Uuid, f32)> {
        // Stub: Requires backend access to concept embeddings
        // Semantic pathway should query backend directly for concepts
        Vec::new()
    }

    /// Get all bindings from a concept to its episodes
    ///
    /// Returns the full list of episode bindings for semantic pathway aggregation.
    ///
    /// # Arguments
    ///
    /// * `concept_id` - Concept UUID
    ///
    /// # Returns
    ///
    /// Vector of concept bindings (cloned from Arc for downstream processing)
    ///
    /// # Performance
    ///
    /// - Typical latency: <500ns for 100 episodes
    /// - Uses Arc cloning (cheap pointer copy, no deep clone)
    #[must_use]
    pub fn get_bindings_from_concept(&self, concept_id: &Uuid) -> Vec<ConceptBinding> {
        self.concept_to_episodes
            .get(concept_id)
            .map(|episodes_arc| {
                // Clone the Arc'd bindings into owned ConceptBinding instances
                episodes_arc
                    .iter()
                    .map(|binding_arc| {
                        // Create owned ConceptBinding from Arc<ConceptBinding>
                        ConceptBinding::new(
                            binding_arc.episode_id,
                            binding_arc.concept_id,
                            binding_arc.get_strength(),
                            binding_arc.contribution,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::bindings::ConceptBinding;

    #[test]
    fn test_index_creation() {
        let index = BindingIndex::new(0.1);
        assert_eq!(index.count(), 0);
    }

    #[test]
    fn test_add_binding() {
        let index = BindingIndex::new(0.1);
        let ep_id = Uuid::new_v4();
        let con_id = Uuid::new_v4();

        let binding = ConceptBinding::new(ep_id, con_id, 0.8, 0.6);
        index.add_binding(binding);

        assert_eq!(index.count(), 1);
    }

    #[test]
    fn test_bidirectional_lookup() {
        let index = BindingIndex::new(0.1);
        let ep_id = Uuid::new_v4();
        let con_id = Uuid::new_v4();

        let binding = ConceptBinding::new(ep_id, con_id, 0.8, 0.6);
        index.add_binding(binding);

        // Episode → Concepts
        let concepts = index.get_concepts_for_episode(&ep_id);
        assert_eq!(concepts.len(), 1);
        assert_eq!(concepts[0].concept_id, con_id);

        // Concept → Episodes
        let episodes = index.get_episodes_for_concept(&con_id);
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].episode_id, ep_id);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn test_strength_update() {
        let index = BindingIndex::new(0.1);
        let ep_id = Uuid::new_v4();
        let con_id = Uuid::new_v4();

        let binding = ConceptBinding::new(ep_id, con_id, 0.5, 0.6);
        index.add_binding(binding);

        // Update strength
        let updated = index.update_binding_strength(&ep_id, &con_id, |s| s + 0.2);
        assert!(updated);

        // Verify new strength
        let strength = index
            .get_binding_strength(&ep_id, &con_id)
            .expect("Binding should exist after update");
        assert!((strength - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_remove_episode_bindings() {
        let index = BindingIndex::new(0.1);
        let ep_id = Uuid::new_v4();
        let con_id1 = Uuid::new_v4();
        let con_id2 = Uuid::new_v4();

        index.add_binding(ConceptBinding::new(ep_id, con_id1, 0.8, 0.6));
        index.add_binding(ConceptBinding::new(ep_id, con_id2, 0.7, 0.5));

        assert_eq!(index.count(), 2);

        let removed = index.remove_episode_bindings(&ep_id);
        assert_eq!(removed, 2);
        assert_eq!(index.count(), 0);

        // Verify removed from both indices
        assert!(index.get_concepts_for_episode(&ep_id).is_empty());
        assert!(index.get_episodes_for_concept(&con_id1).is_empty());
        assert!(index.get_episodes_for_concept(&con_id2).is_empty());
    }

    #[test]
    fn test_garbage_collection() {
        let index = BindingIndex::new(0.3);
        let ep_id1 = Uuid::new_v4();
        let ep_id2 = Uuid::new_v4();
        let con_id = Uuid::new_v4();

        // Add strong and weak bindings
        let strong = ConceptBinding::new(ep_id1, con_id, 0.8, 0.6);
        let weak = ConceptBinding::new(ep_id2, con_id, 0.2, 0.4);

        index.add_binding(strong);
        index.add_binding(weak);

        assert_eq!(index.count(), 2);

        // GC should remove weak binding
        let removed = index.garbage_collect();
        assert_eq!(removed, 1);
        assert_eq!(index.count(), 1);

        // Verify strong binding remains
        let concepts = index.get_concepts_for_episode(&ep_id1);
        assert_eq!(concepts.len(), 1);

        // Verify weak binding removed
        let concepts = index.get_concepts_for_episode(&ep_id2);
        assert!(concepts.is_empty());
    }

    #[test]
    fn test_memory_stats() {
        let index = BindingIndex::with_capacity(1000, 500, 0.1);

        for _ in 0..10 {
            let binding = ConceptBinding::new(Uuid::new_v4(), Uuid::new_v4(), 0.8, 0.6);
            index.add_binding(binding);
        }

        let stats = index.memory_stats();
        assert_eq!(stats.total_bindings, 10);
        assert!(stats.total_bytes > 0);
        assert!(stats.binding_memory_bytes > 0);
    }

    #[test]
    #[allow(clippy::panic)]
    fn test_concurrent_add() {
        use std::thread;

        let index = Arc::new(BindingIndex::new(0.1));
        let con_id = Uuid::new_v4();

        let threads: Vec<_> = (0..10)
            .map(|_| {
                let idx = Arc::clone(&index);
                let c_id = con_id;
                thread::spawn(move || {
                    for _ in 0..10 {
                        let binding = ConceptBinding::new(Uuid::new_v4(), c_id, 0.8, 0.6);
                        idx.add_binding(binding);
                    }
                })
            })
            .collect();

        for thread in threads {
            if let Err(e) = thread.join() {
                panic!("Thread panicked during concurrent binding test: {e:?}");
            }
        }

        assert_eq!(index.count(), 100);

        // Verify all episodes bound to concept
        let episodes = index.get_episodes_for_concept(&con_id);
        assert_eq!(episodes.len(), 100);
    }

    #[test]
    fn test_get_bindings_from_concept() {
        let index = BindingIndex::new(0.1);
        let con_id = Uuid::new_v4();
        let ep_id1 = Uuid::new_v4();
        let ep_id2 = Uuid::new_v4();
        let ep_id3 = Uuid::new_v4();

        // Add bindings to concept
        index.add_binding(ConceptBinding::new(ep_id1, con_id, 0.8, 0.7));
        index.add_binding(ConceptBinding::new(ep_id2, con_id, 0.6, 0.5));
        index.add_binding(ConceptBinding::new(ep_id3, con_id, 0.9, 0.8));

        // Get bindings from concept
        let concept_bindings = index.get_bindings_from_concept(&con_id);
        assert_eq!(concept_bindings.len(), 3);

        // Verify bindings contain expected episode IDs
        let episode_ids: Vec<Uuid> = concept_bindings.iter().map(|b| b.episode_id).collect();
        assert!(episode_ids.contains(&ep_id1));
        assert!(episode_ids.contains(&ep_id2));
        assert!(episode_ids.contains(&ep_id3));

        // Verify strengths are preserved
        let binding1 = concept_bindings.iter().find(|b| b.episode_id == ep_id1);
        assert!(binding1.is_some(), "binding should exist for ep_id1");
        assert!((binding1.as_ref().map_or(0.0, |b| b.get_strength()) - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_get_bindings_from_nonexistent_concept() {
        let index = BindingIndex::new(0.1);
        let con_id = Uuid::new_v4();

        // Get bindings for concept with no bindings
        let bindings = index.get_bindings_from_concept(&con_id);
        assert!(bindings.is_empty());
    }

    #[test]
    fn test_find_concepts_by_embedding_stub() {
        let embedding = [0.5; 768];

        // This is intentionally stubbed - should return empty
        let concepts = BindingIndex::find_concepts_by_embedding(&embedding);
        assert!(concepts.is_empty());
    }
}
