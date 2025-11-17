use std::array;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use atomic_float::AtomicF32;
use dashmap::{DashMap, mapref::entry::Entry};
use uuid::Uuid;

use crate::EMBEDDING_DIM;

const BINDING_SHARDS: usize = 16;

#[repr(align(64))]
#[derive(Debug)]
/// Cache-aligned metadata describing a concept node.
pub struct ConceptMetadata {
    fan_out_count: AtomicU32,
    last_activation: AtomicF32,
    binding_version: AtomicU32,
}

impl ConceptMetadata {
    /// Return the cached fan-out count for a concept.
    #[must_use]
    pub fn fan_out(&self) -> u32 {
        self.fan_out_count.load(Ordering::Relaxed)
    }

    /// Increment and return the updated fan-out count.
    pub fn increment_fan_out(&self) -> u32 {
        self.increment_fan_out_by(1)
    }

    /// Increment the fan-out counter by an arbitrary amount and return the new value.
    pub fn increment_fan_out_by(&self, amount: u32) -> u32 {
        if amount == 0 {
            return self.fan_out();
        }
        self.fan_out_count.fetch_add(amount, Ordering::Relaxed) + amount
    }

    /// Decrement the fan-out counter and return the new value (saturating at zero).
    pub fn decrement_fan_out_by(&self, amount: u32) -> u32 {
        if amount == 0 {
            return self.fan_out();
        }
        loop {
            let current = self.fan_out_count.load(Ordering::Relaxed);
            let new_value = current.saturating_sub(amount);
            if self
                .fan_out_count
                .compare_exchange(current, new_value, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return new_value;
            }
        }
    }

    /// Overwrite the cached fan-out count.
    pub fn set_fan_out(&self, fan_out: u32) {
        self.fan_out_count.store(fan_out, Ordering::Relaxed);
    }

    /// Store the latest activation score observed for the concept.
    pub fn set_last_activation(&self, activation: f32) {
        self.last_activation.store(activation, Ordering::Relaxed);
    }

    /// Load the cached activation score.
    #[must_use]
    pub fn last_activation(&self) -> f32 {
        self.last_activation.load(Ordering::Relaxed)
    }

    /// Increment the binding version used for cache invalidation.
    pub fn bump_binding_version(&self) -> u32 {
        self.binding_version.fetch_add(1, Ordering::Relaxed) + 1
    }
}

impl Default for ConceptMetadata {
    fn default() -> Self {
        Self {
            fan_out_count: AtomicU32::new(0),
            last_activation: AtomicF32::new(0.0),
            binding_version: AtomicU32::new(0),
        }
    }
}

#[derive(Debug)]
/// Aggregated counters describing cache effectiveness.
pub struct CacheStatistics {
    /// Cached fan-out lookups served from memory.
    pub fan_out_hits: AtomicU64,
    /// Fan-out lookups that required recomputation.
    pub fan_out_misses: AtomicU64,
    /// Cached centroid lookups served from memory.
    pub centroid_hits: AtomicU64,
    /// Centroid requests that missed in the cache.
    pub centroid_misses: AtomicU64,
    /// Approximate number of cached binding strengths.
    pub binding_cache_size: AtomicUsize,
}

impl CacheStatistics {
    /// Create a zeroed statistics snapshot.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            fan_out_hits: AtomicU64::new(0),
            fan_out_misses: AtomicU64::new(0),
            centroid_hits: AtomicU64::new(0),
            centroid_misses: AtomicU64::new(0),
            binding_cache_size: AtomicUsize::new(0),
        }
    }

    #[must_use]
    /// Percentage of lookups served from cache.
    pub fn hit_rate(&self) -> f64 {
        let hits =
            self.fan_out_hits.load(Ordering::Relaxed) + self.centroid_hits.load(Ordering::Relaxed);
        let misses = self.fan_out_misses.load(Ordering::Relaxed)
            + self.centroid_misses.load(Ordering::Relaxed);
        if hits + misses == 0 {
            return 0.0;
        }
        hits as f64 / (hits + misses) as f64
    }
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self::new()
    }
}

type BindingMap = Arc<DashMap<Uuid, Arc<AtomicF32>>>;

type BindingShard = DashMap<Uuid, BindingMap>;

#[derive(Debug)]
/// Lock-free cache of concept metadata, centroids, and bindings.
pub struct DualMemoryCache {
    concept_metadata: DashMap<Uuid, Arc<ConceptMetadata>>,
    centroid_cache: DashMap<Uuid, Arc<[f32; EMBEDDING_DIM]>>,
    binding_index: Arc<[BindingShard; BINDING_SHARDS]>,
    cache_stats: CacheStatistics,
}

impl DualMemoryCache {
    #[must_use]
    /// Construct a cache with the default number of shards.
    pub fn new() -> Self {
        let shards: [BindingShard; BINDING_SHARDS] = array::from_fn(|_| DashMap::new());
        Self {
            concept_metadata: DashMap::new(),
            centroid_cache: DashMap::new(),
            binding_index: Arc::new(shards),
            cache_stats: CacheStatistics::new(),
        }
    }

    #[inline]
    const fn shard_for(concept_id: &Uuid) -> usize {
        (concept_id.as_u128() as usize) & (BINDING_SHARDS - 1)
    }

    fn metadata_entry(&self, concept_id: &Uuid) -> Arc<ConceptMetadata> {
        match self.concept_metadata.entry(*concept_id) {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => {
                let inserted = entry.insert(Arc::new(ConceptMetadata::default()));
                Arc::clone(&*inserted)
            }
        }
    }

    /// Update the last activation value touched for the concept.
    pub fn set_last_activation(&self, concept_id: &Uuid, activation: f32) {
        let metadata = self.metadata_entry(concept_id);
        metadata.set_last_activation(activation);
    }

    /// Fetch the cached fan-out count if available.
    #[must_use]
    pub fn get_fan_out(&self, concept_id: &Uuid) -> Option<u32> {
        self.concept_metadata
            .get(concept_id)
            .map(|meta| {
                self.cache_stats
                    .fan_out_hits
                    .fetch_add(1, Ordering::Relaxed);
                meta.fan_out()
            })
            .or_else(|| {
                self.cache_stats
                    .fan_out_misses
                    .fetch_add(1, Ordering::Relaxed);
                None
            })
    }

    /// Increment the fan-out counter and return the new value.
    pub fn increment_fan_out(&self, concept_id: &Uuid) -> u32 {
        self.increment_fan_out_by(concept_id, 1)
    }

    /// Increment the fan-out counter by the provided amount.
    pub fn increment_fan_out_by(&self, concept_id: &Uuid, amount: u32) -> u32 {
        let metadata = self.metadata_entry(concept_id);
        metadata.increment_fan_out_by(amount)
    }

    /// Decrement the fan-out counter (saturating at zero).
    pub fn decrement_fan_out_by(&self, concept_id: &Uuid, amount: u32) -> u32 {
        let metadata = self.metadata_entry(concept_id);
        metadata.decrement_fan_out_by(amount)
    }

    /// Persist an authoritative fan-out count from the binding index.
    pub fn cache_fan_out(&self, concept_id: &Uuid, fan_out: u32) {
        let metadata = self.metadata_entry(concept_id);
        metadata.set_fan_out(fan_out);
    }

    /// Insert or replace the centroid cache entry for a concept.
    pub fn upsert_centroid(&self, concept_id: Uuid, centroid: &[f32; EMBEDDING_DIM]) {
        self.centroid_cache.insert(concept_id, Arc::new(*centroid));
    }

    /// Cache a centroid using an existing reference-counted allocation.
    pub fn cache_centroid_arc(&self, concept_id: Uuid, centroid: Arc<[f32; EMBEDDING_DIM]>) {
        self.centroid_cache.insert(concept_id, centroid);
    }

    #[must_use]
    /// Fetch a cached centroid for the provided concept identifier.
    pub fn get_centroid(&self, concept_id: &Uuid) -> Option<Arc<[f32; EMBEDDING_DIM]>> {
        self.centroid_cache
            .get(concept_id)
            .map(|entry| {
                self.cache_stats
                    .centroid_hits
                    .fetch_add(1, Ordering::Relaxed);
                Arc::clone(entry.value())
            })
            .or_else(|| {
                self.cache_stats
                    .centroid_misses
                    .fetch_add(1, Ordering::Relaxed);
                None
            })
    }

    fn binding_map(&self, concept_id: &Uuid) -> BindingMap {
        let shard = &self.binding_index[Self::shard_for(concept_id)];
        match shard.entry(*concept_id) {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => {
                let inserted = entry.insert(Arc::new(DashMap::new()));
                Arc::clone(&*inserted)
            }
        }
    }

    /// Record a binding strength for a concept/episode pair.
    pub fn update_binding(&self, concept_id: Uuid, episode_id: Uuid, strength: f32) {
        let map = self.binding_map(&concept_id);
        if let Some(existing) = map.get(&episode_id) {
            existing.value().store(strength, Ordering::Relaxed);
            return;
        }
        self.cache_stats
            .binding_cache_size
            .fetch_add(1, Ordering::Relaxed);
        let arc = Arc::new(AtomicF32::new(strength));
        map.insert(episode_id, Arc::clone(&arc));
    }

    /// Look up a cached binding strength.
    #[must_use]
    pub fn binding_strength(&self, concept_id: &Uuid, episode_id: &Uuid) -> Option<f32> {
        let shard = &self.binding_index[Self::shard_for(concept_id)];
        shard.get(concept_id).and_then(|map| {
            map.value()
                .get(episode_id)
                .map(|value| value.load(Ordering::Relaxed))
        })
    }

    #[cfg(all(target_arch = "x86_64", feature = "dual_memory_cache"))]
    /// Software prefetch for concept metadata on x86 targets.
    pub fn prefetch_concepts(&self, concept_ids: &[Uuid]) {
        for chunk in concept_ids.chunks(8) {
            for id in chunk {
                if let Some(entry) = self.concept_metadata.get(id) {
                    unsafe {
                        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                        _mm_prefetch(
                            entry.value().as_ref() as *const ConceptMetadata as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                }
            }
        }
    }

    #[cfg(not(all(target_arch = "x86_64", feature = "dual_memory_cache")))]
    /// No-op fallback on non x86 targets.
    pub const fn prefetch_concepts(&self, _concept_ids: &[Uuid]) {
        let _ = self;
    }

    #[must_use]
    /// Expose the shared cache statistics.
    pub const fn stats(&self) -> &CacheStatistics {
        &self.cache_stats
    }
}

impl Default for DualMemoryCache {
    fn default() -> Self {
        Self::new()
    }
}
