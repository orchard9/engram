//! Infallible memory store with graceful degradation
//!
//! Implements cognitive design principles where operations never fail but
//! degrade gracefully under pressure, returning activation levels that
//! indicate store quality.

use crate::{Episode, Memory};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Activation level returned by store operations
///
/// Indicates the quality of a store operation from 0.0 to 1.0
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Activation(f32);

impl Activation {
    /// Create a new activation level
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Get the raw activation value
    pub fn value(&self) -> f32 {
        self.0
    }

    /// Check if activation indicates successful store
    pub fn is_successful(&self) -> bool {
        self.0 > 0.5
    }

    /// Check if activation indicates degraded store
    pub fn is_degraded(&self) -> bool {
        self.0 < 0.8
    }
}

/// Memory store that never fails, degrading gracefully under pressure
///
/// # Cognitive Design
///
/// This store follows human memory formation patterns:
/// - Store quality varies based on system state (like attention/fatigue)
/// - Returns activation levels instead of Result types
/// - Graceful degradation mirrors biological memory under stress
/// - Concurrent stores don't block (like parallel memory formation)
pub struct MemoryStore {
    /// Lock-free map for high-activation memories
    hot_memories: DashMap<String, Arc<Memory>>,

    /// Sorted map for eviction candidates (by activation level)
    eviction_queue: RwLock<BTreeMap<(OrderedFloat, String), Arc<Memory>>>,

    /// Current memory count
    memory_count: AtomicUsize,

    /// Maximum memories before eviction
    max_memories: usize,

    /// System pressure indicator (0.0 = no pressure, 1.0 = max pressure)
    pressure: RwLock<f32>,

    /// Write-ahead log for durability (non-blocking)
    wal_buffer: Arc<DashMap<String, Episode>>,
}

/// Wrapper for f32 that implements Ord for BTreeMap
#[derive(Clone, Copy, Debug)]
struct OrderedFloat(f32);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl MemoryStore {
    /// Create a new memory store with specified capacity
    pub fn new(max_memories: usize) -> Self {
        Self {
            hot_memories: DashMap::new(),
            eviction_queue: RwLock::new(BTreeMap::new()),
            memory_count: AtomicUsize::new(0),
            max_memories,
            pressure: RwLock::new(0.0),
            wal_buffer: Arc::new(DashMap::new()),
        }
    }

    /// Store an episode, returning activation level indicating store quality
    ///
    /// # Returns
    ///
    /// Activation level from 0.0 to 1.0 indicating:
    /// - 1.0: Perfect store with full confidence
    /// - 0.8-0.9: Normal store with slight system pressure
    /// - 0.5-0.7: Degraded store under memory pressure
    /// - 0.3-0.4: Heavily degraded, may be evicted soon
    /// - < 0.3: Critical pressure, immediate eviction likely
    ///
    /// # Cognitive Design
    ///
    /// Never returns errors because human memory formation doesn't "fail" -
    /// it degrades. Under stress or fatigue, memories form with lower
    /// confidence and are more likely to be forgotten.
    pub fn store(&self, episode: Episode) -> Activation {
        // Calculate system pressure
        let current_count = self.memory_count.load(Ordering::Relaxed);
        let pressure = (current_count as f32 / self.max_memories as f32).min(1.0);

        // Update system pressure
        {
            let mut p = self.pressure.write();
            *p = pressure;
        }

        // Calculate base activation from episode encoding confidence and pressure
        let base_activation = episode.encoding_confidence.raw() * (1.0 - pressure * 0.5);

        // Check if we need to evict
        if current_count >= self.max_memories {
            self.evict_lowest_activation();
        }

        // Convert episode to memory
        let memory = Memory::from_episode(episode.clone(), base_activation);
        let memory_id = memory.id.clone();
        let memory_arc = Arc::new(memory);

        // Store in hot tier (lock-free)
        self.hot_memories
            .insert(memory_id.clone(), memory_arc.clone());

        // Add to eviction queue
        {
            let mut queue = self.eviction_queue.write();
            queue.insert(
                (OrderedFloat(base_activation), memory_id.clone()),
                memory_arc,
            );
        }

        // Store in WAL buffer (non-blocking)
        self.wal_buffer.insert(memory_id, episode);

        // Increment count
        self.memory_count.fetch_add(1, Ordering::Relaxed);

        // Return activation adjusted for any degradation
        Activation::new(base_activation)
    }

    /// Evict the memory with lowest activation
    fn evict_lowest_activation(&self) {
        let mut queue = self.eviction_queue.write();

        if let Some(((_, id), _)) = queue.iter().next() {
            let id = id.clone();

            // Remove from hot memories
            self.hot_memories.remove(&id);

            // Remove from eviction queue
            queue.retain(|k, _| k.1 != id);

            // Remove from WAL buffer
            self.wal_buffer.remove(&id);

            // Decrement count
            self.memory_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Get a memory by ID
    pub fn get(&self, id: &str) -> Option<Arc<Memory>> {
        self.hot_memories.get(id).map(|entry| entry.clone())
    }

    /// Get current system pressure
    pub fn pressure(&self) -> f32 {
        *self.pressure.read()
    }

    /// Get current memory count
    pub fn count(&self) -> usize {
        self.memory_count.load(Ordering::Relaxed)
    }

    /// Check if store can accept more memories without eviction
    pub fn has_capacity(&self) -> bool {
        self.count() < self.max_memories
    }
}

/// Extension trait to convert Episode to Memory
trait EpisodeToMemory {
    fn from_episode(episode: Episode, activation: f32) -> Self;
}

impl EpisodeToMemory for Memory {
    fn from_episode(episode: Episode, activation: f32) -> Self {
        let mut memory = Memory::new(
            format!("mem_{}", episode.id),
            episode.embedding,
            episode.encoding_confidence,
        );

        memory.set_activation(activation);
        memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Confidence, EpisodeBuilder};
    use chrono::Utc;

    #[test]
    fn test_store_returns_activation() {
        let store = MemoryStore::new(10);

        let episode = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(Utc::now())
            .what("test episode".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .build();

        let activation = store.store(episode);

        // Should return high activation with no pressure
        assert!(activation.value() > 0.8);
        assert!(activation.value() <= 1.0);
    }

    #[test]
    fn test_store_never_panics() {
        let store = MemoryStore::new(10);

        // Store many episodes to trigger eviction
        for i in 0..20 {
            let episode = EpisodeBuilder::new()
                .id(format!("ep{}", i))
                .when(Utc::now())
                .what("test episode".to_string())
                .embedding([0.1; 768])
                .confidence(Confidence::MEDIUM)
                .build();

            let activation = store.store(episode);

            // Should always return valid activation
            assert!(activation.value() >= 0.0);
            assert!(activation.value() <= 1.0);
        }

        // Should have evicted old memories
        assert_eq!(store.count(), 10);
    }

    #[test]
    fn test_concurrent_stores_dont_block() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(MemoryStore::new(100));
        let mut handles = vec![];

        // Spawn multiple threads storing concurrently
        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = thread::spawn(move || {
                let episode = EpisodeBuilder::new()
                    .id(format!("ep_thread_{}", i))
                    .when(Utc::now())
                    .what("concurrent episode".to_string())
                    .embedding([0.1; 768])
                    .confidence(Confidence::HIGH)
                    .build();

                store_clone.store(episode)
            });
            handles.push(handle);
        }

        // All stores should complete without blocking
        for handle in handles {
            let activation = handle.join().unwrap();
            assert!(activation.value() > 0.0);
        }

        // All episodes should be stored
        assert_eq!(store.count(), 10);
    }

    #[test]
    fn test_degraded_store_under_pressure() {
        let store = MemoryStore::new(10);

        // Fill store to capacity
        for i in 0..9 {
            let episode = EpisodeBuilder::new()
                .id(format!("ep{}", i))
                .when(Utc::now())
                .what("test episode".to_string())
                .embedding([0.1; 768])
                .confidence(Confidence::MEDIUM)
                .build();

            store.store(episode);
        }

        // Store at near capacity - should show degradation
        let episode = EpisodeBuilder::new()
            .id("ep_pressure".to_string())
            .when(Utc::now())
            .what("pressure test".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .build();

        let activation = store.store(episode);

        // Activation should be degraded due to pressure
        assert!(activation.value() < 0.9);
        assert!(activation.value() > 0.4);

        // Pressure should be high
        assert!(store.pressure() > 0.8);
    }

    #[test]
    fn test_eviction_of_low_activation() {
        let store = MemoryStore::new(3);

        // Store episodes with different confidence levels
        let low_conf = EpisodeBuilder::new()
            .id("low".to_string())
            .when(Utc::now())
            .what("low confidence episode".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::LOW)
            .build();

        let med_conf = EpisodeBuilder::new()
            .id("med".to_string())
            .when(Utc::now())
            .what("medium confidence episode".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::MEDIUM)
            .build();

        let high_conf = EpisodeBuilder::new()
            .id("high".to_string())
            .when(Utc::now())
            .what("high confidence episode".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .build();

        store.store(low_conf);
        store.store(med_conf);
        store.store(high_conf);

        // Store one more to trigger eviction
        let new_episode = EpisodeBuilder::new()
            .id("new".to_string())
            .when(Utc::now())
            .what("new episode".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .build();

        store.store(new_episode);

        // Low confidence memory should be evicted
        assert!(store.get("mem_low").is_none());
        assert!(store.get("mem_med").is_some());
        assert!(store.get("mem_high").is_some());
        assert!(store.get("mem_new").is_some());
    }
}
