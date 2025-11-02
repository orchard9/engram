//! Hot tier storage implementation using `DashMap` for concurrent access
//!
//! This module provides ultra-fast in-memory storage for highly active memories.
//! Key features:
//! - Lock-free concurrent access via `DashMap`
//! - Sub-100μs retrieval latency
//! - SIMD-optimized similarity search
//! - Automatic activation tracking and decay
//! - Memory pressure-aware eviction

use super::confidence::{ConfidenceTier, StorageConfidenceCalibrator};
use super::{StorageError, StorageTierBackend, TierStatistics};
use crate::{
    Confidence, Cue, CueType, Episode, EpisodeBuilder, Memory, compute, numeric::unit_ratio_to_f32,
};
use dashmap::DashMap;
use std::convert::TryFrom;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

/// Hot tier storage using lock-free `DashMap` for maximum performance
pub struct HotTier {
    /// Primary storage for active memories
    pub data: DashMap<String, Arc<Memory>>,
    /// Access timestamps for LRU eviction
    pub access_times: DashMap<String, u64>,
    /// Performance metrics
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    total_accesses: AtomicU64,
    /// Capacity limit
    pub max_capacity: usize,
    /// Confidence calibrator for storage tier adjustments
    confidence_calibrator: StorageConfidenceCalibrator,
}

impl HotTier {
    /// Create a new hot tier with specified capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            data: DashMap::with_capacity(capacity),
            access_times: DashMap::with_capacity(capacity),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            total_accesses: AtomicU64::new(0),
            max_capacity: capacity,
            confidence_calibrator: StorageConfidenceCalibrator::new(),
        }
    }

    /// Create a new hot tier with custom confidence calibrator
    #[must_use]
    pub fn with_confidence_calibrator(
        capacity: usize,
        calibrator: StorageConfidenceCalibrator,
    ) -> Self {
        Self {
            data: DashMap::with_capacity(capacity),
            access_times: DashMap::with_capacity(capacity),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            total_accesses: AtomicU64::new(0),
            max_capacity: capacity,
            confidence_calibrator: calibrator,
        }
    }

    /// Get current timestamp as nanoseconds since UNIX epoch
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .try_into()
            .unwrap_or(u64::MAX)
    }

    /// Record access to a memory for LRU tracking
    fn record_access(&self, memory_id: &str) {
        let timestamp = Self::current_timestamp();
        self.access_times.insert(memory_id.to_string(), timestamp);
        self.total_accesses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get storage duration for a memory (time since first stored)
    #[must_use]
    pub fn get_storage_duration(&self, memory_id: &str) -> std::time::Duration {
        self.access_times.get(memory_id).map_or_else(
            || std::time::Duration::from_secs(0),
            |entry| {
                let stored_timestamp = *entry.value();
                let current_timestamp = Self::current_timestamp();
                let duration_nanos = current_timestamp.saturating_sub(stored_timestamp);
                std::time::Duration::from_nanos(duration_nanos)
            },
        )
    }

    /// Apply confidence calibration for hot tier retrieval
    #[must_use]
    pub fn calibrate_confidence(&self, raw_confidence: Confidence, memory_id: &str) -> Confidence {
        let storage_duration = self.get_storage_duration(memory_id);
        self.confidence_calibrator.adjust_for_storage_tier(
            raw_confidence,
            ConfidenceTier::Hot,
            storage_duration,
        )
    }

    /// Batch calibrate confidence for multiple results
    pub fn calibrate_confidence_batch(&self, results: &mut [(Episode, Confidence)]) {
        for (episode, confidence) in results.iter_mut() {
            let storage_duration = self.get_storage_duration(&episode.id);
            *confidence = self.confidence_calibrator.adjust_for_storage_tier(
                *confidence,
                ConfidenceTier::Hot,
                storage_duration,
            );
        }
    }

    /// Check if tier is approaching capacity
    #[must_use]
    pub fn is_near_capacity(&self) -> bool {
        if self.max_capacity == 0 {
            return false;
        }

        let current = u64::try_from(self.data.len()).unwrap_or(u64::MAX);
        let capacity = u64::try_from(self.max_capacity).unwrap_or(u64::MAX);

        if capacity == 0 {
            false
        } else {
            current.saturating_mul(100) > capacity.saturating_mul(80)
        }
    }

    /// Check whether the tier currently holds the provided memory identifier.
    #[must_use]
    pub fn contains_memory(&self, memory_id: &str) -> bool {
        self.data.contains_key(memory_id)
    }

    /// Get least recently used memories for eviction
    #[must_use]
    pub fn get_lru_candidates(&self, count: usize) -> Vec<String> {
        let mut candidates: Vec<(String, u64)> = self
            .access_times
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();

        // Sort by access time (oldest first)
        candidates.sort_by_key(|&(_, timestamp)| timestamp);

        candidates
            .into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect()
    }

    /// Perform fast embedding similarity search using SIMD
    fn embedding_similarity_search(
        &self,
        query_vector: &[f32; 768],
        threshold: Confidence,
    ) -> Vec<(Episode, Confidence)> {
        let mut results = Vec::new();

        // Batch process memories for SIMD efficiency
        let memories: Vec<_> = self.data.iter().collect();

        for entry in memories {
            let memory = entry.value();

            // Use SIMD-optimized cosine similarity from compute module
            let similarity = compute::cosine_similarity_768(query_vector, &memory.embedding);

            if similarity >= threshold.raw() {
                let episode = Self::memory_to_episode(memory);
                let raw_confidence = Confidence::from_raw(similarity);

                // Apply storage tier confidence calibration
                let storage_duration = self.get_storage_duration(&memory.id);
                let calibrated_confidence = self.confidence_calibrator.adjust_for_storage_tier(
                    raw_confidence,
                    ConfidenceTier::Hot,
                    storage_duration,
                );

                results.push((episode, calibrated_confidence));
            }
        }

        // Sort by confidence (highest first)
        results.sort_by(|a, b| {
            b.1.raw()
                .partial_cmp(&a.1.raw())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Perform semantic content search
    fn semantic_content_search(
        &self,
        content: &str,
        threshold: Confidence,
    ) -> Vec<(Episode, Confidence)> {
        let mut results = Vec::new();
        let content_lower = content.to_lowercase();

        for entry in &self.data {
            let memory = entry.value();

            // Check if memory has content and matches
            if let Some(memory_content) = &memory.content {
                let memory_lower = memory_content.to_lowercase();

                // Simple substring matching (could be enhanced with fuzzy matching)
                let relevance = if memory_lower.contains(&content_lower) {
                    1.0
                } else if content_lower.contains(&memory_lower) {
                    0.8
                } else {
                    // Check for word overlap
                    let query_words: std::collections::HashSet<&str> =
                        content_lower.split_whitespace().collect();
                    let memory_words: std::collections::HashSet<&str> =
                        memory_lower.split_whitespace().collect();

                    let intersection_size = query_words.intersection(&memory_words).count();
                    let union_size = query_words.union(&memory_words).count();

                    if union_size > 0 {
                        Self::ratio_usize(intersection_size, union_size)
                    } else {
                        0.0
                    }
                };

                if relevance >= threshold.raw() {
                    let episode = Self::memory_to_episode(memory);
                    let confidence = Confidence::exact(relevance);
                    results.push((episode, confidence));
                }
            }
        }

        results.sort_by(|a, b| {
            b.1.raw()
                .partial_cmp(&a.1.raw())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Convert Memory to Episode for results
    fn memory_to_episode(memory: &Memory) -> Episode {
        EpisodeBuilder::new()
            .id(memory.id.clone())
            .when(memory.created_at)
            .what(
                memory
                    .content
                    .clone()
                    .unwrap_or_else(|| format!("Memory: {id}", id = memory.id)),
            )
            .embedding(memory.embedding)
            .confidence(memory.confidence)
            .build()
    }

    /// Get memory by ID with access tracking
    #[must_use = "Return value indicates whether the memory was found"]
    pub fn get_memory(&self, memory_id: &str) -> Option<Arc<Memory>> {
        self.data.get(memory_id).map_or_else(
            || {
                self.cache_misses.fetch_add(1, Ordering::Relaxed);
                None
            },
            |memory| {
                self.record_access(memory_id);
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                Some(memory.clone())
            },
        )
    }

    /// Remove memory from hot tier
    #[must_use = "Callers rely on the removed memory"]
    pub fn evict_memory(&self, memory_id: &str) -> Option<Arc<Memory>> {
        self.access_times.remove(memory_id);
        self.data.remove(memory_id).map(|(_, memory)| memory)
    }

    /// Get current memory count
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tier is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl HotTier {
    fn ratio_usize(numerator: usize, denominator: usize) -> f32 {
        let numerator_u64 = u64::try_from(numerator).unwrap_or(u64::MAX);
        let denominator_u64 = u64::try_from(denominator).unwrap_or(u64::MAX);
        unit_ratio_to_f32(numerator_u64, denominator_u64)
    }

    fn ratio_u64(numerator: u64, denominator: u64) -> f32 {
        unit_ratio_to_f32(numerator, denominator)
    }

    fn f32_from_usize(value: usize) -> f32 {
        let value_u64 = u64::try_from(value).unwrap_or(u64::MAX);
        Self::f32_from_u64(value_u64)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    const fn f32_from_u64(value: u64) -> f32 {
        if value == u64::MAX {
            f32::MAX
        } else {
            value as f32
        }
    }
}

impl StorageTierBackend for HotTier {
    type Error = StorageError;

    async fn store(&self, memory: Arc<Memory>) -> Result<(), Self::Error> {
        let memory_id = memory.id.clone();

        // Store memory and record access
        self.data.insert(memory_id.clone(), memory);
        self.record_access(&memory_id);

        Ok(())
    }

    async fn recall(&self, cue: &Cue) -> Result<Vec<(Episode, Confidence)>, Self::Error> {
        self.total_accesses.fetch_add(1, Ordering::Relaxed);

        let results = match &cue.cue_type {
            CueType::Embedding { vector, threshold } => {
                self.embedding_similarity_search(vector, *threshold)
            }
            CueType::Semantic {
                content,
                fuzzy_threshold,
            } => self.semantic_content_search(content, *fuzzy_threshold),
            CueType::Context {
                confidence_threshold,
                ..
            } => {
                // For context cues, return all memories above threshold
                let mut results = Vec::new();
                for entry in &self.data {
                    let memory = entry.value();
                    if memory.confidence.raw() >= confidence_threshold.raw() {
                        let episode = Self::memory_to_episode(memory);
                        results.push((episode, memory.confidence));
                    }
                }
                results.sort_by(|a, b| {
                    b.1.raw()
                        .partial_cmp(&a.1.raw())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                results
            }
            CueType::Temporal {
                confidence_threshold,
                ..
            } => {
                // For temporal cues, return recent memories above threshold
                let mut results = Vec::new();
                for entry in &self.data {
                    let memory = entry.value();
                    if memory.confidence.raw() >= confidence_threshold.raw() {
                        let episode = Self::memory_to_episode(memory);
                        results.push((episode, memory.confidence));
                    }
                }
                results.sort_by(|a, b| {
                    b.1.raw()
                        .partial_cmp(&a.1.raw())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                results
            }
        };

        // Limit results to max requested
        let limited_results = results.into_iter().take(cue.max_results).collect();

        Ok(limited_results)
    }

    async fn update_activation(&self, memory_id: &str, activation: f32) -> Result<(), Self::Error> {
        self.data.get(memory_id).map_or_else(
            || {
                Err(StorageError::AllocationFailed(format!(
                    "Memory {memory_id} not found in hot tier"
                )))
            },
            |memory_ref| {
                memory_ref.set_activation(activation);
                self.record_access(memory_id);
                Ok(())
            },
        )
    }

    async fn remove(&self, memory_id: &str) -> Result<(), Self::Error> {
        self.data.remove(memory_id);
        self.access_times.remove(memory_id);
        Ok(())
    }

    fn statistics(&self) -> TierStatistics {
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);
        let total_accesses = cache_hits + cache_misses;

        let cache_hit_rate = Self::ratio_u64(cache_hits, total_accesses);

        // Calculate average activation
        let total_activation: f32 = self
            .data
            .iter()
            .map(|entry| entry.value().activation())
            .sum();

        let average_activation = if self.data.is_empty() {
            0.0
        } else {
            let denominator = Self::f32_from_usize(self.data.len()).max(f32::EPSILON);
            total_activation / denominator
        };

        let memory_count = self.data.len();
        let memory_count_u64 = u64::try_from(memory_count).unwrap_or(u64::MAX);
        let memory_size = u64::try_from(std::mem::size_of::<Memory>()).unwrap_or(0);
        let total_size_bytes = memory_count_u64.saturating_mul(memory_size);

        TierStatistics {
            memory_count,
            total_size_bytes,
            average_activation,
            last_access_time: SystemTime::now(),
            cache_hit_rate,
            compaction_ratio: 1.0, // No compaction in hot tier
        }
    }

    async fn maintenance(&self) -> Result<(), Self::Error> {
        // Hot tier maintenance is minimal - just update statistics
        // Actual eviction is handled by the tier coordinator
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CueBuilder, EpisodeBuilder};
    use chrono::Utc;
    use std::fmt::Debug;

    type TestResult<T = ()> = Result<T, String>;

    fn ensure(condition: bool, message: impl Into<String>) -> TestResult {
        if condition {
            Ok(())
        } else {
            Err(message.into())
        }
    }

    fn ensure_eq<T>(actual: &T, expected: &T, context: &str) -> TestResult
    where
        T: PartialEq + Debug,
    {
        if actual == expected {
            Ok(())
        } else {
            Err(format!("{context}: expected {expected:?}, got {actual:?}"))
        }
    }

    trait IntoTestResult<T> {
        fn into_test_result(self, context: &str) -> TestResult<T>;
    }

    impl<T, E: std::fmt::Debug> IntoTestResult<T> for Result<T, E> {
        fn into_test_result(self, context: &str) -> TestResult<T> {
            self.map_err(|err| format!("{context}: {err:?}"))
        }
    }

    fn create_test_memory(id: &str, activation: f32, content: Option<String>) -> Arc<Memory> {
        let content_str = content
            .clone()
            .unwrap_or_else(|| format!("test memory {id}"));
        let episode = EpisodeBuilder::new()
            .id(id.to_string())
            .when(Utc::now())
            .what(content_str)
            .embedding([0.5f32; 768])
            .confidence(Confidence::HIGH)
            .build();

        let mut memory = Memory::from_episode(episode, activation);
        if let Some(content) = content {
            memory.content = Some(content);
        }

        Arc::new(memory)
    }

    #[tokio::test]
    async fn test_hot_tier_store_and_recall() -> TestResult {
        let hot_tier = HotTier::new(100);

        // Store test memory
        let memory = create_test_memory("test1", 0.9, Some("test content".to_string()));
        hot_tier
            .store(memory)
            .await
            .into_test_result("store memory in hot tier")?;

        // Test embedding recall
        let cue = CueBuilder::new()
            .id("test_cue".to_string())
            .embedding_search([0.5f32; 768], Confidence::MEDIUM)
            .max_results(10)
            .build();

        let results = hot_tier
            .recall(&cue)
            .await
            .into_test_result("recall from hot tier should succeed")?;
        ensure(!results.is_empty(), "recall should yield results")?;
        ensure_eq(&results[0].0.id, &"test1".to_string(), "recalled id")?;

        Ok(())
    }

    #[tokio::test]
    async fn test_hot_tier_semantic_search() -> TestResult {
        let hot_tier = HotTier::new(100);

        // Store memories with different content
        let memory1 = create_test_memory("m1", 0.9, Some("cognitive memory system".to_string()));
        let memory2 =
            create_test_memory("m2", 0.8, Some("neural network architecture".to_string()));

        hot_tier
            .store(memory1)
            .await
            .into_test_result("store first semantic memory")?;
        hot_tier
            .store(memory2)
            .await
            .into_test_result("store second semantic memory")?;

        // Search for cognitive content
        let cue = CueBuilder::new()
            .id("semantic_cue".to_string())
            .semantic_search("cognitive".to_string(), Confidence::LOW)
            .max_results(10)
            .build();

        let results = hot_tier
            .recall(&cue)
            .await
            .into_test_result("recall from hot tier should succeed")?;
        ensure(!results.is_empty(), "semantic recall results")?;
        ensure_eq(&results[0].0.id, &"m1".to_string(), "semantic match id")?;

        Ok(())
    }

    #[tokio::test]
    async fn test_hot_tier_capacity_management() -> TestResult {
        let hot_tier = HotTier::new(10);

        // Fill beyond capacity
        for i in 0..15 {
            let memory = create_test_memory(&format!("mem_{i}"), 0.5, None);
            hot_tier
                .store(memory)
                .await
                .into_test_result("store memory during capacity test")?;
        }

        ensure(
            hot_tier.is_near_capacity(),
            "tier should report near capacity",
        )?;

        // Test LRU candidate selection
        let candidates = hot_tier.get_lru_candidates(5);
        ensure_eq(&candidates.len(), &5_usize, "candidate count")?;

        Ok(())
    }

    #[test]
    fn test_hot_tier_statistics() {
        let hot_tier = HotTier::new(100);
        let stats = hot_tier.statistics();

        assert_eq!(stats.memory_count, 0);
        assert!(stats.cache_hit_rate.abs() < f32::EPSILON);
        assert!((stats.compaction_ratio - 1.0).abs() < f32::EPSILON);
    }
}
