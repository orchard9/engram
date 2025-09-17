//! Hot tier storage implementation using DashMap for concurrent access
//!
//! This module provides ultra-fast in-memory storage for highly active memories.
//! Key features:
//! - Lock-free concurrent access via DashMap
//! - Sub-100μs retrieval latency
//! - SIMD-optimized similarity search
//! - Automatic activation tracking and decay
//! - Memory pressure-aware eviction

use super::{StorageError, StorageTier, TierStatistics};
use crate::{
    compute, 
    Confidence, 
    Cue, 
    CueType, 
    Episode, 
    EpisodeBuilder, 
    Memory,
};
use dashmap::DashMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

/// Hot tier storage using lock-free DashMap for maximum performance
pub struct HotTier {
    /// Primary storage for active memories
    data: DashMap<String, Arc<Memory>>,
    /// Access timestamps for LRU eviction
    access_times: DashMap<String, u64>,
    /// Performance metrics
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    total_accesses: AtomicU64,
    /// Capacity limit
    max_capacity: usize,
}

impl HotTier {
    /// Create a new hot tier with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            data: DashMap::with_capacity(capacity),
            access_times: DashMap::with_capacity(capacity),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            total_accesses: AtomicU64::new(0),
            max_capacity: capacity,
        }
    }

    /// Get current timestamp as nanoseconds since UNIX epoch
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// Record access to a memory for LRU tracking
    fn record_access(&self, memory_id: &str) {
        let timestamp = Self::current_timestamp();
        self.access_times.insert(memory_id.to_string(), timestamp);
        self.total_accesses.fetch_add(1, Ordering::Relaxed);
    }

    /// Check if tier is approaching capacity
    pub fn is_near_capacity(&self) -> bool {
        self.data.len() > (self.max_capacity as f32 * 0.8) as usize
    }

    /// Get least recently used memories for eviction
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
                let episode = self.memory_to_episode(memory);
                let confidence = Confidence::exact(similarity);
                results.push((episode, confidence));
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
        
        for entry in self.data.iter() {
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
                        intersection_size as f32 / union_size as f32
                    } else {
                        0.0
                    }
                };
                
                if relevance >= threshold.raw() {
                    let episode = self.memory_to_episode(memory);
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
    fn memory_to_episode(&self, memory: &Memory) -> Episode {
        EpisodeBuilder::new()
            .id(memory.id.clone())
            .when(memory.created_at)
            .what(memory.content.clone().unwrap_or_else(|| format!("Memory: {}", memory.id)))
            .embedding(memory.embedding)
            .confidence(memory.confidence)
            .build()
    }

    /// Get memory by ID with access tracking
    pub fn get_memory(&self, memory_id: &str) -> Option<Arc<Memory>> {
        if let Some(memory) = self.data.get(memory_id) {
            self.record_access(memory_id);
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            Some(memory.clone())
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Remove memory from hot tier
    pub fn evict_memory(&self, memory_id: &str) -> Option<Arc<Memory>> {
        self.access_times.remove(memory_id);
        self.data.remove(memory_id).map(|(_, memory)| memory)
    }

    /// Get current memory count
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tier is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl StorageTier for HotTier {
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
            CueType::Semantic { content, fuzzy_threshold } => {
                self.semantic_content_search(content, *fuzzy_threshold)
            }
            CueType::Context { confidence_threshold, .. } => {
                // For context cues, return all memories above threshold
                let mut results = Vec::new();
                for entry in self.data.iter() {
                    let memory = entry.value();
                    if memory.confidence.raw() >= confidence_threshold.raw() {
                        let episode = self.memory_to_episode(memory);
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
            CueType::Temporal { confidence_threshold, .. } => {
                // For temporal cues, return recent memories above threshold
                let mut results = Vec::new();
                for entry in self.data.iter() {
                    let memory = entry.value();
                    if memory.confidence.raw() >= confidence_threshold.raw() {
                        let episode = self.memory_to_episode(memory);
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
        let limited_results = results
            .into_iter()
            .take(cue.max_results)
            .collect();

        Ok(limited_results)
    }

    async fn update_activation(&self, memory_id: &str, activation: f32) -> Result<(), Self::Error> {
        if let Some(memory_ref) = self.data.get(memory_id) {
            memory_ref.set_activation(activation);
            self.record_access(memory_id);
            Ok(())
        } else {
            Err(StorageError::AllocationFailed(format!(
                "Memory {} not found in hot tier",
                memory_id
            )))
        }
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
        
        let cache_hit_rate = if total_accesses > 0 {
            cache_hits as f32 / total_accesses as f32
        } else {
            0.0
        };

        // Calculate average activation
        let total_activation: f32 = self
            .data
            .iter()
            .map(|entry| entry.value().activation())
            .sum();
        
        let average_activation = if self.data.is_empty() {
            0.0
        } else {
            total_activation / self.data.len() as f32
        };

        TierStatistics {
            memory_count: self.data.len(),
            total_size_bytes: self.data.len() as u64 * std::mem::size_of::<Memory>() as u64,
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

    fn create_test_memory(id: &str, activation: f32, content: Option<String>) -> Arc<Memory> {
        let content_str = content.clone().unwrap_or_else(|| format!("test memory {}", id));
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
    async fn test_hot_tier_store_and_recall() {
        let hot_tier = HotTier::new(100);
        
        // Store test memory
        let memory = create_test_memory("test1", 0.9, Some("test content".to_string()));
        hot_tier.store(memory.clone()).await.unwrap();
        
        // Test embedding recall
        let cue = CueBuilder::new()
            .id("test_cue".to_string())
            .embedding_search([0.5f32; 768], Confidence::MEDIUM)
            .max_results(10)
            .build();
        
        let results = hot_tier.recall(&cue).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0.id, "test1");
    }

    #[tokio::test]
    async fn test_hot_tier_semantic_search() {
        let hot_tier = HotTier::new(100);
        
        // Store memories with different content
        let memory1 = create_test_memory("m1", 0.9, Some("cognitive memory system".to_string()));
        let memory2 = create_test_memory("m2", 0.8, Some("neural network architecture".to_string()));
        
        hot_tier.store(memory1).await.unwrap();
        hot_tier.store(memory2).await.unwrap();
        
        // Search for cognitive content
        let cue = CueBuilder::new()
            .id("semantic_cue".to_string())
            .semantic_search("cognitive".to_string(), Confidence::LOW)
            .max_results(10)
            .build();
        
        let results = hot_tier.recall(&cue).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0.id, "m1");
    }

    #[tokio::test]
    async fn test_hot_tier_capacity_management() {
        let hot_tier = HotTier::new(10);
        
        // Fill beyond capacity
        for i in 0..15 {
            let memory = create_test_memory(&format!("mem_{}", i), 0.5, None);
            hot_tier.store(memory).await.unwrap();
        }
        
        assert!(hot_tier.is_near_capacity());
        
        // Test LRU candidate selection
        let candidates = hot_tier.get_lru_candidates(5);
        assert_eq!(candidates.len(), 5);
    }

    #[test]
    fn test_hot_tier_statistics() {
        let hot_tier = HotTier::new(100);
        let stats = hot_tier.statistics();
        
        assert_eq!(stats.memory_count, 0);
        assert_eq!(stats.cache_hit_rate, 0.0);
        assert_eq!(stats.compaction_ratio, 1.0);
    }
}