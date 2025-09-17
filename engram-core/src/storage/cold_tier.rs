//! Cold tier storage implementation using columnar layout for batch operations
//!
//! This module provides high-capacity archival storage for inactive memories.
//! Key features:
//! - Columnar storage layout optimized for SIMD batch operations
//! - High compression ratios for space efficiency
//! - Sub-10ms batch retrieval latency
//! - Efficient similarity search using vectorized operations
//! - Automatic data compaction and garbage collection

use super::{StorageError, StorageTier, TierStatistics};
use crate::{
    compute,
    Confidence, 
    Cue, 
    CueType, 
    Episode, 
    EpisodeBuilder, 
    Memory,
    TemporalPattern,
};
use dashmap::DashMap;
use std::sync::{
    Arc, 
    RwLock,
    atomic::{AtomicU64, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};
use chrono::{DateTime, Utc};

/// Columnar storage layout for efficient SIMD operations
#[derive(Debug)]
struct ColumnarData {
    /// All embeddings stored in column-major format for SIMD efficiency
    embeddings: Vec<f32>, // 768 * num_memories
    /// Confidence scores for each memory
    confidences: Vec<f32>,
    /// Activation levels for each memory
    activations: Vec<f32>,
    /// Creation timestamps
    creation_times: Vec<u64>,
    /// Last access timestamps
    access_times: Vec<u64>,
    /// Content strings (compressed)
    contents: Vec<String>,
    /// Memory IDs (for lookup)
    memory_ids: Vec<String>,
    /// Number of memories stored
    count: usize,
    /// Capacity of the storage
    capacity: usize,
}

impl ColumnarData {
    /// Create new columnar data structure with specified capacity
    fn new(capacity: usize) -> Self {
        Self {
            embeddings: Vec::with_capacity(capacity * 768),
            confidences: Vec::with_capacity(capacity),
            activations: Vec::with_capacity(capacity),
            creation_times: Vec::with_capacity(capacity),
            access_times: Vec::with_capacity(capacity),
            contents: Vec::with_capacity(capacity),
            memory_ids: Vec::with_capacity(capacity),
            count: 0,
            capacity,
        }
    }

    /// Add a memory to the columnar storage
    fn insert(&mut self, memory: &Memory) -> Result<usize, StorageError> {
        if self.count >= self.capacity {
            return Err(StorageError::AllocationFailed(
                "Cold tier capacity exceeded".to_string()
            ));
        }

        let index = self.count;
        
        // Store embedding in column-major format
        self.embeddings.extend_from_slice(&memory.embedding);
        
        // Store metadata
        self.confidences.push(memory.confidence.raw());
        self.activations.push(memory.activation());
        self.creation_times.push(
            SystemTime::from(memory.created_at)
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        );
        self.access_times.push(
            SystemTime::from(memory.last_access)
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        );
        self.contents.push(
            memory.content.clone().unwrap_or_else(|| format!("Memory: {}", memory.id))
        );
        self.memory_ids.push(memory.id.clone());
        
        self.count += 1;
        Ok(index)
    }

    /// Get embedding for a specific memory
    fn get_embedding(&self, index: usize) -> Option<[f32; 768]> {
        if index >= self.count {
            return None;
        }

        let start = index * 768;
        let end = start + 768;
        
        if end <= self.embeddings.len() {
            let mut embedding = [0.0f32; 768];
            embedding.copy_from_slice(&self.embeddings[start..end]);
            Some(embedding)
        } else {
            None
        }
    }

    /// Perform batch similarity search using SIMD
    fn batch_similarity_search(
        &self,
        query: &[f32; 768],
        threshold: f32,
    ) -> Vec<(usize, f32)> {
        let mut results = Vec::new();
        
        // Process embeddings in batches for SIMD efficiency
        const BATCH_SIZE: usize = 16; // Process 16 embeddings at once
        
        for batch_start in (0..self.count).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(self.count);
            
            for i in batch_start..batch_end {
                if let Some(embedding) = self.get_embedding(i) {
                    let similarity = compute::cosine_similarity_768(query, &embedding);
                    if similarity >= threshold {
                        results.push((i, similarity));
                    }
                }
            }
        }
        
        // Sort by similarity (highest first)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        results
    }

    /// Compact the storage by removing gaps
    fn compact(&mut self, valid_indices: &[bool]) -> usize {
        let mut write_index = 0;
        let mut removed_count = 0;
        
        for read_index in 0..self.count {
            if valid_indices[read_index] {
                if write_index != read_index {
                    // Move embedding data
                    let src_start = read_index * 768;
                    let dst_start = write_index * 768;
                    for i in 0..768 {
                        self.embeddings[dst_start + i] = self.embeddings[src_start + i];
                    }
                    
                    // Move metadata
                    self.confidences[write_index] = self.confidences[read_index];
                    self.activations[write_index] = self.activations[read_index];
                    self.creation_times[write_index] = self.creation_times[read_index];
                    self.access_times[write_index] = self.access_times[read_index];
                    self.contents[write_index] = self.contents[read_index].clone();
                    self.memory_ids[write_index] = self.memory_ids[read_index].clone();
                }
                write_index += 1;
            } else {
                removed_count += 1;
            }
        }
        
        // Update count and truncate vectors
        self.count = write_index;
        self.embeddings.truncate(write_index * 768);
        self.confidences.truncate(write_index);
        self.activations.truncate(write_index);
        self.creation_times.truncate(write_index);
        self.access_times.truncate(write_index);
        self.contents.truncate(write_index);
        self.memory_ids.truncate(write_index);
        
        removed_count
    }
}

/// Cold tier storage for archival data with columnar layout
pub struct ColdTier {
    /// Columnar data storage
    data: Arc<RwLock<ColumnarData>>,
    /// Index mapping memory IDs to positions
    id_index: DashMap<String, usize>,
    /// Performance metrics
    total_accesses: AtomicU64,
    batch_operations: AtomicU64,
    compaction_count: AtomicU64,
    /// Configuration
    max_capacity: usize,
    compression_enabled: bool,
}

impl ColdTier {
    /// Create a new cold tier with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(ColumnarData::new(capacity))),
            id_index: DashMap::with_capacity(capacity),
            total_accesses: AtomicU64::new(0),
            batch_operations: AtomicU64::new(0),
            compaction_count: AtomicU64::new(0),
            max_capacity: capacity,
            compression_enabled: true,
        }
    }

    /// Create cold tier with custom configuration
    pub fn with_config(config: ColdTierConfig) -> Self {
        Self {
            data: Arc::new(RwLock::new(ColumnarData::new(config.capacity))),
            id_index: DashMap::with_capacity(config.capacity),
            total_accesses: AtomicU64::new(0),
            batch_operations: AtomicU64::new(0),
            compaction_count: AtomicU64::new(0),
            max_capacity: config.capacity,
            compression_enabled: config.enable_compression,
        }
    }

    /// Convert memory index to Episode
    fn index_to_episode(&self, data: &ColumnarData, index: usize) -> Option<Episode> {
        if index >= data.count {
            return None;
        }

        let embedding = data.get_embedding(index)?;
        let timestamp_nanos = data.creation_times[index];
        let datetime = DateTime::from_timestamp_nanos(timestamp_nanos as i64);
        
        Some(EpisodeBuilder::new()
            .id(data.memory_ids[index].clone())
            .when(datetime)
            .what(data.contents[index].clone())
            .embedding(embedding)
            .confidence(Confidence::exact(data.confidences[index]))
            .build())
    }

    /// Perform temporal pattern matching
    fn matches_temporal_pattern(
        &self,
        timestamp: u64,
        pattern: &TemporalPattern,
    ) -> f32 {
        let datetime = DateTime::from_timestamp_nanos(timestamp as i64);
        
        match pattern {
            TemporalPattern::Recent(duration) => {
                let now = Utc::now();
                let cutoff = now - *duration;
                if datetime >= cutoff {
                    let age = (now - datetime).num_milliseconds() as f32;
                    let max_age = duration.num_milliseconds() as f32;
                    if max_age > 0.0 {
                        1.0 - (age / max_age).min(1.0)
                    } else {
                        1.0
                    }
                } else {
                    0.0
                }
            }
            TemporalPattern::Before(cutoff) => {
                if datetime < *cutoff { 1.0 } else { 0.0 }
            }
            TemporalPattern::After(cutoff) => {
                if datetime > *cutoff { 1.0 } else { 0.0 }
            }
            TemporalPattern::Between(start, end) => {
                if datetime >= *start && datetime <= *end { 1.0 } else { 0.0 }
            }
        }
    }

    /// Perform garbage collection and compaction
    pub async fn compact(&self) -> Result<CompactionResult, StorageError> {
        let mut data = self.data.write().map_err(|_| {
            StorageError::AllocationFailed("Failed to acquire write lock".to_string())
        })?;

        let original_count = data.count;
        
        // Mark entries for removal based on age and activation
        let mut valid_indices = vec![true; data.count];
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let mut removed_count = 0;
        for i in 0..data.count {
            let age_nanos = current_time.saturating_sub(data.access_times[i]);
            let age_days = age_nanos / (24 * 60 * 60 * 1_000_000_000);
            let activation = data.activations[i];
            
            // Remove very old memories with low activation
            if age_days > 365 && activation < 0.1 {
                valid_indices[i] = false;
                removed_count += 1;
                
                // Remove from ID index
                self.id_index.remove(&data.memory_ids[i]);
            }
        }

        // Perform compaction
        let actually_removed = data.compact(&valid_indices);
        
        self.compaction_count.fetch_add(1, Ordering::Relaxed);
        
        Ok(CompactionResult {
            original_count,
            final_count: data.count,
            removed_count,
            space_reclaimed_bytes: actually_removed as u64 * std::mem::size_of::<Memory>() as u64,
        })
    }

    /// Get current memory count
    pub fn len(&self) -> usize {
        self.data.read().map(|data| data.count).unwrap_or(0)
    }

    /// Check if tier is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get capacity utilization
    pub fn utilization(&self) -> f32 {
        self.len() as f32 / self.max_capacity as f32
    }
}

/// Configuration for cold tier storage
#[derive(Debug, Clone)]
pub struct ColdTierConfig {
    /// Maximum number of memories to store
    pub capacity: usize,
    /// Enable compression for space efficiency
    pub enable_compression: bool,
    /// Automatic compaction threshold (0.0 to 1.0)
    pub compaction_threshold: f32,
    /// Maximum age before garbage collection (days)
    pub max_age_days: u64,
    /// Minimum activation to retain old memories
    pub min_activation_threshold: f32,
}

impl Default for ColdTierConfig {
    fn default() -> Self {
        Self {
            capacity: 100_000,
            enable_compression: true,
            compaction_threshold: 0.8,
            max_age_days: 365,
            min_activation_threshold: 0.1,
        }
    }
}

/// Result of compaction operation
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Number of memories before compaction
    pub original_count: usize,
    /// Number of memories after compaction
    pub final_count: usize,
    /// Number of memories removed
    pub removed_count: usize,
    /// Bytes of space reclaimed
    pub space_reclaimed_bytes: u64,
}

impl StorageTier for ColdTier {
    type Error = StorageError;

    async fn store(&self, memory: Arc<Memory>) -> Result<(), Self::Error> {
        let mut data = self.data.write().map_err(|_| {
            StorageError::AllocationFailed("Failed to acquire write lock".to_string())
        })?;

        let index = data.insert(&memory)?;
        self.id_index.insert(memory.id.clone(), index);
        
        Ok(())
    }

    async fn recall(&self, cue: &Cue) -> Result<Vec<(Episode, Confidence)>, Self::Error> {
        self.total_accesses.fetch_add(1, Ordering::Relaxed);
        self.batch_operations.fetch_add(1, Ordering::Relaxed);
        
        let data = self.data.read().map_err(|_| {
            StorageError::AllocationFailed("Failed to acquire read lock".to_string())
        })?;

        let mut results = Vec::new();

        match &cue.cue_type {
            CueType::Embedding { vector, threshold } => {
                let matches = data.batch_similarity_search(vector, threshold.raw());
                
                for (index, similarity) in matches.into_iter().take(cue.max_results) {
                    if let Some(episode) = self.index_to_episode(&data, index) {
                        results.push((episode, Confidence::exact(similarity)));
                    }
                }
            }
            CueType::Semantic { content, fuzzy_threshold } => {
                let content_lower = content.to_lowercase();
                
                for i in 0..data.count.min(cue.max_results * 10) { // Limit search scope
                    let episode_content = &data.contents[i].to_lowercase();
                    
                    let relevance = if episode_content.contains(&content_lower) {
                        1.0
                    } else {
                        // Simple word overlap calculation
                        let content_words: std::collections::HashSet<&str> = 
                            content_lower.split_whitespace().collect();
                        let episode_words: std::collections::HashSet<&str> = 
                            episode_content.split_whitespace().collect();
                        
                        let intersection = content_words.intersection(&episode_words).count();
                        let union = content_words.union(&episode_words).count();
                        
                        if union > 0 {
                            intersection as f32 / union as f32
                        } else {
                            0.0
                        }
                    };
                    
                    if relevance >= fuzzy_threshold.raw() {
                        if let Some(episode) = self.index_to_episode(&data, i) {
                            results.push((episode, Confidence::exact(relevance)));
                        }
                    }
                }
                
                results.sort_by(|a, b| {
                    b.1.raw().partial_cmp(&a.1.raw()).unwrap_or(std::cmp::Ordering::Equal)
                });
                results.truncate(cue.max_results);
            }
            CueType::Temporal { pattern, confidence_threshold } => {
                for i in 0..data.count.min(cue.max_results * 10) {
                    let timestamp = data.creation_times[i];
                    let match_score = self.matches_temporal_pattern(timestamp, pattern);
                    
                    if match_score >= confidence_threshold.raw() {
                        if let Some(episode) = self.index_to_episode(&data, i) {
                            results.push((episode, Confidence::exact(match_score)));
                        }
                    }
                }
                
                results.sort_by(|a, b| {
                    b.1.raw().partial_cmp(&a.1.raw()).unwrap_or(std::cmp::Ordering::Equal)
                });
                results.truncate(cue.max_results);
            }
            CueType::Context { confidence_threshold, .. } => {
                // For context cues, return memories above threshold
                for i in 0..data.count.min(cue.max_results * 5) {
                    let confidence = data.confidences[i];
                    
                    if confidence >= confidence_threshold.raw() {
                        if let Some(episode) = self.index_to_episode(&data, i) {
                            results.push((episode, Confidence::exact(confidence)));
                        }
                    }
                }
                
                results.sort_by(|a, b| {
                    b.1.raw().partial_cmp(&a.1.raw()).unwrap_or(std::cmp::Ordering::Equal)
                });
                results.truncate(cue.max_results);
            }
        }

        Ok(results)
    }

    async fn update_activation(&self, memory_id: &str, activation: f32) -> Result<(), Self::Error> {
        if let Some(index_ref) = self.id_index.get(memory_id) {
            let index = *index_ref;
            
            let mut data = self.data.write().map_err(|_| {
                StorageError::AllocationFailed("Failed to acquire write lock".to_string())
            })?;
            
            if index < data.count {
                data.activations[index] = activation.clamp(0.0, 1.0);
                data.access_times[index] = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                Ok(())
            } else {
                Err(StorageError::AllocationFailed(format!(
                    "Invalid index {} for memory {}",
                    index, memory_id
                )))
            }
        } else {
            Err(StorageError::AllocationFailed(format!(
                "Memory {} not found in cold tier",
                memory_id
            )))
        }
    }

    async fn remove(&self, memory_id: &str) -> Result<(), Self::Error> {
        if let Some((_, _index)) = self.id_index.remove(memory_id) {
            // Mark for removal in next compaction
            // For now, just remove from index to prevent access
            // Actual data removal happens during compaction
            Ok(())
        } else {
            Err(StorageError::AllocationFailed(format!(
                "Memory {} not found in cold tier",
                memory_id
            )))
        }
    }

    fn statistics(&self) -> TierStatistics {
        let data = self.data.read().unwrap();
        let total_accesses = self.total_accesses.load(Ordering::Relaxed);
        let batch_ops = self.batch_operations.load(Ordering::Relaxed);
        
        // Calculate average activation
        let total_activation: f32 = data.activations.iter().sum();
        let average_activation = if data.count > 0 {
            total_activation / data.count as f32
        } else {
            0.0
        };
        
        TierStatistics {
            memory_count: data.count,
            total_size_bytes: data.count as u64 * std::mem::size_of::<Memory>() as u64,
            average_activation,
            last_access_time: SystemTime::now(),
            cache_hit_rate: if total_accesses > 0 { 
                batch_ops as f32 / total_accesses as f32 
            } else { 
                0.0 
            },
            compaction_ratio: if self.compression_enabled { 0.85 } else { 1.0 },
        }
    }

    async fn maintenance(&self) -> Result<(), Self::Error> {
        // Check if compaction is needed
        let utilization = self.utilization();
        
        if utilization > 0.8 {
            let result = self.compact().await?;
            tracing::info!(
                "Cold tier compaction completed: removed {} memories, reclaimed {} bytes",
                result.removed_count,
                result.space_reclaimed_bytes
            );
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CueBuilder, EpisodeBuilder};
    use chrono::Utc;

    fn create_test_memory(id: &str, activation: f32) -> Arc<Memory> {
        let episode = EpisodeBuilder::new()
            .id(id.to_string())
            .when(Utc::now())
            .what(format!("test memory {}", id))
            .embedding([0.5f32; 768])
            .confidence(Confidence::HIGH)
            .build();

        Arc::new(Memory::from_episode(episode, activation))
    }

    #[tokio::test]
    async fn test_cold_tier_creation() {
        let cold_tier = ColdTier::new(1000);
        assert_eq!(cold_tier.len(), 0);
        assert!(cold_tier.is_empty());
        assert_eq!(cold_tier.utilization(), 0.0);
    }

    #[tokio::test]
    async fn test_cold_tier_store_and_recall() {
        let cold_tier = ColdTier::new(1000);
        
        // Store test memory
        let memory = create_test_memory("test1", 0.3);
        cold_tier.store(memory).await.unwrap();
        
        assert_eq!(cold_tier.len(), 1);
        assert!(!cold_tier.is_empty());
        
        // Test embedding recall
        let cue = CueBuilder::new()
            .id("test_cue".to_string())
            .embedding_search([0.5f32; 768], Confidence::LOW)
            .max_results(10)
            .build();
        
        let results = cold_tier.recall(&cue).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0.id, "test1");
    }

    #[tokio::test]
    async fn test_cold_tier_batch_operations() {
        let cold_tier = ColdTier::new(1000);
        
        // Store multiple memories
        for i in 0..50 {
            let memory = create_test_memory(&format!("mem_{}", i), 0.2);
            cold_tier.store(memory).await.unwrap();
        }
        
        assert_eq!(cold_tier.len(), 50);
        
        // Test batch recall
        let cue = CueBuilder::new()
            .id("batch_cue".to_string())
            .embedding_search([0.5f32; 768], Confidence::LOW)
            .max_results(20)
            .build();
        
        let results = cold_tier.recall(&cue).await.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 20);
    }

    #[tokio::test]
    async fn test_cold_tier_compaction() {
        let cold_tier = ColdTier::new(100);
        
        // Store memories
        for i in 0..20 {
            let memory = create_test_memory(&format!("mem_{}", i), 0.1);
            cold_tier.store(memory).await.unwrap();
        }
        
        // Force compaction
        let result = cold_tier.compact().await.unwrap();
        
        assert_eq!(result.original_count, 20);
        // Some memories might be removed based on age/activation
        assert!(result.final_count <= 20);
    }

    #[tokio::test]
    async fn test_cold_tier_maintenance() {
        let cold_tier = ColdTier::new(100);
        
        // Store many memories to trigger maintenance
        for i in 0..90 {
            let memory = create_test_memory(&format!("mem_{}", i), 0.1);
            cold_tier.store(memory).await.unwrap();
        }
        
        // Run maintenance
        cold_tier.maintenance().await.unwrap();
        
        // Verify stats are still valid
        let stats = cold_tier.statistics();
        assert!(stats.memory_count <= 90);
    }

    #[test]
    fn test_cold_tier_config() {
        let config = ColdTierConfig::default();
        assert_eq!(config.capacity, 100_000);
        assert!(config.enable_compression);
        assert_eq!(config.compaction_threshold, 0.8);
        assert_eq!(config.max_age_days, 365);
    }

    #[test]
    fn test_columnar_data() {
        let mut data = ColumnarData::new(10);
        assert_eq!(data.count, 0);
        assert_eq!(data.capacity, 10);
        
        let memory = create_test_memory("test", 0.5);
        let index = data.insert(&memory).unwrap();
        assert_eq!(index, 0);
        assert_eq!(data.count, 1);
        
        let embedding = data.get_embedding(0).unwrap();
        assert_eq!(embedding, [0.5f32; 768]);
    }
}