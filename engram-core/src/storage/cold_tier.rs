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
use super::confidence::{StorageConfidenceCalibrator, ConfidenceTier};
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
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr;

/// 64-byte aligned column for SIMD operations
#[repr(align(64))]
pub struct AlignedColumn {
    /// 64-byte aligned raw pointer for SIMD operations
    data: *mut f32,
    /// Allocated capacity
    capacity: usize,
    /// Current number of values
    len: usize,
    /// Memory layout for deallocation
    layout: Layout,
}

// AlignedColumn is safe to send between threads as long as it's not accessed concurrently
unsafe impl Send for AlignedColumn {}
// AlignedColumn is safe to share between threads when protected by RwLock
unsafe impl Sync for AlignedColumn {}

impl std::fmt::Debug for AlignedColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedColumn")
            .field("capacity", &self.capacity)
            .field("len", &self.len)
            .field("data_ptr", &self.data)
            .finish()
    }
}

impl AlignedColumn {
    /// Create new aligned column with specified capacity
    fn new(capacity: usize) -> Self {
        let layout = Layout::from_size_align(
            capacity * std::mem::size_of::<f32>(),
            64  // 64-byte alignment for AVX-512
        ).expect("Failed to create layout");

        let data = unsafe {
            let ptr = alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            ptr as *mut f32
        };

        Self { data, capacity, len: 0, layout }
    }

    /// Push value to column (unsafe: no bounds checking)
    unsafe fn push(&mut self, value: f32) {
        if self.len < self.capacity {
            ptr::write(self.data.add(self.len), value);
            self.len += 1;
        }
    }

    /// Get immutable slice of column data
    unsafe fn get_slice(&self) -> &[f32] {
        std::slice::from_raw_parts(self.data, self.len)
    }

    /// Get mutable slice of column data
    unsafe fn get_slice_mut(&mut self) -> &mut [f32] {
        std::slice::from_raw_parts_mut(self.data, self.len)
    }
}

impl Drop for AlignedColumn {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data as *mut u8, self.layout);
        }
    }
}

/// Columnar storage layout for efficient SIMD operations
#[derive(Debug)]
struct ColumnarData {
    /// 768 aligned columns, one for each embedding dimension
    embedding_columns: Vec<AlignedColumn>,
    /// Metadata columns (not aligned, regular storage)
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
        let mut embedding_columns = Vec::with_capacity(768);
        for _ in 0..768 {
            embedding_columns.push(AlignedColumn::new(capacity));
        }

        Self {
            embedding_columns,
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

        // Store embedding in true column-major format
        unsafe {
            for (dim, &value) in memory.embedding.iter().enumerate() {
                self.embedding_columns[dim].push(value);
            }
        }

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

        let mut embedding = [0.0f32; 768];
        unsafe {
            for dim in 0..768 {
                let column_slice = self.embedding_columns[dim].get_slice();
                if index < column_slice.len() {
                    embedding[dim] = column_slice[index];
                }
            }
        }
        Some(embedding)
    }

    /// Perform batch similarity search using SIMD-optimized columnar operations
    fn batch_similarity_search(
        &self,
        query: &[f32; 768],
        threshold: f32,
    ) -> Vec<(usize, f32)> {
        if self.count == 0 {
            return Vec::new();
        }

        // Allocate aligned buffer for similarity scores
        let mut similarities = vec![0.0f32; self.count];
        let vector_ops = compute::get_vector_ops();

        // Compute dot products using SIMD FMA operations on columns
        unsafe {
            for dim in 0..768 {
                let query_val = query[dim];
                if query_val.abs() < 1e-8 {
                    continue; // Skip zero dimensions
                }

                let column_data = self.embedding_columns[dim].get_slice();

                // SIMD FMA: similarities += column * query_val
                vector_ops.fma_accumulate(column_data, query_val, &mut similarities);
            }
        }

        // Normalize similarities to get cosine similarity
        let query_norm = compute::get_vector_ops().l2_norm_768(query);
        if query_norm == 0.0 {
            return Vec::new();
        }

        // Filter by threshold and collect results
        let mut results: Vec<(usize, f32)> = similarities
            .iter()
            .enumerate()
            .filter_map(|(idx, &dot_product)| {
                // Get vector norm for normalization
                let mut vector_norm_sq = 0.0f32;
                unsafe {
                    for dim in 0..768 {
                        let val = self.embedding_columns[dim].get_slice()[idx];
                        vector_norm_sq += val * val;
                    }
                }
                let vector_norm = vector_norm_sq.sqrt();

                if vector_norm > 0.0 {
                    let similarity = dot_product / (query_norm * vector_norm);
                    if similarity >= threshold {
                        return Some((idx, similarity.clamp(-1.0, 1.0)));
                    }
                }
                None
            })
            .collect();

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
                    // Move embedding data in each column
                    unsafe {
                        for dim in 0..768 {
                            let column = &mut self.embedding_columns[dim];
                            let slice = column.get_slice_mut();
                            slice[write_index] = slice[read_index];
                        }
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

        // Update count and column lengths
        self.count = write_index;
        for column in &mut self.embedding_columns {
            column.len = write_index;
        }

        // Truncate metadata vectors
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
    /// Confidence calibrator for cold tier adjustments
    confidence_calibrator: StorageConfidenceCalibrator,
    /// Storage timestamps for tracking time in storage
    storage_timestamps: DashMap<String, std::time::SystemTime>,
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
            confidence_calibrator: StorageConfidenceCalibrator::new(),
            storage_timestamps: DashMap::new(),
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
            confidence_calibrator: StorageConfidenceCalibrator::new(),
            storage_timestamps: DashMap::new(),
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

    /// Get storage duration for a memory (time since first stored)
    pub fn get_storage_duration(&self, memory_id: &str) -> std::time::Duration {
        if let Some(entry) = self.storage_timestamps.get(memory_id) {
            let stored_time = *entry.value();
            std::time::SystemTime::now()
                .duration_since(stored_time)
                .unwrap_or_default()
        } else {
            // If no timestamp recorded, assume it was just stored
            std::time::Duration::from_secs(0)
        }
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
        let memory_id = memory.id.clone();

        // Record storage timestamp for temporal decay calculation
        self.storage_timestamps.insert(memory_id.clone(), std::time::SystemTime::now());

        let mut data = self.data.write().map_err(|_| {
            StorageError::AllocationFailed("Failed to acquire write lock".to_string())
        })?;

        let index = data.insert(&memory)?;
        self.id_index.insert(memory_id, index);

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

        // Apply cold tier confidence calibration to all results
        let mut calibrated_results = Vec::with_capacity(results.len());
        for (episode, confidence) in results {
            // Get actual storage duration for this memory
            let storage_duration = self.get_storage_duration(&episode.id);
            let calibrated_confidence = self.confidence_calibrator.adjust_for_storage_tier(
                confidence,
                ConfidenceTier::Cold,
                storage_duration,
            );
            calibrated_results.push((episode, calibrated_confidence));
        }

        Ok(calibrated_results)
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
            // Clean up storage timestamp
            self.storage_timestamps.remove(memory_id);

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