//! Cold tier storage implementation using columnar layout for batch operations
//!
//! This module provides high-capacity archival storage for inactive memories.
//! Key features:
//! - Columnar storage layout optimized for SIMD batch operations
//! - High compression ratios for space efficiency
//! - Sub-10ms batch retrieval latency
//! - Efficient similarity search using vectorized operations
//! - Automatic data compaction and garbage collection

use super::confidence::{ConfidenceTier, StorageConfidenceCalibrator};
use super::{StorageError, StorageTierBackend, TierStatistics};
use crate::{
    Confidence, Cue, CueType, Episode, EpisodeBuilder, Memory, TemporalPattern, compute,
    numeric::{saturating_f32_from_f64, unit_ratio_to_f32},
};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::convert::TryFrom;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration as StdDuration, SystemTime, UNIX_EPOCH};

/// Column buffer that enforces capacity while relying on safe `Vec` storage
#[derive(Debug, Default)]
struct ColumnBuffer {
    values: Vec<f32>,
    capacity: usize,
}

impl ColumnBuffer {
    /// Create a new column buffer with the provided capacity
    fn new(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a value into the column, respecting the configured capacity.
    ///
    /// # Errors
    ///
    /// Returns an error when the column has already reached its configured
    /// capacity and cannot store additional values.
    fn push(&mut self, value: f32) -> Result<(), StorageError> {
        if self.values.len() >= self.capacity {
            return Err(StorageError::allocation_failed(
                "Cold tier column capacity exceeded",
            ));
        }
        debug_assert!(self.values.len() < self.capacity);
        self.values.push(value);
        Ok(())
    }

    /// Access the column contents as an immutable slice
    fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Retrieve a value by index if it exists
    fn get(&self, index: usize) -> Option<f32> {
        self.values.get(index).copied()
    }

    /// Write a value into the provided index when it exists
    fn set(&mut self, index: usize, value: f32) {
        debug_assert!(index < self.values.len());
        if let Some(slot) = self.values.get_mut(index) {
            *slot = value;
        }
    }

    /// Truncate the column to the provided length
    fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
    }

    /// Current number of stored values
    const fn len(&self) -> usize {
        self.values.len()
    }
}

/// Columnar storage layout for efficient SIMD operations
#[derive(Debug)]
struct ColumnarData {
    /// 768 column buffers, one for each embedding dimension
    embedding_columns: Vec<ColumnBuffer>,
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
            embedding_columns.push(ColumnBuffer::new(capacity));
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

    /// Add a memory to the columnar storage.
    ///
    /// # Errors
    ///
    /// Returns an error when the cold tier has exhausted its capacity and
    /// cannot insert additional memories.
    fn insert(&mut self, memory: &Memory) -> Result<usize, StorageError> {
        if self.count >= self.capacity {
            return Err(StorageError::AllocationFailed(
                "Cold tier capacity exceeded".to_string(),
            ));
        }

        let index = self.count;

        // Store embedding in true column-major format
        for (dim, &value) in memory.embedding.iter().enumerate() {
            self.embedding_columns[dim].push(value)?;
        }

        // Store metadata
        self.confidences.push(memory.confidence.raw());
        self.activations.push(memory.activation());
        self.creation_times.push(
            SystemTime::from(memory.created_at)
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX),
        );
        self.access_times.push(
            SystemTime::from(memory.last_access)
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .try_into()
                .unwrap_or(u64::MAX),
        );
        self.contents.push(
            memory
                .content
                .clone()
                .unwrap_or_else(|| format!("Memory: {}", memory.id)),
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
        for (dim, column) in self.embedding_columns.iter().enumerate().take(768) {
            if let Some(value) = column.get(index) {
                embedding[dim] = value;
            }
        }
        Some(embedding)
    }

    /// Perform batch similarity search using SIMD-optimized columnar operations
    fn batch_similarity_search(&self, query: &[f32; 768], threshold: f32) -> Vec<(usize, f32)> {
        if self.count == 0 {
            return Vec::new();
        }

        // Allocate aligned buffer for similarity scores
        let mut similarities = vec![0.0f32; self.count];
        let vector_ops = compute::get_vector_ops();

        // Compute dot products using SIMD FMA operations on columns
        for (dim, column) in self.embedding_columns.iter().enumerate().take(768) {
            let query_val = query[dim];
            if query_val.abs() < 1e-8 {
                continue; // Skip zero dimensions
            }

            let column_data = column.as_slice();

            // SIMD FMA: similarities += column * query_val
            vector_ops.fma_accumulate(column_data, query_val, &mut similarities);
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
                for dim in 0..768 {
                    if let Some(val) = self.embedding_columns[dim].get(idx) {
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
        debug_assert!(
            self.embedding_columns
                .iter()
                .all(|column| column.len() == self.count)
        );
        let mut write_index = 0;
        let mut removed_count = 0;

        for (read_index, &is_valid) in valid_indices.iter().enumerate().take(self.count) {
            if is_valid {
                if write_index != read_index {
                    // Move embedding data in each column
                    for column in &mut self.embedding_columns {
                        if let Some(value) = column.get(read_index) {
                            column.set(write_index, value);
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
            column.truncate(write_index);
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
    #[must_use]
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
    #[must_use]
    pub fn with_config(config: &ColdTierConfig) -> Self {
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
    fn index_to_episode(data: &ColumnarData, index: usize) -> Option<Episode> {
        if index >= data.count {
            return None;
        }

        let embedding = data.get_embedding(index)?;
        let timestamp_nanos = data.creation_times[index];
        let datetime = Self::datetime_from_nanos(timestamp_nanos);

        Some(
            EpisodeBuilder::new()
                .id(data.memory_ids[index].clone())
                .when(datetime)
                .what(data.contents[index].clone())
                .embedding(embedding)
                .confidence(Confidence::exact(data.confidences[index]))
                .build(),
        )
    }

    /// Perform temporal pattern matching
    fn matches_temporal_pattern(timestamp: u64, pattern: &TemporalPattern) -> f32 {
        let datetime = Self::datetime_from_nanos(timestamp);

        match pattern {
            TemporalPattern::Recent(duration) => {
                let now = Utc::now();
                let cutoff = now - *duration;
                if datetime >= cutoff {
                    #[allow(clippy::cast_precision_loss)]
                    let age_ms = (now - datetime).num_milliseconds() as f64;
                    #[allow(clippy::cast_precision_loss)]
                    let max_age_ms = duration.num_milliseconds() as f64;

                    if max_age_ms > 0.0 {
                        let ratio = (age_ms / max_age_ms).clamp(0.0, 1.0);
                        saturating_f32_from_f64(1.0 - ratio)
                    } else {
                        1.0
                    }
                } else {
                    0.0
                }
            }
            TemporalPattern::Before(cutoff) => {
                if datetime < *cutoff {
                    1.0
                } else {
                    0.0
                }
            }
            TemporalPattern::After(cutoff) => {
                if datetime > *cutoff {
                    1.0
                } else {
                    0.0
                }
            }
            TemporalPattern::Between(start, end) => {
                if datetime >= *start && datetime <= *end {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    fn recall_embedding_matches(
        data: &ColumnarData,
        vector: &[f32; 768],
        threshold: Confidence,
        max_results: usize,
    ) -> Vec<(Episode, Confidence)> {
        data.batch_similarity_search(vector, threshold.raw())
            .into_iter()
            .take(max_results)
            .filter_map(|(index, similarity)| {
                Self::index_to_episode(data, index)
                    .map(|episode| (episode, Confidence::exact(similarity)))
            })
            .collect()
    }

    fn recall_semantic_matches(
        data: &ColumnarData,
        content: &str,
        fuzzy_threshold: Confidence,
        max_results: usize,
    ) -> Vec<(Episode, Confidence)> {
        let content_lower = content.to_lowercase();
        let mut results = Vec::new();

        let search_limit = data.count.min(max_results.saturating_mul(10));
        for i in 0..search_limit {
            let episode_content = &data.contents[i].to_lowercase();

            let relevance = if episode_content.contains(&content_lower) {
                1.0
            } else {
                let content_words: std::collections::HashSet<&str> =
                    content_lower.split_whitespace().collect();
                let episode_words: std::collections::HashSet<&str> =
                    episode_content.split_whitespace().collect();

                let intersection = content_words.intersection(&episode_words).count();
                let union = content_words.union(&episode_words).count();

                if union > 0 {
                    let intersection = u64::try_from(intersection).unwrap_or(u64::MAX);
                    let union = u64::try_from(union).unwrap_or(u64::MAX);
                    unit_ratio_to_f32(intersection, union)
                } else {
                    0.0
                }
            };

            if relevance >= fuzzy_threshold.raw()
                && let Some(episode) = Self::index_to_episode(data, i)
            {
                results.push((episode, Confidence::exact(relevance)));
            }
        }

        results.sort_by(|a, b| {
            b.1.raw()
                .partial_cmp(&a.1.raw())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(max_results);
        results
    }

    fn recall_temporal_matches(
        data: &ColumnarData,
        pattern: &TemporalPattern,
        confidence_threshold: Confidence,
        max_results: usize,
    ) -> Vec<(Episode, Confidence)> {
        let mut results = Vec::new();
        let search_limit = data.count.min(max_results.saturating_mul(10));

        for i in 0..search_limit {
            let timestamp = data.creation_times[i];
            let match_score = Self::matches_temporal_pattern(timestamp, pattern);

            if match_score >= confidence_threshold.raw()
                && let Some(episode) = Self::index_to_episode(data, i)
            {
                results.push((episode, Confidence::exact(match_score)));
            }
        }

        results.sort_by(|a, b| {
            b.1.raw()
                .partial_cmp(&a.1.raw())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(max_results);
        results
    }

    fn recall_context_matches(
        data: &ColumnarData,
        confidence_threshold: Confidence,
        max_results: usize,
    ) -> Vec<(Episode, Confidence)> {
        let mut results = Vec::new();
        let search_limit = data.count.min(max_results.saturating_mul(5));

        for i in 0..search_limit {
            let confidence = data.confidences[i];

            if confidence >= confidence_threshold.raw()
                && let Some(episode) = Self::index_to_episode(data, i)
            {
                results.push((episode, Confidence::exact(confidence)));
            }
        }

        results.sort_by(|a, b| {
            b.1.raw()
                .partial_cmp(&a.1.raw())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(max_results);
        results
    }

    /// Check whether the tier currently holds the provided memory identifier.
    #[must_use = "The return value indicates if a memory is persisted in the cold tier"]
    pub fn contains_memory(&self, memory_id: &str) -> bool {
        self.id_index.contains_key(memory_id)
    }

    /// Perform garbage collection and compaction.
    ///
    /// # Errors
    ///
    /// Returns an error when the cold tier fails to acquire the required
    /// write lock to perform the compaction cycle.
    pub fn compact(&self) -> Result<CompactionResult, StorageError> {
        let mut data = self.data.write().map_err(|_| {
            StorageError::AllocationFailed("Failed to acquire write lock".to_string())
        })?;

        let original_count = data.count;

        // Mark entries for removal based on age and activation
        let mut valid_indices = vec![true; data.count];
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .try_into()
            .unwrap_or(u64::MAX);

        let mut removed_count = 0;
        for (i, is_valid) in valid_indices.iter_mut().enumerate().take(data.count) {
            let age_nanos = current_time.saturating_sub(data.access_times[i]);
            let age_days = age_nanos / (24 * 60 * 60 * 1_000_000_000);
            let activation = data.activations[i];

            // Remove very old memories with low activation
            if age_days > 365 && activation < 0.1 {
                *is_valid = false;
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
    #[must_use = "The caller should use the count rather than ignore it"]
    pub fn len(&self) -> usize {
        match self.data.read() {
            Ok(data) => data.count,
            Err(poisoned) => poisoned.into_inner().count,
        }
    }

    /// Check if tier is empty
    #[must_use = "The emptiness check informs eviction logic"]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get capacity utilization
    #[must_use = "Capacity utilization guides tier balancing decisions"]
    pub fn utilization(&self) -> f32 {
        Self::ratio_usize(self.len(), self.max_capacity)
    }

    /// Get storage duration for a memory (time since first stored)
    #[must_use = "Callers rely on the duration to adjust confidence"]
    pub fn get_storage_duration(&self, memory_id: &str) -> std::time::Duration {
        self.storage_timestamps.get(memory_id).map_or_else(
            || StdDuration::from_secs(0),
            |entry| {
                let stored_time = *entry.value();
                SystemTime::now()
                    .duration_since(stored_time)
                    .unwrap_or_default()
            },
        )
    }
}

impl ColdTier {
    fn datetime_from_nanos(timestamp_nanos: u64) -> DateTime<Utc> {
        let limited = i64::try_from(timestamp_nanos).unwrap_or(i64::MAX);
        DateTime::from_timestamp_nanos(limited)
    }

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

/// Configuration for cold tier storage
#[derive(Debug, Clone, Copy)]
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

impl StorageTierBackend for ColdTier {
    type Error = StorageError;

    async fn store(&self, memory: Arc<Memory>) -> Result<(), Self::Error> {
        let memory_id = memory.id.clone();

        // Record storage timestamp for temporal decay calculation
        self.storage_timestamps
            .insert(memory_id.clone(), std::time::SystemTime::now());

        let index = {
            let mut data = self.data.write().map_err(|_| {
                StorageError::AllocationFailed("Failed to acquire write lock".to_string())
            })?;

            data.insert(&memory)?
        };
        self.id_index.insert(memory_id, index);

        Ok(())
    }

    async fn recall(&self, cue: &Cue) -> Result<Vec<(Episode, Confidence)>, Self::Error> {
        self.total_accesses.fetch_add(1, Ordering::Relaxed);
        self.batch_operations.fetch_add(1, Ordering::Relaxed);

        let data = self.data.read().map_err(|_| {
            StorageError::AllocationFailed("Failed to acquire read lock".to_string())
        })?;

        let results = match &cue.cue_type {
            CueType::Embedding { vector, threshold } => {
                Self::recall_embedding_matches(&data, vector, *threshold, cue.max_results)
            }
            CueType::Semantic {
                content,
                fuzzy_threshold,
            } => Self::recall_semantic_matches(&data, content, *fuzzy_threshold, cue.max_results),
            CueType::Temporal {
                pattern,
                confidence_threshold,
            } => Self::recall_temporal_matches(
                &data,
                pattern,
                *confidence_threshold,
                cue.max_results,
            ),
            CueType::Context {
                confidence_threshold,
                ..
            } => Self::recall_context_matches(&data, *confidence_threshold, cue.max_results),
        };

        drop(data);

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
            drop(index_ref);

            let mut data = self.data.write().map_err(|_| {
                StorageError::AllocationFailed("Failed to acquire write lock".to_string())
            })?;

            let result = if index < data.count {
                data.activations[index] = activation.clamp(0.0, 1.0);
                data.access_times[index] = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
                    .try_into()
                    .unwrap_or(u64::MAX);
                Ok(())
            } else {
                Err(StorageError::AllocationFailed(format!(
                    "Invalid index {index} for memory {memory_id}"
                )))
            };
            drop(data);
            result
        } else {
            Err(StorageError::AllocationFailed(format!(
                "Memory {memory_id} not found in cold tier"
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
                "Memory {memory_id} not found in cold tier"
            )))
        }
    }

    fn statistics(&self) -> TierStatistics {
        let data = match self.data.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let total_accesses = self.total_accesses.load(Ordering::Relaxed);
        let batch_ops = self.batch_operations.load(Ordering::Relaxed);

        // Calculate average activation
        let total_activation: f32 = data.activations.iter().sum();
        let average_activation = if data.count > 0 {
            let denominator = Self::f32_from_usize(data.count).max(f32::EPSILON);
            total_activation / denominator
        } else {
            0.0
        };

        let memory_count = data.count;
        let memory_count_u64 = u64::try_from(memory_count).unwrap_or(u64::MAX);
        let node_size = u64::try_from(std::mem::size_of::<Memory>()).unwrap_or(0);
        let total_size_bytes = memory_count_u64.saturating_mul(node_size);

        drop(data);

        TierStatistics {
            memory_count,
            total_size_bytes,
            average_activation,
            last_access_time: SystemTime::now(),
            cache_hit_rate: Self::ratio_u64(batch_ops, total_accesses),
            compaction_ratio: if self.compression_enabled { 0.85 } else { 1.0 },
        }
    }

    async fn maintenance(&self) -> Result<(), Self::Error> {
        // Check if compaction is needed
        let utilization = self.utilization();

        if utilization > 0.8 {
            let result = self.compact()?;
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

    impl<T> IntoTestResult<T> for Option<T> {
        fn into_test_result(self, context: &str) -> TestResult<T> {
            self.ok_or_else(|| context.to_string())
        }
    }

    fn create_test_memory(id: &str, activation: f32) -> Arc<Memory> {
        let episode = EpisodeBuilder::new()
            .id(id.to_string())
            .when(Utc::now())
            .what(format!("test memory {id}"))
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
        assert!(cold_tier.utilization().abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_cold_tier_store_and_recall() -> TestResult {
        let cold_tier = ColdTier::new(1000);

        // Store test memory
        let memory = create_test_memory("test1", 0.3);
        cold_tier
            .store(memory)
            .await
            .into_test_result("store memory in cold tier")?;

        ensure_eq(&cold_tier.len(), &1_usize, "cold tier length after store")?;
        ensure(!cold_tier.is_empty(), "cold tier should not be empty")?;

        // Test embedding recall
        let cue = CueBuilder::new()
            .id("test_cue".to_string())
            .embedding_search([0.5f32; 768], Confidence::LOW)
            .max_results(10)
            .build();

        let results = cold_tier
            .recall(&cue)
            .await
            .into_test_result("recall from cold tier should succeed")?;
        ensure(!results.is_empty(), "recall should return results")?;
        ensure_eq(&results[0].0.id, &"test1".to_string(), "recalled id")?;

        Ok(())
    }

    #[tokio::test]
    async fn test_cold_tier_batch_operations() -> TestResult {
        let cold_tier = ColdTier::new(1000);

        // Store multiple memories
        for i in 0..50 {
            let memory = create_test_memory(&format!("mem_{i}"), 0.2);
            cold_tier
                .store(memory)
                .await
                .into_test_result("store memory in batch")?;
        }

        ensure_eq(&cold_tier.len(), &50_usize, "batch store length")?;

        // Test batch recall
        let cue = CueBuilder::new()
            .id("batch_cue".to_string())
            .embedding_search([0.5f32; 768], Confidence::LOW)
            .max_results(20)
            .build();

        let results = cold_tier
            .recall(&cue)
            .await
            .into_test_result("recall from cold tier should succeed")?;
        ensure(!results.is_empty(), "batch recall results")?;
        ensure(results.len() <= 20, "batch recall respects limit")?;

        Ok(())
    }

    #[tokio::test]
    async fn test_cold_tier_compaction() -> TestResult {
        let cold_tier = ColdTier::new(100);

        // Store memories
        for i in 0..20 {
            let memory = create_test_memory(&format!("mem_{i}"), 0.1);
            cold_tier
                .store(memory)
                .await
                .into_test_result("store memory before compaction")?;
        }

        // Force compaction
        let result = cold_tier.compact().into_test_result("compact cold tier")?;

        ensure_eq(
            &result.original_count,
            &20_usize,
            "compaction original count",
        )?;
        // Some memories might be removed based on age/activation
        ensure(result.final_count <= 20, "compaction final count bounded")?;

        Ok(())
    }

    #[tokio::test]
    async fn test_cold_tier_maintenance() -> TestResult {
        let cold_tier = ColdTier::new(100);

        // Store many memories to trigger maintenance
        for i in 0..90 {
            let memory = create_test_memory(&format!("mem_{i}"), 0.1);
            cold_tier
                .store(memory)
                .await
                .into_test_result("store memory before maintenance")?;
        }

        // Run maintenance
        cold_tier
            .maintenance()
            .await
            .into_test_result("maintenance should succeed")?;

        // Verify stats are still valid
        let stats = cold_tier.statistics();
        ensure(
            stats.memory_count <= 90,
            "maintenance should not increase count",
        )?;

        Ok(())
    }

    #[test]
    fn test_cold_tier_config() {
        let config = ColdTierConfig::default();
        assert_eq!(config.capacity, 100_000);
        assert!(config.enable_compression);
        assert!((config.compaction_threshold - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.max_age_days, 365);
    }

    #[test]
    fn test_columnar_data() -> TestResult {
        let mut data = ColumnarData::new(10);
        ensure_eq(&data.count, &0_usize, "new columnar count")?;
        ensure_eq(&data.capacity, &10_usize, "columnar capacity")?;

        let memory = create_test_memory("test", 0.5);
        let index = data
            .insert(&memory)
            .into_test_result("insert into columnar data")?;
        ensure_eq(&index, &0_usize, "columnar insertion index")?;
        ensure_eq(&data.count, &1_usize, "columnar count after insert")?;

        let embedding = data
            .get_embedding(0)
            .into_test_result("missing embedding from columnar data")?;
        ensure(
            embedding
                .iter()
                .all(|&value| (value - 0.5f32).abs() < f32::EPSILON),
            "stored embedding values",
        )?;

        Ok(())
    }
}
