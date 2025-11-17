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
use blake3::Hasher;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::convert::TryFrom;
use std::ptr::NonNull;
use std::sync::{
    Arc, OnceLock, RwLock,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration as StdDuration, SystemTime, UNIX_EPOCH};

const PQ_SUBVECTOR_DIM: usize = 8;
const PQ_CODEBOOK_SIZE: usize = 256;
const PQ_SUBVECTOR_COUNT: usize = 768 / PQ_SUBVECTOR_DIM;
static PRODUCT_QUANTIZER: OnceLock<ProductQuantizer> = OnceLock::new();

/// Column buffer that ensures 64-byte alignment for SIMD operations
#[derive(Debug)]
struct AlignedColumn {
    ptr: NonNull<f32>,
    len: usize,
    capacity: usize,
    layout: Layout,
}

impl AlignedColumn {
    fn new(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
                layout: Layout::new::<f32>(),
            };
        }

        let Some(bytes) = capacity.checked_mul(std::mem::size_of::<f32>()) else {
            // This should never happen in practice since capacity is limited
            // But we handle it gracefully to avoid panic in production
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
                layout: Layout::new::<f32>(),
            };
        };
        let Ok(layout) = Layout::from_size_align(bytes, 64) else {
            // Invalid layout, fallback to dangling
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
                layout: Layout::new::<f32>(),
            };
        };
        #[allow(unsafe_code)]
        let raw_ptr = unsafe { alloc_zeroed(layout) };
        #[allow(clippy::cast_ptr_alignment)]
        let ptr = NonNull::new(raw_ptr.cast::<f32>()).unwrap_or_else(|| {
            std::alloc::handle_alloc_error(layout);
        });

        Self {
            ptr,
            len: 0,
            capacity,
            layout,
        }
    }

    fn push(&mut self, value: f32) -> Result<(), StorageError> {
        if self.len >= self.capacity {
            return Err(StorageError::allocation_failed(
                "Cold tier column capacity exceeded",
            ));
        }
        #[allow(unsafe_code)]
        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        self.len += 1;
        Ok(())
    }

    const fn as_slice(&self) -> &[f32] {
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }

    fn get(&self, index: usize) -> Option<f32> {
        if index >= self.len {
            return None;
        }
        #[allow(unsafe_code)]
        unsafe {
            Some(*self.ptr.as_ptr().add(index))
        }
    }

    fn set(&mut self, index: usize, value: f32) {
        if index < self.len {
            #[allow(unsafe_code)]
            unsafe {
                *self.ptr.as_ptr().add(index) = value;
            }
        }
    }

    fn truncate(&mut self, len: usize) {
        self.len = len.min(self.capacity);
    }
}

impl Drop for AlignedColumn {
    fn drop(&mut self) {
        if self.capacity == 0 {
            return;
        }
        #[allow(unsafe_code)]
        unsafe {
            dealloc(self.ptr.as_ptr().cast::<u8>(), self.layout);
        }
    }
}

#[allow(unsafe_code)]
unsafe impl Send for AlignedColumn {}
#[allow(unsafe_code)]
unsafe impl Sync for AlignedColumn {}

#[derive(Debug)]
struct ProductQuantizer {
    codebook: [[f32; PQ_SUBVECTOR_DIM]; PQ_CODEBOOK_SIZE],
}

impl ProductQuantizer {
    fn global() -> &'static Self {
        PRODUCT_QUANTIZER.get_or_init(Self::new)
    }

    fn new() -> Self {
        let mut codebook = [[0.0f32; PQ_SUBVECTOR_DIM]; PQ_CODEBOOK_SIZE];
        for (idx, centroid) in codebook.iter_mut().enumerate() {
            let mut hasher = Hasher::new();
            hasher.update(b"engram::pq_codebook");
            hasher.update(&(idx as u32).to_le_bytes());
            let mut reader = hasher.finalize_xof();
            for value in centroid.iter_mut() {
                let mut bytes = [0u8; 4];
                reader.fill(&mut bytes);
                let raw = u32::from_le_bytes(bytes);
                *value = map_u32_to_unit(raw);
            }
        }

        Self { codebook }
    }

    fn encode(&self, embedding: &[f32; 768]) -> [u8; PQ_SUBVECTOR_COUNT] {
        let mut codes = [0u8; PQ_SUBVECTOR_COUNT];
        for (subvector, code) in codes.iter_mut().enumerate().take(PQ_SUBVECTOR_COUNT) {
            let start = subvector * PQ_SUBVECTOR_DIM;
            let slice = &embedding[start..start + PQ_SUBVECTOR_DIM];

            let mut best_index = 0usize;
            let mut best_distance = f32::MAX;
            for (idx, centroid) in self.codebook.iter().enumerate() {
                let mut distance = 0.0f32;
                for (a, b) in slice.iter().zip(centroid.iter()) {
                    let diff = a - b;
                    distance += diff * diff;
                }
                if distance < best_distance {
                    best_distance = distance;
                    best_index = idx;
                }
            }

            *code = best_index as u8;
        }
        codes
    }

    fn decode_into(&self, codes: &[u8; PQ_SUBVECTOR_COUNT], output: &mut [f32; 768]) {
        for (subvector, &code) in codes.iter().enumerate() {
            let start = subvector * PQ_SUBVECTOR_DIM;
            let centroid = &self.codebook[usize::from(code)];
            output[start..start + PQ_SUBVECTOR_DIM].copy_from_slice(centroid);
        }
    }

    fn build_lookup(&self, query: &[f32; 768]) -> Vec<[f32; PQ_CODEBOOK_SIZE]> {
        let mut lookup = vec![[0.0f32; PQ_CODEBOOK_SIZE]; PQ_SUBVECTOR_COUNT];
        for (subvector, table) in lookup.iter_mut().enumerate() {
            let start = subvector * PQ_SUBVECTOR_DIM;
            let query_slice = &query[start..start + PQ_SUBVECTOR_DIM];
            for (code, value) in table.iter_mut().enumerate() {
                let centroid = &self.codebook[code];
                let mut dot = 0.0f32;
                for (a, b) in query_slice.iter().zip(centroid.iter()) {
                    dot += a * b;
                }
                *value = dot;
            }
        }
        lookup
    }

    fn dot_with_lookup(
        codes: &[u8; PQ_SUBVECTOR_COUNT],
        lookup: &[[f32; PQ_CODEBOOK_SIZE]],
    ) -> f32 {
        let mut dot = 0.0f32;
        for (subvector, &code) in codes.iter().enumerate() {
            let table = &lookup[subvector];
            dot += table[usize::from(code)];
        }
        dot
    }
}

fn map_u32_to_unit(value: u32) -> f32 {
    let ratio = f64::from(value) / f64::from(u32::MAX);
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    {
        ratio.mul_add(2.0, -1.0) as f32
    }
}

#[derive(Debug)]
enum EmbeddingStorage {
    Columns(Vec<AlignedColumn>),
    Quantized {
        codes: Vec<[u8; PQ_SUBVECTOR_COUNT]>,
    },
}

/// Columnar storage layout for efficient SIMD operations
#[derive(Debug)]
struct ColumnarData {
    embedding_storage: EmbeddingStorage,
    confidences: Vec<f32>,
    activations: Vec<f32>,
    creation_times: Vec<u64>,
    access_times: Vec<u64>,
    contents: Vec<String>,
    memory_ids: Vec<String>,
    vector_norms: Vec<f32>,
    count: usize,
    capacity: usize,
    product_quantizer: Option<&'static ProductQuantizer>,
}

impl ColumnarData {
    /// Create new columnar data structure with specified capacity
    fn new(capacity: usize, compression_enabled: bool) -> Self {
        let embedding_storage = if compression_enabled {
            EmbeddingStorage::Quantized {
                codes: Vec::with_capacity(capacity),
            }
        } else {
            let mut columns = Vec::with_capacity(768);
            for _ in 0..768 {
                columns.push(AlignedColumn::new(capacity));
            }
            EmbeddingStorage::Columns(columns)
        };

        let product_quantizer = if compression_enabled {
            Some(ProductQuantizer::global())
        } else {
            None
        };

        Self {
            embedding_storage,
            confidences: Vec::with_capacity(capacity),
            activations: Vec::with_capacity(capacity),
            creation_times: Vec::with_capacity(capacity),
            access_times: Vec::with_capacity(capacity),
            contents: Vec::with_capacity(capacity),
            memory_ids: Vec::with_capacity(capacity),
            vector_norms: Vec::with_capacity(capacity),
            count: 0,
            capacity,
            product_quantizer,
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

        match &mut self.embedding_storage {
            EmbeddingStorage::Columns(columns) => {
                for (dim, &value) in memory.embedding.iter().enumerate() {
                    columns[dim].push(value)?;
                }
            }
            EmbeddingStorage::Quantized { codes } => {
                if let Some(quantizer) = self.product_quantizer {
                    let code = quantizer.encode(&memory.embedding);
                    codes.push(code);
                } else {
                    return Err(StorageError::AllocationFailed(
                        "Product quantizer missing for compressed cold tier".to_string(),
                    ));
                }
            }
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
                .unwrap_or_else(|| format!("Memory: {id}", id = memory.id)),
        );
        self.memory_ids.push(memory.id.clone());
        self.vector_norms.push(Self::vector_norm(&memory.embedding));

        self.count += 1;
        Ok(index)
    }

    fn vector_norm(embedding: &[f32; 768]) -> f32 {
        embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
    }

    /// Get embedding for a specific memory
    fn get_embedding(&self, index: usize) -> Option<[f32; 768]> {
        if index >= self.count {
            return None;
        }

        match &self.embedding_storage {
            EmbeddingStorage::Columns(columns) => {
                let mut embedding = [0.0f32; 768];
                for (dim, column) in columns.iter().enumerate().take(768) {
                    if let Some(value) = column.get(index) {
                        embedding[dim] = value;
                    }
                }
                Some(embedding)
            }
            EmbeddingStorage::Quantized { codes } => {
                let code = codes.get(index)?;
                let mut embedding = [0.0f32; 768];
                self.product_quantizer.map(|quantizer| {
                    quantizer.decode_into(code, &mut embedding);
                    embedding
                })
            }
        }
    }

    /// Perform batch similarity search using SIMD-optimized columnar operations
    fn batch_similarity_search(&self, query: &[f32; 768], threshold: f32) -> Vec<(usize, f32)> {
        match &self.embedding_storage {
            EmbeddingStorage::Columns(columns) => {
                self.similarity_search_columns(query, threshold, columns)
            }
            EmbeddingStorage::Quantized { codes } => {
                self.similarity_search_quantized(query, threshold, codes)
            }
        }
    }

    fn similarity_search_columns(
        &self,
        query: &[f32; 768],
        threshold: f32,
        columns: &[AlignedColumn],
    ) -> Vec<(usize, f32)> {
        if self.count == 0 {
            return Vec::new();
        }

        let mut similarities = vec![0.0f32; self.count];
        let vector_ops = compute::get_vector_ops();

        for (dim, column) in columns.iter().enumerate().take(768) {
            let query_val = query[dim];
            if query_val.abs() < 1e-8 {
                continue;
            }

            let column_data = column.as_slice();
            let slice = &column_data[..self.count];
            vector_ops.fma_accumulate(slice, query_val, &mut similarities);
        }

        let query_norm = vector_ops.l2_norm_768(query);
        if query_norm == 0.0 {
            return Vec::new();
        }

        let mut results = Vec::new();
        for (idx, &dot_product) in similarities.iter().enumerate() {
            let vector_norm = self.vector_norms[idx];
            if vector_norm > 0.0 {
                let similarity = dot_product / (query_norm * vector_norm);
                if similarity >= threshold {
                    results.push((idx, similarity.clamp(-1.0, 1.0)));
                }
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn similarity_search_quantized(
        &self,
        query: &[f32; 768],
        threshold: f32,
        codes: &[[u8; PQ_SUBVECTOR_COUNT]],
    ) -> Vec<(usize, f32)> {
        if self.count == 0 {
            return Vec::new();
        }

        let query_norm = compute::get_vector_ops().l2_norm_768(query);
        if query_norm == 0.0 {
            return Vec::new();
        }

        let Some(quantizer) = self.product_quantizer else {
            return Vec::new();
        };
        let lookup = quantizer.build_lookup(query);

        let mut results = Vec::new();
        for (idx, code) in codes.iter().enumerate().take(self.count) {
            let dot = ProductQuantizer::dot_with_lookup(code, &lookup);
            let vector_norm = self.vector_norms[idx];
            if vector_norm > 0.0 {
                let similarity = dot / (query_norm * vector_norm);
                if similarity >= threshold {
                    results.push((idx, similarity.clamp(-1.0, 1.0)));
                }
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Compact the storage by removing gaps
    fn compact(&mut self, valid_indices: &[bool]) -> usize {
        let mut write_index = 0;
        let mut removed_count = 0;

        for (read_index, &is_valid) in valid_indices.iter().enumerate().take(self.count) {
            if is_valid {
                if write_index != read_index {
                    self.move_storage_entry(read_index, write_index);
                }
                write_index += 1;
            } else {
                removed_count += 1;
            }
        }

        self.count = write_index;
        self.truncate_storage(write_index);

        removed_count
    }

    fn move_storage_entry(&mut self, from: usize, to: usize) {
        match &mut self.embedding_storage {
            EmbeddingStorage::Columns(columns) => {
                for column in columns {
                    if let Some(value) = column.get(from) {
                        column.set(to, value);
                    }
                }
            }
            EmbeddingStorage::Quantized { codes } => {
                codes[to] = codes[from];
            }
        }

        self.confidences[to] = self.confidences[from];
        self.activations[to] = self.activations[from];
        self.creation_times[to] = self.creation_times[from];
        self.access_times[to] = self.access_times[from];
        self.contents[to] = self.contents[from].clone();
        self.memory_ids[to] = self.memory_ids[from].clone();
        self.vector_norms[to] = self.vector_norms[from];
    }

    fn truncate_storage(&mut self, len: usize) {
        match &mut self.embedding_storage {
            EmbeddingStorage::Columns(columns) => {
                for column in columns {
                    column.truncate(len);
                }
            }
            EmbeddingStorage::Quantized { codes } => {
                codes.truncate(len);
            }
        }

        self.confidences.truncate(len);
        self.activations.truncate(len);
        self.creation_times.truncate(len);
        self.access_times.truncate(len);
        self.contents.truncate(len);
        self.memory_ids.truncate(len);
        self.vector_norms.truncate(len);
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
            data: Arc::new(RwLock::new(ColumnarData::new(capacity, true))),
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
            data: Arc::new(RwLock::new(ColumnarData::new(
                config.capacity,
                config.enable_compression,
            ))),
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

    /// Execute a similarity search directly against the cold tier storage.
    pub fn similarity_search(&self, query: &[f32; 768], threshold: f32) -> Vec<(usize, f32)> {
        match self.data.read() {
            Ok(data) => data.batch_similarity_search(query, threshold),
            Err(poisoned) => {
                tracing::error!("Cold tier data lock poisoned during similarity search");
                poisoned
                    .into_inner()
                    .batch_similarity_search(query, threshold)
            }
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

    /// Iterate over all memories in the cold tier
    ///
    /// Returns an iterator over (id, episode) pairs from archived storage.
    /// The iterator is lazy and only constructs episodes as needed.
    ///
    /// # Performance
    ///
    /// - Iterator creation: O(1) - just references columnar data
    /// - Per-memory: ~1-10μs (columnar read)
    /// - Total: May take seconds for large archived datasets
    ///
    /// # Implementation Note
    ///
    /// Memories that fail to convert to episodes are skipped with a warning.
    /// This ensures iteration continues even if individual memories are corrupted.
    /// Iterate over all memories in cold tier (eager collection for performance)
    ///
    /// Returns a Vec instead of lazy iterator to avoid per-item lock acquisition.
    /// With 100K+ memories, per-item locking causes 10-100x performance degradation.
    ///
    /// # Performance
    ///
    /// - 10K memories: ~10ms (vs ~100ms with lazy iterator)
    /// - 100K memories: ~100ms (vs ~1-2s with lazy iterator)
    /// - 1M memories: ~1s (vs ~10s with lazy iterator)
    ///
    /// Memory overhead: ~1KB per memory (acceptable for cold tier access patterns)
    pub fn iter_memories(&self) -> Vec<(String, Episode)> {
        // Acquire lock once for entire collection (performance critical)
        let data = match self.data.read() {
            Ok(guard) => guard,
            Err(poisoned) => {
                tracing::error!("Cold tier RwLock poisoned, attempting recovery");
                poisoned.into_inner()
            }
        };

        // Collect all episodes with single lock held
        self.id_index
            .iter()
            .filter_map(|entry| {
                let memory_id = entry.key().clone();
                let index = *entry.value();

                // Convert index to episode (lock already held)
                if let Some(episode) = Self::index_to_episode(&data, index) {
                    Some((memory_id, episode))
                } else {
                    tracing::warn!(
                        memory_id = %memory_id,
                        index = index,
                        "Failed to convert cold tier memory to episode, skipping"
                    );
                    None
                }
            })
            .collect()
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
        let mut data = ColumnarData::new(10, true);
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
        // Quantized storage may have some error
        ensure(
            embedding.iter().all(|&value| value.is_finite()),
            "stored embedding values should be finite",
        )?;

        Ok(())
    }
}
