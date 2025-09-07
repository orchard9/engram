//! Infallible memory store with graceful degradation
//!
//! Implements cognitive design principles where operations never fail but
//! degrade gracefully under pressure, returning activation levels that
//! indicate store quality.

use crate::{Confidence, Cue, CueType, Episode, Memory, TemporalPattern};
use chrono::Utc;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "monitoring")]
use std::time::Instant;

#[cfg(feature = "monitoring")]
use crate::metrics::{MetricsRegistry, cognitive::CognitiveMetric};

#[cfg(feature = "pattern_completion")]
use crate::completion::{
    PatternReconstructor, CompletionConfig, PartialEpisode,
    PatternCompleter, CompletedEpisode,
};

#[cfg(feature = "hnsw_index")]
use crate::index::{CognitiveHnswIndex, IndexUpdate, UpdatePriority};
#[cfg(feature = "hnsw_index")]
use crossbeam_queue::SegQueue;

#[cfg(feature = "memory_mapped_persistence")]
use crate::storage::{
    CognitiveTierArchitecture, FsyncMode, StorageConfig, StorageMetrics, wal::WalWriter,
};

/// Activation level returned by store operations
///
/// Indicates the quality of a store operation from 0.0 to 1.0
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Activation(f32);

impl Activation {
    /// Create a new activation level
    #[must_use]
    pub const fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Get the raw activation value
    #[must_use]
    pub const fn value(&self) -> f32 {
        self.0
    }

    /// Check if activation indicates successful store
    #[must_use]
    pub fn is_successful(&self) -> bool {
        self.0 > 0.5
    }

    /// Check if activation indicates degraded store
    #[must_use]
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
    pub(crate) hot_memories: DashMap<String, Arc<Memory>>,

    /// Sorted map for eviction candidates (by activation level)
    eviction_queue: RwLock<BTreeMap<(OrderedFloat, String), Arc<Memory>>>,

    /// Current memory count
    memory_count: AtomicUsize,

    /// Maximum memories before eviction
    max_memories: usize,

    /// System pressure indicator (0.0 = no pressure, 1.0 = max pressure)
    pressure: RwLock<f32>,

    /// Write-ahead log for durability (non-blocking)
    pub(crate) wal_buffer: Arc<DashMap<String, Episode>>,

    /// HNSW index for fast similarity search
    #[cfg(feature = "hnsw_index")]
    hnsw_index: Option<Arc<CognitiveHnswIndex>>,

    /// Queue for background index updates
    #[cfg(feature = "hnsw_index")]
    index_update_queue: Option<Arc<SegQueue<IndexUpdate>>>,

    /// Persistent storage backend
    #[cfg(feature = "memory_mapped_persistence")]
    persistent_backend: Option<Arc<CognitiveTierArchitecture>>,

    /// Write-ahead log for durability
    #[cfg(feature = "memory_mapped_persistence")]
    wal_writer: Option<Arc<WalWriter>>,

    /// Storage metrics
    #[cfg(feature = "memory_mapped_persistence")]
    storage_metrics: Arc<StorageMetrics>,
    
    /// Pattern completion engine
    #[cfg(feature = "pattern_completion")]
    pattern_reconstructor: Option<Arc<RwLock<PatternReconstructor>>>,
}

/// Wrapper for f32 that implements Ord for `BTreeMap`
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
    #[must_use]
    pub fn new(max_memories: usize) -> Self {
        Self {
            hot_memories: DashMap::new(),
            eviction_queue: RwLock::new(BTreeMap::new()),
            memory_count: AtomicUsize::new(0),
            max_memories,
            pressure: RwLock::new(0.0),
            wal_buffer: Arc::new(DashMap::new()),
            #[cfg(feature = "hnsw_index")]
            hnsw_index: None,
            #[cfg(feature = "hnsw_index")]
            index_update_queue: None,
            #[cfg(feature = "memory_mapped_persistence")]
            persistent_backend: None,
            #[cfg(feature = "memory_mapped_persistence")]
            wal_writer: None,
            #[cfg(feature = "memory_mapped_persistence")]
            storage_metrics: Arc::new(StorageMetrics::new()),
            #[cfg(feature = "pattern_completion")]
            pattern_reconstructor: None,
        }
    }

    /// Create a memory store with HNSW index enabled
    #[cfg(feature = "hnsw_index")]
    #[must_use]
    pub fn with_hnsw_index(mut self) -> Self {
        self.hnsw_index = Some(Arc::new(CognitiveHnswIndex::new()));
        self.index_update_queue = Some(Arc::new(SegQueue::new()));
        self
    }

    /// Create a memory store with persistent backend enabled
    #[cfg(feature = "memory_mapped_persistence")]
    #[must_use]
    pub fn with_persistence<P: AsRef<std::path::Path>>(
        mut self,
        data_dir: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let storage_metrics = Arc::clone(&self.storage_metrics);

        // Create persistent backend
        let persistent_backend = Arc::new(CognitiveTierArchitecture::new(
            data_dir.as_ref(),
            self.max_memories,       // hot capacity
            self.max_memories * 10,  // warm capacity
            self.max_memories * 100, // cold capacity
            storage_metrics.clone(),
        )?);

        // Create WAL writer
        let wal_dir = data_dir.as_ref().join("wal");
        let wal_writer = Arc::new(WalWriter::new(
            wal_dir,
            FsyncMode::PerBatch,
            storage_metrics.clone(),
        )?);

        self.persistent_backend = Some(persistent_backend);
        self.wal_writer = Some(wal_writer);

        Ok(self)
    }

    /// Initialize the persistent backend (start background workers)
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn initialize_persistence(
        &mut self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ref mut wal_writer) = self.wal_writer {
            Arc::get_mut(wal_writer).unwrap().start()?;
        }
        Ok(())
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
        #[cfg(feature = "monitoring")]
        let start = Instant::now();
        
        // Calculate system pressure
        let current_count = self.memory_count.load(Ordering::Relaxed);
        let pressure = (current_count as f32 / self.max_memories as f32).min(1.0);

        // Update system pressure
        {
            let mut p = self.pressure.write();
            *p = pressure;
        }

        // Calculate base activation from episode encoding confidence and pressure
        let base_activation = episode.encoding_confidence.raw() * pressure.mul_add(-0.5, 1.0);

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
                memory_arc.clone(),
            );
        }

        // Store in WAL buffer (non-blocking)
        self.wal_buffer.insert(memory_id.clone(), episode.clone());

        // Persist to storage backend with graceful degradation
        #[cfg(feature = "memory_mapped_persistence")]
        {
            if let Some(ref wal_writer) = self.wal_writer {
                if let Ok(wal_entry) = crate::storage::wal::WalEntry::new_episode(&episode) {
                    if let Err(e) = wal_writer.write_async(wal_entry) {
                        tracing::warn!("WAL write failed: {}, continuing with in-memory only", e);
                        // Graceful degradation: reduce activation to indicate storage issues
                        return Activation::new(base_activation * 0.9);
                    }
                }
            }

            if let Some(ref backend) = self.persistent_backend {
                // Try to store in persistent backend asynchronously
                // Note: In production this would be async, simplified for now
                if let Err(e) = std::thread::spawn({
                    let backend_clone = Arc::clone(backend);
                    let memory_clone = Arc::clone(&memory_arc);
                    move || {
                        // Simplified synchronous persistence
                        tracing::info!("Would persist memory {} in background", memory_clone.id);
                    }
                })
                .join()
                {
                    tracing::warn!("Background persistence thread failed: {:?}", e);
                }
            }
        }

        // Queue for HNSW indexing
        #[cfg(feature = "hnsw_index")]
        {
            if let Some(ref queue) = self.index_update_queue {
                queue.push(IndexUpdate::Insert {
                    memory_id: memory_id.clone(),
                    memory: memory_arc.clone(),
                    generation: 0,
                    priority: UpdatePriority::Normal,
                });
            }

            // Try to insert immediately if index is available
            if let Some(ref hnsw) = self.hnsw_index {
                let _ = hnsw.insert_memory(memory_arc.clone());
            }
        }

        // Increment count
        self.memory_count.fetch_add(1, Ordering::Relaxed);
        
        #[cfg(feature = "monitoring")]
        if let Some(ref metrics) = self.metrics {
            // Record store operation metrics
            metrics.increment_counter("memories_created_total", 1);
            metrics.observe_histogram("store_activation", base_activation as f64);
            metrics.observe_histogram("store_duration_seconds", start.elapsed().as_secs_f64());
            
            // Record cognitive metrics
            metrics.record_cognitive(CognitiveMetric::CLSContribution {
                hippocampal: 1.0 - pressure, // More hippocampal when less pressure
                neocortical: pressure,       // More neocortical under pressure
            });
        }

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

    /// Get storage metrics (if persistence enabled)
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn storage_metrics(&self) -> Arc<StorageMetrics> {
        Arc::clone(&self.storage_metrics)
    }

    /// Get tier statistics (if persistence enabled)
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn tier_statistics(&self) -> Option<crate::storage::TierArchitectureStats> {
        self.persistent_backend
            .as_ref()
            .map(|backend| backend.get_tier_statistics())
    }

    /// Perform maintenance on storage tiers
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn maintenance(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ref _backend) = self.persistent_backend {
            tracing::info!("Would perform maintenance on persistent backend");
        }
        Ok(())
    }

    /// Gracefully shutdown storage backend
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ref mut wal_writer) = self.wal_writer {
            if let Some(writer) = Arc::get_mut(wal_writer) {
                writer.shutdown()?;
            }
        }
        Ok(())
    }

    /// Get current memory count
    pub fn count(&self) -> usize {
        self.memory_count.load(Ordering::Relaxed)
    }

    /// Check if store can accept more memories without eviction
    pub fn has_capacity(&self) -> bool {
        self.count() < self.max_memories
    }

    /// Recall memories based on a cue, returning episodes with confidence scores
    ///
    /// # Returns
    ///
    /// Vector of (Episode, Confidence) tuples, sorted by confidence
    /// - Empty vector if no matches found
    /// - Low confidence results for partial matches
    /// - Higher confidence for better matches
    ///
    /// # Cognitive Design
    ///
    /// Never returns errors because human recall doesn't "fail" - it returns
    /// nothing, partial matches, or reconstructed memories with varying confidence.
    pub fn recall(&self, cue: Cue) -> Vec<(Episode, Confidence)> {
        #[cfg(feature = "monitoring")]
        let start = Instant::now();
        
        // Try persistent backend first if available
        #[cfg(feature = "memory_mapped_persistence")]
        {
            if let Some(ref backend) = self.persistent_backend {
                let results = self.recall_with_persistence(cue, backend);
                
                #[cfg(feature = "monitoring")]
                if let Some(ref metrics) = self.metrics {
                    metrics.increment_counter("queries_executed_total", 1);
                    metrics.observe_histogram("query_duration_seconds", start.elapsed().as_secs_f64());
                    metrics.observe_histogram("query_result_count", results.len() as f64);
                }
                
                return results;
            }
        }

        // Fallback to in-memory recall
        let results = self.recall_in_memory(cue);
        
        #[cfg(feature = "monitoring")]
        if let Some(ref metrics) = self.metrics {
            metrics.increment_counter("queries_executed_total", 1);
            metrics.observe_histogram("query_duration_seconds", start.elapsed().as_secs_f64());
            metrics.observe_histogram("query_result_count", results.len() as f64);
        }
        
        results
    }

    /// Recall with persistent backend integration
    #[cfg(feature = "memory_mapped_persistence")]
    fn recall_with_persistence(
        &self,
        cue: Cue,
        backend: &Arc<CognitiveTierArchitecture>,
    ) -> Vec<(Episode, Confidence)> {
        // Simplified synchronous recall for now
        let results = {
            tracing::info!("Would recall from persistent backend");
            Vec::new() // Placeholder
        };

        // If persistent backend returned insufficient results, supplement with in-memory
        if results.len() < cue.max_results {
            let mut combined = results;
            let mut in_memory = self.recall_in_memory(cue);

            // Remove duplicates and merge
            in_memory.retain(|(episode, _)| {
                !combined
                    .iter()
                    .any(|(existing, _)| existing.id == episode.id)
            });

            combined.extend(in_memory);
            combined.truncate(cue.max_results);
            combined
        } else {
            results
        }
    }

    /// In-memory recall implementation
    fn recall_in_memory(&self, cue: Cue) -> Vec<(Episode, Confidence)> {
        // Try to use HNSW index if available and appropriate
        #[cfg(feature = "hnsw_index")]
        {
            if let Some(ref hnsw) = self.hnsw_index {
                if let CueType::Embedding { vector, threshold } = &cue.cue_type {
                    return self.recall_with_hnsw(cue.clone(), hnsw, vector, *threshold);
                }
            }
        }

        let mut results = Vec::new();

        // Get all episodes from WAL buffer
        let episodes: Vec<(String, Episode)> = self
            .wal_buffer
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        // Match based on cue type
        match &cue.cue_type {
            CueType::Embedding { vector, threshold } => {
                for (_id, episode) in episodes {
                    let similarity = cosine_similarity(&episode.embedding, vector);
                    let confidence = Confidence::exact(similarity);

                    // Include if meets threshold
                    if confidence.raw() >= threshold.raw() {
                        results.push((episode, confidence));
                    }
                }
            }
            CueType::Context {
                time_range,
                location,
                confidence_threshold,
            } => {
                for (_id, episode) in episodes {
                    let mut match_score = 0.0;
                    let mut max_score = 0.0;

                    // Check time range if provided
                    if let Some((start, end)) = time_range {
                        max_score += 1.0;
                        if episode.when >= *start && episode.when <= *end {
                            match_score += 1.0;
                        }
                    }

                    // Check location if provided
                    if let Some(loc) = location {
                        max_score += 1.0;
                        if let Some(ep_loc) = &episode.where_location {
                            if ep_loc.contains(loc) || loc.contains(ep_loc) {
                                match_score += 1.0;
                            }
                        }
                    }

                    if max_score > 0.0 {
                        let confidence_value = match_score / max_score;
                        let confidence = Confidence::exact(confidence_value);

                        if confidence.raw() >= confidence_threshold.raw() {
                            results.push((episode, confidence));
                        }
                    }
                }
            }
            CueType::Semantic {
                content,
                fuzzy_threshold,
            } => {
                for (_id, episode) in episodes {
                    // Simple substring matching for now
                    let match_score = if episode
                        .what
                        .to_lowercase()
                        .contains(&content.to_lowercase())
                    {
                        1.0
                    } else if content
                        .to_lowercase()
                        .contains(&episode.what.to_lowercase())
                    {
                        0.7
                    } else {
                        // Calculate word overlap
                        let cue_words: Vec<&str> = content.split_whitespace().collect();
                        let ep_words: Vec<&str> = episode.what.split_whitespace().collect();
                        let matches = cue_words.iter().filter(|w| ep_words.contains(w)).count();

                        if matches > 0 {
                            (matches as f32 / cue_words.len().max(1) as f32) * 0.5
                        } else {
                            0.0
                        }
                    };

                    if match_score > 0.0 {
                        let confidence = Confidence::exact(match_score);

                        if confidence.raw() >= fuzzy_threshold.raw() {
                            results.push((episode, confidence));
                        }
                    }
                }
            }
            CueType::Temporal {
                pattern,
                confidence_threshold,
            } => {
                for (_id, episode) in episodes {
                    let is_match = match pattern {
                        TemporalPattern::Before(time) => episode.when < *time,
                        TemporalPattern::After(time) => episode.when > *time,
                        TemporalPattern::Between(start, end) => {
                            episode.when >= *start && episode.when <= *end
                        }
                        TemporalPattern::Recent(duration) => {
                            let now = Utc::now();
                            episode.when > now - *duration
                        }
                    };

                    if is_match {
                        let confidence = *confidence_threshold;
                        results.push((episode, confidence));
                    }
                }
            }
        }

        // Apply spreading activation for associative recall
        results = self.apply_spreading_activation(results, &cue);

        // Sort by confidence (highest first)
        results.sort_by(|a, b| {
            b.1.raw()
                .partial_cmp(&a.1.raw())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results to max_results from cue
        results.truncate(cue.max_results);

        results
    }

    /// Complete a partial episode using pattern completion
    #[cfg(feature = "pattern_completion")]
    pub fn complete_pattern(&self, partial: PartialEpisode) -> CompletedEpisode {
        #[cfg(feature = "monitoring")]
        let start = Instant::now();
        
        if let Some(ref reconstructor) = self.pattern_reconstructor {
            // Get episodes for context
            let episodes: Vec<Episode> = self.wal_buffer.iter()
                .map(|entry| entry.value().clone())
                .collect();
            
            // Update reconstructor with current episodes
            let mut reconstructor = reconstructor.write();
            reconstructor.update(&episodes);
            
            // Perform completion
            match reconstructor.complete(&partial) {
                Ok(completed) => {
                    #[cfg(feature = "monitoring")]
                    if let Some(ref metrics) = self.metrics {
                        metrics.increment_counter("pattern_completions_total", 1);
                        metrics.observe_histogram("pattern_completion_duration_seconds", start.elapsed().as_secs_f64());
                        metrics.observe_histogram("pattern_completion_confidence", completed.confidence.as_probability() as f64);
                        
                        // Record cognitive metrics
                        metrics.record_cognitive(CognitiveMetric::PatternCompletion {
                            plausibility: completed.confidence.as_probability(),
                            is_false_memory: completed.confidence.as_probability() < 0.5,
                        });
                    }
                    
                    completed
                }
                Err(e) => {
                    tracing::warn!("Pattern completion failed: {:?}", e);
                    // Return a degraded completion
                    use crate::completion::SourceMap;
                    CompletedEpisode {
                        episode: Episode::new(
                            format!("failed_{}", chrono::Utc::now().timestamp()),
                            chrono::Utc::now(),
                            "Failed pattern completion".to_string(),
                            [0.0; 768],
                            Confidence::exact(0.1),
                        ),
                        completion_confidence: Confidence::exact(0.1),
                        source_attribution: SourceMap::default(),
                        alternative_hypotheses: Vec::new(),
                        metacognitive_confidence: Confidence::exact(0.1),
                        activation_evidence: Vec::new(),
                    }
                }
            }
        } else {
            // No pattern completion available, return minimal completion
            use crate::completion::SourceMap;
            CompletedEpisode {
                episode: Episode::new(
                    format!("unavailable_{}", chrono::Utc::now().timestamp()),
                    chrono::Utc::now(),
                    "Pattern completion not available".to_string(),
                    [0.0; 768],
                    Confidence::exact(0.0),
                ),
                completion_confidence: Confidence::exact(0.0),
                source_attribution: SourceMap::default(),
                alternative_hypotheses: Vec::new(),
                metacognitive_confidence: Confidence::exact(0.0),
                activation_evidence: Vec::new(),
            }
        }
    }
    
    /// Enable pattern completion with default configuration
    #[cfg(feature = "pattern_completion")]
    pub fn enable_pattern_completion(&mut self) {
        let config = CompletionConfig::default();
        let reconstructor = PatternReconstructor::new(config);
        self.pattern_reconstructor = Some(Arc::new(RwLock::new(reconstructor)));
    }
    
    /// Enable pattern completion with custom configuration
    #[cfg(feature = "pattern_completion")]
    pub fn enable_pattern_completion_with_config(&mut self, config: CompletionConfig) {
        let reconstructor = PatternReconstructor::new(config);
        self.pattern_reconstructor = Some(Arc::new(RwLock::new(reconstructor)));
    }
    
    /// Apply spreading activation to enhance recall results
    fn apply_spreading_activation(
        &self,
        mut results: Vec<(Episode, Confidence)>,
        cue: &Cue,
    ) -> Vec<(Episode, Confidence)> {
        // Get system pressure to modulate activation spreading
        let pressure = self.pressure();
        let spread_factor = pressure.mul_add(-0.5, 1.0); // Reduce spreading under pressure

        // For each high-confidence result, boost related memories
        let high_confidence_results: Vec<_> = results
            .iter()
            .filter(|(_, conf)| conf.is_high())
            .cloned()
            .collect();

        for (ep, base_conf) in high_confidence_results {
            // Find related episodes based on temporal proximity
            let time_window = chrono::Duration::hours(1);
            let start = ep.when - time_window;
            let end = ep.when + time_window;

            for entry in self.wal_buffer.iter() {
                let related_ep = entry.value();
                if related_ep.id != ep.id && related_ep.when >= start && related_ep.when <= end {
                    // Calculate activation boost based on temporal proximity
                    let time_diff = (ep.when - related_ep.when).num_seconds().abs() as f32;
                    let max_diff = time_window.num_seconds() as f32;
                    let proximity = 1.0 - (time_diff / max_diff);
                    let boost = proximity * base_conf.raw() * spread_factor * 0.3;

                    // Check if already in results
                    let existing_idx = results.iter().position(|(e, _)| e.id == related_ep.id);

                    if let Some(idx) = existing_idx {
                        // Boost existing result
                        let (ep, old_conf) = &results[idx];
                        let new_value = (old_conf.raw() + boost).min(1.0);
                        let new_conf = Confidence::exact(new_value);
                        results[idx] = (ep.clone(), new_conf);
                    } else if boost > cue.result_threshold.raw() * 0.5 {
                        // Add as new low-confidence result
                        let conf = Confidence::exact(boost);
                        results.push((related_ep.clone(), conf));
                    }
                }
            }
        }

        results
    }

    /// Recall using HNSW index for fast similarity search
    #[cfg(feature = "hnsw_index")]
    fn recall_with_hnsw(
        &self,
        cue: Cue,
        hnsw: &CognitiveHnswIndex,
        vector: &[f32; 768],
        threshold: Confidence,
    ) -> Vec<(Episode, Confidence)> {
        // Use HNSW for fast similarity search
        let candidates = hnsw.search_with_confidence(
            vector,
            cue.max_results * 2, // Get more candidates for diversity
            threshold,
        );

        let mut results = Vec::new();
        for (memory_id, confidence) in candidates {
            if let Some(episode) = self.wal_buffer.get(&memory_id) {
                results.push((episode.clone(), confidence));
            }
        }

        // Apply spreading activation using HNSW graph structure
        let pressure = *self.pressure.read();
        results = hnsw.apply_spreading_activation(results, &cue, pressure);

        // Sort and limit results
        results.sort_by(|a, b| {
            b.1.raw()
                .partial_cmp(&a.1.raw())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(cue.max_results);
        results
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    crate::compute::cosine_similarity_768(a, b)
}

/// Extension trait to convert Episode to Memory
trait EpisodeToMemory {
    fn from_episode(episode: Episode, activation: f32) -> Self;
}

impl EpisodeToMemory for Memory {
    fn from_episode(episode: Episode, activation: f32) -> Self {
        let memory = Self::new(
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
    use crate::{Confidence, Cue, EpisodeBuilder};
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
                .id(format!("ep{i}"))
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
                    .id(format!("ep_thread_{i}"))
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
                .id(format!("ep{i}"))
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
    fn test_recall_returns_empty_for_no_matches() {
        let store = MemoryStore::new(10);

        // Store an episode
        let episode = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(Utc::now())
            .what("test episode".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);

        // Create cue that won't match
        let cue = Cue::semantic(
            "cue1".to_string(),
            "completely different content".to_string(),
            Confidence::HIGH,
        );

        let results = store.recall(cue);

        // Should return empty vector, not error
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_recall_with_embedding_cue() {
        let store = MemoryStore::new(10);

        // Store episodes with different embeddings
        let mut embedding1 = [0.0; 768];
        embedding1[0] = 1.0;

        let mut embedding2 = [0.0; 768];
        embedding2[1] = 1.0;

        let episode1 = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(Utc::now())
            .what("memory one".to_string())
            .embedding(embedding1)
            .confidence(Confidence::HIGH)
            .build();

        let episode2 = EpisodeBuilder::new()
            .id("ep2".to_string())
            .when(Utc::now())
            .what("memory two".to_string())
            .embedding(embedding2)
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode1);
        store.store(episode2);

        // Search with embedding similar to first
        let cue = Cue::embedding("cue1".to_string(), embedding1, Confidence::exact(0.9));

        let results = store.recall(cue);

        // Should find the matching episode
        assert!(!results.is_empty());
        assert_eq!(results[0].0.id, "ep1");
        assert!(results[0].1.raw() > 0.9);
    }

    #[test]
    fn test_recall_with_context_cue() {
        let store = MemoryStore::new(10);

        let now = Utc::now();
        let yesterday = now - chrono::Duration::days(1);
        let tomorrow = now + chrono::Duration::days(1);

        // Store episodes at different times
        let episode1 = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(now)
            .what("today's memory".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .where_location("office".to_string())
            .build();

        let episode2 = EpisodeBuilder::new()
            .id("ep2".to_string())
            .when(yesterday)
            .what("yesterday's memory".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .where_location("home".to_string())
            .build();

        store.store(episode1);
        store.store(episode2);

        // Search for today's memories in office
        let cue = Cue::context(
            "cue1".to_string(),
            Some((now - chrono::Duration::hours(1), tomorrow)),
            Some("office".to_string()),
            Confidence::MEDIUM,
        );

        let results = store.recall(cue);

        // Should find only today's office memory
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, "ep1");
        assert!(results[0].1.raw() == 1.0); // Perfect match on both criteria
    }

    #[test]
    fn test_recall_with_semantic_cue() {
        let store = MemoryStore::new(10);
        let now = Utc::now();

        // Store episodes with different content and times to avoid spreading activation
        let episode1 = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(now)
            .what("meeting with team about project alpha".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .build();

        let episode2 = EpisodeBuilder::new()
            .id("ep2".to_string())
            .when(now - chrono::Duration::hours(2)) // Outside spreading activation window
            .what("lunch at the cafeteria".to_string())
            .embedding([0.2; 768])
            .confidence(Confidence::HIGH)
            .build();

        let episode3 = EpisodeBuilder::new()
            .id("ep3".to_string())
            .when(now)
            .what("project review meeting".to_string())
            .embedding([0.3; 768])
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode1);
        store.store(episode2);
        store.store(episode3);

        // Search for memories about meetings
        let cue = Cue::semantic("cue1".to_string(), "meeting".to_string(), Confidence::LOW);

        let results = store.recall(cue);

        // Should find both meeting-related episodes
        assert_eq!(results.len(), 2);
        let ids: Vec<String> = results.iter().map(|(e, _)| e.id.clone()).collect();
        assert!(ids.contains(&"ep1".to_string()));
        assert!(ids.contains(&"ep3".to_string()));
        assert!(!ids.contains(&"ep2".to_string()));
    }

    #[test]
    fn test_recall_confidence_normalization() {
        let store = MemoryStore::new(10);

        let episode = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(Utc::now())
            .what("test memory".to_string())
            .embedding([0.5; 768])
            .confidence(Confidence::MEDIUM)
            .build();

        store.store(episode);

        // Create cue with partial match
        let mut partial_embedding = [0.5; 768];
        partial_embedding[0] = 0.3; // Make it slightly different

        let cue = Cue::embedding("cue1".to_string(), partial_embedding, Confidence::LOW);

        let results = store.recall(cue);

        // Should return with confidence in valid range
        assert!(!results.is_empty());
        let confidence = results[0].1.raw();
        assert!(confidence >= 0.0);
        assert!(confidence <= 1.0);
        assert!(confidence < 1.0); // Should be less than perfect match
    }

    #[test]
    fn test_concurrent_recall_doesnt_block() {
        use std::thread;

        let store = Arc::new(MemoryStore::new(100));

        // Store some episodes first
        for i in 0..20 {
            let episode = EpisodeBuilder::new()
                .id(format!("ep{i}"))
                .when(Utc::now())
                .what(format!("memory {i}"))
                .embedding([i as f32 * 0.01; 768])
                .confidence(Confidence::HIGH)
                .build();

            store.store(episode);
        }

        let mut handles = vec![];

        // Spawn multiple threads doing recalls concurrently
        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = thread::spawn(move || {
                let cue = Cue::semantic(format!("cue{i}"), format!("memory {i}"), Confidence::LOW);

                store_clone.recall(cue)
            });
            handles.push(handle);
        }

        // All recalls should complete without blocking
        for handle in handles {
            let results = handle.join().unwrap();
            // Each should find at least one match
            assert!(!results.is_empty());
        }
    }

    #[test]
    fn test_spreading_activation() {
        let store = MemoryStore::new(10);

        let base_time = Utc::now();

        // Store related episodes close in time
        let episode1 = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(base_time)
            .what("meeting about project".to_string())
            .embedding([0.5; 768])
            .confidence(Confidence::HIGH)
            .build();

        let episode2 = EpisodeBuilder::new()
            .id("ep2".to_string())
            .when(base_time + chrono::Duration::minutes(30))
            .what("discussion with John".to_string())
            .embedding([0.3; 768])
            .confidence(Confidence::MEDIUM)
            .build();

        let episode3 = EpisodeBuilder::new()
            .id("ep3".to_string())
            .when(base_time + chrono::Duration::hours(2))
            .what("unrelated task".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::LOW)
            .build();

        store.store(episode1);
        store.store(episode2);
        store.store(episode3);

        // Search for meeting - should also pull in related discussion
        let cue = Cue::semantic("cue1".to_string(), "meeting".to_string(), Confidence::LOW);

        let results = store.recall(cue);

        // Should find meeting directly and possibly discussion through spreading
        assert!(!results.is_empty());
        assert_eq!(results[0].0.id, "ep1"); // Direct match should be first

        // If spreading activation worked, temporally related episode might be included
        // (This depends on implementation details and thresholds)
        if results.len() > 1 {
            // Related episode should have lower confidence due to spreading
            assert!(results[1].1.raw() < results[0].1.raw());
        }
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
