//! Lock-free batch processing engine with work-stealing parallelism

use crate::{
    Activation, Confidence, Cue, Episode,
    store::MemoryStore,
    compute::{self, VectorOps},
};
use crate::batch::{
    BatchOperations, BatchConfig, BatchStoreResult, BatchRecallResult, 
    BatchSimilarityResult, BatchOperation, BatchOperationResult,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use rayon::prelude::*;
use crossbeam_queue::SegQueue;

/// High-performance batch processing engine
pub struct BatchEngine {
    /// Reference to memory store
    memory_store: Arc<MemoryStore>,
    /// SIMD vector operations
    vector_ops: &'static dyn VectorOps,
    /// Work queue for batch operations
    work_queue: Arc<SegQueue<BatchOperation>>,
    /// Metrics tracking
    metrics: Arc<BatchMetrics>,
}

#[derive(Debug)]
struct BatchMetrics {
    total_operations: AtomicU64,
    simd_operations: AtomicU64,
    total_processing_time_us: AtomicU64,
}

impl BatchEngine {
    /// Create a new batch processing engine with the given memory store
    pub fn new(memory_store: Arc<MemoryStore>) -> Self {
        Self {
            memory_store,
            vector_ops: compute::get_vector_ops(),
            work_queue: Arc::new(SegQueue::new()),
            metrics: Arc::new(BatchMetrics {
                total_operations: AtomicU64::new(0),
                simd_operations: AtomicU64::new(0),
                total_processing_time_us: AtomicU64::new(0),
            }),
        }
    }
    
    /// Process batch operations with parallel execution
    pub fn process_batch(&self, operations: Vec<BatchOperation>, config: &BatchConfig) -> Vec<BatchOperationResult> {
        
        let start = Instant::now();
        
        // Process operations in parallel using rayon
        let results: Vec<BatchOperationResult> = operations
            .into_par_iter()
            .map(|op| match op {
                BatchOperation::Store(episode) => {
                    let activation = self.memory_store.store(episode);
                    self.metrics.total_operations.fetch_add(1, Ordering::Relaxed);
                    BatchOperationResult::Store {
                        activation,
                        memory_id: format!("mem_{}", self.metrics.total_operations.load(Ordering::Relaxed)),
                    }
                },
                BatchOperation::Recall(cue) => {
                    let results = self.memory_store.recall(cue);
                    self.metrics.total_operations.fetch_add(1, Ordering::Relaxed);
                    BatchOperationResult::Recall(results)
                },
                BatchOperation::SimilaritySearch { embedding, k, threshold } => {
                    let results = self.batch_similarity_search_single(&embedding, k, threshold);
                    self.metrics.simd_operations.fetch_add(1, Ordering::Relaxed);
                    BatchOperationResult::SimilaritySearch(results)
                },
            })
            .collect();
        
        let elapsed = start.elapsed().as_micros() as u64;
        self.metrics.total_processing_time_us.fetch_add(elapsed, Ordering::Relaxed);
        
        results
    }
    
    /// SIMD-accelerated similarity search for a single query
    fn batch_similarity_search_single(
        &self,
        query: &[f32; 768],
        k: usize,
        threshold: Confidence,
    ) -> Vec<(String, Confidence)> {
        // Get all memory embeddings from the store
        let memories: Vec<_> = self.memory_store.hot_memories
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        
        if memories.is_empty() {
            return vec![];
        }
        
        // Extract embeddings for batch similarity computation
        let embeddings: Vec<[f32; 768]> = memories.iter()
            .map(|(_, mem)| mem.embedding)
            .collect();
        
        if embeddings.is_empty() {
            return vec![];
        }
        
        // Use SIMD batch similarity computation
        let similarities = self.vector_ops.cosine_similarity_batch_768(query, &embeddings);
        
        // Combine with memory IDs and filter by threshold
        let mut results: Vec<(String, Confidence)> = similarities
            .into_iter()
            .zip(memories.iter())
            .filter_map(|(sim, (id, _))| {
                let confidence = Confidence::from_raw(sim);
                if confidence >= threshold {
                    Some((id.clone(), confidence))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by confidence and take top k
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        
        results
    }
}

impl BatchOperations for BatchEngine {
    fn batch_store(&self, episodes: Vec<Episode>, config: BatchConfig) -> BatchStoreResult {
        let start = Instant::now();
        let initial_pressure = self.memory_store.pressure();
        
        // Process episodes in parallel
        let activations: Vec<Activation> = episodes
            .into_par_iter()
            .map(|episode| self.memory_store.store(episode))
            .collect();
        
        let successful_count = activations.iter().filter(|a| a.is_successful()).count();
        let degraded_count = activations.iter().filter(|a| a.is_degraded()).count();
        let peak_pressure = self.memory_store.pressure().max(initial_pressure);
        
        BatchStoreResult {
            activations,
            successful_count,
            degraded_count,
            processing_time_ms: start.elapsed().as_millis() as u64,
            peak_memory_pressure: peak_pressure,
        }
    }
    
    fn batch_recall(&self, cues: Vec<Cue>, config: BatchConfig) -> BatchRecallResult {
        let start = Instant::now();
        
        // Separate embedding cues for SIMD processing
        let (embedding_cues, other_cues): (Vec<_>, Vec<_>) = cues.into_iter()
            .partition(|cue| matches!(cue.cue_type, crate::CueType::Embedding { .. }));
        
        let mut results = Vec::new();
        let mut simd_accelerated_count = 0;
        
        // Process embedding cues with SIMD if available
        if !embedding_cues.is_empty() && config.use_simd {
            results.extend(self.batch_recall_embeddings(embedding_cues));
            simd_accelerated_count = results.len();
        }
        
        // Process other cue types in parallel
        let other_results: Vec<_> = other_cues
            .into_par_iter()
            .map(|cue| self.memory_store.recall(cue))
            .collect();
        
        results.extend(other_results);
        
        BatchRecallResult {
            results,
            processing_time_ms: start.elapsed().as_millis() as u64,
            simd_accelerated_count,
        }
    }
    
    fn batch_similarity_search(
        &self,
        query_embeddings: &[[f32; 768]],
        k: usize,
        threshold: Confidence,
    ) -> BatchSimilarityResult {
        let start = Instant::now();
        
        // Process each query in parallel
        let results: Vec<Vec<(String, Confidence)>> = query_embeddings
            .par_iter()
            .map(|query| self.batch_similarity_search_single(query, k, threshold))
            .collect();
        
        let simd_efficiency = if self.metrics.total_operations.load(Ordering::Relaxed) > 0 {
            self.metrics.simd_operations.load(Ordering::Relaxed) as f32 /
            self.metrics.total_operations.load(Ordering::Relaxed) as f32
        } else {
            0.0
        };
        
        BatchSimilarityResult {
            results,
            processing_time_ms: start.elapsed().as_millis() as u64,
            simd_efficiency,
        }
    }
}

impl BatchEngine {
    /// SIMD-accelerated batch recall for embedding cues
    fn batch_recall_embeddings(&self, cues: Vec<Cue>) -> Vec<Vec<(Episode, Confidence)>> {
        // Extract embeddings from cues
        let query_embeddings: Vec<[f32; 768]> = cues.iter()
            .filter_map(|cue| match &cue.cue_type {
                crate::CueType::Embedding { vector, .. } => {
                    if vector.len() == 768 {
                        let mut arr = [0.0f32; 768];
                        arr.copy_from_slice(vector);
                        Some(arr)
                    } else {
                        None
                    }
                },
                _ => None,
            })
            .collect();
        
        if query_embeddings.is_empty() {
            return vec![];
        }
        
        // Get all episodes for similarity comparison from WAL buffer
        let all_episodes: Vec<Episode> = self.memory_store.wal_buffer
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        
        // Process each query embedding
        query_embeddings.into_iter()
            .zip(cues.iter())
            .map(|(query_embedding, cue)| {
                // Extract embeddings from episodes
                let episode_embeddings: Vec<[f32; 768]> = all_episodes.iter()
                    .map(|ep| ep.embedding)
                    .collect();
                
                if episode_embeddings.is_empty() {
                    return vec![];
                }
                
                // SIMD batch similarity
                let similarities = self.vector_ops.cosine_similarity_batch_768(
                    &query_embedding,
                    &episode_embeddings
                );
                
                // Combine with episodes and filter
                let mut results: Vec<(Episode, Confidence)> = similarities
                    .into_iter()
                    .zip(all_episodes.iter())
                    .filter_map(|(sim, ep)| {
                        let confidence = Confidence::from_raw(sim);
                        if confidence >= cue.result_threshold {
                            Some((ep.clone(), confidence))
                        } else {
                            None
                        }
                    })
                    .collect();
                
                // Sort by confidence and limit results
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                results.truncate(cue.max_results);
                
                results
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_engine_creation() {
        let store = Arc::new(MemoryStore::default());
        let engine = BatchEngine::new(store);
        assert_eq!(engine.metrics.total_operations.load(Ordering::Relaxed), 0);
    }
}