//! Streaming batch processor with bounded memory usage

use crate::{
    Activation, Confidence, Cue, Episode,
    store::MemoryStore,
};
use crate::batch::{
    BatchConfig, BatchOperation, BatchOperationResult, BatchResult, 
    BatchMetadata, BackpressureStrategy, BatchError,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt};

/// Streaming batch processor with backpressure handling
pub struct StreamingBatchProcessor {
    /// Reference to memory store
    memory_store: Arc<MemoryStore>,
    /// Batch configuration
    config: BatchConfig,
    /// Current memory usage
    memory_usage: AtomicUsize,
}

impl StreamingBatchProcessor {
    /// Create a new streaming processor
    pub fn new(memory_store: Arc<MemoryStore>, config: BatchConfig) -> Self {
        Self {
            memory_store,
            config,
            memory_usage: AtomicUsize::new(0),
        }
    }
    
    /// Process a stream of batch operations with bounded memory
    pub async fn process_stream<S>(
        &self,
        mut operations: S,
    ) -> impl Stream<Item = BatchResult>
    where
        S: Stream<Item = BatchOperation> + Unpin,
    {
        let (tx, rx) = mpsc::channel(self.config.batch_size);
        let memory_store = Arc::clone(&self.memory_store);
        let config = self.config.clone();
        
        // Spawn processor task
        tokio::spawn(async move {
            let mut batch = Vec::with_capacity(config.batch_size);
            let mut operation_id = 0usize;
            
            while let Some(operation) = operations.next().await {
                batch.push((operation_id, operation));
                operation_id += 1;
                
                // Process batch when full or stream ends
                if batch.len() >= config.batch_size {
                    Self::process_batch_async(&memory_store, &batch, &config, &tx).await;
                    batch.clear();
                }
            }
            
            // Process remaining operations
            if !batch.is_empty() {
                Self::process_batch_async(&memory_store, &batch, &config, &tx).await;
            }
        });
        
        tokio_stream::wrappers::ReceiverStream::new(rx)
    }
    
    /// Process a batch asynchronously
    async fn process_batch_async(
        memory_store: &Arc<MemoryStore>,
        batch: &[(usize, BatchOperation)],
        config: &BatchConfig,
        tx: &mpsc::Sender<BatchResult>,
    ) {
        let start = Instant::now();
        
        for (id, operation) in batch {
            let op_start = Instant::now();
            let simd_used = config.use_simd && matches!(operation, BatchOperation::SimilaritySearch { .. });
            
            let result = match operation {
                BatchOperation::Store(episode) => {
                    let activation = memory_store.store(episode.clone());
                    BatchOperationResult::Store {
                        activation,
                        memory_id: format!("mem_{}", id),
                    }
                },
                BatchOperation::Recall(cue) => {
                    let results = memory_store.recall(cue.clone());
                    BatchOperationResult::Recall(results)
                },
                BatchOperation::SimilaritySearch { embedding, k, threshold } => {
                    let results = Self::similarity_search(memory_store, embedding, *k, *threshold);
                    BatchOperationResult::SimilaritySearch(results)
                },
            };
            
            let metadata = BatchMetadata {
                processing_time_us: op_start.elapsed().as_micros() as u64,
                simd_used,
                memory_pressure: memory_store.pressure(),
            };
            
            let batch_result = BatchResult {
                operation_id: *id,
                result,
                metadata,
            };
            
            // Send result, applying backpressure if needed
            if tx.send(batch_result).await.is_err() {
                // Receiver dropped, stop processing
                break;
            }
        }
    }
    
    /// Perform similarity search
    fn similarity_search(
        memory_store: &MemoryStore,
        query: &[f32; 768],
        k: usize,
        threshold: Confidence,
    ) -> Vec<(String, Confidence)> {
        use crate::compute;
        
        // Get all episodes
        let episodes: Vec<(String, Episode)> = memory_store.wal_buffer
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        
        if episodes.is_empty() {
            return vec![];
        }
        
        // Extract embeddings
        let embeddings: Vec<[f32; 768]> = episodes.iter()
            .map(|(_, ep)| ep.embedding)
            .collect();
        
        // Compute similarities using SIMD
        let similarities = compute::cosine_similarity_batch_768(query, &embeddings);
        
        // Filter and sort results
        let mut results: Vec<(String, Confidence)> = similarities
            .into_iter()
            .zip(episodes.iter())
            .filter_map(|(sim, (id, _))| {
                let confidence = Confidence::from_raw(sim);
                if confidence >= threshold {
                    Some((id.clone(), confidence))
                } else {
                    None
                }
            })
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        
        results
    }
    
    /// Apply backpressure strategy
    pub fn apply_backpressure(&self, current_size: usize) -> bool {
        match self.config.backpressure_strategy {
            BackpressureStrategy::DropOldest => {
                // Allow overflow, oldest will be dropped
                true
            },
            BackpressureStrategy::AdaptiveResize => {
                // Resize if under pressure
                let pressure = self.memory_store.pressure();
                current_size < (self.config.batch_size as f32 * (1.0 - pressure)) as usize
            },
            BackpressureStrategy::Block => {
                // Block until below limit
                current_size < self.config.batch_size
            },
            BackpressureStrategy::FallbackMode => {
                // Switch to single operation mode under pressure
                self.memory_store.pressure() < 0.8 || current_size == 0
            },
        }
    }
    
    /// Check memory limit
    pub fn check_memory_limit(&self) -> Result<(), BatchError> {
        let current = self.memory_usage.load(Ordering::Relaxed);
        let limit = self.config.memory_limit_mb * 1024 * 1024;
        
        if current > limit {
            Err(BatchError::MemoryLimitExceeded {
                current_mb: current / (1024 * 1024),
                limit_mb: self.config.memory_limit_mb,
            })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EpisodeBuilder, Cue};
    use chrono::Utc;
    use tokio_stream::StreamExt;
    
    #[tokio::test]
    async fn test_streaming_processor() {
        let store = Arc::new(MemoryStore::new(100));
        let config = BatchConfig {
            batch_size: 5,
            ..Default::default()
        };
        let processor = StreamingBatchProcessor::new(Arc::clone(&store), config);
        
        // Create stream of operations
        let operations: Vec<BatchOperation> = (0..10)
            .map(|i| {
                let episode = EpisodeBuilder::new()
                    .id(format!("ep{}", i))
                    .when(Utc::now())
                    .what(format!("Episode {}", i))
                    .embedding([i as f32 * 0.1; 768])
                    .confidence(Confidence::HIGH)
                    .build();
                BatchOperation::Store(episode)
            })
            .collect();
        
        let stream = tokio_stream::iter(operations);
        let mut result_stream = processor.process_stream(stream).await;
        
        let mut count = 0;
        while let Some(result) = result_stream.next().await {
            assert!(result.operation_id < 10);
            count += 1;
        }
        
        assert_eq!(count, 10);
        assert_eq!(store.count(), 10);
    }
}