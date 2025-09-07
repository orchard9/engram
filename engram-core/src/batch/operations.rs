//! Core batch operations implementation

use crate::{
    Activation, Confidence, Cue, Episode,
    store::MemoryStore,
};
use crate::batch::{
    BatchOperations, BatchConfig, BatchStoreResult, BatchRecallResult,
    BatchSimilarityResult,
};
use std::time::Instant;
use rayon::prelude::*;

/// Extension trait to add batch operations to MemoryStore
impl BatchOperations for MemoryStore {
    fn batch_store(&self, episodes: Vec<Episode>, _config: BatchConfig) -> BatchStoreResult {
        let start = Instant::now();
        let initial_pressure = self.pressure();
        
        // Process episodes in parallel while maintaining cognitive semantics
        let activations: Vec<Activation> = episodes
            .into_par_iter()
            .map(|episode| {
                self.store(episode)
            })
            .collect();
        
        // Get peak pressure after all operations
        let peak_pressure = self.pressure();
        
        let successful_count = activations.iter().filter(|a| a.is_successful()).count();
        let degraded_count = activations.iter().filter(|a| a.is_degraded()).count();
        
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
        
        // Separate embedding cues for potential SIMD optimization
        let (embedding_cues, other_cues): (Vec<_>, Vec<_>) = cues.into_iter()
            .partition(|cue| matches!(cue.cue_type, crate::CueType::Embedding { .. }));
        
        let mut results = Vec::new();
        let mut simd_accelerated_count = 0;
        
        // Process embedding cues with SIMD if available and beneficial
        if !embedding_cues.is_empty() && config.use_simd {
            let embedding_results = self.batch_recall_embeddings_simd(embedding_cues, &config);
            simd_accelerated_count = embedding_results.len();
            results.extend(embedding_results);
        } else if !embedding_cues.is_empty() {
            // Process embedding cues without SIMD
            let embedding_results: Vec<_> = embedding_cues
                .into_par_iter()
                .map(|cue| self.recall(cue))
                .collect();
            results.extend(embedding_results);
        }
        
        // Process other cue types in parallel
        if !other_cues.is_empty() {
            let other_results: Vec<_> = other_cues
                .into_par_iter()
                .map(|cue| self.recall(cue))
                .collect();
            results.extend(other_results);
        }
        
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
            .map(|query| self.similarity_search_single(query, k, threshold))
            .collect();
        
        // Calculate SIMD efficiency based on whether we used vector operations
        let simd_efficiency = if cfg!(feature = "force_scalar_compute") {
            0.0
        } else {
            match crate::compute::detect_cpu_features() {
                #[cfg(target_arch = "x86_64")]
                crate::compute::CpuCapability::Avx512F => 1.0,
                #[cfg(target_arch = "x86_64")]
                crate::compute::CpuCapability::Avx2Fma | crate::compute::CpuCapability::Avx2 => 0.8,
                _ => 0.2,
            }
        };
        
        BatchSimilarityResult {
            results,
            processing_time_ms: start.elapsed().as_millis() as u64,
            simd_efficiency,
        }
    }
}

impl MemoryStore {
    /// SIMD-optimized batch recall for embedding cues
    fn batch_recall_embeddings_simd(&self, cues: Vec<Cue>, _config: &BatchConfig) -> Vec<Vec<(Episode, Confidence)>> {
        use crate::compute;
        
        // Extract query embeddings
        let query_embeddings: Vec<[f32; 768]> = cues.iter()
            .filter_map(|cue| match &cue.cue_type {
                crate::CueType::Embedding { vector, .. } => Some(*vector),
                _ => None,
            })
            .collect();
        
        if query_embeddings.is_empty() {
            return vec![];
        }
        
        // Get all episodes from WAL buffer for comparison
        let all_episodes: Vec<Episode> = self.wal_buffer
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        
        if all_episodes.is_empty() {
            return vec![vec![]; cues.len()];
        }
        
        // Extract episode embeddings
        let episode_embeddings: Vec<[f32; 768]> = all_episodes.iter()
            .map(|ep| ep.embedding)
            .collect();
        
        // Process each query
        query_embeddings.into_iter()
            .zip(cues.iter())
            .map(|(query_embedding, cue)| {
                // Use SIMD batch similarity computation
                let similarities = compute::cosine_similarity_batch_768(&query_embedding, &episode_embeddings);
                
                // Get threshold from cue
                let threshold = match &cue.cue_type {
                    crate::CueType::Embedding { threshold, .. } => *threshold,
                    _ => Confidence::LOW,
                };
                
                // Combine with episodes and filter
                let mut results: Vec<(Episode, Confidence)> = similarities
                    .into_iter()
                    .zip(all_episodes.iter())
                    .filter_map(|(sim, ep)| {
                        let confidence = Confidence::from_raw(sim);
                        if confidence >= threshold {
                            Some((ep.clone(), confidence))
                        } else {
                            None
                        }
                    })
                    .collect();
                
                // Sort by confidence and limit
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                results.truncate(cue.max_results);
                
                results
            })
            .collect()
    }
    
    /// Single similarity search operation
    fn similarity_search_single(
        &self,
        query: &[f32; 768],
        k: usize,
        threshold: Confidence,
    ) -> Vec<(String, Confidence)> {
        use crate::compute;
        
        // Get all episodes with their IDs
        let episodes: Vec<(String, Episode)> = self.wal_buffer
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        
        if episodes.is_empty() {
            return vec![];
        }
        
        // Extract embeddings for batch similarity
        let embeddings: Vec<[f32; 768]> = episodes.iter()
            .map(|(_, ep)| ep.embedding)
            .collect();
        
        // Compute similarities using SIMD
        let similarities = compute::cosine_similarity_batch_768(query, &embeddings);
        
        // Combine with IDs and filter by threshold
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
        
        // Sort by confidence and take top k
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EpisodeBuilder, Cue};
    use chrono::Utc;
    
    #[test]
    fn test_batch_store_maintains_semantics() {
        let store = MemoryStore::new(100);
        
        // Create batch of episodes
        let episodes: Vec<Episode> = (0..10)
            .map(|i| {
                EpisodeBuilder::new()
                    .id(format!("ep{}", i))
                    .when(Utc::now())
                    .what(format!("Episode {}", i))
                    .embedding([i as f32 * 0.1; 768])
                    .confidence(Confidence::HIGH)
                    .build()
            })
            .collect();
        
        // Batch store
        let result = store.batch_store(episodes.clone(), BatchConfig::default());
        
        // Verify all episodes stored
        assert_eq!(result.activations.len(), 10);
        assert_eq!(result.successful_count, 10);
        assert_eq!(result.degraded_count, 0);
        
        // Verify cognitive semantics preserved
        for activation in &result.activations {
            assert!(activation.is_successful());
            assert!(!activation.is_degraded());
        }
    }
    
    #[test]
    fn test_batch_recall_with_mixed_cues() {
        let store = MemoryStore::new(100);
        
        // Store some episodes
        for i in 0..5 {
            let episode = EpisodeBuilder::new()
                .id(format!("ep{}", i))
                .when(Utc::now())
                .what(format!("Memory content {}", i))
                .embedding([i as f32 * 0.2; 768])
                .confidence(Confidence::HIGH)
                .build();
            store.store(episode);
        }
        
        // Create mixed cues
        let cues = vec![
            Cue::embedding("cue1".to_string(), [0.0; 768], Confidence::LOW),
            Cue::semantic("cue2".to_string(), "Memory content 2".to_string(), Confidence::LOW),
        ];
        
        // Batch recall
        let result = store.batch_recall(cues, BatchConfig::default());
        
        // Verify results
        assert_eq!(result.results.len(), 2);
        assert!(result.results[0].len() > 0); // Embedding cue should find matches
        assert!(result.results[1].len() > 0); // Semantic cue should find exact match
    }
    
    #[test]
    fn test_batch_similarity_search() {
        let store = MemoryStore::new(100);
        
        // Store episodes with distinct embeddings
        for i in 0..10 {
            let mut embedding = [0.0; 768];
            embedding[i] = 1.0;
            
            let episode = EpisodeBuilder::new()
                .id(format!("ep{}", i))
                .when(Utc::now())
                .what(format!("Memory {}", i))
                .embedding(embedding)
                .confidence(Confidence::HIGH)
                .build();
            store.store(episode);
        }
        
        // Create query embeddings
        let mut query1 = [0.0; 768];
        query1[0] = 1.0;
        let mut query2 = [0.0; 768];
        query2[5] = 1.0;
        
        let queries = vec![query1, query2];
        
        // Batch similarity search
        let result = store.batch_similarity_search(&queries, 3, Confidence::LOW);
        
        // Verify results
        assert_eq!(result.results.len(), 2);
        assert!(result.results[0].len() <= 3);
        assert!(result.results[1].len() <= 3);
        
        // First query should match ep0 best
        if !result.results[0].is_empty() {
            assert_eq!(result.results[0][0].0, "ep0");
        }
        
        // Second query should match ep5 best
        if !result.results[1].is_empty() {
            assert_eq!(result.results[1][0].0, "ep5");
        }
    }
}