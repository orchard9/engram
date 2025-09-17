//! Semantic deduplication for vector-based memories
//!
//! This module provides similarity-based deduplication to prevent storing
//! near-identical memories, with configurable merge strategies and thresholds.

use crate::{compute, Confidence, Memory};
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Strategy for handling duplicate memories
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MergeStrategy {
    /// Keep the memory with highest confidence
    KeepHighestConfidence,
    /// Merge metadata from both memories
    MergeMetadata,
    /// Create a composite memory from both
    CreateComposite,
    /// Keep the most recent memory
    KeepMostRecent,
    /// Keep the oldest memory
    KeepOldest,
}

impl Default for MergeStrategy {
    fn default() -> Self {
        Self::KeepHighestConfidence
    }
}

/// Result of a deduplication check
#[derive(Debug, Clone)]
pub struct DeduplicationResult {
    /// Whether a duplicate was found
    pub is_duplicate: bool,
    /// ID of the existing duplicate memory if found
    pub existing_id: Option<String>,
    /// Similarity score if duplicate found
    pub similarity: f32,
    /// Suggested action based on merge strategy
    pub action: DeduplicationAction,
}

/// Action to take based on deduplication result
#[derive(Debug, Clone)]
pub enum DeduplicationAction {
    /// Store as new memory (no duplicate)
    StoreNew,
    /// Skip storing (exact duplicate)
    Skip,
    /// Replace existing memory with new one
    Replace(String),
    /// Merge with existing memory
    Merge(String),
    /// Create composite from both
    CreateComposite(String),
}

/// Semantic deduplicator for memory storage
pub struct SemanticDeduplicator {
    /// Similarity threshold for considering memories duplicates
    similarity_threshold: f32,
    
    /// Strategy for handling duplicates
    merge_strategy: MergeStrategy,
    
    /// Statistics tracking
    stats: DeduplicationStats,
    
    /// Cache of recent similarity comparisons for efficiency
    similarity_cache: HashMap<(String, String), f32>,
    
    /// Maximum cache size
    max_cache_size: usize,
}

impl SemanticDeduplicator {
    /// Create a new deduplicator with specified threshold
    pub fn new(similarity_threshold: f32, merge_strategy: MergeStrategy) -> Self {
        Self {
            similarity_threshold: similarity_threshold.clamp(0.0, 1.0),
            merge_strategy,
            stats: DeduplicationStats::default(),
            similarity_cache: HashMap::new(),
            max_cache_size: 1000,
        }
    }
    
    /// Create with default settings (0.95 threshold, keep highest confidence)
    pub fn default() -> Self {
        Self::new(0.95, MergeStrategy::default())
    }
    
    /// Check if a new memory is a duplicate of any existing memories
    pub fn check_duplicate(
        &mut self,
        new_memory: &Memory,
        existing_memories: &[Arc<Memory>],
    ) -> DeduplicationResult {
        let mut best_match: Option<(usize, f32)> = None;
        
        for (idx, existing) in existing_memories.iter().enumerate() {
            // Check cache first
            let cache_key = (new_memory.id.clone(), existing.id.clone());
            let similarity = if let Some(&cached) = self.similarity_cache.get(&cache_key) {
                cached
            } else {
                // Compute similarity using SIMD-optimized function
                let sim = compute::cosine_similarity_768(
                    &new_memory.embedding,
                    &existing.embedding,
                );
                
                // Cache the result
                if self.similarity_cache.len() < self.max_cache_size {
                    self.similarity_cache.insert(cache_key, sim);
                }
                
                sim
            };
            
            // Track best match
            if similarity >= self.similarity_threshold {
                if best_match.is_none() || similarity > best_match.unwrap().1 {
                    best_match = Some((idx, similarity));
                }
            }
        }
        
        // Handle the best match if found
        if let Some((idx, similarity)) = best_match {
            let existing = &existing_memories[idx];
            self.stats.duplicates_found.fetch_add(1, Ordering::Relaxed);
            
            let action = self.determine_action(new_memory, existing, similarity);
            
            DeduplicationResult {
                is_duplicate: true,
                existing_id: Some(existing.id.clone()),
                similarity,
                action,
            }
        } else {
            self.stats.unique_memories.fetch_add(1, Ordering::Relaxed);
            
            DeduplicationResult {
                is_duplicate: false,
                existing_id: None,
                similarity: 0.0,
                action: DeduplicationAction::StoreNew,
            }
        }
    }
    
    /// Determine what action to take for a duplicate
    fn determine_action(
        &self,
        new_memory: &Memory,
        existing: &Memory,
        similarity: f32,
    ) -> DeduplicationAction {
        // Exact duplicate (>0.99 similarity)
        if similarity > 0.99 {
            return DeduplicationAction::Skip;
        }
        
        match self.merge_strategy {
            MergeStrategy::KeepHighestConfidence => {
                if new_memory.confidence.raw() > existing.confidence.raw() {
                    DeduplicationAction::Replace(existing.id.clone())
                } else {
                    DeduplicationAction::Skip
                }
            }
            MergeStrategy::KeepMostRecent => {
                if new_memory.created_at > existing.created_at {
                    DeduplicationAction::Replace(existing.id.clone())
                } else {
                    DeduplicationAction::Skip
                }
            }
            MergeStrategy::KeepOldest => {
                if new_memory.created_at < existing.created_at {
                    DeduplicationAction::Replace(existing.id.clone())
                } else {
                    DeduplicationAction::Skip
                }
            }
            MergeStrategy::MergeMetadata => {
                DeduplicationAction::Merge(existing.id.clone())
            }
            MergeStrategy::CreateComposite => {
                DeduplicationAction::CreateComposite(existing.id.clone())
            }
        }
    }
    
    /// Merge two memories according to the merge strategy
    pub fn merge_memories(
        &self,
        memory1: &Memory,
        memory2: &Memory,
    ) -> Memory {
        match self.merge_strategy {
            MergeStrategy::MergeMetadata => {
                // Keep embedding from higher confidence memory
                let (primary, secondary) = if memory1.confidence.raw() > memory2.confidence.raw() {
                    (memory1, memory2)
                } else {
                    (memory2, memory1)
                };
                
                let mut merged = primary.clone();
                
                // Merge content if available
                if merged.content.is_none() && secondary.content.is_some() {
                    merged.content = secondary.content.clone();
                } else if let (Some(content1), Some(content2)) = 
                    (&merged.content, &secondary.content) {
                    // Combine content strings
                    merged.content = Some(format!("{} | {}", content1, content2));
                }
                
                // Use higher confidence
                merged.confidence = Confidence::exact(
                    memory1.confidence.raw().max(memory2.confidence.raw())
                );
                
                // Average activation
                merged.set_activation(
                    (memory1.activation() + memory2.activation()) / 2.0
                );
                
                merged
            }
            MergeStrategy::CreateComposite => {
                // Average the embeddings
                let mut composite_embedding = [0.0f32; 768];
                for i in 0..768 {
                    composite_embedding[i] = 
                        (memory1.embedding[i] + memory2.embedding[i]) / 2.0;
                }
                
                // Normalize the composite embedding
                let norm = composite_embedding.iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();
                
                if norm > 0.0 {
                    for val in &mut composite_embedding {
                        *val /= norm;
                    }
                }
                
                let mut composite = memory1.clone();
                composite.embedding = composite_embedding;
                composite.id = format!("{}_composite_{}", memory1.id, memory2.id);
                
                // Combine confidence with slight reduction for uncertainty
                composite.confidence = Confidence::exact(
                    (memory1.confidence.raw() + memory2.confidence.raw()) / 2.0 * 0.95
                );
                
                composite
            }
            _ => {
                // For other strategies, just return the preferred memory
                if memory1.confidence.raw() > memory2.confidence.raw() {
                    memory1.clone()
                } else {
                    memory2.clone()
                }
            }
        }
    }
    
    /// Clear the similarity cache
    pub fn clear_cache(&mut self) {
        self.similarity_cache.clear();
    }
    
    /// Get deduplication statistics
    pub fn stats(&self) -> DeduplicationStats {
        self.stats.clone()
    }
    
    /// Set the similarity threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
    }
    
    /// Set the merge strategy
    pub fn set_strategy(&mut self, strategy: MergeStrategy) {
        self.merge_strategy = strategy;
    }
}

/// Statistics for deduplication monitoring
#[derive(Debug, Default)]
pub struct DeduplicationStats {
    /// Number of unique memories processed
    pub unique_memories: AtomicUsize,
    
    /// Number of duplicates found
    pub duplicates_found: AtomicUsize,
    
    /// Number of near-duplicates (similarity > 0.9 but < threshold)
    pub near_duplicates: AtomicUsize,
    
    /// Number of memories merged
    pub memories_merged: AtomicUsize,
    
    /// Number of memories replaced
    pub memories_replaced: AtomicUsize,
}

impl Clone for DeduplicationStats {
    fn clone(&self) -> Self {
        Self {
            unique_memories: AtomicUsize::new(self.unique_memories.load(Ordering::Relaxed)),
            duplicates_found: AtomicUsize::new(self.duplicates_found.load(Ordering::Relaxed)),
            near_duplicates: AtomicUsize::new(self.near_duplicates.load(Ordering::Relaxed)),
            memories_merged: AtomicUsize::new(self.memories_merged.load(Ordering::Relaxed)),
            memories_replaced: AtomicUsize::new(self.memories_replaced.load(Ordering::Relaxed)),
        }
    }
}

impl DeduplicationStats {
    /// Get the deduplication rate as a percentage
    pub fn deduplication_rate(&self) -> f32 {
        let total = self.unique_memories.load(Ordering::Relaxed) 
            + self.duplicates_found.load(Ordering::Relaxed);
        
        if total == 0 {
            0.0
        } else {
            (self.duplicates_found.load(Ordering::Relaxed) as f32 / total as f32) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryBuilder;
    use chrono::Utc;
    
    fn create_test_memory(id: &str, embedding_val: f32, confidence: f32) -> Memory {
        let mut embedding = [0.0f32; 768];
        embedding[0] = embedding_val;
        
        MemoryBuilder::new()
            .id(id.to_string())
            .embedding(embedding)
            .confidence(Confidence::exact(confidence))
            .content(format!("Test memory {}", id))
            .build()
    }
    
    #[test]
    fn test_exact_duplicate_detection() {
        let mut dedup = SemanticDeduplicator::new(0.95, MergeStrategy::KeepHighestConfidence);
        
        let memory1 = create_test_memory("mem1", 0.5, 0.8);
        let memory2 = create_test_memory("mem2", 0.5, 0.9);
        
        let existing = vec![Arc::new(memory1.clone())];
        let result = dedup.check_duplicate(&memory2, &existing);
        
        assert!(result.is_duplicate);
        assert_eq!(result.existing_id, Some("mem1".to_string()));
        assert!(result.similarity > 0.99);
    }
    
    #[test]
    fn test_non_duplicate() {
        let mut dedup = SemanticDeduplicator::new(0.95, MergeStrategy::KeepHighestConfidence);
        
        let memory1 = create_test_memory("mem1", 0.1, 0.8);
        let memory2 = create_test_memory("mem2", 0.9, 0.9);
        
        let existing = vec![Arc::new(memory1)];
        let result = dedup.check_duplicate(&memory2, &existing);
        
        assert!(!result.is_duplicate);
        assert_eq!(result.existing_id, None);
    }
    
    #[test]
    fn test_merge_strategy_highest_confidence() {
        let mut dedup = SemanticDeduplicator::new(0.95, MergeStrategy::KeepHighestConfidence);
        
        let memory1 = create_test_memory("mem1", 0.5, 0.7);
        let memory2 = create_test_memory("mem2", 0.5, 0.9);
        
        let existing = vec![Arc::new(memory1)];
        let result = dedup.check_duplicate(&memory2, &existing);
        
        assert!(result.is_duplicate);
        match result.action {
            DeduplicationAction::Replace(_) => (),
            _ => panic!("Expected Replace action for higher confidence"),
        }
    }
    
    #[test]
    fn test_merge_memories() {
        let dedup = SemanticDeduplicator::new(0.95, MergeStrategy::MergeMetadata);
        
        let mut memory1 = create_test_memory("mem1", 0.5, 0.7);
        memory1.content = Some("Content 1".to_string());
        
        let mut memory2 = create_test_memory("mem2", 0.6, 0.9);
        memory2.content = Some("Content 2".to_string());
        
        let merged = dedup.merge_memories(&memory1, &memory2);
        
        assert_eq!(merged.confidence.raw(), 0.9);
        assert!(merged.content.unwrap().contains("Content 1"));
    }
    
    #[test]
    fn test_composite_creation() {
        let dedup = SemanticDeduplicator::new(0.95, MergeStrategy::CreateComposite);
        
        let memory1 = create_test_memory("mem1", 0.4, 0.8);
        let memory2 = create_test_memory("mem2", 0.6, 0.8);
        
        let composite = dedup.merge_memories(&memory1, &memory2);
        
        // Composite embedding should be average and normalized
        assert!(composite.embedding[0] > 0.4 && composite.embedding[0] < 0.6);
        assert!(composite.id.contains("composite"));
    }
    
    #[test]
    fn test_deduplication_stats() {
        let mut dedup = SemanticDeduplicator::default();
        
        let memory1 = create_test_memory("mem1", 0.5, 0.8);
        let memory2 = create_test_memory("mem2", 0.5, 0.9);
        let memory3 = create_test_memory("mem3", 0.1, 0.7);
        
        let mut existing = vec![];
        
        // First memory is unique
        dedup.check_duplicate(&memory1, &existing);
        existing.push(Arc::new(memory1));
        
        // Second is duplicate
        dedup.check_duplicate(&memory2, &existing);
        
        // Third is unique
        dedup.check_duplicate(&memory3, &existing);
        
        let stats = dedup.stats();
        assert_eq!(stats.unique_memories.load(Ordering::Relaxed), 2);
        assert_eq!(stats.duplicates_found.load(Ordering::Relaxed), 1);
        assert!(stats.deduplication_rate() > 30.0);
    }
}