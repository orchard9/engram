//! Semantic deduplication for vector-based memories
//!
//! This module provides similarity-based deduplication to prevent storing
//! near-identical memories, with configurable merge strategies and thresholds.

use crate::{Confidence, Memory, compute};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Strategy for handling duplicate memories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeStrategy {
    /// Keep the memory with highest confidence
    #[default]
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
    #[must_use]
    pub fn new(similarity_threshold: f32, merge_strategy: MergeStrategy) -> Self {
        Self {
            similarity_threshold: similarity_threshold.clamp(0.0, 1.0),
            merge_strategy,
            stats: DeduplicationStats::default(),
            similarity_cache: HashMap::new(),
            max_cache_size: 1000,
        }
    }

    /// Check if a new memory is a duplicate of any existing memories
    #[must_use]
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
                let sim =
                    compute::cosine_similarity_768(&new_memory.embedding, &existing.embedding);

                // Cache the result
                if self.similarity_cache.len() < self.max_cache_size {
                    self.similarity_cache.insert(cache_key, sim);
                }

                sim
            };

            // Track best match
            if similarity >= self.similarity_threshold
                && match best_match {
                    None => true,
                    Some((_, best_sim)) => similarity > best_sim,
                }
            {
                best_match = Some((idx, similarity));
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
        // For exact duplicates with same ID, always skip
        if similarity > 0.999 && new_memory.id == existing.id {
            return DeduplicationAction::Skip;
        }

        // For high similarity, apply merge strategy
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
            MergeStrategy::MergeMetadata => DeduplicationAction::Merge(existing.id.clone()),
            MergeStrategy::CreateComposite => {
                DeduplicationAction::CreateComposite(existing.id.clone())
            }
        }
    }

    /// Merge two memories according to the merge strategy
    #[must_use]
    pub fn merge_memories(&self, memory1: &Memory, memory2: &Memory) -> Memory {
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
                if merged.content.is_none() {
                    merged.content.clone_from(&secondary.content);
                } else if let (Some(content1), Some(content2)) =
                    (&merged.content, &secondary.content)
                {
                    // Combine content strings
                    merged.content = Some(format!("{content1} | {content2}"));
                }

                // Use higher confidence
                merged.confidence =
                    Confidence::exact(memory1.confidence.raw().max(memory2.confidence.raw()));

                // Average activation
                merged.set_activation(f32::midpoint(memory1.activation(), memory2.activation()));

                merged
            }
            MergeStrategy::CreateComposite => {
                // Average the embeddings
                let mut composite_embedding = [0.0f32; 768];
                for (dest, (left, right)) in composite_embedding
                    .iter_mut()
                    .zip(memory1.embedding.iter().zip(&memory2.embedding))
                {
                    *dest = f32::midpoint(*left, *right);
                }

                // Normalize the composite embedding
                let norm = composite_embedding
                    .iter()
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
                composite.id = format!("{id1}_composite_{id2}", id1 = memory1.id, id2 = memory2.id);

                // Combine confidence with slight reduction for uncertainty
                composite.confidence = Confidence::exact(
                    f32::midpoint(memory1.confidence.raw(), memory2.confidence.raw()) * 0.95,
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
    #[must_use]
    pub fn stats(&self) -> DeduplicationStats {
        self.stats.clone()
    }

    /// Set the similarity threshold
    pub const fn set_threshold(&mut self, threshold: f32) {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set the merge strategy
    pub const fn set_strategy(&mut self, strategy: MergeStrategy) {
        self.merge_strategy = strategy;
    }
}

impl Default for SemanticDeduplicator {
    fn default() -> Self {
        Self::new(0.95, MergeStrategy::default())
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
    #[must_use]
    pub fn deduplication_rate(&self) -> f32 {
        let total = self.unique_memories.load(Ordering::Relaxed)
            + self.duplicates_found.load(Ordering::Relaxed);

        if total == 0 {
            0.0
        } else {
            let duplicates = self.duplicates_found.load(Ordering::Relaxed);
            match (usize_to_f32(duplicates), usize_to_f32(total)) {
                (Some(duplicates_f32), Some(total_f32)) if total_f32 > f32::EPSILON => {
                    (duplicates_f32 / total_f32) * 100.0
                }
                _ => 0.0,
            }
        }
    }
}

#[inline]
#[must_use]
const fn usize_to_f32(value: usize) -> Option<f32> {
    if value <= u32::MAX as usize {
        #[allow(clippy::cast_precision_loss)]
        let truncated = value as f32;
        Some(truncated)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult<T = ()> = Result<T, String>;
    use crate::MemoryBuilder;
    fn ensure(condition: bool, message: impl Into<String>) -> TestResult {
        if condition {
            Ok(())
        } else {
            Err(message.into())
        }
    }

    fn create_test_memory(id: &str, embedding_val: f32, confidence: f32) -> Memory {
        let mut embedding = [0.0f32; 768];

        // Create orthogonal embeddings using different basis vectors
        // This ensures different embeddings have low cosine similarity
        match id {
            "mem1" => {
                // First basis vector: concentrated in first third
                for (i, value) in embedding.iter_mut().enumerate().take(256) {
                    if let Ok(idx) = u32::try_from(i) {
                        #[allow(clippy::cast_precision_loss)]
                        let idx = idx as f32;
                        *value = ((idx + embedding_val) * 0.01).sin();
                    }
                }
            }
            "mem2" => {
                // Second basis vector: concentrated in middle third
                for (i, value) in embedding.iter_mut().enumerate().take(512).skip(256) {
                    if let Ok(idx) = u32::try_from(i) {
                        #[allow(clippy::cast_precision_loss)]
                        let idx = idx as f32;
                        *value = ((idx + embedding_val) * 0.01).cos();
                    }
                }
            }
            _ => {
                // Third basis vector: concentrated in last third
                for (i, value) in embedding.iter_mut().enumerate().skip(512) {
                    if let Ok(idx) = u32::try_from(i) {
                        #[allow(clippy::cast_precision_loss)]
                        let idx = idx as f32;
                        *value = ((idx * embedding_val) * 0.01).sin() * 0.5;
                    }
                }
            }
        }

        // Normalize to unit vector
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        MemoryBuilder::new()
            .id(id.to_string())
            .embedding(embedding)
            .confidence(Confidence::exact(confidence))
            .content(format!("Test memory {id}"))
            .build()
    }

    #[test]
    fn test_exact_duplicate_detection() {
        let mut dedup = SemanticDeduplicator::new(0.95, MergeStrategy::KeepHighestConfidence);

        let memory1 = create_test_memory("mem1", 0.5, 0.8);
        // Create actual duplicate with same embedding but different confidence
        let mut memory2 = memory1.clone();
        memory2.id = "mem2".to_string();
        memory2.confidence = Confidence::exact(0.9);

        let existing = vec![Arc::new(memory1)];
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
        // Create similar memory with same embedding but higher confidence
        let mut memory2 = memory1.clone();
        memory2.id = "mem2".to_string();
        memory2.confidence = Confidence::exact(0.9);

        let existing = vec![Arc::new(memory1)];
        let result = dedup.check_duplicate(&memory2, &existing);

        assert!(result.is_duplicate);
        assert!(
            matches!(result.action, DeduplicationAction::Replace(_)),
            "expected Replace action for higher confidence"
        );
    }

    #[test]
    fn test_merge_memories() -> TestResult {
        let dedup = SemanticDeduplicator::new(0.95, MergeStrategy::MergeMetadata);

        let mut memory1 = create_test_memory("mem1", 0.5, 0.7);
        memory1.content = Some("Content 1".to_string());

        let mut memory2 = create_test_memory("mem2", 0.6, 0.9);
        memory2.content = Some("Content 2".to_string());

        let merged = dedup.merge_memories(&memory1, &memory2);

        ensure(
            (merged.confidence.raw() - 0.9).abs() < 0.001,
            "merged confidence should prioritize higher value",
        )?;
        let content = merged
            .content
            .as_ref()
            .ok_or_else(|| "expected merged memory to retain content".to_string())?;
        ensure(
            content.contains("Content 1"),
            "merged content should include original",
        )?;

        Ok(())
    }

    #[test]
    fn test_composite_creation() {
        let dedup = SemanticDeduplicator::new(0.95, MergeStrategy::CreateComposite);

        // Create two different embeddings for compositing
        let mut embedding1 = [0.0f32; 768];
        let mut embedding2 = [0.0f32; 768];
        embedding1[0] = 0.4;
        embedding2[0] = 0.6;

        let memory1 = MemoryBuilder::new()
            .id("mem1".to_string())
            .embedding(embedding1)
            .confidence(Confidence::exact(0.8))
            .content("Memory 1".to_string())
            .build();

        let memory2 = MemoryBuilder::new()
            .id("mem2".to_string())
            .embedding(embedding2)
            .confidence(Confidence::exact(0.8))
            .content("Memory 2".to_string())
            .build();

        let composite = dedup.merge_memories(&memory1, &memory2);

        // Composite embedding should be average and normalized
        // After averaging (0.4 + 0.6) / 2 = 0.5 and normalizing
        assert!(composite.id.contains("composite"));
        // Check that it was normalized (length should be 1)
        let norm: f32 = composite
            .embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Composite should be normalized");
    }

    #[test]
    fn test_deduplication_stats() {
        let mut dedup = SemanticDeduplicator::default();

        let memory1 = create_test_memory("mem1", 0.5, 0.8);
        // Create actual duplicate
        let mut memory2 = memory1.clone();
        memory2.id = "mem2".to_string();
        memory2.confidence = Confidence::exact(0.9);

        let memory3 = create_test_memory("mem3", 0.1, 0.7);

        let mut existing = vec![];

        // First memory is unique
        let _ = dedup.check_duplicate(&memory1, &existing);
        existing.push(Arc::new(memory1.clone()));

        // Second is duplicate
        let _ = dedup.check_duplicate(&memory2, &existing);

        // Third is unique
        let _ = dedup.check_duplicate(&memory3, &existing);

        let stats = dedup.stats();
        assert_eq!(stats.unique_memories.load(Ordering::Relaxed), 2);
        assert_eq!(stats.duplicates_found.load(Ordering::Relaxed), 1);
        assert!(stats.deduplication_rate() > 30.0);
    }
}
