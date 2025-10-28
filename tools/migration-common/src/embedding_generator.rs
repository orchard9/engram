//! Embedding generation with caching for migration

use crate::error::MigrationResult;
use dashmap::DashMap;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Content hash for deduplication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Create a content hash from text
    #[must_use]
    pub fn from_text(text: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        Self(hash)
    }

    /// Get hex representation
    #[must_use]
    pub fn as_hex(&self) -> String {
        hex::encode(self.0)
    }
}

/// Embedding cache for deduplication
pub struct EmbeddingCache {
    cache: Arc<DashMap<ContentHash, [f32; 768]>>,
}

impl EmbeddingCache {
    /// Create a new embedding cache
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
        }
    }

    /// Get embedding from cache
    #[must_use]
    pub fn get(&self, hash: &ContentHash) -> Option<[f32; 768]> {
        self.cache.get(hash).map(|entry| *entry.value())
    }

    /// Insert embedding into cache
    pub fn insert(&self, hash: ContentHash, embedding: [f32; 768]) {
        self.cache.insert(hash, embedding);
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.len(),
        }
    }
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Number of entries in cache
    pub size: usize,
}

/// Embedding generator with batching and caching
pub struct EmbeddingGenerator {
    cache: Arc<EmbeddingCache>,
    batch_size: usize,
}

impl EmbeddingGenerator {
    /// Create a new embedding generator
    #[must_use]
    pub fn new(batch_size: usize) -> Self {
        Self {
            cache: Arc::new(EmbeddingCache::new()),
            batch_size,
        }
    }

    /// Generate embedding for a single text
    pub fn generate(&self, text: &str) -> MigrationResult<[f32; 768]> {
        let hash = ContentHash::from_text(text);

        // Check cache first
        if let Some(embedding) = self.cache.get(&hash) {
            return Ok(embedding);
        }

        // Generate new embedding (simple implementation - in production would use actual model)
        let embedding = self.generate_simple_embedding(text);

        // Cache for future use
        self.cache.insert(hash, embedding);

        Ok(embedding)
    }

    /// Generate embeddings for a batch of texts
    pub fn generate_batch(&self, texts: &[String]) -> MigrationResult<Vec<[f32; 768]>> {
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached = Vec::new();
        let mut uncached_indices = Vec::new();

        // Check cache for all texts
        for (i, text) in texts.iter().enumerate() {
            let hash = ContentHash::from_text(text);
            if let Some(embedding) = self.cache.get(&hash) {
                results.push(Some(embedding));
            } else {
                results.push(None);
                uncached.push((text.clone(), hash));
                uncached_indices.push(i);
            }
        }

        // Generate embeddings for uncached texts
        for (text, hash) in uncached {
            let embedding = self.generate_simple_embedding(&text);
            self.cache.insert(hash, embedding);

            // Find and update the result
            for &idx in &uncached_indices {
                if results[idx].is_none() {
                    results[idx] = Some(embedding);
                    break;
                }
            }
        }

        // Convert Option<[f32; 768]> to [f32; 768]
        let final_results: Vec<[f32; 768]> = results
            .into_iter()
            .map(|opt| opt.expect("All embeddings should be generated"))
            .collect();

        Ok(final_results)
    }

    /// Simple embedding generation (placeholder - in production would use real model)
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn generate_simple_embedding(&self, text: &str) -> [f32; 768] {
        let mut embedding = [0.0f32; 768];

        // Create a deterministic but simple embedding based on text content
        let bytes = text.as_bytes();
        for (i, chunk) in bytes.chunks(8).enumerate() {
            if i >= 96 {
                break; // 96 chunks * 8 = 768
            }

            let mut sum = 0u64;
            for &byte in chunk {
                sum = sum.wrapping_add(u64::from(byte));
            }

            for j in 0..8 {
                if i * 8 + j < 768 {
                    embedding[i * 8 + j] = (sum.wrapping_mul(j as u64 + 1) % 1000) as f32 / 1000.0;
                }
            }
        }

        // Normalize to unit length
        let norm = embedding
            .iter()
            .map(|&x| f64::from(x * x))
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val = (*val as f64 / norm) as f32;
            }
        }

        embedding
    }

    /// Get cache statistics
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Get batch size
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }
}
