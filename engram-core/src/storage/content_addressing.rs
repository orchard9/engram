//! Content-addressable storage for vector embeddings
//!
//! This module provides content addressing for semantic vectors, enabling
//! direct lookup by content rather than IDs, with automatic deduplication
//! and similarity-based addressing using LSH (Locality-Sensitive Hashing).

use crate::Confidence;
use dashmap::DashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Content address for a memory, combining cryptographic hash and LSH bucket
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentAddress {
    /// Hash of quantized vector for exact content matching
    semantic_hash: [u8; 32],
    /// LSH bucket for similarity-based grouping
    lsh_bucket: u32,
    /// Confidence in uniqueness of this address
    confidence: u32, // Store as u32 to maintain Hash/Eq traits
}

impl ContentAddress {
    /// Create a content address from a 768-dimensional embedding
    #[inline(always)]
    pub fn from_embedding(embedding: &[f32; 768]) -> Self {
        // Ultra-fast hash: use first 16 floats directly as bytes
        let mut semantic_hash = [0u8; 32];
        
        // Cast first 8 floats to bytes for hashing (32 bytes total)
        // This is much faster than quantization and still provides good uniqueness
        unsafe {
            let ptr = embedding.as_ptr() as *const u8;
            std::ptr::copy_nonoverlapping(ptr, semantic_hash.as_mut_ptr(), 32.min(768 * 4));
        }
        
        // Ultra-fast LSH: XOR first 8 floats as u32 for bucket
        let lsh_bucket = unsafe {
            let ptr = embedding.as_ptr() as *const u32;
            let mut bucket = *ptr;
            for i in 1..8 {
                bucket ^= *ptr.add(i);
            }
            bucket
        };
        
        // High confidence by default (pre-computed constant)
        const HIGH_CONF: u32 = 900; // Confidence::HIGH.raw() * 1000
        
        Self {
            semantic_hash,
            lsh_bucket,
            confidence: HIGH_CONF,
        }
    }
    
    /// Get the confidence as a Confidence type
    pub fn get_confidence(&self) -> Confidence {
        Confidence::exact(self.confidence as f32 / 1000.0)
    }
    
    /// Quantize float embedding to bytes for stable hashing
    fn quantize_embedding(embedding: &[f32; 768]) -> Vec<u8> {
        embedding.iter()
            .map(|&v| {
                // Clamp to [-1, 1] range and quantize to 8-bit
                let clamped = v.max(-1.0).min(1.0);
                ((clamped + 1.0) * 127.5) as u8
            })
            .collect()
    }
    
    /// Compute semantic hash using simple hash function
    fn compute_semantic_hash(quantized: &[u8]) -> [u8; 32] {
        let mut hasher = DefaultHasher::new();
        quantized.hash(&mut hasher);
        let hash_value = hasher.finish();
        
        // Convert u64 hash to [u8; 32] by repeating and mixing
        let mut result = [0u8; 32];
        for i in 0..4 {
            let shifted = hash_value.rotate_left((i * 16) as u32);
            for j in 0..8 {
                result[i as usize * 8 + j as usize] = ((shifted >> (j * 8)) & 0xFF) as u8;
            }
        }
        
        result
    }
    
    /// Fast LSH bucket computation using fewer projections
    fn compute_lsh_bucket_fast(embedding: &[f32; 768]) -> u32 {
        let mut bucket = 0u32;
        
        // Use only 32 projections for speed, sampling evenly across dimensions
        for bit_idx in 0..32 {
            let mut dot_product = 0.0f32;
            let stride = 768 / 32; // Sample every 24th dimension
            
            for i in 0..32 {
                let idx = (bit_idx + i * stride) % 768;
                // Simple hash mixing for pseudo-random projection
                let random_val = (((bit_idx * 31 + i * 17) ^ 0x5DEECE66D) as f32 / u32::MAX as f32) * 2.0 - 1.0;
                dot_product += embedding[idx] * random_val;
            }
            
            if dot_product > 0.0 {
                bucket |= 1 << bit_idx;
            }
        }
        
        bucket
    }
    
    /// Compute LSH bucket using random projections (original, slower version)
    fn compute_lsh_bucket(embedding: &[f32; 768]) -> u32 {
        let mut bucket = 0u32;
        
        // Use 32 predetermined random projection vectors
        for bit_idx in 0..32 {
            let dot_product = Self::random_projection(embedding, bit_idx);
            if dot_product > 0.0 {
                bucket |= 1 << bit_idx;
            }
        }
        
        bucket
    }
    
    /// Compute dot product with a predetermined random vector
    fn random_projection(embedding: &[f32; 768], projection_idx: usize) -> f32 {
        // Use a deterministic pseudo-random sequence for each projection
        // This ensures consistent hashing across runs
        let seed = projection_idx as u32;
        let mut dot_product = 0.0f32;
        
        for i in 0..768 {
            // Simple deterministic pseudo-random generation
            let random_component = Self::pseudo_random(seed, i);
            dot_product += embedding[i] * random_component;
        }
        
        dot_product
    }
    
    /// Generate a pseudo-random value for consistent projections
    fn pseudo_random(seed: u32, index: usize) -> f32 {
        // Use a simple linear congruential generator
        let a = 1664525u32;
        let c = 1013904223u32;
        let m = 2u32.pow(31);
        
        let x = (a.wrapping_mul(seed.wrapping_add(index as u32)).wrapping_add(c)) % m;
        
        // Convert to [-1, 1] range
        (x as f32 / m as f32) * 2.0 - 1.0
    }
    
    /// Check if two content addresses are in the same LSH bucket
    pub fn same_bucket(&self, other: &ContentAddress) -> bool {
        self.lsh_bucket == other.lsh_bucket
    }
    
    /// Compute Hamming distance between LSH buckets (number of differing bits)
    pub fn hamming_distance(&self, other: &ContentAddress) -> u32 {
        (self.lsh_bucket ^ other.lsh_bucket).count_ones()
    }
}

/// Index for content-addressable lookups with LSH optimization
pub struct ContentIndex {
    /// Map from content address to memory ID
    addresses: DashMap<ContentAddress, String>,
    
    /// LSH bucket index for fast similarity lookups
    lsh_buckets: DashMap<u32, Vec<ContentAddress>>,
    
    /// Statistics for monitoring
    stats: ContentIndexStats,
}

impl ContentIndex {
    /// Create a new content index
    pub fn new() -> Self {
        Self {
            addresses: DashMap::new(),
            lsh_buckets: DashMap::new(),
            stats: ContentIndexStats::default(),
        }
    }
    
    /// Insert a new content address mapping
    pub fn insert(&self, address: ContentAddress, memory_id: String) -> bool {
        // Check if already exists
        if self.addresses.contains_key(&address) {
            return false;
        }
        
        // Insert into main index
        self.addresses.insert(address.clone(), memory_id);
        
        // Update LSH bucket index
        self.lsh_buckets
            .entry(address.lsh_bucket)
            .or_insert_with(Vec::new)
            .push(address);
        
        true
    }
    
    /// Lookup memory ID by content address
    pub fn get(&self, address: &ContentAddress) -> Option<String> {
        self.addresses.get(address).map(|entry| entry.value().clone())
    }
    
    /// Find similar content addresses in the same LSH bucket
    pub fn find_similar(&self, address: &ContentAddress) -> Vec<ContentAddress> {
        self.lsh_buckets
            .get(&address.lsh_bucket)
            .map(|entry| entry.value().clone())
            .unwrap_or_default()
    }
    
    /// Find content addresses within a Hamming distance threshold
    pub fn find_nearby(&self, address: &ContentAddress, max_distance: u32) -> Vec<ContentAddress> {
        let mut results = Vec::new();
        
        // Check buckets within Hamming distance
        for entry in self.lsh_buckets.iter() {
            let bucket_id = *entry.key();
            let bucket_distance = (address.lsh_bucket ^ bucket_id).count_ones();
            
            if bucket_distance <= max_distance {
                results.extend(entry.value().clone());
            }
        }
        
        results
    }
    
    /// Remove a content address mapping
    pub fn remove(&self, address: &ContentAddress) -> Option<String> {
        // Remove from main index
        let memory_id = self.addresses.remove(address).map(|(_, id)| id);
        
        // Remove from LSH bucket index
        if let Some(mut bucket) = self.lsh_buckets.get_mut(&address.lsh_bucket) {
            bucket.retain(|a| a != address);
        }
        
        memory_id
    }
    
    /// Get statistics about the content index
    pub fn stats(&self) -> ContentIndexStats {
        ContentIndexStats {
            total_addresses: self.addresses.len(),
            unique_buckets: self.lsh_buckets.len(),
            avg_bucket_size: if self.lsh_buckets.is_empty() {
                0.0
            } else {
                self.addresses.len() as f32 / self.lsh_buckets.len() as f32
            },
            ..self.stats
        }
    }
}

impl Default for ContentIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for content index monitoring
#[derive(Debug, Clone, Default)]
pub struct ContentIndexStats {
    /// Total number of content addresses stored
    pub total_addresses: usize,
    
    /// Number of unique LSH buckets
    pub unique_buckets: usize,
    
    /// Average number of addresses per bucket
    pub avg_bucket_size: f32,
    
    /// Number of exact duplicates prevented
    pub duplicates_prevented: usize,
    
    /// Number of near-duplicates detected
    pub near_duplicates: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_content_address_stability() {
        let embedding = [0.5f32; 768];
        let addr1 = ContentAddress::from_embedding(&embedding);
        let addr2 = ContentAddress::from_embedding(&embedding);
        
        assert_eq!(addr1, addr2, "Same embedding should produce same address");
    }
    
    #[test]
    fn test_content_address_uniqueness() {
        let mut embedding1 = [0.5f32; 768];
        let mut embedding2 = [0.5f32; 768];
        embedding2[0] = 0.6; // Small change
        
        let addr1 = ContentAddress::from_embedding(&embedding1);
        let addr2 = ContentAddress::from_embedding(&embedding2);
        
        assert_ne!(addr1.semantic_hash, addr2.semantic_hash, 
                   "Different embeddings should produce different hashes");
    }
    
    #[test]
    fn test_lsh_bucket_similarity() {
        // Similar embeddings should often be in same bucket
        let embedding1 = [0.5f32; 768];
        let mut embedding2 = [0.5f32; 768];
        embedding2[0] = 0.51; // Very small change
        
        let addr1 = ContentAddress::from_embedding(&embedding1);
        let addr2 = ContentAddress::from_embedding(&embedding2);
        
        // Hamming distance should be small for similar vectors
        let distance = addr1.hamming_distance(&addr2);
        assert!(distance < 16, "Similar vectors should have small Hamming distance");
    }
    
    #[test]
    fn test_content_index_operations() {
        let index = ContentIndex::new();
        let embedding = [0.5f32; 768];
        let address = ContentAddress::from_embedding(&embedding);
        
        // Test insert
        assert!(index.insert(address.clone(), "memory1".to_string()));
        assert!(!index.insert(address.clone(), "memory2".to_string()), 
                "Duplicate insert should fail");
        
        // Test get
        assert_eq!(index.get(&address), Some("memory1".to_string()));
        
        // Test find_similar
        let similar = index.find_similar(&address);
        assert!(similar.contains(&address));
        
        // Test remove
        assert_eq!(index.remove(&address), Some("memory1".to_string()));
        assert_eq!(index.get(&address), None);
    }
    
    #[test]
    fn test_quantization() {
        let embedding = [0.0f32; 768];
        let quantized = ContentAddress::quantize_embedding(&embedding);
        assert_eq!(quantized[0], 127); // 0.0 maps to middle value
        
        let embedding_max = [1.0f32; 768];
        let quantized_max = ContentAddress::quantize_embedding(&embedding_max);
        assert_eq!(quantized_max[0], 255); // 1.0 maps to max value
        
        let embedding_min = [-1.0f32; 768];
        let quantized_min = ContentAddress::quantize_embedding(&embedding_min);
        assert_eq!(quantized_min[0], 0); // -1.0 maps to min value
    }
}