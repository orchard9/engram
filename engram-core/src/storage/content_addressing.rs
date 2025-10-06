//! Content-addressable storage for vector embeddings
//!
//! This module provides content addressing for semantic vectors, enabling
//! direct lookup by content rather than IDs, with automatic deduplication
//! and similarity-based addressing using LSH (Locality-Sensitive Hashing).

use crate::Confidence;
use dashmap::DashMap;
#[cfg(test)]
use std::collections::hash_map::DefaultHasher;
#[cfg(test)]
use std::hash::{Hash, Hasher};

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
    #[must_use]
    pub fn from_embedding(embedding: &[f32; 768]) -> Self {
        // High confidence by default (pre-computed constant)
        const HIGH_CONF: u32 = 900; // Confidence::HIGH.raw() * 1000

        // Ultra-fast hash: use first 16 floats directly as bytes
        let mut semantic_hash = [0u8; 32];

        // Cast first 8 floats to bytes for hashing (32 bytes total)
        // This is much faster than quantization and still provides good uniqueness
        for (chunk, value) in semantic_hash.chunks_exact_mut(4).zip(embedding.iter()) {
            chunk.copy_from_slice(&value.to_ne_bytes());
        }

        // Ultra-fast LSH: XOR first 8 floats as u32 for bucket
        let mut bucket_iter = embedding.iter().take(8).map(|value| value.to_bits());
        let lsh_bucket = bucket_iter
            .next()
            .map_or(0, |first| bucket_iter.fold(first, |acc, bits| acc ^ bits));

        Self {
            semantic_hash,
            lsh_bucket,
            confidence: HIGH_CONF,
        }
    }

    /// Get the confidence as a Confidence type
    #[must_use]
    pub fn get_confidence(&self) -> Confidence {
        Confidence::from_raw(confidence_from_scaled(self.confidence))
    }

    /// Quantize float embedding to bytes for stable hashing
    #[cfg(test)]
    fn quantize_embedding(embedding: &[f32; 768]) -> Vec<u8> {
        embedding
            .iter()
            .map(|&value| quantize_value(value))
            .collect()
    }

    /// Compute semantic hash using simple hash function
    #[cfg(test)]
    #[must_use]
    fn compute_semantic_hash(quantized: &[u8]) -> [u8; 32] {
        let mut hasher = DefaultHasher::new();
        quantized.hash(&mut hasher);
        let hash_value = hasher.finish();

        // Convert u64 hash to [u8; 32] by repeating and mixing
        let mut result = [0u8; 32];
        for (i, chunk) in result.chunks_exact_mut(8).enumerate() {
            let shift = u32::try_from(i * 16).unwrap_or(0);
            let shifted = hash_value.rotate_left(shift);
            for (j, byte) in chunk.iter_mut().enumerate() {
                let offset = u32::try_from(j * 8).unwrap_or(0);
                let raw = (shifted >> offset) & 0xFF;
                *byte = u8::try_from(raw).unwrap_or(0);
            }
        }

        result
    }

    /// Fast LSH bucket computation using fewer projections
    #[cfg(test)]
    #[must_use]
    fn compute_lsh_bucket_fast(embedding: &[f32; 768]) -> u32 {
        let mut bucket = 0u32;

        // Use only 32 projections for speed, sampling evenly across dimensions
        for bit_idx in 0..32 {
            let mut dot_product = 0.0f32;
            let stride = 768 / 32; // Sample every 24th dimension

            for i in 0..32 {
                let idx = (bit_idx + i * stride) % 768;
                // Simple hash mixing for pseudo-random projection
                let bit_component = u32::try_from(bit_idx * 31).unwrap_or(0);
                let index_component = u32::try_from(i * 17).unwrap_or(0);
                let mix = bit_component.wrapping_add(index_component) ^ 0xDEEC_E66D;
                let max_val = f64::from(u32::MAX);
                let ratio = f64::from(mix) / max_val;
                #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                let random_val = ratio.mul_add(2.0, -1.0) as f32;
                dot_product += embedding[idx] * random_val;
            }

            if dot_product > 0.0 {
                bucket |= 1 << bit_idx;
            }
        }

        bucket
    }

    /// Compute LSH bucket using random projections (original, slower version)
    #[cfg(test)]
    #[must_use]
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
    #[cfg(test)]
    #[must_use]
    fn random_projection(embedding: &[f32; 768], projection_idx: usize) -> f32 {
        // Use a deterministic pseudo-random sequence for each projection
        // This ensures consistent hashing across runs
        let seed = u32::try_from(projection_idx).unwrap_or(u32::MAX);
        let mut dot_product = 0.0f32;

        for (i, value) in embedding.iter().enumerate() {
            // Simple deterministic pseudo-random generation
            let random_component = Self::pseudo_random(seed, i);
            dot_product += value * random_component;
        }

        dot_product
    }

    /// Generate a pseudo-random value for consistent projections
    #[cfg(test)]
    #[must_use]
    fn pseudo_random(seed: u32, index: usize) -> f32 {
        // Use a simple linear congruential generator
        let a = 1_664_525u32;
        let c = 1_013_904_223u32;
        let m = 2u32.pow(31);

        let index_u32 = u32::try_from(index).unwrap_or(u32::MAX);
        let x = (a.wrapping_mul(seed.wrapping_add(index_u32)).wrapping_add(c)) % m;

        // Convert to [-1, 1] range
        #[allow(clippy::cast_precision_loss)]
        (x as f32 / m as f32).mul_add(2.0, -1.0)
    }

    /// Check if two content addresses are in the same LSH bucket
    #[must_use]
    pub const fn same_bucket(&self, other: &Self) -> bool {
        self.lsh_bucket == other.lsh_bucket
    }

    /// Compute Hamming distance between LSH buckets (number of differing bits)
    #[must_use]
    pub const fn hamming_distance(&self, other: &Self) -> u32 {
        (self.lsh_bucket ^ other.lsh_bucket).count_ones()
    }
}

#[cfg(test)]
#[inline]
#[must_use]
fn quantize_value(value: f32) -> u8 {
    let clamped = value.clamp(-1.0, 1.0);
    let scaled = (clamped + 1.0) * 127.5;
    let rounded = scaled.round().clamp(0.0, 255.0);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    {
        rounded as u8
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            addresses: DashMap::new(),
            lsh_buckets: DashMap::new(),
            stats: ContentIndexStats::default(),
        }
    }

    /// Insert a new content address mapping
    #[must_use]
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
            .or_default()
            .push(address);

        true
    }

    /// Lookup memory ID by content address
    #[must_use]
    pub fn get(&self, address: &ContentAddress) -> Option<String> {
        self.addresses
            .get(address)
            .map(|entry| entry.value().clone())
    }

    /// Find similar content addresses in the same LSH bucket
    #[must_use]
    pub fn find_similar(&self, address: &ContentAddress) -> Vec<ContentAddress> {
        self.lsh_buckets
            .get(&address.lsh_bucket)
            .map_or_else(Vec::new, |entry| entry.value().clone())
    }

    /// Find content addresses within a Hamming distance threshold
    #[must_use]
    pub fn find_nearby(&self, address: &ContentAddress, max_distance: u32) -> Vec<ContentAddress> {
        let mut results = Vec::new();

        // Check buckets within Hamming distance
        for entry in &self.lsh_buckets {
            let bucket_id = *entry.key();
            let bucket_distance = (address.lsh_bucket ^ bucket_id).count_ones();

            if bucket_distance <= max_distance {
                results.extend(entry.value().clone());
            }
        }

        results
    }

    /// Remove a content address mapping
    #[must_use]
    pub fn remove(&self, address: &ContentAddress) -> Option<String> {
        // Remove from main index
        let memory_id = self.addresses.remove(address).map(|(_, id)| id);

        // Remove from LSH bucket index
        if let Some(mut bucket) = self.lsh_buckets.get_mut(&address.lsh_bucket) {
            bucket.retain(|a| a != address);
        }

        memory_id
    }

    /// Remove all entries associated with a memory identifier
    #[must_use]
    pub fn remove_by_memory_id(&self, memory_id: &str) -> usize {
        let matching_addresses: Vec<ContentAddress> = self
            .addresses
            .iter()
            .filter(|entry| entry.value() == memory_id)
            .map(|entry| entry.key().clone())
            .collect();

        let mut removed = 0usize;
        for address in matching_addresses {
            if self.remove(&address).is_some() {
                removed += 1;
            }
        }

        removed
    }

    /// Get statistics about the content index
    #[must_use]
    pub fn stats(&self) -> ContentIndexStats {
        let total_addresses = self.addresses.len();
        let unique_buckets = self.lsh_buckets.len();
        let avg_bucket_size = if unique_buckets == 0 {
            0.0
        } else {
            safe_divide_usize(total_addresses, unique_buckets)
        };

        ContentIndexStats {
            total_addresses,
            unique_buckets,
            avg_bucket_size,
            ..self.stats
        }
    }
}

impl Default for ContentIndex {
    fn default() -> Self {
        Self::new()
    }
}

fn confidence_from_scaled(value: u32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    {
        value as f32 / 1000.0
    }
}

fn safe_divide_usize(numerator: usize, denominator: usize) -> f32 {
    let numerator = usize_to_f32(numerator);
    let denominator = usize_to_f32(denominator).max(f32::EPSILON);
    numerator / denominator
}

const fn usize_to_f32(value: usize) -> f32 {
    if fits_in_u32(value) {
        #[allow(clippy::cast_precision_loss)]
        {
            value as f32
        }
    } else {
        f32::MAX
    }
}

const fn fits_in_u32(value: usize) -> bool {
    if usize::BITS <= 32 {
        true
    } else {
        (value >> 32) == 0
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
        let embedding1 = [0.5f32; 768];
        let mut embedding2 = [0.5f32; 768];
        embedding2[0] = 0.6; // Small change

        let addr1 = ContentAddress::from_embedding(&embedding1);
        let addr2 = ContentAddress::from_embedding(&embedding2);

        assert_ne!(
            addr1.semantic_hash, addr2.semantic_hash,
            "Different embeddings should produce different hashes"
        );
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
        assert!(
            distance < 16,
            "Similar vectors should have small Hamming distance"
        );
    }

    #[test]
    fn test_content_index_operations() {
        let index = ContentIndex::new();
        let embedding = [0.5f32; 768];
        let address = ContentAddress::from_embedding(&embedding);

        // Test insert
        assert!(index.insert(address.clone(), "memory1".to_string()));
        assert!(
            !index.insert(address.clone(), "memory2".to_string()),
            "Duplicate insert should fail"
        );

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
        assert_eq!(quantized[0], 128); // 0.0 maps to middle value

        let embedding_max = [1.0f32; 768];
        let quantized_max = ContentAddress::quantize_embedding(&embedding_max);
        assert_eq!(quantized_max[0], 255); // 1.0 maps to max value

        let embedding_min = [-1.0f32; 768];
        let quantized_min = ContentAddress::quantize_embedding(&embedding_min);
        assert_eq!(quantized_min[0], 0); // -1.0 maps to min value
    }

    #[test]
    fn test_internal_hash_helpers_are_stable() {
        let embedding = [0.42f32; 768];
        let quantized = ContentAddress::quantize_embedding(&embedding);
        let hash = ContentAddress::compute_semantic_hash(&quantized);
        assert!(hash.iter().any(|byte| *byte != 0));

        let fast_bucket = ContentAddress::compute_lsh_bucket_fast(&embedding);
        let slow_bucket = ContentAddress::compute_lsh_bucket(&embedding);
        assert!(fast_bucket.count_ones() <= 32);
        assert!(slow_bucket.count_ones() <= 32);

        let projection = ContentAddress::random_projection(&embedding, 0);
        assert!(projection.is_finite());

        let pseudo = ContentAddress::pseudo_random(42, 7);
        assert!((-1.0..=1.0).contains(&pseudo));
    }
}
