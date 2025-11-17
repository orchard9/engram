//! Content-addressable storage for vector embeddings
//!
//! This module provides content addressing for semantic vectors, enabling
//! direct lookup by content rather than IDs, with automatic deduplication
//! and similarity-based addressing using LSH (Locality-Sensitive Hashing).

use crate::Confidence;
use blake3::Hasher;
use dashmap::DashMap;
use std::sync::OnceLock;

const EMBEDDING_DIM: usize = 768;
const LSH_PROJECTION_COUNT: usize = 32;
const PROJECTION_DOMAIN: &[u8] = b"engram::content_address::projection";
const HIGH_CONFIDENCE_SCALED: u32 = 900; // Confidence::HIGH.raw() * 1000

static LSH_PROJECTIONS: OnceLock<LshProjectionTable> = OnceLock::new();

#[derive(Debug)]
struct LshProjectionTable {
    seeds: [u64; LSH_PROJECTION_COUNT],
    vectors: [[f32; EMBEDDING_DIM]; LSH_PROJECTION_COUNT],
}

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
    #[must_use]
    pub fn from_embedding(embedding: &[f32; EMBEDDING_DIM]) -> Self {
        let quantized = quantize_embedding(embedding);
        let semantic_hash = compute_semantic_hash(&quantized);
        let lsh_bucket = compute_lsh_bucket(embedding);

        Self {
            semantic_hash,
            lsh_bucket,
            confidence: HIGH_CONFIDENCE_SCALED,
        }
    }

    /// Get the confidence as a Confidence type
    #[must_use]
    pub fn get_confidence(&self) -> Confidence {
        Confidence::from_raw(confidence_from_scaled(self.confidence))
    }

    /// Expose the deterministic LSH projection seeds for observability.
    #[must_use]
    pub fn projection_seeds() -> &'static [u64; LSH_PROJECTION_COUNT] {
        &projection_table().seeds
    }

    /// Expose the projection vectors for deterministic replay/testing.
    #[must_use]
    pub fn projection_vectors() -> &'static [[f32; EMBEDDING_DIM]; LSH_PROJECTION_COUNT] {
        &projection_table().vectors
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

fn quantize_embedding(embedding: &[f32; EMBEDDING_DIM]) -> [u8; EMBEDDING_DIM] {
    let mut quantized = [0u8; EMBEDDING_DIM];
    for (idx, value) in embedding.iter().enumerate() {
        quantized[idx] = quantize_value(*value);
    }
    quantized
}

fn compute_semantic_hash(quantized: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(quantized);
    let hash = hasher.finalize();
    *hash.as_bytes()
}

fn compute_lsh_bucket(embedding: &[f32; EMBEDDING_DIM]) -> u32 {
    let table = projection_table();
    table
        .vectors
        .iter()
        .enumerate()
        .fold(0u32, |mut bucket, (bit_idx, projection)| {
            if dot_product(projection, embedding) >= 0.0 {
                bucket |= 1 << bit_idx;
            }
            bucket
        })
}

fn dot_product(a: &[f32; EMBEDDING_DIM], b: &[f32; EMBEDDING_DIM]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn projection_table() -> &'static LshProjectionTable {
    LSH_PROJECTIONS.get_or_init(build_projection_table)
}

#[allow(clippy::large_stack_arrays)]
fn build_projection_table() -> LshProjectionTable {
    let mut seeds = [0u64; LSH_PROJECTION_COUNT];
    let mut vectors = [[0.0f32; EMBEDDING_DIM]; LSH_PROJECTION_COUNT];
    for bit_idx in 0..LSH_PROJECTION_COUNT {
        let seed = derive_seed(bit_idx as u64);
        seeds[bit_idx] = seed;
        vectors[bit_idx] = build_projection_vector(seed);
    }

    LshProjectionTable { seeds, vectors }
}

const fn derive_seed(index: u64) -> u64 {
    const BASE: u64 = 0xB6E0_C10F_F3A5_A1D3;
    const STEP: u64 = 0x9E37_79B9_7F4A_7C15;
    BASE ^ STEP.wrapping_mul(index + 1)
}

fn build_projection_vector(seed: u64) -> [f32; EMBEDDING_DIM] {
    let mut vector = [0.0f32; EMBEDDING_DIM];
    let mut hasher = Hasher::new();
    hasher.update(PROJECTION_DOMAIN);
    hasher.update(&seed.to_le_bytes());
    let mut reader = hasher.finalize_xof();
    for value in &mut vector {
        let mut bytes = [0u8; 4];
        reader.fill(&mut bytes);
        let raw = u32::from_le_bytes(bytes);
        *value = map_u32_to_unit(raw);
    }
    vector
}

fn map_u32_to_unit(value: u32) -> f32 {
    let ratio = f64::from(value) / f64::from(u32::MAX);
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    {
        ratio.mul_add(2.0, -1.0) as f32
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
        let quantized = quantize_embedding(&embedding);
        assert_eq!(quantized[0], 128); // 0.0 maps to middle value

        let embedding_max = [1.0f32; 768];
        let quantized_max = quantize_embedding(&embedding_max);
        assert_eq!(quantized_max[0], 255); // 1.0 maps to max value

        let embedding_min = [-1.0f32; 768];
        let quantized_min = quantize_embedding(&embedding_min);
        assert_eq!(quantized_min[0], 0); // -1.0 maps to min value
    }

    #[test]
    fn test_internal_hash_helpers_are_stable() {
        let embedding = [0.42f32; 768];
        let quantized = quantize_embedding(&embedding);
        let hash = compute_semantic_hash(&quantized);
        assert!(hash.iter().any(|byte| *byte != 0));

        let bucket_a = compute_lsh_bucket(&embedding);
        let bucket_b = compute_lsh_bucket(&embedding);
        assert_eq!(bucket_a, bucket_b);

        let seeds = ContentAddress::projection_seeds();
        assert_eq!(seeds.len(), LSH_PROJECTION_COUNT);
        let vectors = ContentAddress::projection_vectors();
        assert_eq!(vectors.len(), LSH_PROJECTION_COUNT);
        assert!(vectors[0].iter().all(|value| value.is_finite()));
    }
}
