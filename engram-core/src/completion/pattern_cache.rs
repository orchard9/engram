//! LRU cache for semantic patterns
//!
//! Implements efficient caching of pattern retrieval results to reduce
//! consolidation engine lookups during pattern completion operations.

use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::pattern_retrieval::RankedPattern;

/// LRU cache for semantic patterns
pub struct PatternCache {
    /// LRU cache with configurable capacity
    cache: Mutex<LruCache<u64, Vec<RankedPattern>>>,

    /// Cache hit/miss statistics
    stats: CacheStats,
}

impl PatternCache {
    /// Create new pattern cache with capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        #[allow(clippy::unwrap_used)]
        let capacity = NonZeroUsize::new(capacity).unwrap_or_else(|| NonZeroUsize::new(1).unwrap());
        Self {
            cache: Mutex::new(LruCache::new(capacity)),
            stats: CacheStats::default(),
        }
    }

    /// Get patterns from cache
    #[must_use]
    pub fn get(&self, key: u64) -> Option<Vec<RankedPattern>> {
        let mut cache = self.cache.lock().ok()?;
        cache.get(&key).map_or_else(
            || {
                self.stats.record_miss();
                None
            },
            |patterns| {
                self.stats.record_hit();
                Some(patterns.clone())
            },
        )
    }

    /// Insert patterns into cache
    pub fn insert(&self, key: u64, patterns: Vec<RankedPattern>) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.put(key, patterns);
        }
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    /// Clear cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Number of cache hits
    hits: AtomicUsize,
    /// Number of cache misses
    misses: AtomicUsize,
}

impl Clone for CacheStats {
    fn clone(&self) -> Self {
        Self {
            hits: AtomicUsize::new(self.hits.load(Ordering::Relaxed)),
            misses: AtomicUsize::new(self.misses.load(Ordering::Relaxed)),
        }
    }
}

impl CacheStats {
    /// Record a cache hit
    pub(crate) fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub(crate) fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get cache hit rate (0.0-1.0)
    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                hits as f32 / total as f32
            }
        }
    }

    /// Get total number of hits
    #[must_use]
    pub fn hits(&self) -> usize {
        self.hits.load(Ordering::Relaxed)
    }

    /// Get total number of misses
    #[must_use]
    pub fn misses(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Confidence;
    use crate::completion::SemanticPattern;
    use chrono::Utc;

    // Test helper to create a ranked pattern
    fn create_test_ranked_pattern(id: &str) -> RankedPattern {
        let pattern = SemanticPattern {
            id: id.to_string(),
            embedding: [0.5; 768],
            source_episodes: vec!["ep1".to_string(), "ep2".to_string()],
            strength: 0.9,
            schema_confidence: Confidence::exact(0.8),
            last_consolidated: Utc::now(),
        };

        RankedPattern {
            pattern,
            relevance: 0.85,
            strength: 0.9,
            match_source: super::super::pattern_retrieval::MatchSource::Embedding,
            support_count: 2,
        }
    }

    #[test]
    fn test_pattern_cache_hit() {
        let cache = PatternCache::new(100);
        let key = 12345_u64;
        let patterns = vec![create_test_ranked_pattern("pattern1")];

        cache.insert(key, patterns.clone());

        let retrieved = cache.get(key);
        assert!(retrieved.is_some());
        if let Some(patterns_vec) = retrieved {
            assert_eq!(patterns_vec.len(), patterns.len());
        }

        // Hit rate should be 1.0 (1 hit, 0 misses)
        assert!((cache.stats().hit_rate() - 1.0).abs() < 1e-6);
        assert_eq!(cache.stats().hits(), 1);
        assert_eq!(cache.stats().misses(), 0);
    }

    #[test]
    fn test_pattern_cache_miss() {
        let cache = PatternCache::new(100);
        let key = 12345_u64;

        let retrieved = cache.get(key);
        assert!(retrieved.is_none());

        // Hit rate should be 0.0 (0 hits, 1 miss)
        assert!(cache.stats().hit_rate().abs() < 1e-6);
        assert_eq!(cache.stats().hits(), 0);
        assert_eq!(cache.stats().misses(), 1);
    }

    #[test]
    fn test_pattern_cache_lru_eviction() {
        let cache = PatternCache::new(2); // Small capacity

        cache.insert(1, vec![create_test_ranked_pattern("pattern1")]);
        cache.insert(2, vec![create_test_ranked_pattern("pattern2")]);
        cache.insert(3, vec![create_test_ranked_pattern("pattern3")]); // Evicts 1

        assert!(cache.get(1).is_none()); // Evicted
        assert!(cache.get(2).is_some());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_pattern_cache_clear() {
        let cache = PatternCache::new(100);

        cache.insert(1, vec![create_test_ranked_pattern("pattern1")]);
        cache.insert(2, vec![create_test_ranked_pattern("pattern2")]);

        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_some());

        cache.clear();

        // After clear, all entries should be gone
        assert!(cache.get(1).is_none());
        assert!(cache.get(2).is_none());
    }

    #[test]
    fn test_cache_stats_accumulation() {
        let cache = PatternCache::new(100);

        // Record several hits and misses
        cache.insert(1, vec![create_test_ranked_pattern("pattern1")]);

        let _ = cache.get(1); // hit
        let _ = cache.get(2); // miss
        let _ = cache.get(1); // hit
        let _ = cache.get(3); // miss

        assert_eq!(cache.stats().hits(), 2);
        assert_eq!(cache.stats().misses(), 2);
        assert!((cache.stats().hit_rate() - 0.5).abs() < 1e-6); // 2/4 = 0.5
    }
}
