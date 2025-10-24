# Task 003: Semantic Pattern Retrieval

**Status:** Pending
**Priority:** P0 (Critical Path)
**Estimated Effort:** 2 days
**Dependencies:** None (parallel track with Task 001-002)

## Objective

Integrate consolidated semantic patterns from Milestone 6 consolidation system into pattern completion. Implement efficient retrieval of relevant semantic patterns based on partial episode cues, enabling global pattern knowledge to augment local context reconstruction.

## Integration Points

**Uses:**
- `/engram-core/src/consolidation/pattern_detector.rs` - EpisodicPattern from M6
- `/engram-core/src/completion/consolidation.rs` - SemanticPattern and ConsolidationEngine from M6
- `/engram-core/src/index/hnsw.rs` - HNSW index for fast pattern retrieval (if available)
- `/engram-core/src/embedding/similarity.rs` - SIMD similarity operations

**Creates:**
- `/engram-core/src/completion/pattern_retrieval.rs` - Semantic pattern retrieval engine
- `/engram-core/src/completion/pattern_cache.rs` - LRU cache for frequently-used patterns
- `/engram-core/tests/pattern_retrieval_tests.rs` - Retrieval correctness and performance tests

## Detailed Specification

### 1. Semantic Pattern Retrieval Engine

```rust
// /engram-core/src/completion/pattern_retrieval.rs

use crate::completion::{SemanticPattern, PartialEpisode};
use crate::Confidence;
use dashmap::DashMap;
use std::sync::Arc;

/// Retrieves relevant semantic patterns for completion
pub struct PatternRetriever {
    /// Consolidation engine with learned patterns
    consolidation: Arc<ConsolidationEngine>,

    /// LRU cache for recently-used patterns
    pattern_cache: Arc<PatternCache>,

    /// Minimum pattern strength for retrieval (default: 0.01 p-value)
    min_pattern_strength: f32,

    /// Maximum patterns to retrieve (default: 10)
    max_patterns: usize,

    /// Similarity threshold for pattern matching (default: 0.6)
    similarity_threshold: f32,
}

impl PatternRetriever {
    /// Create new pattern retriever
    pub fn new(consolidation: Arc<ConsolidationEngine>) -> Self;

    /// Retrieve semantic patterns relevant to partial episode
    ///
    /// Returns patterns ranked by relevance (similarity * strength)
    pub fn retrieve_patterns(
        &self,
        partial: &PartialEpisode,
    ) -> Vec<RankedPattern>;

    /// Match partial episode to semantic patterns using embedding similarity
    fn match_by_embedding(
        &self,
        partial_embedding: &[Option<f32>],
    ) -> Vec<(String, f32)>; // (pattern_id, similarity)

    /// Match partial episode to patterns using temporal features
    fn match_by_temporal_context(
        &self,
        temporal_context: &[String],
    ) -> Vec<(String, f32)>; // (pattern_id, relevance)

    /// Combine embedding and temporal matches with adaptive weighting
    fn merge_match_scores(
        &self,
        embedding_matches: Vec<(String, f32)>,
        temporal_matches: Vec<(String, f32)>,
        cue_completeness: f32, // 0.0-1.0
    ) -> Vec<RankedPattern>;

    /// Get pattern from cache or consolidation storage
    fn get_pattern(&self, pattern_id: &str) -> Option<SemanticPattern>;

    /// Compute cue completeness (fraction of non-null embedding dimensions)
    fn cue_completeness(partial_embedding: &[Option<f32>]) -> f32 {
        let total = partial_embedding.len();
        let present = partial_embedding.iter().filter(|v| v.is_some()).count();
        present as f32 / total as f32
    }
}

/// Semantic pattern with relevance ranking
#[derive(Debug, Clone)]
pub struct RankedPattern {
    /// Pattern data
    pub pattern: SemanticPattern,

    /// Relevance score (0.0-1.0)
    pub relevance: f32,

    /// Pattern strength from consolidation (p-value)
    pub strength: f32,

    /// Matching source (embedding, temporal, or both)
    pub match_source: MatchSource,

    /// Number of source episodes in pattern
    pub support_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchSource {
    Embedding,
    Temporal,
    Combined,
}

impl PatternRetriever {
    pub fn retrieve_patterns(
        &self,
        partial: &PartialEpisode,
    ) -> Vec<RankedPattern> {
        // Check cache first
        let cache_key = Self::compute_cache_key(partial);
        if let Some(cached) = self.pattern_cache.get(&cache_key) {
            return cached;
        }

        // Compute cue completeness for adaptive weighting
        let completeness = Self::cue_completeness(&partial.partial_embedding);

        // Match by embedding similarity
        let embedding_matches = self.match_by_embedding(&partial.partial_embedding);

        // Match by temporal context
        let temporal_matches = self.match_by_temporal_context(&partial.temporal_context);

        // Merge and rank
        let ranked = self.merge_match_scores(
            embedding_matches,
            temporal_matches,
            completeness,
        );

        // Cache result
        self.pattern_cache.insert(cache_key, ranked.clone());

        ranked
    }

    fn match_by_embedding(
        &self,
        partial_embedding: &[Option<f32>],
    ) -> Vec<(String, f32)> {
        // Get all consolidated patterns
        let patterns = self.consolidation.get_semantic_patterns();

        let mut matches = Vec::new();

        for pattern in patterns {
            // Compute similarity using only non-null dimensions
            let similarity = Self::masked_similarity(
                partial_embedding,
                &pattern.embedding,
            );

            if similarity >= self.similarity_threshold {
                matches.push((pattern.id.clone(), similarity));
            }
        }

        // Sort by similarity (descending)
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        matches.truncate(self.max_patterns);
        matches
    }

    fn masked_similarity(
        partial: &[Option<f32>],
        full: &[f32; 768],
    ) -> f32 {
        let mut dot = 0.0;
        let mut norm_p = 0.0;
        let mut norm_f = 0.0;
        let mut count = 0;

        for (i, p_opt) in partial.iter().enumerate() {
            if let Some(p) = p_opt {
                let f = full[i];
                dot += p * f;
                norm_p += p * p;
                norm_f += f * f;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        // Cosine similarity on masked dimensions
        dot / (norm_p.sqrt() * norm_f.sqrt())
    }

    fn merge_match_scores(
        &self,
        embedding_matches: Vec<(String, f32)>,
        temporal_matches: Vec<(String, f32)>,
        cue_completeness: f32,
    ) -> Vec<RankedPattern> {
        // Adaptive weighting: sparse cues favor temporal, rich cues favor embedding
        let embedding_weight = cue_completeness;
        let temporal_weight = 1.0 - cue_completeness;

        // Combine scores
        let mut score_map: HashMap<String, (f32, MatchSource)> = HashMap::new();

        for (pattern_id, score) in embedding_matches {
            score_map.insert(
                pattern_id,
                (score * embedding_weight, MatchSource::Embedding),
            );
        }

        for (pattern_id, score) in temporal_matches {
            score_map.entry(pattern_id.clone())
                .and_modify(|(s, src)| {
                    *s += score * temporal_weight;
                    *src = MatchSource::Combined;
                })
                .or_insert((score * temporal_weight, MatchSource::Temporal));
        }

        // Convert to RankedPattern
        let mut ranked: Vec<RankedPattern> = score_map
            .into_iter()
            .filter_map(|(pattern_id, (relevance, match_source))| {
                self.get_pattern(&pattern_id).map(|pattern| {
                    RankedPattern {
                        strength: pattern.statistical_strength,
                        support_count: pattern.source_episodes.len(),
                        relevance,
                        match_source,
                        pattern,
                    }
                })
            })
            .collect();

        // Rank by relevance * strength (multiplicative combination)
        ranked.sort_by(|a, b| {
            let score_a = a.relevance * a.strength;
            let score_b = b.relevance * b.strength;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        ranked.truncate(self.max_patterns);
        ranked
    }
}
```

### 2. Pattern Cache Implementation

```rust
// /engram-core/src/completion/pattern_cache.rs

use lru::LruCache;
use std::sync::Mutex;
use std::num::NonZeroUsize;

/// LRU cache for semantic patterns
pub struct PatternCache {
    /// LRU cache with configurable capacity
    cache: Mutex<LruCache<u64, Vec<RankedPattern>>>,

    /// Cache hit/miss statistics
    stats: CacheStats,
}

impl PatternCache {
    /// Create new pattern cache with capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(capacity).unwrap())),
            stats: CacheStats::default(),
        }
    }

    /// Get patterns from cache
    pub fn get(&self, key: &u64) -> Option<Vec<RankedPattern>> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(patterns) = cache.get(key) {
            self.stats.record_hit();
            Some(patterns.clone())
        } else {
            self.stats.record_miss();
            None
        }
    }

    /// Insert patterns into cache
    pub fn insert(&self, key: u64, patterns: Vec<RankedPattern>) {
        let mut cache = self.cache.lock().unwrap();
        cache.put(key, patterns);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    /// Clear cache
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl CacheStats {
    fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn hit_rate(&self) -> f32 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f32 / total as f32
        }
    }
}
```

### 3. Adaptive Weighting Strategy

**Completeness-Based Weighting:**
```
cue_completeness = (non-null dimensions) / 768

embedding_weight = cue_completeness
temporal_weight = 1.0 - cue_completeness

final_score = (embedding_similarity * embedding_weight) + (temporal_relevance * temporal_weight)
```

**Rationale:**
- **Sparse cues (30% complete):** Embedding similarity unreliable → favor temporal context
- **Rich cues (70% complete):** Embedding similarity informative → favor embedding matching
- **Medium cues (50%):** Balanced weighting between both sources

## Acceptance Criteria

1. **Retrieval Accuracy:**
   - Retrieve >85% of ground-truth relevant patterns in top-10 results
   - Ranking quality: nDCG@10 > 0.80 on validation sets
   - Pattern strength filter prevents spurious patterns (p > 0.01 excluded)

2. **Performance:**
   - Pattern retrieval <5ms P95 for 1000 consolidated patterns
   - Cache hit rate >60% in production workloads
   - Masked similarity computation <100μs for 768-dim embeddings

3. **Adaptive Weighting:**
   - Sparse cues (30%) produce temporal_weight > 0.65
   - Rich cues (80%) produce embedding_weight > 0.75
   - Combined matches outperform single-source by >15% accuracy

4. **Integration Quality:**
   - Seamlessly retrieves patterns from consolidation engine
   - Handles empty pattern sets gracefully (returns empty vec)
   - Thread-safe concurrent retrieval operations

5. **Cache Effectiveness:**
   - LRU eviction maintains bounded memory (<50MB for 1000 patterns)
   - Cache invalidation on new consolidation cycles
   - Statistics tracking for monitoring

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_masked_similarity_computation() {
    let partial = vec![Some(1.0), None, Some(0.5), None];
    let full = [1.0, 0.8, 0.5, 0.3];

    let similarity = PatternRetriever::masked_similarity(&partial, &full);

    // Should use only dimensions 0 and 2
    // dot = 1.0*1.0 + 0.5*0.5 = 1.25
    // norm_p = sqrt(1.0 + 0.25) = sqrt(1.25)
    // norm_f = sqrt(1.0 + 0.25) = sqrt(1.25)
    // similarity = 1.25 / 1.25 = 1.0
    assert!((similarity - 1.0).abs() < 1e-6);
}

#[test]
fn test_adaptive_weighting_sparse_cue() {
    // 30% complete cue
    let partial_embedding = vec![Some(1.0); 230]; // 230/768 ≈ 0.3
    partial_embedding.extend(vec![None; 538]);

    let completeness = PatternRetriever::cue_completeness(&partial_embedding);
    assert!((completeness - 0.3).abs() < 0.01);

    // Temporal weight should dominate
    let temporal_weight = 1.0 - completeness;
    assert!(temporal_weight > 0.65);
}

#[test]
fn test_pattern_retrieval_ranking() {
    let consolidation = setup_test_consolidation_with_patterns();
    let retriever = PatternRetriever::new(Arc::new(consolidation));

    let partial = create_test_partial_episode();
    let ranked = retriever.retrieve_patterns(&partial);

    // Patterns should be sorted by relevance * strength
    for i in 1..ranked.len() {
        let score_prev = ranked[i - 1].relevance * ranked[i - 1].strength;
        let score_curr = ranked[i].relevance * ranked[i].strength;
        assert!(score_prev >= score_curr);
    }
}

#[test]
fn test_pattern_cache_hit() {
    let cache = PatternCache::new(100);
    let key = 12345u64;
    let patterns = vec![create_test_ranked_pattern()];

    cache.insert(key, patterns.clone());

    let retrieved = cache.get(&key);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().len(), patterns.len());

    // Hit rate should be 1.0 (1 hit, 0 misses)
    assert!((cache.stats().hit_rate() - 1.0).abs() < 1e-6);
}

#[test]
fn test_pattern_cache_lru_eviction() {
    let cache = PatternCache::new(2); // Small capacity

    cache.insert(1, vec![create_test_ranked_pattern()]);
    cache.insert(2, vec![create_test_ranked_pattern()]);
    cache.insert(3, vec![create_test_ranked_pattern()]); // Evicts 1

    assert!(cache.get(&1).is_none()); // Evicted
    assert!(cache.get(&2).is_some());
    assert!(cache.get(&3).is_some());
}
```

### Integration Tests

```rust
#[test]
fn test_end_to_end_pattern_retrieval() {
    // Setup consolidation with 100 patterns
    let mut consolidation = ConsolidationEngine::new();
    let patterns = generate_test_patterns(100);
    for pattern in patterns {
        consolidation.store_semantic_pattern(pattern);
    }

    let retriever = PatternRetriever::new(Arc::new(consolidation));

    // Retrieve patterns for partial episode
    let partial = create_breakfast_partial_episode();
    let ranked = retriever.retrieve_patterns(&partial);

    // Should retrieve breakfast-related patterns
    assert!(!ranked.is_empty());
    assert!(ranked[0].pattern.id.contains("breakfast") ||
            ranked[0].pattern.id.contains("morning"));
}

#[test]
fn test_retrieval_with_consolidation_update() {
    let consolidation = Arc::new(Mutex::new(ConsolidationEngine::new()));
    let retriever = PatternRetriever::new(consolidation.clone());

    // Initial retrieval (empty)
    let partial = create_test_partial_episode();
    let ranked1 = retriever.retrieve_patterns(&partial);
    assert!(ranked1.is_empty());

    // Add patterns to consolidation
    {
        let mut consol = consolidation.lock().unwrap();
        consol.store_semantic_pattern(create_matching_pattern(&partial));
    }

    // Clear cache to force re-retrieval
    retriever.pattern_cache.clear();

    // Retrieval after consolidation
    let ranked2 = retriever.retrieve_patterns(&partial);
    assert!(!ranked2.is_empty());
}
```

### Performance Benchmarks

```rust
#[bench]
fn bench_pattern_retrieval_1000_patterns(b: &mut Bencher) {
    let consolidation = setup_consolidation_with_patterns(1000);
    let retriever = PatternRetriever::new(Arc::new(consolidation));
    let partial = create_test_partial_episode();

    b.iter(|| {
        retriever.retrieve_patterns(&partial)
    });
}

#[bench]
fn bench_masked_similarity(b: &mut Bencher) {
    let partial = create_partial_embedding(0.5); // 50% complete
    let full = [0.5f32; 768];

    b.iter(|| {
        PatternRetriever::masked_similarity(&partial, &full)
    });
}
```

## Integration with Hierarchical Evidence Aggregation (Task 004)

### Semantic Patterns as Bayesian Priors
Task 004 research (Hemmer & Steyvers, 2009) defines hierarchical Bayesian reconstruction:
- **Prior:** Semantic knowledge from consolidated patterns (this task)
- **Likelihood:** Episodic evidence from local context (Task 001)
- **Posterior:** Reconstructed memory (Task 004 integration)

**RankedPattern struct provides prior strength:**
```rust
pub struct RankedPattern {
    pub relevance: f32,              // → P(pattern | cue)
    pub strength: f32,               // → P(pattern) from statistical testing
    pub support_count: usize,        // → number of episodes supporting pattern
}
```

### Adaptive Weighting Based on Cue Completeness
Research from Task 004 (Maximum Entropy Principle, Jaynes 1957) informs adaptive weighting strategy:

**Sparse cues (30% complete):**
- Embedding similarity unreliable (high variance)
- temporal_weight = 0.7 (favor context over embedding)
- Semantic patterns provide regularization against noise

**Rich cues (80% complete):**
- Embedding similarity highly informative
- embedding_weight = 0.8 (favor embedding matching)
- Semantic patterns serve as consistency check

**Implementation in merge_match_scores():**
```rust
let embedding_weight = cue_completeness;
let temporal_weight = 1.0 - cue_completeness;
```

This implements adaptive weighting from hierarchical Bayesian models (Griffiths et al., 2008): trust more reliable evidence source based on cue quality.

## Risk Mitigation

**Risk: Retrieval too slow with >10K patterns**
- **Mitigation:** HNSW indexing for approximate nearest neighbor search
- **Contingency:** Limit consolidation to top-K strongest patterns
- **Performance Target (Task 008):** Pattern retrieval <5ms P95 tracked via `engram_pattern_retrieval_duration_seconds`

**Risk: Cache memory growth unbounded**
- **Mitigation:** LRU eviction with configurable capacity (default: 1000 entries)
- **Contingency:** Adaptive cache sizing based on memory pressure
- **Monitoring (Task 008):** Cache hit rate tracked via `engram_pattern_cache_hit_ratio{memory_space}`

**Risk: Masked similarity computation expensive**
- **Mitigation:** SIMD optimization for dot product on non-null dimensions
- **Contingency:** Approximate similarity using hash-based bucketing
- **Validation (Task 009):** Benchmark masked similarity <100μs for 768-dim embeddings

## Implementation Notes

1. Use `rayon` for parallel pattern matching if >1000 patterns
2. Pre-compute pattern embeddings once during consolidation
3. Cache key = hash of (partial embedding non-null indices + temporal context)
4. Monitor cache hit rate; adjust capacity if <50%
5. Invalidate cache entries on consolidation updates (version number tracking)

## Success Criteria Validation

- [ ] Retrieval accuracy >85% (top-10 recall)
- [ ] nDCG@10 > 0.80 on validation sets
- [ ] Pattern retrieval <5ms P95
- [ ] Cache hit rate >60%
- [ ] Adaptive weighting correctly balances embedding/temporal
- [ ] All unit and integration tests pass
- [ ] Performance benchmarks meet targets
