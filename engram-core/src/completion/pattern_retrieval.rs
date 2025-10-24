//! Semantic pattern retrieval for pattern completion
//!
//! Retrieves relevant semantic patterns from consolidated memory based on
//! partial episode cues, using adaptive weighting between embedding similarity
//! and temporal context matching.

use crate::completion::{ConsolidationEngine, PartialEpisode, SemanticPattern};
use std::collections::{HashMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::pattern_cache::PatternCache;

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
    #[must_use]
    pub fn new(consolidation: Arc<ConsolidationEngine>) -> Self {
        Self {
            consolidation,
            pattern_cache: Arc::new(PatternCache::new(1000)),
            min_pattern_strength: 0.01,
            max_patterns: 10,
            similarity_threshold: 0.6,
        }
    }

    /// Create pattern retriever with custom parameters
    #[must_use]
    pub fn with_params(
        consolidation: Arc<ConsolidationEngine>,
        cache_capacity: usize,
        min_pattern_strength: f32,
        max_patterns: usize,
        similarity_threshold: f32,
    ) -> Self {
        Self {
            consolidation,
            pattern_cache: Arc::new(PatternCache::new(cache_capacity)),
            min_pattern_strength,
            max_patterns,
            similarity_threshold,
        }
    }

    /// Retrieve semantic patterns relevant to partial episode
    ///
    /// Returns patterns ranked by relevance (similarity * strength)
    #[must_use]
    pub fn retrieve_patterns(&self, partial: &PartialEpisode) -> Vec<RankedPattern> {
        // Check cache first
        let cache_key = Self::compute_cache_key(partial);
        if let Some(cached) = self.pattern_cache.get(cache_key) {
            return cached;
        }

        // Compute cue completeness for adaptive weighting
        let completeness = Self::cue_completeness(&partial.partial_embedding);

        // Match by embedding similarity
        let embedding_matches = self.match_by_embedding(&partial.partial_embedding);

        // Match by temporal context
        let temporal_matches = self.match_by_temporal_context(&partial.temporal_context);

        // Merge and rank
        let ranked = self.merge_match_scores(embedding_matches, temporal_matches, completeness);

        // Cache result
        self.pattern_cache.insert(cache_key, ranked.clone());

        ranked
    }

    /// Match partial episode to semantic patterns using embedding similarity
    fn match_by_embedding(&self, partial_embedding: &[Option<f32>]) -> Vec<(String, f32)> {
        // Get all consolidated patterns
        let patterns = self.consolidation.patterns();

        let mut matches = Vec::new();

        for pattern in patterns {
            // Skip weak patterns
            if pattern.strength < self.min_pattern_strength {
                continue;
            }

            // Compute similarity using only non-null dimensions
            let similarity = Self::masked_similarity(partial_embedding, &pattern.embedding);

            if similarity >= self.similarity_threshold {
                matches.push((pattern.id.clone(), similarity));
            }
        }

        // Sort by similarity (descending)
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        matches.truncate(self.max_patterns);
        matches
    }

    /// Match partial episode to patterns using temporal features
    fn match_by_temporal_context(&self, temporal_context: &[String]) -> Vec<(String, f32)> {
        if temporal_context.is_empty() {
            return Vec::new();
        }

        let patterns = self.consolidation.patterns();
        let mut matches = Vec::new();

        for pattern in patterns {
            // Skip weak patterns
            if pattern.strength < self.min_pattern_strength {
                continue;
            }

            // Simple temporal matching: check if any context tags appear in pattern ID
            // This is a placeholder - in production, would use more sophisticated matching
            let mut relevance = 0.0;
            for context_tag in temporal_context {
                if pattern.id.contains(context_tag) {
                    relevance += 0.5;
                }
            }

            // Normalize by number of context tags
            #[allow(clippy::cast_precision_loss)]
            let normalized_relevance = relevance / temporal_context.len() as f32;

            if normalized_relevance > 0.0 {
                matches.push((pattern.id.clone(), normalized_relevance.min(1.0)));
            }
        }

        // Sort by relevance (descending)
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        matches.truncate(self.max_patterns);
        matches
    }

    /// Combine embedding and temporal matches with adaptive weighting
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
            score_map
                .entry(pattern_id.clone())
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
                self.get_pattern(&pattern_id).map(|pattern| RankedPattern {
                    strength: pattern.strength,
                    support_count: pattern.source_episodes.len(),
                    relevance,
                    match_source,
                    pattern,
                })
            })
            .collect();

        // Rank by relevance * strength (multiplicative combination)
        ranked.sort_by(|a, b| {
            let score_a = a.relevance * a.strength;
            let score_b = b.relevance * b.strength;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ranked.truncate(self.max_patterns);
        ranked
    }

    /// Get pattern from cache or consolidation storage
    fn get_pattern(&self, pattern_id: &str) -> Option<SemanticPattern> {
        self.consolidation.pattern_by_id(pattern_id)
    }

    /// Compute masked similarity on non-null dimensions
    fn masked_similarity(partial: &[Option<f32>], full: &[f32; 768]) -> f32 {
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
        let partial_norm = norm_p.sqrt();
        let full_norm = norm_f.sqrt();

        if partial_norm > 0.0 && full_norm > 0.0 {
            dot / (partial_norm * full_norm)
        } else {
            0.0
        }
    }

    /// Compute cue completeness (fraction of non-null embedding dimensions)
    fn cue_completeness(partial_embedding: &[Option<f32>]) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let total = partial_embedding.len() as f32;
        #[allow(clippy::cast_precision_loss)]
        let present = partial_embedding.iter().filter(|v| v.is_some()).count() as f32;
        present / total
    }

    /// Compute cache key for partial episode
    fn compute_cache_key(partial: &PartialEpisode) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash non-null embedding indices
        for (i, val) in partial.partial_embedding.iter().enumerate() {
            if val.is_some() {
                i.hash(&mut hasher);
            }
        }

        // Hash temporal context
        for context in &partial.temporal_context {
            context.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Get cache statistics
    #[must_use]
    pub fn cache_stats(&self) -> super::pattern_cache::CacheStats {
        self.pattern_cache.stats()
    }

    /// Clear pattern cache
    pub fn clear_cache(&self) {
        self.pattern_cache.clear();
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

/// Source of pattern matching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchSource {
    /// Matched by embedding similarity
    Embedding,
    /// Matched by temporal context
    Temporal,
    /// Matched by both embedding and temporal
    Combined,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Confidence;
    use crate::completion::CompletionConfig;

    // Test helper to create test consolidation engine
    fn create_test_consolidation() -> ConsolidationEngine {
        let config = CompletionConfig::default();
        ConsolidationEngine::new(config)
    }

    // Test helper to create partial episode
    fn create_test_partial() -> PartialEpisode {
        PartialEpisode {
            known_fields: HashMap::new(),
            partial_embedding: vec![Some(1.0); 384]
                .into_iter()
                .chain(vec![None; 384])
                .collect(), // 50% complete
            cue_strength: Confidence::exact(0.7),
            temporal_context: vec!["morning".to_string(), "breakfast".to_string()],
        }
    }

    #[test]
    fn test_masked_similarity_computation() {
        let partial = vec![Some(1.0), None, Some(0.5), None];
        let mut full = [0.0; 768];
        full[0] = 1.0;
        full[1] = 0.8;
        full[2] = 0.5;
        full[3] = 0.3;

        let similarity = PatternRetriever::masked_similarity(&partial, &full);

        // Should use only dimensions 0 and 2
        // dot = 1.0*1.0 + 0.5*0.5 = 1.25
        // norm_p = sqrt(1.0 + 0.25) = sqrt(1.25)
        // norm_f = sqrt(1.0 + 0.25) = sqrt(1.25)
        // similarity = 1.25 / 1.25 = 1.0
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cue_completeness_sparse() {
        // 30% complete cue
        let partial_embedding: Vec<Option<f32>> = vec![Some(1.0); 230]
            .into_iter()
            .chain(vec![None; 538])
            .collect();

        let completeness = PatternRetriever::cue_completeness(&partial_embedding);
        assert!((completeness - 0.299_479_17).abs() < 0.01);

        // Temporal weight should dominate
        let temporal_weight = 1.0 - completeness;
        assert!(temporal_weight > 0.65);
    }

    #[test]
    fn test_cue_completeness_rich() {
        // 80% complete cue
        let partial_embedding: Vec<Option<f32>> = vec![Some(1.0); 614]
            .into_iter()
            .chain(vec![None; 154])
            .collect();

        let completeness = PatternRetriever::cue_completeness(&partial_embedding);
        assert!((completeness - 0.799_479_2).abs() < 0.01);

        // Embedding weight should dominate
        let embedding_weight = completeness;
        assert!(embedding_weight > 0.75);
    }

    #[test]
    fn test_pattern_retrieval_empty_consolidation() {
        let consolidation = Arc::new(create_test_consolidation());
        let retriever = PatternRetriever::new(consolidation);

        let partial = create_test_partial();
        let ranked = retriever.retrieve_patterns(&partial);

        // Should return empty for empty consolidation
        assert!(ranked.is_empty());
    }

    #[test]
    fn test_cache_key_consistency() {
        let partial1 = create_test_partial();
        let partial2 = create_test_partial();

        let key1 = PatternRetriever::compute_cache_key(&partial1);
        let key2 = PatternRetriever::compute_cache_key(&partial2);

        // Same partial episodes should produce same cache key
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_stats_tracking() {
        let consolidation = Arc::new(create_test_consolidation());
        let retriever = PatternRetriever::new(consolidation);

        let partial = create_test_partial();

        // First retrieval (miss + insert)
        let _ = retriever.retrieve_patterns(&partial);

        // Second retrieval (hit)
        let _ = retriever.retrieve_patterns(&partial);

        let stats = retriever.cache_stats();

        // Should have 1 hit after the second retrieval
        assert_eq!(stats.hits(), 1);
        // First retrieval was a miss
        assert_eq!(stats.misses(), 1);
        // Hit rate should be 0.5
        assert!((stats.hit_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_clear_cache() {
        let consolidation = Arc::new(create_test_consolidation());
        let retriever = PatternRetriever::new(consolidation);

        let partial = create_test_partial();

        // First retrieval (populates cache)
        let _ = retriever.retrieve_patterns(&partial);

        // Second retrieval (should hit cache)
        let _ = retriever.retrieve_patterns(&partial);
        assert_eq!(retriever.cache_stats().hits(), 1);

        // Clear cache
        retriever.clear_cache();

        // Next retrieval should miss
        let _ = retriever.retrieve_patterns(&partial);
        assert_eq!(retriever.cache_stats().hits(), 1); // Still 1 from before clear
        assert_eq!(retriever.cache_stats().misses(), 2); // One before clear, one after
    }

    #[test]
    fn test_masked_similarity_all_null() {
        let partial = vec![None; 768];
        let full = [1.0; 768];

        let similarity = PatternRetriever::masked_similarity(&partial, &full);

        // Should return 0 for all-null embedding
        assert!(similarity.abs() < 1e-6);
    }

    #[test]
    fn test_masked_similarity_zero_norm() {
        let partial = vec![Some(0.0); 768];
        let full = [1.0; 768];

        let similarity = PatternRetriever::masked_similarity(&partial, &full);

        // Should return 0 when partial has zero norm
        assert!(similarity.abs() < 1e-6);
    }
}
