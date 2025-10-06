use super::{
    multi_cue::{AggregatedCue, CueAggregationStrategy, MultiCueAggregator},
    simd_optimization::SimdActivationMapper,
    similarity_config::SimilarityConfig,
};
use crate::{
    Confidence, Cue, CueType,
    index::{CognitiveHnswIndex, SearchResult, SearchResults, SearchStats},
};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Tier classification used for confidence adjustments during seeding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationTier {
    /// Hot tier - recently accessed, high confidence
    Hot,
    /// Warm tier - moderately accessed, medium confidence
    Warm,
    /// Cold tier - rarely accessed, lower confidence
    Cold,
}

impl ActivationTier {
    const fn weight(self) -> u8 {
        match self {
            Self::Hot => 3,
            Self::Warm => 2,
            Self::Cold => 1,
        }
    }

    /// Returns the confidence adjustment factor for this tier
    #[must_use]
    pub const fn confidence_factor(self) -> f32 {
        match self {
            Self::Hot => 1.0,
            Self::Warm => 0.95,
            Self::Cold => 0.9,
        }
    }

    /// Merges two tiers, preferring the higher priority tier
    #[must_use]
    pub const fn merge(self, other: Self) -> Self {
        if self.weight() >= other.weight() {
            self
        } else {
            other
        }
    }
}

/// Error produced while seeding activation from similarity search
#[derive(Debug, Error)]
pub enum SeedingError {
    /// Cue does not contain an embedding vector
    #[error("cue does not contain an embedding vector")]
    MissingEmbedding,
    /// No embeddings available for seeding
    #[error("no embeddings available for seeding")]
    NoEmbeddings,
}

/// Seed produced from HNSW similarity search
#[derive(Debug, Clone)]
pub struct SeededActivation {
    /// ID of the memory to seed with activation
    pub memory_id: String,
    /// Initial activation level (0.0 to 1.0)
    pub activation: f32,
    /// Confidence in this activation seed
    pub confidence: Confidence,
    /// Storage tier of the memory
    pub tier: ActivationTier,
    /// Similarity score to the cue
    pub similarity: f32,
}

/// Outcome of a seeding run (activation seeds plus search stats)
#[derive(Debug, Clone)]
pub struct SeedingOutcome {
    /// List of activation seeds generated
    pub seeds: Vec<SeededActivation>,
    /// Search statistics from HNSW index
    pub stats: SearchStats,
}

/// Converts HNSW similarity search results into activation seeds
pub struct VectorActivationSeeder {
    index: Arc<CognitiveHnswIndex>,
    config: SimilarityConfig,
    aggregator: MultiCueAggregator,
    tier_resolver: Arc<dyn Fn(&str) -> ActivationTier + Send + Sync>,
}

impl VectorActivationSeeder {
    /// Creates a new seeder with custom tier resolver
    #[must_use]
    pub fn new(
        index: Arc<CognitiveHnswIndex>,
        config: SimilarityConfig,
        tier_resolver: Arc<dyn Fn(&str) -> ActivationTier + Send + Sync>,
    ) -> Self {
        Self {
            index,
            config,
            aggregator: MultiCueAggregator::new(),
            tier_resolver,
        }
    }

    /// Creates a new seeder with default hot tier for all memories
    #[must_use]
    pub fn with_default_resolver(index: Arc<CognitiveHnswIndex>, config: SimilarityConfig) -> Self {
        Self::new(index, config, Arc::new(|_| ActivationTier::Hot))
    }

    /// Returns the similarity configuration
    #[must_use]
    pub const fn config(&self) -> &SimilarityConfig {
        &self.config
    }

    /// Seeds activation from a single cue
    ///
    /// # Errors
    ///
    /// Returns `SeedingError::MissingEmbedding` when the provided cue lacks an embedding vector.
    pub fn seed_from_cue(&self, cue: &Cue) -> Result<SeedingOutcome, SeedingError> {
        match &cue.cue_type {
            CueType::Embedding { vector, threshold } => {
                let embeddings = vec![(vector, 1.0f32)];
                self.seed_embeddings(&embeddings, threshold.raw(), cue.max_results)
            }
            _ => Err(SeedingError::MissingEmbedding),
        }
    }

    /// Seeds activation from multiple cues using aggregation strategy
    ///
    /// # Errors
    ///
    /// Returns `SeedingError::NoEmbeddings` if the cues do not yield any embedding vectors.
    pub fn seed_from_multi_cue(
        &self,
        cues: &[Cue],
        strategy: CueAggregationStrategy,
    ) -> Result<SeedingOutcome, SeedingError> {
        let aggregated = self.aggregator.aggregate(cues, strategy);
        if aggregated.is_empty() {
            return Err(SeedingError::NoEmbeddings);
        }

        let mut views = Vec::with_capacity(aggregated.len());
        for AggregatedCue { embedding, weight } in &aggregated {
            views.push((embedding, *weight));
        }

        // Use the strictest threshold from provided cues to avoid over-activation
        let cue_threshold = cues
            .iter()
            .filter_map(|cue| match &cue.cue_type {
                CueType::Embedding { threshold, .. } => Some(threshold.raw()),
                _ => None,
            })
            .fold(0.0, f32::max);

        let max_results = cues.iter().map(|cue| cue.max_results).max().unwrap_or(64);
        self.seed_embeddings(&views, cue_threshold, max_results)
    }

    fn seed_embeddings(
        &self,
        embeddings: &[(&[f32; 768], f32)],
        cue_threshold: f32,
        max_results: usize,
    ) -> Result<SeedingOutcome, SeedingError> {
        if embeddings.is_empty() {
            return Err(SeedingError::NoEmbeddings);
        }

        let mut aggregate: HashMap<String, SeedAccumulator> = HashMap::new();
        let mut total_nodes = 0usize;
        let mut sum_thoroughness = 0.0f32;
        let mut min_approx = 1.0f32;
        let mut stats_count = 0usize;
        let mut last_ef = self.config.ef_search;

        for (embedding, weight) in embeddings {
            let effective_k = self.config.candidate_limit().min(max_results.max(1));
            let cue_conf = Confidence::from_raw(cue_threshold);
            let threshold_value = self.config.effective_threshold(cue_conf);
            let threshold_conf = Confidence::from_raw(threshold_value);
            let SearchResults { hits, stats } =
                self.index
                    .search_with_details(embedding, effective_k, threshold_conf);

            if hits.is_empty() {
                continue;
            }

            let similarities: Vec<f32> = hits.iter().map(SearchResult::similarity).collect();
            let activations = SimdActivationMapper::batch_sigmoid_activation(
                &similarities,
                self.config.clamped_temperature(),
                threshold_value,
            );

            for (index, hit) in hits.into_iter().enumerate() {
                let activation = activations.get(index).copied().unwrap_or_default();
                let similarity = similarities.get(index).copied().unwrap_or_default();
                let tier = (self.tier_resolver)(hit.memory_id.as_str());
                let search_quality = stats
                    .approximation_ratio
                    .mul_add(0.6, stats.thoroughness * 0.4);
                let tier_factor = tier.confidence_factor();
                let seed_activation = (activation * *weight).clamp(0.0, 1.0);
                if seed_activation < 0.01 {
                    continue;
                }

                let confidence_value = (hit.confidence.raw() * search_quality * tier_factor)
                    .max(seed_activation * 0.5)
                    .min(1.0);
                let entry = aggregate
                    .entry(hit.memory_id.clone())
                    .or_insert_with(|| SeedAccumulator::new(tier));

                entry.accumulate(seed_activation, confidence_value, similarity, tier);
            }

            total_nodes += stats.nodes_visited;
            sum_thoroughness += stats.thoroughness;
            min_approx = min_approx.min(stats.approximation_ratio);
            last_ef = stats.ef_used;
            stats_count += 1;
        }

        let mut seeds = Vec::with_capacity(aggregate.len());
        for (memory_id, acc) in aggregate {
            seeds.push(SeededActivation {
                memory_id,
                activation: acc.activation.min(1.0),
                confidence: Confidence::from_raw(acc.confidence.min(1.0)),
                tier: acc.tier,
                similarity: acc.similarity,
            });
        }

        seeds.sort_by(|a, b| b.activation.total_cmp(&a.activation));

        let mut stats = SearchStats::with_ef(last_ef);
        stats.nodes_visited = total_nodes;
        if stats_count > 0 {
            stats.approximation_ratio = min_approx;
            #[allow(clippy::cast_precision_loss)]
            let count = stats_count as f32;
            stats.thoroughness = (sum_thoroughness / count).min(1.0);
        } else {
            stats.approximation_ratio = 1.0;
            stats.thoroughness = 0.0;
        }

        Ok(SeedingOutcome { seeds, stats })
    }
}

#[derive(Debug)]
struct SeedAccumulator {
    activation: f32,
    confidence: f32,
    similarity: f32,
    tier: ActivationTier,
}

impl SeedAccumulator {
    const fn new(tier: ActivationTier) -> Self {
        Self {
            activation: 0.0,
            confidence: 0.0,
            similarity: 0.0,
            tier,
        }
    }

    fn accumulate(
        &mut self,
        activation: f32,
        confidence: f32,
        similarity: f32,
        tier: ActivationTier,
    ) {
        self.activation = (self.activation + activation).min(1.0);
        self.confidence = self.confidence.max(confidence);
        self.similarity = self.similarity.max(similarity);
        self.tier = self.tier.merge(tier);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Confidence, Cue, index::CognitiveHnswIndex};
    use std::sync::Arc;

    fn dummy_index() -> Arc<CognitiveHnswIndex> {
        Arc::new(CognitiveHnswIndex::new())
    }

    #[test]
    fn seeding_without_embedding_returns_error() {
        let seeder = VectorActivationSeeder::with_default_resolver(
            dummy_index(),
            SimilarityConfig::default(),
        );
        let cue = Cue::context("context".to_string(), None, None, Confidence::MEDIUM);
        let result = seeder.seed_from_cue(&cue);
        assert!(matches!(result, Err(SeedingError::MissingEmbedding)));
    }
}
