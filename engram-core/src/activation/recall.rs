//! Integrated Cognitive Recall Pipeline
//!
//! Orchestrates vector-similarity activation seeding, tier-aware spreading activation,
//! confidence aggregation, and result ranking into a production-ready recall API.

use super::{
    cycle_detector::CycleDetector,
    parallel::ParallelSpreadingEngine,
    seeding::{self, SeededActivation, VectorActivationSeeder},
    ActivationError, ActivationResult, ConfidenceAggregator, NodeId,
};
use crate::{Confidence, Cue, Episode, MemoryStore};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{error, warn};

/// Mode of recall operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallMode {
    /// Use only vector similarity for recall
    Similarity,
    /// Use activation spreading for recall
    Spreading,
    /// Use both methods with fallback
    Hybrid,
}

impl Default for RecallMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

/// Configuration for cognitive recall
#[derive(Debug, Clone)]
pub struct RecallConfig {
    /// Recall mode to use
    pub recall_mode: RecallMode,
    /// Maximum time budget for recall operation
    pub time_budget: Duration,
    /// Minimum confidence threshold for results
    pub min_confidence: f32,
    /// Maximum number of results to return
    pub max_results: usize,
    /// Enable recency boosting
    pub enable_recency_boost: bool,
    /// Recency boost factor (1.0 = no boost, 2.0 = double weight for recent)
    pub recency_boost_factor: f32,
    /// Time window for recency boost (memories within this window get boosted)
    pub recency_window: Duration,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            recall_mode: RecallMode::default(),
            time_budget: Duration::from_millis(10), // 10ms P95 target
            min_confidence: 0.1,
            max_results: 100,
            enable_recency_boost: true,
            recency_boost_factor: 1.2,
            recency_window: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Ranked memory result with activation and confidence scores
#[derive(Debug, Clone)]
pub struct RankedMemory {
    /// The recalled episode
    pub episode: Episode,
    /// Activation level from spreading
    pub activation: f32,
    /// Aggregated confidence score
    pub confidence: Confidence,
    /// Similarity score if available
    pub similarity: Option<f32>,
    /// Recency boost applied
    pub recency_boost: f32,
    /// Final ranking score
    pub rank_score: f32,
}

impl RankedMemory {
    /// Calculate recency boost for an episode
    fn calculate_recency_boost(episode: &Episode, config: &RecallConfig) -> f32 {
        if !config.enable_recency_boost {
            return 1.0;
        }

        let age = Utc::now().signed_duration_since(episode.when);
        let window_secs = config.recency_window.as_secs() as i64;

        if age.num_seconds() < window_secs {
            // Linear decay within window
            let ratio = 1.0 - (age.num_seconds() as f32 / window_secs as f32);
            1.0 + (config.recency_boost_factor - 1.0) * ratio
        } else {
            1.0
        }
    }

    /// Create a ranked memory from components
    fn new(
        episode: Episode,
        activation: f32,
        confidence: Confidence,
        similarity: Option<f32>,
        config: &RecallConfig,
    ) -> Self {
        let recency_boost = Self::calculate_recency_boost(&episode, config);

        // Calculate final rank score combining all factors
        let mut rank_score = activation * confidence.raw();
        if let Some(sim) = similarity {
            rank_score = (rank_score + sim) / 2.0;
        }
        rank_score *= recency_boost;

        Self {
            episode,
            activation,
            confidence,
            similarity,
            recency_boost,
            rank_score,
        }
    }
}

/// Cognitive recall facade that orchestrates the recall pipeline
pub struct CognitiveRecall {
    /// Vector seeder for initial activation
    vector_seeder: Arc<VectorActivationSeeder>,
    /// Parallel spreading engine
    spreading_engine: Arc<ParallelSpreadingEngine>,
    /// Confidence aggregator
    confidence_aggregator: Arc<ConfidenceAggregator>,
    /// Cycle detector
    cycle_detector: Arc<CycleDetector>,
    /// Recall configuration
    config: RecallConfig,
}

impl CognitiveRecall {
    /// Create a new cognitive recall instance
    pub fn new(
        vector_seeder: Arc<VectorActivationSeeder>,
        spreading_engine: Arc<ParallelSpreadingEngine>,
        confidence_aggregator: Arc<ConfidenceAggregator>,
        cycle_detector: Arc<CycleDetector>,
        config: RecallConfig,
    ) -> Self {
        Self {
            vector_seeder,
            spreading_engine,
            confidence_aggregator,
            cycle_detector,
            config,
        }
    }

    /// Get the recall configuration
    #[must_use]
    pub fn config(&self) -> &RecallConfig {
        &self.config
    }

    /// Main recall method that orchestrates the pipeline
    pub fn recall(
        &self,
        cue: &Cue,
        store: &MemoryStore,
    ) -> ActivationResult<Vec<RankedMemory>> {
        let start_time = Instant::now();

        // Step 1: Seed activation from vector similarity
        let seeded_activations = match self.seed_from_cue(cue) {
            Ok(seeds) => seeds,
            Err(e) => {
                warn!(target: "engram::recall", error = ?e, "Failed to seed activation");
                return self.fallback_to_similarity(cue, store);
            }
        };

        if seeded_activations.is_empty() {
            return Ok(Vec::new());
        }

        // Check time budget
        if start_time.elapsed() > self.config.time_budget {
            warn!(target: "engram::recall", "Time budget exceeded during seeding");
            return self.rank_seeded_results(seeded_activations, store);
        }

        // Step 2: Spread activation through graph
        let spreading_results = match self.spread_activation(seeded_activations.clone()) {
            Ok(results) => results,
            Err(e) => {
                error!(target: "engram::recall", error = ?e, "Spreading activation failed");
                return self.rank_seeded_results(seeded_activations, store);
            }
        };

        // Check time budget - return early with seeded results if exceeded
        if start_time.elapsed() > self.config.time_budget {
            warn!(target: "engram::recall", "Time budget exceeded during spreading");
            return self.rank_seeded_results(seeded_activations, store);
        }

        // Step 3: Aggregate confidence scores
        let aggregated_results = self.aggregate_confidence(spreading_results, seeded_activations);

        // Step 4: Rank and filter results
        let ranked_results = self.rank_results(aggregated_results, store);

        // Record metrics
        let elapsed = start_time.elapsed();
        if elapsed > self.config.time_budget {
            warn!(
                target: "engram::recall::performance",
                elapsed_ms = elapsed.as_millis(),
                budget_ms = self.config.time_budget.as_millis(),
                "Recall exceeded time budget"
            );
        }

        Ok(ranked_results)
    }

    /// Seed activation from cue using vector similarity
    fn seed_from_cue(
        &self,
        cue: &Cue,
    ) -> Result<Vec<SeededActivation>, seeding::SeedingError> {
        let outcome = self.vector_seeder.seed_from_cue(cue)?;
        Ok(outcome.seeds)
    }

    /// Spread activation through the memory graph
    fn spread_activation(
        &self,
        seeds: Vec<SeededActivation>,
    ) -> ActivationResult<HashMap<NodeId, (f32, Confidence)>> {
        // Convert seeds to format expected by spreading engine
        let seed_nodes: Vec<(NodeId, f32)> = seeds
            .iter()
            .map(|s| (s.memory_id.clone(), s.activation))
            .collect();

        // Run spreading
        let results = self.spreading_engine.spread_activation(&seed_nodes)?;

        // Convert results to hashmap
        let mut activation_map = HashMap::new();
        for activation in results.activations {
            let confidence = Confidence::from_raw(
                activation.confidence.load(std::sync::atomic::Ordering::Relaxed),
            );
            activation_map.insert(
                activation.memory_id.clone(),
                (
                    activation.activation_level.load(std::sync::atomic::Ordering::Relaxed),
                    confidence,
                ),
            );
        }

        Ok(activation_map)
    }

    /// Aggregate confidence from multiple sources
    ///
    /// Note: This method does simple confidence boosting rather than using the full
    /// ConfidenceAggregator because spreading_results don't include path information
    /// (hop counts, tiers, weights) needed for proper path aggregation. The aggregator
    /// is designed for scenarios where multiple convergent paths with different characteristics
    /// arrive at the same node. Here we only have final activation and confidence values.
    fn aggregate_confidence(
        &self,
        spreading_results: HashMap<NodeId, (f32, Confidence)>,
        seeds: Vec<SeededActivation>,
    ) -> Vec<(String, f32, Confidence, Option<f32>)> {
        let mut aggregated = Vec::new();

        // Create a map of seed similarities for lookup
        let seed_map: HashMap<String, f32> = seeds
            .into_iter()
            .map(|s| (s.memory_id, s.similarity))
            .collect();

        // Combine spreading results with seed similarities
        for (node_id, (activation, confidence)) in spreading_results {
            let similarity = seed_map.get(&node_id).copied();

            // Boost confidence if this was a seed node
            let final_confidence = if similarity.is_some() {
                // Small boost for seed nodes (10% increase)
                Confidence::from_raw((confidence.raw() * 1.1).min(1.0))
            } else {
                confidence
            };

            aggregated.push((node_id, activation, final_confidence, similarity));
        }

        aggregated
    }

    /// Rank and filter results
    fn rank_results(
        &self,
        aggregated: Vec<(String, f32, Confidence, Option<f32>)>,
        store: &MemoryStore,
    ) -> Vec<RankedMemory> {
        let mut ranked: Vec<RankedMemory> = aggregated
            .into_iter()
            .filter_map(|(node_id, activation, confidence, similarity)| {
                // Try to retrieve the episode from store
                store
                    .get_episode(&node_id)
                    .map(|episode| {
                        RankedMemory::new(
                            episode,
                            activation,
                            confidence,
                            similarity,
                            &self.config,
                        )
                    })
            })
            .filter(|r| r.confidence.raw() >= self.config.min_confidence)
            .collect();

        // Sort by rank score (highest first)
        ranked.sort_by(|a, b| {
            b.rank_score
                .partial_cmp(&a.rank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        ranked.truncate(self.config.max_results);

        ranked
    }

    /// Fallback to similarity-based recall using vector seeder only
    fn fallback_to_similarity(
        &self,
        cue: &Cue,
        store: &MemoryStore,
    ) -> ActivationResult<Vec<RankedMemory>> {
        // Use vector seeder to get similar memories without spreading
        let seeded_activations = self.seed_from_cue(cue).map_err(|e| {
            ActivationError::InvalidConfig(format!("Seeding failed in fallback: {:?}", e))
        })?;
        self.rank_seeded_results(seeded_activations, store)
    }

    /// Rank seeded results without spreading
    fn rank_seeded_results(
        &self,
        seeds: Vec<SeededActivation>,
        store: &MemoryStore,
    ) -> ActivationResult<Vec<RankedMemory>> {
        let mut ranked: Vec<RankedMemory> = seeds
            .into_iter()
            .filter_map(|seed| {
                store.get_episode(&seed.memory_id).map(|episode| {
                    RankedMemory::new(
                        episode,
                        seed.activation,
                        seed.confidence,
                        Some(seed.similarity),
                        &self.config,
                    )
                })
            })
            .collect();

        ranked.sort_by(|a, b| {
            b.rank_score
                .partial_cmp(&a.rank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ranked.truncate(self.config.max_results);

        Ok(ranked)
    }
}

/// Builder for CognitiveRecall with sensible defaults
pub struct CognitiveRecallBuilder {
    vector_seeder: Option<Arc<VectorActivationSeeder>>,
    spreading_engine: Option<Arc<ParallelSpreadingEngine>>,
    confidence_aggregator: Option<Arc<ConfidenceAggregator>>,
    cycle_detector: Option<Arc<CycleDetector>>,
    config: RecallConfig,
}

impl Default for CognitiveRecallBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveRecallBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            vector_seeder: None,
            spreading_engine: None,
            confidence_aggregator: None,
            cycle_detector: None,
            config: RecallConfig::default(),
        }
    }

    /// Set the vector seeder
    pub fn vector_seeder(mut self, seeder: Arc<VectorActivationSeeder>) -> Self {
        self.vector_seeder = Some(seeder);
        self
    }

    /// Set the spreading engine
    pub fn spreading_engine(mut self, engine: Arc<ParallelSpreadingEngine>) -> Self {
        self.spreading_engine = Some(engine);
        self
    }

    /// Set the confidence aggregator
    pub fn confidence_aggregator(mut self, aggregator: Arc<ConfidenceAggregator>) -> Self {
        self.confidence_aggregator = Some(aggregator);
        self
    }

    /// Set the cycle detector
    pub fn cycle_detector(mut self, detector: Arc<CycleDetector>) -> Self {
        self.cycle_detector = Some(detector);
        self
    }

    /// Set the recall configuration
    pub fn config(mut self, config: RecallConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the recall mode
    pub fn recall_mode(mut self, mode: RecallMode) -> Self {
        self.config.recall_mode = mode;
        self
    }

    /// Set the time budget
    pub fn time_budget(mut self, budget: Duration) -> Self {
        self.config.time_budget = budget;
        self
    }

    /// Build the CognitiveRecall instance
    pub fn build(self) -> Result<CognitiveRecall, &'static str> {
        let vector_seeder = self.vector_seeder.ok_or("Vector seeder required")?;
        let spreading_engine = self.spreading_engine.ok_or("Spreading engine required")?;
        let confidence_aggregator = self
            .confidence_aggregator
            .ok_or("Confidence aggregator required")?;
        let cycle_detector = self.cycle_detector.ok_or("Cycle detector required")?;

        Ok(CognitiveRecall::new(
            vector_seeder,
            spreading_engine,
            confidence_aggregator,
            cycle_detector,
            self.config,
        ))
    }
}