//! Integrated Cognitive Recall Pipeline
//!
//! Orchestrates vector-similarity activation seeding, tier-aware spreading activation,
//! confidence aggregation, and result ranking into a production-ready recall API.

use super::{
    ActivationError, ActivationResult, BreakerSettings, ConfidenceAggregator, NodeId,
    SpreadingCircuitBreaker,
    cycle_detector::CycleDetector,
    parallel::ParallelSpreadingEngine,
    seeding::{self, SeededActivation, VectorActivationSeeder},
    semantic_seeder::{SemanticActivationSeeder, SemanticError},
};
use crate::{Confidence, Cue, Episode, MemoryStore};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::{error, warn};

/// Metrics for cognitive recall operations
#[derive(Debug, Default)]
pub struct RecallMetrics {
    /// Total number of recall operations performed
    pub total_recalls: AtomicU64,
    /// Number of recalls using similarity mode
    pub similarity_mode_count: AtomicU64,
    /// Number of recalls using spreading mode
    pub spreading_mode_count: AtomicU64,
    /// Number of recalls using hybrid mode
    pub hybrid_mode_count: AtomicU64,
    /// Number of times spreading fallback was triggered
    pub fallbacks_total: AtomicU64,
    /// Number of times time budget was exceeded
    pub time_budget_violations: AtomicU64,
    /// Total activation mass across all recalls
    pub recall_activation_mass: AtomicU64,
    /// Number of seeding failures
    pub seeding_failures: AtomicU64,
    /// Number of spreading failures
    pub spreading_failures: AtomicU64,
}

impl RecallMetrics {
    /// Create a new metrics instance
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a recall operation
    pub fn record_recall(&self, mode: RecallMode) {
        self.total_recalls.fetch_add(1, Ordering::Relaxed);
        match mode {
            RecallMode::Similarity => {
                self.similarity_mode_count.fetch_add(1, Ordering::Relaxed);
            }
            RecallMode::Spreading => {
                self.spreading_mode_count.fetch_add(1, Ordering::Relaxed);
            }
            RecallMode::Hybrid => {
                self.hybrid_mode_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Record a fallback event
    pub fn record_fallback(&self) {
        self.fallbacks_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a time budget violation
    pub fn record_time_budget_violation(&self) {
        self.time_budget_violations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record activation mass
    pub fn record_activation_mass(&self, mass: f32) {
        let mass_u64 = (mass * 1000.0) as u64; // Store as fixed-point
        self.recall_activation_mass
            .fetch_add(mass_u64, Ordering::Relaxed);
    }

    /// Record a seeding failure
    pub fn record_seeding_failure(&self) {
        self.seeding_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a spreading failure
    pub fn record_spreading_failure(&self) {
        self.spreading_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Get fallback rate (0.0 to 1.0)
    #[must_use]
    pub fn fallback_rate(&self) -> f32 {
        let total = self.total_recalls.load(Ordering::Relaxed);
        let fallbacks = self.fallbacks_total.load(Ordering::Relaxed);
        if total > 0 {
            fallbacks as f32 / total as f32
        } else {
            0.0
        }
    }
}

/// Mode of recall operation
///
/// # Examples
///
/// ```
/// use engram_core::activation::RecallMode;
///
/// // For fast lookups with known queries
/// let similarity = RecallMode::Similarity;
///
/// // For exploratory, context-aware retrieval
/// let spreading = RecallMode::Spreading;
///
/// // For production: spreading with similarity fallback
/// let hybrid = RecallMode::Hybrid;  // Recommended
/// ```
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
///
/// # Examples
///
/// ```
/// use engram_core::activation::{RecallConfig, RecallMode};
/// use std::time::Duration;
///
/// // Production configuration with hybrid mode
/// let config = RecallConfig {
///     recall_mode: RecallMode::Hybrid,
///     time_budget: Duration::from_millis(10),  // P95 < 10ms
///     min_confidence: 0.15,
///     max_results: 20,
///     enable_recency_boost: true,
///     recency_boost_factor: 1.2,
///     recency_window: Duration::from_secs(3600),  // 1 hour
/// };
/// ```
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
            rank_score = f32::midpoint(rank_score, sim);
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
    /// Semantic seeder for text-based queries (optional)
    semantic_seeder: Option<Arc<SemanticActivationSeeder>>,
    /// Parallel spreading engine
    spreading_engine: Arc<ParallelSpreadingEngine>,
    /// Confidence aggregator reserved for future multi-path confidence calibration
    #[allow(dead_code)]
    confidence_aggregator: Arc<ConfidenceAggregator>,
    /// Cycle detector reserved for future graph cycle prevention
    #[allow(dead_code)]
    cycle_detector: Arc<CycleDetector>,
    /// Recall configuration
    config: RecallConfig,
    /// Metrics for monitoring recall operations
    metrics: Arc<RecallMetrics>,
    /// Circuit breaker guarding spreading activation
    breaker: Arc<SpreadingCircuitBreaker>,
    /// Optional temporal decay system (Milestone 4)
    decay_system: Option<Arc<crate::decay::BiologicalDecaySystem>>,
}

impl CognitiveRecall {
    /// Create a new cognitive recall instance
    #[must_use]
    pub fn new(
        vector_seeder: Arc<VectorActivationSeeder>,
        spreading_engine: Arc<ParallelSpreadingEngine>,
        confidence_aggregator: Arc<ConfidenceAggregator>,
        cycle_detector: Arc<CycleDetector>,
        config: RecallConfig,
    ) -> Self {
        let breaker = Arc::new(SpreadingCircuitBreaker::new(
            spreading_engine.metrics_handle(),
            &BreakerSettings::default(),
        ));

        Self {
            vector_seeder,
            semantic_seeder: None,
            spreading_engine,
            confidence_aggregator,
            cycle_detector,
            config,
            metrics: Arc::new(RecallMetrics::new()),
            breaker,
            decay_system: None,
        }
    }

    /// Get the recall configuration
    #[must_use]
    pub const fn config(&self) -> &RecallConfig {
        &self.config
    }

    /// Get metrics for this recall instance
    #[must_use]
    pub fn metrics(&self) -> &RecallMetrics {
        &self.metrics
    }

    /// Access the underlying spreading engine.
    #[must_use]
    pub fn spreading_engine(&self) -> Arc<ParallelSpreadingEngine> {
        Arc::clone(&self.spreading_engine)
    }

    /// Set the semantic seeder for text-based queries
    pub fn set_semantic_seeder(&mut self, seeder: Arc<SemanticActivationSeeder>) {
        self.semantic_seeder = Some(seeder);
    }

    /// Main recall method that orchestrates the pipeline
    pub fn recall(&self, cue: &Cue, store: &MemoryStore) -> ActivationResult<Vec<RankedMemory>> {
        let start_time = Instant::now();

        // Record recall operation
        self.metrics.record_recall(self.config.recall_mode);

        // Step 1: Seed activation from vector similarity
        let seeded_activations = match self.seed_from_cue(cue) {
            Ok(seeds) => seeds,
            Err(e) => {
                warn!(target: "engram::recall", error = ?e, "Failed to seed activation");
                self.metrics.record_seeding_failure();
                self.metrics.record_fallback();
                self.spreading_engine.get_metrics().record_fallback();
                self.breaker
                    .on_result(false, start_time.elapsed(), self.config.time_budget);
                return self.fallback_to_similarity(cue, store);
            }
        };

        if seeded_activations.is_empty() {
            return Ok(Vec::new());
        }

        // Check time budget
        if start_time.elapsed() > self.config.time_budget {
            warn!(target: "engram::recall", "Time budget exceeded during seeding");
            self.metrics.record_time_budget_violation();
            self.metrics.record_fallback();
            self.spreading_engine.get_metrics().record_fallback();
            self.breaker
                .on_result(false, start_time.elapsed(), self.config.time_budget);
            return Ok(self.rank_seeded_results(seeded_activations, store));
        }

        if !self.breaker.should_attempt() {
            warn!(
                target: "engram::recall",
                "Circuit breaker open - using similarity fallback"
            );
            self.metrics.record_fallback();
            self.spreading_engine.get_metrics().record_fallback();
            return Ok(self.rank_seeded_results(seeded_activations, store));
        }

        // Step 2: Spread activation through graph
        let spreading_results = match self.spread_activation(&seeded_activations) {
            Ok(results) => results,
            Err(e) => {
                error!(target: "engram::recall", error = ?e, "Spreading activation failed");
                self.metrics.record_spreading_failure();
                self.spreading_engine.get_metrics().record_spread_failure();
                self.metrics.record_fallback();
                self.spreading_engine.get_metrics().record_fallback();
                self.breaker
                    .on_result(false, start_time.elapsed(), self.config.time_budget);
                return Ok(self.rank_seeded_results(seeded_activations, store));
            }
        };

        // Check time budget - return early with seeded results if exceeded
        if start_time.elapsed() > self.config.time_budget {
            warn!(target: "engram::recall", "Time budget exceeded during spreading");
            self.metrics.record_time_budget_violation();
            self.metrics.record_fallback();
            self.spreading_engine.get_metrics().record_fallback();
            self.breaker
                .on_result(false, start_time.elapsed(), self.config.time_budget);
            return Ok(self.rank_seeded_results(seeded_activations, store));
        }

        let spread_latency = start_time.elapsed();
        self.breaker
            .on_result(true, spread_latency, self.config.time_budget);

        // Step 3: Aggregate confidence scores
        let aggregated_results = Self::aggregate_confidence(spreading_results, seeded_activations);

        // Step 4: Rank and filter results
        let ranked_results = self.rank_results(aggregated_results, store);

        // Record activation mass
        let total_activation: f32 = ranked_results.iter().map(|r| r.activation).sum();
        self.metrics.record_activation_mass(total_activation);

        // Record metrics
        let elapsed = start_time.elapsed();
        if elapsed > self.config.time_budget {
            warn!(
                target: "engram::recall::performance",
                elapsed_ms = elapsed.as_millis(),
                budget_ms = self.config.time_budget.as_millis(),
                "Recall exceeded time budget"
            );
            self.metrics.record_time_budget_violation();
        }

        Ok(ranked_results)
    }

    /// Seed activation from cue using vector similarity
    fn seed_from_cue(&self, cue: &Cue) -> Result<Vec<SeededActivation>, seeding::SeedingError> {
        let outcome = self.vector_seeder.seed_from_cue(cue);
        match outcome {
            Ok(outcome) => Ok(outcome.seeds),
            Err(seeding::SeedingError::MissingEmbedding) => {
                // No embedding provided; return empty seed set so caller can fallback
                Ok(Vec::new())
            }
            Err(err) => Err(err),
        }
    }

    /// Spread activation through the memory graph
    fn spread_activation(
        &self,
        seeds: &[SeededActivation],
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
                activation
                    .confidence
                    .load(std::sync::atomic::Ordering::Relaxed),
            );
            activation_map.insert(
                activation.memory_id.clone(),
                (
                    activation
                        .activation_level
                        .load(std::sync::atomic::Ordering::Relaxed),
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
        let now = Utc::now();

        let mut ranked: Vec<RankedMemory> = aggregated
            .into_iter()
            .filter_map(|(node_id, activation, confidence, similarity)| {
                // Try to retrieve the episode from store
                store.get_episode(&node_id).map(|episode| {
                    // Apply temporal decay lazily if decay system is configured
                    let final_confidence = if let Some(decay_system) = &self.decay_system {
                        let elapsed = now.signed_duration_since(episode.last_recall);
                        let elapsed_std = elapsed.to_std().unwrap_or_default();

                        decay_system.compute_decayed_confidence(
                            confidence,
                            elapsed_std,
                            u64::from(episode.recall_count),
                            episode.when,
                        )
                    } else {
                        confidence
                    };

                    RankedMemory::new(episode, activation, final_confidence, similarity, &self.config)
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
            ActivationError::InvalidConfig(format!("Seeding failed in fallback: {e:?}"))
        })?;
        Ok(self.rank_seeded_results(seeded_activations, store))
    }

    /// Rank seeded results without spreading
    fn rank_seeded_results(
        &self,
        seeds: Vec<SeededActivation>,
        store: &MemoryStore,
    ) -> Vec<RankedMemory> {
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

        ranked
    }

    /// Semantic recall from text query
    ///
    /// This method provides a natural language interface to the recall system.
    /// It converts the text query to an embedding using the semantic seeder,
    /// then follows the same pipeline as the vector-based recall.
    ///
    /// # Arguments
    ///
    /// * `query` - The text query from the user
    /// * `language` - Optional ISO 639-1 language code (e.g., "en", "es")
    /// * `store` - Memory store to retrieve episodes from
    ///
    /// # Returns
    ///
    /// Ranked list of memories matching the query
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Semantic seeder is not configured
    /// - Embedding generation fails
    /// - Spreading activation fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = recall.recall_semantic(
    ///     "automobile safety features",
    ///     Some("en"),
    ///     &store
    /// ).await?;
    /// ```
    pub async fn recall_semantic(
        &self,
        query: &str,
        language: Option<&str>,
        store: &MemoryStore,
    ) -> ActivationResult<Vec<RankedMemory>> {
        let start_time = Instant::now();

        // Check if semantic seeder is available
        let semantic_seeder = self.semantic_seeder.as_ref().ok_or_else(|| {
            ActivationError::InvalidConfig(
                "Semantic seeder not configured. Use set_semantic_seeder() first.".to_string(),
            )
        })?;

        // Record recall operation
        self.metrics.record_recall(self.config.recall_mode);

        // Step 1: Seed activation from text query
        let seeding_outcome = semantic_seeder
            .seed_from_query(
                query,
                language,
                self.config.min_confidence,
                self.config.max_results,
            )
            .await
            .map_err(|e| match e {
                SemanticError::EmbeddingFailed(err) => {
                    warn!(target: "engram::recall::semantic", error = ?err, "Embedding generation failed");
                    self.metrics.record_seeding_failure();
                    ActivationError::InvalidConfig(format!("Embedding generation failed: {err}"))
                }
                SemanticError::NoEmbeddings => {
                    warn!(target: "engram::recall::semantic", "No embeddings available");
                    self.metrics.record_seeding_failure();
                    ActivationError::InvalidConfig("No embeddings available".to_string())
                }
                SemanticError::SeedingFailed(err) => {
                    warn!(target: "engram::recall::semantic", error = ?err, "Seeding failed");
                    self.metrics.record_seeding_failure();
                    ActivationError::InvalidConfig(format!("Seeding failed: {err}"))
                }
            })?;

        let seeded_activations = seeding_outcome.seeds;

        if seeded_activations.is_empty() {
            return Ok(Vec::new());
        }

        // Check time budget after seeding
        if start_time.elapsed() > self.config.time_budget {
            warn!(target: "engram::recall::semantic", "Time budget exceeded during seeding");
            self.metrics.record_time_budget_violation();
            self.metrics.record_fallback();
            self.spreading_engine.get_metrics().record_fallback();
            self.breaker
                .on_result(false, start_time.elapsed(), self.config.time_budget);
            return Ok(self.rank_seeded_results(seeded_activations, store));
        }

        // Check circuit breaker
        if !self.breaker.should_attempt() {
            warn!(
                target: "engram::recall::semantic",
                "Circuit breaker open - using similarity fallback"
            );
            self.metrics.record_fallback();
            self.spreading_engine.get_metrics().record_fallback();
            return Ok(self.rank_seeded_results(seeded_activations, store));
        }

        // Step 2: Spread activation through graph
        let spreading_results = match self.spread_activation(&seeded_activations) {
            Ok(results) => results,
            Err(e) => {
                error!(target: "engram::recall::semantic", error = ?e, "Spreading activation failed");
                self.metrics.record_spreading_failure();
                self.spreading_engine.get_metrics().record_spread_failure();
                self.metrics.record_fallback();
                self.spreading_engine.get_metrics().record_fallback();
                self.breaker
                    .on_result(false, start_time.elapsed(), self.config.time_budget);
                return Ok(self.rank_seeded_results(seeded_activations, store));
            }
        };

        // Check time budget after spreading
        if start_time.elapsed() > self.config.time_budget {
            warn!(target: "engram::recall::semantic", "Time budget exceeded during spreading");
            self.metrics.record_time_budget_violation();
            self.metrics.record_fallback();
            self.spreading_engine.get_metrics().record_fallback();
            self.breaker
                .on_result(false, start_time.elapsed(), self.config.time_budget);
            return Ok(self.rank_seeded_results(seeded_activations, store));
        }

        let spread_latency = start_time.elapsed();
        self.breaker
            .on_result(true, spread_latency, self.config.time_budget);

        // Step 3: Aggregate confidence scores
        let aggregated_results = Self::aggregate_confidence(spreading_results, seeded_activations);

        // Step 4: Rank and filter results
        let ranked_results = self.rank_results(aggregated_results, store);

        // Record activation mass
        let total_activation: f32 = ranked_results.iter().map(|r| r.activation).sum();
        self.metrics.record_activation_mass(total_activation);

        // Record metrics
        let elapsed = start_time.elapsed();
        if elapsed > self.config.time_budget {
            warn!(
                target: "engram::recall::semantic::performance",
                elapsed_ms = elapsed.as_millis(),
                budget_ms = self.config.time_budget.as_millis(),
                "Semantic recall exceeded time budget"
            );
            self.metrics.record_time_budget_violation();
        }

        Ok(ranked_results)
    }
}

/// Builder for CognitiveRecall with sensible defaults
pub struct CognitiveRecallBuilder {
    vector_seeder: Option<Arc<VectorActivationSeeder>>,
    semantic_seeder: Option<Arc<SemanticActivationSeeder>>,
    spreading_engine: Option<Arc<ParallelSpreadingEngine>>,
    confidence_aggregator: Option<Arc<ConfidenceAggregator>>,
    cycle_detector: Option<Arc<CycleDetector>>,
    config: RecallConfig,
    metrics: Option<Arc<RecallMetrics>>,
    breaker: Option<Arc<SpreadingCircuitBreaker>>,
    decay_system: Option<Arc<crate::decay::BiologicalDecaySystem>>,
}

impl Default for CognitiveRecallBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveRecallBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            vector_seeder: None,
            semantic_seeder: None,
            spreading_engine: None,
            confidence_aggregator: None,
            cycle_detector: None,
            config: RecallConfig::default(),
            metrics: None,
            breaker: None,
            decay_system: None,
        }
    }

    /// Set the vector seeder
    #[must_use]
    pub fn vector_seeder(mut self, seeder: Arc<VectorActivationSeeder>) -> Self {
        self.vector_seeder = Some(seeder);
        self
    }

    /// Set the semantic seeder for text-based queries
    #[must_use]
    pub fn semantic_seeder(mut self, seeder: Arc<SemanticActivationSeeder>) -> Self {
        self.semantic_seeder = Some(seeder);
        self
    }

    /// Set custom metrics instance
    #[must_use]
    pub fn metrics(mut self, metrics: Arc<RecallMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Configure the spreading circuit breaker
    #[must_use]
    pub fn breaker(mut self, breaker: Arc<SpreadingCircuitBreaker>) -> Self {
        self.breaker = Some(breaker);
        self
    }

    /// Set temporal decay system (Milestone 4)
    ///
    /// Enables lazy temporal decay during recall operations. Decay is computed
    /// at query time based on time since last access, not via background threads.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use engram_core::decay::BiologicalDecaySystem;
    ///
    /// let decay_system = Arc::new(BiologicalDecaySystem::default());
    /// let recall = CognitiveRecallBuilder::new()
    ///     .decay_system(decay_system)
    ///     .build()?;
    /// ```
    #[must_use]
    pub fn decay_system(mut self, system: Arc<crate::decay::BiologicalDecaySystem>) -> Self {
        self.decay_system = Some(system);
        self
    }

    /// Set the spreading engine
    #[must_use]
    pub fn spreading_engine(mut self, engine: Arc<ParallelSpreadingEngine>) -> Self {
        self.spreading_engine = Some(engine);
        self
    }

    /// Set the confidence aggregator
    #[must_use]
    pub fn confidence_aggregator(mut self, aggregator: Arc<ConfidenceAggregator>) -> Self {
        self.confidence_aggregator = Some(aggregator);
        self
    }

    /// Set the cycle detector
    #[must_use]
    pub fn cycle_detector(mut self, detector: Arc<CycleDetector>) -> Self {
        self.cycle_detector = Some(detector);
        self
    }

    /// Set the recall configuration
    #[must_use]
    pub const fn config(mut self, config: RecallConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the recall mode
    #[must_use]
    pub const fn recall_mode(mut self, mode: RecallMode) -> Self {
        self.config.recall_mode = mode;
        self
    }

    /// Set the time budget
    #[must_use]
    pub const fn time_budget(mut self, budget: Duration) -> Self {
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
        let metrics = self
            .metrics
            .unwrap_or_else(|| Arc::new(RecallMetrics::new()));

        let breaker = self.breaker.unwrap_or_else(|| {
            Arc::new(SpreadingCircuitBreaker::new(
                spreading_engine.metrics_handle(),
                &BreakerSettings::default(),
            ))
        });

        Ok(CognitiveRecall {
            vector_seeder,
            semantic_seeder: self.semantic_seeder,
            spreading_engine,
            confidence_aggregator,
            cycle_detector,
            config: self.config,
            metrics,
            breaker,
            decay_system: self.decay_system,
        })
    }
}
