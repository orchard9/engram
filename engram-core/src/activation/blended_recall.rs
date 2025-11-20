//! Blended Recall with Dual-Process Integration
//!
//! Implements cognitively-inspired blended recall that combines fast episodic (System 1)
//! and slower semantic (System 2) memory sources, with provenance tracking, confidence
//! calibration, and adaptive blending strategies based on dual-process theories.
//!
//! # Cognitive Foundations
//!
//! ## Dual-Process Theory (Kahneman 2011, Evans 2008)
//! - **System 1 (Fast)**: Episodic recall via spreading activation through direct associations
//!   - Latency: ~100-300ms for cached memories
//!   - Automatic, parallel, associative
//!   - High confidence when direct match exists
//!
//! - **System 2 (Slow)**: Semantic reasoning via concept hierarchies and pattern completion
//!   - Latency: ~300-1000ms due to hierarchical traversal
//!   - Controlled, serial, rule-based
//!   - Useful for partial cues and generalization
//!
//! ## Complementary Learning Systems (McClelland et al. 1995)
//! - Episodic system provides specific instances (hippocampal pattern separation)
//! - Semantic system provides generalized schemas (neocortical pattern completion)
//! - Blending enables both retrieval of specific memories AND generalization from concepts
//! - Interaction between systems improves both accuracy and robustness

use super::{ActivationResult, CognitiveRecall, RankedMemory};
#[cfg(feature = "dual_memory_types")]
use crate::memory_graph::BindingIndex;
use crate::{Confidence, Cue, CueType, MemoryStore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for blended recall with adaptive weighting
#[derive(Debug, Clone)]
pub struct BlendedRecallConfig {
    /// Initial episodic pathway weight (System 1 baseline)
    pub base_episodic_weight: f32,

    /// Initial semantic pathway weight (System 2 baseline)
    pub base_semantic_weight: f32,

    /// Enable adaptive weight adjustment based on pathway performance
    pub adaptive_weighting: bool,

    /// Enable pattern completion from concepts for low-confidence episodic recall
    pub enable_pattern_completion: bool,

    /// Minimum concept coherence to trust semantic pathway (0.0-1.0)
    pub min_concept_coherence: f32,

    /// Time budget for semantic pathway before timeout
    pub semantic_timeout: Duration,

    /// Confidence threshold below which to attempt pattern completion
    pub completion_threshold: Confidence,

    /// Maximum number of concepts to retrieve for semantic pathway
    pub max_concepts: usize,

    /// Blending mode strategy
    pub blend_mode: BlendMode,
}

impl Default for BlendedRecallConfig {
    fn default() -> Self {
        Self {
            base_episodic_weight: 0.7,
            base_semantic_weight: 0.3,
            adaptive_weighting: true,
            enable_pattern_completion: true,
            min_concept_coherence: 0.6,
            semantic_timeout: Duration::from_millis(8),
            completion_threshold: Confidence::from_raw(0.4),
            max_concepts: 20,
            blend_mode: BlendMode::AdaptiveWeighted,
        }
    }
}

/// Blending strategy modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Use fixed episodic/semantic weights
    FixedWeights,

    /// Adjust weights based on pathway performance and timing
    AdaptiveWeighted,

    /// Use episodic only, fall back to semantic if insufficient results
    EpisodicPriority,

    /// Use semantic for generalization, episodic for specifics
    ComplementaryRoles,
}

/// Direction of memory recall pathway
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallPathway {
    /// Fast episodic pathway (System 1)
    Episodic,
    /// Slower semantic pathway (System 2)
    Semantic,
}

/// Detailed provenance showing recall pathway contributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallProvenance {
    /// Episode ID being tracked
    pub episode_id: String,

    /// Episodic pathway contribution (0.0-1.0)
    pub episodic_contribution: f32,

    /// Semantic pathway contribution (0.0-1.0)
    pub semantic_contribution: f32,

    /// Which concepts contributed to semantic pathway (if any)
    pub contributing_concepts: Vec<ConceptContribution>,

    /// Final blended source classification
    pub final_source: RecallSource,

    /// Pathway latencies for performance analysis
    pub episodic_latency_ms: f32,
    /// Semantic pathway latency if executed
    pub semantic_latency_ms: Option<f32>,
}

/// Concept contribution to semantic pathway
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptContribution {
    /// Concept UUID (stored as string for serialization)
    pub concept_id: String,
    /// Similarity of concept to cue
    pub similarity_to_cue: f32,
    /// Binding strength to episode
    pub binding_strength: f32,
    /// Concept coherence quality metric
    pub coherence: f32,
    /// Weight of this concept's contribution
    pub contribution_weight: f32,
}

/// Classification of memory source for recall result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecallSource {
    /// Retrieved purely from episodic pathway
    Episodic,

    /// Retrieved purely from semantic pathway
    Semantic,

    /// Blended from both pathways (convergent retrieval)
    Blended {
        /// Episodic weight as percentage (0-100)
        episodic_weight: u8,
        /// Semantic weight as percentage (0-100)
        semantic_weight: u8,
    },

    /// Pattern completed from concepts when episodic failed
    PatternCompleted,
}

/// Extended RankedMemory with blending metadata
#[derive(Debug, Clone)]
pub struct BlendedRankedMemory {
    /// Base ranked memory from existing system
    pub base: RankedMemory,

    /// Detailed provenance for this result
    pub provenance: RecallProvenance,

    /// Calibrated confidence accounting for pathway convergence
    pub blended_confidence: Confidence,

    /// Novelty score if pattern completed from concepts
    pub novelty_score: Option<f32>,
}

impl BlendedRankedMemory {
    /// Check if this result came from both pathways (high reliability signal)
    #[must_use]
    pub const fn is_convergent(&self) -> bool {
        matches!(self.provenance.final_source, RecallSource::Blended { .. })
    }

    /// Get total pathway contribution (should sum to ~1.0)
    #[must_use]
    pub fn total_contribution(&self) -> f32 {
        self.provenance.episodic_contribution + self.provenance.semantic_contribution
    }
}

/// Result of episodic pathway execution
struct EpisodicPathwayResult {
    results: Vec<RankedMemory>,
    latency: Duration,
    confidence: Confidence,
}

/// Result of semantic pathway execution
struct SemanticPathwayResult {
    /// Episode scores from concept-mediated retrieval (using episode ID strings)
    episode_scores: HashMap<String, f32>,
    /// Concept contributions per episode (using episode ID strings)
    concept_contributions: HashMap<String, Vec<ConceptContribution>>,
    /// Semantic pathway latency
    latency: Duration,
    /// Average concept coherence quality
    average_concept_coherence: f32,
}

/// Adaptive blend weights calculated from pathway performance
struct BlendWeights {
    episodic: f32,
    semantic: f32,
}

/// Metrics for blended recall operations
#[derive(Debug, Default)]
pub struct BlendedRecallMetrics {
    /// Total blended recalls performed
    pub total_recalls: std::sync::atomic::AtomicU64,
    /// Number of times semantic pathway timed out
    pub semantic_timeouts: std::sync::atomic::AtomicU64,
    /// Number of times concept quality was too low
    pub low_concept_quality: std::sync::atomic::AtomicU64,
    /// Number of pattern completion attempts
    pub pattern_completions: std::sync::atomic::AtomicU64,
    /// Number of convergent retrievals (both pathways)
    pub convergent_retrievals: std::sync::atomic::AtomicU64,
}

impl BlendedRecallMetrics {
    /// Record a semantic timeout event
    fn record_semantic_timeout(&self) {
        self.semantic_timeouts
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record low concept quality event
    #[allow(dead_code)] // Reserved for future semantic pathway integration
    fn record_low_concept_quality(&self) {
        self.low_concept_quality
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record pattern completion attempt
    fn record_pattern_completion(&self) {
        self.pattern_completions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record convergent retrieval event
    fn record_convergent_retrieval(&self) {
        self.convergent_retrievals
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a blended recall with result metadata
    fn record_blended_recall(
        &self,
        #[cfg_attr(not(feature = "dual_memory_types"), allow(unused_variables))] elapsed: Duration,
        #[cfg_attr(not(feature = "dual_memory_types"), allow(unused_variables))]
        episodic_latency: Duration,
        #[cfg_attr(not(feature = "dual_memory_types"), allow(unused_variables))]
        semantic_latency: Option<Duration>,
        results: &[BlendedRankedMemory],
    ) {
        self.total_recalls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Count convergent retrievals
        let convergent_count = results.iter().filter(|r| r.is_convergent()).count();
        self.convergent_retrievals.fetch_add(
            convergent_count as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Record global metrics for dual-memory monitoring
        #[cfg(feature = "dual_memory_types")]
        if let Some(metrics) = crate::metrics::metrics() {
            let elapsed_ms = elapsed.as_millis().try_into().unwrap_or(u64::MAX);
            let episodic_ms = episodic_latency.as_millis().try_into().unwrap_or(u64::MAX);

            metrics.record_gauge("engram_blended_recall_latency_ms", elapsed_ms as f64);
            metrics.record_gauge("engram_episodic_recall_latency_ms", episodic_ms as f64);

            if let Some(semantic_dur) = semantic_latency {
                let semantic_ms = semantic_dur.as_millis().try_into().unwrap_or(u64::MAX);
                metrics.record_gauge("engram_semantic_recall_latency_ms", semantic_ms as f64);
            }

            // Calculate recall quality metrics if results are non-empty
            if !results.is_empty() {
                let avg_confidence: f32 = results
                    .iter()
                    .map(|r| r.blended_confidence.raw())
                    .sum::<f32>()
                    / results.len() as f32;
                metrics.record_gauge("engram_recall_accuracy", f64::from(avg_confidence));

                // Precision = convergent / total
                let precision = convergent_count as f32 / results.len() as f32;
                metrics.record_gauge("engram_recall_precision", f64::from(precision));
            }
        }
    }
}

/// Blended recall engine combining episodic and semantic pathways
pub struct BlendedRecallEngine {
    /// Existing cognitive recall engine for episodic pathway
    cognitive_recall: Arc<CognitiveRecall>,

    /// Binding index for concept-episode lookups (optional, gated by dual_memory_types)
    /// NOTE: Currently unused - will be used when semantic pathway is implemented (Task D)
    #[cfg(feature = "dual_memory_types")]
    #[allow(dead_code)]
    binding_index: Arc<BindingIndex>,

    /// Configuration
    config: BlendedRecallConfig,

    /// Metrics
    metrics: Arc<BlendedRecallMetrics>,
}

impl BlendedRecallEngine {
    /// Create new blended recall engine
    #[must_use]
    #[cfg(feature = "dual_memory_types")]
    pub fn new(
        cognitive_recall: Arc<CognitiveRecall>,
        binding_index: Arc<BindingIndex>,
        config: BlendedRecallConfig,
    ) -> Self {
        Self {
            cognitive_recall,
            binding_index,
            config,
            metrics: Arc::new(BlendedRecallMetrics::default()),
        }
    }

    /// Create new blended recall engine (without binding index for non-dual-memory builds)
    #[must_use]
    #[cfg(not(feature = "dual_memory_types"))]
    pub fn new(
        cognitive_recall: Arc<CognitiveRecall>,
        _binding_index: (),
        config: BlendedRecallConfig,
    ) -> Self {
        Self {
            cognitive_recall,
            config,
            metrics: Arc::new(BlendedRecallMetrics::default()),
        }
    }

    /// Get metrics for this engine
    #[must_use]
    pub fn metrics(&self) -> Arc<BlendedRecallMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Main blended recall entrypoint
    pub fn recall_blended(
        &self,
        cue: &Cue,
        store: &MemoryStore,
    ) -> ActivationResult<Vec<BlendedRankedMemory>> {
        let start_time = Instant::now();
        let time_budget = self.cognitive_recall.config().time_budget;

        // Phase 1: Parallel pathway execution with timeout
        let (episodic_pathway, semantic_pathway) =
            self.execute_dual_pathways(cue, store, time_budget)?;

        // Phase 2: Adaptive weight calculation
        let blend_weights = self.calculate_adaptive_weights(
            &episodic_pathway,
            semantic_pathway.as_ref(),
            time_budget,
            start_time.elapsed(),
        );

        // Phase 3: Blend results with provenance tracking
        let blended_results = self.blend_with_provenance(
            &episodic_pathway,
            semantic_pathway.as_ref(),
            &blend_weights,
            store,
        );

        // Phase 4: Confidence calibration for blended results
        let mut calibrated_results = Self::calibrate_blended_confidence(blended_results);

        // Phase 5: Optional pattern completion if needed
        if self.should_attempt_completion(&calibrated_results) {
            self.metrics.record_pattern_completion();
            calibrated_results =
                Self::pattern_complete_from_concepts(&calibrated_results, cue, store);
        }

        // Record metrics
        let episodic_latency = Duration::from_millis(0); // Will be filled from results
        let semantic_latency = None; // Will be filled from results
        self.metrics.record_blended_recall(
            start_time.elapsed(),
            episodic_latency,
            semantic_latency,
            &calibrated_results,
        );

        Ok(calibrated_results)
    }

    /// Execute episodic and semantic pathways in parallel with timeout
    fn execute_dual_pathways(
        &self,
        cue: &Cue,
        store: &MemoryStore,
        time_budget: Duration,
    ) -> ActivationResult<(EpisodicPathwayResult, Option<SemanticPathwayResult>)> {
        let episodic_start = Instant::now();

        // Episodic pathway: Use existing CognitiveRecall
        let episodic_results = self.cognitive_recall.recall(cue, store)?;
        let episodic_latency = episodic_start.elapsed();

        let episodic_pathway = EpisodicPathwayResult {
            results: episodic_results.clone(),
            latency: episodic_latency,
            confidence: Self::calculate_pathway_confidence(&episodic_results),
        };

        // Check if we have time budget for semantic pathway
        let remaining_budget = time_budget.saturating_sub(episodic_latency);
        if remaining_budget < self.config.semantic_timeout {
            self.metrics.record_semantic_timeout();
            return Ok((episodic_pathway, None));
        }

        // Semantic pathway: Concept-mediated recall with timeout
        let semantic_pathway =
            Self::execute_semantic_pathway_with_timeout(cue, store, self.config.semantic_timeout);

        Ok((episodic_pathway, semantic_pathway))
    }

    /// Execute semantic pathway with timeout
    ///
    /// Implements concept-mediated episode retrieval using semantic similarity.
    /// This represents the neocortical semantic pathway (System 2) complementing
    /// the hippocampal episodic pathway (System 1).
    ///
    /// # Algorithm (Complementary Learning Systems Theory)
    ///
    /// 1. **Concept Search**: Find concepts similar to query embedding
    ///    - Represents neocortical pattern completion
    ///    - Uses centroid similarity (slow cortical learning)
    ///
    /// 2. **Binding Traversal**: Map concepts → episodes via binding strength
    ///    - Represents hippocampal-neocortical communication
    ///    - Weighted by concept coherence and binding strength
    ///
    /// 3. **Score Aggregation**: Combine multi-concept evidence
    ///    - Implements evidence accumulation across semantic space
    ///    - Biologically plausible weighted voting
    ///
    /// # Biological Plausibility
    ///
    /// - **Timing**: 8ms default timeout matches slower cortical processing
    /// - **Coherence gating**: Only high-quality concepts contribute (cortical reliability)
    /// - **Binding strength**: Models synaptic weight between hippocampus and cortex
    ///
    /// # Current Implementation Status
    ///
    /// This is a **stub implementation** that returns None because:
    /// - BindingIndex doesn't have direct access to concept embeddings
    /// - Requires DualMemoryBackend iteration (Task 009 follow-up)
    /// - Full implementation deferred to maintain separation of concerns
    ///
    /// **Proper Implementation Path**:
    /// 1. Add `backend: &dyn DualMemoryBackend` parameter to BlendedRecallEngine
    /// 2. Use `backend.iter_concepts()` to access concept embeddings
    /// 3. Compute cosine similarity for top-K concepts
    /// 4. Use `binding_index.get_bindings_from_concept()` for episode mapping
    ///
    /// **Why Not Implement Now**:
    /// - Requires significant API changes to pass backend through recall stack
    /// - Episodic pathway already works and provides baseline performance
    /// - Semantic pathway is optional enhancement, not blocking
    fn execute_semantic_pathway_with_timeout(
        cue: &Cue,
        _store: &MemoryStore,
        timeout: Duration,
    ) -> Option<SemanticPathwayResult> {
        let start = Instant::now();

        // Extract embedding from cue
        let CueType::Embedding {
            vector: cue_embedding,
            ..
        } = &cue.cue_type
        else {
            return None; // Semantic pathway requires embedding
        };

        // Check timeout before any work
        if start.elapsed() >= timeout {
            return None;
        }

        // STUB: Concept search requires backend access
        //
        // Full implementation would:
        // 1. Iterate concepts from DualDashMapBackend
        // 2. Compute cosine similarity: dot(cue, concept.centroid) / (norm(cue) * norm(centroid))
        // 3. Take top max_concepts by similarity
        // 4. For each concept, get bindings via binding_index.get_bindings_from_concept()
        // 5. Aggregate episode scores: score = similarity * coherence * binding_strength
        // 6. Group by episode_id and sum contributions
        //
        // Example pseudocode:
        // ```
        // let concepts = backend.iter_concepts()
        //     .map(|c| (c.id, cosine_similarity(cue_embedding, c.centroid), c.coherence))
        //     .sorted_by_similarity()
        //     .take(max_concepts);
        //
        // let mut episode_scores = HashMap::new();
        // for (concept_id, similarity, coherence) in concepts {
        //     let bindings = binding_index.get_bindings_from_concept(&concept_id);
        //     for binding in bindings {
        //         let score = similarity * coherence * binding.get_strength();
        //         episode_scores.entry(binding.episode_id)
        //             .and_modify(|s| *s += score)
        //             .or_insert(score);
        //     }
        // }
        // ```

        let _ = (cue_embedding, timeout, start);

        // Return None to indicate semantic pathway not executed
        // Blending logic will handle pure episodic results gracefully
        None
    }

    /// Calculate pathway confidence from results
    fn calculate_pathway_confidence(results: &[RankedMemory]) -> Confidence {
        if results.is_empty() {
            return Confidence::from_raw(0.0);
        }

        let avg_confidence: f32 =
            results.iter().map(|r| r.confidence.raw()).sum::<f32>() / results.len() as f32;

        Confidence::from_raw(avg_confidence)
    }

    /// Calculate adaptive weights based on pathway performance
    fn calculate_adaptive_weights(
        &self,
        episodic: &EpisodicPathwayResult,
        semantic: Option<&SemanticPathwayResult>,
        time_budget: Duration,
        _elapsed: Duration,
    ) -> BlendWeights {
        match self.config.blend_mode {
            BlendMode::FixedWeights => BlendWeights {
                episodic: self.config.base_episodic_weight,
                semantic: self.config.base_semantic_weight,
            },

            BlendMode::AdaptiveWeighted => {
                let mut episodic_weight = self.config.base_episodic_weight;
                let mut semantic_weight = self.config.base_semantic_weight;

                // Factor 1: Episodic pathway confidence
                if episodic.confidence.raw() > 0.8 {
                    episodic_weight *= 1.2; // Boost confident episodic
                } else if episodic.confidence.raw() < 0.3 {
                    semantic_weight *= 1.3; // Rely more on semantic when episodic weak
                }

                // Factor 2: Semantic pathway quality (if available)
                if let Some(sem) = &semantic {
                    if sem.average_concept_coherence < self.config.min_concept_coherence {
                        episodic_weight *= 1.5; // Don't trust low-quality concepts
                        semantic_weight *= 0.5;
                    }

                    // Factor 3: Timing penalty for slow semantic
                    let semantic_ratio = sem.latency.as_secs_f32() / time_budget.as_secs_f32();
                    if semantic_ratio > 0.6 {
                        semantic_weight *= 0.7; // Penalize slow semantic pathway
                    }
                } else {
                    // No semantic pathway - pure episodic
                    episodic_weight = 1.0;
                    semantic_weight = 0.0;
                }

                // Normalize to sum to 1.0
                let total = episodic_weight + semantic_weight;
                BlendWeights {
                    episodic: episodic_weight / total,
                    semantic: semantic_weight / total,
                }
            }

            BlendMode::EpisodicPriority => {
                // Use episodic unless it's weak, then semantic
                if episodic.confidence.raw() > 0.5 || semantic.is_none() {
                    BlendWeights {
                        episodic: 1.0,
                        semantic: 0.0,
                    }
                } else {
                    BlendWeights {
                        episodic: 0.3,
                        semantic: 0.7,
                    }
                }
            }

            BlendMode::ComplementaryRoles => {
                // Balance based on complementary strengths
                BlendWeights {
                    episodic: 0.6,
                    semantic: 0.4,
                }
            }
        }
    }

    /// Blend results with provenance tracking
    fn blend_with_provenance(
        &self,
        episodic: &EpisodicPathwayResult,
        semantic: Option<&SemanticPathwayResult>,
        weights: &BlendWeights,
        _store: &MemoryStore,
    ) -> Vec<BlendedRankedMemory> {
        let mut blended_results = Vec::new();

        // Create a map of episodic results by episode ID
        let episodic_map: HashMap<String, &RankedMemory> = episodic
            .results
            .iter()
            .map(|r| (r.episode.id.clone(), r))
            .collect();

        // Process episodic results
        for ranked_memory in &episodic.results {
            let episode_id = &ranked_memory.episode.id;

            // Check if this episode also appeared in semantic pathway
            let (semantic_contrib, contributing_concepts, final_source) = semantic
                .and_then(|sem| {
                    sem.episode_scores.get(episode_id).map(|score| {
                        let concepts = sem
                            .concept_contributions
                            .get(episode_id)
                            .cloned()
                            .unwrap_or_default();

                        (
                            *score,
                            concepts,
                            RecallSource::Blended {
                                episodic_weight: (weights.episodic * 100.0) as u8,
                                semantic_weight: (weights.semantic * 100.0) as u8,
                            },
                        )
                    })
                })
                .unwrap_or((0.0, Vec::new(), RecallSource::Episodic));

            // Track convergent retrievals
            if matches!(final_source, RecallSource::Blended { .. }) {
                self.metrics.record_convergent_retrieval();
            }

            let provenance = RecallProvenance {
                episode_id: episode_id.clone(),
                episodic_contribution: weights.episodic,
                semantic_contribution: if semantic_contrib > 0.0 {
                    weights.semantic
                } else {
                    0.0
                },
                contributing_concepts,
                final_source,
                episodic_latency_ms: episodic.latency.as_secs_f32() * 1000.0,
                semantic_latency_ms: semantic.as_ref().map(|s| s.latency.as_secs_f32() * 1000.0),
            };

            blended_results.push(BlendedRankedMemory {
                base: ranked_memory.clone(),
                provenance,
                blended_confidence: ranked_memory.confidence,
                novelty_score: None,
            });
        }

        // Add any semantic-only results if semantic pathway exists
        if let Some(sem) = semantic {
            for (episode_id, score) in &sem.episode_scores {
                // Skip if already in episodic results
                if episodic_map.contains_key(episode_id) {
                    continue;
                }

                // This is a semantic-only result
                let concepts = sem
                    .concept_contributions
                    .get(episode_id)
                    .cloned()
                    .unwrap_or_default();

                let provenance = RecallProvenance {
                    episode_id: episode_id.clone(),
                    episodic_contribution: 0.0,
                    semantic_contribution: weights.semantic,
                    contributing_concepts: concepts,
                    final_source: RecallSource::Semantic,
                    episodic_latency_ms: episodic.latency.as_secs_f32() * 1000.0,
                    semantic_latency_ms: Some(sem.latency.as_secs_f32() * 1000.0),
                };

                // Create a synthetic RankedMemory for semantic-only result
                // Note: This requires fetching the episode from store, which we skip for now
                // as it would require store.get_episode() which returns Option<Episode>
                let _ = (score, provenance);
            }
        }

        // Sort by rank score
        blended_results.sort_by(|a, b| {
            b.base
                .rank_score
                .partial_cmp(&a.base.rank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to max results
        let max_results = self.cognitive_recall.config().max_results;
        blended_results.truncate(max_results);

        blended_results
    }

    /// Calibrate confidence for blended results
    ///
    /// Convergent retrieval (both pathways agree) increases confidence.
    /// Divergent retrieval (only one pathway) decreases confidence.
    fn calibrate_blended_confidence(
        mut results: Vec<BlendedRankedMemory>,
    ) -> Vec<BlendedRankedMemory> {
        for result in &mut results {
            let base_confidence = result.base.confidence.raw();

            let calibrated = match result.provenance.final_source {
                RecallSource::Blended {
                    episodic_weight,
                    semantic_weight,
                } => {
                    // Convergent retrieval: Both pathways found this episode
                    // This is a strong reliability signal - boost confidence
                    let convergence_boost = 1.15;

                    // Weight by pathway balance (more balanced = more confident)
                    let balance = 1.0
                        - ((f32::from(episodic_weight) - f32::from(semantic_weight)).abs() / 100.0);
                    let balance_factor = 1.0 + (balance * 0.1);

                    (base_confidence * convergence_boost * balance_factor).min(1.0)
                }

                RecallSource::Episodic => {
                    // Pure episodic: Use base confidence (no adjustment)
                    base_confidence
                }

                RecallSource::Semantic => {
                    // Pure semantic: Slight penalty for lack of episodic confirmation
                    // But boost if concept coherence is high
                    let coherence_avg = if result.provenance.contributing_concepts.is_empty() {
                        0.5
                    } else {
                        result
                            .provenance
                            .contributing_concepts
                            .iter()
                            .map(|c| c.coherence)
                            .sum::<f32>()
                            / result.provenance.contributing_concepts.len() as f32
                    };

                    if coherence_avg > 0.8 {
                        base_confidence * 1.05 // High-quality concepts are reliable
                    } else {
                        base_confidence * 0.9 // Lower confidence without episodic support
                    }
                }

                RecallSource::PatternCompleted => {
                    // Pattern completion: Lower confidence, this is reconstruction
                    base_confidence * 0.7
                }
            };

            result.blended_confidence = Confidence::from_raw(calibrated);
        }

        results
    }

    /// Determine if pattern completion should be attempted
    fn should_attempt_completion(&self, results: &[BlendedRankedMemory]) -> bool {
        if !self.config.enable_pattern_completion {
            return false;
        }

        // Attempt completion if:
        // 1. Results are sparse (< 5 results)
        // 2. Top result confidence is low
        // 3. We have high-quality concepts available (checked during completion)

        results.len() < 5
            || results
                .first()
                .is_none_or(|r| r.blended_confidence.raw() < self.config.completion_threshold.raw())
    }

    /// Pattern completion from concepts when episodic recall is insufficient
    ///
    /// Implements CA3-inspired pattern completion using semantic concepts to
    /// reconstruct episodes that were not directly retrieved via episodic pathway.
    ///
    /// # Biological Basis (Hippocampal CA3 Auto-Association)
    ///
    /// Models the hippocampal CA3 region's ability to complete partial patterns:
    /// - **Recurrent collaterals**: CA3-CA3 connections enable pattern completion
    /// - **Pattern separation threshold**: DG filters input, CA3 completes patterns
    /// - **Confidence gating**: Only high-coherence concepts trigger completion
    ///
    /// # Algorithm
    ///
    /// 1. **Detect sparse results**: Trigger if <5 results or low confidence
    /// 2. **Find completion concepts**: High-coherence concepts matching cue
    /// 3. **Reconstruct episodes**: Traverse concept → episode bindings
    /// 4. **Confidence penalty**: Mark as PatternCompleted with reduced confidence
    ///
    /// # When Pattern Completion Occurs
    ///
    /// - Episodic pathway returns <5 results
    /// - Top result confidence < completion_threshold (default 0.4)
    /// - High-quality concepts available (coherence > min_concept_coherence)
    ///
    /// # Current Implementation Status
    ///
    /// This is a **stub implementation** that passes through existing results because:
    /// - Requires same backend access as semantic pathway
    /// - Depends on concept embedding search (not yet available)
    /// - Pattern completion is optional enhancement for edge cases
    ///
    /// **Full Implementation**:
    /// Would use same approach as semantic pathway stub to find concepts,
    /// then add new episodes not in existing results, marked with:
    /// - `provenance.final_source = RecallSource::PatternCompleted`
    /// - `blended_confidence` reduced by 30% (0.7x factor)
    /// - `novelty_score` set to indicate reconstruction
    fn pattern_complete_from_concepts(
        results: &[BlendedRankedMemory],
        cue: &Cue,
        _store: &MemoryStore,
    ) -> Vec<BlendedRankedMemory> {
        // Extract embedding from cue
        let CueType::Embedding {
            vector: cue_embedding,
            ..
        } = &cue.cue_type
        else {
            return results.to_vec();
        };

        // STUB: Pattern completion requires backend access
        //
        // Full implementation would:
        // 1. Check if completion needed (sparse results or low confidence)
        // 2. Find high-coherence concepts via backend.iter_concepts()
        // 3. Get episode bindings via binding_index.get_bindings_from_concept()
        // 4. Filter out episodes already in results
        // 5. Add new episodes with PatternCompleted provenance
        // 6. Apply confidence penalty (0.7x) to indicate reconstruction
        //
        // Example pseudocode:
        // ```
        // if results.len() < 5 || results[0].confidence < 0.4 {
        //     let high_coherence_concepts = backend.iter_concepts()
        //         .filter(|c| c.coherence > 0.8)
        //         .map(|c| (c.id, cosine_similarity(cue_embedding, c.centroid)))
        //         .sorted_by_similarity()
        //         .take(5);
        //
        //     let existing_ids: HashSet<_> = results.iter().map(|r| &r.base.episode.id).collect();
        //     let mut completed_results = results.to_vec();
        //
        //     for (concept_id, similarity) in high_coherence_concepts {
        //         let bindings = binding_index.get_bindings_from_concept(&concept_id);
        //         for binding in bindings {
        //             if !existing_ids.contains(&binding.episode_id.to_string()) {
        //                 // Fetch episode from store and create BlendedRankedMemory
        //                 // with PatternCompleted source and reduced confidence
        //             }
        //         }
        //     }
        //     return completed_results;
        // }
        // ```

        let _ = cue_embedding;

        // Return results unmodified (no pattern completion)
        results.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blend_mode_variants() {
        // Test that all blend modes are distinct
        assert_ne!(BlendMode::FixedWeights, BlendMode::AdaptiveWeighted);
        assert_ne!(BlendMode::EpisodicPriority, BlendMode::ComplementaryRoles);
    }

    #[test]
    fn test_default_config() {
        let config = BlendedRecallConfig::default();
        assert!((config.base_episodic_weight - 0.7).abs() < f32::EPSILON);
        assert!((config.base_semantic_weight - 0.3).abs() < f32::EPSILON);
        assert!(config.adaptive_weighting);
        assert!(config.enable_pattern_completion);
        assert!((config.min_concept_coherence - 0.6).abs() < f32::EPSILON);
        assert_eq!(config.blend_mode, BlendMode::AdaptiveWeighted);
    }

    #[test]
    fn test_recall_source_variants() {
        let episodic = RecallSource::Episodic;
        let semantic = RecallSource::Semantic;
        let blended = RecallSource::Blended {
            episodic_weight: 60,
            semantic_weight: 40,
        };
        let completed = RecallSource::PatternCompleted;

        assert_ne!(episodic, semantic);
        assert_ne!(episodic, blended);
        assert_ne!(episodic, completed);
    }

    #[test]
    fn test_metrics_initialization() {
        let metrics = BlendedRecallMetrics::default();
        assert_eq!(
            metrics
                .total_recalls
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            metrics
                .semantic_timeouts
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(
            metrics
                .low_concept_quality
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }
}
