# Task 009: Blended Recall with Dual-Process Integration

## Objective
Implement cognitively-inspired blended recall that combines fast episodic (System 1) and slower semantic (System 2) memory sources, with provenance tracking, confidence calibration, and adaptive blending strategies based on dual-process theories.

## Background

Human memory retrieval is not a single mechanism but a blend of multiple processes operating at different speeds:

### Dual-Process Theory (Kahneman, 2011; Evans, 2008)
- **System 1 (Fast)**: Episodic recall via spreading activation through direct associations
  - Latency: ~100-300ms for cached memories
  - Automatic, parallel, associative
  - High confidence when direct match exists
- **System 2 (Slow)**: Semantic reasoning via concept hierarchies and pattern completion
  - Latency: ~300-1000ms due to hierarchical traversal
  - Controlled, serial, rule-based
  - Useful for partial cues and generalization

### Complementary Learning Systems (McClelland et al., 1995)
- Episodic system provides specific instances (hippocampal pattern separation)
- Semantic system provides generalized schemas (neocortical pattern completion)
- Blending enables both retrieval of specific memories AND generalization from concepts
- Interaction between systems improves both accuracy and robustness

### Current Engram Architecture
The existing CognitiveRecall system (recall.rs) implements pure episodic recall:
1. Vector similarity seeding (VectorActivationSeeder)
2. Spreading activation through episode graph (ParallelSpreadingEngine)
3. Confidence aggregation and ranking
4. Returns Vec<RankedMemory> with episode, activation, confidence, similarity

With dual memory types (Milestone 17), we can enhance this to blend:
- Episodic pathways: Direct episode-to-episode spreading
- Semantic pathways: Episode→Concept→Episode spreading via generalization

## Requirements

### Core Functionality
1. Implement dual-pathway recall executing episodic and semantic searches in parallel
2. Design cognitively-plausible blending strategy with timing-aware weights
3. Track detailed provenance showing which results came from episodic vs semantic pathways
4. Implement adaptive blend weight optimization based on query characteristics
5. Add confidence calibration for blended results accounting for pathway convergence
6. Support graceful fallback to episodic-only when concept quality is insufficient

### Cognitive Validity
7. Model System 1/System 2 timing differences (fast episodic, slower semantic)
8. Implement pattern completion from concepts for partial/degraded cues
9. Validate against psychological data (priming effects, semantic generalization)
10. Ensure blended recall improves robustness without degrading precision

### Performance Constraints
11. Total recall latency budget: 15ms P95 (5ms overhead vs pure episodic)
12. Semantic pathway timeout: 8ms (allows fallback if slow)
13. Blending computation: <500μs
14. Memory overhead: <20% vs episodic-only recall

## Technical Specification

### Files to Create

#### `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/activation/blended_recall.rs`
Core blending logic with dual-process architecture:

### Files to Modify
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/activation/recall.rs` - Add BlendedRecallMode
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/activation/mod.rs` - Export blended_recall module

## Cognitive Architecture Design

### Dual-Process Recall Pathways

The blending strategy mirrors human memory's complementary systems:

**Episodic Pathway (System 1 - Fast)**
```
Cue → Vector Similarity → Direct Spreading → Episodes
      (HNSW)              (Hot tier cache)

Latency: 100-300ms
Strength: Direct associations, high precision
Weakness: Fails on partial cues, no generalization
```

**Semantic Pathway (System 2 - Slower)**
```
Cue → Concept Matching → Hierarchical Spreading → Bound Episodes
      (Centroid search)   (Multi-hop traversal)

Latency: 300-1000ms
Strength: Generalization, pattern completion, robustness
Weakness: Lower precision, higher latency
```

**Blending Strategy**
```
Blended Result = α·Episodic + (1-α)·Semantic

Where α adapts based on:
1. Cue completeness (complete cue → higher α)
2. Episodic confidence (high confidence → higher α)
3. Time budget remaining (tight budget → higher α for fast path)
4. Concept quality (low coherence → higher α, fallback to episodic)
```

### Timing-Aware Blending

Following dual-process theory, we model different latencies:

```rust
pub enum RecallPathway {
    Episodic {
        expected_latency_ms: f32,  // 100-300ms
        actual_latency_ms: f32,
        confidence: Confidence,
    },
    Semantic {
        expected_latency_ms: f32,  // 300-1000ms
        actual_latency_ms: f32,
        confidence: Confidence,
        concept_coherence: f32,    // Quality metric
    },
}

impl RecallPathway {
    /// Calculate pathway weight based on timing and quality
    fn pathway_weight(&self, time_budget: Duration) -> f32 {
        match self {
            Self::Episodic { actual_latency_ms, confidence, .. } => {
                // Fast pathway gets bonus for speed
                let speed_bonus = if *actual_latency_ms < 200.0 { 1.2 } else { 1.0 };
                confidence.raw() * speed_bonus
            }
            Self::Semantic { actual_latency_ms, confidence, concept_coherence, .. } => {
                // Slow pathway penalized if exceeds budget
                let time_penalty = if *actual_latency_ms > time_budget.as_millis() as f32 * 0.6 {
                    0.8 // 20% penalty for slow semantic pathway
                } else {
                    1.0
                };
                // Require high concept quality for semantic pathway to contribute
                confidence.raw() * concept_coherence * time_penalty
            }
        }
    }
}
```

### Blended Recall Configuration

```rust
/// Configuration for blended recall with adaptive weighting
#[derive(Debug, Clone)]
pub struct BlendedRecallConfig {
    /// Initial episodic pathway weight (System 1 baseline)
    pub base_episodic_weight: f32,  // Default 0.7

    /// Initial semantic pathway weight (System 2 baseline)
    pub base_semantic_weight: f32,  // Default 0.3

    /// Enable adaptive weight adjustment based on pathway performance
    pub adaptive_weighting: bool,  // Default true

    /// Enable pattern completion from concepts for low-confidence episodic recall
    pub enable_pattern_completion: bool,  // Default true

    /// Minimum concept coherence to trust semantic pathway (0.0-1.0)
    pub min_concept_coherence: f32,  // Default 0.6

    /// Time budget for semantic pathway before timeout
    pub semantic_timeout: Duration,  // Default 8ms

    /// Confidence threshold below which to attempt pattern completion
    pub completion_threshold: Confidence,  // Default 0.4

    /// Maximum number of concepts to retrieve for semantic pathway
    pub max_concepts: usize,  // Default 20

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
```

### Provenance Tracking for Explainability

```rust
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
    pub semantic_latency_ms: Option<f32>,  // None if semantic timed out
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptContribution {
    pub concept_id: String,
    pub similarity_to_cue: f32,
    pub binding_strength: f32,
    pub coherence: f32,
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
    Blended { episodic_weight: u8, semantic_weight: u8 },  // Percentages

    /// Pattern completed from concepts when episodic failed
    PatternCompleted,
}
```

### Blended Result Type Extension

```rust
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
    pub fn is_convergent(&self) -> bool {
        matches!(self.provenance.final_source, RecallSource::Blended { .. })
    }

    /// Get total pathway contribution (should sum to ~1.0)
    #[must_use]
    pub fn total_contribution(&self) -> f32 {
        self.provenance.episodic_contribution + self.provenance.semantic_contribution
    }
}
```

### Core Blending Implementation

```rust
pub struct BlendedRecallEngine {
    /// Existing cognitive recall engine for episodic pathway
    cognitive_recall: Arc<CognitiveRecall>,

    /// Binding index for concept-episode lookups
    binding_index: Arc<BindingIndex>,

    /// Hierarchical spreading for semantic pathway
    hierarchical_spreading: Arc<HierarchicalSpreading>,

    /// Configuration
    config: BlendedRecallConfig,

    /// Metrics
    metrics: Arc<BlendedRecallMetrics>,
}

impl BlendedRecallEngine {
    /// Main blended recall entrypoint
    pub fn recall_blended(
        &self,
        cue: &Cue,
        store: &MemoryStore,
    ) -> ActivationResult<Vec<BlendedRankedMemory>> {
        let start_time = Instant::now();
        let time_budget = self.cognitive_recall.config().time_budget;

        // Phase 1: Parallel pathway execution with timeout
        let (episodic_pathway, semantic_pathway) = self.execute_dual_pathways(cue, store, time_budget)?;

        // Phase 2: Adaptive weight calculation
        let blend_weights = self.calculate_adaptive_weights(
            &episodic_pathway,
            &semantic_pathway,
            time_budget,
            start_time.elapsed(),
        );

        // Phase 3: Blend results with provenance tracking
        let blended_results = self.blend_with_provenance(
            episodic_pathway,
            semantic_pathway,
            blend_weights,
            store,
        )?;

        // Phase 4: Confidence calibration for blended results
        let calibrated_results = self.calibrate_blended_confidence(blended_results);

        // Phase 5: Optional pattern completion if needed
        let final_results = if self.should_attempt_completion(&calibrated_results) {
            self.pattern_complete_from_concepts(&calibrated_results, cue, store)?
        } else {
            calibrated_results
        };

        // Record metrics
        self.metrics.record_blended_recall(
            start_time.elapsed(),
            episodic_pathway.latency,
            semantic_pathway.map(|p| p.latency),
            &final_results,
        );

        Ok(final_results)
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
            results: episodic_results,
            latency: episodic_latency,
            confidence: self.calculate_pathway_confidence(&episodic_results),
        };

        // Check if we have time budget for semantic pathway
        let remaining_budget = time_budget.saturating_sub(episodic_latency);
        if remaining_budget < self.config.semantic_timeout {
            self.metrics.record_semantic_timeout();
            return Ok((episodic_pathway, None));
        }

        // Semantic pathway: Concept-mediated recall with timeout
        let semantic_pathway = self.execute_semantic_pathway_with_timeout(
            cue,
            store,
            self.config.semantic_timeout,
        );

        Ok((episodic_pathway, semantic_pathway))
    }

    /// Semantic pathway: Find concepts, traverse bindings, reconstruct episodes
    fn execute_semantic_pathway_with_timeout(
        &self,
        cue: &Cue,
        store: &MemoryStore,
        timeout: Duration,
    ) -> Option<SemanticPathwayResult> {
        let start = Instant::now();

        // Extract embedding from cue
        let cue_embedding = match &cue.cue_type {
            CueType::Embedding { vector, .. } => vector,
            _ => return None,  // Semantic pathway requires embedding
        };

        // Step 1: Find relevant concepts via centroid similarity
        let concept_matches = self.binding_index.find_concepts_by_embedding(
            cue_embedding,
            self.config.max_concepts,
        ).ok()?;

        if concept_matches.is_empty() {
            return None;
        }

        // Step 2: Filter concepts by minimum coherence
        let quality_concepts: Vec<_> = concept_matches
            .into_iter()
            .filter(|(_, _, coherence)| *coherence >= self.config.min_concept_coherence)
            .collect();

        if quality_concepts.is_empty() {
            self.metrics.record_low_concept_quality();
            return None;
        }

        // Step 3: Traverse bindings from concepts to episodes
        let mut episode_scores: HashMap<String, f32> = HashMap::new();
        let mut concept_contributions: HashMap<String, Vec<ConceptContribution>> = HashMap::new();

        for (concept_id, similarity, coherence) in quality_concepts {
            if start.elapsed() > timeout {
                self.metrics.record_semantic_timeout();
                break;
            }

            // Get bindings from this concept
            let bindings = self.binding_index.get_bindings_from_concept(&concept_id).ok()?;

            for binding in bindings {
                let episode_id = &binding.episode_id;
                let binding_strength = binding.strength.load(Ordering::Relaxed);

                // Contribution = concept similarity * binding strength * coherence
                let contribution = similarity * binding_strength * coherence;

                episode_scores
                    .entry(episode_id.clone())
                    .and_modify(|score| *score += contribution)
                    .or_insert(contribution);

                concept_contributions
                    .entry(episode_id.clone())
                    .or_insert_with(Vec::new)
                    .push(ConceptContribution {
                        concept_id: concept_id.clone(),
                        similarity_to_cue: similarity,
                        binding_strength,
                        coherence,
                        contribution_weight: contribution,
                    });
            }
        }

        let latency = start.elapsed();

        Some(SemanticPathwayResult {
            episode_scores,
            concept_contributions,
            latency,
            confidence: self.calculate_semantic_confidence(&quality_concepts),
            average_concept_coherence: quality_concepts.iter()
                .map(|(_, _, c)| c)
                .sum::<f32>() / quality_concepts.len() as f32,
        })
    }
}
```

### Adaptive Weight Calculation

```rust
impl BlendedRecallEngine {
    /// Calculate adaptive weights based on pathway performance
    fn calculate_adaptive_weights(
        &self,
        episodic: &EpisodicPathwayResult,
        semantic: &Option<SemanticPathwayResult>,
        time_budget: Duration,
        elapsed: Duration,
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
                    episodic_weight *= 1.2;  // Boost confident episodic
                } else if episodic.confidence.raw() < 0.3 {
                    semantic_weight *= 1.3;  // Rely more on semantic when episodic weak
                }

                // Factor 2: Semantic pathway quality (if available)
                if let Some(sem) = semantic {
                    if sem.average_concept_coherence < self.config.min_concept_coherence {
                        episodic_weight *= 1.5;  // Don't trust low-quality concepts
                        semantic_weight *= 0.5;
                    }

                    // Factor 3: Timing penalty for slow semantic
                    let semantic_ratio = sem.latency.as_secs_f32() / time_budget.as_secs_f32();
                    if semantic_ratio > 0.6 {
                        semantic_weight *= 0.7;  // Penalize slow semantic pathway
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
            },

            BlendMode::EpisodicPriority => {
                // Use episodic unless it's weak, then semantic
                if episodic.confidence.raw() > 0.5 || semantic.is_none() {
                    BlendWeights { episodic: 1.0, semantic: 0.0 }
                } else {
                    BlendWeights { episodic: 0.3, semantic: 0.7 }
                }
            },

            BlendMode::ComplementaryRoles => {
                // Balance based on complementary strengths
                BlendWeights { episodic: 0.6, semantic: 0.4 }
            },
        }
    }
}
```

### Confidence Calibration for Blended Results

```rust
impl BlendedRecallEngine {
    /// Calibrate confidence for blended results
    ///
    /// Convergent retrieval (both pathways agree) increases confidence
    /// Divergent retrieval (only one pathway) decreases confidence
    fn calibrate_blended_confidence(
        &self,
        mut results: Vec<BlendedRankedMemory>,
    ) -> Vec<BlendedRankedMemory> {
        for result in &mut results {
            let base_confidence = result.base.confidence.raw();

            let calibrated = match result.provenance.final_source {
                RecallSource::Blended { episodic_weight, semantic_weight } => {
                    // Convergent retrieval: Both pathways found this episode
                    // This is a strong reliability signal - boost confidence
                    let convergence_boost = 1.15;

                    // Weight by pathway balance (more balanced = more confident)
                    let balance = 1.0 - ((episodic_weight as f32 - semantic_weight as f32).abs() / 100.0);
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
                    let coherence_avg = result.provenance.contributing_concepts
                        .iter()
                        .map(|c| c.coherence)
                        .sum::<f32>() / result.provenance.contributing_concepts.len() as f32;

                    if coherence_avg > 0.8 {
                        base_confidence * 1.05  // High-quality concepts are reliable
                    } else {
                        base_confidence * 0.9  // Lower confidence without episodic support
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
        // 3. We have high-quality concepts available

        results.len() < 5 ||
            results.first()
                .map(|r| r.blended_confidence.raw() < self.config.completion_threshold.raw())
                .unwrap_or(true)
    }

    /// Pattern completion from concepts when episodic recall is insufficient
    fn pattern_complete_from_concepts(
        &self,
        results: &[BlendedRankedMemory],
        cue: &Cue,
        store: &MemoryStore,
    ) -> ActivationResult<Vec<BlendedRankedMemory>> {
        // Use hierarchical spreading to find additional episodes via concepts
        // This implements pattern completion similar to hippocampal CA3 auto-association

        let cue_embedding = match &cue.cue_type {
            CueType::Embedding { vector, .. } => vector,
            _ => return Ok(results.to_vec()),
        };

        // Find high-coherence concepts matching the cue
        let concepts = self.binding_index
            .find_concepts_by_embedding(cue_embedding, 10)
            .unwrap_or_default();

        let high_coherence_concepts: Vec<_> = concepts
            .into_iter()
            .filter(|(_, _, coherence)| *coherence > 0.8)
            .collect();

        if high_coherence_concepts.is_empty() {
            return Ok(results.to_vec());
        }

        // Retrieve episodes bound to these high-quality concepts
        let mut completed_results = results.to_vec();

        for (concept_id, similarity, coherence) in high_coherence_concepts {
            let bindings = self.binding_index
                .get_bindings_from_concept(&concept_id)
                .unwrap_or_default();

            for binding in bindings {
                // Skip if we already have this episode
                if completed_results.iter().any(|r| r.base.episode.id == binding.episode_id) {
                    continue;
                }

                // Retrieve episode from store
                if let Some(episode) = store.get_episode(&binding.episode_id) {
                    let strength = binding.strength.load(Ordering::Relaxed);
                    let activation = similarity * strength * coherence;

                    // Create pattern-completed result with lower confidence
                    let base = RankedMemory::new(
                        episode,
                        activation,
                        Confidence::from_raw(0.5),
                        None,
                        self.cognitive_recall.config(),
                    );

                    let provenance = RecallProvenance {
                        episode_id: binding.episode_id.clone(),
                        episodic_contribution: 0.0,
                        semantic_contribution: 1.0,
                        contributing_concepts: vec![ConceptContribution {
                            concept_id: concept_id.clone(),
                            similarity_to_cue: similarity,
                            binding_strength: strength,
                            coherence,
                            contribution_weight: activation,
                        }],
                        final_source: RecallSource::PatternCompleted,
                        episodic_latency_ms: 0.0,
                        semantic_latency_ms: None,
                    };

                    completed_results.push(BlendedRankedMemory {
                        base,
                        provenance,
                        blended_confidence: Confidence::from_raw(0.4),  // Low confidence for completion
                        novelty_score: Some(0.8),  // High novelty - this is reconstruction
                    });
                }
            }
        }

        // Re-rank with pattern-completed results
        completed_results.sort_by(|a, b| {
            b.base.rank_score
                .partial_cmp(&a.base.rank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        completed_results.truncate(self.cognitive_recall.config().max_results);

        Ok(completed_results)
    }
}
```

## Implementation Notes

### Biological Plausibility
- **Dual pathways mirror hippocampal-neocortical interaction**: Episodic (hippocampus) provides specific instances, semantic (neocortex) provides schemas
- **Timing differences reflect neural architecture**: Direct associations are faster than multi-hop semantic traversal
- **Convergent retrieval boosts confidence**: When both pathways agree, this mirrors increased firing synchrony
- **Pattern completion from concepts**: Similar to CA3 auto-association for partial cue reconstruction

### Performance Optimization
- **Parallel pathway execution**: Episodic and semantic run concurrently with timeout
- **Early termination**: Skip semantic pathway if episodic latency exceeds budget
- **Concept quality filtering**: Only use concepts with coherence > threshold
- **Caching**: Concept centroids cached in binding index for fast lookup
- **SIMD centroid comparison**: Batch vector similarity for concept matching

### Fallback Strategy
When to fall back to episodic-only:
1. **Low concept quality**: Average coherence < 0.6
2. **Time-critical queries**: Semantic timeout would exceed budget
3. **High episodic confidence**: Episodic pathway already found good results
4. **No embeddings**: Text-only cues cannot use semantic pathway

### Integration with Existing Systems
- **CognitiveRecall**: Used directly for episodic pathway (no changes needed)
- **BindingIndex**: Provides concept→episode lookups (Task 005)
- **HierarchicalSpreading**: Enables multi-hop concept traversal (Task 008)
- **RankedMemory**: Extended with BlendedRankedMemory wrapper
- **RecallConfig**: Time budgets and thresholds reused

## Testing Approach

### Unit Tests (engram-core/src/activation/blended_recall.rs)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_weight_calculation() {
        // High episodic confidence should increase episodic weight
        let config = BlendedRecallConfig::default();
        let engine = create_test_engine(config);

        let episodic = EpisodicPathwayResult {
            confidence: Confidence::from_raw(0.9),
            latency: Duration::from_millis(2),
            results: vec![],
        };

        let semantic = Some(SemanticPathwayResult {
            confidence: Confidence::from_raw(0.6),
            average_concept_coherence: 0.7,
            latency: Duration::from_millis(5),
            episode_scores: HashMap::new(),
            concept_contributions: HashMap::new(),
        });

        let weights = engine.calculate_adaptive_weights(
            &episodic,
            &semantic,
            Duration::from_millis(10),
            Duration::from_millis(7),
        );

        assert!(weights.episodic > 0.7);  // Should be boosted
        assert!(weights.semantic < 0.3);  // Should be reduced
    }

    #[test]
    fn test_confidence_calibration_convergent() {
        // Convergent retrieval should boost confidence
        let result = BlendedRankedMemory {
            base: create_test_ranked_memory(Confidence::from_raw(0.7)),
            provenance: RecallProvenance {
                final_source: RecallSource::Blended {
                    episodic_weight: 60,
                    semantic_weight: 40,
                },
                episodic_contribution: 0.6,
                semantic_contribution: 0.4,
                ..Default::default()
            },
            blended_confidence: Confidence::from_raw(0.7),
            novelty_score: None,
        };

        let calibrated = engine.calibrate_blended_confidence(vec![result]);
        assert!(calibrated[0].blended_confidence.raw() > 0.7);  // Boosted
    }

    #[test]
    fn test_pattern_completion_threshold() {
        // Should attempt completion when results are sparse
        let sparse_results = vec![
            create_blended_result(Confidence::from_raw(0.3)),
            create_blended_result(Confidence::from_raw(0.2)),
        ];

        assert!(engine.should_attempt_completion(&sparse_results));

        // Should not attempt when results are good
        let good_results = vec![
            create_blended_result(Confidence::from_raw(0.9)),
            create_blended_result(Confidence::from_raw(0.8)),
            create_blended_result(Confidence::from_raw(0.7)),
            create_blended_result(Confidence::from_raw(0.6)),
            create_blended_result(Confidence::from_raw(0.5)),
        ];

        assert!(!engine.should_attempt_completion(&good_results));
    }

    #[test]
    fn test_fallback_to_episodic_only() {
        // Low concept quality should trigger episodic-only fallback
        let config = BlendedRecallConfig {
            min_concept_coherence: 0.6,
            ..Default::default()
        };

        let episodic = create_test_episodic_result();
        let semantic = Some(SemanticPathwayResult {
            average_concept_coherence: 0.3,  // Below threshold
            ..Default::default()
        });

        let weights = engine.calculate_adaptive_weights(&episodic, &semantic, ...);
        assert!(weights.episodic > 0.9);  // Almost pure episodic
        assert!(weights.semantic < 0.1);
    }

    #[test]
    fn test_provenance_tracking() {
        // Verify provenance correctly tracks concept contributions
        let result = engine.recall_blended(&cue, &store).unwrap();

        for memory in &result {
            // Contributions should sum to ~1.0
            let total = memory.provenance.episodic_contribution
                + memory.provenance.semantic_contribution;
            assert!((total - 1.0).abs() < 0.1);

            // Semantic contribution should have concept details
            if memory.provenance.semantic_contribution > 0.0 {
                assert!(!memory.provenance.contributing_concepts.is_empty());
            }
        }
    }
}
```

### Integration Tests

#### 1. Blended vs Episodic-Only Comparison
```rust
#[test]
fn test_blended_recall_outperforms_episodic_for_partial_cues() {
    // Create a scenario where semantic pathway helps
    let graph = create_test_graph_with_concepts();

    // Partial cue that doesn't match episodes directly but matches concepts
    let partial_cue = create_degraded_cue(original_embedding, noise=0.3);

    let episodic_results = episodic_only_recall(&partial_cue, &graph);
    let blended_results = blended_recall(&partial_cue, &graph);

    // Blended should return more results via concept-mediated retrieval
    assert!(blended_results.len() > episodic_results.len());

    // Blended should have higher average confidence
    let blended_avg_conf = blended_results.iter()
        .map(|r| r.blended_confidence.raw())
        .sum::<f32>() / blended_results.len() as f32;

    let episodic_avg_conf = episodic_results.iter()
        .map(|r| r.confidence.raw())
        .sum::<f32>() / episodic_results.len() as f32;

    assert!(blended_avg_conf > episodic_avg_conf * 0.9);
}
```

#### 2. Semantic Priming Effect
```rust
#[test]
fn test_semantic_priming_via_concepts() {
    // Test Neely (1977) semantic priming paradigm
    // "DOCTOR" primes "NURSE" through medical concept

    let graph = create_medical_concept_graph();

    // Prime: Store "DOCTOR" with high activation
    graph.activate_episode("doctor_episode", 1.0);

    // Target: Retrieve "NURSE"
    let cue = Cue::embedding("nurse_cue", nurse_embedding, ...);
    let results = blended_recall(&cue, &graph);

    // Should retrieve nurse-related episodes faster/more confidently
    // due to shared medical concept activation
    let nurse_result = results.iter()
        .find(|r| r.base.episode.what.contains("nurse"))
        .unwrap();

    assert!(nurse_result.base.activation > 0.3);  // Boosted by priming
}
```

#### 3. Pattern Completion Validation
```rust
#[test]
fn test_pattern_completion_reconstructs_missing_episodes() {
    let graph = create_graph_with_concept_cluster();

    // Remove some episodes from episodic store but keep in concepts
    let removed_episodes = remove_random_episodes(&graph, 0.3);

    let cue = create_concept_matching_cue();
    let results = blended_recall_with_completion(&cue, &graph);

    // Should reconstruct some of the removed episodes via concepts
    let reconstructed_count = results.iter()
        .filter(|r| matches!(r.provenance.final_source, RecallSource::PatternCompleted))
        .filter(|r| removed_episodes.contains(&r.base.episode.id))
        .count();

    assert!(reconstructed_count > 0);
    assert!(reconstructed_count < removed_episodes.len());  // Not perfect
}
```

### Performance Benchmarks

```rust
#[bench]
fn bench_blended_recall_latency(b: &mut Bencher) {
    let (engine, store, cue) = setup_benchmark_scenario();

    b.iter(|| {
        black_box(engine.recall_blended(&cue, &store))
    });

    // Assert P95 < 15ms
    let p95_latency = b.results.percentile(95);
    assert!(p95_latency < Duration::from_millis(15));
}

#[bench]
fn bench_episodic_pathway_only(b: &mut Bencher) {
    // Baseline for comparison
    let (recall, store, cue) = setup_benchmark_scenario();

    b.iter(|| {
        black_box(recall.recall(&cue, &store))
    });
}

#[bench]
fn bench_semantic_pathway_timeout(b: &mut Bencher) {
    // Verify timeout mechanism works
    let (engine, store, cue) = setup_slow_semantic_scenario();

    b.iter(|| {
        let start = Instant::now();
        black_box(engine.recall_blended(&cue, &store));
        assert!(start.elapsed() < Duration::from_millis(20));  // Should timeout
    });
}
```

### Psychological Validation Tests

#### Convergent Retrieval Benefit
```rust
#[test]
fn test_convergent_retrieval_increases_confidence() {
    // Memories found by both pathways should be more confident
    let results = blended_recall(&cue, &graph);

    let convergent = results.iter()
        .filter(|r| r.is_convergent())
        .collect::<Vec<_>>();

    let divergent = results.iter()
        .filter(|r| !r.is_convergent())
        .collect::<Vec<_>>();

    let convergent_avg = convergent.iter()
        .map(|r| r.blended_confidence.raw())
        .sum::<f32>() / convergent.len() as f32;

    let divergent_avg = divergent.iter()
        .map(|r| r.blended_confidence.raw())
        .sum::<f32>() / divergent.len() as f32;

    assert!(convergent_avg > divergent_avg * 1.1);  // At least 10% boost
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Blended recall combines episodic and semantic pathways correctly
- [ ] Provenance tracking captures all pathway contributions with concept details
- [ ] Adaptive weighting adjusts based on pathway performance and timing
- [ ] Pattern completion activates when episodic recall is insufficient
- [ ] Confidence calibration boosts convergent results, penalizes divergent
- [ ] Graceful fallback to episodic-only when concepts are low quality

### Cognitive Validity
- [ ] System 1/System 2 timing differences observable (episodic 2-5ms, semantic 5-10ms)
- [ ] Convergent retrieval (both pathways) shows higher confidence than divergent
- [ ] Blended recall improves robustness to partial/noisy cues (>20% more results)
- [ ] Pattern completion from concepts matches psychological data (r > 0.7 correlation)
- [ ] Semantic priming effect demonstrable via concept-mediated spreading

### Performance Requirements
- [ ] Total blended recall latency P95 < 15ms (vs 10ms episodic-only baseline)
- [ ] Semantic pathway timeout enforced (<8ms)
- [ ] Blending computation overhead <500μs
- [ ] Memory footprint increase <20% vs episodic-only recall
- [ ] Throughput degradation <10% vs episodic-only (at least 9K recalls/sec)

### Integration & Compatibility
- [ ] Integrates with existing CognitiveRecall without breaking changes
- [ ] Works with BindingIndex from Task 005
- [ ] Uses HierarchicalSpreading from Task 008
- [ ] Extends RankedMemory type with backward compatibility
- [ ] RecallConfig settings respected (time budgets, thresholds)

### Code Quality
- [ ] All unit tests pass (>95% code coverage)
- [ ] Integration tests validate end-to-end scenarios
- [ ] Benchmarks confirm performance targets
- [ ] Clippy warnings resolved (zero warnings required)
- [ ] Documentation complete with cognitive justification

## Dependencies

### Required (Blocking)
- Task 005 (Binding Formation) - Provides BindingIndex for concept→episode lookups
- Task 007 (Fan Effect Spreading) - Concept fan-out penalty during semantic pathway
- Task 008 (Hierarchical Spreading) - Multi-hop concept traversal for pattern completion

### Optional (Enhancing)
- Task 004 (Concept Formation) - Higher quality concepts improve semantic pathway
- Task 006 (Consolidation Integration) - More concepts available over time
- Existing CognitiveRecall system (recall.rs) - Used directly for episodic pathway

## Estimated Time
4 days (increased from original 3 days due to comprehensive cognitive validation)

- Day 1: Core blending logic, provenance tracking, adaptive weighting
- Day 2: Confidence calibration, pattern completion, fallback strategies
- Day 3: Integration with existing systems, comprehensive unit tests
- Day 4: Psychological validation tests, performance benchmarks, documentation

## References

### Dual-Process Theory
- Kahneman, D. (2011). Thinking, Fast and Slow. System 1 (fast, automatic) vs System 2 (slow, deliberate)
- Evans, J. St. B. T. (2008). Dual-processing accounts of reasoning, judgment, and social cognition. Annual Review of Psychology, 59, 255-278.

### Complementary Learning Systems
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. Psychological Review, 102(3), 419-457.
- Norman, K. A., & O'Reilly, R. C. (2003). Modeling hippocampal and neocortical contributions to recognition memory. Psychological Review, 110(4), 611-646.

### Pattern Completion & Convergence
- Marr, D. (1971). Simple memory: a theory for archicortex. Philosophical Transactions of the Royal Society B, 262(841), 23-81.
- Neely, J. H. (1977). Semantic priming and retrieval from lexical memory. Journal of Experimental Psychology: General, 106(3), 226-254.

### Biological Plausibility
- Treves, A., & Rolls, E. T. (1994). Computational analysis of the role of the hippocampus in memory. Hippocampus, 4(3), 374-391.
- O'Reilly, R. C., & Rudy, J. W. (2001). Conjunctive representations in learning and memory: principles of cortical and hippocampal function. Psychological Review, 108(2), 311-345.