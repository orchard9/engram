# Task 004: Concept Formation Engine

## Objective
Implement the clustering algorithm that identifies patterns in episodic memories and creates concept nodes during consolidation, based on hippocampal-neocortical consolidation mechanisms.

## Background
Concept formation mirrors the biological process of systems consolidation where repeated patterns in hippocampal episodic memories gradually form neocortical semantic representations. Research shows this happens through:
- Sharp-wave ripple (SWR) mediated replay during quiet waking and NREM sleep
- Gradual extraction of statistical regularities across multiple experiences
- Progressive independence from contextual details (semanticization)
- Complementary learning systems with different learning rates

## Requirements
1. Implement biologically-inspired clustering that reflects overlapping neural representations
2. Calculate cluster centroids using replay-weighted averaging (simulating SWR replay frequency)
3. Implement gradual concept strength based on consolidation iterations (slow cortical learning)
4. Use coherence thresholds derived from pattern separation/completion literature
5. Integrate with sleep-stage-aware consolidation scheduler
6. Track concept formation metrics aligned with memory consolidation timescales

## Technical Specification

### Files to Create
- `engram-core/src/consolidation/concept_formation.rs` - Main algorithm
- `engram-core/src/consolidation/clustering.rs` - Clustering implementation

### Files to Modify
- `engram-core/src/consolidation/mod.rs` - Integration point
- `engram-core/src/consolidation/dream.rs` - Add concept formation

### Core Algorithm
```rust
pub struct ConceptFormationEngine {
    // Biologically-inspired parameters
    min_cluster_size: usize,           // Default: 3 (minimum overlapping experiences)
    coherence_threshold: f32,          // Default: 0.65 (pattern completion threshold)
    similarity_threshold: f32,         // Default: 0.55 (CA3 pattern separation boundary)
    max_concepts_per_cycle: usize,     // Default: 5 (limited by sleep spindle density)
    consolidation_rate: f32,           // Default: 0.02 (slow cortical learning rate)
    replay_weight_decay: f32,          // Default: 0.9 (SWR replay frequency decay)
}

pub struct ProtoConcept {
    centroid: [f32; 768],
    coherence_score: f32,
    replay_count: u32,              // Simulates SWR replay frequency
    consolidation_strength: f32,     // Gradually increases with iterations
    episode_indices: Vec<usize>,
    temporal_span: Duration,         // Time span of contributing episodes
    semantic_distance: f32,          // Average distance from episodes to centroid
}

impl ConceptFormationEngine {
    pub fn form_concepts(&self, episodes: &[Episode], sleep_stage: SleepStage) -> Vec<ProtoConcept> {
        // 1. Apply biologically-inspired similarity calculation
        let similarity_matrix = self.calculate_neural_overlap(episodes);
        
        // 2. Use soft clustering to allow overlapping representations
        let clusters = self.hierarchical_cluster_with_overlap(
            &similarity_matrix,
            self.similarity_threshold
        );
        
        // 3. Filter by consolidation-relevant coherence
        let viable_clusters = clusters.into_iter()
            .filter(|c| self.calculate_coherence(c, episodes) > self.coherence_threshold)
            .filter(|c| c.len() >= self.min_cluster_size)
            .collect::<Vec<_>>();
        
        // 4. Extract concepts with replay weighting
        viable_clusters.into_iter()
            .take(self.max_concepts_per_cycle)
            .map(|cluster| self.extract_concept(&cluster, episodes, sleep_stage))
            .collect()
    }
    
    fn calculate_neural_overlap(&self, episodes: &[Episode]) -> Vec<Vec<f32>> {
        // Similarity based on overlapping neural representations
        // Uses dot product with sigmoid transformation to model neural activation overlap
        let mut matrix = vec![vec![0.0; episodes.len()]; episodes.len()];
        
        for i in 0..episodes.len() {
            for j in i+1..episodes.len() {
                let similarity = self.neural_similarity(
                    &episodes[i].embedding,
                    &episodes[j].embedding,
                    episodes[i].timestamp.duration_since(episodes[j].timestamp)
                );
                matrix[i][j] = similarity;
                matrix[j][i] = similarity;
            }
        }
        matrix
    }
    
    fn neural_similarity(&self, emb1: &[f32; 768], emb2: &[f32; 768], time_diff: Duration) -> f32 {
        // Model temporal gradient of overlapping representations
        let base_similarity = dot_product(emb1, emb2);
        let temporal_factor = (-time_diff.as_secs_f32() / 86400.0).exp(); // Daily decay
        
        // Sigmoid transformation models neural activation threshold
        1.0 / (1.0 + (-10.0 * (base_similarity * temporal_factor - 0.5)).exp())
    }
    
    fn calculate_coherence(&self, cluster: &[usize], episodes: &[Episode]) -> f32 {
        // Models pattern completion capability of the cluster
        if cluster.len() < 2 {
            return 0.0;
        }
        
        let mut coherence_sum = 0.0;
        let mut count = 0;
        
        for &i in cluster {
            for &j in cluster {
                if i != j {
                    let similarity = self.neural_similarity(
                        &episodes[i].embedding,
                        &episodes[j].embedding,
                        Duration::ZERO // Within-cluster, ignore time
                    );
                    coherence_sum += similarity;
                    count += 1;
                }
            }
        }
        
        coherence_sum / count as f32
    }
    
    fn extract_concept(&self, cluster: &[usize], episodes: &[Episode], sleep_stage: SleepStage) -> ProtoConcept {
        // Weighted centroid extraction based on replay probability
        let replay_weights = self.calculate_replay_weights(cluster, episodes, sleep_stage);
        let centroid = self.weighted_centroid(cluster, episodes, &replay_weights);
        
        // Calculate semantic distance (degree of abstraction)
        let semantic_distance = cluster.iter()
            .zip(&replay_weights)
            .map(|(&idx, &weight)| {
                let dist = euclidean_distance(&episodes[idx].embedding, &centroid);
                dist * weight
            })
            .sum::<f32>() / replay_weights.iter().sum::<f32>();
        
        ProtoConcept {
            centroid,
            coherence_score: self.calculate_coherence(cluster, episodes),
            replay_count: 1, // Initial replay
            consolidation_strength: self.consolidation_rate, // Initial strength
            episode_indices: cluster.to_vec(),
            temporal_span: self.calculate_temporal_span(cluster, episodes),
            semantic_distance,
        }
    }
    
    fn calculate_replay_weights(&self, cluster: &[usize], episodes: &[Episode], sleep_stage: SleepStage) -> Vec<f32> {
        // Model SWR replay probability based on recency and importance
        cluster.iter().map(|&idx| {
            let recency = episodes[idx].timestamp.elapsed().as_secs_f32() / 3600.0; // Hours
            let recency_weight = (-recency / 24.0).exp(); // 24-hour decay
            
            // Sleep stage modulates replay probability
            let stage_factor = match sleep_stage {
                SleepStage::NREM2 => 1.5,  // Peak replay during NREM2
                SleepStage::NREM3 => 1.2,  // High replay in SWS
                SleepStage::REM => 0.8,     // Lower but still present
                SleepStage::Wake => 0.5,    // Quiet waking replay
            };
            
            recency_weight * stage_factor * episodes[idx].importance
        }).collect()
    }
}
```

### Integration with Consolidation
```rust
impl ConsolidationEngine {
    pub fn consolidate_with_concepts(&self, sleep_stage: SleepStage) -> ConsolidationResult {
        // Select episodes based on replay priority (recency + importance)
        let episodes = self.select_episodes_for_replay(sleep_stage);
        
        // Form concepts with stage-appropriate parameters
        let mut concepts = self.concept_engine.form_concepts(&episodes, sleep_stage);
        
        // Update existing concepts through iterative consolidation
        for concept in &mut concepts {
            if let Some(existing) = self.find_similar_concept(&concept.centroid) {
                // Gradual update simulating slow cortical learning
                self.update_concept_strength(existing, concept);
                concept.replay_count = existing.replay_count + 1;
                concept.consolidation_strength = 
                    (existing.consolidation_strength + self.concept_engine.consolidation_rate)
                    .min(1.0); // Asymptotic approach to full consolidation
            }
        }
        
        // Create or update concept nodes
        for concept in concepts {
            if concept.consolidation_strength > 0.1 { // Threshold for cortical representation
                self.graph.add_or_update_concept(concept);
            }
        }
        
        ConsolidationResult {
            episodes_consolidated: episodes.len(),
            concepts_formed: concepts.len(),
            sleep_stage,
            consolidation_cycle: self.current_cycle,
        }
    }
    
    fn select_episodes_for_replay(&self, sleep_stage: SleepStage) -> Vec<Episode> {
        // Biologically-inspired episode selection
        let replay_capacity = match sleep_stage {
            SleepStage::NREM2 => 100,  // High capacity during spindles
            SleepStage::NREM3 => 80,   // Sustained during SWS
            SleepStage::REM => 50,     // Selective during REM
            SleepStage::Wake => 20,    // Limited during quiet wake
        };
        
        self.episode_buffer.iter()
            .filter(|ep| ep.age() < Duration::from_secs(86400 * 7)) // 7-day window
            .sorted_by_key(|ep| OrderedFloat(-ep.replay_priority()))
            .take(replay_capacity)
            .cloned()
            .collect()
    }
}
```

### Biologically-Inspired Parameters

Based on hippocampal-neocortical consolidation research:

1. **Coherence Threshold (0.65)**: Derived from CA3 pattern completion studies showing ~65% overlap needed for successful retrieval
2. **Similarity Threshold (0.55)**: Based on dentate gyrus pattern separation boundary
3. **Min Cluster Size (3)**: Minimum experiences for statistical regularity extraction
4. **Consolidation Rate (0.02)**: Slow cortical learning rate preventing catastrophic interference
5. **Replay Weight Decay (0.9)**: Models decreasing SWR frequency over successive nights
6. **Temporal Decay (24h)**: Aligns with circadian consolidation rhythms

## Implementation Notes
- Use hierarchical clustering with soft boundaries to model overlapping neural representations
- Implement replay-based weighting that respects SWR replay statistics
- Model temporal gradient of memory overlap (memories become less distinct over time)
- Enforce biologically plausible formation rates (1-5% matching cortical spine dynamics)
- Include refractory period between concept formations (models resource limitations)
- Track consolidation strength over multiple sleep cycles for gradual semanticization

## Testing Approach

### Unit Tests
1. **Neural Overlap Calculation**
   - Test sigmoid activation function matches neural response curves
   - Verify temporal decay follows forgetting curve literature
   - Validate similarity matrix symmetry and bounds

2. **Clustering Validation**
   - Test with known patterns from memory research (e.g., AB-AC paradigm)
   - Verify soft boundaries allow appropriate overlap
   - Confirm minimum cluster sizes respect biological constraints

3. **Replay Weight Testing**
   - Validate decay curves match SWR frequency data
   - Test sleep stage modulation factors
   - Ensure recency bias follows power law

### Integration Tests
1. **Multi-cycle Consolidation**
   - Test gradual concept strengthening over 5-10 cycles
   - Verify asymptotic approach to full consolidation
   - Validate no catastrophic interference with existing concepts

2. **Sleep Stage Transitions**
   - Test appropriate parameter adjustments between stages
   - Verify concept formation rates match stage-specific capacity
   - Validate temporal dynamics across full sleep cycle

### Validation Against Neuroscience Literature
1. **Pattern Completion Tests**
   - Concepts should complete patterns with 65% cue overlap
   - Test degraded input reconstruction capabilities
   - Validate against CA3 autoassociative memory properties

2. **Consolidation Timeline Validation**
   - Fast initial consolidation (hours)
   - Slow semantic extraction (days to weeks)
   - Remote memory independence (months)

3. **Interference Patterns**
   - Test retroactive interference (new learning affects old)
   - Test proactive interference (old learning affects new)
   - Verify consolidation reduces interference

## Acceptance Criteria
- [ ] Neural overlap calculation produces biologically plausible similarity values (0.5-0.7 for related memories)
- [ ] Clustering respects pattern separation boundaries (>55% similarity for same cluster)
- [ ] Coherence threshold enables pattern completion (>65% within-cluster similarity)
- [ ] Concept formation rate matches cortical spine dynamics (1-5% of episodes)
- [ ] Replay weighting follows SWR frequency distributions (power law decay)
- [ ] Consolidation strength increases logarithmically over cycles (half-life ~5 cycles)
- [ ] Sleep stage modulation matches empirical replay rates
- [ ] Temporal gradients produce appropriate semanticization over days
- [ ] No catastrophic interference with existing semantic memories
- [ ] Performance scales linearly with episode count up to 10,000 episodes

## Dependencies
- Task 001 (Dual Memory Types)
- Existing consolidation system

## Neuroscience References for Implementation

### Key Papers to Guide Parameters
1. **Pattern Separation/Completion**: Yassa & Stark (2011) - Pattern separation in the hippocampus
2. **SWR Replay**: Girardeau et al. (2009) - Selective suppression of hippocampal ripples
3. **Systems Consolidation**: Frankland & Bontempi (2005) - The organization of recent and remote memories
4. **Sleep Stages**: Diekelmann & Born (2010) - The memory function of sleep
5. **Complementary Learning**: McClelland et al. (1995) - Why there are complementary learning systems

### Implementation Guidelines
1. **Similarity Metrics**: Use cosine similarity with sigmoid transformation to match neural activation functions
2. **Temporal Dynamics**: Implement exponential decay with 24-48 hour time constant
3. **Replay Probability**: Weight by recency^(-0.5) * importance following empirical SWR data
4. **Consolidation Cycles**: Model 5-10 iterations for initial consolidation, 30-100 for full semanticization
5. **Interference Prevention**: Keep learning rate <0.05 to prevent catastrophic forgetting

### Biological Constraints to Enforce
- Maximum 100-200 replays per night (SWR frequency limit)
- Concept formation rate limited by synaptic plasticity timescales
- Temporal contiguity window of ~30 seconds for episode association
- Gradual transition from episodic to semantic over 7-30 days
- Sleep stage transitions every 90-120 minutes affecting parameters

## Estimated Time
3 days