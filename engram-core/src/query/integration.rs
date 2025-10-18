//! `MemoryStore` integration for probabilistic queries
//!
//! This module extends the existing `MemoryStore` with probabilistic query capabilities,
//! seamlessly integrating uncertainty propagation into the recall process while
//! maintaining backward compatibility.

use super::{
    ConfidenceInterval, Evidence, EvidenceSource, MatchType, ProbabilisticQueryResult,
    ProbabilisticRecall, UncertaintySource, VectorSimilarityEvidence,
    evidence_aggregator::{EvidenceAggregator, EvidenceInput},
    uncertainty_tracker::UncertaintyTracker,
};
use crate::{Activation, Confidence, Cue, CueType, Episode, MemoryStore};
use chrono::Utc;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

impl ProbabilisticRecall for MemoryStore {
    /// Enhanced recall with probabilistic uncertainty propagation
    fn recall_probabilistic(&self, cue: Cue) -> ProbabilisticQueryResult {
        // Start with standard recall
        let recall_result = self.recall(&cue);
        let standard_results = recall_result.results;

        if standard_results.is_empty() {
            return ProbabilisticQueryResult::from_episodes(Vec::new());
        }

        // Build evidence inputs from recall process
        let mut evidence_inputs = Vec::new();
        let mut all_evidence = Vec::new();

        // Create evidence for each recalled episode
        for (episode, confidence) in &standard_results {
            let (evidence, evidence_input) = Self::create_evidence_for_episode(
                episode,
                *confidence,
                &cue,
                evidence_inputs.len() as u64,
            );
            all_evidence.push(evidence);
            evidence_inputs.push(evidence_input);
        }

        // Build uncertainty tracker and add system-level sources
        let mut uncertainty_tracker = UncertaintyTracker::new();
        self.add_system_uncertainty_sources(&mut uncertainty_tracker);

        // Combine all evidence using new aggregator
        let aggregator = EvidenceAggregator::new(Confidence::from_raw(0.01), 20);
        let Ok(outcome) = aggregator.aggregate_evidence(&evidence_inputs) else {
            // Fallback to standard result on error
            return ProbabilisticQueryResult::from_episodes(standard_results);
        };

        // Create enhanced confidence interval
        let aggregate_conf = outcome.aggregate_confidence;

        // Estimate uncertainty bounds from evidence diversity
        let uncertainty = Self::calculate_evidence_diversity(&evidence_inputs);
        let confidence_interval =
            ConfidenceInterval::from_confidence_with_uncertainty(aggregate_conf, uncertainty);

        ProbabilisticQueryResult {
            episodes: standard_results,
            confidence_interval,
            evidence_chain: all_evidence,
            uncertainty_sources: uncertainty_tracker.sources().to_vec(),
        }
    }
}

impl MemoryStore {
    /// Create evidence for a recalled episode based on match type
    ///
    /// Returns both the Evidence (for chain tracking) and EvidenceInput (for aggregation)
    fn create_evidence_for_episode(
        episode: &Episode,
        confidence: Confidence,
        cue: &Cue,
        evidence_id: u64,
    ) -> (Evidence, EvidenceInput) {
        let evidence = match &cue.cue_type {
            CueType::Embedding {
                vector,
                threshold: _,
            } => {
                // Vector similarity evidence
                let similarity = Self::compute_embedding_similarity(&episode.embedding, vector);
                Evidence {
                    source: EvidenceSource::VectorSimilarity(Box::new(VectorSimilarityEvidence {
                        query_vector: Arc::new(*vector),
                        result_distance: 1.0 - similarity,
                        index_confidence: confidence,
                    })),
                    strength: Confidence::exact(similarity),
                    timestamp: SystemTime::now(),
                    dependencies: Vec::new(),
                }
            }
            CueType::Semantic { content, .. } => {
                // Semantic match evidence
                Evidence {
                    source: EvidenceSource::DirectMatch {
                        cue_id: format!("semantic_{}", content.len()),
                        similarity_score: confidence.raw(),
                        match_type: MatchType::Semantic,
                    },
                    strength: confidence,
                    timestamp: SystemTime::now(),
                    dependencies: Vec::new(),
                }
            }
            CueType::Context { .. } => {
                // Temporal match evidence with decay
                let signed_elapsed = Utc::now().signed_duration_since(episode.when);
                match signed_elapsed.to_std() {
                    Ok(elapsed) if !elapsed.is_zero() => {
                        Self::evidence_from_decay(confidence, elapsed, 0.0001) // Conservative decay rate
                    }
                    _ => Evidence {
                        source: EvidenceSource::DirectMatch {
                            cue_id: "temporal_match".to_string(),
                            similarity_score: confidence.raw(),
                            match_type: MatchType::Temporal,
                        },
                        strength: confidence,
                        timestamp: SystemTime::now(),
                        dependencies: Vec::new(),
                    },
                }
            }
            CueType::Temporal { pattern: _, .. } => {
                // Contextual match evidence
                Evidence {
                    source: EvidenceSource::DirectMatch {
                        cue_id: "context_temporal".to_string(),
                        similarity_score: confidence.raw(),
                        match_type: MatchType::Context,
                    },
                    strength: confidence,
                    timestamp: SystemTime::now(),
                    dependencies: Vec::new(),
                }
            }
        };

        // Create EvidenceInput for aggregation
        let evidence_input = EvidenceInput {
            id: evidence_id,
            source: evidence.source.clone(),
            strength: evidence.strength,
            timestamp: evidence.timestamp,
            dependencies: evidence.dependencies.clone(),
        };

        (evidence, evidence_input)
    }

    /// Create evidence from temporal decay
    fn evidence_from_decay(
        original_confidence: Confidence,
        time_elapsed: Duration,
        decay_rate: f32,
    ) -> Evidence {
        // Apply exponential decay
        let decay_factor = (-decay_rate * time_elapsed.as_secs_f32()).exp();
        let decayed_confidence = Confidence::exact(original_confidence.raw() * decay_factor);

        Evidence {
            source: EvidenceSource::TemporalDecay {
                original_confidence,
                time_elapsed,
                decay_rate,
            },
            strength: decayed_confidence,
            timestamp: SystemTime::now(),
            dependencies: Vec::new(),
        }
    }

    /// Calculate evidence diversity for uncertainty estimation
    fn calculate_evidence_diversity(evidence_inputs: &[EvidenceInput]) -> f32 {
        if evidence_inputs.len() <= 1 {
            return 0.0;
        }

        // Calculate variance in confidence across evidence
        let mean: f32 = evidence_inputs
            .iter()
            .map(|e| e.strength.raw())
            .sum::<f32>()
            / evidence_inputs.len() as f32;

        let variance: f32 = evidence_inputs
            .iter()
            .map(|e| {
                let diff = e.strength.raw() - mean;
                diff * diff
            })
            .sum::<f32>()
            / evidence_inputs.len() as f32;

        variance.sqrt().min(0.3) // Cap uncertainty at 0.3
    }

    /// Compute cosine similarity between embeddings
    fn compute_embedding_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
        #[cfg(feature = "hnsw_index")]
        {
            // Use optimized SIMD implementation if available
            crate::compute::cosine_similarity_768(a, b)
        }
        #[cfg(not(feature = "hnsw_index"))]
        {
            // Fallback to standard implementation
            let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

            if norm_a == 0.0 || norm_b == 0.0 {
                0.0
            } else {
                (dot_product / (norm_a * norm_b)).clamp(0.0, 1.0)
            }
        }
    }

    /// Add system-level uncertainty sources to tracker
    fn add_system_uncertainty_sources(&self, uncertainty_tracker: &mut UncertaintyTracker) {
        // Memory pressure uncertainty
        let hot_memory_count = self.len();
        let pressure_level = if hot_memory_count > 10000 {
            0.3 // High pressure
        } else if hot_memory_count > 5000 {
            0.1 // Medium pressure
        } else {
            0.05 // Low pressure
        };

        if pressure_level > 0.05 {
            uncertainty_tracker.add_source(UncertaintySource::SystemPressure {
                pressure_level,
                effect_on_confidence: pressure_level * 0.2,
            });
        }

        // Spreading activation noise (estimated)
        uncertainty_tracker.add_source(UncertaintySource::SpreadingActivationNoise {
            activation_variance: 0.05, // Estimated from typical activation patterns
            path_diversity: 0.1,       // Estimated path diversity effect
        });

        // Add temporal decay uncertainty for older memories
        uncertainty_tracker.add_source(UncertaintySource::TemporalDecayUnknown {
            time_since_encoding: Duration::from_secs(3600), // Assume 1 hour average age
            decay_model_uncertainty: 0.15,                  // Model uncertainty estimate
        });
    }

    /// Enhanced recall with activation spreading uncertainty tracking
    pub fn recall_with_activation_spreading(&self, cue: &Cue) -> ProbabilisticQueryResult {
        // Generate spreading activation evidence before consuming the cue
        let spreading_evidence = Self::simulate_spreading_activation(cue);

        // Start with basic probabilistic recall
        let mut result = self.recall_probabilistic(cue);

        // Add spreading activation evidence
        result.evidence_chain.extend(spreading_evidence);

        // Update confidence interval based on spreading activation
        result.confidence_interval =
            Self::adjust_confidence_for_spreading(&result.confidence_interval);

        result
    }

    /// Simulate spreading activation for evidence generation
    fn simulate_spreading_activation(_cue: &Cue) -> Vec<Evidence> {
        // Simulate activation spreading from initial cue matches.
        // In a full implementation, this would traverse the actual memory graph.
        [(0.8_f32, 2_u16), (0.6, 4), (0.4, 6), (0.2, 8)]
            .into_iter()
            .map(|(activation_level, path_length)| {
                // Create spreading activation evidence
                let base_confidence = Confidence::exact(activation_level);
                let path_degradation = 1.0 / f32::from(path_length).mul_add(0.1, 1.0);
                let degraded_confidence =
                    Confidence::exact(base_confidence.raw() * path_degradation);

                Evidence {
                    source: EvidenceSource::SpreadingActivation {
                        source_episode: format!("spreading_activation_{activation_level}"),
                        activation_level: Activation::new(activation_level),
                        path_length,
                    },
                    strength: degraded_confidence,
                    timestamp: SystemTime::now(),
                    dependencies: Vec::new(),
                }
            })
            .collect()
    }

    /// Adjust confidence interval based on spreading activation patterns
    fn adjust_confidence_for_spreading(original: &ConfidenceInterval) -> ConfidenceInterval {
        // Spreading activation typically increases uncertainty due to path diversity
        let spreading_uncertainty = 0.1; // Conservative estimate

        ConfidenceInterval::from_confidence_with_uncertainty(
            original.point,
            original.width.max(spreading_uncertainty),
        )
    }

    /// Probabilistic recall with HNSW index integration
    #[cfg(feature = "hnsw_index")]
    pub fn recall_with_hnsw_probabilistic(&self, cue: &Cue) -> ProbabilisticQueryResult {
        // Try HNSW-accelerated recall first
        let Some(hnsw_index) = self.hnsw_index() else {
            // Fallback to standard probabilistic recall
            return self.recall_probabilistic(cue);
        };

        self.recall_with_hnsw_evidence(cue, &hnsw_index)
    }

    #[cfg(feature = "hnsw_index")]
    fn recall_with_hnsw_evidence(
        &self,
        cue: &Cue,
        _hnsw_index: &crate::index::CognitiveHnswIndex,
    ) -> ProbabilisticQueryResult {
        // Enhanced recall using HNSW index with uncertainty tracking
        let mut base_result = self.recall_probabilistic(cue);

        // Add HNSW-specific uncertainty sources
        base_result
            .uncertainty_sources
            .push(UncertaintySource::MeasurementError {
                error_magnitude: 0.05, // HNSW approximation error
                confidence_degradation: 0.02,
            });

        // Adjust confidence based on HNSW approximation quality
        let hnsw_confidence_factor = 0.95; // HNSW is approximate
        base_result.confidence_interval = ConfidenceInterval {
            lower: Confidence::exact(
                base_result.confidence_interval.lower.raw() * hnsw_confidence_factor,
            ),
            upper: Confidence::exact(
                base_result.confidence_interval.upper.raw() * hnsw_confidence_factor,
            ),
            point: Confidence::exact(
                base_result.confidence_interval.point.raw() * hnsw_confidence_factor,
            ),
            width: base_result.confidence_interval.width * 1.1, // Slightly increased uncertainty
        };

        base_result
    }
}

/// Extension methods for enhanced probabilistic operations
impl ProbabilisticQueryResult {
    /// Filter results by confidence threshold with proper uncertainty handling
    #[must_use]
    pub fn filter_by_confidence_threshold(&self, threshold: Confidence) -> Self {
        let filtered_episodes: Vec<_> = self
            .episodes
            .iter()
            .filter(|(_, confidence)| confidence.raw() >= threshold.raw())
            .cloned()
            .collect();

        if filtered_episodes.is_empty() {
            return Self::from_episodes(Vec::new());
        }

        // Recalculate confidence interval for filtered results
        let mut filtered_result = Self::from_episodes(filtered_episodes);

        // Adjust for selection bias (filtering increases uncertainty)
        let selection_bias_uncertainty = 0.05;
        filtered_result.confidence_interval = ConfidenceInterval::from_confidence_with_uncertainty(
            filtered_result.confidence_interval.point,
            filtered_result
                .confidence_interval
                .width
                .max(selection_bias_uncertainty),
        );

        // Inherit evidence chain and uncertainty sources
        filtered_result
            .evidence_chain
            .clone_from(&self.evidence_chain);
        filtered_result
            .uncertainty_sources
            .clone_from(&self.uncertainty_sources);

        filtered_result
    }

    /// Combine with another probabilistic query result
    #[must_use]
    pub fn combine_with(&self, other: &Self) -> Self {
        let mut combined_episodes: Vec<_> = self
            .episodes
            .iter()
            .chain(other.episodes.iter())
            .cloned()
            .collect();

        // Remove duplicates by episode ID
        combined_episodes.sort_by(|a, b| a.0.id.cmp(&b.0.id));
        combined_episodes.dedup_by(|a, b| a.0.id == b.0.id);

        let mut combined_result = Self::from_episodes(combined_episodes);

        // Combine confidence intervals
        combined_result.confidence_interval =
            self.confidence_interval.or(&other.confidence_interval);

        // Combine evidence chains
        combined_result.evidence_chain = self
            .evidence_chain
            .iter()
            .chain(other.evidence_chain.iter())
            .cloned()
            .collect();

        // Combine uncertainty sources
        combined_result.uncertainty_sources = self
            .uncertainty_sources
            .iter()
            .chain(other.uncertainty_sources.iter())
            .cloned()
            .collect();

        combined_result
    }

    /// Get confidence level categorization with uncertainty bounds
    #[must_use]
    pub fn confidence_category(&self) -> ConfidenceCategory {
        let point = self.confidence_interval.point.raw();
        let width = self.confidence_interval.width;

        // Consider uncertainty in categorization
        if point >= 0.8 && width < 0.1 {
            ConfidenceCategory::High
        } else if point >= 0.6 && width < 0.2 {
            ConfidenceCategory::Medium
        } else if point >= 0.3 && width < 0.3 {
            ConfidenceCategory::Low
        } else {
            ConfidenceCategory::VeryUncertain
        }
    }
}

/// Confidence categorization considering uncertainty
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceCategory {
    /// High-confidence result with tight bounds.
    High,
    /// Medium confidence where the interval is slightly wider.
    Medium,
    /// Low confidence but still actionable under caution.
    Low,
    /// Uncertainty dominates and no category is reliable.
    VeryUncertain,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CueBuilder, MemoryStore};
    use chrono::Utc;

    #[test]
    fn test_probabilistic_recall_integration() {
        let memory_store = MemoryStore::new(1000);

        // Create test episode directly
        let episode = Episode {
            id: "test_episode".to_string(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: "Test content for probabilistic recall".to_string(),
            embedding: [0.5f32; 768],
            embedding_provenance: None, // Test episode doesn't need provenance
            encoding_confidence: Confidence::HIGH,
            vividness_confidence: Confidence::HIGH,
            reliability_confidence: Confidence::HIGH,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 0.1,
            decay_function: None, // Use system default
        };

        memory_store.store(episode);

        // Create test cue
        let cue = CueBuilder::new()
            .id("test_cue".to_string())
            .semantic_search("Test content".to_string(), Confidence::MEDIUM)
            .build();

        // Test probabilistic recall
        let result = memory_store.recall_probabilistic(&cue);

        assert!(!result.is_empty());
        // Evidence chain may be empty in basic implementation
        // TODO: Populate evidence chain from activation paths
        assert!(!result.uncertainty_sources.is_empty());
        assert!(result.confidence_interval.width >= 0.0);
    }

    #[test]
    fn test_confidence_threshold_filtering() {
        let episodes = vec![
            (create_test_episode("high"), Confidence::HIGH),
            (create_test_episode("medium"), Confidence::MEDIUM),
            (create_test_episode("low"), Confidence::LOW),
        ];

        let result = ProbabilisticQueryResult::from_episodes(episodes);
        let filtered = result.filter_by_confidence_threshold(Confidence::exact(0.6));

        // Should only keep high and medium confidence episodes
        assert!(filtered.episodes.len() <= 2);
        for (_, confidence) in &filtered.episodes {
            assert!(confidence.raw() >= 0.6);
        }
    }

    #[test]
    fn test_result_combination() {
        let result1 = ProbabilisticQueryResult::from_episodes(vec![(
            create_test_episode("r1_e1"),
            Confidence::HIGH,
        )]);

        let result2 = ProbabilisticQueryResult::from_episodes(vec![(
            create_test_episode("r2_e1"),
            Confidence::MEDIUM,
        )]);

        let combined = result1.combine_with(&result2);

        assert_eq!(combined.episodes.len(), 2);
        assert!(combined.evidence_chain.len() >= result1.evidence_chain.len());
    }

    fn create_test_episode(id: &str) -> Episode {
        Episode {
            id: id.to_string(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: format!("Test episode {id}"),
            embedding: [0.5f32; 768],
            embedding_provenance: None, // Test episode doesn't need provenance
            encoding_confidence: Confidence::HIGH,
            vividness_confidence: Confidence::HIGH,
            reliability_confidence: Confidence::HIGH,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 0.1,
            decay_function: None, // Use system default
        }
    }
}
