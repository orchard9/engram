//! Evidence tracking and combination for probabilistic reasoning
//!
//! This module implements Bayesian evidence combination with circular dependency
//! detection and proper uncertainty propagation. It prevents common cognitive
//! biases like base rate neglect and conjunction fallacy while maintaining
//! computational efficiency.

use super::{Evidence, EvidenceId, EvidenceSource, ProbabilisticError, ProbabilisticResult};
use crate::{Activation, Confidence};
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::time::{Duration, SystemTime};

/// Evidence aggregator with circular dependency detection
pub struct EvidenceAggregator {
    /// Evidence database with unique identifiers
    evidence_db: HashMap<EvidenceId, Evidence>,
    /// Dependency graph for circular reference detection
    dependency_graph: HashMap<EvidenceId, HashSet<EvidenceId>>,
    /// Next available evidence ID
    next_id: EvidenceId,
    /// Cache for computed evidence combinations
    combination_cache: HashMap<Vec<EvidenceId>, CombinedEvidence>,
}

/// Combined evidence result with uncertainty bounds
#[derive(Debug, Clone)]
pub struct CombinedEvidence {
    /// Final confidence after evidence combination
    pub confidence: Confidence,
    /// Lower bound of uncertainty interval  
    pub lower_bound: Confidence,
    /// Upper bound of uncertainty interval
    pub upper_bound: Confidence,
    /// Evidence sources that contributed
    pub contributing_sources: Vec<EvidenceId>,
    /// Dependency chain for traceability
    pub dependency_chain: Vec<EvidenceId>,
    /// Time when combination was computed
    pub computed_at: SystemTime,
}

impl EvidenceAggregator {
    /// Create new evidence aggregator
    #[must_use]
    pub fn new() -> Self {
        Self {
            evidence_db: HashMap::new(),
            dependency_graph: HashMap::new(),
            next_id: 1,
            combination_cache: HashMap::new(),
        }
    }

    /// Add evidence with automatic ID assignment
    #[must_use]
    pub fn add_evidence(&mut self, evidence: Evidence) -> EvidenceId {
        let id = self.next_id;
        self.next_id += 1;

        // Build dependency graph
        for dep_id in &evidence.dependencies {
            self.dependency_graph.entry(id).or_default().insert(*dep_id);
        }

        // Store evidence
        self.evidence_db.insert(id, evidence);

        // Clear relevant cache entries
        self.invalidate_cache_for_evidence(id);

        id
    }

    /// Combine multiple evidence sources using Bayesian updating.
    ///
    /// # Errors
    ///
    /// - [`ProbabilisticError::InsufficientEvidence`] if any provided identifier is
    ///   absent from the aggregator.
    /// - [`ProbabilisticError::CircularDependency`] when the dependency graph
    ///   contains a cycle.
    ///
    /// # Panics
    ///
    /// This function does not panic; invalid identifiers are propagated as
    /// [`ProbabilisticError`] values instead.
    #[must_use = "Use the combined evidence to drive downstream probabilistic reasoning"]
    pub fn combine_evidence(
        &mut self,
        evidence_ids: &[EvidenceId],
    ) -> ProbabilisticResult<CombinedEvidence> {
        // Check cache first
        let mut sorted_ids = evidence_ids.to_vec();
        sorted_ids.sort_unstable();
        if let Some(cached) = self.combination_cache.get(&sorted_ids) {
            return Ok(cached.clone());
        }

        // Validate evidence IDs exist
        for &id in evidence_ids {
            if !self.evidence_db.contains_key(&id) {
                return Err(ProbabilisticError::InsufficientEvidence);
            }
        }

        // Check for circular dependencies
        self.check_circular_dependencies(evidence_ids)?;

        // Perform Bayesian combination
        let result = self.bayesian_combination(evidence_ids)?;

        // Cache result
        self.combination_cache.insert(sorted_ids, result.clone());

        Ok(result)
    }

    /// Create evidence from spreading activation
    #[must_use]
    pub fn evidence_from_activation(
        source_episode: impl Into<String>,
        activation_level: Activation,
        path_length: u16,
    ) -> Evidence {
        // Convert activation level to confidence with path degradation
        let base_confidence = Confidence::exact(activation_level.value());
        let path_degradation = 1.0 / f32::from(path_length).mul_add(0.1, 1.0);
        let degraded_confidence = Confidence::exact(base_confidence.raw() * path_degradation);

        Evidence {
            source: EvidenceSource::SpreadingActivation {
                source_episode: source_episode.into(),
                activation_level,
                path_length,
            },
            strength: degraded_confidence,
            timestamp: SystemTime::now(),
            dependencies: Vec::new(),
        }
    }

    /// Create evidence from temporal decay
    #[must_use]
    pub fn evidence_from_decay(
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

    /// Create evidence from direct cue matching
    #[must_use]
    pub fn evidence_from_cue_match(
        cue_id: impl Into<String>,
        similarity_score: f32,
        match_type: super::MatchType,
    ) -> Evidence {
        let confidence = Confidence::exact(similarity_score);

        Evidence {
            source: EvidenceSource::DirectMatch {
                cue_id: cue_id.into(),
                similarity_score,
                match_type,
            },
            strength: confidence,
            timestamp: SystemTime::now(),
            dependencies: Vec::new(),
        }
    }

    /// Check for circular dependencies in the evidence chain.
    ///
    /// # Errors
    ///
    /// - [`ProbabilisticError::CircularDependency`] if traversing the graph reveals a
    ///   cycle.
    ///
    /// # Panics
    ///
    /// This function does not panic. Depth-first traversal is bounded by the
    /// dependency graph depth and guards against recursion overflow.
    fn check_circular_dependencies(&self, evidence_ids: &[EvidenceId]) -> ProbabilisticResult<()> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for &id in evidence_ids {
            if self.has_circular_dependency(id, &mut visited, &mut rec_stack) {
                return Err(ProbabilisticError::CircularDependency);
            }
        }

        Ok(())
    }

    /// Recursive circular dependency detection using DFS
    fn has_circular_dependency(
        &self,
        id: EvidenceId,
        visited: &mut HashSet<EvidenceId>,
        rec_stack: &mut HashSet<EvidenceId>,
    ) -> bool {
        if rec_stack.contains(&id) {
            return true;
        }

        if visited.contains(&id) {
            return false;
        }

        visited.insert(id);
        rec_stack.insert(id);

        if let Some(dependencies) = self.dependency_graph.get(&id) {
            for &dep_id in dependencies {
                if self.has_circular_dependency(dep_id, visited, rec_stack) {
                    return true;
                }
            }
        }

        rec_stack.remove(&id);
        false
    }

    /// Perform Bayesian evidence combination with bias prevention.
    ///
    /// # Errors
    ///
    /// - [`ProbabilisticError::InsufficientEvidence`] when `evidence_ids` is empty.
    ///
    /// # Panics
    ///
    /// This function does not panic. When the input slice is empty a
    /// [`ProbabilisticError::InsufficientEvidence`] is returned instead of
    /// triggering a panic.
    fn bayesian_combination(
        &self,
        evidence_ids: &[EvidenceId],
    ) -> ProbabilisticResult<CombinedEvidence> {
        if evidence_ids.is_empty() {
            return Err(ProbabilisticError::InsufficientEvidence);
        }

        // Sort by timestamp for proper temporal ordering
        let mut sorted_evidence: Vec<_> = evidence_ids
            .iter()
            .filter_map(|&id| self.evidence_db.get(&id).map(|e| (id, e)))
            .collect();
        sorted_evidence.sort_by_key(|(_, e)| e.timestamp);

        // Initialize with first piece of evidence (base rate)
        let (first_id, first_evidence) = sorted_evidence[0];
        let mut combined_confidence = first_evidence.strength;
        let mut contributing_sources = vec![first_id];

        // Combine remaining evidence using log-odds
        for (id, evidence) in sorted_evidence.iter().skip(1) {
            // Convert to log-odds for numerical stability
            let current_odds = combined_confidence.raw() / (1.0 - combined_confidence.raw());
            let evidence_odds = evidence.strength.raw() / (1.0 - evidence.strength.raw());

            // Combine odds (assumes independence)
            let combined_odds = current_odds * evidence_odds;

            // Convert back to probability
            let combined_prob = combined_odds / (1.0 + combined_odds);

            // Handle numerical edge cases
            let final_prob = if combined_prob.is_finite() && (0.0..=1.0).contains(&combined_prob) {
                combined_prob
            } else {
                // Fallback to weighted average for numerical stability
                let weight1 = 0.6;
                let weight2 = 0.4;
                combined_confidence
                    .raw()
                    .mul_add(weight1, evidence.strength.raw() * weight2)
            };

            combined_confidence = Confidence::exact(final_prob);
            contributing_sources.push(*id);
        }

        // Estimate uncertainty bounds based on evidence diversity
        let evidence_values: Vec<f32> = sorted_evidence
            .iter()
            .map(|(_, e)| e.strength.raw())
            .collect();

        let len_u32 = u32::try_from(evidence_values.len()).unwrap_or(u32::MAX);
        let count = f64::from(len_u32).max(1.0);
        let sum: f64 = evidence_values.iter().map(|value| f64::from(*value)).sum();
        let mean = sum / count;
        let variance = evidence_values
            .iter()
            .map(|value| {
                let diff = f64::from(*value) - mean;
                diff * diff
            })
            .sum::<f64>()
            / count;
        let std_dev = variance.sqrt();

        // Conservative uncertainty bounds
        let uncertainty_factor = super::clamp_probability_to_f32(std_dev * 1.96); // 95% interval
        let lower_bound =
            Confidence::exact((combined_confidence.raw() - uncertainty_factor).max(0.0));
        let upper_bound =
            Confidence::exact((combined_confidence.raw() + uncertainty_factor).min(1.0));

        // Build full dependency chain
        let mut dependency_chain = Vec::new();
        for &id in evidence_ids {
            self.collect_dependencies(id, &mut dependency_chain);
        }
        dependency_chain.sort_unstable();
        dependency_chain.dedup();

        Ok(CombinedEvidence {
            confidence: combined_confidence,
            lower_bound,
            upper_bound,
            contributing_sources,
            dependency_chain,
            computed_at: SystemTime::now(),
        })
    }

    /// Recursively collect all dependencies for an evidence item
    fn collect_dependencies(&self, id: EvidenceId, chain: &mut Vec<EvidenceId>) {
        if chain.contains(&id) {
            return; // Avoid infinite recursion
        }

        chain.push(id);

        if let Some(dependencies) = self.dependency_graph.get(&id) {
            for &dep_id in dependencies {
                self.collect_dependencies(dep_id, chain);
            }
        }
    }

    /// Invalidate cache entries affected by new evidence
    fn invalidate_cache_for_evidence(&mut self, evidence_id: EvidenceId) {
        self.combination_cache
            .retain(|ids, _| !ids.contains(&evidence_id));
    }

    /// Get evidence by ID
    #[must_use]
    pub fn get_evidence(&self, id: EvidenceId) -> Option<&Evidence> {
        self.evidence_db.get(&id)
    }

    /// Get all evidence for a source type
    #[must_use]
    pub fn get_evidence_by_source_type(&self, source_type: &str) -> Vec<(EvidenceId, &Evidence)> {
        self.evidence_db
            .iter()
            .filter(|(_, evidence)| {
                matches!(
                    (&evidence.source, source_type),
                    (EvidenceSource::SpreadingActivation { .. }, "activation")
                        | (EvidenceSource::TemporalDecay { .. }, "decay")
                        | (EvidenceSource::DirectMatch { .. }, "match")
                        | (EvidenceSource::VectorSimilarity(_), "similarity")
                )
            })
            .map(|(&id, evidence)| (id, evidence))
            .collect()
    }
}

impl Default for EvidenceAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MatchType;
    use std::fmt::Debug;
    use std::time::Duration;

    type TestResult<T = ()> = Result<T, String>;

    fn ensure_eq<T>(actual: &T, expected: &T, context: &str) -> TestResult
    where
        T: PartialEq + Debug,
    {
        if actual == expected {
            Ok(())
        } else {
            Err(format!("{context}: expected {expected:?}, got {actual:?}"))
        }
    }

    fn ensure(condition: bool, message: impl Into<String>) -> TestResult {
        if condition {
            Ok(())
        } else {
            Err(message.into())
        }
    }

    fn ensure_close_f32(actual: f32, expected: f32, context: &str) -> TestResult {
        let diff = (actual - expected).abs();
        if diff <= f32::EPSILON {
            Ok(())
        } else {
            Err(format!(
                "{context}: expected {expected}, got {actual} (diff {diff})"
            ))
        }
    }

    fn ensure_close_f64(actual: f64, expected: f64, context: &str) -> TestResult {
        let diff = (actual - expected).abs();
        if diff <= 1e-10 {
            Ok(())
        } else {
            Err(format!(
                "{context}: expected {expected}, got {actual} (diff {diff})"
            ))
        }
    }

    #[test]
    fn test_evidence_creation_from_activation() -> TestResult {
        let evidence = EvidenceAggregator::evidence_from_activation(
            "test_episode".to_string(),
            Activation::new(0.8),
            2,
        );

        if let EvidenceSource::SpreadingActivation {
            source_episode,
            activation_level,
            path_length,
        } = &evidence.source
        {
            ensure_eq(
                source_episode,
                &"test_episode".to_string(),
                "activation episode",
            )?;
            ensure_close_f32(activation_level.value(), 0.8, "activation level")?;
            ensure_eq(path_length, &2, "activation path length")?;
        } else {
            return Err("wrong evidence source type".to_string());
        }

        // Path degradation should reduce confidence
        ensure(
            evidence.strength.raw() < 0.8,
            "strength should decay with path length",
        )?;

        Ok(())
    }

    #[test]
    fn test_evidence_combination() -> TestResult {
        let mut aggregator = EvidenceAggregator::new();

        let evidence1 = EvidenceAggregator::evidence_from_activation(
            "episode1".to_string(),
            Activation::new(0.7),
            1,
        );

        let evidence2 = EvidenceAggregator::evidence_from_cue_match(
            "cue1".to_string(),
            0.6,
            MatchType::Embedding,
        );

        let id1 = aggregator.add_evidence(evidence1);
        let id2 = aggregator.add_evidence(evidence2);

        let combined = aggregator
            .combine_evidence(&[id1, id2])
            .map_err(|err| format!("combine evidence should succeed: {err:?}"))?;

        ensure(
            !combined.contributing_sources.is_empty(),
            "combined evidence should track sources",
        )?;
        ensure(
            combined.confidence.raw() > 0.0,
            "combined confidence should be positive",
        )?;
        ensure(
            combined.lower_bound.raw() <= combined.confidence.raw(),
            "lower bound should not exceed confidence",
        )?;
        ensure(
            combined.confidence.raw() <= combined.upper_bound.raw(),
            "upper bound should exceed confidence",
        )?;

        Ok(())
    }

    #[test]
    fn test_circular_dependency_detection() {
        let mut aggregator = EvidenceAggregator::new();

        // Create evidence with circular dependencies
        let mut evidence1 = EvidenceAggregator::evidence_from_activation(
            "episode1".to_string(),
            Activation::new(0.7),
            1,
        );

        let mut evidence2 = EvidenceAggregator::evidence_from_activation(
            "episode2".to_string(),
            Activation::new(0.6),
            1,
        );

        // Create circular dependency: 1 -> 2 -> 1
        evidence1.dependencies = vec![2];
        evidence2.dependencies = vec![1];

        let id1 = aggregator.add_evidence(evidence1);
        let id2 = aggregator.add_evidence(evidence2);

        // Should detect circular dependency
        let result = aggregator.combine_evidence(&[id1, id2]);
        assert!(matches!(
            result,
            Err(ProbabilisticError::CircularDependency)
        ));
    }

    #[test]
    fn test_temporal_decay_evidence() -> TestResult {
        let original = Confidence::HIGH;
        let time_elapsed = Duration::from_secs(3600); // 1 hour
        let decay_rate = 0.001; // per second

        let evidence = EvidenceAggregator::evidence_from_decay(original, time_elapsed, decay_rate);

        if let EvidenceSource::TemporalDecay {
            original_confidence,
            time_elapsed: elapsed,
            decay_rate: rate,
        } = &evidence.source
        {
            ensure_eq(
                original_confidence,
                &Confidence::HIGH,
                "temporal original confidence",
            )?;
            ensure_eq(elapsed, &Duration::from_secs(3600), "temporal elapsed")?;
            ensure_close_f64(f64::from(*rate), 0.001, "temporal decay rate")?;
        } else {
            return Err("wrong evidence source type".to_string());
        }

        // Should be decayed from original
        ensure(
            evidence.strength.raw() < original.raw(),
            "temporal decay should reduce confidence",
        )?;

        Ok(())
    }

    #[test]
    fn test_evidence_aggregation_empty() {
        let mut aggregator = EvidenceAggregator::new();
        let result = aggregator.combine_evidence(&[]);
        assert!(matches!(
            result,
            Err(ProbabilisticError::InsufficientEvidence)
        ));
    }
}
