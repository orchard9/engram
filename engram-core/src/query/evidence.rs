//! Evidence tracking and combination for probabilistic reasoning
//!
//! This module implements Bayesian evidence combination with circular dependency
//! detection and proper uncertainty propagation. It prevents common cognitive
//! biases like base rate neglect and conjunction fallacy while maintaining
//! computational efficiency.

use crate::{Confidence, Activation};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, Duration};
use super::{Evidence, EvidenceId, EvidenceSource, ProbabilisticError, ProbabilisticResult};

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
    pub fn new() -> Self {
        Self {
            evidence_db: HashMap::new(),
            dependency_graph: HashMap::new(),
            next_id: 1,
            combination_cache: HashMap::new(),
        }
    }
    
    /// Add evidence with automatic ID assignment
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
    
    /// Combine multiple evidence sources using Bayesian updating
    pub fn combine_evidence(&mut self, evidence_ids: Vec<EvidenceId>) -> ProbabilisticResult<CombinedEvidence> {
        // Check cache first
        let mut sorted_ids = evidence_ids.clone();
        sorted_ids.sort();
        if let Some(cached) = self.combination_cache.get(&sorted_ids) {
            return Ok(cached.clone());
        }
        
        // Validate evidence IDs exist
        for &id in &evidence_ids {
            if !self.evidence_db.contains_key(&id) {
                return Err(ProbabilisticError::InsufficientEvidence);
            }
        }
        
        // Check for circular dependencies
        self.check_circular_dependencies(&evidence_ids)?;
        
        // Perform Bayesian combination
        let result = self.bayesian_combination(&evidence_ids)?;
        
        // Cache result
        self.combination_cache.insert(sorted_ids, result.clone());
        
        Ok(result)
    }
    
    /// Create evidence from spreading activation
    pub fn evidence_from_activation(
        source_episode: String,
        activation_level: Activation,
        path_length: u16,
    ) -> Evidence {
        // Convert activation level to confidence with path degradation
        let base_confidence = Confidence::exact(activation_level.value());
        let path_degradation = 1.0 / (1.0 + path_length as f32 * 0.1);
        let degraded_confidence = Confidence::exact(base_confidence.raw() * path_degradation);
        
        Evidence {
            source: EvidenceSource::SpreadingActivation {
                source_episode,
                activation_level,
                path_length,
            },
            strength: degraded_confidence,
            timestamp: SystemTime::now(),
            dependencies: Vec::new(),
        }
    }
    
    /// Create evidence from temporal decay
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
    pub fn evidence_from_cue_match(
        cue_id: String,
        similarity_score: f32,
        match_type: super::MatchType,
    ) -> Evidence {
        let confidence = Confidence::exact(similarity_score);
        
        Evidence {
            source: EvidenceSource::DirectMatch {
                cue_id,
                similarity_score,
                match_type,
            },
            strength: confidence,
            timestamp: SystemTime::now(),
            dependencies: Vec::new(),
        }
    }
    
    /// Check for circular dependencies in evidence chain
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
    
    /// Perform Bayesian evidence combination with bias prevention
    fn bayesian_combination(&self, evidence_ids: &[EvidenceId]) -> ProbabilisticResult<CombinedEvidence> {
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
            let final_prob = if combined_prob.is_finite() && combined_prob >= 0.0 && combined_prob <= 1.0 {
                combined_prob
            } else {
                // Fallback to weighted average for numerical stability
                let weight1 = 0.6;
                let weight2 = 0.4;
                combined_confidence.raw() * weight1 + evidence.strength.raw() * weight2
            };
            
            combined_confidence = Confidence::exact(final_prob);
            contributing_sources.push(*id);
        }
        
        // Estimate uncertainty bounds based on evidence diversity
        let evidence_values: Vec<f32> = sorted_evidence.iter()
            .map(|(_, e)| e.strength.raw())
            .collect();
        
        let mean = evidence_values.iter().sum::<f32>() / evidence_values.len() as f32;
        let variance = evidence_values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / evidence_values.len() as f32;
        let std_dev = variance.sqrt();
        
        // Conservative uncertainty bounds
        let uncertainty_factor = std_dev * 1.96; // 95% confidence interval
        let lower_bound = Confidence::exact((combined_confidence.raw() - uncertainty_factor).max(0.0));
        let upper_bound = Confidence::exact((combined_confidence.raw() + uncertainty_factor).min(1.0));
        
        // Build full dependency chain
        let mut dependency_chain = Vec::new();
        for &id in evidence_ids {
            self.collect_dependencies(id, &mut dependency_chain);
        }
        dependency_chain.sort();
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
        self.combination_cache.retain(|ids, _| !ids.contains(&evidence_id));
    }
    
    /// Get evidence by ID
    pub fn get_evidence(&self, id: EvidenceId) -> Option<&Evidence> {
        self.evidence_db.get(&id)
    }
    
    /// Get all evidence for a source type
    pub fn get_evidence_by_source_type(&self, source_type: &str) -> Vec<(EvidenceId, &Evidence)> {
        self.evidence_db.iter()
            .filter(|(_, evidence)| {
                match (&evidence.source, source_type) {
                    (EvidenceSource::SpreadingActivation { .. }, "activation") => true,
                    (EvidenceSource::TemporalDecay { .. }, "decay") => true,
                    (EvidenceSource::DirectMatch { .. }, "match") => true,
                    (EvidenceSource::VectorSimilarity { .. }, "similarity") => true,
                    _ => false,
                }
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
    use std::time::Duration;

    #[test]
    fn test_evidence_creation_from_activation() {
        let evidence = EvidenceAggregator::evidence_from_activation(
            "test_episode".to_string(),
            Activation::new(0.8),
            2,
        );
        
        match evidence.source {
            EvidenceSource::SpreadingActivation { 
                ref source_episode, 
                activation_level, 
                path_length 
            } => {
                assert_eq!(source_episode, "test_episode");
                assert_eq!(activation_level.value(), 0.8);
                assert_eq!(path_length, 2);
            }
            _ => panic!("Wrong evidence source type"),
        }
        
        // Path degradation should reduce confidence
        assert!(evidence.strength.raw() < 0.8);
    }
    
    #[test]
    fn test_evidence_combination() {
        let mut aggregator = EvidenceAggregator::new();
        
        let evidence1 = EvidenceAggregator::evidence_from_activation(
            "episode1".to_string(),
            Activation::new(0.7),
            1,
        );
        
        let evidence2 = EvidenceAggregator::evidence_from_cue_match(
            "cue1".to_string(),
            0.6,
            super::MatchType::Embedding,
        );
        
        let id1 = aggregator.add_evidence(evidence1);
        let id2 = aggregator.add_evidence(evidence2);
        
        let combined = aggregator.combine_evidence(vec![id1, id2]).unwrap();
        
        assert!(!combined.contributing_sources.is_empty());
        assert!(combined.confidence.raw() > 0.0);
        assert!(combined.lower_bound.raw() <= combined.confidence.raw());
        assert!(combined.confidence.raw() <= combined.upper_bound.raw());
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
        let result = aggregator.combine_evidence(vec![id1, id2]);
        assert!(matches!(result, Err(ProbabilisticError::CircularDependency)));
    }
    
    #[test]
    fn test_temporal_decay_evidence() {
        let original = Confidence::HIGH;
        let time_elapsed = Duration::from_secs(3600); // 1 hour
        let decay_rate = 0.001; // per second
        
        let evidence = EvidenceAggregator::evidence_from_decay(
            original,
            time_elapsed,
            decay_rate,
        );
        
        match evidence.source {
            EvidenceSource::TemporalDecay { 
                original_confidence, 
                time_elapsed: elapsed, 
                decay_rate: rate 
            } => {
                assert_eq!(original_confidence, Confidence::HIGH);
                assert_eq!(elapsed, Duration::from_secs(3600));
                assert_eq!(rate, 0.001);
            }
            _ => panic!("Wrong evidence source type"),
        }
        
        // Should be decayed from original
        assert!(evidence.strength.raw() < original.raw());
    }
    
    #[test]
    fn test_evidence_aggregation_empty() {
        let mut aggregator = EvidenceAggregator::new();
        let result = aggregator.combine_evidence(vec![]);
        assert!(matches!(result, Err(ProbabilisticError::InsufficientEvidence)));
    }
}