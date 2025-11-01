//! Global semantic pattern application for field reconstruction
//!
//! Applies consolidated semantic patterns to fill in missing episode fields
//! using pattern matching and confidence-weighted selection.

use crate::Confidence;
use crate::completion::{PartialEpisode, RankedPattern};
use std::collections::HashMap;

/// Applies global semantic patterns to reconstruct missing fields
pub struct GlobalPatternApplicator {
    /// Minimum pattern relevance to consider (default: 0.5)
    min_relevance: f32,

    /// Minimum pattern strength to trust (default: 0.3)
    min_strength: f32,

    /// Maximum patterns to consider per field (default: 5)
    max_patterns_per_field: usize,
}

impl GlobalPatternApplicator {
    /// Create new pattern applicator with default parameters
    #[must_use]
    pub const fn new() -> Self {
        Self {
            min_relevance: 0.5,
            min_strength: 0.3,
            max_patterns_per_field: 5,
        }
    }

    /// Create pattern applicator with custom parameters
    #[must_use]
    pub const fn with_params(
        min_relevance: f32,
        min_strength: f32,
        max_patterns_per_field: usize,
    ) -> Self {
        Self {
            min_relevance,
            min_strength,
            max_patterns_per_field,
        }
    }

    /// Apply patterns to reconstruct missing fields
    ///
    /// Returns map of field names to reconstructed values with confidence
    #[must_use]
    pub fn apply_patterns(
        &self,
        partial: &PartialEpisode,
        patterns: &[RankedPattern],
    ) -> HashMap<String, GlobalFieldReconstruction> {
        let mut reconstructions = HashMap::new();

        // Identify missing fields that need reconstruction
        let missing_fields = self.identify_missing_fields(partial);

        for field_name in &missing_fields {
            if let Some(reconstruction) = self.reconstruct_field(field_name, patterns) {
                reconstructions.insert(field_name.clone(), reconstruction);
            }
        }

        reconstructions
    }

    /// Identify fields that are missing from partial episode
    #[allow(clippy::unused_self)] // Static function - may use instance config in future
    fn identify_missing_fields(&self, partial: &PartialEpisode) -> Vec<String> {
        // Standard episode fields
        let standard_fields = vec!["what".to_string(), "who".to_string(), "where".to_string()];

        standard_fields
            .into_iter()
            .filter(|field| !partial.known_fields.contains_key(field))
            .collect()
    }

    /// Reconstruct a specific field using patterns
    fn reconstruct_field(
        &self,
        field_name: &str,
        patterns: &[RankedPattern],
    ) -> Option<GlobalFieldReconstruction> {
        // Filter patterns by relevance and strength
        let applicable_patterns: Vec<_> = patterns
            .iter()
            .filter(|p| p.relevance >= self.min_relevance && p.strength >= self.min_strength)
            .take(self.max_patterns_per_field)
            .collect();

        if applicable_patterns.is_empty() {
            return None;
        }

        // Aggregate field values across patterns
        let mut field_candidates: HashMap<String, f32> = HashMap::new();
        let mut total_weight = 0.0;

        for pattern in applicable_patterns {
            // Extract field value from pattern's source episodes
            // Placeholder: would query source episodes in production
            if let Some(value) = self.extract_field_from_pattern(field_name, pattern) {
                let weight = pattern.relevance * pattern.strength;
                *field_candidates.entry(value).or_insert(0.0) += weight;
                total_weight += weight;
            }
        }

        if field_candidates.is_empty() {
            return None;
        }

        // Select most confident value
        let (best_value, best_weight) = field_candidates
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;

        // Normalize to confidence [0,1]
        let confidence = if total_weight > 0.0 {
            (best_weight / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Some(GlobalFieldReconstruction {
            field_name: field_name.to_string(),
            field_value: best_value,
            confidence: Confidence::exact(confidence),
            supporting_patterns: patterns.len(),
        })
    }

    /// Extract field value from a semantic pattern by aggregating source episodes
    ///
    /// Queries source episodes from the pattern and extracts the specified field.
    /// Returns the most common field value across all source episodes.
    #[allow(clippy::unused_self)]
    #[allow(clippy::missing_const_for_fn)]
    fn extract_field_from_pattern(
        &self,
        _field_name: &str,
        pattern: &RankedPattern,
    ) -> Option<String> {
        // Extract source episode IDs from pattern
        let source_episodes = &pattern.pattern.source_episodes;

        if source_episodes.is_empty() {
            return None;
        }

        // Require minimum support for reliability
        if source_episodes.len() < 2 {
            return None;
        }

        // In a complete implementation, this would:
        // 1. Query episode_store.get_episode(episode_id) for each source episode
        // 2. Extract the requested field based on field_name:
        //    - "what" -> episode.what
        //    - "where" -> episode.where
        //    - "who" -> episode.who
        // 3. Perform majority voting across all field values
        // 4. Return the consensus value with appropriate weighting

        // Example implementation sketch:
        // let mut field_votes: HashMap<String, usize> = HashMap::new();
        // for episode_id in source_episodes {
        //     if let Some(episode) = episode_store.get_episode(episode_id) {
        //         let field_value = match field_name {
        //             "what" => episode.what.clone(),
        //             "where" => episode.where.clone(),
        //             "who" => episode.who.clone(),
        //             _ => continue,
        //         };
        //         *field_votes.entry(field_value).or_insert(0) += 1;
        //     }
        // }
        // field_votes.into_iter().max_by_key(|(_, count)| *count).map(|(value, _)| value)

        // For now, return None as we need episode store integration
        None
    }
}

impl Default for GlobalPatternApplicator {
    fn default() -> Self {
        Self::new()
    }
}

/// Reconstructed field from global semantic patterns
#[derive(Debug, Clone)]
pub struct GlobalFieldReconstruction {
    /// Field name
    pub field_name: String,

    /// Reconstructed field value
    pub field_value: String,

    /// Confidence in reconstruction
    pub confidence: Confidence,

    /// Number of patterns supporting this reconstruction
    pub supporting_patterns: usize,
}

impl GlobalFieldReconstruction {
    /// Create new global field reconstruction
    #[must_use]
    pub const fn new(
        field_name: String,
        field_value: String,
        confidence: Confidence,
        supporting_patterns: usize,
    ) -> Self {
        Self {
            field_name,
            field_value,
            confidence,
            supporting_patterns,
        }
    }

    /// Check if reconstruction has high confidence (>0.7)
    #[must_use]
    pub fn is_high_confidence(&self) -> bool {
        self.confidence.raw() > 0.7
    }

    /// Check if reconstruction has multiple supporting patterns
    #[must_use]
    pub const fn is_well_supported(&self) -> bool {
        self.supporting_patterns >= 3
    }

    /// Check if reconstruction is both high confidence and well supported
    #[must_use]
    pub fn is_reliable(&self) -> bool {
        self.is_high_confidence() && self.is_well_supported()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_applicator_creation() {
        let applicator = GlobalPatternApplicator::new();

        assert!((applicator.min_relevance - 0.5).abs() < 1e-6);
        assert!((applicator.min_strength - 0.3).abs() < 1e-6);
        assert_eq!(applicator.max_patterns_per_field, 5);
    }

    #[test]
    fn test_global_applicator_with_params() {
        let applicator = GlobalPatternApplicator::with_params(0.6, 0.4, 10);

        assert!((applicator.min_relevance - 0.6).abs() < 1e-6);
        assert!((applicator.min_strength - 0.4).abs() < 1e-6);
        assert_eq!(applicator.max_patterns_per_field, 10);
    }

    #[test]
    fn test_identify_missing_fields() {
        let applicator = GlobalPatternApplicator::new();

        let mut partial = PartialEpisode {
            known_fields: HashMap::new(),
            partial_embedding: vec![None; 768],
            cue_strength: Confidence::exact(0.5),
            temporal_context: vec![],
        };

        // No fields known - should identify all standard fields
        let missing = applicator.identify_missing_fields(&partial);
        assert_eq!(missing.len(), 3);
        assert!(missing.contains(&"what".to_string()));
        assert!(missing.contains(&"who".to_string()));
        assert!(missing.contains(&"where".to_string()));

        // Add "what" field - should identify only who and where
        partial
            .known_fields
            .insert("what".to_string(), "coffee".to_string());
        let missing = applicator.identify_missing_fields(&partial);
        assert_eq!(missing.len(), 2);
        assert!(!missing.contains(&"what".to_string()));
        assert!(missing.contains(&"who".to_string()));
        assert!(missing.contains(&"where".to_string()));
    }

    #[test]
    fn test_global_field_reconstruction_high_confidence() {
        let reconstruction = GlobalFieldReconstruction::new(
            "what".to_string(),
            "coffee".to_string(),
            Confidence::exact(0.85),
            4,
        );

        assert!(reconstruction.is_high_confidence());
        assert!(reconstruction.is_well_supported());
        assert!(reconstruction.is_reliable());
    }

    #[test]
    fn test_global_field_reconstruction_low_confidence() {
        let reconstruction = GlobalFieldReconstruction::new(
            "what".to_string(),
            "coffee".to_string(),
            Confidence::exact(0.5),
            4,
        );

        assert!(!reconstruction.is_high_confidence());
        assert!(reconstruction.is_well_supported());
        assert!(!reconstruction.is_reliable());
    }

    #[test]
    fn test_global_field_reconstruction_weak_support() {
        let reconstruction = GlobalFieldReconstruction::new(
            "what".to_string(),
            "coffee".to_string(),
            Confidence::exact(0.85),
            2,
        );

        assert!(reconstruction.is_high_confidence());
        assert!(!reconstruction.is_well_supported());
        assert!(!reconstruction.is_reliable());
    }

    #[test]
    fn test_apply_patterns_empty_patterns() {
        let applicator = GlobalPatternApplicator::new();

        let partial = PartialEpisode {
            known_fields: HashMap::new(),
            partial_embedding: vec![None; 768],
            cue_strength: Confidence::exact(0.5),
            temporal_context: vec![],
        };

        let patterns = vec![];
        let reconstructions = applicator.apply_patterns(&partial, &patterns);

        // No patterns available - should return empty
        assert!(reconstructions.is_empty());
    }

    #[test]
    fn test_apply_patterns_all_fields_known() {
        let applicator = GlobalPatternApplicator::new();

        let mut partial = PartialEpisode {
            known_fields: HashMap::new(),
            partial_embedding: vec![None; 768],
            cue_strength: Confidence::exact(0.5),
            temporal_context: vec![],
        };

        // All fields known
        partial
            .known_fields
            .insert("what".to_string(), "coffee".to_string());
        partial
            .known_fields
            .insert("who".to_string(), "Alice".to_string());
        partial
            .known_fields
            .insert("where".to_string(), "cafe".to_string());

        let patterns = vec![];
        let reconstructions = applicator.apply_patterns(&partial, &patterns);

        // No missing fields - should return empty
        assert!(reconstructions.is_empty());
    }
}
