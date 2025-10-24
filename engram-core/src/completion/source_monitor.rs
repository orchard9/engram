//! Source monitoring for memory field attribution
//!
//! Implements the Source Monitoring Framework (Johnson, Hashtroudi, & Lindsay, 1993)
//! to track whether episode fields are recalled, reconstructed, imagined, or consolidated.
//! Prevents false memory formation through explicit source tracking.

use crate::Confidence;
use crate::completion::{IntegratedField, MemorySource, PartialEpisode, SourceMap};
use std::collections::HashMap;

/// Source attribution engine for pattern completion
///
/// Classifies memory sources based on evidence pathways to prevent false memories.
/// Uses independent confidence scores for source attribution (separate from field confidence).
#[allow(clippy::struct_field_names)] // All fields have _threshold suffix to clarify their purpose
pub struct SourceMonitor {
    /// Minimum confidence for "recalled" label (default: 0.85)
    recalled_threshold: f32,

    /// Minimum confidence for "consolidated" label (default: 0.70)
    consolidated_threshold: f32,

    /// Minimum confidence for "reconstructed" label (default: 0.50)
    reconstructed_threshold: f32,
}

impl SourceMonitor {
    /// Create new source monitor with default thresholds
    #[must_use]
    pub const fn new() -> Self {
        Self {
            recalled_threshold: 0.85,
            consolidated_threshold: 0.70,
            reconstructed_threshold: 0.50,
        }
    }

    /// Create source monitor with custom confidence thresholds
    #[must_use]
    pub const fn with_thresholds(recalled: f32, consolidated: f32, reconstructed: f32) -> Self {
        Self {
            recalled_threshold: recalled,
            consolidated_threshold: consolidated,
            reconstructed_threshold: reconstructed,
        }
    }

    /// Attribute sources for all completed fields
    ///
    /// Classifies each field as Recalled, Reconstructed, Consolidated, or Imagined
    /// based on evidence pathways and confidence levels.
    ///
    /// # Arguments
    ///
    /// * `partial` - Original partial episode (known fields are "recalled")
    /// * `integrated_fields` - Fields completed via evidence integration
    ///
    /// # Returns
    ///
    /// `SourceMap` with source classifications and confidence scores
    #[must_use]
    pub fn attribute_sources(
        &self,
        partial: &PartialEpisode,
        integrated_fields: &HashMap<String, IntegratedField>,
    ) -> SourceMap {
        let mut field_sources = HashMap::new();
        let mut source_confidence = HashMap::new();

        for (field_name, integrated) in integrated_fields {
            // Check if field was in original partial episode
            let in_partial = partial.known_fields.contains_key(field_name);

            let (source, conf) = self.classify_source(field_name, integrated, in_partial);

            field_sources.insert(field_name.clone(), source);
            source_confidence.insert(field_name.clone(), conf);
        }

        SourceMap {
            field_sources,
            source_confidence,
        }
    }

    /// Classify memory source for a single field
    ///
    /// Implements evidence pathway attribution from Johnson et al. (1993):
    /// - Direct Recall: Field present in original cue
    /// - Consolidation: Derived from global semantic patterns
    /// - Reconstruction: Completed from local temporal neighbors
    /// - Imagination: Speculative completion with low confidence
    #[must_use]
    pub fn classify_source(
        &self,
        _field_name: &str,
        integrated: &IntegratedField,
        in_partial: bool,
    ) -> (MemorySource, Confidence) {
        // Recalled: Field was in original partial episode
        if in_partial {
            return (
                MemorySource::Recalled,
                Confidence::exact(self.recalled_threshold),
            );
        }

        let field_confidence = integrated.confidence.raw();

        // Consolidated: Dominated by global patterns with high confidence
        if integrated.global_contribution > 0.7 && field_confidence >= self.consolidated_threshold {
            return (MemorySource::Consolidated, integrated.confidence);
        }

        // Reconstructed: Dominated by local context with moderate confidence
        if integrated.local_contribution > 0.7 && field_confidence >= self.reconstructed_threshold {
            return (MemorySource::Reconstructed, integrated.confidence);
        }

        // Imagined: Low confidence completion (speculative)
        (MemorySource::Imagined, integrated.confidence)
    }

    /// Compute source attribution confidence
    ///
    /// Source confidence is distinct from field confidence. Measures how certain
    /// we are about the source classification itself (not field accuracy).
    ///
    /// # Arguments
    ///
    /// * `source` - Classified memory source
    /// * `field_confidence` - Confidence in field value
    /// * `evidence_consensus` - Agreement ratio among evidence sources [0,1]
    ///
    /// # Returns
    ///
    /// Confidence in source attribution
    #[must_use]
    pub fn compute_source_confidence(
        &self,
        source: MemorySource,
        field_confidence: Confidence,
        evidence_consensus: f32,
    ) -> Confidence {
        let base_confidence = field_confidence.raw();

        // Source confidence increases with evidence consensus
        let source_conf = match source {
            // Recalled: High confidence if field was explicitly provided
            MemorySource::Recalled => self.recalled_threshold,

            // Consolidated: Confidence based on pattern strength + consensus
            MemorySource::Consolidated => {
                base_confidence * evidence_consensus * 1.1 // Slight boost for pattern agreement
            }

            // Reconstructed: Confidence based on neighbor consensus
            MemorySource::Reconstructed => base_confidence * evidence_consensus,

            // Imagined: Low confidence by definition
            MemorySource::Imagined => base_confidence * 0.8, // Penalty for speculation
        };

        Confidence::exact(source_conf.clamp(0.0, 1.0))
    }

    /// Check if source is reliable for retrieval
    ///
    /// Returns true if source is Recalled, Consolidated, or high-confidence Reconstructed.
    /// Filters out Imagined and low-confidence Reconstructed to prevent confabulation.
    #[must_use]
    pub fn is_reliable_source(&self, source: MemorySource, confidence: Confidence) -> bool {
        match source {
            MemorySource::Recalled | MemorySource::Consolidated => true,
            MemorySource::Reconstructed => confidence.raw() >= self.reconstructed_threshold,
            MemorySource::Imagined => false,
        }
    }
}

impl Default for SourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_monitor_creation() {
        let monitor = SourceMonitor::new();

        assert!((monitor.recalled_threshold - 0.85).abs() < 1e-6);
        assert!((monitor.consolidated_threshold - 0.70).abs() < 1e-6);
        assert!((monitor.reconstructed_threshold - 0.50).abs() < 1e-6);
    }

    #[test]
    fn test_source_monitor_custom_thresholds() {
        let monitor = SourceMonitor::with_thresholds(0.9, 0.75, 0.55);

        assert!((monitor.recalled_threshold - 0.9).abs() < 1e-6);
        assert!((monitor.consolidated_threshold - 0.75).abs() < 1e-6);
        assert!((monitor.reconstructed_threshold - 0.55).abs() < 1e-6);
    }

    #[test]
    fn test_classify_recalled_source() {
        let monitor = SourceMonitor::new();

        let integrated = IntegratedField::new(
            "what".to_string(),
            "coffee".to_string(),
            Confidence::exact(0.9),
            0.7,
            0.3,
            true,
        );

        let (source, _) = monitor.classify_source("what", &integrated, true);
        assert_eq!(source, MemorySource::Recalled);
    }

    #[test]
    fn test_classify_consolidated_source() {
        let monitor = SourceMonitor::new();

        let integrated = IntegratedField::new(
            "what".to_string(),
            "coffee".to_string(),
            Confidence::exact(0.8),
            0.2,
            0.8, // Dominated by global patterns
            true,
        );

        let (source, _) = monitor.classify_source("what", &integrated, false);
        assert_eq!(source, MemorySource::Consolidated);
    }

    #[test]
    fn test_classify_reconstructed_source() {
        let monitor = SourceMonitor::new();

        let integrated = IntegratedField::new(
            "what".to_string(),
            "coffee".to_string(),
            Confidence::exact(0.6),
            0.8, // Dominated by local context
            0.2,
            true,
        );

        let (source, _) = monitor.classify_source("what", &integrated, false);
        assert_eq!(source, MemorySource::Reconstructed);
    }

    #[test]
    fn test_classify_imagined_source() {
        let monitor = SourceMonitor::new();

        let integrated = IntegratedField::new(
            "what".to_string(),
            "coffee".to_string(),
            Confidence::exact(0.3), // Low confidence
            0.5,
            0.5,
            false,
        );

        let (source, _) = monitor.classify_source("what", &integrated, false);
        assert_eq!(source, MemorySource::Imagined);
    }

    #[test]
    fn test_compute_source_confidence_recalled() {
        let monitor = SourceMonitor::new();

        let conf =
            monitor.compute_source_confidence(MemorySource::Recalled, Confidence::exact(0.9), 1.0);

        assert!((conf.raw() - 0.85).abs() < 1e-6); // recalled_threshold
    }

    #[test]
    fn test_compute_source_confidence_consolidated() {
        let monitor = SourceMonitor::new();

        let conf = monitor.compute_source_confidence(
            MemorySource::Consolidated,
            Confidence::exact(0.8),
            0.9, // High consensus
        );

        // 0.8 * 0.9 * 1.1 = 0.792
        assert!((conf.raw() - 0.792).abs() < 0.01);
    }

    #[test]
    fn test_compute_source_confidence_reconstructed() {
        let monitor = SourceMonitor::new();

        let conf = monitor.compute_source_confidence(
            MemorySource::Reconstructed,
            Confidence::exact(0.6),
            0.8, // Moderate consensus
        );

        // 0.6 * 0.8 = 0.48
        assert!((conf.raw() - 0.48).abs() < 0.01);
    }

    #[test]
    fn test_compute_source_confidence_imagined() {
        let monitor = SourceMonitor::new();

        let conf =
            monitor.compute_source_confidence(MemorySource::Imagined, Confidence::exact(0.4), 0.5);

        // 0.4 * 0.8 = 0.32 (penalty for speculation)
        assert!((conf.raw() - 0.32).abs() < 0.01);
    }

    #[test]
    fn test_is_reliable_source_recalled() {
        let monitor = SourceMonitor::new();

        assert!(monitor.is_reliable_source(MemorySource::Recalled, Confidence::exact(0.9)));
        assert!(monitor.is_reliable_source(MemorySource::Recalled, Confidence::exact(0.5)));
    }

    #[test]
    fn test_is_reliable_source_consolidated() {
        let monitor = SourceMonitor::new();

        assert!(monitor.is_reliable_source(MemorySource::Consolidated, Confidence::exact(0.8)));
        assert!(monitor.is_reliable_source(MemorySource::Consolidated, Confidence::exact(0.6)));
    }

    #[test]
    fn test_is_reliable_source_reconstructed() {
        let monitor = SourceMonitor::new();

        // Above threshold
        assert!(monitor.is_reliable_source(MemorySource::Reconstructed, Confidence::exact(0.6)));

        // Below threshold
        assert!(!monitor.is_reliable_source(MemorySource::Reconstructed, Confidence::exact(0.4)));
    }

    #[test]
    fn test_is_reliable_source_imagined() {
        let monitor = SourceMonitor::new();

        // Always unreliable
        assert!(!monitor.is_reliable_source(MemorySource::Imagined, Confidence::exact(0.9)));
        assert!(!monitor.is_reliable_source(MemorySource::Imagined, Confidence::exact(0.3)));
    }

    #[test]
    fn test_attribute_sources_integration() {
        let monitor = SourceMonitor::new();

        // Create partial episode with one known field
        let mut known_fields = HashMap::new();
        known_fields.insert("what".to_string(), "coffee".to_string());

        let partial = PartialEpisode {
            known_fields,
            partial_embedding: vec![None; 768],
            cue_strength: Confidence::exact(0.7),
            temporal_context: vec![],
        };

        // Create integrated fields (what=recalled, who=reconstructed)
        let mut integrated_fields = HashMap::new();

        integrated_fields.insert(
            "what".to_string(),
            IntegratedField::new(
                "what".to_string(),
                "coffee".to_string(),
                Confidence::exact(0.9),
                1.0,
                0.0,
                false,
            ),
        );

        integrated_fields.insert(
            "who".to_string(),
            IntegratedField::new(
                "who".to_string(),
                "Alice".to_string(),
                Confidence::exact(0.6),
                0.8,
                0.2,
                false,
            ),
        );

        let source_map = monitor.attribute_sources(&partial, &integrated_fields);

        // "what" should be Recalled (was in partial)
        assert_eq!(
            source_map.field_sources.get("what"),
            Some(&MemorySource::Recalled)
        );

        // "who" should be Reconstructed (local dominated, moderate confidence)
        assert_eq!(
            source_map.field_sources.get("who"),
            Some(&MemorySource::Reconstructed)
        );

        // Both should have source confidence
        assert!(source_map.source_confidence.contains_key("what"));
        assert!(source_map.source_confidence.contains_key("who"));
    }
}
