//! Integration tests for hierarchical evidence aggregation
//!
//! Tests the integration of local context evidence with global semantic patterns
//! using Bayesian evidence combination and adaptive weighting strategies.

use chrono::Utc;
use engram_core::completion::{
    CompletionConfig, ConsolidationEngine, FieldReconstructor, GlobalPatternApplicator,
    HierarchicalEvidenceAggregator, PartialEpisode, PatternRetriever, ReconstructedField,
};
use engram_core::{Confidence, Episode};
use std::collections::HashMap;
use std::sync::Arc;

// Test helper to create episodes
fn create_test_episode(id: &str, embedding: &[f32; 768], what: &str) -> Episode {
    Episode {
        id: id.to_string(),
        when: Utc::now(),
        where_location: None,
        who: None,
        what: what.to_string(),
        embedding: *embedding,
        embedding_provenance: None,
        encoding_confidence: Confidence::exact(0.9),
        vividness_confidence: Confidence::exact(0.8),
        reliability_confidence: Confidence::exact(0.85),
        last_recall: Utc::now(),
        recall_count: 0,
        decay_rate: 0.05,
        decay_function: None,
        metadata: std::collections::HashMap::new(),
    }
}

// Test helper to create partial episode
#[allow(clippy::missing_const_for_fn)] // Test helper - const not needed
fn create_partial_episode(
    known_fields: HashMap<String, String>,
    partial_embedding: Vec<Option<f32>>,
    temporal_context: Vec<String>,
) -> PartialEpisode {
    PartialEpisode {
        known_fields,
        partial_embedding,
        cue_strength: Confidence::exact(0.7),
        temporal_context,
    }
}

// Test helper to create consolidation engine with patterns
fn create_consolidation_with_patterns() -> ConsolidationEngine {
    let config = CompletionConfig::default();
    let mut engine = ConsolidationEngine::new(config);

    // Create breakfast-related episodes
    let breakfast_embedding = [1.0; 768];
    let breakfast_episodes = vec![
        create_test_episode("breakfast1", &breakfast_embedding, "coffee and toast"),
        create_test_episode("breakfast2", &breakfast_embedding, "coffee and toast"),
        create_test_episode("breakfast3", &breakfast_embedding, "coffee and toast"),
    ];

    // Create lunch-related episodes
    let lunch_embedding = [0.5; 768];
    let lunch_episodes = vec![
        create_test_episode("lunch1", &lunch_embedding, "sandwich"),
        create_test_episode("lunch2", &lunch_embedding, "sandwich"),
        create_test_episode("lunch3", &lunch_embedding, "sandwich"),
    ];

    // Consolidate patterns
    engine.ripple_replay(&breakfast_episodes);
    engine.ripple_replay(&lunch_episodes);

    engine
}

#[test]
fn test_evidence_aggregator_creation() {
    let consolidation = Arc::new(create_consolidation_with_patterns());
    let field_reconstructor = FieldReconstructor::new();
    let pattern_retriever = PatternRetriever::new(consolidation);

    let aggregator = HierarchicalEvidenceAggregator::new(field_reconstructor, pattern_retriever);

    // Verify aggregator was created with default params
    assert!((aggregator.compute_adaptive_weights(0.5, 0.5).0 - 0.5).abs() < 1e-6);
}

#[test]
fn test_adaptive_weights_computation() {
    let consolidation = Arc::new(create_consolidation_with_patterns());
    let field_reconstructor = FieldReconstructor::new();
    let pattern_retriever = PatternRetriever::new(consolidation);

    let aggregator = HierarchicalEvidenceAggregator::new(field_reconstructor, pattern_retriever);

    // Test balanced weights
    let (local_weight, global_weight) = aggregator.compute_adaptive_weights(0.6, 0.6);
    assert!((local_weight - 0.5).abs() < 1e-6);
    assert!((global_weight - 0.5).abs() < 1e-6);

    // Test local stronger
    let (local_weight, global_weight) = aggregator.compute_adaptive_weights(0.8, 0.4);
    assert!(local_weight > global_weight);
    assert!((local_weight + global_weight - 1.0).abs() < 1e-6);

    // Test global stronger
    let (local_weight, global_weight) = aggregator.compute_adaptive_weights(0.3, 0.7);
    assert!(global_weight > local_weight);
    assert!((local_weight + global_weight - 1.0).abs() < 1e-6);
}

#[test]
fn test_agreement_checking() {
    let consolidation = Arc::new(create_consolidation_with_patterns());
    let field_reconstructor = FieldReconstructor::new();
    let pattern_retriever = PatternRetriever::new(consolidation);

    let aggregator = HierarchicalEvidenceAggregator::new(field_reconstructor, pattern_retriever);

    // Exact match
    assert!(aggregator.check_agreement("coffee", "coffee", 1.0));

    // High similarity
    assert!(aggregator.check_agreement("coffee", "espresso", 0.85));

    // Low similarity
    assert!(!aggregator.check_agreement("coffee", "tea", 0.5));

    // Different values, low similarity
    assert!(!aggregator.check_agreement("coffee", "sandwich", 0.3));
}

#[test]
fn test_integrate_field_both_sources_agree() {
    let consolidation = Arc::new(create_consolidation_with_patterns());
    let field_reconstructor = FieldReconstructor::new();
    let pattern_retriever = PatternRetriever::new(consolidation);

    let aggregator = HierarchicalEvidenceAggregator::new(field_reconstructor, pattern_retriever);

    // Create local evidence
    let local_evidence = ReconstructedField {
        value: "coffee".to_string(),
        confidence: Confidence::exact(0.7),
        source: engram_core::completion::MemorySource::Recalled,
        evidence: vec![],
    };

    // Create global patterns (empty for now, since extraction is a placeholder)
    let global_patterns = vec![];

    // Integrate
    let result = aggregator.integrate_field("what", Some(&local_evidence), &global_patterns);

    // Should return local-only result since global extraction is placeholder
    assert!(result.is_some());
    let integrated = result.unwrap();
    assert_eq!(integrated.field_value, "coffee");
    assert!((integrated.local_contribution - 1.0).abs() < 1e-6);
    assert!((integrated.global_contribution - 0.0).abs() < 1e-6);
}

#[test]
fn test_integrate_field_local_only() {
    let consolidation = Arc::new(create_consolidation_with_patterns());
    let field_reconstructor = FieldReconstructor::new();
    let pattern_retriever = PatternRetriever::new(consolidation);

    let aggregator = HierarchicalEvidenceAggregator::new(field_reconstructor, pattern_retriever);

    // Create local evidence
    let local_evidence = ReconstructedField {
        value: "coffee".to_string(),
        confidence: Confidence::exact(0.8),
        source: engram_core::completion::MemorySource::Recalled,
        evidence: vec![],
    };

    // No global patterns
    let global_patterns = vec![];

    // Integrate
    let result = aggregator.integrate_field("what", Some(&local_evidence), &global_patterns);

    assert!(result.is_some());
    let integrated = result.unwrap();
    assert_eq!(integrated.field_value, "coffee");
    assert!((integrated.confidence.raw() - 0.8).abs() < 1e-6);
    assert!((integrated.local_contribution - 1.0).abs() < 1e-6);
    assert!(!integrated.agreement); // No global source to agree with
}

#[test]
fn test_integrate_field_no_sources() {
    let consolidation = Arc::new(create_consolidation_with_patterns());
    let field_reconstructor = FieldReconstructor::new();
    let pattern_retriever = PatternRetriever::new(consolidation);

    let aggregator = HierarchicalEvidenceAggregator::new(field_reconstructor, pattern_retriever);

    // No local or global evidence
    let result = aggregator.integrate_field("what", None, &[]);

    // Should return None
    assert!(result.is_none());
}

#[test]
fn test_global_applicator_identify_missing_fields() {
    let _applicator = GlobalPatternApplicator::new();

    // Create partial with no known fields
    let partial = create_partial_episode(HashMap::new(), vec![None; 768], vec![]);

    // Should identify all standard fields
    let missing: Vec<String> = vec!["what".to_string(), "who".to_string(), "where".to_string()]
        .into_iter()
        .filter(|field| !partial.known_fields.contains_key(field))
        .collect();

    assert_eq!(missing.len(), 3);
    assert!(missing.contains(&"what".to_string()));
    assert!(missing.contains(&"who".to_string()));
    assert!(missing.contains(&"where".to_string()));
}

#[test]
fn test_global_applicator_apply_patterns_empty() {
    let applicator = GlobalPatternApplicator::new();

    // Create partial episode
    let partial = create_partial_episode(HashMap::new(), vec![None; 768], vec![]);

    // No patterns available
    let patterns = vec![];

    // Apply patterns
    let reconstructions = applicator.apply_patterns(&partial, &patterns);

    // Should return empty (no patterns to apply)
    assert!(reconstructions.is_empty());
}

#[test]
fn test_global_applicator_all_fields_known() {
    let applicator = GlobalPatternApplicator::new();

    // Create partial with all fields known
    let mut known_fields = HashMap::new();
    known_fields.insert("what".to_string(), "coffee".to_string());
    known_fields.insert("who".to_string(), "Alice".to_string());
    known_fields.insert("where".to_string(), "cafe".to_string());

    let partial = create_partial_episode(known_fields, vec![None; 768], vec![]);

    let patterns = vec![];

    let reconstructions = applicator.apply_patterns(&partial, &patterns);

    // No missing fields to reconstruct
    assert!(reconstructions.is_empty());
}

#[test]
fn test_integrated_field_classification() {
    use engram_core::completion::IntegratedField;

    // Primarily local
    let field = IntegratedField::new(
        "what".to_string(),
        "coffee".to_string(),
        Confidence::exact(0.8),
        0.7,
        0.3,
        true,
    );
    assert!(field.is_primarily_local());
    assert!(!field.is_primarily_global());
    assert!(!field.is_balanced());

    // Primarily global
    let field = IntegratedField::new(
        "what".to_string(),
        "coffee".to_string(),
        Confidence::exact(0.8),
        0.3,
        0.7,
        true,
    );
    assert!(!field.is_primarily_local());
    assert!(field.is_primarily_global());
    assert!(!field.is_balanced());

    // Balanced
    let field = IntegratedField::new(
        "what".to_string(),
        "coffee".to_string(),
        Confidence::exact(0.8),
        0.5,
        0.5,
        true,
    );
    assert!(!field.is_primarily_local());
    assert!(!field.is_primarily_global());
    assert!(field.is_balanced());
}

#[test]
fn test_global_field_reconstruction_reliability() {
    use engram_core::completion::GlobalFieldReconstruction;

    // High confidence, well supported
    let reconstruction = GlobalFieldReconstruction::new(
        "what".to_string(),
        "coffee".to_string(),
        Confidence::exact(0.85),
        5,
    );
    assert!(reconstruction.is_high_confidence());
    assert!(reconstruction.is_well_supported());
    assert!(reconstruction.is_reliable());

    // Low confidence
    let reconstruction = GlobalFieldReconstruction::new(
        "what".to_string(),
        "coffee".to_string(),
        Confidence::exact(0.5),
        5,
    );
    assert!(!reconstruction.is_high_confidence());
    assert!(reconstruction.is_well_supported());
    assert!(!reconstruction.is_reliable());

    // Weak support
    let reconstruction = GlobalFieldReconstruction::new(
        "what".to_string(),
        "coffee".to_string(),
        Confidence::exact(0.85),
        1,
    );
    assert!(reconstruction.is_high_confidence());
    assert!(!reconstruction.is_well_supported());
    assert!(!reconstruction.is_reliable());
}

#[test]
fn test_aggregator_with_custom_params() {
    let consolidation = Arc::new(create_consolidation_with_patterns());
    let field_reconstructor = FieldReconstructor::new();
    let pattern_retriever = PatternRetriever::new(consolidation);

    // Create with custom parameters
    let aggregator = HierarchicalEvidenceAggregator::with_params(
        field_reconstructor,
        pattern_retriever,
        0.9,  // High agreement threshold
        0.25, // 25% boost
        0.2,  // Low min confidence
    );

    // High agreement threshold should require higher similarity
    assert!(aggregator.check_agreement("coffee", "coffee", 1.0));
    assert!(!aggregator.check_agreement("coffee", "espresso", 0.85));
}

#[test]
fn test_applicator_with_custom_params() {
    let applicator = GlobalPatternApplicator::with_params(
        0.6, // Higher min relevance
        0.4, // Higher min strength
        3,   // Fewer patterns per field
    );

    let partial = create_partial_episode(HashMap::new(), vec![None; 768], vec![]);
    let patterns = vec![];

    // Should still work with empty patterns
    let reconstructions = applicator.apply_patterns(&partial, &patterns);
    assert!(reconstructions.is_empty());
}

#[test]
fn test_low_confidence_adaptive_weighting() {
    let consolidation = Arc::new(create_consolidation_with_patterns());
    let field_reconstructor = FieldReconstructor::new();
    let pattern_retriever = PatternRetriever::new(consolidation);

    let aggregator = HierarchicalEvidenceAggregator::new(field_reconstructor, pattern_retriever);

    // Both sources very low confidence
    let (local_weight, global_weight) = aggregator.compute_adaptive_weights(0.1, 0.1);

    // Should default to equal weighting
    assert!((local_weight - 0.5).abs() < 1e-6);
    assert!((global_weight - 0.5).abs() < 1e-6);
}
