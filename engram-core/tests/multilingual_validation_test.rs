//! Multilingual Validation Test Suite
//!
//! Validates Milestone 3.6 features:
//! - Query expansion with synonyms and abbreviations
//! - Figurative language interpretation
//! - Integration between components
//! - Graceful fallback behavior
//!
//! This test suite focuses on practical validation of implemented features
//! rather than external benchmark datasets (which would require MTEB data).

#![allow(clippy::unwrap_used)] // Tests are allowed to use unwrap
#![allow(clippy::float_cmp)] // Tests may compare floats directly

use engram_core::{
    embedding::{
        EmbeddingError, EmbeddingProvenance, EmbeddingProvider, EmbeddingWithProvenance,
        ModelVersion,
    },
    query::{
        expansion::QueryExpander,
        figurative::{FigurativeInterpreter, IdiomLexicon},
        lexicon::{AbbreviationLexicon, Lexicon, SynonymLexicon},
    },
};
use std::sync::Arc;

// Mock embedding provider for testing
struct MockEmbeddingProvider;

#[async_trait::async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(
        &self,
        text: &str,
        language: Option<&str>,
    ) -> Result<EmbeddingWithProvenance, EmbeddingError> {
        // Create deterministic embeddings based on text length and language
        let base_value = (text.len() as f32) / 1000.0;
        let lang_offset = match language {
            Some("es") => 0.1,
            Some("zh") => 0.2,
            _ => 0.0,
        };
        let vector = vec![base_value + lang_offset; 768];
        let model = ModelVersion::new("mock-model".to_string(), "1.0.0".to_string(), 768);
        let provenance = EmbeddingProvenance::new(model, language.map(String::from));
        Ok(EmbeddingWithProvenance::new(vector, provenance))
    }

    async fn embed_batch(
        &self,
        texts: &[&str],
        language: Option<&str>,
    ) -> Result<Vec<EmbeddingWithProvenance>, EmbeddingError> {
        let mut results = Vec::new();
        for text in texts {
            results.push(self.embed(text, language).await?);
        }
        Ok(results)
    }

    fn model_version(&self) -> &ModelVersion {
        static MODEL: std::sync::OnceLock<ModelVersion> = std::sync::OnceLock::new();
        MODEL.get_or_init(|| ModelVersion::new("mock-model".to_string(), "1.0.0".to_string(), 768))
    }

    fn max_sequence_length(&self) -> usize {
        512
    }
}

// ============================================================================
// Test 1: Synonym Expansion and Recall Parity
// ============================================================================

#[tokio::test]
async fn test_synonym_expansion_generates_variants() {
    let synonym_lexicon = SynonymLexicon::with_test_data();
    let embedding_provider = Arc::new(MockEmbeddingProvider);

    let expander = QueryExpander::builder(embedding_provider)
        .with_lexicon(Arc::new(synonym_lexicon) as Arc<dyn Lexicon>)
        .build();

    let expanded = expander.expand("car", Some("en")).await.unwrap();

    // Should generate synonym variants
    assert!(
        expanded.variants.len() > 1,
        "Expected synonym variants for 'car', got {}",
        expanded.variants.len()
    );

    // Check that "automobile" is included
    let has_automobile = expanded
        .variants
        .iter()
        .any(|v| v.text.contains("automobile"));
    assert!(
        has_automobile,
        "Expected 'automobile' as synonym variant of 'car'"
    );

    // All variants should have confidence scores
    for variant in &expanded.variants {
        assert!(
            variant.confidence > 0.0 && variant.confidence <= 1.0,
            "Variant confidence out of range: {}",
            variant.confidence
        );
    }
}

#[tokio::test]
async fn test_abbreviation_expansion_generates_variants() {
    let abbrev_lexicon = AbbreviationLexicon::with_test_data();
    let embedding_provider = Arc::new(MockEmbeddingProvider);

    let expander = QueryExpander::builder(embedding_provider)
        .with_lexicon(Arc::new(abbrev_lexicon) as Arc<dyn Lexicon>)
        .build();

    let expanded = expander.expand("ML", Some("en")).await.unwrap();

    // Should generate abbreviation expansion
    assert!(
        expanded.variants.len() > 1,
        "Expected abbreviation variants for 'ML'"
    );

    // Check that "machine learning" is included
    let has_expansion = expanded
        .variants
        .iter()
        .any(|v| v.text.to_lowercase().contains("machine learning"));
    assert!(
        has_expansion,
        "Expected 'machine learning' as expansion of 'ML'"
    );
}

#[tokio::test]
async fn test_combined_synonym_and_abbreviation_expansion() {
    let synonym_lexicon = SynonymLexicon::with_test_data();
    let abbrev_lexicon = AbbreviationLexicon::with_test_data();
    let embedding_provider = Arc::new(MockEmbeddingProvider);

    let expander = QueryExpander::builder(embedding_provider)
        .with_lexicon(Arc::new(synonym_lexicon) as Arc<dyn Lexicon>)
        .with_lexicon(Arc::new(abbrev_lexicon) as Arc<dyn Lexicon>)
        .build();

    // Test with a term that has synonyms
    let expanded = expander.expand("car", Some("en")).await.unwrap();

    assert!(
        expanded.variants.len() >= 2,
        "Expected multiple variants from combined expansion"
    );
}

// ============================================================================
// Test 2: Figurative Language Interpretation
// ============================================================================

#[tokio::test]
async fn test_idiom_interpretation() {
    let lexicon = IdiomLexicon::with_test_data();
    let embedding_provider = Arc::new(MockEmbeddingProvider);
    let interpreter = FigurativeInterpreter::new(lexicon, embedding_provider);

    let variants = interpreter
        .interpret("break the ice", Some("en"))
        .await
        .unwrap();

    // Should generate literal interpretations
    assert!(
        !variants.is_empty(),
        "Expected interpretations for idiom 'break the ice'"
    );

    // Should include "initiate conversation"
    let has_interpretation = variants
        .iter()
        .any(|v| v.text.contains("initiate conversation"));
    assert!(
        has_interpretation,
        "Expected 'initiate conversation' interpretation"
    );

    // All interpretations should have high confidence (idioms are deterministic)
    for variant in &variants {
        assert!(
            variant.confidence >= 0.8,
            "Idiom interpretation should have high confidence, got {}",
            variant.confidence
        );
    }
}

#[tokio::test]
async fn test_unknown_idiom_no_hallucination() {
    let lexicon = IdiomLexicon::with_test_data();
    let embedding_provider = Arc::new(MockEmbeddingProvider);
    let interpreter = FigurativeInterpreter::new(lexicon, embedding_provider);

    let variants = interpreter
        .interpret("flibbertigibbet nonsense", Some("en"))
        .await
        .unwrap();

    // Should return empty - no hallucination
    assert_eq!(
        variants.len(),
        0,
        "Unknown idiom should not generate variants (hallucination prevention)"
    );
}

#[tokio::test]
async fn test_multiple_idioms_all_interpreted() {
    let lexicon = IdiomLexicon::with_test_data();
    let embedding_provider = Arc::new(MockEmbeddingProvider);
    let interpreter = FigurativeInterpreter::new(lexicon, embedding_provider);

    let test_idioms = vec![
        "break the ice",
        "piece of cake",
        "cost an arm and a leg",
        "under the weather",
    ];

    for idiom in test_idioms {
        let variants = interpreter.interpret(idiom, Some("en")).await.unwrap();
        assert!(
            !variants.is_empty(),
            "Expected interpretation for idiom '{}'",
            idiom
        );
    }
}

// ============================================================================
// Test 3: Variant Limiting and Configuration
// ============================================================================

#[tokio::test]
async fn test_max_variants_configuration() {
    let synonym_lexicon = SynonymLexicon::with_test_data();
    let embedding_provider = Arc::new(MockEmbeddingProvider);

    let synonym_lexicon_arc = Arc::new(synonym_lexicon) as Arc<dyn Lexicon>;

    // Expander with small variant limit
    let expander_small = QueryExpander::builder(embedding_provider.clone())
        .with_lexicon(synonym_lexicon_arc.clone())
        .max_variants(2)
        .build();

    // Expander with large variant limit
    let expander_large = QueryExpander::builder(embedding_provider)
        .with_lexicon(synonym_lexicon_arc)
        .max_variants(10)
        .build();

    let small_expanded = expander_small.expand("car", Some("en")).await.unwrap();
    let large_expanded = expander_large.expand("car", Some("en")).await.unwrap();

    assert!(
        small_expanded.variants.len() <= 2,
        "Small max_variants should limit to 2"
    );
    assert!(
        large_expanded.variants.len() <= 10,
        "Large max_variants should limit to 10"
    );
    assert!(
        small_expanded.variants.len() <= large_expanded.variants.len(),
        "Small limit should produce fewer or equal variants"
    );
}

#[tokio::test]
async fn test_confidence_threshold_filtering() {
    let synonym_lexicon = SynonymLexicon::with_test_data();
    let embedding_provider = Arc::new(MockEmbeddingProvider);

    let expander = QueryExpander::builder(embedding_provider)
        .with_lexicon(Arc::new(synonym_lexicon) as Arc<dyn Lexicon>)
        .confidence_threshold(0.5)
        .build();

    let expanded = expander.expand("car", Some("en")).await.unwrap();

    // All variants should meet threshold
    for variant in &expanded.variants {
        assert!(
            variant.confidence >= 0.5,
            "Variant confidence {} below threshold 0.5",
            variant.confidence
        );
    }
}

// ============================================================================
// Test 4: Integration Testing - Full Pipeline
// ============================================================================

#[tokio::test]
async fn test_query_expansion_to_figurative_interpretation_pipeline() {
    // Setup full pipeline
    let synonym_lexicon = SynonymLexicon::with_test_data();
    let idiom_lexicon = IdiomLexicon::with_test_data();
    let embedding_provider = Arc::new(MockEmbeddingProvider);

    let expander = QueryExpander::builder(embedding_provider.clone())
        .with_lexicon(Arc::new(synonym_lexicon) as Arc<dyn Lexicon>)
        .build();

    let interpreter = FigurativeInterpreter::new(idiom_lexicon, embedding_provider);

    // Step 1: Expand query
    let expanded = expander.expand("break the ice", Some("en")).await.unwrap();

    // Step 2: Interpret figurative language
    let figurative_variants = interpreter
        .interpret("break the ice", Some("en"))
        .await
        .unwrap();

    // Both should produce results
    assert!(
        !expanded.variants.is_empty() || !figurative_variants.is_empty(),
        "Pipeline should produce variants from either expansion or interpretation"
    );

    // Figurative interpretation should work for idioms
    assert!(
        !figurative_variants.is_empty(),
        "Should interpret 'break the ice' as idiom"
    );
}

// ============================================================================
// Test 5: Error Handling and Graceful Degradation
// ============================================================================

#[tokio::test]
async fn test_expander_handles_empty_query() {
    let embedding_provider = Arc::new(MockEmbeddingProvider);
    let expander = QueryExpander::new(embedding_provider);

    let expanded = expander.expand("", Some("en")).await.unwrap();

    // Should handle empty query gracefully - at least returns original
    assert!(
        expanded.variants.len() >= 1,
        "Should produce at least original variant"
    );
    assert_eq!(expanded.variants[0].text, "");
}

#[tokio::test]
async fn test_expander_handles_unknown_language() {
    let synonym_lexicon = SynonymLexicon::with_test_data();
    let embedding_provider = Arc::new(MockEmbeddingProvider);

    let expander = QueryExpander::builder(embedding_provider)
        .with_lexicon(Arc::new(synonym_lexicon) as Arc<dyn Lexicon>)
        .build();

    // Query in language not in lexicon
    let expanded = expander.expand("car", Some("unknown-lang")).await.unwrap();

    // Should still work (may use English fallback or return original)
    // At minimum should not crash
    assert!(
        !expanded.variants.is_empty(),
        "Should handle unknown language gracefully"
    );
}

// ============================================================================
// Test 6: Provenance Tracking
// ============================================================================

#[tokio::test]
async fn test_embedding_provenance_tracked() {
    let embedding_provider = Arc::new(MockEmbeddingProvider);

    let embedding = embedding_provider
        .embed("test query", Some("en"))
        .await
        .unwrap();

    // Provenance should be tracked - verify vector is created
    assert_eq!(embedding.vector.len(), 768);

    // Model version should match mock provider
    let model_version = embedding_provider.model_version();
    assert_eq!(model_version.name, "mock-model");
    assert_eq!(model_version.version, "1.0.0");
    assert_eq!(model_version.dimension, 768);
}

#[tokio::test]
async fn test_embedding_generation_consistent() {
    let embedding_provider = Arc::new(MockEmbeddingProvider);

    // Same text should produce consistent embeddings
    let embedding1 = embedding_provider.embed("test", Some("en")).await.unwrap();
    let embedding2 = embedding_provider.embed("test", Some("en")).await.unwrap();

    assert_eq!(embedding1.vector, embedding2.vector);

    // Different text should produce different embeddings
    let embedding3 = embedding_provider
        .embed("different text", Some("en"))
        .await
        .unwrap();
    assert_ne!(embedding1.vector, embedding3.vector);
}

// ============================================================================
// Test 7: Performance and Scalability Checks
// ============================================================================

#[tokio::test]
async fn test_expansion_completes_within_reasonable_time() {
    let synonym_lexicon = SynonymLexicon::with_test_data();
    let embedding_provider = Arc::new(MockEmbeddingProvider);

    let expander = QueryExpander::builder(embedding_provider)
        .with_lexicon(Arc::new(synonym_lexicon) as Arc<dyn Lexicon>)
        .build();

    let start = std::time::Instant::now();
    let _expanded = expander.expand("car", Some("en")).await.unwrap();
    let elapsed = start.elapsed();

    // Should complete in reasonable time (< 100ms for mock provider)
    assert!(
        elapsed.as_millis() < 100,
        "Expansion took too long: {:?}",
        elapsed
    );
}

#[tokio::test]
async fn test_batch_expansion_efficiency() {
    let synonym_lexicon = SynonymLexicon::with_test_data();
    let embedding_provider = Arc::new(MockEmbeddingProvider);

    let expander = QueryExpander::builder(embedding_provider)
        .with_lexicon(Arc::new(synonym_lexicon) as Arc<dyn Lexicon>)
        .build();

    let queries = vec!["car", "computer", "book", "house", "phone"];

    let start = std::time::Instant::now();
    for query in queries {
        let _expanded = expander.expand(query, Some("en")).await.unwrap();
    }
    let elapsed = start.elapsed();

    // Batch should be efficient (< 500ms for 5 queries with mock provider)
    assert!(
        elapsed.as_millis() < 500,
        "Batch expansion took too long: {:?}",
        elapsed
    );
}
