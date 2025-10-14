//! Integration tests for query expansion with confidence budgets.
//!
//! Tests the full query expansion pipeline including:
//! - Lexicon consultation (synonyms, abbreviations)
//! - Confidence budget enforcement
//! - Variant ranking and truncation
//! - Embedding computation
//! - Integration with semantic activation seeding

use engram_core::{
    ConfidenceBudget,
    embedding::{
        EmbeddingError, EmbeddingProvenance, EmbeddingProvider, EmbeddingWithProvenance,
        ModelVersion,
    },
    query::expansion::{QueryExpander, VariantType},
    query::lexicon::{AbbreviationLexicon, CompositeLexicon, Lexicon, SynonymLexicon},
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
        let value = (text.len() as f32) / 1000.0;
        let vector = vec![value; 768];
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

#[tokio::test]
async fn test_query_expansion_with_synonyms() {
    let provider = Arc::new(MockEmbeddingProvider);
    let synonym_lexicon = Arc::new(SynonymLexicon::with_test_data()) as Arc<dyn Lexicon>;

    let expander = QueryExpander::builder(provider)
        .with_lexicon(synonym_lexicon)
        .max_variants(10)
        .confidence_threshold(0.3)
        .build();

    let expanded = expander.expand("car", Some("en")).await;
    assert!(expanded.is_ok(), "expansion should succeed");

    let result = expanded.unwrap();

    // Should include original + synonyms
    assert!(
        result.variants.len() >= 2,
        "should have at least original + synonyms"
    );

    // Original should be first (highest confidence)
    assert_eq!(result.variants[0].text, "car");
    assert_eq!(result.variants[0].variant_type, VariantType::Original);
    assert_eq!(result.variants[0].confidence, 1.0);

    // Should include "automobile" synonym
    let has_automobile = result.variants.iter().any(|v| v.text == "automobile");
    assert!(has_automobile, "should include 'automobile' synonym");

    // Should include "vehicle" synonym
    let has_vehicle = result.variants.iter().any(|v| v.text == "vehicle");
    assert!(has_vehicle, "should include 'vehicle' synonym");

    // All variants should have embeddings
    for variant in &result.variants {
        assert!(
            variant.has_embedding(),
            "variant should have embedding: {}",
            variant.text
        );
    }
}

#[tokio::test]
async fn test_query_expansion_with_abbreviations() {
    let provider = Arc::new(MockEmbeddingProvider);
    let abbrev_lexicon = Arc::new(AbbreviationLexicon::with_test_data()) as Arc<dyn Lexicon>;

    let expander = QueryExpander::builder(provider)
        .with_lexicon(abbrev_lexicon)
        .max_variants(10)
        .confidence_threshold(0.3)
        .build();

    let expanded = expander.expand("ML", Some("en")).await;
    assert!(expanded.is_ok(), "expansion should succeed");

    let result = expanded.unwrap();

    // Should include original + expansions
    assert!(
        result.variants.len() >= 2,
        "should have at least original + expansions"
    );

    // Original should be first
    assert_eq!(result.variants[0].text, "ML");

    // Should include "machine learning" expansion
    let has_ml = result.variants.iter().any(|v| v.text == "machine learning");
    assert!(has_ml, "should include 'machine learning' expansion");

    // May include "maximum likelihood" if confidence threshold allows
    let has_maxlik = result
        .variants
        .iter()
        .any(|v| v.text == "maximum likelihood");
    // This is optional depending on confidence threshold
    if has_maxlik {
        println!("Included 'maximum likelihood' expansion");
    }
}

#[tokio::test]
async fn test_query_expansion_with_composite_lexicon() {
    let provider = Arc::new(MockEmbeddingProvider);
    let synonym_lex = Arc::new(SynonymLexicon::with_test_data()) as Arc<dyn Lexicon>;
    let abbrev_lex = Arc::new(AbbreviationLexicon::with_test_data()) as Arc<dyn Lexicon>;

    let composite = Arc::new(CompositeLexicon::new(
        "composite".to_string(),
        vec![synonym_lex, abbrev_lex],
    )) as Arc<dyn Lexicon>;

    let expander = QueryExpander::builder(provider)
        .with_lexicon(composite)
        .max_variants(10)
        .confidence_threshold(0.3)
        .build();

    // Test with a word that has synonyms
    let expanded = expander.expand("car", Some("en")).await;
    assert!(expanded.is_ok());
    let result = expanded.unwrap();
    assert!(result.variants.len() >= 2);
    assert_eq!(result.variants[0].text, "car");
}

#[tokio::test]
async fn test_query_expansion_respects_max_variants() {
    let provider = Arc::new(MockEmbeddingProvider);
    let synonym_lexicon = Arc::new(SynonymLexicon::with_test_data()) as Arc<dyn Lexicon>;

    let expander = QueryExpander::builder(provider)
        .with_lexicon(synonym_lexicon)
        .max_variants(2) // Limit to 2 variants
        .confidence_threshold(0.0) // Include all
        .build();

    let expanded = expander.expand("big", Some("en")).await;
    assert!(expanded.is_ok());

    let result = expanded.unwrap();

    // Should respect max_variants limit
    assert_eq!(result.variants.len(), 2, "should respect max_variants=2");

    // Should have truncation flag if more variants were generated
    if result.expansion_metadata.total_variants_generated > 2 {
        assert!(
            result.expansion_metadata.truncated,
            "should be marked as truncated"
        );
    }
}

#[tokio::test]
async fn test_query_expansion_respects_confidence_threshold() {
    let provider = Arc::new(MockEmbeddingProvider);
    let synonym_lexicon = Arc::new(SynonymLexicon::with_test_data()) as Arc<dyn Lexicon>;

    let expander = QueryExpander::builder(provider)
        .with_lexicon(synonym_lexicon)
        .max_variants(10)
        .confidence_threshold(0.75) // High threshold
        .build();

    let expanded = expander.expand("car", Some("en")).await;
    assert!(expanded.is_ok());

    let result = expanded.unwrap();

    // All variants should meet threshold (except original which is always 1.0)
    for variant in &result.variants {
        if variant.variant_type != VariantType::Original {
            assert!(
                variant.confidence >= 0.75,
                "variant '{}' has confidence {} < threshold 0.75",
                variant.text,
                variant.confidence
            );
        }
    }
}

#[tokio::test]
async fn test_query_expansion_metadata() {
    let provider = Arc::new(MockEmbeddingProvider);
    let synonym_lexicon = Arc::new(SynonymLexicon::with_test_data()) as Arc<dyn Lexicon>;

    let expander = QueryExpander::builder(provider)
        .with_lexicon(synonym_lexicon)
        .build();

    let expanded = expander.expand("car", Some("en")).await;
    assert!(expanded.is_ok());

    let result = expanded.unwrap();
    let metadata = &result.expansion_metadata;

    // Should have timing information
    assert!(metadata.expansion_time_us > 0, "should have expansion time");

    // Should list lexicons consulted
    assert!(
        !metadata.lexicons_consulted.is_empty(),
        "should list lexicons"
    );

    // Should track total variants generated
    assert!(
        metadata.total_variants_generated >= result.variants.len(),
        "total_variants_generated should be >= final variant count"
    );
}

#[tokio::test]
async fn test_query_expansion_deduplication() {
    let provider = Arc::new(MockEmbeddingProvider);

    // Create two lexicons with overlapping synonyms
    let mut lex1 = SynonymLexicon::new("lex1".to_string(), Some("en".to_string()));
    lex1.add_synonym("test".to_string(), "example".to_string(), 0.7);

    let mut lex2 = SynonymLexicon::new("lex2".to_string(), Some("en".to_string()));
    lex2.add_synonym("test".to_string(), "example".to_string(), 0.9); // Higher confidence

    let composite = Arc::new(CompositeLexicon::new(
        "test-composite".to_string(),
        vec![Arc::new(lex1), Arc::new(lex2)],
    )) as Arc<dyn Lexicon>;

    let expander = QueryExpander::builder(provider)
        .with_lexicon(composite)
        .build();

    let expanded = expander.expand("test", Some("en")).await;
    assert!(expanded.is_ok());

    let result = expanded.unwrap();

    // Count occurrences of "example"
    let example_count = result
        .variants
        .iter()
        .filter(|v| v.text == "example")
        .count();
    assert_eq!(
        example_count, 1,
        "should deduplicate 'example' to single variant"
    );

    // Should keep the higher confidence
    let example_variant = result.variants.iter().find(|v| v.text == "example");
    assert!(example_variant.is_some());
    assert_eq!(
        example_variant.unwrap().confidence,
        0.9,
        "should keep higher confidence"
    );
}

#[test]
fn test_confidence_budget_basic() {
    let budget = ConfidenceBudget::new(1.0);

    assert!(budget.consume(0.3), "should consume 0.3");
    assert!(
        (budget.remaining() - 0.7).abs() < 0.01,
        "remaining should be ~0.7"
    );

    assert!(budget.consume(0.5), "should consume 0.5");
    assert!(
        (budget.remaining() - 0.2).abs() < 0.01,
        "remaining should be ~0.2"
    );

    assert!(
        !budget.consume(0.3),
        "should fail to consume 0.3 (would exceed)"
    );
    assert!(
        (budget.remaining() - 0.2).abs() < 0.01,
        "remaining should still be ~0.2"
    );
}

#[test]
fn test_confidence_budget_exhaustion() {
    let budget = ConfidenceBudget::new(1.0);

    assert!(budget.consume(1.0), "should consume full budget");
    assert!(budget.is_exhausted(), "should be exhausted");
    assert!(
        !budget.consume(0.01),
        "should fail to consume after exhaustion"
    );
}

#[test]
fn test_confidence_budget_reset() {
    let budget = ConfidenceBudget::new(1.0);

    assert!(budget.consume(0.8));
    assert!((budget.remaining() - 0.2).abs() < 0.01);

    budget.reset();
    assert!(
        (budget.remaining() - 1.0).abs() < 0.01,
        "should reset to initial"
    );
    assert!(budget.consume(0.8), "should be able to consume after reset");
}

#[test]
fn test_confidence_budget_negative_consumption() {
    let budget = ConfidenceBudget::new(1.0);

    assert!(!budget.consume(-0.5), "should reject negative consumption");
    assert!(
        (budget.remaining() - 1.0).abs() < 0.01,
        "budget should be unchanged"
    );
}
