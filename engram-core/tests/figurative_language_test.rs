//! Integration tests for figurative language interpretation.
//!
//! Tests the complete figurative language interpretation system including:
//! - Idiom lexicon loading from JSON
//! - Known idiom expansion (high confidence)
//! - Unknown idiom handling (no hallucination)
//! - Analogy pattern detection (similes, metaphors)
//! - Vector analogy computation (mathematical correctness)

#![allow(clippy::unwrap_used)] // Tests are allowed to use unwrap
#![allow(clippy::float_cmp)] // Tests may compare floats directly for exact values

use engram_core::{
    embedding::{
        EmbeddingError, EmbeddingProvenance, EmbeddingProvider, EmbeddingWithProvenance,
        ModelVersion,
    },
    query::analogy::{AnalogyEngine, AnalogyPattern, AnalogyRelation},
    query::figurative::{FigurativeInterpreter, IdiomLexicon},
};
use std::path::PathBuf;
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

#[test]
fn test_idiom_lexicon_from_json_file() {
    // Find the idiom lexicon file
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../engram-data/figurative/idiom_lexicon.json");

    if !path.exists() {
        // Skip test if file doesn't exist (CI environment may not have data files)
        println!("Skipping test: idiom lexicon file not found at {path:?}");
        return;
    }

    let lexicon = IdiomLexicon::from_json_file(&path).expect("Failed to load idiom lexicon");

    // Verify lexicon loaded correctly
    assert!(
        lexicon.len() >= 50,
        "Expected at least 50 idioms, got {}",
        lexicon.len()
    );
    assert_eq!(lexicon.version(), "1.0.0");
    assert_eq!(lexicon.culture(), "en-US");

    // Test some known idioms
    assert!(lexicon.lookup("break the ice").is_some());
    assert!(lexicon.lookup("piece of cake").is_some());
    assert!(lexicon.lookup("cost an arm and a leg").is_some());
}

#[tokio::test]
async fn test_known_idiom_interpretation() {
    let lexicon = IdiomLexicon::with_test_data();
    let interpreter = FigurativeInterpreter::new(lexicon, Arc::new(MockEmbeddingProvider));

    let variants = interpreter
        .interpret("break the ice", Some("en"))
        .await
        .unwrap();

    // Should return idiom expansions with high confidence
    assert!(!variants.is_empty(), "Expected variants for known idiom");
    assert_eq!(variants[0].text, "initiate conversation");
    assert_eq!(variants[0].confidence, 0.9); // Default idiom confidence
}

#[tokio::test]
async fn test_unknown_idiom_no_hallucination() {
    let lexicon = IdiomLexicon::with_test_data();
    let interpreter = FigurativeInterpreter::new(lexicon, Arc::new(MockEmbeddingProvider));

    let variants = interpreter
        .interpret("flibbertigibbet", Some("en"))
        .await
        .unwrap();

    // Should return empty (no hallucination)
    assert_eq!(
        variants.len(),
        0,
        "Unknown idiom should not generate variants"
    );
}

#[tokio::test]
async fn test_idiom_case_insensitive() {
    let lexicon = IdiomLexicon::with_test_data();
    let interpreter = FigurativeInterpreter::new(lexicon, Arc::new(MockEmbeddingProvider));

    let variants_lower = interpreter
        .interpret("break the ice", Some("en"))
        .await
        .unwrap();
    let variants_upper = interpreter
        .interpret("BREAK THE ICE", Some("en"))
        .await
        .unwrap();
    let variants_mixed = interpreter
        .interpret("Break The Ice", Some("en"))
        .await
        .unwrap();

    assert_eq!(variants_lower.len(), variants_upper.len());
    assert_eq!(variants_lower.len(), variants_mixed.len());
    assert!(!variants_lower.is_empty());
}

#[test]
fn test_analogy_pattern_detection_as() {
    let lexicon = IdiomLexicon::with_test_data();
    let interpreter = FigurativeInterpreter::new(lexicon, Arc::new(MockEmbeddingProvider));

    let pattern = interpreter
        .detect_analogy_pattern("fast as cheetah")
        .unwrap();
    assert_eq!(pattern.target, "fast");
    assert_eq!(pattern.relation, AnalogyRelation::As);
    assert_eq!(pattern.source, "cheetah");
}

#[test]
fn test_analogy_pattern_detection_like() {
    let lexicon = IdiomLexicon::with_test_data();
    let interpreter = FigurativeInterpreter::new(lexicon, Arc::new(MockEmbeddingProvider));

    let pattern = interpreter
        .detect_analogy_pattern("brave like lion")
        .unwrap();
    assert_eq!(pattern.target, "brave");
    assert_eq!(pattern.relation, AnalogyRelation::Like);
    assert_eq!(pattern.source, "lion");
}

#[test]
fn test_analogy_pattern_detection_is() {
    let lexicon = IdiomLexicon::with_test_data();
    let interpreter = FigurativeInterpreter::new(lexicon, Arc::new(MockEmbeddingProvider));

    let pattern = interpreter.detect_analogy_pattern("time is money").unwrap();
    assert_eq!(pattern.target, "time");
    assert_eq!(pattern.relation, AnalogyRelation::Is);
    assert_eq!(pattern.source, "money");
}

#[test]
fn test_analogy_pattern_rejects_common_phrases() {
    let lexicon = IdiomLexicon::with_test_data();
    let interpreter = FigurativeInterpreter::new(lexicon, Arc::new(MockEmbeddingProvider));

    // Should not match trivial phrases
    assert!(interpreter.detect_analogy_pattern("it is good").is_none());
    assert!(interpreter.detect_analogy_pattern("this is test").is_none());
    assert!(
        interpreter
            .detect_analogy_pattern("there is problem")
            .is_none()
    );
}

#[test]
fn test_vector_analogy_subtract() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![0.5, 1.0, 1.5];

    let result = AnalogyEngine::subtract(&a, &b).unwrap();

    assert_eq!(result.len(), 3);
    assert!((result[0] - 0.5).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 1.5).abs() < 1e-6);
}

#[test]
fn test_vector_analogy_add() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![0.5, 1.0, 1.5];

    let result = AnalogyEngine::add(&a, &b).unwrap();

    assert_eq!(result.len(), 3);
    assert!((result[0] - 1.5).abs() < 1e-6);
    assert!((result[1] - 3.0).abs() < 1e-6);
    assert!((result[2] - 4.5).abs() < 1e-6);
}

#[test]
fn test_vector_normalize() {
    let v = vec![3.0, 4.0]; // magnitude = 5.0
    let normalized = AnalogyEngine::normalize(&v).unwrap();

    // Should be unit vector [0.6, 0.8]
    assert!((normalized[0] - 0.6).abs() < 1e-6);
    assert!((normalized[1] - 0.8).abs() < 1e-6);

    // Verify magnitude is 1.0
    let magnitude: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((magnitude - 1.0).abs() < 1e-6);
}

#[test]
fn test_vector_normalize_zero_vector() {
    let zero = vec![0.0, 0.0, 0.0];
    let result = AnalogyEngine::normalize(&zero);

    assert!(result.is_err(), "Zero vector should fail normalization");
}

#[test]
fn test_vector_analogy_dimension_mismatch() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];

    let result = AnalogyEngine::subtract(&a, &b);
    assert!(result.is_err(), "Dimension mismatch should fail");
}

#[test]
fn test_compute_analogy() {
    // Simple analogy: target=2, source1=1, source2=4
    // Result: 2 + (4 - 1) = 5 (normalized)
    let target = vec![2.0, 0.0];
    let source1 = vec![1.0, 0.0];
    let source2 = vec![4.0, 0.0];

    let result = AnalogyEngine::compute_analogy(&target, &source1, &source2).unwrap();

    // Result should be [5,0] normalized to [1,0]
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!(result[1].abs() < 1e-6);
}

#[test]
fn test_cosine_similarity() {
    // Test identical vectors (similarity = 1.0)
    let v1 = vec![1.0, 0.0];
    let similarity = AnalogyEngine::cosine_similarity(&v1, &v1).unwrap();
    assert!((similarity - 1.0).abs() < 1e-6);

    // Test orthogonal vectors (similarity = 0.0)
    let v2 = vec![0.0, 1.0];
    let similarity = AnalogyEngine::cosine_similarity(&v1, &v2).unwrap();
    assert!(similarity.abs() < 1e-6);

    // Test opposite vectors (similarity = -1.0)
    let v3 = vec![-1.0, 0.0];
    let similarity = AnalogyEngine::cosine_similarity(&v1, &v3).unwrap();
    assert!((similarity - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_analogy_pattern_display() {
    let pattern = AnalogyPattern::new(
        "fast".to_string(),
        AnalogyRelation::As,
        "cheetah".to_string(),
    );
    assert_eq!(pattern.to_string(), "fast as cheetah");
}

#[tokio::test]
async fn test_custom_confidence_thresholds() {
    let lexicon = IdiomLexicon::with_test_data();
    let interpreter = FigurativeInterpreter::new(lexicon, Arc::new(MockEmbeddingProvider))
        .with_min_confidence(0.7)
        .with_idiom_confidence(0.95);

    let variants = interpreter
        .interpret("break the ice", Some("en"))
        .await
        .unwrap();

    assert!(!variants.is_empty());
    assert_eq!(variants[0].confidence, 0.95); // Custom idiom confidence
}

#[test]
fn test_idiom_lexicon_metadata() {
    let lexicon = IdiomLexicon::with_test_data();

    assert!(!lexicon.is_empty());
    assert!(lexicon.len() >= 5); // Test data has at least 5 idioms
    assert!(!lexicon.version().is_empty());
    assert!(!lexicon.culture().is_empty());
}

#[tokio::test]
async fn test_multiple_idiom_expansions() {
    let lexicon = IdiomLexicon::with_test_data();
    let interpreter = FigurativeInterpreter::new(lexicon, Arc::new(MockEmbeddingProvider));

    let variants = interpreter
        .interpret("break the ice", Some("en"))
        .await
        .unwrap();

    // Should have multiple expansions
    assert!(
        variants.len() >= 3,
        "Expected multiple expansions for 'break the ice'"
    );

    // All should have idiom confidence
    for variant in &variants {
        assert_eq!(variant.confidence, 0.9);
    }
}
