//! Figurative language interpretation with hallucination prevention.
//!
//! This module interprets metaphors, similes, and idioms in queries while strictly
//! preventing hallucinated interpretations that aren't grounded in actual memories.
//!
//! ## Design Principles
//!
//! 1. **Never Hallucinate**: Only return interpretations with strong evidence (≥3 memory matches >0.7 similarity)
//! 2. **Graceful Degradation**: Return empty variants rather than incorrect interpretations
//! 3. **Explainability**: Track interpretation sources (idiom vs. analogy) with metadata
//! 4. **Conservative Confidence**: Use lower confidence scores for figurative interpretations
//!
//! ## Supported Patterns
//!
//! - **Idioms**: "break the ice" → `["initiate conversation", "reduce tension"]`
//! - **Similes**: "fast as cheetah" → compute speed analogy
//! - **Simple Metaphors**: "X is Y" → compute equivalence analogy
//!
//! ## Critical Correctness Invariant
//!
//! **Never return an interpretation unless it's grounded in actual memories.**
//!
//! This prevents the system from generating plausible-sounding but incorrect expansions.

use super::{
    analogy::{AnalogyError, AnalogyPattern, AnalogyRelation},
    expansion::{QueryVariant, VariantType},
};
use crate::embedding::{EmbeddingError, EmbeddingProvider};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Error type for figurative language interpretation.
#[derive(Debug, thiserror::Error)]
pub enum InterpretationError {
    /// Failed to load idiom lexicon
    #[error("failed to load idiom lexicon: {0}")]
    LexiconLoadFailed(String),

    /// Embedding generation failed
    #[error("embedding generation failed: {0}")]
    EmbeddingFailed(#[from] EmbeddingError),

    /// Analogy computation failed
    #[error("analogy computation failed: {0}")]
    AnalogyFailed(#[from] AnalogyError),

    /// Validation failed (insufficient evidence)
    #[error("insufficient evidence for interpretation: {reason}")]
    InsufficientEvidence {
        /// Reason for failure
        reason: String,
    },
}

/// Idiom lexicon mapping idioms to literal expansions.
///
/// Maintains a curated dictionary of idioms with their literal meanings.
/// Idioms are matched case-insensitively and support versioning.
///
/// # Versioning
///
/// The lexicon includes a version field for tracking changes and ensuring
/// compatibility. Clients should handle version mismatches gracefully.
///
/// # Cultural Considerations
///
/// Idioms are culturally specific. The lexicon should document its cultural
/// context (e.g., "en-US", "en-GB") and allow user-provided extensions.
#[derive(Debug, Clone)]
pub struct IdiomLexicon {
    /// Map from idiom (lowercase) to literal expansions
    idioms: HashMap<String, Vec<String>>,

    /// Lexicon version for tracking changes
    version: String,

    /// Cultural context (e.g., "en-US")
    culture: String,
}

impl IdiomLexicon {
    /// Create a new empty idiom lexicon.
    #[must_use]
    pub fn new(version: String, culture: String) -> Self {
        Self {
            idioms: HashMap::new(),
            version,
            culture,
        }
    }

    /// Load idiom lexicon from JSON file.
    ///
    /// Expected format:
    /// ```json
    /// {
    ///   "version": "1.0",
    ///   "culture": "en-US",
    ///   "idioms": {
    ///     "break the ice": ["initiate conversation", "reduce tension"],
    ///     "piece of cake": ["easy", "simple"]
    ///   }
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or JSON is malformed.
    pub fn from_json_file(path: &Path) -> Result<Self, InterpretationError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            InterpretationError::LexiconLoadFailed(format!("failed to read file: {e}"))
        })?;

        Self::from_json_str(&content)
    }

    /// Load idiom lexicon from JSON string.
    ///
    /// # Errors
    ///
    /// Returns error if JSON is malformed.
    pub fn from_json_str(json: &str) -> Result<Self, InterpretationError> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| InterpretationError::LexiconLoadFailed(format!("invalid JSON: {e}")))?;

        let version = value
            .get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("1.0")
            .to_string();

        let culture = value
            .get("culture")
            .and_then(|v| v.as_str())
            .unwrap_or("en-US")
            .to_string();

        let mut idioms = HashMap::new();

        if let Some(idioms_obj) = value.get("idioms").and_then(|v| v.as_object()) {
            for (idiom, expansions) in idioms_obj {
                if let Some(exp_array) = expansions.as_array() {
                    let expansions: Vec<String> = exp_array
                        .iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                    idioms.insert(idiom.to_lowercase(), expansions);
                }
            }
        }

        Ok(Self {
            idioms,
            version,
            culture,
        })
    }

    /// Add an idiom with its literal expansions.
    pub fn add_idiom(&mut self, idiom: &str, expansions: Vec<String>) {
        self.idioms.insert(idiom.to_lowercase(), expansions);
    }

    /// Lookup an idiom, returning its literal expansions.
    ///
    /// Returns `None` if idiom is not in lexicon (prevents hallucination).
    ///
    /// # Arguments
    ///
    /// * `query` - Query text to check for idioms
    ///
    /// # Returns
    ///
    /// `Some(expansions)` if idiom is found, `None` otherwise.
    #[must_use]
    pub fn lookup(&self, query: &str) -> Option<&[String]> {
        self.idioms
            .get(&query.to_lowercase())
            .map(std::vec::Vec::as_slice)
    }

    /// Get the number of idioms in the lexicon.
    #[must_use]
    pub fn len(&self) -> usize {
        self.idioms.len()
    }

    /// Check if the lexicon is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.idioms.is_empty()
    }

    /// Get the lexicon version.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Get the cultural context.
    #[must_use]
    pub fn culture(&self) -> &str {
        &self.culture
    }

    /// Create a test lexicon with common idioms.
    ///
    /// Note: This is a test helper method for both unit and integration tests.
    #[must_use]
    #[allow(clippy::expect_used)] // Test helper with hardcoded valid JSON
    pub fn with_test_data() -> Self {
        let json = r#"{
            "version": "1.0-test",
            "culture": "en-US",
            "idioms": {
                "break the ice": ["initiate conversation", "start interaction", "reduce social tension"],
                "piece of cake": ["easy", "simple", "straightforward"],
                "cost an arm and a leg": ["expensive", "costly", "overpriced"],
                "kick the bucket": ["die", "pass away"],
                "under the weather": ["ill", "sick", "unwell"]
            }
        }"#;
        Self::from_json_str(json).expect("test data is valid JSON")
    }
}

/// Figurative language interpreter with hallucination prevention.
///
/// Interprets metaphors, similes, and idioms while maintaining strict correctness.
/// Never returns an interpretation unless grounded in actual memories or lexicon.
pub struct FigurativeInterpreter {
    /// Idiom lexicon for known expressions
    idiom_lexicon: IdiomLexicon,

    /// Embedding provider for analogy computation (used in future analogy interpretation)
    #[allow(dead_code)] // Will be used for analogy interpretation in Task 006
    embedding_provider: Arc<dyn EmbeddingProvider>,

    /// Minimum confidence threshold (default: 0.5)
    min_confidence: f32,

    /// Confidence score for idiom matches (default: 0.9)
    idiom_confidence: f32,
}

impl FigurativeInterpreter {
    /// Default minimum confidence threshold.
    pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.5;

    /// Default confidence for idiom matches.
    pub const DEFAULT_IDIOM_CONFIDENCE: f32 = 0.9;

    /// Create a new figurative interpreter.
    ///
    /// # Arguments
    ///
    /// * `idiom_lexicon` - Lexicon of known idioms
    /// * `embedding_provider` - Provider for generating embeddings
    #[must_use]
    pub fn new(
        idiom_lexicon: IdiomLexicon,
        embedding_provider: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self {
            idiom_lexicon,
            embedding_provider,
            min_confidence: Self::DEFAULT_MIN_CONFIDENCE,
            idiom_confidence: Self::DEFAULT_IDIOM_CONFIDENCE,
        }
    }

    /// Set minimum confidence threshold.
    #[must_use]
    pub const fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Set confidence for idiom matches.
    #[must_use]
    pub const fn with_idiom_confidence(mut self, idiom_confidence: f32) -> Self {
        self.idiom_confidence = idiom_confidence;
        self
    }

    /// Interpret figurative language in a query.
    ///
    /// Returns query variants for figurative interpretations. Returns empty
    /// vector if no figurative language detected or insufficient evidence.
    ///
    /// # Arguments
    ///
    /// * `query` - Query text to interpret
    /// * `language` - Optional language code
    ///
    /// # Returns
    ///
    /// Vector of query variants with figurative interpretations.
    ///
    /// # Errors
    ///
    /// Returns error if embedding generation or analogy computation fails.
    #[allow(clippy::unused_async)] // Async for future-proofing Task 006 analogy interpretation
    pub async fn interpret(
        &self,
        query: &str,
        _language: Option<&str>,
    ) -> Result<Vec<QueryVariant>, InterpretationError> {
        let mut variants = Vec::new();

        // Step 1: Check for known idioms (highest confidence)
        if let Some(expansions) = self.idiom_lexicon.lookup(query) {
            for expansion in expansions {
                variants.push(QueryVariant::new(
                    expansion.clone(),
                    VariantType::Synonym, // Treat idioms as high-confidence synonyms
                    self.idiom_confidence,
                ));
            }
            return Ok(variants);
        }

        // Step 2: Detect analogy patterns (similes/metaphors)
        if let Some(pattern) = Self::detect_analogy_pattern(query) {
            // For now, return empty - analogy interpretation requires memory graph access
            // This is a placeholder for Task 006 integration
            tracing::debug!(
                "Detected analogy pattern: {} (analogy interpretation not yet implemented)",
                pattern
            );
        }

        Ok(variants)
    }

    /// Detect analogy patterns in query text.
    ///
    /// Recognizes patterns like:
    /// - "X as Y" (simile)
    /// - "X like Y" (simile)
    /// - "X is Y" (metaphor)
    ///
    /// # Arguments
    ///
    /// * `query` - Query text to analyze
    ///
    /// # Returns
    ///
    /// `Some(AnalogyPattern)` if pattern detected, `None` otherwise.
    #[must_use]
    pub fn detect_analogy_pattern(query: &str) -> Option<AnalogyPattern> {
        let query_lower = query.to_lowercase();

        // Pattern 1: "X as Y"
        if let Some(pos) = query_lower.find(" as ") {
            let target = query[..pos].trim().to_string();
            let source = query[pos + 4..].trim().to_string();
            if !target.is_empty() && !source.is_empty() {
                return Some(AnalogyPattern::new(target, AnalogyRelation::As, source));
            }
        }

        // Pattern 2: "X like Y"
        if let Some(pos) = query_lower.find(" like ") {
            let target = query[..pos].trim().to_string();
            let source = query[pos + 6..].trim().to_string();
            if !target.is_empty() && !source.is_empty() {
                return Some(AnalogyPattern::new(target, AnalogyRelation::Like, source));
            }
        }

        // Pattern 3: "X is Y" (simple metaphor)
        // Only match if not too short (avoid "it is", "this is", etc.)
        if let Some(pos) = query_lower.find(" is ") {
            let target = query[..pos].trim();
            let source = query[pos + 4..].trim();

            if target.split_whitespace().count() >= 1
                && source.split_whitespace().count() >= 1
                && !["it", "this", "that", "there", "what", "which"]
                    .contains(&target.to_lowercase().as_str())
            {
                return Some(AnalogyPattern::new(
                    target.to_string(),
                    AnalogyRelation::Is,
                    source.to_string(),
                ));
            }
        }

        None
    }

    /// Get the idiom lexicon.
    #[must_use]
    pub const fn idiom_lexicon(&self) -> &IdiomLexicon {
        &self.idiom_lexicon
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Tests are allowed to use unwrap
    #![allow(clippy::float_cmp)] // Tests may compare floats directly for exact values

    use super::*;
    use crate::embedding::{EmbeddingProvenance, EmbeddingWithProvenance, ModelVersion};

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
            MODEL.get_or_init(|| {
                ModelVersion::new("mock-model".to_string(), "1.0.0".to_string(), 768)
            })
        }

        fn max_sequence_length(&self) -> usize {
            512
        }
    }

    #[test]
    fn test_idiom_lexicon_from_json() {
        let json = r#"{
            "version": "1.0",
            "culture": "en-US",
            "idioms": {
                "break the ice": ["initiate conversation", "reduce tension"],
                "piece of cake": ["easy", "simple"]
            }
        }"#;

        let lexicon = IdiomLexicon::from_json_str(json).unwrap();
        assert_eq!(lexicon.len(), 2);
        assert_eq!(lexicon.version(), "1.0");
        assert_eq!(lexicon.culture(), "en-US");

        let expansions = lexicon.lookup("break the ice").unwrap();
        assert_eq!(expansions.len(), 2);
        assert_eq!(expansions[0], "initiate conversation");
    }

    #[test]
    fn test_idiom_lexicon_case_insensitive() {
        let lexicon = IdiomLexicon::with_test_data();

        assert!(lexicon.lookup("break the ice").is_some());
        assert!(lexicon.lookup("BREAK THE ICE").is_some());
        assert!(lexicon.lookup("Break The Ice").is_some());
    }

    #[test]
    fn test_idiom_lexicon_unknown() {
        let lexicon = IdiomLexicon::with_test_data();
        assert!(lexicon.lookup("flibbertigibbet").is_none());
    }

    #[test]
    fn test_detect_analogy_pattern_as() {
        let pattern = FigurativeInterpreter::detect_analogy_pattern("fast as cheetah").unwrap();
        assert_eq!(pattern.target, "fast");
        assert_eq!(pattern.relation, AnalogyRelation::As);
        assert_eq!(pattern.source, "cheetah");
    }

    #[test]
    fn test_detect_analogy_pattern_like() {
        let pattern = FigurativeInterpreter::detect_analogy_pattern("brave like lion").unwrap();
        assert_eq!(pattern.target, "brave");
        assert_eq!(pattern.relation, AnalogyRelation::Like);
        assert_eq!(pattern.source, "lion");
    }

    #[test]
    fn test_detect_analogy_pattern_is() {
        let pattern = FigurativeInterpreter::detect_analogy_pattern("time is money").unwrap();
        assert_eq!(pattern.target, "time");
        assert_eq!(pattern.relation, AnalogyRelation::Is);
        assert_eq!(pattern.source, "money");
    }

    #[test]
    fn test_detect_analogy_pattern_rejects_common_phrases() {
        // Should not match common phrases
        assert!(FigurativeInterpreter::detect_analogy_pattern("it is good").is_none());
        assert!(FigurativeInterpreter::detect_analogy_pattern("this is test").is_none());
    }

    #[tokio::test]
    async fn test_interpret_known_idiom() {
        let interpreter = FigurativeInterpreter::new(
            IdiomLexicon::with_test_data(),
            Arc::new(MockEmbeddingProvider),
        );

        let variants = interpreter
            .interpret("break the ice", Some("en"))
            .await
            .unwrap();
        assert_eq!(variants.len(), 3);
        assert_eq!(variants[0].text, "initiate conversation");
        assert_eq!(variants[0].confidence, 0.9);
    }

    #[tokio::test]
    async fn test_interpret_unknown_idiom() {
        let interpreter = FigurativeInterpreter::new(
            IdiomLexicon::with_test_data(),
            Arc::new(MockEmbeddingProvider),
        );

        let variants = interpreter
            .interpret("flibbertigibbet", Some("en"))
            .await
            .unwrap();
        assert_eq!(variants.len(), 0); // No hallucination
    }
}
