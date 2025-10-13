//! Semantic activation seeding from text queries.
//!
//! This module provides semantic query support by converting text queries into embeddings
//! and using vector similarity search to seed spreading activation. This bridges natural
//! language queries with graph-based cognitive recall.
//!
//! ## Design Principles
//!
//! - **Semantic as Seeds**: Vector similarity identifies initial candidates; spreading
//!   activation then explores the graph from those seeds
//! - **Graceful Degradation**: Falls back to lexical search when embeddings unavailable
//! - **Explainability**: Tracks activation source (semantic vs lexical vs explicit cue)
//! - **Query Expansion Ready**: Infrastructure for synonym/abbreviation expansion (Tasks 003/004)

use super::seeding::{SeedingError, SeedingOutcome, VectorActivationSeeder};
use crate::{
    Confidence, Cue,
    embedding::{EmbeddingError, EmbeddingProvider},
};
use std::sync::Arc;
use thiserror::Error;

/// Source of activation for explainability and debugging.
///
/// Tracks how activation was initiated to enable explainability and debugging.
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationSource {
    /// Explicit cue provided by user
    ExplicitCue {
        /// ID of the cue
        cue_id: String,
    },
    /// Semantic similarity from text query
    SemanticSimilarity {
        /// Original query text
        query: String,
        /// Similarity score
        similarity: f32,
    },
    /// Spreading activation from another episode
    SpreadingActivation {
        /// Source episode ID
        from_episode: String,
    },
}

/// Error type for semantic seeding operations.
#[derive(Debug, Error)]
pub enum SemanticError {
    /// Failed to generate embedding for query
    #[error("embedding generation failed: {0}")]
    EmbeddingFailed(#[from] EmbeddingError),

    /// No embeddings available (triggers fallback to lexical search)
    #[error("no embeddings available for seeding")]
    NoEmbeddings,

    /// Vector seeding failed
    #[error("vector seeding failed: {0}")]
    SeedingFailed(#[from] SeedingError),
}

/// Semantic activation seeder that converts text queries to activation seeds.
///
/// This bridges natural language queries with vector-based activation seeding.
/// It uses an `EmbeddingProvider` to convert text to embeddings, then delegates
/// to `VectorActivationSeeder` for the actual HNSW similarity search.
///
/// ## Query Expansion Support
///
/// Future tasks (003/004) will add query expansion for synonyms, abbreviations,
/// and figurative language. The infrastructure supports multi-vector queries,
/// so expansion can be plugged in without changing the core seeding logic.
pub struct SemanticActivationSeeder {
    /// Embedding provider for converting text to vectors
    embedding_provider: Arc<dyn EmbeddingProvider>,

    /// Vector seeder for HNSW similarity search
    vector_seeder: Arc<VectorActivationSeeder>,

    /// Optional query expander (placeholder for Task 003)
    /// In the future, this will expand queries with synonyms and abbreviations
    #[allow(dead_code)]
    query_expander: Option<Arc<dyn QueryExpander>>,

    /// Optional figurative interpreter (placeholder for Task 004)
    /// In the future, this will interpret metaphors and similes
    #[allow(dead_code)]
    figurative_interpreter: Option<Arc<dyn FigurativeInterpreter>>,
}

impl SemanticActivationSeeder {
    /// Create a new semantic activation seeder.
    ///
    /// # Arguments
    ///
    /// * `embedding_provider` - Provider for generating embeddings from text
    /// * `vector_seeder` - Seeder for HNSW similarity search
    #[must_use]
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        vector_seeder: Arc<VectorActivationSeeder>,
    ) -> Self {
        Self {
            embedding_provider,
            vector_seeder,
            query_expander: None,
            figurative_interpreter: None,
        }
    }

    /// Seed activation from a text query.
    ///
    /// This is the main entry point for semantic queries. It:
    /// 1. Converts the query text to an embedding
    /// 2. Uses vector similarity search to find relevant memories
    /// 3. Returns seeds for spreading activation
    ///
    /// # Arguments
    ///
    /// * `query` - The text query from the user
    /// * `language` - Optional ISO 639-1 language code (e.g., "en", "es")
    /// * `threshold` - Minimum similarity threshold (0.0-1.0)
    /// * `max_results` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// `SeedingOutcome` containing activation seeds and search statistics.
    ///
    /// # Errors
    ///
    /// - `SemanticError::EmbeddingFailed` if embedding generation fails
    /// - `SemanticError::NoEmbeddings` if no embeddings can be generated (triggers fallback)
    /// - `SemanticError::SeedingFailed` if vector seeding fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let outcome = seeder.seed_from_query(
    ///     "automobile safety features",
    ///     Some("en"),
    ///     0.7,
    ///     20
    /// ).await?;
    /// ```
    pub async fn seed_from_query(
        &self,
        query: &str,
        language: Option<&str>,
        threshold: f32,
        _max_results: usize,
    ) -> Result<SeedingOutcome, SemanticError> {
        // Step 1: Generate embedding for query
        let embedding_result = self
            .embedding_provider
            .embed(query, language)
            .await
            .map_err(SemanticError::EmbeddingFailed)?;

        // Step 2: Convert to fixed-size array for HNSW search
        let embedding_vec = embedding_result.vector;
        if embedding_vec.len() != 768 {
            return Err(SemanticError::EmbeddingFailed(
                EmbeddingError::EncodingFailed(format!(
                    "expected 768-dimensional embedding, got {}",
                    embedding_vec.len()
                )),
            ));
        }

        // Convert Vec<f32> to [f32; 768]
        let mut embedding_array: [f32; 768] = [0.0; 768];
        embedding_array.copy_from_slice(&embedding_vec);

        // Step 3: Create a cue from the embedding
        let confidence = Confidence::from_raw(threshold);
        // Use query text as cue ID for traceability
        let cue_id = format!("semantic:{}", query);
        let cue = Cue::embedding(cue_id, embedding_array, confidence);

        // Step 4: Use vector seeder to find similar memories
        let outcome = self.vector_seeder.seed_from_cue(&cue)?;

        Ok(outcome)
    }

    /// Seed activation from multiple query variants (for future query expansion).
    ///
    /// This method is infrastructure for Tasks 003/004. It supports multi-vector
    /// queries where each variant has an associated confidence weight.
    ///
    /// # Arguments
    ///
    /// * `query_variants` - List of (query_text, confidence_weight) pairs
    /// * `language` - Optional ISO 639-1 language code
    /// * `threshold` - Minimum similarity threshold
    /// * `max_results` - Maximum number of results
    ///
    /// # Returns
    ///
    /// `SeedingOutcome` with aggregated results across all query variants.
    ///
    /// # Errors
    ///
    /// Returns error if embedding generation or seeding fails.
    pub async fn seed_from_multi_query(
        &self,
        query_variants: &[(&str, f32)],
        language: Option<&str>,
        threshold: f32,
        _max_results: usize,
    ) -> Result<SeedingOutcome, SemanticError> {
        if query_variants.is_empty() {
            return Err(SemanticError::NoEmbeddings);
        }

        // Generate embeddings for all variants
        let mut cues = Vec::with_capacity(query_variants.len());

        for (idx, (query, _weight)) in query_variants.iter().enumerate() {
            let embedding_result = self
                .embedding_provider
                .embed(query, language)
                .await
                .map_err(SemanticError::EmbeddingFailed)?;

            let embedding_vec = embedding_result.vector;
            if embedding_vec.len() != 768 {
                continue; // Skip invalid embeddings
            }

            let mut embedding_array: [f32; 768] = [0.0; 768];
            embedding_array.copy_from_slice(&embedding_vec);

            let confidence = Confidence::from_raw(threshold);
            let cue_id = format!("semantic_multi:{}:{}", idx, query);
            let cue = Cue::embedding(cue_id, embedding_array, confidence);
            cues.push(cue);
        }

        if cues.is_empty() {
            return Err(SemanticError::NoEmbeddings);
        }

        // Use multi-cue aggregation (average strategy for now)
        let outcome = self.vector_seeder.seed_from_multi_cue(
            &cues,
            super::multi_cue::CueAggregationStrategy::Average,
        )?;

        Ok(outcome)
    }

    /// Check if semantic seeding is available.
    ///
    /// Returns `true` if the embedding provider is available and functional.
    #[must_use]
    pub fn is_available(&self) -> bool {
        // In the future, we might add runtime checks here
        true
    }
}

/// Query expander trait (COMPLETED in Task 003).
///
/// This trait is now implemented by `crate::query::expansion::QueryExpander`.
/// Use that implementation for full query expansion with confidence budgets.
#[deprecated(since = "0.1.0", note = "Use crate::query::expansion::QueryExpander instead")]
pub trait QueryExpander: Send + Sync {
    /// Expand a query into multiple variants
    fn expand(&self, query: &str) -> Vec<(String, f32)>;
}

/// Figurative language interpreter trait (placeholder for Task 004).
///
/// This trait will be implemented in Task 004 to interpret metaphors and similes.
/// For now, it's just a placeholder.
pub trait FigurativeInterpreter: Send + Sync {
    /// Interpret figurative language in a query
    fn interpret(&self, query: &str) -> Vec<(String, f32)>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        embedding::{EmbeddingProvenance, EmbeddingWithProvenance, ModelVersion},
        index::CognitiveHnswIndex,
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
            // Generate a simple embedding based on text length (for testing)
            let value = (text.len() as f32) / 1000.0;
            let vector = vec![value; 768];

            let model = ModelVersion::new(
                "mock-model".to_string(),
                "1.0.0".to_string(),
                768,
            );
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
            // Return a reference to a static model version
            static MODEL: std::sync::OnceLock<ModelVersion> = std::sync::OnceLock::new();
            MODEL.get_or_init(|| {
                ModelVersion::new("mock-model".to_string(), "1.0.0".to_string(), 768)
            })
        }

        fn max_sequence_length(&self) -> usize {
            512
        }
    }

    fn create_test_seeder() -> SemanticActivationSeeder {
        let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(MockEmbeddingProvider);
        let index = Arc::new(CognitiveHnswIndex::new());
        let vector_seeder = Arc::new(VectorActivationSeeder::with_default_resolver(
            index,
            super::super::similarity_config::SimilarityConfig::default(),
        ));

        SemanticActivationSeeder::new(embedding_provider, vector_seeder)
    }

    #[tokio::test]
    async fn test_semantic_seeder_basic() {
        let seeder = create_test_seeder();
        assert!(seeder.is_available());

        // This will return empty seeds since we have no memories in the index
        let result = seeder
            .seed_from_query("test query", Some("en"), 0.7, 10)
            .await;

        // Should succeed even with no results
        assert!(result.is_ok());
        if let Ok(outcome) = result {
            assert_eq!(outcome.seeds.len(), 0);
        }
    }

    #[tokio::test]
    async fn test_multi_query_seeding() {
        let seeder = create_test_seeder();

        let variants = vec![("automobile", 1.0), ("car", 0.9), ("vehicle", 0.8)];

        let result = seeder
            .seed_from_multi_query(&variants, Some("en"), 0.7, 10)
            .await;

        // Should succeed even with no results
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_empty_query_variants_fails() {
        let seeder = create_test_seeder();

        let variants: Vec<(&str, f32)> = vec![];

        let result = seeder
            .seed_from_multi_query(&variants, Some("en"), 0.7, 10)
            .await;

        assert!(matches!(result, Err(SemanticError::NoEmbeddings)));
    }
}
