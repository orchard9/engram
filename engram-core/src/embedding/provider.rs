//! Embedding provider trait for abstracting over different embedding models.
//!
//! This module defines the `EmbeddingProvider` trait that all embedding implementations
//! must satisfy. The trait is designed to be:
//!
//! - **Async-first**: All operations return futures to avoid blocking
//! - **Error-explicit**: Clear error types with actionable messages
//! - **Batch-friendly**: Support for efficient batch processing
//! - **Model-agnostic**: Works with any embedding model that produces vectors
//!
//! ## Error Handling Philosophy
//!
//! Embedding errors should never block memory operations. Callers should:
//! 1. Log the error with structured metadata
//! 2. Continue the operation without the embedding
//! 3. Emit metrics for monitoring embedding coverage
//!
//! This aligns with our principle: embeddings are optional metadata, never required.

use super::provenance::{EmbeddingWithProvenance, ModelVersion};
use thiserror::Error;

/// Errors that can occur during embedding generation.
///
/// All error variants include actionable information to help diagnose and fix issues.
/// Error messages follow cognitive ergonomics principles: they explain what went wrong
/// and suggest how to fix it.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// Text exceeds the maximum sequence length supported by the model.
    ///
    /// **How to fix**: Truncate the text or use a model with longer context window.
    #[error("text exceeds maximum sequence length: got {0} tokens, max {1}")]
    TextTooLong(usize, usize),

    /// Language is not supported by this embedding model.
    ///
    /// **How to fix**: Use a multilingual model or pre-process text to a supported language.
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// Model failed to initialize (file not found, invalid format, etc.).
    ///
    /// **How to fix**: Check that model files exist and are in the expected format.
    #[error("model initialization failed: {0}")]
    InitializationFailed(String),

    /// Encoding operation failed (tokenization, inference, or post-processing error).
    ///
    /// **How to fix**: Check model logs for detailed error. Text may contain invalid UTF-8.
    #[error("encoding failed: {0}")]
    EncodingFailed(String),

    /// Model file format is invalid or corrupted.
    ///
    /// **How to fix**: Re-download the model or verify file integrity.
    #[error("invalid model format: {0}")]
    InvalidModelFormat(String),

    /// Operation timed out.
    ///
    /// **How to fix**: Increase timeout or check if model inference is hanging.
    #[error("operation timed out after {0:?}")]
    Timeout(std::time::Duration),
}

/// Trait for generating embeddings with provenance tracking.
///
/// Implementations must be `Send + Sync` to support concurrent embedding generation.
/// All methods are async to avoid blocking the runtime.
///
/// ## Implementation Notes
///
/// - **Model Loading**: Expensive initialization should happen in constructor, not per-request
/// - **Batch Processing**: Implement `embed_batch` efficiently to amortize inference costs
/// - **Error Handling**: Use specific error variants to help debugging
/// - **Provenance**: Always include complete provenance metadata
///
/// ## Example Implementation
///
/// ```rust,ignore
/// use engram_core::embedding::*;
/// use async_trait::async_trait;
///
/// struct MyEmbeddingModel {
///     model_version: ModelVersion,
///     // ... model state
/// }
///
/// #[async_trait]
/// impl EmbeddingProvider for MyEmbeddingModel {
///     async fn embed(&self, text: &str, language: Option<&str>) -> Result<EmbeddingWithProvenance, EmbeddingError> {
///         // 1. Tokenize text
///         // 2. Run inference
///         // 3. Create provenance
///         // 4. Return EmbeddingWithProvenance
///         todo!()
///     }
///
///     // ... other methods
/// }
/// ```
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding for the given text with provenance tracking.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    /// * `language` - Optional ISO 639-1 language code (e.g., "en", "es", "zh")
    ///
    /// # Returns
    ///
    /// An embedding with complete provenance metadata, or an error if generation fails.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Text is too long (`TextTooLong`)
    /// - Language is unsupported (`UnsupportedLanguage`)
    /// - Encoding fails for any reason (`EncodingFailed`)
    ///
    /// # Performance
    ///
    /// Single-text embedding should complete in <10ms p95 on CPU, <2ms on GPU.
    async fn embed(
        &self,
        text: &str,
        language: Option<&str>,
    ) -> Result<EmbeddingWithProvenance, EmbeddingError>;

    /// Generate embeddings for multiple texts in batch for efficiency.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of texts to embed
    /// * `language` - Optional ISO 639-1 language code applied to all texts
    ///
    /// # Returns
    ///
    /// Vector of embeddings with provenance, one per input text.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any text fails to encode. In this case, no embeddings are returned.
    /// Implementations may choose to skip failed texts and return partial results in the future.
    ///
    /// # Performance
    ///
    /// Batch processing should be significantly faster than sequential embedding.
    /// Target: 10x speedup for batches >50 texts.
    async fn embed_batch(
        &self,
        texts: &[&str],
        language: Option<&str>,
    ) -> Result<Vec<EmbeddingWithProvenance>, EmbeddingError>;

    /// Get the model version information.
    ///
    /// This is used for compatibility checking and provenance tracking.
    fn model_version(&self) -> &ModelVersion;

    /// Get the maximum sequence length supported by this model.
    ///
    /// Texts longer than this will trigger `TextTooLong` errors.
    fn max_sequence_length(&self) -> usize;

    /// Check if a specific language is supported by this model.
    ///
    /// Default implementation returns `true` (assume multilingual).
    /// Implementations should override if they have language restrictions.
    fn supports_language(&self, _language: &str) -> bool {
        true // Default: assume multilingual
    }

    /// Get model-specific configuration or capabilities.
    ///
    /// Returns a map of configuration keys to values for debugging and monitoring.
    /// Default implementation returns empty map.
    fn capabilities(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::provenance::{EmbeddingProvenance, ModelVersion};

    // Mock implementation for testing
    struct MockEmbeddingProvider {
        model_version: ModelVersion,
        fail_on_long_text: bool,
    }

    impl MockEmbeddingProvider {
        fn new() -> Self {
            Self {
                model_version: ModelVersion::new("mock-model".to_string(), "1.0.0".to_string(), 4),
                fail_on_long_text: false,
            }
        }

        fn with_length_limit(mut self) -> Self {
            self.fail_on_long_text = true;
            self
        }
    }

    #[async_trait::async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        async fn embed(
            &self,
            text: &str,
            language: Option<&str>,
        ) -> Result<EmbeddingWithProvenance, EmbeddingError> {
            if self.fail_on_long_text && text.len() > 100 {
                return Err(EmbeddingError::TextTooLong(text.len(), 100));
            }

            let vector = vec![0.1, 0.2, 0.3, 0.4];
            let provenance =
                EmbeddingProvenance::new(self.model_version.clone(), language.map(String::from));

            Ok(EmbeddingWithProvenance::new(vector, provenance))
        }

        async fn embed_batch(
            &self,
            texts: &[&str],
            language: Option<&str>,
        ) -> Result<Vec<EmbeddingWithProvenance>, EmbeddingError> {
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                results.push(self.embed(text, language).await?);
            }
            Ok(results)
        }

        fn model_version(&self) -> &ModelVersion {
            &self.model_version
        }

        fn max_sequence_length(&self) -> usize {
            if self.fail_on_long_text { 100 } else { 512 }
        }
    }

    #[tokio::test]
    async fn test_mock_provider_basic_embedding() {
        let provider = MockEmbeddingProvider::new();
        let result = provider.embed("test text", Some("en")).await;

        assert!(result.is_ok());
        if let Ok(embedding) = result {
            assert_eq!(embedding.vector.len(), 4);
            assert_eq!(embedding.language(), Some("en"));
            assert_eq!(embedding.model().name, "mock-model");
        }
    }

    #[tokio::test]
    async fn test_mock_provider_text_too_long() {
        let provider = MockEmbeddingProvider::new().with_length_limit();
        let long_text = "a".repeat(200);
        let result = provider.embed(&long_text, None).await;

        assert!(result.is_err());
        if let Err(EmbeddingError::TextTooLong(got, max)) = result {
            assert_eq!(got, 200);
            assert_eq!(max, 100);
        }
    }

    #[tokio::test]
    async fn test_mock_provider_batch_embedding() {
        let provider = MockEmbeddingProvider::new();
        let texts = vec!["text1", "text2", "text3"];
        let result = provider.embed_batch(&texts, Some("es")).await;

        assert!(result.is_ok());
        if let Ok(embeddings) = result {
            assert_eq!(embeddings.len(), 3);
            for embedding in embeddings {
                assert_eq!(embedding.language(), Some("es"));
            }
        }
    }

    #[tokio::test]
    async fn test_mock_provider_default_language_support() {
        let provider = MockEmbeddingProvider::new();

        // Default implementation should support all languages
        assert!(provider.supports_language("en"));
        assert!(provider.supports_language("es"));
        assert!(provider.supports_language("unknown"));
    }

    #[tokio::test]
    async fn test_mock_provider_capabilities() {
        let provider = MockEmbeddingProvider::new();
        let capabilities = provider.capabilities();

        // Default implementation returns empty map
        assert!(capabilities.is_empty());
    }

    #[test]
    fn test_embedding_error_display() {
        let err1 = EmbeddingError::TextTooLong(1000, 512);
        assert_eq!(
            err1.to_string(),
            "text exceeds maximum sequence length: got 1000 tokens, max 512"
        );

        let err2 = EmbeddingError::UnsupportedLanguage("xyz".to_string());
        assert_eq!(err2.to_string(), "unsupported language: xyz");

        let err3 = EmbeddingError::InitializationFailed("model not found".to_string());
        assert_eq!(
            err3.to_string(),
            "model initialization failed: model not found"
        );

        let err4 = EmbeddingError::EncodingFailed("tokenization error".to_string());
        assert_eq!(err4.to_string(), "encoding failed: tokenization error");
    }
}
