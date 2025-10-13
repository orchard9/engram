//! Tokenization abstraction for multilingual text processing.
//!
//! This module provides a safe abstraction over the HuggingFace `tokenizers` library,
//! with specific support for sentence-transformers compatible models.
//!
//! ## Design Principles
//!
//! - **Unicode Correctness**: Proper handling of emoji, CJK characters, and RTL scripts
//! - **Truncation Strategy**: Match sentence-transformers behavior exactly
//! - **Error Handling**: Clear error messages for tokenization failures
//! - **Performance**: Minimize allocations for single-text tokenization

#[cfg(feature = "multilingual_embeddings")]
use tokenizers::{EncodeInput, Tokenizer as HfTokenizer};

use super::provider::EmbeddingError;
use std::path::Path;

/// Tokenization result containing token IDs and attention mask.
///
/// The attention mask indicates which tokens are actual content (1) vs padding (0).
#[derive(Debug, Clone)]
pub struct TokenizationResult {
    /// Token IDs for model input
    pub token_ids: Vec<u32>,

    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<u32>,

    /// Original text length in Unicode scalar values
    pub original_length: usize,

    /// Whether the text was truncated
    pub truncated: bool,
}

/// Tokenizer abstraction for multilingual text processing.
///
/// This wraps the HuggingFace tokenizer with a simpler API that matches our needs.
pub struct SentenceTokenizer {
    #[cfg(feature = "multilingual_embeddings")]
    inner: HfTokenizer,

    max_length: usize,
}

impl SentenceTokenizer {
    /// Load a tokenizer from a JSON file (HuggingFace format).
    ///
    /// # Arguments
    ///
    /// * `tokenizer_path` - Path to the tokenizer.json file
    /// * `max_length` - Maximum sequence length (typically 512 for sentence-transformers)
    ///
    /// # Errors
    ///
    /// Returns `InitializationFailed` if the tokenizer file cannot be loaded or parsed.
    pub fn from_file(tokenizer_path: &Path, max_length: usize) -> Result<Self, EmbeddingError> {
        #[cfg(feature = "multilingual_embeddings")]
        {
            let inner = HfTokenizer::from_file(tokenizer_path).map_err(|e| {
                EmbeddingError::InitializationFailed(format!(
                    "failed to load tokenizer from {:?}: {}",
                    tokenizer_path, e
                ))
            })?;

            Ok(Self { inner, max_length })
        }

        #[cfg(not(feature = "multilingual_embeddings"))]
        {
            let _ = (tokenizer_path, max_length);
            Err(EmbeddingError::InitializationFailed(
                "multilingual_embeddings feature not enabled".to_string(),
            ))
        }
    }

    /// Tokenize a single text string.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to tokenize
    ///
    /// # Returns
    ///
    /// A `TokenizationResult` containing token IDs, attention mask, and metadata.
    ///
    /// # Errors
    ///
    /// Returns `EncodingFailed` if tokenization fails (e.g., invalid UTF-8).
    pub fn tokenize(&self, text: &str) -> Result<TokenizationResult, EmbeddingError> {
        #[cfg(feature = "multilingual_embeddings")]
        {
            let original_length = text.chars().count();

            // Encode with truncation enabled
            let encoding = self
                .inner
                .encode(text, true)
                .map_err(|e| EmbeddingError::EncodingFailed(format!("tokenization failed: {}", e)))?;

            let token_ids = encoding.get_ids().to_vec();
            let attention_mask = encoding.get_attention_mask().to_vec();
            let truncated = token_ids.len() >= self.max_length;

            Ok(TokenizationResult {
                token_ids,
                attention_mask,
                original_length,
                truncated,
            })
        }

        #[cfg(not(feature = "multilingual_embeddings"))]
        {
            let _ = text;
            Err(EmbeddingError::EncodingFailed(
                "multilingual_embeddings feature not enabled".to_string(),
            ))
        }
    }

    /// Tokenize multiple texts in batch.
    ///
    /// This is more efficient than calling `tokenize` repeatedly.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of texts to tokenize
    ///
    /// # Returns
    ///
    /// Vector of tokenization results, one per input text.
    ///
    /// # Errors
    ///
    /// Returns `EncodingFailed` if any tokenization fails.
    pub fn tokenize_batch(&self, texts: &[&str]) -> Result<Vec<TokenizationResult>, EmbeddingError> {
        #[cfg(feature = "multilingual_embeddings")]
        {
            let original_lengths: Vec<usize> = texts.iter().map(|t| t.chars().count()).collect();

            // Convert to EncodeInput
            let inputs: Vec<EncodeInput> = texts.iter().map(|&t| t.into()).collect();

            // Batch encode
            let encodings = self.inner.encode_batch(inputs, true).map_err(|e| {
                EmbeddingError::EncodingFailed(format!("batch tokenization failed: {}", e))
            })?;

            // Convert encodings to results
            let results: Vec<TokenizationResult> = encodings
                .into_iter()
                .zip(original_lengths.iter())
                .map(|(encoding, &original_length)| {
                    let token_ids = encoding.get_ids().to_vec();
                    let attention_mask = encoding.get_attention_mask().to_vec();
                    let truncated = token_ids.len() >= self.max_length;

                    TokenizationResult {
                        token_ids,
                        attention_mask,
                        original_length,
                        truncated,
                    }
                })
                .collect();

            Ok(results)
        }

        #[cfg(not(feature = "multilingual_embeddings"))]
        {
            let _ = texts;
            Err(EmbeddingError::EncodingFailed(
                "multilingual_embeddings feature not enabled".to_string(),
            ))
        }
    }

    /// Get the maximum sequence length supported by this tokenizer.
    #[must_use]
    pub const fn max_length(&self) -> usize {
        self.max_length
    }
}

#[cfg(test)]
#[cfg(feature = "multilingual_embeddings")]
mod tests {
    use super::*;

    // Note: These tests require a tokenizer file to be present.
    // In a real implementation, we'd include a test fixture.

    #[test]
    fn test_tokenization_result_structure() {
        // Test that TokenizationResult has the correct structure
        let result = TokenizationResult {
            token_ids: vec![101, 2023, 2003, 1037, 3231, 102],
            attention_mask: vec![1, 1, 1, 1, 1, 1],
            original_length: 12,
            truncated: false,
        };

        assert_eq!(result.token_ids.len(), 6);
        assert_eq!(result.attention_mask.len(), 6);
        assert_eq!(result.original_length, 12);
        assert!(!result.truncated);
    }

    #[test]
    fn test_unicode_length_counting() {
        // Verify we count Unicode scalar values correctly
        let text = "Hello 世界 🌍";
        let char_count = text.chars().count();

        assert_eq!(char_count, 10); // "Hello " (6) + "世界 " (3) + "🌍" (1)
    }
}
