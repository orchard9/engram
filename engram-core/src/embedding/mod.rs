//! Embedding infrastructure with provenance tracking for multilingual semantic recall.
//!
//! This module provides the foundational infrastructure for generating and tracking
//! embeddings with strict provenance metadata. All embeddings include model version,
//! language, timestamp, and quality information for auditability and reproducibility.
//!
//! ## Key Design Principles
//!
//! 1. **Optional Metadata**: Embeddings are optional metadata on memories, never required
//! 2. **Provenance Tracking**: Every embedding tracks its origin for auditability
//! 3. **Graceful Degradation**: Missing embeddings trigger warnings but never block operations
//! 4. **Zero Unsafe Code**: All trait definitions use safe Rust
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use engram_core::embedding::{EmbeddingProvider, EmbeddingProvenance};
//!
//! async fn generate_embedding(provider: &impl EmbeddingProvider, text: &str) {
//!     match provider.embed(text, Some("en")).await {
//!         Ok(embedding) => {
//!             println!("Generated embedding with model: {}", embedding.provenance.model.name);
//!         }
//!         Err(e) => {
//!             eprintln!("Embedding generation failed: {}, continuing without embedding", e);
//!         }
//!     }
//! }
//! ```

pub mod provenance;
pub mod provider;
pub mod tokenizer;

#[cfg(feature = "multilingual_embeddings")]
pub mod multilingual;

pub use provenance::{EmbeddingProvenance, EmbeddingWithProvenance, ModelVersion};
pub use provider::{EmbeddingError, EmbeddingProvider};
pub use tokenizer::{SentenceTokenizer, TokenizationResult};
