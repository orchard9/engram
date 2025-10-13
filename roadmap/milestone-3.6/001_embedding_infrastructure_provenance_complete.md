# Task 001: Embedding Infrastructure and Provenance Tracking

## Objective
Establish the foundational embedding infrastructure with strict provenance tracking, ensuring every memory's semantic representation is auditable and versioned.

## Priority
P0 (blocking)

## Effort Estimate
2 days

## Dependencies
None

## Technical Approach

### Core Module Structure

**Files to Create**:
- `engram-core/src/embedding/mod.rs` - Module root
- `engram-core/src/embedding/provider.rs` - Trait definitions
- `engram-core/src/embedding/provenance.rs` - Metadata tracking
- `engram-core/src/embedding/encoder.rs` - Concrete implementations

**Files to Modify**:
- `engram-core/src/episode.rs` - Add `EmbeddingMetadata` field
- `engram-core/src/cue.rs` - Add embedding support
- `engram-core/src/lib.rs` - Export new module

### Core Data Structures

```rust
// engram-core/src/embedding/provenance.rs
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelVersion {
    name: String,           // e.g., "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    version: String,        // semantic version or git hash
    dimension: usize,       // embedding vector size
}

#[derive(Debug, Clone)]
pub struct EmbeddingProvenance {
    model: ModelVersion,
    language: Option<String>,           // ISO 639-1 code (en, es, zh)
    created_at: std::time::SystemTime,
    quality_score: Option<f32>,         // model's self-reported confidence
    metadata: HashMap<String, String>,  // extensible k-v pairs
}

#[derive(Debug, Clone)]
pub struct EmbeddingWithProvenance {
    vector: Vec<f32>,
    provenance: EmbeddingProvenance,
}

// engram-core/src/embedding/provider.rs
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding for text with provenance
    async fn embed(&self, text: &str, language: Option<&str>) -> Result<EmbeddingWithProvenance, EmbeddingError>;

    /// Batch embedding for efficiency
    async fn embed_batch(&self, texts: &[&str], language: Option<&str>) -> Result<Vec<EmbeddingWithProvenance>, EmbeddingError>;

    /// Provider metadata
    fn model_version(&self) -> &ModelVersion;

    /// Maximum sequence length supported
    fn max_sequence_length(&self) -> usize;
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("text exceeds maximum sequence length: {0} > {1}")]
    TextTooLong(usize, usize),

    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),

    #[error("model initialization failed: {0}")]
    InitializationFailed(String),

    #[error("encoding failed: {0}")]
    EncodingFailed(String),
}
```

### Integration Strategy

1. Modify `Episode` to store optional `EmbeddingWithProvenance`
2. Add builder method `Episode::with_embedding()`
3. Implement fallback logic: if embedding missing, log warning but continue operation
4. Add metrics for embedding coverage: `engram_episodes_with_embeddings_total{model_version}`

### Error Handling

- Never fail memory storage if embedding generation fails
- Emit structured warnings to monitoring system
- Maintain separate audit log for embedding errors

## Acceptance Criteria

- [ ] `EmbeddingProvider` trait compiles with zero unsafe code
- [ ] Provenance includes model version, language, timestamp, quality score
- [ ] `Episode` supports optional embeddings without breaking existing tests
- [ ] Embedding errors are logged but don't propagate to memory operations
- [ ] Metrics exported via existing Prometheus integration

## Testing Approach

**Unit Tests** (`engram-core/src/embedding/tests.rs`):
- Provenance serialization/deserialization (serde)
- Model version comparison and equality
- Error type construction and formatting

**Integration Tests** (`engram-core/tests/embedding_integration_test.rs`):
- Store episodes with and without embeddings
- Verify fallback behavior when embeddings absent
- Check metrics emission for embedding coverage

**Property Tests** (using `proptest`):
- Embedding vectors always match declared dimension
- Provenance timestamps monotonically increase
- Quality scores always in [0.0, 1.0] range

## Risk Mitigation

**Risk**: Embedding generation is slow and blocks memory operations
**Mitigation**: All embedding operations are async, with timeouts and circuit breakers. Store operations never wait on embeddings.

**Risk**: Model version mismatches cause index corruption
**Mitigation**: HNSW index includes model version in metadata. Queries against wrong version trigger warnings and fallback to lexical search.

**Risk**: Large embedding vectors consume excessive memory
**Mitigation**: Implement lazy loading for embeddings, storing only provenance in hot path. Load vectors on-demand from storage tier.

## Notes

This task establishes the foundational embedding infrastructure for Milestone 3.6. All subsequent tasks depend on this implementation.

**Key Principles**:
- Embeddings are optional metadata, never required for operation
- Provenance tracking ensures auditability and reproducibility
- Graceful degradation when embeddings unavailable
- Zero unsafe code in trait definitions
