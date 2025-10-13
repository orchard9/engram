# Task 002: Multilingual Sentence Encoder Implementation

## Objective
Implement a production-ready multilingual sentence encoder that integrates with existing sentence-transformers models, with deterministic tokenization and explicit language handling.

## Priority
P0 (blocking)

## Effort Estimate
2.5 days

## Dependencies
- Task 001: Embedding Infrastructure and Provenance Tracking

## Technical Approach

### Core Implementation Files

**Files to Create**:
- `engram-core/src/embedding/multilingual.rs` - Encoder implementation
- `engram-core/src/embedding/tokenizer.rs` - Tokenization abstraction
- `engram-cli/examples/embedding_validation.rs` - Validation tool

**Files to Modify**:
- `Cargo.toml` - Add dependencies
- `engram-core/Cargo.toml` - Feature flags

### Dependencies

Add to `engram-core/Cargo.toml` (verify against `chosen_libraries.md`):

```toml
[dependencies]
tokenizers = "0.19"  # HuggingFace tokenizers (Rust native)
ndarray = "0.15"     # N-dimensional arrays (already used)
ort = { version = "2.0", optional = true }  # ONNX Runtime (for model inference)
```

### Model Selection

- **Primary**: `paraphrase-multilingual-mpnet-base-v2` (768-dim, 50+ languages)
- Export to ONNX format using Python script (not part of runtime, tooling only)
- Store ONNX model in `engram-data/models/multilingual/`

### Core Architecture

```rust
// engram-core/src/embedding/multilingual.rs
pub struct MultilingualEncoder {
    tokenizer: Arc<Tokenizer>,
    session: Arc<ort::Session>,  // ONNX Runtime session
    model_version: ModelVersion,
    max_length: usize,
}

impl MultilingualEncoder {
    /// Load model from ONNX file (pre-exported from sentence-transformers)
    pub fn from_onnx_path(model_path: &Path, tokenizer_path: &Path) -> Result<Self, EmbeddingError>;

    /// Encode single text to embedding
    fn encode_internal(&self, text: &str, language: Option<&str>) -> Result<Vec<f32>, EmbeddingError>;

    /// Mean pooling over sequence length
    fn mean_pool(embeddings: &ndarray::ArrayView2<f32>, mask: &ndarray::Array2<i64>) -> Vec<f32>;
}

#[async_trait::async_trait]
impl EmbeddingProvider for MultilingualEncoder {
    async fn embed(&self, text: &str, language: Option<&str>) -> Result<EmbeddingWithProvenance, EmbeddingError>;
    async fn embed_batch(&self, texts: &[&str], language: Option<&str>) -> Result<Vec<EmbeddingWithProvenance>, EmbeddingError>;
    fn model_version(&self) -> &ModelVersion;
    fn max_sequence_length(&self) -> usize;
}
```

### Batch Processing Strategy

- Use rayon for parallel encoding if >10 texts
- Single encoding path: tokenize → ONNX inference → mean pooling
- Batch path: parallel tokenization → batched ONNX inference → parallel pooling

### Performance Constraints

- Single encoding: <10ms p95 (CPU), <2ms (GPU)
- Batch encoding (100 texts): <200ms p95
- Memory: <500MB model weights, <1GB inference workspace

## Acceptance Criteria

- [ ] Load ONNX model from filesystem without Python dependencies
- [ ] Generate 768-dimensional embeddings for English, Spanish, Mandarin
- [ ] Batch encoding 10x faster than sequential for >50 texts
- [ ] Reproduce sentence-transformers embeddings within ε=0.001 (validation test)
- [ ] Handle text truncation gracefully with clear error messages
- [ ] No unsafe code except ONNX FFI boundary (unavoidable)

## Testing Approach

**Unit Tests**:
- Tokenizer correctly handles Unicode (emoji, CJK, RTL scripts)
- Truncation behavior matches sentence-transformers
- Mean pooling implementation matches reference

**Integration Tests** (`engram-core/tests/multilingual_embedding_test.rs`):
- Load model from test fixtures
- Generate embeddings for multilingual test corpus
- Verify similarity scores match expected ranges (e.g., "cat" and "gato" have high similarity)

**Validation Tests** (`engram-cli/examples/embedding_validation.rs`):
- Compare against sentence-transformers Python library
- Test MTEB benchmark subset (1000 query/document pairs per language)
- Measure nDCG@10 and ensure ≥0.75 (baseline, will improve with query expansion)

**Performance Tests**:
- Benchmark single/batch encoding latency
- Profile memory allocation patterns
- Validate no memory leaks over 10,000 encodings

## Risk Mitigation

**Risk**: ONNX Runtime has platform-specific quirks
**Mitigation**: Provide pre-built binaries for Linux/macOS/Windows. Document platform-specific build flags. Fallback to dummy encoder for tests.

**Risk**: Model file is large (>400MB), slows startup
**Mitigation**: Lazy load model on first use. Provide CDN download script. Support memory-mapped model loading.

**Risk**: Embeddings drift between sentence-transformers versions
**Mitigation**: Pin exact model version in provenance. Provide migration tool to re-embed memories when upgrading models.

## Notes

This task implements the core multilingual embedding capability. The encoder must be deterministic and reproducible across platforms.

**Tooling**: A Python script will be provided in `scripts/export_onnx_model.py` to convert sentence-transformers models to ONNX format. This is build tooling only, not runtime.

**Model Storage**: The ONNX model and tokenizer files should be downloaded separately and placed in `engram-data/models/multilingual/`. Consider providing a download script or documenting the process clearly.
