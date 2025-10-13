# Task 003: Query Expansion with Confidence Budgets

## Objective
Implement query expansion for synonyms, abbreviations, and lexical variations with strict confidence budget enforcement, ensuring expanded queries never exceed latency budgets.

## Priority
P1 (critical)

## Effort Estimate
2 days

## Dependencies
- Task 001: Embedding Infrastructure and Provenance Tracking
- Task 002: Multilingual Sentence Encoder Implementation

## Technical Approach

### Core Module Structure

**Files to Create**:
- `engram-core/src/query/expansion.rs` - Expansion engine
- `engram-core/src/query/lexicon.rs` - Lexicon abstraction
- `engram-core/tests/query_expansion_test.rs` - Integration tests
- `engram-data/lexicons/` - Domain lexicons

**Files to Modify**:
- `engram-core/src/query/mod.rs` - Add expansion module
- `engram-core/src/cue.rs` - Add expansion metadata
- `engram-core/src/confidence.rs` - Add ConfidenceBudget

### Dependencies

Add to `engram-core/Cargo.toml`:

```toml
[dependencies]
unicode-segmentation = "1.11"  # Word boundary detection
fst = "0.4"  # Finite state transducers for fast lexicon lookup
```

### Core Data Structures

```rust
// engram-core/src/query/expansion.rs
#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    original: String,
    variants: Vec<QueryVariant>,
    total_confidence: f32,
    expansion_metadata: ExpansionMetadata,
}

#[derive(Debug, Clone)]
pub struct QueryVariant {
    text: String,
    variant_type: VariantType,
    confidence: f32,           // how likely this variant matches user intent
    embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariantType {
    Original,           // query as provided
    Synonym,            // lexical synonym (e.g., "car" -> "automobile")
    Abbreviation,       // expansion (e.g., "ML" -> "machine learning")
    Morphological,      // stemmed/lemmatized form
}

#[derive(Debug, Clone)]
pub struct ExpansionMetadata {
    expansion_time_us: u64,
    lexicons_consulted: Vec<String>,
    truncated: bool,            // true if budget exhausted before full expansion
}

pub struct QueryExpander {
    lexicons: Vec<Arc<dyn Lexicon>>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    max_variants: usize,        // hard limit on expansion size (default: 10)
    confidence_threshold: f32,  // min confidence to include variant (default: 0.3)
}
```

### Lexicon Implementations

**1. SynonymLexicon**: Use WordNet synsets (English-only, expand later)
- Pre-build FST from WordNet data
- Store in `engram-data/lexicons/wordnet_en.fst`
- Confidence: 0.8 for direct synonyms, 0.5 for hypernyms

**2. AbbreviationLexicon**: Domain-specific abbreviations
- Medical: "MI" -> "myocardial infarction" (0.9)
- Technical: "ML" -> "machine learning" (0.85), "maximum likelihood" (0.4)
- Ambiguity handling: return multiple variants with confidence scores

### Confidence Budget Integration

```rust
// engram-core/src/confidence.rs (modify existing)
#[derive(Debug, Clone)]
pub struct ConfidenceBudget {
    initial: f32,
    consumed: Arc<AtomicU32>,  // track usage across async operations
}

impl ConfidenceBudget {
    pub fn new(initial: f32) -> Self;
    pub fn consume(&self, amount: f32) -> bool;
    pub fn remaining(&self) -> f32;
}
```

### Expansion Algorithm

1. Start with original query as highest confidence variant (1.0)
2. Consult lexicons in priority order (synonyms → abbreviations → morphological)
3. Sort variants by confidence descending
4. Truncate to `max_variants` limit
5. Compute embeddings up to remaining confidence budget
6. Return `ExpandedQuery` with metadata about truncation

## Acceptance Criteria

- [ ] Expand "car" to ["automobile", "vehicle"] with appropriate confidences
- [ ] Handle abbreviations: "ML" -> ["machine learning", "maximum likelihood"]
- [ ] Respect max_variants limit (never return >10 variants)
- [ ] Expansion completes in <1ms p95 (lexicon lookup only, excluding embedding)
- [ ] Budget exhaustion logged but doesn't fail query
- [ ] Metadata includes lexicons consulted and truncation flag

## Testing Approach

**Unit Tests**:
- Lexicon lookup correctness (known synonym pairs)
- Confidence budget enforcement (consume until exhausted)
- Variant deduplication (same text from multiple lexicons)

**Integration Tests** (`engram-core/tests/query_expansion_test.rs`):
- Full expansion pipeline with multiple lexicons
- Verify embeddings computed for top-N variants
- Check expansion metadata accuracy

**Regression Tests**:
- Suite of 100 known queries with gold-standard expansions
- Measure recall: % of expected variants in top-10
- Target: ≥95% recall on synonym suite

## Risk Mitigation

**Risk**: Synonym explosion for common words (e.g., "get" has 20+ synonyms)
**Mitigation**: Rank by corpus frequency, prefer domain-specific meanings. Cap variants at 10 with clear truncation signal.

**Risk**: Abbreviation ambiguity causes irrelevant expansions
**Mitigation**: Use context-aware disambiguation (future task). For now, return multiple variants with calibrated confidence scores.

**Risk**: Lexicon updates require redeployment
**Mitigation**: Load lexicons from filesystem at startup. Provide hot-reload mechanism via configuration file watcher.

## Notes

This task implements query expansion with strict confidence budget enforcement. The expansion system must be fast (<1ms) and respect latency budgets.

**Lexicon Format**: All lexicons use FST (finite state transducers) for O(|query|) lookup time. Build scripts provided in `scripts/build_lexicons.rs`.

**Configuration**: Max variants and confidence threshold should be configurable via CLI config system.
