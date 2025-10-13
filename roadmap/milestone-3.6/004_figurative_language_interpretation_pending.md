# Task 004: Figurative Language Interpretation

## Objective
Implement metaphor and simile interpretation via semantic analogy search, with strict hallucination prevention and explicit interpretation confidence scoring.

## Priority
P2 (important, but not blocking)

## Effort Estimate
3 days

## Dependencies
- Task 001: Embedding Infrastructure and Provenance Tracking
- Task 002: Multilingual Sentence Encoder Implementation
- Task 003: Query Expansion with Confidence Budgets

## Technical Approach

### Core Module Structure

**Files to Create**:
- `engram-core/src/query/figurative.rs` - Interpretation engine
- `engram-core/src/query/analogy.rs` - Vector analogy operations
- `engram-core/tests/figurative_language_test.rs` - Tests
- `engram-data/figurative/idiom_lexicon.json` - Known idioms

**Files to Modify**:
- `engram-core/src/query/expansion.rs` - Integrate figurative variants

### Hallucination Prevention Strategy

Figurative language poses a correctness challenge: we must **never hallucinate analogies** that aren't grounded in the actual memory graph. Our strategy:

1. **Idiom Lexicon**: Maintain curated list of known idioms with literal expansions
2. **Metaphor/Simile Analogy**: Use vector algebra for novel analogies
3. **Validation**: Only return analogies if ≥3 memories match with similarity >0.7
4. **Graceful Degradation**: If confidence <0.5, skip figurative interpretation entirely

### Core Architecture

```rust
// engram-core/src/query/figurative.rs
pub struct FigurativeInterpreter {
    idiom_lexicon: IdiomLexicon,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    analogy_engine: AnalogyEngine,
    min_confidence: f32,  // 0.5 default
}

impl FigurativeInterpreter {
    /// Interpret figurative language in query
    pub async fn interpret(&self, query: &str) -> Result<Vec<QueryVariant>, InterpretationError>;

    /// Detect "X like/as Y" or "X is Y" patterns
    fn detect_analogy_pattern(&self, query: &str) -> Option<AnalogyPattern>;

    /// Interpret analogy via vector algebra
    async fn interpret_analogy(&self, analogy: AnalogyPattern) -> Result<Vec<QueryVariant>, InterpretationError>;

    fn vector_subtract(a: &[f32], b: &[f32]) -> Vec<f32>;
    fn vector_add(a: &[f32], b: &[f32]) -> Vec<f32>;
}

#[derive(Debug, Clone)]
struct AnalogyPattern {
    target: String,
    relation: String,  // "as", "like", "is"
    source: String,
}

// Idiom lexicon (JSON-backed)
pub struct IdiomLexicon {
    idioms: HashMap<String, Vec<String>>,  // idiom -> literal expansions
}
```

### Idiom Lexicon Format

```json
// engram-data/figurative/idiom_lexicon.json
{
  "break the ice": ["initiate conversation", "start interaction", "reduce social tension"],
  "kick the bucket": ["die", "pass away"],
  "cost an arm and a leg": ["expensive", "costly"],
  "piece of cake": ["easy", "simple", "straightforward"]
}
```

### Analogy Interpretation Algorithm

1. Detect pattern: "X as/like Y" or "X is Y"
2. Compute embeddings for X and Y
3. Calculate offset: offset = Y_embedding - X_embedding
4. Apply offset to query embedding: analogy_embedding = query_embedding + offset
5. Search memory graph for nearest neighbors to analogy_embedding
6. **Validation**: Require ≥3 memory matches with similarity >0.7
7. If validation fails, return empty variants (no hallucination)

### Critical Validation Step

```rust
// Validation step (in interpret_analogy)
let memory_matches = self.memory_store.hnsw_search(&analogy_emb, k=10)?;
let high_confidence_matches = memory_matches.iter()
    .filter(|m| m.similarity > 0.7)
    .count();

if high_confidence_matches < 3 {
    tracing::warn!(
        "Analogy '{}' not supported by memory graph ({} matches < 3 threshold)",
        analogy.target, high_confidence_matches
    );
    return Ok(vec![]);  // return empty, don't hallucinate
}
```

## Acceptance Criteria

- [ ] Idiom lexicon loaded from JSON with ≥50 common idioms
- [ ] Known idioms expanded with 0.9 confidence
- [ ] Unknown idioms return empty variants (no hallucination)
- [ ] Analogy pattern detection matches "X as Y" and "X is Y" structures
- [ ] Vector analogy computation mathematically correct (unit tested)
- [ ] Analogy interpretations require ≥3 memory matches with >0.7 similarity
- [ ] Interpretation time <5ms p95 (excluding memory search)

## Testing Approach

**Unit Tests**:
- Idiom lexicon lookup correctness
- Analogy pattern regex matching
- Vector arithmetic (subtract, add, normalize)

**Integration Tests** (`engram-core/tests/figurative_language_test.rs`):
- Known idioms: "break the ice" -> ["initiate conversation"]
- Unknown idioms: "flibbertigibbet" -> [] (no hallucination)
- Simple analogies: "fast as cheetah" with seeded memories

**Human Evaluation**:
- Curate 100 metaphor/simile queries
- Manual labeling of acceptable interpretations
- Measure % acceptable ≥80% (milestone success criterion)

## Risk Mitigation

**Risk**: Analogy vector algebra produces nonsensical embeddings
**Mitigation**: Normalize all vectors to unit length. Validate analogy embeddings have reasonable similarity to known concept embeddings.

**Risk**: Idiom lexicon becomes stale or culturally biased
**Mitigation**: Version idiom lexicon, allow user-provided extensions. Document cultural assumptions clearly.

**Risk**: Complex figurative language (nested metaphors, irony) fails
**Mitigation**: Explicitly document unsupported patterns. Return empty variants with warning rather than incorrect interpretations.

## Notes

This task implements figurative language interpretation with strict hallucination prevention. The critical correctness invariant: **never return an interpretation unless it's grounded in actual memories**.

**Supported Patterns**:
- Simple similes: "X as Y", "X like Y"
- Simple metaphors: "X is Y"
- Known idioms from curated lexicon

**Unsupported Patterns** (documented limitations):
- Nested metaphors: "the world is a stage, and we are actors"
- Irony and sarcasm: "oh great, another meeting"
- Cultural-specific expressions without explicit lexicon entries

**Future Work**: Context-aware disambiguation, cultural adaptation, online learning from user feedback.
