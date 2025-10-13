# Task 005: Semantic Search Integration with Spreading Activation

## Objective
Integrate vector similarity search as an activation seeding mechanism for spreading activation, ensuring semantic and graph-based recall operate cohesively.

## Priority
P0 (blocking)

## Effort Estimate
2 days

## Dependencies
- Task 001: Embedding Infrastructure and Provenance Tracking
- Task 002: Multilingual Sentence Encoder Implementation

## Technical Approach

### Core Module Structure

**Files to Create**:
- `engram-core/src/activation/semantic_seeder.rs` - Semantic activation seeding

**Files to Modify**:
- `engram-core/src/activation/recall.rs` - Integrate semantic seeding
- `engram-core/src/activation/seeding.rs` - Add semantic seeding trait impl
- `engram-core/src/store.rs` - Add multi-vector search

### Architecture Insight

The existing `CognitiveRecall` system uses `ActivationSeeder` to initialize spreading activation. We extend this with `SemanticActivationSeeder`:

**Key Insight**: Semantic recall is not a replacement for graph-based spreading activation, but a complementary seeding mechanism.

### Core Data Structures

```rust
// engram-core/src/activation/semantic_seeder.rs
pub struct SemanticActivationSeeder {
    embedding_provider: Arc<dyn EmbeddingProvider>,
    query_expander: Arc<QueryExpander>,
    figurative_interpreter: Arc<FigurativeInterpreter>,
}

impl SemanticActivationSeeder {
    /// Seed activation from semantic query
    pub async fn seed_from_query(
        &self,
        query: &str,
        budget: ConfidenceBudget,
        store: &MemoryStore,
    ) -> Result<Vec<ActivationSeed>, SemanticError>;
}

#[derive(Debug, Clone)]
pub struct ActivationSeed {
    pub episode_id: EpisodeId,
    pub initial_activation: f32,
    pub source: ActivationSource,
}

#[derive(Debug, Clone)]
pub enum ActivationSource {
    ExplicitCue { cue_id: CueId },
    SemanticSimilarity { query: String, similarity: f32 },
    SpreadingActivation { from_episode: EpisodeId },
}
```

### Multi-Vector Search

Extend `MemoryStore` to support multi-vector queries with confidence weighting:

```rust
// engram-core/src/store.rs (modify existing)
impl MemoryStore {
    /// Search HNSW index with multiple query vectors
    pub fn multi_vector_search(
        &self,
        queries: &[(Vec<f32>, f32)],  // (embedding, confidence)
        k: usize,
    ) -> Result<Vec<SearchResult>, StorageError>;
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub episode_id: EpisodeId,
    pub similarity: f32,
    pub query_confidence: f32,
}
```

### Aggregation Strategy

When multiple query variants match the same episode:
- Use **maximum similarity** (not sum) to avoid double-counting
- Weight by query confidence: `score = similarity * query_confidence`
- Sort by weighted score and return top-k

### Integration with CognitiveRecall

```rust
// engram-core/src/activation/recall.rs (modify existing)
impl CognitiveRecall {
    /// Recall memories via semantic query
    pub async fn recall_semantic(
        &self,
        query: &str,
        budget: ConfidenceBudget,
    ) -> Result<RecallResult, RecallError> {
        // Step 1: Semantic seeding
        let seeds = self.semantic_seeder.seed_from_query(query, budget.clone(), &self.store).await?;

        if seeds.is_empty() {
            tracing::warn!("Semantic seeding failed, trying lexical fallback");
            return self.recall_lexical(query, budget);  // existing fallback
        }

        // Step 2: Standard spreading activation from semantic seeds
        let activation_map = self.spread_activation(&seeds, budget)?;

        // Step 3: Rank and return
        let ranked_episodes = self.rank_by_activation(&activation_map)?;

        Ok(RecallResult {
            episodes: ranked_episodes,
            activation_metadata: ActivationMetadata {
                seed_source: "semantic",
                num_seeds: seeds.len(),
                spreading_rounds: self.config.max_spreading_rounds,
            },
        })
    }

    /// Lexical fallback (existing implementation)
    fn recall_lexical(&self, query: &str, budget: ConfidenceBudget) -> Result<RecallResult, RecallError>;
}
```

### Semantic Seeding Algorithm

1. Expand query (synonyms, abbreviations, figurative language from Tasks 003/004)
2. Collect embeddings for all variants
3. If no embeddings available, return empty seeds (triggers lexical fallback)
4. Multi-vector HNSW search (k=20 candidates)
5. Convert to ActivationSeeds with confidence weighting
6. Return seeds for spreading activation

## Acceptance Criteria

- [ ] `SemanticActivationSeeder` generates seeds from query embeddings
- [ ] Multi-vector search aggregates results across expanded queries
- [ ] Semantic seeds integrate seamlessly with existing spreading activation
- [ ] Graceful fallback to lexical search when embeddings unavailable
- [ ] Activation source tracked in metadata (semantic vs lexical vs explicit)
- [ ] End-to-end recall latency <50ms p95 (including expansion + search + spreading)

## Testing Approach

**Integration Tests** (`engram-core/tests/semantic_recall_integration_test.rs`):
- Store 100 memories with embeddings
- Query: "automobile" (semantic) -> should recall "car" memories
- Query: "ML applications" (abbreviation expansion) -> recall machine learning memories
- Verify activation spreads from semantic seeds to connected memories

**Regression Tests**:
- Compare semantic recall against lexical recall for same queries
- Measure nDCG@10 improvement (expect ≥10% improvement)
- Ensure existing graph traversal tests still pass

**Performance Tests**:
- Benchmark multi-vector search latency (target <10ms for 10 vectors)
- Profile memory allocation patterns
- Validate spreading activation performance unchanged

## Risk Mitigation

**Risk**: Semantic seeds dominate spreading activation, drowning out explicit cues
**Mitigation**: Weight semantic seeds lower (0.5x) than explicit cues. Allow configuration of seed weighting strategy.

**Risk**: Multi-vector search is too slow for real-time queries
**Mitigation**: Batch HNSW searches if backend supports it. Cache recent query embeddings with TTL.

**Risk**: Query expansion creates too many vectors, exceeding HNSW performance budget
**Mitigation**: Cap expansion to 10 variants (Task 003). Implement early termination when top-k results stabilize.

## Notes

This task bridges semantic vector search with graph-based spreading activation. The integration must be seamless and maintain performance characteristics of both systems.

**Key Design Decisions**:
- Semantic search seeds activation, doesn't replace spreading
- Graceful degradation to lexical search when embeddings absent
- Activation source tracking for explainability
- Configurable seed weighting strategies

**Integration Points**:
- `CognitiveRecall::recall_semantic()` - new entry point
- `MemoryStore::multi_vector_search()` - HNSW extension
- `ActivationSeeder` trait - new semantic implementation
- Confidence propagation system - unchanged, operates on seeds
