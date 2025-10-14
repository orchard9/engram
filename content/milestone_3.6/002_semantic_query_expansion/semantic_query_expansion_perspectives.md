# Semantic Query Expansion: Multiple Architectural Perspectives

## 1. Cognitive-Architecture Perspective

### The Mental Model: Spreading Activation and Semantic Priming

Query expansion mirrors how human memory retrieval works through spreading activation in semantic networks. When you think of "dog," your mind automatically primes related concepts like "pet," "bark," "puppy" without conscious effort. This is spreading activation.

**Core Insight**: Semantic query expansion is computational spreading activation, where:
- The initial query activates nodes in the semantic embedding space
- Activation spreads to nearby concepts (synonyms, related terms)
- Activation strength decays with semantic distance (confidence weighting)
- The system retrieves memories that exceed an activation threshold

**Cognitive Parallels**:

1. **Semantic Priming**: Seeing "doctor" makes "nurse" easier to recall (faster response times in psychology experiments). Query expansion implements this by retrieving "nurse" even when only "doctor" was queried.

2. **Fan Effect**: The more associations a concept has, the slower retrieval becomes. This maps to query explosion - unlimited expansion degrades performance and precision.

3. **Context-Dependent Retrieval**: Memory recall improves when retrieval context matches encoding context. Multilingual embeddings preserve this by mapping cross-lingual concepts to the same contextual region.

**Implementation Implications**:

- Use confidence scores as activation levels, with thresholds mimicking neural firing thresholds
- Implement refractory periods: don't re-expand recently expanded terms (prevents cycles)
- Model temporal decay: recent memories should activate more strongly than distant ones
- Support inhibition: allow negative expansion terms to suppress irrelevant results

**Validation Against Cognitive Science**:

Test whether expansion patterns match human free association:
- Collins & Loftus (1975) spreading activation model predictions
- Semantic priming effect magnitudes from psychology literature
- Recall curves following Ebbinghaus forgetting patterns

### System 1 vs System 2 in Query Processing

- **System 1 (Fast, Automatic)**: Semantic embedding similarity - instant, intuitive matches
- **System 2 (Slow, Deliberate)**: Lexical fallback with explicit term matching - logical, verifiable

Hybrid search implements both systems: semantic for broad recall, lexical for precise verification.

---

## 2. Memory-Systems Perspective

### Episodic-Semantic Distinction in Retrieval

Query expansion sits at the intersection of episodic (specific experiences) and semantic (general knowledge) memory:

**Episodic Queries**: "What did I discuss in the meeting with Sarah last Tuesday?"
- Requires exact lexical matching on proper nouns, dates
- Expansion should be minimal - context is highly specific

**Semantic Queries**: "What do I know about machine learning optimization?"
- Benefits massively from expansion - many paraphrases possible
- Can cross linguistic boundaries (retrieve "optimisation" from British sources)

**Dynamic Switching**: The system must detect query type and adjust expansion strategy:

```rust
match query_type {
    QueryType::Episodic => {
        // Minimal expansion, high precision
        expansion_threshold: 0.9,
        max_synonyms: 2
    },
    QueryType::Semantic => {
        // Broad expansion, high recall
        expansion_threshold: 0.7,
        max_synonyms: 10
    }
}
```

### Consolidation and Query Patterns

Memory consolidation transforms episodic traces into semantic knowledge. Query expansion should mirror this:

1. **Fresh Memories**: Recent episodes retain verbatim wording, so exact lexical match works well
2. **Consolidated Memories**: Old memories are reconstructed from semantic gist, requiring expanded retrieval

**Implementation**: Bias toward semantic search for older memories, lexical search for recent memories:

```rust
let semantic_weight = min(1.0, age_in_days / 30.0);
let lexical_weight = 1.0 - semantic_weight;
hybrid_score = semantic_weight * embedding_sim + lexical_weight * lexical_match;
```

### Pattern Completion through Expansion

Query expansion enables pattern completion - a core memory function where partial cues retrieve complete memories:

- **Partial Cue**: "The patient with chest pain"
- **Expanded**: "chest pain" → ["angina", "myocardial infarction", "cardiac arrest"]
- **Completed Pattern**: Retrieve full case history despite different terminology

This mirrors hippocampal pattern completion where partial CA3 inputs reconstruct full episodic representations.

---

## 3. Rust-Graph-Engine Perspective

### HNSW as Multi-Scale Semantic Network

Query expansion leverages the hierarchical structure of HNSW (Hierarchical Navigable Small World) indices:

**Expansion at Multiple Scales**:

1. **Layer 0 (Fine-Grained)**: Exact synonym matching - "automobile" ↔ "car"
2. **Higher Layers (Coarse)**: Broader semantic fields - "automobile" → "transportation"
3. **Top Layer**: Conceptual domains - "automobile" → "manufacturing," "engineering"

**Implementation Strategy**:

```rust
struct HNSWQueryExpander {
    index: HnswIndex,
    layer_thresholds: Vec<f32>, // Similarity thresholds per layer
}

impl HNSWQueryExpander {
    fn expand_multilayer(&self, query_embedding: &[f32]) -> Vec<(NodeId, f32)> {
        let mut expansions = Vec::new();

        // Start at top layer for broad concepts
        for layer in (0..self.index.max_layer()).rev() {
            let candidates = self.index.search_layer(query_embedding, layer, k=20);
            let threshold = self.layer_thresholds[layer];

            expansions.extend(
                candidates.into_iter()
                    .filter(|(_, sim)| *sim >= threshold)
            );
        }

        expansions
    }
}
```

### Lock-Free Expansion Dictionary Updates

Hot-reloading synonym dictionaries without blocking query processing requires lock-free data structures:

**Arc-Swap Pattern**:

```rust
use arc_swap::ArcSwap;
use std::sync::Arc;

struct ExpansionDictionary {
    // Current dictionary - can be atomically swapped
    current: Arc<ArcSwap<HashMap<String, Vec<(String, f32)>>>>,
}

impl ExpansionDictionary {
    fn reload(&self, new_dict: HashMap<String, Vec<(String, f32)>>) {
        // Atomic swap - readers see old or new, never partial state
        self.current.store(Arc::new(new_dict));
    }

    fn expand(&self, term: &str) -> Vec<(String, f32)> {
        // Load returns Arc<HashMap>, keeps dict alive during read
        let dict = self.current.load();
        dict.get(term).cloned().unwrap_or_default()
    }
}
```

**Zero-Copy Reads**: Multiple concurrent queries can read the dictionary without contention.

### Cache-Optimal Expansion Pipeline

Query expansion can fragment memory access patterns. Optimize via:

1. **Batch Expansion**: Group similar queries, expand all simultaneously
2. **Trie-Based Dictionary**: Prefix tree enables cache-friendly string matching
3. **SIMD String Matching**: AVX-512 for parallel substring checks

```rust
// Cache-friendly batch expansion
fn expand_batch(queries: &[Query], dict: &ExpansionDict) -> Vec<Vec<Term>> {
    // Sort queries by first character for cache locality
    let mut sorted = queries.iter().enumerate().collect::<Vec<_>>();
    sorted.sort_by_key(|(_, q)| q.terms[0].chars().next());

    // Process in order - dictionary lookups hit same cache lines
    let mut results = vec![Vec::new(); queries.len()];
    for (orig_idx, query) in sorted {
        results[orig_idx] = dict.expand(&query.terms);
    }

    results
}
```

---

## 4. Systems-Architecture Perspective

### Tiered Storage for Expansion Resources

Synonym dictionaries and embedding models have different access patterns:

**Hot Tier (RAM)**:
- Core language pairs (e.g., English-Spanish)
- High-frequency terms (top 10K words)
- Recent user-defined expansions
- Access: <1μs lookup

**Warm Tier (Memory-Mapped Files)**:
- Full multilingual dictionaries
- Domain-specific terminologies
- Access: <100μs with page cache

**Cold Tier (Disk)**:
- Historical expansion logs
- Rare language pairs
- Archive of old dictionary versions
- Access: <10ms

**Adaptive Promotion**: Track dictionary term access frequency, promote hot terms to RAM tier.

### Confidence Propagation Architecture

Query expansion creates derived terms with propagated confidence. This requires careful probability management:

**Confidence Decay Model**:

```rust
struct ConfidenceDecay {
    base_confidence: f32,
    hop_penalty: f32,
    source_trust: f32,
}

impl ConfidenceDecay {
    fn propagate(&self, source_conf: f32, similarity: f32, hop_count: u8) -> f32 {
        let hop_factor = self.hop_penalty.powi(hop_count as i32);
        let derived_conf = source_conf * similarity * hop_factor * self.source_trust;

        // Ensure valid probability
        derived_conf.clamp(0.0, 1.0)
    }
}
```

**Calibration**: Validate derived confidences against ground truth relevance judgments. Adjust `hop_penalty` until confidence scores match precision.

### NUMA-Aware Embedding Search

On multi-socket systems, embedding similarity search must respect NUMA topology:

**NUMA-Local Shard Assignment**:

```rust
struct NumaAwareIndex {
    shards: Vec<HnswShard>,
    numa_nodes: Vec<NumaNode>,
}

impl NumaAwareIndex {
    fn search_parallel(&self, query: &[f32]) -> Vec<(MemoryId, f32)> {
        // Pin search threads to NUMA nodes where shards reside
        self.shards.par_iter()
            .zip(self.numa_nodes.iter())
            .map(|(shard, node)| {
                numa::pin_to_node(node.id);
                shard.search(query)
            })
            .flatten()
            .collect()
    }
}
```

**Remote Access Penalty**: Cross-NUMA memory access is 2-3x slower. Keep query embeddings replicated on each NUMA node.

### Write-Ahead Log for Expansion Auditing

Enterprise compliance requires auditing which expansions led to which results:

**Append-Only Expansion Log**:

```rust
struct ExpansionAuditLog {
    wal: WriteAheadLog,
}

struct ExpansionEvent {
    timestamp: u64,
    query_id: Uuid,
    original_term: String,
    expanded_terms: Vec<(String, f32)>,
    source: ExpansionSource, // Dictionary, Embedding, LLM
}

impl ExpansionAuditLog {
    fn log_expansion(&mut self, event: ExpansionEvent) -> Result<()> {
        self.wal.append(&bincode::serialize(&event)?)?;
        Ok(())
    }

    fn query_provenance(&self, query_id: Uuid) -> Vec<ExpansionEvent> {
        self.wal.iter()
            .filter_map(|entry| bincode::deserialize(entry).ok())
            .filter(|e: &ExpansionEvent| e.query_id == query_id)
            .collect()
    }
}
```

**Compaction**: Periodically merge WAL into time-partitioned Parquet files for efficient analytical queries.

---

## 5. Comparative Analysis

| Aspect | Cognitive | Memory-Systems | Rust-Graph-Engine | Systems-Architecture |
|--------|-----------|----------------|-------------------|---------------------|
| **Focus** | Biological plausibility | Memory type dynamics | Data structure efficiency | Resource management |
| **Expansion Metric** | Activation level | Episodic vs semantic weight | HNSW layer distance | Cache hit rate |
| **Confidence Model** | Neural firing rate | Memory consolidation strength | Graph edge weight | Probability calibration |
| **Key Optimization** | Match priming effects | Dynamic episodic/semantic blend | Lock-free dictionary swaps | NUMA-aware sharding |
| **Validation** | Psychology experiments | Memory recall curves | HNSW benchmark | Latency percentiles |

---

## 6. Synthesis: Unified Implementation Strategy

The optimal implementation integrates insights from all perspectives:

1. **From Cognitive-Architecture**: Use spreading activation model with refractory periods and confidence thresholds
2. **From Memory-Systems**: Dynamically adjust expansion based on memory age and query type
3. **From Rust-Graph-Engine**: Leverage HNSW layers for multi-scale expansion, use lock-free dictionary updates
4. **From Systems-Architecture**: Implement tiered storage for dictionaries, NUMA-aware search, and audit logging

**Concrete Architecture**:

```rust
struct UnifiedQueryExpander {
    // Cognitive: spreading activation engine
    activation_engine: SpreadingActivation,

    // Memory-systems: episodic/semantic router
    query_classifier: EpisodicSemanticClassifier,

    // Rust-graph-engine: HNSW-backed semantic expansion
    hnsw_expander: HNSWMultiLayerExpander,
    dictionary: Arc<ArcSwap<ExpansionDictionary>>,

    // Systems-architecture: tiered resources and audit
    tiered_storage: TieredDictionaryStorage,
    audit_log: ExpansionAuditLog,
}

impl UnifiedQueryExpander {
    async fn expand(&self, query: Query) -> ExpandedQuery {
        // Classify query type (episodic vs semantic)
        let query_type = self.query_classifier.classify(&query);

        // Adjust expansion parameters based on type
        let params = self.activation_engine.params_for_type(query_type);

        // Multi-layer semantic expansion via HNSW
        let semantic_expansions = self.hnsw_expander.expand_multilayer(
            &query.embedding,
            params
        );

        // Lexical synonym expansion via dictionary
        let lexical_expansions = self.dictionary.load()
            .expand_terms(&query.terms, params.max_synonyms);

        // Combine and propagate confidence
        let combined = self.activation_engine.merge_expansions(
            semantic_expansions,
            lexical_expansions,
            params.confidence_decay
        );

        // Audit for provenance
        self.audit_log.log_expansion(ExpansionEvent {
            query_id: query.id,
            expansions: combined.clone(),
        }).await?;

        ExpandedQuery {
            original: query,
            expansions: combined,
        }
    }
}
```

---

## 7. Recommended Perspective for Medium Article

**Choose: Systems-Architecture Perspective**

**Rationale**:
- Most actionable for practitioners building production systems
- Covers concrete implementation challenges (NUMA, tiering, audit logging)
- Bridges theory (confidence propagation) with practice (cache optimization)
- Appeals to senior engineers concerned with reliability and observability
- Natural arc: problem → architectural solution → validation strategy

**Article Structure**:
1. Introduction: Query expansion vs query explosion problem
2. Tiered storage: Hot/warm/cold for dictionaries
3. Confidence propagation: Calibrated probability management
4. NUMA-aware search: Multi-socket performance
5. Audit logging: Compliance and debugging
6. Benchmarks: Real-world performance validation
7. Conclusion: Balancing recall, latency, and explainability
