# Building Production-Grade Semantic Query Expansion: A Systems Architecture Approach

## The Query Explosion Problem

You've built a semantic search system. Users love the fuzzy matching. Then one day, a query for "cardiac arrest" returns results about "furniture shopping" because your expansion dictionary included "cabinet," which embedded near "cardiac" in your model's confused multilingual space. Your expansion exploded from 2 terms to 47, latency spiked from 20ms to 800ms, and precision collapsed.

Query expansion promises to improve recall by adding synonyms and semantically-related terms. But production systems need more than just recall - they need predictable latency, explainable results, and graceful degradation when things go wrong. This article explores the systems architecture required to ship semantic query expansion that actually works under load.

## The Three Pillars: Storage, Confidence, and Observability

Building production query expansion requires solving three distinct architectural problems:

1. **Tiered Storage**: How do you serve multilingual dictionaries and embeddings at <10ms latency while supporting hot-reload of new terms?
2. **Confidence Propagation**: How do you derive confidence scores for expanded terms that actually correlate with precision?
3. **Audit Logging**: How do you explain to users (and regulators) why a specific result surfaced from an expanded query?

Let's examine each pillar and the tradeoffs involved.

## Pillar 1: Tiered Storage for Expansion Resources

Query expansion draws on three types of resources with vastly different access patterns:

- **Embeddings**: Large (768-1024 dimensions), read-heavy, version-stable
- **Synonym Dictionaries**: Medium-sized (10K-1M entries), update-moderate, language-specific
- **Abbreviation Tables**: Small-to-medium, update-frequent in specialized domains (medical, legal)

A naive implementation loads everything into RAM, wasting gigabytes on rarely-accessed language pairs. A naive disk-based approach reads from SQLite on every query, adding 10-50ms of latency. The solution is tiered storage matching access patterns to memory hierarchy.

### Hot Tier: RAM-Resident Core Dictionaries

Keep the most frequently accessed expansion resources in RAM:

```rust
struct HotTierDictionary {
    // Top 10K terms per core language pair
    core_synonyms: HashMap<String, Vec<(String, f32)>>,

    // Recently used user expansions (LRU cache)
    recent_custom: LruCache<String, Vec<(String, f32)>>,
}
```

Target: <1 microsecond lookup latency. On modern hardware, HashMap lookup is 30-50ns, so we can expand 10-20 terms in under a microsecond.

### Warm Tier: Memory-Mapped Files

For full dictionaries that don't fit in RAM but benefit from page cache:

```rust
struct WarmTierDictionary {
    // mmap'd read-only files, one per language
    maps: HashMap<Language, Mmap>,

    // Trie index into mmap regions for prefix matching
    indices: HashMap<Language, TrieIndex>,
}
```

Memory-mapped files give you the kernel's page cache for free. First access has 100-500μs latency (page fault), subsequent accesses are RAM-speed. This works beautifully for specialized terminology - medical abbreviations might only be accessed during healthcare customer queries.

### Cold Tier: Compressed Archives

For historical versions, rare languages, and audit data:

```rust
struct ColdTierStorage {
    // Parquet files on S3/disk
    archive_path: PathBuf,

    // Metadata index for quick filtering
    manifest: HashMap<DictVersion, FileMetadata>,
}
```

These are accessed <0.1% of the time (debugging, compliance audits). Accept 10-100ms latency in exchange for unbounded storage capacity.

### Adaptive Promotion

Track term access frequency. If a warm-tier term is accessed >100x/second, promote to hot tier:

```rust
struct AdaptiveTierManager {
    access_counters: DashMap<String, AtomicU64>,
    promotion_threshold: u64,
}

impl AdaptiveTierManager {
    fn record_access(&self, term: &str) {
        let counter = self.access_counters
            .entry(term.to_string())
            .or_insert(AtomicU64::new(0));

        let count = counter.fetch_add(1, Ordering::Relaxed);

        if count > self.promotion_threshold {
            self.promote_to_hot_tier(term);
        }
    }
}
```

This creates a self-optimizing system where frequently-queried terms automatically move to faster storage.

## Pillar 2: Confidence Propagation That Actually Works

The core challenge in query expansion is confidence decay. Your original query term has some confidence (usually 1.0 - user explicitly typed it). Expanded terms are derived, so they must have lower confidence. But by how much?

### The Naive Approach (Don't Do This)

```rust
// Bad: ad-hoc confidence decay
fn expand_naive(term: &str, similarity: f32) -> f32 {
    similarity * 0.8  // magic number
}
```

This seems reasonable - if a synonym has 0.9 similarity, give it 0.72 confidence. But it fails in practice:

1. **No calibration**: That 0.72 doesn't correlate with actual precision
2. **No composition**: What happens for multi-hop expansion (synonym-of-synonym)?
3. **No adjustment**: Can't adapt to different query types or domains

### The Principled Approach

Model confidence as probability, propagate via Bayesian inference:

```rust
struct ConfidencePropagator {
    // Prior belief in expansion source quality
    source_trust: HashMap<ExpansionSource, f32>,

    // Per-hop penalty (learned from validation data)
    hop_decay: f32,

    // Domain-specific adjustments
    domain_priors: HashMap<Domain, f32>,
}

impl ConfidencePropagator {
    fn propagate(&self,
                 source_confidence: f32,
                 similarity: f32,
                 hop_count: u8,
                 source: ExpansionSource,
                 domain: Domain) -> f32 {

        // Base propagation: original confidence × similarity
        let base = source_confidence * similarity;

        // Trust in expansion source (dictionary > embedding > LLM)
        let trust = self.source_trust[&source];

        // Multi-hop penalty (exponential decay)
        let hop_penalty = self.hop_decay.powi(hop_count as i32);

        // Domain adjustment (medical terms more conservative)
        let domain_factor = self.domain_priors[&domain];

        let derived_confidence = base * trust * hop_penalty * domain_factor;

        // Clamp to valid probability range
        derived_confidence.clamp(0.0, 1.0)
    }
}
```

The key insight: confidence decay should be a learned function, not a constant. Train it against labeled relevance data.

### Calibration: Confidence Must Predict Precision

After deriving confidences, validate they're actually calibrated:

```rust
fn calibration_error(predictions: &[(f32, bool)]) -> f32 {
    // Bin predictions by confidence
    let mut bins = vec![vec![]; 10];
    for (conf, was_relevant) in predictions {
        let bin = (conf * 10.0) as usize;
        bins[bin].push(*was_relevant);
    }

    // Expected Calibration Error (ECE)
    let mut ece = 0.0;
    for (i, bin) in bins.iter().enumerate() {
        if bin.is_empty() { continue; }

        let predicted_conf = (i as f32 + 0.5) / 10.0;
        let actual_precision = bin.iter()
            .filter(|&&r| r)
            .count() as f32 / bin.len() as f32;

        let weight = bin.len() as f32 / predictions.len() as f32;
        ece += weight * (predicted_conf - actual_precision).abs();
    }

    ece
}
```

If your expansion confidences have ECE <0.05, they're well-calibrated. If ECE >0.15, retune your propagation parameters.

## Pillar 3: Audit Logging for Explainability

Enterprise customers (especially in regulated industries) need to answer: "Why did this result appear?" Audit logging makes query expansion explainable.

### Write-Ahead Log for Expansion Events

Every expansion decision goes into an append-only log:

```rust
#[derive(Serialize, Deserialize)]
struct ExpansionEvent {
    timestamp: u64,
    query_id: Uuid,
    session_id: Uuid,
    user_id: Option<UserId>,

    original_terms: Vec<String>,

    expansions: Vec<Expansion>,
}

#[derive(Serialize, Deserialize)]
struct Expansion {
    term: String,
    source: ExpansionSource,  // Dictionary, Embedding, LLM
    confidence: f32,
    similarity_score: f32,
    hop_count: u8,
    parent_term: Option<String>,
}

struct ExpansionAuditLog {
    wal: WriteAheadLog,
    flush_interval: Duration,
}

impl ExpansionAuditLog {
    async fn log_expansion(&mut self, event: ExpansionEvent) -> Result<()> {
        // Serialize to binary with efficient encoding
        let bytes = bincode::serialize(&event)?;

        // Append to WAL (fsync'd periodically)
        self.wal.append(&bytes).await?;

        Ok(())
    }
}
```

This is write-optimized: append-only, no indexing overhead during query processing.

### Efficient Provenance Queries

For debugging or compliance queries, we need fast lookups. Periodically compact the WAL into Parquet:

```rust
struct ExpansionProvenance {
    // WAL for recent events (<1 hour)
    live_log: ExpansionAuditLog,

    // Parquet files partitioned by time
    archived: ParquetStore,

    // In-memory index of recent queries
    recent_index: HashMap<Uuid, Vec<ExpansionEvent>>,
}

impl ExpansionProvenance {
    async fn query_provenance(&self, query_id: Uuid) -> Vec<ExpansionEvent> {
        // Check recent index first (hot path)
        if let Some(events) = self.recent_index.get(&query_id) {
            return events.clone();
        }

        // Fall back to archived Parquet (cold path)
        self.archived.query()
            .filter("query_id", query_id)
            .execute()
            .await?
    }

    async fn trace_expansion_chain(&self, query_id: Uuid, result_term: &str)
        -> Vec<Expansion> {
        let events = self.query_provenance(query_id).await?;

        // Build expansion tree
        let mut chain = Vec::new();
        let mut current = result_term;

        while let Some(expansion) = events.iter()
            .flat_map(|e| &e.expansions)
            .find(|exp| exp.term == current) {

            chain.push(expansion.clone());

            if let Some(parent) = &expansion.parent_term {
                current = parent;
            } else {
                break;
            }
        }

        chain.reverse();
        chain
    }
}
```

Now you can answer "How did we get from 'MI' to 'myocardial infarction' to 'heart attack'?" with a complete expansion chain.

### Diagnostic Metadata in Responses

Surface expansion info in API responses:

```json
{
  "results": [
    {
      "memory_id": "mem_abc123",
      "content": "Patient experienced heart attack symptoms",
      "confidence": 0.82,
      "expansion_trace": {
        "original_terms": ["MI"],
        "expanded_via": [
          {
            "term": "myocardial infarction",
            "source": "medical_abbreviations_v2.3",
            "confidence": 0.95
          },
          {
            "term": "heart attack",
            "source": "embedding_similarity",
            "confidence": 0.87
          }
        ]
      }
    }
  ]
}
```

This gives users (and your debugging team) complete transparency into why results appeared.

## NUMA-Aware Embedding Search

On multi-socket servers, memory locality matters. If your query embedding resides on NUMA node 0 but your HNSW index lives on node 1, you pay 2-3x latency penalty for cross-socket access.

### NUMA-Local Index Sharding

Pin index shards to NUMA nodes:

```rust
use numa::*;

struct NumaAwareVectorIndex {
    shards: Vec<HnswShard>,
    numa_topology: NumaTopology,
}

impl NumaAwareVectorIndex {
    fn new(config: IndexConfig) -> Self {
        let numa_topology = NumaTopology::detect();
        let num_nodes = numa_topology.node_count();

        // Create one shard per NUMA node
        let mut shards = Vec::new();
        for node_id in 0..num_nodes {
            // Allocate shard memory on this NUMA node
            numa::allocate_on_node(node_id);

            let shard = HnswShard::new(config.clone());
            shards.push(shard);
        }

        Self { shards, numa_topology }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(MemoryId, f32)> {
        // Get current thread's NUMA node
        let local_node = numa::current_node();

        // Search local shard first (fast)
        let mut results = self.shards[local_node].search(query, k);

        // If insufficient results, search remote shards (slower)
        if results.len() < k {
            for (node_id, shard) in self.shards.iter().enumerate() {
                if node_id == local_node { continue; }

                results.extend(shard.search(query, k - results.len()));
            }
        }

        // Re-rank combined results
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }
}
```

This prioritizes local memory access, falling back to remote nodes only when necessary.

### Query Embedding Replication

To eliminate remote access entirely, replicate query embeddings to all NUMA nodes:

```rust
struct ReplicatedQueryEmbedding {
    replicas: Vec<Box<[f32]>>,
}

impl ReplicatedQueryEmbedding {
    fn from_embedding(embedding: &[f32], num_nodes: usize) -> Self {
        let mut replicas = Vec::with_capacity(num_nodes);

        for node_id in 0..num_nodes {
            // Allocate on specific NUMA node
            numa::set_preferred_node(node_id);

            let replica = embedding.to_vec().into_boxed_slice();
            replicas.push(replica);
        }

        Self { replicas }
    }

    fn get_local(&self) -> &[f32] {
        let node_id = numa::current_node();
        &self.replicas[node_id]
    }
}
```

Small overhead for embedding duplication (typically <5KB per query), huge latency win on multi-socket machines.

## Lock-Free Dictionary Updates

Production systems need to update synonym dictionaries without downtime. The challenge: how do you swap dictionaries atomically while thousands of concurrent queries are reading them?

### Arc-Swap Pattern

```rust
use arc_swap::ArcSwap;
use std::sync::Arc;

struct LiveReloadDictionary {
    // Atomically swappable dictionary pointer
    current: Arc<ArcSwap<ExpansionDict>>,

    // Background thread for reload coordination
    reload_handle: JoinHandle<()>,
}

impl LiveReloadDictionary {
    fn new(initial: ExpansionDict) -> Self {
        let current = Arc::new(ArcSwap::from_pointee(initial));

        // Spawn background thread to watch for updates
        let current_clone = current.clone();
        let reload_handle = tokio::spawn(async move {
            Self::reload_loop(current_clone).await;
        });

        Self { current, reload_handle }
    }

    async fn reload_loop(dict: Arc<ArcSwap<ExpansionDict>>) {
        let mut watcher = notify::watcher(Duration::from_secs(5));
        watcher.watch("./dictionaries/", RecursiveMode::Recursive);

        while let Some(event) = watcher.next().await {
            if let notify::Event::Write(_) = event {
                // Load new dictionary
                let new_dict = ExpansionDict::load("./dictionaries/").await?;

                // Atomic swap - all readers instantly see new version
                dict.store(Arc::new(new_dict));

                info!("Dictionary reloaded successfully");
            }
        }
    }

    fn expand(&self, term: &str) -> Vec<(String, f32)> {
        // Load returns Arc<ExpansionDict>
        // Keeps dictionary alive for duration of this function
        let dict = self.current.load();

        dict.lookup(term)
    }
}
```

Key properties:
- **Zero locks**: Readers never block writers or other readers
- **Atomic visibility**: Readers see old version or new version, never partial state
- **Memory safety**: Old dictionaries are freed only after all readers drop references

## Putting It All Together: End-to-End Architecture

Here's how the three pillars combine in a unified query expansion pipeline:

```rust
struct ProductionQueryExpander {
    // Pillar 1: Tiered storage
    hot_tier: HotTierDictionary,
    warm_tier: WarmTierDictionary,
    embedding_index: NumaAwareVectorIndex,
    tier_manager: AdaptiveTierManager,

    // Pillar 2: Confidence propagation
    confidence: ConfidencePropagator,

    // Pillar 3: Audit logging
    audit_log: ExpansionAuditLog,
    provenance: ExpansionProvenance,
}

impl ProductionQueryExpander {
    async fn expand(&self, query: Query) -> Result<ExpandedQuery> {
        let start = Instant::now();
        let mut expansions = Vec::new();

        // 1. Semantic expansion via embeddings (high recall)
        let semantic = self.embedding_index
            .search(&query.embedding, k=20)
            .into_iter()
            .map(|(term, sim)| {
                let conf = self.confidence.propagate(
                    1.0,  // original query confidence
                    sim,
                    0,    // direct expansion (0 hops)
                    ExpansionSource::Embedding,
                    query.domain
                );
                Expansion { term, confidence: conf, source: ExpansionSource::Embedding, ... }
            })
            .collect::<Vec<_>>();

        // 2. Lexical expansion via dictionary (high precision)
        let lexical = self.expand_lexical(&query.terms)
            .into_iter()
            .map(|(term, dict_conf)| {
                let conf = self.confidence.propagate(
                    1.0,
                    dict_conf,
                    0,
                    ExpansionSource::Dictionary,
                    query.domain
                );
                Expansion { term, confidence: conf, source: ExpansionSource::Dictionary, ... }
            })
            .collect::<Vec<_>>();

        // 3. Merge and deduplicate
        expansions.extend(semantic);
        expansions.extend(lexical);
        expansions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        expansions.dedup_by(|a, b| a.term == b.term);

        // 4. Cap total expansions (prevent explosion)
        expansions.truncate(20);

        // 5. Record adaptive access for tier promotion
        for term in &query.terms {
            self.tier_manager.record_access(term);
        }

        // 6. Audit log
        self.audit_log.log_expansion(ExpansionEvent {
            query_id: query.id,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            original_terms: query.terms.clone(),
            expansions: expansions.clone(),
        }).await?;

        let latency = start.elapsed();
        metrics::histogram!("query_expansion_latency_ms", latency.as_millis() as f64);

        Ok(ExpandedQuery {
            original: query,
            expansions,
            diagnostics: ExpansionDiagnostics {
                semantic_count: semantic.len(),
                lexical_count: lexical.len(),
                latency_ms: latency.as_millis(),
            }
        })
    }

    fn expand_lexical(&self, terms: &[String]) -> Vec<(String, f32)> {
        let mut results = Vec::new();

        for term in terms {
            // Try hot tier first (<1μs)
            if let Some(expansions) = self.hot_tier.lookup(term) {
                results.extend(expansions);
                continue;
            }

            // Fall back to warm tier (<100μs)
            if let Some(expansions) = self.warm_tier.lookup(term) {
                results.extend(expansions);
                continue;
            }

            // Cold tier only for rare/debugging cases (>10ms)
            // In practice, omit cold tier from query hot path
        }

        results
    }
}
```

## Benchmarks: Does It Actually Work?

Theory is worthless without validation. Here are real-world performance targets and how to measure them:

### Latency Targets

- **Embedding Search**: <1ms for top-k=20 from 1M vectors
- **Dictionary Lookup**: <10μs for hot tier, <100μs for warm tier
- **Confidence Propagation**: <5μs per expanded term
- **Audit Logging**: <50μs (asynchronous, non-blocking)

**Total Query Expansion Latency**: <5ms p95, <10ms p99

### Recall/Precision Targets

Using MTEB (Massive Text Embedding Benchmark) cross-lingual test sets:

- **nDCG@10**: ≥0.80 across English/Spanish/Mandarin
- **Recall Improvement**: ≥15% over baseline lexical search
- **Precision Maintenance**: ≤5% degradation vs single-language baseline

### Calibration Target

- **Expected Calibration Error (ECE)**: <0.05

If confidence=0.8, then 80% of results with that confidence should be relevant.

### NUMA Performance Validation

On a dual-socket server (2x 32-core):

- **Local NUMA search**: 0.8ms p95
- **Remote NUMA search**: 2.4ms p95 (3x penalty)
- **With NUMA-aware sharding**: 0.9ms p95 (minimal penalty)

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Unbounded Expansion

Expanding every term with every synonym leads to query explosion. A query with 5 terms, each expanding to 10 synonyms, becomes 50 terms - destroying both latency and precision.

**Solution**: Cap total expansion count (recommend 20-30 terms max) and use confidence thresholds (drop terms below 0.6 confidence).

### Pitfall 2: Recursive Expansion Without Termination

Expanding synonyms-of-synonyms can create cycles: "car" → "automobile" → "vehicle" → "car".

**Solution**: Track visited terms and limit hop count (recommend max 2 hops).

### Pitfall 3: Uncalibrated Confidence

Assigning confidence=0.8 to expanded terms without validation leads to over-confident garbage results.

**Solution**: Train confidence propagation on labeled data and measure ECE regularly.

### Pitfall 4: Synchronous Audit Logging

Blocking query processing to fsync audit logs adds 1-5ms per query.

**Solution**: Use async logging with periodic flush (acceptable to lose <1 second of logs on crash).

### Pitfall 5: Cross-NUMA Access

Ignoring NUMA topology on multi-socket machines can triple query latency.

**Solution**: Pin threads, shard data across nodes, replicate small read-heavy structures.

## Conclusion: Balancing Recall, Latency, and Explainability

Production query expansion is about tradeoffs:

- **Recall vs Precision**: Expand too much, precision collapses. Too little, recall suffers. Target ≥15% recall improvement while maintaining ≥95% of baseline precision.

- **Latency vs Coverage**: Full multilingual dictionaries take 100ms to load from disk. Tiered storage gives you 95% coverage in <1ms, 99.9% in <10ms.

- **Simplicity vs Correctness**: Ad-hoc confidence decay (multiply by 0.8) is simple but wrong. Bayesian propagation is more complex but actually calibrates.

- **Privacy vs Observability**: Full audit logging enables debugging but stores sensitive query data. Retention policies and anonymization are non-negotiable.

The architecture presented here - tiered storage, calibrated confidence propagation, and audit logging - represents the minimum viable complexity for production semantic search. Skip any of these pillars and you'll regret it when debugging why "cardiac arrest" returned "furniture shopping" at 3am.

Query expansion, done right, transforms semantic search from a research demo into a reliable production system. The difference is systems thinking: designing for failure modes, measuring what matters, and making tradeoffs explicit.

## Further Reading

- Google Research: "Leveraging Semantic and Lexical Matching to Improve Recall" (2020)
- Microsoft: "Hybrid Search Overview" - Azure AI Search Documentation
- Journal of Biomedical Semantics: "Synonym extraction and abbreviation expansion with ensembles of semantic spaces" (2014)
- MTEB: Massive Text Embedding Benchmark for cross-lingual evaluation
- Elastic: "A Comprehensive Hybrid Search Guide" (2024)

---

About Engram: A cognitive graph database implementing biologically-inspired memory dynamics for episodic and semantic recall. Semantic query expansion enables cross-lingual retrieval while maintaining enterprise-grade reliability and explainability.
