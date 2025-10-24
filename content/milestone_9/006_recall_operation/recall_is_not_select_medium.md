# RECALL Is Not SELECT: Why Cognitive Queries Need Different Semantics

When we built Engram's query language, we faced a tempting choice: make it look like SQL. After all, developers know SQL. SELECT, WHERE, JOIN - these are comfortable patterns.

But here's the thing: memory recall isn't database retrieval. And pretending it is creates fundamental semantic mismatches that no amount of syntactic sugar can fix.

Let me show you what I mean.

## The Confidence Problem

Consider this innocent-looking SQL query:

```sql
SELECT * FROM memories
WHERE confidence > 0.7
  AND created_at > '2024-01-01';
```

Seems reasonable, right? Get memories with high confidence created this year.

But there's a subtle problem. In SQL, `confidence` is just another column - user-supplied data that sits in storage. It doesn't change. If you inserted a row with confidence 0.9 three months ago, it's still 0.9 today.

That's not how memory works.

Memory confidence decays with time. An episode you were 90% sure about yesterday might be 75% sure about today and 60% sure about next week. The confidence isn't a stored value - it's a computed property that depends on when you ask.

Here's what RECALL does instead:

```
RECALL episode
WHERE confidence > 0.7
  AND created > "2024-01-01"
```

Under the hood, this computes confidence dynamically:

```rust
fn compute_confidence(
    base_confidence: f32,
    age: Duration,
    decay_rate: f32,
) -> f32 {
    base_confidence * (-decay_rate * age.as_secs_f32()).exp()
}
```

That episode stored with 0.9 confidence three days ago? At 5% daily decay, it's now 0.77. RECALL factors this in automatically. SQL doesn't.

You could add this logic to SQL with triggers or computed columns, but now you're fighting the data model. The database thinks confidence is data. We know it's metadata about data quality.

## The Similarity Problem

Let's try another query. Find meetings where we discussed the API design:

```sql
SELECT * FROM events
WHERE type = 'meeting'
  AND description LIKE '%API%design%';
```

This finds exact string matches. But what if someone wrote "interface specification" instead of "API design"? What about "REST endpoint planning"? These are semantically similar, but SQL's string matching can't see that.

You might think: "Just use full-text search!" Sure, that helps with synonyms. But it doesn't capture semantic meaning.

RECALL operates in embedding space:

```
RECALL episode
WHERE content SIMILAR TO api_design_embedding
THRESHOLD 0.7
```

Here's what's happening:

```rust
fn similarity_filter(
    episodes: Vec<Episode>,
    query_embedding: &[f32],
    threshold: f32,
) -> Vec<(Episode, Confidence)> {
    episodes.into_iter()
        .map(|ep| {
            let sim = cosine_similarity(&ep.embedding, query_embedding);
            (ep, Confidence::new(sim))
        })
        .filter(|(_, conf)| conf.value() >= threshold)
        .collect()
}
```

This finds "interface specification" and "REST endpoint planning" because they're semantically similar in embedding space, even though they share no tokens with "API design".

SQL can't express this naturally. You'd need to bolt on vector extensions (pgvector, etc.), store embeddings separately, and write complex procedural code. The query language isn't designed for semantic similarity - it's designed for discrete matching.

## The Uncertainty Problem

Here's where it gets really different.

SQL queries return discrete results:

```sql
SELECT * FROM users WHERE age > 30;
-- Returns: [Alice, Bob, Charlie]
```

These users either match or they don't. There's no ambiguity, no uncertainty about whether Bob should be in the result set.

RECALL returns probabilistic distributions:

```
RECALL episode WHERE confidence > 0.7
```

Returns:
```rust
vec![
    (episode_1, Confidence::new(0.92)),
    (episode_2, Confidence::new(0.81)),
    (episode_3, Confidence::new(0.73)),
]
```

Each result comes with a confidence score. But more importantly, the entire result set has uncertainty:

```rust
pub struct ProbabilisticQueryResult {
    pub episodes: Vec<(Episode, Confidence)>,
    pub aggregate_confidence: ConfidenceInterval,  // [0.73, 0.92]
    pub uncertainty_sources: Vec<UncertaintySource>,
}
```

The system tells you: "I found these three episodes, and I'm between 73% and 92% confident this is the complete answer."

SQL doesn't have a concept of aggregate uncertainty. Results are either present or absent, never "probably present with 80% confidence."

Why does this matter? Because when you're building AI systems, uncertainty propagation is critical. If episode retrieval is 80% confident, and subsequent reasoning depends on that retrieval, you need to know the uncertainty compounds.

## The Evidence Problem

Let's say a SQL query returns unexpected results. You run EXPLAIN ANALYZE:

```sql
EXPLAIN ANALYZE
SELECT * FROM memories WHERE confidence > 0.7;

Seq Scan on memories  (cost=0.00..10.75 rows=5 width=32)
  Filter: (confidence > 0.7)
  Rows Removed by Filter: 3
```

This tells you HOW the query ran - sequential scan, 5 rows matched, 3 filtered out. Useful for performance tuning.

But it doesn't tell you WHY specific rows returned. Why did episode A match but episode B didn't? What made the system confident about episode A?

RECALL provides evidence chains:

```rust
Evidence::Confidence(ConfidenceEvidence {
    base_confidence: 0.9,
    uncertainty_sources: vec![
        UncertaintySource::TemporalDecay {
            age: Duration::from_days(3),
            decay_rate: 0.05,
        },
        UncertaintySource::RetrievalAmbiguity {
            similar_nodes: 5,
        },
    ],
    calibrated_confidence: 0.77,
    method: CalibrationMethod::Platt,
})
```

This tells you:
- Base confidence was 0.9
- Dropped to 0.77 due to 3-day temporal decay
- 5 similar nodes created retrieval ambiguity
- Platt scaling calibrated the score

Now you understand not just WHAT returned, but WHY and HOW CONFIDENT to be about it.

This is the difference between database introspection (performance-focused) and cognitive introspection (correctness-focused).

## The Performance Problem

You might think: "This all sounds nice, but SQL is fast. Databases have decades of optimization."

True. But SQL is optimized for the wrong access patterns.

SQL excels at:
- B-tree lookups (indexed exact matches)
- Hash joins (equi-joins on keys)
- Sequential scans (full table scans with filters)

RECALL needs:
- K-nearest neighbor search (find similar embeddings)
- Graph traversal (spreading activation)
- Probabilistic ranking (confidence-weighted results)

These are fundamentally different algorithms. You can implement K-NN on top of SQL (using pgvector or custom indexes), but you're fighting the query planner. The database doesn't understand semantic similarity, so it can't optimize for it.

Here's RECALL's performance profile:

```rust
// K-NN search with HNSW index
let neighbors = hnsw_index.search(query_embedding, k=10);
// Typical: 10-100 microseconds, O(log N) complexity

// Filter by confidence threshold
let filtered = neighbors.into_iter()
    .filter(|(_, conf)| conf >= threshold);

// Apply temporal decay
let with_decay = filtered.map(|(ep, conf)| {
    let age = now - ep.created_at;
    let decayed = conf * exp(-decay_rate * age);
    (ep, decayed)
});
```

Total time for typical query: 100 microseconds for K-NN + 10 microseconds for filtering/ranking = 110 microseconds.

SQL equivalent (using pgvector):
```sql
SELECT *, 1 - (embedding <=> query_embedding) AS similarity
FROM memories
WHERE 1 - (embedding <=> query_embedding) > 0.7
ORDER BY embedding <=> query_embedding
LIMIT 10;
```

This works, but the query planner treats it as an opaque operator. It can't push down filters, reorder operations, or parallelize intelligently. Typical time: 500 microseconds to 2 milliseconds.

RECALL is faster because it's designed for this access pattern from the ground up.

## The Cognitive Semantics

Here's the core insight: SQL was designed for data retrieval. RECALL was designed for memory retrieval.

These sound similar, but they're fundamentally different:

**Data Retrieval** (SQL):
- Data is ground truth
- Queries find facts
- Results are deterministic
- Correctness = matches predicate

**Memory Retrieval** (RECALL):
- Memory is reconstruction
- Queries find likely experiences
- Results are probabilistic
- Correctness = reflects confidence

SQL's relational algebra is perfect for the first. But it's a mismatch for the second.

When you ask your brain "What did I have for breakfast?" you're not running a SELECT query on a breakfast facts table. You're reconstructing a recent experience from distributed memory traces, weighted by recency, relevance, and confidence.

That's what RECALL does.

## A Real Example

Let me show you a concrete example from Engram.

Task: Find high-confidence meetings from the past month where the project was discussed.

**SQL Approach**:
```sql
SELECT e.id, e.description, e.created_at,
       e.base_confidence * EXP(-0.05 * EXTRACT(EPOCH FROM (NOW() - e.created_at)) / 86400) AS decayed_confidence
FROM events e
WHERE e.type = 'meeting'
  AND e.created_at > NOW() - INTERVAL '30 days'
  AND e.base_confidence * EXP(-0.05 * EXTRACT(EPOCH FROM (NOW() - e.created_at)) / 86400) > 0.7
  AND (
    e.description ILIKE '%project%'
    OR EXISTS (
      SELECT 1 FROM event_embeddings ee
      WHERE ee.event_id = e.id
        AND 1 - (ee.embedding <=> (SELECT embedding FROM embeddings WHERE name = 'project_discussion')) > 0.7
    )
  )
ORDER BY e.created_at DESC;
```

This is:
- 200+ characters
- Requires manual decay calculation
- Uses subqueries for embedding similarity
- Mixes string matching with semantic search
- No aggregate confidence or uncertainty tracking

**RECALL Approach**:
```
RECALL episode
WHERE type = "meeting"
  AND created > "2024-09-24"
  AND content SIMILAR TO project_embedding THRESHOLD 0.7
  AND confidence > 0.7
```

This is:
- 140 characters
- Automatic decay handling
- Native embedding similarity
- Clean semantic constraints
- Returns confidence intervals and evidence

Same semantics, clearer intent, better performance.

## When to Use Each

I'm not saying SQL is bad. It's excellent for what it was designed for.

**Use SQL when**:
- You have relational data (users, products, orders)
- You need transactional guarantees (ACID)
- You're querying facts, not experiences
- Determinism is critical
- The data model is naturally tabular

**Use RECALL when**:
- You have episodic memory (events, experiences)
- You need probabilistic results (confidence-weighted)
- You're querying semantics, not strings
- Uncertainty is inherent
- The data model is naturally graph-based

For Engram, we need both. We use PostgreSQL for system metadata (user accounts, access control, configuration). We use RECALL for cognitive queries (episodic memory, pattern retrieval, semantic search).

## The Implementation

If you're curious how this works in practice, here's the RECALL execution pipeline:

```rust
impl QueryExecutor {
    pub fn execute_recall(
        &self,
        query: RecallQuery,
        space: &SpaceHandle,
    ) -> Result<ProbabilisticQueryResult> {
        // 1. Parse query into AST
        let ast = Parser::parse(&query.text)?;

        // 2. Convert pattern to embedding cue
        let cue = self.pattern_to_cue(&ast.pattern)?;

        // 3. K-NN search in HNSW index
        let candidates = space.store.knn_search(&cue, k=100)?;

        // 4. Apply constraints (confidence, temporal, similarity)
        let filtered = self.apply_constraints(candidates, &ast.constraints)?;

        // 5. Apply temporal decay
        let with_decay = filtered.into_iter()
            .map(|(ep, conf)| {
                let age = SystemTime::now().duration_since(ep.created_at)?;
                let decayed = conf.value() * (-self.config.decay_rate * age.as_secs_f32()).exp();
                Ok((ep, Confidence::new(decayed)))
            })
            .collect::<Result<Vec<_>>>()?;

        // 6. Calibrate confidences (Platt scaling)
        let calibrated = self.calibrator.calibrate(with_decay)?;

        // 7. Build evidence chain
        let evidence = self.build_evidence(&calibrated)?;

        // 8. Return probabilistic result
        Ok(ProbabilisticQueryResult {
            episodes: calibrated,
            aggregate_confidence: self.compute_aggregate_confidence(&calibrated),
            uncertainty_sources: vec![
                UncertaintySource::TemporalDecay { age, decay_rate },
                UncertaintySource::RetrievalAmbiguity { similar_nodes: calibrated.len() },
            ],
            evidence_chain: evidence,
        })
    }
}
```

Each step is optimized for cognitive semantics:
- K-NN search for semantic similarity
- Confidence filtering for uncertainty
- Temporal decay for forgetting
- Calibration for accuracy
- Evidence for explainability

You could build all this on top of SQL, but you'd be fighting the abstractions at every step. With RECALL, the abstractions match the problem domain.

## The Future

RECALL is the first cognitive operation in Engram's query language. But it's not the only one.

Coming next:
- SPREAD: Spreading activation for associative retrieval
- PREDICT: Temporal prediction with uncertainty
- IMAGINE: Pattern completion from partial cues
- CONSOLIDATE: Episodic-to-semantic memory transfer

None of these map cleanly to SQL primitives. They're cognitive operations that need cognitive semantics.

This is the future of query languages for AI systems: not adapting database queries to handle uncertainty, but designing queries around cognitive operations from the start.

## Takeaways

If you remember nothing else:

1. Memory confidence decays. Store it as metadata, not data.
2. Semantic similarity beats string matching for cognitive queries.
3. Probabilistic results need aggregate uncertainty, not just per-item scores.
4. Evidence chains explain WHY results returned, not just HOW they were found.
5. Cognitive operations need cognitive semantics, not relational algebra.

RECALL isn't SELECT with fuzzy matching. It's a fundamentally different operation for a fundamentally different data model.

And that's exactly what makes it powerful.

---

Want to try RECALL yourself? Engram is open source:
https://github.com/engram-memory/engram

Questions? Find me on Twitter: @engrammemory

Code samples from this article:
https://github.com/engram-memory/engram/tree/main/examples/query_examples.rs
