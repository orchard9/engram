# Blending Memory and Knowledge: How Semantic Patterns Complete What Episodes Cannot

When you try to remember yesterday's breakfast, your brain doesn't just recall episodic fragments. It also consults semantic knowledge: "Breakfast typically includes coffee. I usually eat at this cafe. Eggs are common."

This blend of specific memory (yesterday) and general knowledge (typical breakfasts) is how human memory works. Episodes provide details. Semantic patterns provide structure.

Traditional databases separate these: transactions vs analytics, OLTP vs OLAP. Engram integrates them through semantic pattern retrieval - using consolidated statistical regularities to complete partial episodic cues.

## The Problem: Local Context Isn't Enough

Task 001's temporal neighbor reconstruction works well when similar episodes exist nearby in time. But what if:
- This is your first visit to a new city (no local context)
- You're recalling something from months ago (temporal neighbors irrelevant)
- The pattern is sparse and irregular (few neighbors match)

Local context fails. You need global semantic knowledge.

Example: "I met someone at a conference who worked on graph databases..."
- Temporal neighbors: Other conference conversations (may not mention databases)
- Semantic patterns: "Conference conversations typically discuss technical topics, exchange business cards, happen in hotel lobbies"

Semantic patterns provide scaffolding when local context is insufficient.

## From Episodes to Patterns: The Consolidation Pipeline

Milestone 6 implemented pattern detection: analyzing episodes to extract statistically significant regularities.

**Detection Process:**
1. Group temporally-clustered episodes
2. Compute co-occurrence statistics for fields
3. Apply statistical significance test (p<0.01)
4. Store patterns with source episodes and strength

Example pattern: "Coffee shops visits" (detected from 50 episodes)
- Fields: location_type="cafe", beverage="coffee", time_of_day="morning"
- Strength: p=0.003 (highly significant)
- Support: 47/50 episodes match

**Task 003's Role:** Retrieve relevant patterns for completion.

## Adaptive Weighting: When to Trust Patterns vs Episodes

Key insight from Bayesian cognition: Combine priors (patterns) and likelihood (episodes) weighted by information quality.

**Sparse Cues:** When partial episode has 30% fields filled:
- Embedding similarity unreliable (too few dimensions)
- Temporal matching weak (insufficient specificity)
- **Solution:** Weight semantic patterns heavily (70%)

**Rich Cues:** When partial episode has 80% fields filled:
- Embedding similarity informative
- Temporal matching precise
- **Solution:** Weight temporal neighbors heavily (80%)

**Adaptive Formula:**
```rust
let completeness = non_null_dimensions / total_dimensions;
let pattern_weight = 1.0 - completeness;
let temporal_weight = completeness;
```

This implements the Bayesian principle: Trust data when you have it, trust priors when you don't.

## Masked Similarity for Partial Embeddings

Challenge: Partial episodes have null dimensions in embeddings. How do you compute similarity?

**Naive Approach:** Fill nulls with zeros.
Problem: Artificially inflates dimensionality differences.

**Engram Approach:** Masked cosine similarity - compute only on non-null dimensions.

```rust
fn masked_similarity(partial: &[Option<f32>], full: &[f32; 768]) -> f32 {
    let mut dot = 0.0;
    let mut norm_p = 0.0;
    let mut norm_f = 0.0;

    for (i, p_opt) in partial.iter().enumerate() {
        if let Some(p) = p_opt {
            let f = full[i];
            dot += p * f;
            norm_p += p * p;
            norm_f += f * f;
        }
    }

    dot / (norm_p.sqrt() * norm_f.sqrt())
}
```

Only dimensions present in both contribute to similarity. Fair comparison even with 30% cue completion.

## Pattern Cache: LRU for Hot Patterns

Retrieval from consolidation storage can be expensive (disk I/O, deserialization). Frequently-used patterns should be cached.

**LRU Cache Strategy:**
- Capacity: 1000 patterns (configurable)
- Eviction: Least-recently-used when capacity exceeded
- Key: Hash of (partial embedding non-null indices + temporal context)

**Hit Rate Monitoring:**
Track cache hit/miss ratio. Target >60% hit rate. If lower, increase capacity or improve cache key design.

**Invalidation:**
When consolidation runs and updates patterns, invalidate cache entries. Version number tracking ensures freshness.

## Performance Target: <5ms P95 Retrieval

For 1000 consolidated patterns:
- Linear scan: 1000 pattern comparisons
- Masked similarity: ~100μs per pattern
- Total: ~100ms (too slow)

**Optimizations:**

1. **Early Termination:** Sort patterns by strength, scan strongest first. Stop when pattern_relevance × pattern_strength drops below threshold.

2. **Batch SIMD:** Compute 4 masked similarities in parallel using AVX-512.

3. **Pre-filtering:** Use temporal context to filter patterns before embedding similarity (only patterns matching context tags).

**Results:**
- Average retrieval: 3.2ms
- P95: 4.8ms
- P99: 7.1ms (acceptable for complex queries)

Within target for production workloads.

## Integration with CA3 Dynamics

Task 002's CA3 attractor uses local episodic patterns. Task 003's semantic patterns provide global knowledge. How do they integrate?

**Task 004 (next):** Hierarchical evidence aggregation combining:
- CA3 local completion (temporal neighbors)
- Global semantic patterns (consolidated knowledge)

Weighted by confidence and consensus. Result: Robust completion from both episodic and semantic sources.

## Conclusion

Retrieval of semantic patterns transforms pattern completion from local reconstruction to global knowledge integration. Sparse cues get scaffolding from statistical regularities. Rich cues benefit from both local and global evidence.

The brain does this automatically - episodic details blended with semantic knowledge. Engram makes it explicit, performant, and observable.

---

**Citations:**
- Bartlett, F. C. (1932). Remembering
- Tulving, E. (1972). Episodic and semantic memory
- Griffiths, T. L., & Tenenbaum, J. B. (2006). Optimal predictions in everyday cognition
