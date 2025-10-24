# Semantic Pattern Retrieval: Research Foundations

## Schema Theory and Semantic Memory

### Bartlett (1932): Schemas as Organized Knowledge
Schemas are mental frameworks that organize knowledge and guide reconstruction. Memory recall blends episodic details with schema-based expectations.

**Key Insight:** People "recall" schema-consistent details that were never encoded. This isn't error - it's efficient memory using statistical regularities.

### Tulving (1972): Episodic vs Semantic Memory
- Episodic: Specific events with temporal/spatial context ("breakfast at Joe's on Tuesday")
- Semantic: General knowledge without context ("breakfast typically includes coffee")

**Consolidation transforms episodic → semantic:** Repeated patterns extracted from episodes become semantic knowledge.

## Statistical Pattern Detection

### Turk-Browne et al. (2010): Statistical Learning
Brain automatically detects statistical regularities in experience. Occurs implicitly, without conscious awareness.

**Neural Substrate:** Hippocampus tracks co-occurrence statistics. Med temporal lobe extracts patterns. Leads to semantic abstraction.

**Engram Application:** Milestone 6 consolidation detects patterns (p<0.01 significance). Task 003 retrieves these patterns for completion.

### Griffiths & Tenenbaum (2006): Bayesian Models of Cognition
Human cognition implements Bayesian inference: combine priors (semantic knowledge) with likelihood (current evidence).

**Pattern Retrieval as Bayesian Updating:**
```
P(pattern | cue) ∝ P(cue | pattern) × P(pattern)
```
- Likelihood: How well pattern matches partial cue
- Prior: Pattern strength from consolidation (p-value)

## Adaptive Weighting Based on Cue Quality

### Principle of Indifference (Jaynes 1957)
When information is scarce, rely on priors. When information is rich, rely on data.

**Application:** Sparse cues (30% complete) should weight semantic patterns heavily. Rich cues (80% complete) should weight embedding similarity.

**Adaptive Formula:**
```
embedding_weight = cue_completeness
temporal_weight = 1.0 - cue_completeness
```

## Pattern Retrieval Efficiency

### HNSW (Malkov & Yashunin 2018): Hierarchical Navigable Small World
Approximate nearest neighbor search in logarithmic time. Critical for fast semantic pattern retrieval.

**Structure:** Multi-layer graph with skip connections. Navigate from top (coarse) to bottom (fine).

**Performance:** Sub-ms retrieval from millions of patterns.

**Engram Integration:** If consolidation produces >10K patterns, use HNSW indexing. Otherwise, linear scan sufficient (<5ms for 1000 patterns).

## References

1. Bartlett, F. C. (1932). Remembering: A study in experimental and social psychology.
2. Tulving, E. (1972). Episodic and semantic memory. Organization of Memory, 1, 381-403.
3. Turk-Browne, N. B., Scholl, B. J., Chun, M. M., & Johnson, M. K. (2010). Neural evidence of statistical learning. Trends in Cognitive Sciences, 13(2), 47-53.
4. Griffiths, T. L., & Tenenbaum, J. B. (2006). Optimal predictions in everyday cognition. Psychological Science, 17(9), 767-773.
5. Jaynes, E. T. (1957). Information theory and statistical mechanics. Physical Review, 106(4), 620.
6. Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE TPAMI.
