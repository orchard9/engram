# Pattern Completion Primitives: Research Foundations

## Memory Reconstruction in Cognitive Psychology

### Bartlett's Schema Theory (1932)
Frederic Bartlett's seminal work "Remembering" demonstrated that human memory is fundamentally reconstructive, not reproductive. His "War of the Ghosts" study showed that people systematically transform memories to fit cultural schemas, adding details that weren't present and omitting incongruous elements.

**Key Findings:**
- Memory recall involves active reconstruction using schemas
- Missing details filled in based on semantic knowledge
- Reconstruction confidence doesn't correlate with accuracy
- People unaware of which details are recalled vs reconstructed

**Implications for Engram:**
Field-level reconstruction must track provenance - which details came from actual recall vs pattern completion. Users need explicit source attribution to avoid false memory formation.

### Pattern Completion in Hippocampal Networks

**Marr's Tri-Level Framework (1971)**
David Marr proposed the hippocampus as an autoassociative memory capable of pattern completion from partial cues. CA3 recurrent collaterals enable completion through attractor dynamics.

**Treves & Rolls Capacity Analysis (1994)**
Mathematical analysis of CA3 pattern completion capacity:
- Completion requires ~30% cue overlap for reliable recall
- Sparse coding (5% active neurons) maximizes storage capacity
- Recurrent weight strength determines attractor basin depth
- Completion accuracy degrades gracefully with cue degradation

**Norman & O'Reilly CLS Theory (2003)**
Complementary Learning Systems theory explains interaction between:
- Hippocampus: Fast episodic pattern completion from partial cues
- Neocortex: Slow semantic knowledge extracted from episodes
- Pattern completion uses both systems: local context + global patterns

## Temporal Context in Memory

### Howard & Kahana Temporal Context Model (2002)
The Temporal Context Model (TCM) explains how temporal proximity drives memory associations:
- Episodes encoded with temporal context representations
- Context drifts slowly over time (exponential decay)
- Recall uses temporal context as retrieval cue
- Accounts for contiguity effects in free recall

**Temporal Weighting Function:**
```
context_weight(t) = exp(-t/tau)
```
where tau controls decay rate (typically 1-2 hours for autobiographical memory).

**Application to Engram:**
Reconstruction should weight temporal neighbors by recency with exponential decay. Recent episodes (minutes-hours ago) provide stronger evidence than distant episodes (days-weeks ago).

### Kahana Contiguity Effects (1996)
Strong empirical finding: Items studied together are recalled together. Temporal contiguity is one of the strongest predictors of memory association.

**Measurement:**
Conditional Response Probability (CRP) shows elevated recall probability for temporally adjacent items (+1, -1 lag) compared to distant items.

**Engram Implementation:**
Field-level reconstruction should prioritize temporal neighbors within sliding windows (default: 1 hour before/after). Similarity threshold (0.7) ensures only contextually-related episodes contribute.

## Consensus and Agreement in Distributed Systems

### Paxos Consensus Algorithm
While designed for distributed systems, Paxos's voting mechanism applies to memory reconstruction:
- Multiple sources propose values (temporal neighbors)
- Weighted voting determines consensus value
- Quorum requirements ensure reliability
- Failure handling via fallback mechanisms

**Adaptation for Memory:**
Replace node voting with neighbor evidence weighting. Similarity × recency × decay determines vote strength. Consensus value is weighted majority, not simple majority.

### Confidence from Agreement
Bayesian framework for computing confidence from source agreement:
- High consensus (90%+ agreement) → confidence ~0.95
- Medium consensus (60-80%) → confidence ~0.75
- Low consensus (<60%) → confidence ~0.50

**Shannon Entropy as Agreement Metric:**
```
H(X) = -Σ p(x) log p(x)
```
Lower entropy (higher agreement) should produce higher confidence. Uniform distribution (maximum entropy) indicates conflicting evidence.

## Embeddings and Similarity Metrics

### Cosine Similarity for Semantic Relatedness
Embedding-based similarity is standard in modern memory systems:
- Cosine similarity in [0, 1] for normalized embeddings
- Threshold (typically 0.6-0.8) separates related from unrelated
- SIMD optimization critical for performance (AVX-512)

**Performance Considerations:**
Task 001 specifies <2ms P95 for 5 neighbors. Requires:
- SIMD dot product: ~50ns for 768-dim vectors
- Batch processing: 5 neighbors in parallel
- Pre-allocation: zero-copy similarity computation

### Partial Embedding Matching
Novel challenge for Engram: matching partial embeddings with null dimensions.

**Approaches:**
1. Masked cosine similarity (only non-null dimensions)
2. Embedding imputation (fill nulls with mean/zero)
3. Probabilistic similarity (treat nulls as distributions)

Task 001 uses masked cosine similarity for simplicity and interpretability.

## Source Monitoring Framework

### Johnson, Hashtroudi, & Lindsay (1993)
Source Monitoring Framework explains how people attribute memories to sources:
- Reality monitoring: Distinguishing perceived from imagined
- Internal source monitoring: Which thought generated a memory
- External source monitoring: Which person/context provided information

**Cues for Source Attribution:**
- Perceptual details → external source (recalled)
- Cognitive operations → internal source (reconstructed/imagined)
- Confidence doesn't predict source accuracy (common misconception)

**Engram Application:**
Source attribution must be independent of completion confidence. High-confidence reconstructions can still be wrong. Explicit source tracking (MemorySource enum) prevents conflation.

### Lindsay & Johnson Reality Monitoring (2000)
Key finding: People struggle to distinguish real memories from imagined/suggested ones, especially when:
- Time delay increases between encoding and test
- Suggested information plausible and schema-consistent
- Source information not explicitly encoded at retrieval

**Design Implications:**
Engram must make source attribution first-class:
- SourceMap data structure per field
- Independent source confidence scores
- Alternative hypotheses to prevent single-path confabulation

## Field-Level vs Episode-Level Reconstruction

### Granularity Trade-offs

**Episode-Level Completion:**
- Pros: Simpler API, single confidence score
- Cons: Loses provenance, can't track mixed sources

**Field-Level Completion:**
- Pros: Precise provenance, mixed-source support
- Cons: More complex, higher memory overhead

Task 001 chooses field-level for accuracy. Real memory involves mixed sources - some details recalled, others reconstructed.

### Metadata Preservation
Each reconstructed field needs:
- Value (string)
- Confidence (0.0-1.0)
- Source (Recalled/Reconstructed/Imagined/Consolidated)
- Evidence trail (contributing neighbors with weights)

Metadata enables:
- User transparency (which details are genuine)
- Confidence calibration (source consensus tracking)
- Debugging (why this value was chosen)
- Temporal analysis (evidence pathway visualization)

## Performance Considerations

### Latency Budgets
Task 001 specifies:
- Field reconstruction: <2ms P95 for 5 neighbors
- Temporal context extraction: <3ms P95 for 100-episode window
- Zero allocations in consensus hot path

**Breakdown:**
- Similarity computation: 5 × 50ns = 250ns (SIMD)
- Temporal filtering: ~1ms (binary search on sorted episodes)
- Field consensus: ~500μs (voting algorithm)
- Evidence packaging: ~300μs (struct allocation)
- Total: ~2ms (tight but achievable)

### Memory Efficiency
Reconstruction primitives should avoid allocations:
- Pre-allocate neighbor buffers (capacity: 10)
- Reuse similarity computation scratch space
- Arena allocation for temporary structures
- Copy-on-write for returned data

**Benchmark Target:**
Process 1000 reconstructions/sec/core with <10MB working memory.

## Validation Against Human Performance

### Serial Position Curves
Classic finding in memory research: Recall varies by study position
- Primacy effect: First items recalled better (consolidated)
- Recency effect: Last items recalled better (still in working memory)
- Middle items weakest (interference)

**Engram Validation:**
Reconstruction accuracy should follow similar pattern when completing episode sequences. Recent episodes (recency) and strongly-consolidated episodes (primacy) should contribute more than middle episodes.

### Cue Overload Effect
As more items associated with a cue, retrieval becomes less effective.

**Prediction for Engram:**
When temporal window contains 50+ similar episodes, reconstruction confidence should decrease due to cue overload. Adaptive window sizing may be necessary.

## References

1. Bartlett, F. C. (1932). Remembering: A study in experimental and social psychology.
2. Marr, D. (1971). Simple memory: A theory for archicortex. Philosophical Transactions of the Royal Society B.
3. Treves, A., & Rolls, E. T. (1994). Computational analysis of the role of the hippocampus in memory. Hippocampus, 4(3), 374-391.
4. Howard, M. W., & Kahana, M. J. (2002). A distributed representation of temporal context. Journal of Mathematical Psychology, 46(3), 269-299.
5. Norman, K. A., & O'Reilly, R. C. (2003). Modeling hippocampal and neocortical contributions to recognition memory. Psychological Review, 110(4), 611.
6. Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. Psychological Bulletin, 114(1), 3.
7. Kahana, M. J. (1996). Associative retrieval processes in free recall. Memory & Cognition, 24(1), 103-109.
8. Lindsay, D. S., & Johnson, M. K. (2000). False memories and the source monitoring framework: Reply to Reyna and Lloyd (1997). Learning and Individual Differences, 12(2), 145-161.
