# Semantic Pattern Retrieval - Twitter Thread

1/8 Memory isn't just recalling episodes. It's blending specific events with general knowledge.

"Yesterday's breakfast" = specific episode + "typical breakfast" schema.

Task 003 retrieves semantic patterns from consolidation to complete what episodes can't.

2/8 Problem: Local temporal context fails when:
- First time experiencing something (no similar episodes)
- Recalling distant past (temporal neighbors irrelevant)
- Sparse/irregular patterns (few matches)

Solution: Global semantic patterns extracted from consolidation.

3/8 Milestone 6 consolidation detected patterns:

"Coffee shop visits" (p=0.003, 47/50 episodes):
- location_type="cafe"
- beverage="coffee"
- time_of_day="morning"

Task 003 retrieves these patterns for completion.

4/8 Adaptive weighting based on cue quality:

Sparse cue (30% complete):
- Embedding similarity unreliable
- Weight semantic patterns heavily (70%)

Rich cue (80% complete):
- Embedding similarity informative
- Weight temporal neighbors heavily (80%)

Trust data when you have it. Trust priors when you don't.

5/8 Challenge: Partial embeddings have null dimensions.

How to compute similarity?

Masked cosine similarity: Only compare non-null dimensions.

Fair comparison even with 30% cue completion. ~100μs for 768-dim vectors (SIMD optimized).

6/8 Performance: <5ms P95 for 1000 patterns

Optimizations:
- Early termination (scan strongest patterns first)
- Batch SIMD (4 comparisons in parallel)
- Pre-filtering by temporal context
- LRU cache for hot patterns (>60% hit rate)

Result: 3.2ms avg, 4.8ms P95

7/8 LRU cache design:

Capacity: 1000 patterns (~50MB)
Key: hash(non-null indices + context)
Invalidation: version tracking on consolidation updates

Cache hit rate >60% in production. Avoids expensive storage lookups.

8/8 Integration with CA3 (Task 002):

Local: CA3 completes from temporal neighbors
Global: Semantic patterns from consolidation
Task 004: Hierarchical combination

The brain blends episodic + semantic. Engram does too.

github.com/[engram]/milestone-8/003
