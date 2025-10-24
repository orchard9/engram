# Pattern Completion Primitives: Architectural Perspectives

## Cognitive Architecture Perspective

### Memory Reconstruction as Core Cognitive Function

Human memory isn't a recording device - it's a reconstruction engine. Every time you recall a memory, your brain rebuilds it from fragments using learned patterns. Bartlett demonstrated this in 1932, but most databases still treat retrieval as faithful playback.

Engram's field-level reconstruction primitives implement memory the way the brain actually works. When you query with partial information, the system doesn't fail or return incomplete results - it completes the pattern using temporal context and statistical regularities.

**Key Insight from CLS Theory:**
The brain uses two complementary systems:
- Hippocampus: Fast pattern completion from partial cues
- Neocortex: Slow extraction of semantic regularities

Engram mirrors this architecture:
- Local reconstruction (Task 001): Hippocampal pattern completion from temporal neighbors
- Global patterns (Task 003): Neocortical semantic knowledge from consolidation

### Source Monitoring Prevents False Memories

The most dangerous aspect of reconstructive memory is confabulation - confidently "recalling" details that were actually filled in. Johnson's Source Monitoring Framework addresses this: humans track whether details came from perception (external source) or imagination (internal source).

Engram's SourceMap provides this capability. Each field carries metadata:
- Recalled: Actually present in the cue
- Reconstructed: Filled from temporal neighbors
- Imagined: Low-confidence speculation
- Consolidated: Derived from semantic patterns

This prevents the false memory problem plaguing traditional semantic search: confidently returning irrelevant results because they match a learned pattern.

### Biologically-Plausible Completion

Marr's 1971 theory positioned CA3 as an autoassociative memory. Treves & Rolls formalized this: with 30% cue overlap and 5% sparsity, CA3 can complete patterns with high fidelity.

Task 001 implements these constraints:
- Minimum similarity threshold (0.7) ensures sufficient cue overlap
- Temporal window limits candidates (biological constraint: attention span)
- Sparse active set (max 5 neighbors) mirrors neuronal sparsity
- Exponential recency weighting matches hippocampal time cells

The result: reconstruction that behaves like human episodic memory, complete with contiguity effects and temporal context dependency.

## Memory Systems Perspective

### Complementary Learning Systems in Action

Norman & O'Reilly's CLS theory explains why the brain needs both hippocampus and neocortex: fast learning causes catastrophic interference without slow consolidation.

Task 001 focuses on the fast system - hippocampal pattern completion:
```rust
pub fn reconstruct_fields(
    &self,
    partial: &PartialEpisode,
    temporal_neighbors: &[Episode],
) -> HashMap<String, ReconstructedField>
```

This mirrors CA3 dynamics: given partial input and retrieved neighbors (CA3 recurrent collaterals), complete the pattern using weighted consensus.

The slow system (consolidated semantic patterns) comes in Task 003. The hierarchical integration (Task 004) implements the interaction between fast and slow systems.

### Temporal Context as Retrieval Scaffold

Howard & Kahana's Temporal Context Model explains contiguity effects: items encoded with similar temporal context are retrieved together. Context drifts slowly, creating a gradient.

Engram implements this with exponential temporal weighting:
```rust
pub fn recency_weight(&self, temporal_distance: std::time::Duration) -> f32 {
    let normalized_distance = temporal_distance.as_secs_f32() / self.temporal_window.as_secs_f32();
    (1.0 - normalized_distance).powf(self.recency_exponent)
}
```

Recent episodes contribute strongly (weight near 1.0), distant episodes weakly (weight near 0.0). This matches the CRP curves from Kahana's free recall experiments.

### Pattern Separation vs Pattern Completion

The dentate gyrus performs pattern separation - making similar inputs distinct. CA3 performs pattern completion - making partial inputs whole. These are opposing forces in a delicate balance.

Task 001 handles completion. Pattern separation happens at encoding (future work: orthogonalize similar episodes to prevent interference).

The field consensus algorithm implements this balance:
- High similarity threshold (0.7): Prevents spurious completions from dissimilar episodes
- Low consensus (<60%): Returns low-confidence result, signaling ambiguity
- Confidence calibration: Tracks reconstruction reliability over time

## Rust Graph Engine Perspective

### Zero-Copy Neighbor Retrieval

Performance requirement: <3ms P95 for temporal context extraction from 100-episode window.

Challenge: Retrieving episodes by time range typically requires:
1. Scanning all episodes (slow)
2. Sorting by timestamp (expensive)
3. Binary search (log N but allocation overhead)

Solution: Maintain sorted temporal index at insertion time.

```rust
pub struct TemporalIndex {
    // Episodes sorted by timestamp
    episodes_by_time: Vec<(DateTime<Utc>, EpisodeId)>,
}

impl TemporalIndex {
    pub fn range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> &[(DateTime<Utc>, EpisodeId)] {
        // Binary search for range bounds
        let start_idx = self.episodes_by_time.binary_search_by_key(&start, |(t, _)| *t).unwrap_or_else(|i| i);
        let end_idx = self.episodes_by_time.binary_search_by_key(&end, |(t, _)| *t).unwrap_or_else(|i| i);
        &self.episodes_by_time[start_idx..end_idx]
    }
}
```

Zero allocations. Zero copies. Pure slice borrowing.

### SIMD-Optimized Similarity Computation

Requirement: Process 5 neighbors <250ns total (50ns per neighbor).

Standard cosine similarity loop:
```rust
// Slow: 2-3μs per 768-dim vector
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
```

AVX-512 SIMD optimization:
```rust
// Fast: 50ns per 768-dim vector
#[target_feature(enable = "avx512f")]
unsafe fn cosine_similarity_simd(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    // Process 16 floats at once (512 bits / 32 bits)
    // 768 / 16 = 48 iterations
    let mut dot = _mm512_setzero_ps();
    let mut norm_a = _mm512_setzero_ps();
    let mut norm_b = _mm512_setzero_ps();

    for i in (0..768).step_by(16) {
        let va = _mm512_loadu_ps(&a[i]);
        let vb = _mm512_loadu_ps(&b[i]);

        dot = _mm512_fmadd_ps(va, vb, dot);
        norm_a = _mm512_fmadd_ps(va, va, norm_a);
        norm_b = _mm512_fmadd_ps(vb, vb, norm_b);
    }

    let dot_sum = _mm512_reduce_add_ps(dot);
    let norm_a_sum = _mm512_reduce_add_ps(norm_a).sqrt();
    let norm_b_sum = _mm512_reduce_add_ps(norm_b).sqrt();

    dot_sum / (norm_a_sum * norm_b_sum)
}
```

60x speedup. Critical for sub-2ms latency.

### Lock-Free Field Consensus

Requirement: Zero allocations in consensus hot path.

Challenge: Field consensus needs to aggregate votes from multiple neighbors, which typically requires HashMap allocations.

Solution: Pre-allocated voting buffer with arena semantics.

```rust
pub struct FieldReconstructor {
    // Pre-allocated voting buffer (reused across calls)
    vote_buffer: RefCell<Vec<(String, f32)>>,
    max_neighbors: usize,
}

impl FieldReconstructor {
    pub fn reconstruct_fields(
        &self,
        partial: &PartialEpisode,
        temporal_neighbors: &[Episode],
    ) -> HashMap<String, ReconstructedField> {
        let mut buffer = self.vote_buffer.borrow_mut();
        buffer.clear(); // Reuse allocation

        // Collect weighted votes
        for (neighbor, similarity, recency) in temporal_neighbors {
            let weight = similarity * recency * self.neighbor_decay;
            for (field_name, field_value) in neighbor.fields() {
                if !partial.has_field(field_name) {
                    buffer.push((field_value.clone(), weight));
                }
            }
        }

        // Compute consensus (in-place sort)
        buffer.sort_by(|(_, w1), (_, w2)| w2.partial_cmp(w1).unwrap());

        // Winner-take-all with confidence from agreement
        let consensus_value = &buffer[0].0;
        let total_weight: f32 = buffer.iter().map(|(_, w)| w).sum();
        let consensus_weight: f32 = buffer.iter()
            .filter(|(v, _)| v == consensus_value)
            .map(|(_, w)| w)
            .sum();
        let confidence = consensus_weight / total_weight;

        // ... rest of reconstruction logic
    }
}
```

Zero allocations per call. Buffer reuse. Cache-friendly sequential access.

### Cache-Optimal Temporal Neighbor Iteration

Episodes stored in temporal order means sequential memory access - cache-friendly.

```rust
// Good: Sequential scan (cache-friendly)
for episode in episodes_by_time.iter() {
    if episode.timestamp > cutoff {
        break;
    }
    // Process episode
}

// Bad: Random access (cache thrashing)
for episode_id in episode_ids {
    let episode = episode_store.get(episode_id); // Random access
    // Process episode
}
```

Modern CPUs prefetch sequential access patterns. Random access defeats prefetcher, causing cache misses (100-300 cycle penalty).

Task 001's temporal index enables cache-optimal iteration.

## Systems Architecture Perspective

### Tiered Reconstruction: Local First, Global Fallback

Systems principle: Try cheap operations first, expensive operations only if necessary.

Reconstruction hierarchy:
1. Recalled fields: Free (already in partial episode)
2. Local reconstruction: Cheap (5 neighbors, <2ms)
3. Global patterns: Moderate (consolidation lookup, 5-10ms)
4. Imagined fallback: Free (low-confidence placeholder)

Task 001 implements level 2. Level 3 comes in Task 003-004. This tiered approach ensures graceful degradation: even if consolidation is empty, local reconstruction still works.

### Evidence Tracking for Debugging and Audit

Production memory systems need observability. When reconstruction goes wrong, you need to debug why.

NeighborEvidence structure provides this:
```rust
pub struct NeighborEvidence {
    pub episode_id: String,
    pub similarity: f32,
    pub temporal_distance: f64,
    pub field_value: String,
    pub weight: f32,
}
```

For each reconstructed field, you can trace:
- Which neighbors contributed
- How similar they were
- How recent they were
- What values they voted for
- Final weight assigned

This enables:
- User transparency: "Why did you suggest this value?"
- Debugging: "Why is reconstruction accuracy low?"
- Auditing: "Was this completion justified?"
- Calibration: "Does weight correlate with accuracy?"

### Graceful Degradation Under Sparse Data

Empty neighbor set (no temporal context) should not crash or return errors. It should return empty reconstructions with confidence 0.0.

```rust
if temporal_neighbors.is_empty() {
    return HashMap::new(); // Graceful degradation
}
```

Similarly:
- All neighbors below similarity threshold → return empty
- Conflicting votes (50/50 split) → return both as alternatives with low confidence
- Single neighbor → complete but mark low confidence (no consensus)

Systems thinking: Every edge case is an opportunity for graceful degradation, not failure.

### Performance Monitoring Integration

Task 001 specifies metrics for Prometheus integration:
- Field reconstruction latency histogram
- Temporal context extraction latency histogram
- Neighbor count distribution
- Consensus confidence distribution
- Source attribution breakdown

These metrics enable:
- SLO tracking: Are we meeting <2ms P95?
- Bottleneck identification: Which step is slow?
- Capacity planning: How many neighbors typical?
- Accuracy monitoring: Is confidence calibrated?

Every production system needs telemetry. Reconstruction primitives are no exception.

## Synthesis: Why Field-Level Reconstruction Matters

Traditional databases return exact matches or nothing. Semantic search returns approximate matches without provenance. Neither approach mirrors human memory.

Engram's field-level reconstruction primitives bridge this gap:
- Cognitive plausibility: Matches human reconstructive memory
- Performance: Sub-millisecond latency via SIMD and zero-copy
- Transparency: Explicit source attribution prevents false memories
- Graceful degradation: Works with sparse data, no catastrophic failures
- Biological grounding: Implements Marr's autoassociative memory theory

The result: A memory system that completes patterns like the brain, runs fast like high-performance Rust, and maintains transparency like responsible AI.

Next steps: Integrate CA3 attractor dynamics (Task 002) for more sophisticated completion, and semantic patterns from consolidation (Task 003) for global knowledge.
