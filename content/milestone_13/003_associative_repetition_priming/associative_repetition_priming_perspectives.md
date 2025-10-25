# Associative and Repetition Priming: Four Architectural Perspectives

## Cognitive Architecture Designer

Tulving & Schacter (1990) made a crucial distinction between explicit and implicit memory systems, with repetition priming being a hallmark of implicit memory. Unlike semantic priming, which operates on meaning-based networks, repetition priming is fundamentally perceptual. Seeing "TABLE" makes you faster at recognizing "TABLE" again, but doesn't fully transfer to "table" or hearing the word spoken.

This specificity reveals that repetition priming operates at the level of perceptual representations, not abstract concepts. In neural terms, it likely involves facilitation in sensory cortices rather than just semantic areas. Our implementation must capture this by storing modality-specific traces with format details.

Associative priming, meanwhile, reflects statistical learning. The brain is extraordinarily good at detecting co-occurrence patterns. When "thunder" and "lightning" appear together repeatedly, neural circuits wire together regardless of whether they're semantically related. This is Hebbian learning at the systems level: neurons that fire together, wire together.

The independence of these priming types is computationally elegant. Semantic priming operates on graph structure, repetition priming on trace strengthening, and associative priming on co-occurrence statistics. They combine additively because they're implemented by different neural mechanisms. This modularity makes implementation cleaner and more biologically plausible.

## Memory Systems Researcher

The empirical data on repetition priming is remarkably robust. Jacoby & Dallas (1981) found 30-50ms facilitation for immediate re-presentation, with effects persisting for days. What's fascinating is the decay function: it's logarithmic, not exponential. A 24-hour delay produces perhaps 50% reduction in effect size, while the first hour produces only 10-15% reduction.

This logarithmic decay suggests repetition priming isn't just residual activation (which would decay exponentially). It's genuine memory trace strengthening. Each repetition makes the trace more permanent, following a power law of practice. This has clear implications for implementation: we need persistent storage, not just transient activation.

Associative priming data from McKoon & Ratcliff (1992) shows different patterns. Effects are intermediate in duration - longer than semantic priming but shorter than repetition priming. Peak facilitation is 40-60ms for high-PMI pairs, with effects visible up to 1500ms SOA. This suggests a hybrid mechanism: activation spreading through learned associations that decay faster than perceptual traces but slower than pure semantic activation.

The validation challenge is ensuring independence. If we manipulate semantic relatedness while holding association constant, we should see semantic priming vary while associative priming stays stable. Similarly, repetition should boost recognition regardless of whether the repeated word is semantically related to context. Our test suite must verify this independence empirically.

## Rust Graph Engine Architect

From an implementation standpoint, repetition priming requires a different data structure than semantic priming. We're not spreading activation through a graph - we're matching current input against stored traces and computing similarity.

The efficient approach is maintaining an index from nodes to their presentation traces:

```rust
pub struct RepetitionIndex {
    // Map from node ID to all its traces
    traces: HashMap<NodeId, Vec<RepetitionTrace>>,
}

impl RepetitionIndex {
    pub fn find_matching_traces(
        &self,
        node: NodeId,
        query: &PresentationDetails,
    ) -> impl Iterator<Item = &RepetitionTrace> {
        self.traces
            .get(&node)
            .into_iter()
            .flat_map(|traces| traces.iter())
            .filter(move |trace| self.matches(trace, query))
    }
}
```

This gives O(1) access to relevant traces plus O(k) filtering where k is the number of traces for that node (typically 1-10). Computing the match quality and decay for 10 traces takes approximately 1-2 microseconds.

For associative priming, the challenge is storing a sparse co-occurrence matrix efficiently. With 100K nodes, a dense matrix would be 10 billion entries. But actual co-occurrences are sparse - perhaps 0.01% density. We use a HashMap for O(1) lookup of existing pairs:

```rust
pub struct CooccurrenceMatrix {
    // Only store non-zero entries
    pairs: HashMap<(NodeId, NodeId), f32>,
}
```

With 1M actual co-occurring pairs at 20 bytes each (16-byte key, 4-byte value), that's 20MB - reasonable. Lookup is single hash table access, approximately 50-100 nanoseconds.

The memory layout matters for cache efficiency. Traces for frequently accessed nodes should be cache-resident. We can achieve this by maintaining a small LRU cache of hot traces, keeping the working set under 32KB (L1 cache size).

## Systems Architecture Optimizer

The systems challenge is managing trace lifetime and memory bounds. Repetition priming traces can persist for days, but we can't keep infinite history. We need eviction policies that maintain statistical validity while bounding memory.

Strategy: probabilistic trace retention. Strong traces (high initial strength, recent presentations) are always kept. Weak traces are randomly sampled with probability proportional to their strength. This ensures we keep the traces most likely to produce measurable priming while bounding total memory.

```rust
impl RepetitionPrimingEngine {
    pub fn evict_weak_traces(&mut self, max_traces: usize) {
        if self.traces.len() <= max_traces {
            return;
        }

        // Sort by strength * recency score
        self.traces.sort_by_key(|t| {
            let age_secs = t.timestamp.elapsed().as_secs_f32();
            let decay = self.logarithmic_decay(age_secs);
            OrderedFloat(-(t.strength * decay))
        });

        // Keep top max_traces
        self.traces.truncate(max_traces);
    }
}
```

Run periodically (every 1000 presentations), this maintains bounded memory while preserving strong traces that contribute most to priming effects.

For associative priming, the challenge is incremental learning. As new co-occurrences are observed, we must update the matrix without expensive recomputation. The temporal weighting approach handles this elegantly: recent observations get weight 1.0, older observations decay by 5% per time window. This creates a sliding window effect where associations naturally age out.

Performance validation requires microbenchmarks specifically for these operations:
- Trace matching: should be under 2μs for typical node
- Co-occurrence lookup: under 100ns
- Combined overhead: under 3μs per retrieval
- Memory: under 500MB total for both systems

Meeting these budgets ensures that adding repetition and associative priming doesn't compromise Engram's core retrieval performance.
