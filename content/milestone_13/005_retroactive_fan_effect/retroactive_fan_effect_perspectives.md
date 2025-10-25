# Retroactive Fan Effect: Architectural Perspectives

## Cognitive Architecture Designer

The fan effect, discovered by Anderson (1974), represents one of the most robust findings in cognitive psychology: retrieval time increases linearly with the number of associations to a concept. When "lawyer" appears in one sentence, recognition takes 1.11s; with four sentences, it takes 1.17s - a 54ms penalty per additional fan. This isn't a bug in human memory; it's a fundamental constraint of spreading activation.

From a biological perspective, the fan effect emerges from limited attentional resources during memory retrieval. The hippocampus CA3 region implements pattern completion through recurrent connections, but activation must be distributed across all matching associations. Higher fan means thinner activation spread, requiring more time to reach threshold for successful retrieval. The prefrontal cortex must then resolve competition between multiple active representations.

Engram's implementation must capture both components: the automatic spreading of activation (handled by our existing activation dynamics from Milestone 5) and the controlled selection process that resolves interference. This creates a natural integration point where fan effect overhead appears in retrieval latency, making it observable and measurable just like in human subjects.

The temporal dynamics matter critically. Anderson's ACT-R model predicts logarithmic retrieval time as a function of fan, but empirical data shows linear relationships in most experimental paradigms. We implement the linear model based on more recent meta-analyses (Schneider & Anderson, 2012), ensuring our validation criteria match modern understanding rather than early theoretical predictions.

## Memory Systems Researcher

Reder & Ross (1983) demonstrated that the fan effect isn't just about retrieval - it affects recognition confidence. High-fan items show reduced hit rates (from 0.88 to 0.76) and increased false alarm rates (from 0.12 to 0.24). This confidence degradation happens because activation is divided among competing associations, reducing the distinctiveness of any single match.

The retroactive component is crucial: adding new associations to an existing concept retroactively impairs retrieval of old associations. If you learn "The lawyer is in the park" today and "The lawyer is in the bank" tomorrow, retrieval of the park association becomes slower and less confident. This retroactive interference distinguishes fan effect from proactive interference (which affects encoding of new associations).

Statistical validation requires careful experimental design. Anderson's original studies used factorial designs with fan levels 1-4 and measured reaction times with millisecond precision. Our acceptance criteria must replicate the key findings:

1. Linear RT increase: 50-60ms per fan increment (Anderson 1974)
2. Confidence degradation: 12-15% hit rate reduction from fan 1 to fan 4 (Reder & Ross 1983)
3. Error rate increase: 8-12% false alarm increase with high fan (Schneider & Anderson 2012)

The validation must control for semantic similarity, temporal decay, and baseline activation levels - all confounds that can artificially inflate or deflate fan effects. We need within-subjects designs where each concept serves as its own control across different fan conditions.

## Rust Graph Engine Architect

Implementing retroactive fan effect requires efficient counting of association fan during retrieval operations. The naive approach - counting outgoing edges on every retrieval - adds unacceptable latency. We need cached fan counts that update incrementally when associations are added or removed.

The architecture uses atomic fan counters co-located with node metadata:

```rust
pub struct NodeMetadata {
    id: NodeId,
    fan_count: AtomicU32,  // Incremented on association add, decremented on prune
    last_access: AtomicU64,
    base_activation: f32,
}
```

This enables O(1) fan lookups during retrieval with memory ordering Relaxed (no synchronization needed since approximate counts are acceptable). The fan counter is updated during edge insertion:

```rust
pub fn add_association(&self, source: NodeId, target: NodeId) {
    self.edges.insert(source, target);
    self.metadata.get(source).fan_count.fetch_add(1, Ordering::Relaxed);
}
```

The retrieval penalty computation becomes a simple arithmetic operation:

```rust
pub fn calculate_retrieval_time(&self, node: NodeId) -> Duration {
    let fan = self.metadata.get(node).fan_count.load(Ordering::Relaxed);
    let base_time = Duration::from_micros(50);  // Baseline retrieval
    let fan_penalty = Duration::from_micros(12 * (fan - 1) as u64);  // 12us per extra fan
    base_time + fan_penalty
}
```

Performance targets: fan count lookup in <5ns (L1 cache hit on atomic read), retrieval time calculation in <10ns (single multiply-add). Total overhead for fan effect implementation: <15ns per retrieval operation, negligible compared to actual spreading activation costs (500-800us).

## Systems Architecture Optimizer

The fan effect creates an interesting optimization opportunity: we can predict retrieval difficulty before initiating spreading activation. High-fan nodes will require more iterations to resolve, so we can allocate different computational budgets based on fan counts.

Consider a tiered retrieval strategy:

- Fan 1-2 (low competition): Standard spreading activation, 3 iterations max
- Fan 3-5 (moderate competition): Extended activation, 5 iterations, narrower beam search
- Fan 6+ (high competition): Deliberate retrieval with focused attention, 8 iterations, context-guided selection

This adaptive approach keeps average-case retrieval fast (most nodes have fan 1-3) while ensuring high-fan retrievals don't fail due to insufficient activation time. The performance budget is allocated where interference is highest.

Memory layout optimization: fan counters should be cache-line aligned with other hot metadata (last access time, base activation). This ensures a single cache line fetch provides all data needed for retrieval decision-making:

```rust
#[repr(C, align(64))]  // Cache line alignment
pub struct HotNodeMetadata {
    fan_count: AtomicU32,
    last_access: AtomicU64,
    base_activation: f32,
    _padding: [u8; 44],  // Pad to 64 bytes
}
```

The retroactive component requires tracking association timestamps to distinguish old vs new associations to the same concept. This adds 8 bytes per edge for timestamp storage, but enables analysis of how recently added associations affect retrieval of earlier associations. The tradeoff is worth it for cognitive validation, even if production systems might elide timestamps in memory-constrained scenarios.
