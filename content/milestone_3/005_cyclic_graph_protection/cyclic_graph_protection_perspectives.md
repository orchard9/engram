# Cyclic Graph Protection Perspectives

## Multiple Architectural Perspectives on Task 005: Cyclic Graph Protection

### Cognitive-Architecture Perspective

**Dynamic Inhibition as Cognitive Control:**
Cycle protection mirrors inhibitory control observed in dorsolateral prefrontal cortex, where perseverative loops are suppressed to preserve goal-directed reasoning (Miller & Cohen, 2001). The spreading engine should implement a context-aware inhibitor: the first encounter of a loop attenuates activation, but repeated traversals trigger aggressive suppression.

```rust
pub fn cognitive_inhibition(&mut self, visit: &VisitRecord) {
    let inhibition = self.cycle_penalty
        * (1.0 + f32::from(visit.visit_count) * 0.25)
        * tier_modifier(visit.first_visit_tier);
    self.activation_level.fetch_mul(1.0 - inhibition, Ordering::Relaxed);
    self.confidence.fetch_mul(1.0 - inhibition * 0.5, Ordering::Relaxed);
}
```

**Tier-Specific Executive Control:**
Hot-tier memories act like working memory traces; they require fast inhibition to avoid rumination. Cold-tier memories align with schema reconstruction and can tolerate deeper exploration. By mapping the cognitive control literature onto storage tiers, we preserve plausibility while preventing runaway loops (Baddeley, 2012).

### Rust Graph Engine Perspective

**Cache-Local Visit Tracking:**
Visit records should be laid out contiguously with 32-byte alignment to minimize false sharing. Using `#[repr(C)]` ensures predictable layout when we reuse records from an object pool.

```rust
#[repr(C, align(32))]
pub struct VisitRecord {
    memory_id: MemoryId,
    tier: StorageTier,
    hop_count: AtomicU16,
    visit_count: AtomicU8,
    activation_snapshot: AtomicF32,
}
```

DashMap shards protect write-heavy workloads, but we can reduce contention further with per-thread visit caches flushed at synchronization points. Each worker records first-touch visits in a small `SmallVec` (capacity 16) before promoting them to the shared structure, cutting cross-core chatter for shallow spreads.

**Deterministic Ordering:**
Cycle penalties must be deterministic to preserve repeatable tests. Apply penalties using stable ordering (sorted by `MemoryId`) at the end of each hop, so unit tests and property tests get consistent results across runs.

### Systems Architecture Perspective

**Hot Path Budgeting:**
Cycle detection sits on the critical path of activation updates. Budget: < 200ns per visit on hot tier. We reach this by combining a lock-free Bloom filter (3 hash functions, 1% FP rate) with sharded state. Bloom filter positive ⇒ check DashMap; negative ⇒ allocate record lazily. Cold tier can tolerate more overhead, so we throttle the Bloom refresh interval to reduce memory pressure.

**Instrumentation Hooks:**
Expose counters under `metrics::activation`:
- `cycles_detected_total`
- `cycle_penalties_applied_total`
- `max_cycle_length`
- `termination_time_ns`

Hook them into the monitoring agent defined in Task 012 so regression dashboards highlight pathological graphs quickly.

### Verification & Testing Perspective

**Property-Based Guarantees:**
Use `proptest` to generate random directed graphs with configurable clustering coefficients. Properties:
1. Spreading terminates within `max_hops` even when cycles exist.
2. Activation monotonically decreases when revisiting the same memory.
3. Confidence penalty is proportional to visit count.

```rust
proptest! {
    #[test]
    fn spreading_always_terminates(graph in random_graphs()) {
        let result = run_spreading(&graph, params());
        prop_assert!(result.hops <= params().max_hops);
        prop_assert!(result.activation_norm() < 1e-3);
    }
}
```

**Deterministic Simulation Harness:**
Create golden graphs (triangles, figure-eight, strongly connected clusters). Record expected hop sequences and penalty applications to verify no regressions. This aligns with the acceptance tester's requirement to validate bounded execution (Task 011).

### Memory Systems Perspective

**Adaptive Hop Limits Based on Consolidation Stage:**
Hippocampal memories (hot tier) exhibit short refractory windows; we map this to a `max_hops = 3`. Neocortical representations (warm tier) permit deeper traversal with `max_hops = 5`, while schema reconstruction (cold tier) tolerates `max_hops = 7` for thorough exploration (McClelland et al., 1995).

**Metacognitive Feedback Loop:**
Cycle penalties feed back into the confidence aggregation engine (Task 004). A high penalty signals uncertainty, enabling downstream recall modules to deprioritize cyclic paths. This reflects human metacognition where recognition of repetitive thought reduces perceived reliability (Koriat, 2012).

## Cross-Perspective Synthesis
Cyclic graph protection is not just a correctness safeguard; it encodes cognitive control, maintains deterministic engine behavior, and preserves performance envelopes. By coordinating inhibition logic, cache-friendly data structures, probabilistic guards, and property-based verification, we ensure spreading activation remains biologically plausible and operationally robust.

## Key Citations
- Miller, E. K., & Cohen, J. D. "An integrative theory of prefrontal cortex function." *Annual Review of Neuroscience* (2001).
- Baddeley, A. *Working Memory: Theories, Models, and Controversies.* (2012).
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. "Why there are complementary learning systems in the hippocampus and neocortex." *Psychological Review* (1995).
- Koriat, A. "The subjective confidence in judgments: A metacognitive analysis." *Psychological Review* (2012).
