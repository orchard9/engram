# Deterministic Spreading Execution Perspectives

## Multiple Architectural Perspectives on Task 006: Deterministic Spreading Execution

### Cognitive-Architecture Perspective

Deterministic spreading lets researchers replicate emergent cognitive behaviors precisely. When a recall path surfaces an unexpected association, investigators can replay the exact activation sequence to understand why—a requirement for scientific validity (Gluck & Myers, 2019). Deterministic mode effectively freezes the internal "thought process," enabling step-by-step inspection of activation levels, confidence adjustments, and inhibition triggers.

```rust
pub struct CognitiveTrace {
    pub hop_index: u16,
    pub memory_id: MemoryId,
    pub activation: f32,
    pub confidence: f32,
    pub tier: StorageTier,
}
```

By emitting `CognitiveTrace` events in deterministic mode, integrated recall tooling (Task 008) can visualize spreading as a narrative.

### Rust Graph Engine Perspective

**Stable Scheduling Primitive:**
A deterministic scheduler is implemented as a thin layer around Rayon. We partition activation slices by key and process them in sorted order, reusing Rayon to parallelize batches while preserving order via chunked iterators.

```rust
pub fn deterministic_for_each<T: Send>(items: &mut [T], f: impl Fn(&mut T) + Sync) {
    items.sort_unstable_by(deterministic_cmp);
    items.chunks(chunk_size())
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|chunk| chunk.iter().for_each(|item| f(item.clone_mut())));
}
```

`deterministic_cmp` sorts by `(tier, activation_bucket, memory_id)` so ties consistently resolve.

**Floating-Point Discipline:**
We use Kahan summation for aggregation and store activation deltas as 32-bit floats but accumulate in 64-bit to avoid rounding discrepancies (Higham, 2002). After reduction we quantize results back to 32-bit using `f32::from_bits` on deterministic rounding (round-to-nearest-even).

### Systems Architecture Perspective

Deterministic mode introduces phase barriers per hop. Each worker signals completion via an atomic counter; the last worker releases the next phase by toggling a `Phaser` instance (Lea, 2010). We bound the number of phases to `max_hops`, avoiding unbounded synchronization.

```rust
pub fn wait_for_phase(&self, phase: usize) {
    self.barriers[phase % self.barriers.len()].wait();
}
```

For optional overhead control, we expose `ExecutionMode::Performance` that bypasses barriers and stable sorting. Switching modes is a runtime flag so operators can enable determinism when debugging production incidents.

### Verification & Testing Perspective

Determinism needs regression guarantees:
- `determinism_roundtrip`: run spread twice, serialize results, assert byte equality
- `determinism_parallel`: run across varying thread counts (1, 2, N) to ensure invariance
- `tie_breaking_regression`: fixed dataset with known tie resolutions, hashed to golden output

```rust
#[test]
fn determinism_roundtrip() {
    let results_a = deterministic_spread(seed, graph.clone());
    let results_b = deterministic_spread(seed, graph);
    assert_eq!(results_a.hash(), results_b.hash());
}
```

Continuous integration should run determinism suites nightly to catch accidental divergences early.

### Technical Communication Perspective

Deterministic execution is a messaging asset. In developer docs we can promise "every emergent inference can be replayed." That resonates with data scientists accustomed to reproducible notebooks and fosters trust among stakeholders with compliance obligations.

## Key Citations
- Higham, N. J. *Accuracy and Stability of Numerical Algorithms.* (2002).
- Gluck, K. A., & Myers, C. W. *Computational Models of Cognitive Processes.* (2019).
- Lea, D. *The JSR-166 Concurrency Utilities.* (2010).
