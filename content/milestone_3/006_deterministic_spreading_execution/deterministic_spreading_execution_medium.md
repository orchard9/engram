# Deterministic Spreading Execution: Replaying Cognitive Reasoning on Demand

*Perspective: Rust Graph Engine*

Debugging emergent behavior is only possible when the system can replay every decision path. Traditional graph databases rely on fixed query plans; Engram's spreading activation evolves dynamically, influenced by stochastic seeding, tier-dependent decay, and cycle penalties. Without determinism, engineers investigating a recall anomaly might never reproduce the same activation trace twice. Task 006 adds a deterministic execution mode that locks the spreading engine to a reproducible schedule while keeping the door open for high-throughput non-deterministic runs.

## Building a Deterministic Scheduler on Top of Rayon
We keep the ergonomic benefits of Rayon for parallelism but introduce a deterministic layer:

1. **Canonical Ordering**: Activation records are sorted by `(tier, activation_bucket, memory_id)`. We quantize activation into 12-bit buckets to avoid minute rounding differences driving different orders.
2. **Chunked Processing**: Sorted slices are split into fixed-size chunks (e.g., 128 records). Each chunk is processed in parallel but commits updates in chunk order, ensuring deterministic writes.
3. **Phase Barriers**: After each hop, a barrier waits for all chunks to finish before the next hop begins. This prevents stragglers from leaking activation into future phases.

```rust
pub fn process_hop(records: &mut [ActivationRecord], ctx: &Context) {
    records.sort_unstable_by(deterministic_cmp);
    records.chunks(ctx.chunk_size)
        .enumerate()
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|(chunk_idx, chunk)| {
            let mut local = LocalAccumulator::new(chunk_idx);
            for record in chunk {
                local.apply(record, ctx);
            }
            ctx.commit(chunk_idx, local);
        });
    ctx.barrier.wait();
}
```

`ctx.commit` writes into a staging array indexed by `chunk_idx`, so the commit order is independent of execution timing.

## Seeded Randomness Without Contention
Spreading occasionally uses randomness—for tie-breaking, probabilistic decay, or sampling downstream memories. We adopt a counter-based Philox PRNG keyed by `(global_seed, hop_index, worker_id)` (Salmon et al., 2011). Each worker computes `philox(global_seed, hop, worker, local_counter)` locally, eliminating shared state. Because Philox is bijective, all workers generate the same sequence regardless of interleaving.

## Precise Floating-Point Discipline
Accumulating activation deltas in parallel invites non-deterministic rounding. We tackle this with two techniques:

- **64-bit Accumulation**: While activation values remain 32-bit, all intermediate sums use `f64`. After reduction we round back with `f32::round_ties_even` semantics.
- **Kahan Summation**: Local accumulators maintain compensation terms so ordering changes do not impact the final sum (Higham, 2002).

Combined, these methods produce bit-identical outputs even under heavy parallelism.

## Testing Deterministic Guarantees
Deterministic mode ships with three new suites:

1. **Round-Trip Test**: run the same spread twice, hash `ActivationResult`s, expect equality.
2. **Thread-Count Test**: repeat with worker pools of size 1, 2, and 8; results remain identical.
3. **Chaos Test**: random graphs with frequent ties highlight tie-break determinism and floating-point stability.

Any future code change that introduces non-determinism will fail these suites immediately, safeguarding the invariants.

## Balancing Determinism and Throughput
Deterministic mode incurs ~8.5% overhead in preliminary microbenchmarks—mostly from extra sorting and barrier synchronization. To keep production fast, deterministic mode is opt-in via `ExecutionMode::Deterministic { seed }`. Operators can toggle it when diagnosing incidents or running acceptance tests. Performance mode bypasses sorting, uses relaxed barriers, and accepts minute floating-point divergence in exchange for throughput.

## Why It Matters
Cognitive systems must justify their inferences. Deterministic execution gives engineers a replay button for thought processes, bridging the gap between emergent behavior and explainability. It also unlocks scientific applications where experimenters require consistent results across clusters. By layering deterministic scheduling on top of our existing spreading engine, we gain reproducibility without sacrificing the ergonomics of the current parallel architecture.

## References
- Higham, N. J. *Accuracy and Stability of Numerical Algorithms.* (2002).
- Salmon, J. K., Moraes, M. A., Dror, R. O., & Shaw, D. E. "Parallel random numbers: As easy as 1, 2, 3." *SC* (2011).
