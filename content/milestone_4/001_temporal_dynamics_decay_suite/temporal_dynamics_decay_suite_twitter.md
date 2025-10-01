# Temporal Dynamics Decay Suite Twitter Content

## Thread: Teaching Engram to Forget (On Purpose)

**Tweet 1/10**
Most databases never forget. Your brain does. Milestone 4 for Engram is about importing the Ebbinghaus forgetting curve into our graph engine so memories fade realistically instead of rotting silently. (Ebbinghaus, 1885)

**Tweet 2/10**
Why? Because recall quality depends on WHEN you last touched the memory. Wixted & Ebbesen showed that long-term retention follows a power law, not just an exponential drop. We need both kernels selectable per memory. (Wixted & Ebbesen, 1991)

**Tweet 3/10**
Every episode stores `last_access`, decay params, and activation mass. On recall we compute `elapsed`, ask the kernel for a decay factor, and scale activation lazily. Zero background workers, zero drift.

**Tweet 4/10**
Spaced repetition matters. Cepeda et al. mapped the "temporal ridgeline" of optimal review intervals. Our decay defaults learn from that ridgeline so Engram schedules refreshes before memories slip away. (Cepeda et al., 2008)

**Tweet 5/10**
Bjork & Bjork split memory into storage strength vs retrieval strength. We mirror that: decay touches retrieval strength; storage strength tracks how fast the next recall rebounds. (Bjork & Bjork, 1992)

**Tweet 6/10**
Implementation snippet:
```rust
let elapsed = now - episode.last_access;
let factor = kernel.factor(elapsed, &episode.decay);
let decayed = episode.activation * factor;
```
Panic-safe, SIMD friendly, and always clamped to `[0.0, 1.0]`.

**Tweet 7/10**
Temporal dynamics feed storage tiers. Hot tier keeps high retrieval strength items nearby; cold tier holds low-urgency episodes. Migration now depends on actual decay, not arbitrary timers.

**Tweet 8/10**
Validation plan: regress Engram against historical forgetting curves, run `proptest` to ensure activation never increases with time, and finish with valgrind marathons to prove zero leaks during lazy decay.

**Tweet 9/10**
Instrumentation hooks land with this milestone: `decay_compute_time`, `decay_error_percent`, `decay_kernel_usage`. Observability keeps us inside the ±5% error budget.

**Tweet 10/10**
Temporal dynamics make Engram feel alive. Memories fade unless you revisit them, spacing keeps them sharp, and every recall is deterministic. That's cognitive realism, delivered in Rust.
