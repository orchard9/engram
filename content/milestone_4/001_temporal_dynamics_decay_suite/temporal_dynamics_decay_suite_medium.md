# Teaching Engram to Forget Gracefully

Milestone 4 asks us to give Engram something every human memory system already has: a sense of time. Instead of treating memories as static records, we want them to ebb and fade based on how often we revisit them and how long they have been dormant. Done right, Engram will reproduce classic forgetting curves within ±5% error while keeping recall deterministic and lazily evaluated. Done poorly, we would either cling to stale memories forever or watch valuable context disappear overnight. This article walks through the psychology that inspired the milestone, the engineering patterns we adopted, and the validation work we will need before calling temporal dynamics complete.

## Why Forgetting Matters

Hermann Ebbinghaus' meticulous self-experiments produced the curve that still anchors every forgetting conversation today: steep decline in the first hours, then a gradually flattening tail (Ebbinghaus, 1885). John Wixted and Ebbe Ebbesen later showed that long-term retention is better modeled by power-law curves, especially when observations span days or weeks (Wixted & Ebbesen, 1991). These findings tell us two things:

1. Engram must be able to reproduce both exponential and power-law decay.
2. Forgetting is not a bug—it is how memory prioritizes relevance under resource limits.

But forgetting is not the whole story. Research on the spacing effect shows that revisiting information at expanding intervals dramatically improves retention (Cepeda et al., 2008). Robert and Elizabeth Bjork's new theory of disuse reframes this as the interplay between *retrieval strength* (how easy it is to recall now) and *storage strength* (how well the memory is encoded) (Bjork & Bjork, 1992). A memory can be hard to retrieve yet still have high storage strength if it was deeply learned; one successful recall can renew its retrieval strength. This dual lens gives us a blueprint for Engram's data model: we should record both a decay rate and the most recent activation mass so we can separate long-term value from immediate accessibility.

John Anderson and Lynne Schooler added a pragmatic constraint. Their rational analysis of memory argues that our brains tune retention to match the statistics of our environment (Anderson & Schooler, 1991). If a fact is encountered frequently, the brain keeps it handy; otherwise it decays. Engram should do the same, learning decay parameters from observed access patterns rather than locking users into hand-tuned constants.

## Translating Research into Engram's Design

Milestone 4 implements these insights through three key moves.

### 1. Configurable Decay Kernels

We introduce a `DecayKernel` trait that encapsulates decay functions:

```rust
pub trait DecayKernel {
    fn factor(&self, elapsed: Duration, params: &DecayParams) -> f32;
}
```

Two built-in kernels ship with the milestone:

- `ExponentialKernel`: `exp(-lambda * elapsed)` for situations where short-term drop-off matters most.
- `PowerLawKernel`: `(1.0 + elapsed / sigma).powf(-alpha)` for longer horizons inspired by Wixted & Ebbesen (1991).

Each memory stores both the kernel type and its parameters (`lambda`, `alpha`, `sigma`). Rehearsal events—successful recalls—update these parameters using ACT-R style rules to reflect the spacing effect (Pavlik & Anderson, 2008). Because kernels are pure functions, we can evaluate them lazily without background threads chewing CPU cycles.

### 2. Lazy Evaluation in the Recall Pipeline

Decay is applied when we actually try to recall something. The recall pipeline will:

1. Fetch the episode metadata including `last_access`, `decay_params`, and `activation_mass`.
2. Compute `elapsed = now - last_access` using monotonic clocks to avoid skew.
3. Ask the kernel for a decay factor and multiply it by the stored activation mass.
4. Continue spreading activation if the decayed value stays above threshold.

This approach keeps Engram deterministic: given the same timestamps, every node reproduces the same activation. It also sidesteps the performance tax of ticking decay for millions of memories that no one is asking about.

### 3. Feedback into Tiered Storage

Decay outputs feed the promotion and demotion rules in Engram's three-tier storage model. Episodes whose decayed activation falls below a tier's threshold migrate toward cold storage, freeing the hot tier for fresher knowledge. Because migrations now depend on temporal dynamics, tier controllers need the same kernel metadata to avoid jitter or thrashing.

## Instrumentation and Validation

Hitting the ±5% target requires disciplined measurement.

- **Curve Fitting**: We will generate synthetic recall logs that follow published forgetting curves and verify Engram's decay stays within tolerance across 1 minute to 30 day horizons.
- **Property-Based Testing**: `proptest` suites will vary decay modes, parameters, and elapsed times to guarantee activation never increases over time and never becomes NaN.
- **Long-Run Leak Detection**: Continuous recall simulations under `valgrind` watch for allocations or reference cycles that could surface during months of uptime.
- **Observability**: New metrics—`decay_compute_time`, `decay_factor_avg`, `decay_error_percent`—reveal whether kernels diverge or numerical drift appears in production.

These checks map directly to Milestone 4's validation clause: match published psychology within ±5% and prove no leaks during long-running decay.

## Developer Experience

Temporal dynamics should not burden application developers. We will expose intuitive configuration APIs:

```rust
store.set_decay_defaults(DecayDefaults::new()
    .with_kernel(DecayKernelKind::PowerLaw)
    .with_alpha(0.9)
    .with_sigma(Duration::from_hours(12)));
```

Individual memories can override the defaults:

```rust
let episode = EpisodeBuilder::new()
    .with_decay_kernel(DecayKernelKind::Exponential)
    .with_lambda(0.35)
    .commit();
```

Documentation will link decay presets to real-world scenarios—language flashcards vs system log retention—so teams can make informed choices without reading academic papers.

## Looking Ahead

Temporal dynamics unlock downstream milestones. Accurate decay lets us schedule consolidation intelligently, generate realistic priming effects, and keep the activation spreading engine grounded in cognitive science instead of arbitrary heuristics. By combining classic psychology (Ebbinghaus, 1885; Cepeda et al., 2008) with modern learning models (Bjork & Bjork, 1992; Pavlik & Anderson, 2008), Engram gains a memory of its own history.

Milestone 4 is our promise that Engram will forget gracefully—never so quickly that knowledge evaporates, never so slowly that stale memories clog the graph. With configurable decay kernels, lazy evaluation, and rigorous validation, we are ready to teach Engram the art of letting go.
