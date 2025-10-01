# Temporal Dynamics Decay Suite Perspectives

## Cognitive-Architecture Perspective
- Model Engram's decay functions after empirically grounded forgetting curves so that recall aligns with human retrievability curves within ±5% as required by Milestone 4 (Ebbinghaus, 1885; Wixted & Ebbesen, 1991).
- Represent storage strength vs retrieval strength separately to avoid collapsing long-term consolidation effects into short-lived activation spikes (Bjork & Bjork, 1992).
- Provide narrative hooks for spaced repetition cues so UX surfaces can explain why a memory resurfaces now, tying schedule recommendations to rational analysis principles (Anderson & Schooler, 1991).

## Memory-Systems Perspective
- Track `last_access`, `decay_rate`, and `recall_successes` per episode to allow ACT-R style adaptive scheduling without global background jobs (Pavlik & Anderson, 2008).
- Allow domain-specific decay kernels—semantic knowledge may use power-law decay while episodic flashbulb memories use slower exponential tails.
- Validate decay outcomes using curated datasets (e.g., language learning, autobiographical recall) to demonstrate Engram captures spacing benefits reported by Cepeda et al. (2008).

## Rust Graph Engine Perspective
- Implement `DecayKernel` as a trait with zero-cost abstraction over exponential vs power-law evaluation; expose SIMD-ready batch evaluation for recall fan-out.
- Compute decay lazily in the recall pipeline, using monotonic clocks and saturating arithmetic to guarantee non-negative activation even under numerical jitter.
- Integrate with activation spreading engine via `ActivationMass::apply_decay(elapsed, kernel)` so tier-aware pipelines can switch kernels without branching.

## Systems-Architecture Perspective
- Keep decay evaluation stateless and idempotent so it scales in distributed deployments; nodes must reproduce identical activation given the same `last_access` timestamp.
- Cache kernel coefficients per tier when beneficial, invalidating on write to avoid stale decay parameters; consider storing precomputed `decay_factor` snapshots in warm tier metadata.
- Instrument decay metrics (`decay_compute_time`, `decay_error_percent`) and feed into observability pipelines to catch drifts before they exceed validation thresholds.

## Cross-Perspective Risks & Mitigations
- **Risk**: Power-law evaluation may be numerically unstable for long spans. **Mitigation**: Evaluate in log-space with fused multiply-add instructions when available.
- **Risk**: Inconsistent clocks across nodes can skew decay. **Mitigation**: Normalize timestamps via logical clock or GPS-synchronized wall clock, and guard against backward deltas.
- **Risk**: Valgrind validation may flag allocations in hot loops. **Mitigation**: Reuse stack-allocated buffers and audit `Vec` growth in decay calculators.

## References
- Anderson, J. R., & Schooler, L. J. (1991). *Reflections of the environment in memory*. Psychological Science, 2(6), 396-408.
- Bjork, R. A., & Bjork, E. L. (1992). *A new theory of disuse and an old theory of stimulus fluctuation*.
- Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2008). *Spacing effects in learning: A temporal ridgeline of optimal intervals*. Psychological Science, 19(11), 1095-1102.
- Ebbinghaus, H. (1885). *Memory: A Contribution to Experimental Psychology*.
- Pavlik, P. I., & Anderson, J. R. (2008). *How to optimize distributed practice: Learner, task, and schedule*. Journal of Experimental Psychology: Applied, 14(2), 101-117.
- Wixted, J. T., & Ebbesen, E. B. (1991). *On the form of forgetting*. Psychological Science, 2(6), 409-415.
