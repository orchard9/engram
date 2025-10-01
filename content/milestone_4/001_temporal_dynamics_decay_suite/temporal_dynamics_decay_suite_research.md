# Temporal Dynamics Decay Suite Research

## Overview
Milestone 4 pushes Engram beyond static activation by introducing temporal decay that matches established forgetting curves. Building directly on the foundations from Milestones 1–3—typed confidence-bearing memories, vector-native storage, and activation spreading—we now extend the recall stack so every activation budget also respects time. The goal is to apply decay lazily at recall using each memory's `last_access` metadata while supporting selectable decay functions (exponential, power-law, spaced repetition). This research digest summarizes cognitive science, computational modeling, and systems engineering findings that inform the milestone's architecture and validation strategy.

## Continuity with Milestones 1–3
- **Milestone 1 (Core Memory Types)**: Confidence-wrapped memories and cues established the data contracts that temporal decay will modulate, ensuring that decayed activation still reports probabilistic confidence rather than raw scores.
- **Milestone 2 (Vector-Native Storage)**: SIMD-friendly embedding storage gives us the throughput needed for lazy decay evaluation; decay kernels must integrate with the same vector math paths introduced for similarity search and HNSW seeding.
- **Milestone 3 (Activation Spreading Engine)**: Parallel spreading introduced deterministic hop budgets and latency managers. Temporal dynamics slot into this pipeline by scaling activation before hops propagate, letting the existing scheduler balance spatial and temporal attenuation.

Together these milestones provide the substrate for Milestone 4: we already know how to store, seed, and propagate activation—now we teach those pathways to respect time-dependent weakening while preserving reproducibility.

## Core Questions
- What empirical forgetting curves should Engram reproduce to stay within the ±5% error margin mandated in `milestones.md`?
- How can configurable decay functions reflect both exponential and power-law dynamics observed in human memory research?
- Which scheduling heuristics from spaced repetition literature translate into Engram's lazy recall path without background workers?
- What data structures and numerical practices protect against underflow, drift, or leaks during long-running decay calculations?

## Cognitive Foundations of Forgetting
- **Ebbinghaus Forgetting Curve**: The original exponential-like forgetting curve decays rapidly within the first hours before flattening, establishing baseline parameters for `DecayFunction::exponential` (Ebbinghaus, 1885).
- **Power-Law vs Exponential Debate**: Large-scale retention studies show that power-law functions fit longer-term forgetting better than pure exponentials, motivating a configurable `DecayKernel` abstraction (Wixted & Ebbesen, 1991).
- **Spacing Effect and Retrieval Strength**: Distributed practice extends retention; Engram should expose decay parameters tied to review intervals to match the "temporal ridgeline" surfaces reported in meta-analyses (Cepeda et al., 2008).
- **Storage Strength vs Retrieval Strength**: The new theory of disuse distinguishes latent storage strength from immediate retrievability, implying Engram must track both intrinsic decay rate and activation mass (Bjork & Bjork, 1992).
- **Rational Analysis of Memory**: Memory access follows environmental statistics; Engram's default decay constants should align with observed access patterns to stay ecologically valid (Anderson & Schooler, 1991).

## Computational Models and Scheduling
- **Adaptive Scheduling**: ACT-R based optimizations compute optimal review spacing from success history, informing Engram's ability to parameterize decay per episode using power-law updating (Pavlik & Anderson, 2008).
- **Lazy Decay Evaluation**: Instead of decrementing activation continuously, Engram will compute decay on-demand as `activation * kernel(elapsed)` using vectorized math introduced in Milestone 2 to avoid drift and to keep CPU cost proportional to query load.
- **Per-Memory Configurations**: Decay kernels should accept memory-specific parameters (`decay_rate`, `strength`), enabling domain-specific forgetting curves without branching logic in the hot path.
- **Numerical Stability**: Implement decay calculations in log-space when activation spans orders of magnitude; clamp results to `[0.0, 1.0]` and reserve epsilon floors to prevent denormal slowdowns.

## Systems & Storage Considerations
- **Tier-Aware Decay**: Hot-tier episodes can recompute decay inline, while warm/cold tiers may cache decay coefficients to avoid recomputing expensive power terms—yet caches must invalidate on access to honor "last_access" semantics established in Milestone 2's tiering work.
- **Metadata Consistency**: `last_access` updates must be monotonic even under concurrent recalls; use atomic compare-and-swap or monotonic clocks to prevent backward jumps.
- **Lazy Migration Hooks**: Decay outcomes feed storage migration policies (promote/demote tiers) and activation thresholds; thresholds should be configurable per tier to avoid cache thrash.
- **Leak Prevention**: Long-running decay computations risk accumulating temporary allocations. Adopt scoped buffers or stack allocations and verify with `valgrind` runs as required by milestone validation notes.

## Validation Strategies
- **Curve Fit Benchmarks**: Generate synthetic recall logs and verify that Engram's computed decay stays within ±5% of empirical curves from Ebbinghaus and spacing literature across sampled intervals.
- **Property Tests**: Ensure decay never increases activation over time; use `proptest` to vary `elapsed` durations, decay modes, and parameter ranges.
- **Regression Seeds**: Capture boundary cases (e.g., near-zero elapsed, very long elapsed, mixed decay modes) in `engram-core/proptest-regressions/` for repeatability.
- **Leak Checks**: Run `valgrind` (or `cargo valgrind` wrapper) over long-lived recall loops to guarantee no allocation leaks accumulate during lazy decay.

## Citations
- Anderson, J. R., & Schooler, L. J. (1991). *Reflections of the environment in memory*. Psychological Science, 2(6), 396-408.
- Bjork, R. A., & Bjork, E. L. (1992). *A new theory of disuse and an old theory of stimulus fluctuation*. In A. Healy et al. (Eds.), *From Learning Processes to Cognitive Processes*.
- Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2008). *Spacing effects in learning: A temporal ridgeline of optimal intervals*. Psychological Science, 19(11), 1095-1102.
- Ebbinghaus, H. (1885). *Memory: A Contribution to Experimental Psychology*. Leipzig: Duncker & Humblot.
- Pavlik, P. I., & Anderson, J. R. (2008). *How to optimize distributed practice: Learner, task, and schedule*. Journal of Experimental Psychology: Applied, 14(2), 101-117.
- Wixted, J. T., & Ebbesen, E. B. (1991). *On the form of forgetting*. Psychological Science, 2(6), 409-415.
