---
title: Cognitive Spreading Explanation
outline: deep
---

# Cognitive Spreading

Spreading activation mirrors associative recall in human cognition. Engram’s implementation combines semantic priming, episodic reconstruction, and deterministic traces to make the behaviour observable.

## Semantic Priming

- **Mechanism:** Activation spreads along weighted edges, favouring strong semantic links (doctor → nurse).

- **Implementation:** `HnswActivationEngine::spread_activation` applies `similarity_threshold` and `distance_decay` to seed the graph.

- **Tuning:** Lower thresholds increase recall breadth but also latency; see the [Performance Guide](../howto/spreading_performance.md).

## Episodic Reconstruction

Partial cues trigger reconstruction when activation crosses `RecallConfig::min_confidence`.

1. Cue enters the graph via vector similarity seeding.

2. Activation traverses episodic edges (e.g., event timelines).

3. Reconstructed episodes surface through `RankedMemory::rank_score` combining activation, confidence, similarity, and recency boosts.

## Confidence and Decay

- `SpreadingMetrics::parallel_efficiency` captures how quickly activation dissipates.

- Cognitive decay functions (Task 011) map to `ParallelSpreadingConfig::decay_function`, letting teams mimic human forgetting curves.

## Deterministic Debugging

When `ParallelSpreadingConfig::deterministic(true)` is enabled with a seed:

- Work-stealing becomes deterministic (phase-synchronised batches).

- `TraceEntry` records each hop (depth, source, target, activation, confidence).

- The new visualizer turns traces into GraphViz diagrams, colouring node activation and edge confidence.

![Spreading example](../assets/spreading_example.png)

> The legend is stored in `docs/assets/spreading_legend.svg` and ensures accessibility compliance.

## Integration With Monitoring

Task 012 instrumentation extends the cognitive view into Prometheus:

- **Activation mass:** `RecallMetrics::recall_activation_mass`

- **Fallback rate:** `RecallMetrics::fallback_rate`

- **Cycle detection:** `engram_spreading_breaker_transitions_total`

Combine metrics with traces to validate that spreading matches psychological expectations such as the fan effect (activation dilution across many links).

## Further Reading

- Newell, *Unified Theories of Cognition* (1990)

- Kanerva, *Sparse Distributed Memory* (1988)

- Kostecki, “The Diátaxis documentation framework” (2020)
