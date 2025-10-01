# Integrated Recall: From Vector Matches to Cognitive Retrieval

*Perspective: Systems Architecture*

Engram's recall used to be a pure similarity search. Given a cue, it returned the most similar embeddings. Accurate? Sometimes. Context-aware? Rarely. Task 008 transforms recall into a cognitive pipeline that mirrors how humans remember: quick familiarity check, then recollection that reconstructs context. The payoff is higher-quality results and a foundation for explainable retrieval.

## Step 1: Familiarity via Vector Seeding
We start with the vector seeder introduced in Task 002. Given a cue embedding, it retrieves top-K candidate memories per tier using ANN search. This familiar stage is deterministic and fast (~1 ms). For each candidate, we record similarity scores, tiers, and metadata required for spreading.

## Step 2: Recollection through Spreading Activation
Candidates become seeds for the spreading engine. Tier-aware scheduling (Task 003) iteratively expands activation within the time budget. Cycle protection (Task 005) ensures termination, while SIMD batch kernels (Task 007) accelerate activation propagation. Each hop produces path evidence: activation contributions, tier transitions, cycle penalties, and hop counts.

```rust
let seeds = vector_seeder.seed_from_cue(cue, store).await?;
let results = scheduler.spread_with_budget(
    seeds,
    &cycle_detector,
    &budget_tracker,
    DeterministicMode::from(seed_option),
).await?;
```

The budget tracker enforces the 10 ms P95 target. If time runs out, we freeze the current activation state and move to ranking.

## Step 3: Confidence Aggregation
The aggregator consolidates path evidence by summing activation mass with Kahan compensation and blending multiple signals. Confidence is computed as:

```
confidence = sigmoid( w1 * activation + w2 * similarity + w3 * recency + w4 * tier_weight )
```

Weights come from Milestone 2 calibration datasets. We also capture provenance metadata (`contributing_paths`, `cycle_penalties`) for explainability.

## Step 4: Ranking and Hybrid Fallback
`SpreadingResultRanker` sorts results primarily by activation, then confidence, then similarity. We add a small recency boost (`exp(-age / tau)`, tau = 30 days). If the pipeline encounters errors—or the time budget expires before spreading completes—we return similarity-only results while logging the incident.

```rust
match ranking.rank(results, max_results) {
    Ok(ranked) => Ok(ranked),
    Err(err) => {
        tracing::warn!(?err, "recall_spreading_failed, falling back");
        self.recall_similarity(cue).await
    }
}
```

## Observability
Integrated recall exports metrics under `metrics::recall`:
- `recall_latency_ms` (P50/P95)
- `recall_mode_counts` (similarity, spreading, fallback)
- `recall_activation_mass`
- `recall_confidence_avg`

We also emit structured logs with top-K memory IDs, tiers, and confidence for audit trails. Deterministic mode (Task 006) lets engineers replay the entire pipeline when debugging.

## Rollout Plan
The new recall path ships behind `RecallMode::Spreading` with Hybrid fallback. Operators enable it per tenant, compare quality metrics, then graduate to full spreading once latency and accuracy targets are met. Feature flags and migration docs ensure rollback remains one configuration change away.

## References
- Anderson, J. R. "A spreading activation theory of memory." *Journal of Verbal Learning and Verbal Behavior* (1983).
- Tulving, E. *Elements of Episodic Memory.* (1983).
- Yonelinas, A. P. "The nature of recollection and familiarity: A review." *Journal of Memory and Language* (2002).
