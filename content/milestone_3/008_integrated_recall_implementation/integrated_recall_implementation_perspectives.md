# Integrated Recall Implementation Perspectives

## Multiple Architectural Perspectives on Task 008: Integrated Recall Implementation

### Cognitive-Architecture Perspective

Integrated recall operationalizes dual-process memory: *familiarity* via vector seeding and *recollection* via spreading activation (Yonelinas, 2002). Cue representations trigger immediate candidates; spreading reconstructs context by traversing episodic links. Confidence aggregation provides a metacognitive signal—how strongly the system "believes" the recall reflects the cue—a mechanism supported by metamemory research (Koriat, 2012).

```rust
pub struct RecallEvidence {
    pub familiarity_score: f32,
    pub recollection_activation: f32,
    pub confidence: f32,
    pub contributing_paths: SmallVec<[PathSummary; 8]>,
}
```

Each `PathSummary` captures tier transitions and cycle penalties, enabling explanation interfaces to narrate the recall.

### Systems Architecture Perspective

**Pipeline Orchestration:**
The recall API orchestrates a three-stage pipeline with time budgeting:
1. Vector seeding (fast, deterministic)
2. Spreading iterations (budget-aware)
3. Ranking + fallback

We instrument each stage with latency histograms exported to Prometheus. If spreading exceeds `time_budget`, the scheduler emits partial results and signals fallback to similarity.

### Rust Graph Engine Perspective

Integrating recall demands careful ownership and borrowing rules. `MemoryStore::recall` constructs a `RecallContext` containing references to tier caches, CPU feature flags, and cycle detection state. We use `async fn recall` with `Send` futures, so all components must be `Send + Sync`. The pipeline streams intermediate results via channels, enabling early termination on cancellation signals.

```rust
pub struct RecallContext<'a> {
    pub tiers: &'a TierCache,
    pub rng: DeterministicRng,
    pub budget: BudgetTracker,
}
```

### Verification & Testing Perspective

Create golden recall fixtures: load curated memories, issue cues, and snapshot ranked outputs. We run the recall pipeline in deterministic mode (Task 006) to ensure reproducibility. Differential tests compare similarity-only results vs. integrated recall to quantify improvements in precision/recall metrics.

### Systems Product Planner Perspective

Rollout strategy uses a feature flag `recall_mode`. Staging environments run in Hybrid mode (similarity + spreading) while logging divergence. Once metrics (accuracy uplift, latency budget) meet thresholds, production tenants enable Spreading mode. Documentation must highlight configuration migration paths and backup plan (quick rollback to similarity).

## Key Citations
- Yonelinas, A. P. "The nature of recollection and familiarity: A review." *Journal of Memory and Language* (2002).
- Koriat, A. "The subjective confidence in judgments: A metacognitive analysis." *Psychological Review* (2012).
