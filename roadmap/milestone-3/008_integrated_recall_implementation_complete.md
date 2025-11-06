# Task 008: Integrated Recall Implementation

## Objective
Wire the `MemoryStore::recall` pipeline to drive spreading activation, confidence aggregation, and ranking, turning milestone components into a production-ready `recall()` API.

## Priority
P0 (Critical Path)

## Effort Estimate
2 days

## Dependencies
- Task 002: Vector-Similarity Activation Seeding
- Task 003: Tier-Aware Spreading Scheduler
- Task 004: Confidence Aggregation Engine
- Task 005: Cyclic Graph Protection

## Technical Approach

### Current Baseline
- `engram-core/src/query/integration.rs` implements probabilistic recall (`recall_probabilistic`) but still routes through similarity-only evidence.
- `activation/seeding.rs` provides `VectorSeeder` APIs used by Task 002, returning `SeededActivation` batches.
- `activation/parallel.rs` and `activation/accumulator.rs` compute activation but are invoked manually in tests only.

### Implementation Plan
1. **CognitiveRecall Facade**
   - Create `activation/recall.rs` with `pub struct CognitiveRecall` holding references to `VectorSeeder`, `ParallelSpreadingEngine`, `ConfidenceAggregator`, and `CycleDetector`.
   - Expose `impl CognitiveRecall { pub async fn recall(&self, cue: &Cue, store: &MemoryStore) -> ActivationResult<Vec<RankedMemory>> }` calling each subsystem in sequence.

2. **MemoryStore Integration**
   - Extend `MemoryStore::recall` (`query/integration.rs`) to dispatch on `RecallMode`:
     ```rust
     match self.config.recall_mode {
         RecallMode::Similarity => self.recall_similarity(cue).await,
         RecallMode::Spreading => self.cognitive_recall.recall(cue, self).await,
         RecallMode::Hybrid => self.recall_hybrid(cue).await,
     }
     ```
   - Populate `self.cognitive_recall` during store initialization; reuse existing builder pattern that currently injects `VectorOps`.

3. **Spreading Execution**
   - Use `VectorSeeder::seed_from_cue` (from `activation/seeding.rs`) to map cues into initial activations. Convert seeds into the `ActivationTask` format consumed by `ParallelSpreadingEngine::spread` (exposed via `activation/parallel.rs`).
   - Pass `CycleDetector` from Task 005 to enforce tier-aware hop budgets, and reuse `LatencyBudgetManager` (`activation/latency_budget.rs`) for time budgeting.

4. **Confidence Aggregation & Ranking**
   - Pipe `ActivationResult` into `ConfidenceAggregator::aggregate_paths` (Task 004, `activation/accumulator.rs`). Combine similarities from seeding (`SeededActivation::similarity`) with activation mass and recency (call `Episode::recency_boost()` if available).
   - Implement `ResultRanker` that sorts by `activation`, then `confidence`, then `similarity`. Add recency boost using metadata already stored on `Episode` (`episode.when`).

5. **Fallback & Error Handling**
   - On `ActivationError`, log via `tracing::error!` and fall back to `recall_similarity`. Mark fallback in metrics (`ActivationMetrics::fallbacks_total`).
   - Honor `SpreadingConfig::time_budget`: if `LatencyBudgetManager::within_budget` fails, return partial results collected so far.

6. **API Docs & Examples**
   - Document `MemoryStore::recall` with cognitive context and add example in new file `examples/cognitive_recall_patterns.rs` (references Task 013).

### Acceptance Criteria
- [ ] `MemoryStore::recall` calls spreading pipeline when `RecallMode::Spreading | Hybrid`
- [ ] `CognitiveRecall::recall` orchestrates seeding → spreading → aggregation → ranking using existing modules
- [ ] Results include activation, confidence, similarity, recency metadata for each episode
- [ ] Hybrid mode returns similarity fallback when spreading exceeds time budget or fails
- [ ] Integration tests cover similarity vs spreading outputs (`tests/recall_integration.rs`)
- [ ] P95 latency <10 ms on warm-tier dataset of 10 k episodes when spreading enabled

### Testing Approach
- Deterministic snapshots using Task 006 to validate ranking stability
- A/B tests comparing `RecallMode::Similarity` and `RecallMode::Spreading` recall quality on fixture datasets
- Error injection test forcing `CycleDetector` to reject nodes and verifying fallback path

## Risk Mitigation
- **Performance regressions** → honor `time_budget` and keep hybrid fallback default in production
- **Quality regressions** → track metrics `recall_mode_counts`, `recall_activation_mass`, and run acceptance suite from Task 011 nightly
- **API churn** → guard new pipeline behind configuration flags and document migration steps

## Notes
Relevant modules:
- `VectorSeeder` (`engram-core/src/activation/seeding.rs`)
- `ParallelSpreadingEngine` (`engram-core/src/activation/parallel.rs`)
- `ConfidenceAggregator` (`engram-core/src/activation/accumulator.rs`)
- `MemoryStore` integration (`engram-core/src/query/integration.rs`)
