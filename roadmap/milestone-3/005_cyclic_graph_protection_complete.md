# Task 005: Cyclic Graph Protection

## Objective
Implement cycle detection and prevention so spreading terminates in bounded time without infinite loops while preserving biologically-inspired activation dynamics.

## Priority
P0 (Critical Path)

## Effort Estimate
1.5 days

## Dependencies
- Task 003: Tier-Aware Spreading Scheduler

## Technical Approach

### Current Baseline
- `engram-core/src/activation/cycle_detector.rs` already ships a `CycleDetector` backed by `DashSet<NodeId>` plus a thread-local `LocalCycleDetector`. Today it only gates visit-count checks.
- `engram-core/src/activation/traversal.rs` (`BreadthFirstTraversal::traverse`) keeps a per-node `AtomicUsize` visit counter but does not enforce tier-aware limits or confidence penalties.
- `ParallelSpreadingConfig` in `engram-core/src/activation/mod.rs` exposes `cycle_detection: bool` yet defaults to a fixed max depth without tier-specific hop budgets.

### Implementation Details
1. **Tier-Aware Visit Budgets**
   - Extend `CycleDetector::new` to accept a `HashMap<StorageTier, usize>` and expose `CycleDetector::max_visits_for_tier`. Use the existing `StorageTier::from_depth` mapping in `activation/parallel.rs` when calculating hop depth.
   - Update `BreadthFirstTraversal::traverse` and the parallel worker in `activation/parallel.rs::process_task` to call `cycle_detector.should_visit(&node_id, visit_count, tier)` before queueing neighbors. Reuse the existing visit count in `DashMap<NodeId, AtomicUsize>` to avoid duplicate atomics.

2. **Confidence Penalty Pipeline**
   - Augment `ActivationRecord` (`activation/accumulator.rs`) with `fn apply_cycle_penalty(&self, factor: f32)` that scales both `activation` and `confidence` atomics. Invoke it from `process_task` whenever `CycleDetector::should_visit` returns false.
   - Surface penalties via `ActivationMetrics`: increment `cycles_detected` (`activation/mod.rs::ActivationMetrics::cycles_detected`) and record per-tier counts for Prometheus integration in Task 012.

3. **Cycle Evidence Export**
   - Track cycle paths in `CycleDetector::cycle_nodes` (already an `Arc<DashSet<NodeId>>`). Extend `ActivationResult` (`activation/mod.rs::ActivationResultData`) with `cycle_paths: Vec<Vec<NodeId>>` so integrated recall (Task 008) can annotate low-confidence paths.

4. **Debug Instrumentation**
   - When `ParallelSpreadingConfig::trace_activation_flow` is enabled, dump detected cycles into `tracing::warn!` with node IDs and tier metadata to drive the visualization tooling in Task 013.

### Acceptance Criteria
- [x] `CycleDetector` supports tier-specific hop budgets and enforces them inside both sequential (`BreadthFirstTraversal`) and parallel spreading
- [x] `ActivationRecord::apply_cycle_penalty` reduces activation/confidence and increments `ActivationMetrics::cycles_detected`
- [x] `ActivationResult` carries `cycle_paths` for downstream recall heuristics
- [ ] Tier-aware cycle budgets reduce per-hop latency variance to <100 ms worst case in pathological graphs
- [x] Unit tests cover hop-budget enforcement and confidence penalties (`activation/cycle_detector.rs`, `activation/parallel.rs`)
- [ ] Benchmark regression: cycle detection adds <2 % overhead compared to baseline (measure with `cargo bench --bench spreading`)

### Testing Approach
- Extend `activation/cycle_detector.rs` tests to cover tier budgets and penalty application
- Add deterministic integration test in `engram-core/tests/spreading_validation.rs` that exercises warm-tier cycles with deterministic mode enabled
- Benchmark pathological graphs using 1 000-node strongly connected component and verify bounded termination

## Risk Mitigation
- **Bottleneck in DashSet lookups** → amortize with thread-local `LocalCycleDetector` cache and only promote to shared state on high fan-out nodes
- **Over-aggressive penalties** → expose configuration (`CyclePenaltyConfig`) so QA can tune per tier; default to 7 % first revisit, +2 % per revisit, capped at 35 %
- **Metrics flood Prometheus** → aggregate per-tier counters (`cycle_detected_total{tier}`) instead of per-node labels

## Notes
This task hardens the activation engine’s stability. Reference code:
- `CycleDetector::should_visit` and `LocalCycleDetector::visit` (`engram-core/src/activation/cycle_detector.rs`)
- `BreadthFirstTraversal::traverse` (`engram-core/src/activation/traversal.rs`)
- `ParallelSpreadingConfig::cycle_detection` and `ActivationMetrics::cycles_detected` (`engram-core/src/activation/mod.rs`)

## Implementation Summary
- Introduced tier-aware visit budgets in `CycleDetector` with per-tier DashMap accounting and applied them in both sequential and parallel paths.
- Added activation/confidence penalty handling along with tracing and per-tier metrics counters when cycles are detected.
- Extended spreading results to expose `cycle_paths` and ensured new regression tests exercise tier budgets and penalties.

## Verification
- `cargo fmt`
- `cargo check -p engram-core`
- `cargo test -p engram-core cycle_detection_penalises_revisits` (warnings only)
