# Task 006: Deterministic Spreading Execution

## Objective
Guarantee that spreading produces bit-identical results for a fixed seed, enabling reproducible debugging, validation, and scientific experiments.

## Priority
P1 (Quality Critical)

## Effort Estimate
1 day

## Dependencies
- Task 005: Cyclic Graph Protection

## Technical Approach

### Current Baseline
- `ParallelSpreadingConfig` (`engram-core/src/activation/mod.rs`) exposes `deterministic: bool`, `seed: Option<u64>`, and `phase_sync_interval` but the default execution path still relies on opportunistic work stealing.
- `activation/parallel.rs` seeds `StdRng` when `config.deterministic` is true and synchronizes workers through `context.phase_barrier.wait()` each loop iteration, yet task queues remain order-dependent.
- Unit test `activation/parallel.rs::test_deterministic_spreading` already compares two runs, providing an entry point for regression coverage.

### Implementation Details
1. **Canonical Task Ordering**
   - Before processing, sort local work queues by `(depth, target_node, contribution)` when `config.deterministic` is enabled. Use `Vec::sort_by` with stable ordering in `ActivationQueue` (`activation/queue.rs`). When pushing tasks, insert into a `BinaryHeap` or run a `sort_unstable_by` after batch pushes.
   - In `process_task`, replace `context.local_queue.push(new_task)` with `context.local_queue.insert_deterministic(new_task)` under deterministic mode to avoid interleaving differences.

2. **Seeded Work Stealing**
   - The existing `rng: Option<StdRng>` in `WorkerContext` seeds from `config.seed`. Ensure all randomness (work stealing threshold, decay noise) calls `context.rng.as_mut().unwrap()` to avoid falling back to `thread_rng`. Add `debug_assert!(context.config.deterministic => context.rng.is_some())` to catch regressions.

3. **Barrier Granularity**
   - Promote `phase_barrier` from the worker loop into hop-level synchronization: after each hop, wait on a `Barrier` sized to `num_threads`. Expose this through `PhaseBarrier` wrapper in `activation/parallel.rs` (already defined) and integrate with the scheduler so deterministic tests can control hop transitions.

4. **Trace Capture**
   - Extend `ActivationResultData` (`activation/mod.rs`) with `deterministic_trace: Vec<TraceEntry>` when deterministic mode is on. Populate inside `process_task` with `(depth, target_node.clone(), record.get_activation())`. This powers recall explanations and Task 011 snapshots.

5. **API Surface**
   - Document `ParallelSpreadingConfig::deterministic` and `seed` in Rustdoc (`activation/mod.rs`) and expose convenience constructor `ParallelSpreadingConfig::deterministic(seed: u64)`.

### Acceptance Criteria
- [ ] Deterministic runs produce identical `ActivationResult` hashes across executions and thread counts (1, 2, N)
- [ ] All randomness routed through `StdRng` seeded from `ParallelSpreadingConfig::seed`
- [ ] Work queues maintain canonical ordering when deterministic mode is enabled
- [ ] Deterministic trace data captured for downstream tooling
- [ ] Regression tests updated (`activation/parallel.rs::test_deterministic_spreading`) to cover multi-thread determinism and hash comparison
- [ ] Overhead <10 % relative to performance mode (benchmark with `cargo bench --bench spreading`)

### Testing Approach
- Extend existing deterministic test to vary `num_threads` and assert equality of serialized `ActivationResultData`
- Add property test in `tests/spreading_validation.rs` running 10 repetitions per seed
- Benchmark deterministic vs performance mode on 10 k-node graph to ensure overhead target

## Risk Mitigation
- **Sorting overhead** → only enabled when `config.deterministic` is true; default path remains unchanged
- **Seed misuse** → fallback to `seed.unwrap_or(ParallelSpreadingConfig::DEFAULT_SEED)` and log warnings if deterministic mode requested without seed
- **Trace growth** → gate on `config.trace_activation_flow` and compress entries beyond max depth

## Notes
Reference code:
- `WorkerContext` and `ParallelSpreadingEngine::process_task` (`engram-core/src/activation/parallel.rs`)
- `ActivationQueue` primitives (`engram-core/src/activation/queue.rs`)
- `ParallelSpreadingConfig` defaults and tests (`engram-core/src/activation/mod.rs`)
