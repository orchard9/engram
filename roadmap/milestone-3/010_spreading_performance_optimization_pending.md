# Task 010: Spreading Performance Optimization

## Objective
Hit the <10 ms P95 latency target by tightening allocation, cache behavior, batching, and latency prediction in the spreading engine.

## Priority
P1 (Performance Critical)

## Effort Estimate
1.5 days

## Dependencies
- Task 008: Integrated Recall Implementation

## Technical Approach

### Current Baseline
- `ActivationMemoryPool` (`engram-core/src/activation/memory_pool.rs`) exposes arena allocation but is not wired into the parallel engine; `process_task` still allocates vectors per hop.
- `BreadthFirstTraversal` and `parallel.rs` rely on `DashMap`/`VecDeque` without cache-aligned node representations.
- `LatencyBudgetManager` (`activation/latency_budget.rs`) provides tier budgets but no predictive tuning.

### Implementation Details
1. **Lock-Free Activation Pool**
   - Add `ActivationRecordPool` wrapping `crossbeam_epoch::Stack<NonNull<ActivationRecord>>` with per-thread caches (`thread_local!`).
   - Replace `ActivationRecord::new` allocations in `parallel.rs::process_task` with `ActivationRecordPool::acquire`. On recycle, call `record.reset()` and push back into the stack.
   - Expose metrics: `activation_pool_hit_rate`, `pool_high_water_mark` via `ActivationMetrics`.

2. **Cache-Optimized Node Layout**
   - Introduce `#[repr(C, align(64))] struct CacheOptimizedNode` storing hot fields (ID, activation, confidence, tier) in first cache line; move metadata into a companion struct. Update `MemoryGraph::get_neighbors` to hand out references to the hot struct for spreading loops.
   - In `parallel.rs`, prefetch upcoming nodes using `_mm_prefetch` with distance from `ParallelSpreadingConfig::prefetch_distance`.

3. **Adaptive Batching**
   - Implement `AdaptiveBatcher` that consumes CPU topology (via `std::thread::available_parallelism`, optional `numa` crate) and historical metrics. Compute recommended batch size using geometric mean of cache capacity, memory bandwidth, and logical cores.
   - Store the batch size in `ActivationMetrics::parallel_efficiency` and feed into `ParallelSpreadingConfig::batch_size` at runtime.

4. **Latency Prediction Loop**
   - Create `LatencyPredictor` that records `(batch_size, hop_count, tier_mix, observed)` tuples and fits a simple linear model. Integrate with `LatencyBudgetManager::within_budget` before launching another hop; if predicted latency would exceed budget, truncate spreading and flag partial results.

5. **Metrics + Observability**
   - Publish new Prometheus metrics (`engram_spreading_latency_prediction_error`, `engram_spreading_cache_miss_rate`, `engram_spreading_pool_utilization`). Gather cache miss rate by sampling hardware counters via `metrics::hardware::HardwareMetrics::last_cache_stats()`.

### Acceptance Criteria
- [ ] `ActivationRecordPool` reduces allocator calls by ≥50 % (measure `activation_pool_hit_rate`)
- [ ] Cache-aligned nodes improve L2 miss rate to <5 % on synthetic workload (perf counter test)
- [ ] Adaptive batching converges within 3 iterations and stabilizes `batch_size` per topology
- [ ] Latency prediction error <20 % for 95 % of requests
- [ ] Recall P95 latency <10 ms on 10 k warm-tier dataset with spreading enabled
- [ ] Metrics exported for pool utilization, cache miss rate, latency prediction error

### Testing Approach
- Benchmarks: `cargo bench --bench spreading` before/after optimization; track IPC and miss rates with `perf stat`
- Long-running soak test (100 k spreads) to ensure pool does not leak; monitor via `PoolStats::utilization`
- Unit tests for `AdaptiveBatcher::compute_optimal_batch_size` and `LatencyPredictor::predict_latency`

## Risk Mitigation
- **Pool fragmentation** → add periodic `ActivationMemoryPool::reset` on engine idle and log high-water marks
- **Oscillating batch size** → apply EWMA damping factor 0.5 and minimum update interval of 500 spreads
- **Prediction mistakes** → fallback to similarity recall when predicted latency exceeds budget by >50 %

## Notes
Relevant modules:
- Memory pool primitives (`engram-core/src/activation/memory_pool.rs`)
- Parallel engine (`engram-core/src/activation/parallel.rs`)
- Latency budgets (`engram-core/src/activation/latency_budget.rs`)
