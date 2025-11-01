# Task 010: Performance Benchmarking and Tuning

**Status:** Pending
**Estimated Effort:** 2 days
**Dependencies:** Tasks 001-009 (full streaming pipeline + chaos validation)
**Priority:** OPTIMIZATION

## Objective

Validate 100K observations/sec target with concurrent recalls. Identify bottlenecks, tune parameters (worker count, batch size, queue capacity), establish production baselines for P50/P99/P99.9 latency.

## Deliverables

### 1. Throughput Benchmark Suite
**File:** `engram-core/benches/streaming_throughput.rs` (300 lines)

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn bench_streaming_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_throughput");

    for rate in [10_000, 50_000, 100_000, 200_000] {
        group.throughput(Throughput::Elements(rate as u64));
        group.bench_function(format!("{}_obs_per_sec", rate), |b| {
            b.iter(|| {
                // Benchmark streaming at target rate for 10 seconds
            });
        });
    }
}

fn bench_worker_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("worker_scaling");

    for workers in [1, 2, 4, 8] {
        group.bench_function(format!("{}_workers", workers), |b| {
            b.iter(|| {
                // Measure throughput with N workers
            });
        });
    }
}

fn bench_batch_size_tuning(c: &mut Criterion) {
    for batch_size in [10, 50, 100, 500, 1000] {
        group.bench_function(format!("batch_{}", batch_size), |b| {
            b.iter(|| {
                // Measure latency vs throughput tradeoff
            });
        });
    }
}

criterion_group!(benches, bench_streaming_throughput, bench_worker_scaling);
criterion_main!(benches);
```

**Benchmarks:**
1. Throughput ramp: 10K → 50K → 100K → 200K obs/sec
2. Worker scaling: 1, 2, 4, 8 workers (linear scaling expected)
3. Batch size: 10, 50, 100, 500, 1000 (sweet spot identification)
4. Queue capacity: 1K, 10K, 100K (memory vs latency tradeoff)

### 2. Concurrent Recall Benchmark
**File:** `engram-core/benches/concurrent_recall.rs` (250 lines)

```rust
#[bench]
fn bench_concurrent_streaming_and_recall() {
    // Streaming: 100K observations/sec
    // Concurrent recalls: 10 recalls/sec
    // Duration: 60 seconds
    // Measure: P50/P99 latency for both operations
}

#[bench]
fn bench_recall_under_load() {
    // Measure recall latency at different streaming loads:
    // - 10K obs/sec: baseline
    // - 50K obs/sec: moderate
    // - 100K obs/sec: target
    // - 150K obs/sec: overload
}
```

**Metrics:**
- Observation ingestion latency: P50/P99/P99.9
- Recall query latency: P50/P99/P99.9
- Worker utilization: percentage busy
- Queue depth: min/avg/max
- Memory footprint: RSS during 1M observation test

### 3. Parameter Tuning Framework
**File:** `engram-core/benches/streaming_parameter_tuning.rs` (200 lines)

```rust
struct TuningResult {
    worker_count: usize,
    batch_size: usize,
    queue_capacity: usize,
    throughput_obs_per_sec: f64,
    p99_latency_ms: f64,
    memory_mb: f64,
}

fn grid_search_optimal_params() -> TuningResult {
    // Grid search over parameter space:
    // - Worker count: 1, 2, 4, 8
    // - Batch size: 10, 50, 100, 500
    // - Queue capacity: 1K, 10K, 100K
    // Return Pareto frontier of throughput vs latency
}
```

## Acceptance Criteria

### Performance Targets

✓ **Sustained 100K observations/sec for 60s** with 4-core CPU
- Measured throughput: ≥100K obs/sec
- CPU usage: <80% (headroom for spikes)
- Memory: <2GB for 1M observations

✓ **Concurrent recalls: 10/sec with <20ms P99 latency**
- Recall latency during streaming: P99 <20ms
- No interference between observations and recalls
- Isolation via priority queues

✓ **Worker scaling efficiency**
- Linear scaling up to core count (1 → 2 → 4 workers)
- Diminishing returns beyond physical cores
- Work stealing overhead: <5% throughput penalty

✓ **Optimal batch size identified**
- Latency vs throughput tradeoff characterized
- Sweet spot expected: 100-500 (amortizes lock overhead)
- Below 10: high lock contention
- Above 1000: high tail latency

## Bottleneck Analysis

### Profiling Strategy

1. **CPU Profiling** (flamegraph)
```bash
cargo flamegraph --bench streaming_throughput -- --bench
```
Expected hotspots:
- HNSW insertion: 60-70% (inherent graph algorithm cost)
- Vector operations: 10-20% (distance calculations)
- Lock acquisition: <5% (lock-free design should minimize)

2. **Memory Profiling** (valgrind/massif)
```bash
valgrind --tool=massif ./target/release/engram-bench
```
Expected allocations:
- Episode objects: ~3KB each (768 floats + metadata)
- HNSW nodes: ~4KB each (edges + layer metadata)
- Queue overhead: <1MB (lock-free SegQueue)

3. **Cache Analysis** (perf stat)
```bash
perf stat -e cache-references,cache-misses ./target/release/engram-bench
```
Cache miss rate targets:
- L1 cache: <5% miss rate (hot path data)
- L2 cache: <10% miss rate (HNSW neighbors)
- L3 cache: <20% miss rate (cold data)

## Production Baselines

### Latency Distribution Targets

| Percentile | Target | Acceptable | Degraded |
|------------|--------|------------|----------|
| P50        | 5ms    | 10ms       | 20ms     |
| P90        | 15ms   | 30ms       | 50ms     |
| P99        | 50ms   | 100ms      | 200ms    |
| P99.9      | 100ms  | 200ms      | 500ms    |

### Throughput Targets

| Load Level | Target | CPU % | Memory | Notes |
|------------|--------|-------|--------|-------|
| Baseline   | 10K/s  | 20%   | 500MB  | Steady state |
| Normal     | 50K/s  | 50%   | 1GB    | Typical workload |
| Peak       | 100K/s | 75%   | 2GB    | Target capacity |
| Overload   | 150K/s | 90%   | 3GB    | Admission control kicks in |

## Tuning Results (Expected)

### Optimal Configuration (4-core CPU)

```rust
WorkerPoolConfig {
    num_workers: 4,               // Match core count
    queue_config: QueueConfig {
        high_capacity: 1000,      // Rare immediate operations
        normal_capacity: 50_000,  // 0.5s buffer at 100K/sec
        low_capacity: 10_000,     // Background tasks
    },
    min_batch_size: 10,           // Low latency under light load
    max_batch_size: 500,          // High throughput under heavy load
    steal_threshold: 1000,        // Steal when victim has 10s of work
    idle_sleep_ms: 1,             // Responsive to bursts
}
```

### Parameter Sensitivity Analysis

**Worker Count:**
- 1 worker: 25K obs/sec (baseline)
- 2 workers: 48K obs/sec (1.9x, near-linear)
- 4 workers: 92K obs/sec (3.7x, linear scaling)
- 8 workers: 105K obs/sec (4.2x, diminishing returns on 4-core)

**Batch Size:**
- 10: 60K obs/sec, P99 latency 20ms
- 100: 95K obs/sec, P99 latency 50ms (optimal balance)
- 500: 110K obs/sec, P99 latency 150ms
- 1000: 115K obs/sec, P99 latency 300ms (too high latency)

**Queue Capacity:**
- 1K: frequent admission control activations
- 10K: rare activations, good for steady load
- 50K: 0.5s buffer, handles bursts well
- 100K: 1s buffer, excessive memory use

## Research Foundation

- Amdahl's Law: Worker scaling limits
- Little's Law: Queue depth = throughput × latency
- Batch processing: Amortization of fixed costs (Batcher et al.)
- Lock-free data structures: Michael & Scott (1996)

## Files to Create

- `engram-core/benches/streaming_throughput.rs` (300 lines)
- `engram-core/benches/concurrent_recall.rs` (250 lines)
- `engram-core/benches/streaming_parameter_tuning.rs` (200 lines)
- `docs/operations/performance-tuning.md` (200 lines) - Tuning guide

## Testing Approach

1. **Baseline**: Measure single-worker, small-batch performance
2. **Scaling**: Vary worker count, measure linear scaling
3. **Batching**: Vary batch size, find optimal latency/throughput point
4. **Memory**: Run 1M observation test, measure RSS growth
5. **Concurrent**: Add recall load, measure interference

## Dependencies

- Task 003: Worker pool (tuning target)
- Task 004: Batch HNSW insertion (batch size tuning)
- Task 009: Chaos testing (stress conditions)

## Next Steps

After benchmarking:
- Task 011: Add Prometheus metrics based on identified bottlenecks
- Production: Deploy with optimal parameters from tuning

## Notes

Performance benchmarking uses Criterion for statistical rigor. All benchmarks run
for sufficient iterations to achieve <5% variance. Results establish production
baselines and guide operational decisions (when to scale, when to tune).

The parameter tuning framework enables continuous optimization as workloads evolve.
Recommended: re-run quarterly or after major code changes.
