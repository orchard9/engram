# Integration Performance: Research and Technical Foundation

## The Challenge

Milestone 13 adds extensive cognitive functionality:
- Three priming types (Tasks 002-003)
- Three interference types (Tasks 004-005)
- Reconsolidation (Tasks 006-007)
- Psychology validation (Tasks 008-010)
- Observability (Tasks 001, 011-012)

Each individually meets performance budgets. But do they compose efficiently? This task validates that the integrated system maintains performance targets.

## Performance Targets

**Latency Budgets (from earlier milestones):**
- Encoding: < 50μs
- Retrieval: < 200μs (simple)
- Retrieval: < 2ms (complex pattern completion)
- Spreading activation: < 500μs (depth-3, fan-10)
- Consolidation: < 100μs (STM→LTM check)

**With M13 additions, budgets remain:**
- Encoding: < 75μs (+50% for interference tracking, reconsolidation check)
- Retrieval: < 300μs (+50% for priming boost, interference penalty)
- Spreading: < 600μs (+20% for priming integration)

**Throughput Targets:**
- 10K retrievals/sec (baseline)
- 8K retrievals/sec (with all M13 features enabled)
- 5K encodings/sec

## Benchmark Suite

```rust
pub struct IntegrationPerformanceBench {
    memory: EngramCore,
    workload: RealisticWorkload,
}

impl IntegrationPerformanceBench {
    pub fn bench_encoding_with_all_features(&mut self) -> BenchResult {
        let mut latencies = Vec::new();

        for trial in 0..10_000 {
            let (node, associations) = self.workload.generate_encoding();

            let start = Instant::now();

            // This triggers:
            // - Interference detection (Task 004-005)
            // - Reconsolidation boundary check (Task 006)
            // - Repetition priming trace update (Task 003)
            // - Metrics recording (Task 001)
            // - Event tracing (Task 011)
            self.memory.encode(node, associations);

            latencies.push(start.elapsed().as_nanos() as f64);
        }

        BenchResult {
            mean: mean(&latencies),
            p50: percentile(&latencies, 0.50),
            p95: percentile(&latencies, 0.95),
            p99: percentile(&latencies, 0.99),
            target: 75_000.0,  // 75μs
            pass: percentile(&latencies, 0.95) < 75_000.0,
        }
    }

    pub fn bench_retrieval_with_all_features(&mut self) -> BenchResult {
        let mut latencies = Vec::new();

        for trial in 0..10_000 {
            let cue = self.workload.generate_retrieval_cue();

            let start = Instant::now();

            // This triggers:
            // - Priming boost computation (Tasks 002-003)
            // - Interference penalty (Tasks 004-005)
            // - Fan effect calculation (Task 005)
            // - Reconsolidation window check (Task 006)
            // - Spreading activation with priming
            // - Metrics recording
            // - Event tracing
            let results = self.memory.retrieve(cue);

            latencies.push(start.elapsed().as_nanos() as f64);
        }

        BenchResult {
            mean: mean(&latencies),
            p50: percentile(&latencies, 0.50),
            p95: percentile(&latencies, 0.95),
            p99: percentile(&latencies, 0.99),
            target: 300_000.0,  // 300μs
            pass: percentile(&latencies, 0.95) < 300_000.0,
        }
    }

    pub fn bench_throughput(&mut self) -> ThroughputResult {
        let duration = Duration::from_secs(10);
        let start = Instant::now();
        let mut operations = 0;

        while start.elapsed() < duration {
            let cue = self.workload.generate_retrieval_cue();
            let _ = self.memory.retrieve(cue);
            operations += 1;
        }

        let ops_per_sec = operations as f64 / duration.as_secs_f64();

        ThroughputResult {
            ops_per_sec,
            target: 8_000.0,
            pass: ops_per_sec >= 8_000.0,
        }
    }
}
```

## Realistic Workload Generation

```rust
pub struct RealisticWorkload {
    graph_size: usize,
    avg_fan: f32,
    priming_probability: f32,
    interference_probability: f32,
}

impl RealisticWorkload {
    pub fn generate_retrieval_cue(&self) -> NodeId {
        // Generate retrieval that might trigger:
        // - Semantic priming (30% of retrievals)
        // - Repetition priming (20% - recently seen)
        // - Associative priming (40% - follows common patterns)
        // - Interference (25% - similar to recent learning)
        // - Reconsolidation (15% - meets boundary conditions)

        // Realistic distributions based on typical memory usage
        // ...
    }
}
```

## Memory Overhead Analysis

```rust
pub struct MemoryOverheadAnalysis;

impl MemoryOverheadAnalysis {
    pub fn measure_memory_footprint(&self) -> MemoryReport {
        let baseline = self.measure_baseline_memory();

        let with_priming = self.measure_with_priming();
        let with_interference = self.measure_with_interference();
        let with_reconsolidation = self.measure_with_reconsolidation();
        let with_metrics = self.measure_with_metrics();
        let with_all = self.measure_all_features();

        MemoryReport {
            baseline,
            priming_overhead: with_priming - baseline,
            interference_overhead: with_interference - baseline,
            reconsolidation_overhead: with_reconsolidation - baseline,
            metrics_overhead: with_metrics - baseline,
            total_overhead: with_all - baseline,
            percentage_increase: (with_all - baseline) as f32 / baseline as f32,
            target: 0.20,  // <20% memory overhead
            pass: (with_all - baseline) as f32 / baseline as f32 < 0.20,
        }
    }
}
```

## Validation Criteria

**Must Pass:**
- Encoding p95 latency: < 75μs
- Retrieval p95 latency: < 300μs
- Throughput: >= 8K ops/sec
- Memory overhead: < 20%
- Metrics overhead (when enabled): < 1% CPU

**Should Pass:**
- Encoding p99 latency: < 100μs
- Retrieval p99 latency: < 500μs
- Throughput: >= 10K ops/sec
- Memory overhead: < 15%

## Regression Detection

Set up continuous benchmarking to detect performance regressions:

```rust
#[bench]
fn bench_regression_check(b: &mut Bencher) {
    let memory = EngramCore::new_with_all_features();

    b.iter(|| {
        let cue = random_node();
        let results = memory.retrieve(cue);
        black_box(results);
    });

    // Fail if slower than baseline by >10%
    let baseline_ns = 250_000;  // 250μs baseline
    let current_ns = b.elapsed().as_nanos() / b.iterations();

    assert!(current_ns < baseline_ns * 110 / 100,
        "Performance regression: {}ns vs {}ns baseline",
        current_ns, baseline_ns);
}
```

## Optimization Opportunities

If benchmarks fail, investigate:

1. **Priming boost computation:** Are we recomputing when we could cache?
2. **Interference detection:** Can we batch or lazy-evaluate?
3. **Metrics collection:** Are atomic operations creating cache coherence traffic?
4. **Reconsolidation checks:** Can we use bloom filter for fast negative checks?
5. **Memory allocations:** Profile with heaptrack to find allocation hotspots

## CI Integration

```yaml
# .github/workflows/performance.yml
name: Performance Benchmarks
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run integration benchmarks
        run: cargo bench --features=all -- --test integration_performance
      - name: Check performance budgets
        run: |
          if ! cargo bench --features=all | grep "PASS"; then
            echo "Performance regression detected"
            exit 1
          fi
```

Run on every PR to catch regressions early.
