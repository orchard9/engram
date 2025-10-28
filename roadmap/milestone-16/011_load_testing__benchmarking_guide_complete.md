# Task 011: Load Testing & Benchmarking Guide — pending

**Priority:** P2
**Estimated Effort:** 2 days
**Dependencies:** Task 004 (Performance Tuning)
**Agent:** verification-testing-lead

## Objective

Deliver comprehensive load testing toolkit, benchmark suite, and performance validation framework that validates Engram's performance claims through systematic, reproducible testing. Enable operators to validate capacity, detect regressions, and conduct chaos engineering experiments.

## Performance Targets from Vision.md

These are the claims we must validate:

- **Throughput:** 10,000 activations/second sustained on commodity hardware
- **Latency:** P99 < 10ms for single-hop activation
- **Scale:** 1M+ nodes with 768-dimensional embeddings
- **Concurrency:** Linear scaling with CPU cores (validated up to 32 cores)
- **Memory:** Overhead < 2x raw data size

## Key Deliverables

### 1. Load Testing CLI Tool (`tools/loadtest/`)

**Purpose:** Generate realistic workload patterns for capacity testing and stress testing.

**Implementation:**
```
/tools/loadtest/
├── src/
│   ├── main.rs              # CLI entry point with clap
│   ├── workload_generator.rs # Workload pattern generators
│   ├── distribution.rs       # Statistical distributions for realistic traffic
│   ├── replay.rs             # Deterministic workload replay from traces
│   ├── metrics_collector.rs  # Real-time metrics aggregation
│   └── report.rs             # Statistical analysis and report generation
├── scenarios/
│   ├── write_heavy.toml      # 80% store, 20% recall
│   ├── read_heavy.toml       # 20% store, 80% recall
│   ├── mixed_balanced.toml   # 50/50 read/write
│   ├── burst_traffic.toml    # Periodic load spikes
│   ├── embeddings_search.toml # Similarity search focused
│   ├── consolidation.toml    # Background consolidation during load
│   └── multi_tenant.toml     # Multiple memory spaces concurrently
├── traces/
│   └── production_sample.json # Anonymized production traffic traces
└── Cargo.toml
```

**Workload Characteristics:**

- **Arrival Patterns:** Poisson (random), Bursty (clusters), Periodic (waves), Constant (stress test)
- **Operation Mix:** Store/Recall/Pattern Completion/Consolidation ratios
- **Embedding Distribution:** Clustered (realistic), Uniform (pathological), Single-point (hot-spot)
- **Memory Space Distribution:** Single-tenant vs multi-tenant workloads
- **Query Complexity:** Shallow (1-2 hops) vs Deep (5+ hops) spreading activation
- **Data Size:** Small (100K nodes), Medium (1M nodes), Large (10M nodes)

**Deterministic Replay:**
All workloads accept `--seed` parameter for reproducibility. Load tests can be replayed exactly using saved trace files for regression testing.

### 2. Benchmark Suite (`scripts/run_benchmark.sh`, `scripts/compare_benchmarks.sh`)

**Benchmark Categories:**

**Core Operations:**
- `store_single` - Single memory insertion latency
- `store_batch` - Batch insertion throughput (1K, 10K, 100K)
- `recall_by_id` - Direct memory retrieval by ID
- `recall_by_cue` - Cue-based recall with confidence thresholds
- `embedding_search` - K-nearest neighbor search (k=10, k=100)
- `spreading_activation` - Multi-hop activation spreading (depth=3, depth=5)

**Pattern Completion:**
- `pattern_detection` - Identify recurring patterns in episodes
- `semantic_extraction` - Extract semantic knowledge from patterns
- `reconstruction` - Fill gaps in partial memories
- `consolidation_cycle` - Full consolidation run with metrics

**Concurrent Operations:**
- `concurrent_writes` - Parallel stores across threads (1, 2, 4, 8, 16, 32 threads)
- `concurrent_reads` - Parallel recalls with contention
- `mixed_concurrent` - Concurrent reads + writes
- `multi_space_isolation` - Multiple memory spaces with cross-space non-interference

**Storage Tier Operations:**
- `hot_tier_lookup` - Active memory cache hit
- `warm_tier_scan` - Append-only log scan
- `cold_tier_embedding_batch` - Columnar SIMD operations
- `tier_migration` - Memory promotion/demotion

**Implementation:**
```bash
#!/bin/bash
# scripts/run_benchmark.sh

# Run comprehensive benchmark suite with statistical validation
# Usage: ./scripts/run_benchmark.sh [--baseline baseline.json] [--output results.json]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Benchmark configuration
WARMUP_RUNS=3
BENCHMARK_RUNS=10
CONFIDENCE_LEVEL=0.95
MAX_VARIANCE=0.05  # 5% coefficient of variation threshold

# Statistical validation using Welch's t-test
# Null hypothesis: no performance difference from baseline
# Alternative: performance degraded (one-tailed test)
ALPHA=0.05  # Significance level

# Run criterion benchmarks with statistical rigor
cd "$PROJECT_ROOT"
cargo bench --bench comprehensive -- --save-baseline current

# Generate comparison report if baseline provided
if [ -n "${BASELINE:-}" ]; then
    cargo bench --bench comprehensive -- --baseline "$BASELINE" --save-baseline current
fi

# Export results in machine-readable format
python3 scripts/analyze_benchmarks.py \
    --input target/criterion/ \
    --output "${OUTPUT:-benchmark_results.json}" \
    --confidence "$CONFIDENCE_LEVEL" \
    --alpha "$ALPHA"
```

### 3. Performance Regression Detection

**Methodology:**

**Statistical Framework:**
- Use Welch's t-test for comparing two benchmark runs (current vs baseline)
- Effect size calculation using Cohen's d (small: 0.2, medium: 0.5, large: 0.8)
- Multiple testing correction using Benjamini-Hochberg procedure (control FDR)
- Bootstrap confidence intervals (10,000 resamples) for non-normal distributions

**Regression Criteria:**
- **Critical Regression:** P99 latency increased by >20% OR throughput decreased by >20% (p < 0.01)
- **Warning Regression:** P99 latency increased by >10% OR throughput decreased by >10% (p < 0.05)
- **Nominal Variation:** Changes within 5% are attributed to measurement noise

**Implementation:**
```python
# scripts/analyze_benchmarks.py

import json
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class BenchmarkResult:
    name: str
    mean: float
    std: float
    p50: float
    p95: float
    p99: float
    samples: List[float]

def welch_t_test(baseline: BenchmarkResult, current: BenchmarkResult) -> Tuple[float, float]:
    """
    Perform Welch's t-test (unequal variances) for regression detection.
    Returns (t_statistic, p_value) for one-tailed test (current > baseline).
    """
    t_stat, p_value_two_tailed = stats.ttest_ind(
        current.samples,
        baseline.samples,
        equal_var=False  # Welch's t-test
    )
    # Convert to one-tailed: testing if current is slower
    p_value = p_value_two_tailed / 2 if t_stat > 0 else 1 - (p_value_two_tailed / 2)
    return t_stat, p_value

def cohens_d(baseline: BenchmarkResult, current: BenchmarkResult) -> float:
    """
    Calculate Cohen's d effect size.
    Measures standardized difference between two means.
    """
    pooled_std = np.sqrt(
        (baseline.std ** 2 + current.std ** 2) / 2
    )
    return (current.mean - baseline.mean) / pooled_std

def detect_regression(baseline: BenchmarkResult, current: BenchmarkResult, alpha: float = 0.05) -> dict:
    """
    Detect performance regression using statistical hypothesis testing.
    """
    t_stat, p_value = welch_t_test(baseline, current)
    effect_size = cohens_d(baseline, current)

    percent_change = ((current.mean - baseline.mean) / baseline.mean) * 100

    if p_value < 0.01 and percent_change > 20:
        severity = "CRITICAL"
    elif p_value < alpha and percent_change > 10:
        severity = "WARNING"
    else:
        severity = "NOMINAL"

    return {
        "name": current.name,
        "severity": severity,
        "percent_change": percent_change,
        "p_value": p_value,
        "effect_size": effect_size,
        "baseline_mean": baseline.mean,
        "current_mean": current.mean,
        "statistical_power": calculate_power(baseline, current, alpha)
    }

def calculate_power(baseline: BenchmarkResult, current: BenchmarkResult, alpha: float) -> float:
    """
    Calculate statistical power (1 - beta) of the test.
    Power is probability of detecting a true effect.
    """
    effect_size = cohens_d(baseline, current)
    n = len(current.samples)
    # Use Cohen's power tables or simulation
    # Simplified: power increases with effect size and sample size
    noncentrality = effect_size * np.sqrt(n / 2)
    critical_value = stats.t.ppf(1 - alpha, df=2*n - 2)
    power = 1 - stats.nct.cdf(critical_value, df=2*n - 2, nc=noncentrality)
    return power
```

### 4. Chaos Engineering Scenarios

**Purpose:** Validate resilience and graceful degradation under adverse conditions.

**Fault Injection Framework:**
```
/scripts/chaos/
├── inject_network_latency.sh   # Add 100ms, 500ms, 1s latency
├── inject_packet_loss.sh        # Simulate 1%, 5%, 10% packet loss
├── inject_memory_pressure.sh    # Allocate memory to trigger OOM conditions
├── inject_cpu_contention.sh     # Spawn CPU-intensive processes
├── inject_disk_slow.sh          # Throttle disk I/O bandwidth
├── kill_random_connection.sh    # Randomly terminate client connections
└── corrupt_wal.sh               # Introduce controlled corruption in WAL
```

**Chaos Scenarios:**

1. **Network Partition:** Simulate split-brain scenario with gossip protocol validation
2. **Memory Pressure:** Trigger eviction policies and verify no data loss
3. **Slow Disk:** Validate backpressure and flow control
4. **Cascading Failure:** Kill random nodes and measure MTTR (Mean Time To Recovery)
5. **Thundering Herd:** Simultaneous reconnect of 1000 clients after partition
6. **Byzantine Faults:** Inject corrupted messages and verify isolation

**Validation Criteria:**
- System remains available (degraded performance acceptable)
- No data corruption (checksums validated)
- Graceful recovery within RTO (30 minutes)
- Metrics accurately reflect degraded state
- Alerts fire appropriately

### 5. Comparative Benchmarks vs Other Systems

**Differential Testing Targets:**

**Vector Databases:**
- FAISS (Facebook AI Similarity Search) - CPU baseline
- Pinecone - Managed service comparison
- Weaviate - Hybrid search comparison
- ScaNN (Google) - SIMD comparison

**Graph Databases:**
- Neo4j - Cypher query equivalents
- RedisGraph - In-memory graph comparison

**Benchmark Harness:**
```rust
// tools/loadtest/src/comparative.rs

/// Run identical workload against Engram and baseline systems.
/// Measures throughput, latency, and resource utilization.
pub struct ComparativeBenchmark {
    systems: Vec<Box<dyn BenchmarkTarget>>,
    workload: WorkloadSpec,
    metrics: MetricsCollector,
}

pub trait BenchmarkTarget {
    fn name(&self) -> &str;
    fn setup(&mut self, config: &BenchmarkConfig) -> Result<()>;
    fn store(&mut self, memory: &Memory) -> Result<Duration>;
    fn recall(&mut self, cue: &Cue) -> Result<(Vec<Memory>, Duration)>;
    fn embedding_search(&mut self, query: &[f32], k: usize) -> Result<(Vec<Memory>, Duration)>;
    fn teardown(&mut self) -> Result<()>;
}

impl BenchmarkTarget for EngramTarget { /* ... */ }
impl BenchmarkTarget for FaissTarget { /* ... */ }
impl BenchmarkTarget for Neo4jTarget { /* ... */ }
```

**Comparative Metrics:**
- Throughput at P99 < 10ms latency (ops/second)
- Memory efficiency (bytes per node)
- Index build time (for 1M nodes)
- Query latency distribution (P50, P95, P99, P99.9)
- Concurrency scaling (1, 4, 16, 64 threads)
- Resource utilization (CPU%, Memory%, Disk I/O)

### 6. Load Testing Scenarios (`scenarios/`)

**Scenario Specifications (TOML format):**

```toml
# scenarios/write_heavy.toml

name = "Write-Heavy Workload"
description = "80% store operations, 20% recall - simulates initial data ingestion"

[duration]
total_seconds = 3600  # 1 hour sustained load

[arrival]
pattern = "poisson"
mean_rate = 8000  # ops/second
lambda = 8000

[operations]
store_weight = 80
recall_weight = 15
embedding_search_weight = 5

[data]
num_nodes = 1_000_000
embedding_dim = 768
memory_spaces = 1

[validation]
expected_p99_latency_ms = 10
expected_throughput_ops_sec = 8000
max_error_rate = 0.001  # 0.1% errors allowed

[chaos]
enabled = false
```

```toml
# scenarios/burst_traffic.toml

name = "Burst Traffic Pattern"
description = "Periodic load spikes simulating real-world traffic patterns"

[duration]
total_seconds = 1800

[arrival]
pattern = "periodic_burst"
base_rate = 2000
burst_rate = 15000
burst_duration_sec = 30
burst_period_sec = 300  # Every 5 minutes

[operations]
store_weight = 40
recall_weight = 50
embedding_search_weight = 10

[data]
num_nodes = 500_000
embedding_dim = 768
memory_spaces = 4

[validation]
expected_p99_latency_ms = 15  # Relaxed during bursts
expected_throughput_ops_sec = 10000
max_error_rate = 0.005

[chaos]
enabled = false
```

### 7. Statistical Analysis Framework

**Hypothesis Tests:**

**H1: Throughput Capacity**
- Null: System cannot sustain 10K ops/sec
- Alternative: System sustains >= 10K ops/sec for 1 hour
- Test: Measure throughput in 60-second windows, require 100% of windows >= 10K
- Confidence: 95%

**H2: Latency SLA**
- Null: P99 latency >= 10ms
- Alternative: P99 latency < 10ms under load
- Test: Collect latency samples (n >= 10,000), compute empirical P99
- Confidence: 99%

**H3: Linear Scaling**
- Null: Throughput does not scale linearly with cores
- Alternative: Throughput increases proportionally (within 10% efficiency loss)
- Test: Linear regression R^2 > 0.95 on cores vs throughput
- Confidence: 95%

**H4: Memory Overhead**
- Null: Memory overhead >= 2x raw data size
- Alternative: Memory overhead < 2x
- Test: Measure RSS after loading 1M nodes, compare to 1M * (768*4 + metadata)
- Confidence: 99%

**Output Format:**
```json
{
  "benchmark_id": "2025-10-24T12:00:00Z",
  "hypothesis_tests": [
    {
      "name": "H1_throughput_capacity",
      "result": "PASS",
      "p_value": 0.001,
      "confidence_interval": [10250, 10780],
      "target": 10000,
      "achieved": 10500
    },
    {
      "name": "H2_latency_sla",
      "result": "PASS",
      "p_value": 0.003,
      "confidence_interval": [8.2, 9.7],
      "target": 10.0,
      "achieved": 8.9
    }
  ],
  "performance_summary": {
    "throughput_ops_sec": 10500,
    "p50_latency_ms": 3.2,
    "p95_latency_ms": 7.1,
    "p99_latency_ms": 8.9,
    "p999_latency_ms": 12.3,
    "memory_overhead_ratio": 1.83
  }
}
```

## Documentation Deliverables

### `/docs/operations/load-testing.md`

**Structure:**

1. **Overview** - When and why to load test
2. **Quick Start** - Run your first load test in 5 minutes
3. **Workload Scenarios** - Predefined scenarios and when to use them
4. **Custom Scenarios** - How to define realistic workloads for your use case
5. **Interpreting Results** - Understanding metrics and identifying bottlenecks
6. **Capacity Planning** - Using load test results to size deployments
7. **CI Integration** - Automated load testing in CI/CD pipelines
8. **Troubleshooting** - Common issues and resolutions

**Example Content:**
```markdown
## Quick Start

Run a 1-hour sustained load test at 10K ops/sec:

```bash
# Build the load testing tool
cd tools/loadtest
cargo build --release

# Run predefined mixed workload scenario
./target/release/loadtest \
  --scenario scenarios/mixed_balanced.toml \
  --duration 3600 \
  --target-rate 10000 \
  --output results/test_$(date +%s).json

# View real-time metrics dashboard
open http://localhost:7432/metrics
```

Expected output shows throughput, latency percentiles, and error rate:
```
Load Test: Mixed Balanced Workload
Duration: 3600s | Target Rate: 10000 ops/sec
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[00:15:30] Progress: 25% | Actual: 10234 ops/sec
  Store:    4098 ops/sec | P99: 8.2ms
  Recall:   5102 ops/sec | P99: 9.1ms
  Search:   1034 ops/sec | P99: 7.8ms
  Errors:   0.02%
```
```

### `/docs/operations/benchmarking.md`

**Structure:**

1. **Overview** - Purpose of benchmarking vs load testing
2. **Running Benchmarks** - Execute the comprehensive benchmark suite
3. **Baseline Management** - Establishing and maintaining performance baselines
4. **Regression Detection** - Automated regression testing
5. **Comparative Benchmarks** - Comparing against FAISS, Neo4j, etc.
6. **Interpreting Results** - Understanding benchmark outputs
7. **Performance Debugging** - Using benchmarks to find bottlenecks
8. **CI Integration** - Automated benchmarking on every PR

### `/docs/howto/test-production-capacity.md`

**Format:** Context → Action → Verification

**Example:**
```markdown
## Context

You need to validate that your Engram deployment can handle expected production load of 15K operations/second with P99 latency under 10ms.

## Action

1. Capture production traffic pattern (if available):
```bash
# Export 1 hour of production traffic as trace
engram export-trace \
  --start "2025-10-24T00:00:00Z" \
  --duration 3600 \
  --output prod_trace.json
```

2. Replay trace at increased rate to find capacity limit:
```bash
# Start at 100% production rate
loadtest --replay prod_trace.json --rate-multiplier 1.0

# Gradually increase to find breaking point
loadtest --replay prod_trace.json --rate-multiplier 1.5
loadtest --replay prod_trace.json --rate-multiplier 2.0
```

3. Analyze latency vs throughput curve:
```bash
python3 scripts/plot_capacity.py results/*.json
```

## Verification

Check that at 15K ops/sec:
- [ ] P99 latency < 10ms in all 60-second windows
- [ ] Error rate < 0.1%
- [ ] CPU utilization < 80% (headroom for spikes)
- [ ] Memory growth is bounded (no leaks)
- [ ] No disk I/O bottlenecks (queue depth < 32)

If any check fails, see troubleshooting guide.
```

### `/docs/reference/benchmark-results.md`

**Purpose:** Maintain historical baseline performance results for regression detection.

**Format:**
```markdown
## Baseline Performance Results

Last Updated: 2025-10-24

### Environment
- CPU: AMD EPYC 7763 (64 cores, 2.45 GHz)
- Memory: 512 GB DDR4
- Disk: NVMe SSD (7000 MB/s read, 5000 MB/s write)
- OS: Linux 6.1.0
- Rust: 1.83.0

### Core Operations (Single-Threaded)

| Operation | Mean (ms) | P95 (ms) | P99 (ms) | Throughput (ops/sec) |
|-----------|-----------|----------|----------|----------------------|
| store_single | 0.23 | 0.31 | 0.42 | 4347 |
| recall_by_id | 0.18 | 0.25 | 0.34 | 5556 |
| recall_by_cue | 1.85 | 2.41 | 3.12 | 540 |
| embedding_search_k10 | 1.32 | 1.78 | 2.23 | 758 |
| spreading_activation_d3 | 3.45 | 4.67 | 5.89 | 290 |

### Concurrent Operations (32 Threads)

| Operation | Throughput (ops/sec) | P99 (ms) | Scaling Efficiency |
|-----------|----------------------|----------|--------------------|
| concurrent_writes | 87,234 | 8.7 | 0.93 (93%) |
| concurrent_reads | 124,567 | 6.4 | 0.91 (91%) |
| mixed_concurrent | 102,345 | 9.2 | 0.92 (92%) |

### Memory Efficiency

- 1M nodes (768-dim): 3.2 GB memory usage
- Raw data size: 1M * (768*4 + 64 metadata) = 1.84 GB
- Overhead ratio: 1.74x (within 2x target)
```

## Implementation Notes

### Criterion Integration

Use Criterion.rs for statistical benchmarking with proper warmup and sample size:

```rust
// benches/comprehensive.rs

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
    measurement::WallTime, BatchSize,
};

fn benchmark_store_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("store");

    // Configure statistical parameters
    group.sample_size(100);  // Number of iterations
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    group.confidence_level(0.95);
    group.noise_threshold(0.02);  // 2% noise threshold

    // Benchmark single store operation
    group.bench_function("store_single", |b| {
        let store = setup_store();
        b.iter_batched(
            || generate_random_memory(),
            |memory| store.store(memory),
            BatchSize::SmallInput,
        );
    });

    // Benchmark batch operations with varying sizes
    for batch_size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("store_batch", batch_size),
            &batch_size,
            |b, &size| {
                let store = setup_store();
                b.iter_batched(
                    || generate_memory_batch(size),
                    |batch| store.store_batch(batch),
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_store_operations);
criterion_main!(benches);
```

### Deterministic Workload Generation

All load tests must be reproducible using seeds:

```rust
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub struct WorkloadGenerator {
    rng: ChaCha8Rng,
    config: WorkloadConfig,
}

impl WorkloadGenerator {
    pub fn new(seed: u64, config: WorkloadConfig) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            config,
        }
    }

    pub fn next_operation(&mut self) -> Operation {
        let op_type = self.rng.gen_range(0.0..1.0);

        if op_type < self.config.store_weight {
            Operation::Store(self.generate_memory())
        } else if op_type < self.config.store_weight + self.config.recall_weight {
            Operation::Recall(self.generate_cue())
        } else {
            Operation::Search(self.generate_query())
        }
    }

    pub fn generate_memory(&mut self) -> Memory {
        // Generate deterministic but realistic memory
        let embedding = self.generate_embedding_clustered();
        Memory::new(embedding, Confidence::high())
    }

    fn generate_embedding_clustered(&mut self) -> [f32; 768] {
        // Use mixture of Gaussians to create realistic clusters
        let cluster_id = self.rng.gen_range(0..self.config.num_clusters);
        let cluster_center = &self.config.cluster_centers[cluster_id];

        let mut embedding = [0.0; 768];
        for i in 0..768 {
            let noise = self.rng.sample(Normal::new(0.0, 0.1).unwrap());
            embedding[i] = (cluster_center[i] + noise).clamp(-1.0, 1.0);
        }

        normalize(&mut embedding);
        embedding
    }
}
```

## Acceptance Criteria

### Functional Requirements

- [ ] Load testing CLI tool (`tools/loadtest/`) compiles and runs
- [ ] All 7 predefined workload scenarios execute successfully
- [ ] Benchmark suite covers all core operations (store, recall, search, spreading, consolidation)
- [ ] Regression detection script identifies >5% performance degradation with p < 0.05
- [ ] Chaos engineering scripts can inject all 6 fault types
- [ ] Comparative benchmark harness runs against FAISS and Neo4j

### Performance Validation

- [ ] Load test sustains 10K ops/sec for 1 hour with P99 < 10ms
- [ ] System handles 1M+ nodes with 768-dim embeddings
- [ ] Concurrent operations show linear scaling up to 32 cores (>90% efficiency)
- [ ] Memory overhead measured at < 2x raw data size
- [ ] All hypothesis tests (H1-H4) pass with documented confidence levels

### Chaos Engineering

- [ ] System remains available during network partition (degraded mode acceptable)
- [ ] No data corruption after WAL corruption injection (checksums validated)
- [ ] Recovery from node failure completes within RTO (30 minutes)
- [ ] Thundering herd scenario (1000 reconnects) handled without cascading failure

### Documentation

- [ ] Load testing guide (`docs/operations/load-testing.md`) complete with 8 sections
- [ ] Benchmarking guide (`docs/operations/benchmarking.md`) complete with 8 sections
- [ ] Capacity testing walkthrough (`docs/howto/test-production-capacity.md`) follows Context→Action→Verification
- [ ] Baseline benchmark results (`docs/reference/benchmark-results.md`) documented with environment details

### Statistical Rigor

- [ ] All benchmarks use Criterion with proper warmup (>3 sec) and sample size (>30)
- [ ] Regression detection uses Welch's t-test with Benjamini-Hochberg correction
- [ ] Confidence intervals reported for all performance metrics
- [ ] Statistical power calculated and reported (target >0.80)
- [ ] Effect sizes computed using Cohen's d

## Follow-Up Tasks

If any acceptance criteria fail:

1. **Performance below targets** → Create follow-up task for optimization (link to Task 004)
2. **Chaos tests reveal data corruption** → Create follow-up task for WAL hardening
3. **Regression detection false positives** → Tune statistical parameters (increase sample size, adjust alpha)
4. **Comparative benchmarks show Engram slower** → Create architecture investigation task

## Implementation Sequence

### Day 1: Load Testing Infrastructure

**Morning:**
- Create `tools/loadtest/` directory structure
- Implement workload generator with deterministic seeding
- Add Poisson and bursty arrival patterns

**Afternoon:**
- Implement 7 predefined scenarios (TOML configurations)
- Add metrics collector with real-time aggregation
- Create report generator with statistical summary

**Evening:**
- Test load generator with local Engram instance
- Validate deterministic replay with same seed
- Document CLI usage

### Day 2: Benchmarks and Validation

**Morning:**
- Create comprehensive Criterion benchmark suite (`benches/comprehensive.rs`)
- Implement all core operation benchmarks
- Add concurrent operation benchmarks

**Afternoon:**
- Implement regression detection script (`scripts/analyze_benchmarks.py`)
- Add statistical hypothesis testing framework
- Create baseline management workflow

**Evening:**
- Write chaos engineering scripts (6 fault injection types)
- Run initial validation tests
- Write documentation (4 markdown files)
- Generate baseline benchmark results

## Testing Strategy

### Unit Tests

Test workload generation determinism:
```rust
#[test]
fn test_workload_determinism() {
    let config = WorkloadConfig::default();
    let mut gen1 = WorkloadGenerator::new(42, config.clone());
    let mut gen2 = WorkloadGenerator::new(42, config.clone());

    for _ in 0..1000 {
        let op1 = gen1.next_operation();
        let op2 = gen2.next_operation();
        assert_eq!(op1, op2, "Same seed must produce identical operations");
    }
}
```

### Integration Tests

Validate load test execution:
```rust
#[test]
fn test_load_test_execution() {
    let scenario = load_scenario("scenarios/mixed_balanced.toml");
    let results = run_load_test(scenario, Duration::from_secs(60));

    assert!(results.throughput_ops_sec > 5000);
    assert!(results.p99_latency_ms < 15.0);
    assert!(results.error_rate < 0.01);
}
```

### Statistical Validation

Verify regression detection:
```rust
#[test]
fn test_regression_detection_sensitivity() {
    // Generate baseline with mean=10ms, std=1ms
    let baseline = generate_samples(10.0, 1.0, 100);

    // Generate regressed samples with mean=12ms (20% slower)
    let regressed = generate_samples(12.0, 1.0, 100);

    let result = detect_regression(&baseline, &regressed, 0.05);
    assert_eq!(result.severity, "WARNING");
    assert!(result.p_value < 0.05);
}
```

## Technical Notes

### Avoiding Measurement Bias

1. **Warmup:** Run 3+ seconds of warmup to stabilize JIT, caches, and branch prediction
2. **Outlier Detection:** Use Tukey's fence (IQR method) to remove outliers before computing percentiles
3. **Timer Resolution:** Use high-resolution timers (TSC or RDTSC on x86)
4. **CPU Isolation:** Pin benchmark threads to specific cores using `taskset`
5. **Background Load:** Disable background tasks (cron, indexing) during benchmarks

### Performance Targets Validation Matrix

| Claim | Test Method | Pass Criteria | Current Status |
|-------|-------------|---------------|----------------|
| 10K ops/sec | 1-hour sustained load test | 100% of 1-min windows >= 10K | TBD |
| P99 < 10ms | Latency measurement under load | Empirical P99 < 10ms | TBD |
| 1M+ nodes | Scale test with embeddings | System operational at 1M nodes | TBD |
| Linear scaling | Concurrent benchmark 1-32 cores | R^2 > 0.95 on throughput | TBD |
| <2x memory overhead | Memory measurement | RSS / raw_data < 2.0 | TBD |

## References

### Academic Literature

- Welch, B. L. (1947). "The generalization of 'Student's' problem when several different population variances are involved"
- Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences"
- Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False Discovery Rate"
- Basiri, A. et al. (2016). "Chaos Engineering" - Netflix TechBlog

### Industry Standards

- SNIA IOPS benchmarking guidelines
- TPC-C transaction processing benchmarks
- YCSB (Yahoo! Cloud Serving Benchmark) methodology

### Tools

- Criterion.rs - Statistical benchmarking framework
- hyperfine - Command-line benchmarking tool
- wrk2 - HTTP load testing with coordinated omission correction
- tc (traffic control) - Linux network fault injection
- stress-ng - CPU/memory/disk stress testing
