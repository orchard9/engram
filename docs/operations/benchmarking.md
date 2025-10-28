# Benchmarking

Microbenchmark suite for measuring specific operation performance and detecting regressions.

## Overview

Benchmarking complements load testing by providing precise measurements of individual operations. Use benchmarks to:

- **Regression Detection**: Identify performance degradation in new code
- **Optimization Validation**: Verify performance improvements
- **Operation Profiling**: Understand cost of specific operations
- **Baseline Establishment**: Document expected performance characteristics
- **Comparative Analysis**: Compare against other systems (FAISS, Neo4j)

## Benchmark vs Load Testing

| Aspect | Benchmarking | Load Testing |
|--------|--------------|--------------|
| **Scope** | Micro (single operations) | Macro (full system) |
| **Duration** | Seconds to minutes | Minutes to hours |
| **Purpose** | Optimization, regression detection | Capacity planning, SLA validation |
| **Methodology** | Criterion statistical framework | Realistic workload simulation |
| **Output** | Latency distributions, throughput | System capacity, error rates |

## Running Benchmarks

### Quick Start

Run the comprehensive benchmark suite:

```bash
# Build and run all benchmarks
./scripts/run_benchmarks.sh

# Save results as baseline for future comparisons
./scripts/run_benchmarks.sh --save-baseline v0.1.0

# Compare against previous baseline
./scripts/run_benchmarks.sh --baseline v0.1.0 --output results.json
```

Expected output:

```
================================
Engram Benchmark Suite
================================

Configuration:
  Warmup time:      3s
  Measurement time: 10s
  Sample size:      50
  Confidence level: 0.95
  Significance:     α = 0.05

Building benchmarks...
Running benchmarks...
This may take 10-15 minutes depending on hardware.

store/store_single      time:   [230.2 µs 235.4 µs 241.1 µs]
store/batch/100         time:   [21.2 ms 22.1 ms 23.3 ms]
                        thrpt:  [4291 elem/s 4525 elem/s 4717 elem/s]
recall/recall_by_id     time:   [182.3 µs 187.9 µs 194.1 µs]
spreading_activation/d3 time:   [3.42 ms 3.51 ms 3.61 ms]

Benchmark execution complete!

✓ No regressions detected
  Analysis saved to: target/benchmark_analysis.json

Benchmark results available at:
  target/criterion/report/index.html
```

### Baseline Management

Establish performance baselines for regression detection:

```bash
# Create baseline before major changes
./scripts/run_benchmarks.sh --save-baseline pre_refactor

# After changes, compare
./scripts/run_benchmarks.sh --baseline pre_refactor

# If regression detected:
# ❌ CRITICAL regressions detected!
#   store/store_single: 20.3% slower (p=0.002)
#   See analysis: target/benchmark_analysis.json
```

Keep baselines organized:
- `main` - Latest passing main branch
- `v0.1.0`, `v0.2.0` - Release versions
- `pre_refactor`, `post_optimization` - Major changes

### Comparing Baselines

Compare any two saved baselines:

```bash
./scripts/compare_benchmarks.sh v0.1.0 v0.2.0 --output v0.1_to_v0.2.json
```

## Benchmark Categories

### 1. Core Operations

Single-threaded operation benchmarks:

**store_single**: Single memory insertion latency
- Target: < 250µs mean
- Measures: Hash insertion, memory allocation

**store_batch**: Batch insertion throughput
- Sizes: 100, 1K, 10K items
- Target: > 4000 ops/sec for batch of 100

**recall_by_id**: Direct memory retrieval by ID
- Target: < 200µs mean
- Measures: Hash lookup, deserialization

**recall_by_cue**: Cue-based recall with confidence thresholds
- Target: < 2ms mean
- Measures: Activation spreading (shallow)

**embedding_search**: K-nearest neighbor search
- k=10, k=100
- Target: < 1.5ms for k=10
- Measures: SIMD vector operations, index traversal

**spreading_activation**: Multi-hop activation spreading
- Depth 3, depth 5
- Target: < 4ms for depth 3
- Measures: Graph traversal, parallel execution

### 2. Pattern Completion

Higher-level cognitive operations:

**pattern_detection**: Identify recurring patterns in episodes
- Complexity: 100-node subgraphs
- Target: < 10ms

**semantic_extraction**: Extract semantic knowledge from patterns
- Consolidation operation
- Target: < 50ms per pattern

**reconstruction**: Fill gaps in partial memories
- Confabulation using pattern completion
- Target: < 15ms

**consolidation_cycle**: Full consolidation run
- Episode to semantic transformation
- Target: < 100ms per episode

### 3. Concurrent Operations

Multi-threaded scaling benchmarks:

**concurrent_writes**: Parallel stores across threads
- Threads: 1, 2, 4, 8, 16, 32
- Target: > 90% scaling efficiency
- Validates lock-free data structures

**concurrent_reads**: Parallel recalls with contention
- Threads: 1, 2, 4, 8, 16, 32
- Target: > 90% scaling efficiency

**concurrent_mixed**: Concurrent reads + writes
- 50/50 mix
- Target: > 90% scaling efficiency
- Tests read-write coordination

**multi_space_isolation**: Multiple memory spaces concurrently
- 4 memory spaces, 8 threads each
- Target: No interference (isolation maintained)

### 4. Storage Tier Operations

Tiered storage performance:

**hot_tier_lookup**: Active memory cache hit
- Target: < 100ns (in-memory hash lookup)

**warm_tier_scan**: Append-only log scan
- Target: < 1ms for 1MB scan
- Measures: Sequential I/O, decompression

**cold_tier_embedding_batch**: Columnar SIMD operations
- Target: > 10K vectors/sec dot product
- Measures: SIMD utilization

**tier_migration**: Memory promotion/demotion
- Target: < 5ms per migration
- Measures: I/O efficiency, consistency

## Interpreting Results

### Criterion Output

Criterion provides statistical analysis:

```
store/store_single      time:   [230.2 µs 235.4 µs 241.1 µs]
                        change: [-3.1% +1.2% +5.8%] (p = 0.23 > 0.05)
                        No change in performance detected.
```

**time**: [lower bound, estimate, upper bound] at 95% confidence
**change**: Percent change from baseline
**p-value**: Statistical significance (< 0.05 indicates significant change)

### Regression Analysis

Automated regression detection classifies changes:

**CRITICAL**: > 20% slower, p < 0.01
- Requires immediate investigation
- Blocks release

**WARNING**: > 10% slower, p < 0.05
- Investigate before merge
- May block release depending on criticality

**NOMINAL**: < 10% change or p > 0.05
- Measurement noise
- No action required

**IMPROVEMENT**: Significantly faster
- Document optimization
- Update baselines

### Statistical Metrics

**Mean**: Average latency across all samples
- Use for throughput calculations

**P95/P99**: Tail latency at 95th/99th percentile
- Critical for SLA validation

**Standard Deviation**: Measurement variability
- High variance indicates non-deterministic performance

**Coefficient of Variation** (CV = std/mean):
- CV < 0.05 (5%): Good reproducibility
- CV > 0.10 (10%): High variance, investigate

## Regression Detection Framework

### Statistical Tests

**Welch's t-test**: Compare two benchmark runs
- Handles unequal variances
- One-tailed test (current > baseline)
- Null hypothesis: no performance difference

**Cohen's d effect size**: Standardized difference
- Small: 0.2, Medium: 0.5, Large: 0.8
- Independent of sample size

**Benjamini-Hochberg correction**: Multiple testing correction
- Controls False Discovery Rate (FDR)
- Prevents spurious regressions from multiple comparisons

### Automated Analysis

Python script performs comprehensive analysis:

```bash
python3 scripts/analyze_benchmarks.py \
  --baseline target/criterion/v0.1.0 \
  --current target/criterion/main \
  --output regression_report.json \
  --alpha 0.05
```

Output format:

```json
{
  "summary": {
    "total_benchmarks": 25,
    "critical_regressions": 0,
    "warning_regressions": 1,
    "improvements": 3,
    "nominal_changes": 21
  },
  "warnings": [{
    "name": "store/store_batch/1000",
    "percent_change": 12.4,
    "p_value": 0.032,
    "effect_size": 0.54,
    "baseline_mean": 22.1,
    "current_mean": 24.8,
    "statistical_power": 0.87
  }]
}
```

## Performance Debugging

### Identifying Bottlenecks

Use benchmarks to find hot paths:

1. **Profile benchmark**:
```bash
cargo bench --bench comprehensive -- --profile-time=10
```

2. **Generate flamegraph**:
```bash
cargo flamegraph --bench comprehensive
```

3. **Analyze results**: Focus on functions consuming > 10% CPU time

### Optimization Workflow

1. **Establish baseline**: Run benchmarks, save as `pre_optimization`
2. **Profile**: Identify hot path using flamegraph
3. **Optimize**: Apply targeted optimization
4. **Validate**: Re-run benchmarks, compare to baseline
5. **Iterate**: Repeat until target met

Example:

```bash
# Baseline
./scripts/run_benchmarks.sh --save-baseline pre_simd

# Optimize (e.g., add SIMD to embedding ops)

# Validate
./scripts/run_benchmarks.sh --baseline pre_simd

# Expected: embedding_search improved by > 2x
```

## CI Integration

Automated benchmarking in pull requests:

```yaml
name: Benchmark

on:
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Checkout base commit
        run: git checkout ${{ github.base_ref }}

      - name: Run baseline benchmarks
        run: ./scripts/run_benchmarks.sh --save-baseline pr_base

      - name: Checkout PR commit
        run: git checkout ${{ github.head_ref }}

      - name: Run PR benchmarks
        run: ./scripts/run_benchmarks.sh --baseline pr_base --output pr_results.json

      - name: Comment on PR
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('pr_results.json'));
            const comment = generateComment(results);
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

## Comparative Benchmarks

Compare Engram against FAISS and Neo4j:

```bash
# Requires FAISS and Neo4j installations
cd tools/loadtest

# Run comparative benchmarks
cargo run --release --features comparative -- compare \
  --systems engram,faiss,neo4j \
  --num-nodes 100000 \
  --embedding-dim 768 \
  --output comparative_results.json
```

**Note**: Comparative benchmarking requires external dependencies (FAISS library, Neo4j instance). See `tools/loadtest/src/comparative.rs` for integration details.

## Best Practices

1. **Stable Environment**: Run on dedicated hardware, minimal background processes
2. **CPU Isolation**: Pin benchmarks to specific cores: `taskset -c 0-7 cargo bench`
3. **Thermal Stability**: Allow CPU to cool between runs
4. **Sample Size**: Use >= 30 samples for statistical validity (Criterion default: 100)
5. **Warmup**: Always include warmup phase (Criterion default: 3s)
6. **Reproducibility**: Document hardware, OS version, compiler version
7. **Version Baselines**: Save baselines for every release
8. **Continuous Monitoring**: Track trends over time, not just point comparisons

## Troubleshooting

### High Variance (CV > 10%)

**Causes**:
- Background processes competing for CPU
- Thermal throttling
- Non-deterministic algorithms (use `--deterministic` flag)
- Insufficient warmup

**Solutions**:
- Close background applications
- Increase warmup time: `group.warm_up_time(Duration::from_secs(10))`
- Use CPU governor: `cpupower frequency-set --governor performance`

### Benchmark Fails to Complete

**Causes**:
- Timeout (default: 2 minutes per benchmark)
- Out of memory
- Deadlock in concurrent benchmarks

**Solutions**:
- Increase timeout: `group.measurement_time(Duration::from_secs(30))`
- Reduce sample size: `group.sample_size(30)`
- Check logs for deadlock backtraces

## Related Documentation

- [Load Testing Guide](load-testing.md) - Full system capacity testing
- [Performance Tuning](performance-tuning.md) - Optimization techniques
- [Profiling Methodology](../performance/profiling_methodology.md) - Detailed profiling guide
