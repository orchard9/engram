# Load Testing

Comprehensive guide to load testing Engram deployments for capacity planning and stress testing.

## Overview

Load testing validates that Engram can handle expected production workloads before deployment. Use load testing to:

- **Capacity Planning**: Determine maximum sustainable throughput
- **Performance Validation**: Verify P99 latency under load meets SLAs
- **Bottleneck Identification**: Find system constraints before they impact production
- **Configuration Tuning**: Optimize settings for your workload patterns
- **Regression Detection**: Ensure new versions maintain performance

## When to Load Test

- Before initial production deployment
- After infrastructure changes (hardware, network, storage)
- When scaling to new capacity levels
- After significant code changes (major versions)
- Quarterly for capacity planning validation

## Quick Start

Run a 1-hour sustained load test at 10K ops/sec:

```bash
# Build the load testing tool
cd tools/loadtest
cargo build --release

# Run predefined mixed workload scenario
./target/release/loadtest run \
  --scenario scenarios/mixed_balanced.toml \
  --duration 3600 \
  --target-rate 10000 \
  --output results/test_$(date +%s).json \
  --endpoint http://localhost:7432
```

Expected output shows real-time metrics:

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

## Workload Scenarios

Engram includes 7 predefined scenarios covering common deployment patterns:

### 1. Write-Heavy (`write_heavy.toml`)

**Use case**: Initial data ingestion, batch imports

**Operation mix**:
- 80% Store operations
- 15% Recall operations
- 5% Embedding search

**Configuration**:
```toml
[operations]
store_weight = 80.0
recall_weight = 15.0
embedding_search_weight = 5.0

[arrival]
pattern = "poisson"
mean_rate = 8000.0

[validation]
expected_p99_latency_ms = 12.0  # Relaxed for write-heavy
expected_throughput_ops_sec = 8000.0
```

**Run**:
```bash
loadtest run --scenario scenarios/write_heavy.toml --duration 1800
```

### 2. Read-Heavy (`read_heavy.toml`)

**Use case**: Query-focused applications, production serving

**Operation mix**:
- 20% Store operations
- 70% Recall operations
- 10% Embedding search

**Configuration**:
```toml
[operations]
store_weight = 20.0
recall_weight = 70.0
embedding_search_weight = 10.0

[arrival]
pattern = "constant"
rate = 12000.0

[validation]
expected_p99_latency_ms = 8.0  # Stricter for read-heavy
expected_throughput_ops_sec = 12000.0
```

**Run**:
```bash
loadtest run --scenario scenarios/read_heavy.toml --duration 1800
```

### 3. Mixed Balanced (`mixed_balanced.toml`)

**Use case**: General-purpose workload, steady-state production

**Operation mix**:
- 40% Store operations
- 50% Recall operations
- 10% Embedding search

**Target**: 10K ops/sec sustained for 1 hour
**Validation**: P99 < 10ms

This is the recommended baseline test for capacity validation.

### 4. Burst Traffic (`burst_traffic.toml`)

**Use case**: Applications with periodic load spikes (e.g., daily patterns)

**Pattern**: 2K ops/sec baseline, 15K ops/sec bursts every 5 minutes for 30 seconds

**Configuration**:
```toml
[arrival]
pattern = "periodic_burst"
base_rate = 2000.0
burst_rate = 15000.0
burst_duration_sec = 30
burst_period_sec = 300
```

Validates system handles traffic variability without degradation.

### 5. Embeddings Search (`embeddings_search.toml`)

**Use case**: Vector similarity search workloads

**Operation mix**:
- 10% Store operations
- 20% Recall operations
- 70% Embedding search (k=10 to k=50)

Tests SIMD optimizations and index performance.

### 6. Consolidation (`consolidation.toml`)

**Use case**: Background consolidation with concurrent load

**Features**:
- Runs consolidation cycles during load test
- Validates no impact on foreground latency
- Tests memory tier migration under pressure

### 7. Multi-Tenant (`multi_tenant.toml`)

**Use case**: Multiple memory spaces with cross-space isolation

**Configuration**:
```toml
[data]
memory_spaces = 4
```

Validates isolation guarantees and per-tenant fairness.

## Custom Scenarios

Create custom TOML files for your specific workload:

```toml
name = "My Custom Workload"
description = "Production traffic pattern captured on 2025-10-24"

[duration]
total_seconds = 7200  # 2 hours

[arrival]
pattern = "poisson"
mean_rate = 15000.0

[operations]
store_weight = 30.0
recall_weight = 60.0
embedding_search_weight = 10.0
pattern_completion_weight = 0.0

[data]
num_nodes = 2_000_000
embedding_dim = 768
memory_spaces = 8

[data.embedding_distribution]
type = "clustered"
num_clusters = 20
std_dev = 0.15

[validation]
expected_p99_latency_ms = 10.0
expected_throughput_ops_sec = 15000.0
max_error_rate = 0.001
```

Run with:
```bash
loadtest run --scenario my_custom.toml
```

## Arrival Patterns

Control how operations are distributed over time:

### Poisson (Random)

Realistic model for independent user requests:

```toml
[arrival]
pattern = "poisson"
lambda = 10000.0  # Mean rate (ops/sec)
```

### Bursty

Clustered requests simulating coordinated load:

```toml
[arrival]
pattern = "bursty"
mean_rate = 10000.0
burst_factor = 3.0
burst_probability = 0.1
```

### Periodic

Sine wave pattern for daily/hourly cycles:

```toml
[arrival]
pattern = "periodic"
base_rate = 5000.0
amplitude = 5000.0
period_secs = 3600  # 1 hour cycle
```

### Constant (Stress Test)

Fixed rate for maximum throughput testing:

```toml
[arrival]
pattern = "constant"
rate = 15000.0
```

## Interpreting Results

### Real-Time Metrics

Monitor during test execution:

- **Throughput**: Actual ops/sec vs target rate
- **P99 Latency**: 99th percentile response time by operation type
- **Error Rate**: Percentage of failed operations
- **Progress**: Time remaining and operations completed

### Final Report

After completion, analyze JSON output:

```json
{
  "summary": {
    "total_operations": 36000000,
    "duration_secs": 3600,
    "throughput_ops_sec": 10000,
    "error_rate": 0.0002
  },
  "latency": {
    "store": {
      "p50": 3.2,
      "p95": 7.1,
      "p99": 8.9,
      "p99.9": 12.3
    },
    "recall": {
      "p50": 2.8,
      "p95": 6.4,
      "p99": 8.2,
      "p99.9": 11.1
    }
  },
  "validation": {
    "met_throughput_target": true,
    "met_latency_target": true,
    "max_error_rate_ok": true
  }
}
```

### Key Metrics

**Throughput**:
- Target: >= 10,000 ops/sec sustained
- Measure: Operations completed / duration
- Pass criteria: >= 95% of target for all 60-second windows

**Latency**:
- Target: P99 < 10ms for single-hop activation
- Measure: 99th percentile of response times
- Pass criteria: P99 below target in steady state

**Error Rate**:
- Target: < 0.1% (1 error per 1000 operations)
- Measure: Failed operations / total operations
- Common errors: Timeouts, connection refused, 5xx responses

## Capacity Planning

Use load test results to size production deployments:

### 1. Find Breaking Point

Run increasing load until latency degrades:

```bash
for rate in 10000 15000 20000 25000 30000; do
  loadtest run \
    --scenario scenarios/mixed_balanced.toml \
    --target-rate $rate \
    --duration 600 \
    --output results/capacity_${rate}.json
done
```

Analyze results:
```bash
python3 scripts/plot_capacity.py results/capacity_*.json
```

### 2. Calculate Headroom

Reserve capacity for traffic spikes:

- **Baseline**: 10K ops/sec sustained
- **Peak**: 30K ops/sec observed during load test
- **Headroom**: 3x capacity for 3x traffic spikes
- **Deployment target**: Configure for 30K ops/sec sustained

### 3. Horizontal Scaling

Test multi-node deployments:

```bash
# Test with 1, 2, 4, 8 nodes
for nodes in 1 2 4 8; do
  # Deploy Engram cluster with $nodes instances
  deploy_cluster $nodes

  # Run load test
  loadtest run \
    --scenario scenarios/mixed_balanced.toml \
    --duration 1800 \
    --output results/scaling_${nodes}nodes.json
done

# Analyze scaling efficiency
python3 scripts/analyze_scaling.py results/scaling_*.json
```

Expected scaling efficiency: > 90% up to 8 nodes

## CI Integration

Automate load testing in CI/CD pipelines:

```yaml
# .github/workflows/load-test.yml
name: Load Test

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly at 2 AM Sunday
  workflow_dispatch:  # Manual trigger

jobs:
  load-test:
    runs-on: [self-hosted, large]
    timeout-minutes: 120

    steps:
      - uses: actions/checkout@v3

      - name: Build Engram
        run: cargo build --release

      - name: Start Engram
        run: ./target/release/engram-server --config ci-test.toml &

      - name: Wait for startup
        run: sleep 10

      - name: Run load test
        run: |
          cd tools/loadtest
          cargo run --release -- run \
            --scenario scenarios/mixed_balanced.toml \
            --duration 3600 \
            --output ../../results/ci_loadtest.json

      - name: Validate results
        run: |
          python3 scripts/validate_loadtest.py results/ci_loadtest.json

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: results/ci_loadtest.json
```

## Troubleshooting

### High Latency (P99 > target)

**Symptoms**: P99 latency exceeds 10ms target

**Diagnosis**:
1. Check CPU utilization: `top` or `htop`
2. Check memory pressure: `free -h` or `vm_stat`
3. Check disk I/O: `iostat -x 1`
4. Review Engram logs for slow operations

**Solutions**:
- Increase CPU cores (target < 80% utilization)
- Add memory (target < 90% usage)
- Optimize spreading activation depth
- Enable SIMD optimizations
- Check network latency between client and server

### Low Throughput (< target)

**Symptoms**: Actual throughput significantly below target rate

**Diagnosis**:
1. Check error rate: High errors indicate system overload
2. Review connection pool settings
3. Check network bandwidth: `iftop` or `nload`
4. Profile Engram with `perf` or `flamegraph`

**Solutions**:
- Increase connection pool size
- Add more Engram instances (horizontal scaling)
- Optimize hot code paths
- Enable batch operations
- Review consolidation settings (background work may impact foreground)

### Connection Errors

**Symptoms**: High error rate with connection refused or timeouts

**Diagnosis**:
1. Check Engram is running: `ps aux | grep engram`
2. Verify endpoint URL is correct
3. Check network connectivity: `curl http://localhost:7432/health`
4. Review firewall rules

**Solutions**:
- Restart Engram server
- Fix network configuration
- Increase connection limits: `ulimit -n 65536`
- Add load balancer for connection distribution

### Memory Leaks

**Symptoms**: Memory usage grows unbounded during test

**Diagnosis**:
1. Monitor RSS over time: `watch -n 1 "ps aux | grep engram"`
2. Check for accumulation in specific data structures
3. Review heap profile: `cargo run --release --features profiling`

**Solutions**:
- Enable memory profiling to identify leak
- Check for circular references in graph structures
- Verify consolidation is running and evicting old memories
- Update to latest version with memory leak fixes

## Best Practices

1. **Start Small**: Begin with short tests (5-10 minutes) before long runs
2. **Use Deterministic Seeds**: Reproducibility is critical for debugging
3. **Monitor System Resources**: Watch CPU, memory, disk, network during test
4. **Test Realistic Workloads**: Use production traffic patterns when possible
5. **Automate**: Integrate load tests into CI for continuous validation
6. **Document Results**: Save baselines for regression detection
7. **Test Failure Modes**: Use chaos engineering to validate resilience

## Related Documentation

- [Benchmarking Guide](benchmarking.md) - Microbenchmarks for specific operations
- [Test Production Capacity](../howto/test-production-capacity.md) - Step-by-step capacity testing
- [Performance Tuning](performance-tuning.md) - Optimization techniques
- [Monitoring](monitoring.md) - Production observability
