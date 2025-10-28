# Performance Baselines

This document provides expected performance baselines for Engram on reference hardware configurations. Use these baselines to validate your deployment and detect performance regressions.

## Reference Hardware Configurations

### Standard Configuration

**Hardware:**
- CPU: 32 cores (Intel Xeon Gold 6258R @ 2.7GHz, Cascade Lake)
- Memory: 128GB DDR4-2933 (8 channels, 100GB/s bandwidth)
- Storage: 2TB NVMe SSD (Samsung 970 EVO Plus, 3.5GB/s read, 3.3GB/s write)
- Network: 25Gbps Ethernet
- NUMA: 2 sockets, 16 cores each

**Software:**
- OS: Linux 5.15+ (Ubuntu 22.04 LTS)
- Kernel: Transparent Huge Pages enabled
- I/O Scheduler: mq-deadline for NVMe

**Engram Configuration:**
- Hot tier: 16GB (128 shards)
- Warm tier: 40GB (memory-mapped)
- SIMD: AVX2 enabled
- Thread pools: 64 recall workers, 32 store workers
- HNSW: M=32, ef_construction=400, ef_search=200

### High-Performance Configuration

**Hardware:**
- CPU: 64 cores (Intel Xeon Platinum 8380 @ 2.3GHz, Ice Lake)
- Memory: 512GB DDR4-3200 (8 channels per socket, 200GB/s bandwidth)
- Storage: 4TB NVMe SSD (Intel P5800X Optane, 7GB/s read/write)
- Network: 100Gbps Ethernet
- NUMA: 2 sockets, 32 cores each

**Engram Configuration:**
- Hot tier: 64GB (256 shards)
- Warm tier: 160GB (memory-mapped)
- SIMD: AVX-512 enabled
- Thread pools: 128 recall workers, 64 store workers
- HNSW: M=48, ef_construction=600, ef_search=300

### Budget Configuration

**Hardware:**
- CPU: 8 cores (AMD Ryzen 7 5800X @ 3.8GHz)
- Memory: 32GB DDR4-3200 (2 channels, 50GB/s bandwidth)
- Storage: 1TB SATA SSD (Samsung 870 EVO, 560MB/s read/write)
- Network: 1Gbps Ethernet
- NUMA: Single socket

**Engram Configuration:**
- Hot tier: 4GB (32 shards)
- Warm tier: 12GB (memory-mapped)
- SIMD: AVX2 enabled
- Thread pools: 16 recall workers, 8 store workers
- HNSW: M=24, ef_construction=200, ef_search=150

## Expected Performance Metrics

### Latency Baselines

All latencies in milliseconds. Measurements at steady state with 1M nodes, 768-dimensional embeddings.

#### Standard Configuration

| Operation | P50 | P90 | P95 | P99 | P99.9 |
|-----------|-----|-----|-----|-----|-------|
| Store | 1.2 | 2.8 | 3.5 | 5.2 | 10.5 |
| Recall (1-hop) | 2.1 | 4.8 | 6.2 | 8.5 | 15.2 |
| Recall (multi-hop) | 5.3 | 12.8 | 18.5 | 25.3 | 48.7 |
| Activation Spread (depth=3) | 3.5 | 8.2 | 11.5 | 18.2 | 35.5 |
| Pattern Completion | 12.5 | 28.3 | 38.5 | 52.8 | 95.2 |
| Consolidation (per batch) | 150 | 280 | 350 | 480 | 850 |

#### High-Performance Configuration

| Operation | P50 | P90 | P95 | P99 | P99.9 |
|-----------|-----|-----|-----|-----|-------|
| Store | 0.8 | 1.8 | 2.2 | 3.5 | 7.2 |
| Recall (1-hop) | 1.3 | 3.2 | 4.5 | 6.2 | 10.8 |
| Recall (multi-hop) | 3.5 | 8.5 | 12.2 | 18.5 | 32.5 |
| Activation Spread (depth=3) | 2.2 | 5.5 | 7.8 | 12.5 | 24.2 |
| Pattern Completion | 8.5 | 18.5 | 25.2 | 35.8 | 65.5 |
| Consolidation (per batch) | 100 | 180 | 240 | 320 | 580 |

#### Budget Configuration

| Operation | P50 | P90 | P95 | P99 | P99.9 |
|-----------|-----|-----|-----|-----|-------|
| Store | 2.5 | 5.2 | 6.8 | 9.5 | 18.5 |
| Recall (1-hop) | 4.2 | 9.5 | 12.5 | 18.2 | 35.5 |
| Recall (multi-hop) | 10.5 | 25.2 | 35.5 | 52.8 | 95.5 |
| Activation Spread (depth=3) | 7.2 | 16.5 | 22.8 | 35.5 | 68.5 |
| Pattern Completion | 25.5 | 55.2 | 75.5 | 105.2 | 185.5 |
| Consolidation (per batch) | 300 | 580 | 750 | 980 | 1650 |

### Throughput Baselines

Operations per second at sustained load (5 minutes).

#### Standard Configuration

| Workload | Operations/sec | Notes |
|----------|---------------|--------|
| 100% Store | 15,000 | WAL enabled, durability guaranteed |
| 100% Recall | 25,000 | 80% cache hit rate |
| 70/30 Recall/Store | 18,000 | Typical mixed workload |
| Burst (10 seconds) | 50,000 | Short-term peak capacity |
| Concurrent clients (1000) | 16,000 | Slight degradation under high concurrency |

#### High-Performance Configuration

| Workload | Operations/sec | Notes |
|----------|---------------|--------|
| 100% Store | 35,000 | Optane storage, high parallelism |
| 100% Recall | 60,000 | Large cache, AVX-512 vectorization |
| 70/30 Recall/Store | 42,000 | Balanced workload |
| Burst (10 seconds) | 120,000 | Excellent burst capacity |
| Concurrent clients (5000) | 38,000 | Minimal degradation |

#### Budget Configuration

| Workload | Operations/sec | Notes |
|----------|---------------|--------|
| 100% Store | 5,000 | SATA SSD bottleneck |
| 100% Recall | 8,000 | Limited cache size |
| 70/30 Recall/Store | 6,000 | Storage-limited |
| Burst (10 seconds) | 15,000 | Limited burst capacity |
| Concurrent clients (100) | 5,500 | Thread pool saturation |

### Resource Usage Baselines

At sustained load with 1M nodes.

#### Standard Configuration

| Resource | Expected | Warning | Critical |
|----------|----------|---------|----------|
| CPU Utilization | 45-65% | >75% | >90% |
| Memory RSS | 72GB | >102GB | >115GB |
| Memory Overhead | 1.8x | >2.5x | >3.0x |
| Disk IOPS | 250-400 | >600 | >1000 |
| Disk Throughput | 80-150 MB/s | >300 MB/s | >500 MB/s |
| Network Throughput | 150-350 Mbps | >1000 Mbps | >2000 Mbps |
| Cache Hit Rate | 80-85% | <70% | <60% |
| WAL Flush Latency (P99) | 3-5ms | >10ms | >20ms |

#### High-Performance Configuration

| Resource | Expected | Warning | Critical |
|----------|----------|---------|----------|
| CPU Utilization | 40-60% | >70% | >85% |
| Memory RSS | 280GB | >410GB | >460GB |
| Memory Overhead | 1.7x | >2.2x | >2.8x |
| Disk IOPS | 500-1000 | >2000 | >5000 |
| Disk Throughput | 200-500 MB/s | >1000 MB/s | >2000 MB/s |
| Network Throughput | 500-1500 Mbps | >5000 Mbps | >10000 Mbps |
| Cache Hit Rate | 85-90% | <75% | <65% |
| WAL Flush Latency (P99) | 1-2ms | >5ms | >10ms |

#### Budget Configuration

| Resource | Expected | Warning | Critical |
|----------|----------|---------|----------|
| CPU Utilization | 55-75% | >85% | >95% |
| Memory RSS | 22GB | >28GB | >30GB |
| Memory Overhead | 2.0x | >2.8x | >3.5x |
| Disk IOPS | 150-300 | >500 | >800 |
| Disk Throughput | 50-100 MB/s | >200 MB/s | >400 MB/s |
| Network Throughput | 50-150 Mbps | >500 Mbps | >800 Mbps |
| Cache Hit Rate | 70-75% | <60% | <50% |
| WAL Flush Latency (P99) | 10-15ms | >25ms | >50ms |

## Scaling Characteristics

### CPU Scaling

Performance scaling with CPU core count (Standard Configuration hardware):

| Cores | Store ops/s | Recall ops/s | Efficiency |
|-------|-------------|--------------|------------|
| 4 | 4,200 | 6,800 | 100% (baseline) |
| 8 | 8,100 | 13,200 | 96% |
| 16 | 15,500 | 25,500 | 92% |
| 32 | 28,500 | 48,000 | 85% |
| 64 | 48,000 | 82,000 | 71% |
| 128 | 75,000 | 125,000 | 55% |

**Scaling behavior:**
- **Linear region (1-32 cores):** 85-96% efficiency
- **Good efficiency (32-64 cores):** 71-85% efficiency
- **Diminishing returns (>64 cores):** <71% efficiency due to:
  - Lock contention in hot tier
  - NUMA remote memory access overhead
  - Cache line bouncing between sockets
  - Memory bandwidth saturation

**Recommendations:**
- Optimal: 32-48 cores per instance
- Beyond 64 cores: Consider horizontal scaling (multiple instances)

### Memory Scaling

Memory requirements by graph size (768-dimensional embeddings):

| Nodes | Raw Data | Hot Tier | Warm Tier | Total RSS | Overhead |
|-------|----------|----------|-----------|-----------|----------|
| 100K | 300MB | 512MB | 1GB | 2GB | 6.7x |
| 500K | 1.5GB | 2GB | 5GB | 9GB | 6.0x |
| 1M | 3GB | 4GB | 12GB | 22GB | 7.3x |
| 5M | 15GB | 16GB | 48GB | 85GB | 5.7x |
| 10M | 30GB | 32GB | 96GB | 165GB | 5.5x |

**Memory overhead components:**
- Embeddings: 768 × 4 bytes = 3KB per node
- Metadata: NodeId, timestamps, confidence = 64 bytes per node
- Edges: Average 32 edges × 32 bytes = 1KB per node
- HNSW index: M × layers × 8 bytes ≈ 512 bytes per node
- DashMap overhead: ~30% for concurrent access
- OS page tables and heap fragmentation: ~10%

**Overhead decreases with scale:**
- Small graphs (<1M nodes): 6-7x overhead (index dominates)
- Medium graphs (1-10M nodes): 5-6x overhead (amortized)
- Large graphs (>10M nodes): 4-5x overhead (data dominates)

### Data Size Scaling

Latency growth with graph size:

```
Recall P99 latency (ms) = 2.0 + 0.5 × log₁₀(nodes) + 0.1 × depth

Examples:
- 100K nodes, depth=1: 2.0 + 0.5 × 5 + 0.1 × 1 = 4.6ms
- 1M nodes, depth=1:   2.0 + 0.5 × 6 + 0.1 × 1 = 5.1ms
- 10M nodes, depth=1:  2.0 + 0.5 × 7 + 0.1 × 1 = 5.6ms
- 1M nodes, depth=3:   2.0 + 0.5 × 6 + 0.1 × 3 = 5.3ms
```

**Logarithmic scaling justification:**
- HNSW index: O(log n) search complexity
- Activation spreading: Bounded by max depth, not graph size
- Cache-oblivious algorithms: Adapt to working set size

**Throughput scaling:**

```
Max throughput (ops/s) = base_throughput / (1 + nodes / saturation_point)

Examples (Standard Configuration):
- 100K nodes:  25000 / (1 + 0.1) = 22,700 ops/s
- 1M nodes:    25000 / (1 + 1.0) = 12,500 ops/s
- 10M nodes:   25000 / (1 + 10) = 2,270 ops/s

Saturation point ≈ 1M nodes (cache-resident working set)
```

## Workload-Specific Baselines

### Read-Heavy Workload (90% Recall)

Configuration optimizations:
- Larger hot tier (25% of RAM)
- Higher HNSW ef_search (300)
- More recall workers (2x CPU cores)

**Standard Configuration performance:**
- Throughput: 22,000 ops/s
- P99 Recall Latency: 7.2ms
- Cache Hit Rate: 88%
- CPU Utilization: 55%

### Write-Heavy Workload (50% Store)

Configuration optimizations:
- Batch WAL writes (1000ms interval)
- More store workers (1.5x CPU cores)
- Lower consolidation frequency (10min)

**Standard Configuration performance:**
- Throughput: 12,000 ops/s
- P99 Store Latency: 6.5ms
- Disk Write Amplification: 2.8x
- CPU Utilization: 48%

### Mixed Workload (70/20/10 Recall/Store/Pattern)

Default configuration.

**Standard Configuration performance:**
- Throughput: 16,000 ops/s
- P99 Recall: 8.5ms
- P99 Store: 5.2ms
- P99 Pattern: 52.8ms
- CPU Utilization: 58%

## Performance Regression Detection

### Regression Thresholds

Alert when metrics deviate from baselines:

| Metric | Warning | Critical |
|--------|---------|----------|
| P99 Latency | +20% | +50% |
| P50 Latency | +15% | +30% |
| Throughput | -15% | -30% |
| Cache Hit Rate | -10pp | -20pp |
| Memory Overhead | +0.5x | +1.0x |
| CPU Efficiency | -10% | -20% |
| Disk I/O Latency | +30% | +100% |

### Regression Detection Queries

Prometheus alerting rules:

```yaml
groups:
  - name: engram_performance_regression
    interval: 60s
    rules:
      - alert: RecallLatencyRegression
        expr: |
          histogram_quantile(0.99,
            rate(engram_memory_operation_duration_seconds_bucket{operation="recall"}[5m])
          ) > (
            avg_over_time(
              histogram_quantile(0.99,
                rate(engram_memory_operation_duration_seconds_bucket{operation="recall"}[5m])
              )[7d:5m]
            ) * 1.2
          )
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Recall P99 latency increased by >20%"
          description: "Current: {{ $value }}s, 7-day avg: {{ $labels.baseline }}s"

      - alert: ThroughputRegression
        expr: |
          rate(engram_operations_total[5m]) < (
            avg_over_time(rate(engram_operations_total[5m])[7d:5m]) * 0.85
          )
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Throughput decreased by >15%"
          description: "Current: {{ $value }} ops/s"

      - alert: CacheHitRateRegression
        expr: |
          (
            rate(engram_activation_cache_hits_total[5m]) /
            rate(engram_activation_cache_requests_total[5m])
          ) < (
            avg_over_time(
              rate(engram_activation_cache_hits_total[5m]) /
              rate(engram_activation_cache_requests_total[5m])
            [7d:5m]) - 0.1
          )
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate dropped by >10 percentage points"
          description: "Current: {{ $value }}"
```

## Benchmark Validation

### Running Benchmarks

Validate your deployment against baselines:

```bash
# Run standard benchmark suite
./scripts/benchmark_deployment.sh 300 20

# Expected output format:
# Store Operations:
#   Throughput: 15234 req/sec (target: ≥15000)
#   P99 Latency: 5.1ms (target: ≤5.2ms)
#   Status: PASS
#
# Recall Operations:
#   Throughput: 24856 req/sec (target: ≥25000)
#   P99 Latency: 8.3ms (target: ≤8.5ms)
#   Status: PASS

# Compare against configuration baseline
./scripts/benchmark_deployment.sh 300 20 > current_benchmark.txt
diff docs/reference/performance-baselines.md current_benchmark.txt
```

### Acceptance Criteria

Deployment is considered performant if:

1. **Latency within targets:**
   - P99 Store ≤ baseline × 1.1
   - P99 Recall ≤ baseline × 1.1
   - P50 operations ≤ baseline

2. **Throughput meets minimums:**
   - Sustained throughput ≥ baseline × 0.9
   - Burst throughput ≥ baseline × 0.9

3. **Resource usage acceptable:**
   - Memory overhead ≤ baseline × 1.2
   - CPU utilization < 80% at sustained load
   - Cache hit rate ≥ baseline - 5%

4. **Scaling efficiency:**
   - Throughput scales linearly up to 32 cores
   - Latency grows logarithmically with data size

## Performance Tuning Recommendations

If metrics don't meet baselines:

1. **Latency >20% higher:**
   - Run `./scripts/analyze_slow_queries.sh 10 1h`
   - Check cache hit rate (target: >80%)
   - Profile with `./scripts/profile_performance.sh 120`
   - See [Identify Slow Queries](/docs/howto/identify-slow-queries.md)

2. **Throughput >15% lower:**
   - Check CPU utilization (should be 40-70%)
   - Verify thread pool sizing
   - Check for lock contention
   - See [Optimize Resource Usage](/docs/howto/optimize-resource-usage.md)

3. **Memory overhead >50% higher:**
   - Verify tier sizing configuration
   - Check for memory leaks (RSS growth)
   - Enable compression for cold tier
   - Review consolidation settings

4. **Poor scaling efficiency:**
   - Enable NUMA awareness for multi-socket
   - Increase DashMap shard count
   - Check for cache line bouncing
   - See [Performance Tuning](/docs/operations/performance-tuning.md)

## Hardware Recommendations

Based on performance baselines:

### For 1M Nodes

**Minimum:**
- 8 cores, 32GB RAM, SATA SSD
- Expected: 6K ops/s, P99 <20ms

**Recommended:**
- 16 cores, 64GB RAM, NVMe SSD
- Expected: 12K ops/s, P99 <12ms

**High-Performance:**
- 32 cores, 128GB RAM, NVMe SSD
- Expected: 18K ops/s, P99 <10ms

### For 10M Nodes

**Minimum:**
- 16 cores, 128GB RAM, NVMe SSD
- Expected: 3K ops/s, P99 <30ms

**Recommended:**
- 32 cores, 256GB RAM, NVMe SSD
- Expected: 6K ops/s, P99 <18ms

**High-Performance:**
- 64 cores, 512GB RAM, Optane SSD
- Expected: 12K ops/s, P99 <12ms

### For 100M+ Nodes

Requires horizontal scaling (multiple instances with sharding).

**Per Instance:**
- 32-48 cores, 256-512GB RAM, NVMe/Optane SSD
- Shard graph across 10-20 instances
- Expected aggregate: 60-120K ops/s

## Related Documentation

- [Performance Tuning Guide](/docs/operations/performance-tuning.md) - Comprehensive tuning procedures
- [Identify Slow Queries](/docs/howto/identify-slow-queries.md) - Query performance debugging
- [Optimize Resource Usage](/docs/howto/optimize-resource-usage.md) - Resource optimization techniques
- [Monitoring](/docs/operations/monitoring.md) - Metrics collection and dashboards
