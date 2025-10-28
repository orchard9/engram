# How to Identify Slow Queries

This guide provides step-by-step procedures for identifying, analyzing, and resolving slow queries in Engram.

## Quick Start

Use the automated slow query analyzer:

```bash
# Find queries slower than 10ms in last hour
./scripts/analyze_slow_queries.sh 10 1h

# Full detail mode for deep investigation
./scripts/analyze_slow_queries.sh 10 1h full

# Summary mode for quick overview
./scripts/analyze_slow_queries.sh 10 1h summary
```

## Manual Investigation

### Step 1: Check Current Latencies

Query Prometheus for operation-specific latency percentiles:

```bash
# View P99 latencies by operation type
curl -s http://localhost:9090/api/v1/query \
  -d 'query=histogram_quantile(0.99, rate(engram_memory_operation_duration_seconds_bucket[5m]))' \
  | jq '.data.result[] | {op: .metric.operation, p99: .value[1]}'

# View all percentiles for recall operations
for p in 0.50 0.90 0.95 0.99 0.999; do
  echo -n "P$(echo "$p * 100" | bc | cut -d. -f1): "
  curl -s http://localhost:9090/api/v1/query \
    -d "query=histogram_quantile($p, rate(engram_memory_operation_duration_seconds_bucket{operation=\"recall\"}[5m]))" \
    | jq -r '.data.result[0].value[1]'
done
```

### Step 2: Identify Patterns

Look for these common patterns that indicate performance issues:

#### Time-Based Degradation

Performance degrades during specific time periods (peak hours, after consolidation):

```bash
# Compare latency over time windows
curl -s http://localhost:9090/api/v1/query_range \
  -d 'query=histogram_quantile(0.99, rate(engram_memory_operation_duration_seconds_bucket[5m]))' \
  -d 'start=2025-10-27T00:00:00Z' \
  -d 'end=2025-10-27T23:59:59Z' \
  -d 'step=300' \
  | jq '.data.result[] | {op: .metric.operation, values: .values}'
```

**Common Causes:**
- Memory tier migrations during off-peak hours
- Consolidation background jobs competing for resources
- Cache warmup after restart
- Log rotation or backup operations

#### Size-Based Degradation

Performance correlates with memory space or graph size:

```bash
# Check node count per memory space
curl -s http://localhost:9090/api/v1/query \
  -d 'query=engram_graph_nodes_total' \
  | jq '.data.result[] | {space: .metric.memory_space, nodes: .value[1]}'

# Check if latency correlates with size
curl -s http://localhost:9090/api/v1/query \
  -d 'query=histogram_quantile(0.99, rate(engram_memory_operation_duration_seconds_bucket[5m])) by (memory_space)' \
  | jq '.data.result[] | {space: .metric.memory_space, p99: .value[1]}'
```

**Common Causes:**
- Index quality degradation for large graphs
- Cache hit rate decreases with working set size
- NUMA memory placement suboptimal
- Graph traversal depth exceeds cache-friendly range

#### Operation-Specific Patterns

Only certain operations are slow:

```bash
# Compare operation types
curl -s http://localhost:9090/api/v1/query \
  -d 'query=histogram_quantile(0.99, rate(engram_memory_operation_duration_seconds_bucket[5m])) by (operation)' \
  | jq '.data.result[] | {op: .metric.operation, p99: .value[1]}'
```

**Common Causes:**
- Store operations: WAL flush latency, lock contention
- Recall operations: Index coverage, cache misses
- Activation spreading: Prefetch distance, SIMD underutilization
- Pattern completion: Search depth too high, candidate set too large

### Step 3: Root Cause Analysis

For each slow operation, perform targeted analysis:

#### Cache Analysis

Check cache hit rates and efficiency:

```bash
# Overall cache hit rate (target: >80%)
curl -s http://localhost:9090/api/v1/query \
  -d 'query=rate(engram_activation_cache_hits_total[5m]) / rate(engram_activation_cache_requests_total[5m])' \
  | jq -r '.data.result[0].value[1]'

# Cache hit rate by tier
curl -s http://localhost:9090/api/v1/query \
  -d 'query=rate(engram_activation_cache_hits_total[5m]) / rate(engram_activation_cache_requests_total[5m]) by (tier)' \
  | jq '.data.result[] | {tier: .metric.tier, hit_rate: .value[1]}'
```

**Low cache hit rate (<70%):**
- Increase hot tier size: `storage.hot_tier_size_mb`
- Adjust eviction policy: Check memory space access patterns
- Review prefetch distance: `activation.prefetch_distance`

#### Index State Analysis

Verify HNSW index quality:

```bash
# Average edges per node (target: 16-32)
curl -s http://localhost:9090/api/v1/query \
  -d 'query=avg(engram_graph_edges_per_node)' \
  | jq -r '.data.result[0].value[1]'

# Index coverage percentage
curl -s http://localhost:9090/api/v1/query \
  -d 'query=engram_hnsw_indexed_nodes / engram_graph_nodes_total' \
  | jq -r '.data.result[0].value[1]'
```

**Low connectivity or coverage:**
- Rebuild index with higher M: `hnsw_index.M = 32`
- Increase build quality: `hnsw_index.ef_construction = 400`
- Verify background indexing is enabled

#### Resource Bottleneck Analysis

Check CPU, memory, and I/O utilization:

```bash
# CPU usage by Engram process
top -b -n 1 -p $(pgrep engram) | tail -1 | awk '{print "CPU: " $9 "%"}'

# Memory usage and pressure
ps -p $(pgrep engram) -o pid,vsz,rss,pmem | tail -1

# I/O wait time
iostat -x 1 2 | grep -A 1 "^Device" | tail -1 | awk '{print "I/O Util: " $14 "%"}'

# Engram-specific metrics
curl -s http://localhost:9090/api/v1/query \
  -d 'query=rate(engram_cpu_seconds_total[5m])' \
  | jq '.data.result[0].value[1]'
```

**Resource saturation indicators:**
- CPU >90%: Lock contention or insufficient parallelism
- Memory >90%: Tier eviction too aggressive, OOM risk
- I/O >80%: WAL flush bottleneck, storage tier too slow

#### Concurrency Analysis

Detect lock contention and thread starvation:

```bash
# Thread state distribution
for i in {1..5}; do
  echo "=== Sample $i ==="
  ps -eLo state,comm -p $(pgrep engram) | grep engram | awk '{print $1}' | sort | uniq -c
  sleep 1
done

# Lock wait time from Prometheus
curl -s http://localhost:9090/api/v1/query \
  -d 'query=rate(engram_lock_wait_seconds_total[5m])' \
  | jq '.data.result[] | {lock: .metric.lock_type, wait_time: .value[1]}'
```

**Contention indicators:**
- Many threads in D (disk sleep) state: I/O bottleneck
- Many threads in R (running) state but low throughput: Lock contention
- High lock wait time: Increase shard count or use lock-free structures

### Step 4: Apply Fixes

Based on root cause, apply targeted fixes:

| Root Cause | Configuration Fix | Verification Query |
|------------|------------------|-------------------|
| Low cache hits | `storage.hot_tier_size_mb = 4096` | Cache hit rate >80% |
| Missing indices | `hnsw_index.M = 32, ef_construction = 400` | Index coverage 100% |
| I/O bottleneck | `wal.sync_mode = "batch", wal.flush_interval_ms = 1000` | IOPS <500 |
| Lock contention | `storage.hot_tier_shards = <CPU_CORES * 4>` | CPU linear with load |
| NUMA locality | `numa.prefer_local_node = true` | numastat remote <10% |
| SIMD underutilization | `activation.simd_batch_size = 8, simd_enabled = true` | perf stat AVX events |

## Operation-Specific Troubleshooting

### Slow Recall Operations

**Diagnosis:**

```bash
# Check recall latency breakdown
./scripts/analyze_slow_queries.sh 10 1h full | grep -A 20 "recall"
```

**Common issues and fixes:**

1. **HNSW search quality**
   - Symptom: P99 >20ms, low recall accuracy
   - Fix: Increase `hnsw_index.ef_search` from 100 to 200
   - Verification: Recall quality improves, latency acceptable

2. **Embedding computation overhead**
   - Symptom: High CPU, embedding ops in profile
   - Fix: Enable AVX2/AVX-512 with `activation.avx_version`
   - Verification: `perf stat` shows vectorized instructions

3. **Cache misses on embedding data**
   - Symptom: High L3 cache miss rate
   - Fix: Increase `activation.prefetch_distance` to 12-16
   - Verification: Cache miss rate <10%

### Slow Store Operations

**Diagnosis:**

```bash
# Check WAL flush latency
curl -s http://localhost:9090/api/v1/query \
  -d 'query=histogram_quantile(0.99, rate(engram_wal_flush_duration_seconds_bucket[5m]))' \
  | jq -r '.data.result[0].value[1]'
```

**Common issues and fixes:**

1. **WAL synchronous flushes**
   - Symptom: WAL flush P99 >5ms
   - Fix: `wal.sync_mode = "batch"`, `wal.flush_interval_ms = 1000`
   - Verification: Flush latency <2ms, throughput increases

2. **Lock contention in hot tier**
   - Symptom: Non-linear scaling with concurrency
   - Fix: `storage.hot_tier_shards = <CPU_CORES * 4>`
   - Verification: Linear throughput scaling

3. **Index update overhead**
   - Symptom: Store latency increases with graph size
   - Fix: Defer index updates, batch insertions
   - Verification: Constant store latency regardless of size

### Slow Activation Spreading

**Diagnosis:**

```bash
# Profile activation spreading
./scripts/profile_performance.sh 60 ./profile-activation
grep "activation" ./profile-activation/cpu_profile.txt | head -20
```

**Common issues and fixes:**

1. **Poor prefetch effectiveness**
   - Symptom: Cache miss rate >15% during traversal
   - Fix: Tune `activation.prefetch_distance` (4-16)
   - Verification: Cache miss rate <10%

2. **Traversal depth too high**
   - Symptom: Exponential latency with depth
   - Fix: `activation.max_traversal_depth = 5`, adjust threshold
   - Verification: Bounded latency regardless of graph size

3. **SIMD batch too small**
   - Symptom: Scalar similarity computations
   - Fix: `activation.simd_batch_size = 8`, ensure batch accumulation
   - Verification: Vectorized ops in `perf stat`

### Slow Pattern Completion

**Diagnosis:**

```bash
# Check pattern completion metrics
curl -s http://localhost:9090/api/v1/query \
  -d 'query=histogram_quantile(0.99, rate(engram_memory_operation_duration_seconds_bucket{operation="pattern_completion"}[5m]))' \
  | jq -r '.data.result[0].value[1]'
```

**Common issues and fixes:**

1. **Search space explosion**
   - Symptom: P99 >100ms, increases with graph size
   - Fix: `pattern_completion.max_depth = 3`, `max_candidates = 100`
   - Verification: Bounded search space

2. **Candidate scoring overhead**
   - Symptom: High CPU in scoring phase
   - Fix: Early pruning, top-k optimization
   - Verification: Logarithmic complexity

## Continuous Monitoring

Set up alerts for slow query detection:

```yaml
# Prometheus alert rules
groups:
  - name: engram_performance
    rules:
      - alert: SlowRecallQueries
        expr: histogram_quantile(0.99, rate(engram_memory_operation_duration_seconds_bucket{operation="recall"}[5m])) > 0.010
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Recall P99 latency exceeds 10ms target"
          description: "Current P99: {{ $value }}s, investigate cache hit rate and index quality"

      - alert: SlowStoreQueries
        expr: histogram_quantile(0.99, rate(engram_memory_operation_duration_seconds_bucket{operation="store"}[5m])) > 0.005
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Store P99 latency exceeds 5ms target"
          description: "Current P99: {{ $value }}s, check WAL flush latency and lock contention"
```

## Advanced Debugging Techniques

### Flame Graph Analysis

Generate flame graphs for visual profiling:

```bash
# Capture profile
./scripts/profile_performance.sh 120 ./profile-output

# Generate flame graph (requires flamegraph.pl)
flamegraph.pl ./profile-output/collapsed.txt > ./profile-output/flamegraph.svg

# View in browser
open ./profile-output/flamegraph.svg
```

**Reading flame graphs:**
- Width: Time spent in function (including children)
- Height: Call stack depth
- Color: Random (for differentiation only)
- Hot paths: Wide boxes at top of stack

### Cache-to-Cache Transfer Analysis

Detect false sharing and cache line bouncing:

```bash
# Capture cache-to-cache transfers
sudo perf c2c record -p $(pgrep engram) -- sleep 30

# Analyze HITM (cache line bouncing)
sudo perf c2c report --stdio > c2c_report.txt

# Look for high "Remote Hitm" percentages
grep "Remote Hitm" c2c_report.txt
```

**Fix false sharing:**
- Pad atomic fields to separate cache lines
- Use thread-local storage for hot counters
- Enable NUMA-aware memory allocation

### NUMA Memory Access Patterns

Analyze NUMA locality:

```bash
# Per-NUMA node memory statistics
numastat -p $(pgrep engram)

# Detailed memory access profile
perf mem record -p $(pgrep engram) -- sleep 30
perf mem report --stdio
```

**Optimize NUMA placement:**
- Local node accesses should be >90%
- Pin threads to NUMA nodes: `numactl --cpunodebind=0 --membind=0`
- Enable `numa.prefer_local_node = true`

## Related Documentation

- [Performance Tuning Guide](/docs/operations/performance-tuning.md) - Comprehensive tuning procedures
- [Optimize Resource Usage](/docs/howto/optimize-resource-usage.md) - Resource optimization techniques
- [Performance Baselines](/docs/reference/performance-baselines.md) - Expected performance metrics
- [Monitoring](/docs/operations/monitoring.md) - Metrics collection and dashboards
