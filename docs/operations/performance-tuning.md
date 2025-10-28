# Performance Tuning Guide

This guide provides comprehensive instructions for optimizing Engram's performance in production environments. It covers profiling, bottleneck identification, configuration tuning, and benchmarking.

## Quick Start

Run the automated tuning wizard to generate an optimized configuration:

```bash
./scripts/tune_config.sh /etc/engram/engram.toml
systemctl restart engram
./scripts/benchmark_deployment.sh 60 10

```

## Performance Targets

Engram is designed to meet the following performance targets on standard hardware (32 cores, 128GB RAM, NVMe SSD):

| Metric | Target | Measurement Command |
|--------|--------|-------------------|
| P99 Recall Latency | <10ms | `curl http://localhost:9090/api/v1/query -d 'query=histogram_quantile(0.99, engram_memory_operation_duration_seconds_bucket{operation="recall"})'` |
| Sustained Throughput | 10,000 ops/s | `curl http://localhost:9090/api/v1/query -d 'query=rate(engram_operations_total[1m])'` |
| Memory Overhead | <2x data size | `curl http://localhost:9090/api/v1/query -d 'query=engram_memory_bytes / engram_data_bytes'` |
| CPU Utilization | <70% at sustained load | `top` or `htop` |

## Performance Profiling

### Comprehensive System Profile

Use the profiling script to capture CPU, memory, NUMA, cache, and I/O metrics:

```bash
./scripts/profile_performance.sh 120 ./profile-output

```

This generates:

- `cpu_profile.txt` - CPU hotspots and cache miss statistics

- `memory_usage.txt` - RSS, VSZ, and huge page usage

- `numa_memory.txt` - NUMA node memory distribution

- `cache_stats.txt` - L1/L2/L3 cache efficiency

- `thread_states.txt` - Thread contention analysis

- `io_stats.txt` - Disk I/O patterns

- `latency_percentiles.txt` - Query latency percentiles (P50, P90, P95, P99, P99.9)

- `bottleneck_report.txt` - Automated bottleneck identification with recommended actions

### Interpreting Profile Results

**Cache Efficiency:**

- Target: <10% L3 cache miss rate

- High miss rate (>15%): Increase `activation.prefetch_distance` or optimize data layout

**Memory Bandwidth:**

- Target: <80% of theoretical bandwidth

- Saturation: Enable compression, reduce embedding dimensions, or add memory channels

**NUMA Locality:**

- Target: >90% local node accesses

- High remote accesses: Enable `numa.prefer_local_node = true` and pin threads

**Lock Contention:**

- Target: <100us average contention time

- High contention: Increase `storage.hot_tier_shards` to 2-4x CPU cores

## Common Performance Issues

### 1. High Recall Latency (P99 >10ms)

**Symptoms:**

- Slow recall operations

- High query latency percentiles

- Poor user experience

**Root Causes:**

1. **Insufficient hot tier size** - Frequently accessed memories evicted to slower tiers

2. **Poor HNSW index configuration** - Low connectivity reduces recall quality

3. **Cache thrashing** - Working set exceeds cache size

**Solutions:**

```bash
# Increase hot tier to 20% of RAM
./scripts/tune_config.sh /etc/engram/engram.toml

# Or manually edit config:
# [storage]
# hot_tier_size_mb = 4096  # 4GB for frequently accessed data

# Rebuild HNSW index with better parameters
# [hnsw_index]
# M = 32  # Higher connectivity improves recall
# ef_construction = 400
# ef_search = 200

# Restart to apply changes
systemctl restart engram

```

**Verification:**

```bash
# Check cache hit rate (target: >80%)
curl http://localhost:9090/api/v1/query \
  -d 'query=rate(engram_activation_cache_hits_total[5m]) / rate(engram_activation_cache_requests_total[5m])'

# Monitor P99 latency improvement
./scripts/analyze_slow_queries.sh 10 5m

```

### 2. Memory Bandwidth Saturation

**Symptoms:**

- Memory bandwidth >80% of theoretical maximum

- CPU waiting on memory (high IPC stalls)

- Poor scaling with increased load

**Root Causes:**

1. **Unoptimized memory access patterns** - Poor spatial locality

2. **No SIMD utilization** - Scalar operations for embeddings

3. **Poor NUMA locality** - Cross-socket memory accesses

**Solutions:**

```bash
# Enable huge pages for reduced TLB misses
echo 1024 > /proc/sys/vm/nr_hugepages

# Pin to NUMA nodes for local memory access
numactl --cpunodebind=0 --membind=0 engram-server

# Enable SIMD vectorization
# [activation]
# simd_enabled = true
# simd_batch_size = 8
# avx_version = "avx2"  # or "avx512" if available

# Enable cold tier compression to reduce bandwidth
# [storage]
# cold_tier_compression = "zstd"
# cold_tier_compression_level = 3

```

**Verification:**

```bash
# Check memory bandwidth utilization
sudo perf stat -e memory_bandwidth_read,memory_bandwidth_write -p $(pgrep engram) sleep 10

# Verify NUMA locality
numastat -p $(pgrep engram)

# Check SIMD instruction usage
sudo perf stat -e fp_arith_inst_retired.256b_packed_single -p $(pgrep engram) sleep 10

```

### 3. Lock Contention

**Symptoms:**

- High CPU usage with low throughput

- Many threads in BLOCKED state

- Non-linear scaling with thread count

**Root Causes:**

1. **Insufficient DashMap shards** - Too few shards for concurrency level

2. **Hot spots in data** - Popular keys creating contention

3. **Read-write lock conflicts** - Writers blocking readers

**Solutions:**

```toml
# Increase sharding in config
[storage]
hot_tier_shards = 128  # 2-4x CPU cores

# Use NUMA-aware sharding
[numa]
numa_aware = true
socket_memory_mb = [65536]  # Split memory across NUMA nodes

```

**Verification:**

```bash
# Profile lock contention
sudo perf record -g -e lock:contention_begin -p $(pgrep engram) -- sleep 10
sudo perf report

# Check thread utilization
top -H -p $(pgrep engram)

```

### 4. Slow Store Operations

**Symptoms:**

- High store latency (P99 >5ms)

- Low write throughput

- WAL flush bottleneck

**Root Causes:**

1. **WAL synchronous writes** - Every write waits for fsync

2. **Small I/O operations** - Not batching writes

3. **Slow storage** - HDD or saturated SSD

**Solutions:**

```toml
# Enable write batching
[wal]
buffer_size_mb = 128
flush_interval_ms = 1000  # Batch for 1 second
sync_mode = "batch"  # Group commits

# Use direct I/O for large writes
use_direct_io = true

```

**Verification:**

```bash
# Monitor WAL flush latency
curl http://localhost:9090/api/v1/query \
  -d 'query=histogram_quantile(0.99, rate(engram_wal_flush_duration_seconds_bucket[5m]))'

# Check disk I/O patterns
iostat -x 1

```

## Profiling Workflow

### Step 1: Baseline Performance

Establish baseline metrics before optimization:

```bash
# Run benchmark suite
./scripts/benchmark_deployment.sh 300 20
cp benchmark-*/benchmark_report.txt baseline_report.txt

# Capture baseline profile
./scripts/profile_performance.sh 120 ./profile-baseline

```

### Step 2: Identify Bottlenecks

Use profiling tools to find performance bottlenecks:

```bash
# Automated bottleneck detection
./scripts/profile_performance.sh 120 ./profile-current
cat ./profile-current/bottleneck_report.txt

# Analyze slow queries
./scripts/analyze_slow_queries.sh 10 1h full

# Manual investigation with perf
sudo perf top -p $(pgrep engram)

```

### Step 3: Apply Tuning

Generate optimized configuration based on workload:

```bash
# Automated tuning wizard
./scripts/tune_config.sh /etc/engram/engram.toml

# Review generated config
cat /etc/engram/engram.toml

# Restart Engram
systemctl restart engram

```

### Step 4: Verify Improvements

Benchmark and compare against baseline:

```bash
# Re-run benchmark
./scripts/benchmark_deployment.sh 300 20
cp benchmark-*/benchmark_report.txt tuned_report.txt

# Compare results
diff baseline_report.txt tuned_report.txt

# Capture improved profile
./scripts/profile_performance.sh 120 ./profile-tuned
diff ./profile-baseline/bottleneck_report.txt ./profile-tuned/bottleneck_report.txt

```

## Hardware-Specific Tuning

### Intel Xeon (Cascade Lake and newer)

**Advantages:**

- AVX-512 support (2x embedding throughput)

- High memory bandwidth

- Large L3 cache

**Configuration:**

```toml
[activation]
avx_version = "avx512"
simd_batch_size = 16  # AVX-512 handles 16 floats per instruction
prefetch_distance = 16  # Larger cache allows more prefetch

[storage]
hot_tier_size_mb = 8192  # Leverage large cache
use_huge_pages = true

```

**Verification:**

```bash
# Verify AVX-512 usage
sudo perf stat -e fp_arith_inst_retired.512b_packed_single -p $(pgrep engram) sleep 10

```

### AMD EPYC (Rome and newer)

**Advantages:**

- High memory bandwidth (8 memory channels)

- Many cores (up to 128)

- CCX-based architecture

**Configuration:**

```toml
[thread_pools]
recall_workers = 64  # Leverage high core count
store_workers = 32

[numa]
numa_aware = true
prefer_local_node = true  # Critical for multi-CCX

[activation]
avx_version = "avx2"

```

**Verification:**

```bash
# Check CCX-local memory access
numastat -p $(pgrep engram)

```

### ARM Graviton (Graviton3 and newer)

**Advantages:**

- NEON SIMD instructions

- Larger L1 cache (64KB vs 32KB)

- LSE atomic extensions

**Configuration:**

```toml
[activation]
simd_enabled = true
# Uses NEON intrinsics automatically on ARM

[storage]
hot_tier_shards = 128  # Benefit from LSE atomics

```

## Configuration Reference

### Critical Performance Parameters

**Storage Tier Sizing:**

```toml
[storage]
# Hot tier: Frequently accessed memories (DRAM, lock-free concurrent)
hot_tier_size_mb = 4096  # 10-20% of RAM

# Warm tier: Recently accessed memories (memory-mapped files)
warm_tier_size_mb = 12288  # 30-40% of RAM

# Cold tier: Archived memories (columnar storage with compression)
cold_tier_compression = "zstd"
cold_tier_compression_level = 3

```

**Cache Optimization:**

```toml
[activation]
# Prefetch nodes ahead of traversal (4-16 for L2/L3 cache)
prefetch_distance = 12

# SIMD batch size (8 for AVX2, 16 for AVX-512)
simd_batch_size = 8

# Visit budget per tier (limits cache thrashing)
visit_budget_per_tier = { hot = 3, warm = 2, cold = 1 }

```

**HNSW Index Tuning:**

```toml
[hnsw_index]
# Connectivity (higher = better recall, more memory)
M = 32  # 16-32 for production

# Build quality (higher = better index, slower build)
ef_construction = 400  # 200-400 for production

# Search quality (higher = better recall, slower search)
ef_search = 200  # 100-200 for production

```

**Thread Pool Sizing:**

```toml
[thread_pools]
# Read-heavy: 2x CPU cores for I/O concurrency
recall_workers = 64

# Write-heavy: 1x CPU cores
store_workers = 32

# Background operations
consolidation_workers = 2
background_workers = 4

```

## Monitoring Dashboard Queries

Key metrics to monitor in Prometheus/Grafana:

```promql
# Operation latency percentiles
histogram_quantile(0.99,
  rate(engram_memory_operation_duration_seconds_bucket[5m])
) by (operation)

# Cache hit rate (target: >80%)
rate(engram_activation_cache_hits_total[5m]) /
rate(engram_activation_cache_requests_total[5m])

# Memory tier distribution
engram_storage_tier_memories_count by (tier)

# Throughput by operation
rate(engram_operations_total[5m]) by (operation)

# Error rate (target: <0.1%)
rate(engram_operation_errors_total[5m]) /
rate(engram_operations_total[5m])

```

## Troubleshooting

### Performance Degradation Checklist

When performance degrades, check in order:

1. **System Resources**

   ```bash
   htop
   iostat -x 1
   vmstat 1
   ```

2. **Cache Efficiency**

   ```bash
   sudo perf stat -e cache-misses,cache-references -p $(pgrep engram) sleep 10
   ```

3. **Memory Access Patterns**

   ```bash
   numastat -p $(pgrep engram)
   grep Huge /proc/meminfo
   ```

4. **Application Metrics**

   ```bash
   curl http://localhost:9090/api/v1/query \
     -d 'query=engram_memory_operation_duration_seconds'
   ```

5. **Thread Behavior**

   ```bash
   top -H -p $(pgrep engram)
   ```

### Common Bottleneck Patterns

**Pattern 1: Cache Line Bouncing**

- **Symptoms:** High CPU on multi-socket systems, poor scaling

- **Detection:** `perf c2c record` shows high HITM events

- **Fix:** Enable NUMA awareness, pad atomic fields

**Pattern 2: NUMA Memory Stalls**

- **Symptoms:** 3-4x higher memory latency, uneven CPU utilization

- **Detection:** `numastat` shows high remote accesses

- **Fix:** Pin threads and memory to NUMA nodes

**Pattern 3: SIMD Underutilization**

- **Symptoms:** Low IPC (<1.5), embedding operations dominate profile

- **Detection:** AVX registers unused in `perf report`

- **Fix:** Increase batch size, ensure data alignment

See [/docs/reference/performance-baselines.md](/docs/reference/performance-baselines.md) for expected performance on reference hardware.

See [/docs/howto/identify-slow-queries.md](/docs/howto/identify-slow-queries.md) for detailed query analysis procedures.

See [/docs/howto/optimize-resource-usage.md](/docs/howto/optimize-resource-usage.md) for resource optimization techniques.
