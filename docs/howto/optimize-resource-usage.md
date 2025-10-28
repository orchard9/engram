# How to Optimize Resource Usage

This guide provides comprehensive procedures for optimizing CPU, memory, disk I/O, and network resources in Engram deployments.

## Quick Assessment

Run the resource optimization analyzer:

```bash
# Generate optimized configuration for your workload
./scripts/tune_config.sh /etc/engram/engram.toml

# Profile current resource usage
./scripts/profile_performance.sh 120 ./resource-profile

# Review resource utilization
cat ./resource-profile/bottleneck_report.txt
```

## CPU Optimization

### Identifying CPU Bottlenecks

**Check CPU utilization patterns:**

```bash
# Overall CPU usage
top -b -n 1 -p $(pgrep engram) | tail -1 | awk '{print "CPU: " $9 "%"}'

# Per-thread CPU distribution
top -H -b -n 1 -p $(pgrep engram) | head -20

# CPU time by operation type (from Prometheus)
curl -s http://localhost:9090/api/v1/query \
  -d 'query=rate(engram_cpu_seconds_total[5m]) by (operation)' \
  | jq '.data.result[] | {op: .metric.operation, cpu: .value[1]}'
```

**Target utilization:**
- Sustained load: 40-60% CPU
- Peak load: 70-80% CPU
- Alert threshold: >90% CPU for 5+ minutes

### CPU Optimization Strategies

#### 1. Thread Pool Sizing

Configure thread pools based on workload characteristics:

**Read-heavy workload (>70% recall operations):**

```toml
[thread_pools]
# 2x CPU cores for I/O-bound operations
recall_workers = 64  # for 32-core system

# Fewer workers for write operations
store_workers = 16
consolidation_workers = 2
background_workers = 4
```

**Write-heavy workload (>30% store operations):**

```toml
[thread_pools]
# Equal allocation for balanced throughput
recall_workers = 32
store_workers = 32
consolidation_workers = 4
background_workers = 4
```

**Verification:**

```bash
# Check thread utilization
ps -eLo pid,state,comm -p $(pgrep engram) | grep engram | awk '{print $2}' | sort | uniq -c

# R (running) should dominate, D (disk wait) and S (sleeping) indicate underutilization
```

#### 2. SIMD Vectorization

Enable SIMD instructions for embedding operations:

**Auto-detect CPU capabilities:**

```bash
# Check available SIMD instruction sets
lscpu | grep -E "avx|sse"
grep -o 'avx[^ ]*' /proc/cpuinfo | sort -u
```

**Configure SIMD:**

```toml
[activation]
simd_enabled = true

# Choose based on CPU capability
avx_version = "avx512"  # Intel Xeon Cascade Lake+, Ice Lake
# avx_version = "avx2"   # Intel Haswell+, AMD Zen+
# avx_version = "sse4"   # Older processors

# Batch size must match SIMD width
simd_batch_size = 16  # AVX-512: 16 floats
# simd_batch_size = 8  # AVX2: 8 floats
```

**Verification:**

```bash
# Check SIMD instruction usage
sudo perf stat -e fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.512b_packed_single \
  -p $(pgrep engram) sleep 10

# Expect high counts for enabled SIMD width
```

**Expected speedup:**
- AVX2 (8-wide): 4-6x vs scalar
- AVX-512 (16-wide): 8-12x vs scalar

#### 3. Lock Contention Reduction

Reduce lock contention in concurrent data structures:

**Increase DashMap sharding:**

```toml
[storage]
# Formula: 2-4x CPU cores for lock-free concurrency
hot_tier_shards = 128  # 4x for 32-core system
```

**Profile lock contention:**

```bash
# Capture lock contention events
sudo perf record -e lock:contention_begin -g -p $(pgrep engram) -- sleep 30
sudo perf report --stdio | head -50

# Check contention time
curl -s http://localhost:9090/api/v1/query \
  -d 'query=rate(engram_lock_wait_seconds_total[5m])' \
  | jq '.data.result[] | {lock: .metric.lock_type, wait: .value[1]}'
```

**Target:** <100μs average lock wait time

#### 4. NUMA-Aware Scheduling

Optimize for multi-socket systems:

**Detect NUMA topology:**

```bash
numactl --hardware
lscpu | grep NUMA
```

**Configure NUMA awareness:**

```toml
[numa]
numa_aware = true
prefer_local_node = true
interleave_memory = false  # Prefer local allocation

# Memory per NUMA node (optional, auto-detected)
socket_memory_mb = [65536, 65536]  # 64GB per socket
```

**Pin Engram to NUMA node:**

```bash
# Pin to node 0 (CPU + memory)
numactl --cpunodebind=0 --membind=0 engram-server

# Verify locality
numastat -p $(pgrep engram)
```

**Target:** >90% local node memory accesses

## Memory Optimization

### Identifying Memory Bottlenecks

**Check memory utilization:**

```bash
# Overall memory usage
ps -p $(pgrep engram) -o pid,vsz,rss,pmem

# Memory breakdown by tier (from Prometheus)
curl -s http://localhost:9090/api/v1/query \
  -d 'query=engram_storage_tier_bytes by (tier)' \
  | jq '.data.result[] | {tier: .metric.tier, bytes: .value[1]}'

# Memory overhead ratio
curl -s http://localhost:9090/api/v1/query \
  -d 'query=engram_memory_bytes / engram_data_bytes' \
  | jq -r '.data.result[0].value[1]'
```

**Target metrics:**
- Memory overhead: <2x raw data size
- RSS growth: <5% per day (indicating no leaks)
- Huge page usage: >80% of hot tier

### Memory Optimization Strategies

#### 1. Tier Size Tuning

Configure storage tier sizes based on available RAM:

**Formula-based sizing:**

```bash
# Total RAM: 128GB
# Hot tier: 10-20% of RAM = 12.8-25.6GB
# Warm tier: 30-40% of RAM = 38.4-51.2GB
# Cold tier: Disk-based (unlimited)
```

**Configuration:**

```toml
[storage]
# Hot tier: Frequently accessed (lock-free DashMap)
hot_tier_size_mb = 16384  # 16GB for 128GB system

# Warm tier: Recently accessed (memory-mapped files)
warm_tier_size_mb = 40960  # 40GB

# Cold tier: Archived (columnar with compression)
cold_tier_compression = "zstd"
cold_tier_compression_level = 3
```

**Monitor tier distribution:**

```bash
curl -s http://localhost:9090/api/v1/query \
  -d 'query=engram_storage_tier_memories_count by (tier)' \
  | jq '.data.result[] | {tier: .metric.tier, count: .value[1]}'
```

**Optimal distribution:**
- Hot tier: 5-10% of total memories (most frequently accessed)
- Warm tier: 20-30% of total memories
- Cold tier: 60-75% of total memories

#### 2. Huge Page Enablement

Use transparent huge pages to reduce TLB misses:

**Enable huge pages:**

```bash
# Check current huge page configuration
cat /proc/meminfo | grep Huge

# Allocate huge pages (2MB pages, need 8192 for 16GB)
echo 8192 > /proc/sys/vm/nr_hugepages

# Verify allocation
cat /proc/meminfo | grep HugePages_Total
```

**Configure Engram:**

```toml
[storage]
use_huge_pages = true
mmap_populate = true  # Populate pages at mmap time (requires SSD)
```

**Verification:**

```bash
# Check huge page usage by Engram
grep -E "AnonHugePages|ShmemPmdMapped" /proc/$(pgrep engram)/smaps | \
  awk '{sum+=$2} END {print "Huge Pages: " sum/1024 " MB"}'
```

**Expected improvement:**
- TLB miss rate: 50-70% reduction
- Memory access latency: 10-15% reduction

#### 3. Memory Tier Migration Tuning

Optimize tier migration thresholds:

**Current migration rules:**

```toml
[storage.migration]
# Hot -> Warm: Low access frequency
hot_to_warm_accesses = 10  # <10 accesses in 1 hour
hot_to_warm_window_seconds = 3600

# Warm -> Cold: Very low access
warm_to_cold_accesses = 1  # <1 access in 24 hours
warm_to_cold_window_seconds = 86400

# Cold -> Warm: Reactivation
cold_to_warm_accesses = 5  # >5 accesses in 1 hour
cold_to_warm_window_seconds = 3600

# Warm -> Hot: High access
warm_to_hot_accesses = 20  # >20 accesses in 1 hour
warm_to_hot_window_seconds = 3600
```

**Workload-specific tuning:**

- **Read-heavy:** Lower promotion thresholds (faster promotion to hot tier)
- **Write-heavy:** Higher eviction thresholds (keep in hot tier longer)
- **Mixed:** Default thresholds

#### 4. Compression for Cold Tier

Enable compression to reduce memory footprint:

```toml
[storage]
cold_tier_compression = "zstd"

# Compression level trade-off
# 1: Fast compression, lower ratio (good for write-heavy)
# 3: Balanced (default)
# 9: Slow compression, higher ratio (good for read-heavy with infrequent writes)
cold_tier_compression_level = 3
```

**Compression ratios:**
- Embedding data: 1.5-2.0x (limited compressibility)
- Metadata/strings: 3.0-5.0x
- Overall: 2.0-2.5x average

## Disk I/O Optimization

### Identifying I/O Bottlenecks

**Check I/O utilization:**

```bash
# Disk I/O statistics
iostat -x 1 5

# Look for:
# - %util >80%: Disk saturation
# - await >10ms: High latency
# - r/s + w/s: IOPS rate

# Engram process I/O
sudo iotop -b -n 5 -p $(pgrep engram)
```

**From Prometheus:**

```bash
# WAL flush latency (target: <5ms P99)
curl -s http://localhost:9090/api/v1/query \
  -d 'query=histogram_quantile(0.99, rate(engram_wal_flush_duration_seconds_bucket[5m]))' \
  | jq -r '.data.result[0].value[1]'

# Disk I/O rate
curl -s http://localhost:9090/api/v1/query \
  -d 'query=rate(engram_disk_io_bytes_total[5m]) by (operation)' \
  | jq '.data.result[] | {op: .metric.operation, bytes_per_sec: .value[1]}'
```

### I/O Optimization Strategies

#### 1. Write-Ahead Log (WAL) Tuning

Optimize WAL for your storage device:

**SSD/NVMe configuration (low latency):**

```toml
[wal]
buffer_size_mb = 64  # Smaller buffer for low latency
flush_interval_ms = 100  # Flush frequently
sync_mode = "immediate"  # Immediate durability
use_direct_io = true  # Bypass page cache

segment_size_mb = 64
max_segments = 100
```

**HDD configuration (high throughput):**

```toml
[wal]
buffer_size_mb = 128  # Larger buffer for batching
flush_interval_ms = 5000  # Batch for 5 seconds
sync_mode = "batch"  # Group commits
use_direct_io = false  # Use page cache

segment_size_mb = 128
max_segments = 50
```

**Verification:**

```bash
# Monitor WAL performance
curl -s http://localhost:9090/api/v1/query \
  -d 'query=rate(engram_wal_writes_total[5m])' \
  | jq -r '.data.result[0].value[1]'

# Check flush latency
curl -s http://localhost:9090/api/v1/query \
  -d 'query=histogram_quantile(0.99, rate(engram_wal_flush_duration_seconds_bucket[5m]))' \
  | jq -r '.data.result[0].value[1]'
```

**Target metrics:**
- WAL flush P99: <5ms (SSD), <20ms (HDD)
- Write throughput: >1000 ops/sec

#### 2. Memory-Mapped File Optimization

Tune mmap behavior for warm tier:

```toml
[storage]
# Populate pages at mmap time (reduces page faults, requires good I/O)
mmap_populate = true  # Enable for SSD
# mmap_populate = false  # Disable for HDD

# Use huge pages for mmap
use_huge_pages = true

# Readahead window (OS tuning)
# sysctl -w vm.max_map_count=262144
```

**OS-level tuning:**

```bash
# Increase dirty page threshold for better batching
sysctl -w vm.dirty_ratio=40
sysctl -w vm.dirty_background_ratio=10

# Tune readahead for sequential access
blockdev --setra 2048 /dev/nvme0n1
```

#### 3. I/O Scheduler Selection

Choose appropriate I/O scheduler for storage device:

**For SSDs/NVMe:**

```bash
# Use none/noop for low latency
echo none > /sys/block/nvme0n1/queue/scheduler

# Or mq-deadline for better fairness
echo mq-deadline > /sys/block/nvme0n1/queue/scheduler
```

**For HDDs:**

```bash
# Use BFQ for good latency under load
echo bfq > /sys/block/sda/queue/scheduler
```

#### 4. Reduce Write Amplification

Minimize unnecessary writes:

**Configuration:**

```toml
[consolidation]
# Reduce consolidation frequency
interval_seconds = 600  # 10 minutes instead of 5

# Increase batch size
batch_size = 5000  # Process more at once, less frequently

# Higher confidence threshold (fewer updates)
min_confidence_threshold = 0.4  # Up from 0.3

[wal]
# Use group commits to reduce fsync calls
sync_mode = "batch"
flush_interval_ms = 1000
```

**Monitor write amplification:**

```bash
# Bytes written vs data size
curl -s http://localhost:9090/api/v1/query \
  -d 'query=rate(engram_disk_writes_bytes_total[1h]) / rate(engram_data_size_bytes[1h])' \
  | jq -r '.data.result[0].value[1]'
```

**Target:** <3x write amplification

## Network Optimization

### Identifying Network Bottlenecks

**Check network utilization:**

```bash
# Interface statistics
ifstat -i eth0 1 5

# Connection count
ss -s | grep TCP

# Engram network metrics
curl -s http://localhost:9090/api/v1/query \
  -d 'query=rate(engram_network_bytes_total[5m]) by (direction)' \
  | jq '.data.result[] | {dir: .metric.direction, bytes_per_sec: .value[1]}'
```

### Network Optimization Strategies

#### 1. gRPC Connection Pooling

Configure connection pools for client libraries:

**Server-side:**

```toml
[grpc]
max_concurrent_streams = 1000
keepalive_interval_seconds = 30
keepalive_timeout_seconds = 10
max_connection_age_seconds = 3600
```

**Client-side (example Python):**

```python
import grpc

# Connection pool
channel = grpc.insecure_channel(
    'localhost:7432',
    options=[
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ('grpc.keepalive_time_ms', 30000),
        ('grpc.keepalive_timeout_ms', 10000),
        ('grpc.http2.max_pings_without_data', 0),
    ]
)
```

#### 2. Batch Operations

Use batch APIs to reduce network round-trips:

**Instead of individual operations:**

```bash
# BAD: N round-trips
for memory in memories:
    store(memory)
```

**Use batch operations:**

```bash
# GOOD: 1 round-trip
store_batch(memories)
```

**Expected improvement:**
- Latency: 5-10x reduction
- Throughput: 10-20x increase

#### 3. Compression

Enable compression for large payloads:

```toml
[grpc]
compression_enabled = true
compression_algorithm = "gzip"  # or "zstd" for better ratio
compression_level = 6  # 1-9, higher = better compression, more CPU
```

**Trade-off:**
- CPU: +5-10% for compression
- Network: 60-70% reduction (typical)
- Latency: Neutral to slightly better (less data transfer)

## Resource Monitoring Dashboard

Set up comprehensive resource monitoring:

```promql
# CPU Utilization
rate(engram_cpu_seconds_total[5m])

# Memory Usage
engram_memory_bytes

# Memory Overhead Ratio
engram_memory_bytes / engram_data_bytes

# Cache Hit Rate
rate(engram_activation_cache_hits_total[5m]) / rate(engram_activation_cache_requests_total[5m])

# Disk I/O
rate(engram_disk_io_bytes_total[5m]) by (operation)

# Network Traffic
rate(engram_network_bytes_total[5m]) by (direction)

# Thread Pool Utilization
engram_thread_pool_active_threads / engram_thread_pool_total_threads by (pool)
```

## Optimization Workflow

### 1. Baseline Assessment

Capture baseline resource usage:

```bash
# Run comprehensive profile
./scripts/profile_performance.sh 300 ./baseline

# Capture metrics snapshot
curl -s http://localhost:9090/api/v1/query \
  -d 'query=engram_memory_bytes' > baseline_metrics.json
```

### 2. Identify Top Resource Consumer

Analyze profile results:

```bash
# Check bottleneck report
cat ./baseline/bottleneck_report.txt

# Look for:
# - CPU >70%: Thread pool or SIMD optimization needed
# - Memory >80%: Tier sizing or compression needed
# - I/O >80%: WAL tuning or storage upgrade needed
```

### 3. Apply Targeted Optimization

Based on bottleneck, apply one optimization at a time:

```bash
# Example: High CPU usage
./scripts/tune_config.sh /etc/engram/engram.toml
# Review and edit config
vi /etc/engram/engram.toml
# Restart
systemctl restart engram
```

### 4. Measure Improvement

Verify optimization impact:

```bash
# Re-profile
./scripts/profile_performance.sh 300 ./optimized

# Compare
diff ./baseline/bottleneck_report.txt ./optimized/bottleneck_report.txt

# Benchmark
./scripts/benchmark_deployment.sh 300 20
```

### 5. Iterate

Repeat for next bottleneck until all metrics within targets.

## Optimization Checklist

- [ ] CPU utilization 40-70% at sustained load
- [ ] Memory overhead <2x raw data size
- [ ] Cache hit rate >80%
- [ ] Disk I/O P99 latency <10ms
- [ ] WAL flush P99 <5ms
- [ ] NUMA local accesses >90%
- [ ] SIMD instructions enabled and utilized
- [ ] Thread pool sized for workload
- [ ] Huge pages allocated and used
- [ ] Lock contention <100μs average

## Related Documentation

- [Performance Tuning Guide](/docs/operations/performance-tuning.md) - Comprehensive tuning procedures
- [Identify Slow Queries](/docs/howto/identify-slow-queries.md) - Query performance debugging
- [Performance Baselines](/docs/reference/performance-baselines.md) - Expected performance metrics
- [Monitoring](/docs/operations/monitoring.md) - Metrics collection and dashboards
