# Capacity Planning

Comprehensive capacity planning guide for Engram deployments. Use this document to predict resource requirements, forecast growth, and optimize infrastructure costs based on workload characteristics.

## Overview

Capacity planning for Engram requires understanding:

1. **Workload Characteristics**: Write/read ratio, operation rate, graph depth
2. **Data Volume**: Number of nodes, edges, embeddings
3. **Performance Requirements**: Latency SLAs, throughput targets
4. **Growth Projections**: Expected data growth rate over time
5. **Cost Constraints**: Budget limitations and optimization opportunities

This guide provides formulas, calculators, and worksheets to accurately predict resource needs.

## Quick Start: Capacity Calculator

The fastest way to estimate resources is using the capacity calculator script:

```bash
# Basic usage (1M nodes, mixed workload)
./scripts/estimate_capacity.sh

# Custom workload (100M nodes, read-heavy, 10K ops/sec)
./scripts/estimate_capacity.sh 100000000 0.15 75 10000 90 300 read-heavy

# See all options
./scripts/estimate_capacity.sh --help
```

The calculator outputs:
- CPU core requirements
- Memory (RAM) requirements by tier
- Storage capacity and IOPS requirements
- Network bandwidth requirements
- Cloud instance recommendations (AWS, GCP, Azure)
- Scaling triggers and optimization tips

## Capacity Planning Formulas

Use these formulas to manually calculate resource requirements or validate calculator outputs.

### Memory Requirements

Memory is the most critical resource for Engram's tiered architecture.

#### Hot Tier Memory (In-Memory)

The hot tier keeps the most frequently accessed nodes in RAM for sub-millisecond access.

```
HOT_TIER_MEMORY = active_nodes * (EMBEDDING_SIZE + NODE_METADATA) * HASH_TABLE_OVERHEAD
                + active_edges * EDGE_OVERHEAD * HASH_TABLE_OVERHEAD
                + activation_pool_size * 8KB

Where:
  EMBEDDING_SIZE = 768 floats * 4 bytes = 3,072 bytes
  NODE_METADATA = 64 bytes (activation, confidence, timestamp, decay)
  EDGE_OVERHEAD = 32 bytes (weight, timestamp, confidence)
  HASH_TABLE_OVERHEAD = 1.5x (DashMap bucket overhead + tombstones)
  active_nodes = total_nodes * active_ratio (typically 0.05-0.15)
  activation_pool_size = ops_per_sec * 0.01 (1% concurrent operations)
```

**Example Calculation:**
```
1,000,000 total nodes
0.1 active ratio (100,000 hot nodes)
50 edges per node (5,000,000 hot edges)
5,000 ops/sec

HOT_TIER_MEMORY = 100,000 * (3,072 + 64) * 1.5
                + 5,000,000 * 32 * 1.5
                + (5,000 * 0.01) * 8,192

                = 470,400,000 + 240,000,000 + 409,600
                = 710,809,600 bytes
                = 678 MB
```

#### Warm Tier Memory (Memory-Mapped)

The warm tier uses memory-mapped files, so memory usage depends on OS page cache behavior.

```
WARM_TIER_MEMORY = mmap_windows * MMAP_PAGE_SIZE * prefetch_ratio
                 + page_cache_size

Where:
  warm_nodes = active_nodes * 3 (warm tier is 3x hot tier)
  mmap_windows = warm_nodes / nodes_per_window (typically 1,000)
  MMAP_PAGE_SIZE = 4,096 bytes
  prefetch_ratio = 0.1 (prefetch 10% of warm tier)
  page_cache_size = warm_tier_file_size * 0.25 (OS keeps 25% in cache)
```

**Example Calculation:**
```
100,000 hot nodes
300,000 warm nodes (3x hot)
1,000 nodes per mmap window

WARM_TIER_MEMORY = (300,000 / 1,000) * 4,096 * 0.1
                 + (300,000 * 3,072) * 0.25

                 = 122,880 + 230,400,000
                 = 230,522,880 bytes
                 = 220 MB
```

#### Cold Tier Memory (Indexes Only)

The cold tier stores only indices in memory, with data compressed on disk.

```
COLD_TIER_MEMORY = bloom_filter_size + skip_list_index_size

Where:
  bloom_filter_size = cold_nodes * 10 bits / 8 (bloom filter for existence checks)
  skip_list_index_size = cold_nodes / 128 * 64 bytes (skip list for range queries)
  cold_nodes = total_nodes - hot_nodes - warm_nodes
```

**Example Calculation:**
```
1,000,000 total nodes
100,000 hot nodes
300,000 warm nodes
600,000 cold nodes

COLD_TIER_MEMORY = (600,000 * 10 / 8) + (600,000 / 128 * 64)
                 = 750,000 + 300,000
                 = 1,050,000 bytes
                 = 1 MB
```

#### Total Memory Requirement

```
TOTAL_MEMORY = HOT_TIER_MEMORY + WARM_TIER_MEMORY + COLD_TIER_MEMORY
             + WAL_BUFFER_MEMORY + RUNTIME_OVERHEAD

Where:
  WAL_BUFFER_MEMORY = wal_buffer_size * 256 bytes (default 10,000 * 256 = 2.5MB)
  RUNTIME_OVERHEAD = 500 MB (tokio runtime, stack, heap fragmentation)
```

**Example Total:**
```
TOTAL_MEMORY = 678 + 220 + 1 + 2.5 + 500
             = 1,401.5 MB
             ≈ 1.4 GB minimum

Recommended (with 25% headroom): 1.75 GB
Production (with 50% headroom): 2.1 GB
```

### CPU Requirements

CPU requirements depend on the workload type and operation rate.

#### Spreading Activation CPU

Spreading activation is the primary CPU-intensive operation.

```
SPREADING_CPU = spreading_ops_per_sec / single_core_throughput
              * (1 + gpu_fallback_ratio * 0.5)

Where:
  single_core_throughput = 2,500 ops/sec (without GPU)
  spreading_ops_per_sec = total_ops_per_sec * read_ratio
  gpu_fallback_ratio = 0.0 (with GPU) or 1.0 (without GPU)
  read_ratio = 0.6 (for mixed), 0.8 (read-heavy), 0.2 (write-heavy)
```

**Example Calculation:**
```
5,000 ops/sec total
0.6 read ratio (mixed workload)
No GPU acceleration

SPREADING_CPU = (5,000 * 0.6) / 2,500 * (1 + 1.0 * 0.5)
              = 3,000 / 2,500 * 1.5
              = 1.8 cores
```

#### Consolidation CPU

Memory consolidation processes nodes periodically to strengthen frequently co-activated patterns.

```
CONSOLIDATION_CPU = consolidation_rate * nodes_per_consolidation / throughput
                  / consolidation_interval

Where:
  consolidation_rate = 1.0 (always run)
  nodes_per_consolidation = active_nodes (process all hot nodes)
  throughput = 50,000 nodes/sec/core
  consolidation_interval = 300 seconds (5 minutes default)
```

**Example Calculation:**
```
100,000 active nodes
300 second interval
50,000 nodes/sec/core throughput

CONSOLIDATION_CPU = 1.0 * 100,000 / 50,000 / 300
                  = 0.0067 cores
                  ≈ negligible (amortized over 5 minutes)
```

#### API Overhead CPU

HTTP/gRPC request handling overhead.

```
API_OVERHEAD_CPU = http_requests_per_sec / 5,000
                 + grpc_requests_per_sec / 8,000

Where:
  http_requests_per_sec = ops_per_sec * http_ratio
  grpc_requests_per_sec = ops_per_sec * grpc_ratio
  5,000 = HTTP requests per core (JSON overhead)
  8,000 = gRPC requests per core (protobuf more efficient)
```

**Example Calculation:**
```
5,000 ops/sec total
100% HTTP (0% gRPC)

API_OVERHEAD_CPU = 5,000 / 5,000 + 0 / 8,000
                 = 1.0 cores
```

#### Total CPU Requirement

```
TOTAL_CPU = SPREADING_CPU + CONSOLIDATION_CPU + API_OVERHEAD_CPU + SYSTEM_OVERHEAD

Where:
  SYSTEM_OVERHEAD = 1.0 core (OS, monitoring, logging)
```

**Example Total:**
```
TOTAL_CPU = 1.8 + 0.0067 + 1.0 + 1.0
          = 3.8 cores minimum

Recommended (NUMA-aligned): 4 cores
With headroom (1.5x): 6 cores → round to 8 cores
```

### Storage Requirements

Storage capacity depends on tier distribution and compression ratios.

#### Warm Tier Storage

Warm tier uses uncompressed storage with version retention.

```
WARM_TIER_STORAGE = warm_nodes * (EMBEDDING_SIZE + NODE_METADATA)
                  + warm_edges * EDGE_OVERHEAD
                  + segment_headers * 4KB
                  * (1 + version_retention_ratio)

Where:
  warm_nodes = active_nodes * 3
  warm_edges = warm_nodes * edges_per_node
  segment_headers = warm_nodes / 10,000 (one header per 10K nodes)
  version_retention_ratio = 0.3 (keep 30% old versions)
```

**Example Calculation:**
```
300,000 warm nodes
50 edges per node (15,000,000 warm edges)
30 segment headers

WARM_TIER_STORAGE = 300,000 * (3,072 + 64)
                  + 15,000,000 * 32
                  + 30 * 4,096
                  * (1 + 0.3)

                  = (940,800,000 + 480,000,000 + 122,880) * 1.3
                  = 1,846,363,744 bytes
                  = 1.72 GB
```

#### Cold Tier Storage

Cold tier uses aggressive compression.

```
COLD_TIER_STORAGE = cold_nodes * EMBEDDING_SIZE * compression_ratio
                  + columnar_index_size

Where:
  cold_nodes = total_nodes - hot_nodes - warm_nodes
  compression_ratio = 0.4 (zstd level 3 achieves 60% compression)
  columnar_index_size = cold_nodes * 128 bytes (skip list + bloom filter)
```

**Example Calculation:**
```
600,000 cold nodes
0.4 compression ratio (zstd level 3)

COLD_TIER_STORAGE = 600,000 * 3,072 * 0.4
                  + 600,000 * 128

                  = 737,280,000 + 76,800,000
                  = 814,080,000 bytes
                  = 776 MB
```

#### Write-Ahead Log (WAL)

WAL size depends on write rate and retention policy.

```
WAL_STORAGE = wal_retention_hours * writes_per_hour * WAL_RECORD_SIZE
            * (1 + wal_compression_ratio)

Where:
  WAL_RECORD_SIZE = 256 bytes (average)
  writes_per_hour = ops_per_sec * write_ratio * 3,600
  wal_retention_hours = retention_days * 24
  wal_compression_ratio = 0.7 (snappy compression, 30% reduction)
```

**Example Calculation:**
```
5,000 ops/sec
0.4 write ratio (mixed workload)
30 days retention

WAL_STORAGE = (30 * 24) * (5,000 * 0.4 * 3,600) * 256 * (1 + 0.7)
            = 720 * 7,200,000 * 256 * 1.7
            = 2,239,488,000,000 bytes
            = 2,086 GB
            ≈ 2.1 TB

Note: This is excessive - typically WAL is compacted/archived frequently
Realistic with daily compaction: 7 days * 297 GB/day = 2.1 GB
```

**Realistic WAL with Compaction:**
```
7 days retention (daily compaction)
WAL_STORAGE = 7 * (5,000 * 0.4 * 3,600 * 24) * 256 * 1.7 / 1,073,741,824
            = 23 GB
```

#### Snapshots

Incremental snapshots for point-in-time recovery.

```
SNAPSHOT_STORAGE = num_snapshots * total_data_size * incremental_ratio

Where:
  num_snapshots = 5 (daily for a week)
  total_data_size = WARM_TIER_STORAGE + COLD_TIER_STORAGE
  incremental_ratio = 0.3 (only 30% changes between snapshots)
```

**Example Calculation:**
```
5 snapshots
1.72 GB warm + 0.776 GB cold = 2.5 GB total

SNAPSHOT_STORAGE = 5 * 2.5 * 0.3
                 = 3.75 GB
```

#### Total Storage Requirement

```
TOTAL_STORAGE = WARM_TIER_STORAGE + COLD_TIER_STORAGE + WAL_STORAGE
              + SNAPSHOT_STORAGE + TEMP_SPACE

Where:
  TEMP_SPACE = max(largest_segment * 2, total_data_size * 0.1)
```

**Example Total:**
```
TOTAL_STORAGE = 1.72 + 0.776 + 23 + 3.75 + (2.5 * 0.1)
              = 29.5 GB minimum

Recommended (with 50% headroom): 44 GB → provision 50 GB
```

### Network Bandwidth

Network requirements depend on client traffic and backup/replication.

```
CLIENT_BANDWIDTH = (request_size_avg * request_rate)
                 + (response_size_avg * request_rate)

BACKUP_BANDWIDTH = backup_size / backup_window_seconds

PEAK_BANDWIDTH = max(CLIENT_BANDWIDTH * 1.5, BACKUP_BANDWIDTH) + CLIENT_BANDWIDTH

Where:
  request_size_avg = 512 bytes (typical query)
  response_size_avg = 4,096 bytes (includes embedding)
  backup_size = TOTAL_STORAGE
  backup_window_seconds = 14,400 (4 hours)
```

**Example Calculation:**
```
5,000 requests/sec
50 GB backup size
4 hour backup window

CLIENT_BANDWIDTH = (512 + 4,096) * 5,000 / 1,048,576
                 = 22 MB/s

BACKUP_BANDWIDTH = 50 * 1,024 / 14,400
                 = 3.6 MB/s

PEAK_BANDWIDTH = max(22 * 1.5, 3.6) + 22
               = 33 + 22
               = 55 MB/s
               = 440 Mbps

Recommended: 1 Gbps link (headroom for spikes)
```

## Workload-Specific Capacity Models

Different workload types have different resource profiles.

### Write-Heavy Workload (80/20 Write/Read)

Characteristics:
- Continuous memory creation (ingestion pipeline)
- High WAL growth rate
- Rapid hot tier expansion
- Moderate spreading CPU usage

**Resource Profile:**
```
CPU: 0.9x baseline (less spreading, more consolidation)
Memory: 1.2x baseline (more buffering, faster hot tier growth)
Storage: 1.5x baseline (larger WAL, more frequent snapshots)
Network: 1.0x baseline (requests are smaller than responses)
```

**Optimization Tips:**
- Increase WAL buffer size to 50 MB (from 2.5 MB default)
- Batch writes aggressively (100-1,000 operations per batch)
- Lower hot tier threshold (0.2 vs 0.3) to evict faster
- Pre-allocate warm tier segments
- Run compaction more frequently (hourly vs every 5 minutes)

### Read-Heavy Workload (20/80 Write/Read)

Characteristics:
- Spreading activation dominant
- Stable memory size
- Low WAL growth
- High network response traffic

**Resource Profile:**
```
CPU: 1.3x baseline (more spreading, less consolidation)
Memory: 1.0x baseline (stable hot tier)
Storage: 0.5x baseline (minimal WAL growth)
Network: 1.2x baseline (larger response payloads)
```

**Optimization Tips:**
- Maximize hot tier retention (activation threshold 0.4 vs 0.3)
- Enable GPU acceleration for parallel spreading
- Increase activation pool to 100 MB (from default)
- Use read replicas for horizontal scaling (future)
- Pin spreading threads to dedicated CPU cores

### Analytical Workload (Complex Traversals)

Characteristics:
- Deep graph traversals (5-10 hops)
- Large working sets
- Long-running queries
- CPU-intensive pattern completion

**Resource Profile:**
```
CPU: 1.8x baseline (heavy recursive activation)
Memory: 1.5x baseline (large working sets must fit in RAM)
Storage: 0.3x baseline (minimal writes)
Network: 0.8x baseline (fewer but larger queries)
```

**Optimization Tips:**
- Pin hot tier in memory (disable eviction during queries)
- Dedicate 75% of cores to spreading threads
- Increase refractory period to 500ms (from 100ms) for stability
- Batch analytical queries to amortize graph warming
- Pre-warm critical paths before query execution

### Mixed Workload (40/60 Write/Read)

Characteristics:
- Balanced write and read operations
- Moderate consolidation activity
- Typical production usage pattern

**Resource Profile:**
```
CPU: 1.0x baseline (balanced)
Memory: 1.0x baseline (balanced)
Storage: 1.0x baseline (balanced)
Network: 1.0x baseline (balanced)
```

**Optimization Tips:**
- Use adaptive batch controller (auto-tune batch size)
- Schedule consolidation during low-traffic windows (2-6 AM)
- Implement tier rebalancing policies based on access patterns
- Monitor and adjust thresholds dynamically
- Reserve 20% CPU headroom for traffic spikes

## Capacity Planning Worksheet

Use this worksheet to plan capacity for your deployment.

### Step 1: Define Workload Parameters

```
Total Nodes Expected: _____________
Active Ratio (0.05-0.20): _____________
Edges per Node (10-100): _____________
Operations per Second: _____________
Write/Read Ratio: _____________/_____________
Retention Days: _____________
Workload Type: [ ] write-heavy [ ] read-heavy [ ] analytical [ ] mixed
```

### Step 2: Calculate Resource Requirements

Run the capacity calculator:
```bash
./scripts/estimate_capacity.sh \
  [total_nodes] \
  [active_ratio] \
  [edges_per_node] \
  [ops_per_sec] \
  [retention_days] \
  300 \
  [workload_type]
```

Record outputs:
```
CPU Cores Required: _____________
Memory (RAM) Required: _____________
Storage Required: _____________
Network Bandwidth Required: _____________
```

### Step 3: Apply Headroom Multipliers

```
Phase 1 (Initial Deployment, Day 0-7):
  CPU: Required × 2.0 = _____________
  Memory: Required × 2.0 = _____________
  Storage: Required × 2.0 = _____________

Phase 2 (Optimization, Day 7-30):
  CPU: Peak Observed × 1.3 = _____________
  Memory: Peak Observed × 1.3 = _____________
  Storage: Peak Observed × 1.5 = _____________

Phase 3 (Steady State, Day 30+):
  Target 70% avg, 85% peak utilization
  CPU: Required × 1.2 = _____________
  Memory: Required × 1.2 = _____________
  Storage: Required × 1.5 = _____________
```

### Step 4: Select Cloud Instance Type

Based on calculated requirements, select instance type:

**AWS EC2:**
- 4 cores, 16 GB: `m6i.xlarge` (~$150/month)
- 8 cores, 32 GB: `m6i.2xlarge` (~$300/month)
- 16 cores, 64 GB: `m6i.4xlarge` (~$600/month)
- 32 cores, 128 GB: `m6i.8xlarge` (~$1,200/month)

**GCP Compute Engine:**
- 4 cores, 16 GB: `n2-standard-4`
- 8 cores, 32 GB: `n2-standard-8`
- 16 cores, 64 GB: `n2-standard-16`
- 32 cores, 128 GB: `n2-standard-32`

**Azure Virtual Machines:**
- 4 cores, 16 GB: `Standard_D4s_v5`
- 8 cores, 32 GB: `Standard_D8s_v5`
- 16 cores, 64 GB: `Standard_D16s_v5`
- 32 cores, 128 GB: `Standard_D32s_v5`

Selected Instance: _____________
Monthly Cost: _____________

### Step 5: Plan Growth and Scaling Triggers

```
Expected Monthly Growth Rate: _____________ % nodes/month

Scaling Triggers:
  CPU > 70% for 10 minutes → Scale to: _____________
  Memory > 80% for 5 minutes → Scale to: _____________
  Storage > 70% → Expand to: _____________

Forecast 3 months:
  Month 1: _____________ nodes, _____________ cost
  Month 2: _____________ nodes, _____________ cost
  Month 3: _____________ nodes, _____________ cost
```

## Growth Forecasting

Predict future capacity needs based on growth trends.

### Linear Growth Model

```
future_nodes(t) = current_nodes + (growth_rate * t)

Where:
  t = time in days
  growth_rate = nodes added per day (constant)
```

**Example:**
```
Current: 1,000,000 nodes
Growth: 10,000 nodes/day
30 days: 1,000,000 + (10,000 * 30) = 1,300,000 nodes (30% growth)
```

### Exponential Growth Model

```
future_nodes(t) = current_nodes * (1 + growth_rate)^t

Where:
  t = time in days
  growth_rate = daily percentage growth (compounding)
```

**Example:**
```
Current: 1,000,000 nodes
Growth: 2% per day
30 days: 1,000,000 * (1.02)^30 = 1,811,362 nodes (81% growth)
```

### Capacity Exhaustion Prediction

```
days_until_full = (capacity_limit - current_usage) / daily_growth_rate

Alert when: days_until_full < 30
Scale when: days_until_full < 14
Emergency when: days_until_full < 7
```

**Example:**
```
Storage capacity: 100 GB
Current usage: 50 GB
Daily growth: 2 GB/day

days_until_full = (100 - 50) / 2 = 25 days

Action: Schedule scaling within 14 days (before critical threshold)
```

## Cost Optimization Strategies

Reduce infrastructure costs through right-sizing and tier optimization.

### Tier Optimization Cost Savings

```
RAM Cost: $10/GB/month (typical cloud pricing)
SSD Cost: $0.10/GB/month
Object Storage: $0.02/GB/month

Savings by moving 1 TB from hot → warm:
  $10,000/month (RAM) → $100/month (SSD) = $9,900/month saved

Savings by moving 1 TB from warm → cold (archived):
  $100/month (SSD) → $20/month (object storage) = $80/month saved
```

**Optimization Target:**
- Hot tier: <5% of total nodes (current: _____ %)
- Warm tier: 20-30% of total nodes (current: _____ %)
- Cold tier: >65% of total nodes (current: _____ %)

**Actions:**
1. Lower hot tier threshold if ratio >10%
2. Archive cold tier data older than _____ days
3. Enable compression for warm tier nodes inactive >1 hour

### Right-Sizing Cost Savings

```
Over-provisioned (2x required):
  8 cores @ $300/month = $300
  Actual usage: 40% (4 cores worth)
  Waste: $150/month (50%)

Right-sized (1.3x peak):
  4 cores @ $150/month = $150
  Actual usage: 70% (efficient)
  Savings: $150/month (50%)
```

**Right-Sizing Process:**
1. Deploy with 2x calculated baseline
2. Monitor for 7 days
3. Identify peak usage (max 1-hour window)
4. Right-size to 1.3x peak
5. Expected savings: 30-50%

### Spot Instance Cost Savings (AWS/GCP/Azure)

For non-critical or batch workloads:

```
On-Demand Cost: $300/month (m6i.2xlarge)
Spot Cost: $90/month (70% discount)
Savings: $210/month

Use spot instances for:
- Read replicas (with automatic failover)
- Batch analytical queries
- Development/staging environments
- GPU acceleration (70% cheaper)
```

**Risk Mitigation:**
- Implement graceful shutdown on spot interruption
- Fallback to on-demand for critical path
- Diversify across multiple spot pools

## Monitoring Capacity Utilization

Track these metrics to validate capacity planning and adjust forecasts.

### Key Metrics

```
# Resource utilization
engram_cpu_utilization_percent
engram_memory_usage_bytes{tier}
engram_storage_usage_bytes{tier}
engram_network_bandwidth_mbps

# Growth indicators
rate(engram_total_nodes[1d])  # Nodes added per day
rate(engram_storage_usage_bytes[1d])  # Storage growth per day

# Tier distribution
engram_hot_tier_ratio  # Should be <0.10
engram_warm_tier_ratio  # Should be 0.20-0.30
engram_cold_tier_ratio  # Should be >0.65

# Capacity exhaustion
(engram_storage_limit_bytes - engram_storage_usage_bytes)
  / rate(engram_storage_usage_bytes[7d]) / 86400  # Days until full
```

### Capacity Alerts

```yaml
# Alert when capacity will be exhausted within 30 days
- alert: CapacityExhaustionWarning
  expr: |
    (engram_storage_limit_bytes - engram_storage_usage_bytes)
    / rate(engram_storage_usage_bytes[7d]) / 86400 < 30
  annotations:
    action: "Plan capacity expansion within 14 days"

# Alert when capacity will be exhausted within 7 days
- alert: CapacityExhaustionCritical
  expr: |
    (engram_storage_limit_bytes - engram_storage_usage_bytes)
    / rate(engram_storage_usage_bytes[7d]) / 86400 < 7
  annotations:
    action: "Emergency capacity expansion required"
```

## Summary

- Use `/scripts/estimate_capacity.sh` for quick estimates
- Apply formulas for detailed capacity planning
- Choose workload-specific optimizations
- Forecast growth and plan scaling triggers
- Right-size deployments to reduce costs by 30-50%
- Monitor capacity utilization continuously
- Update capacity plans quarterly

## Next Steps

1. Run capacity calculator for your expected workload
2. Complete capacity planning worksheet
3. Select cloud instance type and provision infrastructure
4. Deploy with 2x headroom for initial period
5. Monitor for 7 days and right-size
6. Set up capacity exhaustion alerts
7. Review and adjust quarterly

## Related Documentation

- [Scaling Guide](/operations/scaling.md)
- [How-to: Scale Vertically](/howto/scale-vertically.md)
- [Resource Requirements](/reference/resource-requirements.md)
- [Performance Tuning](/operations/performance-tuning.md)
- [Cost Optimization](/operations/cost-optimization.md)
