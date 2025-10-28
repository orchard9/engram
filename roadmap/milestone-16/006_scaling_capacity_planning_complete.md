# Task 006: Scaling Strategies & Capacity Planning — pending

**Priority:** P1 (High)
**Estimated Effort:** 2 days
**Dependencies:** Tasks 001 (Deployment), 003 (Monitoring)
**Reviewed By:** systems-architecture-optimizer (Margo Seltzer)
**Review Date:** 2025-10-24

## Objective

Define vertical and horizontal scaling procedures with capacity planning guidance. Enable operators to predict resource needs, scale proactively, and optimize infrastructure costs while meeting SLAs. Provide deterministic capacity models based on Engram's tiered storage architecture and cognitive workload patterns.

## Key Deliverables

- `/scripts/estimate_capacity.sh` - Capacity planning calculator with workload profiling
- `/docs/operations/scaling.md` - Complete scaling guide with decision matrices
- `/docs/operations/capacity-planning.md` - Capacity planning worksheet with formulas
- `/docs/howto/scale-vertically.md` - Step-by-step vertical scaling procedures
- `/docs/reference/resource-requirements.md` - Resource specifications and sizing tables

## Technical Specifications

### Memory Hierarchy and Resource Model

**Fundamental Resource Constants:**
```
EMBEDDING_SIZE = 768 * 4 bytes = 3,072 bytes per node
NODE_METADATA = 64 bytes (activation, confidence, timestamp, decay)
EDGE_OVERHEAD = 32 bytes (weight, timestamp, confidence)
HASH_TABLE_OVERHEAD = 1.5x (DashMap bucket overhead + tombstones)
MMAP_PAGE_SIZE = 4,096 bytes
WAL_RECORD_SIZE = 256 bytes (average)
COMPACTION_WORKSPACE = 2x largest segment size
```

### Comprehensive Capacity Planning Formulas

**1. Memory Requirements (RAM):**
```
HOT_TIER_MEMORY = active_nodes * (EMBEDDING_SIZE + NODE_METADATA) * HASH_TABLE_OVERHEAD
                + active_edges * EDGE_OVERHEAD * HASH_TABLE_OVERHEAD
                + activation_pool_size * 8KB  # Pre-allocated activation records

WARM_TIER_MEMORY = mmap_windows * MMAP_PAGE_SIZE * prefetch_ratio
                 + page_cache_size  # OS-managed, typically 25% of warm tier size

COLD_TIER_MEMORY = bloom_filter_size + skip_list_index_size
                 = (total_cold_nodes * 10 bits / 8) + (total_cold_nodes / 128 * 64 bytes)

WAL_BUFFER_MEMORY = wal_buffer_size * WAL_RECORD_SIZE  # Default: 10,000 * 256 = 2.5MB

TOTAL_MEMORY = HOT_TIER_MEMORY + WARM_TIER_MEMORY + COLD_TIER_MEMORY
             + WAL_BUFFER_MEMORY + RUNTIME_OVERHEAD

Where:
- active_nodes: Nodes with activation > 0.1 in last hour
- prefetch_ratio: Fraction of warm tier prefetched (default: 0.1)
- RUNTIME_OVERHEAD: Stack, heap fragmentation, tokio runtime (~500MB base)
```

**2. CPU Requirements (Cores):**
```
SPREADING_CPU = spreading_ops_per_sec / 2500  # Single core handles ~2,500 ops/sec
              * (1 + gpu_fallback_ratio * 0.5)  # GPU fallback adds 50% overhead

CONSOLIDATION_CPU = consolidation_rate * nodes_per_consolidation / 50000
                  # Single core processes ~50K nodes/sec during consolidation

COMPACTION_CPU = compaction_frequency * segment_size / (compaction_throughput * 3600)
               # Amortized over hourly compaction windows

API_OVERHEAD_CPU = http_requests_per_sec / 5000  # HTTP endpoint overhead
                 + grpc_requests_per_sec / 8000   # gRPC more efficient

TOTAL_CPU = ceiling(SPREADING_CPU + CONSOLIDATION_CPU + COMPACTION_CPU + API_OVERHEAD_CPU)

Constraints:
- Maximum useful cores: 32 (beyond this, lock contention dominates)
- NUMA considerations: Prefer even core counts (2, 4, 8, 16) for NUMA symmetry
```

**3. Storage Requirements (Disk):**
```
HOT_TIER_STORAGE = 0  # Memory-only, no persistent storage

WARM_TIER_STORAGE = warm_nodes * (EMBEDDING_SIZE + NODE_METADATA)
                  + warm_edges * EDGE_OVERHEAD
                  + segment_headers * 4KB
                  * (1 + version_retention_ratio)  # Keep old versions

COLD_TIER_STORAGE = cold_nodes * EMBEDDING_SIZE * compression_ratio
                  + columnar_index_size
                  Where compression_ratio = 0.4 (zstd level 3)

WAL_STORAGE = wal_retention_hours * writes_per_hour * WAL_RECORD_SIZE
            * (1 + wal_compression_ratio)  # Usually 0.7 with snappy

SNAPSHOT_STORAGE = num_snapshots * total_data_size * 0.3  # Incremental snapshots

TOTAL_STORAGE = WARM_TIER_STORAGE + COLD_TIER_STORAGE + WAL_STORAGE
              + SNAPSHOT_STORAGE + TEMP_SPACE

Where TEMP_SPACE = max(largest_segment * 2, total_data_size * 0.1)
```

**4. Network Bandwidth Requirements:**
```
REPLICATION_BANDWIDTH = write_throughput * replication_factor  # Future: distributed mode
BACKUP_BANDWIDTH = backup_size / backup_window_seconds
CLIENT_BANDWIDTH = (request_size_avg * request_rate) + (response_size_avg * request_rate)

PEAK_BANDWIDTH = max(REPLICATION_BANDWIDTH, BACKUP_BANDWIDTH) + CLIENT_BANDWIDTH * 1.5
```

### Vertical Scaling Procedures

**CPU Scaling Decision Matrix:**
| Current Utilization | P99 Latency | Action | Implementation |
|---------------------|-------------|---------|----------------|
| <50% | <10ms | No action | Monitor for trends |
| 50-70% | <10ms | Plan scaling | Schedule maintenance window |
| 70-85% | 10-50ms | Scale immediately | Add 2x cores (up to 8) |
| 70-85% | >50ms | Scale urgently | Add 4x cores (up to 16) |
| >85% | Any | Emergency scale | Maximum available cores + investigate bottleneck |

**Memory Scaling Decision Matrix:**
| Memory Pressure | Tier Distribution | Action | Implementation |
|-----------------|-------------------|---------|----------------|
| <60% | Balanced | No action | Monitor growth rate |
| 60-75% | Hot-heavy (>50%) | Optimize | Increase eviction rate to warm tier |
| 75-85% | Balanced | Scale RAM | Add 50% more RAM |
| 75-85% | Cold-heavy | Migrate | Move cold data to disk, reduce mmap windows |
| >85% | Any | Emergency | Double RAM + immediate tier rebalancing |

**Storage Scaling Triggers:**
| Metric | Threshold | Action | Procedure |
|--------|-----------|--------|-----------|
| Disk utilization | >70% | Expand volume | Add 2x current size |
| WAL size | >10GB | Increase compaction | Reduce WAL retention, increase compaction frequency |
| IOPS utilization | >80% | Upgrade disk type | Move to NVMe or increase IOPS allocation |
| Snapshot count | >10 | Prune snapshots | Keep only last 5 + weekly archives |

### Capacity Planning Calculator Implementation

**`/scripts/estimate_capacity.sh`:**
```bash
#!/bin/bash
set -euo pipefail

# Input parameters
EXPECTED_NODES="${1:-1000000}"        # Total nodes expected
ACTIVE_RATIO="${2:-0.1}"              # Fraction of nodes active (hot tier)
EDGES_PER_NODE="${3:-50}"             # Average edges per node
OPS_PER_SEC="${4:-5000}"              # Target operations per second
RETENTION_DAYS="${5:-30}"             # Data retention period
CONSOLIDATION_INTERVAL="${6:-300}"     # Seconds between consolidations

# Constants from Engram architecture
EMBEDDING_SIZE=3072                    # 768 floats * 4 bytes
NODE_METADATA=64                       # Activation, confidence, timestamp, decay
EDGE_OVERHEAD=32                       # Weight, timestamp, confidence
HASH_OVERHEAD=1.5                      # DashMap overhead factor
WAL_RECORD=256                         # Average WAL record size
COMPRESSION_RATIO=0.4                  # Cold tier compression with zstd

# Calculate tier distribution
HOT_NODES=$(echo "$EXPECTED_NODES * $ACTIVE_RATIO" | bc -l | cut -d. -f1)
WARM_NODES=$(echo "$EXPECTED_NODES * $ACTIVE_RATIO * 3" | bc -l | cut -d. -f1)  # 3x hot
COLD_NODES=$(echo "$EXPECTED_NODES - $HOT_NODES - $WARM_NODES" | bc -l | cut -d. -f1)

TOTAL_EDGES=$(echo "$EXPECTED_NODES * $EDGES_PER_NODE" | bc -l | cut -d. -f1)
HOT_EDGES=$(echo "$HOT_NODES * $EDGES_PER_NODE" | bc -l | cut -d. -f1)

# Memory calculations (in MB)
HOT_MEMORY=$(echo "($HOT_NODES * ($EMBEDDING_SIZE + $NODE_METADATA) + \
             $HOT_EDGES * $EDGE_OVERHEAD) * $HASH_OVERHEAD / 1048576" | bc -l)

WARM_MEMORY=$(echo "$WARM_NODES * $EMBEDDING_SIZE * 0.1 / 1048576" | bc -l)  # 10% prefetch

COLD_MEMORY=$(echo "($COLD_NODES * 10 / 8 + $COLD_NODES * 64 / 128) / 1048576" | bc -l)

ACTIVATION_POOL=$(echo "$OPS_PER_SEC * 0.01 * 8192 / 1048576" | bc -l)  # 1% concurrent * 8KB

RUNTIME_OVERHEAD=500  # MB - Base runtime, tokio, system

TOTAL_MEMORY=$(echo "$HOT_MEMORY + $WARM_MEMORY + $COLD_MEMORY + \
              $ACTIVATION_POOL + $RUNTIME_OVERHEAD" | bc -l)

# CPU calculations
SPREADING_CORES=$(echo "$OPS_PER_SEC / 2500" | bc -l)
CONSOLIDATION_CORES=$(echo "$HOT_NODES / $CONSOLIDATION_INTERVAL / 50000" | bc -l)
API_CORES=$(echo "$OPS_PER_SEC / 5000" | bc -l)
TOTAL_CORES=$(echo "$SPREADING_CORES + $CONSOLIDATION_CORES + $API_CORES + 1" | bc -l)

# Storage calculations (in GB)
WARM_STORAGE=$(echo "$WARM_NODES * ($EMBEDDING_SIZE + $NODE_METADATA) * 1.3 / 1073741824" | bc -l)
COLD_STORAGE=$(echo "$COLD_NODES * $EMBEDDING_SIZE * $COMPRESSION_RATIO / 1073741824" | bc -l)
WAL_STORAGE=$(echo "$OPS_PER_SEC * 86400 * $RETENTION_DAYS * $WAL_RECORD * 0.7 / 1073741824" | bc -l)
SNAPSHOT_STORAGE=$(echo "($WARM_STORAGE + $COLD_STORAGE) * 5 * 0.3" | bc -l)  # 5 snapshots, 30% incremental
TOTAL_STORAGE=$(echo "$WARM_STORAGE + $COLD_STORAGE + $WAL_STORAGE + $SNAPSHOT_STORAGE" | bc -l)

# Format output
echo "======================================"
echo "Engram Capacity Planning Estimate"
echo "======================================"
echo ""
echo "Input Parameters:"
echo "  Expected Nodes: $(printf "%'d" $EXPECTED_NODES)"
echo "  Active Ratio: ${ACTIVE_RATIO} ($(printf "%'d" $HOT_NODES) hot nodes)"
echo "  Edges per Node: ${EDGES_PER_NODE}"
echo "  Target Ops/sec: $(printf "%'d" $OPS_PER_SEC)"
echo "  Retention Days: ${RETENTION_DAYS}"
echo ""
echo "Tier Distribution:"
echo "  Hot Tier:  $(printf "%'d" $HOT_NODES) nodes (in-memory)"
echo "  Warm Tier: $(printf "%'d" $WARM_NODES) nodes (mmap)"
echo "  Cold Tier: $(printf "%'d" $COLD_NODES) nodes (compressed)"
echo ""
echo "Resource Requirements:"
echo "  CPU Cores: $(printf "%.1f" $TOTAL_CORES) cores"
echo "    - Spreading: $(printf "%.1f" $SPREADING_CORES) cores"
echo "    - Consolidation: $(printf "%.1f" $CONSOLIDATION_CORES) cores"
echo "    - API/System: $(printf "%.1f" $API_CORES) cores"
echo "    - Recommended: $(echo "scale=0; $TOTAL_CORES * 1.5" | bc) cores (with headroom)"
echo ""
echo "  Memory: $(printf "%.1f" $TOTAL_MEMORY) MB"
echo "    - Hot Tier: $(printf "%.1f" $HOT_MEMORY) MB"
echo "    - Warm Tier: $(printf "%.1f" $WARM_MEMORY) MB"
echo "    - Cold Tier: $(printf "%.1f" $COLD_MEMORY) MB"
echo "    - Activation Pool: $(printf "%.1f" $ACTIVATION_POOL) MB"
echo "    - Runtime: ${RUNTIME_OVERHEAD} MB"
echo "    - Recommended: $(echo "scale=0; $TOTAL_MEMORY * 1.25 / 1024" | bc) GB (with headroom)"
echo ""
echo "  Storage: $(printf "%.1f" $TOTAL_STORAGE) GB"
echo "    - Warm Tier: $(printf "%.1f" $WARM_STORAGE) GB"
echo "    - Cold Tier: $(printf "%.1f" $COLD_STORAGE) GB (compressed)"
echo "    - WAL: $(printf "%.1f" $WAL_STORAGE) GB"
echo "    - Snapshots: $(printf "%.1f" $SNAPSHOT_STORAGE) GB"
echo "    - Recommended: $(echo "scale=0; $TOTAL_STORAGE * 1.5" | bc) GB (with headroom)"
echo ""
echo "Scaling Recommendations:"
if (( $(echo "$TOTAL_CORES > 8" | bc -l) )); then
    echo "  ⚠ CPU: Consider horizontal scaling beyond 8 cores"
fi
if (( $(echo "$TOTAL_MEMORY > 32768" | bc -l) )); then
    echo "  ⚠ Memory: Large memory footprint - ensure NUMA-aware allocation"
fi
if (( $(echo "$HOT_NODES > 10000000" | bc -l) )); then
    echo "  ⚠ Hot Tier: Consider sharding hot tier across multiple instances"
fi
echo "  ✓ Deploy with these minimum resources for baseline performance"
echo "  ✓ Monitor actual utilization and adjust based on workload"
echo ""
echo "Cost Optimization Tips:"
echo "  • Reduce hot tier size by tuning eviction threshold"
echo "  • Use spot instances for read replicas (future)"
echo "  • Archive cold tier to object storage after ${RETENTION_DAYS} days"
echo "  • Increase consolidation interval to reduce CPU usage"
```

### Resource Utilization Patterns by Workload Type

**1. Write-Heavy Workload (Ingestion):**
```
Characteristics:
- 80% writes, 20% reads
- Continuous memory creation
- High WAL growth

Resource Profile:
- CPU: Moderate (consolidation dominates)
- Memory: High (hot tier grows rapidly)
- Storage: Very High (WAL + tier migration)
- Network: Moderate

Optimization:
- Increase WAL buffer size
- Batch writes aggressively
- Tune hot→warm eviction threshold lower (0.3 → 0.2)
- Pre-allocate warm tier segments
```

**2. Read-Heavy Workload (Retrieval):**
```
Characteristics:
- 20% writes, 80% reads
- Spreading activation dominant
- Cache-friendly access patterns

Resource Profile:
- CPU: High (spreading computation)
- Memory: Moderate (stable hot tier)
- Storage: Low (minimal WAL growth)
- Network: High (response traffic)

Optimization:
- Maximize hot tier retention
- Enable GPU acceleration for spreading
- Increase activation pool size
- Use read replicas for scaling
```

**3. Analytical Workload (Pattern Mining):**
```
Characteristics:
- Complex graph traversals
- Deep spreading activation
- Pattern completion operations

Resource Profile:
- CPU: Very High (recursive activation)
- Memory: High (large working sets)
- Storage: Low
- Network: Low

Optimization:
- Dedicated CPU cores for spreading
- Pin hot tier in memory (disable eviction)
- Increase refractory period for stability
- Batch analytical queries
```

**4. Mixed Workload (Typical Production):**
```
Characteristics:
- 40% writes, 60% reads
- Periodic consolidation spikes
- Variable access patterns

Resource Profile:
- CPU: Moderate with spikes
- Memory: Moderate, growing slowly
- Storage: Moderate
- Network: Moderate

Optimization:
- Use adaptive batch controller
- Schedule consolidation during low traffic
- Implement tier rebalancing policies
- Monitor and adjust dynamically
```

### Cost Optimization Strategies

**1. Right-Sizing Strategy:**
```
Phase 1 (Initial Deployment):
- Start with 2x calculated baseline
- Monitor actual utilization for 7 days
- Identify usage patterns and peaks

Phase 2 (Optimization):
- Reduce resources to 1.3x peak observed
- Implement auto-scaling policies
- Fine-tune tier thresholds

Phase 3 (Steady State):
- Target 70% average utilization
- 85% peak utilization threshold
- Automated scaling triggers

Savings: 30-50% vs over-provisioning
```

**2. Tier Optimization Strategy:**
```
Hot Tier Minimization:
- Target: <5% of total nodes in hot tier
- Method: Aggressive eviction (activation < 0.2)
- Savings: $500/TB/month (RAM vs SSD cost)

Warm Tier Compression:
- Enable compression for nodes inactive >1 hour
- Use zstd level 3 (40% compression, low CPU)
- Savings: 60% storage cost reduction

Cold Tier Archival:
- Move to object storage after 30 days
- Access via async restoration
- Savings: $20/TB/month vs $100/TB/month SSD
```

**3. Compute Optimization:**
```
CPU Scheduling:
- Consolidation: Run during 2-6 AM low traffic window
- Compaction: Spread throughout day, limit to 1 core
- Batch operations: Accumulate and process in bursts

GPU Offloading:
- Use spot GPU instances for batch spreading
- Fallback to CPU during spot interruption
- Savings: 70% for spreading computation

Workload Isolation:
- Dedicate cores to critical path (spreading)
- Use CPU affinity to reduce context switching
- Improvement: 20% throughput increase
```

### Performance Prediction Models

**Linear Scaling Model (up to 8 cores):**
```
throughput(cores) = base_throughput * cores * efficiency(cores)
where efficiency(cores) = 1.0 - (cores - 1) * 0.05

Example:
- 1 core: 2,500 ops/sec * 1.0 = 2,500 ops/sec
- 4 cores: 2,500 * 4 * 0.85 = 8,500 ops/sec
- 8 cores: 2,500 * 8 * 0.65 = 13,000 ops/sec
```

**Memory Latency Model:**
```
latency_ms = base_latency * tier_multiplier * size_factor

where:
- base_latency = 0.1ms (hot tier)
- tier_multiplier: hot=1, warm=10, cold=100
- size_factor = 1 + log10(nodes_activated / 1000)

Example (10,000 nodes activated):
- Hot: 0.1 * 1 * 2 = 0.2ms
- Warm: 0.1 * 10 * 2 = 2ms
- Cold: 0.1 * 100 * 2 = 20ms
```

**Storage Throughput Model:**
```
write_throughput_MB/s = min(
    disk_iops * 4KB / 1024,  # IOPS limit
    disk_bandwidth_MB/s,      # Bandwidth limit
    wal_buffer_size / wal_flush_interval
)

read_throughput_MB/s = min(
    disk_iops * avg_read_size / 1024,
    disk_bandwidth_MB/s,
    cache_hit_rate * memory_bandwidth + (1 - cache_hit_rate) * disk_bandwidth
)
```

### Monitoring Metrics for Scaling Decisions

**Key Metrics to Track:**
```
# CPU Saturation
engram_cpu_utilization_percent{core}
engram_spreading_queue_depth
engram_consolidation_lag_seconds

# Memory Pressure
engram_memory_usage_bytes{tier}
engram_eviction_rate{from_tier,to_tier}
engram_allocation_failures_total

# Storage Saturation
engram_storage_usage_bytes{tier}
engram_wal_size_bytes
engram_compaction_pending_segments

# Throughput Indicators
engram_operations_per_second{operation}
engram_p99_latency_ms{operation}
engram_tier_miss_rate{tier}
```

**Scaling Alert Rules:**
```yaml
- alert: ScaleUpCPURequired
  expr: |
    (avg_over_time(engram_cpu_utilization_percent[5m]) > 70)
    and
    (rate(engram_spreading_queue_depth[5m]) > 0)
  for: 10m
  annotations:
    action: "Add 2x CPU cores or investigate bottleneck"

- alert: ScaleUpMemoryRequired
  expr: |
    (engram_memory_usage_bytes{tier="hot"} / engram_memory_limit_bytes) > 0.8
  for: 5m
  annotations:
    action: "Add 50% more RAM or increase tier eviction rate"

- alert: ScaleUpStorageRequired
  expr: |
    predict_linear(engram_storage_usage_bytes[6h], 86400) > engram_storage_limit_bytes * 0.9
  for: 30m
  annotations:
    action: "Storage will fill within 24h - expand volume or archive data"
```

### Operational Procedures for Scaling

**Vertical Scaling Procedure (Kubernetes):**
```bash
#!/bin/bash
# Scale Engram deployment vertically

# 1. Check current resources
kubectl get deployment engram -o yaml | grep -A5 resources

# 2. Update resource limits
kubectl set resources deployment engram \
  --limits=cpu=8,memory=16Gi \
  --requests=cpu=4,memory=8Gi

# 3. Trigger rolling update
kubectl rollout restart deployment engram

# 4. Monitor rollout
kubectl rollout status deployment engram

# 5. Verify new resources
kubectl top pods -l app=engram
```

**Vertical Scaling Procedure (Docker):**
```bash
#!/bin/bash
# Scale Engram container vertically

# 1. Stop current container gracefully
docker stop --time 30 engram

# 2. Start with new resources
docker run -d \
  --name engram \
  --cpus="8" \
  --memory="16g" \
  --memory-reservation="8g" \
  -v engram-data:/data \
  engram:latest

# 3. Verify resources
docker stats engram --no-stream
```

**Tier Rebalancing Procedure:**
```bash
#!/bin/bash
# Rebalance storage tiers when memory pressure is high

# 1. Check current tier distribution
curl -s http://localhost:7432/api/v1/admin/storage/stats | jq .

# 2. Adjust eviction threshold
curl -X POST http://localhost:7432/api/v1/admin/storage/eviction \
  -H "Content-Type: application/json" \
  -d '{"hot_threshold": 0.2, "warm_threshold": 0.05}'

# 3. Trigger immediate eviction sweep
curl -X POST http://localhost:7432/api/v1/admin/storage/evict-now

# 4. Monitor tier migration
watch -n 5 'curl -s http://localhost:7432/api/v1/admin/storage/stats | jq .tier_distribution'

# 5. Verify memory usage decreased
curl -s http://localhost:7432/metrics | jq .memory
```

### Capacity Forecasting and Trend Analysis

**Growth Prediction Model:**
```
future_nodes(t) = current_nodes * (1 + growth_rate)^t
future_memory(t) = future_nodes(t) * bytes_per_node * tier_distribution

Where:
- t = time in days
- growth_rate = observed daily growth (typically 0.01-0.05)
- tier_distribution = [0.1 hot, 0.3 warm, 0.6 cold]

Example 30-day forecast:
current: 1M nodes, 0.02 daily growth
30 days: 1M * 1.02^30 = 1.81M nodes
Memory: 1.81M * 3KB * 0.1 = 543MB hot tier growth
```

**Capacity Planning Worksheet:**
```
1. Measure current state (Day 0):
   - Node count by tier
   - Operation rate (ops/sec)
   - Growth rate (nodes/day)

2. Project 30/60/90 day requirements:
   - Apply growth model
   - Add 30% safety margin
   - Account for seasonal patterns

3. Identify scaling triggers:
   - When will CPU hit 70%?
   - When will memory hit 80%?
   - When will storage hit 70%?

4. Plan scaling events:
   - Schedule maintenance windows
   - Order hardware (if bare metal)
   - Update budgets
```

### NUMA-Aware Memory Allocation

**NUMA Architecture Considerations:**
```
Modern servers have Non-Uniform Memory Access:
- Each CPU socket has local memory (fast access)
- Remote memory access is 1.5-2x slower
- Critical for systems >32GB RAM

NUMA Optimization for Engram:
1. Pin hot tier to NUMA node 0
2. Spread warm tier across nodes
3. Use numactl for process binding

Example deployment:
numactl --membind=0 --cpunodebind=0 engram \
  --hot-tier-size=16GB \
  --numa-aware=true
```

**NUMA Monitoring:**
```bash
# Check NUMA topology
numactl --hardware

# Monitor NUMA statistics
numastat engram

# Verify memory allocation
cat /proc/$(pgrep engram)/numa_maps | head -20
```

### Cache Line Optimization for Scaling

**CPU Cache Hierarchy Impact:**
```
L1 Cache: 64KB, 4 cycles (per core)
L2 Cache: 256KB, 12 cycles (per core)
L3 Cache: 8-32MB, 40 cycles (shared)
RAM: 128GB, 100-300 cycles

Cache-Optimized Data Layout:
- Node struct: 64-byte aligned (1 cache line)
- Hot paths: Prefetch next nodes
- Avoid false sharing between cores

Performance Impact:
- Good layout: 2,500 ops/sec/core
- Poor layout: 800 ops/sec/core (3x slower)
```

**Cache-Friendly Batching:**
```
Batch Size Selection:
- L1 fit: 8 nodes (64KB / 8KB per node)
- L2 fit: 32 nodes (256KB / 8KB per node)
- L3 fit: 1,024 nodes (8MB / 8KB per node)

Adaptive batching scales with cache:
- Few nodes: Use L1 batch size
- Many nodes: Use L3 batch size
- Massive graphs: Stream from RAM
```

### Disaster Recovery Scaling Considerations

**Recovery Time Objectives (RTO) by Scale:**
```
Data Size | Cold Start | Warm Start | Hot Standby
----------|------------|------------|-------------
<1GB      | 30 seconds | 10 seconds | <1 second
1-10GB    | 2 minutes  | 30 seconds | <5 seconds
10-100GB  | 10 minutes | 2 minutes  | <30 seconds
100GB-1TB | 1 hour     | 10 minutes | <2 minutes
>1TB      | 4 hours    | 30 minutes | <5 minutes

Scaling impact on recovery:
- WAL replay: 100MB/sec (NVMe SSD)
- Index rebuild: 50K nodes/sec/core
- Memory warmup: 1GB/sec from disk
```

**Backup Scaling Strategy:**
```
Incremental Backup Sizing:
daily_backup_size = daily_new_nodes * 3KB + daily_modified * 1KB

Full Backup Frequency:
- <10GB: Weekly full backup
- 10-100GB: Bi-weekly full backup
- >100GB: Monthly full backup

Parallel Backup for Scale:
- Shard data into 1GB segments
- Backup segments in parallel
- Merge during restore
```

### Auto-Scaling Configuration

**Kubernetes HPA Configuration:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: engram-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: engram
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: engram_spreading_queue_depth
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100  # Double pods
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50  # Halve pods
        periodSeconds: 300
```

**AWS Auto Scaling Configuration:**
```json
{
  "AutoScalingGroupName": "engram-asg",
  "MinSize": 1,
  "MaxSize": 10,
  "DesiredCapacity": 2,
  "TargetGroupARNs": ["arn:aws:elasticloadbalancing:..."],
  "HealthCheckType": "ELB",
  "HealthCheckGracePeriod": 300,
  "MixedInstancesPolicy": {
    "InstancesDistribution": {
      "OnDemandPercentageAboveBaseCapacity": 30,
      "SpotAllocationStrategy": "capacity-optimized"
    },
    "LaunchTemplate": {
      "Overrides": [
        {"InstanceType": "m6i.2xlarge"},
        {"InstanceType": "m6a.2xlarge"},
        {"InstanceType": "m5.2xlarge"}
      ]
    }
  }
}
```

### Capacity Planning for Specific Use Cases

**1. LLM Memory Augmentation:**
```
Requirements:
- 10M memories per agent
- 100ms retrieval latency
- 1,000 concurrent agents

Capacity:
- Nodes: 10M * 1,000 = 10B total
- Hot tier: 100M nodes (1% active)
- Memory: 100M * 3KB = 300GB RAM
- CPU: 40 cores (100K ops/sec)
- Storage: 10B * 3KB * 0.4 = 12TB
```

**2. Knowledge Graph Queries:**
```
Requirements:
- 100M entities
- 1B relationships
- Graph traversals up to 5 hops

Capacity:
- Memory: Keep 10% in hot tier = 30GB
- CPU: 16 cores for traversal
- Specialized: GPU for parallel BFS
- Storage: 500GB with indices
```

**3. Time-Series Pattern Memory:**
```
Requirements:
- 1M events/minute ingestion
- 7-day hot window
- Pattern detection queries

Capacity:
- Hot tier: 7 * 24 * 60 * 1M = 10B events
- Memory: Requires sharding (>1TB)
- CPU: 64 cores across 4 nodes
- Storage: 30TB with compression
```

### Operational Runbook for Scaling Events

**Pre-Scaling Checklist:**
```
□ Current metrics baseline captured
□ Backup completed successfully
□ Maintenance window scheduled
□ Rollback plan documented
□ Team notified
□ Monitoring dashboards ready
```

**Scaling Execution Steps:**
```
1. Enable maintenance mode
   curl -X POST http://localhost:7432/api/v1/admin/maintenance/enable

2. Capture current state
   ./scripts/capture_metrics.sh > pre-scaling-metrics.json

3. Execute scaling operation
   # Follow procedure for your platform

4. Verify new resources
   ./scripts/verify_resources.sh

5. Warm up caches
   curl -X POST http://localhost:7432/api/v1/admin/warmup

6. Run smoke tests
   ./scripts/smoke_test.sh

7. Disable maintenance mode
   curl -X POST http://localhost:7432/api/v1/admin/maintenance/disable

8. Monitor for 30 minutes
   watch -n 10 'curl -s http://localhost:7432/metrics | jq .summary'
```

**Post-Scaling Validation:**
```
□ All health checks passing
□ P99 latency within SLA
□ No error rate increase
□ Memory usage stable
□ CPU usage normalized
□ Customer traffic normal
```

## Acceptance Criteria

### Capacity Planning Calculator
- [ ] `/scripts/estimate_capacity.sh` accurately predicts resource needs within 15% for test workloads
- [ ] Calculator handles edge cases: 0 nodes, 100B nodes, invalid inputs
- [ ] Output includes clear recommendations and warnings for large deployments
- [ ] Script executable without additional dependencies (uses standard Unix tools)

### Scaling Decision Matrices
- [ ] CPU scaling matrix tested with synthetic load (stress-ng)
- [ ] Memory scaling matrix validated against tier distribution scenarios
- [ ] Storage scaling triggers tested with rapid write workloads
- [ ] All matrices included in `/docs/operations/scaling.md`

### Vertical Scaling Procedures
- [ ] Kubernetes scaling procedure tested in minikube
- [ ] Docker scaling procedure tested with resource constraints
- [ ] Tier rebalancing reduces memory by >30% when triggered
- [ ] All procedures complete in <5 minutes (excluding data migration)

### Performance Models
- [ ] Linear scaling model validated up to 8 cores (measured efficiency >65%)
- [ ] Memory latency model predictions within 20% of observed
- [ ] Storage throughput model accounts for cache effects
- [ ] Models documented with real measurement data

### Cost Optimization
- [ ] Right-sizing strategy reduces cost by >30% vs. initial overprovisioning
- [ ] Tier optimization achieves <5% hot tier ratio for stable workloads
- [ ] Compute optimization improves throughput by >20% via CPU affinity
- [ ] Cost savings quantified in dollars for AWS/GCP/Azure

### Documentation
- [ ] `/docs/operations/scaling.md` reviewed by ops team
- [ ] `/docs/operations/capacity-planning.md` includes worked examples
- [ ] `/docs/howto/scale-vertically.md` has step-by-step screenshots
- [ ] `/docs/reference/resource-requirements.md` has sizing tables for 5 workload types

### Operational Validation
- [ ] External operator successfully uses calculator to size deployment
- [ ] Scaling procedures executed without errors in staging environment
- [ ] Alert rules fire appropriately during scaling scenarios
- [ ] Rollback procedure tested and documented

## Follow-Up Tasks

**Milestone 16 Dependencies:**
- Task 004 (Performance Tuning): Use capacity models for bottleneck analysis
- Task 007 (High Availability): Factor in redundancy for capacity planning
- Task 011 (Load Testing): Validate scaling triggers under stress

**Milestone 17 (Distributed Scaling):**
- Horizontal scaling with consistent hashing
- Cross-region replication capacity planning
- Distributed tier migration protocols
- Global capacity orchestration
