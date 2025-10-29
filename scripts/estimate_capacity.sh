#!/usr/bin/env bash
set -euo pipefail

# Engram Capacity Planning Calculator
# Estimates resource requirements based on expected workload characteristics
# Usage: ./estimate_capacity.sh [nodes] [active_ratio] [edges_per_node] [ops_per_sec] [retention_days] [consolidation_interval] [workload_type]

# Show usage if --help flag provided
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    cat <<EOF
Engram Capacity Planning Calculator

Usage: $0 [nodes] [active_ratio] [edges_per_node] [ops_per_sec] [retention_days] [consolidation_interval] [workload_type]

Parameters:
  nodes                  Total nodes expected (default: 1000000)
  active_ratio          Fraction of nodes active in hot tier (default: 0.1)
  edges_per_node        Average edges per node (default: 50)
  ops_per_sec           Target operations per second (default: 5000)
  retention_days        Data retention period (default: 30)
  consolidation_interval Seconds between consolidations (default: 300)
  workload_type         Workload pattern: write-heavy, read-heavy, analytical, mixed (default: mixed)

Workload Profiles:
  write-heavy  - 80% writes, 20% reads (ingestion workload)
  read-heavy   - 20% writes, 80% reads (retrieval workload)
  analytical   - Complex graph traversals and pattern mining
  mixed        - 40% writes, 60% reads (typical production)

Examples:
  # Default configuration (1M nodes, mixed workload)
  $0

  # Large analytical workload
  $0 100000000 0.15 75 10000 90 300 analytical

  # Write-heavy ingestion pipeline
  $0 50000000 0.05 30 20000 7 600 write-heavy

  # Small read-heavy retrieval system
  $0 500000 0.2 40 2000 14 300 read-heavy
EOF
    exit 0
fi

# Input parameters with defaults
EXPECTED_NODES="${1:-1000000}"
ACTIVE_RATIO="${2:-0.1}"
EDGES_PER_NODE="${3:-50}"
OPS_PER_SEC="${4:-5000}"
RETENTION_DAYS="${5:-30}"
CONSOLIDATION_INTERVAL="${6:-300}"
WORKLOAD_TYPE="${7:-mixed}"

# Constants from Engram architecture
EMBEDDING_SIZE=3072                    # 768 floats * 4 bytes
NODE_METADATA=64                       # Activation, confidence, timestamp, decay
EDGE_OVERHEAD=32                       # Weight, timestamp, confidence
HASH_OVERHEAD=1.5                      # DashMap bucket overhead + tombstones
WAL_RECORD=256                         # Average WAL record size
COMPRESSION_RATIO=0.4                  # Cold tier compression with zstd level 3
MMAP_PAGE_SIZE=4096                    # OS page size for warm tier
RUNTIME_BASE=500                       # Base runtime overhead in MB

# Workload-specific multipliers
case "$WORKLOAD_TYPE" in
    write-heavy)
        WRITE_RATIO=0.8
        READ_RATIO=0.2
        WAL_MULTIPLIER=1.5             # More WAL activity
        CPU_MULTIPLIER=0.9             # Less CPU for spreading
        MEMORY_MULTIPLIER=1.2          # More memory for buffering
        ;;
    read-heavy)
        WRITE_RATIO=0.2
        READ_RATIO=0.8
        WAL_MULTIPLIER=0.5             # Less WAL activity
        CPU_MULTIPLIER=1.3             # More CPU for spreading
        MEMORY_MULTIPLIER=1.0          # Standard memory
        ;;
    analytical)
        WRITE_RATIO=0.1
        READ_RATIO=0.9
        WAL_MULTIPLIER=0.3             # Minimal writes
        CPU_MULTIPLIER=1.8             # Heavy CPU for traversals
        MEMORY_MULTIPLIER=1.5          # Large working sets
        ;;
    mixed|*)
        WRITE_RATIO=0.4
        READ_RATIO=0.6
        WAL_MULTIPLIER=1.0             # Standard WAL
        CPU_MULTIPLIER=1.0             # Balanced CPU
        MEMORY_MULTIPLIER=1.0          # Standard memory
        ;;
esac

# Calculate tier distribution
HOT_NODES=$(echo "$EXPECTED_NODES * $ACTIVE_RATIO" | bc -l | cut -d. -f1)
WARM_NODES=$(echo "$EXPECTED_NODES * $ACTIVE_RATIO * 3" | bc -l | cut -d. -f1)
COLD_NODES=$(echo "$EXPECTED_NODES - $HOT_NODES - $WARM_NODES" | bc -l | cut -d. -f1)

TOTAL_EDGES=$(echo "$EXPECTED_NODES * $EDGES_PER_NODE" | bc -l | cut -d. -f1)
HOT_EDGES=$(echo "$HOT_NODES * $EDGES_PER_NODE" | bc -l | cut -d. -f1)

# Memory calculations (in MB)
HOT_MEMORY=$(echo "($HOT_NODES * ($EMBEDDING_SIZE + $NODE_METADATA) + \
             $HOT_EDGES * $EDGE_OVERHEAD) * $HASH_OVERHEAD / 1048576" | bc -l)

WARM_MEMORY=$(echo "$WARM_NODES * $EMBEDDING_SIZE * 0.1 / 1048576" | bc -l)

COLD_MEMORY=$(echo "($COLD_NODES * 10 / 8 + $COLD_NODES * 64 / 128) / 1048576" | bc -l)

ACTIVATION_POOL=$(echo "$OPS_PER_SEC * 0.01 * 8192 / 1048576" | bc -l)

RUNTIME_OVERHEAD=$RUNTIME_BASE

TOTAL_MEMORY=$(echo "($HOT_MEMORY + $WARM_MEMORY + $COLD_MEMORY + \
              $ACTIVATION_POOL + $RUNTIME_OVERHEAD) * $MEMORY_MULTIPLIER" | bc -l)

# CPU calculations with workload adjustments
SPREADING_CORES=$(echo "$OPS_PER_SEC * $READ_RATIO / 2500" | bc -l)
CONSOLIDATION_CORES=$(echo "$HOT_NODES * $WRITE_RATIO / $CONSOLIDATION_INTERVAL / 50000" | bc -l)
API_CORES=$(echo "$OPS_PER_SEC / 5000" | bc -l)
WORKLOAD_CORES=$(echo "($SPREADING_CORES + $CONSOLIDATION_CORES + $API_CORES) * $CPU_MULTIPLIER + 1" | bc -l)
TOTAL_CORES=$WORKLOAD_CORES

# Calculate recommended cores based on NUMA topology
RECOMMENDED_CORES=$(echo "scale=0; ($TOTAL_CORES * 1.5 + 0.5) / 1" | bc)
# Round to next power of 2 or NUMA-friendly value (2, 4, 8, 16, 32)
if (( $(echo "$RECOMMENDED_CORES <= 2" | bc -l) )); then
    NUMA_CORES=2
elif (( $(echo "$RECOMMENDED_CORES <= 4" | bc -l) )); then
    NUMA_CORES=4
elif (( $(echo "$RECOMMENDED_CORES <= 8" | bc -l) )); then
    NUMA_CORES=8
elif (( $(echo "$RECOMMENDED_CORES <= 16" | bc -l) )); then
    NUMA_CORES=16
elif (( $(echo "$RECOMMENDED_CORES <= 32" | bc -l) )); then
    NUMA_CORES=32
else
    NUMA_CORES=$(echo "scale=0; (($RECOMMENDED_CORES + 31) / 32) * 32" | bc)
fi

# Storage calculations (in GB) with workload adjustments
WARM_STORAGE=$(echo "$WARM_NODES * ($EMBEDDING_SIZE + $NODE_METADATA) * 1.3 / 1073741824" | bc -l)
COLD_STORAGE=$(echo "$COLD_NODES * $EMBEDDING_SIZE * $COMPRESSION_RATIO / 1073741824" | bc -l)
WAL_STORAGE=$(echo "$OPS_PER_SEC * $WRITE_RATIO * 86400 * $RETENTION_DAYS * $WAL_RECORD * 0.7 * $WAL_MULTIPLIER / 1073741824" | bc -l)
SNAPSHOT_STORAGE=$(echo "($WARM_STORAGE + $COLD_STORAGE) * 5 * 0.3" | bc -l)
TEMP_SPACE=$(echo "($WARM_STORAGE + $COLD_STORAGE) * 0.1" | bc -l)
TOTAL_STORAGE=$(echo "$WARM_STORAGE + $COLD_STORAGE + $WAL_STORAGE + $SNAPSHOT_STORAGE + $TEMP_SPACE" | bc -l)

# Calculate storage IOPS requirements
WRITE_IOPS=$(echo "$OPS_PER_SEC * $WRITE_RATIO * 1.5" | bc -l | cut -d. -f1)  # 1.5x for WAL overhead
READ_IOPS=$(echo "$OPS_PER_SEC * $READ_RATIO * 0.3" | bc -l | cut -d. -f1)    # 30% cache miss rate
TOTAL_IOPS=$((WRITE_IOPS + READ_IOPS))

# Network bandwidth (MB/s)
AVG_REQUEST_SIZE=512          # bytes
AVG_RESPONSE_SIZE=4096        # bytes (includes embeddings)
CLIENT_BANDWIDTH=$(echo "($AVG_REQUEST_SIZE + $AVG_RESPONSE_SIZE) * $OPS_PER_SEC / 1048576" | bc -l)
BACKUP_BANDWIDTH=$(echo "$TOTAL_STORAGE * 1024 / (3600 * 4)" | bc -l)  # 4-hour backup window
PEAK_BANDWIDTH=$(echo "$CLIENT_BANDWIDTH * 1.5 + $BACKUP_BANDWIDTH" | bc -l)

# Format output
echo "=========================================="
echo "  Engram Capacity Planning Estimate"
echo "=========================================="
echo ""
echo "Workload Profile: ${WORKLOAD_TYPE}"
echo "  Write/Read Ratio: ${WRITE_RATIO}/${READ_RATIO}"
echo ""
echo "Input Parameters:"
echo "  Expected Nodes: $(printf "%'d" $EXPECTED_NODES)"
echo "  Active Ratio: ${ACTIVE_RATIO} ($(printf "%'d" $HOT_NODES) hot nodes)"
echo "  Edges per Node: ${EDGES_PER_NODE}"
echo "  Target Ops/sec: $(printf "%'d" $OPS_PER_SEC)"
echo "  Retention Days: ${RETENTION_DAYS}"
echo "  Consolidation Interval: ${CONSOLIDATION_INTERVAL}s"
echo ""
echo "Tier Distribution:"
echo "  Hot Tier:  $(printf "%'10d" $HOT_NODES) nodes (in-memory, DashMap)"
echo "  Warm Tier: $(printf "%'10d" $WARM_NODES) nodes (mmap files)"
echo "  Cold Tier: $(printf "%'10d" $COLD_NODES) nodes (compressed, zstd)"
echo ""
echo "=========================================="
echo "  Resource Requirements"
echo "=========================================="
echo ""
echo "CPU:"
echo "  Minimum Cores: $(printf "%.1f" $TOTAL_CORES) vCPUs"
echo "    - Spreading Activation: $(printf "%.2f" $SPREADING_CORES) cores"
echo "    - Memory Consolidation: $(printf "%.2f" $CONSOLIDATION_CORES) cores"
echo "    - API/HTTP Overhead: $(printf "%.2f" $API_CORES) cores"
echo "    - System Overhead: 1.0 cores"
echo "  Recommended: ${NUMA_CORES} vCPUs (NUMA-aligned)"
echo "  Maximum Useful: 32 cores (lock contention beyond this)"
echo ""
echo "Memory:"
echo "  Minimum RAM: $(printf "%.1f" $TOTAL_MEMORY) MB ($(echo "scale=1; $TOTAL_MEMORY / 1024" | bc) GB)"
echo "    - Hot Tier: $(printf "%'10.1f" $HOT_MEMORY) MB"
echo "    - Warm Tier Prefetch: $(printf "%'10.1f" $WARM_MEMORY) MB"
echo "    - Cold Tier Index: $(printf "%'10.1f" $COLD_MEMORY) MB"
echo "    - Activation Pool: $(printf "%'10.1f" $ACTIVATION_POOL) MB"
echo "    - Runtime Overhead: $(printf "%'10d" $RUNTIME_OVERHEAD) MB"
echo "  Recommended: $(echo "scale=0; ($TOTAL_MEMORY * 1.25 / 1024 + 0.5) / 1" | bc) GB (25% headroom)"
echo "  Peak with Compaction: $(echo "scale=0; ($TOTAL_MEMORY * 1.5 / 1024 + 0.5) / 1" | bc) GB"
echo ""
echo "Storage:"
echo "  Minimum Disk: $(printf "%.1f" $TOTAL_STORAGE) GB"
echo "    - Warm Tier Data: $(printf "%'10.1f" $WARM_STORAGE) GB"
echo "    - Cold Tier Data: $(printf "%'10.1f" $COLD_STORAGE) GB (compressed)"
echo "    - WAL: $(printf "%'10.1f" $WAL_STORAGE) GB"
echo "    - Snapshots: $(printf "%'10.1f" $SNAPSHOT_STORAGE) GB (incremental)"
echo "    - Temp/Compaction: $(printf "%'10.1f" $TEMP_SPACE) GB"
echo "  Recommended: $(echo "scale=0; ($TOTAL_STORAGE * 1.5 + 0.5) / 1" | bc) GB (50% growth buffer)"
echo "  IOPS Required: $(printf "%'d" $TOTAL_IOPS) IOPS (${WRITE_IOPS} write / ${READ_IOPS} read)"
echo "  Disk Type: NVMe SSD (NVMe > SSD > HDD for best performance)"
echo ""
echo "Network:"
echo "  Client Bandwidth: $(printf "%.1f" $CLIENT_BANDWIDTH) MB/s sustained"
echo "  Peak Bandwidth: $(printf "%.1f" $PEAK_BANDWIDTH) MB/s (client + backup)"
echo "  Recommended: $(echo "scale=0; ($PEAK_BANDWIDTH * 8 + 999) / 1000" | bc) Gbps link"
echo ""
echo "=========================================="
echo "  NUMA & Performance Guidance"
echo "=========================================="
echo ""
if (( $(echo "$TOTAL_MEMORY > 32768" | bc -l) )); then
    echo "NUMA Configuration Required:"
    echo "  Memory Size: >32GB - Use NUMA-aware allocation"
    echo "  Recommended NUMA Policy:"
    echo "    numactl --membind=0 --cpunodebind=0 engram-server"
    echo "  Pin hot tier to NUMA node 0 for fastest access"
    echo "  Distribute warm tier across NUMA nodes"
    echo ""
    echo "  Verify NUMA setup:"
    echo "    numactl --hardware"
    echo "    numastat engram-server"
    echo ""
fi

if (( NUMA_CORES > 8 )); then
    echo "CPU Affinity Recommendations:"
    echo "  Cores: ${NUMA_CORES} - Use CPU pinning for critical paths"
    echo "  Pin spreading threads to dedicated cores (avoid context switching)"
    echo "  Example taskset usage:"
    echo "    taskset -c 0-7 engram-server  # Pin to first 8 cores"
    echo ""
fi

echo "Cache Optimization:"
CACHE_L1_NODES=$(echo "64 / ($EMBEDDING_SIZE / 1024)" | bc)
CACHE_L2_NODES=$(echo "256 / ($EMBEDDING_SIZE / 1024)" | bc)
CACHE_L3_MB=$(echo "$NUMA_CORES * 1.5" | bc)
CACHE_L3_NODES=$(echo "$CACHE_L3_MB * 1024 / ($EMBEDDING_SIZE / 1024)" | bc)
echo "  L1 Cache (64KB): ~${CACHE_L1_NODES} nodes per core"
echo "  L2 Cache (256KB): ~${CACHE_L2_NODES} nodes per core"
echo "  L3 Cache ($(printf "%.0f" $CACHE_L3_MB)MB shared): ~${CACHE_L3_NODES} nodes total"
echo "  Optimize batch sizes to fit working set in L3"
echo ""
echo "=========================================="
echo "  Scaling Triggers & Alerts"
echo "=========================================="
echo ""
echo "Scale Up CPU When:"
echo "  - Average CPU utilization >70% for 10+ minutes"
echo "  - Spreading queue depth growing (>100 pending)"
echo "  - P99 latency >10ms on spreading operations"
echo "  Action: Add $(echo "$NUMA_CORES * 2" | bc) cores (double capacity)"
echo ""
echo "Scale Up Memory When:"
echo "  - Memory utilization >80%"
echo "  - Hot tier eviction rate spiking"
echo "  - Allocation failures or OOM warnings"
echo "  Action: Add $(echo "scale=0; ($TOTAL_MEMORY / 1024 + 0.5) / 1" | bc) GB (double capacity)"
echo ""
echo "Scale Up Storage When:"
echo "  - Disk utilization >70%"
echo "  - WAL size >10GB"
echo "  - Compaction backlog growing"
echo "  Action: Add $(echo "scale=0; ($TOTAL_STORAGE + 0.5) / 1" | bc) GB (double capacity)"
echo ""
echo "=========================================="
echo "  Workload-Specific Recommendations"
echo "=========================================="
echo ""
case "$WORKLOAD_TYPE" in
    write-heavy)
        echo "Write-Heavy Workload Optimizations:"
        echo "  - Increase WAL buffer to $(echo "scale=0; $WAL_STORAGE * 0.1 * 1024" | bc) MB"
        echo "  - Batch writes aggressively (100-1000 operations)"
        echo "  - Reduce hot tier threshold (0.3 → 0.2) to speed eviction"
        echo "  - Pre-allocate warm tier segments to avoid allocation overhead"
        echo "  - Monitor WAL disk write throughput closely"
        ;;
    read-heavy)
        echo "Read-Heavy Workload Optimizations:"
        echo "  - Maximize hot tier retention (increase activation threshold to 0.4)"
        echo "  - Enable GPU acceleration for parallel spreading"
        echo "  - Increase activation pool to $(echo "scale=0; $ACTIVATION_POOL * 2" | bc) MB"
        echo "  - Use read replicas for horizontal scaling (future)"
        echo "  - Consider CPU with AVX-512 for SIMD spreading"
        ;;
    analytical)
        echo "Analytical Workload Optimizations:"
        echo "  - Pin hot tier in memory (disable eviction during queries)"
        echo "  - Dedicate $(echo "scale=0; $NUMA_CORES * 0.75" | bc) cores to spreading threads"
        echo "  - Increase refractory period to 500ms for stable activation"
        echo "  - Batch analytical queries to amortize graph warming"
        echo "  - Pre-warm critical paths before query execution"
        ;;
    mixed|*)
        echo "Mixed Workload Optimizations:"
        echo "  - Use adaptive batch controller (auto-tune batch size)"
        echo "  - Schedule consolidation during low-traffic windows (2-6 AM)"
        echo "  - Implement tier rebalancing policies based on access patterns"
        echo "  - Monitor and adjust thresholds dynamically"
        echo "  - Reserve 20% CPU headroom for traffic spikes"
        ;;
esac
echo ""
echo "=========================================="
echo "  Cost Optimization Strategies"
echo "=========================================="
echo ""
echo "Right-Sizing Approach:"
echo "  1. Deploy with 2x calculated baseline initially"
echo "  2. Monitor for 7 days to identify actual patterns"
echo "  3. Reduce to 1.3x peak observed utilization"
echo "  4. Target 70% average, 85% peak utilization"
echo "  Estimated Savings: 30-50% vs over-provisioning"
echo ""
echo "Tier Optimization:"
echo "  - Keep <5% of nodes in hot tier (current: $(echo "scale=1; $HOT_NODES * 100 / $EXPECTED_NODES" | bc)%)"
echo "  - Compress warm tier nodes inactive >1 hour"
echo "  - Archive cold tier to object storage after ${RETENTION_DAYS} days"
echo "  Estimated Savings: 60% storage costs"
echo ""
echo "Compute Optimization:"
if (( NUMA_CORES > 4 )); then
    echo "  - Run consolidation during off-peak hours only"
    echo "  - Use CPU affinity to reduce context switching (20% improvement)"
    echo "  - Consider spot GPU instances for batch spreading (70% cheaper)"
fi
echo ""
echo "=========================================="
echo "  Instance Recommendations"
echo "=========================================="
echo ""
MEMORY_GB=$(echo "scale=0; ($TOTAL_MEMORY * 1.25 / 1024 + 0.5) / 1" | bc)
STORAGE_GB=$(echo "scale=0; ($TOTAL_STORAGE * 1.5 + 0.5) / 1" | bc)

echo "AWS EC2:"
if (( NUMA_CORES <= 4 )) && (( MEMORY_GB <= 16 )); then
    echo "  - m6i.xlarge (4 vCPU, 16GB RAM) + ${STORAGE_GB}GB gp3"
    echo "  - Cost: ~\$150/month + \$$(echo "scale=0; $STORAGE_GB * 0.08" | bc)/month storage"
elif (( NUMA_CORES <= 8 )) && (( MEMORY_GB <= 32 )); then
    echo "  - m6i.2xlarge (8 vCPU, 32GB RAM) + ${STORAGE_GB}GB gp3"
    echo "  - Cost: ~\$300/month + \$$(echo "scale=0; $STORAGE_GB * 0.08" | bc)/month storage"
elif (( NUMA_CORES <= 16 )) && (( MEMORY_GB <= 64 )); then
    echo "  - m6i.4xlarge (16 vCPU, 64GB RAM) + ${STORAGE_GB}GB gp3"
    echo "  - Cost: ~\$600/month + \$$(echo "scale=0; $STORAGE_GB * 0.08" | bc)/month storage"
else
    echo "  - m6i.8xlarge (32 vCPU, 128GB RAM) + ${STORAGE_GB}GB gp3"
    echo "  - Cost: ~\$1200/month + \$$(echo "scale=0; $STORAGE_GB * 0.08" | bc)/month storage"
fi
echo ""
echo "GCP Compute Engine:"
if (( NUMA_CORES <= 4 )) && (( MEMORY_GB <= 16 )); then
    echo "  - n2-standard-4 (4 vCPU, 16GB RAM) + ${STORAGE_GB}GB SSD"
elif (( NUMA_CORES <= 8 )) && (( MEMORY_GB <= 32 )); then
    echo "  - n2-standard-8 (8 vCPU, 32GB RAM) + ${STORAGE_GB}GB SSD"
elif (( NUMA_CORES <= 16 )) && (( MEMORY_GB <= 64 )); then
    echo "  - n2-standard-16 (16 vCPU, 64GB RAM) + ${STORAGE_GB}GB SSD"
else
    echo "  - n2-standard-32 (32 vCPU, 128GB RAM) + ${STORAGE_GB}GB SSD"
fi
echo ""
echo "Azure Virtual Machines:"
if (( NUMA_CORES <= 4 )) && (( MEMORY_GB <= 16 )); then
    echo "  - Standard_D4s_v5 (4 vCPU, 16GB RAM) + ${STORAGE_GB}GB Premium SSD"
elif (( NUMA_CORES <= 8 )) && (( MEMORY_GB <= 32 )); then
    echo "  - Standard_D8s_v5 (8 vCPU, 32GB RAM) + ${STORAGE_GB}GB Premium SSD"
elif (( NUMA_CORES <= 16 )) && (( MEMORY_GB <= 64 )); then
    echo "  - Standard_D16s_v5 (16 vCPU, 64GB RAM) + ${STORAGE_GB}GB Premium SSD"
else
    echo "  - Standard_D32s_v5 (32 vCPU, 128GB RAM) + ${STORAGE_GB}GB Premium SSD"
fi
echo ""
echo "=========================================="
echo "  Next Steps"
echo "=========================================="
echo ""
echo "1. Review capacity plan with team and stakeholders"
echo "2. Provision infrastructure using recommended instance types"
echo "3. Deploy Engram with configuration:"
echo "     --hot-tier-size=$(echo "scale=0; $HOT_MEMORY + 0.5" | bc)MB \\"
echo "     --warm-tier-size=$(echo "scale=0; $WARM_STORAGE * 1024 + 0.5" | bc)MB \\"
echo "     --cold-tier-path=/data/cold \\"
echo "     --wal-buffer-size=$(echo "scale=0; $WAL_STORAGE * 0.1 * 1024 + 0.5" | bc)MB"
echo "4. Monitor metrics for 7 days: CPU, memory, storage, latency"
echo "5. Adjust resources based on actual utilization patterns"
echo "6. Set up auto-scaling policies (see /docs/operations/scaling.md)"
echo "7. Configure alerts for scaling triggers"
echo "8. Schedule capacity review quarterly"
echo ""
echo "Documentation:"
echo "  - Scaling Guide: /docs/operations/scaling.md"
echo "  - Capacity Planning: /docs/operations/capacity-planning.md"
echo "  - Vertical Scaling: /docs/howto/scale-vertically.md"
echo "  - Resource Reference: /docs/reference/resource-requirements.md"
echo ""
echo "=========================================="
