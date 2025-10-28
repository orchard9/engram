#!/bin/bash
set -euo pipefail

# Engram Capacity Planning Calculator
# Estimates resource requirements based on expected workload characteristics

# Input parameters
EXPECTED_NODES="${1:-1000000}"
ACTIVE_RATIO="${2:-0.1}"
EDGES_PER_NODE="${3:-50}"
OPS_PER_SEC="${4:-5000}"
RETENTION_DAYS="${5:-30}"
CONSOLIDATION_INTERVAL="${6:-300}"

# Constants from Engram architecture
EMBEDDING_SIZE=3072
NODE_METADATA=64
EDGE_OVERHEAD=32
HASH_OVERHEAD=1.5
WAL_RECORD=256
COMPRESSION_RATIO=0.4

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

RUNTIME_OVERHEAD=500

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
SNAPSHOT_STORAGE=$(echo "($WARM_STORAGE + $COLD_STORAGE) * 5 * 0.3" | bc -l)
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
echo "    - Recommended: $(echo "scale=0; ($TOTAL_CORES * 1.5) / 1" | bc) cores (with headroom)"
echo ""
echo "  Memory: $(printf "%.1f" $TOTAL_MEMORY) MB"
echo "    - Hot Tier: $(printf "%.1f" $HOT_MEMORY) MB"
echo "    - Warm Tier: $(printf "%.1f" $WARM_MEMORY) MB"
echo "    - Cold Tier: $(printf "%.1f" $COLD_MEMORY) MB"
echo "    - Activation Pool: $(printf "%.1f" $ACTIVATION_POOL) MB"
echo "    - Runtime: ${RUNTIME_OVERHEAD} MB"
echo "    - Recommended: $(echo "scale=0; ($TOTAL_MEMORY * 1.25 / 1024) / 1" | bc) GB (with headroom)"
echo ""
echo "  Storage: $(printf "%.1f" $TOTAL_STORAGE) GB"
echo "    - Warm Tier: $(printf "%.1f" $WARM_STORAGE) GB"
echo "    - Cold Tier: $(printf "%.1f" $COLD_STORAGE) GB (compressed)"
echo "    - WAL: $(printf "%.1f" $WAL_STORAGE) GB"
echo "    - Snapshots: $(printf "%.1f" $SNAPSHOT_STORAGE) GB"
echo "    - Recommended: $(echo "scale=0; ($TOTAL_STORAGE * 1.5) / 1" | bc) GB (with headroom)"
echo ""
echo "Scaling Recommendations:"
if (( $(echo "$TOTAL_CORES > 8" | bc -l) )); then
    echo "  WARNING: CPU cores exceed 8 - consider horizontal scaling"
fi
if (( $(echo "$TOTAL_MEMORY > 32768" | bc -l) )); then
    echo "  WARNING: Large memory footprint - ensure NUMA-aware allocation"
fi
if (( $(echo "$HOT_NODES > 10000000" | bc -l) )); then
    echo "  WARNING: Hot tier exceeds 10M nodes - consider sharding"
fi
echo "  INFO: Deploy with minimum resources for baseline performance"
echo "  INFO: Monitor actual utilization and adjust based on workload"
echo ""
echo "Cost Optimization Tips:"
echo "  - Reduce hot tier size by tuning eviction threshold"
echo "  - Use spot instances for read replicas (future)"
echo "  - Archive cold tier to object storage after ${RETENTION_DAYS} days"
echo "  - Increase consolidation interval to reduce CPU usage"
