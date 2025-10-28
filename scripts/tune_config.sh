#!/usr/bin/env bash
# Interactive configuration tuning wizard for Engram
#
# This script analyzes the system hardware and current workload patterns to
# generate an optimized configuration file. It detects CPU, memory, storage,
# and NUMA characteristics, then tunes parameters accordingly.
#
# Usage: ./scripts/tune_config.sh [config_file]
# Example: ./scripts/tune_config.sh /etc/engram/engram.toml

set -euo pipefail

CONFIG_FILE="${1:-/etc/engram/engram.toml}"
BACKUP_FILE="${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"

echo "=== Engram Configuration Tuning Wizard ==="
echo "Config file: $CONFIG_FILE"
echo ""

# Backup existing config
if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$BACKUP_FILE"
    echo "Backed up to: $BACKUP_FILE"
fi

# Detect system characteristics
detect_system() {
    echo "Detecting system characteristics..."

    # CPU cores
    if command -v nproc &> /dev/null; then
        CPU_CORES=$(nproc)
    else
        CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo "4")
    fi

    # Memory in GB
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    else
        # macOS fallback
        MEMORY_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "8589934592")
        MEMORY_GB=$((MEMORY_BYTES / 1024 / 1024 / 1024))
    fi

    # NUMA nodes
    if command -v lscpu &> /dev/null; then
        NUMA_NODES=$(lscpu | grep "NUMA node(s):" | awk '{print $3}' || echo "1")
    else
        NUMA_NODES=1
    fi

    # Check for AVX support
    AVX_SUPPORT="none"
    if [ -f /proc/cpuinfo ]; then
        if grep -q avx512 /proc/cpuinfo; then
            AVX_SUPPORT="avx512"
        elif grep -q avx2 /proc/cpuinfo; then
            AVX_SUPPORT="avx2"
        fi
    else
        # macOS fallback
        if sysctl -n machdep.cpu.features 2>/dev/null | grep -q AVX; then
            AVX_SUPPORT="avx2"
        fi
    fi

    # Detect storage type (SSD vs HDD)
    STORAGE_TYPE="ssd"
    if [ -d /sys/block ]; then
        # Check if any block device is rotational
        for dev in /sys/block/*/queue/rotational; do
            if [ -f "$dev" ] && [ "$(cat "$dev")" = "1" ]; then
                STORAGE_TYPE="hdd"
                break
            fi
        done
    fi

    echo "  CPU Cores: $CPU_CORES"
    echo "  Memory: ${MEMORY_GB}GB"
    echo "  NUMA Nodes: $NUMA_NODES"
    echo "  AVX Support: $AVX_SUPPORT"
    echo "  Storage Type: $STORAGE_TYPE"
    echo ""
}

# Analyze workload pattern
analyze_workload() {
    echo "Analyzing current workload pattern..."

    # Check if Prometheus is available
    if ! curl -sf http://localhost:9090/api/v1/query --max-time 5 \
      -d 'query=up' >/dev/null 2>&1; then
        echo "  WARNING: Prometheus not available, using defaults"
        READ_RATIO="0.7"
        AVG_RECALL_SIZE="10"
        WORKLOAD_TYPE="balanced"
        return
    fi

    # Get operation distribution from metrics
    READ_RATIO=$(curl -sf http://localhost:9090/api/v1/query --max-time 5 \
        -d 'query=rate(engram_recall_operations_total[1h]) / (rate(engram_recall_operations_total[1h]) + rate(engram_store_operations_total[1h]))' \
        2>/dev/null | jq -r '.data.result[0].value[1] // "0.7"' || echo "0.7")

    AVG_RECALL_SIZE=$(curl -sf http://localhost:9090/api/v1/query --max-time 5 \
        -d 'query=avg(engram_recall_result_size)' \
        2>/dev/null | jq -r '.data.result[0].value[1] // "10"' || echo "10")

    if command -v bc &> /dev/null; then
        READ_PCT=$(echo "scale=0; $READ_RATIO * 100" | bc 2>/dev/null || echo "70")
    else
        READ_PCT="70"
    fi

    echo "  Read/Write Ratio: ${READ_PCT}% reads"
    echo "  Average Recall Size: $AVG_RECALL_SIZE memories"

    # Classify workload
    if command -v bc &> /dev/null; then
        if (( $(echo "$READ_RATIO > 0.8" | bc -l 2>/dev/null || echo 0) )); then
            WORKLOAD_TYPE="read_heavy"
        elif (( $(echo "$READ_RATIO < 0.3" | bc -l 2>/dev/null || echo 0) )); then
            WORKLOAD_TYPE="write_heavy"
        else
            WORKLOAD_TYPE="balanced"
        fi
    else
        WORKLOAD_TYPE="balanced"
    fi

    echo "  Workload Type: $WORKLOAD_TYPE"
    echo ""
}

# Helper functions for calculating optimal values
calculate_hot_tier_size() {
    # Base calculation: 10% of RAM for hot tier, capped at 8GB
    local size_mb=$((MEMORY_GB * 1024 * 10 / 100))
    [ "$size_mb" -gt 8192 ] && size_mb=8192
    [ "$size_mb" -lt 256 ] && size_mb=256
    echo "$size_mb"
}

calculate_warm_tier_size() {
    # 30% of RAM for warm tier
    echo $((MEMORY_GB * 1024 * 30 / 100))
}

calculate_shards() {
    # Number of shards for DashMap: 2x CPU cores for lock-free concurrency
    echo $((CPU_CORES * 2))
}

calculate_prefetch_distance() {
    # Prefetch distance based on cache size (4-16 nodes)
    if [ "$CPU_CORES" -gt 32 ]; then
        echo "16"
    elif [ "$CPU_CORES" -gt 16 ]; then
        echo "12"
    else
        echo "8"
    fi
}

calculate_hnsw_m() {
    case "$WORKLOAD_TYPE" in
        read_heavy)
            echo "32"  # Higher connectivity for better recall
            ;;
        write_heavy)
            echo "16"  # Lower connectivity for faster insertions
            ;;
        *)
            echo "24"  # Balanced
            ;;
    esac
}

calculate_ef_construction() {
    [ "$WORKLOAD_TYPE" = "read_heavy" ] && echo "400" || echo "200"
}

calculate_ef_search() {
    [ "$WORKLOAD_TYPE" = "read_heavy" ] && echo "200" || echo "100"
}

calculate_wal_flush_interval() {
    case "$WORKLOAD_TYPE" in
        write_heavy)
            echo "5000"  # 5 seconds for batching
            ;;
        read_heavy)
            echo "100"   # 100ms for low latency
            ;;
        *)
            echo "1000"  # 1 second balanced
            ;;
    esac
}

calculate_recall_workers() {
    if [ "$WORKLOAD_TYPE" = "read_heavy" ]; then
        echo $((CPU_CORES * 2))  # 2x for I/O bound reads
    else
        echo "$CPU_CORES"
    fi
}

calculate_store_workers() {
    if [ "$WORKLOAD_TYPE" = "write_heavy" ]; then
        echo "$CPU_CORES"
    else
        echo $((CPU_CORES / 2))
    fi
}

# Generate optimized configuration
generate_config() {
    echo "Generating optimized configuration..."

    local HOT_TIER=$(calculate_hot_tier_size)
    local WARM_TIER=$(calculate_warm_tier_size)
    local SHARDS=$(calculate_shards)
    local PREFETCH=$(calculate_prefetch_distance)
    local HNSW_M=$(calculate_hnsw_m)
    local EF_CONST=$(calculate_ef_construction)
    local EF_SEARCH=$(calculate_ef_search)
    local WAL_FLUSH=$(calculate_wal_flush_interval)
    local RECALL_WORKERS=$(calculate_recall_workers)
    local STORE_WORKERS=$(calculate_store_workers)

    cat > "$CONFIG_FILE" <<EOF
# Engram Performance-Tuned Configuration
# Generated: $(date)
# System: ${CPU_CORES} cores, ${MEMORY_GB}GB RAM, ${NUMA_NODES} NUMA nodes
# Workload: ${WORKLOAD_TYPE}

[storage]
# Hot tier configuration (lock-free concurrent hashmap)
hot_tier_size_mb = ${HOT_TIER}
hot_tier_shards = ${SHARDS}

# Warm tier (memory-mapped files)
warm_tier_size_mb = ${WARM_TIER}
use_huge_pages = $([ "$MEMORY_GB" -gt 16 ] && echo "true" || echo "false")
mmap_populate = $([ "$STORAGE_TYPE" = "ssd" ] && echo "true" || echo "false")

# Cold tier (columnar storage)
cold_tier_compression = "zstd"
cold_tier_compression_level = $([ "$WORKLOAD_TYPE" = "write_heavy" ] && echo "1" || echo "3")

[activation]
# Spreading activation parameters
prefetch_distance = ${PREFETCH}
max_traversal_depth = 5
visit_budget_per_tier = { hot = 3, warm = 2, cold = 1 }

# SIMD optimization
simd_batch_size = 8
simd_enabled = $([ "$AVX_SUPPORT" != "none" ] && echo "true" || echo "false")
avx_version = "$AVX_SUPPORT"

[hnsw_index]
# HNSW parameters optimized for workload
M = ${HNSW_M}
ef_construction = ${EF_CONST}
ef_search = ${EF_SEARCH}
max_m = $((HNSW_M * 2))
max_m0 = $((HNSW_M * 2))

[wal]
# Write-ahead log configuration
buffer_size_mb = $([ "$WORKLOAD_TYPE" = "write_heavy" ] && echo "128" || echo "64")
flush_interval_ms = ${WAL_FLUSH}
sync_mode = $([ "$STORAGE_TYPE" = "ssd" ] && echo "\"batch\"" || echo "\"immediate\"")
segment_size_mb = 64

[thread_pools]
# Thread pool sizing based on system and workload
recall_workers = ${RECALL_WORKERS}
store_workers = ${STORE_WORKERS}
consolidation_workers = 2
background_workers = 4

[numa]
# NUMA-aware memory allocation
numa_aware = $([ "$NUMA_NODES" -gt 1 ] && echo "true" || echo "false")
interleave_memory = false
prefer_local_node = true
$([ "$NUMA_NODES" -gt 1 ] && echo "socket_memory_mb = [$((MEMORY_GB * 1024 / NUMA_NODES))]" || echo "# single NUMA node")

[consolidation]
# Memory consolidation settings
enabled = true
interval_seconds = $([ "$WORKLOAD_TYPE" = "write_heavy" ] && echo "600" || echo "300")
batch_size = $([ "$MEMORY_GB" -gt 32 ] && echo "5000" || echo "1000")
min_confidence_threshold = 0.3

[monitoring]
# Performance monitoring
metrics_enabled = true
metrics_interval_seconds = 10
slow_query_threshold_ms = 10
trace_sampling_rate = 0.001
EOF

    echo "Configuration written to: $CONFIG_FILE"
}

# Main execution
detect_system
analyze_workload
generate_config

echo ""
echo "=== Configuration Tuning Complete ==="
echo ""
echo "Configuration Summary:"
echo "  Hot Tier: $(calculate_hot_tier_size)MB"
echo "  Warm Tier: $(calculate_warm_tier_size)MB"
echo "  DashMap Shards: $(calculate_shards)"
echo "  HNSW M: $(calculate_hnsw_m)"
echo "  Prefetch Distance: $(calculate_prefetch_distance)"
echo "  Recall Workers: $(calculate_recall_workers)"
echo "  Store Workers: $(calculate_store_workers)"
echo ""
echo "Next steps:"
echo "1. Review configuration: cat $CONFIG_FILE"
echo "2. Restart Engram to apply changes"
echo "3. Run benchmark: ./scripts/benchmark_deployment.sh 60 10"
echo "4. Monitor metrics: ./scripts/profile_performance.sh 120"
echo ""
echo "To rollback: cp $BACKUP_FILE $CONFIG_FILE"
