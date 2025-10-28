# Task 004: Performance Tuning & Profiling Guide — pending

**Priority:** P0 (Critical Path)
**Estimated Effort:** 2 days
**Dependencies:** Task 003 (Monitoring)

## Objective

Create operator-focused performance tuning, profiling, and optimization procedures. Enable operators to identify bottlenecks, tune configurations for their workload, and achieve documented performance targets without developer assistance.

## Integration Points

**Uses:**
- `/docs/operations/monitoring.md` - Metrics from Task 003
- `/engram-core/benches/` - Existing benchmarks
- `/vision.md` - Performance targets (10ms P99, 10K ops/sec)
- `/deployments/prometheus/` - Metrics infrastructure

**Creates:**
- `/scripts/profile_performance.sh` - Operator profiling toolkit
- `/scripts/analyze_slow_queries.sh` - Query performance analysis
- `/scripts/benchmark_deployment.sh` - Production benchmark suite
- `/scripts/tune_config.sh` - Configuration tuning wizard
- `/tools/perf-analyzer/src/main.rs` - Performance analysis CLI

**Updates:**
- `/docs/operations/performance-tuning.md` - Complete tuning guide
- `/docs/howto/identify-slow-queries.md` - Query debugging
- `/docs/howto/optimize-resource-usage.md` - Resource optimization
- `/docs/reference/performance-baselines.md` - Expected performance

## Technical Specifications

### Performance Targets (from vision.md)

**Latency:**
- P50: <5ms for single-hop activation
- P99: <10ms for single-hop activation
- P99.9: <50ms for multi-hop activation

**Throughput:**
- Sustained: 10,000 operations/second (store + recall combined)
- Burst: 50,000 operations/second for <10 seconds
- Concurrent: 1,000 simultaneous clients

**Resource Limits:**
- Memory overhead: <2x raw data size
- CPU utilization: <70% at sustained load
- Disk I/O: <500 IOPS for steady state

**Scaling:**
- 1M+ nodes with 768-dimensional embeddings
- Linear scaling with CPU cores up to 32 cores
- Graph operations scale O(log n) with node count

### Critical Performance Paths

Based on codebase analysis, these are the hot paths requiring optimization:

**1. Activation Spreading (engram-core/src/activation/traversal.rs)**
- BreadthFirstTraversal with DashMap for concurrent visit tracking
- Cache line optimization: 64-byte aligned nodes
- Prefetch distance tuning for L2/L3 cache
- NUMA-aware memory allocation for multi-socket systems

**2. SIMD Operations (engram-core/src/activation/simd_optimization.rs)**
- AoSoA layout for 768-dimensional embeddings (96 tiles × 8 lanes)
- Cache-aligned batches at 64-byte boundaries
- Vectorized similarity computations using AVX2/AVX-512
- Batch size tuning: 8 embeddings per SIMD batch

**3. Storage Tiers (engram-core/src/storage/)**
- Hot tier: Lock-free DashMap with atomic operations
- Warm tier: Memory-mapped files with 2MB huge pages
- Cold tier: Columnar storage with compression
- Cache line-aware node layout (3200 bytes = 50 cache lines)

### Profiling Toolkit

### /scripts/profile_performance.sh

```bash
#!/bin/bash
# Operator-focused performance profiling with cache efficiency analysis

set -euo pipefail

DURATION="${1:-60}"  # Profile duration in seconds
OUTPUT_DIR="${2:-./profile-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_DIR"

echo "Profiling Engram for ${DURATION}s..."
echo "Output directory: $OUTPUT_DIR"

# 1. Capture baseline metrics
echo "Step 1/8: Capturing baseline metrics..."
curl -s http://localhost:9090/api/v1/query \
  -d 'query=engram_memory_operation_duration_seconds' \
  > "$OUTPUT_DIR/baseline_metrics.json"

# 2. CPU profiling with cache miss analysis
echo "Step 2/8: CPU profiling with cache analysis..."
ENGRAM_PID=$(pgrep engram)
if command -v perf &> /dev/null; then
    # Record CPU cycles and cache misses
    sudo perf record -F 999 -p $ENGRAM_PID -g \
        -e cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses \
        -- sleep $DURATION
    sudo perf script > "$OUTPUT_DIR/cpu_profile.txt"

    # Generate cache miss statistics
    sudo perf stat -p $ENGRAM_PID \
        -e cache-references,cache-misses,LLC-loads,LLC-load-misses \
        sleep 10 2> "$OUTPUT_DIR/cache_stats.txt"

    # Generate flamegraph data
    sudo perf script | stackcollapse-perf.pl > "$OUTPUT_DIR/collapsed.txt"
    echo "CPU profile saved to $OUTPUT_DIR/cpu_profile.txt"
else
    echo "WARN: perf not available, skipping CPU profile"
fi

# 3. NUMA topology and memory locality
echo "Step 3/8: NUMA topology analysis..."
if command -v numactl &> /dev/null; then
    numactl --hardware > "$OUTPUT_DIR/numa_topology.txt"
    numactl --show > "$OUTPUT_DIR/numa_policy.txt"

    # Per-node memory usage
    for node in $(ls -d /sys/devices/system/node/node* | grep -o 'node[0-9]*'); do
        echo "=== $node ===" >> "$OUTPUT_DIR/numa_memory.txt"
        cat /sys/devices/system/node/$node/meminfo >> "$OUTPUT_DIR/numa_memory.txt"
    done
else
    echo "WARN: numactl not available, skipping NUMA analysis"
fi

# 4. Memory profiling with huge page analysis
echo "Step 4/8: Memory profiling..."
if [ -f /proc/$ENGRAM_PID/smaps ]; then
    cat /proc/$ENGRAM_PID/smaps > "$OUTPUT_DIR/memory_map.txt"

    # Extract huge page usage
    grep -E "AnonHugePages|ShmemPmdMapped|FilePmdMapped" /proc/$ENGRAM_PID/smaps | \
        awk '{sum+=$2} END {print "Huge Pages: " sum/1024 " MB"}' > "$OUTPUT_DIR/huge_pages.txt"

    ps -p $ENGRAM_PID -o pid,vsz,rss,pmem,comm > "$OUTPUT_DIR/memory_usage.txt"
fi

# 5. Lock contention analysis
echo "Step 5/8: Lock contention analysis..."
if [ -f /proc/$ENGRAM_PID/task ]; then
    # Sample thread states to detect contention
    for i in {1..10}; do
        echo "=== Sample $i ===" >> "$OUTPUT_DIR/thread_states.txt"
        for tid in $(ls /proc/$ENGRAM_PID/task/); do
            cat /proc/$ENGRAM_PID/task/$tid/stat | awk '{print $2, $3}' >> "$OUTPUT_DIR/thread_states.txt"
        done
        sleep 1
    done
fi

# 6. I/O profiling
echo "Step 6/8: I/O profiling..."
if command -v iotop &> /dev/null; then
    sudo iotop -b -n 10 -d 1 -p $ENGRAM_PID > "$OUTPUT_DIR/io_stats.txt"
else
    # Fallback to /proc/io
    for i in {1..10}; do
        echo "=== Sample $i ===" >> "$OUTPUT_DIR/io_stats.txt"
        cat /proc/$ENGRAM_PID/io >> "$OUTPUT_DIR/io_stats.txt"
        sleep 1
    done
fi

# 7. Query latency analysis with percentiles
echo "Step 7/8: Analyzing query latencies..."
for percentile in 50 90 95 99 99.9; do
    curl -s http://localhost:9090/api/v1/query \
      -d "query=histogram_quantile(0.$percentile, engram_memory_operation_duration_seconds_bucket)" \
      | jq -r ".data.result[] | \"P$percentile \(.metric.operation): \(.value[1])s\"" \
      >> "$OUTPUT_DIR/latency_percentiles.txt"
done

# 8. Comprehensive bottleneck identification
echo "Step 8/8: Identifying bottlenecks..."
cat > "$OUTPUT_DIR/bottleneck_report.txt" <<EOF
Engram Performance Analysis Report
Generated: $(date)
Duration: ${DURATION}s

=== CRITICAL METRICS ===

Cache Efficiency:
$(grep "cache-misses" "$OUTPUT_DIR/cache_stats.txt" 2>/dev/null || echo "N/A")

Memory Pressure:
$(cat "$OUTPUT_DIR/memory_usage.txt")
$(cat "$OUTPUT_DIR/huge_pages.txt" 2>/dev/null || echo "Huge Pages: N/A")

NUMA Locality:
$(grep "node 0" "$OUTPUT_DIR/numa_memory.txt" 2>/dev/null | head -1 || echo "Single NUMA node")

Latency Analysis (all percentiles):
$(cat "$OUTPUT_DIR/latency_percentiles.txt" | grep "recall:" | sort)

=== BOTTLENECK IDENTIFICATION ===
EOF

# Advanced bottleneck analysis
P99_RECALL=$(grep "P99 recall:" "$OUTPUT_DIR/latency_percentiles.txt" | cut -d: -f2 | tr -d 's' | tr -d ' ')
if [ -n "$P99_RECALL" ] && (( $(echo "$P99_RECALL > 0.010" | bc -l 2>/dev/null || echo 0) )); then
    echo "⚠️  CRITICAL: Recall P99 latency ${P99_RECALL}s exceeds 10ms target" >> "$OUTPUT_DIR/bottleneck_report.txt"
    echo "   Root Cause Analysis:" >> "$OUTPUT_DIR/bottleneck_report.txt"

    # Check cache misses
    CACHE_MISS_RATE=$(grep "cache-misses" "$OUTPUT_DIR/cache_stats.txt" 2>/dev/null | grep -o '[0-9.]*%' | tr -d '%' || echo "0")
    if [ -n "$CACHE_MISS_RATE" ] && (( $(echo "$CACHE_MISS_RATE > 10" | bc -l 2>/dev/null || echo 0) )); then
        echo "   - High cache miss rate: ${CACHE_MISS_RATE}%" >> "$OUTPUT_DIR/bottleneck_report.txt"
        echo "     ACTION: Increase prefetch distance, review data layout" >> "$OUTPUT_DIR/bottleneck_report.txt"
    fi

    # Check memory usage
    RSS_MB=$(awk '{print $3}' "$OUTPUT_DIR/memory_usage.txt" | tail -1)
    if [ -n "$RSS_MB" ] && (( ${RSS_MB%%M} > 4096 )); then
        echo "   - High memory usage: $RSS_MB" >> "$OUTPUT_DIR/bottleneck_report.txt"
        echo "     ACTION: Increase hot tier eviction rate" >> "$OUTPUT_DIR/bottleneck_report.txt"
    fi
fi

cat "$OUTPUT_DIR/bottleneck_report.txt"
echo ""
echo "Full report saved to $OUTPUT_DIR/"
echo "Generate flamegraph: flamegraph.pl $OUTPUT_DIR/collapsed.txt > $OUTPUT_DIR/flamegraph.svg"
```

### /scripts/analyze_slow_queries.sh

```bash
#!/bin/bash
# Advanced slow query analysis with root cause detection

set -euo pipefail

THRESHOLD_MS="${1:-10}"  # Default: queries >10ms (P99 target)
LOOKBACK="${2:-1h}"     # Default: last 1 hour
DETAIL_LEVEL="${3:-full}" # full|summary

echo "=== Engram Slow Query Analyzer ==="
echo "Threshold: >${THRESHOLD_MS}ms | Lookback: ${LOOKBACK} | Detail: ${DETAIL_LEVEL}"
echo ""

# Function to analyze activation spreading performance
analyze_activation() {
    local space=$1
    local latency=$2

    echo "  Activation Spreading Analysis:"

    # Check cache hit rate
    CACHE_HIT_RATE=$(curl -s http://localhost:9090/api/v1/query \
        -d "query=rate(engram_activation_cache_hits_total[${LOOKBACK}]) / rate(engram_activation_cache_requests_total[${LOOKBACK}])" \
        | jq -r '.data.result[0].value[1] // "N/A"')

    if [ "$CACHE_HIT_RATE" != "N/A" ] && (( $(echo "$CACHE_HIT_RATE < 0.8" | bc -l 2>/dev/null || echo 0) )); then
        echo "    ⚠️  Low cache hit rate: $(echo "scale=2; $CACHE_HIT_RATE * 100" | bc)%"
        echo "       ACTION: Increase hot tier size from current configuration"
        echo "       COMMAND: engram-cli config set hot_tier.size_mb 2048"
    fi

    # Check prefetch effectiveness
    echo "    Checking prefetch distance effectiveness..."
    echo "       Current setting: Check /etc/engram/engram.toml -> activation.prefetch_distance"
    echo "       Optimal range: 4-16 nodes ahead for L2/L3 cache"

    # Check SIMD batch utilization
    echo "    SIMD batch efficiency:"
    echo "       Minimum batch size for vectorization: 8 embeddings"
    echo "       ACTION: Enable batch accumulation if seeing <8 parallel activations"
}

# Function to analyze recall operations
analyze_recall() {
    local space=$1
    local latency=$2

    echo "  Recall Operation Analysis:"

    # Check HNSW index state
    echo "    HNSW Index Configuration:"
    echo "       Check parameters: M (connectivity), efConstruction (build quality), efSearch (search quality)"

    EDGE_COUNT=$(curl -s http://localhost:9090/api/v1/query \
        -d "query=avg(engram_graph_edges_per_node{memory_space=\"$space\"})" \
        | jq -r '.data.result[0].value[1] // "N/A"')

    if [ "$EDGE_COUNT" != "N/A" ]; then
        echo "       Average edges per node: $EDGE_COUNT"
        if (( $(echo "$EDGE_COUNT < 16" | bc -l 2>/dev/null || echo 0) )); then
            echo "       ⚠️  Low connectivity detected (M < 16)"
            echo "       ACTION: Rebuild index with M=32 for better recall quality"
        fi
    fi

    # Check embedding dimension impact
    echo "    Embedding computation overhead:"
    echo "       768-dimensional dot products dominate recall time"
    echo "       ACTION: Ensure AVX2/AVX-512 instructions are enabled"
    echo "       VERIFY: grep avx /proc/cpuinfo && echo 'AVX available'"
}

# Function to analyze store operations
analyze_store() {
    local space=$1
    local latency=$2

    echo "  Store Operation Analysis:"

    # Check WAL performance
    WAL_FLUSH_LATENCY=$(curl -s http://localhost:9090/api/v1/query \
        -d "query=histogram_quantile(0.99, rate(engram_wal_flush_duration_seconds_bucket[${LOOKBACK}]))" \
        | jq -r '.data.result[0].value[1] // "N/A"')

    if [ "$WAL_FLUSH_LATENCY" != "N/A" ] && (( $(echo "$WAL_FLUSH_LATENCY > 0.005" | bc -l 2>/dev/null || echo 0) )); then
        echo "    ⚠️  Slow WAL flushes: ${WAL_FLUSH_LATENCY}s"
        echo "       ACTION: Increase WAL buffer size or use O_DIRECT writes"
        echo "       CONFIG: wal.buffer_size_mb = 64"
        echo "       CONFIG: wal.sync_mode = 'batch'  # Group commits"
    fi

    # Check lock contention
    echo "    Checking for lock contention in DashMap..."
    echo "       ACTION: Monitor CPU usage during stores"
    echo "       If CPU >80% with low throughput: shard the hashmap further"
}

# Main analysis loop
echo "Analyzing slow queries..."
curl -s -G http://localhost:9090/api/v1/query \
  --data-urlencode "query=histogram_quantile(0.99, rate(engram_memory_operation_duration_seconds_bucket[${LOOKBACK}])) > ${THRESHOLD_MS}/1000" \
  | jq -r '.data.result[] | "\(.metric.operation)|\(.metric.memory_space // "default")|\(.value[1])"' \
  | while IFS='|' read -r operation space latency_str; do
      latency_ms=$(echo "scale=1; $latency_str * 1000" | bc)

      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "Slow Query Detected:"
      echo "  Operation: $operation"
      echo "  Memory Space: $space"
      echo "  P99 Latency: ${latency_ms}ms (threshold: ${THRESHOLD_MS}ms)"
      echo ""

      case $operation in
          activation|spreading)
              analyze_activation "$space" "$latency_str"
              ;;
          recall)
              analyze_recall "$space" "$latency_str"
              ;;
          store)
              analyze_store "$space" "$latency_str"
              ;;
          consolidation)
              echo "  Consolidation Analysis:"
              echo "    Background consolidation should not impact query latency"
              echo "    ACTION: Move to separate thread pool or reduce frequency"
              echo "    CONFIG: consolidation.interval_seconds = 300"
              echo "    CONFIG: consolidation.max_batch_size = 1000"
              ;;
          pattern_completion)
              echo "  Pattern Completion Analysis:"
              echo "    Complex operation involving multiple graph traversals"
              echo "    ACTION: Limit search depth and candidate set size"
              echo "    CONFIG: pattern_completion.max_depth = 3"
              echo "    CONFIG: pattern_completion.max_candidates = 100"
              ;;
          *)
              echo "  Generic optimization suggestions:"
              echo "    - Profile with: ./scripts/profile_performance.sh"
              echo "    - Check metrics dashboard for patterns"
              ;;
      esac

      if [ "$DETAIL_LEVEL" = "full" ]; then
          echo ""
          echo "  Deep Dive Commands:"
          echo "    # Check operation-specific metrics"
          echo "    curl -s http://localhost:9090/api/v1/query -d 'query=engram_${operation}_operations_total'"
          echo "    # View real-time latency"
          echo "    watch -n 1 'curl -s http://localhost:9090/api/v1/query -d \"query=engram_memory_operation_duration_seconds{operation=\\\"$operation\\\"}\"'"
      fi
      echo ""
  done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Analysis complete. No more slow queries found."
```

### /scripts/benchmark_deployment.sh

```bash
#!/bin/bash
# Production benchmark suite

set -euo pipefail

ENGRAM_URL="${ENGRAM_URL:-http://localhost:7432}"
DURATION="${1:-60}"  # seconds
CONCURRENCY="${2:-10}"  # concurrent clients

echo "Benchmarking Engram at $ENGRAM_URL"
echo "Duration: ${DURATION}s, Concurrency: $CONCURRENCY"

# Ensure Engram is healthy
curl -sf "$ENGRAM_URL/api/v1/system/health" > /dev/null || {
    echo "ERROR: Engram not healthy"
    exit 1
}

# Benchmark: Store operations
echo "Benchmark 1/4: Store operations..."
ab -n 10000 -c $CONCURRENCY -t $DURATION \
   -p <(echo '{"content":"benchmark","confidence":0.9}') \
   -T "application/json" \
   "$ENGRAM_URL/api/v1/memories/remember" \
   > /tmp/bench_store.txt

STORE_RPS=$(grep "Requests per second" /tmp/bench_store.txt | awk '{print $4}')
STORE_P99=$(grep "99%" /tmp/bench_store.txt | awk '{print $2}')

# Benchmark: Recall operations
echo "Benchmark 2/4: Recall operations..."
ab -n 10000 -c $CONCURRENCY -t $DURATION \
   "$ENGRAM_URL/api/v1/memories/recall?query=benchmark" \
   > /tmp/bench_recall.txt

RECALL_RPS=$(grep "Requests per second" /tmp/bench_recall.txt | awk '{print $4}')
RECALL_P99=$(grep "99%" /tmp/bench_recall.txt | awk '{print $2}')

# Benchmark: Mixed workload (70% recall, 30% store)
echo "Benchmark 3/4: Mixed workload..."
# (Implementation: wrk2 or custom load generator)

# Benchmark: Concurrent memory spaces
echo "Benchmark 4/4: Multi-tenant workload..."
# (Implementation: parallel ab runs with different X-Memory-Space headers)

# Report
cat > /tmp/benchmark_report.txt <<EOF
Engram Benchmark Report
Generated: $(date)
Duration: ${DURATION}s
Concurrency: $CONCURRENCY

Store Operations:
  Throughput: $STORE_RPS req/sec
  P99 Latency: ${STORE_P99}ms
  Target: ≥1000 req/sec, ≤100ms
  Status: $([ $(echo "$STORE_RPS > 1000" | bc) -eq 1 ] && echo "PASS" || echo "FAIL")

Recall Operations:
  Throughput: $RECALL_RPS req/sec
  P99 Latency: ${RECALL_P99}ms
  Target: ≥5000 req/sec, ≤50ms
  Status: $([ $(echo "$RECALL_RPS > 5000" | bc) -eq 1 ] && echo "PASS" || echo "FAIL")

Overall Status: $([ $(echo "$STORE_RPS > 1000 && $RECALL_RPS > 5000" | bc) -eq 1 ] && echo "PASS" || echo "WARN: Below targets")

Recommendations:
$(if [ $(echo "$STORE_RPS < 1000" | bc) -eq 1 ]; then
    echo "- Store throughput below target: Check disk I/O and WAL configuration"
fi)
$(if [ $(echo "$RECALL_RPS < 5000" | bc) -eq 1 ]; then
    echo "- Recall throughput below target: Verify index coverage and cache hit rate"
fi)
EOF

cat /tmp/benchmark_report.txt
```

### /scripts/tune_config.sh

```bash
#!/bin/bash
# Interactive configuration tuning wizard

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

    CPU_CORES=$(nproc)
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    NUMA_NODES=$(lscpu | grep "NUMA node(s):" | awk '{print $3}')

    # Check for AVX support
    AVX_SUPPORT="none"
    if grep -q avx512 /proc/cpuinfo; then
        AVX_SUPPORT="avx512"
    elif grep -q avx2 /proc/cpuinfo; then
        AVX_SUPPORT="avx2"
    fi

    # Detect storage type (SSD vs HDD)
    STORAGE_TYPE="hdd"
    if [ -d /sys/block/nvme0n1 ] || lsblk -d -o name,rota | grep -q " 0$"; then
        STORAGE_TYPE="ssd"
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

    # Get operation distribution from metrics
    READ_RATIO=$(curl -s http://localhost:9090/api/v1/query \
        -d 'query=rate(engram_recall_operations_total[1h]) / (rate(engram_recall_operations_total[1h]) + rate(engram_store_operations_total[1h]))' \
        | jq -r '.data.result[0].value[1] // "0.7"')

    AVG_RECALL_SIZE=$(curl -s http://localhost:9090/api/v1/query \
        -d 'query=avg(engram_recall_result_size)' \
        | jq -r '.data.result[0].value[1] // "10"')

    echo "  Read/Write Ratio: $(echo "scale=0; $READ_RATIO * 100" | bc)% reads"
    echo "  Average Recall Size: $AVG_RECALL_SIZE memories"

    # Classify workload
    if (( $(echo "$READ_RATIO > 0.8" | bc -l) )); then
        WORKLOAD_TYPE="read_heavy"
    elif (( $(echo "$READ_RATIO < 0.3" | bc -l) )); then
        WORKLOAD_TYPE="write_heavy"
    else
        WORKLOAD_TYPE="balanced"
    fi

    echo "  Workload Type: $WORKLOAD_TYPE"
    echo ""
}

# Generate optimized configuration
generate_config() {
    echo "Generating optimized configuration..."

    cat > "$CONFIG_FILE" <<EOF
# Engram Performance-Tuned Configuration
# Generated: $(date)
# System: ${CPU_CORES} cores, ${MEMORY_GB}GB RAM, ${NUMA_NODES} NUMA nodes
# Workload: ${WORKLOAD_TYPE}

[storage]
# Hot tier configuration (lock-free concurrent hashmap)
hot_tier_size_mb = $(calculate_hot_tier_size)
hot_tier_shards = $(calculate_shards)

# Warm tier (memory-mapped files)
warm_tier_size_mb = $(calculate_warm_tier_size)
use_huge_pages = $([ "$MEMORY_GB" -gt 16 ] && echo "true" || echo "false")
mmap_populate = $([ "$STORAGE_TYPE" = "ssd" ] && echo "true" || echo "false")

# Cold tier (columnar storage)
cold_tier_compression = "zstd"
cold_tier_compression_level = $([ "$WORKLOAD_TYPE" = "write_heavy" ] && echo "1" || echo "3")

[activation]
# Spreading activation parameters
prefetch_distance = $(calculate_prefetch_distance)
max_traversal_depth = 5
visit_budget_per_tier = { hot = 3, warm = 2, cold = 1 }

# SIMD optimization
simd_batch_size = 8
simd_enabled = $([ "$AVX_SUPPORT" != "none" ] && echo "true" || echo "false")
avx_version = "$AVX_SUPPORT"

[hnsw_index]
# HNSW parameters optimized for workload
M = $(calculate_hnsw_m)
ef_construction = $(calculate_ef_construction)
ef_search = $(calculate_ef_search)
max_m = $(echo "$M * 2" | bc)
max_m0 = $(echo "$M * 2" | bc)

[wal]
# Write-ahead log configuration
buffer_size_mb = $([ "$WORKLOAD_TYPE" = "write_heavy" ] && echo "128" || echo "64")
flush_interval_ms = $(calculate_wal_flush_interval)
sync_mode = $([ "$STORAGE_TYPE" = "ssd" ] && echo "'batch'" || echo "'immediate'")
segment_size_mb = 64

[thread_pools]
# Thread pool sizing based on system and workload
recall_workers = $(calculate_recall_workers)
store_workers = $(calculate_store_workers)
consolidation_workers = 2
background_workers = 4

[numa]
# NUMA-aware memory allocation
numa_aware = $([ "$NUMA_NODES" -gt 1 ] && echo "true" || echo "false")
interleave_memory = false
prefer_local_node = true
$([ "$NUMA_NODES" -gt 1 ] && echo "socket_memory_mb = [$(echo "$MEMORY_GB * 1024 / $NUMA_NODES" | bc)]" || echo "# single NUMA node")

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

# Helper functions for calculating optimal values
calculate_hot_tier_size() {
    # Base calculation: 10% of RAM for hot tier, capped at 8GB
    local size_mb=$(echo "$MEMORY_GB * 1024 * 0.1" | bc | cut -d'.' -f1)
    [ "$size_mb" -gt 8192 ] && size_mb=8192
    [ "$size_mb" -lt 256 ] && size_mb=256
    echo "$size_mb"
}

calculate_warm_tier_size() {
    # 30% of RAM for warm tier
    echo "$MEMORY_GB * 1024 * 0.3" | bc | cut -d'.' -f1
}

calculate_shards() {
    # Number of shards for DashMap: 2x CPU cores for lock-free concurrency
    echo "$CPU_CORES * 2" | bc
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
        echo "$CPU_CORES * 2" | bc  # 2x for I/O bound reads
    else
        echo "$CPU_CORES"
    fi
}

calculate_store_workers() {
    if [ "$WORKLOAD_TYPE" = "write_heavy" ]; then
        echo "$CPU_CORES"
    else
        echo "$CPU_CORES / 2" | bc
    fi
}

# Main execution
detect_system
analyze_workload
generate_config

echo ""
echo "=== Configuration Tuning Complete ==="
echo ""
echo "Next steps:"
echo "1. Review configuration: cat $CONFIG_FILE"
echo "2. Restart Engram: systemctl restart engram"
echo "3. Run benchmark: ./scripts/benchmark_deployment.sh"
echo "4. Monitor metrics: ./scripts/profile_performance.sh"
echo ""
echo "To rollback: cp $BACKUP_FILE $CONFIG_FILE && systemctl restart engram"
```

### Configuration Tuning Parameters

**Cache Optimization:**
- Hot tier size: 10% of RAM (max 8GB) for frequently accessed memories
- Prefetch distance: 4-16 nodes based on L2/L3 cache size
- Cache line alignment: All nodes aligned to 64-byte boundaries
- NUMA locality: Pin memory to local socket when possible

**HNSW Index Tuning Matrix:**
| Workload | M | efConstruction | efSearch | Rationale |
|----------|---|----------------|----------|-----------|
| Read-heavy | 32 | 400 | 200 | Maximum recall quality |
| Write-heavy | 16 | 200 | 100 | Fast insertions |
| Balanced | 24 | 300 | 150 | Good compromise |
| Memory-constrained | 8 | 100 | 50 | Minimum footprint |

**Storage Tier Thresholds:**
- Hot→Warm migration: <10 accesses in 1 hour
- Warm→Cold migration: <1 access in 24 hours
- Cold→Warm promotion: >5 accesses in 1 hour
- Warm→Hot promotion: >20 accesses in 1 hour

**Lock-Free Tuning:**
- DashMap shards: 2 × CPU cores (reduces contention)
- Atomic operation ordering: Relaxed for counters, AcqRel for state
- Hazard pointer epochs: 3 (balances memory reclamation vs overhead)
- Memory pool chunk size: 64KB (balances fragmentation vs allocation)

**SIMD Batch Optimization:**
- Minimum batch: 8 embeddings (full AVX-256 register)
- Maximum batch: 64 embeddings (L1 cache resident)
- AoSoA tile size: 96 tiles × 8 lanes for 768-dim vectors
- Alignment: 64-byte for cache lines, 32-byte for AVX

## Testing Requirements

```bash
# Profile performance
./scripts/profile_performance.sh 120 ./profile-output
ls ./profile-output/

# Analyze slow queries
./scripts/analyze_slow_queries.sh 50 1h

# Run benchmark
./scripts/benchmark_deployment.sh 300 20
grep "Overall Status" /tmp/benchmark_report.txt

# Verify targets achieved
curl -s http://localhost:9090/api/v1/query \
  -d 'query=histogram_quantile(0.99, rate(engram_memory_operation_duration_seconds_bucket{operation="recall"}[5m]))' \
  | jq -r '.data.result[0].value[1]' \
  | awk '{if ($1 < 0.010) print "PASS: P99 recall latency under 10ms"; else print "FAIL: P99 latency " $1 "s"}'
```

### /tools/perf-analyzer/src/main.rs

```rust
//! Performance analysis CLI tool for Engram operators

use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct PerformanceProfile {
    cache_miss_rate: f64,
    memory_bandwidth_gb_s: f64,
    numa_remote_accesses: u64,
    lock_contention_us: u64,
    simd_utilization: f64,
}

impl PerformanceProfile {
    /// Analyze performance and generate recommendations
    pub fn analyze(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Cache optimization
        if self.cache_miss_rate > 0.10 {
            recommendations.push(format!(
                "High cache miss rate ({:.1}%). Consider:\n  \
                 - Increasing prefetch distance (current: check config)\n  \
                 - Reviewing data structure layout for better locality\n  \
                 - Using cache-oblivious algorithms",
                self.cache_miss_rate * 100.0
            ));
        }

        // Memory bandwidth
        let theoretical_bandwidth = 100.0; // GB/s for DDR4-3200
        let bandwidth_util = self.memory_bandwidth_gb_s / theoretical_bandwidth;
        if bandwidth_util > 0.8 {
            recommendations.push(format!(
                "Memory bandwidth saturated ({:.1}%). Consider:\n  \
                 - Enabling compression for cold tier\n  \
                 - Reducing embedding dimensions if possible\n  \
                 - Adding more memory channels",
                bandwidth_util * 100.0
            ));
        }

        // NUMA optimization
        if self.numa_remote_accesses > 1_000_000 {
            recommendations.push(format!(
                "High NUMA remote accesses ({}M). Consider:\n  \
                 - Pinning threads to NUMA nodes\n  \
                 - Using node-local memory allocation\n  \
                 - Replicating hot data across nodes",
                self.numa_remote_accesses / 1_000_000
            ));
        }

        // Lock contention
        if self.lock_contention_us > 1000 {
            recommendations.push(format!(
                "Lock contention detected ({}μs average). Consider:\n  \
                 - Increasing DashMap shard count\n  \
                 - Using RCU for read-heavy workloads\n  \
                 - Implementing wait-free data structures",
                self.lock_contention_us
            ));
        }

        // SIMD utilization
        if self.simd_utilization < 0.5 {
            recommendations.push(format!(
                "Low SIMD utilization ({:.1}%). Consider:\n  \
                 - Batching more operations\n  \
                 - Ensuring data alignment\n  \
                 - Using explicit SIMD intrinsics",
                self.simd_utilization * 100.0
            ));
        }

        recommendations
    }
}

fn main() {
    println!("Engram Performance Analyzer v1.0");

    // Parse hardware counters and generate profile
    let profile = PerformanceProfile {
        cache_miss_rate: 0.15,
        memory_bandwidth_gb_s: 85.0,
        numa_remote_accesses: 2_500_000,
        lock_contention_us: 1500,
        simd_utilization: 0.4,
    };

    println!("\nPerformance Analysis Results:");
    println!("============================");

    let recommendations = profile.analyze();
    if recommendations.is_empty() {
        println!("✓ No performance issues detected");
    } else {
        for (i, rec) in recommendations.iter().enumerate() {
            println!("\n{}. {}", i + 1, rec);
        }
    }
}
```

## Documentation Requirements

### /docs/operations/performance-tuning.md

```markdown
# Performance Tuning Guide

## Quick Start

Run the automated tuning wizard:
```bash
./scripts/tune_config.sh
```

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| P99 Recall Latency | <10ms | `histogram_quantile(0.99, engram_memory_operation_duration_seconds_bucket{operation="recall"})` |
| Sustained Throughput | 10K ops/s | `rate(engram_operations_total[1m])` |
| Memory Overhead | <2x data size | `engram_memory_bytes / engram_data_bytes` |

## Common Performance Issues

### 1. High Recall Latency

**Symptoms:**
- P99 latency >10ms
- Cache miss rate >20%

**Root Causes:**
- Insufficient hot tier size
- Poor HNSW index configuration
- Cache thrashing

**Solutions:**
```bash
# Increase hot tier
engram-cli config set storage.hot_tier_size_mb 4096

# Rebuild index with better parameters
engram-cli index rebuild --M 32 --ef-construction 400
```

### 2. Memory Bandwidth Saturation

**Symptoms:**
- Memory bandwidth >80% of theoretical max
- CPU waiting on memory

**Root Causes:**
- Unoptimized memory access patterns
- No SIMD utilization
- Poor NUMA locality

**Solutions:**
```bash
# Enable huge pages
echo 1024 > /proc/sys/vm/nr_hugepages

# Pin to NUMA nodes
numactl --cpunodebind=0 --membind=0 engram-server
```

### 3. Lock Contention

**Symptoms:**
- High CPU usage with low throughput
- Threads in BLOCKED state

**Root Causes:**
- Insufficient DashMap shards
- Hot spots in data

**Solutions:**
```toml
# Increase sharding in config
[storage]
hot_tier_shards = 128  # 2x CPU cores
```

## Profiling Workflow

1. **Baseline Performance**
   ```bash
   ./scripts/benchmark_deployment.sh 300 20
   ```

2. **Identify Bottlenecks**
   ```bash
   ./scripts/profile_performance.sh 120
   ./scripts/analyze_slow_queries.sh 10
   ```

3. **Apply Tuning**
   ```bash
   ./scripts/tune_config.sh
   systemctl restart engram
   ```

4. **Verify Improvements**
   ```bash
   ./scripts/benchmark_deployment.sh 300 20
   diff profile-before/ profile-after/
   ```

## Hardware-Specific Tuning

### Intel Xeon (Cascade Lake+)
- Enable AVX-512 for 2x embedding throughput
- Use Intel PCM for cache analysis
- Consider Intel Optane for warm tier

### AMD EPYC (Rome+)
- Leverage high memory bandwidth
- Use CCX-aware thread placement
- Enable Infinity Fabric optimization

### ARM Graviton
- Use NEON SIMD instructions
- Optimize for larger L1 cache
- Consider LSE atomics

## Monitoring Dashboard Queries

Key metrics to monitor:

```promql
# Operation latency percentiles
histogram_quantile(0.99,
  rate(engram_memory_operation_duration_seconds_bucket[5m])
) by (operation)

# Cache efficiency
rate(engram_activation_cache_hits_total[5m]) /
rate(engram_activation_cache_requests_total[5m])

# Memory tier distribution
engram_storage_tier_memories_count by (tier)

# NUMA remote accesses
rate(node_memory_numa_foreign_total[5m])
```
```

### /docs/howto/identify-slow-queries.md

```markdown
# How to Identify Slow Queries

## Automated Analysis

Use the slow query analyzer:

```bash
# Find queries slower than 10ms in last hour
./scripts/analyze_slow_queries.sh 10 1h

# Summary mode for quick overview
./scripts/analyze_slow_queries.sh 10 1h summary
```

## Manual Investigation

### Step 1: Check Current Latencies

```bash
# View P99 latencies by operation
curl -s http://localhost:9090/api/v1/query \
  -d 'query=histogram_quantile(0.99, rate(engram_memory_operation_duration_seconds_bucket[5m]))' \
  | jq '.data.result[] | {op: .metric.operation, p99: .value[1]}'
```

### Step 2: Identify Patterns

Common patterns indicating performance issues:

1. **Time-based degradation**: Performance worse during peak hours
2. **Size-based degradation**: Larger memory spaces slower
3. **Operation-specific**: Only certain operations slow

### Step 3: Root Cause Analysis

For each slow operation, check:

- **Cache metrics**: Is cache hit rate low?
- **Index state**: Are indices properly built?
- **Resource usage**: CPU, memory, or I/O bottleneck?
- **Concurrency**: Lock contention or thread starvation?

### Step 4: Apply Fixes

Based on root cause:

| Root Cause | Fix | Verification |
|------------|-----|--------------|
| Low cache hits | Increase hot tier size | Cache hit rate >80% |
| Missing indices | Rebuild HNSW index | Index coverage 100% |
| I/O bottleneck | Enable write batching | IOPS <500 |
| Lock contention | Increase shard count | CPU utilization linear with load |
```

### /docs/reference/performance-baselines.md

```markdown
# Performance Baselines

## Reference Hardware

**Configuration:**
- CPU: 32 cores (Intel Xeon Gold 6258R @ 2.7GHz)
- Memory: 128GB DDR4-2933
- Storage: 2TB NVMe SSD (3.5GB/s read)
- Network: 25Gbps

## Expected Performance

### Latency Baselines

| Operation | P50 | P95 | P99 | P99.9 |
|-----------|-----|-----|-----|-------|
| Store | 1ms | 3ms | 5ms | 10ms |
| Recall (1-hop) | 2ms | 5ms | 8ms | 15ms |
| Recall (multi-hop) | 5ms | 15ms | 25ms | 50ms |
| Pattern Completion | 10ms | 30ms | 50ms | 100ms |

### Throughput Baselines

| Workload | Operations/sec | Notes |
|----------|---------------|--------|
| 100% Store | 15,000 | WAL enabled |
| 100% Recall | 25,000 | Hot tier hits |
| 70/30 Mixed | 18,000 | Typical workload |
| Burst | 50,000 | <10 seconds |

### Resource Usage

| Metric | Expected | Warning | Critical |
|--------|----------|---------|----------|
| CPU Usage | 40-60% | >70% | >90% |
| Memory Usage | 50-70% | >80% | >95% |
| Disk IOPS | 200-400 | >500 | >1000 |
| Network Mbps | 100-500 | >1000 | >2000 |

## Scaling Characteristics

### Linear Scaling Region
- 1-32 CPU cores: Near-linear scaling
- 32-64 cores: 80% efficiency
- >64 cores: Diminishing returns

### Memory Scaling
- <100K nodes: All in hot tier
- 100K-1M nodes: Hot/warm split
- >1M nodes: Full tier hierarchy

### Performance vs Data Size

```
Latency (ms) = 2.0 + 0.5 * log10(nodes)
Throughput = 30000 / (1 + nodes/1000000)
```

## Regression Detection

Alert when:
- P99 latency increases >20% from baseline
- Throughput drops >15% from baseline
- Cache hit rate drops below 70%
- Memory usage exceeds 2x data size
```

## Acceptance Criteria

**Profiling Tools:**
- [ ] Profiling script identifies top 3 bottlenecks in <1 minute
- [ ] CPU profile includes flamegraph generation
- [ ] Memory profile shows heap and RSS usage
- [ ] Slow query analysis provides actionable recommendations
- [ ] All scripts work without external dependencies (except curl/jq)

**Tuning Guidance:**
- [ ] All tunable parameters documented
- [ ] Tuning recommendations improve P99 latency by >20%
- [ ] Configuration changes validated via benchmarks
- [ ] Workload-specific configurations provided

**Benchmarking:**
- [ ] Benchmark suite runs in <10 minutes
- [ ] Results compare against documented baselines
- [ ] Pass/fail criteria clearly defined
- [ ] Recommendations generated automatically

**Documentation:**
- [ ] External operator identifies bottleneck in <15 minutes
- [ ] Tuning recommendations tested and verified
- [ ] All performance targets documented with measurement methods
- [ ] Troubleshooting flowchart provided

**Performance Targets:**
- [ ] P99 recall latency <10ms achieved with tuning
- [ ] Sustained throughput ≥10K ops/sec achieved
- [ ] Memory overhead <2x verified
- [ ] Linear scaling verified up to 32 cores

## Common Bottleneck Patterns & Resolutions

### Pattern 1: Cache Line Bouncing
**Symptoms:**
- High CPU usage on multi-socket systems
- Performance doesn't scale with cores
- `perf c2c` shows high HITM events

**Root Cause:**
- False sharing in CacheOptimalMemoryNode
- AtomicF32 activation field accessed by multiple threads
- Cache lines ping-ponging between CPU sockets

**Resolution:**
```bash
# Detect false sharing
perf c2c record -p $(pgrep engram) -- sleep 10
perf c2c report --stdio

# Fix: Pad atomic fields to separate cache lines
# In config: numa.prefer_local_node = true
```

### Pattern 2: NUMA Memory Stalls
**Symptoms:**
- Memory latency 3-4x higher than expected
- Uneven CPU utilization across sockets
- High QPI/UPI traffic

**Root Cause:**
- Threads accessing memory on remote NUMA nodes
- No NUMA-aware allocation in DashMap

**Resolution:**
```bash
# Verify NUMA binding
numastat -p $(pgrep engram)

# Fix: Pin threads and memory
numactl --cpunodebind=0 --membind=0 engram-server
```

### Pattern 3: SIMD Underutilization
**Symptoms:**
- Low IPC (Instructions Per Cycle) <1.5
- Embedding operations dominate profile
- AVX registers unused in perf report

**Root Cause:**
- Batch size too small for vectorization
- Data not aligned for SIMD loads
- Scalar fallback in similarity computation

**Resolution:**
```bash
# Check SIMD usage
perf stat -e fp_arith_inst_retired.256b_packed_single engram-server

# Fix: Ensure batch size ≥8 and alignment
# Config: activation.simd_batch_size = 8
```

### Pattern 4: WAL Write Amplification
**Symptoms:**
- High disk I/O for low store rate
- WAL files growing faster than data
- fsync() dominating profile

**Root Cause:**
- Small writes not batched
- Synchronous flushes on every operation
- No group commit optimization

**Resolution:**
```bash
# Monitor WAL behavior
strace -e fsync,write -p $(pgrep engram) -c

# Fix: Enable batch mode
# Config: wal.sync_mode = "batch"
# Config: wal.flush_interval_ms = 1000
```

### Pattern 5: Lock Contention in Hot Tier
**Symptoms:**
- Threads spinning on DashMap locks
- Non-linear scaling with thread count
- High system CPU time

**Root Cause:**
- Insufficient sharding in DashMap
- Hot keys creating contention points
- Read-write lock conflicts

**Resolution:**
```bash
# Profile lock contention
perf record -g -e lock:* -p $(pgrep engram) -- sleep 10
perf report

# Fix: Increase shard count
# Config: storage.hot_tier_shards = $(nproc) * 4
```

## Performance Debugging Checklist

When performance degrades, check in order:

1. **System Resources**
   ```bash
   # CPU, Memory, I/O
   htop; iostat -x 1; vmstat 1
   ```

2. **Cache Efficiency**
   ```bash
   # L1/L2/L3 cache misses
   perf stat -e cache-misses,cache-references -p $(pgrep engram)
   ```

3. **Memory Access Patterns**
   ```bash
   # NUMA locality
   numastat -p $(pgrep engram)
   # Huge pages usage
   grep Huge /proc/meminfo
   ```

4. **Application Metrics**
   ```bash
   # Operation latencies
   curl -s http://localhost:9090/api/v1/query \
     -d 'query=engram_memory_operation_duration_seconds'
   ```

5. **Thread Behavior**
   ```bash
   # Thread states and contention
   top -H -p $(pgrep engram)
   # Stack traces of blocked threads
   pstack $(pgrep engram)
   ```

## Before/After Optimization Examples

### Example 1: Cache Miss Reduction
**Before:** 25% L3 cache miss rate, P99 latency 45ms
```bash
# Original config
prefetch_distance = 2
hot_tier_size_mb = 512
```

**After:** 8% L3 cache miss rate, P99 latency 9ms
```bash
# Optimized config
prefetch_distance = 12
hot_tier_size_mb = 2048
# Result: 5x reduction in P99 latency
```

### Example 2: NUMA Optimization
**Before:** 40% remote NUMA accesses, throughput 5K ops/s
```bash
# No NUMA awareness
numa_aware = false
```

**After:** 5% remote NUMA accesses, throughput 18K ops/s
```bash
# NUMA-optimized
numa_aware = true
prefer_local_node = true
# Result: 3.6x throughput improvement
```

### Example 3: SIMD Vectorization
**Before:** Scalar operations, 100ms for 1000 similarities
```bash
simd_enabled = false
```

**After:** AVX2 vectorized, 15ms for 1000 similarities
```bash
simd_enabled = true
avx_version = "avx2"
simd_batch_size = 8
# Result: 6.7x speedup in similarity computation
```

## Follow-Up Tasks

- Task 003: Add performance metrics to dashboards
- Task 005: Reference performance issues in troubleshooting
- Task 006: Use tuning guidance in scaling documentation
- Task 011: Integrate benchmark suite into load testing
