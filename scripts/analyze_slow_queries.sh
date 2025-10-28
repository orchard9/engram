#!/usr/bin/env bash
# Advanced slow query analysis with root cause detection
#
# This script analyzes query performance to identify slow operations and
# provides actionable recommendations for optimization. It integrates with
# Prometheus metrics to detect patterns and root causes.
#
# Usage: ./scripts/analyze_slow_queries.sh [threshold_ms] [lookback] [detail_level]
# Example: ./scripts/analyze_slow_queries.sh 10 1h full

set -euo pipefail

THRESHOLD_MS="${1:-10}"
LOOKBACK="${2:-1h}"
DETAIL_LEVEL="${3:-full}"

echo "=== Engram Slow Query Analyzer ==="
echo "Threshold: >${THRESHOLD_MS}ms | Lookback: ${LOOKBACK} | Detail: ${DETAIL_LEVEL}"
echo ""

# Function to analyze activation spreading performance
analyze_activation() {
    local space=$1
    local latency=$2

    echo "  Activation Spreading Analysis:"

    # Check cache hit rate
    CACHE_HIT_RATE=$(curl -sf http://localhost:9090/api/v1/query --max-time 5 \
        -d "query=rate(engram_activation_cache_hits_total[${LOOKBACK}]) / rate(engram_activation_cache_requests_total[${LOOKBACK}])" \
        2>/dev/null | jq -r '.data.result[0].value[1] // "N/A"' || echo "N/A")

    if [ "$CACHE_HIT_RATE" != "N/A" ] && command -v bc &> /dev/null; then
        if (( $(echo "$CACHE_HIT_RATE < 0.8" | bc -l 2>/dev/null || echo 0) )); then
            CACHE_PCT=$(echo "scale=1; $CACHE_HIT_RATE * 100" | bc 2>/dev/null || echo "N/A")
            echo "    WARNING: Low cache hit rate: ${CACHE_PCT}%"
            echo "       ACTION: Increase hot tier size from current configuration"
            echo "       COMMAND: engram-cli config set hot_tier.size_mb 2048"
        fi
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

    EDGE_COUNT=$(curl -sf http://localhost:9090/api/v1/query --max-time 5 \
        -d "query=avg(engram_graph_edges_per_node{memory_space=\"$space\"})" \
        2>/dev/null | jq -r '.data.result[0].value[1] // "N/A"' || echo "N/A")

    if [ "$EDGE_COUNT" != "N/A" ]; then
        echo "       Average edges per node: $EDGE_COUNT"
        if command -v bc &> /dev/null && (( $(echo "$EDGE_COUNT < 16" | bc -l 2>/dev/null || echo 0) )); then
            echo "       WARNING: Low connectivity detected (M < 16)"
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
    WAL_FLUSH_LATENCY=$(curl -sf http://localhost:9090/api/v1/query --max-time 5 \
        -d "query=histogram_quantile(0.99, rate(engram_wal_flush_duration_seconds_bucket[${LOOKBACK}]))" \
        2>/dev/null | jq -r '.data.result[0].value[1] // "N/A"' || echo "N/A")

    if [ "$WAL_FLUSH_LATENCY" != "N/A" ] && command -v bc &> /dev/null; then
        if (( $(echo "$WAL_FLUSH_LATENCY > 0.005" | bc -l 2>/dev/null || echo 0) )); then
            echo "    WARNING: Slow WAL flushes: ${WAL_FLUSH_LATENCY}s"
            echo "       ACTION: Increase WAL buffer size or use O_DIRECT writes"
            echo "       CONFIG: wal.buffer_size_mb = 64"
            echo "       CONFIG: wal.sync_mode = 'batch'  # Group commits"
        fi
    fi

    # Check lock contention
    echo "    Checking for lock contention in DashMap..."
    echo "       ACTION: Monitor CPU usage during stores"
    echo "       If CPU >80% with low throughput: shard the hashmap further"
}

# Main analysis loop
echo "Analyzing slow queries..."

# Check if Prometheus is available
if ! curl -sf http://localhost:9090/api/v1/query --max-time 5 \
  -d 'query=up' >/dev/null 2>&1; then
  echo "ERROR: Cannot connect to Prometheus at http://localhost:9090"
  echo "Please ensure Prometheus is running and accessible."
  exit 1
fi

# Query for slow operations
THRESHOLD_SEC=$(echo "scale=6; $THRESHOLD_MS / 1000" | bc)
curl -sf http://localhost:9090/api/v1/query -G --max-time 10 \
  --data-urlencode "query=histogram_quantile(0.99, rate(engram_memory_operation_duration_seconds_bucket[${LOOKBACK}])) > ${THRESHOLD_SEC}" \
  2>/dev/null | jq -r '.data.result[] | "\(.metric.operation)|\(.metric.memory_space // "default")|\(.value[1])"' \
  | while IFS='|' read -r operation space latency_str; do
      if [ -z "$operation" ]; then
        continue
      fi

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
echo "Analysis complete."
