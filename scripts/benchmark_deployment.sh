#!/usr/bin/env bash
# Production benchmark suite for Engram
#
# This script runs comprehensive benchmarks against a deployed Engram instance
# to validate performance targets and identify regressions. It tests store,
# recall, mixed workloads, and multi-tenant scenarios.
#
# Usage: ./scripts/benchmark_deployment.sh [duration_seconds] [concurrency]
# Example: ./scripts/benchmark_deployment.sh 60 10

set -euo pipefail

ENGRAM_URL="${ENGRAM_URL:-http://localhost:7432}"
DURATION="${1:-60}"
CONCURRENCY="${2:-10}"
OUTPUT_DIR="./benchmark-$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

echo "Benchmarking Engram at $ENGRAM_URL"
echo "Duration: ${DURATION}s, Concurrency: $CONCURRENCY"
echo "Output: $OUTPUT_DIR"
echo ""

# Ensure Engram is healthy
echo "Checking Engram health..."
if curl -sf "$ENGRAM_URL/api/v1/system/health" --max-time 5 > /dev/null 2>&1; then
    echo "  Engram is healthy"
else
    echo "ERROR: Engram not healthy at $ENGRAM_URL/api/v1/system/health"
    exit 1
fi

# Create test data payload
TEST_MEMORY='{"content":"benchmark test memory","confidence":0.9,"embedding":[0.1,0.2,0.3]}'
TEST_QUERY='benchmark'

# Benchmark 1: Store operations
echo ""
echo "Benchmark 1/4: Store operations..."
if command -v ab &> /dev/null; then
    ab -n 10000 -c "$CONCURRENCY" -t "$DURATION" \
       -p <(echo "$TEST_MEMORY") \
       -T "application/json" \
       "$ENGRAM_URL/api/v1/memories/remember" \
       > "$OUTPUT_DIR/bench_store.txt" 2>&1 || true

    if [ -f "$OUTPUT_DIR/bench_store.txt" ]; then
        STORE_RPS=$(grep "Requests per second" "$OUTPUT_DIR/bench_store.txt" | awk '{print $4}' | cut -d'.' -f1 || echo "0")
        STORE_P99=$(grep "99%" "$OUTPUT_DIR/bench_store.txt" | awk '{print $2}' || echo "N/A")
        echo "  Store: ${STORE_RPS} req/s, P99: ${STORE_P99}ms"
    else
        STORE_RPS=0
        STORE_P99="N/A"
        echo "  Store benchmark failed"
    fi
else
    echo "  WARNING: Apache Bench (ab) not available, skipping store benchmark"
    STORE_RPS=0
    STORE_P99="N/A"
fi

# Benchmark 2: Recall operations
echo ""
echo "Benchmark 2/4: Recall operations..."
if command -v ab &> /dev/null; then
    ab -n 10000 -c "$CONCURRENCY" -t "$DURATION" \
       "$ENGRAM_URL/api/v1/memories/recall?query=$TEST_QUERY" \
       > "$OUTPUT_DIR/bench_recall.txt" 2>&1 || true

    if [ -f "$OUTPUT_DIR/bench_recall.txt" ]; then
        RECALL_RPS=$(grep "Requests per second" "$OUTPUT_DIR/bench_recall.txt" | awk '{print $4}' | cut -d'.' -f1 || echo "0")
        RECALL_P99=$(grep "99%" "$OUTPUT_DIR/bench_recall.txt" | awk '{print $2}' || echo "N/A")
        echo "  Recall: ${RECALL_RPS} req/s, P99: ${RECALL_P99}ms"
    else
        RECALL_RPS=0
        RECALL_P99="N/A"
        echo "  Recall benchmark failed"
    fi
else
    echo "  WARNING: Apache Bench (ab) not available, skipping recall benchmark"
    RECALL_RPS=0
    RECALL_P99="N/A"
fi

# Benchmark 3: Mixed workload (70% recall, 30% store)
echo ""
echo "Benchmark 3/4: Mixed workload (70% recall, 30% store)..."
echo "  Simulating realistic workload pattern..."

# Launch background recall workers (70%)
RECALL_WORKERS=$((CONCURRENCY * 7 / 10))
if [ "$RECALL_WORKERS" -lt 1 ]; then
    RECALL_WORKERS=1
fi

# Launch background store workers (30%)
STORE_WORKERS=$((CONCURRENCY * 3 / 10))
if [ "$STORE_WORKERS" -lt 1 ]; then
    STORE_WORKERS=1
fi

echo "  Recall workers: $RECALL_WORKERS, Store workers: $STORE_WORKERS"

MIXED_START=$(date +%s)
MIXED_REQUESTS=0

# Simple mixed workload generator
{
    for i in $(seq 1 "$RECALL_WORKERS"); do
        (
            while [ $(($(date +%s) - MIXED_START)) -lt "$DURATION" ]; do
                curl -sf "$ENGRAM_URL/api/v1/memories/recall?query=$TEST_QUERY" > /dev/null 2>&1 || true
            done
        ) &
    done

    for i in $(seq 1 "$STORE_WORKERS"); do
        (
            while [ $(($(date +%s) - MIXED_START)) -lt "$DURATION" ]; do
                curl -sf -X POST "$ENGRAM_URL/api/v1/memories/remember" \
                    -H "Content-Type: application/json" \
                    -d "$TEST_MEMORY" > /dev/null 2>&1 || true
            done
        ) &
    done

    wait
} 2>/dev/null

MIXED_END=$(date +%s)
MIXED_DURATION=$((MIXED_END - MIXED_START))
echo "  Mixed workload completed in ${MIXED_DURATION}s"

# Benchmark 4: Multi-tenant workload
echo ""
echo "Benchmark 4/4: Multi-tenant workload..."
echo "  Testing multiple memory spaces concurrently..."

SPACES=("user-123" "session-456" "context-789")
for space in "${SPACES[@]}"; do
    (
        if command -v ab &> /dev/null; then
            ab -n 1000 -c 5 -t 30 \
               -H "X-Memory-Space: $space" \
               "$ENGRAM_URL/api/v1/memories/recall?query=$TEST_QUERY" \
               > "$OUTPUT_DIR/bench_tenant_${space}.txt" 2>&1 || true
        fi
    ) &
done
wait

echo "  Multi-tenant benchmark completed"

# Generate comprehensive report
echo ""
echo "Generating benchmark report..."

cat > "$OUTPUT_DIR/benchmark_report.txt" <<EOF
Engram Benchmark Report
Generated: $(date)
Target URL: $ENGRAM_URL
Duration: ${DURATION}s
Concurrency: $CONCURRENCY

========================================
STORE OPERATIONS
========================================
Throughput: $STORE_RPS req/sec
P99 Latency: ${STORE_P99}ms
Target: ≥1000 req/sec, ≤100ms
EOF

if command -v bc &> /dev/null && [ "$STORE_RPS" -gt 0 ]; then
    if [ "$STORE_RPS" -ge 1000 ]; then
        echo "Status: PASS" >> "$OUTPUT_DIR/benchmark_report.txt"
    else
        echo "Status: FAIL (throughput below target)" >> "$OUTPUT_DIR/benchmark_report.txt"
    fi
else
    echo "Status: UNKNOWN (benchmark tool unavailable)" >> "$OUTPUT_DIR/benchmark_report.txt"
fi

cat >> "$OUTPUT_DIR/benchmark_report.txt" <<EOF

========================================
RECALL OPERATIONS
========================================
Throughput: $RECALL_RPS req/sec
P99 Latency: ${RECALL_P99}ms
Target: ≥5000 req/sec, ≤50ms
EOF

if command -v bc &> /dev/null && [ "$RECALL_RPS" -gt 0 ]; then
    if [ "$RECALL_RPS" -ge 5000 ]; then
        echo "Status: PASS" >> "$OUTPUT_DIR/benchmark_report.txt"
    else
        echo "Status: FAIL (throughput below target)" >> "$OUTPUT_DIR/benchmark_report.txt"
    fi
else
    echo "Status: UNKNOWN (benchmark tool unavailable)" >> "$OUTPUT_DIR/benchmark_report.txt"
fi

cat >> "$OUTPUT_DIR/benchmark_report.txt" <<EOF

========================================
OVERALL ASSESSMENT
========================================
EOF

OVERALL_STATUS="PASS"
if [ "$STORE_RPS" -gt 0 ] && [ "$STORE_RPS" -lt 1000 ]; then
    OVERALL_STATUS="WARN: Store throughput below target"
fi
if [ "$RECALL_RPS" -gt 0 ] && [ "$RECALL_RPS" -lt 5000 ]; then
    OVERALL_STATUS="WARN: Recall throughput below target"
fi
if [ "$STORE_RPS" -eq 0 ] || [ "$RECALL_RPS" -eq 0 ]; then
    OVERALL_STATUS="FAIL: Benchmarks could not run"
fi

echo "Overall Status: $OVERALL_STATUS" >> "$OUTPUT_DIR/benchmark_report.txt"

cat >> "$OUTPUT_DIR/benchmark_report.txt" <<EOF

========================================
RECOMMENDATIONS
========================================
EOF

if [ "$STORE_RPS" -gt 0 ] && [ "$STORE_RPS" -lt 1000 ]; then
    cat >> "$OUTPUT_DIR/benchmark_report.txt" <<EOF
Store Throughput Below Target:
  - Check disk I/O and WAL configuration
  - Consider increasing WAL buffer size
  - Review fsync settings for batching
  - Monitor: iostat -x 1

EOF
fi

if [ "$RECALL_RPS" -gt 0 ] && [ "$RECALL_RPS" -lt 5000 ]; then
    cat >> "$OUTPUT_DIR/benchmark_report.txt" <<EOF
Recall Throughput Below Target:
  - Verify index coverage and cache hit rate
  - Check HNSW index configuration (M, ef_search)
  - Increase hot tier size if needed
  - Monitor: ./scripts/analyze_slow_queries.sh

EOF
fi

cat >> "$OUTPUT_DIR/benchmark_report.txt" <<EOF

========================================
NEXT STEPS
========================================
1. Review detailed logs in: $OUTPUT_DIR/
2. Compare against baseline: ./scripts/benchmark_regression.sh
3. Profile performance: ./scripts/profile_performance.sh 120
4. Analyze slow queries: ./scripts/analyze_slow_queries.sh 10 1h

For assistance: See /docs/operations/performance-tuning.md
EOF

# Display report
cat "$OUTPUT_DIR/benchmark_report.txt"

# Save summary metrics
echo "store_rps=$STORE_RPS" > "$OUTPUT_DIR/metrics.env"
echo "recall_rps=$RECALL_RPS" >> "$OUTPUT_DIR/metrics.env"
echo "overall_status=$OVERALL_STATUS" >> "$OUTPUT_DIR/metrics.env"

echo ""
echo "Benchmark complete. Report saved to: $OUTPUT_DIR/benchmark_report.txt"
