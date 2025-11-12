#!/bin/bash
# Compare before/after performance for Milestone 17 regression detection
#
# Usage: ./scripts/compare_m17_performance.sh <task_id>
#   task_id: Task number (e.g., 001, 002)
#
# Exit codes:
#   0 - No regressions detected
#   1 - Regressions detected (>5% increase in latency or >5% decrease in throughput)
#   2 - Error (missing files or invalid data)

set -e

TASK_ID=$1
PERF_DIR="tmp/m17_performance"

if [[ -z "$TASK_ID" ]]; then
    echo "Usage: $0 <task_id>"
    echo "Example: $0 001"
    exit 2
fi

# Find most recent before/after files
BEFORE_FILE=$(ls -t "${PERF_DIR}/${TASK_ID}_before_"*.json 2>/dev/null | head -1)
AFTER_FILE=$(ls -t "${PERF_DIR}/${TASK_ID}_after_"*.json 2>/dev/null | head -1)

if [[ ! -f "$BEFORE_FILE" ]]; then
    echo "ERROR: No 'before' results found for task $TASK_ID"
    echo "Run: ./scripts/m17_performance_check.sh $TASK_ID before"
    exit 2
fi

if [[ ! -f "$AFTER_FILE" ]]; then
    echo "ERROR: No 'after' results found for task $TASK_ID"
    echo "Run: ./scripts/m17_performance_check.sh $TASK_ID after"
    exit 2
fi

# Check for jq
if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is required for performance comparison"
    echo "Install with: brew install jq (macOS) or apt-get install jq (Linux)"
    exit 2
fi

echo "=== Performance Comparison: Task $TASK_ID ==="
echo "Before: $(basename "$BEFORE_FILE")"
echo "After:  $(basename "$AFTER_FILE")"
echo

# Extract metrics using jq (with fallback defaults)
BEFORE_P50=$(jq -r '.p50_latency_ms // 0' "$BEFORE_FILE")
AFTER_P50=$(jq -r '.p50_latency_ms // 0' "$AFTER_FILE")

BEFORE_P95=$(jq -r '.p95_latency_ms // 0' "$BEFORE_FILE")
AFTER_P95=$(jq -r '.p95_latency_ms // 0' "$AFTER_FILE")

BEFORE_P99=$(jq -r '.p99_latency_ms // 0' "$BEFORE_FILE")
AFTER_P99=$(jq -r '.p99_latency_ms // 0' "$AFTER_FILE")

BEFORE_THROUGHPUT=$(jq -r '.overall_throughput // 0' "$BEFORE_FILE")
AFTER_THROUGHPUT=$(jq -r '.overall_throughput // 0' "$AFTER_FILE")

BEFORE_ERRORS=$(jq -r '.total_errors // 0' "$BEFORE_FILE")
AFTER_ERRORS=$(jq -r '.total_errors // 0' "$AFTER_FILE")

BEFORE_ERROR_RATE=$(jq -r '.overall_error_rate // 1.0' "$BEFORE_FILE")
AFTER_ERROR_RATE=$(jq -r '.overall_error_rate // 1.0' "$AFTER_FILE")

# Helper function to calculate percentage change with divide-by-zero protection
calc_pct_change() {
    local before=$1
    local after=$2

    # Check if before is zero or very small
    if awk "BEGIN {exit !($before < 0.001 && $before > -0.001)}"; then
        echo "N/A"
    else
        awk "BEGIN {printf \"%.2f\", (($after - $before) / $before) * 100}"
    fi
}

# Calculate percentage changes with divide-by-zero protection
p50_change=$(calc_pct_change "$BEFORE_P50" "$AFTER_P50")
p95_change=$(calc_pct_change "$BEFORE_P95" "$AFTER_P95")
p99_change=$(calc_pct_change "$BEFORE_P99" "$AFTER_P99")
throughput_change=$(calc_pct_change "$BEFORE_THROUGHPUT" "$AFTER_THROUGHPUT")

# Display results table
printf "%-20s %10s %10s %12s\n" "Metric" "Before" "After" "Change"
printf "%-20s %10s %10s %12s\n" "--------------------" "----------" "----------" "------------"
printf "%-20s %10.3f %10.3f %12s\n" "P50 latency (ms)" "$BEFORE_P50" "$AFTER_P50" "${p50_change}%"
printf "%-20s %10.3f %10.3f %12s\n" "P95 latency (ms)" "$BEFORE_P95" "$AFTER_P95" "${p95_change}%"
printf "%-20s %10.3f %10.3f %12s\n" "P99 latency (ms)" "$BEFORE_P99" "$AFTER_P99" "${p99_change}%"
printf "%-20s %10.1f %10.1f %12s\n" "Throughput (ops/s)" "$BEFORE_THROUGHPUT" "$AFTER_THROUGHPUT" "${throughput_change}%"
printf "%-20s %10d %10d %+11d\n" "Errors" "$BEFORE_ERRORS" "$AFTER_ERRORS" "$((AFTER_ERRORS - BEFORE_ERRORS))"

# Calculate error rate display values
BEFORE_ERR_DISPLAY=$(awk "BEGIN {printf \"%.1f\", $BEFORE_ERROR_RATE * 100}")
AFTER_ERR_DISPLAY=$(awk "BEGIN {printf \"%.1f\", $AFTER_ERROR_RATE * 100}")
ERR_DELTA_DISPLAY=$(awk "BEGIN {printf \"%.1f\", ($AFTER_ERROR_RATE - $BEFORE_ERROR_RATE) * 100}")

printf "%-20s %9s%% %9s%% %10spp\n" "Error rate" "$BEFORE_ERR_DISPLAY" "$AFTER_ERR_DISPLAY" "$ERR_DELTA_DISPLAY"
echo

# Check for regressions (>5% threshold)
REGRESSION=0

# Function to compare floats (returns 0 if a > b)
compare_float() {
    awk -v a="$1" -v b="$2" 'BEGIN { if (a > b) exit 0; else exit 1 }'
}

# Check if either baseline has high error rate (>10% makes comparison invalid)
BEFORE_ERROR_PCT=$(awk "BEGIN {printf \"%.1f\", $BEFORE_ERROR_RATE * 100}")
AFTER_ERROR_PCT=$(awk "BEGIN {printf \"%.1f\", $AFTER_ERROR_RATE * 100}")

if awk "BEGIN {exit !($BEFORE_ERROR_RATE > 0.1 || $AFTER_ERROR_RATE > 0.1)}"; then
    echo "⚠️  WARNING: High error rate detected (before: ${BEFORE_ERROR_PCT}%, after: ${AFTER_ERROR_PCT}%)"
    echo "          Comparison may not be valid - fix API errors first"
    echo
fi

# Only check regressions if changes are numeric (not "N/A")
if [[ "$p99_change" != "N/A" ]] && compare_float "$p99_change" "5.0"; then
    echo "⚠️  REGRESSION: P99 latency increased by ${p99_change}% (threshold: +5%)"
    REGRESSION=1
fi

if [[ "$throughput_change" != "N/A" ]] && compare_float "-5.0" "$throughput_change"; then
    echo "⚠️  REGRESSION: Throughput decreased by ${throughput_change}% (threshold: -5%)"
    REGRESSION=1
fi

# Check for error rate increase (must be <5% in both tests for valid comparison)
if awk "BEGIN {exit !($BEFORE_ERROR_RATE < 0.05 && $AFTER_ERROR_RATE < 0.05)}"; then
    # Both error rates acceptable - valid comparison
    :
elif awk "BEGIN {exit !($AFTER_ERROR_RATE > $BEFORE_ERROR_RATE + 0.05)}"; then
    ERROR_INCREASE=$(awk "BEGIN {printf \"%.1f\", ($AFTER_ERROR_RATE - $BEFORE_ERROR_RATE) * 100}")
    echo "⚠️  REGRESSION: Error rate increased by ${ERROR_INCREASE}pp"
    REGRESSION=1
fi

if [[ $REGRESSION -eq 0 ]]; then
    echo "✓ No significant regressions detected (within 5% threshold)"
    echo
    echo "Summary for PERFORMANCE_LOG.md:"
    echo "- Before: P50=${BEFORE_P50}ms, P95=${BEFORE_P95}ms, P99=${BEFORE_P99}ms, ${BEFORE_THROUGHPUT} ops/s"
    echo "- After:  P50=${AFTER_P50}ms, P95=${AFTER_P95}ms, P99=${AFTER_P99}ms, ${AFTER_THROUGHPUT} ops/s"
    echo "- Change: ${p99_change}% P99 latency, ${throughput_change}% throughput"
    echo "- Status: ✓ Within 5% target"
else
    echo
    echo "Action required: Investigate and fix performance regressions before completing task"
    echo "Suggested steps:"
    echo "  1. Profile with: cargo flamegraph --bin engram"
    echo "  2. Check diagnostics: cat ${PERF_DIR}/${TASK_ID}_after_*_diag.txt"
    echo "  3. Review system stats: cat ${PERF_DIR}/${TASK_ID}_after_*_sys.txt"
fi

exit $REGRESSION
