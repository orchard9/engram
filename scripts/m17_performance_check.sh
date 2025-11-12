#!/bin/bash
# 60-second standardized load test for Milestone 17 regression detection
#
# Usage: ./scripts/m17_performance_check.sh <task_id> <phase> [--competitive]
#   task_id:       Task number (e.g., 001, 002)
#   phase:         "before" or "after"
#   --competitive: Optional flag to run competitive scenario instead of default
#
# Examples:
#   ./scripts/m17_performance_check.sh 001 before
#   ./scripts/m17_performance_check.sh 001 after --competitive

set -e

TASK_ID=${1:-"unknown"}
PHASE=${2:-"before"}
COMPETITIVE=false
OUTPUT_DIR="tmp/m17_performance"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse optional --competitive flag
if [[ "${3}" == "--competitive" ]] || [[ "${2}" == "--competitive" && -z "${3}" ]]; then
    COMPETITIVE=true
    # If --competitive is 2nd arg, shift phase detection
    if [[ "${2}" == "--competitive" ]]; then
        PHASE="before"  # Default if only flag provided
    fi
fi

# Determine file prefix based on competitive mode
if [[ "$COMPETITIVE" == true ]]; then
    FILE_PREFIX="${OUTPUT_DIR}/competitive_${TASK_ID}_${PHASE}"
    SCENARIO="scenarios/competitive/hybrid_production_100k.toml"
else
    FILE_PREFIX="${OUTPUT_DIR}/${TASK_ID}_${PHASE}"
    SCENARIO="scenarios/m17_baseline.toml"
fi

RESULT_FILE="${FILE_PREFIX}_${TIMESTAMP}.json"
SYS_FILE="${FILE_PREFIX}_${TIMESTAMP}_sys.txt"
DIAG_FILE="${FILE_PREFIX}_${TIMESTAMP}_diag.txt"

if [[ "$PHASE" != "before" && "$PHASE" != "after" ]]; then
    echo "ERROR: Phase must be 'before' or 'after'"
    echo "Usage: $0 <task_id> <phase> [--competitive]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ "$COMPETITIVE" == true ]]; then
    echo "=== M17 Competitive Performance Check: Task ${TASK_ID} (${PHASE}) ==="
else
    echo "=== M17 Performance Check: Task ${TASK_ID} (${PHASE}) ==="
fi
echo "Scenario: $SCENARIO"
echo "Results will be saved to: $RESULT_FILE"
echo

# 1. Build in release mode
echo "Building Engram in release mode..."
cargo build --release --quiet

# 2. Start Engram server in background
echo "Starting Engram server on port 7432..."
./target/release/engram start --port 7432 &
SERVER_PID=$!

# Wait for server to be ready (with health check)
echo "Waiting for server startup..."
RETRIES=10
RETRY_DELAY=1
for i in $(seq 1 $RETRIES); do
    sleep $RETRY_DELAY
    if curl -sf http://localhost:7432/health > /dev/null 2>&1; then
        echo "✓ Server healthy after ${i} seconds"
        break
    fi

    # Check if process still exists
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process terminated during startup"
        exit 1
    fi

    if [ "$i" -eq "$RETRIES" ]; then
        echo "ERROR: Server failed to respond to health check after ${RETRIES} seconds"
        echo "Server process is running (PID $SERVER_PID) but not accepting connections"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
done

# 3. Run 60s load test with deterministic seed
if [[ "$COMPETITIVE" == true ]]; then
    SEED=2882400044  # 0xABCD1234 for competitive scenarios
    echo "Running 60-second competitive load test (deterministic seed: 0xABCD1234)..."
else
    SEED=3735928559  # 0xDEADBEEF for default scenarios
    echo "Running 60-second load test (deterministic seed: 0xDEADBEEF)..."
fi

if ./target/release/loadtest run \
    --scenario "$SCENARIO" \
    --duration 60 \
    --seed "$SEED" \
    --endpoint http://localhost:7432 \
    --output "$RESULT_FILE" 2>&1 | tee "${FILE_PREFIX}_${TIMESTAMP}_loadtest.log"; then
    echo "✓ Load test completed successfully"
else
    echo "⚠️  Load test completed with errors (check log for details)"
fi

# 4. Capture system metrics
echo "Capturing system metrics..."
if ps -p $SERVER_PID -o rss,vsz,%cpu,%mem > "$SYS_FILE" 2>/dev/null; then
    echo "✓ System metrics captured"
else
    echo "⚠️  Could not capture system metrics (server may have stopped)"
fi

# 5. Stop server gracefully
echo "Stopping Engram server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# 6. Run diagnostics
echo "Running diagnostics..."
if ./scripts/engram_diagnostics.sh 2>/dev/null | head -20 > "$DIAG_FILE"; then
    echo "✓ Diagnostics captured"
else
    echo "⚠️  Could not run diagnostics"
fi

echo
echo "=== Performance Check Complete ==="
echo "Results:      $RESULT_FILE"
echo "System stats: $SYS_FILE"
echo "Diagnostics:  $DIAG_FILE"
echo

# 7. Quick summary if result file exists
if [[ -f "$RESULT_FILE" ]]; then
    echo "Quick Summary:"
    if command -v jq >/dev/null 2>&1; then
        echo "  P50 latency:  $(jq -r '.latency.p50_ms // "N/A"' "$RESULT_FILE") ms"
        echo "  P95 latency:  $(jq -r '.latency.p95_ms // "N/A"' "$RESULT_FILE") ms"
        echo "  P99 latency:  $(jq -r '.latency.p99_ms // "N/A"' "$RESULT_FILE") ms"
        echo "  Throughput:   $(jq -r '.throughput.ops_per_sec // "N/A"' "$RESULT_FILE") ops/s"
        echo "  Errors:       $(jq -r '.errors.total // 0' "$RESULT_FILE")"
    else
        echo "  (Install jq for summary statistics)"
    fi
fi
