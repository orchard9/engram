#!/bin/bash
# Determinism validation script for competitive scenarios
#
# Validates that scenarios produce reproducible results across multiple runs
# with the same seed. This is critical for fair competitive comparison.
#
# Usage: ./determinism_test.sh <scenario_path>
# Example: ./determinism_test.sh scenarios/competitive/qdrant_ann_1m_768d.toml

set -euo pipefail

SCENARIO="${1:-}"
DURATION=10  # Short test for faster iteration (full validation uses 60s)
NUM_RUNS=3
TMP_DIR="/tmp/engram_determinism_$$"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Cleanup on exit
trap 'rm -rf "$TMP_DIR"' EXIT

if [ -z "$SCENARIO" ]; then
    echo "Usage: $0 <scenario_path>"
    echo "Example: $0 scenarios/competitive/qdrant_ann_1m_768d.toml"
    exit 1
fi

if [ ! -f "$SCENARIO" ]; then
    echo "Error: Scenario file not found: $SCENARIO"
    exit 1
fi

echo "=== Determinism Test: $SCENARIO ==="
echo "Testing with $NUM_RUNS consecutive runs (${DURATION}s each)"
echo ""

# Create temporary directory
mkdir -p "$TMP_DIR"

# Extract seed from TOML (if not specified, determinism test is meaningless)
SEED=$(grep -E "^seed = " "$SCENARIO" 2>/dev/null | awk -F'=' '{print $2}' | tr -d ' ' || echo "")
if [ -z "$SEED" ]; then
    # Try to extract from filename convention (e.g., qdrant_ann_1m_768d.toml -> seed 42)
    BASENAME=$(basename "$SCENARIO" .toml)
    case "$BASENAME" in
        qdrant_*) SEED=42 ;;
        neo4j_*) SEED=43 ;;
        hybrid_*) SEED=44 ;;
        milvus_*) SEED=45 ;;
        *) SEED=42 ;;  # Default fallback
    esac
fi

echo "Using seed: $SEED"
echo ""

# Check if loadtest binary exists
if [ ! -f "target/release/loadtest" ]; then
    echo -e "${YELLOW}Warning: Release binary not found. Building...${NC}"
    cargo build --release --bin loadtest || {
        echo -e "${RED}FAIL: Failed to build loadtest binary${NC}"
        exit 1
    }
fi

# Run scenario multiple times
echo "Running scenario $NUM_RUNS times..."
for i in $(seq 1 $NUM_RUNS); do
    echo "  Run $i/$NUM_RUNS..."

    # Run loadtest with deterministic seed
    cargo run --release --bin loadtest -- run \
        --scenario "$SCENARIO" \
        --duration "$DURATION" \
        --seed "$SEED" \
        --output "$TMP_DIR/run_${i}.json" \
        > "$TMP_DIR/run_${i}.log" 2>&1 || {
            echo -e "${RED}FAIL: Run $i failed to complete${NC}"
            cat "$TMP_DIR/run_${i}.log"
            exit 1
        }
done

echo ""
echo "=== Results Comparison ==="

# Extract key metrics from each run
echo "P99 Latency (ms):"
for i in $(seq 1 $NUM_RUNS); do
    if [ -f "$TMP_DIR/run_${i}.json" ]; then
        P99=$(jq -r '.p99_latency_ms // "N/A"' "$TMP_DIR/run_${i}.json")
        echo "  Run $i: $P99"
    fi
done
echo ""

echo "Total Operations:"
for i in $(seq 1 $NUM_RUNS); do
    if [ -f "$TMP_DIR/run_${i}.json" ]; then
        OPS=$(jq -r '.total_operations // "N/A"' "$TMP_DIR/run_${i}.json")
        echo "  Run $i: $OPS"
    fi
done
echo ""

echo "Throughput (ops/sec):"
for i in $(seq 1 $NUM_RUNS); do
    if [ -f "$TMP_DIR/run_${i}.json" ]; then
        THROUGHPUT=$(jq -r '.throughput_ops_sec // "N/A"' "$TMP_DIR/run_${i}.json")
        echo "  Run $i: $THROUGHPUT"
    fi
done
echo ""

# Statistical validation using Python
echo "=== Statistical Validation ==="
python3 - <<EOF
import json
import sys
from pathlib import Path

tmp_dir = Path("$TMP_DIR")
num_runs = $NUM_RUNS

# Load all run results
runs = []
for i in range(1, num_runs + 1):
    json_path = tmp_dir / f"run_{i}.json"
    if not json_path.exists():
        print(f"ERROR: Missing results for run {i}")
        sys.exit(1)

    with open(json_path) as f:
        runs.append(json.load(f))

all_passed = True

# Check 1: Operation counts must be exactly identical
print("1. Operation Count Determinism:")
op_counts = [r.get('total_operations', 0) for r in runs]
if len(set(op_counts)) == 1:
    print(f"   PASS: All runs have identical operation count: {op_counts[0]}")
else:
    print(f"   FAIL: Operation counts differ: {op_counts}")
    all_passed = False
print()

# Check 2: P99 latency variance within tolerance
print("2. P99 Latency Stability:")
p99_latencies = [r.get('p99_latency_ms', 0.0) for r in runs]
if not p99_latencies or all(p == 0 for p in p99_latencies):
    print("   SKIP: No P99 data available")
else:
    mean_p99 = sum(p99_latencies) / len(p99_latencies)
    max_deviation = max(abs(p - mean_p99) for p in p99_latencies)

    # Tolerance: 0.5ms absolute OR 1% relative, whichever is larger
    tolerance_abs = 0.5
    tolerance_rel = mean_p99 * 0.01
    tolerance = max(tolerance_abs, tolerance_rel)

    if max_deviation <= tolerance:
        print(f"   PASS: P99 within tolerance (max deviation: {max_deviation:.3f}ms <= {tolerance:.3f}ms)")
        print(f"         Values: {[f'{p:.2f}' for p in p99_latencies]}")
    else:
        print(f"   FAIL: P99 exceeds tolerance (max deviation: {max_deviation:.3f}ms > {tolerance:.3f}ms)")
        print(f"         Values: {[f'{p:.2f}' for p in p99_latencies]}")
        all_passed = False
print()

# Check 3: Throughput variance within tolerance
print("3. Throughput Stability:")
throughputs = [r.get('throughput_ops_sec', 0.0) for r in runs]
if not throughputs or all(t == 0 for t in throughputs):
    print("   SKIP: No throughput data available")
else:
    mean_throughput = sum(throughputs) / len(throughputs)
    max_deviation = max(abs(t - mean_throughput) for t in throughputs)
    tolerance = mean_throughput * 0.02  # 2% tolerance

    if max_deviation <= tolerance:
        print(f"   PASS: Throughput within 2% tolerance (max deviation: {max_deviation:.2f} ops/sec)")
        print(f"         Values: {[f'{t:.2f}' for t in throughputs]}")
    else:
        print(f"   FAIL: Throughput exceeds 2% tolerance (max deviation: {max_deviation:.2f} ops/sec > {tolerance:.2f})")
        print(f"         Values: {[f'{t:.2f}' for t in throughputs]}")
        all_passed = False
print()

# Check 4: Operation distribution consistency
print("4. Operation Distribution Consistency:")
if 'per_operation_stats' in runs[0]:
    # Get operation types from first run
    op_types = list(runs[0]['per_operation_stats'].keys())

    consistent = True
    for op_type in op_types:
        counts = [
            r.get('per_operation_stats', {}).get(op_type, {}).get('count', 0)
            for r in runs
        ]

        if len(set(counts)) == 1:
            print(f"   PASS: {op_type} count consistent: {counts[0]}")
        else:
            print(f"   FAIL: {op_type} counts differ: {counts}")
            consistent = False
            all_passed = False

    if consistent:
        print()
else:
    print("   SKIP: No per-operation stats available")
    print()

# Final verdict
print("=" * 50)
if all_passed:
    print("RESULT: Determinism test PASSED")
    print()
    print("All metrics are within acceptable variance:")
    print("  - Operation counts: Exactly identical")
    print("  - P99 latency: <0.5ms or <1% variance")
    print("  - Throughput: <2% variance")
    print("  - Operation distribution: Exactly identical")
    sys.exit(0)
else:
    print("RESULT: Determinism test FAILED")
    print()
    print("One or more metrics exceeded variance tolerance.")
    print("This scenario is not suitable for competitive comparison.")
    print()
    print("Possible causes:")
    print("  - Non-deterministic RNG (check seed usage)")
    print("  - System load variation (run during idle time)")
    print("  - Thermal throttling (check CPU temperature)")
    sys.exit(1)
EOF

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Determinism test PASSED${NC}"
else
    echo -e "${RED}Determinism test FAILED${NC}"
fi

exit $EXIT_CODE
