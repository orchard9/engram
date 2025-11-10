#!/bin/bash
# Correctness validation script for competitive scenarios
#
# Validates that operation distribution matches configured weights
# and that operation generation follows specified patterns.
#
# Usage: ./correctness_test.sh <scenario_path>
# Example: ./correctness_test.sh scenarios/competitive/hybrid_production_100k.toml

set -euo pipefail

SCENARIO="${1:-}"
DURATION=60  # Full 60s test for statistical significance

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ -z "$SCENARIO" ]; then
    echo "Usage: $0 <scenario_path>"
    echo "Example: $0 scenarios/competitive/hybrid_production_100k.toml"
    exit 1
fi

if [ ! -f "$SCENARIO" ]; then
    echo "Error: Scenario file not found: $SCENARIO"
    exit 1
fi

echo "=== Correctness Test: $SCENARIO ==="
echo "Running ${DURATION}s test for statistical significance..."
echo ""

# Check if loadtest binary exists
if [ ! -f "target/release/loadtest" ]; then
    echo -e "${YELLOW}Warning: Release binary not found. Building...${NC}"
    cargo build --release --bin loadtest || {
        echo -e "${RED}FAIL: Failed to build loadtest binary${NC}"
        exit 1
    }
fi

# Run scenario and capture detailed output
TMP_JSON="/tmp/correctness_test_$$.json"
TMP_LOG="/tmp/correctness_test_$$.log"

echo "Executing scenario..."
cargo run --release --bin loadtest -- run \
    --scenario "$SCENARIO" \
    --duration "$DURATION" \
    --output "$TMP_JSON" \
    > "$TMP_LOG" 2>&1 || {
        echo -e "${RED}FAIL: Loadtest execution failed${NC}"
        cat "$TMP_LOG"
        rm -f "$TMP_JSON" "$TMP_LOG"
        exit 1
    }

echo "Loadtest completed successfully"
echo ""

# Validate operation distribution using Python
echo "=== Operation Distribution Validation ==="

python3 - "$SCENARIO" "$TMP_JSON" <<'EOF'
import json
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print("Usage: script <scenario_path> <results_json>")
    sys.exit(1)

scenario_path = sys.argv[1]
results_path = sys.argv[2]

# Parse scenario TOML manually (avoiding toml library dependency)
# Extract operation weights
weights = {}
with open(scenario_path) as f:
    in_operations = False
    for line in f:
        line = line.strip()
        if line == '[operations]':
            in_operations = True
            continue
        if in_operations:
            if line.startswith('['):  # New section
                break
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove comments
                if '#' in value:
                    value = value.split('#')[0].strip()
                try:
                    weights[key] = float(value)
                except ValueError:
                    pass

if not weights:
    print("ERROR: Could not extract operation weights from scenario")
    sys.exit(1)

print("Configured operation weights:")
for k, v in weights.items():
    print(f"  {k}: {v}")
print()

# Load test results
with open(results_path) as f:
    report = json.load(f)

total_ops = report.get('total_operations', 0)
if total_ops == 0:
    print("ERROR: No operations recorded in test results")
    sys.exit(1)

print(f"Total operations executed: {total_ops}")
print()

# Extract actual counts from results
# Map TOML weight names to operation types
weight_to_op_map = {
    'store_weight': 'Store',
    'recall_weight': 'Recall',
    'embedding_search_weight': 'EmbeddingSearch',
    'pattern_completion_weight': 'PatternCompletion',
}

actual_counts = {}
per_op_stats = report.get('per_operation_stats', {})

# Handle both possible formats
for toml_key, op_type in weight_to_op_map.items():
    # Try both the mapped op_type and the toml_key stem
    count = 0
    if op_type in per_op_stats:
        count = per_op_stats[op_type].get('count', 0)
    elif toml_key.replace('_weight', '') in per_op_stats:
        count = per_op_stats[toml_key.replace('_weight', '')].get('count', 0)

    actual_counts[op_type] = count

print("Actual operation counts:")
for op_type, count in actual_counts.items():
    pct = (count / total_ops * 100) if total_ops > 0 else 0
    print(f"  {op_type}: {count} ({pct:.1f}%)")
print()

# Calculate expected ratios
total_weight = sum(weights.values())
if total_weight == 0:
    print("ERROR: Total weight is zero")
    sys.exit(1)

expected_ratios = {}
for toml_key, weight in weights.items():
    op_type = weight_to_op_map.get(toml_key)
    if op_type:
        expected_ratios[op_type] = weight / total_weight

print("=== Expected vs Actual Comparison ===")
print()

all_passed = True
tolerance = 0.03  # 3% tolerance for statistical variance

for op_type in sorted(expected_ratios.keys()):
    expected_ratio = expected_ratios[op_type]
    actual_count = actual_counts.get(op_type, 0)
    actual_ratio = actual_count / total_ops if total_ops > 0 else 0.0

    # Special case: zero weight operations
    if expected_ratio == 0.0:
        if actual_count == 0:
            print(f"{op_type}:")
            print(f"  Expected: 0 operations (0.0%)")
            print(f"  Actual:   {actual_count} operations ({actual_ratio*100:.1f}%)")
            print(f"  Status:   PASS (zero as expected)")
        else:
            print(f"{op_type}:")
            print(f"  Expected: 0 operations (0.0%)")
            print(f"  Actual:   {actual_count} operations ({actual_ratio*100:.1f}%)")
            print(f"  Status:   FAIL (should be zero)")
            all_passed = False
        print()
        continue

    # Non-zero weight operations
    deviation = abs(actual_ratio - expected_ratio)
    deviation_pct = deviation * 100

    print(f"{op_type}:")
    print(f"  Expected: {expected_ratio*100:.1f}% ({int(expected_ratio * total_ops)} operations)")
    print(f"  Actual:   {actual_ratio*100:.1f}% ({actual_count} operations)")
    print(f"  Deviation: {deviation_pct:.2f} percentage points")

    if deviation <= tolerance:
        print(f"  Status:   PASS (within {tolerance*100:.0f}% tolerance)")
    else:
        print(f"  Status:   FAIL (exceeds {tolerance*100:.0f}% tolerance)")
        all_passed = False
    print()

# Additional checks
print("=== Additional Validation Checks ===")
print()

# Check 1: Total operations should be close to rate * duration
expected_ops_min = report.get('expected_operations', total_ops) * 0.9  # 90% threshold
if total_ops < expected_ops_min:
    print(f"WARNING: Total operations ({total_ops}) is significantly below expected")
    print(f"         This may indicate errors or throttling during test")
    print()

# Check 2: Error rate validation
error_rate = report.get('error_rate', 0.0)
max_error_rate = 0.05  # 5% max acceptable error rate
print(f"Error rate: {error_rate*100:.2f}%")
if error_rate > max_error_rate:
    print(f"  FAIL: Error rate exceeds {max_error_rate*100:.0f}% threshold")
    all_passed = False
else:
    print(f"  PASS: Error rate within acceptable bounds")
print()

# Final verdict
print("=" * 60)
if all_passed:
    print("RESULT: Operation distribution validation PASSED")
    print()
    print("All operation types are within ±3% of configured weights.")
    print("The scenario generates a valid workload for competitive comparison.")
    sys.exit(0)
else:
    print("RESULT: Operation distribution validation FAILED")
    print()
    print("One or more operation types deviated from expected distribution.")
    print("This may indicate:")
    print("  - Bug in workload generator")
    print("  - Insufficient test duration (increase to 60s+)")
    print("  - Incorrect TOML configuration")
    print()
    print("Review the deviation details above and fix before proceeding.")
    sys.exit(1)
EOF

EXIT_CODE=$?

# Cleanup
rm -f "$TMP_JSON" "$TMP_LOG"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Correctness test PASSED${NC}"
else
    echo -e "${RED}Correctness test FAILED${NC}"
fi

exit $EXIT_CODE
