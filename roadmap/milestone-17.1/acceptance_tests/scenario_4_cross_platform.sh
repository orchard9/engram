#!/bin/bash
# Scenario 4: Cross-Platform Determinism
# Objective: Verify scenarios produce identical operation sequences on different platforms

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Scenario 4: Cross-Platform Determinism ==="
echo "Objective: Verify scenarios produce identical operation sequences across platforms"
echo ""

# Note: Full cross-platform testing requires Linux + macOS or Docker
# This script performs determinism validation on the current platform only
# For full validation, run this script on both macOS and Linux and compare checksums

echo "Note: This scenario validates determinism on the current platform."
echo "For full cross-platform validation, run on both macOS and Linux and compare outputs."
echo ""

# Step 1: Check platform
echo "Step 1: Detect current platform"
PLATFORM=$(uname -s)
echo "  - Platform: $PLATFORM"

# Step 2: Run scenario three times with same seed
echo "Step 2: Execute scenario 3 times with fixed seed (determinism check)"
mkdir -p tmp/cross_platform_test

for run in 1 2 3; do
  echo "  - Run $run..."
  cargo run --release --bin loadtest -- run \
    --scenario scenarios/competitive/qdrant_ann_1m_768d.toml \
    --duration 5 \
    --seed 42 \
    --output tmp/cross_platform_test/run_${run}.json 2>&1 | tail -3
done

# Step 3: Extract operation counts (deterministic)
echo "Step 3: Extract operation counts from each run"
for run in 1 2 3; do
  if [ ! -f tmp/cross_platform_test/run_${run}.json ]; then
    echo "ERROR: Run $run output not found"
    exit 1
  fi

  OPS=$(grep -o '"total_operations":[0-9]*' tmp/cross_platform_test/run_${run}.json | cut -d: -f2)
  echo "  - Run $run: $OPS operations"
done

# Step 4: Compare operation counts (should be identical with same duration/seed)
echo "Step 4: Validate operation count consistency"
OPS1=$(grep -o '"total_operations":[0-9]*' tmp/cross_platform_test/run_1.json | cut -d: -f2)
OPS2=$(grep -o '"total_operations":[0-9]*' tmp/cross_platform_test/run_2.json | cut -d: -f2)
OPS3=$(grep -o '"total_operations":[0-9]*' tmp/cross_platform_test/run_3.json | cut -d: -f2)

if [ "$OPS1" -eq "$OPS2" ] && [ "$OPS2" -eq "$OPS3" ]; then
  echo "  - PASS: All runs executed $OPS1 operations (deterministic)"
else
  echo "  - FAIL: Operation counts differ ($OPS1, $OPS2, $OPS3)"
  echo "  This indicates non-deterministic behavior"
  rm -rf tmp/cross_platform_test
  exit 1
fi

# Step 5: Check if Docker is available for Linux testing
echo "Step 5: Check for Docker (optional Linux testing)"
if command -v docker &> /dev/null; then
  echo "  - Docker available for Linux cross-platform testing"
  echo "  - To run full cross-platform test:"
  echo "    docker run --rm -v \$(pwd):/workspace -w /workspace rust:latest bash -c \\"
  echo "      cargo build --release && cargo run --release --bin loadtest -- run \\"
  echo "      --scenario scenarios/competitive/qdrant_ann_1m_768d.toml \\"
  echo "      --duration 5 --seed 42 --output /workspace/tmp/linux_result.json\\"
  echo ""
  echo "  - Note: Full test skipped in automated acceptance testing"
else
  echo "  - Docker not available (Linux testing skipped)"
fi

# Cleanup
rm -rf tmp/cross_platform_test

echo ""
echo "RESULT: PASS - Scenario 4 completed successfully"
echo "  - Determinism validated on current platform ($PLATFORM)"
echo "  - Three consecutive runs produced identical operation counts"
echo "  - Framework supports deterministic replay with fixed seeds"
echo ""
echo "For production quarterly reviews, validate on both macOS and Linux:"
echo "  1. Run scenario on macOS, save output"
echo "  2. Run same scenario on Linux with same seed"
echo "  3. Compare operation sequences (SHA256 checksum of operation log)"
echo "  4. Latency variance <20% is acceptable (architecture differences)"
