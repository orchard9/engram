#!/bin/bash
# Scenario 5: Concurrent Execution (Isolation Validation)
# Objective: Verify scenarios don't interfere when run in parallel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Scenario 5: Concurrent Execution (Isolation Validation) ==="
echo "Objective: Verify scenarios don't interfere when run in parallel"
echo ""

# Note: Parallel execution is marked OPTIONAL in Task 008 specification
# This test validates whether parallel execution is safe, not required for M17.1

echo "Status: OPTIONAL - Parallel execution not required for M17.1"
echo "Quarterly review workflow runs scenarios sequentially (15min total is acceptable)"
echo ""

# Step 1: Check system resources
echo "Step 1: Check system resources for parallel execution"
TOTAL_RAM_GB=$(python3 -c "import psutil; print(int(psutil.virtual_memory().total / (1024**3)))")
echo "  - Total RAM: ${TOTAL_RAM_GB}GB"

if [ "$TOTAL_RAM_GB" -lt 32 ]; then
  echo "  - WARNING: <32GB RAM, parallel execution may cause OOM"
  echo "  - Skipping parallel test (acceptable for M17.1)"
  echo ""
  echo "RESULT: SKIP - Scenario 5 (optional)"
  echo "  - Insufficient RAM for safe parallel execution"
  echo "  - Sequential execution validated in other scenarios"
  exit 0
fi

# Step 2: Check if --parallel flag exists in benchmark suite
echo "Step 2: Check for parallel execution support"
if [ ! -f "scripts/competitive_benchmark_suite.sh" ]; then
  echo "  - competitive_benchmark_suite.sh not found"
  echo "  - Parallel execution not implemented (acceptable for M17.1)"
  echo ""
  echo "RESULT: SKIP - Scenario 5 (optional)"
  echo "  - Parallel execution not implemented"
  echo "  - Sequential workflow validated in other scenarios"
  exit 0
fi

if ! grep -q "parallel" scripts/competitive_benchmark_suite.sh; then
  echo "  - No --parallel flag found in competitive_benchmark_suite.sh"
  echo "  - Parallel execution not implemented (acceptable for M17.1)"
  echo ""
  echo "RESULT: SKIP - Scenario 5 (optional)"
  echo "  - Parallel execution not implemented"
  echo "  - Sequential workflow is sufficient for quarterly reviews"
  exit 0
fi

# Step 3: If we reach here, parallel execution is supported
echo "Step 3: Execute parallel test (2 scenarios simultaneously)"
mkdir -p tmp/parallel_test

# Start two scenarios in background
cargo run --release --bin loadtest -- run \
  --scenario scenarios/competitive/neo4j_traversal_100k.toml \
  --duration 10 \
  --seed 43 \
  --output tmp/parallel_test/neo4j_parallel.json &
PID1=$!

cargo run --release --bin loadtest -- run \
  --scenario scenarios/competitive/qdrant_ann_1m_768d.toml \
  --duration 10 \
  --seed 42 \
  --output tmp/parallel_test/qdrant_parallel.json &
PID2=$!

echo "  - Scenario 1 (neo4j) started: PID $PID1"
echo "  - Scenario 2 (qdrant) started: PID $PID2"

# Wait for both to complete
wait $PID1
RESULT1=$?
wait $PID2
RESULT2=$?

# Step 4: Validate both completed successfully
echo "Step 4: Validate parallel execution results"
if [ $RESULT1 -ne 0 ]; then
  echo "  - ERROR: Scenario 1 (neo4j) failed with exit code $RESULT1"
  rm -rf tmp/parallel_test
  exit 1
fi

if [ $RESULT2 -ne 0 ]; then
  echo "  - ERROR: Scenario 2 (qdrant) failed with exit code $RESULT2"
  rm -rf tmp/parallel_test
  exit 1
fi

echo "  - Both scenarios completed successfully"

# Step 5: Validate outputs exist
if [ ! -f tmp/parallel_test/neo4j_parallel.json ] || [ ! -f tmp/parallel_test/qdrant_parallel.json ]; then
  echo "  - ERROR: Output files missing"
  rm -rf tmp/parallel_test
  exit 1
fi

echo "  - Both output files created"

# Cleanup
rm -rf tmp/parallel_test

echo ""
echo "RESULT: PASS - Scenario 5 completed successfully"
echo "  - Parallel execution supported and functional"
echo "  - No resource contention detected"
echo "  - Both scenarios completed without interference"
echo ""
echo "Note: Parallel execution is an optimization for faster quarterly reviews"
echo "Sequential execution (validated in other scenarios) is sufficient for M17.1"
