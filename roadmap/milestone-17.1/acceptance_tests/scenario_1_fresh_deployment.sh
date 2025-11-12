#!/bin/bash
# Scenario 1: Fresh Deployment (Cold Start)
# Objective: Verify framework works from scratch on clean system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Scenario 1: Fresh Deployment (Cold Start) ==="
echo "Objective: Verify framework works from scratch on clean system"
echo ""

# Step 1: Cleanup preconditions
echo "Step 1: Setup preconditions (clean slate)"
rm -rf tmp/competitive_benchmarks/
echo "  - Cleaned tmp/competitive_benchmarks/"

# Step 2: Build from scratch
echo "Step 2: Build project"
cargo clean 2>&1 | head -5
cargo build --release 2>&1 | tail -5
echo "  - Build complete"

# Step 3: Check that scenarios exist
echo "Step 3: Verify scenario files exist"
SCENARIOS=(
  "scenarios/competitive/qdrant_ann_1m_768d.toml"
  "scenarios/competitive/neo4j_traversal_100k.toml"
  "scenarios/competitive/hybrid_production_100k.toml"
  "scenarios/competitive/milvus_ann_10m_768d.toml"
)

for scenario in "${SCENARIOS[@]}"; do
  if [ ! -f "$scenario" ]; then
    echo "ERROR: Missing scenario file: $scenario"
    exit 1
  fi
done
echo "  - All 4 scenario files found"

# Step 4: Check that loadtest binary exists
echo "Step 4: Verify loadtest binary"
if [ ! -f "target/release/loadtest" ]; then
  echo "ERROR: Loadtest binary not found at target/release/loadtest"
  exit 1
fi
echo "  - Loadtest binary verified"

# Step 5: Run a minimal test (just verify it starts and completes)
echo "Step 5: Execute minimal test (neo4j_traversal_100k for 5 seconds)"
timeout 30s cargo run --release --bin loadtest -- run \
  --scenario scenarios/competitive/neo4j_traversal_100k.toml \
  --duration 5 \
  --seed 43 \
  --output tmp/scenario1_test.json 2>&1 | tail -10

if [ ! -f tmp/scenario1_test.json ]; then
  echo "ERROR: Test output file not created"
  exit 1
fi
echo "  - Test completed successfully"

# Step 6: Validate output format
echo "Step 6: Validate output file structure"
if ! grep -q "p99_latency_ms" tmp/scenario1_test.json; then
  echo "ERROR: Output missing p99_latency_ms field"
  exit 1
fi
echo "  - Output format validated"

# Cleanup test artifacts
rm -f tmp/scenario1_test.json

echo ""
echo "RESULT: PASS - Scenario 1 completed successfully"
echo "  - Fresh deployment can build and execute scenarios"
echo "  - All scenario files present"
echo "  - Loadtest binary functional"
echo "  - Output format correct"
