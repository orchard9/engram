#!/usr/bin/env bash
# Performance regression benchmark script for CI/CD
#
# This script runs the regression benchmark suite and fails if performance
# regressions >5% are detected compared to established baselines.
#
# Exit codes:
#   0 - All benchmarks passed, no regressions detected
#   1 - Performance regressions detected or benchmark failed
#
# Usage:
#   ./scripts/benchmark_regression.sh

set -euo pipefail

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "Performance Regression Benchmark Suite"
echo "================================================"
echo ""

# Check if baselines exist
BASELINES_FILE="engram-core/benches/regression/baselines.json"
if [ ! -f "$BASELINES_FILE" ]; then
    echo -e "${YELLOW}Warning: No baselines found at $BASELINES_FILE${NC}"
    echo "Run ./scripts/update_baselines.sh to establish baselines first"
    echo ""
fi

# Display platform information
echo "Platform: $(uname -s) $(uname -m)"
echo "Date: $(date)"
echo ""

# Build with release optimizations
echo "Building Engram with release optimizations..."
cargo build --release --package engram-core
echo ""

# Run regression benchmarks
echo "Running regression benchmarks..."
echo "This will take several minutes..."
echo ""

# Run the benchmark - it will exit with code 1 if regressions detected
if cargo bench --package engram-core --bench regression -- --noplot; then
    echo ""
    echo "================================================"
    echo -e "${GREEN}✓ Performance regression checks PASSED${NC}"
    echo "================================================"
    exit 0
else
    BENCH_EXIT_CODE=$?
    echo ""
    echo "================================================"
    echo -e "${RED}✗ Performance regression checks FAILED${NC}"
    echo "================================================"
    echo ""
    echo "Detected performance regressions >5% from baseline."
    echo ""
    echo "If these regressions are intentional due to API changes,"
    echo "update baselines with:"
    echo "  ./scripts/update_baselines.sh"
    echo ""
    echo "If these are unintentional regressions, investigate with:"
    echo "  cargo bench --package engram-core --bench profiling_harness"
    echo "  cargo flamegraph --bench profiling_harness"
    echo ""
    exit "$BENCH_EXIT_CODE"
fi
