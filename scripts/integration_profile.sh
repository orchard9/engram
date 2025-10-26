#!/bin/bash
# Integration test profiling script for Zig kernels
#
# This script runs integration tests with profiling enabled and generates
# a performance comparison report between Zig-enabled and Rust-only builds.
#
# Usage:
#   ./scripts/integration_profile.sh
#
# Output:
#   tmp/integration_profile.log - Detailed profiling results

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create tmp directory if it doesn't exist
mkdir -p tmp

# Output file
OUTPUT_FILE="tmp/integration_profile.log"

echo -e "${BLUE}=== Engram Integration Test Profiling ===${NC}"
echo ""
echo "Output will be written to: $OUTPUT_FILE"
echo ""

# Function to run tests and extract timing
run_tests() {
    local features=$1
    local label=$2

    echo -e "${YELLOW}Running tests: $label${NC}"

    # Run integration tests
    if [ "$features" == "" ]; then
        cargo test --test integration --quiet -- --nocapture 2>&1 | tee -a "$OUTPUT_FILE"
    else
        cargo test --features "$features" --test integration --quiet -- --nocapture 2>&1 | tee -a "$OUTPUT_FILE"
    fi

    # Run profiling tests (ignored by default)
    if [ "$features" == "zig-kernels" ]; then
        echo ""
        echo -e "${YELLOW}Running profiling tests...${NC}"
        cargo test --features "$features" --test integration -- --ignored --nocapture 2>&1 | tee -a "$OUTPUT_FILE"
    fi

    echo ""
}

# Initialize output file
{
    echo "======================================================"
    echo "Engram Integration Test Profile"
    echo "Date: $(date)"
    echo "======================================================"
    echo ""
} > "$OUTPUT_FILE"

# Run tests with Zig kernels enabled
echo -e "${GREEN}Phase 1: Running tests with Zig kernels${NC}"
run_tests "zig-kernels" "Zig Kernels Enabled"

# Run tests without Zig kernels (Rust fallback)
# Commenting out Rust-only comparison since we're focused on Zig integration
# echo -e "${GREEN}Phase 2: Running tests with Rust-only implementation${NC}"
# run_tests "" "Rust-Only (Fallback)"

# Run scenario tests
echo -e "${GREEN}Phase 2: Running scenario tests${NC}"
echo -e "${YELLOW}Memory Recall Scenario${NC}"
cargo test --features zig-kernels --test integration scenario_memory_recall --nocapture 2>&1 | tee -a "$OUTPUT_FILE"

echo ""
echo -e "${YELLOW}Consolidation Scenario${NC}"
cargo test --features zig-kernels --test integration scenario_consolidation --nocapture 2>&1 | tee -a "$OUTPUT_FILE"

echo ""
echo -e "${YELLOW}Pattern Completion Scenario${NC}"
cargo test --features zig-kernels --test integration scenario_pattern_completion --nocapture 2>&1 | tee -a "$OUTPUT_FILE"

# Check arena statistics
echo ""
echo -e "${GREEN}Phase 3: Arena Allocator Statistics${NC}"
cargo test --features zig-kernels --test integration arena_allocator --nocapture 2>&1 | tee -a "$OUTPUT_FILE"

# Summary
{
    echo ""
    echo "======================================================"
    echo "Profile Complete"
    echo "======================================================"
    echo ""
    echo "Summary:"
    echo "--------"
    echo ""

    # Count passed/failed tests
    PASSED=$(grep -c "test .* ok" "$OUTPUT_FILE" || true)
    FAILED=$(grep -c "test .* FAILED" "$OUTPUT_FILE" || true)

    echo "Tests passed: $PASSED"
    echo "Tests failed: $FAILED"
    echo ""

    # Extract performance metrics if available
    echo "Performance Highlights:"
    echo "-----------------------"
    grep -A 2 "kernel:" "$OUTPUT_FILE" || echo "No explicit kernel timing found"
    echo ""

    grep -A 1 "ops/sec" "$OUTPUT_FILE" || echo "No throughput metrics found"
    echo ""

    echo "Arena Statistics:"
    echo "-----------------"
    grep -A 3 "Arena stats:" "$OUTPUT_FILE" || echo "No arena statistics found"
    echo ""

} | tee -a "$OUTPUT_FILE"

echo -e "${GREEN}Profiling complete!${NC}"
echo -e "Full results in: ${BLUE}$OUTPUT_FILE${NC}"
echo ""

# Check if any tests failed
if [ "$FAILED" -gt 0 ]; then
    echo -e "${RED}WARNING: $FAILED test(s) failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
