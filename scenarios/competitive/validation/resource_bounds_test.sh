#!/bin/bash
# Resource bounds validation script for competitive scenarios
#
# Validates that memory footprint matches theoretical predictions
# based on node count, embedding dimension, and overhead factor.
#
# Usage: ./resource_bounds_test.sh <scenario_path>
# Example: ./resource_bounds_test.sh scenarios/competitive/qdrant_ann_1m_768d.toml

set -euo pipefail

SCENARIO="${1:-}"
DURATION=30  # 30s test for memory stability

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ -z "$SCENARIO" ]; then
    echo "Usage: $0 <scenario_path>"
    echo "Example: $0 scenarios/competitive/qdrant_ann_1m_768d.toml"
    exit 1
fi

if [ ! -f "$SCENARIO" ]; then
    echo "Error: Scenario file not found: $SCENARIO"
    exit 1
fi

echo "=== Resource Bounds Test: $SCENARIO ==="
echo ""

# Extract configuration from TOML
echo "Extracting scenario configuration..."
NUM_NODES=$(grep -E "^num_nodes = " "$SCENARIO" | awk -F'=' '{print $2}' | tr -d ' ' | sed 's/_//g')
EMBEDDING_DIM=$(grep -E "^embedding_dim = " "$SCENARIO" | awk -F'=' '{print $2}' | tr -d ' ')

if [ -z "$NUM_NODES" ] || [ -z "$EMBEDDING_DIM" ]; then
    echo -e "${RED}FAIL: Could not extract num_nodes or embedding_dim from scenario${NC}"
    exit 1
fi

echo "  Nodes: $(printf "%'d" "$NUM_NODES")"
echo "  Embedding dimension: $EMBEDDING_DIM"
echo ""

# Calculate expected memory footprint
# Formula: num_nodes * embedding_dim * 4 bytes/float * 1.3 overhead
EXPECTED_MB=$(python3 -c "print(int($NUM_NODES * $EMBEDDING_DIM * 4 * 1.3 / 1024 / 1024))")

echo "Expected memory footprint: ~${EXPECTED_MB}MB"
echo "  Formula: num_nodes * embedding_dim * 4 bytes * 1.3 overhead"
echo "  Breakdown:"
echo "    Base vectors: $(python3 -c "print(int($NUM_NODES * $EMBEDDING_DIM * 4 / 1024 / 1024))")MB"
echo "    Overhead (30%): $(python3 -c "print(int($NUM_NODES * $EMBEDDING_DIM * 4 * 0.3 / 1024 / 1024))")MB"
echo ""

# Check available system memory
echo "Checking available system memory..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    TOTAL_MEM_MB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024)}')
    # Available memory is harder on macOS, use vm_stat
    FREE_PAGES=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
    INACTIVE_PAGES=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | tr -d '.')
    PAGE_SIZE=4096
    AVAILABLE_MB=$(( (FREE_PAGES + INACTIVE_PAGES) * PAGE_SIZE / 1024 / 1024 ))
else
    # Linux
    TOTAL_MEM_MB=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024)}')
    AVAILABLE_MB=$(grep MemAvailable /proc/meminfo | awk '{print int($2/1024)}')
fi

echo "  Total memory: ${TOTAL_MEM_MB}MB"
echo "  Available memory: ${AVAILABLE_MB}MB"
echo ""

# Warn if insufficient memory
REQUIRED_MB=$((EXPECTED_MB * 2))  # 2x for safety margin
if [ "$AVAILABLE_MB" -lt "$REQUIRED_MB" ]; then
    echo -e "${YELLOW}WARNING: Available memory (${AVAILABLE_MB}MB) is less than 2x expected (${REQUIRED_MB}MB)${NC}"
    echo -e "${YELLOW}This test may fail with OOM or trigger swap, affecting performance.${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Test aborted by user"
        exit 1
    fi
fi

# Check if loadtest binary exists
if [ ! -f "target/release/loadtest" ]; then
    echo -e "${YELLOW}Warning: Release binary not found. Building...${NC}"
    cargo build --release --bin loadtest || {
        echo -e "${RED}FAIL: Failed to build loadtest binary${NC}"
        exit 1
    }
fi

# Run with memory tracking
echo "Running scenario with memory tracking (${DURATION}s)..."
TMP_OUTPUT="/tmp/resource_test_$$.txt"

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: use /usr/bin/time -l
    /usr/bin/time -l cargo run --release --bin loadtest -- run \
        --scenario "$SCENARIO" \
        --duration "$DURATION" \
        --output /tmp/resource_test_$$.json \
        > /tmp/resource_test_$$.log 2> "$TMP_OUTPUT" || {
            echo -e "${RED}FAIL: Loadtest execution failed${NC}"
            cat /tmp/resource_test_$$.log
            cat "$TMP_OUTPUT"
            rm -f "$TMP_OUTPUT" /tmp/resource_test_$$.json /tmp/resource_test_$$.log
            exit 1
        }

    # Extract max RSS (in bytes on macOS)
    MAX_RSS_BYTES=$(grep "maximum resident set size" "$TMP_OUTPUT" | awk '{print $1}')
    if [ -z "$MAX_RSS_BYTES" ]; then
        echo -e "${RED}FAIL: Could not extract RSS from time output${NC}"
        cat "$TMP_OUTPUT"
        rm -f "$TMP_OUTPUT" /tmp/resource_test_$$.json /tmp/resource_test_$$.log
        exit 1
    fi
    MAX_RSS_MB=$((MAX_RSS_BYTES / 1024 / 1024))
else
    # Linux: use /usr/bin/time -v
    /usr/bin/time -v cargo run --release --bin loadtest -- run \
        --scenario "$SCENARIO" \
        --duration "$DURATION" \
        --output /tmp/resource_test_$$.json \
        > /tmp/resource_test_$$.log 2> "$TMP_OUTPUT" || {
            echo -e "${RED}FAIL: Loadtest execution failed${NC}"
            cat /tmp/resource_test_$$.log
            cat "$TMP_OUTPUT"
            rm -f "$TMP_OUTPUT" /tmp/resource_test_$$.json /tmp/resource_test_$$.log
            exit 1
        }

    # Extract max RSS (in KB on Linux)
    MAX_RSS_KB=$(grep "Maximum resident set size" "$TMP_OUTPUT" | awk '{print $6}')
    if [ -z "$MAX_RSS_KB" ]; then
        echo -e "${RED}FAIL: Could not extract RSS from time output${NC}"
        cat "$TMP_OUTPUT"
        rm -f "$TMP_OUTPUT" /tmp/resource_test_$$.json /tmp/resource_test_$$.log
        exit 1
    fi
    MAX_RSS_MB=$((MAX_RSS_KB / 1024))
fi

echo ""
echo "=== Memory Usage Results ==="
echo "Measured max RSS: ${MAX_RSS_MB}MB"
echo "Expected footprint: ${EXPECTED_MB}MB"
echo ""

# Validate memory is within expected bounds (0.5x - 2x tolerance)
LOWER_BOUND=$((EXPECTED_MB / 2))
UPPER_BOUND=$((EXPECTED_MB * 2))

echo "=== Validation ==="
echo "Acceptable range: ${LOWER_BOUND}MB - ${UPPER_BOUND}MB (0.5x - 2x expected)"
echo ""

ALL_PASSED=true

if [ "$MAX_RSS_MB" -lt "$LOWER_BOUND" ]; then
    echo -e "${RED}FAIL: Memory usage too low (${MAX_RSS_MB}MB < ${LOWER_BOUND}MB)${NC}"
    echo "  This may indicate nodes are not being created properly"
    echo "  or the test duration was too short for full initialization"
    ALL_PASSED=false
elif [ "$MAX_RSS_MB" -gt "$UPPER_BOUND" ]; then
    echo -e "${RED}FAIL: Memory usage too high (${MAX_RSS_MB}MB > ${UPPER_BOUND}MB)${NC}"
    echo "  This may indicate:"
    echo "    - Memory leaks"
    echo "    - Inefficient storage (excessive overhead)"
    echo "    - Unexpected data duplication"
    ALL_PASSED=false
else
    RATIO=$(python3 -c "print(f'{$MAX_RSS_MB / $EXPECTED_MB:.2f}x')")
    echo -e "${GREEN}PASS: Memory usage within expected bounds${NC}"
    echo "  Actual: ${MAX_RSS_MB}MB (${RATIO} of expected)"
    echo "  Range: ${LOWER_BOUND}MB - ${UPPER_BOUND}MB"
fi

echo ""

# Check for swap usage (indicates memory pressure)
if [[ "$OSTYPE" == "darwin"* ]]; then
    SWAP_USED=$(sysctl vm.swapusage | awk '{print $7}' | tr -d 'M')
    if (( $(echo "$SWAP_USED > 100" | bc -l) )); then
        echo -e "${YELLOW}WARNING: Swap usage detected (${SWAP_USED}MB)${NC}"
        echo "  Memory pressure may affect performance measurements"
        ALL_PASSED=false
    fi
else
    SWAP_USED=$(free -m | grep Swap | awk '{print $3}')
    if [ "$SWAP_USED" -gt 100 ]; then
        echo -e "${YELLOW}WARNING: Swap usage detected (${SWAP_USED}MB)${NC}"
        echo "  Memory pressure may affect performance measurements"
        ALL_PASSED=false
    fi
fi

# Cleanup
rm -f "$TMP_OUTPUT" /tmp/resource_test_$$.json /tmp/resource_test_$$.log

echo "=" * 50
if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}Resource bounds test PASSED${NC}"
    echo ""
    echo "Memory footprint is within acceptable range for competitive comparison."
    exit 0
else
    echo -e "${RED}Resource bounds test FAILED${NC}"
    echo ""
    echo "Memory usage is outside acceptable bounds or swap was used."
    echo "This scenario may not be suitable for performance comparison."
    exit 1
fi
