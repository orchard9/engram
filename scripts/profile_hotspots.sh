#!/usr/bin/env bash
#
# Profiling script for identifying performance hotspots using flamegraphs
#
# This script:
# 1. Installs cargo-flamegraph if not present
# 2. Runs the profiling harness benchmark with flamegraph profiling
# 3. Generates flamegraph SVG in tmp/flamegraph.svg
# 4. Extracts top 10 functions by cumulative time
#
# Usage: ./scripts/profile_hotspots.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Engram Performance Profiling ===${NC}"
echo

# Create tmp directory if it doesn't exist
mkdir -p tmp

# Check if cargo-flamegraph is installed
if ! command -v cargo-flamegraph &> /dev/null; then
    echo -e "${YELLOW}cargo-flamegraph not found. Installing...${NC}"
    cargo install flamegraph
    echo -e "${GREEN}cargo-flamegraph installed successfully${NC}"
    echo
fi

# Check platform-specific requirements
case "$(uname -s)" in
    Darwin*)
        echo -e "${YELLOW}macOS detected. Ensuring DTrace is available...${NC}"
        if ! command -v dtrace &> /dev/null; then
            echo -e "${RED}Error: DTrace not found. Please enable DTrace in System Preferences.${NC}"
            echo "On macOS, you may need to:"
            echo "  1. Run: sudo dtruss ls (and allow in System Preferences)"
            echo "  2. Grant Terminal full disk access in System Preferences"
            exit 1
        fi
        PROFILER_ARGS="--root"
        ;;
    Linux*)
        echo -e "${YELLOW}Linux detected. Using perf for profiling...${NC}"
        if ! command -v perf &> /dev/null; then
            echo -e "${RED}Error: perf not found. Install with: sudo apt-get install linux-tools-generic${NC}"
            exit 1
        fi
        PROFILER_ARGS=""
        ;;
    *)
        echo -e "${RED}Unsupported platform: $(uname -s)${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}Running profiling harness with flamegraph...${NC}"
echo "This will take several minutes. The benchmark runs 30 seconds of measurement."
echo

# Run cargo-flamegraph on the profiling harness
# Use --release for realistic profiling with optimizations enabled
# Use --bench to run in benchmark mode (not test mode)
cd "$(dirname "$0")/.."

if cargo flamegraph ${PROFILER_ARGS} \
    --bench profiling_harness \
    --output tmp/flamegraph.svg \
    -- --bench 2>&1 | tee tmp/profiling_output.log; then
    echo
    echo -e "${GREEN}Flamegraph generated successfully: tmp/flamegraph.svg${NC}"
else
    echo
    echo -e "${RED}Error: Flamegraph generation failed${NC}"
    echo "Check tmp/profiling_output.log for details"
    exit 1
fi

echo
echo -e "${GREEN}=== Profiling Complete ===${NC}"
echo
echo "Results:"
echo "  Flamegraph: tmp/flamegraph.svg"
echo "  Raw output: tmp/profiling_output.log"
echo
echo "To view the flamegraph, open tmp/flamegraph.svg in a web browser:"
echo "  open tmp/flamegraph.svg          # macOS"
echo "  xdg-open tmp/flamegraph.svg      # Linux"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review the flamegraph to identify the widest bars (most time spent)"
echo "2. Look for functions that occupy >5% of total time"
echo "3. Document findings in docs/internal/profiling_results.md"
echo "4. Prioritize optimization of the top 10 functions by cumulative time"
echo
echo "Expected hotspots:"
echo "  - Vector similarity: 15-25% of compute time"
echo "  - Activation spreading: 20-30% of compute time"
echo "  - Memory decay: 10-15% of compute time"
