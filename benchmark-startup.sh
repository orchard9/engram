#!/bin/bash

# Engram Startup Benchmark Script
# Measures time from git clone to fully operational cluster
# Target: <60 seconds on modern hardware

set -e

# Colors for cognitive-friendly output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Phase timing variables
PHASE_TIMES=()
PHASE_NAMES=()
START_TIME=$(date +%s.%N)

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to record phase timing
record_phase() {
    local phase_name="$1"
    local phase_time=$(date +%s.%N)
    PHASE_TIMES+=("$phase_time")
    PHASE_NAMES+=("$phase_name")
}

# Function to print progress
print_progress() {
    local icon="$1"
    local message="$2"
    echo -e "${icon} ${message}"
}

# Function to calculate elapsed time
elapsed_time() {
    local start=$1
    local end=$2
    echo "scale=2; $end - $start" | bc
}

# Function to print time breakdown
print_breakdown() {
    print_header "⏱️  Performance Breakdown"
    
    local total_time=$(elapsed_time $START_TIME $(date +%s.%N))
    
    echo -e "\n${BOLD}Phase Timings:${NC}"
    echo -e "─────────────────────────────────────────────────────────────"
    
    local prev_time=$START_TIME
    for i in "${!PHASE_NAMES[@]}"; do
        local phase_duration=$(elapsed_time $prev_time ${PHASE_TIMES[$i]})
        local percentage=$(echo "scale=1; $phase_duration * 100 / $total_time" | bc)
        
        # Create visual bar
        local bar_length=$(echo "scale=0; $percentage / 2" | bc)
        local bar=""
        for ((j=0; j<bar_length; j++)); do
            bar="${bar}█"
        done
        
        printf "%-25s %6.2fs %5.1f%% %s\n" \
            "${PHASE_NAMES[$i]}" \
            "$phase_duration" \
            "$percentage" \
            "$bar"
        
        prev_time=${PHASE_TIMES[$i]}
    done
    
    echo -e "─────────────────────────────────────────────────────────────"
    echo -e "${BOLD}Total Time: ${total_time}s${NC}"
    
    # Performance assessment
    echo ""
    if (( $(echo "$total_time < 60" | bc -l) )); then
        echo -e "${GREEN}✅ PASS: Startup completed in under 60 seconds!${NC}"
    else
        echo -e "${RED}❌ FAIL: Startup exceeded 60 second target${NC}"
    fi
}

# Function to print optimization recommendations
print_recommendations() {
    print_header "💡 Optimization Recommendations"
    
    local build_time=$(elapsed_time $BUILD_START $BUILD_END)
    
    if (( $(echo "$build_time > 30" | bc -l) )); then
        echo -e "${YELLOW}⚠️  Build phase is slow (${build_time}s)${NC}"
        echo "   Recommendations:"
        echo "   • Enable sccache for build caching"
        echo "   • Use 'cargo build --release' with lto='thin'"
        echo "   • Consider pre-built binaries for CI"
    fi
    
    local clone_time=$(elapsed_time $CLONE_START $CLONE_END)
    if (( $(echo "$clone_time > 10" | bc -l) )); then
        echo -e "${YELLOW}⚠️  Clone phase is slow (${clone_time}s)${NC}"
        echo "   Recommendations:"
        echo "   • Use shallow clone with --depth=1"
        echo "   • Consider Git LFS for large files"
        echo "   • Use CDN-backed mirrors"
    fi
}

# Check for required tools
check_requirements() {
    print_header "🔍 Checking Requirements"
    
    local missing_tools=()
    
    if ! command -v git &> /dev/null; then
        missing_tools+=("git")
    fi
    
    if ! command -v cargo &> /dev/null; then
        missing_tools+=("cargo")
    fi
    
    if ! command -v bc &> /dev/null; then
        missing_tools+=("bc")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        echo -e "${RED}❌ Missing required tools: ${missing_tools[*]}${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All requirements met${NC}"
}

# Main benchmark execution
main() {
    print_header "🚀 Engram Startup Benchmark"
    echo "Target: Git clone to operational in <60 seconds"
    echo "Platform: $(uname -s) $(uname -m)"
    echo "Date: $(date)"
    
    check_requirements
    
    # Create temporary directory for benchmark
    BENCH_DIR=$(mktemp -d -t engram-bench-XXXXXX)
    cd "$BENCH_DIR"
    
    echo -e "\nWorking directory: $BENCH_DIR"
    
    # Phase 1: Clone
    print_header "📦 Phase 1: Clone Repository"
    CLONE_START=$(date +%s.%N)
    print_progress "⏳" "Cloning repository..."
    
    if [ -n "$ENGRAM_REPO_URL" ]; then
        REPO_URL="$ENGRAM_REPO_URL"
    else
        REPO_URL="https://github.com/orchard9/engram.git"
    fi
    
    git clone --depth=1 "$REPO_URL" engram 2>&1 | \
        while IFS= read -r line; do
            if [[ $line == *"Receiving objects"* ]]; then
                echo -e "\r${CYAN}📥 Receiving objects...${NC}"
            elif [[ $line == *"Resolving deltas"* ]]; then
                echo -e "\r${CYAN}🔗 Resolving deltas...${NC}"
            fi
        done
    
    CLONE_END=$(date +%s.%N)
    record_phase "Clone Repository"
    print_progress "✅" "Clone completed in $(elapsed_time $CLONE_START $CLONE_END)s"
    
    cd engram
    
    # Phase 2: Build
    print_header "🔨 Phase 2: Build"
    BUILD_START=$(date +%s.%N)
    print_progress "⏳" "Building Engram..."
    
    # Use release build for benchmarking
    RUSTFLAGS="-C target-cpu=native" cargo build --release 2>&1 | \
        while IFS= read -r line; do
            if [[ $line == *"Compiling"* ]]; then
                pkg=$(echo $line | grep -oP 'Compiling \K[^ ]+' || true)
                if [ -n "$pkg" ]; then
                    echo -ne "\r${CYAN}📦 Compiling $pkg...${NC}        "
                fi
            elif [[ $line == *"Finished"* ]]; then
                echo -e "\r${GREEN}✅ Build finished!${NC}                    "
            fi
        done
    
    BUILD_END=$(date +%s.%N)
    record_phase "Build"
    print_progress "✅" "Build completed in $(elapsed_time $BUILD_START $BUILD_END)s"
    
    # Phase 3: Start Server
    print_header "🌟 Phase 3: Start Server"
    START_SERVER=$(date +%s.%N)
    print_progress "⏳" "Starting Engram server..."
    
    # Start server in background
    ./target/release/engram start --port 7432 --grpc-port 50051 > server.log 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to be ready
    MAX_WAIT=30
    WAITED=0
    while ! curl -s http://localhost:7432/health > /dev/null 2>&1; do
        sleep 0.5
        WAITED=$((WAITED + 1))
        echo -ne "\r${CYAN}⏳ Waiting for server... ${WAITED}s${NC}"
        
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo -e "\n${RED}❌ Server failed to start within ${MAX_WAIT}s${NC}"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
    done
    
    END_START=$(date +%s.%N)
    record_phase "Start Server"
    print_progress "✅" "Server started in $(elapsed_time $START_SERVER $END_START)s"
    
    # Phase 4: First Query
    print_header "🔍 Phase 4: First Query"
    QUERY_START=$(date +%s.%N)
    print_progress "⏳" "Executing first query..."
    
    # Execute a simple health check query
    RESPONSE=$(curl -s -X GET http://localhost:7432/health)
    
    if [[ $RESPONSE == *"healthy"* ]]; then
        QUERY_END=$(date +%s.%N)
        record_phase "First Query"
        print_progress "✅" "First query completed in $(elapsed_time $QUERY_START $QUERY_END)s"
    else
        echo -e "${RED}❌ First query failed${NC}"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    
    # Stop server
    kill $SERVER_PID 2>/dev/null || true
    
    # Print final results
    print_breakdown
    print_recommendations
    
    # Cleanup
    cd /
    rm -rf "$BENCH_DIR"
    
    # Exit with appropriate code
    TOTAL_TIME=$(elapsed_time $START_TIME $(date +%s.%N))
    if (( $(echo "$TOTAL_TIME < 60" | bc -l) )); then
        exit 0
    else
        exit 1
    fi
}

# Handle script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)
            ENGRAM_REPO_URL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--repo URL]"
            echo "  --repo URL    Use alternative repository URL"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run the benchmark
main