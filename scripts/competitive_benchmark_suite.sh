#!/usr/bin/env bash
# Competitive Benchmark Suite Runner
# Production-grade orchestration script for quarterly baseline comparisons
#
# This script runs all competitive scenarios with comprehensive diagnostics,
# robust error handling, and system resource management. Designed for reliable
# quarterly automated execution to track Engram's performance against competitors.

set -euo pipefail
IFS=$'\n\t'

# Configuration
# shellcheck disable=SC2155
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC2155
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly OUTPUT_DIR="$PROJECT_ROOT/tmp/competitive_benchmarks"
readonly SCENARIOS_DIR="$PROJECT_ROOT/scenarios/competitive"
readonly LOADTEST_BIN="$PROJECT_ROOT/target/release/loadtest"
readonly ENGRAM_BIN="$PROJECT_ROOT/target/release/engram"
readonly DIAGNOSTICS_SCRIPT="$SCRIPT_DIR/engram_diagnostics.sh"
readonly ENGRAM_PORT=7432
readonly SCENARIO_DURATION=60
readonly COOLDOWN_PERIOD=30
readonly SERVER_TIMEOUT=120

# Runtime state
TIMESTAMP=""
FAILED_SCENARIOS=()
FAILURE_REASONS=()
ENGRAM_PID=""
MONITOR_PID=""
SUITE_START_TIME=0

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Command-line flags
PREFLIGHT_ONLY=false
DRY_RUN=false
SINGLE_SCENARIO=""

# Parse command-line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --preflight-only)
                PREFLIGHT_ONLY=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --scenario)
                SINGLE_SCENARIO="$2"
                shift 2
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                print_usage
                exit 1
                ;;
        esac
    done
}

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Production-hardened orchestration script for competitive benchmarks.

Options:
    --preflight-only    Run pre-flight checks only, don't execute benchmarks
    --dry-run           Print execution plan without running benchmarks
    --scenario NAME     Run only the specified scenario
    -h, --help          Show this help message

Exit Codes:
    0   All scenarios passed
    1   One or more scenarios failed
    2   Pre-flight checks failed
    130 Interrupted by user (SIGINT/SIGTERM)
EOF
}

# Cleanup trap (always runs on exit)
cleanup() {
    local exit_code=$?

    # Kill monitor process if running
    if [[ -n "$MONITOR_PID" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi

    # Kill Engram server if running
    if [[ -n "$ENGRAM_PID" ]]; then
        log_info "Stopping Engram server (PID: $ENGRAM_PID)..."
        kill -TERM "$ENGRAM_PID" 2>/dev/null || true

        # Wait up to 5 seconds for graceful shutdown
        for i in {1..5}; do
            if ! kill -0 "$ENGRAM_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        # Force kill if still running
        if kill -0 "$ENGRAM_PID" 2>/dev/null; then
            kill -KILL "$ENGRAM_PID" 2>/dev/null || true
        fi
    fi

    # Kill any other orphaned Engram processes
    pkill -9 engram 2>/dev/null || true

    # Flush logs
    sync 2>/dev/null || true

    # Print partial results if interrupted
    if [[ $exit_code -eq 130 ]] && [[ -n "$TIMESTAMP" ]]; then
        log_warning "Suite interrupted. Partial results in $OUTPUT_DIR/$TIMESTAMP/"
    fi

    exit "$exit_code"
}

# Signal handlers
handle_interrupt() {
    log_warning "Received interrupt signal. Cleaning up..."
    exit 130
}

trap cleanup EXIT
trap handle_interrupt INT TERM

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Cross-platform timeout function (macOS doesn't have timeout command)
# Usage: timeout_run <seconds> <command> [args...]
# Returns: command exit code, or 124 if timeout reached
timeout_run() {
    local timeout_duration=$1
    shift

    # Run command in background
    "$@" &
    local cmd_pid=$!

    # Wait for timeout or completion
    local elapsed=0
    while kill -0 "$cmd_pid" 2>/dev/null; do
        if [[ $elapsed -ge $timeout_duration ]]; then
            # Timeout reached, kill the process
            kill -TERM "$cmd_pid" 2>/dev/null || true
            sleep 1
            if kill -0 "$cmd_pid" 2>/dev/null; then
                kill -KILL "$cmd_pid" 2>/dev/null || true
            fi
            wait "$cmd_pid" 2>/dev/null || true
            return 124  # Same exit code as GNU timeout
        fi
        sleep 1
        ((elapsed++))
    done

    # Command completed, get exit code
    wait "$cmd_pid"
    return $?
}

# Atomic file write helper
atomic_write() {
    local content=$1
    local output_file=$2
    local temp_file="${output_file}.tmp.$$"

    printf "%s" "$content" > "$temp_file"
    mv "$temp_file" "$output_file"
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

preflight_checks() {
    log_info "Running pre-flight system checks..."

    local checks_passed=true

    # Binary verification
    if ! check_binary_verification; then
        checks_passed=false
    fi

    # Scenario validation
    if ! check_scenario_validation; then
        checks_passed=false
    fi

    # Output directory
    if ! check_output_directory; then
        checks_passed=false
    fi

    # System resources
    if ! check_system_resources; then
        checks_passed=false
    fi

    # Tooling dependencies
    if ! check_tooling_dependencies; then
        checks_passed=false
    fi

    if [[ "$checks_passed" == "true" ]]; then
        log_success "All pre-flight checks passed"
        return 0
    else
        log_error "Pre-flight checks failed"
        return 2
    fi
}

check_binary_verification() {
    log_info "Checking loadtest binary..."

    if [[ ! -f "$LOADTEST_BIN" ]]; then
        log_error "Loadtest binary not found at $LOADTEST_BIN"
        log_error "Build it with: cargo build --release -p loadtest"
        return 1
    fi

    # Verify it's a release build (check for optimization)
    if file "$LOADTEST_BIN" | grep -q "debug"; then
        log_error "Loadtest binary appears to be a debug build"
        log_error "Rebuild with: cargo build --release -p loadtest"
        return 1
    fi

    # Check binary is executable
    if [[ ! -x "$LOADTEST_BIN" ]]; then
        log_error "Loadtest binary is not executable"
        return 1
    fi

    # Extract version from cargo metadata
    local version
    version=$(cargo metadata --format-version 1 --no-deps 2>/dev/null | \
        jq -r '.packages[] | select(.name == "loadtest") | .version' || echo "unknown")

    log_success "Loadtest binary verified (version: $version)"
    return 0
}

check_scenario_validation() {
    log_info "Validating scenarios..."

    if [[ ! -d "$SCENARIOS_DIR" ]]; then
        log_error "Scenarios directory not found at $SCENARIOS_DIR"
        return 1
    fi

    # Count TOML files
    local scenario_count
    scenario_count=$(find "$SCENARIOS_DIR" -name "*.toml" -type f | wc -l | tr -d ' ')

    if [[ "$scenario_count" -ne 4 ]]; then
        log_error "Expected 4 scenario files, found $scenario_count"
        return 1
    fi

    # Basic TOML syntax validation
    local scenarios
    local scenario_files
    scenario_files=$(find "$SCENARIOS_DIR" -name "*.toml" -type f | sort)

    # Read into array (compatible with bash 3.x)
    scenarios=()
    while IFS= read -r file; do
        scenarios+=("$file")
    done <<< "$scenario_files"

    local names_seen=()
    for scenario_file in "${scenarios[@]}"; do
        # Check if file is readable
        if [[ ! -r "$scenario_file" ]]; then
            log_error "Scenario file not readable: $scenario_file"
            return 1
        fi

        # Basic TOML validation - check for required fields
        if ! grep -q "^name = " "$scenario_file"; then
            log_error "Scenario missing 'name' field: $scenario_file"
            return 1
        fi

        # Extract scenario name for uniqueness check
        local scenario_name
        scenario_name=$(basename "$scenario_file" .toml)

        # Check for duplicate names
        if [[ ${#names_seen[@]} -gt 0 ]]; then
            for seen in "${names_seen[@]}"; do
                if [[ "$seen" == "$scenario_name" ]]; then
                    log_error "Duplicate scenario name: $scenario_name"
                    return 1
                fi
            done
        fi
        names_seen+=("$scenario_name")
    done

    log_success "All 4 scenarios validated"
    return 0
}

check_output_directory() {
    log_info "Checking output directory..."

    # Create if missing
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        mkdir -p "$OUTPUT_DIR"
    fi

    # Verify write permissions
    if [[ ! -w "$OUTPUT_DIR" ]]; then
        log_error "No write permission to $OUTPUT_DIR"
        return 1
    fi

    # Check disk space
    local available_gb
    if [[ "$OSTYPE" == "darwin"* ]]; then
        available_gb=$(df -g "$OUTPUT_DIR" | awk 'NR==2 {print $4}')
    else
        available_gb=$(df -BG "$OUTPUT_DIR" | awk 'NR==2 {print $4}' | tr -d 'G')
    fi

    if [[ "$available_gb" -lt 5 ]]; then
        log_error "Insufficient disk space: ${available_gb}GB (need at least 5GB)"
        return 1
    elif [[ "$available_gb" -lt 10 ]]; then
        log_warning "Low disk space: ${available_gb}GB (recommended: 10GB+)"
    fi

    # Clean up partial results from interrupted previous runs
    find "$OUTPUT_DIR" -name "*.tmp.*" -type f -delete 2>/dev/null || true
    find "$OUTPUT_DIR" -name "*.partial" -type f -delete 2>/dev/null || true

    log_success "Output directory ready ($available_gb GB available)"
    return 0
}

check_system_resources() {
    log_info "Checking system resources..."

    local warnings=0

    # Check RAM
    local total_gb available_mb
    if [[ "$OSTYPE" == "darwin"* ]]; then
        total_gb=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
        available_mb=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//' | awk '{print int($1*4096/1024/1024)}')
    else
        total_gb=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024/1024)}')
        available_mb=$(grep MemAvailable /proc/meminfo | awk '{print int($2/1024)}')
    fi

    if [[ "$total_gb" -lt 8 ]]; then
        log_error "Insufficient total RAM: ${total_gb}GB (need at least 8GB)"
        return 1
    elif [[ "$total_gb" -lt 16 ]]; then
        log_warning "Low total RAM: ${total_gb}GB (recommended: 16GB+)"
        ((warnings++))
    fi

    local available_gb=$((available_mb / 1024))
    if [[ "$available_gb" -lt 4 ]]; then
        log_error "Insufficient available RAM: ${available_gb}GB (need at least 4GB)"
        return 1
    fi

    # Check CPU load
    local load_1min
    load_1min=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ' | cut -d'.' -f1)

    if [[ "$load_1min" -gt 2 ]]; then
        log_warning "High system load: $load_1min (recommended: <2.0)"
        ((warnings++))
    fi

    # Check for other Engram processes
    if pgrep -f engram >/dev/null 2>&1; then
        log_warning "Existing Engram processes detected. Cleaning up..."
        pkill -9 engram 2>/dev/null || true
        sleep 2

        if pgrep -f engram >/dev/null 2>&1; then
            log_error "Failed to kill existing Engram processes"
            return 1
        fi
    fi

    # Check localhost connectivity
    if ! ping -c 1 127.0.0.1 >/dev/null 2>&1; then
        log_error "Localhost not reachable (network issue)"
        return 1
    fi

    # CPU frequency scaling check (Linux only)
    if [[ "$OSTYPE" == "linux-gnu"* ]] && [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
        local governor
        governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
        if [[ "$governor" != "performance" ]]; then
            log_warning "CPU governor is '$governor' (recommended: 'performance')"
            ((warnings++))
        fi
    fi

    # Thermal check (macOS only)
    if [[ "$OSTYPE" == "darwin"* ]] && command -v powermetrics >/dev/null 2>&1; then
        # Note: powermetrics requires sudo, skip if not available
        :
    fi

    if [[ $warnings -gt 0 ]]; then
        log_warning "System checks completed with $warnings warning(s)"
    else
        log_success "System resources adequate"
    fi

    return 0
}

check_tooling_dependencies() {
    log_info "Checking tooling dependencies..."

    local missing_critical=()
    local missing_optional=()

    # Critical dependencies
    if ! command -v jq >/dev/null 2>&1; then
        missing_critical+=("jq")
    fi

    if ! command -v bc >/dev/null 2>&1; then
        missing_critical+=("bc")
    fi

    # Optional dependencies
    if ! command -v perf >/dev/null 2>&1; then
        missing_optional+=("perf")
    fi

    if ! command -v flamegraph >/dev/null 2>&1; then
        missing_optional+=("flamegraph")
    fi

    if [[ ${#missing_critical[@]} -gt 0 ]]; then
        log_error "Missing critical tools: ${missing_critical[*]}"
        log_error "Install with: brew install jq bc (macOS) or apt-get install jq bc (Linux)"
        return 1
    fi

    if [[ ${#missing_optional[@]} -gt 0 ]]; then
        log_warning "Missing optional tools: ${missing_optional[*]}"
        log_warning "Some diagnostic features will be unavailable"
    fi

    log_success "All critical tools available"
    return 0
}

# ============================================================================
# SCENARIO EXECUTION
# ============================================================================

wait_for_server() {
    log_info "Waiting for Engram server to be ready..."

    local max_attempts=30
    for i in $(seq 1 $max_attempts); do
        if curl -s "http://localhost:$ENGRAM_PORT/health" >/dev/null 2>&1; then
            log_success "Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done

    log_error "Server failed to become ready after ${max_attempts}s"
    return 1
}

start_engram_server() {
    log_info "Starting Engram server on port $ENGRAM_PORT..."

    # Start server in background
    "$ENGRAM_BIN" start --port "$ENGRAM_PORT" >/dev/null 2>&1 &
    ENGRAM_PID=$!

    # Verify process started
    if ! kill -0 "$ENGRAM_PID" 2>/dev/null; then
        log_error "Failed to start Engram server"
        ENGRAM_PID=""
        return 1
    fi

    # Wait for server to be ready
    if ! wait_for_server; then
        return 1
    fi

    log_success "Engram server started (PID: $ENGRAM_PID)"
    return 0
}

stop_engram_server() {
    if [[ -z "$ENGRAM_PID" ]]; then
        return 0
    fi

    log_info "Stopping Engram server (PID: $ENGRAM_PID)..."

    # Send SIGTERM
    kill -TERM "$ENGRAM_PID" 2>/dev/null || true

    # Wait up to 5 seconds for graceful shutdown
    for i in {1..5}; do
        if ! kill -0 "$ENGRAM_PID" 2>/dev/null; then
            log_success "Server stopped gracefully"
            ENGRAM_PID=""
            return 0
        fi
        sleep 1
    done

    # Force kill if still running
    log_warning "Server not responding, force killing..."
    kill -KILL "$ENGRAM_PID" 2>/dev/null || true
    ENGRAM_PID=""

    return 0
}

monitor_system_metrics() {
    local output_file=$1
    local pid=$2
    local interval=5

    {
        printf "%-10s | %-6s | %-10s | %-8s | %-10s | %-10s\n" \
            "Timestamp" "CPU%" "Mem(MB)" "Threads" "IO_Wait%" "Net(KB/s)"
        printf "%s\n" "$(printf '%.0s-' {1..70})"

        local sample_count=0
        local max_samples=$((SCENARIO_DURATION / interval))

        while [[ $sample_count -lt $max_samples ]]; do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi

            local elapsed=$((sample_count * interval))
            local timestamp
            timestamp=$(printf "%02d:%02d:%02d" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)))

            # Get process stats
            local stats cpu_pct mem_mb threads io_wait net_kbps
            if [[ "$OSTYPE" == "darwin"* ]]; then
                stats=$(ps -p "$pid" -o %cpu,rss,thcount 2>/dev/null || echo "0.0 0 0")
                cpu_pct=$(echo "$stats" | awk '{print $1}')
                mem_mb=$(echo "$stats" | awk '{print int($2/1024)}')
                threads=$(echo "$stats" | awk '{print $3}')
                io_wait="N/A"
                net_kbps="N/A"
            else
                stats=$(ps -p "$pid" -o %cpu,rss,nlwp 2>/dev/null || echo "0.0 0 0")
                cpu_pct=$(echo "$stats" | awk '{print $1}')
                mem_mb=$(echo "$stats" | awk '{print int($2/1024)}')
                threads=$(echo "$stats" | awk '{print $3}')
                io_wait="N/A"
                net_kbps="N/A"
            fi

            printf "%-10s | %-6s | %-10s | %-8s | %-10s | %-10s\n" \
                "$timestamp" "$cpu_pct" "$mem_mb" "$threads" "$io_wait" "$net_kbps"

            ((sample_count++))
            sleep "$interval"
        done
    } > "$output_file" 2>&1 &

    MONITOR_PID=$!
}

collect_diagnostics() {
    local scenario_name=$1
    local output_file=$2

    log_info "Collecting diagnostics for $scenario_name..."

    {
        echo "=== Engram Diagnostics for $scenario_name ==="
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""

        if [[ -x "$DIAGNOSTICS_SCRIPT" ]]; then
            "$DIAGNOSTICS_SCRIPT" 2>&1 || true
        else
            echo "Diagnostics script not found or not executable"
        fi
    } > "$output_file"
}

verify_system_stable() {
    # Check load average
    local load_1min
    load_1min=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ' | cut -d'.' -f1)

    if [[ "$load_1min" -gt 4 ]]; then
        log_warning "System under high load: $load_1min"
        return 1
    fi

    # Check available memory
    local available_mb
    if [[ "$OSTYPE" == "darwin"* ]]; then
        available_mb=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//' | awk '{print int($1*4096/1024/1024)}')
    else
        available_mb=$(grep MemAvailable /proc/meminfo | awk '{print int($2/1024)}')
    fi

    local available_gb=$((available_mb / 1024))
    if [[ "$available_gb" -lt 4 ]]; then
        log_error "Insufficient available memory: ${available_gb}GB"
        return 1
    fi

    return 0
}

run_scenario() {
    local scenario_file=$1
    local scenario_name
    scenario_name=$(basename "$scenario_file" .toml)

    log_info "========================================="
    log_info "Running scenario: $scenario_name"
    log_info "========================================="

    # Pre-scenario validation
    if ! verify_system_stable; then
        log_warning "System not stable, waiting 30s..."
        sleep 30

        if ! verify_system_stable; then
            log_error "System still unstable, skipping scenario"
            FAILED_SCENARIOS+=("$scenario_name")
            FAILURE_REASONS+=("System unstable")
            return 1
        fi
    fi

    # Output files
    local output_file="$OUTPUT_DIR/$TIMESTAMP/${TIMESTAMP}_${scenario_name}.txt"
    local stderr_file="$OUTPUT_DIR/$TIMESTAMP/${TIMESTAMP}_${scenario_name}_stderr.txt"
    local diag_file="$OUTPUT_DIR/$TIMESTAMP/${TIMESTAMP}_${scenario_name}_diag.txt"
    local sys_file="$OUTPUT_DIR/$TIMESTAMP/${TIMESTAMP}_${scenario_name}_sys.txt"

    # Start Engram server
    if ! start_engram_server; then
        log_error "Failed to start Engram server"
        FAILED_SCENARIOS+=("$scenario_name")
        FAILURE_REASONS+=("Server startup failed")
        return 1
    fi

    # Start system monitoring
    monitor_system_metrics "$sys_file" "$ENGRAM_PID"

    # Run loadtest
    log_info "Executing loadtest (${SCENARIO_DURATION}s)..."
    local loadtest_exit=0

    timeout_run "$SERVER_TIMEOUT" "$LOADTEST_BIN" run \
        --scenario "$scenario_file" \
        --duration "$SCENARIO_DURATION" \
        --endpoint "http://localhost:$ENGRAM_PORT" \
        > "$output_file" 2> "$stderr_file" || loadtest_exit=$?

    # Stop monitoring
    if [[ -n "$MONITOR_PID" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
        MONITOR_PID=""
    fi

    # Collect diagnostics
    collect_diagnostics "$scenario_name" "$diag_file"

    # Stop Engram server
    stop_engram_server

    # Check results
    if [[ $loadtest_exit -ne 0 ]]; then
        log_error "Loadtest failed with exit code $loadtest_exit"
        FAILED_SCENARIOS+=("$scenario_name")
        FAILURE_REASONS+=("Loadtest exit code $loadtest_exit")

        # Mark partial results
        mv "$output_file" "${output_file}.partial" 2>/dev/null || true

        return 1
    fi

    log_success "Scenario completed successfully"

    # Cooldown before next scenario
    if [[ $COOLDOWN_PERIOD -gt 0 ]]; then
        log_info "Cooling down for ${COOLDOWN_PERIOD}s..."
        sleep "$COOLDOWN_PERIOD"
    fi

    return 0
}

# ============================================================================
# METADATA AND SUMMARY
# ============================================================================

generate_metadata() {
    local metadata_file="$OUTPUT_DIR/$TIMESTAMP/${TIMESTAMP}_metadata.txt"

    log_info "Generating metadata..."

    {
        echo "=== Competitive Benchmark Suite Metadata ==="
        echo "Timestamp: $TIMESTAMP"
        echo ""

        echo "=== Git Context ==="
        echo "Commit Hash: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
        echo "Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')"
        echo "Dirty: $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted changes"
        echo "Last Commit: $(git log -1 --pretty=%B 2>/dev/null | head -1 || echo 'N/A')"
        echo ""

        echo "=== System Information ==="
        echo "OS: $(uname -a)"

        local mem_gb
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'N/A')"
            echo "CPU Cores: $(sysctl -n hw.physicalcpu 2>/dev/null || echo 'N/A') physical, $(sysctl -n hw.logicalcpu 2>/dev/null || echo 'N/A') logical"
            mem_gb=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
            echo "RAM: ${mem_gb}GB"
        else
            echo "CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
            echo "CPU Cores: $(grep -c "physical id" /proc/cpuinfo | sort -u) physical, $(grep -c "processor" /proc/cpuinfo) logical"
            mem_gb=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024/1024)}')
            echo "RAM: ${mem_gb}GB"
        fi

        echo ""

        echo "=== Software Versions ==="
        echo "Engram Core: $(cargo metadata --format-version 1 --no-deps 2>/dev/null | jq -r '.packages[] | select(.name == "engram-core") | .version' || echo 'N/A')"
        echo "Loadtest: $(cargo metadata --format-version 1 --no-deps 2>/dev/null | jq -r '.packages[] | select(.name == "loadtest") | .version' || echo 'N/A')"
        echo "Rust: $(rustc --version 2>/dev/null || echo 'N/A')"
        echo ""

        echo "=== Benchmark Configuration ==="
        echo "Scenarios Executed: ${#FAILED_SCENARIOS[@]} failed, $(($(find "$SCENARIOS_DIR" -name "*.toml" | wc -l | tr -d ' ') - ${#FAILED_SCENARIOS[@]})) passed"
        echo "Duration per Scenario: ${SCENARIO_DURATION}s"
        echo "Cooldown Period: ${COOLDOWN_PERIOD}s"
        echo "Timestamp Format: YYYY-MM-DD_HH-MM-SS"
    } > "$metadata_file"

    log_success "Metadata generated"
}

generate_summary() {
    local summary_file="$OUTPUT_DIR/$TIMESTAMP/${TIMESTAMP}_summary.txt"

    log_info "Generating summary..."

    local total_scenarios
    total_scenarios=$(find "$SCENARIOS_DIR" -name "*.toml" -type f | wc -l | tr -d ' ')
    local passed=$((total_scenarios - ${#FAILED_SCENARIOS[@]}))
    local suite_duration=$(($(date +%s) - SUITE_START_TIME))

    {
        echo "=== Competitive Benchmark Suite Summary ==="
        echo "Execution Time: $(date -r "$SUITE_START_TIME" '+%Y-%m-%d %H:%M:%S') - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Total Duration: ${suite_duration}s ($((suite_duration / 60))m $((suite_duration % 60))s)"
        echo ""

        echo "=== Results ==="
        echo "Total Scenarios: $total_scenarios"
        echo "Passed: $passed"
        echo "Failed: ${#FAILED_SCENARIOS[@]}"
        echo ""

        if [[ ${#FAILED_SCENARIOS[@]} -gt 0 ]]; then
            echo "=== Failed Scenarios ==="
            for i in "${!FAILED_SCENARIOS[@]}"; do
                echo "  - ${FAILED_SCENARIOS[$i]}: ${FAILURE_REASONS[$i]}"
            done
            echo ""
        fi

        echo "=== Output Location ==="
        echo "$OUTPUT_DIR/$TIMESTAMP/"
        echo ""

        echo "=== Next Steps ==="
        if [[ ${#FAILED_SCENARIOS[@]} -eq 0 ]]; then
            echo "All scenarios passed. Run Task 004 report generator to create comparison report."
        else
            echo "Review failed scenarios and re-run if needed."
        fi
    } > "$summary_file"

    # Print summary to stdout as well
    cat "$summary_file"

    log_success "Summary generated"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    echo "============================================="
    echo "  Competitive Benchmark Suite Runner"
    echo "============================================="
    echo ""

    # Parse command-line arguments
    parse_args "$@"

    # Initialize timestamp
    TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
    SUITE_START_TIME=$(date +%s)

    # Create output directory for this run
    mkdir -p "$OUTPUT_DIR/$TIMESTAMP"

    # Run pre-flight checks
    if ! preflight_checks; then
        log_error "Pre-flight checks failed. Aborting."
        exit 2
    fi

    if [[ "$PREFLIGHT_ONLY" == "true" ]]; then
        log_success "Pre-flight checks completed successfully"
        exit 0
    fi

    # Get scenarios to run
    local scenarios
    if [[ -n "$SINGLE_SCENARIO" ]]; then
        scenarios=("$SCENARIOS_DIR/${SINGLE_SCENARIO}.toml")

        if [[ ! -f "${scenarios[0]}" ]]; then
            log_error "Scenario not found: $SINGLE_SCENARIO"
            exit 1
        fi
    else
        # Read into array (compatible with bash 3.x)
        scenarios=()
        while IFS= read -r file; do
            scenarios+=("$file")
        done < <(find "$SCENARIOS_DIR" -name "*.toml" -type f | sort)
    fi

    # Dry run mode
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run mode - would execute the following scenarios:"
        for scenario in "${scenarios[@]}"; do
            echo "  - $(basename "$scenario" .toml)"
        done
        log_info "Total estimated time: $((${#scenarios[@]} * (SCENARIO_DURATION + COOLDOWN_PERIOD) / 60)) minutes"
        exit 0
    fi

    # Execute scenarios
    log_info "Executing ${#scenarios[@]} scenario(s)..."
    echo ""

    for scenario in "${scenarios[@]}"; do
        run_scenario "$scenario" || true
    done

    # Generate metadata and summary
    generate_metadata
    generate_summary

    # Final status
    echo ""
    echo "============================================="
    if [[ ${#FAILED_SCENARIOS[@]} -eq 0 ]]; then
        log_success "Competitive benchmark suite complete."
        log_success "Results in $OUTPUT_DIR/$TIMESTAMP/"
        log_success "Summary: ${#scenarios[@]}/${#scenarios[@]} scenarios passed"
        exit 0
    else
        log_warning "Competitive benchmark suite complete with failures."
        log_warning "Results in $OUTPUT_DIR/$TIMESTAMP/"
        log_warning "Summary: $((${#scenarios[@]} - ${#FAILED_SCENARIOS[@]}))/${#scenarios[@]} scenarios passed, ${#FAILED_SCENARIOS[@]} failed (${FAILED_SCENARIOS[*]})"
        exit 1
    fi
}

# Run main function
main "$@"
