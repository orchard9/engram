#!/usr/bin/env bash
# Engram Process Diagnostics Tool
# Checks for running engram processes and logs CPU/memory usage

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log file with prepend
LOG_FILE="${PWD}/tmp/engram_diagnostics.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Function to prepend to log file
log_diagnostic() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local temp_file=$(mktemp)

    # Write new entry
    echo "=== Diagnostic Run: $timestamp ===" > "$temp_file"
    cat >> "$temp_file"
    echo "" >> "$temp_file"

    # Prepend to existing log (if exists)
    if [ -f "$LOG_FILE" ]; then
        cat "$LOG_FILE" >> "$temp_file"
    fi

    mv "$temp_file" "$LOG_FILE"
}

# Check for engram processes
check_processes() {
    echo "Checking for engram processes..."

    # Find engram binary processes (exclude this script)
    local engram_procs=$(ps aux | grep -E "(engram-cli|engram-core|target/.*/engram)" | grep -v "grep" | grep -v "engram_diagnostics")

    if [ -n "$engram_procs" ]; then
        echo -e "${YELLOW}⚠ Found running engram processes:${NC}"

        # Count and display processes
        local count=0
        while IFS= read -r line; do
            if [ -n "$line" ]; then
                count=$((count + 1))
                local pid=$(echo "$line" | awk '{print $2}')
                local cpu=$(echo "$line" | awk '{print $3}')
                local mem=$(echo "$line" | awk '{print $4}')
                local cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')

                echo -e "  PID: $pid | CPU: ${cpu}% | MEM: ${mem}% | CMD: $cmd"

                # Warn if high CPU
                if command -v bc > /dev/null && (( $(echo "$cpu > 50.0" | bc -l 2>/dev/null || echo "0") )); then
                    echo -e "    ${RED}⚠ HIGH CPU USAGE${NC}"
                fi
            fi
        done <<< "$engram_procs"

        echo -e "\n${YELLOW}Total engram processes: $count${NC}"

        if [ "$count" -gt 2 ]; then
            echo -e "${RED}⚠ WARNING: More than 2 engram processes detected!${NC}"
            echo -e "${RED}  This may indicate leaked test processes.${NC}"
            echo -e "${RED}  Run with --kill flag to clean up: ./scripts/engram_diagnostics.sh --kill${NC}"
        fi

        return 1
    else
        echo -e "${GREEN}✓ No engram processes found${NC}"
        return 0
    fi
}

# Check system resources
check_system() {
    echo -e "\nSystem Resources:"

    # CPU load
    if command -v uptime > /dev/null; then
        load=$(uptime | awk -F'load average:' '{print $2}' | sed 's/^[ \t]*//')
        echo "  Load average: $load"
    fi

    # Memory (macOS)
    if command -v vm_stat > /dev/null; then
        free_mem=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        free_mb=$((free_mem * 4096 / 1024 / 1024))
        echo "  Free memory: ~${free_mb}MB"
    fi
}

# Main execution
main() {
    echo "=================================="
    echo "  Engram Process Diagnostics"
    echo "=================================="
    echo ""

    # Capture output for logging
    {
        check_processes
        process_status=$?

        check_system

        echo ""
        echo "=================================="

        # Exit with error if processes found
        if [ $process_status -ne 0 ]; then
            echo -e "${YELLOW}⚠ Action Required: Clean up engram processes${NC}"
            exit 1
        else
            echo -e "${GREEN}✓ System clean${NC}"
            exit 0
        fi
    } | tee >(log_diagnostic)
}

# Run with optional cleanup flag
if [ "${1:-}" = "--kill" ]; then
    echo "Killing all engram processes..."
    pkill -9 engram || true
    echo "Done. Re-running diagnostics..."
    echo ""
fi

main
