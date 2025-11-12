#!/usr/bin/env bash
# System Requirements Verification Script
# Validates hardware meets minimum specifications for competitive baseline measurement

set -euo pipefail

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

# Requirements
readonly MIN_RAM_GB=16
readonly RECOMMENDED_RAM_GB=32
readonly MIN_CPU_CORES=4
readonly RECOMMENDED_CPU_CORES=8
readonly MIN_DISK_FREE_GB=10

# Track failures
REQUIREMENTS_MET=true

echo "=== System Requirements Verification ==="
echo

# Check RAM
echo "Checking RAM..."
if command -v python3 &>/dev/null; then
    RAM_CHECK=$(python3 <<EOF
import psutil
total_ram_gb = psutil.virtual_memory().total / (1024**3)
available_ram_gb = psutil.virtual_memory().available / (1024**3)
print(f"{total_ram_gb:.1f},{available_ram_gb:.1f}")
EOF
)
    TOTAL_RAM=$(echo "$RAM_CHECK" | cut -d',' -f1)
    AVAILABLE_RAM=$(echo "$RAM_CHECK" | cut -d',' -f2)

    if (( $(echo "$TOTAL_RAM < $MIN_RAM_GB" | bc -l) )); then
        echo -e "${RED}ERROR${NC}: Insufficient RAM (${TOTAL_RAM}GB < ${MIN_RAM_GB}GB required)"
        REQUIREMENTS_MET=false
    elif (( $(echo "$TOTAL_RAM < $RECOMMENDED_RAM_GB" | bc -l) )); then
        echo -e "${YELLOW}WARNING${NC}: RAM below recommended (${TOTAL_RAM}GB < ${RECOMMENDED_RAM_GB}GB recommended)"
    else
        echo -e "${GREEN}PASS${NC}: RAM sufficient (${TOTAL_RAM}GB total, ${AVAILABLE_RAM}GB available)"
    fi

    if (( $(echo "$AVAILABLE_RAM < 12" | bc -l) )); then
        echo -e "${YELLOW}WARNING${NC}: Low available RAM (${AVAILABLE_RAM}GB < 12GB recommended)"
        echo "  Close other applications before running 1M+ scenarios"
    fi
else
    echo -e "${YELLOW}WARNING${NC}: python3 not found, skipping RAM check"
fi
echo

# Check CPU cores
echo "Checking CPU cores..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    CPU_CORES=$(sysctl -n hw.ncpu)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CPU_CORES=$(nproc)
else
    CPU_CORES=$(python3 -c "import os; print(os.cpu_count())" 2>/dev/null || echo "unknown")
fi

if [[ "$CPU_CORES" != "unknown" ]]; then
    if (( CPU_CORES < MIN_CPU_CORES )); then
        echo -e "${RED}ERROR${NC}: Insufficient CPU cores ($CPU_CORES < $MIN_CPU_CORES required)"
        REQUIREMENTS_MET=false
    elif (( CPU_CORES < RECOMMENDED_CPU_CORES )); then
        echo -e "${YELLOW}WARNING${NC}: CPU cores below recommended ($CPU_CORES < $RECOMMENDED_CPU_CORES)"
        echo "  Performance may be lower than documented baselines"
    else
        echo -e "${GREEN}PASS${NC}: CPU cores sufficient ($CPU_CORES cores)"
    fi
else
    echo -e "${YELLOW}WARNING${NC}: Unable to determine CPU core count"
fi
echo

# Check disk space
echo "Checking disk space..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    DISK_FREE_GB=$(df -g . | tail -1 | awk '{print $4}')
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    DISK_FREE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
else
    DISK_FREE_GB="unknown"
fi

if [[ "$DISK_FREE_GB" != "unknown" ]]; then
    if (( DISK_FREE_GB < MIN_DISK_FREE_GB )); then
        echo -e "${RED}ERROR${NC}: Insufficient disk space (${DISK_FREE_GB}GB < ${MIN_DISK_FREE_GB}GB required)"
        REQUIREMENTS_MET=false
    else
        echo -e "${GREEN}PASS${NC}: Disk space sufficient (${DISK_FREE_GB}GB free)"
    fi
else
    echo -e "${YELLOW}WARNING${NC}: Unable to determine disk space"
fi
echo

# Check thermal state (macOS only)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Checking thermal state..."
    # Use timeout to prevent hanging on thermlog
    if timeout 5 pmset -g thermlog 2>/dev/null | grep -q "CPU_Scheduler_Limit" 2>/dev/null; then
        echo -e "${YELLOW}WARNING${NC}: System under thermal pressure"
        echo "  Allow system to cool before running baseline measurement"
        echo "  Wait 5 minutes, close resource-intensive applications"
    else
        echo -e "${GREEN}PASS${NC}: Thermal state acceptable (or cannot determine)"
    fi
    echo
fi

# Summary
echo "=== Summary ==="
if [[ "$REQUIREMENTS_MET" == "true" ]]; then
    echo -e "${GREEN}System meets minimum requirements for baseline measurement${NC}"
    exit 0
else
    echo -e "${RED}System does NOT meet minimum requirements${NC}"
    echo "Fix the errors above before running baseline measurement"
    exit 1
fi
