#!/usr/bin/env bash
# Collect comprehensive debug information for support escalation
# Creates a tarball with system info, logs, config, metrics, and health status

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEBUG_DIR="engram-debug-$TIMESTAMP"
ENGRAM_URL="${ENGRAM_URL:-http://localhost:7432}"
DATA_DIR="${ENGRAM_DATA_DIR:-./data}"
CONFIG_FILE="${ENGRAM_CONFIG:-~/.config/engram/config.toml}"

echo "Collecting debug information..."
echo "This may take up to 1 minute"
echo ""

mkdir -p "$DEBUG_DIR"

# System information
echo "[1/10] Collecting system information..."
{
    echo "=== System Information ==="
    uname -a
    echo ""
    echo "=== OS Release ==="
    cat /etc/os-release 2>/dev/null || sw_vers 2>/dev/null || echo "OS info not available"
    echo ""
    echo "=== Date/Time ==="
    date
    echo ""
} > "$DEBUG_DIR/system_info.txt" 2>&1

# Process information
echo "[2/10] Collecting process information..."
{
    echo "=== Process List ==="
    ps aux | grep -E "engram|PID" | grep -v grep
    echo ""
    echo "=== Process Tree ==="
    if command -v pstree > /dev/null 2>&1 && pgrep engram > /dev/null 2>&1; then
        pstree -p "$(pgrep engram | head -1)" 2>/dev/null || echo "pstree not available"
    else
        echo "Engram process not running or pstree not available"
    fi
} > "$DEBUG_DIR/process_info.txt" 2>&1

# Resource usage
echo "[3/10] Collecting resource usage..."
{
    echo "=== Top Processes ==="
    if command -v top > /dev/null; then
        top -b -n 1 2>/dev/null | head -20 || top -l 1 | head -20
    else
        echo "top not available"
    fi
    echo ""
    echo "=== Memory Info ==="
    if [ -f /proc/meminfo ]; then
        head -20 /proc/meminfo
    elif command -v vm_stat > /dev/null; then
        vm_stat
    else
        echo "Memory info not available"
    fi
} > "$DEBUG_DIR/resource_usage.txt" 2>&1

# Network information
echo "[4/10] Collecting network information..."
{
    echo "=== Network Connections ==="
    if command -v netstat > /dev/null; then
        netstat -tulpn 2>/dev/null | grep engram || netstat -an | grep -E "7432|50051"
    elif command -v ss > /dev/null; then
        ss -tulpn | grep engram
    else
        echo "Network info not available"
    fi
    echo ""
    echo "=== Port Check ==="
    if command -v lsof > /dev/null; then
        lsof -i :7432 2>/dev/null || echo "Port 7432 not in use"
        lsof -i :50051 2>/dev/null || echo "Port 50051 not in use"
    fi
} > "$DEBUG_DIR/network.txt" 2>&1

# Disk usage
echo "[5/10] Collecting disk usage..."
{
    echo "=== Disk Usage ==="
    df -h
    echo ""
    echo "=== Data Directory ==="
    if [ -d "$DATA_DIR" ]; then
        echo "Data directory: $DATA_DIR"
        du -sh "$DATA_DIR" 2>/dev/null || echo "Cannot calculate size"
        echo ""
        echo "=== Directory Structure ==="
        ls -lah "$DATA_DIR" 2>/dev/null || echo "Cannot list directory"
        echo ""
        echo "=== WAL Directory ==="
        if [ -d "$DATA_DIR/wal" ]; then
            find "$DATA_DIR/wal" -name "*.log" 2>/dev/null | wc -l | xargs echo "WAL files:"
            du -sh "$DATA_DIR/wal" 2>/dev/null || echo "Cannot calculate WAL size"
        else
            echo "WAL directory not found"
        fi
    else
        echo "Data directory not found: $DATA_DIR"
    fi
} > "$DEBUG_DIR/disk_usage.txt" 2>&1

# Logs (last 1000 lines)
echo "[6/10] Collecting logs..."
{
    if command -v journalctl &> /dev/null; then
        journalctl -u engram -n 1000 --no-pager 2>/dev/null || echo "Cannot access journalctl"
    elif [ -f /var/log/engram.log ]; then
        tail -1000 /var/log/engram.log 2>/dev/null
    elif [ -f "$DATA_DIR/engram.log" ]; then
        tail -1000 "$DATA_DIR/engram.log" 2>/dev/null
    else
        echo "No logs found"
        echo "Checked locations:"
        echo "  - journalctl -u engram"
        echo "  - /var/log/engram.log"
        echo "  - $DATA_DIR/engram.log"
    fi
} > "$DEBUG_DIR/logs.txt" 2>&1

# Configuration
echo "[7/10] Collecting configuration..."
{
    if [ -f "$CONFIG_FILE" ]; then
        echo "=== Configuration File: $CONFIG_FILE ==="
        cat "$CONFIG_FILE"
    else
        echo "Configuration file not found: $CONFIG_FILE"
    fi
    echo ""
    echo "=== Environment Variables ==="
    env | grep -E "ENGRAM|RUST" | sort
} > "$DEBUG_DIR/config.txt" 2>&1

# Health check
echo "[8/10] Running health check..."
if command -v curl > /dev/null 2>&1; then
    curl -s "$ENGRAM_URL/api/v1/system/health" 2>&1 > "$DEBUG_DIR/health.json" || echo "Health endpoint not accessible" > "$DEBUG_DIR/health.json"
else
    echo "curl not available" > "$DEBUG_DIR/health.json"
fi

# Metrics snapshot
echo "[9/10] Collecting metrics..."
if command -v curl > /dev/null 2>&1; then
    curl -s "$ENGRAM_URL/metrics" 2>&1 > "$DEBUG_DIR/metrics.txt" || echo "Metrics endpoint not accessible" > "$DEBUG_DIR/metrics.txt"
else
    echo "curl not available" > "$DEBUG_DIR/metrics.txt"
fi

# Run diagnostic script if available
echo "[10/10] Running diagnostic script..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/diagnose_health.sh" ]; then
    "$SCRIPT_DIR/diagnose_health.sh" "$DEBUG_DIR/diagnostic_report.txt" 2>&1 || echo "Diagnostic script failed" >> "$DEBUG_DIR/diagnostic_report.txt"
else
    echo "Diagnostic script not found at $SCRIPT_DIR/diagnose_health.sh" > "$DEBUG_DIR/diagnostic_report.txt"
fi

# Create summary
{
    echo "=== Debug Bundle Summary ==="
    echo "Collected: $TIMESTAMP"
    echo "Hostname: $(hostname)"
    echo "Engram URL: $ENGRAM_URL"
    echo "Data Directory: $DATA_DIR"
    echo ""
    echo "=== Bundle Contents ==="
    ls -lh "$DEBUG_DIR"
} > "$DEBUG_DIR/README.txt"

# Package into tarball
echo ""
echo "Creating tarball..."
TARBALL="$DEBUG_DIR.tar.gz"
tar -czf "$TARBALL" "$DEBUG_DIR/" 2>/dev/null

# Cleanup
rm -rf "$DEBUG_DIR"

# Show results
echo ""
echo "========================================"
echo "Debug bundle created successfully!"
echo "========================================"
echo ""
echo "File: $TARBALL"
echo "Size: $(du -h "$TARBALL" | awk '{print $1}')"
echo ""
echo "This bundle contains:"
echo "  - System and process information"
echo "  - Recent logs (last 1000 lines)"
echo "  - Configuration files"
echo "  - Health and metrics snapshots"
echo "  - Diagnostic report"
echo ""
echo "Send this file to support for analysis:"
echo "  support@engram.example.com"
echo ""
echo "To extract: tar -xzf $TARBALL"
echo ""

exit 0
