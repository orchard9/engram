#!/usr/bin/env bash
# Comprehensive health diagnostic for Engram
# Performs 10 health checks and provides actionable recommendations

set -euo pipefail

ENGRAM_URL="${ENGRAM_URL:-http://localhost:7432}"
OUTPUT="${1:-/dev/stdout}"

exec > >(tee -a "$OUTPUT")

echo "========================================"
echo "   Engram Health Diagnostic Report     "
echo "========================================"
echo "Timestamp: $(date)"
echo ""

# Track overall health status
HEALTH_STATUS=0

# Check 1: Process status
echo "[1/10] Checking process status..."
if pgrep -x engram > /dev/null 2>&1; then
    PID=$(pgrep -x engram)
    echo "  Process running (PID: $PID)"
    if command -v ps > /dev/null; then
        UPTIME=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ' || echo "unknown")
        echo "  Uptime: $UPTIME"
    fi
else
    echo "  CRITICAL: Engram process not running"
    echo "  Action: Check systemd status or start manually"
    HEALTH_STATUS=1
fi
echo ""

# Check 2: HTTP health endpoint
echo "[2/10] Checking HTTP health endpoint..."
if command -v curl > /dev/null 2>&1; then
    if HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$ENGRAM_URL/api/v1/system/health" 2>/dev/null); then
        if [ "$HTTP_STATUS" = "200" ]; then
            echo "  HTTP health endpoint responding (200 OK)"
        else
            echo "  WARNING: HTTP health endpoint returned $HTTP_STATUS"
            echo "  Action: Check logs for errors"
            HEALTH_STATUS=1
        fi
    else
        echo "  CRITICAL: HTTP health endpoint not accessible"
        echo "  Action: Check firewall, port bindings, and logs"
        HEALTH_STATUS=1
    fi
else
    echo "  SKIP: curl not installed"
fi
echo ""

# Check 3: gRPC endpoint
echo "[3/10] Checking gRPC endpoint..."
if command -v grpcurl &> /dev/null; then
    if grpcurl -plaintext localhost:50051 list > /dev/null 2>&1; then
        echo "  gRPC endpoint responding"
    else
        echo "  WARNING: gRPC endpoint not accessible"
        echo "  Action: Check gRPC configuration and port"
    fi
else
    echo "  SKIP: grpcurl not installed"
fi
echo ""

# Check 4: Storage tiers
echo "[4/10] Checking storage tiers..."
if [ -n "${ENGRAM_DATA_DIR:-}" ]; then
    DATA_DIR="$ENGRAM_DATA_DIR"
else
    DATA_DIR="./data"
fi

if [ -d "$DATA_DIR" ]; then
    if command -v df > /dev/null; then
        DISK_USAGE=$(df -h "$DATA_DIR" 2>/dev/null | awk 'NR==2 {print $5}' | tr -d '%' || echo "0")
        echo "  Data directory exists: $DATA_DIR"
        echo "  Disk usage: ${DISK_USAGE}%"

        if [ "$DISK_USAGE" -gt 90 ]; then
            echo "  CRITICAL: Disk usage >90%"
            echo "  Action: Free up space or expand volume"
            HEALTH_STATUS=1
        elif [ "$DISK_USAGE" -gt 80 ]; then
            echo "  WARNING: Disk usage >80%"
            echo "  Action: Plan for capacity expansion"
        fi
    else
        echo "  Data directory exists: $DATA_DIR"
    fi
else
    echo "  CRITICAL: Data directory not found: $DATA_DIR"
    echo "  Action: Create data directory or set ENGRAM_DATA_DIR"
    HEALTH_STATUS=1
fi
echo ""

# Check 5: WAL status
echo "[5/10] Checking WAL status..."
if [ -d "$DATA_DIR/wal" ]; then
    WAL_COUNT=$(find "$DATA_DIR/wal" -name "*.log" 2>/dev/null | wc -l | tr -d ' ')
    if command -v du > /dev/null; then
        WAL_SIZE=$(du -sh "$DATA_DIR/wal" 2>/dev/null | awk '{print $1}' || echo "unknown")
    else
        WAL_SIZE="unknown"
    fi
    echo "  WAL directory exists"
    echo "  WAL files: $WAL_COUNT"
    echo "  WAL size: $WAL_SIZE"

    if [ "$WAL_COUNT" -gt 100 ]; then
        echo "  WARNING: Large number of WAL files"
        echo "  Action: Check consolidation is running"
    fi
else
    echo "  WARNING: WAL directory not found"
    echo "  Action: WAL may not be initialized yet"
fi
echo ""

# Check 6: Memory usage
echo "[6/10] Checking memory usage..."
if [ -n "${PID:-}" ] && [ -f /proc/"$PID"/status ] 2>/dev/null; then
    RSS=$(awk '/^VmRSS:/ {print $2}' /proc/"$PID"/status 2>/dev/null || echo "0")
    RSS_MB=$((RSS / 1024))
    echo "  Process memory: ${RSS_MB}MB RSS"

    if [ "$RSS_MB" -gt 4096 ]; then
        echo "  WARNING: High memory usage (>4GB)"
        echo "  Action: Check for memory leaks or reduce hot tier size"
    fi
elif [ -n "${PID:-}" ] && command -v ps > /dev/null; then
    RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ' || echo "0")
    RSS_MB=$((RSS_KB / 1024))
    echo "  Process memory: ${RSS_MB}MB RSS"

    if [ "$RSS_MB" -gt 4096 ]; then
        echo "  WARNING: High memory usage (>4GB)"
        echo "  Action: Check for memory leaks or reduce hot tier size"
    fi
else
    echo "  SKIP: Cannot determine memory usage"
fi
echo ""

# Check 7: Open file descriptors
echo "[7/10] Checking file descriptors..."
if [ -n "${PID:-}" ] && [ -d /proc/"$PID"/fd ] 2>/dev/null; then
    FD_COUNT=$(ls -1 /proc/"$PID"/fd 2>/dev/null | wc -l | tr -d ' ')
    FD_LIMIT=$(ulimit -n)
    echo "  Open file descriptors: $FD_COUNT / $FD_LIMIT"

    FD_PERCENT=$((FD_COUNT * 100 / FD_LIMIT))
    if [ "$FD_PERCENT" -gt 80 ]; then
        echo "  CRITICAL: FD usage >80%"
        echo "  Action: Increase ulimit or check for fd leaks"
        HEALTH_STATUS=1
    fi
elif [ -n "${PID:-}" ] && command -v lsof > /dev/null 2>&1; then
    FD_COUNT=$(lsof -p "$PID" 2>/dev/null | wc -l | tr -d ' ')
    FD_LIMIT=$(ulimit -n)
    echo "  Open file descriptors: ~$FD_COUNT / $FD_LIMIT"
else
    FD_LIMIT=$(ulimit -n)
    echo "  File descriptor limit: $FD_LIMIT"
    echo "  SKIP: Cannot count open file descriptors"
fi
echo ""

# Check 8: Recent errors in logs
echo "[8/10] Checking recent errors..."
if command -v journalctl &> /dev/null; then
    ERROR_COUNT=$(journalctl -u engram --since "5 minutes ago" 2>/dev/null | grep -c ERROR || echo "0")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "  WARNING: $ERROR_COUNT errors in last 5 minutes"
        echo "  Action: Review logs with: journalctl -u engram -n 100"
    else
        echo "  No errors in last 5 minutes"
    fi
elif [ -f /var/log/engram.log ]; then
    ERROR_COUNT=$(tail -1000 /var/log/engram.log 2>/dev/null | grep -c ERROR || echo "0")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo "  WARNING: $ERROR_COUNT errors in recent logs"
        echo "  Action: Review logs at /var/log/engram.log"
    else
        echo "  No recent errors found"
    fi
else
    echo "  SKIP: Cannot access logs"
fi
echo ""

# Check 9: Connectivity
echo "[9/10] Checking network connectivity..."
if command -v curl > /dev/null 2>&1; then
    if curl -sf "$ENGRAM_URL/api/v1/system/health" > /dev/null 2>&1; then
        RESPONSE_TIME=$(curl -s -o /dev/null -w "%{time_total}" "$ENGRAM_URL/api/v1/system/health" 2>/dev/null || echo "0")
        echo "  API responding in ${RESPONSE_TIME}s"

        # Check if response time is slow (>1s)
        if command -v bc > /dev/null 2>&1; then
            if [ "$(echo "$RESPONSE_TIME > 1.0" | bc -l 2>/dev/null || echo "0")" = "1" ]; then
                echo "  WARNING: Slow response time (>1s)"
                echo "  Action: Check CPU/disk load and network latency"
            fi
        fi
    else
        echo "  CRITICAL: Cannot connect to API"
        echo "  Action: Check if service is running and port is accessible"
        HEALTH_STATUS=1
    fi
else
    echo "  SKIP: curl not installed"
fi
echo ""

# Check 10: Metrics availability
echo "[10/10] Checking metrics..."
if command -v curl > /dev/null 2>&1; then
    if curl -sf "$ENGRAM_URL/metrics" > /dev/null 2>&1; then
        echo "  Metrics endpoint accessible"
    else
        echo "  WARNING: Metrics endpoint not accessible"
        echo "  Action: Verify monitoring feature enabled"
    fi
else
    echo "  SKIP: curl not installed"
fi
echo ""

echo "========================================"
echo "Diagnostic complete"
echo ""

if [ $HEALTH_STATUS -eq 0 ]; then
    echo "Status: HEALTHY"
    echo "All critical checks passed"
else
    echo "Status: UNHEALTHY"
    echo "One or more critical issues detected"
    echo "Review recommendations above and take corrective action"
fi

echo ""
echo "Save this report for troubleshooting or support escalation"

exit $HEALTH_STATUS
