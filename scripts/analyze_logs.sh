#!/usr/bin/env bash
# Log analysis and error pattern aggregation for Engram
# Categorizes errors into 7 families and provides actionable recommendations

set -euo pipefail

SINCE="${1:-1 hour ago}"
OUTPUT="${2:-/dev/stdout}"

exec > >(tee -a "$OUTPUT")

echo "========================================"
echo "   Engram Log Analysis Report          "
echo "========================================"
echo "Time Range: $SINCE to now"
echo "Generated: $(date)"
echo ""

# Get logs for analysis
LOGS=""
if command -v journalctl &> /dev/null; then
  LOGS=$(journalctl -u engram --since "$SINCE" --no-pager 2>/dev/null || echo "")
fi

if [ -z "$LOGS" ]; then
  if [ -f /var/log/engram.log ]; then
    LOGS=$(tail -10000 /var/log/engram.log 2>/dev/null || echo "")
  elif [ -f "${ENGRAM_DATA_DIR:-./data}/engram.log" ]; then
    LOGS=$(tail -10000 "${ENGRAM_DATA_DIR:-./data}/engram.log" 2>/dev/null || echo "")
  fi
fi

if [ -z "$LOGS" ]; then
  echo "ERROR: No logs found. Check log location or journalctl access."
  echo ""
  echo "Tried locations:"
  echo "  - journalctl -u engram --since \"$SINCE\""
  echo "  - /var/log/engram.log"
  echo "  - \${ENGRAM_DATA_DIR}/engram.log"
  exit 1
fi

# Error Summary
echo "=== ERROR SUMMARY ==="
echo ""

ERROR_COUNT=$(echo "$LOGS" | grep -c "ERROR" 2>/dev/null || echo "0")
WARN_COUNT=$(echo "$LOGS" | grep -c "WARN" 2>/dev/null || echo "0")
INFO_COUNT=$(echo "$LOGS" | grep -c "INFO" 2>/dev/null || echo "0")
TOTAL_LINES=$(echo "$LOGS" | wc -l | tr -d ' ')

echo "Total Log Lines: $TOTAL_LINES"
echo "Total Errors:    $ERROR_COUNT"
echo "Total Warnings:  $WARN_COUNT"
echo "Total Info:      $INFO_COUNT"
echo ""

if [ "$ERROR_COUNT" -eq 0 ] && [ "$WARN_COUNT" -eq 0 ]; then
  echo "No errors or warnings found. System appears healthy."
  echo ""
  exit 0
fi

# Top Error Patterns
echo "=== TOP ERROR PATTERNS ==="
echo ""

if [ "$ERROR_COUNT" -gt 0 ]; then
  echo "$LOGS" | grep "ERROR" | \
    sed 's/.*ERROR //g' | \
    sed 's/ at .*//g' | \
    sed 's/:.*//' | \
    sort | uniq -c | sort -rn | head -10 | \
    awk '{printf "  %5d  %s\n", $1, substr($0, index($0,$2))}' || echo "  No patterns identified"
else
  echo "  No errors found"
fi

echo ""

# Error Categories (7 families from task specification)
echo "=== ERROR CATEGORIES ==="
echo ""

# Category 1: Node/Memory Access Errors
NODE_ERRORS=$(echo "$LOGS" | grep -ic "node.*not found\|memory.*not found\|invalid.*node" 2>/dev/null || echo "0")

# Category 2: Activation/Confidence Violations
ACTIVATION_ERRORS=$(echo "$LOGS" | grep -ic "invalid activation\|invalid confidence\|activation level\|confidence interval\|out of bounds" 2>/dev/null || echo "0")

# Category 3: Storage/Persistence Errors
STORAGE_ERRORS=$(echo "$LOGS" | grep -ic "WAL\|persist\|deserialize\|serialize\|disk.*full\|no space left" 2>/dev/null || echo "0")

# Category 4: Multi-Space Errors
SPACE_ERRORS=$(echo "$LOGS" | grep -ic "memory space.*not found\|isolation\|space.*error" 2>/dev/null || echo "0")

# Category 5: Index/Query Errors
INDEX_ERRORS=$(echo "$LOGS" | grep -ic "index.*corrupt\|fallback\|query.*failed\|pattern matching.*failed" 2>/dev/null || echo "0")

# Category 6: NaN/Numerical Errors
NAN_ERRORS=$(echo "$LOGS" | grep -ic "NaN\|Infinity\|infinite\|numerical" 2>/dev/null || echo "0")

# Category 7: Resource Exhaustion
RESOURCE_ERRORS=$(echo "$LOGS" | grep -ic "out of memory\|too many open files\|EMFILE\|ENOSPC\|exhausted" 2>/dev/null || echo "0")

echo "1. Node/Memory Access Errors:    $NODE_ERRORS"
echo "2. Activation/Confidence Errors: $ACTIVATION_ERRORS"
echo "3. Storage/Persistence Errors:   $STORAGE_ERRORS"
echo "4. Multi-Space Errors:           $SPACE_ERRORS"
echo "5. Index/Query Errors:           $INDEX_ERRORS"
echo "6. NaN/Numerical Errors:         $NAN_ERRORS"
echo "7. Resource Exhaustion:          $RESOURCE_ERRORS"
echo ""

# Performance Indicators
echo "=== PERFORMANCE INDICATORS ==="
echo ""

SLOW_QUERIES=$(echo "$LOGS" | grep -ic "slow query\|latency.*ms\|timeout" 2>/dev/null || echo "0")
WAL_LAG_WARNS=$(echo "$LOGS" | grep -ic "WAL lag\|WAL.*behind" 2>/dev/null || echo "0")
CONSOLIDATION_ISSUES=$(echo "$LOGS" | grep -ic "consolidation.*timeout\|consolidation.*failed\|consolidation.*stuck" 2>/dev/null || echo "0")

echo "  Slow Query Warnings:          $SLOW_QUERIES"
echo "  WAL Lag Warnings:             $WAL_LAG_WARNS"
echo "  Consolidation Issues:         $CONSOLIDATION_ISSUES"
echo ""

# Recovery Strategy Indicators
echo "=== RECOVERY STRATEGIES ACTIVATED ==="
echo ""

RETRIES=$(echo "$LOGS" | grep -ic "retrying\|retry attempt" 2>/dev/null || echo "0")
FALLBACKS=$(echo "$LOGS" | grep -ic "fallback\|falling back" 2>/dev/null || echo "0")
PARTIAL_RESULTS=$(echo "$LOGS" | grep -ic "partial result\|partial success" 2>/dev/null || echo "0")

echo "  Retry Attempts:               $RETRIES"
echo "  Fallback Activations:         $FALLBACKS"
echo "  Partial Results:              $PARTIAL_RESULTS"
echo ""

# Critical Issues Requiring Attention
echo "=== CRITICAL ISSUES ==="
echo ""

CRITICAL_PATTERNS=(
  "data loss"
  "corruption"
  "out of memory"
  "no space left"
  "panic"
  "segfault"
  "deadlock"
  "cannot allocate"
)

CRITICAL_FOUND=false
for pattern in "${CRITICAL_PATTERNS[@]}"; do
  COUNT=$(echo "$LOGS" | grep -ic "$pattern" 2>/dev/null || echo "0")
  if [ "$COUNT" -gt 0 ]; then
    echo "  ! $pattern: $COUNT occurrences"
    CRITICAL_FOUND=true
  fi
done

if [ "$CRITICAL_FOUND" = false ]; then
  echo "  No critical issues detected"
fi

echo ""

# Recent Error Timeline
echo "=== RECENT ERROR TIMELINE (Last 10) ==="
echo ""

if [ "$ERROR_COUNT" -gt 0 ]; then
  echo "$LOGS" | grep "ERROR" | tail -10 | \
    sed 's/^/  /' | \
    cut -c1-120 || echo "  No errors found"
else
  echo "  No errors found"
fi

echo ""

# Recent Warning Timeline
echo "=== RECENT WARNING TIMELINE (Last 5) ==="
echo ""

if [ "$WARN_COUNT" -gt 0 ]; then
  echo "$LOGS" | grep "WARN" | tail -5 | \
    sed 's/^/  /' | \
    cut -c1-120 || echo "  No warnings found"
else
  echo "  No warnings found"
fi

echo ""
echo "========================================"
echo "Analysis complete"
echo ""

# Provide actionable recommendations based on error patterns
echo "=== ACTIONABLE RECOMMENDATIONS ==="
echo ""

RECOMMENDATIONS=0

if [ "$STORAGE_ERRORS" -gt 100 ]; then
  echo "HIGH PRIORITY: Storage/Persistence Errors"
  echo "  Issue: $STORAGE_ERRORS storage errors detected"
  echo "  Action: Check disk health and WAL lag"
  echo "  Reference: Issue 3, 5 in troubleshooting guide"
  echo "  Command: ./scripts/diagnose_health.sh"
  echo ""
  RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

if [ "$NAN_ERRORS" -gt 0 ]; then
  echo "HIGH PRIORITY: NaN/Numerical Errors"
  echo "  Issue: $NAN_ERRORS NaN errors detected"
  echo "  Action: Run emergency sanitization"
  echo "  Reference: Issue 7 in troubleshooting guide"
  echo "  Command: ./scripts/emergency_recovery.sh --sanitize-nan --dry-run"
  echo ""
  RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

if [ "$SPACE_ERRORS" -gt 10 ]; then
  echo "MEDIUM PRIORITY: Multi-Space Isolation Issues"
  echo "  Issue: $SPACE_ERRORS multi-space errors detected"
  echo "  Action: Verify space creation and isolation"
  echo "  Reference: Issue 6 in troubleshooting guide"
  echo "  Command: curl http://localhost:7432/api/v1/system/health | jq .spaces"
  echo ""
  RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

if [ "$INDEX_ERRORS" -gt 5 ]; then
  echo "MEDIUM PRIORITY: Index Corruption"
  echo "  Issue: $INDEX_ERRORS index errors detected"
  echo "  Action: Rebuild indices"
  echo "  Reference: Issue 9 in troubleshooting guide"
  echo "  Command: ./scripts/emergency_recovery.sh --rebuild-indices --dry-run"
  echo ""
  RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

if [ "$WAL_LAG_WARNS" -gt 20 ]; then
  echo "MEDIUM PRIORITY: Persistent WAL Lag"
  echo "  Issue: $WAL_LAG_WARNS WAL lag warnings detected"
  echo "  Action: Check consolidation status"
  echo "  Reference: Issue 3, 8 in troubleshooting guide"
  echo "  Command: curl http://localhost:9090/api/v1/query --data-urlencode 'query=engram_wal_lag_seconds'"
  echo ""
  RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

if [ "$SLOW_QUERIES" -gt 50 ]; then
  echo "MEDIUM PRIORITY: High Query Latency"
  echo "  Issue: $SLOW_QUERIES slow query warnings detected"
  echo "  Action: Run performance profiling"
  echo "  Reference: Issue 2 in troubleshooting guide"
  echo "  Command: ./scripts/profile_performance.sh 60"
  echo ""
  RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

if [ "$RESOURCE_ERRORS" -gt 0 ]; then
  echo "HIGH PRIORITY: Resource Exhaustion"
  echo "  Issue: $RESOURCE_ERRORS resource exhaustion errors detected"
  echo "  Action: Check memory, disk, and file descriptors"
  echo "  Reference: Issue 4 in troubleshooting guide"
  echo "  Command: ./scripts/diagnose_health.sh"
  echo ""
  RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

if [ "$CONSOLIDATION_ISSUES" -gt 10 ]; then
  echo "MEDIUM PRIORITY: Consolidation Issues"
  echo "  Issue: $CONSOLIDATION_ISSUES consolidation problems detected"
  echo "  Action: Check consolidation configuration and performance"
  echo "  Reference: Issue 8 in troubleshooting guide"
  echo "  Command: journalctl -u engram | grep -i consolidation | tail -50"
  echo ""
  RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

if [ "$CRITICAL_FOUND" = true ]; then
  echo "CRITICAL: Severe System Issues"
  echo "  Issue: Critical patterns detected (see above)"
  echo "  Action: Immediate escalation required"
  echo "  Reference: Incident response guide (SEV1)"
  echo "  Command: ./scripts/collect_debug_info.sh"
  echo ""
  RECOMMENDATIONS=$((RECOMMENDATIONS + 1))
fi

if [ "$RECOMMENDATIONS" -eq 0 ]; then
  if [ "$ERROR_COUNT" -gt 0 ] || [ "$WARN_COUNT" -gt 0 ]; then
    echo "No specific high-priority issues identified."
    echo "Review error patterns above for context."
    echo ""
    echo "General troubleshooting steps:"
    echo "  1. Run health diagnostic: ./scripts/diagnose_health.sh"
    echo "  2. Check system resources: top, df -h, ulimit -n"
    echo "  3. Review recent changes: git log -10"
  else
    echo "System appears healthy. No recommendations at this time."
  fi
fi

echo ""
echo "For detailed troubleshooting, see: docs/operations/troubleshooting.md"
echo ""

# Exit with error code if critical issues found
if [ "$CRITICAL_FOUND" = true ]; then
  exit 1
elif [ "$ERROR_COUNT" -gt 100 ]; then
  exit 1
else
  exit 0
fi
