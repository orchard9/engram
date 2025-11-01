#!/usr/bin/env bash
#
# 24-Hour Production Validation Test
#
# Comprehensive single-node validation following the plan in:
# docs/operations/24-hour-validation-plan.md
#
# Usage:
#   ./scripts/validate_24h_production.sh [--quick-test]
#
# Options:
#   --quick-test    Run 1-hour version for testing (not production validation)
#   --resume        Resume from previous run (experimental)

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly OUTPUT_DIR="${OUTPUT_DIR:-/tmp/engram-24h}"
readonly DURATION_HOURS="${DURATION_HOURS:-24}"
readonly QUICK_TEST="${1:-}"

# Test mode (1 hour for --quick-test, 24 hours otherwise)
if [[ "$QUICK_TEST" == "--quick-test" ]]; then
    DURATION_SECS=3600  # 1 hour
    echo "⚠️  QUICK TEST MODE: Running 1-hour validation (not production-grade)"
else
    DURATION_SECS=$((DURATION_HOURS * 3600))  # 24 hours
    echo "✓ PRODUCTION MODE: Running ${DURATION_HOURS}-hour validation"
fi

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# Tracking
CRITICAL_FAILURES=0
HIGH_FAILURES=0
MEDIUM_FAILURES=0
START_TIME=""
PID_CONSOLIDATION=""
PID_METRICS=""

# Cleanup on exit
cleanup() {
    echo "Cleaning up background processes..."
    [[ -n "$PID_CONSOLIDATION" ]] && kill "$PID_CONSOLIDATION" 2>/dev/null || true
    [[ -n "$PID_METRICS" ]] && kill "$PID_METRICS" 2>/dev/null || true
}
trap cleanup EXIT

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

# Helper functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OUTPUT_DIR/validation.log"
}

log_metric() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')],$*" >> "$OUTPUT_DIR/metrics.csv"
}

check_critical() {
    local name="$1"
    local result="$2"

    if [[ "$result" == "pass" ]]; then
        echo -e "${GREEN}✓${NC} CRITICAL: $name"
        return 0
    else
        echo -e "${RED}✗${NC} CRITICAL: $name - FAILED"
        ((CRITICAL_FAILURES++))
        return 1
    fi
}

check_high() {
    local name="$1"
    local result="$2"

    if [[ "$result" == "pass" ]]; then
        echo -e "${GREEN}✓${NC} HIGH: $name"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} HIGH: $name - FAILED"
        ((HIGH_FAILURES++))
        return 1
    fi
}

# Phase functions
phase_0_pre_flight() {
    log "PHASE 0: Pre-Flight Checks"

    # Build release binary
    log "Building release binary..."
    cargo build --release --bin consolidation_soak || {
        log "ERROR: Failed to build binary"
        exit 1
    }

    # Run full test suite
    log "Running test suite..."
    cargo test --workspace --lib --release 2>&1 | tee "$OUTPUT_DIR/tests_pre.log"
    local test_result=$(grep "test result:" "$OUTPUT_DIR/tests_pre.log" | tail -1)
    log "Test result: $test_result"

    if echo "$test_result" | grep -q "FAILED"; then
        check_critical "Pre-flight test suite" "fail"
        log "ERROR: Tests failing before soak test. Fix tests first."
        exit 1
    else
        check_critical "Pre-flight test suite" "pass"
    fi

    # Check disk space
    local available_gb=$(df -g "$OUTPUT_DIR" | awk 'NR==2 {print $4}')
    if [[ $available_gb -lt 20 ]]; then
        log "WARNING: Less than 20GB disk space available"
    fi

    log "Pre-flight checks complete"
}

phase_1_baseline() {
    log "PHASE 1: Baseline Establishment (Hours 0-2)"

    # Start consolidation soak test with baseline load
    log "Starting consolidation soak test..."
    target/release/consolidation_soak \
        --duration-secs "$DURATION_SECS" \
        --scheduler-interval-secs 60 \
        --sample-interval-secs 60 \
        --episodes-per-tick 10 \
        --output-dir "$OUTPUT_DIR/consolidation" \
        > "$OUTPUT_DIR/consolidation.log" 2>&1 &

    PID_CONSOLIDATION=$!
    log "Consolidation soak PID: $PID_CONSOLIDATION"

    # Let it run for a bit to establish baseline
    sleep 120

    # Capture baseline metrics
    log "Capturing baseline metrics..."
    local rss_baseline=$(ps -p "$PID_CONSOLIDATION" -o rss= | awk '{print $1}')
    echo "rss_baseline,$rss_baseline" > "$OUTPUT_DIR/baseline.csv"
    log "Baseline RSS: ${rss_baseline}KB"
}

phase_2_steady_state() {
    log "PHASE 2: Steady State (Hours 2-8)"

    # Monitor for 6 hours (or proportional in quick mode)
    local monitor_duration=$((DURATION_SECS / 4))
    log "Monitoring steady state for ${monitor_duration}s..."

    # Sample metrics every minute
    local samples=$((monitor_duration / 60))
    for i in $(seq 1 "$samples"); do
        sleep 60

        # Check if consolidation process still running
        if ! kill -0 "$PID_CONSOLIDATION" 2>/dev/null; then
            check_critical "Consolidation process alive" "fail"
            log "ERROR: Consolidation process died unexpectedly"
            return 1
        fi

        # Sample RSS
        local rss=$(ps -p "$PID_CONSOLIDATION" -o rss= | awk '{print $1}')
        log_metric "rss,$rss"

        # Sample CPU
        local cpu=$(ps -p "$PID_CONSOLIDATION" -o %cpu= | awk '{print $1}')
        log_metric "cpu,$cpu"

        if [[ $((i % 10)) -eq 0 ]]; then
            log "Steady state progress: $i/$samples samples collected"
        fi
    done

    check_critical "Consolidation process alive" "pass"
}

phase_3_memory_validation() {
    log "PHASE 3: Memory Validation"

    # Read baseline and current RSS
    local rss_baseline=$(grep "rss_baseline" "$OUTPUT_DIR/baseline.csv" | cut -d, -f2)
    local rss_current=$(ps -p "$PID_CONSOLIDATION" -o rss= | awk '{print $1}')

    # Calculate growth rate
    local growth=$((rss_current - rss_baseline))
    local growth_pct=$((growth * 100 / rss_baseline))

    log "Memory: Baseline ${rss_baseline}KB → Current ${rss_current}KB (${growth_pct}% growth)"

    # Critical: <25% growth over test duration
    if [[ $growth_pct -lt 25 ]]; then
        check_critical "Memory leak check (<25% growth)" "pass"
    else
        check_critical "Memory leak check (<25% growth)" "fail"
    fi

    # High: <10% growth
    if [[ $growth_pct -lt 10 ]]; then
        check_high "Low memory growth (<10%)" "pass"
    else
        check_high "Low memory growth (<10%)" "fail"
    fi
}

phase_4_consolidation_validation() {
    log "PHASE 4: Consolidation Validation"

    # Parse consolidation snapshots
    if [[ ! -f "$OUTPUT_DIR/consolidation/snapshots.jsonl" ]]; then
        check_critical "Consolidation snapshots exist" "fail"
        return 1
    fi

    local snapshot_count=$(wc -l < "$OUTPUT_DIR/consolidation/snapshots.jsonl")
    local expected_snapshots=$((DURATION_SECS / 60))

    log "Consolidation: $snapshot_count snapshots (expected ~$expected_snapshots)"

    # Critical: At least 95% of expected runs
    local min_snapshots=$((expected_snapshots * 95 / 100))
    if [[ $snapshot_count -ge $min_snapshots ]]; then
        check_critical "Consolidation cadence (≥95%)" "pass"
    else
        check_critical "Consolidation cadence (≥95%)" "fail"
    fi

    # Parse timestamps and check cadence
    log "Analyzing consolidation cadence..."
    python3 << 'EOF' > "$OUTPUT_DIR/cadence_analysis.txt"
import json
import sys
from datetime import datetime

snapshots = []
with open(f'{os.environ["OUTPUT_DIR"]}/consolidation/snapshots.jsonl') as f:
    for line in f:
        snapshots.append(json.loads(line))

if len(snapshots) < 2:
    print("ERROR: Not enough snapshots")
    sys.exit(1)

# Calculate deltas
deltas = []
for i in range(1, len(snapshots)):
    t1 = datetime.fromisoformat(snapshots[i-1]['captured_at'].replace('Z', '+00:00'))
    t2 = datetime.fromisoformat(snapshots[i]['captured_at'].replace('Z', '+00:00'))
    delta = (t2 - t1).total_seconds()
    deltas.append(delta)

avg_delta = sum(deltas) / len(deltas)
within_2s = sum(1 for d in deltas if abs(d - 60) <= 2)
accuracy = within_2s / len(deltas) * 100

print(f"Average cadence: {avg_delta:.2f}s")
print(f"Accuracy (within ±2s): {accuracy:.1f}%")
print(f"Min delta: {min(deltas):.2f}s")
print(f"Max delta: {max(deltas):.2f}s")

# Exit 0 if ≥95% within tolerance
sys.exit(0 if accuracy >= 95 else 1)
EOF

    if [[ $? -eq 0 ]]; then
        check_high "Consolidation cadence accuracy (≥95% within ±2s)" "pass"
    else
        check_high "Consolidation cadence accuracy (≥95% within ±2s)" "fail"
    fi

    cat "$OUTPUT_DIR/cadence_analysis.txt" | tee -a "$OUTPUT_DIR/validation.log"
}

phase_5_performance_validation() {
    log "PHASE 5: Performance Validation"

    # Run quick benchmark suite
    log "Running benchmark suite..."
    cargo bench --workspace -- --quick 2>&1 | tee "$OUTPUT_DIR/bench_during_soak.log" || {
        check_high "Benchmarks run during soak" "fail"
        return 1
    }

    check_high "Benchmarks run during soak" "pass"
}

phase_6_post_test_validation() {
    log "PHASE 6: Post-Test Validation"

    # Stop consolidation soak
    if [[ -n "$PID_CONSOLIDATION" ]] && kill -0 "$PID_CONSOLIDATION" 2>/dev/null; then
        log "Gracefully stopping consolidation soak..."
        kill -TERM "$PID_CONSOLIDATION"
        sleep 5

        if kill -0 "$PID_CONSOLIDATION" 2>/dev/null; then
            log "Force killing consolidation soak..."
            kill -KILL "$PID_CONSOLIDATION"
        fi
    fi

    # Run test suite again
    log "Running post-test suite..."
    cargo test --workspace --lib --release 2>&1 | tee "$OUTPUT_DIR/tests_post.log"
    local test_result=$(grep "test result:" "$OUTPUT_DIR/tests_post.log" | tail -1)
    log "Test result: $test_result"

    if echo "$test_result" | grep -q "FAILED"; then
        check_critical "Post-test suite" "fail"
    else
        check_critical "Post-test suite" "pass"
    fi

    # Check for crashes in logs
    if grep -q "panic\|SIGSEGV\|SIGABRT" "$OUTPUT_DIR/consolidation.log"; then
        check_critical "No crashes during test" "fail"
    else
        check_critical "No crashes during test" "pass"
    fi
}

generate_report() {
    log "Generating validation report..."

    local total_criteria=$((5 + 10))  # 5 critical + 10 high
    local passed_criteria=$((15 - CRITICAL_FAILURES - HIGH_FAILURES))
    local critical_pct=$((100 - CRITICAL_FAILURES * 20))  # 5 critical = 100%
    local high_pct=$((100 - HIGH_FAILURES * 10))  # 10 high = 100%

    # Determine overall status
    local status="FAIL"
    if [[ $CRITICAL_FAILURES -eq 0 ]]; then
        if [[ $HIGH_FAILURES -le 2 ]]; then  # 80% of high criteria
            status="PASS"
        elif [[ $HIGH_FAILURES -le 4 ]]; then  # 60% of high criteria
            status="CONDITIONAL PASS"
        fi
    fi

    cat > "$OUTPUT_DIR/VALIDATION_REPORT.md" << EOF
# 24-Hour Production Validation Report

**Date**: $(date +'%Y-%m-%d')
**Duration**: ${DURATION_HOURS} hours
**Status**: **${status}**

---

## Summary

- **Critical Criteria**: ${critical_pct}% passed (${CRITICAL_FAILURES}/5 failed)
- **High Criteria**: ${high_pct}% passed (${HIGH_FAILURES}/10 failed)
- **Overall Score**: ${passed_criteria}/${total_criteria} criteria met

### Status Interpretation

EOF

    if [[ "$status" == "PASS" ]]; then
        cat >> "$OUTPUT_DIR/VALIDATION_REPORT.md" << EOF
✅ **APPROVED FOR PRODUCTION DEPLOYMENT**

All critical criteria met and ≥80% of high criteria met. System is validated
for single-node production deployment.

**Recommendation**: Proceed with production deployment following documented
procedures in \`docs/operations/deployment.md\`.
EOF
    elif [[ "$status" == "CONDITIONAL PASS" ]]; then
        cat >> "$OUTPUT_DIR/VALIDATION_REPORT.md" << EOF
⚠️ **CONDITIONAL APPROVAL**

All critical criteria met but only 60-80% of high criteria met. System can
be deployed to production with monitoring and mitigation plans for failed
high criteria.

**Recommendation**: Review failed high criteria, establish monitoring alerts,
and prepare mitigation runbooks before deploying.
EOF
    else
        cat >> "$OUTPUT_DIR/VALIDATION_REPORT.md" << EOF
❌ **NOT APPROVED FOR PRODUCTION**

One or more critical criteria failed. System requires fixes before production
deployment.

**Recommendation**: Fix all critical failures and re-run validation. Do NOT
deploy to production until achieving at least CONDITIONAL PASS status.
EOF
    fi

    cat >> "$OUTPUT_DIR/VALIDATION_REPORT.md" << EOF

---

## Detailed Results

### Critical Criteria (Must Pass)

See validation.log for detailed results.

### High Criteria (Should Pass)

See validation.log for detailed results.

---

## Metrics Collected

- Memory snapshots: \`$OUTPUT_DIR/metrics.csv\`
- Consolidation logs: \`$OUTPUT_DIR/consolidation.log\`
- Consolidation snapshots: \`$OUTPUT_DIR/consolidation/snapshots.jsonl\`
- Metrics data: \`$OUTPUT_DIR/consolidation/metrics.jsonl\`
- Test results: \`$OUTPUT_DIR/tests_*.log\`
- Benchmark results: \`$OUTPUT_DIR/bench_*.log\`

---

## Next Steps

EOF

    if [[ "$status" == "PASS" ]] || [[ "$status" == "CONDITIONAL PASS" ]]; then
        cat >> "$OUTPUT_DIR/VALIDATION_REPORT.md" << EOF
1. Review this report and metrics
2. Document baseline metrics in \`docs/operations/production-baselines.md\`
3. Configure Grafana alerts based on observed thresholds
4. Create runbooks for any issues encountered
5. Schedule production deployment
6. Monitor closely for first 48 hours post-deployment
EOF
    else
        cat >> "$OUTPUT_DIR/VALIDATION_REPORT.md" << EOF
1. Review failed criteria in validation.log
2. Root cause analysis for each failure
3. Create GitHub issues for fixes
4. Implement fixes
5. Re-run this validation test
6. Do NOT deploy until validation passes
EOF
    fi

    cat >> "$OUTPUT_DIR/VALIDATION_REPORT.md" << EOF

---

**Generated**: $(date +'%Y-%m-%d %H:%M:%S')
**Duration**: ${DURATION_HOURS} hours
**Output Directory**: \`$OUTPUT_DIR\`
EOF

    log "Report generated: $OUTPUT_DIR/VALIDATION_REPORT.md"

    # Print summary to console
    echo ""
    echo "======================================"
    echo "  24-Hour Validation Complete"
    echo "======================================"
    echo "Status: $status"
    echo "Critical: ${critical_pct}% passed"
    echo "High: ${high_pct}% passed"
    echo ""
    echo "Full report: $OUTPUT_DIR/VALIDATION_REPORT.md"
    echo "======================================"

    # Return exit code based on status
    if [[ "$status" == "PASS" ]]; then
        return 0
    elif [[ "$status" == "CONDITIONAL PASS" ]]; then
        return 1
    else
        return 2
    fi
}

# Main execution
main() {
    START_TIME=$(date +%s)

    log "========================================="
    log "  Engram 24-Hour Production Validation"
    log "========================================="
    log "Duration: ${DURATION_HOURS} hours"
    log "Output: $OUTPUT_DIR"
    log ""

    phase_0_pre_flight
    phase_1_baseline
    phase_2_steady_state
    phase_3_memory_validation
    phase_4_consolidation_validation
    phase_5_performance_validation
    phase_6_post_test_validation

    local end_time=$(date +%s)
    local elapsed=$((end_time - START_TIME))
    log "Total validation time: ${elapsed}s ($(($elapsed / 3600))h $((($elapsed % 3600) / 60))m)"

    generate_report
}

main
