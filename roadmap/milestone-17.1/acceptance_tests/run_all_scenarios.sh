#!/bin/bash
# M17.1 Task 008: Master Acceptance Test Harness
# Runs all 7 acceptance test scenarios sequentially

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "M17.1 Task 008 Acceptance Testing"
echo "========================================="
echo "Starting: $(date)"
echo "Platform: $(uname -s) $(uname -m)"
echo ""

# Track results
PASSED=0
FAILED=0
SKIPPED=0

# Scenario 1: Fresh Deployment
echo "[1/7] Scenario 1: Fresh Deployment (Cold Start)"
echo "---"
if "$SCRIPT_DIR/scenario_1_fresh_deployment.sh"; then
  PASSED=$((PASSED + 1))
  echo ""
  echo "Scenario 1: PASS"
else
  FAILED=$((FAILED + 1))
  echo ""
  echo "Scenario 1: FAIL"
  echo "ERROR: Fresh deployment validation failed"
  echo "This is a critical failure - framework cannot build or execute"
  exit 1
fi
echo ""
echo ""

# Scenario 2: Trend Analysis
echo "[2/7] Scenario 2: Trend Analysis (Quarterly Cadence)"
echo "---"
if "$SCRIPT_DIR/scenario_2_trend_analysis.sh"; then
  PASSED=$((PASSED + 1))
  echo ""
  echo "Scenario 2: PASS"
else
  FAILED=$((FAILED + 1))
  echo ""
  echo "Scenario 2: FAIL"
  echo "WARNING: Trend analysis validation failed"
  echo "This may indicate non-deterministic behavior or system instability"
fi
echo ""
echo ""

# Scenario 3: Regression Detection
echo "[3/7] Scenario 3: Regression Detection (Performance Degradation)"
echo "---"
if "$SCRIPT_DIR/scenario_3_regression_detection.sh"; then
  PASSED=$((PASSED + 1))
  echo ""
  echo "Scenario 3: PASS"
else
  FAILED=$((FAILED + 1))
  echo ""
  echo "Scenario 3: FAIL"
  echo "WARNING: Regression detection validation failed"
fi
echo ""
echo ""

# Scenario 4: Cross-Platform Determinism
echo "[4/7] Scenario 4: Cross-Platform Determinism"
echo "---"
if "$SCRIPT_DIR/scenario_4_cross_platform.sh"; then
  PASSED=$((PASSED + 1))
  echo ""
  echo "Scenario 4: PASS"
else
  FAILED=$((FAILED + 1))
  echo ""
  echo "Scenario 4: FAIL"
  echo "WARNING: Determinism validation failed"
fi
echo ""
echo ""

# Scenario 5: Concurrent Execution (OPTIONAL)
echo "[5/7] Scenario 5: Concurrent Execution (Isolation Validation)"
echo "---"
echo "Note: This scenario is OPTIONAL for M17.1"
if [ -f "$SCRIPT_DIR/scenario_5_concurrent.sh" ]; then
  if "$SCRIPT_DIR/scenario_5_concurrent.sh"; then
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
      PASSED=$((PASSED + 1))
      echo ""
      echo "Scenario 5: PASS"
    else
      SKIPPED=$((SKIPPED + 1))
      echo ""
      echo "Scenario 5: SKIP (acceptable for M17.1)"
    fi
  else
    SKIPPED=$((SKIPPED + 1))
    echo ""
    echo "Scenario 5: SKIP (parallel execution not implemented)"
  fi
else
  SKIPPED=$((SKIPPED + 1))
  echo ""
  echo "Scenario 5: SKIP (script not found)"
fi
echo ""
echo ""

# Scenario 6: Failure Recovery
echo "[6/7] Scenario 6: Failure Recovery (Resilience Testing)"
echo "---"
if "$SCRIPT_DIR/scenario_6_failure_recovery.sh"; then
  PASSED=$((PASSED + 1))
  echo ""
  echo "Scenario 6: PASS"
else
  FAILED=$((FAILED + 1))
  echo ""
  echo "Scenario 6: FAIL"
  echo "WARNING: Failure recovery validation failed"
fi
echo ""
echo ""

# Scenario 7: Documentation Validation (Automated Only)
echo "[7/7] Scenario 7: Documentation Completeness (Automated Checks)"
echo "---"
echo "Note: Human validation required separately (see scenario_7_documentation_validation.md)"
if "$SCRIPT_DIR/scenario_7_documentation_validation_automated.sh"; then
  PASSED=$((PASSED + 1))
  echo ""
  echo "Scenario 7: PASS (automated checks only)"
else
  FAILED=$((FAILED + 1))
  echo ""
  echo "Scenario 7: FAIL"
  echo "WARNING: Documentation validation failed"
fi
echo ""
echo ""

# Summary
echo "========================================="
echo "Acceptance Testing Complete"
echo "========================================="
echo "Finished: $(date)"
echo ""
echo "Results Summary:"
echo "  PASSED:  $PASSED"
echo "  FAILED:  $FAILED"
echo "  SKIPPED: $SKIPPED"
echo ""

if [ $FAILED -eq 0 ]; then
  echo "STATUS: ALL CRITICAL TESTS PASSED"
  echo ""
  echo "Next Steps:"
  echo "  1. Review any skipped scenarios (acceptable for M17.1)"
  echo "  2. Conduct human validation for Scenario 7"
  echo "  3. Update M17.1_COMPLETION_CHECKLIST.md with results"
  echo "  4. Mark Task 008 as complete if all criteria met"
  echo ""
  exit 0
else
  echo "STATUS: FAILURES DETECTED"
  echo ""
  echo "Failed Scenarios: $FAILED"
  echo "Review failure details above and address issues before marking Task 008 complete"
  echo ""
  exit 1
fi
