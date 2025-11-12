# M17.1 Task 008: Acceptance Test Suite

Comprehensive acceptance testing for the competitive baseline framework.

## Quick Start

Run all acceptance scenarios:
```bash
./run_all_scenarios.sh
```

This will execute all 7 acceptance scenarios sequentially and provide a summary report.

## Individual Scenarios

### Scenario 1: Fresh Deployment (Cold Start)
```bash
./scenario_1_fresh_deployment.sh
```

Validates that the framework can build and execute from scratch on a clean system.

**Expected Duration**: 2-5 minutes
**Critical**: Yes - Must pass for M17.1 completion

---

### Scenario 2: Trend Analysis (Quarterly Cadence)
```bash
./scenario_2_trend_analysis.sh
```

Verifies longitudinal tracking and comparison across time periods.

**Expected Duration**: 1-2 minutes
**Critical**: Yes

---

### Scenario 3: Regression Detection (Performance Degradation)
```bash
./scenario_3_regression_detection.sh
```

Validates automated detection of performance regressions.

**Expected Duration**: 1-2 minutes
**Critical**: Yes

---

### Scenario 4: Cross-Platform Determinism
```bash
./scenario_4_cross_platform.sh
```

Verifies scenarios produce identical operation sequences across platforms.

**Expected Duration**: 1 minute
**Critical**: No (optional for M17.1)

**Note**: Full cross-platform validation requires running on both macOS and Linux. This script validates determinism on the current platform only.

---

### Scenario 5: Concurrent Execution (Isolation Validation)
```bash
./scenario_5_concurrent.sh
```

Verifies scenarios don't interfere when run in parallel.

**Expected Duration**: 1-2 minutes (if supported)
**Critical**: No (optional for M17.1)

**Note**: Parallel execution is not required for M17.1. Sequential execution (15min quarterly review) is acceptable.

---

### Scenario 6: Failure Recovery (Resilience Testing)
```bash
./scenario_6_failure_recovery.sh
```

Validates graceful handling of various failure modes.

**Expected Duration**: 2-3 minutes
**Critical**: Yes

---

### Scenario 7: Documentation Completeness

**Automated Checks**:
```bash
./scenario_7_documentation_validation_automated.sh
```

**Expected Duration**: <1 minute
**Critical**: Yes

**Human Validation**:
See `scenario_7_documentation_validation.md` for manual testing protocol.

**Expected Duration**: 30 minutes (with recruited engineer)
**Critical**: Yes

---

## Success Criteria

M17.1 Task 008 acceptance testing is complete when:

1. At least 5/7 scenarios pass (scenarios 4 and 5 are optional)
2. All failure modes tested and documented
3. Human validation completed for Scenario 7
4. Results documented in M17.1_COMPLETION_CHECKLIST.md

## Troubleshooting

### Scenario fails with "Loadtest binary not found"
```bash
cd ../../..  # Navigate to project root
cargo build --release
```

### Scenario fails with "Insufficient memory"
Close other applications and re-run. Some scenarios (especially milvus_ann_10m_768d) require 16GB+ RAM.

### Scenario shows high variance between runs
Ensure system is idle with no background processes. CPU throttling and thermal management can affect results.

### Make quality fails
```bash
cd ../../..  # Navigate to project root
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

Fix any clippy warnings before marking Task 008 complete.

## File Structure

```
acceptance_tests/
├── README.md                                          # This file
├── run_all_scenarios.sh                               # Master test harness
├── scenario_1_fresh_deployment.sh                     # Fresh deployment test
├── scenario_2_trend_analysis.sh                       # Trend analysis test
├── scenario_3_regression_detection.sh                 # Regression detection test
├── scenario_4_cross_platform.sh                       # Cross-platform test
├── scenario_5_concurrent.sh                           # Concurrent execution test
├── scenario_6_failure_recovery.sh                     # Failure recovery test
├── scenario_7_documentation_validation.md             # Human validation protocol
└── scenario_7_documentation_validation_automated.sh   # Automated doc checks
```

## Integration with M17.1 Completion

After all scenarios pass:

1. Update `M17.1_COMPLETION_CHECKLIST.md` with test results
2. Document any known limitations discovered during testing
3. Verify `make quality` passes (zero clippy warnings)
4. Mark Task 008 as complete: rename `008_*_in_progress.md` to `008_*_complete.md`
5. Commit all changes with performance summary

## References

- [M17.1 Completion Checklist](../M17.1_COMPLETION_CHECKLIST.md)
- [Task 008 Specification](../008_documentation_and_acceptance_testing_in_progress.md)
- [Competitive Baselines Documentation](../../../docs/reference/competitive_baselines.md)
