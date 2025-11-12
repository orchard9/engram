# Milestone 17 - Performance Baseline Reference

**Date Established**: 2025-11-09
**Baseline File**: `tmp/m17_performance/001_before_20251108_234452.json`
**Codebase State**: Pre-Dual-Memory-Architecture (dev branch)
**Test Duration**: 60 seconds
**Seed**: 0xDEADBEEF (3735928559)

## Executive Summary

This is the official performance baseline for Milestone 17 regression tracking. All M17 tasks must maintain performance within 5% of these metrics.

**Baseline Quality**: EXCELLENT
- Error Rate: 0.0% ✓
- Throughput: 999.9 ops/sec (target: 800) ✓✓
- P99 Latency: 0.501ms (target: <50ms) ✓✓

## Detailed Metrics

### Overall Performance
```json
{
  "total_operations": 59998,
  "total_errors": 0,
  "overall_error_rate": 0.0,
  "overall_throughput": 999.89 ops/sec,
  "p50_latency_ms": 0.208,
  "p95_latency_ms": 0.436,
  "p99_latency_ms": 0.501,
  "p999_latency_ms": 0.584
}
```

### Per-Operation Breakdown

#### Store Operations
```
Count:         30,000 (50.0% of workload)
Success Rate:  100.0% (0 errors)
P50 Latency:   0.178ms
P95 Latency:   0.421ms
P99 Latency:   0.501ms
```

#### Recall Operations
```
Count:         25,429 (42.4% of workload)
Success Rate:  100.0% (0 errors)
P50 Latency:   0.227ms
P95 Latency:   0.445ms
P99 Latency:   0.500ms
```

#### Search Operations
```
Count:         4,569 (7.6% of workload)
Success Rate:  100.0% (0 errors)
P50 Latency:   0.248ms
P95 Latency:   0.459ms
P99 Latency:   0.496ms
```

#### Pattern Completion
```
Status: DISABLED in baseline scenario
Reason: Requires correlated embeddings (baseline uses random data)
Weight: 0.0 (reallocated to Recall +0.025, Search +0.025)
```

## Test Configuration

**Scenario**: `scenarios/m17_baseline.toml`
```toml
[operations]
store_weight = 0.5
recall_weight = 0.425
embedding_search_weight = 0.075
pattern_completion_weight = 0.0  # Disabled (see note)

[data]
num_nodes = 1000
embedding_dim = 768
memory_spaces = 1

[validation]
expected_p99_latency_ms = 50.0
expected_throughput_ops_sec = 800.0
max_error_rate = 0.01
```

## Regression Thresholds

For each M17 task, performance must meet:

**Latency** (5% tolerance):
- P99 must be ≤ 0.526ms (0.501 × 1.05)
- P95 must be ≤ 0.458ms (0.436 × 1.05)

**Throughput** (5% tolerance):
- Overall must be ≥ 949.9 ops/sec (999.9 × 0.95)

**Error Rate**:
- Must remain at 0.0%
- Any errors = immediate investigation required

## How to Use This Baseline

### Before Making M17 Changes
```bash
# Establish "before" snapshot
./scripts/m17_performance_check.sh <task_number> before
```

### After Implementing M17 Task
```bash
# Capture "after" performance
./scripts/m17_performance_check.sh <task_number> after

# Compare against before snapshot
./scripts/compare_m17_performance.sh <task_number>
```

### Interpreting Results

The comparison script will flag regressions:
```
⚠️  REGRESSION: P99 latency increased by 7.2% (threshold: +5%)
```

If regression detected:
1. Profile with `cargo flamegraph --bin engram`
2. Review diagnostics: `tmp/m17_performance/<task>_after_*_diag.txt`
3. Check system metrics: `tmp/m17_performance/<task>_after_*_sys.txt`
4. Fix performance issue before completing task
5. Re-run test until within 5% threshold

## Known Limitations

### PatternCompletion Disabled
PatternCompletion is disabled in the baseline scenario because the workload generator creates random, uncorrelated embeddings. Pattern completion requires semantically similar memories to reconstruct from.

**For PatternCompletion testing**, use dedicated scenario with correlated data:
```bash
# Future: create scenarios/pattern_completion.toml with clustered embeddings
./target/release/loadtest run \
  --scenario scenarios/pattern_completion.toml \
  --duration 60 \
  --seed 0xDEADBEEF \
  --endpoint http://localhost:7432
```

### Server Must Be Running
The performance script now includes server health checks, but verify server is running:
```bash
./target/release/engram status
# or
curl http://localhost:7432/health
```

## Troubleshooting

### High Error Rate (>0%)
Check if server is running:
```bash
./target/release/engram status
./target/release/engram start  # if needed
```

### Low Throughput (<800 ops/sec)
- Check CPU usage during test
- Review system load (concurrent processes)
- Verify database isn't degraded (restart server)

### High Latency (P99 > 50ms)
- System under load from other processes
- Database needs consolidation
- Disk I/O bottleneck

## Change Log

### 2025-11-09: Initial Baseline Established
- Baseline file: `001_before_20251108_234452.json`
- Configuration: Pre-dual-memory architecture (dev branch, commit TBD)
- Quality: 0% errors, 999.9 ops/sec, P99 0.501ms
- Issues Fixed:
  - Server health check added to performance script
  - PatternCompletion disabled (requires correlated data)
  - Enhanced loadtest error logging

### Bug Fixes Applied

**Critical Fixes** (see `tmp/BUG_INVESTIGATION_SUMMARY.md` for details):
1. Server lifecycle management - script now validates server before testing
2. Error logging enhanced - captures HTTP status codes and response bodies
3. Scenario configuration - PatternCompletion properly disabled
4. Investigation tooling - comprehensive diagnostic reports

## References

- Investigation Report: `tmp/LOADTEST_BUG_INVESTIGATION_REPORT.md`
- Executive Summary: `tmp/BUG_INVESTIGATION_SUMMARY.md`
- Comparison Script: `scripts/compare_m17_performance.sh`
- Performance Check Script: `scripts/m17_performance_check.sh`
- Baseline Scenario: `scenarios/m17_baseline.toml`

---

**Baseline Status**: VALIDATED ✓
**Ready for M17 Tasks**: YES
**Next Review**: After first M17 task completion
