# Milestone 17 Performance Testing Workflow

Quick reference for M17 performance tracking process.

## Overview

Milestone 17 implements dual memory architecture with a <5% performance regression target. Every task requires before/after performance validation.

## Workflow

### Before Starting a Task

```bash
# Example for Task 001
./scripts/m17_performance_check.sh 001 before
```

This runs a 60-second deterministic load test and saves results to `tmp/m17_performance/`.

### After Completing a Task

```bash
# Run after test
./scripts/m17_performance_check.sh 001 after

# Compare results
./scripts/compare_m17_performance.sh 001
```

The comparison script will:
- Display before/after metrics table
- Calculate percentage changes
- Detect regressions (>5% P99 latency increase or >5% throughput decrease)
- Exit with code 1 if regressions detected, 0 if acceptable

### If Regression Detected

1. **Profile** to find hot spots:
   ```bash
   cargo flamegraph --bin engram
   ```

2. **Check diagnostics**:
   ```bash
   cat tmp/m17_performance/001_after_*_diag.txt
   ```

3. **Review system metrics**:
   ```bash
   cat tmp/m17_performance/001_after_*_sys.txt
   ```

4. **Fix issues** and re-run after test until regression <5%

### Update Performance Log

After successful validation:

1. Copy summary from compare script output
2. Update `roadmap/milestone-17/PERFORMANCE_LOG.md` with results
3. Mark task status as ✓ or ⚠️ depending on regression

## File Locations

- **Scripts**: `scripts/m17_performance_check.sh`, `scripts/compare_m17_performance.sh`
- **Scenario**: `scenarios/m17_baseline.toml`
- **Results**: `tmp/m17_performance/`
- **Performance log**: `roadmap/milestone-17/PERFORMANCE_LOG.md`

## Load Test Scenario

The baseline scenario runs:
- 40% episode storage operations
- 40% episodic recall operations
- 10% semantic recall (falls back to episodic until M17 enabled)
- 10% blended recall (falls back to episodic until M17 enabled)

Target: 1000 ops/second for 60 seconds (deterministic seed 0xDEADBEEF).

## Performance Targets

| Metric | Threshold |
|--------|-----------|
| P99 latency increase | <5% |
| Throughput decrease | <5% |
| Error count increase | 0 |

## Commit Message Format

```
feat(m17): Complete Task XXX - <task name>

<implementation details>

Performance: P99 latency +X.X%, throughput +Y.Y% (within 5% target)
```

## Troubleshooting

### "jq not found"
```bash
brew install jq  # macOS
```

### "bc not found"
```bash
brew install bc  # macOS
```

### Server won't start
Check diagnostics:
```bash
./scripts/engram_diagnostics.sh
```

### Load test fails
Check loadtest log:
```bash
cat tmp/m17_performance/<task>_<phase>_*_loadtest.log
```

## Example Complete Workflow

```bash
# Starting Task 001
./scripts/m17_performance_check.sh 001 before

# Implement task following CLAUDE.md workflow...
# Write code, tests, run make quality, etc.

# Validate performance
./scripts/m17_performance_check.sh 001 after
./scripts/compare_m17_performance.sh 001

# If regression detected, fix and re-test
# Otherwise, update PERFORMANCE_LOG.md

# Commit
git add .
git commit -m "feat(m17): Complete Task 001 - Dual Memory Types

Implemented cache-aligned DualMemoryNode with Episode/Concept variants.
Added zero-copy conversions and feature flag gating.

Performance: P99 latency +2.3%, throughput -0.8% (within 5% target)"
```
