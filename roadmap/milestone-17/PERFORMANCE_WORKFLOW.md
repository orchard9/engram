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

## Competitive Validation (Optional)

For tasks that modify core graph operations, competitive validation ensures Engram maintains its positioning against Neo4j and other graph databases.

### When to Use Competitive Validation

Run competitive validation if your task modifies:
- **Spreading activation algorithms** - Changes to activation propagation, decay, or thresholds
- **Graph traversal** - Node navigation, edge following, or path finding optimizations
- **Vector search** - Embedding similarity calculations or HNSW modifications
- **Memory consolidation** - Episode-to-concept transformation or schema extraction

**Not required for:**
- Documentation updates
- Test-only changes
- Configuration changes
- Monitoring/metrics additions

### Competitive Validation Workflow

```bash
# 1. Establish competitive baseline (before changes)
./scripts/m17_performance_check.sh 001 before --competitive

# 2. Implement task...

# 3. Run competitive validation (after changes)
./scripts/m17_performance_check.sh 001 after --competitive

# 4. Compare against baseline AND Neo4j
./scripts/compare_m17_performance.sh 001 --competitive
```

### Competitive Thresholds

| Metric | Internal Target | Competitive Target | Baseline |
|--------|----------------|-------------------|----------|
| P99 latency increase | <5% | <10% | Neo4j: 27.96ms |
| Throughput decrease | <5% | <10% | Neo4j: 280 QPS |
| Exit code | 1 (internal regression) | 2 (competitive regression) | 0 (success) |

**Competitive regression (exit code 2)** indicates:
- Performance degraded >10% vs internal baseline
- Engram's competitive positioning weakened vs Neo4j
- Requires investigation before task completion

### Interpreting Competitive Results

**Example output:**
```
Competitive Positioning:
Metric               vs Neo4j
-------------------- ----------
P99 latency            61.2% faster
Neo4j P99             27.96ms (baseline)
Engram P99            10.85ms (Engram)

Checking for competitive regressions (>10% threshold)...

✓ No competitive regressions detected (within 10% threshold)

Summary for PERFORMANCE_LOG.md:
- Before: P50=8.2ms, P95=9.5ms, P99=10.1ms, 490 ops/s
- After:  P50=8.7ms, P95=10.2ms, P99=10.85ms, 475 ops/s
- Change: +7.4% P99 latency, -3.1% throughput
- Status: ✓ Within 10% competitive target
- vs Neo4j: 61.2% (baseline: 27.96ms)
```

**What this means:**
- Internal delta: +7.4% P99 latency (exceeds 5% internal target but acceptable for competitive)
- Competitive delta: Still 61% faster than Neo4j baseline
- Decision: Accept change, document in PERFORMANCE_LOG.md

### Competitive Scenario Details

Competitive validation uses `scenarios/competitive/hybrid_production_100k.toml`:
- **Workload mix**: 30% store, 30% recall, 30% search, 10% pattern completion
- **Dataset**: 100K nodes, scale-free distribution
- **Seed**: 0xABCD1234 (deterministic)
- **Duration**: 60 seconds
- **Comparison**: Neo4j graph traversal baseline (27.96ms P99)

This scenario represents realistic production workloads combining graph and vector operations - Engram's key differentiation vs pure graph or vector databases.

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

## Example Workflow with Competitive Validation

```bash
# Starting Task 005 (modifies spreading activation)
./scripts/m17_performance_check.sh 005 before
./scripts/m17_performance_check.sh 005 before --competitive

# Implement task...

# Standard validation
./scripts/m17_performance_check.sh 005 after
./scripts/compare_m17_performance.sh 005

# Competitive validation
./scripts/m17_performance_check.sh 005 after --competitive
./scripts/compare_m17_performance.sh 005 --competitive

# Both validations pass - update logs and commit
git commit -m "feat(m17): Complete Task 005 - Graph Storage Adaptation

Performance:
- Internal: P99 +3.2%, throughput -1.8% (within 5% target)
- Competitive: P99 +6.8%, still 58% faster than Neo4j (within 10% target)"
```
