# Task 007: Performance Regression Prevention Integration

**Status**: Pending
**Complexity**: Moderate
**Dependencies**: Task 006 (requires initial baseline)
**Estimated Effort**: 4 hours

## Objective

Integrate competitive benchmarks into the existing M17 performance regression framework to ensure future changes don't degrade competitive positioning.

## Specifications

1. **Extend `scripts/m17_performance_check.sh`**:
   - Add optional `--competitive` flag
   - When set, run `scenarios/competitive/hybrid_production_100k.toml` instead of default scenario
   - Store results in `tmp/m17_performance/competitive_<task>_<phase>_*.txt`
   - Use same regression threshold (5%)

2. **Extend `scripts/compare_m17_performance.sh`**:
   - Detect if inputs are from competitive scenarios (check for "competitive" in filename)
   - If competitive scenario, include comparison against baseline from `competitive_baselines.md`
   - Output format:
     ```
     Task XXX Competitive Performance:
     Before: P99 10.2ms, 490 QPS
     After:  P99 10.8ms, 470 QPS
     Delta:  +5.9% latency, -4.1% throughput (REGRESSION DETECTED)
     vs Baseline (Neo4j): 27.96ms (61% faster)
     ```

3. **Update CLAUDE.md Workflow** (Task execution instructions):
   - Add optional step after M17 task completion:
     - "If task modifies core graph operations, run competitive validation:"
     - `./scripts/m17_performance_check.sh <task> before --competitive`
     - `./scripts/m17_performance_check.sh <task> after --competitive`
     - `./scripts/compare_m17_performance.sh <task> --competitive`

4. **Create Competitive Regression Alert**:
   - If competitive scenario regresses >10% (stricter than 5% internal target):
     - Print warning: "COMPETITIVE REGRESSION DETECTED"
     - Print impact: "Engram now X% slower than <competitor>"
     - Suggest profiling: "Run `cargo flamegraph` to identify hot spots"
   - Exit code: 2 (different from normal regression exit code 1)

5. **Documentation Update**:
   - Add section to `roadmap/milestone-17/PERFORMANCE_WORKFLOW.md`:
     - **Competitive Validation** (optional for critical tasks)
     - When to use: Changes to spreading activation, graph traversal, vector search
     - How to interpret: Competitive delta vs internal delta

## File Paths

```
scripts/m17_performance_check.sh (modify existing)
scripts/compare_m17_performance.sh (modify existing)
CLAUDE.md (modify existing)
roadmap/milestone-17/PERFORMANCE_WORKFLOW.md (modify existing)
```

## Acceptance Criteria

1. `--competitive` flag works with both performance check scripts
2. Competitive regression detection triggers distinct exit code (2)
3. Comparison output includes both internal delta and competitive positioning
4. CLAUDE.md clearly explains when to use competitive validation
5. Scripts pass shellcheck linting (zero warnings)
6. Backward compatibility: Scripts work without `--competitive` flag (existing behavior)

## Testing Approach

```bash
# Validate shell syntax
shellcheck scripts/m17_performance_check.sh
shellcheck scripts/compare_m17_performance.sh

# Test competitive flag
./scripts/m17_performance_check.sh 999 before --competitive
# Should run hybrid_production_100k.toml scenario

# Test comparison output
./scripts/m17_performance_check.sh 999 after --competitive
./scripts/compare_m17_performance.sh 999 --competitive
# Should show delta vs Neo4j baseline

# Test regression detection (manual: artificially degrade performance)
# Modify engram to add 10ms sleep in graph traversal
# Run after test, verify exit code 2

# Test backward compatibility
./scripts/m17_performance_check.sh 999 before
# Should run default scenario (not competitive)
```

## Integration Points

- Extends Task 003 benchmark infrastructure
- Uses Task 002 baseline documentation
- Integrates with existing M17 workflow (`PERFORMANCE_WORKFLOW.md`)
- Updates CLAUDE.md task execution instructions
