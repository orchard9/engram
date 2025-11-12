# Task 007: Performance Regression Prevention Integration

**Status**: Pending
**Complexity**: Moderate
**Dependencies**: Task 006 (requires initial baseline)
**Estimated Effort**: 4 hours

## Objective

Integrate competitive benchmarks into the existing M17 performance regression framework to ensure future changes don't degrade competitive positioning. Extends the proven M17 regression detection workflow (scripts/m17_performance_check.sh, scripts/compare_m17_performance.sh) with competitive validation capabilities.

## Problem Statement

Engram has a mature internal regression framework (<5% tolerance) but no mechanism to detect when changes degrade competitive positioning relative to Neo4j, Qdrant, etc. This task integrates competitive benchmarks into the existing workflow without breaking backward compatibility or imposing overhead on tasks that don't touch competitive-critical code paths.

## Integration Architecture

### Design Principles

1. **Backward Compatibility**: Existing M17 workflow continues to work unchanged
2. **Opt-In Competitive Testing**: Use `--competitive` flag only for tasks affecting core graph operations
3. **Consistent Interface**: Same script structure, same output format, same exit code semantics (extended)
4. **Minimal Performance Overhead**: Competitive scenarios only run when explicitly requested

### File Naming Conventions

**Internal regression testing** (existing):
- Before: `tmp/m17_performance/<task>_before_<timestamp>.json`
- After: `tmp/m17_performance/<task>_after_<timestamp>.json`

**Competitive validation** (new):
- Before: `tmp/m17_performance/competitive_<task>_before_<timestamp>.json`
- After: `tmp/m17_performance/competitive_<task>_after_<timestamp>.json`

**Pattern matching**: Compare script detects competitive mode by checking for "competitive_" prefix in filename.

### Script Extension Design

#### `scripts/m17_performance_check.sh` Modifications

Add optional `--competitive` flag as third positional argument:

```bash
# Existing usage (unchanged)
./scripts/m17_performance_check.sh <task_id> <phase>

# New usage (opt-in)
./scripts/m17_performance_check.sh <task_id> <phase> --competitive
```

**Implementation strategy**:
1. Accept optional third argument after positional task_id and phase
2. If `--competitive` present, set scenario to `scenarios/competitive/hybrid_production_100k.toml`
3. If `--competitive` present, modify output filenames to include "competitive_" prefix
4. Default behavior unchanged (uses `scenarios/m17_baseline.toml`)

**Validation logic**:
```bash
COMPETITIVE_MODE=0
if [[ "${3:-}" == "--competitive" ]]; then
    COMPETITIVE_MODE=1
    SCENARIO="scenarios/competitive/hybrid_production_100k.toml"
    PREFIX="competitive_"
else
    SCENARIO="scenarios/m17_baseline.toml"
    PREFIX=""
fi
```

#### `scripts/compare_m17_performance.sh` Modifications

Detect competitive mode by examining input filenames:

```bash
# Detect if comparing competitive results
COMPETITIVE_MODE=0
if [[ "$BEFORE_FILE" == *"competitive_"* ]]; then
    COMPETITIVE_MODE=1
fi
```

**Competitive mode behavior**:
1. Use 10% regression threshold instead of 5% (stricter tolerance for competitive positioning)
2. Load competitor baselines from `docs/reference/competitive_baselines.md`
3. Include "vs Baseline (Neo4j)" comparison in output
4. Exit with code 2 for competitive regression (vs code 1 for internal regression)

### Exit Code Semantics

Existing behavior preserved, extended with new competitive regression code:

| Exit Code | Meaning | When |
|-----------|---------|------|
| 0 | No regression detected | Within threshold (5% internal, 10% competitive) |
| 1 | Internal regression detected | >5% increase in P99 or >5% decrease in throughput (internal scenarios) |
| 2 | Competitive regression detected | >10% worse than before, or positioning degraded vs competitor baseline (competitive scenarios) |
| 3 | Error (missing files or invalid data) | Script execution failure |

**Rationale for 10% competitive threshold**:
- Internal 5% threshold catches micro-regressions during incremental development
- Competitive 10% threshold focuses on macro-level positioning changes
- Avoids false positives from measurement noise in longer competitive scenarios
- Still strict enough to prevent meaningful competitive degradation

## Regression Detection Strategy

### Two-Level Detection

**Level 1: Internal Delta** (before vs after)
- Threshold: 10% (double internal tolerance, accounts for competitive scenario variance)
- Metrics: P99 latency increase, throughput decrease, error rate increase
- Action: Block task completion, trigger profiling workflow

**Level 2: Competitive Positioning** (after vs documented baseline)
- Threshold: Absolute positioning degradation (e.g., from "61% faster than Neo4j" to "50% faster")
- Metrics: P99 latency delta vs competitor baseline
- Action: Warn (not block), suggest optimization follow-up task

### Detection Logic

```bash
# Level 1: Check internal delta (10% threshold for competitive scenarios)
if [[ "$COMPETITIVE_MODE" == "1" ]]; then
    THRESHOLD=10.0
else
    THRESHOLD=5.0
fi

if compare_float "$p99_change" "$THRESHOLD"; then
    echo "REGRESSION: P99 latency increased by ${p99_change}% (threshold: +${THRESHOLD}%)"
    REGRESSION=1
fi

# Level 2: Check competitive positioning (only in competitive mode)
if [[ "$COMPETITIVE_MODE" == "1" ]]; then
    # Load Neo4j baseline from competitive_baselines.md
    NEO4J_BASELINE=27.96  # P99 latency in ms

    # Calculate Engram's advantage
    ADVANTAGE=$(awk "BEGIN {printf \"%.1f\", (($NEO4J_BASELINE - $AFTER_P99) / $NEO4J_BASELINE) * 100}")

    # Check if positioning degraded significantly
    if awk "BEGIN {exit !($ADVANTAGE < 50.0)}"; then
        echo "WARNING: Competitive positioning degraded (now only ${ADVANTAGE}% faster than Neo4j)"
        echo "         Previous baseline: 61% faster"
        echo "         Consider creating optimization follow-up task"
    fi
fi
```

### When to Block vs Warn

**Block task completion** (exit code 2):
- Internal delta exceeds 10% threshold
- Error rate increases by >5 percentage points
- Throughput drops below competitive baseline (e.g., slower than Neo4j)

**Warn but allow completion** (exit code 0, with warning message):
- Competitive positioning degraded but still faster than competitor
- Internal delta within 10% but lost ground relative to competitor
- New competitive weakness discovered (e.g., ANN search now slower than Qdrant)

## Workflow Integration

### Updated CLAUDE.md Instructions

Add new section to "How to do a Milestone 17 task" workflow:

```markdown
11. **Performance validation** - Run 60s load test AFTER changes:
    ```bash
    ./scripts/m17_performance_check.sh <task_number> after
    ./scripts/compare_m17_performance.sh <task_number>
    ```

    **If regression >5% detected:**
    - Profile with `cargo flamegraph --bin engram` to identify hot spots
    - Check `tmp/m17_performance/<task>_after_*_diag.txt` for diagnostics
    - Review `tmp/m17_performance/<task>_after_*_sys.txt` for system metrics
    - Fix performance issues before proceeding
    - Re-run after test until regression <5%

12. **Competitive validation** (OPTIONAL - only for tasks affecting core graph operations):

    **When to use competitive validation:**
    - Task modifies spreading activation algorithms
    - Task changes graph traversal or pathfinding logic
    - Task alters vector search or ANN index structures
    - Task touches memory consolidation or decay functions
    - Task modifies concurrency primitives (locks, atomics, SIMD)

    **Skip competitive validation for:**
    - API changes without algorithmic impact
    - Documentation or test-only changes
    - CLI flag additions or configuration changes
    - Logging, metrics, or monitoring improvements

    **Run competitive tests:**
    ```bash
    ./scripts/m17_performance_check.sh <task_number> before --competitive
    # (implement task as usual)
    ./scripts/m17_performance_check.sh <task_number> after --competitive
    ./scripts/compare_m17_performance.sh <task_number>
    ```

    **If competitive regression >10% detected:**
    - Compare flamegraphs: before vs after
    - Check if regression is algorithmic (O(n^2) instead of O(n)) or constant factor
    - If algorithmic: fix before completing task
    - If constant factor: document in commit message, create follow-up optimization task
    - Re-run after test until regression <10%
```

### When to Use Competitive Validation

**Use `--competitive` flag for these task types**:

1. **Spreading Activation Tasks** (M17: 007, 008, 010)
   - Direct competitor: Neo4j graph traversal
   - Scenario: `hybrid_production_100k.toml` (includes graph queries)

2. **Vector Search Tasks** (Future M18 optimizations)
   - Direct competitor: Qdrant ANN search
   - Scenario: `qdrant_ann_1m_768d.toml`

3. **Memory Consolidation Tasks** (M17: 004, 005, 006)
   - No direct competitor (unique to Engram)
   - Use competitive scenario to ensure no regression in hybrid workload

4. **Core Storage Tasks** (M17: 002, 003)
   - Baseline: Hybrid production workload
   - Ensure storage refactoring doesn't degrade end-to-end latency

**Skip `--competitive` flag for**:
- API compatibility changes (non-performance)
- Testing infrastructure improvements
- Documentation tasks
- Monitoring and metrics additions

### Performance Overhead

Competitive validation adds approximately 2 minutes per task (60s before + 60s after):
- Internal-only testing: 2 minutes total
- Internal + competitive testing: 4 minutes total
- Overhead justified for 20% of tasks (core graph operations)
- Majority of tasks (80%) continue with fast internal-only testing

## Reporting and Alerting Design

### Competitive Regression Alert Format

When competitive regression detected (exit code 2):

```
=== COMPETITIVE REGRESSION DETECTED ===

Task XXX Competitive Performance:
Before: P99 10.2ms, 490 QPS
After:  P99 11.5ms, 450 QPS
Delta:  +12.7% latency, -8.2% throughput (EXCEEDS 10% THRESHOLD)

Competitive Positioning:
vs Neo4j (baseline: 27.96ms)
  Before: 63.5% faster (10.2ms vs 27.96ms)
  After:  58.9% faster (11.5ms vs 27.96ms)
  Status: Still competitive but lost ground (WARNING)

Action Required:
1. Profile to identify hot spots:
   cargo flamegraph --bin engram -- start --port 7432 &
   ./target/release/loadtest run --scenario scenarios/competitive/hybrid_production_100k.toml --duration 60

2. Check diagnostics for anomalies:
   cat tmp/m17_performance/competitive_<task>_after_*_diag.txt

3. Review system metrics for resource contention:
   cat tmp/m17_performance/competitive_<task>_after_*_sys.txt

4. Common causes of competitive regression:
   - Lock contention (check flamegraph for spin time)
   - Cache misses (NUMA-awareness degraded)
   - Algorithmic complexity increase (O(n) -> O(n^2))
   - SIMD vectorization broken (check scalar fallback)

5. Fix issues and re-run:
   ./scripts/m17_performance_check.sh <task> after --competitive
   ./scripts/compare_m17_performance.sh <task>

Cannot proceed to task completion until regression <10%
```

### Normal Competitive Comparison Output

When no regression detected (exit code 0):

```
=== Performance Comparison: Task XXX (Competitive) ===
Before: competitive_XXX_before_20251108_140530.json
After:  competitive_XXX_after_20251108_142130.json

Metric               Before      After        Change
-------------------- ---------- ---------- ------------
P50 latency (ms)          8.234      8.456       +2.70%
P95 latency (ms)          9.821     10.123       +3.08%
P99 latency (ms)         10.234     10.567       +3.25%
Throughput (ops/s)        489.3      485.2       -0.84%
Errors                        0          0           +0
Error rate                  0.0%       0.0%       +0.0pp

Competitive Positioning:
vs Neo4j (baseline: 27.96ms P99 latency)
  Engram:  10.567ms P99
  Speedup: 62.2% faster
  Status:  BETTER (within 10% threshold)

No significant regressions detected (within 10% competitive threshold)

Summary for PERFORMANCE_LOG.md:
- Before: P50=8.234ms, P95=9.821ms, P99=10.234ms, 489.3 ops/s
- After:  P50=8.456ms, P95=10.123ms, P99=10.567ms, 485.2 ops/s
- Change: +3.25% P99 latency, -0.84% throughput
- Status: Competitive advantage maintained (62.2% faster than Neo4j)
```

### Integration with Performance Log

Extend `roadmap/milestone-17/PERFORMANCE_LOG.md` format to include competitive results:

```markdown
## Task 007: Fan Effect Spreading

**Status**: Complete

**Internal Performance** (scenarios/m17_baseline.toml):
- Before: P50=0.167ms, P95=0.31ms, P99=0.458ms, 999.88 ops/s
- After:  P50=0.171ms, P95=0.32ms, P99=0.472ms, 995.24 ops/s
- Change: +3.1% P99 latency, -0.5% throughput
- Status: Within 5% target

**Competitive Performance** (scenarios/competitive/hybrid_production_100k.toml):
- Before: P99=10.2ms, 490 QPS
- After:  P99=10.5ms, 487 QPS
- Change: +2.9% P99 latency, -0.6% throughput
- vs Neo4j: 62.4% faster (10.5ms vs 27.96ms)
- Status: Competitive advantage maintained, within 10% target
```

### Historical Tracking Approach

**Short-term tracking** (task-level):
- Each task creates before/after pairs in `tmp/m17_performance/`
- Comparison script generates summary for PERFORMANCE_LOG.md
- Developer copies summary into log manually after validation

**Long-term tracking** (milestone-level):
- At end of M17, run competitive suite: `./scripts/quarterly_competitive_review.sh`
- Update `docs/reference/competitive_baselines.md` with M17 final measurements
- Create M18 optimization tasks for any >20% worse-than-competitor scenarios

**Quarterly tracking** (production):
- First week of each quarter: Jan, Apr, Jul, Oct
- Owner: Performance engineering lead
- Workflow: Task 005 quarterly review integration
- Output: Updated baselines + trendline analysis (improving/degrading)

## File Paths

```
scripts/m17_performance_check.sh (modify existing - add --competitive flag)
scripts/compare_m17_performance.sh (modify existing - detect competitive mode)
CLAUDE.md (modify existing - add competitive validation section)
roadmap/milestone-17/PERFORMANCE_WORKFLOW.md (modify existing - document --competitive usage)
docs/reference/competitive_baselines.md (reference only - created in Task 002)
```

## Acceptance Criteria

1. `--competitive` flag works with both performance check scripts
2. Competitive regression detection triggers distinct exit code (2 not 1)
3. Comparison output includes both internal delta and competitive positioning
4. CLAUDE.md clearly explains when to use competitive validation (with examples)
5. Scripts pass shellcheck linting (zero warnings)
6. Backward compatibility: Scripts work without `--competitive` flag (existing behavior unchanged)
7. Default behavior unchanged: Existing M17 tasks can ignore competitive validation
8. Competitive mode uses 10% threshold (not 5%)

## Testing Approach

```bash
# Validate shell syntax
shellcheck scripts/m17_performance_check.sh
shellcheck scripts/compare_m17_performance.sh

# Test backward compatibility (no --competitive flag)
./scripts/m17_performance_check.sh 999 before
# Should run scenarios/m17_baseline.toml (verify in output)

./scripts/m17_performance_check.sh 999 after
./scripts/compare_m17_performance.sh 999
# Should use 5% threshold, exit code 0 or 1

# Test competitive flag
./scripts/m17_performance_check.sh 998 before --competitive
# Should run scenarios/competitive/hybrid_production_100k.toml
# Should create tmp/m17_performance/competitive_998_before_*.json

./scripts/m17_performance_check.sh 998 after --competitive
# Should run scenarios/competitive/hybrid_production_100k.toml
# Should create tmp/m17_performance/competitive_998_after_*.json

# Test competitive comparison
./scripts/compare_m17_performance.sh 998
# Should detect competitive mode (check for "vs Neo4j" in output)
# Should use 10% threshold
# Should show competitive positioning

# Test regression detection (manual: artificially degrade performance)
# Option 1: Add 15ms sleep in graph traversal to trigger >10% regression
# Option 2: Modify competitive scenario to increase request rate
./scripts/m17_performance_check.sh 997 before --competitive
# (add sleep or increase load)
./scripts/m17_performance_check.sh 997 after --competitive
./scripts/compare_m17_performance.sh 997
echo "Exit code: $?"
# Should be 2 (competitive regression), not 1 (internal regression)
# Should show "COMPETITIVE REGRESSION DETECTED" alert

# Test file naming conventions
ls tmp/m17_performance/competitive_*
# Should show files with "competitive_" prefix

ls tmp/m17_performance/ | grep -v competitive
# Should show files without "competitive_" prefix (backward compatibility)

# Verify performance overhead (should be ~60s per test)
time ./scripts/m17_performance_check.sh 996 before --competitive
# Should complete in ~70-80s (60s test + 10-20s overhead)
```

## Integration Points

- Extends Task 003 benchmark infrastructure (uses same loadtest binary)
- Uses Task 002 baseline documentation (reads competitive_baselines.md for positioning)
- Integrates with existing M17 workflow (PERFORMANCE_WORKFLOW.md)
- Updates CLAUDE.md task execution instructions (adds optional step 12)
- Prepares for Task 006 initial baseline measurement (validates regression prevention before first baseline)

## Implementation Notes

### Script Modification Strategy

**Phase 1: m17_performance_check.sh**
1. Parse third positional argument for `--competitive` flag
2. Set SCENARIO and PREFIX variables based on flag
3. Modify RESULT_FILE, SYS_FILE, DIAG_FILE to include PREFIX
4. Update echo statements to indicate competitive mode
5. Test backward compatibility (no flag should work unchanged)

**Phase 2: compare_m17_performance.sh**
1. Detect competitive mode from BEFORE_FILE filename (check for "competitive_")
2. Load Neo4j baseline from hardcoded constant (27.96ms) or competitive_baselines.md
3. Calculate competitive positioning (percentage faster/slower)
4. Adjust regression threshold from 5% to 10% in competitive mode
5. Update output format to include "vs Neo4j" comparison
6. Change exit code to 2 for competitive regression
7. Test exit code semantics (0, 1, 2, 3)

**Phase 3: Documentation**
1. Update CLAUDE.md with new optional step 12 (competitive validation)
2. Add "When to use" and "When to skip" guidance
3. Update PERFORMANCE_WORKFLOW.md with competitive section
4. Add examples to both documents
5. Validate markdown formatting

### Baseline Loading Strategy

Hardcode Neo4j baseline initially for simplicity:
```bash
NEO4J_BASELINE_P99=27.96  # ms, from competitive_baselines.md
```

Future enhancement (optional): Parse from competitive_baselines.md dynamically:
```bash
NEO4J_BASELINE_P99=$(grep "Neo4j.*1-hop traversal" docs/reference/competitive_baselines.md | awk '{print $5}' | tr -d 'ms')
```

Trade-off: Hardcoding is simpler and faster, dynamic parsing is more maintainable.
Recommendation: Start with hardcoded, refactor to dynamic in M18 if baselines change frequently.

### Error Handling

Add validation for competitive scenario file existence:
```bash
if [[ "$COMPETITIVE_MODE" == "1" && ! -f "$SCENARIO" ]]; then
    echo "ERROR: Competitive scenario not found: $SCENARIO"
    echo "Run Task 001 to create competitive scenarios first"
    exit 3
fi
```

Add validation for baseline documentation:
```bash
if [[ "$COMPETITIVE_MODE" == "1" && ! -f "docs/reference/competitive_baselines.md" ]]; then
    echo "WARNING: Competitive baselines documentation not found"
    echo "Using hardcoded Neo4j baseline: ${NEO4J_BASELINE_P99}ms"
fi
```

## Success Metrics

1. **Backward Compatibility**: All existing M17 tasks continue to work without modification
2. **Adoption Rate**: 20% of M17 tasks use `--competitive` flag (4 out of 15 tasks)
3. **False Positive Rate**: <10% of competitive tests trigger spurious regressions
4. **Detection Rate**: 100% of competitive regressions caught before merge
5. **Performance Overhead**: <5 minutes added to competitive-validated tasks
