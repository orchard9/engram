# Task 008: Documentation and Acceptance Testing

**Status**: Pending
**Complexity**: Moderate
**Dependencies**: Task 007 (requires all prior tasks)
**Estimated Effort**: 6 hours

## Objective

Validate end-to-end competitive comparison workflow through comprehensive acceptance testing, update reference documentation, and define rigorous completion criteria for M17.1 with clear M18 transition plan.

## Specifications

### 1. End-to-End Validation (System-Level Testing)

Run comprehensive integration tests across all 8 tasks:

**Full Workflow Execution**:
- Run quarterly review workflow: `./scripts/quarterly_competitive_review.sh`
- Verify all 4 scenarios complete successfully within 15 minutes
- Verify report generation with correct comparisons and trend analysis
- Verify baseline documentation automatically updated
- Verify no performance regression vs M17 baseline (run diagnostics)

**Cross-Task Integration Validation**:
- Task 001 scenarios → Task 003 runner: All TOML files discovered and executed
- Task 003 runner → Task 004 generator: Output format consumed correctly
- Task 004 generator → Task 005 workflow: Report integrated into automation
- Task 005 workflow → Task 007 validation: Regression checks triggered
- Task 002 docs ← Task 006 measurements: Baselines documented accurately

**Output Artifact Validation**:
- All output files timestamped and versioned correctly
- Metadata captures complete system context (git hash, system specs, versions)
- Reports are markdown-formatted and render correctly
- Baseline documentation follows Diataxis framework
- No broken links or missing file references

### 2. Comprehensive Acceptance Testing Scenarios

#### Scenario 1: Fresh Deployment (Cold Start)
**Objective**: Verify framework works from scratch on clean system

**Preconditions**:
- Delete `tmp/competitive_benchmarks/` entirely
- Delete `docs/reference/competitive_baselines.md` if exists
- Clean build: `cargo clean && cargo build --release`

**Execution**:
```bash
rm -rf tmp/competitive_benchmarks/
rm -f docs/reference/competitive_baselines.md
cargo clean && cargo build --release
./scripts/quarterly_competitive_review.sh
```

**Expected Outcomes**:
- Report generated without prior data (no trend analysis section)
- Baseline documentation created with initial measurements
- All 4 scenarios complete successfully
- P99 latencies within expected bounds:
  - qdrant_ann_1m_768d: <100ms
  - neo4j_traversal_100k: <50ms
  - hybrid_production_100k: <75ms
  - milvus_ann_10m_768d: <200ms or graceful failure on low-memory systems
- Exit code 0 (success)

**Validation**:
- `ls tmp/competitive_benchmarks/ | grep report` shows exactly 1 report
- `grep "Initial Baseline" tmp/competitive_benchmarks/*_report.md` confirms initial measurement
- `wc -l docs/reference/competitive_baselines.md` shows >100 lines (comprehensive doc)
- No error messages in stderr logs

**Failure Modes to Test**:
- Insufficient disk space (<5GB): Should fail with clear error before execution
- Missing loadtest binary: Should fail pre-flight check with actionable message
- Interrupted during first scenario: Should clean up and be re-runnable

---

#### Scenario 2: Trend Analysis (Quarterly Cadence)
**Objective**: Verify longitudinal tracking and comparison across time periods

**Preconditions**:
- Run scenario 1 first to establish baseline
- Wait 60s to create temporal separation (simulates quarterly gap)
- System in identical state (no code changes)

**Execution**:
```bash
# First run (baseline)
./scripts/quarterly_competitive_review.sh
FIRST=$(ls -t tmp/competitive_benchmarks/*_metadata.txt | head -1 | sed 's/_metadata.txt//')

# Wait to simulate time passing
sleep 60

# Second run (comparison)
./scripts/quarterly_competitive_review.sh
SECOND=$(ls -t tmp/competitive_benchmarks/*_metadata.txt | head -1 | sed 's/_metadata.txt//')

# Generate comparison report
python3 scripts/generate_competitive_report.py \
  --input "$SECOND" \
  --compare-to "$FIRST" \
  --output tmp/competitive_benchmarks/trend_analysis_report.md
```

**Expected Outcomes**:
- Trend analysis section shows minimal delta (<1% for all metrics)
- Report includes both absolute values and percentage changes
- Statistical significance correctly indicated (expected: no significant change)
- Visualization of trends included (text-based sparklines or tables)

**Validation**:
```bash
grep -E "\+[0-9]+\.[0-9]+%" tmp/competitive_benchmarks/trend_analysis_report.md
# Should show deltas like +0.3%, -0.5% (sub-1% variance)

grep "Statistically significant" tmp/competitive_benchmarks/trend_analysis_report.md
# Should return "No statistically significant changes detected" or empty

grep "Trend:" tmp/competitive_benchmarks/trend_analysis_report.md | wc -l
# Should equal number of metrics tracked (at least 12: P50/P95/P99/throughput × 3 scenarios)
```

**Failure Modes to Test**:
- Corrupted first baseline: Should gracefully handle missing comparison data
- Incompatible schema versions: Should warn about non-comparable data
- Clock skew (second run timestamp < first): Should detect and error

---

#### Scenario 3: Regression Detection (Performance Degradation)
**Objective**: Verify automated detection of performance regressions

**Preconditions**:
- Establish clean baseline with scenario 1
- Artificially degrade performance (controlled regression injection)

**Regression Injection**:
```rust
// In engram-core/src/store.rs, temporarily add to store() method:
std::thread::sleep(std::time::Duration::from_millis(10));
```

**Execution**:
```bash
# Establish baseline
./scripts/m17_performance_check.sh 171_competitive before --competitive

# Apply degradation (manually edit store.rs as above)
# Rebuild
cargo build --release

# Run competitive validation
./scripts/m17_performance_check.sh 171_competitive after --competitive
./scripts/compare_m17_performance.sh 171_competitive --competitive
EXIT_CODE=$?
```

**Expected Outcomes**:
- Script detects >5% regression in store operations
- Exit code 2 (regression detected, not failure)
- Warning message printed to stderr:
  ```
  WARNING: Performance regression detected in competitive baseline
  Store operation P99: 12.3ms (baseline: 2.1ms) - 485% slower
  Review tmp/m17_performance/171_competitive_after_*_diag.txt for analysis
  ```
- Detailed diagnostics saved with flame graph (if perf available)

**Validation**:
```bash
echo $EXIT_CODE
# Should be 2 (not 0 or 1)

grep -i "regression" tmp/m17_performance/171_competitive_comparison.txt
# Should contain regression warning

ls tmp/m17_performance/171_competitive_after_*_flamegraph.svg
# Should exist if perf/flamegraph available
```

**Rollback and Revalidation**:
```bash
# Remove degradation, rebuild
git checkout engram-core/src/store.rs
cargo build --release

# Re-run validation
./scripts/m17_performance_check.sh 171_competitive revalidation --competitive
./scripts/compare_m17_performance.sh 171_competitive --competitive
# Should show <1% delta, exit code 0
```

**Failure Modes to Test**:
- Catastrophic regression (>100x slower): Should abort test early with timeout
- Baseline missing: Should error with clear message about running 'before' first
- Inconsistent environment (CPU throttled): Should warn about unstable conditions

---

#### Scenario 4: Cross-Platform Determinism
**Objective**: Verify scenarios produce identical operation sequences on different platforms

**Preconditions**:
- Access to both macOS and Linux systems (or use Docker for Linux)
- Same Engram commit hash on both platforms
- Same scenario file (qdrant_ann_1m_768d.toml) with fixed seed

**Execution (macOS)**:
```bash
cargo build --release
./scripts/competitive_benchmark_suite.sh --scenario qdrant_ann_1m_768d
mv tmp/competitive_benchmarks/latest_qdrant_ann_1m_768d.txt /tmp/macos_results.txt
```

**Execution (Linux via Docker)**:
```bash
docker run --rm -v $(pwd):/workspace -w /workspace rust:latest bash -c "
  cargo build --release
  ./scripts/competitive_benchmark_suite.sh --scenario qdrant_ann_1m_768d
  cat tmp/competitive_benchmarks/latest_qdrant_ann_1m_768d.txt
" > /tmp/linux_results.txt
```

**Validation**:
```bash
# Extract operation sequences (deterministic)
grep "Operation:" /tmp/macos_results.txt | cut -d' ' -f2- | sha256sum > /tmp/macos_ops.sha
grep "Operation:" /tmp/linux_results.txt | cut -d' ' -f2- | sha256sum > /tmp/linux_ops.sha

diff /tmp/macos_ops.sha /tmp/linux_ops.sha
# Should be empty (identical checksums)

# Latency variance expected (architecture differences)
MACOS_P99=$(grep "P99:" /tmp/macos_results.txt | awk '{print $2}')
LINUX_P99=$(grep "P99:" /tmp/linux_results.txt | awk '{print $2}')
echo "P99 latency variance: $MACOS_P99 (macOS) vs $LINUX_P99 (Linux)"
# Variance <20% acceptable due to CPU architecture differences
```

**Expected Outcomes**:
- Operation sequences bitwise-identical (SHA256 match)
- Latency distributions similar but not identical (architecture-dependent)
- Both runs complete successfully without errors
- Memory footprints within 10% of each other

---

#### Scenario 5: Concurrent Execution (Isolation Validation)
**Objective**: Verify scenarios don't interfere when run in parallel

**Preconditions**:
- Modified benchmark suite to support `--parallel` flag
- Sufficient system resources (32GB+ RAM for parallel 1M scenarios)

**Execution**:
```bash
# Sequential baseline
time ./scripts/competitive_benchmark_suite.sh
SEQUENTIAL_DURATION=$?

# Parallel execution
time ./scripts/competitive_benchmark_suite.sh --parallel
PARALLEL_DURATION=$?
```

**Expected Outcomes**:
- Parallel execution faster than sequential (at least 1.5x speedup on 4-core system)
- No resource contention errors (OOM, port conflicts, file locking)
- Results statistically similar to sequential (P99 within 10%)
- No cross-scenario data corruption

**Validation**:
```bash
# Compare parallel vs sequential results
diff -u tmp/competitive_benchmarks/sequential_summary.txt \
        tmp/competitive_benchmarks/parallel_summary.txt
# P99 latencies should be within 10%

# Verify no partial results or corrupted files
find tmp/competitive_benchmarks/ -name "*.partial" -o -name "*.tmp"
# Should be empty (all atomic writes completed)
```

---

#### Scenario 6: Failure Recovery (Resilience Testing)
**Objective**: Verify graceful handling of various failure modes

**Test Cases**:

**6a. Server Crash During Benchmark**:
```bash
# Start benchmark in background
./scripts/competitive_benchmark_suite.sh &
SUITE_PID=$!

# Kill server mid-test (after 30s)
sleep 30
pkill -9 engram

# Wait for suite to complete
wait $SUITE_PID
EXIT_CODE=$?

# Verify graceful handling
echo $EXIT_CODE  # Should be 1 (failure, not crash)
grep "Server crash detected" tmp/competitive_benchmarks/latest_summary.txt
# Should log crash and continue to next scenario
```

**6b. Disk Full During Execution**:
```bash
# Create small disk image (100MB) and run benchmark there
hdiutil create -size 100m -fs HFS+ -volname TestDisk /tmp/testdisk.dmg
hdiutil attach /tmp/testdisk.dmg
ln -sf /Volumes/TestDisk tmp/competitive_benchmarks

./scripts/competitive_benchmark_suite.sh
# Expected: Graceful failure with "Insufficient disk space" error
```

**6c. Network Interruption**:
```bash
# Block localhost connectivity mid-test
sudo ifconfig lo0 down &
sleep 2
./scripts/competitive_benchmark_suite.sh
EXIT_CODE=$?

sudo ifconfig lo0 up
# Expected: Clear error about localhost unreachable, exit code 1
```

**Expected Outcomes**:
- All failure modes handled gracefully (no silent corruption)
- Clear error messages identifying root cause
- Partial results preserved where possible
- Clean exit codes (0=success, 1=failure, 2=warning, 130=interrupted)

---

#### Scenario 7: Documentation Completeness (Human Validation)
**Objective**: Verify another engineer can use the framework without assistance

**Test Protocol**:
1. Recruit engineer unfamiliar with M17.1 competitive framework
2. Provide only: `docs/reference/competitive_baselines.md` and `README.md`
3. Ask them to: "Run a competitive baseline and interpret the results"
4. Observe: Do they succeed without asking questions?

**Success Criteria**:
- Engineer completes task in <30 minutes
- Zero clarifying questions about how to run tools
- Correctly interprets P99 latency and trend data
- Identifies which scenarios Engram wins/loses vs competitors

**Documentation Validation Checklist**:
- [ ] All commands copy-pasteable and run successfully
- [ ] Prerequisites explicitly stated (required tools, system specs)
- [ ] Expected outputs shown with examples
- [ ] Troubleshooting section covers common errors
- [ ] Links to source code/config files all working
- [ ] Terminology defined (P99, QPS, recall, etc.)
- [ ] Comparison to competitors clearly explained

**Automated Validation**:
```bash
# Check all markdown links valid
npx markdown-link-check docs/reference/competitive_baselines.md
# Should report 0 broken links

# Check all code blocks syntactically valid
grep -A 20 '```bash' docs/reference/competitive_baselines.md | \
  bash -n
# Should have no syntax errors

# Check all referenced files exist
grep -o 'scenarios/competitive/[^ ]*\.toml' docs/reference/competitive_baselines.md | \
  while read f; do [ -f "$f" ] || echo "Missing: $f"; done
# Should produce no output
```

---

### 3. Documentation Updates

**docs/reference/competitive_baselines.md** (create new):
```markdown
# Competitive Performance Baselines

## Overview

This document tracks Engram's performance against best-in-class specialized systems
across vector search, graph traversal, and hybrid workloads. Updated quarterly.

## Competitor Baselines

| System | Workload | P99 Latency | Throughput | Recall | Dataset | Source | Last Updated |
|--------|----------|-------------|------------|--------|---------|--------|--------------|
| Qdrant | ANN search | 22-24ms | 626 QPS | 99.5% | 1M 768d | [Qdrant Benchmarks](https://qdrant.tech/benchmarks/) | 2025-Q1 |
| Milvus | ANN search | 708ms | 2,098 QPS | 100% | 10M | [Milvus Docs](https://milvus.io/docs/benchmark.md) | 2025-Q1 |
| Neo4j | 1-hop traversal | 27.96ms | 280 QPS | N/A | 100K nodes | [Neo4j Performance](https://neo4j.com/developer/graph-data-science/performance/) | 2025-Q1 |
| Weaviate | ANN search | 70-150ms | 200-400 QPS | N/A | 1M | [Weaviate Blog](https://weaviate.io/blog) | 2024-Q4 |
| Redis | Vector search | 8ms | N/A | N/A | Unknown | [Redis Docs](https://redis.io/docs/stack/search/reference/vectors/) | 2024-Q4 |

## Engram Baselines (Current)

*Populated after Task 006 initial measurement*

| Workload | P99 Latency | Throughput | Recall | Notes |
|----------|-------------|------------|--------|-------|
| ANN search (1M 768d) | TBD | TBD | TBD | vs Qdrant |
| Graph traversal (100K) | TBD | TBD | N/A | vs Neo4j |
| Hybrid production | TBD | TBD | N/A | Unique to Engram |

## Scenario Mapping

| Competitor | Scenario File | Description |
|------------|---------------|-------------|
| Qdrant | `scenarios/competitive/qdrant_ann_1m_768d.toml` | 1M vectors, pure search, 99.5% recall target |
| Neo4j | `scenarios/competitive/neo4j_traversal_100k.toml` | 100K nodes, 80% 1-hop + 20% 2-hop traversal |
| Milvus | `scenarios/competitive/milvus_ann_10m_768d.toml` | 10M vectors, pure search, 100% recall |
| Engram-specific | `scenarios/competitive/hybrid_production_100k.toml` | Mixed store/recall/search/pattern completion |

## Performance Targets

### ANN Search (1M vectors, 768d)
- **Target**: P99 < 20ms, throughput > 650 QPS, recall > 99.5%
- **Competitor**: Qdrant (22-24ms, 626 QPS, 99.5%)
- **Goal**: Match or exceed Qdrant on pure vector workload

### Graph Traversal (100K nodes)
- **Target**: P99 < 15ms, throughput > 300 QPS
- **Competitor**: Neo4j (27.96ms, 280 QPS)
- **Goal**: 2x faster than Neo4j on graph operations

### Hybrid Workload (100K nodes)
- **Target**: P99 < 10ms, throughput > 500 QPS
- **Competitor**: None (unique capability)
- **Goal**: Demonstrate hybrid operations without performance degradation

## Competitive Positioning

### Strengths
- **Unified Architecture**: Single system for vector + graph + temporal operations
- **Cognitive Realism**: Biologically-inspired spreading activation and decay
- **Probabilistic Reasoning**: Confidence intervals and pattern completion
- **API Simplicity**: No need to orchestrate multiple databases

### Current Limitations
- **Pure Vector Search**: Expected to lag specialized vector DBs by 10-20% initially
- **Large-Scale Data**: 10M+ vectors may require optimization (M18 scope)
- **Write Throughput**: Consolidation overhead may slow batch ingestion

### Differentiation
Engram is the only system supporting:
1. Spreading activation with temporal decay in a single query
2. Pattern completion using hippocampal-neocortical dynamics
3. Memory consolidation integrated with vector search
4. Probabilistic confidence propagation across hybrid operations

## Measurement Methodology

### Hardware Specification
- **Baseline**: M1 Max (10-core CPU, 32GB RAM) or AWS c6i.4xlarge
- **Storage**: NVMe SSD with >1000 MB/s sequential read
- **Network**: Localhost (eliminates network latency)

### Software Configuration
- **Engram Version**: Git commit hash from baseline measurement
- **Rust Version**: 1.75.0 or later
- **Build Mode**: `cargo build --release` with LTO enabled
- **Load Test**: Single-threaded client (eliminates client-side bottlenecks)

### Test Parameters
- **Duration**: 60 seconds per scenario
- **Warm-up**: 10 seconds pre-test (stabilize caches)
- **Seed**: Fixed deterministic seed for reproducibility
- **Isolation**: No concurrent workloads, CPU governor set to performance mode

### Metrics Collection
- **Latency**: HdrHistogram with 3 significant figures (P50, P95, P99)
- **Throughput**: Operations per second (QPS)
- **Recall**: For ANN scenarios only (percentage of true neighbors retrieved)
- **Resource Usage**: Peak memory (RSS), CPU utilization, I/O wait

## Quarterly Review Process

### Schedule
- **Cadence**: First week of each quarter (January, April, July, October)
- **Owner**: Performance engineering lead
- **Duration**: <3 hours including analysis and documentation

### Workflow
1. **Preparation** (30 min):
   - Ensure clean git state (no uncommitted changes)
   - Verify system resources (16GB+ RAM, <50% CPU baseline load)
   - Run pre-flight checks: `./scripts/competitive_benchmark_suite.sh --preflight-only`

2. **Execution** (60 min):
   - Run full suite: `./scripts/quarterly_competitive_review.sh`
   - Monitor progress and capture any anomalies
   - Verify all 4 scenarios complete successfully

3. **Analysis** (60 min):
   - Generate comparison report vs previous quarter
   - Identify regressions or improvements >5%
   - Investigate root causes of significant changes
   - Update baseline documentation with new measurements

4. **Documentation** (30 min):
   - Update this document with latest baselines
   - Commit results: `git add docs/reference/ tmp/competitive_benchmarks/`
   - Create quarterly summary: `docs/reference/quarterly_reviews/YYYY-QN.md`

### Deliverables
- Updated `competitive_baselines.md` with current measurements
- Quarterly comparison report in `tmp/competitive_benchmarks/`
- Actionable recommendations for M18+ optimization tasks

## Troubleshooting

### Common Issues

**Error: "Loadtest binary not found"**
- Solution: Run `cargo build --release` to build all binaries

**Error: "Insufficient memory (8GB available, 16GB required)"**
- Solution: Close other applications or use smaller scenario (neo4j_traversal_100k)

**Warning: "P99 latency variance >5% across runs"**
- Cause: System instability (background processes, thermal throttling)
- Solution: Ensure idle system, check CPU frequency scaling, rerun test

**Error: "Server startup timeout"**
- Solution: Check `tmp/competitive_benchmarks/*_stderr.txt` for server crash logs

### Performance Debugging

If Engram significantly underperforms vs competitors:

1. **Profile hot paths**: `cargo flamegraph --bin loadtest -- run --scenario <file>`
2. **Check diagnostics**: Review `*_diag.txt` for memory leaks or lock contention
3. **Compare to M17 baseline**: Run `./scripts/compare_m17_performance.sh` to isolate regression
4. **Create M18 task**: Document findings in `roadmap/milestone-18/NNN_optimize_<component>.md`

## References

- Scenario definitions: `scenarios/competitive/README.md`
- Benchmark runner: `scripts/competitive_benchmark_suite.sh`
- Report generator: `scripts/generate_competitive_report.py`
- M17 performance framework: `roadmap/milestone-17/PERFORMANCE_WORKFLOW.md`
```

**vision.md** (add competitive positioning section):

Add after "Success Metrics" section:

```markdown
## Competitive Positioning (as of 2025-Q1)

Engram delivers hybrid vector-graph-temporal operations in a unified architecture that outperforms
specialized systems on integrated workloads while maintaining competitive performance on pure operations.

**Performance vs Specialized Systems**:
- Graph traversal: TBD% faster than Neo4j (P99 latency: TBD vs 27.96ms)
- Vector search: TBD% of Qdrant's performance (P99 latency: TBD vs 22-24ms)
- Hybrid workload: No direct competitor (unique capability)

**Key Differentiators**:
1. Only system supporting spreading activation + temporal decay in unified queries
2. Integrated pattern completion using hippocampal-neocortical dynamics
3. Probabilistic confidence propagation across vector, graph, and temporal operations
4. Memory consolidation as first-class operation (not batch ETL)

**Target Markets**:
- Cognitive AI agents requiring human-like memory dynamics
- RAG systems needing temporal context and spreading activation
- Knowledge graphs requiring vector similarity alongside relational queries
- Research platforms studying biologically-plausible memory systems

For detailed competitive baselines and quarterly trends, see `docs/reference/competitive_baselines.md`.
```

**docs/reference/README.md** (create if missing, or add section):

```markdown
# Engram Reference Documentation

## Performance and Benchmarking

- [Competitive Baselines](./competitive_baselines.md) - Comparison vs Qdrant, Neo4j, Milvus
- [Performance Baselines](./performance-baselines.md) - Engram single-node benchmarks
- [Resource Requirements](./resource-requirements.md) - Hardware sizing guide
- [System Requirements](./system-requirements.md) - Minimum and recommended specs

## API Documentation

- [REST API](./rest-api.md) - HTTP endpoints and examples
- [gRPC API](./grpc-api.md) - Protocol Buffers definitions
- [Query Language](./query-language.md) - RECALL/SPREAD/CONSOLIDATE syntax
- [Spreading Activation API](./spreading_api.md) - Advanced spreading parameters

## Configuration

- [Configuration Reference](./configuration.md) - All TOML settings documented
- [GPU Architecture](./gpu-architecture.md) - CUDA/ROCm configuration
- [Error Codes](./error-codes.md) - Complete error catalog

## Cognitive Patterns

- [Cognitive Patterns](./cognitive_patterns.md) - Psychological validation studies
- [API Versioning](./api-versioning.md) - Breaking change policy
```

---

### 4. M17.1 Completion Checklist

Create `roadmap/milestone-17.1/M17.1_COMPLETION_CHECKLIST.md`:

```markdown
# Milestone 17.1 Completion Checklist

## Task Completion Status

- [ ] Task 001: Competitive Scenario Suite (TOML definitions)
  - [ ] All 4 TOML files parse and validate
  - [ ] Determinism tests pass (3 runs produce <1% variance)
  - [ ] README with competitor citations complete
  - [ ] Marked `_complete`

- [ ] Task 002: Competitive Baseline Documentation
  - [ ] `competitive_baselines.md` created with all sections
  - [ ] Competitor baselines table populated with sources
  - [ ] Scenario mapping 1:1 with Task 001 files
  - [ ] Markdown linting passes (zero errors)
  - [ ] Marked `_complete`

- [ ] Task 003: Competitive Benchmark Suite Runner
  - [ ] `competitive_benchmark_suite.sh` passes shellcheck
  - [ ] Pre-flight checks comprehensive (binary, memory, disk, CPU)
  - [ ] All 4 scenarios execute successfully in <10 minutes
  - [ ] Error handling tested (missing binary, SIGINT, server crash)
  - [ ] Marked `_complete`

- [ ] Task 004: Competitive Comparison Report Generator
  - [ ] `generate_competitive_report.py` implements all sections
  - [ ] Report format validated (markdown, renders in GitHub)
  - [ ] Trend analysis calculates percentage deltas correctly
  - [ ] Statistical significance testing implemented
  - [ ] Marked `_complete`

- [ ] Task 005: Quarterly Review Workflow Integration
  - [ ] `quarterly_competitive_review.sh` orchestrates end-to-end
  - [ ] Baseline documentation auto-updated from measurements
  - [ ] Full workflow completes in <15 minutes
  - [ ] Output artifacts archived with timestamps
  - [ ] Marked `_complete`

- [ ] Task 006: Initial Baseline Measurement
  - [ ] All 4 scenarios measured on reference hardware
  - [ ] Results documented in `competitive_baselines.md`
  - [ ] Comparison to competitors analyzed
  - [ ] Performance targets met or documented as M18 tasks
  - [ ] Marked `_complete`

- [ ] Task 007: Performance Regression Prevention
  - [ ] M17 performance scripts extended with `--competitive` flag
  - [ ] Regression detection working (exit code 2 on >5% degradation)
  - [ ] Integration with quarterly workflow validated
  - [ ] Documentation updated with regression workflow
  - [ ] Marked `_complete`

- [ ] Task 008: Documentation and Acceptance Testing
  - [ ] All 7 acceptance scenarios pass (fresh deployment, trend, regression, cross-platform, concurrent, failure recovery, human validation)
  - [ ] Documentation updates complete (vision.md, docs/reference/)
  - [ ] Completion checklist (this file) validated
  - [ ] M18 recommendations documented
  - [ ] Marked `_complete`

## Functional Completeness

### Scenario Execution
- [ ] `qdrant_ann_1m_768d` completes in <90s with P99 <100ms
- [ ] `neo4j_traversal_100k` completes in <90s with P99 <50ms
- [ ] `hybrid_production_100k` completes in <90s with P99 <75ms
- [ ] `milvus_ann_10m_768d` completes or fails gracefully based on RAM

### Report Generation
- [ ] Initial baseline report generated without comparison data
- [ ] Quarterly comparison report shows trend analysis
- [ ] Regression report highlights >5% degradations
- [ ] All reports render correctly in markdown viewers

### Documentation
- [ ] `competitive_baselines.md` includes all competitor sources
- [ ] Vision statement updated with competitive positioning
- [ ] API examples include competitive use cases
- [ ] Troubleshooting guide covers common errors

### Automation
- [ ] Quarterly workflow runs without manual intervention
- [ ] Pre-flight checks prevent invalid test execution
- [ ] Error handling prevents cascading failures
- [ ] Output artifacts versioned and archivable

## Quality Gates

### Code Quality
- [ ] `make quality` passes (zero clippy warnings)
- [ ] All bash scripts pass `shellcheck` (zero warnings)
- [ ] Python scripts type-checked with mypy (if used)
- [ ] No TODO/FIXME comments in production scripts

### Testing Coverage
- [ ] All 7 acceptance scenarios documented and tested
- [ ] Failure modes tested (missing binary, OOM, server crash)
- [ ] Cross-platform validation (macOS + Linux)
- [ ] Determinism verified (3 consecutive runs <1% variance)

### Performance Validation
- [ ] No regression vs M17 baseline (<5% on existing workloads)
- [ ] Competitive scenarios complete within time budget (15min total)
- [ ] Memory footprints within expected bounds (see Task 001)
- [ ] Diagnostics show no resource leaks or contention

### Documentation Quality
- [ ] All markdown files lint-clean (npx markdownlint-cli2)
- [ ] All links validated (npx markdown-link-check)
- [ ] Code blocks syntactically correct (bash -n)
- [ ] Another engineer can follow docs without assistance

## Regression Prevention

### M17 Performance Checks
- [ ] Run baseline comparison: `./scripts/compare_m17_performance.sh 001`
- [ ] Verify <5% regression on all M17 tasks
- [ ] Check diagnostics: `./scripts/engram_diagnostics.sh`
- [ ] Review `tmp/engram_diagnostics.log` for new warnings

### Integration Tests
- [ ] Existing integration tests pass: `cargo test --release integration_tests`
- [ ] No new flaky tests introduced
- [ ] Test duration unchanged (<2 minutes)

### System Integration
- [ ] Quarterly workflow doesn't conflict with M17 workflows
- [ ] Loadtest binary compatible with existing scenarios
- [ ] No breaking changes to engram-core public API

## Final Validation

### End-to-End Smoke Test
```bash
# Clean slate
rm -rf tmp/competitive_benchmarks/
cargo clean && cargo build --release

# Full workflow
time ./scripts/quarterly_competitive_review.sh

# Validation
ls tmp/competitive_benchmarks/ | wc -l
# Expected: >20 files (4 scenarios × 5 files + metadata + summary)

grep "All scenarios passed" tmp/competitive_benchmarks/latest_summary.txt
# Expected: Match found

echo $?
# Expected: 0 (success)
```

### Human Acceptance
- [ ] Performance lead reviews and approves baselines
- [ ] Tech lead reviews code quality (shellcheck, structure, error handling)
- [ ] Documentation lead confirms docs are production-ready
- [ ] Another engineer successfully runs workflow from docs alone

## Sign-Off

- [ ] **Technical Completion**: All 8 tasks marked `_complete`
- [ ] **Functional Validation**: All acceptance scenarios pass
- [ ] **Quality Gates**: Clippy, shellcheck, linting all pass
- [ ] **Performance Validation**: No M17 regressions, baselines documented
- [ ] **Documentation**: Production-ready and validated by peer review
- [ ] **M18 Planning**: Recommendations documented for follow-up optimization

**Completion Date**: _____________

**Completed By**: _____________

**Reviewed By**: _____________

## Known Limitations (Document for M18)

*To be filled during Task 008 based on acceptance test findings*

Example format:
- **10M scenario OOM on 16GB systems**: Requires optimization or 32GB+ RAM requirement documented
- **Parallel execution not implemented**: Sequential only, 15min duration acceptable for quarterly cadence
- **Cross-platform variance**: macOS vs Linux shows 10-15% latency difference (architecture-dependent, acceptable)
- **Qdrant comparison gap**: If >20% slower on pure vector search, create M18 optimization task
```

---

### 5. M18 Optimization Recommendations

Create `roadmap/milestone-17.1/OPTIMIZATION_RECOMMENDATIONS.md`:

```markdown
# M18 Optimization Recommendations

*Generated from M17.1 Task 008 acceptance testing and initial baseline measurements*

## Overview

This document translates competitive baseline findings into actionable M18 tasks for optimization.
Prioritization based on: (1) competitive gap severity, (2) user-facing impact, (3) implementation cost.

## Methodology

### Competitive Gap Analysis

For each scenario, calculate competitive gap:
```
Gap = (Engram_P99 - Competitor_P99) / Competitor_P99 * 100
```

**Severity Classification**:
- **Critical** (Gap >50%): Immediate M18 priority, blocks adoption
- **High** (Gap 20-50%): M18 target, competitive disadvantage
- **Medium** (Gap 5-20%): M19+ optimization, acceptable for specialized system
- **Low** (Gap <5%): No action needed, within measurement variance
- **Advantage** (Gap <0): Publicize as competitive strength

### Prioritization Framework

**Priority Score** = (Gap Severity × User Impact × Implementation Feasibility)

| Factor | Weight | Scoring |
|--------|--------|---------|
| Gap Severity | 40% | Critical=4, High=3, Medium=2, Low=1 |
| User Impact | 40% | Core use case=4, Common=3, Niche=2, Rare=1 |
| Implementation Feasibility | 20% | Easy=4, Moderate=3, Hard=2, Research needed=1 |

**Thresholds**:
- Priority 1 (P1): Score ≥3.0 - Must-have for M18
- Priority 2 (P2): Score 2.0-2.9 - Should-have for M18
- Priority 3 (P3): Score 1.0-1.9 - Nice-to-have for M18 or defer to M19

## Findings Template

*Fill this section after Task 006 initial measurements*

### ANN Search Performance (Qdrant Comparison)

**Measured Performance**:
- Engram P99: ___ms
- Qdrant P99: 22-24ms
- Gap: ___%
- Severity: [Critical|High|Medium|Low|Advantage]

**Root Cause Analysis**:
*Conduct after baseline measurement - use flamegraphs and diagnostics*

Potential areas to investigate:
1. **HNSW Index Efficiency**:
   - Is Engram's HNSW implementation as optimized as Qdrant's?
   - Profile: `cargo flamegraph --bin loadtest -- run --scenario qdrant_ann_1m_768d`
   - Look for hot spots in: `engram-core/src/index/hnsw.rs`

2. **SIMD Utilization**:
   - Are vector distance calculations using AVX2/NEON?
   - Check: Zig kernels being invoked for cosine similarity?
   - Profile CPU instructions: `perf stat -e instructions,cycles,fp_arith_inst_retired.scalar_single`

3. **Memory Layout**:
   - Are embeddings cache-aligned for SIMD loads?
   - Check struct padding in `MemoryNode` (should be 64-byte aligned)
   - Profile cache misses: `perf stat -e cache-references,cache-misses`

4. **Lock Contention**:
   - Is DashMap causing contention on read-heavy workload?
   - Profile: Look for `pthread_mutex_lock` in flamegraph
   - Consider: RwLock or lock-free alternative for read-dominated workloads

**Proposed M18 Tasks**:

*Example template - fill based on actual findings*

**Task M18-001: Optimize HNSW Search Path** (P1 if gap >20%)
- **Objective**: Reduce P99 latency of HNSW search to <20ms (Qdrant parity)
- **Approach**:
  1. Profile hot paths in `hnsw.rs::search()`
  2. Implement prefetching for neighbor traversal
  3. Optimize distance calculation with dedicated SIMD kernel
  4. Benchmark iteratively until <20ms P99
- **Success Criteria**: P99 <20ms on qdrant_ann_1m_768d, no recall degradation
- **Estimated Effort**: 8 hours
- **Dependencies**: None

**Task M18-002: SIMD-Optimized Batch Distance Calculation** (P1 if gap >30%)
- **Objective**: Vectorize cosine similarity for batch queries
- **Approach**:
  1. Extend Zig kernels to support batch (N×768) × (M×768) distance matrix
  2. Add AVX-512 support for x86_64 (fallback to AVX2)
  3. Add NEON support for ARM (M1/M2 optimization)
  4. Integrate into HNSW search path
- **Success Criteria**: 2x throughput on batch queries, <15ms P99
- **Estimated Effort**: 12 hours
- **Dependencies**: Zig integration (already complete in M10)

---

### Graph Traversal Performance (Neo4j Comparison)

**Measured Performance**:
- Engram P99: ___ms
- Neo4j P99: 27.96ms
- Gap: ___%
- Severity: [Critical|High|Medium|Low|Advantage]

**Root Cause Analysis**:

Potential areas to investigate:
1. **Cache Locality**:
   - Are frequently-traversed edges cache-resident?
   - Profile: Check cache hit rate with `perf stat -e LLC-loads,LLC-load-misses`
   - Consider: Adjacency list packing for hot paths

2. **Edge Weight Lookup**:
   - Is edge confidence calculation adding overhead?
   - Profile: Time breakdown of `get_neighbors()` vs `compute_confidence()`
   - Consider: Precompute confidence for static graphs

3. **Spreading Activation Overhead**:
   - Is probabilistic activation slower than Neo4j's deterministic traversal?
   - Profile: Compare pure `get_neighbors()` vs full spreading activation
   - Consider: Fast path for deterministic traversal (confidence=1.0)

**Proposed M18 Tasks**:

*Example template*

**Task M18-003: Cache-Optimized Edge Storage** (P2 if gap >10%)
- **Objective**: Improve cache locality for graph traversal
- **Approach**:
  1. Implement adjacency list with spatial locality (packed edges)
  2. Use NUMA-aware allocation for distributed graphs
  3. Profile cache miss rates before/after optimization
- **Success Criteria**: P99 <15ms (2x faster than Neo4j), <10% memory overhead
- **Estimated Effort**: 10 hours

---

### Hybrid Workload Performance (No Competitor)

**Measured Performance**:
- Engram P99: ___ms
- Target: <10ms (internal goal)
- Gap: ___% vs target
- Severity: [Critical|High|Medium|Low|On Target]

**Analysis**:

This is Engram's unique capability - no direct competitor. Performance targets based on:
1. User expectation: "Hybrid should be <2x slower than pure operations"
2. Internal SLA: P99 <10ms for production RAG applications

**Bottleneck Analysis**:
- Break down P99 by operation type: Store (___ms), Recall (___ms), Search (___ms), Pattern Completion (___ms)
- Identify slowest operation and root cause
- Determine if overhead is from context switching or inherent operation cost

**Proposed M18 Tasks**:

*Create tasks only if P99 >10ms*

**Task M18-004: Optimize Pattern Completion Latency** (P1 if >15ms)
- **Objective**: Reduce pattern completion to <5ms P99
- **Approach**: TBD based on profiling
- **Success Criteria**: Hybrid workload P99 <10ms
- **Estimated Effort**: TBD

---

### Large-Scale Performance (10M Vector Scenario)

**Measured Performance**:
- Engram P99: ___ms (or OOM)
- Milvus P99: 708ms
- Gap: ___%
- Severity: [Critical|High|Medium|Low|Advantage]

**Scalability Analysis**:

If 10M scenario fails with OOM:
1. **Memory Footprint**: Calculate actual RSS vs theoretical minimum
   - Theoretical: 10M × 768 × 4 bytes = 30.7GB
   - Actual: ___GB (measure with `/usr/bin/time -l`)
   - Overhead: ___% (should be <30% for indices/metadata)

2. **Optimization Opportunities**:
   - Quantization: f32 → f16 or int8 (50-75% memory savings)
   - Tiered storage: Hot/warm/cold tiers (M17 feature underutilized?)
   - Lazy loading: Load embeddings on-demand from disk

**Proposed M18 Tasks**:

**Task M18-005: Embedding Quantization (f32 → f16)** (P2)
- **Objective**: Reduce memory footprint by 50% for large-scale deployments
- **Approach**:
  1. Implement f16 storage format with transparent conversion
  2. Validate recall degradation <0.1% on standard benchmarks
  3. Add `quantization: "f16"` config option
- **Success Criteria**: 10M scenario runs on 16GB RAM with <1% recall loss
- **Estimated Effort**: 15 hours
- **Dependencies**: None

**Task M18-006: Tiered Storage for Cold Embeddings** (P3)
- **Objective**: Use M17 tiered storage for 100M+ vector workloads
- **Approach**:
  1. Integrate cold tier with HNSW index (page in on access)
  2. Implement LRU eviction policy for warm→cold promotion
  3. Benchmark latency vs memory tradeoff
- **Success Criteria**: 100M vectors on 32GB RAM with <100ms P99
- **Estimated Effort**: 20 hours
- **Dependencies**: M17 tiered storage complete

---

## Prioritized M18 Task List

*Auto-generate after scoring all findings*

### Must-Have (P1)
1. **M18-001**: Optimize HNSW Search Path (if ANN gap >20%)
2. **M18-002**: SIMD Batch Distance Calculation (if ANN gap >30%)
3. **M18-004**: Pattern Completion Optimization (if hybrid >15ms)

### Should-Have (P2)
4. **M18-003**: Cache-Optimized Edge Storage (if graph gap >10%)
5. **M18-005**: Embedding Quantization (if 10M OOM)

### Nice-to-Have (P3)
6. **M18-006**: Tiered Storage Integration (if 100M+ use case identified)

## Task Template for M18

Use this template when creating optimization tasks in `roadmap/milestone-18/`:

```markdown
# Task M18-XXX: [Optimization Name]

**Status**: Pending
**Complexity**: [Simple|Moderate|Complex]
**Dependencies**: [List M17/M17.1 tasks]
**Estimated Effort**: [Hours]
**Priority**: [P1|P2|P3] (from competitive analysis)

## Objective

Optimize [component] to achieve [target metric] based on competitive baseline gap identified in M17.1.

## Background

**Current Performance**: [Measured P99/throughput from M17.1 Task 006]
**Target Performance**: [Competitor baseline or internal goal]
**Gap**: [Percentage slower, from OPTIMIZATION_RECOMMENDATIONS.md]

## Root Cause Analysis

*Reference flamegraphs, diagnostics, and profiling from M17.1*

Hot spots identified:
1. [Function/module with high CPU %]
2. [Cache misses or lock contention]
3. [SIMD underutilization or memory allocation]

## Optimization Strategy

1. **Phase 1: Profile and Validate** (Xh)
   - Reproduce performance gap with isolated microbenchmark
   - Profile with flamegraph, perf stat, and cachegrind
   - Identify top 3 optimization opportunities (Pareto principle)

2. **Phase 2: Implement Top Optimization** (Xh)
   - [Specific code change, e.g., "Add AVX2 intrinsics to cosine_similarity()"]
   - [File path: engram-core/src/...]
   - Benchmark: Expect X% improvement

3. **Phase 3: Validate and Tune** (Xh)
   - Re-run competitive scenario: `qdrant_ann_1m_768d.toml`
   - Verify P99 meets target (< Xms)
   - Check for regressions on other scenarios

## File Paths

```
engram-core/src/... (modify)
benches/competitive/... (add microbenchmarks)
```

## Acceptance Criteria

1. Competitive scenario P99 < [target]ms (meets/exceeds competitor)
2. No regression on other scenarios (within 5%)
3. Microbenchmark shows [X]% improvement
4. Flamegraph confirms hot spot eliminated
5. Documentation updated in competitive_baselines.md

## Testing Approach

```bash
# Microbenchmark (before optimization)
cargo bench competitive_[scenario]
# Baseline: ____ ns/iter

# Apply optimization
# ...

# Microbenchmark (after optimization)
cargo bench competitive_[scenario]
# Expected: < ____ ns/iter ([X]% improvement)

# Full scenario validation
./scripts/competitive_benchmark_suite.sh --scenario [name]
# Expected: P99 < [target]ms
```

## Integration Points

- Updates engram-core components from M17
- Validates against M17.1 competitive baselines
- Documents improvement in competitive_baselines.md quarterly review
```

## Success Metrics for M18

Define measurable goals for M18 based on M17.1 findings:

| Metric | M17.1 Baseline | M18 Target | Rationale |
|--------|----------------|------------|-----------|
| ANN Search P99 (1M) | ___ms | <20ms | Qdrant parity (22-24ms) |
| Graph Traversal P99 (100K) | ___ms | <15ms | 2x faster than Neo4j (27.96ms) |
| Hybrid Workload P99 (100K) | ___ms | <10ms | Production SLA target |
| Memory Footprint (10M) | ___GB | <20GB | Fit on commodity 32GB servers |
| Throughput (mixed workload) | ___QPS | >1000 QPS | 2x M17 baseline |

## Review Cadence

**Quarterly Check** (align with competitive baseline reviews):
- Run M18 optimizations against latest competitive benchmarks
- Update targets if competitors release performance improvements
- Publish blog post on competitive advantages gained
```

---

## File Paths

```
docs/reference/competitive_baselines.md (create)
docs/reference/README.md (modify - add competitive section)
vision.md (modify - add competitive positioning section)
roadmap/milestone-17.1/M17.1_COMPLETION_CHECKLIST.md (create)
roadmap/milestone-17.1/OPTIMIZATION_RECOMMENDATIONS.md (create)
roadmap/milestone-17.1/acceptance_tests/scenario_*.sh (create test scripts)
```

## Acceptance Criteria

### Documentation Completeness
1. `competitive_baselines.md` includes all required sections (competitor table, scenario mapping, methodology, quarterly process)
2. All competitor baseline sources cited with working URLs
3. Vision statement updated with measurable competitive claims
4. `docs/reference/README.md` links to competitive documentation
5. All markdown files pass linting: `npx markdownlint-cli2 docs/reference/*.md`
6. All links validated: `npx markdown-link-check docs/reference/competitive_baselines.md`

### Acceptance Testing
1. All 7 scenarios pass:
   - Scenario 1 (Fresh Deployment): Report generated from scratch, exit 0
   - Scenario 2 (Trend Analysis): Delta <1% for identical runs, trend section present
   - Scenario 3 (Regression Detection): Exit code 2 when degradation >5%
   - Scenario 4 (Cross-Platform): Operation sequences identical (SHA256 match)
   - Scenario 5 (Concurrent Execution): Parallel runs succeed without interference
   - Scenario 6 (Failure Recovery): Server crash, disk full, network failure handled gracefully
   - Scenario 7 (Documentation Validation): Another engineer completes task in <30min without help
2. Test automation scripts created and executable
3. All expected failure modes tested and logged

### Completion Checklist Validation
1. All 8 tasks listed with granular sub-criteria
2. Quality gates defined (clippy, shellcheck, linting)
3. Performance validation against M17 baseline specified
4. Sign-off section with dates and reviewers
5. Known limitations section populated during testing

### M18 Recommendations
1. `OPTIMIZATION_RECOMMENDATIONS.md` follows template structure
2. Gap analysis methodology clearly explained
3. Prioritization framework applied (P1/P2/P3 scoring)
4. At least 3 concrete M18 tasks proposed with effort estimates
5. Task template provided for consistent M18 planning

### No M17 Regressions
1. Run M17 performance comparison: `./scripts/compare_m17_performance.sh 001`
2. All M17 tasks show <5% regression
3. Diagnostics clean: `./scripts/engram_diagnostics.sh` (no new warnings)
4. Existing integration tests pass: `cargo test --release integration_tests`

## Testing Approach

### Phase 1: Documentation Validation (1h)

```bash
# Markdown linting
npx markdownlint-cli2 'docs/reference/*.md' 'vision.md' 'roadmap/milestone-17.1/*.md'
# Expected: 0 warnings

# Link validation
for file in docs/reference/competitive_baselines.md docs/reference/README.md; do
  npx markdown-link-check "$file"
done
# Expected: All links reachable

# Code block syntax
grep -A 20 '```bash' docs/reference/competitive_baselines.md | bash -n
# Expected: No syntax errors

# File reference validation
grep -o 'scenarios/competitive/[^ ]*\.toml' docs/reference/competitive_baselines.md | \
  while read f; do [ -f "$f" ] || echo "Missing: $f"; done
# Expected: No output (all files exist)
```

### Phase 2: Acceptance Scenario Execution (3h)

Create test harness: `roadmap/milestone-17.1/acceptance_tests/run_all_scenarios.sh`

```bash
#!/bin/bash
# Run all 7 acceptance test scenarios sequentially
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== M17.1 Task 008 Acceptance Testing ==="
echo "Starting: $(date)"
echo ""

# Scenario 1: Fresh Deployment
echo "[1/7] Scenario 1: Fresh Deployment (Cold Start)"
"$SCRIPT_DIR/scenario_1_fresh_deployment.sh" || exit 1
echo ""

# Scenario 2: Trend Analysis
echo "[2/7] Scenario 2: Trend Analysis (Quarterly Cadence)"
"$SCRIPT_DIR/scenario_2_trend_analysis.sh" || exit 1
echo ""

# Scenario 3: Regression Detection
echo "[3/7] Scenario 3: Regression Detection (Performance Degradation)"
"$SCRIPT_DIR/scenario_3_regression_detection.sh" || exit 1
echo ""

# Scenario 4: Cross-Platform Determinism
echo "[4/7] Scenario 4: Cross-Platform Determinism"
"$SCRIPT_DIR/scenario_4_cross_platform.sh" || exit 1
echo ""

# Scenario 5: Concurrent Execution
echo "[5/7] Scenario 5: Concurrent Execution (Isolation Validation)"
if [ -f "$SCRIPT_DIR/scenario_5_concurrent.sh" ]; then
  "$SCRIPT_DIR/scenario_5_concurrent.sh" || echo "WARN: Concurrent test failed (optional)"
else
  echo "SKIP: Parallel execution not implemented (acceptable for M17.1)"
fi
echo ""

# Scenario 6: Failure Recovery
echo "[6/7] Scenario 6: Failure Recovery (Resilience Testing)"
"$SCRIPT_DIR/scenario_6_failure_recovery.sh" || exit 1
echo ""

# Scenario 7: Documentation Validation
echo "[7/7] Scenario 7: Documentation Completeness (Human Validation)"
echo "MANUAL: Recruit engineer unfamiliar with M17.1 to follow docs"
echo "  Hand them: docs/reference/competitive_baselines.md"
echo "  Task: Run a competitive baseline and interpret results"
echo "  Success: Completes in <30min without questions"
echo "  Status: [ ] Passed  [ ] Failed"
echo ""

echo "=== Acceptance Testing Complete ==="
echo "Finished: $(date)"
echo ""
echo "Review results and mark Task 008 as complete if all scenarios passed."
```

Individual scenario scripts in `roadmap/milestone-17.1/acceptance_tests/`:
- `scenario_1_fresh_deployment.sh`
- `scenario_2_trend_analysis.sh`
- `scenario_3_regression_detection.sh`
- `scenario_4_cross_platform.sh`
- `scenario_5_concurrent.sh` (optional)
- `scenario_6_failure_recovery.sh`

Each script follows this template:

```bash
#!/bin/bash
# Scenario N: [Name]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Scenario N: [Name] ==="
echo "Objective: [Brief description]"
echo ""

# Preconditions
echo "Step 1: Setup preconditions"
# [Specific setup commands]

# Execution
echo "Step 2: Execute test"
# [Test commands with output capture]

# Validation
echo "Step 3: Validate outcomes"
# [Assertions with clear PASS/FAIL]

echo ""
if [ $? -eq 0 ]; then
  echo "RESULT: PASS - Scenario N completed successfully"
else
  echo "RESULT: FAIL - Scenario N did not meet acceptance criteria"
  exit 1
fi
```

### Phase 3: Completion Checklist Validation (1h)

```bash
# Verify all tasks marked _complete
ls roadmap/milestone-17.1/00*_complete.md | wc -l
# Expected: 8 (all tasks complete)

# Check for any _in_progress or _blocked
ls roadmap/milestone-17.1/00*_in_progress.md roadmap/milestone-17.1/00*_blocked.md 2>/dev/null
# Expected: No such files (all resolved)

# Validate completion checklist has all required sections
grep -E "^- \[ \]" roadmap/milestone-17.1/M17.1_COMPLETION_CHECKLIST.md | wc -l
# Expected: >50 (comprehensive checklist)

# Run final integration test
./scripts/quarterly_competitive_review.sh
echo $?
# Expected: 0 (all scenarios pass)

# Verify no M17 regressions
for task in 001 002 003 004 005 006 007 008 009 010; do
  if [ -f "tmp/m17_performance/${task}_before_"* ]; then
    ./scripts/compare_m17_performance.sh "$task" || echo "WARN: Task $task regression"
  fi
done
# Expected: All tasks <5% regression
```

### Phase 4: M18 Recommendations (1h)

```bash
# Validate M18 task template is filled
grep "Task M18-" roadmap/milestone-17.1/OPTIMIZATION_RECOMMENDATIONS.md | wc -l
# Expected: >3 (at least 3 proposed M18 tasks)

# Check prioritization is applied
grep -E "Priority [123]" roadmap/milestone-17.1/OPTIMIZATION_RECOMMENDATIONS.md | wc -l
# Expected: >3 (all tasks prioritized)

# Verify gap analysis methodology
grep "Gap Severity" roadmap/milestone-17.1/OPTIMIZATION_RECOMMENDATIONS.md
# Expected: Table with severity classifications found

# Manual review: Do proposed tasks have actionable file paths and estimates?
cat roadmap/milestone-17.1/OPTIMIZATION_RECOMMENDATIONS.md
# Expected: Each M18 task includes file paths, effort estimate, success criteria
```

## Integration Points

### Validates All Prior Tasks
- Task 001: Scenarios execute deterministically in acceptance tests
- Task 002: Baseline documentation complete and accurate
- Task 003: Benchmark suite robust to failure modes
- Task 004: Report generator produces trend analysis
- Task 005: Quarterly workflow orchestrates end-to-end
- Task 006: Initial measurements documented
- Task 007: Regression detection integrated

### Updates Project-Wide Documentation
- `vision.md`: Competitive positioning statement
- `docs/reference/`: New competitive baselines section
- `roadmap/milestone-17.1/`: Completion checklist and M18 recommendations

### Defines M17.1 Completion
- Comprehensive checklist validates all deliverables
- Clear acceptance criteria for each task
- Sign-off process with reviewers
- Known limitations documented for transparency

### Creates M18 Bridge
- Gap analysis prioritizes optimization work
- Task template ensures consistent planning
- Success metrics defined for quarterly tracking
- Framework established for ongoing competitive monitoring

## Performance Expectations

**Acceptance Test Execution Time**:
- Scenario 1 (Fresh): 10-15 minutes (full workflow cold start)
- Scenario 2 (Trend): 15-20 minutes (two full runs + comparison)
- Scenario 3 (Regression): 10 minutes (two runs + validation)
- Scenario 4 (Cross-platform): 20 minutes if Docker used, skip if Linux unavailable
- Scenario 5 (Concurrent): 10 minutes (if implemented, otherwise skip)
- Scenario 6 (Failure): 15 minutes (multiple failure modes)
- Scenario 7 (Documentation): 30 minutes (human validation)

**Total acceptance testing**: ~2 hours (can be partially parallelized)

**Full Task 008 Completion**: 6 hours (including documentation writing and M18 planning)

## Known Limitations

1. **Cross-platform testing**: Requires access to Linux system (Docker acceptable)
2. **Concurrent execution**: Not implemented in M17.1 (sequential acceptable for quarterly cadence)
3. **Human validation**: Requires recruiting another engineer (can be async)
4. **M18 prioritization**: Depends on Task 006 actual measurements (template provided, scoring pending)

## Success Criteria Summary

Task 008 is complete when:

1. All 7 acceptance test scenarios documented with executable scripts
2. At least 5/7 scenarios pass (cross-platform and concurrent are optional)
3. Documentation updates merged:
   - `competitive_baselines.md` created with all sections
   - `vision.md` updated with competitive positioning
   - `docs/reference/README.md` links to competitive docs
4. Completion checklist comprehensive and validated:
   - All 8 tasks listed with granular criteria
   - Quality gates defined (clippy, shellcheck, linting, performance)
   - Sign-off section present
5. M18 recommendations documented:
   - Gap analysis methodology explained
   - At least 3 M18 tasks proposed with priorities
   - Task template provided
6. No M17 performance regressions (<5% on all prior tasks)
7. All markdown files lint-clean and links validated
8. `make quality` passes (zero clippy warnings)

## References

- M17 Performance Framework: `roadmap/milestone-17/PERFORMANCE_WORKFLOW.md`
- Existing Performance Scripts: `scripts/m17_performance_check.sh`, `scripts/compare_m17_performance.sh`
- Diataxis Documentation Framework: https://diataxis.fr/
- HdrHistogram (latency measurement): https://github.com/HdrHistogram/HdrHistogram_rust
- Competitive Baselines (Qdrant): https://qdrant.tech/benchmarks/
- Competitive Baselines (Neo4j): https://neo4j.com/developer/graph-data-science/performance/
- Competitive Baselines (Milvus): https://milvus.io/docs/benchmark.md
