# Milestone 17.1: Competitive Baseline Comparison Framework

**Status**: Pending
**Type**: Comparison & Benchmarking Infrastructure
**Estimated Effort**: 30 hours (~4 working days)
**Dependencies**: Milestone 17 (Dual Memory Architecture)

## Overview

Milestone 17.1 establishes a rigorous competitive comparison framework to position Engram against leading vector databases (Qdrant, Milvus, Weaviate) and graph databases (Neo4j). This framework enables quarterly performance benchmarking at scale (100K-1M nodes) while maintaining compatibility with existing M17 regression tracking infrastructure.

## Motivation

Engram currently tracks internal performance meticulously through M17's regression framework, but lacks external comparison points to validate competitiveness in the market. Without comparative baselines:

- Cannot make credible performance claims ("Engram is X% faster than Neo4j")
- Cannot identify optimization priorities based on competitive gaps
- Cannot track market positioning over time
- Cannot validate hybrid architecture advantages quantitatively

This milestone addresses these gaps with automated, reproducible competitive benchmarks.

## Core Objectives

1. **Standardized Scenarios**: Create TOML scenarios matching competitor benchmarks
   - ANN search (1M vectors) comparable to Qdrant/Milvus
   - Graph traversal (100K nodes) comparable to Neo4j
   - Mixed hybrid workload (Engram's unique capability)

2. **Comparison Documentation**: Track Engram vs competitors
   - Documented competitor baselines with sources
   - Performance positioning (where we lead, where we lag)
   - Market differentiation (hybrid vector+graph+temporal)

3. **Automated Comparison**: Scripts for quarterly execution
   - Competitive benchmark suite runner (bash)
   - Comparison report generator (Python)
   - Integration with existing M17 workflow

4. **Validation & Targets**: Set aspirational performance goals
   - Target: Beat Qdrant P99 (20ms vs their 22ms)
   - Target: Beat Neo4j traversal (15ms vs their 27.96ms)
   - Identify optimization priorities from gaps

## Task Breakdown

### Task 001: Competitive Scenario Suite
**Effort**: 4 hours | **Complexity**: Moderate

Create 4 standardized TOML scenario files:
- `qdrant_ann_1m_768d.toml` - Pure ANN search, 1M vectors
- `neo4j_traversal_100k.toml` - Graph traversal, 100K nodes
- `hybrid_production_100k.toml` - Mixed workload (unique to Engram)
- `milvus_ann_10m_768d.toml` - Large-scale ANN (stretch goal)

Includes comprehensive validation scripts for determinism, correctness, and resource bounds.

### Task 002: Competitive Baseline Documentation
**Effort**: 3 hours | **Complexity**: Simple

Document competitor baselines in `docs/reference/competitive_baselines.md`:
- Qdrant: 22-24ms P99, 626 QPS @ 99.5% recall (1M vectors)
- Neo4j: 27.96ms single-hop traversal, 280 QPS
- Milvus: 708ms P99, 2,098 QPS @ 100% recall (10M vectors)
- Performance targets and measurement methodology

### Task 003: Competitive Benchmark Suite Runner
**Effort**: 5 hours | **Complexity**: Moderate

Production-hardened bash script (`competitive_benchmark_suite.sh`) that:
- Runs all 4 scenarios with comprehensive pre-flight checks
- Collects diagnostics (system metrics, Engram metrics, flame graphs)
- Handles failures gracefully with scenario isolation
- Completes in <10 minutes on standard hardware

### Task 004: Competitive Comparison Report Generator
**Effort**: 8 hours | **Complexity**: Complex

Python script (`generate_competitive_report.py`) that:
- Parses loadtest results with robust error handling
- Performs statistical analysis (t-tests, confidence intervals)
- Generates actionable markdown reports with optimization priorities
- Supports historical trend analysis across quarters

### Task 005: Quarterly Review Workflow Integration
**Effort**: 2 hours | **Complexity**: Simple

Orchestration script and documentation:
- `quarterly_competitive_review.sh` - One-command workflow
- Updated documentation explaining quarterly review process
- Calendar reminder setup for performance engineering team

### Task 006: Initial Baseline Measurement
**Effort**: 3 hours | **Complexity**: Simple

Execute first competitive baseline measurement:
- Run full benchmark suite on stable commit
- Validate results against competitor baselines
- Document Engram's current positioning
- Create follow-up optimization tasks if needed

### Task 007: Performance Regression Prevention
**Effort**: 4 hours | **Complexity**: Moderate

Integrate competitive benchmarks with M17 regression framework:
- Add `--competitive` flag to M17 performance check scripts
- Extend comparison script to include competitive baselines
- Update CLAUDE.md with competitive validation instructions

### Task 008: Documentation and Acceptance Testing
**Effort**: 3 hours | **Complexity**: Simple

End-to-end validation and documentation updates:
- Run 3 acceptance test scenarios (fresh, trend, regression)
- Update project documentation (vision.md, docs/reference/)
- Create M17.1 completion checklist
- Generate optimization recommendations for M18

## Critical Path

```
001 (Scenarios) → 002 (Baselines) → 003 (Runner) → 004 (Report) → 005 (Workflow) → 006 (Initial) → 007 (Integration) → 008 (Acceptance)
```

All tasks are sequential with clear dependencies.

## Success Metrics

1. **Reproducibility**: All scenarios produce identical results across 3 runs (<1% P99 variance)
2. **Automation**: Full workflow completes in <15 minutes without manual intervention
3. **Quality**: Zero clippy warnings, 90%+ test coverage for report generator
4. **Competitiveness**: At least 1 scenario shows Engram as competitive (within 10% of best-in-class)
5. **Actionability**: Report clearly identifies optimization priorities with specific targets

## Expected Outcomes

After M17.1 completion, Engram will have:

1. **Quantitative Positioning**:
   - "Engram achieves X% faster graph traversal than Neo4j"
   - "Engram achieves Y% of Qdrant's vector search performance"
   - "Engram uniquely supports hybrid operations in single query"

2. **Quarterly Tracking**:
   - Automated benchmarks run Q1, Q2, Q3, Q4
   - Historical trends show improvement/degradation over time
   - Competitive gaps inform optimization roadmap

3. **Optimization Priorities**:
   - Data-driven task creation for M18+ based on competitive gaps
   - Clear targets (e.g., "Reduce ANN P99 from 26ms to <20ms")
   - Resource allocation aligned with market positioning

4. **External Validation**:
   - Credible performance claims for documentation/marketing
   - Reproducible benchmarks suitable for publication
   - Transparent methodology builds trust

## Technical Constraints

- **Compatibility**: Must work with existing `loadtest` tool (no breaking changes)
- **Scale**: Test at 100K-1M nodes to match competitor benchmarks
- **Performance**: <5% overhead from competitive validation on M17 tasks
- **Quality**: Zero clippy warnings, passes `make quality` throughout
- **Reproducibility**: Deterministic seeds, documented configurations

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory constraints on 1M scenarios | High | Start with 100K, scale incrementally |
| Competitor baseline data gaps | Medium | Document missing data, use conservative estimates |
| Performance variability | Medium | Use deterministic seeds, multiple runs for validation |
| Scope creep (graphical dashboards) | Low | Limit to ASCII charts, defer visuals to future work |

## Integration with Existing Systems

- **M17 Regression Framework**: Extends with `--competitive` flag
- **Load Test Tool**: Uses existing `tools/loadtest` without modifications
- **Documentation**: Integrates with `docs/reference/` structure
- **Performance Log**: Complements `roadmap/milestone-17/PERFORMANCE_LOG.md`

## Post-Milestone Maintenance

- **Quarterly Execution**: First week of Jan, Apr, Jul, Oct
- **Baseline Updates**: Refresh competitor data when new benchmarks published
- **Scenario Evolution**: Add new scenarios as Engram capabilities grow
- **Automation**: Consider CI integration for automated quarterly runs

## References

- Qdrant benchmarks: https://qdrant.tech/benchmarks/
- Neo4j performance: https://neo4j.com/developer/graph-data-science/performance/
- Milvus benchmarks: https://milvus.io/docs/benchmark.md
- M17 Performance Workflow: `roadmap/milestone-17/PERFORMANCE_WORKFLOW.md`
