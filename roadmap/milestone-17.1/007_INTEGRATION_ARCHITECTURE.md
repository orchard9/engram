# Task 007: Integration Architecture Diagram

## Component Overview

```
┌────────────────────────────────────────────────────────────────┐
│                     Developer Workflow (CLAUDE.md)              │
│                                                                  │
│  Step 11: Internal Performance Validation (required)            │
│  Step 12: Competitive Performance Validation (optional)         │
└────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│              scripts/m17_performance_check.sh                    │
│                                                                  │
│  Inputs:                                                         │
│    - task_id (e.g., "007")                                      │
│    - phase ("before" or "after")                                │
│    - --competitive (optional flag)                              │
│                                                                  │
│  Logic:                                                          │
│    if --competitive:                                            │
│      scenario = scenarios/competitive/hybrid_production_100k    │
│      prefix = "competitive_"                                    │
│    else:                                                         │
│      scenario = scenarios/m17_baseline.toml                     │
│      prefix = ""                                                │
│                                                                  │
│  Output:                                                         │
│    tmp/m17_performance/[prefix]<task>_<phase>_<timestamp>.json │
│    tmp/m17_performance/[prefix]<task>_<phase>_<timestamp>_sys  │
│    tmp/m17_performance/[prefix]<task>_<phase>_<timestamp>_diag │
└────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│             scripts/compare_m17_performance.sh                   │
│                                                                  │
│  Inputs:                                                         │
│    - task_id (e.g., "007")                                      │
│                                                                  │
│  Detection Logic:                                                │
│    if filename contains "competitive_":                         │
│      competitive_mode = true                                    │
│      threshold = 10%                                            │
│      load baseline from competitive_baselines.md                │
│    else:                                                         │
│      competitive_mode = false                                   │
│      threshold = 5%                                             │
│                                                                  │
│  Comparison Logic:                                               │
│    Level 1: Internal Delta (before vs after)                    │
│      - Check P99 latency increase                               │
│      - Check throughput decrease                                │
│      - Check error rate increase                                │
│                                                                  │
│    Level 2: Competitive Positioning (only if competitive_mode)  │
│      - Compare after result vs Neo4j baseline (27.96ms)         │
│      - Calculate speedup percentage                             │
│      - Detect positioning degradation                           │
│                                                                  │
│  Exit Codes:                                                     │
│    0 - No regression (within threshold)                         │
│    1 - Internal regression (>5% or >10% depending on mode)     │
│    2 - Competitive regression (positioning degraded)            │
│    3 - Error (missing files or invalid data)                    │
└────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│              roadmap/milestone-17/PERFORMANCE_LOG.md             │
│                                                                  │
│  Task-by-task tracking:                                         │
│                                                                  │
│  ## Task 007: Fan Effect Spreading                              │
│                                                                  │
│  **Internal Performance**:                                      │
│    - Before: P99=0.458ms, 999.88 ops/s                         │
│    - After:  P99=0.472ms, 995.24 ops/s                         │
│    - Change: +3.1%, -0.5%                                       │
│    - Status: ✓ Within 5% target                                │
│                                                                  │
│  **Competitive Performance**:                                   │
│    - Before: P99=10.2ms, 490 QPS                               │
│    - After:  P99=10.5ms, 487 QPS                               │
│    - Change: +2.9%, -0.6%                                       │
│    - vs Neo4j: 62.4% faster (10.5ms vs 27.96ms)                │
│    - Status: ✓ Competitive advantage maintained                │
└────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

### Internal Testing Flow (Existing)

```
Developer
    │
    └─> ./scripts/m17_performance_check.sh 007 before
            │
            ├─> Build: cargo build --release
            ├─> Start: ./target/release/engram start --port 7432
            ├─> Test:  ./target/release/loadtest run --scenario m17_baseline.toml --duration 60
            └─> Save:  tmp/m17_performance/007_before_<timestamp>.json
    │
    └─> (implement task)
    │
    └─> ./scripts/m17_performance_check.sh 007 after
            │
            ├─> Build: cargo build --release
            ├─> Start: ./target/release/engram start --port 7432
            ├─> Test:  ./target/release/loadtest run --scenario m17_baseline.toml --duration 60
            └─> Save:  tmp/m17_performance/007_after_<timestamp>.json
    │
    └─> ./scripts/compare_m17_performance.sh 007
            │
            ├─> Load: 007_before_*.json, 007_after_*.json
            ├─> Calculate: P99 delta, throughput delta
            ├─> Check: delta > 5%?
            │       ├─> Yes: Exit 1 (REGRESSION)
            │       └─> No:  Exit 0 (SUCCESS)
            └─> Output: Performance summary
```

### Competitive Testing Flow (New)

```
Developer
    │
    └─> ./scripts/m17_performance_check.sh 007 before --competitive
            │
            ├─> Build: cargo build --release
            ├─> Start: ./target/release/engram start --port 7432
            ├─> Test:  ./target/release/loadtest run --scenario competitive/hybrid_production_100k.toml --duration 60
            └─> Save:  tmp/m17_performance/competitive_007_before_<timestamp>.json
    │
    └─> (implement task)
    │
    └─> ./scripts/m17_performance_check.sh 007 after --competitive
            │
            ├─> Build: cargo build --release
            ├─> Start: ./target/release/engram start --port 7432
            ├─> Test:  ./target/release/loadtest run --scenario competitive/hybrid_production_100k.toml --duration 60
            └─> Save:  tmp/m17_performance/competitive_007_after_<timestamp>.json
    │
    └─> ./scripts/compare_m17_performance.sh 007
            │
            ├─> Detect: filename contains "competitive_" → competitive_mode = true
            ├─> Load: competitive_007_before_*.json, competitive_007_after_*.json
            ├─> Calculate: P99 delta, throughput delta
            ├─> Check Level 1: delta > 10%?
            │       ├─> Yes: Exit 2 (COMPETITIVE REGRESSION)
            │       └─> No:  Continue to Level 2
            ├─> Load: Neo4j baseline = 27.96ms (from competitive_baselines.md)
            ├─> Calculate: speedup = (27.96 - after_p99) / 27.96 * 100
            ├─> Check Level 2: positioning degraded?
            │       ├─> Yes: Exit 2 (COMPETITIVE REGRESSION) + warning
            │       └─> No:  Exit 0 (SUCCESS)
            └─> Output: Performance summary + competitive positioning
```

## Regression Detection Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│ START: Load before/after results                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ Detect competitive    │
         │ mode from filename?   │
         └───────┬───────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
  competitive_      (no prefix)
        │                 │
        ▼                 ▼
  Threshold = 10%   Threshold = 5%
        │                 │
        └────────┬────────┘
                 │
                 ▼
         ┌──────────────────┐
         │ Calculate deltas │
         │  - P99 latency   │
         │  - Throughput    │
         │  - Error rate    │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────────┐
         │ P99 increase         │
         │ > threshold?         │
         └──────┬───────────────┘
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
      YES              NO
        │               │
        │               ▼
        │      ┌──────────────────────┐
        │      │ Throughput decrease  │
        │      │ > threshold?         │
        │      └──────┬───────────────┘
        │             │
        │     ┌───────┴───────┐
        │     │               │
        │     ▼               ▼
        │    YES              NO
        │     │               │
        └─────┤               │
              │               ▼
              │      ┌──────────────────────┐
              │      │ Error rate increase  │
              │      │ > 5 percentage pts?  │
              │      └──────┬───────────────┘
              │             │
              │     ┌───────┴───────┐
              │     │               │
              │     ▼               ▼
              │    YES              NO
              │     │               │
              └─────┤               │
                    │               │
                    ▼               ▼
           ┌──────────────┐  ┌─────────────────┐
           │ Competitive  │  │ Competitive     │
           │ mode?        │  │ mode?           │
           └──────┬───────┘  └────┬────────────┘
                  │               │
          ┌───────┴───────┐       │
          │               │       │
          ▼               ▼       ▼
         YES              NO      NO → Exit 0 (SUCCESS)
          │               │       │
          │               │       ▼
          │               │      YES
          │               │       │
          │               ▼       ▼
          │      Exit 1 (INTERNAL  ┌──────────────────────┐
          │       REGRESSION)      │ Load competitor      │
          │                        │ baseline (Neo4j)     │
          ▼                        └──────┬───────────────┘
┌──────────────────────┐                 │
│ Check competitive    │                 ▼
│ positioning          │        ┌──────────────────────┐
└──────┬───────────────┘        │ Calculate speedup    │
       │                        │ vs baseline          │
       ▼                        └──────┬───────────────┘
┌──────────────────────┐               │
│ After P99 >          │               ▼
│ Neo4j baseline?      │        ┌──────────────────────┐
└──────┬───────────────┘        │ Speedup degraded     │
       │                        │ significantly?       │
┌──────┴──────┐                 └──────┬───────────────┘
│             │                        │
▼             ▼                ┌───────┴───────┐
YES           NO               │               │
│             │                ▼               ▼
│             │               YES              NO
│             │                │               │
│             │                │               │
│             └────────────────┤               │
│                              │               │
▼                              ▼               ▼
Exit 2                    Exit 2          Exit 0
(SLOWER THAN              (POSITIONING     (SUCCESS +
COMPETITOR)               DEGRADED)        COMPETITIVE
                                           ADVANTAGE
                                           MAINTAINED)
```

## File System Layout

```
engram/
├── scripts/
│   ├── m17_performance_check.sh         ← Modified in Task 007
│   ├── compare_m17_performance.sh       ← Modified in Task 007
│   └── quarterly_competitive_review.sh  ← Created in Task 005, used by Task 006
│
├── scenarios/
│   ├── m17_baseline.toml                ← Existing (internal testing)
│   └── competitive/                     ← Created in Task 001
│       ├── qdrant_ann_1m_768d.toml
│       ├── milvus_ann_10m.toml
│       ├── neo4j_traversal_100k.toml
│       └── hybrid_production_100k.toml  ← Used by Task 007
│
├── tmp/m17_performance/                 ← Runtime output directory
│   ├── 007_before_20251108_140000.json         ← Internal test (before)
│   ├── 007_before_20251108_140000_sys.txt
│   ├── 007_before_20251108_140000_diag.txt
│   ├── 007_after_20251108_142000.json          ← Internal test (after)
│   ├── 007_after_20251108_142000_sys.txt
│   ├── 007_after_20251108_142000_diag.txt
│   ├── competitive_007_before_20251108_143000.json  ← Competitive test (before)
│   ├── competitive_007_before_20251108_143000_sys.txt
│   ├── competitive_007_before_20251108_143000_diag.txt
│   ├── competitive_007_after_20251108_144000.json   ← Competitive test (after)
│   ├── competitive_007_after_20251108_144000_sys.txt
│   └── competitive_007_after_20251108_144000_diag.txt
│
├── roadmap/milestone-17/
│   ├── PERFORMANCE_LOG.md               ← Modified in Task 007 (tracking)
│   └── PERFORMANCE_WORKFLOW.md          ← Modified in Task 007 (documentation)
│
├── docs/reference/
│   └── competitive_baselines.md         ← Created in Task 002, referenced by Task 007
│
└── CLAUDE.md                            ← Modified in Task 007 (workflow instructions)
```

## Sequence Diagram: Competitive Regression Detection

```
Developer    m17_check.sh    Engram Server   Loadtest    compare.sh    Performance Log
    │              │                │             │            │                │
    │─────before─>│                │             │            │                │
    │ --competitive│                │             │            │                │
    │              │                │             │            │                │
    │              │──build────────>│             │            │                │
    │              │<──binary───────│             │            │                │
    │              │                │             │            │                │
    │              │──start────────>│             │            │                │
    │              │<──listening────│             │            │                │
    │              │                │             │            │                │
    │              │──run hybrid_production_100k─>│            │                │
    │              │                │<───queries──│            │                │
    │              │                │───results──>│            │                │
    │              │<──metrics──────────────────x─│            │                │
    │              │                │             │            │                │
    │              │──stop─────────>│             │            │                │
    │              │                │             │            │                │
    │              │──save: competitive_007_before_*.json      │                │
    │              │                │             │            │                │
    │<─────done────│                │             │            │                │
    │              │                │             │            │                │
    │              │                │             │            │                │
    │──(implement task)              │             │            │                │
    │              │                │             │            │                │
    │              │                │             │            │                │
    │──────after──>│                │             │            │                │
    │ --competitive│                │             │            │                │
    │              │                │             │            │                │
    │              │──build────────>│             │            │                │
    │              │<──binary───────│             │            │                │
    │              │                │             │            │                │
    │              │──start────────>│             │            │                │
    │              │<──listening────│             │            │                │
    │              │                │             │            │                │
    │              │──run hybrid_production_100k─>│            │                │
    │              │                │<───queries──│            │                │
    │              │                │───results──>│            │                │
    │              │<──metrics──────────────────x─│            │                │
    │              │                │             │            │                │
    │              │──stop─────────>│             │            │                │
    │              │                │             │            │                │
    │              │──save: competitive_007_after_*.json       │                │
    │              │                │             │            │                │
    │<─────done────│                │             │            │                │
    │              │                │             │            │                │
    │              │                │             │            │                │
    │────compare──────────────────────────────────────────────>│                │
    │      007     │                │             │            │                │
    │              │                │             │            │                │
    │              │                │             │            │──detect: competitive mode
    │              │                │             │            │  (filename contains "competitive_")
    │              │                │             │            │                │
    │              │                │             │            │──load: before/after JSON
    │              │                │             │            │                │
    │              │                │             │            │──calculate: P99 delta
    │              │                │             │            │  (11.5ms - 10.2ms) / 10.2ms
    │              │                │             │            │  = +12.7%
    │              │                │             │            │                │
    │              │                │             │            │──check: 12.7% > 10%
    │              │                │             │            │  → REGRESSION!
    │              │                │             │            │                │
    │              │                │             │            │──load: Neo4j baseline
    │              │                │             │            │  27.96ms
    │              │                │             │            │                │
    │              │                │             │            │──calculate: speedup
    │              │                │             │            │  (27.96 - 11.5) / 27.96
    │              │                │             │            │  = 58.9% faster
    │              │                │             │            │                │
    │              │                │             │            │──compare: before 63.5% → after 58.9%
    │              │                │             │            │  → POSITIONING DEGRADED (warning)
    │              │                │             │            │                │
    │<─────────────────────────alert: COMPETITIVE REGRESSION───│                │
    │  Exit Code 2 │                │             │            │                │
    │  P99: +12.7% │                │             │            │                │
    │  Positioning:│                │             │            │                │
    │  58.9% faster│                │             │            │                │
    │  (vs 63.5%)  │                │             │            │                │
    │              │                │             │            │                │
    │──────────────────────────────────────────────────────────────────────────>│
    │ (manual) Update PERFORMANCE_LOG.md with competitive results               │
    │              │                │             │            │                │
```

## State Transition Diagram: Task Completion States

```
┌─────────────────┐
│  Task Pending   │
└────────┬────────┘
         │
         │ Developer starts task
         │
         ▼
┌─────────────────┐
│ Run Internal    │
│ Before Test     │
└────────┬────────┘
         │
         │ P99, throughput captured
         │
         ▼
┌─────────────────┐        ┌──────────────────────┐
│  Implement      │───────>│ Core graph operation │
│  Task           │        │ affected?            │
└────────┬────────┘        └──────┬───────────────┘
         │                        │
         │                ┌───────┴───────┐
         │                │               │
         │                ▼               ▼
         │               YES              NO
         │                │               │
         │                │               │
         │                ▼               │
         │       ┌─────────────────┐     │
         │       │ Run Competitive │     │
         │       │ Before Test     │     │
         │       └────────┬────────┘     │
         │                │               │
         │                │               │
         ▼                ▼               │
┌─────────────────┐  ┌─────────────────┐ │
│ Run Internal    │  │ Run Competitive │ │
│ After Test      │  │ After Test      │ │
└────────┬────────┘  └────────┬────────┘ │
         │                    │           │
         │                    │           │
         ▼                    ▼           │
┌─────────────────┐  ┌─────────────────┐ │
│ Compare         │  │ Compare         │ │
│ Internal        │  │ Competitive     │ │
│ (5% threshold)  │  │ (10% threshold) │ │
└────────┬────────┘  └────────┬────────┘ │
         │                    │           │
         │            ┌───────┴───────┐   │
         │            │               │   │
         │            ▼               ▼   │
         │       Exit 0, 2       Exit 0, 2│
         │            │               │   │
         └────────────┴───────────────┴───┘
                      │
              ┌───────┴───────┐
              │               │
              ▼               ▼
         Exit Code 2     Exit Code 0
         (Regression)    (Success)
              │               │
              │               │
              ▼               ▼
    ┌─────────────────┐  ┌─────────────────┐
    │ Fix Regression  │  │ Update          │
    │ or Create       │  │ Performance Log │
    │ Optimization    │  │                 │
    │ Follow-up Task  │  └────────┬────────┘
    └────────┬────────┘           │
             │                    │
             │                    │
             └────────────────────┘
                      │
                      ▼
              ┌─────────────────┐
              │ Task Complete   │
              │ (rename to      │
              │ _complete.md)   │
              └─────────────────┘
```

## Integration Points Summary

### Inputs to Task 007

1. **From Task 001** (Competitive Scenario Suite):
   - `scenarios/competitive/hybrid_production_100k.toml` (scenario definition)

2. **From Task 002** (Competitive Baseline Documentation):
   - `docs/reference/competitive_baselines.md` (Neo4j baseline: 27.96ms P99)

3. **From Existing M17 Framework**:
   - `scripts/m17_performance_check.sh` (script to extend)
   - `scripts/compare_m17_performance.sh` (script to extend)
   - `roadmap/milestone-17/PERFORMANCE_WORKFLOW.md` (documentation to update)
   - `CLAUDE.md` (workflow to update)

### Outputs from Task 007

1. **Modified Scripts**:
   - `scripts/m17_performance_check.sh` (added --competitive flag)
   - `scripts/compare_m17_performance.sh` (added competitive detection and positioning)

2. **Updated Documentation**:
   - `CLAUDE.md` (added step 12: competitive validation)
   - `roadmap/milestone-17/PERFORMANCE_WORKFLOW.md` (added competitive section)

3. **Runtime Artifacts**:
   - `tmp/m17_performance/competitive_<task>_<phase>_*.json` (results)
   - `tmp/m17_performance/competitive_<task>_<phase>_*_sys.txt` (system metrics)
   - `tmp/m17_performance/competitive_<task>_<phase>_*_diag.txt` (diagnostics)

4. **Tracking Data**:
   - `roadmap/milestone-17/PERFORMANCE_LOG.md` (updated with competitive results)

### Dependencies from Task 007

1. **Task 006** depends on Task 007:
   - Needs regression prevention in place before measuring initial baseline
   - Ensures baseline measurement is stable (not mid-regression)

2. **Task 008** (Documentation and Acceptance Testing) depends on Task 007:
   - Validates regression prevention workflow works end-to-end
   - Uses Task 007 scripts to verify competitive performance maintained

3. **Future M17 Tasks** (007-012) depend on Task 007:
   - Core graph operations use competitive validation
   - Regression prevention ensures no cumulative performance drift

## Document Metadata

- **Author**: Systems Architecture Optimizer (Margo Seltzer persona)
- **Date**: 2025-11-08
- **Version**: 1.0
- **Related Documents**:
  - `007_performance_regression_prevention_pending_ENHANCED.md` (implementation specification)
  - `007_ENHANCEMENT_SUMMARY.md` (architectural decisions and trade-offs)
