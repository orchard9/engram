# Task 011 Expansion - Replacement Instructions

## Overview

The expanded Task 011 (Jepsen-Style Consistency Testing) is now available in:
**`/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/011_jepsen_consistency_testing_expanded.md`**

This expansion transforms the original 65-line task description (lines 226-291) into a comprehensive 800+ line implementation specification with complete code examples.

## What Was Added

### 1. Research Foundation (Significantly Expanded)
- **Jepsen Methodology**: Detailed explanation of history-based verification, black-box testing approach
- **Consistency Models**: Clear table comparing linearizability vs eventual consistency, what Engram guarantees vs doesn't guarantee
- **Elle Checker**: Complete algorithm explanation (dependency graphs, cycle detection, minimal counterexamples)
- **Knossos Linearizability**: World exploration, state-space search, optimization techniques
- **PBS (Probabilistically Bounded Staleness)**: Quantifying eventual consistency with probabilistic bounds
- **Nemesis Strategies**: Comprehensive coverage of network partitions, clock skew, process failures, combined failures

### 2. Complete Rust Implementations (New)

**Operation History Recording** (`engram-core/src/cluster/jepsen/history.rs`):
- `HistoryEvent` enum: Invoke/Complete events with wall-clock timestamps
- `Operation` enum: Store/Recall/RecallSimilar/Consolidate operations
- `Outcome` enum: Ok/Fail/Info (handles network partition uncertainty)
- `HistoryRecorder` struct: Thread-safe lock-free history collection
- JSON export for Jepsen analysis

**Consistency Checker** (`engram-core/src/cluster/jepsen/checker.rs`):
- `ConsistencyViolation` enum: DataLoss, ConvergenceFailure, ConfidenceCalibrationError, SplitBrain
- `EventualConsistencyChecker`: Implements 4 core validation algorithms:
  1. `check_no_data_loss()`: Verify all acknowledged writes survive partition healing
  2. `check_bounded_convergence()`: Verify convergence within 60s
  3. `check_confidence_calibration()`: Verify confidence scores match actual correctness
  4. `check_split_brain_resolution()`: Verify deterministic conflict resolution

### 3. Complete Clojure Implementations (New)

**Jepsen Test Harness** (`jepsen/engram/src/engram/core.clj`):
- `db` lifecycle: Install Rust, clone Engram, configure cluster, start daemons
- `workload`: 3-phase test (population, partition + writes, convergence check)
- `engram-test`: Main test composition with checkers and nemesis
- CLI integration with `lein run`

**Engram Client** (`jepsen/engram/src/engram/client.clj`):
- HTTP client using Java HttpClient (5s timeout)
- Operations: `:store`, `:recall`, `:consolidate`
- Generators: `store-gen`, `recall-gen`, `consolidate-gen`, `recall-all-gen`
- Error handling: timeout → :info, exceptions → :fail

**Custom Checkers** (`jepsen/engram/src/engram/checker.clj`):
- `eventual-consistency`: Verify all nodes converge to same state
- `no-data-loss`: Verify no acknowledged writes lost
- `bounded-convergence`: Verify convergence within 60s
- `confidence-calibration`: Verify success rate > 90%

**Nemesis** (`jepsen/engram/src/engram/nemesis.clj`):
- `partition-nemesis`: Network partition using bridge grudge
- `full-nemesis`: Combined faults (partition, kill, pause, clock)

### 4. System Architecture Diagram (New)
Visual representation of Jepsen control node → Engram nodes → history aggregation flow

### 5. Files to Create (Detailed List)
**Rust**: 5 files (mod.rs, history.rs, checker.rs, export.rs, jepsen_harness.rs)
**Clojure**: 6 files (project.clj, core.clj, client.clj, checker.clj, nemesis.clj, cluster.toml)
**Infrastructure**: 3 files (docker-compose.yml, run_jepsen.sh, jepsen-testing.md)

### 6. Files to Modify (New Section)
5 files with specific modifications needed

### 7. Testing Strategy (Comprehensive)

**Local Simulation**: Complete Rust test showing 3-node in-process cluster with partition injection

**Full Jepsen Suite**: Bash script for CI integration (5 nodes, 300s time limit, 10 concurrent clients)

**CI/CD Integration**: Makefile target `make jepsen`

### 8. Dependencies (Detailed)

**Rust**: crossbeam-queue (already have)

**Clojure**: Complete project.clj with Jepsen 0.3.8, Cheshire 5.12.0

**System**: JDK 17+, Leiningen 2.9+, Gnuplot, Graphviz, SSH

### 9. Acceptance Criteria (Quantified)
10 specific criteria with measurable thresholds:
- 1000 test runs zero violations
- 100% write survival
- 99% convergence within 60s
- 10% confidence calibration tolerance
- 5-minute CI timeout
- 1M operation history support

### 10. Performance Targets (New Section)
- <1% CPU overhead
- <100MB memory per node
- <2 minutes analysis time for 100K operations
- <5 minutes CI, <30 minutes full suite

### 11. Integration Points (New Section)
- Hook into Task 010 NetworkSimulator
- Export metrics to Prometheus
- Reuse existing monitoring infrastructure

### 12. References (New Section)
Academic papers, implementation resources, Engram-specific docs

## Comparison: Before vs After

| Aspect | Original (65 lines) | Expanded (800+ lines) |
|--------|-------------------|----------------------|
| Research depth | 4 paragraphs | 6 comprehensive sections with academic citations |
| Code examples | 1 Clojure pseudocode | 10 complete implementations (Rust + Clojure) |
| Data structures | None | 8 core Rust types, 4 Clojure namespaces |
| Algorithms | Mentioned | Fully implemented with comments |
| Test strategy | Generic | 3 levels (local, full, CI) with complete code |
| Dependencies | Generic | Specific versions with justification |
| Acceptance criteria | 4 generic items | 10 quantified metrics |
| Integration | Not specified | Detailed hooks into existing codebase |
| References | None | 8 academic papers + implementation guides |

## How to Use This Expansion

### Option 1: Replace Section in Original File

Edit `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/004-012_remaining_tasks_pending.md`:

1. Delete lines 226-291 (current Task 011)
2. Insert content from `011_jepsen_consistency_testing_expanded.md` at line 226

### Option 2: Create Standalone Task File (Recommended)

Keep the standalone file as the canonical reference:
- `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/011_jepsen_consistency_testing_expanded.md`

Update the main file to reference it:
```markdown
## Task 011: Jepsen-Style Consistency Testing (4-5 days)

See detailed implementation specification in:
`roadmap/milestone-14/011_jepsen_consistency_testing_expanded.md`

**Summary**: Formal validation of distributed consistency properties using Jepsen framework.
Implements history-based verification, nemesis fault injection, and eventual consistency checking.
Includes complete Rust history recorder, Clojure test harness, and CI integration.
```

## Key Improvements from Original

1. **Actionable Implementation**: Every component has complete code, not pseudocode
2. **Academic Rigor**: Cites original papers, explains algorithms from first principles
3. **Engram-Specific**: Adapted for eventual consistency vs linearizability
4. **Production-Ready**: CI integration, performance targets, operational runbook
5. **Testable**: Local simulation for fast iteration, full suite for comprehensive validation
6. **Educational**: Teaches Jepsen methodology while providing implementation

## Next Steps

1. Review expanded task with team for technical accuracy
2. Decide on file organization (standalone vs inline)
3. Begin implementation following the file creation order
4. Use local simulation tests first before full Jepsen cluster
5. Integrate with Task 010 (Network Simulator) as specified
6. Document any deviations from specification in task file

---

**Author**: Professor John Regehr (verification-testing-lead agent)
**Date**: 2025-11-01
**Milestone**: M14 Distributed System Implementation
**Task**: 011 - Jepsen-Style Consistency Testing
