# Task 010 Expansion Instructions

## Overview

The comprehensive expansion of Task 010 (Network Partition Testing Framework) has been completed and saved to:

**Location**: `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/010_network_partition_testing_framework_expanded.md`

## What Was Done

### Research Conducted

1. **Network Simulation Frameworks**
   - Jepsen (history-based linearizability checking)
   - Toxiproxy (TCP proxy fault injection)
   - Chaos Mesh (Kubernetes-native chaos engineering)
   - Madsim (deterministic simulation in Rust)

2. **Deterministic Simulation Testing**
   - FoundationDB's approach (Will Wilson, Strange Loop 2014)
   - Antithesis hypervisor-based determinism
   - Perfect reproducibility through controlled non-determinism

3. **Chaos Engineering Patterns**
   - 5 partition scenarios from Jepsen literature
   - Fault injection types and their impact
   - Invariant-based validation

### What Was Created

A comprehensive 700+ line task specification including:

1. **Research Foundation** (150 lines)
   - Oracle problem in distributed systems
   - Three simulation paradigms comparison
   - Academic paper references (SWIM, Jepsen, DST)
   - Why hybrid approach for Engram

2. **Technical Specification** (400 lines)
   - Complete `NetworkSimulator` with fault injection
   - `FaultInjector` with 7 fault types (packet loss, latency, partition, throttle, reorder, duplicate, corrupt)
   - Deterministic replay via event recording
   - Scenario definition DSL
   - Test orchestrator

3. **Concrete Implementations** (150 lines)
   - Scenario 1: Clean 3|2 partition
   - Scenario 2: Asymmetric partition (A→B works, B→A fails)
   - Scenario 3: Flapping partition (10 rapid cycles)
   - Scenario 4: Cascading failures (sequential node deaths)
   - Scenario 5: Network congestion (latency + packet loss)

4. **Testing Strategy** (100 lines)
   - Unit tests for determinism validation
   - Integration tests with SWIM
   - Property-based tests for eventual consistency

5. **CI Integration** (50 lines)
   - Makefile targets (`make chaos-test`)
   - Shell script with JSON reporting
   - GitHub Actions workflow (optional)

## How to Apply This Expansion

### Option 1: Replace in Existing File

Replace lines 195-224 in `/Users/jordan/Workspace/orchard9/engram/roadmap/milestone-14/004-012_remaining_tasks_pending.md` with the contents of the expanded file.

**Current lines to replace:**
```markdown
## Task 010: Network Partition Testing Framework (3 days)

**Objective**: Build testing framework for simulating network failures.

**Key Components**:
- `NetworkSimulator` with configurable partition scenarios
- Packet loss, latency injection, partition simulation
- Deterministic replay for debugging
- Chaos testing harness

**Test Scenarios**:
1. Clean split (two halves can't communicate)
2. Asymmetric partition (A→B works, B→A fails)
3. Flapping partition (intermittent failures)
4. Cascading failures (nodes fail sequentially)
5. Network congestion (high latency, packet loss)

**Files**:
- `engram-core/tests/network_simulator.rs`
- `engram-core/tests/partition_scenarios.rs`
- `engram-core/tests/chaos/mod.rs`

**Acceptance Criteria**:
- All 5 scenarios testable
- Deterministic replay from seed
- Integration with existing test suite
- CI runs subset of chaos tests
```

**Replace with:** Contents of `010_network_partition_testing_framework_expanded.md`

### Option 2: Create Standalone Task File

Alternatively, create a standalone task file:
```bash
mv roadmap/milestone-14/010_network_partition_testing_framework_expanded.md \
   roadmap/milestone-14/010_network_partition_testing_framework_pending.md
```

Then update the summary file to reference the standalone task.

## Key Improvements Over Original

### Original (30 lines)
- High-level summary
- No research foundation
- No concrete implementations
- No deterministic replay design

### Expanded (700+ lines)
- Comprehensive research foundation with academic citations
- Complete data structures with Rust code
- 5 fully implemented test scenarios
- Deterministic replay mechanism with event recording
- Scenario definition DSL for declarative tests
- Test orchestrator with invariant checking
- CI integration strategy with scripts
- Performance targets and risk mitigation

## Structure Matches Tasks 001-003

The expansion follows the exact structure of the comprehensive tasks:

- ✅ Research Foundation section (like Task 001)
- ✅ Technical Specification with core data structures (like Task 001)
- ✅ Core Operations with Rust implementations (like Task 001)
- ✅ Concrete test scenarios (5 scenarios implemented)
- ✅ Files to Create/Modify lists
- ✅ Testing Strategy with unit/integration/property tests
- ✅ Dependencies section
- ✅ Acceptance Criteria (6 detailed criteria)
- ✅ CI Integration section
- ✅ Performance Targets
- ✅ References section

## Validation Checklist

Before marking as complete, verify:

- [ ] All 5 scenarios have concrete Rust implementations
- [ ] Deterministic replay works (same seed = same results)
- [ ] Event log records all non-deterministic inputs
- [ ] Simulator controls time, randomness, and message delivery
- [ ] Invariant checkers validate eventual consistency, no data loss, no split-brain
- [ ] CI integration runs chaos tests on every PR
- [ ] Documentation explains how to write new scenarios
- [ ] Performance targets met (<10s for 60s simulation)

## Next Steps

1. Review the expanded task file
2. Apply to the milestone planning document
3. Share with rust-graph-engine-architect agent for technical review
4. Share with verification-testing-lead agent for testing strategy review
5. Begin implementation once approved

## Estimated Implementation Effort

With this comprehensive specification:
- Day 1: Implement `NetworkSimulator` core (300 lines)
- Day 2: Implement scenario DSL and orchestrator (400 lines)
- Day 3: Implement all 5 test scenarios + CI integration (300 lines)

Total: 3 days (as estimated in original task)

## Notes

This expansion provides everything needed to implement Task 010 without further design work. The specification is production-ready and includes:

- Complete API surface with method signatures
- Error handling strategy
- Deterministic replay mechanism
- CI integration scripts
- Performance benchmarks
- Risk mitigation strategies

The implementation can proceed directly from this specification.
