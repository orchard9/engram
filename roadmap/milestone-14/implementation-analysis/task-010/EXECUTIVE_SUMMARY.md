# Task 010: Network Partition Testing Framework - Executive Summary

**Status**: Analysis Complete - Ready for Implementation  
**Date**: November 1, 2025  
**Codebase**: Engram (Rust 2024 Edition)

---

## What Was Found

### Pre-Existing Infrastructure (Ready to Use)

The Engram codebase already contains:

1. **Chaos Testing Framework** (`engram-core/tests/chaos/`)
   - Fault injectors for network delays, packet loss, clock skew
   - Validators for eventual consistency, sequence ordering, graph integrity
   - Pattern: `Arc<Mutex<>>` for shared test state
   - Deterministic RNG (seeded `StdRng`)

2. **Async Testing Infrastructure**
   - Tokio 1.47 with full features (in workspace)
   - `#[tokio::test]` macro available
   - Timeout utilities for test deadlines
   - 20+ async tests in codebase for patterns

3. **Test Utilities**
   - Builder pattern for test configuration (`GraphFixture`)
   - Helper modules for shared utilities
   - Deterministic test data generators

4. **All Required Dependencies**
   - `tokio`, `rand`, `serde`, `arc/mutex`, `chrono` - all present
   - No new dependencies needed for Task 010

### What Needs to Be Built

1. **NetworkSimulator Core** (~400 lines)
   - Logical clock for deterministic time
   - Message queue simulation
   - Fault injection system
   - Event recording for replay

2. **Test Infrastructure** (~500 lines)
   - Scenario DSL with builder pattern
   - Test orchestrator
   - 5 concrete test scenarios
   - Invariant validators

3. **Network Transport Layer** (~150 lines)
   - `NetworkTransport` trait
   - `SimulatedTransport` for testing
   - `RealUdpTransport` stub for future

4. **Documentation & CI** (~130 lines)
   - User guide for running tests
   - Developer guide for writing scenarios
   - Makefile targets
   - CI integration script

**Total Implementation**: ~1,500 lines of Rust code + documentation

---

## Key Architectural Decisions

### 1. Deterministic Simulation, Not Real Networking

**Why**: Allows perfect reproducibility and faster test iteration

**How**: 
- Use logical time (simulated clock), not wall-clock
- Seed RNG for deterministic packet drops
- Record all events for replay debugging
- Same seed = identical test execution

### 2. Test Double Pattern for Network Transport

**Why**: Tests use simulated transport, production uses real UDP

**Architecture**:
```
NetworkTransport (trait)
├── RealUdpTransport (production)
└── SimulatedTransport (tests, wraps NetworkSimulator)
```

This enables testing SWIM membership protocol and other distributed features without real networking.

### 3. Builder Pattern for Scenario Definition

**Why**: Declarative, readable test definitions

**Example**:
```rust
ChaosScenario::builder("clean_partition")
    .nodes(5)
    .duration(Duration::from_secs(60))
    .inject_fault(at, fault_spec, duration)
    .invariant(invariant_spec)
    .build()
```

### 4. Logical Clock for Determinism

**Why**: Real time is non-deterministic; logical time ensures reproducibility

**Implementation**:
- `clock: Arc<Mutex<u64>>` (milliseconds)
- `advance_time(ms)` progresses simulation
- Messages delivered when their scheduled time arrives
- No reliance on wall-clock time

---

## Integration Points with Existing Code

### 1. Chaos Testing Framework Expansion

**Current**: `tests/chaos/` has fault injectors and validators  
**New**: Add scenario DSL, orchestrator, and 5 test scenarios  
**Reuse**: Existing `DelayInjector`, `PacketLossSimulator`, `EventualConsistencyValidator`

### 2. Async Testing Patterns

**Current**: 20+ tests use `#[tokio::test]`, `Arc<Mutex<>>`, `timeout()`  
**New**: Partition scenarios follow same patterns  
**Alignment**: No special async handling needed; standard tokio patterns suffice

### 3. Test Utilities

**Current**: Builder patterns in `tests/support/`  
**New**: `ChaosScenarioBuilder` follows same pattern  
**Reuse**: Arc<Mutex<>> for shared simulator state

### 4. Dependencies

**Finding**: All needed dependencies already in workspace Cargo.toml  
- ✓ tokio 1.47
- ✓ rand 0.8
- ✓ serde 1.0
- ✓ chrono 0.4
- ✓ parking_lot, crossbeam, dashmap

**Action**: No new dependencies needed

---

## Implementation Approach

### Phase 1: Core Simulator (Day 1)
Create `engram-core/tests/network_simulator.rs` with:
- NetworkSimulator struct with clock, message queues, fault injectors
- Message delivery simulation with latency/packet loss
- Event recording for deterministic replay

### Phase 2: Scenario DSL (Day 1-2)
Create `engram-core/tests/chaos/scenario.rs` and `orchestrator.rs`:
- ChaosScenario builder for declarative tests
- FaultSpec enum for partition types
- ChaosOrchestrator to run tests and check invariants

### Phase 3: Concrete Scenarios (Day 2)
Create `engram-core/tests/partition_scenarios.rs` with:
- Clean partition (3|2 split)
- Asymmetric partition (one-way failure)
- Flapping partition (rapid cycles)
- Cascading failures (sequential node death)
- Network congestion (latency + packet loss)

### Phase 4: Transport Layer (Day 2-3)
Create `engram-core/src/cluster/`:
- NetworkTransport trait
- SimulatedTransport implementation
- RealUdpTransport stub

### Phase 5: CI & Docs (Day 3)
- Add `make chaos-test` targets
- Create user and developer guides
- CI integration script

---

## Success Criteria (from Task 010 spec)

1. **Deterministic Replay**
   - Same seed produces identical packet drops and delivery order
   - Events can be replayed exactly from recording
   - Failures reproduce 100% of the time

2. **All 5 Scenarios Implemented**
   - Each scenario has concrete Rust implementation
   - Each scenario tests specific failure mode
   - All scenarios pass consistently

3. **Invariant Validation**
   - Eventual consistency checker implemented
   - Data loss detector implemented
   - Split-brain detector implemented
   - Confidence bounds validator implemented

4. **CI Integration**
   - Chaos tests run in CI on every PR
   - Tests complete within 5 minutes
   - Failures include seed for reproduction

5. **Performance**
   - 5-node cluster 60s test: <5s wall-clock time
   - 100-node cluster 60s test: <30s wall-clock time
   - Event recording overhead <5%

6. **Documentation**
   - User guide for running tests
   - Developer guide for writing scenarios
   - CI integration instructions

---

## Critical Implementation Notes

### 1. Logical Time is Key
- Use `Arc<Mutex<u64>>` for milliseconds, not `Instant::now()`
- This ensures determinism across runs
- All message delivery based on logical time advancement

### 2. Seeded RNG for Reproducibility
- Use `StdRng::seed_from_u64(config.seed)` 
- Same seed in `SimulatorConfig` ensures same packet drops
- Store seed with test results for reproduction

### 3. Lock Management
- Always drop locks before acquiring other locks
- Use scoped blocks: `{ let lock = ...; } // drops here`
- Prevents deadlocks in message delivery

### 4. Message Ordering
- Deliver messages in order when multiple are ready
- Track message IDs for debugging
- Record all events (send, deliver, drop) for replay

### 5. Test Isolation
- Each test gets fresh NetworkSimulator instance
- No shared state between tests
- Use Arc to share simulator within single test

---

## Files to Create

```
New Source Files:
- engram-core/tests/network_simulator.rs          (400 lines)
- engram-core/tests/chaos/scenario.rs             (200 lines)
- engram-core/tests/chaos/orchestrator.rs         (300 lines)
- engram-core/tests/partition_scenarios.rs        (300 lines)
- engram-core/src/cluster/mod.rs                  (20 lines)
- engram-core/src/cluster/transport.rs            (100 lines)

Documentation:
- docs/guide/chaos-testing-quickstart.md          (100 lines)
- docs/howto/write-new-chaos-scenario.md          (80 lines)

CI/Scripts:
- scripts/run_chaos_tests.sh                      (30 lines)

Modified Files:
- engram-core/tests/chaos/mod.rs                  (add exports)
- engram-core/src/lib.rs                          (add cluster module)
- Makefile                                        (add chaos-test targets)
```

---

## Related Tasks (Dependency Graph)

### Task 010 → Task 003 (SWIM Membership)
- Task 010 provides NetworkSimulator and NetworkTransport trait
- Task 003 implements SWIM protocol using NetworkTransport
- Tests in Task 003 use NetworkSimulator for partition testing

### Task 010 → Task 011 (Jepsen Validation)
- Task 010 provides test framework and 5 scenarios
- Task 011 implements formal consistency validation using Jepsen principles
- Extends invariants and validators from Task 010

### Task 010 → Task 012 (Runbook)
- Task 010 provides reproduction procedures and debugging techniques
- Task 012 documents how to use chaos tests in production operations

---

## Risk Mitigation

### Risk: Simulator diverges from real network behavior
**Mitigation**: Validate against 3-node real cluster, tune latency parameters to match observed behavior

### Risk: Tests become flaky due to timing
**Mitigation**: Use logical time (simulated clock), not wall-clock time. All non-determinism is seeded.

### Risk: Deterministic replay breaks on code changes
**Mitigation**: Event log includes version metadata, warn on mismatch

### Risk: Chaos tests too slow for CI
**Mitigation**: Quick suite (5 scenarios, 60s each) for PR, full suite nightly

---

## Performance Targets

- 5-node cluster simulation (60s): < 5 seconds wall-clock time
- 100-node cluster simulation (60s): < 30 seconds wall-clock time
- Event recording overhead: < 5% of test time
- Memory overhead for 60s simulation: < 100MB

These are aggressive but achievable with:
- Logical time (no actual sleep)
- Deterministic RNG (one PRNG per simulator)
- Efficient message queues (VecDeque)
- Event log batching

---

## Getting Started

1. **Review existing code**:
   - `engram-core/tests/chaos/` - existing framework
   - `engram-core/tests/error_recovery_integration.rs` - async test patterns
   - `engram-core/tests/support/graph_builders.rs` - builder patterns

2. **Read Task 010 specification**:
   - `roadmap/milestone-14/010_network_partition_testing_framework_expanded.md`

3. **Implement in phases**:
   - Phase 1: NetworkSimulator core
   - Phase 2: Scenario DSL + Orchestrator
   - Phase 3: 5 test scenarios
   - Phase 4: Transport trait
   - Phase 5: CI + Documentation

4. **Validate with make quality**:
   - `cargo test --test partition_scenarios`
   - `cargo fmt && cargo clippy`
   - All warnings must be eliminated

---

## Deliverables Checklist

- [ ] NetworkSimulator with deterministic message delivery
- [ ] 5 concrete partition test scenarios (all passing)
- [ ] Scenario DSL with builder pattern
- [ ] Test orchestrator with invariant checking
- [ ] NetworkTransport trait and implementations
- [ ] Unit tests for simulator (determinism, message ordering)
- [ ] User guide documentation
- [ ] Makefile targets for chaos tests
- [ ] CI integration script
- [ ] All clippy warnings fixed
- [ ] Performance targets validated

---

## Estimated Timeline

**Best Case (experienced Rust developer, familiar with codebase)**: 3 days  
**Typical Case (good Rust skills, first time with this codebase)**: 4-5 days  
**With Debugging**: 5-7 days

Each phase should be code-reviewed and tested independently.

---

## References

The specification is based on:
- Kyle Kingsbury's Jepsen distributed systems testing (2013-present)
- Will Wilson's Deterministic Simulation Testing (FoundationDB, Strange Loop 2014)
- Netflix's Chaos Engineering Principles
- SWIM protocol (Das et al. 2002)
- Antithesis hypervisor-based determinism research

See `roadmap/milestone-14/010_network_partition_testing_framework_expanded.md` for full research citations.

