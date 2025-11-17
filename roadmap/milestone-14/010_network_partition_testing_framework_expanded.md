# Task 010: Network Partition Testing Framework

**Status**: Pending
**Estimated Duration**: 3 days
**Dependencies**: Task 001 (SWIM), Task 003 (Partition handling), Task 006 (Routing)
**Owner**: TBD

## Objective

Create a deterministic network simulation and fault-injection harness so we can test SWIM, routing, replication, and gossip logic under partitions, latency spikes, and packet loss. The framework should support scripted scenarios (minority/majority splits, flapping links, asymmetric partitions) and expose assertions about membership convergence, partition detector behavior, and routing outcomes.

## Current Implementation Snapshot

- No simulation harness exists. SWIM integration tests only spin up real sockets on localhost (slow, non-deterministic).
- There is no automated way to reproduce network partitions or log sequences for replay.

## Technical Specification

### Requirements

1. **Deterministic Simulation**: Provide a `SimulatedTransport` that implements the same trait as the real UDP transport but schedules deliveries via a simulated clock and seeded RNG.
2. **Scenario DSL**: Allow tests to define sequences like “partition nodes A/B from C/D for 30s, then heal” or “drop packets from node X to node Y only”.
3. **Assertions**: Provide helpers to assert membership convergence, partition detector state, router error behavior, etc.
4. **Replay**: Optionally record event logs (messages sent, delivered, dropped) and replay them for debugging.

### Implementation Plan

1. **Abstract Transport**: Add a `ClusterTransport` trait (`send`, `recv`) and implement both `UdpTransport` (existing) and `SimulatedTransport` for tests.
2. **Network Simulator Module** (`tests/network_simulator.rs` or `engram-core/src/test_support/network.rs`):
   - Simulated clock (`u64 ms`)
   - Message queues per destination
   - Fault injectors per link with drop/latency configs
   - Scenario runner that manipulates fault injectors over simulated time
3. **Scenario Definitions**: Provide a simple builder API:

```rust
enum FaultType { DropAll, Delay { ms: u64 }, Asymmetric, Flapping { period_ms: u64 }, Cascading }
struct ScenarioStep { start_ms: u64, end_ms: u64, fault: FaultType, nodes: Vec<NodeId> }
```

4. **Assertions**:
   - `assert_membership_converges(sim, duration_ms)`
   - `assert_partition_detected(detector, expected_state)`
   - `assert_router_confidence(decision, expected_penalty)`

5. **Integration Tests**: Cover the five key Jepsen-style partitions:
   - Majority/minority split
   - Asymmetric partition
   - Flapping partition
   - Cascading failures
   - Partial partitions (3-way split)

6. **Deterministic RNG**: Seed `StdRng` per test so results are reproducible.

### Observability

- Record event logs (JSON) for failed runs so we can replay.
- Provide metrics (counts of dropped/delayed packets) to feed into `cargo test` output or CLI logs.

### Testing Strategy

- Unit tests for `SimulatedTransport` ensuring deterministic delivery ordering.
- Scenario tests verifying SWIM marks nodes dead/suspect as expected under each pattern.
- Routing tests ensuring partition detector triggers local-only mode when the simulator injects faults.

## Acceptance Criteria

1. `SimulatedTransport` can replace real UDP in tests without affecting production code.
2. Scenario builder can express at least the five partition patterns listed above.
3. SWIM membership tests run under the simulator and validate suspect/dead transitions deterministically.
4. Partition detector and router tests use the simulator to verify behavior under partitions.
5. Event logging/replay is available for debugging failing scenarios.
