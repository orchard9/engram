# Task 011: Jepsen-Style Consistency Testing

**Status**: Pending
**Estimated Duration**: 4-5 days
**Dependencies**: Task 010 (Network Partition Testing Framework), Task 009 (Distributed Query Execution)
**Owner**: TBD

## Objective

Use the network simulator (Task 010) plus history-based checking to validate Engram’s eventual-consistency guarantees. We need workloads that generate STORE/RECALL operations across partitions, a nemesis that injects partitions/drops packets, and checkers that verify convergence, no acknowledged-write loss, and bounded staleness.

## Current Implementation Snapshot

- No history logging exists; API handlers don’t emit operation traces.
- There is no workload harness to drive multiple nodes in tests.
- Task 010’s simulator is not yet built, so this task depends on it.

## Technical Specification

### Components

1. **Workload driver** (`tests/jepsen/workload.rs`): spawns multiple async clients performing randomized STORE/RECALL/RECALL_SIMILAR operations against the simulated cluster.
2. **Nemesis**: uses `NetworkSimulator` to apply scripted failures (same as Task 010). Expose a trait so tests can plug different schedules.
3. **History recorder**: capture every operation with fields `{op_id, process_id, op_type, start_time, end_time, result}` and persist to a JSON log for analysis.
4. **Checkers**:
   - **Eventual convergence**: After the nemesis heals and we wait Δ (configurable), reads from all nodes should return identical results for the same `memory_id`.
   - **No acknowledged-write loss**: Any STORE that returned success must be readable on some node after healing.
   - **Bounded staleness**: Measure read age distribution and ensure it stays below configured thresholds (e.g., 95% of reads see writes ≤ 5s old).
   - **Conflict detection**: verify `PartitionDetector` prevents multi-primary behavior (no conflicting responses with same `memory_id`).

### History Format

Use a simple JSONL schema (compatible with Elle-like analysis later):

```json
{
  "op_id": 42,
  "process": 1,
  "op": "store",
  "space": "alpha",
  "memory_id": "mem123",
  "start": 1670000000.123,
  "end": 1670000000.456,
  "ok": true,
  "value": { "content": "hello" }
}
```

### Integration with Simulator

- Tests run entirely in-process: spin up simulated nodes (SWIM + router + storage) using `SimulatedTransport`.
- Workload driver calls HTTP/gRPC handlers directly or through router.
- Nemesis manipulates the simulator while workload runs (e.g., partition nodes 1/2 from 3/4 every 5 seconds).

### Phases per Test

1. **Warmup**: run workload without faults for X seconds to populate data.
2. **Chaos**: enable nemesis for Y seconds.
3. **Healing**: remove faults, continue workload for Z seconds.
4. **Validation**: run checkers on collected history.

### Metrics/Output

- Publish summary after each run: counts of completed ops, failures, detected anomalies.
- If a checker fails, print the minimal offending history (subset of ops) to aid debugging.

### Testing Strategy

- Implement a base scenario with majority/minority partition + read/write workload and ensure checkers pass.
- Add targeted tests for each failure pattern (asymmetric, flapping, cascading, partial).
- Include a regression test that intentionally injects a bug (e.g., disable partition detector) to ensure checkers catch it.

## Acceptance Criteria

1. Workload harness + nemesis run deterministically with the simulator.
2. Operation histories are captured and can be replayed/analyzed offline.
3. Checkers detect violations (fail tests) when we disable key safety mechanisms.
4. Default scenario passes, demonstrating Engram’s eventual consistency under simulated partitions.
5. Documentation describes how to run the Jepsen-style tests and interpret results.
