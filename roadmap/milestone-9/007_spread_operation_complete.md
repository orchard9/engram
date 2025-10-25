# Task 007: SPREAD Operation Implementation

**Status**: Pending
**Duration**: 1.5 days
**Dependencies**: Task 005
**Owner**: TBD

---

## Objective

Map SpreadQuery AST to ActivationSpread::spread_from() with configurable parameters (max_hops, decay_rate, threshold). Return activation paths as evidence.

---

## Files

`engram-core/src/query/executor/spread.rs`

---

## Acceptance Criteria

- [ ] SPREAD queries activate correct nodes
- [ ] Configurable parameters (hops, decay, threshold) work
- [ ] Activation paths included in evidence chain
- [ ] Performance: <5% overhead vs direct API call
- [ ] Integration tests pass
