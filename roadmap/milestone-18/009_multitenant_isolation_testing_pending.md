# Task 009: Multi-Tenant Isolation Testing

**Status**: Pending
**Estimated Duration**: 3-4 days
**Priority**: Medium - Validates memory space isolation

## Objective

Measure cross-space interference under concurrent multi-tenant load. Validate <1% performance impact when neighbor space experiences 10x load spike. Ensure spreading activation never crosses space boundaries.

## Test Scenarios

1. **Noisy Neighbor**: Space A runs 10x load, measure impact on Space B baseline
2. **Balanced Load**: 10 spaces with uniform load (isolation baseline)
3. **Cascading Failure**: Space A crashes, verify other spaces unaffected

## Key Metrics

```rust
pub struct IsolationMetrics {
    pub cross_space_interference_pct: f64,  // Target <1%
    pub memory_isolation: bool,              // No shared nodes
    pub spreading_isolation: bool,           // No cross-space activation
    pub independent_failure: bool,           // Crash doesn't propagate
}
```

## Success Criteria

- **Performance Isolation**: <1% latency impact from noisy neighbor
- **Memory Isolation**: Zero shared nodes between spaces
- **Failure Isolation**: Space crash doesn't affect others
- **Spreading Isolation**: Activation never crosses boundaries (property test)

## Files

- `tools/loadtest/src/multitenant/isolation_tester.rs` (380 lines)
- `scenarios/multitenant/noisy_neighbor.toml`
- `engram-core/tests/space_isolation_property_tests.rs` (250 lines)
