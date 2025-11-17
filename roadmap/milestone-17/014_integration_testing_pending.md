# Task 014: Dual Memory Integration Testing

**Status**: Pending
**Estimated Duration**: 4 days
**Dependencies**: Tasks 001-013
**Owner**: TBD

## Objective

Design a comprehensive integration test matrix for the dual-memory rollout covering migration correctness, feature-flag combinations, differential tests against the single-type engine, load/chaos scenarios, and backwards compatibility. The goal is to ensure the dual-memory feature flag can be enabled safely in production.

## Current Implementation Snapshot

- There are unit/property tests for confidence, binding, etc., but no end-to-end integration tests specific to dual-memory.
- Migration code exists but is untested in large scenarios.

## Technical Specification

### Test Suites to Implement

1. **Migration Correctness** (`engram-core/tests/dual_memory_migration.rs`): verify offline/online migrations preserve data, recall ordering, checkpoint/rollback behavior.
2. **Feature Flag Matrix** (`engram-cli/tests/dual_memory_feature_flags.rs`): run key workloads with each combination of relevant flags (e.g., dual memory, blending, fan-effect) to ensure no panics/regressions.
3. **Differential Testing** (`engram-core/tests/dual_memory_differential.rs`): run the same recall workloads through single-type and dual-type engines and assert top-K overlap and confidence within tolerances.
4. **Load & Chaos Tests** (`engram-core/tests/dual_memory_load.rs` + `dual_memory_chaos.rs`): concurrent writes + consolidations, inject simulated partitions via Task 010’s framework.
5. **Backwards Compatibility** (`engram-cli/tests/dual_memory_backcompat.rs`): ensure legacy clients (without the new feature flags) can still operate.

### Support Infrastructure

- `tests/support/dual_memory_fixtures.rs`: utilities to generate synthetic episodes, cues, embeddings deterministically.
- Load test tool (`tools/dual_memory_load_test.rs`) to soak test for 24h.

### Validation

- Ensure tests run in CI; some may be gated behind `--ignored` due to runtime.
- Document how to run longer chaos/load tests locally.

## Acceptance Criteria

1. Migration tests cover data integrity, recall ordering, checkpoint/rollback, and property-based embedding checks.
2. Feature flag tests exercise all relevant combinations without failures.
3. Differential tests demonstrate equivalence between single-type and dual-type engines for key workloads.
4. Load/chaos tests run using the network simulator and validate behavior under stress.
5. Backwards compatibility tests confirm legacy clients continue to work.
