# Perspectives: Integration Testing Query Languages

## Verification Testing Lead Perspective

Unit tests verify components. Integration tests verify CONNECTIONS.

**Example**: Parser tests verify AST correctness. Executor tests verify execution logic. But do they work TOGETHER?

Integration test catches:
- Parser produces AST executor can't handle
- Executor expects fields parser doesn't provide
- Performance degrades when composed

Pattern: Setup → Execute full pipeline → Verify → Cleanup

Critical for query languages: Many moving parts, must compose correctly.

## Systems Architecture Perspective

Multi-tenant isolation is non-negotiable. One bug = data breach.

**Test strategy**:
```rust
// Populate two tenants with distinct data
space_a.insert("secret_a");
space_b.insert("secret_b");

// Query from space_a
let result = query("RECALL episode", space_a)?;

// Verify: ONLY space_a data returned
assert_no_leakage(result, "secret_b");
```

Test every code path: RECALL, SPREAD, PREDICT, etc.

One leaked episode = security vulnerability. Test aggressively.

## Rust Graph Engine Perspective

Concurrency bugs hide in unit tests, surface in integration tests.

**Scenario**: 100 concurrent queries
- Unit test: Single-threaded, no races
- Integration test: Multi-threaded, races visible

Found 2 data races during concurrent integration testing:
1. Shared activation map (fixed with DashMap)
2. Confidence calibrator cache (fixed with RwLock)

Unit tests missed both. Integration caught both.

## Memory Systems Perspective

Performance under sustained load reveals memory leaks.

**Pattern**: Run 10k queries, measure memory growth

Acceptable: <10% growth (caching, buffering)
Unacceptable: >50% growth (leak)

Found leak: AST nodes not freed after query execution.
Fixed: Explicit arena drop after result serialization.

Leak invisible in single-query unit test. Obvious in load test.
