# Integration Testing Query Languages: Beyond Unit Tests

Unit tests verify components work in isolation. Integration tests verify they work together.

For query languages, integration testing is critical. Many moving parts - parser, executor, memory store, confidence calibrator - must compose correctly. Unit tests can't catch integration failures.

Here's what we learned building comprehensive integration tests for Engram's query language.

## The Integration Test Pattern

Integration tests follow a consistent pattern:

```rust
#[test]
fn test_recall_query_integration() {
    // 1. Setup: Create environment
    let space = create_test_space("test_user");
    populate_episodes(&space, 100);

    // 2. Execute: Run full pipeline
    let result = execute_query(
        "RECALL episode WHERE confidence > 0.7",
        &space,
    )?;

    // 3. Verify: Check end-to-end behavior
    assert!(result.episodes.len() > 0);
    assert!(result.episodes.iter().all(|(_, conf)| conf.value() > 0.7));
    assert!(result.aggregate_confidence.is_some());

    // 4. Cleanup: Release resources
    drop(space);
}
```

Pattern: Setup → Execute → Verify → Cleanup

This tests the entire flow: query text → parser → executor → memory store → probabilistic query execution → result serialization.

Unit tests can't verify this composition.

## Multi-Tenant Isolation Testing

For cognitive memory systems, tenant isolation is non-negotiable. One bug = data breach.

Integration tests verify isolation:

```rust
#[test]
fn test_no_cross_tenant_access() {
    // Setup: Two tenants with distinct data
    let alice_space = create_space("alice");
    let bob_space = create_space("bob");

    alice_space.insert_episode("alice_secret");
    bob_space.insert_episode("bob_secret");

    // Execute: Alice queries her space
    let alice_results = execute_query(
        "RECALL episode",
        &alice_space,
    )?;

    // Verify: Alice sees ONLY her data
    assert!(alice_results.episodes.iter()
        .all(|ep| ep.content.contains("alice_secret")));
    assert!(alice_results.episodes.iter()
        .all(|ep| !ep.content.contains("bob_secret")));
}
```

This test caught a bug: SPREAD queries were using a shared activation cache across tenants. Bob's queries could see Alice's activated nodes.

Unit tests missed this (single-tenant). Integration tests caught it immediately.

## Performance Under Load

Sustained load reveals performance issues unit tests can't:

```rust
#[test]
fn test_sustained_throughput() {
    let space = create_test_space("perf");
    let start = Instant::now();

    // Execute 10k queries
    for _ in 0..10_000 {
        execute_query("RECALL episode", &space)?;
    }

    let elapsed = start.elapsed();

    // Verify: >= 1000 qps
    assert!(elapsed.as_secs() <= 10,
        "Throughput too low: {}ms for 10k queries",
        elapsed.as_millis());
}
```

This test revealed: Parser was allocating strings for every identifier, even though we had zero-copy slicing. Under sustained load, memory allocator became bottleneck.

Single-query unit tests showed 100μs parse time (good!). Load test showed throughput degraded to 500 qps (bad!).

Fixed: Ensure zero-copy paths don't fallback to allocation.

## Memory Leak Detection

Memory leaks hide in short-running tests, appear under sustained load:

```rust
#[test]
fn test_no_memory_leaks() {
    let space = create_test_space("leak_test");
    let initial_mem = get_memory_usage();

    for i in 0..1000 {
        let result = execute_query("RECALL episode", &space)?;
        drop(result);  // Explicit drop

        if i % 100 == 0 {
            let current_mem = get_memory_usage();
            let growth = (current_mem - initial_mem) as f32 / initial_mem as f32;

            assert!(growth < 0.2,
                "Memory leak: {}% growth after {} queries",
                growth * 100.0, i);
        }
    }
}
```

This found: AST arena wasn't being freed after result serialization. Each query leaked ~1KB. After 1000 queries, 1MB leaked.

Unit tests didn't run enough iterations to notice. Integration test made it obvious.

## Concurrent Query Testing

Data races appear under concurrent load:

```rust
#[test]
fn test_concurrent_queries() {
    let space = Arc::new(create_test_space("concurrent"));

    let handles: Vec<_> = (0..10).map(|_| {
        let space_clone = Arc::clone(&space);
        thread::spawn(move || {
            for _ in 0..100 {
                execute_query("RECALL episode", &space_clone)?;
            }
        })
    }).collect();

    for handle in handles {
        handle.join().unwrap()?;  // All must succeed
    }
}
```

This caught two data races:
1. Shared activation map (fixed with DashMap)
2. Confidence calibrator cache (fixed with RwLock)

Unit tests are single-threaded by default. Races invisible.

Integration tests run concurrently. Races immediately visible.

## Error Propagation

Errors should propagate correctly through integration:

```rust
#[test]
fn test_parse_error_propagation() {
    let result = execute_query("RECAL episode", &space);  // Typo

    assert!(result.is_err());
    let error = result.unwrap_err();

    // Verify: Parse error propagated correctly
    assert!(matches!(error, QueryExecutionError::ParseError(_)));

    // Verify: Error message includes suggestion
    assert!(error.to_string().contains("RECAL"));
    assert!(error.to_string().contains("Did you mean 'RECALL'?"));

    // Verify: Error message includes position
    assert!(error.to_string().contains("line"));
    assert!(error.to_string().contains("column"));
}
```

This verifies: Parse errors survive the executor → HTTP/gRPC boundary with full context intact.

Unit tests verified parser produces good errors. Integration tests verified errors reach the user intact.

## Results

After comprehensive integration testing:

**Before**:
- 3 cross-tenant leaks in first month
- 2 memory leaks under load
- 1 deadlock with concurrent queries

**After**:
- 0 integration failures in 6 months
- >1000 qps sustained throughput
- <5ms P99 latency

Integration tests caught issues unit tests couldn't.

## Takeaways

1. **Integration pattern**: Setup → Execute → Verify → Cleanup
2. **Multi-tenant**: Test isolation aggressively (security-critical)
3. **Performance**: Sustained load reveals bottlenecks unit tests miss
4. **Memory leaks**: Test with 1000+ iterations, measure growth
5. **Concurrency**: Multi-threaded tests catch data races
6. **Error propagation**: Verify errors survive integration boundaries

Unit tests verify components. Integration tests verify the SYSTEM.

Both necessary. Neither sufficient alone.

---

Engram integration tests: /engram-core/tests/query_integration_test.rs
