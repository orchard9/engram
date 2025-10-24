# Research: Integration Testing Query Languages

## Key Findings

### 1. Integration Test Scope

**Unit Tests**: Individual components (parser, executor, etc.)
**Integration Tests**: End-to-end flow (query text → parse → execute → result)

Integration tests verify:
- Parser → Executor integration
- Executor → MemoryStore integration
- Multi-tenant isolation
- Performance under load
- No memory leaks

### 2. Test Flow Pattern

```rust
#[test]
fn test_recall_integration() {
    // 1. Setup: Create memory space with test data
    let space = create_test_space("user_123");
    populate_test_episodes(&space, 100);

    // 2. Execute: Run query end-to-end
    let result = execute_query(
        "RECALL episode WHERE confidence > 0.7",
        &space,
    )?;

    // 3. Verify: Check results match expectations
    assert!(result.episodes.len() > 0);
    assert!(result.episodes.iter().all(|(_, conf)| conf.value() > 0.7));

    // 4. Cleanup: Drop space
    drop(space);
}
```

Pattern: Setup → Execute → Verify → Cleanup

### 3. Multi-Tenant Isolation Testing

**Critical**: Queries must not access other tenants' data

```rust
#[test]
fn test_tenant_isolation() {
    let space_a = create_space("tenant_a");
    let space_b = create_space("tenant_b");

    populate_episodes(&space_a, vec!["secret_a"]);
    populate_episodes(&space_b, vec!["secret_b"]);

    // Query from tenant_a should NOT see tenant_b's data
    let result = execute_query("RECALL episode", &space_a)?;
    assert!(result.episodes.iter().all(|ep| ep.content.contains("secret_a")));
    assert!(result.episodes.iter().all(|ep| !ep.content.contains("secret_b")));
}
```

**Failure mode**: Cross-tenant data leakage (catastrophic security issue)

### 4. Performance Under Load

**Sustained Load Testing**:
```rust
#[test]
fn test_sustained_load() {
    let space = create_test_space("perf_test");

    // Execute 10k queries over 10 seconds
    let start = Instant::now();
    for _ in 0..10_000 {
        let _ = execute_query("RECALL episode", &space)?;
    }
    let elapsed = start.elapsed();

    // Verify: Throughput >= 1000 qps
    assert!(elapsed.as_secs() <= 10);

    // Verify: No memory leaks
    let memory_after = get_memory_usage();
    assert!(memory_after < initial_memory * 1.1);  // <10% growth
}
```

Target: 1000 queries/sec, <10% memory growth

### 5. Memory Leak Detection

**Strategy**: Run queries in loop, measure memory growth

```rust
#[test]
fn test_no_memory_leaks() {
    let initial_mem = get_memory_usage();

    for i in 0..1000 {
        let result = execute_query("RECALL episode", &space)?;
        drop(result);  // Explicit drop to verify cleanup

        if i % 100 == 0 {
            let current_mem = get_memory_usage();
            let growth = (current_mem - initial_mem) as f32 / initial_mem as f32;
            assert!(growth < 0.2, "Memory leak detected: {}% growth", growth * 100.0);
        }
    }
}
```

**Tools**: Valgrind, Miri (Rust's interpreter with leak detection)

### 6. Error Propagation Testing

Errors should propagate correctly through integration:

```rust
#[test]
fn test_parse_error_propagation() {
    let result = execute_query("RECAL episode", &space);  // Typo

    assert!(result.is_err());
    let error = result.unwrap_err();

    assert!(matches!(error, QueryExecutionError::ParseError(_)));
    assert!(error.to_string().contains("RECAL"));
    assert!(error.to_string().contains("RECALL"));  // Suggestion
}
```

Verify error messages survive integration boundary.

### 7. Concurrent Query Testing

**Critical**: Multiple queries must not interfere

```rust
#[test]
fn test_concurrent_queries() {
    let space = Arc::new(create_test_space("concurrent"));

    // Spawn 10 threads, each executing 100 queries
    let handles: Vec<_> = (0..10).map(|_| {
        let space_clone = Arc::clone(&space);
        thread::spawn(move || {
            for _ in 0..100 {
                let _ = execute_query("RECALL episode", &space_clone)?;
            }
            Ok::<_, QueryExecutionError>(())
        })
    }).collect();

    // All threads should complete without error
    for handle in handles {
        handle.join().unwrap()?;
    }
}
```

Verify: No data races, no deadlocks, correct results.

### 8. Integration with Pattern Completion (Milestone 8)

When M8 complete, test IMAGINE integration:

```rust
#[test]
#[cfg(feature = "pattern_completion")]
fn test_imagine_integration() {
    let result = execute_query(
        "IMAGINE episode BASED ON partial_episode NOVELTY 0.3",
        &space,
    )?;

    assert!(result.episodes.len() > 0);
    assert!(result.episodes[0].confidence.value() > 0.5);
}
```

Feature-gated until M8 complete.

## Synthesis

Integration testing verifies:
1. **End-to-end flow**: Query → Parse → Execute → Result
2. **Multi-tenant isolation**: No cross-tenant data access
3. **Performance**: 1000 qps sustained, <5ms P99 latency
4. **Memory safety**: No leaks under load
5. **Error propagation**: Errors survive integration boundaries
6. **Concurrency**: No data races, correct results
7. **Pattern completion**: IMAGINE works when M8 ready

Goal: Catch integration issues that unit tests miss.

## References

1. "Testing Strategies in a Microservice Architecture", Toby Clemson
2. "Property-Based Testing", Claessen & Hughes (integration properties)
3. "Rust Memory Safety", Rust Book (Valgrind, Miri)
4. "Concurrency Testing", Herlihy & Shavit (concurrent correctness)
