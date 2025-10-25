# Task 005: Query Executor Infrastructure

**Status**: Pending
**Duration**: 2 days
**Dependencies**: Task 003 (Parser)
**Owner**: TBD

---

## Objective

Build executor that maps parsed AST to existing engine operations (MemoryStore, ActivationSpread, ProbabilisticQueryExecutor) with multi-tenant isolation and evidence tracking.

---

## Files to Create

1. `engram-core/src/query/executor/mod.rs` - Executor trait and dispatcher
2. `engram-core/src/query/executor/query_executor.rs` - Main executor implementation
3. `engram-core/src/query/executor/context.rs` - QueryContext with memory_space_id

---

## Key Types

```rust
pub struct QueryExecutor {
    registry: Arc<MemorySpaceRegistry>,
    config: QueryExecutorConfig,
}

pub struct QueryContext {
    pub memory_space_id: MemorySpaceId,
    pub timeout: Option<Duration>,
}

impl QueryExecutor {
    pub fn execute(
        &self,
        query: Query,
        context: QueryContext,
    ) -> Result<ProbabilisticQueryResult, QueryExecutionError>;
}
```

---

## Integration Points

- Memory space validation via MemorySpaceRegistry
- Evidence chain construction from query source
- Timeout enforcement for long-running queries

---

## Acceptance Criteria

- [ ] Executor routes queries to correct handlers
- [ ] Multi-tenant isolation enforced
- [ ] Evidence chain includes query AST
- [ ] Timeout handling works
- [ ] Integration tests pass
