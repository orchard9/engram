# Task 006: RECALL Operation Implementation

**Status**: Pending
**Duration**: 1.5 days
**Dependencies**: Task 005 (Query Executor)
**Owner**: TBD

---

## Objective

Implement RECALL query execution: map RecallQuery AST to MemoryStore::recall() and ProbabilisticQueryExecutor with constraint application and confidence filtering.

---

## Files to Create

`engram-core/src/query/executor/recall.rs`

---

## Implementation

```rust
impl QueryExecutor {
    pub fn execute_recall(
        &self,
        query: RecallQuery,
        space: &SpaceHandle,
    ) -> Result<ProbabilisticQueryResult, QueryExecutionError> {
        // 1. Convert pattern to cue
        let cue = self.pattern_to_cue(&query.pattern)?;

        // 2. Recall from memory store
        let episodes = space.store.recall(&cue)?;

        // 3. Apply constraints
        let filtered = self.apply_constraints(episodes, &query.constraints)?;

        // 4. Execute probabilistic query
        let result = self.probabilistic_executor.execute(
            filtered,
            &[], // activation_paths
            vec![], // uncertainty_sources
        );

        Ok(result)
    }
}
```

---

## Acceptance Criteria

- [ ] RECALL queries return correct episodes
- [ ] Confidence filtering works
- [ ] Embedding similarity constraints work
- [ ] Temporal constraints work
- [ ] Integration tests pass
