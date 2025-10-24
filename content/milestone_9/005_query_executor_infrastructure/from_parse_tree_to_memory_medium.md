# From Parse Tree to Memory: Building a Cognitive Query Executor

When you type `RECALL episode WHERE confidence > 0.7`, what actually happens? How does a string of text become a memory retrieval operation?

This is the job of the query executor - the bridge between syntax (what you wrote) and semantics (what the system does). Let's explore how Engram implements this translation layer, with a focus on multi-tenant isolation, evidence chains, and zero-cost abstractions.

---

## The Journey of a Query

Let's follow a query through the system:

**Step 1: Syntax → AST** (Parser's job)
```
"RECALL episode WHERE confidence > 0.7"
↓
RecallQuery {
    pattern: Pattern::ContentMatch("episode"),
    constraints: vec![
        Constraint::ConfidenceAbove(0.7)
    ],
}
```

**Step 2: AST → Operations** (Executor's job)
```rust
RecallQuery
↓ pattern_to_cue()
Cue::Content("episode")
↓ MemoryStore::recall()
Vec<Episode>  // All episodes matching "episode"
↓ apply_constraints()
Vec<Episode>  // Filtered to confidence > 0.7
↓ ProbabilisticQueryExecutor::execute()
ProbabilisticQueryResult  // With evidence chains
```

**Step 3: Operations → Results** (Engine's job)
```
MemoryStore uses SIMD operations (M1)
ProbabilisticQueryExecutor calibrates confidence (M6)
ActivationSpread traces paths (M2)
↓
Final result with confidence intervals and evidence
```

The query executor sits in Step 2, translating high-level intentions into concrete engine operations.

---

## Architecture: Pattern Matching as Dispatch

How does the executor know which operation to run? Rust's pattern matching:

```rust
pub struct QueryExecutor {
    registry: Arc<MemorySpaceRegistry>,
    probabilistic_executor: Arc<ProbabilisticQueryExecutor>,
    config: QueryExecutorConfig,
}

impl QueryExecutor {
    pub fn execute(
        &self,
        query: Query,
        context: QueryContext,
    ) -> Result<ProbabilisticQueryResult, QueryExecutionError> {
        // Validate memory space
        let space = self.registry
            .get(&context.memory_space_id)
            .ok_or(QueryExecutionError::InvalidMemorySpace)?;

        // Dispatch based on query type
        match query {
            Query::Recall(q) => self.execute_recall(q, &space, &context),
            Query::Spread(q) => self.execute_spread(q, &space, &context),
            Query::Predict(q) => self.execute_predict(q, &space, &context),
            Query::Imagine(q) => self.execute_imagine(q, &space, &context),
            Query::Consolidate(q) => self.execute_consolidate(q, &space, &context),
        }
    }
}
```

Why pattern matching instead of virtual dispatch (trait objects)?

**Performance**: Pattern matching compiles to direct function calls (zero overhead). Virtual dispatch requires vtable lookup (10ns overhead per call).

**Safety**: Compiler enforces exhaustiveness - add new query type? Compiler error until you handle it.

**Clarity**: The dispatch logic is explicit, not hidden behind trait implementations.

---

## Multi-Tenant Isolation: Compile-Time Safety

Every query operates within a memory space. This is not optional:

```rust
pub struct QueryContext {
    pub memory_space_id: MemorySpaceId,
    pub timeout: Option<Duration>,
    pub max_results: Option<usize>,
}
```

The memory_space_id field is required. You can't create a QueryContext without it.

**Why This Matters**: Security boundaries are enforced at compile time, not runtime.

Compare with SQL:
```sql
-- Dangerous: Easy to forget WHERE clause
SELECT * FROM memories WHERE user_id = 'user_123';

-- OOPS: Forgot WHERE clause, leaks all memories
SELECT * FROM memories;
```

Engram:
```rust
// MUST provide memory_space_id
executor.execute(query, QueryContext {
    memory_space_id: user_id,  // Required field
});

// Forgot it? Compiler error, not runtime leak!
```

**Multi-Tenant Architecture**:
```rust
pub struct MemorySpaceRegistry {
    spaces: DashMap<MemorySpaceId, Arc<SpaceHandle>>,
}

pub struct SpaceHandle {
    pub id: MemorySpaceId,
    pub store: Arc<MemoryStore>,
    pub graph: Arc<Graph>,
    pub config: MemorySpaceConfig,
}
```

Each memory space has its own MemoryStore and Graph. No shared mutable state. Isolation is structural, not procedural.

**Performance**: Registry lookup takes ~50ns. Arc clone takes ~10ns. Total overhead: 60ns for compile-time multi-tenant safety.

---

## Evidence Chains: The "Why" Behind Results

When a query returns results, users want to know WHY. Evidence chains provide an audit trail:

**Evidence Types**:

**1. Query Source**
```rust
Evidence::Query(QueryEvidence {
    source: EvidenceSource::UserQuery { user_id },
    query_ast: Query::Recall(/* ... */),
    parsed_at: SystemTime::now(),
})
```
"This result came from a user query, parsed at time T"

**2. Activation Path** (for SPREAD queries)
```rust
Evidence::Activation(ActivationEvidence {
    source_node: cue_node,
    target_node: result_node,
    activation_strength: 0.8,
    path: vec![cue_node, intermediate, result_node],
    hops: 2,
})
```
"This episode was activated via 2-hop path with strength 0.8"

**3. Confidence Computation**
```rust
Evidence::Confidence(ConfidenceEvidence {
    base_confidence: 0.9,
    uncertainty_sources: vec![
        UncertaintySource::TemporalDecay { age: 3 days, decay_rate: 0.05 },
    ],
    calibrated_confidence: 0.85,
})
```
"Base confidence 0.9, reduced to 0.85 due to 3-day temporal decay"

**4. Consolidation Lineage**
```rust
Evidence::Consolidation(ConsolidationEvidence {
    source_episodes: vec![ep1, ep2, ep3],
    consolidation_time: SystemTime::now(),
    semantic_node: semantic_123,
})
```
"This semantic memory consolidated from 3 episodes"

**Complete Result**:
```rust
pub struct ProbabilisticQueryResult {
    pub episodes: Vec<(Episode, Confidence)>,
    pub evidence_chain: Vec<Evidence>,
    pub query_metadata: QueryMetadata,
}
```

Every result includes not just WHAT was returned, but WHY and HOW.

---

## Example: RECALL Query Execution

Let's trace a complete query execution:

**Query**: `RECALL meeting_notes WHERE confidence > 0.7`

**Step 1: Parse** (already done, we have AST)
```rust
RecallQuery {
    pattern: Pattern::ContentMatch("meeting_notes"),
    constraints: vec![Constraint::ConfidenceAbove(0.7)],
    confidence_threshold: Some(0.7),
    base_rate: None,
}
```

**Step 2: Execute** (executor's job)
```rust
fn execute_recall(
    &self,
    query: RecallQuery,
    space: &SpaceHandle,
    context: &QueryContext,
) -> Result<ProbabilisticQueryResult> {
    // 1. Convert pattern to memory cue
    let cue = self.pattern_to_cue(&query.pattern)?;
    // cue = Cue::Content("meeting_notes")

    // 2. Recall from memory store
    let episodes = space.store.recall(&cue)?;
    // episodes = [ep1, ep2, ep3, ep4, ep5]
    // with confidences [0.9, 0.85, 0.75, 0.65, 0.6]

    // 3. Apply constraints
    let filtered = self.apply_constraints(episodes, &query.constraints)?;
    // filtered = [ep1, ep2, ep3] (confidence > 0.7)

    // 4. Execute probabilistic query
    let result = self.probabilistic_executor.execute(
        filtered,
        &[], // No activation paths for direct recall
        vec![], // Uncertainty from retrieval ambiguity
    );

    // 5. Build evidence chain
    let evidence = vec![
        Evidence::Query(QueryEvidence {
            source: EvidenceSource::UserQuery,
            query_ast: Query::Recall(query.clone()),
            parsed_at: SystemTime::now(),
        }),
        Evidence::Confidence(ConfidenceEvidence {
            base_confidence: result.confidence,
            uncertainty_sources: result.uncertainty_sources,
            calibrated_confidence: result.calibrated_confidence,
            method: CalibrationMethod::Platt,
        }),
    ];

    Ok(ProbabilisticQueryResult {
        episodes: result.episodes,
        evidence_chain: evidence,
        metadata: QueryMetadata {
            execution_time: elapsed,
            memory_space_id: context.memory_space_id,
        },
    })
}
```

**Result**:
```rust
ProbabilisticQueryResult {
    episodes: vec![
        (ep1, Confidence::new(0.9)),
        (ep2, Confidence::new(0.85)),
        (ep3, Confidence::new(0.75)),
    ],
    evidence_chain: vec![
        Evidence::Query(/* query source */),
        Evidence::Confidence(/* how confidence computed */),
    ],
    metadata: QueryMetadata { /* timing, memory space */ },
}
```

User sees: 3 episodes with their confidence scores, plus evidence explaining why these were returned and how confidence was calculated.

---

## Timeout Mechanisms: Preventing Runaway Queries

Long-running queries can exhaust resources. Timeouts prevent this:

```rust
pub struct QueryContext {
    pub timeout: Option<Duration>, // e.g., Some(5 seconds)
}

impl QueryExecutor {
    fn execute_with_deadline(
        &self,
        query: Query,
        context: QueryContext,
    ) -> Result<QueryResult> {
        let deadline = context.timeout.map(|d| Instant::now() + d);

        // Check deadline before expensive operations
        self.check_deadline(deadline)?;

        let result = match query {
            Query::Recall(q) => {
                let episodes = self.recall_with_deadline(q, deadline)?;
                self.check_deadline(deadline)?;
                self.process_results(episodes)?
            }
            // ... other query types
        };

        Ok(result)
    }

    fn check_deadline(&self, deadline: Option<Instant>) -> Result<()> {
        if let Some(d) = deadline {
            if Instant::now() > d {
                return Err(QueryExecutionError::Timeout);
            }
        }
        Ok(())
    }
}
```

**Performance Impact**:
- `Instant::now()`: ~20ns (reads time-stamp counter)
- Comparison: ~1ns
- Total per check: ~21ns
- With 5 checks per query: ~100ns (<0.1% of 100μs parse time)

**Placement Strategy**: Check at natural boundaries
1. Before recall from MemoryStore
2. After recall, before constraint application
3. After constraints, before probabilistic query
4. After probabilistic query, before evidence construction

This gives fine-grained timeout control without hot-path overhead (checks only happen between operations, not inside tight loops).

---

## Integration with Existing Engine Components

The executor is a thin translation layer. It doesn't re-implement cognitive operations, just maps queries to existing APIs:

**Memory Store** (from Milestone 3):
```rust
// Existing API
pub trait MemoryStore {
    fn recall(&self, cue: &Cue) -> Result<Vec<Episode>>;
}

// Executor uses directly
let episodes = space.store.recall(&cue)?;
```

**Activation Spread** (from Milestone 2):
```rust
// Existing API
pub trait ActivationSpread {
    fn spread_from(&self, source: NodeId, config: SpreadConfig) -> Vec<ActivationPath>;
}

// Executor maps SPREAD query to this
let paths = space.graph.spread_from(query.source, SpreadConfig {
    max_hops: query.max_hops.unwrap_or(3),
    decay_rate: query.decay_rate.unwrap_or(0.15),
    threshold: query.activation_threshold.unwrap_or(0.1),
});
```

**Probabilistic Query Executor** (from Milestone 6):
```rust
// Existing API
pub trait ProbabilisticQueryExecutor {
    fn execute(
        &self,
        candidates: Vec<Episode>,
        activation_paths: &[ActivationPath],
        uncertainty_sources: Vec<UncertaintySource>,
    ) -> ProbabilisticQueryResult;
}

// Executor uses for all queries
let result = self.probabilistic_executor.execute(episodes, &[], vec![]);
```

**Key Insight**: The executor is glue code, not new functionality. It translates between query language semantics and engine operation calls.

---

## Error Handling: Context Preservation

When queries fail, preserve context for debugging:

```rust
#[derive(Debug, Clone)]
pub enum QueryExecutionError {
    InvalidMemorySpace(MemorySpaceId),
    Timeout,
    MemoryStoreError(String),
    ConstraintViolation(String),
    FeatureNotAvailable(String),
}

impl QueryExecutor {
    fn execute(&self, query: Query, ctx: QueryContext) -> Result<QueryResult> {
        let space = self.registry
            .get(&ctx.memory_space_id)
            .ok_or_else(|| {
                log::error!("Invalid memory space: {:?}, context: {:?}", ctx.memory_space_id, ctx);
                QueryExecutionError::InvalidMemorySpace(ctx.memory_space_id)
            })?;

        self.dispatch_query(query, &space, &ctx)
            .map_err(|e| {
                log::error!("Query execution failed: {:?}, context: {:?}", e, ctx);
                e
            })
    }
}
```

Every error includes the QueryContext, so you know:
- Which memory space was accessed
- What timeout was set
- What query was executed

This makes production debugging tractable.

---

## Graceful Degradation: Feature Flags

Not all features are ready simultaneously (e.g., IMAGINE requires Milestone 8):

```rust
impl QueryExecutor {
    fn execute_imagine(&self, query: ImagineQuery, space: &SpaceHandle) -> Result<QueryResult> {
        #[cfg(feature = "pattern_completion")]
        {
            let completer = PatternCompleter::new(self.config);
            completer.complete(query.pattern, query.seeds)
        }

        #[cfg(not(feature = "pattern_completion"))]
        {
            Err(QueryExecutionError::FeatureNotAvailable(
                "Pattern completion (IMAGINE) requires Milestone 8. \
                 Enable with --features pattern_completion".to_string()
            ))
        }
    }
}
```

Users get clear feedback: "This feature isn't ready yet, here's what to do when it is."

---

## Testing Strategy

How do we test the executor?

**Unit Tests**: Mock dependencies
```rust
#[test]
fn test_recall_execution() {
    let mock_store = MockMemoryStore::new();
    mock_store.expect_recall()
        .returning(|_| Ok(vec![episode1, episode2]));

    let executor = QueryExecutor::new(mock_store, /* ... */);

    let query = Query::Recall(RecallQuery {
        pattern: Pattern::ContentMatch("test"),
        constraints: vec![],
        confidence_threshold: None,
        base_rate: None,
    });

    let result = executor.execute(query, test_context()).unwrap();

    assert_eq!(result.episodes.len(), 2);
    assert!(!result.evidence_chain.is_empty());
}
```

**Integration Tests**: Real components
```rust
#[test]
fn test_end_to_end_recall() {
    let store = MemoryStore::new();
    store.insert(episode1);
    store.insert(episode2);

    let executor = QueryExecutor::new(store, /* ... */);

    let query_text = "RECALL episode WHERE confidence > 0.7";
    let query = Parser::parse(query_text).unwrap();

    let result = executor.execute(query, test_context()).unwrap();

    assert!(result.episodes.len() > 0);
    assert!(result.evidence_chain.iter().any(|e| matches!(e, Evidence::Query(_))));
}
```

**Property Tests**: Invariants
```rust
#[proptest]
fn executor_never_leaks_across_memory_spaces(
    query: Query,
    space1: MemorySpaceId,
    space2: MemorySpaceId,
) {
    let executor = QueryExecutor::new(/* ... */);

    let result1 = executor.execute(query.clone(), QueryContext { memory_space_id: space1 })?;
    let result2 = executor.execute(query, QueryContext { memory_space_id: space2 })?;

    // Results from different memory spaces must be disjoint
    let ids1: HashSet<_> = result1.episodes.iter().map(|(e, _)| e.id).collect();
    let ids2: HashSet<_> = result2.episodes.iter().map(|(e, _)| e.id).collect();

    prop_assert!(ids1.is_disjoint(&ids2));
}
```

---

## Performance Characteristics

What's the overhead of the executor layer?

**Parsing**: <100μs (Target from Milestone 9)
**Executor dispatch**: ~100ns (pattern matching + registry lookup)
**Memory operation**: 1-10ms (MemoryStore, ActivationSpread - dominant cost)
**Evidence construction**: ~10μs (small allocations)

**Total**: Executor overhead is <1% of total query latency. The expensive part is the actual memory operations, which is correct - that's where the work happens.

---

## Conclusion: The Executor as Translation Layer

The query executor's job is simple but critical: translate high-level query intentions into low-level engine operations while maintaining safety and explainability.

Key design principles:
1. **Thin layer**: Don't re-implement, just translate
2. **Multi-tenant by default**: Memory space ID required, not optional
3. **Evidence-driven**: Always explain why results returned
4. **Zero-cost where possible**: Pattern matching, no vtables
5. **Fail-fast**: Invalid memory space = immediate error
6. **Graceful degradation**: Missing features return clear errors

The result: A query language that feels natural (RECALL, SPREAD, PREDICT) while maintaining the precision and safety of a well-engineered cognitive architecture.

When a user types `RECALL episode WHERE confidence > 0.7`, they're not just running a query - they're engaging in a conversation with a memory system that can explain its reasoning. That's what the executor enables.

---

## Further Reading

1. "Interpreter Pattern", Gang of Four Design Patterns (1994)
2. "Multi-Tenancy Patterns in SaaS Applications", Microsoft Azure Architecture Center
3. PostgreSQL Row-Level Security: https://www.postgresql.org/docs/current/ddl-rowsecurity.html
4. Rust Error Handling: https://doc.rust-lang.org/book/ch09-00-error-handling.html
5. "Building Secure Multi-Tenant Applications", OWASP (2023)
