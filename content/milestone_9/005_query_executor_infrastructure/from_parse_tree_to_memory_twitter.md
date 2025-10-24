# Twitter Thread: From Parse Tree to Memory

**Tweet 1/10** (Hook)

You type: "RECALL episode WHERE confidence > 0.7"

What happens next? How does text become memory retrieval?

Let's trace a query through Engram's executor - the bridge between syntax and semantics.

Thread on query execution, multi-tenant isolation, and evidence chains.

---

**Tweet 2/10** (The Journey)

Query journey in 3 steps:

1. Parser: Text → AST (syntax tree)
2. Executor: AST → Engine operations
3. Engine: Operations → Results

Today we're talking about step 2: the translator.

The executor doesn't implement new features. It maps intentions to existing operations.

---

**Tweet 3/10** (Dispatch Mechanism)

How does executor know which operation to run?

Rust pattern matching:

```rust
match query {
    Query::Recall(q) =>
        execute_recall(q),
    Query::Spread(q) =>
        execute_spread(q),
}
```

Why not virtual dispatch (traits)?
- Pattern matching = zero cost
- Virtual dispatch = 10ns overhead
- Bonus: Compiler enforces exhaustiveness

---

**Tweet 4/10** (Multi-Tenant Isolation)

Every query needs a memory space ID:

```rust
pub struct QueryContext {
    memory_space_id: MemorySpaceId,
    timeout: Option<Duration>,
}
```

Not optional. Can't create QueryContext without it.

Forgot it? Compiler error, not runtime leak.

Compare with SQL's "forgot WHERE user_id" disaster.

---

**Tweet 5/10** (Compile-Time Safety)

SQL approach (runtime check):
```sql
SELECT * FROM memories
WHERE user_id = 'user_123';

-- OOPS: Forgot WHERE = leak all data
SELECT * FROM memories;
```

Engram approach (compile-time):
```rust
executor.execute(query, QueryContext {
    memory_space_id: /* required */
});
```

Security boundaries = type system, not discipline.

---

**Tweet 6/10** (Evidence Chains)

Every query returns not just WHAT, but WHY:

Evidence types:
- Query source (who asked?)
- Activation paths (how found?)
- Confidence computation (why confident?)
- Consolidation lineage (where from?)

Black box → Glass box.

Users understand the reasoning.

---

**Tweet 7/10** (Example Evidence)

Query: "RECALL meeting_notes"

Result:
```rust
ProbabilisticQueryResult {
    episodes: [ep1, ep2, ep3],
    evidence_chain: [
        Query(/* parsed at time T */),
        Confidence(/* base 0.9,
            decayed to 0.85 */),
    ],
}
```

Not just answers. Explanations.

This is cognitive query execution.

---

**Tweet 8/10** (Timeout Mechanism)

Prevent runaway queries with deadlines:

```rust
fn execute_with_deadline(
    query: Query,
    deadline: Option<Instant>,
) -> Result<QueryResult> {
    self.check_deadline(deadline)?;
    let episodes = self.recall()?;
    self.check_deadline(deadline)?;
    self.process(episodes)
}
```

Cost: ~21ns per check
Checks: At operation boundaries
Overhead: <0.1% of query time

---

**Tweet 9/10** (Integration)

Executor is thin glue code:

```rust
// Existing MemoryStore API
space.store.recall(&cue)?;

// Existing ActivationSpread API
space.graph.spread_from(source, config);

// Existing ProbabilisticQuery API
probabilistic_executor.execute(
    candidates,
    paths,
    uncertainty
);
```

Don't re-implement. Translate.

---

**Tweet 10/10** (Graceful Degradation)

Not all features ready simultaneously:

```rust
#[cfg(feature = "pattern_completion")]
{
    // IMAGINE query works
}

#[cfg(not(/* ... */))]
{
    Err(FeatureNotAvailable(
        "IMAGINE requires Milestone 8"
    ))
}
```

Clear feedback when features unavailable.

Users know what to expect.

---

**Bonus Tweet** (Performance)

Executor overhead breakdown:

Parse: <100μs
Executor dispatch: ~100ns
Memory operation: 1-10ms ← dominant
Evidence construction: ~10μs

Total executor overhead: <1%

The work happens where it should: in the memory operations.

---

**Thread Summary**

New blog post: "From Parse Tree to Memory: Building a Cognitive Query Executor"

Topics:
- Pattern matching as zero-cost dispatch
- Multi-tenant compile-time safety
- Evidence chains for explainability
- Timeout mechanisms
- Integration patterns

Building cognitive systems: https://engram.dev/blog/query-executor

---

**Code Example Tweet**

RECALL query execution:

```rust
fn execute_recall(
    query: RecallQuery,
    space: &SpaceHandle,
) -> Result<QueryResult> {
    // 1. Pattern → Cue
    let cue = pattern_to_cue(&query.pattern)?;

    // 2. Recall from memory
    let episodes = space.store.recall(&cue)?;

    // 3. Apply constraints
    let filtered = apply_constraints(
        episodes,
        &query.constraints
    )?;

    // 4. Probabilistic query
    let result = probabilistic_executor
        .execute(filtered, &[], vec![]);

    // 5. Build evidence
    let evidence = build_evidence_chain(
        query,
        result
    );

    Ok(ProbabilisticQueryResult {
        episodes: result.episodes,
        evidence_chain: evidence,
    })
}
```

Simple translation layer.

---

**Security Tweet**

Multi-tenant architecture:

```rust
pub struct MemorySpaceRegistry {
    spaces: DashMap<
        MemorySpaceId,
        Arc<SpaceHandle>
    >,
}

pub struct SpaceHandle {
    store: Arc<MemoryStore>,
    graph: Arc<Graph>,
}
```

Each memory space:
- Independent MemoryStore
- Independent Graph
- No shared mutable state

Isolation is structural, not procedural.

---

**Testing Tweet**

How we test the executor:

Unit tests: Mock dependencies
Integration tests: Real components
Property tests: Invariants

Key property:
"Results from different memory spaces must be disjoint"

```rust
#[proptest]
fn no_cross_space_leaks(
    query: Query,
    space1: MemorySpaceId,
    space2: MemorySpaceId,
) {
    let r1 = execute(query, space1)?;
    let r2 = execute(query, space2)?;

    assert!(r1.disjoint(&r2));
}
```

---

**Error Handling Tweet**

When queries fail, preserve context:

```rust
QueryExecutionError::
    InvalidMemorySpace(id)

// Logs include:
// - Query AST
// - QueryContext (memory space,
//   timeout)
// - Execution state

// Debugging: "What was the user
// trying to do?"
// Answer: Everything preserved
// in error
```

Context = debuggability.

---

**Evidence Chain Tweet**

Evidence enables:

1. Debugging: "Why did query return episode X?"
   → See activation path

2. Compliance: "How did system decide?"
   → See consolidation lineage

3. Model improvement: "What uncertainty sources?"
   → See confidence computation

Not just results. Reasoning.

---

**Design Philosophy Tweet**

Executor design principles:

1. Thin layer (don't re-implement)
2. Multi-tenant by default (required field)
3. Evidence-driven (always explain)
4. Zero-cost where possible (pattern matching)
5. Fail-fast (invalid space = error)
6. Graceful degradation (clear errors)

Translate intentions, preserve safety.

---

**Comparison Tweet**

Query executors in other systems:

SQL: Query planner + executor
- Optimizes for performance
- EXPLAIN shows plan
- No result provenance

GraphQL: Resolver chain
- Fetches nested data
- No multi-tenant enforcement
- Limited explainability

Engram: Translation layer
- Maps to cognitive operations
- Multi-tenant compile-time enforced
- Evidence chains for every result

---

**Memory Analogy Tweet**

Multi-tenant memory spaces = biological brains:

You can't accidentally recall someone else's memories.

Consolidation happens within your brain.

No cross-contamination.

Engram implements this digitally:
- Compile-time: Memory space ID required
- Runtime: Isolated stores and graphs
- Result: Structural isolation

---

**Performance Tweet**

Why pattern matching over trait objects?

Trait object (virtual dispatch):
```rust
let strategy: Box<dyn QueryStrategy>;
strategy.execute() // vtable lookup
```
Cost: ~10ns overhead

Pattern matching:
```rust
match query {
    Query::Recall(q) => execute_recall(q)
}
```
Cost: ~0ns (direct call, inlined)

Zero-cost abstraction FTW.

---

**Final Thought Tweet**

The query executor is conceptually simple:

Take AST, call existing APIs, return results.

But the details matter:
- Multi-tenant safety
- Evidence construction
- Timeout handling
- Error context
- Zero-cost dispatch

Simple ≠ Easy.

Good architecture is invisible when it works.
