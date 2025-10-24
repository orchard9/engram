# Research: From Parse Tree to Memory - Building a Cognitive Query Executor

## Research Questions

1. How do AST interpreters map syntax trees to runtime operations?
2. What are the design patterns for multi-tenant query execution?
3. How do evidence chains enable query explainability?
4. What security boundaries must query executors enforce?
5. How do timeout mechanisms prevent runaway queries?

---

## Key Findings

### 1. AST Interpretation Patterns

**Definition**: An AST interpreter walks a parsed syntax tree and executes corresponding operations at each node.

**Classic Approaches**:

**Visitor Pattern**:
```rust
trait QueryVisitor {
    fn visit_recall(&mut self, query: &RecallQuery) -> Result<QueryResult>;
    fn visit_spread(&mut self, query: &SpreadQuery) -> Result<QueryResult>;
    // ... one method per query type
}

impl QueryVisitor for QueryExecutor {
    fn visit_recall(&mut self, query: &RecallQuery) -> Result<QueryResult> {
        // Map to MemoryStore::recall()
    }
}
```

**Pattern Matching (Rust idiom)**:
```rust
impl QueryExecutor {
    pub fn execute(&self, query: Query, ctx: QueryContext) -> Result<QueryResult> {
        match query {
            Query::Recall(q) => self.execute_recall(q, ctx),
            Query::Spread(q) => self.execute_spread(q, ctx),
            Query::Predict(q) => self.execute_predict(q, ctx),
            // ... exhaustive match
        }
    }
}
```

**Strategy Pattern**:
```rust
trait QueryStrategy {
    fn execute(&self, ctx: &QueryContext) -> Result<QueryResult>;
}

struct RecallStrategy(RecallQuery);
impl QueryStrategy for RecallStrategy {
    fn execute(&self, ctx: &QueryContext) -> Result<QueryResult> {
        // ...
    }
}
```

**Engram's Choice: Pattern Matching**
- Most idiomatic Rust (exhaustive match enforcement)
- Direct mapping without indirection
- Type-safe dispatch at compile time
- Easy to extend (add new Query variant = compiler error until handled)

**Why not Visitor?**
- Adds indirection (trait dispatch overhead)
- Requires mutable state (self: &mut self)
- Less natural in Rust (OOP pattern)

**Why not Strategy?**
- Runtime polymorphism (dynamic dispatch)
- More complex lifetime management
- Overkill for simple dispatch

---

### 2. Multi-Tenant Query Execution

**Problem**: Multiple users share same Engram instance, but queries must not cross memory space boundaries.

**Threat Model**:
1. **Unauthorized reads**: User A queries User B's memories
2. **Unauthorized writes**: Query consolidates across memory spaces
3. **Resource exhaustion**: One user's expensive query impacts others
4. **Timing attacks**: Infer presence of data via query timing

**Security Architecture**:

**Memory Space Isolation**:
```rust
pub struct QueryContext {
    pub memory_space_id: MemorySpaceId,
    pub timeout: Option<Duration>,
    pub max_results: Option<usize>,
}

impl QueryExecutor {
    pub fn execute(
        &self,
        query: Query,
        context: QueryContext,
    ) -> Result<ProbabilisticQueryResult, QueryExecutionError> {
        // 1. Validate memory space exists and user has access
        let space = self.registry
            .get(&context.memory_space_id)
            .ok_or(QueryExecutionError::InvalidMemorySpace)?;

        // 2. All subsequent operations scoped to this space
        match query {
            Query::Recall(q) => self.execute_recall(q, &space, &context),
            // ... all operations receive `space` handle
        }
    }
}
```

**Key Properties**:
- **No ambient authority**: Every operation requires explicit memory space ID
- **Fail-closed**: Invalid memory space = immediate error, no fallback
- **Immutable context**: QueryContext cannot be modified during execution
- **Scope enforcement**: All engine operations receive space handle

**Memory Space Registry**:
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

impl MemorySpaceRegistry {
    pub fn get(&self, id: &MemorySpaceId) -> Option<Arc<SpaceHandle>> {
        self.spaces.get(id).map(|entry| Arc::clone(&entry))
    }
}
```

**Isolation Properties**:
- Each memory space has independent `MemoryStore` and `Graph`
- No shared mutable state between spaces
- Arc for reference counting (no cross-space pointers)
- DashMap for concurrent access without global lock

**Comparison with Other Systems**:

**PostgreSQL Row-Level Security**:
```sql
CREATE POLICY user_isolation ON memories
    USING (memory_space_id = current_user_id());
```
- Declarative, enforced at query planning
- Performance overhead: additional WHERE clause on every query
- Runtime check (not compile-time)

**Kubernetes Namespaces**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: user-123
```
- Process-level isolation (separate address spaces)
- Strong isolation but high overhead (separate API server instances)
- Not practical for in-process multi-tenancy

**Engram Approach: Lightweight Isolation**:
- In-process (no process overhead)
- Compile-time enforcement (QueryContext must contain memory_space_id)
- Runtime validation (registry lookup)
- Arc-based sharing (no data copies)

**Performance**: Registry lookup ~50ns, Arc clone ~10ns. Acceptable overhead for security.

---

### 3. Evidence Chain Construction

**Problem**: Users need to understand WHY query returned specific results (explainability).

**Evidence Types**:

**Query Source**:
```rust
pub struct QueryEvidence {
    pub source: EvidenceSource,
    pub query_ast: Query,
    pub parsed_at: SystemTime,
}

pub enum EvidenceSource {
    UserQuery { user_id: String },
    ApiCall { endpoint: String, request_id: String },
    ConsolidationJob { job_id: String },
}
```

**Memory Activation Path** (from ActivationSpread):
```rust
pub struct ActivationEvidence {
    pub source_node: NodeId,
    pub target_node: NodeId,
    pub activation_strength: f32,
    pub path: Vec<NodeId>, // Nodes traversed
    pub hops: usize,
}
```

**Confidence Computation** (from ProbabilisticQueryExecutor):
```rust
pub struct ConfidenceEvidence {
    pub base_confidence: Confidence,
    pub uncertainty_sources: Vec<UncertaintySource>,
    pub calibrated_confidence: Confidence,
    pub method: CalibrationMethod,
}

pub enum UncertaintySource {
    TemporalDecay { age: Duration, decay_rate: f32 },
    RetrievalAmbiguity { similar_nodes: usize },
    IncompletePattern { completeness: f32 },
}
```

**Consolidation Lineage** (if result from consolidated memory):
```rust
pub struct ConsolidationEvidence {
    pub source_episodes: Vec<NodeId>,
    pub consolidation_time: SystemTime,
    pub semantic_node: NodeId,
    pub compression_ratio: f32,
}
```

**Complete Evidence Chain**:
```rust
pub struct ProbabilisticQueryResult {
    pub episodes: Vec<(Episode, Confidence)>,
    pub evidence_chain: Vec<Evidence>,
    pub query_metadata: QueryMetadata,
}

pub enum Evidence {
    Query(QueryEvidence),
    Activation(ActivationEvidence),
    Confidence(ConfidenceEvidence),
    Consolidation(ConsolidationEvidence),
}
```

**Why Evidence Matters**:

**Debugging**: "Why did this query return episode X?"
- Evidence shows: activated via spreading from cue Y
- Activation strength: 0.8
- Path: cue_Y → intermediate_Z → episode_X
- Hops: 2

**Compliance**: "How did the system arrive at this recommendation?"
- Evidence shows: query parsed from user_123's API call
- Consolidated from episodes A, B, C at time T
- Confidence calibrated using base rate 0.15

**Model Improvement**: "What uncertainty sources dominate?"
- Evidence aggregation shows: TemporalDecay accounts for 60% of uncertainty
- Suggests: tune decay rates or consolidate older memories more aggressively

**Comparison with SQL EXPLAIN**:
```sql
EXPLAIN ANALYZE SELECT * FROM memories WHERE confidence > 0.7;

Seq Scan on memories  (cost=0.00..10.75 rows=5 width=32)
  Filter: (confidence > 0.7)
  Rows Removed by Filter: 3
```

**Similarities**:
- Shows execution strategy (Seq Scan vs Index Scan)
- Reveals cost estimates and actual rows

**Differences**:
- SQL EXPLAIN: execution plan (HOW query ran)
- Evidence chain: result provenance (WHY result returned)
- SQL: performance optimization tool
- Evidence: explainability and debugging tool

**Engram's Evidence Chain: More Comprehensive**
- Includes probabilistic reasoning steps
- Tracks activation paths through graph
- Documents uncertainty sources
- Preserves query AST for auditability

---

### 4. Query Timeout Mechanisms

**Problem**: Long-running queries can exhaust resources or indicate runaway logic.

**Timeout Strategies**:

**Deadline-Based (Engram's Approach)**:
```rust
pub struct QueryContext {
    pub timeout: Option<Duration>,
    // ...
}

impl QueryExecutor {
    fn execute_with_timeout(
        &self,
        query: Query,
        context: QueryContext,
    ) -> Result<QueryResult, QueryExecutionError> {
        let deadline = context.timeout.map(|d| Instant::now() + d);

        // Pass deadline to all operations
        match query {
            Query::Recall(q) => self.execute_recall_with_deadline(q, deadline),
            // ...
        }
    }

    fn execute_recall_with_deadline(
        &self,
        query: RecallQuery,
        deadline: Option<Instant>,
    ) -> Result<QueryResult> {
        // Periodic deadline checks
        let episodes = self.store.recall_with_deadline(&query.pattern, deadline)?;

        // Check before expensive operations
        if let Some(d) = deadline {
            if Instant::now() > d {
                return Err(QueryExecutionError::Timeout);
            }
        }

        // Continue execution...
    }
}
```

**Cancellation Token Pattern** (alternative):
```rust
use tokio::sync::CancellationToken;

impl QueryExecutor {
    async fn execute_with_cancellation(
        &self,
        query: Query,
        cancel: CancellationToken,
    ) -> Result<QueryResult> {
        tokio::select! {
            result = self.execute_inner(query) => result,
            _ = cancel.cancelled() => Err(QueryExecutionError::Cancelled),
        }
    }
}
```

**Trade-offs**:

**Deadline-Based**:
+ Simple synchronous implementation
+ No async overhead
+ Clear timeout semantics
- Requires cooperative checking (can't interrupt CPU-bound work)
- Polling overhead on hot path

**Cancellation Token**:
+ True cancellation (can interrupt async work)
+ Composable (parent can cancel child)
- Requires async/await (complexity)
- Tokio dependency (not needed otherwise)

**Engram's Choice: Deadline-Based**
- Query executor is synchronous (simpler)
- Operations are I/O bound (MemoryStore lookups), not CPU-bound
- Timeout checks at natural boundaries (between operations)
- No need for async machinery

**Timeout Check Placement**:
```rust
// Check before expensive operations
fn execute_recall(/* ... */) -> Result<QueryResult> {
    self.check_deadline(ctx.deadline)?; // 1. Before recall

    let episodes = store.recall(&pattern)?;

    self.check_deadline(ctx.deadline)?; // 2. Before constraint application

    let filtered = self.apply_constraints(episodes, &constraints)?;

    self.check_deadline(ctx.deadline)?; // 3. Before probabilistic query

    let result = self.probabilistic_executor.execute(filtered)?;

    Ok(result)
}
```

**Performance Impact**:
- `Instant::now()`: ~20ns (TSC on x86)
- Comparison: ~1ns
- Total overhead: ~21ns per check
- With 5 checks per query: ~100ns
- Acceptable (<0.1% of 100μs parse time)

**Timeout Configuration**:
- Default: 5 seconds (generous for typical queries)
- Max: 30 seconds (prevent long-running jobs)
- None: Allowed for batch consolidation jobs

---

### 5. Query Executor Architecture Patterns

**Layered Architecture**:

**Layer 1: Dispatcher**
```rust
pub struct QueryExecutor {
    registry: Arc<MemorySpaceRegistry>,
    probabilistic_executor: Arc<ProbabilisticQueryExecutor>,
    config: QueryExecutorConfig,
}

impl QueryExecutor {
    pub fn execute(&self, query: Query, ctx: QueryContext) -> Result<QueryResult> {
        // Validate and dispatch
        let space = self.validate_context(&ctx)?;
        self.dispatch_query(query, &space, &ctx)
    }

    fn dispatch_query(
        &self,
        query: Query,
        space: &SpaceHandle,
        ctx: &QueryContext,
    ) -> Result<QueryResult> {
        match query {
            Query::Recall(q) => self.execute_recall(q, space, ctx),
            Query::Spread(q) => self.execute_spread(q, space, ctx),
            Query::Predict(q) => self.execute_predict(q, space, ctx),
            Query::Imagine(q) => self.execute_imagine(q, space, ctx),
            Query::Consolidate(q) => self.execute_consolidate(q, space, ctx),
        }
    }
}
```

**Layer 2: Operation Handlers** (one per query type)
```rust
impl QueryExecutor {
    fn execute_recall(
        &self,
        query: RecallQuery,
        space: &SpaceHandle,
        ctx: &QueryContext,
    ) -> Result<QueryResult> {
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

        // 5. Construct evidence chain
        let evidence = self.build_evidence_chain(query, result);

        Ok(ProbabilisticQueryResult {
            episodes: result.episodes,
            evidence_chain: evidence,
            metadata: self.build_metadata(ctx),
        })
    }
}
```

**Layer 3: Helper Methods**
```rust
impl QueryExecutor {
    fn pattern_to_cue(&self, pattern: &Pattern) -> Result<Cue> {
        match pattern {
            Pattern::Embedding(vec) => Ok(Cue::Embedding(vec.clone())),
            Pattern::ContentMatch(text) => Ok(Cue::Content(text.clone())),
            Pattern::NodeId(id) => Ok(Cue::NodeId(id.clone())),
        }
    }

    fn apply_constraints(
        &self,
        episodes: Vec<Episode>,
        constraints: &[Constraint],
    ) -> Result<Vec<Episode>> {
        let mut filtered = episodes;
        for constraint in constraints {
            filtered = self.apply_single_constraint(filtered, constraint)?;
        }
        Ok(filtered)
    }

    fn build_evidence_chain(
        &self,
        query: Query,
        result: QueryResult,
    ) -> Vec<Evidence> {
        vec![
            Evidence::Query(QueryEvidence {
                source: EvidenceSource::UserQuery,
                query_ast: query,
                parsed_at: SystemTime::now(),
            }),
            // ... more evidence types
        ]
    }
}
```

**Design Principles**:
1. **Single Responsibility**: Each layer has one job
2. **Dependency Injection**: Registry and executor injected, not hard-coded
3. **Error Propagation**: Use `?` operator, don't swallow errors
4. **Explicit Context**: QueryContext passed explicitly, not ambient
5. **Immutability**: Context and space handles are immutable

---

### 6. Integration with Existing Engine Components

**Memory Store Integration**:
```rust
// Existing API (from Milestone 3)
pub trait MemoryStore {
    fn recall(&self, cue: &Cue) -> Result<Vec<Episode>>;
    fn consolidate(&self, episodes: &[Episode]) -> Result<NodeId>;
    // ...
}

// Query executor uses this directly
impl QueryExecutor {
    fn execute_recall(&self, query: RecallQuery, space: &SpaceHandle) -> Result<QueryResult> {
        let cue = self.pattern_to_cue(&query.pattern)?;
        let episodes = space.store.recall(&cue)?; // Direct call
        // ...
    }
}
```

**Activation Spread Integration**:
```rust
// Existing API (from Milestone 2)
pub trait ActivationSpread {
    fn spread_from(&self, source: NodeId, config: SpreadConfig) -> Vec<ActivationPath>;
}

// Query executor maps SPREAD query to this
impl QueryExecutor {
    fn execute_spread(&self, query: SpreadQuery, space: &SpaceHandle) -> Result<QueryResult> {
        let config = SpreadConfig {
            max_hops: query.max_hops.unwrap_or(3),
            decay_rate: query.decay_rate.unwrap_or(0.15),
            threshold: query.activation_threshold.unwrap_or(0.1),
        };

        let paths = space.graph.spread_from(query.source, config);

        // Convert to ProbabilisticQueryResult
        // ...
    }
}
```

**Probabilistic Query Executor Integration**:
```rust
// Existing API (from Milestone 6)
pub trait ProbabilisticQueryExecutor {
    fn execute(
        &self,
        candidates: Vec<Episode>,
        activation_paths: &[ActivationPath],
        uncertainty_sources: Vec<UncertaintySource>,
    ) -> ProbabilisticQueryResult;
}

// Query executor uses this for all cognitive operations
impl QueryExecutor {
    fn execute_recall(&self, query: RecallQuery, space: &SpaceHandle) -> Result<QueryResult> {
        // Recall and filter episodes
        let episodes = /* ... */;

        // Probabilistic query execution
        let result = self.probabilistic_executor.execute(
            episodes,
            &[], // No activation paths for direct recall
            vec![], // Uncertainty from retrieval ambiguity
        );

        Ok(result)
    }
}
```

**Key Insight**: Query executor is a **thin translation layer** between syntax (AST) and semantics (engine operations). It doesn't re-implement cognitive operations, just maps queries to existing APIs.

---

### 7. Error Handling Strategy

**Error Types**:
```rust
#[derive(Debug, Clone)]
pub enum QueryExecutionError {
    InvalidMemorySpace(MemorySpaceId),
    Timeout,
    MemoryStoreError(String),
    ConstraintViolation(String),
    FeatureNotAvailable(String), // e.g., IMAGINE when M8 not ready
}

impl std::error::Error for QueryExecutionError {}

impl Display for QueryExecutionError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::InvalidMemorySpace(id) =>
                write!(f, "Memory space '{}' not found or access denied", id),
            Self::Timeout =>
                write!(f, "Query execution exceeded timeout limit"),
            Self::MemoryStoreError(e) =>
                write!(f, "Memory store error: {}", e),
            Self::ConstraintViolation(e) =>
                write!(f, "Constraint violation: {}", e),
            Self::FeatureNotAvailable(e) =>
                write!(f, "Feature not available: {}", e),
        }
    }
}
```

**Error Context Preservation**:
```rust
impl QueryExecutor {
    fn execute(&self, query: Query, ctx: QueryContext) -> Result<QueryResult, QueryExecutionError> {
        let space = self.registry
            .get(&ctx.memory_space_id)
            .ok_or_else(|| QueryExecutionError::InvalidMemorySpace(ctx.memory_space_id))?;

        self.dispatch_query(query, &space, &ctx)
            .map_err(|e| {
                // Log error with context
                log::error!("Query execution failed: {:?}, context: {:?}", e, ctx);
                e
            })
    }
}
```

**Graceful Degradation**:
```rust
impl QueryExecutor {
    fn execute_imagine(&self, query: ImagineQuery, space: &SpaceHandle) -> Result<QueryResult> {
        #[cfg(feature = "pattern_completion")]
        {
            // M8 pattern completion available
            let completer = PatternCompleter::new(self.config);
            completer.complete(query.pattern, query.seeds)
        }

        #[cfg(not(feature = "pattern_completion"))]
        {
            // M8 not ready yet, return graceful error
            Err(QueryExecutionError::FeatureNotAvailable(
                "Pattern completion (IMAGINE) requires Milestone 8".to_string()
            ))
        }
    }
}
```

---

## Synthesis: Query Executor Design Principles

1. **Thin Translation Layer**: Don't re-implement cognitive operations, just map queries to existing APIs

2. **Multi-Tenant by Default**: Every operation requires explicit memory space context, no ambient authority

3. **Evidence-Driven**: Construct evidence chain showing why results returned (explainability)

4. **Fail-Fast**: Invalid memory space or timeout = immediate error, no fallback

5. **Graceful Degradation**: Features not ready (IMAGINE, PREDICT) return clear errors, don't crash

6. **Performance-Conscious**: Timeout checks at boundaries, registry lookups amortized

7. **Testable**: Each layer independently testable, dependency injection for mocking

---

## References

1. "Interpreter Pattern", Gang of Four Design Patterns (1994)
2. "Multi-Tenancy Patterns in SaaS Applications", Microsoft Azure Architecture Center
3. "Evidence-Based Software Engineering", Kitchenham et al. (2004)
4. PostgreSQL Row-Level Security: https://www.postgresql.org/docs/current/ddl-rowsecurity.html
5. Tokio Cancellation Tokens: https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html
6. Rust Error Handling: https://doc.rust-lang.org/book/ch09-00-error-handling.html
7. "Building Secure Multi-Tenant Applications", OWASP (2023)
