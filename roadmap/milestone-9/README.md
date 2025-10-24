# Milestone 9: Query Language Parser

## Executive Summary

Implement a cognitive-first query language for Engram that maps directly to memory operations, not SQL-like syntax. Parser must achieve <100μs parse time with production-grade error messages showing location and suggestions.

**Duration**: 14 days
**Dependencies**: Milestones 1-7 complete, Milestone 8 in progress
**Risk Level**: Medium (parser performance, error message quality)

---

## 1. Query Language Design

### 1.1 Core Cognitive Operations

The query language expresses memory operations using verbs that map to biological memory processes:

```
RECALL <pattern> [WHERE <constraints>] [CONFIDENCE <threshold>]
PREDICT <pattern> GIVEN <context> [HORIZON <duration>]
IMAGINE <pattern> [BASED ON <seeds>] [NOVELTY <level>]
CONSOLIDATE <episodes> INTO <semantic_node> [SCHEDULER <policy>]
SPREAD FROM <cue> [MAX_HOPS <n>] [DECAY <rate>] [THRESHOLD <activation>]
```

NOT SQL-like. No SELECT/FROM/JOIN - these are database abstractions that obscure cognitive semantics.

### 1.2 Syntax Examples with Probabilistic Semantics

```
# Episodic recall with confidence filtering
RECALL episode WHERE content SIMILAR TO [0.1, 0.3, ...] CONFIDENCE > 0.7

# Pattern completion (integrates with M8)
IMAGINE episode BASED ON partial_episode NOVELTY 0.3

# Spreading activation with temporal decay
SPREAD FROM cue_node MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1

# Consolidation trigger
CONSOLIDATE episodes WHERE created < "2024-10-20" INTO semantic_memory

# Predictive query with uncertainty
PREDICT episode GIVEN context_embedding HORIZON 3600 CONFIDENCE [0.6, 0.8]
```

### 1.3 Probabilistic Semantics Representation

Confidence is first-class in the syntax:

- **Point estimates**: `CONFIDENCE > 0.7` (threshold filtering)
- **Interval constraints**: `CONFIDENCE [0.6, 0.8]` (range specification)
- **Uncertainty propagation**: Implicit through query execution, explicit in results
- **Base rate integration**: `WITH BASE_RATE 0.1` (optional Bayesian prior)

Design principle: Make uncertainty visible and actionable, not hidden.

### 1.4 Integration with Existing Systems

- `RECALL` maps to `MemoryStore::recall()` + `ProbabilisticQueryExecutor::execute()`
- `SPREAD` maps to `ActivationSpread::spread_from()` with configurable parameters
- `CONSOLIDATE` maps to `ConsolidationScheduler::trigger_consolidation()`
- `IMAGINE` maps to `PatternCompleter::complete()` (Milestone 8)
- `PREDICT` combines activation spreading with temporal projection

All operations return `ProbabilisticQueryResult` with evidence chains.

---

## 2. Parser Architecture

### 2.1 Implementation Approach: Hand-Written Recursive Descent

**Decision**: Hand-written recursive descent parser, NOT parser generator.

**Rationale**:
1. **Performance**: Direct control over allocation, zero-copy string handling
2. **Error Recovery**: Precise control over error messages and suggestions
3. **Maintenance**: Simpler debugging, no generated code to inspect
4. **Dependency**: No external parser DSL (nom, pest, lalrpop) - reduces build complexity

**Trade-offs Accepted**:
- More manual work upfront
- Must implement error recovery manually
- Grammar changes require code changes (acceptable - grammar is stable)

### 2.2 AST Design

```rust
// File: engram-core/src/query/parser/ast.rs

pub enum Query {
    Recall(RecallQuery),
    Predict(PredictQuery),
    Imagine(ImagineQuery),
    Consolidate(ConsolidateQuery),
    Spread(SpreadQuery),
}

pub struct RecallQuery {
    pub pattern: Pattern,
    pub constraints: Vec<Constraint>,
    pub confidence_threshold: Option<Confidence>,
    pub base_rate: Option<Confidence>,
}

pub struct SpreadQuery {
    pub source: NodeId,
    pub max_hops: Option<u16>,
    pub decay_rate: Option<f32>,
    pub activation_threshold: Option<f32>,
}

pub enum Pattern {
    Embedding(Vec<f32>),
    ContentMatch(String),
    NodeId(String),
}

pub enum Constraint {
    SimilarTo { embedding: Vec<f32>, threshold: f32 },
    CreatedBefore(SystemTime),
    ConfidenceAbove(Confidence),
    MemorySpace(MemorySpaceId),
}
```

AST maps 1:1 to cognitive operations, not relational algebra.

### 2.3 Error Recovery Strategy

Position tracking for every token:

```rust
pub struct ParseError {
    pub position: usize,       // Character offset in query
    pub line: usize,           // Line number (1-indexed)
    pub column: usize,         // Column number (1-indexed)
    pub found: String,         // What we found
    pub expected: Vec<String>, // What we expected
    pub suggestion: String,    // Actionable fix
    pub example: String,       // Correct usage example
}

impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f,
            "Parse error at line {}, column {}:\n\
             Found: {}\n\
             Expected: {}\n\
             Suggestion: {}\n\
             Example: {}",
            self.line, self.column, self.found,
            self.expected.join(" or "), self.suggestion, self.example
        )
    }
}
```

Error recovery:
1. **Typo detection**: Levenshtein distance for keyword suggestions
2. **Context-aware**: Track parser state for better "expected" messages
3. **Example-driven**: Every error includes correct syntax example

### 2.4 Performance Optimizations

Target: <100μs parse time for typical queries

**Techniques**:
1. **Zero-copy parsing**: Use string slices, avoid allocations
2. **Inline hot paths**: `#[inline]` on token matching functions
3. **Bump allocator**: Arena allocation for AST nodes
4. **Lazy evaluation**: Parse constraints only when needed
5. **Constant folding**: Pre-compute keyword hash maps

**Measurement**:
- Criterion benchmarks for parser operations
- Flamegraph profiling to identify hot spots
- CI regression tests to prevent performance degradation

---

## 3. Integration Points

### 3.1 Query Execution Flow

```
┌─────────────┐
│ Query Text  │
└──────┬──────┘
       │ parse()
       ▼
┌─────────────┐
│ AST (Query) │
└──────┬──────┘
       │ execute()
       ▼
┌──────────────────────────┐
│ QueryExecutor            │
│ - Maps AST to engine ops │
│ - Handles multi-tenant   │
│ - Tracks evidence        │
└──────┬───────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ Existing Systems                       │
│ - MemoryStore::recall()                │
│ - ActivationSpread::spread_from()      │
│ - ProbabilisticQueryExecutor::execute()│
│ - PatternCompleter::complete() (M8)    │
└────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│ ProbabilisticQueryResult │
│ - Episodes + confidence  │
│ - Evidence chain         │
│ - Uncertainty sources    │
└──────────────────────────┘
```

### 3.2 Multi-Tenant Integration (M7)

Every query operates within a memory space context:

```rust
pub struct QueryContext {
    pub memory_space_id: MemorySpaceId,
    pub executor_config: QueryExecutorConfig,
}

impl QueryExecutor {
    pub fn execute(&self, query: Query, context: QueryContext)
        -> Result<ProbabilisticQueryResult, QueryExecutionError>
    {
        // 1. Validate memory_space_id exists
        let space = self.registry.get(&context.memory_space_id)?;

        // 2. Execute query within space boundary
        match query {
            Query::Recall(q) => self.execute_recall(q, &space),
            Query::Spread(q) => self.execute_spread(q, &space),
            // ...
        }
    }
}
```

Integration with HTTP/gRPC: `X-Memory-Space` header sets `QueryContext::memory_space_id`.

### 3.3 Pattern Completion Integration (M8)

`IMAGINE` queries map directly to M8's `PatternCompleter`:

```rust
// When M8 is complete
Query::Imagine(imagine_query) => {
    let partial = imagine_query.pattern.to_partial_episode();
    let seeds = imagine_query.seeds.unwrap_or_default();

    let completer = PatternCompleter::new(config);
    let completed = completer.complete(partial, seeds)?;

    ProbabilisticQueryResult {
        episodes: vec![(completed.episode, completed.confidence)],
        evidence_chain: completed.source_map.to_evidence(),
        // ...
    }
}
```

Graceful degradation: If M8 not ready, `IMAGINE` returns `FeatureNotAvailable` error.

### 3.4 gRPC/HTTP API Surface

```protobuf
// File: engram-proto/engram.proto

message QueryRequest {
    string query_text = 1;
    string memory_space_id = 2;
    QueryOptions options = 3;
}

message QueryResponse {
    repeated Episode episodes = 1;
    repeated float confidences = 2;
    ConfidenceInterval aggregate_confidence = 3;
    repeated Evidence evidence_chain = 4;
    QueryMetrics metrics = 5;
}

service EngramQuery {
    rpc ExecuteQuery(QueryRequest) returns (QueryResponse);
    rpc StreamQuery(QueryRequest) returns (stream QueryResponse);
}
```

HTTP equivalent: `POST /api/v1/query` with JSON body.

---

## 4. Task Breakdown

### Task 001: Parser Infrastructure (2 days)
**File**: `engram-core/src/query/parser/mod.rs`

Implement tokenizer and position tracking:
- Tokenize query string into `Token` enum
- Track `(line, column, offset)` for every token
- Zero-copy string slicing for identifiers/literals
- Keyword recognition with hash map lookup

**Acceptance Criteria**:
- Tokenize 1000-character query in <10μs
- Position tracking accurate for multi-line queries
- Unit tests for all token types

**Dependencies**: None

---

### Task 002: AST Definition (1 day)
**File**: `engram-core/src/query/parser/ast.rs`

Define AST types for all cognitive operations:
- `Query` enum with variants for RECALL/PREDICT/IMAGINE/CONSOLIDATE/SPREAD
- `Pattern`, `Constraint`, `ConfidenceSpec` types
- `FromStr` implementations for embedding literals
- Serde support for AST serialization (debugging)

**Acceptance Criteria**:
- AST types compile with zero warnings
- Serde round-trip tests pass
- Type sizes optimized (<256 bytes per Query)

**Dependencies**: Task 001

---

### Task 003: Recursive Descent Parser (3 days)
**File**: `engram-core/src/query/parser/parser.rs`

Hand-written recursive descent parser:
- `parse_query()` entry point
- Per-operation parsers: `parse_recall()`, `parse_spread()`, etc.
- Constraint parsing with backtracking
- Embedding literal parsing `[0.1, 0.2, ...]`

**Acceptance Criteria**:
- Parse all example queries from section 1.2
- <100μs parse time for typical queries
- Zero allocations on hot path (arena allocator)

**Dependencies**: Task 002

---

### Task 004: Error Recovery and Messages (2 days)
**File**: `engram-core/src/query/parser/error.rs`

Production-grade error messages:
- Levenshtein distance for keyword typo detection
- Context-aware "expected" suggestions
- Example generation for each error type
- Error message templates

**Acceptance Criteria**:
- 100% of parser errors have actionable suggestions
- Typo detection works for keywords (distance ≤2)
- Error messages include line/column/example

**Dependencies**: Task 003

---

### Task 005: Query Executor Infrastructure (2 days)
**File**: `engram-core/src/query/executor/query_executor.rs`

Map AST to engine operations:
- `QueryExecutor` struct with `MemorySpaceRegistry` reference
- `execute()` dispatcher to per-operation handlers
- Memory space validation and boundary enforcement
- Evidence chain construction

**Acceptance Criteria**:
- Execute RECALL queries end-to-end
- Multi-tenant isolation verified
- Evidence chain includes query source

**Dependencies**: Task 003

---

### Task 006: RECALL Operation Implementation (1.5 days)
**File**: `engram-core/src/query/executor/recall.rs`

Implement RECALL query execution:
- Map `RecallQuery` to `MemoryStore::recall()`
- Apply constraints (confidence threshold, similarity)
- Integrate with `ProbabilisticQueryExecutor`
- Handle empty results vs. low confidence

**Acceptance Criteria**:
- RECALL queries return correct episodes
- Confidence filtering works
- Integration tests with existing MemoryStore

**Dependencies**: Task 005

---

### Task 007: SPREAD Operation Implementation (1.5 days)
**File**: `engram-core/src/query/executor/spread.rs`

Implement SPREAD query execution:
- Map `SpreadQuery` to `ActivationSpread::spread_from()`
- Configure max hops, decay rate, threshold
- Return activation paths as evidence
- Performance: must not degrade spreading performance

**Acceptance Criteria**:
- SPREAD queries activate correct nodes
- Configurable parameters work
- Benchmarks show <5% overhead vs. direct API

**Dependencies**: Task 005

---

### Task 008: Query Language Validation Suite (1 day)
**File**: `engram-core/tests/query_language_corpus.rs`

Comprehensive test corpus:
- 50+ valid queries covering all operations
- 50+ invalid queries with expected errors
- Regression tests for error messages
- Property-based tests for parser invariants

**Acceptance Criteria**:
- 100% of invalid queries produce actionable errors
- All valid queries parse in <100μs
- Corpus covers all syntax features

**Dependencies**: Task 004, 006, 007

---

### Task 009: HTTP/gRPC Query Endpoints (1.5 days)
**Files**:
- `engram-proto/engram.proto` (protocol definition)
- `engram-storage/src/http/query.rs` (HTTP handler)
- `engram-storage/src/grpc/query.rs` (gRPC service)

Add query endpoints to existing HTTP/gRPC services:
- `POST /api/v1/query` HTTP endpoint
- `ExecuteQuery` gRPC method
- JSON/Protobuf serialization
- Memory space routing via `X-Memory-Space` header

**Acceptance Criteria**:
- HTTP endpoint returns JSON query results
- gRPC endpoint returns protobuf results
- Multi-tenant routing works
- OpenAPI spec updated

**Dependencies**: Task 005, 006, 007

---

### Task 010: Parser Performance Optimization (1 day)
**File**: `engram-core/benches/query_parser.rs`

Optimize parser to meet <100μs target:
- Criterion benchmarks for all query types
- Flamegraph profiling
- Inline hot paths
- Benchmark regression tests in CI

**Acceptance Criteria**:
- Parse time <100μs for 90% of queries
- Parse time <200μs for 99% of queries
- CI fails if parse time regresses >10%

**Dependencies**: Task 003, 008

---

### Task 011: Documentation and Examples (0.5 days)
**Files**:
- `docs/reference/query-language.md`
- `examples/query_examples.rs`

User-facing documentation:
- Query language reference with all operations
- Examples for each cognitive operation
- Error message catalog
- Performance characteristics

**Acceptance Criteria**:
- Reference doc covers all syntax
- Examples compile and run
- Error catalog shows actual error output

**Dependencies**: Task 009

---

### Task 012: Integration Testing and Validation (1 day)
**File**: `engram-core/tests/query_integration_test.rs`

End-to-end integration tests:
- Query → Parse → Execute → Result flow
- Multi-tenant query isolation
- Pattern completion integration (when M8 ready)
- Performance under load (1000 queries/sec)

**Acceptance Criteria**:
- All integration tests pass
- Multi-tenant isolation verified
- No memory leaks under sustained load
- P99 latency <5ms (parse + execute)

**Dependencies**: All tasks

---

## 5. Risk Analysis and Mitigation

### Risk 1: Parser Performance (<100μs target)
**Probability**: Medium
**Impact**: High (user-facing latency)

**Mitigation**:
1. Benchmark early (Task 010 parallel with Task 003)
2. Use arena allocation to minimize overhead
3. Zero-copy parsing for string slices
4. Fall back to cached parse results if needed

**Contingency**: If <100μs proves infeasible, adjust target to <500μs and document trade-offs.

---

### Risk 2: Error Message Quality
**Probability**: Low
**Impact**: Medium (developer experience)

**Mitigation**:
1. Comprehensive error test corpus (Task 008)
2. User testing with sample queries
3. Typo detection with Levenshtein distance
4. Context-aware suggestions

**Contingency**: Iterate on error messages post-M9 based on user feedback.

---

### Risk 3: M8 Integration Timing
**Probability**: High
**Impact**: Low (feature flag available)

**Mitigation**:
1. Design `IMAGINE` syntax now, implement stubs
2. Feature flag: `#[cfg(feature = "pattern_completion")]`
3. Graceful error if M8 not available

**Contingency**: Ship M9 without `IMAGINE` support, add in M9.1 patch.

---

### Risk 4: Syntax Stability
**Probability**: Medium
**Impact**: Medium (breaking changes expensive)

**Mitigation**:
1. Review syntax with stakeholders before implementation
2. Reserve keywords for future operations
3. Design for extension (not modification)
4. Version query language explicitly

**Contingency**: Support multiple syntax versions if needed (query prefix: `v1:RECALL ...`).

---

## 6. Performance Targets

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Parse time (typical query) | <100μs P90 | Criterion benchmark |
| Parse time (complex query) | <200μs P99 | Criterion benchmark |
| Execute RECALL | <5ms P99 | Integration test |
| Execute SPREAD | <10ms P99 | Integration test |
| Throughput | >1000 queries/sec | Load test |
| Memory overhead | <1KB per query | Allocation profiling |

All targets measured on commodity hardware (4-core CPU, 16GB RAM).

---

## 7. Acceptance Criteria

Milestone 9 complete when:

1. **Parser Functionality**
   - All cognitive operations (RECALL, PREDICT, IMAGINE, CONSOLIDATE, SPREAD) parse correctly
   - 100% of invalid queries produce actionable error messages
   - Parse time <100μs for 90% of queries

2. **Integration**
   - HTTP and gRPC query endpoints operational
   - Multi-tenant memory space routing works
   - Results return `ProbabilisticQueryResult` with evidence chains

3. **Testing**
   - 100+ query corpus (valid + invalid) tests pass
   - Integration tests cover end-to-end flow
   - Property-based tests verify parser invariants
   - Benchmarks show <100μs parse time

4. **Documentation**
   - Query language reference complete
   - Examples for all operations
   - Error message catalog published

5. **Production Readiness**
   - Zero clippy warnings
   - No memory leaks under sustained load
   - CI regression tests prevent performance degradation

---

## 8. Future Extensions (Out of Scope)

Explicitly deferred to future milestones:

- **Query optimization**: AST rewriting for performance (M13)
- **Distributed queries**: Cross-node query execution (M14)
- **Query caching**: Parse result caching (M16)
- **Visual query builder**: UI for query construction (M17+)
- **Query explain**: EXPLAIN-style execution plan output (M13)

These are valuable but not critical for production deployment. Milestone 9 delivers core query language functionality.

---

## 9. Critical Path

```
001 Parser Infrastructure (2d)
  ↓
002 AST Definition (1d)
  ↓
003 Recursive Descent Parser (3d)
  ↓
004 Error Recovery (2d)
  ↓
005 Query Executor Infrastructure (2d)
  ↓
006 RECALL Implementation (1.5d) ──┐
007 SPREAD Implementation (1.5d) ──┤
  ↓                                 ↓
009 HTTP/gRPC Endpoints (1.5d) ────┘
  ↓
010 Performance Optimization (1d) ──┐
008 Validation Suite (1d) ──────────┤
  ↓                                  ↓
012 Integration Testing (1d)
  ↓
011 Documentation (0.5d)
```

**Total Duration**: 14 days on critical path
**Parallelization Opportunities**: Tasks 006/007 can run in parallel, Task 010 can start when 003 completes

---

## 10. Dependencies

**Hard Dependencies** (must be complete):
- Milestone 1-7: Core memory types, activation spreading, multi-tenant support

**Soft Dependencies** (can work around):
- Milestone 8: Pattern completion (IMAGINE queries stub if not ready)

**No Dependencies On**:
- Milestone 10+: Zig kernels, GPU, distributed (query language independent)

---

## 11. Success Metrics

Milestone 9 succeeds if:

1. **Functionality**: All cognitive operations parse and execute correctly
2. **Performance**: Parse time <100μs P90, total query latency <5ms P99
3. **Quality**: 100% of errors actionable, zero memory leaks
4. **Integration**: HTTP/gRPC endpoints operational, multi-tenant routing works
5. **Maintainability**: Zero clippy warnings, comprehensive test coverage

Post-deployment, track:
- Query language adoption (% queries using DSL vs. direct API)
- Error message effectiveness (support ticket reduction)
- Performance in production (P99 latency, throughput)
