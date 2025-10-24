# Research: AST Design for Cognitive Query Languages

## Topic Overview

Abstract Syntax Trees (ASTs) for cognitive operations differ fundamentally from database query ASTs. Instead of representing relational algebra (SELECT, JOIN, WHERE), they must represent memory operations (RECALL, SPREAD, CONSOLIDATE) that map to biological processes.

## Key Research Areas

### 1. AST Design Principles

**Traditional Database AST**
```
SELECT * FROM users WHERE age > 18
└─ SelectStatement
   ├─ Projection: [*]
   ├─ From: [users]
   └─ Filter: BinaryOp(age, >, 18)
```

**Cognitive Query AST**
```
RECALL episode WHERE confidence > 0.7
└─ RecallQuery
   ├─ Pattern: NodeId("episode")
   └─ Constraints: [ConfidenceAbove(0.7)]
```

The difference: Cognitive ASTs model operations on memory, not data transformations.

**Reference**: "Cognitive Architectures for Language Processing" - Crocker et al. (2010)

### 2. Type-State Pattern for Compile-Time Validation

**Problem**: Invalid queries can be constructed at runtime

```rust
// Bad: Can forget to set pattern
let query = RecallQuery {
    pattern: None,  // Oops, forgot this!
    constraints: vec![],
};
```

**Solution**: Phantom type parameters track builder state

```rust
struct Builder<State> {
    pattern: Option<Pattern>,
    _phantom: PhantomData<State>,
}

// Only BuilderWithPattern has build() method
impl Builder<WithPattern> {
    fn build(self) -> RecallQuery { ... }
}
```

Now you literally cannot call `build()` until pattern is set - compiler error at compile time, not runtime error.

**Reference**: "Type-State Pattern" - Rust API Guidelines
https://rust-lang.github.io/api-guidelines/predictability.html#c-builder-state

**Real-world examples**:
- hyper::Client builder (TLS state tracking)
- tokio::net::TcpStream (connection state)
- reqwest::RequestBuilder (method/URL state)

### 3. Memory Layout Optimization

**Enum Size Formula**:
```
size = discriminant + max(variant_sizes) + padding
```

For cognitive queries:
```rust
enum Query {
    Recall(RecallQuery),   // ~120 bytes
    Spread(SpreadQuery),   // ~64 bytes
    Predict(PredictQuery), // ~128 bytes
}
```

Total size: 1 byte (discriminant) + 128 bytes (largest variant) + 7 bytes (padding) = 136 bytes

Target: <256 bytes (4 cache lines)

**Reference**: "Understanding Rust Enum Memory Layout"
https://doc.rust-lang.org/reference/type-layout.html#the-c-representation

**Optimization techniques**:
1. Box large variants: `Recall(Box<RecallQuery>)` - adds indirection but shrinks enum
2. Inline small data: Use arrays for small vectors
3. Compress fields: u16 instead of usize when possible

### 4. Zero-Copy AST with Lifetimes

**Challenge**: AST should reference source text when possible

```rust
// Without lifetime: Always owns strings
struct Pattern {
    node_id: String,  // Always allocates
}

// With lifetime: Can reference source
struct Pattern<'a> {
    node_id: Cow<'a, str>,  // Borrows when possible
}
```

**Lifetime propagation**:
```rust
RecallQuery<'a> contains Pattern<'a>
Pattern<'a> contains Cow<'a, str>
Cow<'a, str> references source: &'a str
```

The entire AST is parameterized by source lifetime.

**Reference**: "Lifetimes in Rust" - Rust Book Chapter 10
https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html

**Trade-off**: Complexity vs. performance
- Pro: Zero allocations during parsing
- Con: Lifetime parameters propagate through entire codebase
- Mitigation: Provide `into_owned()` escape hatch

### 5. Validation Patterns

**When to validate**:

1. **Construction time**: Prevent invalid ASTs from existing
2. **Parse time**: Validate after constructing AST
3. **Execution time**: Validate before running query

We validate at parse time (option 2) because:
- Parsing is one-time cost
- Execution may happen multiple times (query caching)
- Error messages are better with source context

**Validation strategies**:

```rust
impl RecallQuery {
    fn validate(&self) -> Result<(), ValidationError> {
        // 1. Structural validation
        self.pattern.validate()?;

        // 2. Constraint validation
        for constraint in &self.constraints {
            constraint.validate()?;
        }

        // 3. Cross-field validation
        if self.confidence_threshold.is_some() && self.base_rate.is_some() {
            // Check Bayesian consistency
        }

        Ok(())
    }
}
```

**Reference**: "Parse, Don't Validate" - Alexis King
https://lexi-lambda.github.io/blog/2019/11/05/parse-dont-validate/

Key insight: Use types to make invalid states unrepresentable, but validate at boundaries.

### 6. Serde Integration for Debugging

**Why serialize ASTs**:
1. Debugging: Pretty-print parsed queries
2. Logging: Audit query history
3. Caching: Store parsed queries
4. RPC: Send queries across network

**Lifetime-compatible serialization**:

```rust
#[derive(Serialize, Deserialize)]
struct NodeIdentifier<'a>(
    #[serde(borrow)] Cow<'a, str>
);
```

The `#[serde(borrow)]` annotation tells serde to preserve zero-copy semantics when deserializing.

**Reference**: Serde documentation - Zero-copy deserialization
https://serde.rs/lifetimes.html

**Gotcha**: Deserializing into borrowed data requires the source buffer to outlive the deserialized object. Often easier to deserialize into owned types:

```rust
let query: Query<'static> = serde_json::from_str(json)?;
```

### 7. Cognitive Operation Semantics

**RECALL**: Episodic memory retrieval
- Pattern: What to recall (cue)
- Constraints: Filtering criteria
- Confidence: Threshold for retrieval strength
- Base rate: Prior probability (Bayesian)

**SPREAD**: Activation spreading
- Source: Starting node
- Max hops: Limit propagation depth
- Decay rate: Activation falloff per edge
- Threshold: Minimum activation to propagate

**IMAGINE**: Pattern completion
- Pattern: Partial template
- Seeds: Context nodes
- Novelty: Balance between familiarity and creativity
- Confidence: Required completion certainty

**CONSOLIDATE**: Episodic to semantic transfer
- Episodes: What to consolidate
- Target: Semantic memory destination
- Scheduler: When to trigger consolidation

**PREDICT**: Temporal projection
- Pattern: What to predict
- Context: Given information
- Horizon: Time window
- Confidence interval: Prediction uncertainty

Each operation maps to computational neuroscience literature:

**Reference**: "Computational Models of Memory" - O'Reilly & Munakata (2000)

### 8. Constraint Representation

**Constraint types in cognitive systems**:

1. **Similarity constraints**: Vector distance thresholds
   ```rust
   SimilarTo { embedding: Vec<f32>, threshold: f32 }
   ```

2. **Temporal constraints**: Event ordering
   ```rust
   CreatedBefore(SystemTime)
   CreatedAfter(SystemTime)
   ```

3. **Confidence constraints**: Retrieval certainty
   ```rust
   ConfidenceAbove(Confidence)
   ```

4. **Spatial constraints**: Memory space boundaries
   ```rust
   InMemorySpace(MemorySpaceId)
   ```

5. **Content constraints**: Semantic matching
   ```rust
   ContentContains(String)
   ```

**Design principle**: Constraints should compose naturally

```rust
WHERE confidence > 0.7
  AND created_before "2024-01-01"
  AND content CONTAINS "neural network"
```

Maps to:
```rust
constraints: vec![
    ConfidenceAbove(0.7),
    CreatedBefore(timestamp),
    ContentContains("neural network"),
]
```

**Reference**: "Constraint-Based Reasoning" - Dechter (2003)

### 9. Pattern Matching Strategies

**Pattern types for memory retrieval**:

1. **Exact match**: Node ID
   ```rust
   Pattern::NodeId("episode_123")
   ```

2. **Similarity match**: Embedding vector
   ```rust
   Pattern::Embedding {
       vector: embedding,
       threshold: 0.8
   }
   ```

3. **Content match**: Substring search
   ```rust
   Pattern::ContentMatch("neural network")
   ```

4. **Wildcard**: Match any
   ```rust
   Pattern::Any
   ```

**Cognitive parallel**: These mirror how episodic memory retrieval works:
- Exact: "The meeting on Tuesday" (specific episodic recall)
- Similarity: "Meetings like the one on Tuesday" (pattern-based retrieval)
- Content: "Meetings about neural networks" (semantic cue)
- Wildcard: "All meetings" (unconstrained retrieval)

**Reference**: "Episodic Memory Retrieval" - Tulving (1983)

### 10. Confidence Representation

**Confidence is first-class in cognitive queries**:

```rust
enum ConfidenceThreshold {
    Above(Confidence),           // confidence > 0.7
    Below(Confidence),           // confidence < 0.3
    Between { lower, upper },    // confidence IN [0.6, 0.8]
}
```

**Why not just use f32**:
1. Type safety: Confidence is in [0, 1]
2. Semantics: Confidence has special meaning (probability)
3. Integration: Confidence type used throughout Engram

**Bayesian priors**:
```rust
struct RecallQuery {
    confidence_threshold: Option<ConfidenceThreshold>,
    base_rate: Option<Confidence>,  // Prior P(recall)
}
```

Base rate enables Bayesian reasoning:
```
P(A|B) = P(B|A) * P(A) / P(B)

base_rate = P(recall succeeds)
threshold = minimum P(recall|evidence)
```

**Reference**: "Probabilistic Models of Cognition" - Griffiths et al.
https://probmods.org/

### 11. Error Message Design

**Validation errors should be actionable**:

```rust
#[error("Invalid embedding dimension: expected {expected}, got {actual}
  Expected: 768-dimensional vector (current system default)
  Suggestion: Ensure embedding model matches system configuration
  Example: Use text-embedding-ada-002 or compatible 768-dim model")]
InvalidEmbeddingDimension { expected: usize, actual: usize }
```

Each error includes:
1. **What went wrong**: Clear description
2. **What was expected**: Specification
3. **How to fix it**: Actionable suggestion
4. **Example**: Concrete usage

**Reference**: "Writing Great Error Messages" - Rust Error Handling Project Group
https://blog.rust-lang.org/2021/05/15/seven-years-of-rust.html

**Cognitive principle**: Error messages are a form of pattern completion - the system predicts what the user meant and suggests the corrected form.

### 12. Builder Pattern Ergonomics

**Problem**: Validating ASTs manually is tedious

```rust
let query = RecallQuery { ... };
query.validate()?;  // Easy to forget
```

**Solution**: Builder enforces validation

```rust
let query = RecallQueryBuilder::new()
    .pattern(Pattern::NodeId("episode".into()))
    .constraint(Constraint::ConfidenceAbove(0.7))
    .build()?;  // Validation happens here
```

**Type-state prevents invalid builds**:

```rust
// Won't compile - no pattern set
let query = RecallQueryBuilder::new()
    .build();  // ERROR: method not found

// Compiles - pattern is set
let query = RecallQueryBuilder::new()
    .pattern(...)
    .build();  // OK
```

**Reference**: "The Builder Pattern" - Rust Design Patterns
https://rust-unofficial.github.io/patterns/patterns/creational/builder.html

## Performance Considerations

### 1. AST Size Budgets

**Cache efficiency requires size limits**:

```rust
// Compile-time size assertions
const _: () = assert!(size_of::<Query>() < 256);
const _: () = assert!(size_of::<Pattern>() < 128);
const _: () = assert!(size_of::<Constraint>() < 64);
```

**Why 256 bytes**: Fits in 4 cache lines (4 × 64 bytes)

**Why it matters**: AST traversal during validation/execution should stay cache-resident.

### 2. Allocation Profiling

**Where ASTs allocate**:
1. Vec for constraints (heap)
2. String for node IDs (heap if owned)
3. Vec for embeddings (heap)

**Measurement**:
```rust
#[test]
fn test_parse_allocations() {
    let before = allocation_count();
    let query = parse("RECALL episode");
    let after = allocation_count();

    // Should only allocate: Vec for constraints (1 alloc)
    assert_eq!(after - before, 1);
}
```

### 3. Copy-on-Write Semantics

**When to clone AST**:
- Never during parsing (use borrowed data)
- Only when storing beyond source lifetime
- Use `into_owned()` method explicitly

```rust
let query: Query<'a> = parse(source)?;
let owned: Query<'static> = query.into_owned();  // Clone happens here
```

## Open Questions

1. **Should constraints be a Vec or SmallVec**?
   - Pro (SmallVec): Inline 2-3 constraints (common case)
   - Con: More complex code, potential waste for queries with >3 constraints

2. **Should we use interned strings for node IDs**?
   - Pro: Deduplication if same node appears multiple times
   - Con: Complexity, global state, thread safety

3. **Should Pattern be generic over embedding dimension**?
   - Pro: Type-level guarantee of dimension (Pattern<768>)
   - Con: Increased complexity, monomorphization cost

4. **Should we support query composition** (sub-queries)?
   - Pro: Expressiveness (RECALL x WHERE x IN (SPREAD FROM y))
   - Con: Complexity, harder to optimize

## References

### Academic Papers
1. "Cognitive Architectures" - Anderson et al. (2004)
2. "Computational Memory Models" - Norman & O'Reilly (2003)
3. "Type-State for Typestate" - Strom & Yemini (1986)

### Rust Design Patterns
1. The Rust Programming Language: https://doc.rust-lang.org/book/
2. Rust API Guidelines: https://rust-lang.github.io/api-guidelines/
3. Effective Rust: https://www.lurklurk.org/effective-rust/

### Performance Resources
1. Rust Performance Book: https://nnethercote.github.io/perf-book/
2. "Computer Systems: A Programmer's Perspective" - Bryant & O'Hallaron
3. "Systems Performance" - Brendan Gregg

### Cognitive Science
1. "How Memory Works" - Baddeley et al. (2009)
2. "The Cambridge Handbook of Computational Psychology" (2008)
3. Computational Cognitive Neuroscience: https://grey.colorado.edu/CompCogNeuro/
