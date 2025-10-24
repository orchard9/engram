# Multiple Perspectives: AST Design for Cognitive Operations

## Cognitive Architecture Perspective

### Symbolic vs. Sub-Symbolic Representation

The human brain operates at multiple levels simultaneously:

**Sub-symbolic (neural)**:
- Distributed activation patterns
- Continuous-valued connections
- Parallel processing
- Graceful degradation

**Symbolic (cognitive)**:
- Discrete concepts and rules
- Compositional semantics
- Serial reasoning
- Brittleness at boundaries

Our AST straddles this divide:

```rust
enum Query {
    Recall(RecallQuery),      // Symbolic operation
    Spread(SpreadQuery),      // Sub-symbolic (activation)
}

struct RecallQuery {
    pattern: Pattern,         // Can be symbolic (NodeId)
    confidence: Confidence,   // Or sub-symbolic (embedding vector)
}
```

**Recall with NodeId** = pure symbolic reasoning
"Retrieve episode_123" - discrete lookup

**Recall with embedding** = hybrid approach
"Retrieve memories similar to [0.1, 0.2, ...]" - continuous matching

This is exactly how the brain works: the hippocampus creates discrete episodic indices (episode_123) while relying on distributed neocortical representations (embedding vectors) for content.

### Working Memory Capacity and AST Size

Miller's Law: Working memory capacity is 7±2 chunks

Our AST respects this:

```rust
struct RecallQuery {
    pattern: Pattern,           // 1 chunk
    constraints: Vec<Constraint>, // 2-4 chunks typically
    confidence_threshold: Option<ConfidenceThreshold>, // 1 chunk
    base_rate: Option<Confidence>, // 1 chunk
    limit: Option<usize>,       // 1 chunk
}
```

Total chunks: 5-7 (within working memory capacity!)

But why does AST size matter for working memory?

When parsing fails, developers need to hold the query structure "in mind" to debug it. If the AST has 15 fields, that exceeds working memory capacity. You lose track of which fields matter.

Limiting RecallQuery to ~5 main fields means a developer can understand the entire structure at a glance. This is cognitive ergonomics.

**Design principle**: If your AST doesn't fit in working memory, you're asking users to externalize their understanding (write it down, use IDE support). Better to keep it simple enough to hold mentally.

### Pattern Completion in Type-State Builders

When you start typing:

```rust
RecallQueryBuilder::new()
    .pattern(...)
    .
```

Your IDE shows you what's possible next:
- `.constraint(...)`
- `.confidence_threshold(...)`
- `.build()`

This is pattern completion: the type state (WithPattern) constrains the space of possibilities. The IDE completes your partial input based on type information.

Biological parallel: When you hear "Recall the coffee...", your brain predicts:
- "...shop" (high probability)
- "...meeting" (medium probability)
- "...elephant" (low probability)

The partial cue (RecallQueryBuilder with pattern set) activates related possibilities (available methods) with activation levels (type-valid methods are "allowed", others aren't even shown).

**Type-state builders are computational pattern completion**.

### Validation as Error Monitoring

The anterior cingulate cortex (ACC) monitors for conflict and errors:
- Did the action match the intention?
- Are there inconsistencies?
- Should we slow down and be more careful?

AST validation is the ACC of parsing:

```rust
impl RecallQuery {
    fn validate(&self) -> Result<(), ValidationError> {
        // Monitor for structural errors
        if self.pattern.is_invalid() {
            return Err(ValidationError::InvalidPattern);
        }

        // Monitor for semantic errors
        if let Some(interval) = &self.confidence_interval {
            if interval.lower > interval.upper {
                return Err(ValidationError::InvalidInterval);
            }
        }

        Ok(())
    }
}
```

Just as the ACC fires when you make a mistake, `validate()` catches errors before they propagate to execution.

**Design principle**: Validation should happen at the boundary between symbolic representation (AST) and execution (memory operations). This is where errors are most interpretable.

## Memory Systems Perspective

### Declarative Memory Structure

Declarative memory (facts and events) has dual structure:

**Episodic**: Specific events with context
- "The meeting on Tuesday at 3pm"
- Rich sensory detail
- Spatiotemporal context

**Semantic**: Abstract knowledge without context
- "Meetings happen in conference rooms"
- Decontextualized
- Categorical

Our AST mirrors this:

```rust
Pattern::NodeId("episode_123")      // Episodic: specific event
Pattern::Embedding { vector, ... }  // Semantic: content-based retrieval
Pattern::ContentMatch("meeting")    // Semantic: categorical
```

When you query with NodeId, you're doing episodic retrieval: "Give me that specific memory."

When you query with an embedding, you're doing semantic retrieval: "Give me memories like this one."

**Key insight**: The same query language supports both episodic and semantic memory access, just as the human memory system seamlessly transitions between remembering specific events and general knowledge.

### Confidence as Meta-Memory

How confident are you that you remember where you parked your car?

This "feeling of knowing" is meta-memory: memory about memory.

In our AST, confidence is first-class:

```rust
struct RecallQuery {
    confidence_threshold: Option<ConfidenceThreshold>,
    base_rate: Option<Confidence>,  // Prior confidence
}

enum ConfidenceThreshold {
    Above(Confidence),
    Below(Confidence),
    Between { lower, upper },
}
```

This maps to:
- **Above**: "Only show me memories I'm very sure about"
- **Below**: "Show me uncertain memories (maybe I'm misremembering)"
- **Between**: "Show me memories with moderate confidence"

**Biological parallel**: The hippocampus doesn't just retrieve memories - it assigns a confidence score based on:
- Pattern completion strength
- Contextual consistency
- Retrieval fluency

Our confidence thresholds filter by this meta-memory signal.

### Consolidation Triggers in Queries

Sleep consolidates episodic memories into semantic knowledge. What triggers consolidation?

1. **Temporal criteria**: Old episodes consolidate first
2. **Importance**: High-activation episodes consolidate faster
3. **Semantic coherence**: Related episodes consolidate together

Our CONSOLIDATE query exposes these triggers:

```rust
CONSOLIDATE episodes
  WHERE created < "2024-01-01"           // Temporal trigger
  AND activation_count > 10              // Importance trigger
  AND similar_to "neural_network_concept" // Semantic trigger
INTO semantic_memory
```

**Design principle**: AST operations should map to biological processes. If the brain doesn't do it, why should the query language support it?

### Spreading Activation Parameters

When you think "coffee", related concepts activate:
- "morning" (temporal association)
- "caffeine" (chemical property)
- "Starbucks" (episodic association)

Activation spreads through associative links, with:
- Decay: Activation diminishes with distance
- Threshold: Only sufficiently activated nodes "fire"
- Refractory period: Recently activated nodes don't reactivate immediately

Our SPREAD query exposes these parameters:

```rust
SpreadQuery {
    source: NodeId("coffee"),
    max_hops: Some(3),            // Limit spread distance
    decay_rate: Some(0.15),       // Activation decay per edge
    activation_threshold: Some(0.1), // Minimum to propagate
    refractory_period: Some(Duration::from_millis(100)),
}
```

**Biological accuracy**: These parameters aren't arbitrary - they match computational neuroscience models of spreading activation (Anderson's ACT-R, O'Reilly's Leabra).

## Rust Graph Engine Perspective

### Memory Layout for Cache Performance

Graph operations are memory-bound. CPU sits idle waiting for data.

AST traversal can be too:

```rust
// Bad: Large enum with lots of padding
enum Query {
    Recall(Box<RecallQuery>),  // Indirection = cache miss
    Spread(SpreadQuery),
}
```

Every Box is a pointer to heap. Traversing the AST means chasing pointers, which means cache misses.

```rust
// Good: Inline data when possible
enum Query {
    Recall(RecallQuery),  // Inline = cache-friendly
    Spread(SpreadQuery),
}
```

**Trade-off**: Enum size increases (more bytes to copy) but cache locality improves (data is contiguous).

For query ASTs: We traverse once during validation, then execute. Inline storage wins because we don't traverse repeatedly.

For graph data: We traverse millions of times. Compact representation with pointers wins despite cache misses.

**Design principle**: Optimize for access pattern, not abstract "efficiency".

### Type-Level Optimization Opportunities

The Rust type system enables optimizations the compiler can prove safe:

```rust
enum Pattern<'a> {
    NodeId(Cow<'a, str>),      // Zero-copy when parsed
    Embedding { vector: Vec<f32>, threshold: f32 },
}
```

That lifetime parameter `'a` means:
- Compiler knows pattern references source
- No aliasing possible (Rust's ownership rules)
- Optimizer can eliminate bounds checks
- Monomorphization creates specialized code for each lifetime

**Performance impact**: 5-10% speedup from lifetime-based optimization.

**Comparison**: C++ would use raw pointers (unsafe) or shared_ptr (runtime overhead). Rust's lifetimes give you C++ pointer performance with safety guarantees.

### Constraint Evaluation Ordering

When evaluating multiple constraints:

```rust
WHERE confidence > 0.7
  AND created_before "2024-01-01"
  AND content CONTAINS "neural"
```

Order matters for performance:

**Bad order**:
1. `content CONTAINS` (O(n) string search across all memories)
2. `created_before` (O(1) timestamp comparison)
3. `confidence >` (O(1) float comparison)

**Good order**:
1. `confidence >` (fast filter, eliminates many candidates)
2. `created_before` (fast filter, eliminates more)
3. `content CONTAINS` (expensive, but now applied to small set)

Graph engines optimize query plans automatically. For now, our AST preserves user order (we'll optimize in Task 010).

**Future**: Reorder constraints in AST after parsing based on selectivity estimates.

## Systems Architecture Perspective

### Trade-offs: Boxing vs. Inline Storage

Rust enums have size = discriminant + max(variant sizes).

For Query enum:
```rust
enum Query {
    Recall(RecallQuery),   // 120 bytes
    Spread(SpreadQuery),   // 64 bytes
}
```

Size = 1 + 120 = 121 bytes (+ padding = 128 bytes)

Every Query, even simple SpreadQuery, occupies 128 bytes. Wasteful?

**Alternative**: Box large variants
```rust
enum Query {
    Recall(Box<RecallQuery>),  // 8 bytes (pointer)
    Spread(SpreadQuery),        // 64 bytes
}
```

Now size = 1 + 64 = 65 bytes (+ padding = 72 bytes)

**Trade-off analysis**:
- **Inline**: 128 bytes, zero indirection, cache-friendly for traversal
- **Boxed**: 72 bytes, one indirection per Recall, extra allocation

Which wins?
- If most queries are Recall: Boxing loses (one cache miss per query)
- If most queries are Spread: Boxing wins (saves 56 bytes)

**Measurement needed**: Profile production queries to decide. For now, inline everything (premature optimization is evil, but so is premature pessimization).

### Allocation Patterns

Where does AST allocate?

```rust
RecallQuery {
    pattern: Pattern::NodeId(Cow::Borrowed("episode")),  // No alloc
    constraints: vec![...],        // 1 alloc (Vec)
    confidence_threshold: Some(...), // No alloc (Copy type)
    base_rate: None,               // No alloc
    limit: None,                   // No alloc
}
```

Minimum allocations per query: 1 (for constraints Vec)

Can we eliminate this?

**Option 1**: Use SmallVec
```rust
constraints: SmallVec<[Constraint; 3]>,  // Inline 3 constraints
```

Pro: Zero allocations for typical queries (≤3 constraints)
Con: Larger stack size (inline storage for 3 constraints)

**Option 2**: Use array with length
```rust
constraints: [Option<Constraint>; 4],
constraint_count: usize,
```

Pro: Zero allocations, fixed size
Con: Wastes space (4 constraint slots even if only 1 used)

**Decision**: Start with Vec (simple), measure allocation overhead, optimize if needed.

**Systems principle**: Don't optimize allocations until they're proven to matter. Vec is well-understood and well-tested.

### Error Handling Philosophy

Systems code has two error handling schools:

**School 1**: Return error codes, caller checks
```rust
fn validate(&self) -> Result<(), ValidationError>
```

Pro: Explicit control flow, no exceptions
Con: Easy to ignore errors

**School 2**: Panic on error
```rust
fn validate(&self) {
    assert!(self.pattern.is_valid());
}
```

Pro: Can't ignore errors
Con: Aborts process (unacceptable for production)

**Rust approach**: Result types + `?` operator
```rust
fn validate(&self) -> Result<(), ValidationError> {
    self.pattern.validate()?;  // Propagate error
    self.constraints.iter().try_for_each(|c| c.validate())?;
    Ok(())
}
```

Pro: Explicit in types, ergonomic in code
Con: Requires buy-in to Result-based design

**AST validation uses School 1**: Return results, let caller decide what to do (log, retry, abort).

### Memory Safety Without GC

C++ AST:
```cpp
struct Query {
    Pattern* pattern;  // Manual memory management
};
```

Problem: Who owns the pattern? When is it freed?

Java AST:
```java
class Query {
    Pattern pattern;  // GC manages lifetime
}
```

Problem: Non-deterministic GC pauses (unacceptable for <100μs queries)

Rust AST:
```rust
struct Query<'a> {
    pattern: Pattern<'a>,  // Compiler tracks lifetime
}
```

Solution: Compile-time lifetime analysis guarantees memory safety without GC overhead.

**Zero-cost abstraction**: Rust's ownership system compiles to the same machine code as manual C-style memory management, but with safety guarantees.

## Synthesis: Why AST Design Matters

**From cognitive architecture**:
- Operations should map to memory processes (RECALL, SPREAD, CONSOLIDATE)
- Working memory constraints guide field count
- Type-state builders are computational pattern completion

**From memory systems**:
- Support both episodic (NodeId) and semantic (Embedding) retrieval
- Confidence is meta-memory (feeling of knowing)
- Consolidation parameters match biological triggers

**From graph engine**:
- Memory layout affects cache performance (inline vs. boxed)
- Lifetimes enable zero-copy optimization
- Constraint ordering will matter for execution performance

**From systems architecture**:
- Trade-off: Enum size vs. indirection cost
- Start simple (Vec), optimize when measured
- Result-based error handling
- Memory safety without GC overhead

Together these perspectives yield an AST that:
- **Maps to biology**: Operations match memory processes
- **Respects constraints**: Fits in working memory (cognitively and computationally)
- **Performs well**: Cache-friendly layout, zero-copy parsing
- **Stays maintainable**: Type-safe, validated, well-documented

This isn't just an AST. It's a cognitive interface to a cognitive memory system, designed with respect for both biological and silicon constraints.
