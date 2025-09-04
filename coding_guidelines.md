# Coding Guidelines

## Rust Edition 2024

All code uses Rust Edition 2024. Leverage these edition features:
- Native async traits (no `async-trait` crate needed)
- `let-else` for cleaner error handling
- Improved pattern matching and type inference
- Field defaults with `#[derive(Default)]`

## Rust Conventions

### Error Handling (Cognitive Guidance Principle)

#### Core Philosophy
Errors are not exceptions—they're cognitive signals that guide developers toward correct behavior. Every error must pass the "3am tired developer test" by supporting System 1 (pattern recognition) when System 2 (deliberate reasoning) is degraded.

#### Requirements
- Never use `unwrap()` except in tests
- Never use `expect()` except with proof of impossibility  
- All errors must include context, suggestion, and example
- Use `CognitiveError` type with mandatory fields (no Options!)
- Progressive disclosure: one-line summary → details → examples
- Graceful degradation: return partial results with confidence scores over hard failures

#### Implementation
```rust
// BAD: Just states the problem
#[error("Node not found: {0}")]
NodeNotFound(String),

// GOOD: Context + Suggestion + Example
cognitive_error!(
    context: expected = "valid node ID",
             actual = node_id.clone(),
    suggestion: "Use graph.nodes() to list available nodes",
    example: "let node = graph.get_node(\"user_124\").or_insert_default();",
    confidence: Confidence::high(),
    similar: graph.find_similar_nodes(&node_id) // "Did you mean?"
);
```

#### Error Response Time
- Error detection: <100ms (mirrors brain's ERN response)
- Error formatting: <1ms (maintains <60s startup target)
- Suggestion computation: <10ms (including edit distance)

#### Biological Alignment
- **ERN Principle**: Detect errors before conscious awareness (compile-time > runtime)
- **Cerebellar Model**: Signal → Adjustment → Pattern (context → suggestion → example)
- **Hippocampal Pattern Completion**: Partial results better than none
- **Prediction Error**: Errors update future suggestions

### Memory Management
- Prefer `Arc<T>` over `Rc<T>` for future concurrent access
- Use `Box<[T]>` over `Vec<T>` for fixed-size collections
- Implement `Drop` explicitly for resources requiring cleanup
- Never leak memory in hot paths - verify with `valgrind`

### Concurrency
- Document every `unsafe` block with safety proof
- Prefer `parking_lot` over `std::sync` primitives
- Use `crossbeam` channels over `std::sync::mpsc`
- All shared state requires explicit synchronization strategy documentation
- Prefer lock-free structures where contention is expected

### Performance
- `#[inline]` hot path functions under 5 lines
- `#[cold]` error handling paths
- Use `SmallVec` for collections typically < 32 elements
- Benchmark before optimizing - no speculative performance code
- SIMD operations require fallback scalar implementation

### API Design
- Builder pattern for types with >3 constructor parameters
- All public types implement `Debug`
- Implement `Default` only when meaningful default exists
- Use newtypes for domain concepts, not primitive aliases
- Return `impl Trait` for iterator chains

### Type-State Pattern (Compile-Time Validation)
Make invalid states unrepresentable using phantom types:

```rust
// Memory states encoded in type system
pub struct Memory<State> {
    confidence: Confidence, // Always present, never Option<f32>
    data: Vec<u8>,
    _state: PhantomData<State>,
}

// States as zero-sized types
pub struct Unvalidated;
pub struct Validated;
pub struct Consolidated;

// State transitions enforced at compile time
impl Memory<Unvalidated> {
    pub fn validate(self) -> Result<Memory<Validated>, CognitiveError> {
        // Validation logic
    }
}

impl Memory<Validated> {
    pub fn consolidate(self) -> Memory<Consolidated> {
        // Can only consolidate validated memories
    }
}
```

Benefits:
- 60% reduction in runtime errors (Aldrich et al., 2009)
- Zero runtime overhead - types erased at compilation
- Invalid state transitions become compilation errors
- Self-documenting API through type signatures

## Zig Conventions

### Memory Safety
- Every allocation paired with explicit deallocation
- Use arena allocators for batch operations
- Document ownership transfer at function boundaries
- Compile-time verify buffer sizes where possible
- Runtime bounds checks in debug, removed in release

### Performance
- `@setRuntimeSafety(false)` only in measured hot paths
- Align data structures to cache lines (64 bytes)
- Use `@prefetch` for predictable access patterns
- Vectorize with `@Vector` types, not manual SIMD
- Comptime generate specialized functions over runtime dispatch

## General Principles

### Comments
- Document why, not what
- Every `unsafe` requires safety justification
- Public API requires doctest examples
- Invariants documented at type definition
- Performance assumptions documented with benchmarks

### Testing
- Unit test pure functions exhaustively
- Property-based testing for probabilistic operations
- Fuzzing for parser and serialization code
- Integration tests use deterministic seeds
- Benchmarks track allocations, not just time

### Code Organization
- One concept per module
- Implementation details in submodules
- Public API surface minimal
- Dependencies injected, not global
- Traits for behavior, structs for data

### Naming
- Types: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Type parameters: single capital letter or meaningful name
- Lifetimes: `'a`, `'b` or semantic like `'txn`, `'node`

### Commits
- Imperative mood: "Add X", not "Added X"
- First line < 50 characters
- Body explains why, not what changed
- Reference issue number in footer
- Atomic: one logical change per commit

## Forbidden Patterns

1. **No global mutable state** - Use dependency injection
2. **No stringly-typed APIs** - Use enums and newtypes
3. **No boolean parameters** - Use enums for clarity
4. **No primitive obsession** - Wrap domain concepts
5. **No async in library code** - Let applications choose runtime
6. **No macros for syntax sugar** - Only for compile-time code generation
7. **No dependencies on unstable features** - Edition 2024 stable features only
8. **No synchronous I/O in hot paths** - Always async or memory-mapped
9. **No arbitrary limits** - Make configurable or unbounded
10. **No silent failures** - Log or propagate every error

## Review Checklist

- [ ] No `unwrap()` outside tests
- [ ] All `unsafe` blocks documented
- [ ] Public types implement `Debug`
- [ ] Tests include failure cases
- [ ] Benchmarks included for performance claims
- [ ] Documentation includes examples
- [ ] **Error messages pass "3am tired developer test"**
- [ ] **All errors include context, suggestion, and example**
- [ ] **Confidence scores for partial results (no hard failures)**
- [ ] **Type-state pattern used for compile-time validation**
- [ ] **"Did you mean?" suggestions for similar options**
- [ ] No TODO comments in production code
- [ ] Resource cleanup verified
- [ ] Thread safety documented

## Performance Requirements

Every merge requires:
- Benchmark comparison against main branch
- Memory usage regression test
- Latency percentiles (P50, P95, P99)
- Allocation counts for hot paths
- Flamegraph for CPU usage over 5%

## Breaking Changes

Semver strictly enforced:
- Major: Any public API change
- Minor: New public API additions
- Patch: Internal changes only

Document migration path for every breaking change.
