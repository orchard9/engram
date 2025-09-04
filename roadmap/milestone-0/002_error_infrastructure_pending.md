# Create error infrastructure with context, suggestions, and examples for every error

## Status: PENDING

## Description
Build comprehensive error handling system that mirrors biological error processing, providing context, actionable suggestions, and code examples for every error condition. Must pass the "3am tired developer" test by supporting System 1 (pattern recognition) thinking when System 2 (deliberate reasoning) is degraded.

## Requirements
- Design CognitiveError struct with mandatory fields:
  - `context`: What the system expected vs what it received
  - `suggestion`: Concrete action to resolve the issue  
  - `example`: Working code snippet showing correct usage
  - `confidence`: Error confidence score (for partial failures)
- Implement Display with progressive disclosure:
  - One-line summary for scanning (System 1)
  - Detailed context on demand (System 2)
  - Examples and docs as third level
- Create error builder macros that enforce all fields at compile time
- Add error context enrichment through call chains (not just wrapping)
- Support graceful degradation: return partial results with confidence scores instead of hard failures
- Response time: error formatting must complete in <1ms to maintain <60s startup

## Acceptance Criteria
- [ ] CognitiveError type with mandatory context/suggestion/example/confidence
- [ ] Display shows progressive disclosure (summary → details → examples)
- [ ] `cognitive_error!` macro enforces all fields at compile time
- [ ] Error chaining enriches rather than obscures context
- [ ] Partial results with confidence for recoverable errors
- [ ] Similar error detection: "Did you mean X?" suggestions using edit distance
- [ ] Example passing "3am test": 
  ```
  Memory node 'user_123' not found (confidence: 0.9)
  Expected: Valid node ID in current graph
  Suggestion: Use graph.nodes() to list available nodes, or did you mean 'user_124'?
  Example: let node = graph.get_node("user_124").or_insert_default();
  ```

## Implementation Guide
```rust
pub struct CognitiveError {
    // Required fields - no Options!
    context: ErrorContext,
    suggestion: String,
    example: String,
    confidence: Confidence,
    
    // For progressive disclosure
    details: Option<String>,
    documentation: Option<String>,
    similar: Vec<String>, // "Did you mean?" candidates
}

// Macro enforces all fields
cognitive_error!(
    context: expected = "embedding dimension 768",
             actual = "512",
    suggestion: "Use Config::embedding_dim(512)",
    example: "config.embedding_dim(512).build()",
    confidence: Confidence::exact(1.0)
);
```

## Dependencies
- Task 001 (workspace setup) ✓

## Notes
- Use thiserror for base derive, but wrap with our CognitiveError
- NO color-eyre in library code (only CLI)
- Must support infallible APIs through graceful degradation
- Align with ERN (Error-Related Negativity) principle: detect errors in <100ms
- Follow biological error correction: signal → adjustment → learned pattern