# Example 05: Error Handling

**Learning Goal**: Handle errors gracefully with retry logic, fallback strategies, and educational error messages.

**Difficulty**: Intermediate
**Time**: 20 minutes
**Prerequisites**: Completed Example 01

## Cognitive Concept

Unlike traditional databases that fail hard, Engram embraces uncertainty:

```
Traditional: Success | Failure (binary)
Engram:      Success | Partial | Low Confidence | Retriable Error | Fatal Error
```

Errors teach cognitive concepts and suggest remediation strategies.

## What You'll Learn

- Distinguish retriable vs fatal errors
- Implement exponential backoff
- Handle activation timeouts gracefully
- Use progressive fallback strategies
- Parse educational error messages

## Example Error Scenarios

1. **ERR-2003**: Activation timeout - reduce max_hops
2. **ERR-5003**: Rate limit - implement backoff
3. **ERR-2001**: No results - lower confidence threshold
4. **ERR-1001**: Embedding mismatch - check model dimensions
5. **ERR-4004**: Authentication failure - refresh token

## Code Examples

See language-specific implementations in this directory:

- `rust.rs` - Rust with Result types and retry crate
- `python.py` - Python with tenacity library
- `typescript.ts` - TypeScript with async retry
- `go.go` - Go with context and backoff
- `java.java` - Java with Resilience4j

## Error Handling Patterns

**Pattern 1: Graceful Degradation**
```
Try high threshold → lower threshold → pattern completion → return partial
```

**Pattern 2: Retry with Backoff**
```
Attempt 1: immediate
Attempt 2: wait 1s
Attempt 3: wait 2s
Attempt 4: wait 4s
```

**Pattern 3: Circuit Breaker**
```
After N failures: stop trying, return cached/default
After timeout: try again
```

## Next Steps

- [Error Codes Reference](/reference/error-codes.md) - Complete error catalog
- [Troubleshooting Guide](/operations/troubleshooting.md) - Diagnostic procedures
