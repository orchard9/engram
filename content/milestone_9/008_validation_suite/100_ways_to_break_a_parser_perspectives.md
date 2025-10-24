# Perspectives: 100 Ways to Break a Parser

## Verification Testing Lead Perspective

Parser testing isn't about proving correctness - it's about finding failures systematically.

**Example-Based Tests**: "Does this specific query work?"
**Property-Based Tests**: "Does ANY query satisfy this invariant?"

The shift from examples to properties:
```rust
// Example: Test one case
#[test]
fn test_recall_with_confidence() {
    let result = Parser::parse("RECALL episode WHERE confidence > 0.7");
    assert!(result.is_ok());
}

// Property: Test 1000 cases
proptest! {
    fn parse_roundtrip(query in valid_query_generator()) {
        let ast1 = Parser::parse(&query)?;
        let serialized = ast1.to_string();
        let ast2 = Parser::parse(&serialized)?;
        assert_eq!(ast1, ast2);  // Round-trip invariant
    }
}
```

Properties catch bugs examples miss. One property = 1000+ test cases.

## Systems Architecture Perspective

Parser performance is often ignored until production. Bad idea.

**Cold Path** (no optimization):
```
Parse "RECALL episode WHERE ..." → 500μs
10k queries/sec → parser bottleneck
```

**Hot Path** (optimized):
```
Parse same query → 50μs
100k queries/sec → parser nearly free
```

Optimizations:
1. Keyword hash map (not linear search)
2. Arena allocation (not heap per AST node)
3. Zero-copy string slicing (not clones)
4. Inline hot paths

Result: 10x faster parsing, 10x higher throughput.

## Rust Graph Engine Perspective

Fuzzing parsers prevents memory safety issues.

Rust guarantees memory safety... unless you have logic bugs:
```rust
// Bug: Index out of bounds
fn parse_embedding(&self) -> Result<Vec<f32>> {
    let values = self.tokens[self.pos + 1..self.pos + 1000];  // What if only 10 tokens?
    // Panic at runtime!
}
```

Fuzzer finds this:
```
Input: "RECALL [0.1, 0.2]"
Tokens: [RECALL, LBracket, Float(0.1), Comma, Float(0.2), RBracket]
Expected: 1000 tokens after pos
Actual: 5 tokens
CRASH
```

1M fuzzing iterations finds bugs unit tests miss.

## Cognitive Architecture Perspective

Error messages are cognitive interfaces.

Bad error (cryptic):
```
Parse error at position 45
```

Good error (actionable):
```
Parse error at line 2, column 10:
Found: 'RECAL'
Expected: RECALL

Suggestion: Did you mean 'RECALL'?
Example: RECALL episode WHERE confidence > 0.7
```

User mental model: "I made a typo" → "Here's how to fix it"

Design principle: Errors should teach, not confuse.
