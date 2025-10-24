# Twitter Thread: 100 Ways to Break a Parser

**Tweet 1/7**

Parsers fail. A lot.

The question isn't IF your parser will see invalid input. It will.

The question is: when it fails, does it fail gracefully or catastrophically?

Here's how we built a parser that fails elegantly 100 ways, instead of crashing once.

---

**Tweet 2/7**

Property-based testing generates 1000+ test cases from one property.

Example:
```rust
proptest! {
    fn parse_roundtrip(query in valid_query_generator()) {
        let ast1 = parse(query)?;
        let ast2 = parse(ast1.to_string())?;
        assert_eq!(ast1, ast2);
    }
}
```

One property. 1000 random queries. Automatic.

---

**Tweet 3/7**

Fuzzing finds bugs unit tests miss.

Feed parser 1M random inputs:
- Arbitrary bytes
- Grammar-aware mutations
- Edge cases (unicode, long inputs)

Found: Float overflow, unterminated strings, exponential backtracking

Fixed: All of them.

0 parser panics in 6 months.

---

**Tweet 4/7**

Error messages are your UI.

Bad: "syntax error at position 45"
Better: "Found RECAL, expected keyword"
Best: "Found RECAL, did you mean RECALL? Example: RECALL episode WHERE..."

Levenshtein distance for typo detection.
Context-aware suggestions.
Always include example.

20% → 2% support tickets.

---

**Tweet 5/7**

Parser performance matters.

Slow: 500μs/query → 2k qps
Fast: 50μs/query → 20k qps

Optimizations:
- Keyword hash map: 5x faster
- Arena allocation: 2x faster
- Zero-copy slicing: 1.5x faster

Regression testing: CI fails if >10% slower.

---

**Tweet 6/7**

Test corpus: 150+ queries

75 valid (must parse)
75 invalid (must error correctly)

Each invalid specifies:
- Expected error type
- Required error message content
- Suggested fix

Ensures error quality doesn't regress.

---

**Tweet 7/7**

Results:

Before: 5 panics/month, "syntax error" messages
After: 0 panics, position + suggestion + example

10x reduction in support burden.

Break your parser 100 ways in testing → users break it 0 ways in production.

Code: https://github.com/engram-memory/engram

---

**Hashtags**: #Rust #ParserTesting #PropertyBasedTesting #Fuzzing #SystemsDesign
