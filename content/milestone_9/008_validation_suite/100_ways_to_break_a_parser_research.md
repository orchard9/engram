# Research: 100 Ways to Break a Parser

## Research Questions

1. What are the most common parser failure modes in production systems?
2. How do property-based tests differ from example-based tests for parsers?
3. What makes error messages actionable vs confusing?
4. How do we systematically generate invalid queries that expose edge cases?
5. What are realistic performance targets for production query parsers?

## Key Findings

### 1. Parser Failure Taxonomy

**Syntax Errors** (40% of failures):
- Missing keywords: `episode WHERE confidence > 0.7` (forgot RECALL)
- Typos: `RECAL` instead of `RECALL`
- Wrong keyword order: `WHERE RECALL episode`
- Unterminated literals: `[0.1, 0.2` (missing `]`)
- Invalid operators: `confidence >> 0.7`

**Semantic Errors** (30% of failures):
- Out-of-range values: `confidence > 1.5` (confidence must be 0-1)
- Type mismatches: `MAX_HOPS "five"` (expects integer)
- Conflicting constraints: `created > "2024-01-01" AND created < "2023-01-01"`
- Invalid temporal formats: `created < "not-a-date"`

**Edge Cases** (20% of failures):
- Extremely long inputs (10k+ characters)
- Unicode edge cases (emoji, RTL text)
- Deeply nested structures
- Empty inputs
- Whitespace-only inputs

**Performance Issues** (10% of failures):
- Queries that take >100ms to parse
- Exponential backtracking in constraint parsing
- Memory exhaustion from large embeddings

### 2. Property-Based Testing for Parsers

**Key Properties**:

1. **Round-trip**: `parse(unparse(parse(q))) == parse(q)`
2. **Determinism**: `parse(q)` always returns same result
3. **No panics**: Parser never crashes, always returns Ok or Err
4. **Actionable errors**: Every error has suggestion + example
5. **Position accuracy**: Error positions within ±5 chars of actual error

**Generator Strategy**:
```rust
fn valid_query_generator() -> impl Strategy<Value = String> {
    prop_oneof![
        recall_query_gen(),   // 40% RECALL
        spread_query_gen(),   // 30% SPREAD
        predict_query_gen(),  // 15% PREDICT
        imagine_query_gen(),  // 10% IMAGINE
        consolidate_query_gen(), // 5% CONSOLIDATE
    ]
}
```

**Shrinking**: When test fails, reduce query to minimal failing example
```
Failed: RECALL episode WHERE confidence > 2.0 AND created > "2024" AND similar
Shrunk: RECALL episode WHERE confidence > 2.0
```

### 3. Error Message Quality Criteria

**Bad Error**:
```
Parse error at position 45
```

**Better Error**:
```
Parse error at line 2, column 10:
Found: 'RECAL'
Expected: keyword
```

**Best Error**:
```
Parse error at line 2, column 10:
Found: 'RECAL'
Expected: RECALL, SPREAD, PREDICT, IMAGINE, or CONSOLIDATE

Suggestion: Did you mean 'RECALL'?

Example: RECALL episode WHERE confidence > 0.7
```

**Key Components**:
1. Position (line, column, offset)
2. Found token (what parser saw)
3. Expected tokens (valid alternatives)
4. Suggestion (likely fix using Levenshtein distance)
5. Example (correct usage)

### 4. Fuzzing Strategies

**Unstructured Fuzzing**: Arbitrary byte sequences
```rust
fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = Parser::parse(s);  // Should never panic
    }
});
```

**Structured Fuzzing**: Grammar-aware generation
```rust
#[derive(Arbitrary)]
struct FuzzQuery {
    operation: FuzzOperation,
    pattern: String,
    constraints: Vec<FuzzConstraint>,
}

fuzz_target!(|query: FuzzQuery| {
    let query_str = query.to_query_string();
    let _ = Parser::parse(&query_str);
});
```

**Mutation Fuzzing**: Modify valid queries
```rust
fn mutate_query(valid_query: &str) -> String {
    // Insert random character
    // Delete random character
    // Swap adjacent characters
    // Replace keyword with typo
}
```

**Target**: 1M iterations without crash

### 5. Test Corpus Organization

**150+ Test Queries**:
- 75 valid queries (must parse correctly)
- 75 invalid queries (must produce actionable errors)

**Categories**:
1. RECALL operations (20 valid, 20 invalid)
2. SPREAD operations (15 valid, 15 invalid)
3. PREDICT operations (10 valid, 10 invalid)
4. IMAGINE operations (10 valid, 10 invalid)
5. CONSOLIDATE operations (10 valid, 10 invalid)
6. Edge cases (10 valid, 10 invalid)

**Valid Query Examples**:
```
RECALL episode
RECALL episode WHERE confidence > 0.7
RECALL episode WHERE content SIMILAR TO [0.1, 0.2, 0.3]
SPREAD FROM node_123 MAX_HOPS 5 DECAY 0.15
PREDICT episode GIVEN context HORIZON 3600
```

**Invalid Query Examples**:
```
RECAL episode  // Typo
RECALL  // Missing pattern
SPREAD node_123  // Missing FROM
RECALL episode WHERE confidence > 2.0  // Out of range
RECALL episode WHERE created < "invalid-date"  // Bad format
```

### 6. Performance Targets

**Parse Time**:
- Simple query (<50 tokens): <50μs P90
- Complex query (50-200 tokens): <100μs P90
- Large embedding (1536 floats): <200μs P99

**Measurement**:
```rust
fn benchmark_parse(query: &str) {
    let start = Instant::now();
    let _ = Parser::parse(query);
    let elapsed = start.elapsed().as_micros();
    assert!(elapsed < 100, "Parse time exceeded 100μs: {}μs", elapsed);
}
```

**Regression Testing**: CI fails if parse time increases >10%

### 7. Error Recovery Strategies

**Typo Detection** (Levenshtein distance ≤2):
```rust
fn suggest_keyword(typo: &str) -> Option<&'static str> {
    const KEYWORDS: &[&str] = &["RECALL", "SPREAD", "PREDICT", "IMAGINE"];

    KEYWORDS.iter()
        .filter(|kw| levenshtein_distance(typo, kw) <= 2)
        .min_by_key(|kw| levenshtein_distance(typo, kw))
        .copied()
}

// "RECAL" → "RECALL" (distance 1)
// "SPRED" → "SPREAD" (distance 1)
```

**Context-Aware Expected Tokens**:
```rust
// Parser knows current state
match parser.state {
    ParserState::ExpectingOperation => {
        expected = vec!["RECALL", "SPREAD", "PREDICT"]
    }
    ParserState::AfterRecall => {
        expected = vec!["pattern", "WHERE", "CONFIDENCE"]
    }
    ParserState::AfterWhere => {
        expected = vec!["constraint", "AND"]
    }
}
```

**Panic-Free Parsing**:
```rust
// Never unwrap(), always propagate errors
fn parse_constraint(&mut self) -> Result<Constraint, ParseError> {
    let field = self.expect_identifier()?;
    let operator = self.expect_operator()?;
    let value = self.expect_value()?;

    Ok(Constraint { field, operator, value })
}
```

## Synthesis

Parser validation requires:
1. **Comprehensive corpus**: 150+ queries covering all features
2. **Property-based testing**: 7 properties × 1000 cases = 7000 tests
3. **Fuzzing**: 1M iterations of random/structured inputs
4. **Error quality**: 100% of errors have actionable messages
5. **Performance**: <100μs P90, CI regression guards
6. **Recovery**: Typo detection, context-aware suggestions

Goal: Zero parser panics, zero confusing errors, zero performance regressions.

## References

1. "Property-Based Testing", Claessen & Hughes (QuickCheck paper)
2. "Compiler Errors for Humans", Elm language design
3. "Modern Parser Generator", Matklad (rust-analyzer blog)
4. "Fuzzing Book", Zeller et al. (comprehensive fuzzing guide)
5. "Levenshtein Distance for Typo Detection", standard algorithm
