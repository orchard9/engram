# Perspectives: Error Messages That Teach

## Cognitive Architecture Perspective

### How Error Messages Map to Human Learning

When a developer encounters a parse error, their cognitive system engages in:

1. **Error Detection**: Recognizing mismatch between intention and outcome
2. **Diagnosis**: Building mental model of what went wrong
3. **Correction**: Generating alternative approach
4. **Learning**: Updating mental model to avoid future errors

**Good error messages accelerate this cycle by**:
- Reducing diagnosis time (show exact location + nature of error)
- Suggesting corrections (activate correct memory patterns)
- Providing examples (episodic memory for future recall)

**Bad error messages force expensive search**:
- Developer must build error model from scratch
- Trial-and-error correction (high cognitive load)
- No learning transfer to similar errors

**Analogy**: Error messages as a "semantic memory aid"
- Like having a knowledgeable colleague point out: "You probably meant X, because Y"
- Externalizes expertise into the system
- Reduces working memory load during debugging

**Key Insight**: Error messages are a form of "teaching" - they're training developers on the language's grammar and semantics. The quality of teaching determines learning efficiency.

---

## Memory Systems Perspective

### Error Messages as Memory Consolidation Triggers

**Episodic Memory Formation**:
- Developer encounters "RECAL episode" typo
- Gets error: "Did you mean 'RECALL'?"
- Forms episodic memory: "That one time I typed RECAL and the parser caught it"

**Semantic Memory Extraction**:
- After 3-4 similar corrections (SPRED → SPREAD, IMAGIN → IMAGINE)
- Developer extracts rule: "Parser suggests corrections for typos"
- Builds confidence in parser's helpfulness
- Reduces fear of experimentation ("parser will catch mistakes")

**Pattern Completion**:
- Error shows "Example: RECALL episode WHERE confidence > 0.7"
- Developer's brain pattern-completes: "Ah, WHERE goes after the pattern"
- Learns grammar through examples, not just error descriptions

**Analogy**: Typo detection as "spell-check for code"
- Just as spell-check teaches vocabulary through corrections
- Parser teaches grammar through actionable suggestions
- Reduces cognitive load of "remembering exact syntax"

**Key Insight**: Error messages should support both immediate correction (episodic) and long-term learning (semantic). Examples serve as "memory anchors" for pattern completion.

---

## Rust Graph Engine Perspective

### Performance Trade-offs in Error Recovery

**Hot Path vs Error Path**:
```rust
// Hot path: Valid query parsing
// Target: <100μs, zero allocations
pub fn parse(query: &str) -> Result<Query, ParseError> {
    let tokens = tokenize(query); // Zero-copy slicing
    parse_query(&tokens) // Arena allocation
}

// Error path: Invalid query + error construction
// Target: <1ms (10x slower acceptable)
pub fn create_error(
    position: Position,
    found: &str,
    keywords: &[&str],
) -> ParseError {
    // Levenshtein distance: ~1-2μs per keyword
    let suggestion = find_closest_keyword(found, keywords);

    // String allocations: acceptable on error path
    ParseError {
        suggestion: format!("Did you mean '{}'?", suggestion),
        example: get_example_for_context(state),
        // ...
    }
}
```

**Design Principle**: Error path can be 10-100x slower than hot path
- Valid queries: <100μs parse time (zero allocations)
- Invalid queries: <1ms error construction (allocations allowed)
- Developers encounter errors 1% of the time (99% hot path)

**Levenshtein Distance Performance**:
```rust
// O(m×n) algorithm, but m,n small for keywords
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    // Typical: s1="RECAL" (5 chars), s2="RECALL" (6 chars)
    // Matrix: 6×7 = 42 cells
    // Time: ~150ns on modern CPU

    // For 10 keywords: ~1.5μs total
    // Acceptable overhead for error path
}
```

**Optimization Strategy**:
1. **Lazy evaluation**: Only compute suggestions on error
2. **Short-circuit**: Skip Levenshtein if length differs by >2
3. **Pre-computed keywords**: Static array, no runtime allocation
4. **Cache-friendly**: Small matrices fit in L1 cache

**Key Insight**: Error recovery doesn't need to be fast, it needs to be helpful. Spend microseconds to save developer minutes.

---

## Systems Architecture Perspective

### Error Message Infrastructure Design

**Three-Layer Architecture**:

**Layer 1: Error Detection (Parser)**
```rust
impl Parser {
    fn expect_keyword(&mut self, keyword: &str) -> Result<(), ParseError> {
        let token = self.current_token();
        if !matches!(token, Token::Keyword(k) if k == keyword) {
            return Err(self.create_error(
                ErrorKind::UnexpectedToken {
                    found: token.to_string(),
                    expected: vec![keyword.to_string()],
                }
            ));
        }
        Ok(())
    }
}
```

**Layer 2: Error Enrichment (Context)**
```rust
impl Parser {
    fn create_error(&self, kind: ErrorKind) -> ParseError {
        let suggestion = self.get_suggestion_for_context();
        let example = self.get_example_for_context();

        ParseError {
            kind,
            position: self.position(),
            suggestion,
            example,
        }
    }

    fn get_suggestion_for_context(&self) -> String {
        match self.state {
            ParserState::QueryStart =>
                "Query must start with a cognitive operation keyword".into(),
            ParserState::AfterRecall =>
                "RECALL requires a pattern (node ID, embedding, or content match)".into(),
            // ... context-specific suggestions
        }
    }
}
```

**Layer 3: Error Presentation (Display)**
```rust
impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Structured output: position, found/expected, suggestion, example
        writeln!(f, "Parse error at line {}, column {}:",
                 self.position.line, self.position.column)?;
        // ... format based on error kind
    }
}
```

**Design Principles**:
1. **Separation of concerns**: Detection ≠ Enrichment ≠ Presentation
2. **Testability**: Each layer independently testable
3. **Extensibility**: Add new error kinds without changing presentation
4. **Context preservation**: Parser state flows through layers

**Error Message Testing Architecture**:
```rust
#[test]
fn test_error_message_quality() {
    let test_cases = vec![
        InvalidQueryTest {
            name: "typo_in_recall",
            query: "RECAL episode",
            expected_error_kind: ErrorKind::UnknownKeyword,
            must_contain: vec!["RECAL", "RECALL"],
            must_suggest: Some("Did you mean: 'RECALL'?"),
        },
        // ... 75+ test cases
    ];

    for test in test_cases {
        let error = Parser::parse(test.query).unwrap_err();

        // Verify structure
        assert!(error.position.line > 0);
        assert!(!error.suggestion.is_empty());
        assert!(!error.example.is_empty());

        // Verify content
        for required in test.must_contain {
            assert!(error.to_string().contains(required));
        }
    }
}
```

**Key Insight**: Error messages are not an afterthought - they're a first-class system component with dedicated architecture, testing, and performance targets.

---

## Synthesis: Cross-Cutting Insights

### 1. Error Messages as API

Error messages are an API between the parser and developers:
- **Contract**: Parser promises actionable guidance on every error
- **Documentation**: Examples teach grammar by showing correct usage
- **Feedback loop**: Suggestions guide toward valid syntax
- **Quality metric**: % of errors fixed on first try

### 2. The "Tiredness Test" as Design Principle

Would a developer at 3am understand what to do?
- **Cognitive load**: Minimize mental effort to diagnose and fix
- **Clarity**: No parser jargon, no implementation details
- **Actionability**: Clear next step, not just description
- **Examples**: Show, don't just tell

### 3. Levenshtein Distance as "Fuzzy Matching"

Typo detection is pattern matching with tolerance:
- **Distance ≤1**: Single character mistake (high confidence suggestion)
- **Distance ≤2**: Two mistakes or transposition (moderate confidence)
- **Distance >2**: No suggestion (would be unreliable guess)
- **Analogy**: Like autocorrect in messaging - catches common typos without false positives

### 4. Context as Compressed Domain Knowledge

Parser state encodes grammar rules:
- **State = "AfterRecall"** → Expected = pattern (node ID, embedding, content)
- **State = "InConstraints"** → Expected = field name, operator, value
- **State = "AfterSpread"** → Expected = FROM keyword

Context transforms generic errors into specific guidance.

### 5. Examples as Memory Anchors

Showing correct syntax serves multiple purposes:
- **Immediate**: Copy-paste template for fixing current error
- **Learning**: Episodic memory for future similar situations
- **Pattern completion**: Brain extrapolates grammar rules from examples
- **Confidence**: "This is what good looks like"

---

## Recommendations for Implementation

### High Priority (Core Quality)
1. **Typo detection**: Levenshtein distance ≤2 for all keywords
2. **Position tracking**: Accurate line, column for every error
3. **Context-aware suggestions**: Use parser state for specific guidance
4. **Example generation**: Every error shows correct syntax

### Medium Priority (Enhanced UX)
1. **Color output**: Highlight errors in terminal (if TTY detected)
2. **Multi-line context**: Show query with error highlighted
3. **Error codes**: Unique IDs for documentation lookup (e.g., E001-E099)
4. **Suggestion ranking**: Multiple possibilities ranked by likelihood

### Low Priority (Nice-to-Have)
1. **Fix-it hints**: Automated corrections (like Clang)
2. **IDE integration**: Language server protocol for inline errors
3. **Error recovery**: Continue parsing after first error (cascading detection)
4. **Internationalization**: Error messages in multiple languages

---

## Validation Metrics

How do we know error messages are good?

**Quantitative**:
- 100% of errors have non-empty suggestion and example
- Typo detection covers all keywords (100% coverage)
- Position tracking accurate within ±1 character
- Error construction time <1ms (10x slower than parse OK)

**Qualitative**:
- Errors pass "tiredness test" (understandable when stressed)
- Examples show realistic usage (not toy queries)
- Suggestions are actionable (specific steps, not vague guidance)
- No parser jargon (no "unexpected token", "lookahead", "AST")

**User-Facing**:
- >80% of errors fixed on first attempt
- <5% of support tickets about "unclear error messages"
- Developer sentiment: positive feedback on error quality

---

## Conclusion

Error messages are not a secondary concern - they're a primary interface between the parser and its users. Good error messages:

1. **Reduce cognitive load** (tell me exactly what to do)
2. **Accelerate learning** (teach grammar through examples)
3. **Build confidence** (parser catches mistakes, reduces fear of experimentation)
4. **Save time** (fix on first try vs trial-and-error)

The investment in error message quality pays dividends in developer productivity, satisfaction, and system adoption. Engram's query parser should set a new standard for error message excellence in cognitive architecture systems.
