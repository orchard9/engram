# Task 004: Error Recovery and Messages

**Status**: Pending
**Duration**: 2 days
**Dependencies**: Task 003 (Recursive Descent Parser)
**Owner**: TBD

---

## Objective

Implement production-grade error messages with position tracking, typo detection, and actionable suggestions. Every parse error must guide users toward correct syntax.

---

## Technical Specification

### 1. Enhanced Error Type

```rust
// File: engram-core/src/query/parser/error.rs

#[derive(Debug, Clone)]
pub struct ParseError {
    pub kind: ErrorKind,
    pub position: Position,
    pub suggestion: String,
    pub example: String,
}

#[derive(Debug, Clone)]
pub enum ErrorKind {
    UnexpectedToken { found: String, expected: Vec<String> },
    UnknownKeyword { found: String, did_you_mean: Option<String> },
    InvalidSyntax { message: String },
    ValidationError { message: String },
    UnexpectedEof,
}

impl ParseError {
    pub fn with_suggestion(self, suggestion: impl Into<String>) -> Self {
        Self {
            suggestion: suggestion.into(),
            ..self
        }
    }

    pub fn with_example(self, example: impl Into<String>) -> Self {
        Self {
            example: example.into(),
            ..self
        }
    }
}
```

### 2. Typo Detection with Levenshtein Distance

```rust
// File: engram-core/src/query/parser/typo_detection.rs

pub fn find_closest_keyword(input: &str, keywords: &[&str]) -> Option<String> {
    let input_lower = input.to_lowercase();

    let mut closest = None;
    let mut min_distance = usize::MAX;

    for keyword in keywords {
        let distance = levenshtein_distance(&input_lower, &keyword.to_lowercase());

        // Only suggest if distance <= 2 (1-2 typos)
        if distance <= 2 && distance < min_distance {
            min_distance = distance;
            closest = Some((*keyword).to_string());
        }
    }

    closest
}

fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for (i, c1) in s1.chars().enumerate() {
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            matrix[i + 1][j + 1] = std::cmp::min(
                std::cmp::min(
                    matrix[i][j + 1] + 1,     // deletion
                    matrix[i + 1][j] + 1,     // insertion
                ),
                matrix[i][j] + cost,          // substitution
            );
        }
    }

    matrix[len1][len2]
}
```

### 3. Context-Aware Error Messages

```rust
impl<'a> Parser<'a> {
    fn unexpected_token_error(
        &self,
        found: &Token,
        context: ParserContext,
    ) -> ParseError {
        let (expected, suggestion, example) = match context {
            ParserContext::QueryStart => (
                vec!["RECALL", "PREDICT", "IMAGINE", "CONSOLIDATE", "SPREAD"],
                "Query must start with a cognitive operation keyword",
                "RECALL episode WHERE confidence > 0.7",
            ),
            ParserContext::AfterRecall => (
                vec!["pattern", "identifier", "embedding"],
                "RECALL requires a pattern (node ID, embedding, or content match)",
                "RECALL episode_123",
            ),
            ParserContext::InConstraints => (
                vec!["confidence", "content", "created"],
                "WHERE clause requires field name followed by operator and value",
                "WHERE confidence > 0.7",
            ),
            ParserContext::AfterSpread => (
                vec!["FROM"],
                "SPREAD requires FROM keyword followed by node identifier",
                "SPREAD FROM node_123",
            ),
        };

        ParseError {
            kind: ErrorKind::UnexpectedToken {
                found: format!("{:?}", found),
                expected: expected.into_iter().map(String::from).collect(),
            },
            position: self.position(),
            suggestion: suggestion.to_string(),
            example: example.to_string(),
        }
    }
}

enum ParserContext {
    QueryStart,
    AfterRecall,
    InConstraints,
    AfterSpread,
}
```

### 4. Error Message Formatting

```rust
impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "Parse error at line {}, column {}:",
                 self.position.line, self.position.column)?;

        match &self.kind {
            ErrorKind::UnexpectedToken { found, expected } => {
                writeln!(f, "  Found: {}", found)?;
                writeln!(f, "  Expected: {}", expected.join(" or "))?;
            }
            ErrorKind::UnknownKeyword { found, did_you_mean } => {
                writeln!(f, "  Unknown keyword: '{}'", found)?;
                if let Some(suggestion) = did_you_mean {
                    writeln!(f, "  Did you mean: '{}'?", suggestion)?;
                }
            }
            ErrorKind::InvalidSyntax { message } => {
                writeln!(f, "  {}", message)?;
            }
            ErrorKind::ValidationError { message } => {
                writeln!(f, "  Validation error: {}", message)?;
            }
            ErrorKind::UnexpectedEof => {
                writeln!(f, "  Unexpected end of query")?;
            }
        }

        if !self.suggestion.is_empty() {
            writeln!(f, "\nSuggestion: {}", self.suggestion)?;
        }

        if !self.example.is_empty() {
            writeln!(f, "Example: {}", self.example)?;
        }

        Ok(())
    }
}
```

---

## Files to Create/Modify

1. **Modify**: `engram-core/src/query/parser/error.rs`
   - Enhanced ParseError with suggestions and examples

2. **Create**: `engram-core/src/query/parser/typo_detection.rs`
   - Levenshtein distance implementation
   - Keyword suggestion logic

3. **Modify**: `engram-core/src/query/parser/parser.rs`
   - Use enhanced error messages
   - Add context tracking

---

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_typo_detection() {
        let keywords = vec!["RECALL", "PREDICT", "IMAGINE"];

        assert_eq!(
            find_closest_keyword("RECAL", &keywords),
            Some("RECALL".to_string())
        );

        assert_eq!(
            find_closest_keyword("PREDICR", &keywords),
            Some("PREDICT".to_string())
        );

        // No suggestion if distance > 2
        assert_eq!(
            find_closest_keyword("XYZ", &keywords),
            None
        );
    }

    #[test]
    fn test_error_message_quality() {
        let query = "RECAL episode";  // Typo
        let result = Parser::parse(query);

        let err = result.unwrap_err();
        let msg = err.to_string();

        assert!(msg.contains("RECAL"));
        assert!(msg.contains("RECALL"));  // Suggestion
        assert!(msg.contains("Example:"));
    }

    #[test]
    fn test_error_position_accuracy() {
        let query = "RECALL episode\nWHERE\n  invalid > 0.7";
        let result = Parser::parse(query);

        let err = result.unwrap_err();
        assert_eq!(err.position.line, 3);
        assert!(err.position.column > 0);
    }
}
```

---

## Acceptance Criteria

### Error Message Quality (100% Compliance)
- [ ] 100% of parse errors include actionable suggestions
- [ ] Every error has non-empty suggestion field
- [ ] Every error has non-empty example field
- [ ] Error messages pass the "tiredness test" (3am test from Joe Armstrong)

### Typo Detection (Levenshtein Distance ≤2)
- [ ] Typo detection works for all 17 keywords
- [ ] Distance 1: RECAL → RECALL, SPRED → SPREAD
- [ ] Distance 2: IMAGIN → IMAGINE
- [ ] No suggestions when distance >2 (prevents false positives)
- [ ] Levenshtein computation <200ns per keyword (not on hot path)

### Position Tracking
- [ ] Error messages include line number (1-indexed)
- [ ] Error messages include column number (1-indexed, UTF-8 aware)
- [ ] Error messages include byte offset for slicing
- [ ] Position accuracy: within ±5 characters of actual error
- [ ] Multi-line query support with correct line tracking

### Context Awareness
- [ ] Context-aware expected tokens based on parser state
- [ ] Different messages for QueryStart, AfterRecall, InConstraints, AfterSpread states
- [ ] State-specific examples in error messages
- [ ] Grammar-based expectations (not generic "syntax error")

### Testing & Validation
- [ ] Unit tests for all 8 error kinds
- [ ] 40% typo detection coverage (common developer mistakes)
- [ ] 25% wrong order coverage
- [ ] 20% invalid operator coverage
- [ ] 15% missing keyword coverage
- [ ] Error recovery effectiveness: >85% fixed on first try (research target)
- [ ] Time to fix: measure against baseline (research shows 3x faster with good messages)

### Performance & Optimization
- [ ] Error path can be slower than hot path (acceptable)
- [ ] Levenshtein O(mn) optimized to O(min(m,n)) space
- [ ] Early exit if length difference >2
- [ ] Pre-computed keyword list at parser init

### Code Quality
- [ ] Zero clippy warnings
- [ ] All SAFETY invariants documented
- [ ] Psychological research references in comments (Arnsten 2009 - stress & working memory)

---

## Performance Considerations

From research on error path optimization:

**Levenshtein Distance Benchmarks:**
- Computing distance for "RECAL" vs "RECALL": ~150ns
- Checking 10 keywords: ~1.5μs
- Acceptable overhead for error path (not hot path)

**Memory Layout:**
- Error struct should not bloat (avoid allocation in success path)
- Lazy error construction pattern
- Stack-allocated for common error types

**Caching:**
- Pre-compute all valid keywords at parser init
- Store in static array or PHF (perfect hash function)
- Only compute Levenshtein on error path (cold path optimization acceptable)

Reference: Wagner-Fischer algorithm for Levenshtein distance - O(mn) time, O(min(m,n)) space

---

## User Testing Insights

Based on parser error testing research:

**Common Developer Mistakes Distribution:**
1. Typos in keywords: 40% of errors
2. Wrong keyword order: 25% of errors
3. Invalid operators: 20% of errors
4. Missing required keywords: 15% of errors

**Error Recovery Effectiveness:**
- With good messages: 85% of errors fixed on first try
- With bad messages: 40% of errors fixed on first try
- Time to fix: 3x faster with actionable suggestions

These metrics should be tracked during user acceptance testing.

---

## Psychological Design Principles

From "Mind Your Language: On Novices' Interactions with Error Messages" (Marceau et al. 2011):

1. **Clarity**: No parser jargon (avoid "AST", "token stream", "lookahead")
2. **Actionability**: Every error includes concrete next step
3. **Examples**: Show correct syntax (recognition easier than recall)
4. **Positive Framing**: "Use X" better than "Don't use Y"
5. **Stress Awareness**: Under stress, working memory capacity reduces (Arnsten 2009)

**Implementation:**
- Avoid terms: "unexpected token", "syntax error", "parse failure"
- Use instead: "Found X, expected Y", "Use X keyword here", "Add X to complete query"
- Always provide example of correct usage
- Position error message at eye level (first line most important)

---

## References

### Academic Papers
1. Levenshtein, V. I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals". Soviet Physics Doklady.
2. Wagner, R. A.; Fischer, M. J. (1974). "The String-to-String Correction Problem". Journal of the ACM.
3. Arnsten, A. F. (2009). "Stress signalling pathways that impair prefrontal cortex structure and function". Nature Reviews Neuroscience.
4. Marceau, G., Fisler, K., & Krishnamurthi, S. (2011). "Mind Your Language: On Novices' Interactions with Error Messages". ACM SIGPLAN Conference.
5. Becker, L. et al. (2019). "Compiler Error Messages Considered Unhelpful: The Landscape of Text-Based Programming Error Message Research". ACM ICER.

### Industry Best Practices
- Elm Compiler Error Messages: https://elm-lang.org/news/compiler-errors-for-humans
- Rust Error Handling Guide: https://doc.rust-lang.org/book/ch09-00-error-handling.html
- Clang Diagnostics: https://clang.llvm.org/diagnostics.html

### Theoretical Background
- Levenshtein distance: https://en.wikipedia.org/wiki/Levenshtein_distance
- Wagner-Fischer algorithm: https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm

---

## Success Metrics

Track these metrics during implementation and user testing:

1. **Error Message Quality**: 100% have suggestion + example
2. **Typo Detection Rate**: >95% for distance ≤2
3. **Position Accuracy**: 100% within ±5 chars
4. **First-Try Fix Rate**: >85% (vs. 40% baseline)
5. **Time to Fix**: 3x faster than generic errors (empirical measurement)
6. **Passes Tiredness Test**: 100% pass 3am test review

These should be validated in Task 008 (Validation Suite) with comprehensive error message testing.
