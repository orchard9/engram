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

- [ ] 100% of parse errors include actionable suggestions
- [ ] Typo detection works for all keywords (Levenshtein distance ≤2)
- [ ] Error messages include line, column, and example
- [ ] Context-aware expected tokens
- [ ] Unit tests for all error scenarios
- [ ] Zero clippy warnings

---

## References

- Levenshtein distance: https://en.wikipedia.org/wiki/Levenshtein_distance
- Error message design: https://elm-lang.org/news/compiler-errors-for-humans
