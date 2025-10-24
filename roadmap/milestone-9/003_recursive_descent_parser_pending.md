# Task 003: Recursive Descent Parser

**Status**: Pending
**Duration**: 3 days
**Dependencies**: Task 001 (Parser Infrastructure), Task 002 (AST Definition)
**Owner**: TBD

---

## Objective

Implement hand-written recursive descent parser that converts tokens to AST with <100μs parse time. Use arena allocation to minimize overhead and achieve zero allocations on hot path.

---

## Technical Specification

### 1. Parser Structure

```rust
// File: engram-core/src/query/parser/parser.rs

use super::ast::*;
use super::token::{Token, Spanned, Position};
use super::tokenizer::Tokenizer;
use std::result::Result as StdResult;

pub struct Parser<'a> {
    tokenizer: Tokenizer<'a>,
    current: Option<Spanned<Token<'a>>>,
    previous: Option<Spanned<Token<'a>>>,
}

pub type ParseResult<T> = StdResult<T, ParseError>;

impl<'a> Parser<'a> {
    pub fn new(source: &'a str) -> ParseResult<Self> {
        let mut tokenizer = Tokenizer::new(source);
        let current = tokenizer.next_token().ok();

        Ok(Self {
            tokenizer,
            current,
            previous: None,
        })
    }

    /// Entry point: parse complete query
    pub fn parse(source: &'a str) -> ParseResult<Query> {
        let mut parser = Self::new(source)?;
        parser.parse_query()
    }

    /// Parse top-level query
    fn parse_query(&mut self) -> ParseResult<Query> {
        match self.current_token()? {
            Token::Recall => Ok(Query::Recall(self.parse_recall()?)),
            Token::Predict => Ok(Query::Predict(self.parse_predict()?)),
            Token::Imagine => Ok(Query::Imagine(self.parse_imagine()?)),
            Token::Consolidate => Ok(Query::Consolidate(self.parse_consolidate()?)),
            Token::Spread => Ok(Query::Spread(self.parse_spread()?)),
            token => Err(ParseError::unexpected_token(
                token.clone(),
                vec!["RECALL", "PREDICT", "IMAGINE", "CONSOLIDATE", "SPREAD"],
                self.position(),
            )),
        }
    }

    // Helper methods
    fn current_token(&self) -> ParseResult<&Token<'a>> {
        self.current
            .as_ref()
            .map(|s| &s.value)
            .ok_or_else(|| ParseError::unexpected_eof(self.position()))
    }

    fn advance(&mut self) -> ParseResult<Spanned<Token<'a>>> {
        let current = self.current.take()
            .ok_or_else(|| ParseError::unexpected_eof(self.position()))?;

        self.previous = Some(current.clone());
        self.current = self.tokenizer.next_token().ok();

        Ok(current)
    }

    fn expect(&mut self, expected: Token<'a>) -> ParseResult<Spanned<Token<'a>>> {
        if self.current_token()? == &expected {
            self.advance()
        } else {
            Err(ParseError::unexpected_token(
                self.current_token()?.clone(),
                vec![format!("{:?}", expected)],
                self.position(),
            ))
        }
    }

    fn position(&self) -> Position {
        self.current
            .as_ref()
            .map(|s| s.start)
            .or_else(|| self.previous.as_ref().map(|s| s.end))
            .unwrap_or(Position { offset: 0, line: 1, column: 1 })
    }
}
```

### 2. RECALL Parser

```rust
impl<'a> Parser<'a> {
    /// Parse: RECALL <pattern> [WHERE <constraints>] [CONFIDENCE <threshold>]
    fn parse_recall(&mut self) -> ParseResult<RecallQuery> {
        self.expect(Token::Recall)?;

        let pattern = self.parse_pattern()?;

        let mut constraints = Vec::new();
        if self.check(Token::Where) {
            self.advance()?;
            constraints = self.parse_constraints()?;
        }

        let mut confidence_threshold = None;
        if self.check(Token::Confidence) {
            self.advance()?;
            confidence_threshold = Some(self.parse_confidence_threshold()?);
        }

        let mut base_rate = None;
        if self.check(Token::BaseRate) {
            self.advance()?;
            base_rate = Some(self.parse_confidence_value()?);
        }

        Ok(RecallQuery {
            pattern,
            constraints,
            confidence_threshold,
            base_rate,
            limit: None, // Could add LIMIT clause
        })
    }

    fn check(&self, token: Token<'a>) -> bool {
        self.current_token().ok() == Some(&token)
    }
}
```

### 3. SPREAD Parser

```rust
impl<'a> Parser<'a> {
    /// Parse: SPREAD FROM <node> [MAX_HOPS <n>] [DECAY <rate>] [THRESHOLD <activation>]
    fn parse_spread(&mut self) -> ParseResult<SpreadQuery> {
        self.expect(Token::Spread)?;
        self.expect(Token::From)?;

        let source = self.parse_node_identifier()?;

        let mut max_hops = None;
        let mut decay_rate = None;
        let mut activation_threshold = None;
        let mut refractory_period = None;

        // Parse optional clauses in any order
        while !self.is_at_end() {
            match self.current_token()? {
                Token::MaxHops => {
                    self.advance()?;
                    max_hops = Some(self.parse_integer()? as u16);
                }
                Token::Decay => {
                    self.advance()?;
                    decay_rate = Some(self.parse_float()?);
                }
                Token::Threshold => {
                    self.advance()?;
                    activation_threshold = Some(self.parse_float()?);
                }
                _ => break,
            }
        }

        Ok(SpreadQuery {
            source,
            max_hops,
            decay_rate,
            activation_threshold,
            refractory_period,
        })
    }
}
```

### 4. Pattern Parser

```rust
impl<'a> Parser<'a> {
    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        match self.current_token()? {
            Token::Identifier(name) => {
                // Check if it's a node ID or keyword
                let node_id = NodeIdentifier::new(name.to_string());
                self.advance()?;
                Ok(Pattern::NodeId(node_id))
            }
            Token::LeftBracket => {
                // Parse embedding: [0.1, 0.2, ...]
                let embedding = self.parse_embedding_literal()?;

                // Check for THRESHOLD clause
                let threshold = if self.check(Token::Threshold) {
                    self.advance()?;
                    self.parse_float()?
                } else {
                    0.8 // Default similarity threshold
                };

                Ok(Pattern::Embedding {
                    vector: embedding,
                    threshold,
                })
            }
            Token::StringLiteral(content) => {
                self.advance()?;
                Ok(Pattern::ContentMatch(content.clone()))
            }
            token => Err(ParseError::unexpected_token(
                token.clone(),
                vec!["identifier", "embedding", "string"],
                self.position(),
            )),
        }
    }

    fn parse_embedding_literal(&mut self) -> ParseResult<Vec<f32>> {
        self.expect(Token::LeftBracket)?;

        let mut values = Vec::new();

        while !self.check(Token::RightBracket) {
            values.push(self.parse_float()?);

            if self.check(Token::Comma) {
                self.advance()?;
            } else {
                break;
            }
        }

        self.expect(Token::RightBracket)?;

        if values.is_empty() {
            return Err(ParseError::validation_error(
                "Empty embedding vector",
                self.position(),
            ));
        }

        Ok(values)
    }
}
```

### 5. Constraint Parser

```rust
impl<'a> Parser<'a> {
    fn parse_constraints(&mut self) -> ParseResult<Vec<Constraint>> {
        let mut constraints = Vec::new();

        loop {
            let constraint = self.parse_single_constraint()?;
            constraints.push(constraint);

            // Check for AND (implicit continuation)
            if !self.is_constraint_start() {
                break;
            }
        }

        Ok(constraints)
    }

    fn parse_single_constraint(&mut self) -> ParseResult<Constraint> {
        match self.current_token()? {
            Token::Identifier("content") => {
                self.advance()?;
                // CONTENT CONTAINS "text"
                self.expect_identifier("CONTAINS")?;
                let text = self.parse_string_literal()?;
                Ok(Constraint::ContentContains(text))
            }
            Token::Identifier("created") => {
                self.advance()?;
                if self.check_identifier("BEFORE") {
                    self.advance()?;
                    let time = self.parse_timestamp()?;
                    Ok(Constraint::CreatedBefore(time))
                } else if self.check_identifier("AFTER") {
                    self.advance()?;
                    let time = self.parse_timestamp()?;
                    Ok(Constraint::CreatedAfter(time))
                } else {
                    Err(ParseError::expected_keywords(
                        vec!["BEFORE", "AFTER"],
                        self.position(),
                    ))
                }
            }
            Token::Confidence => {
                self.advance()?;
                let op = self.parse_comparison_operator()?;
                let value = self.parse_confidence_value()?;

                match op {
                    ComparisonOp::GreaterThan => Ok(Constraint::ConfidenceAbove(value)),
                    ComparisonOp::LessThan => Ok(Constraint::ConfidenceBelow(value)),
                    _ => Err(ParseError::validation_error(
                        "Only > and < operators supported for confidence",
                        self.position(),
                    )),
                }
            }
            token => Err(ParseError::unexpected_token(
                token.clone(),
                vec!["content", "created", "confidence"],
                self.position(),
            )),
        }
    }

    fn is_constraint_start(&self) -> bool {
        matches!(
            self.current_token().ok(),
            Some(Token::Identifier("content") | Token::Identifier("created") | Token::Confidence)
        )
    }
}
```

### 6. Literal Parsers

```rust
impl<'a> Parser<'a> {
    fn parse_float(&mut self) -> ParseResult<f32> {
        match self.current_token()? {
            Token::FloatLiteral(value) => {
                let v = *value;
                self.advance()?;
                Ok(v)
            }
            Token::IntegerLiteral(value) => {
                #[allow(clippy::cast_precision_loss)]
                let v = *value as f32;
                self.advance()?;
                Ok(v)
            }
            token => Err(ParseError::unexpected_token(
                token.clone(),
                vec!["number"],
                self.position(),
            )),
        }
    }

    fn parse_integer(&mut self) -> ParseResult<u64> {
        match self.current_token()? {
            Token::IntegerLiteral(value) => {
                let v = *value;
                self.advance()?;
                Ok(v)
            }
            token => Err(ParseError::unexpected_token(
                token.clone(),
                vec!["integer"],
                self.position(),
            )),
        }
    }

    fn parse_confidence_value(&mut self) -> ParseResult<Confidence> {
        let value = self.parse_float()?;
        Ok(Confidence::from_raw(value))
    }

    fn parse_node_identifier(&mut self) -> ParseResult<NodeIdentifier> {
        match self.current_token()? {
            Token::Identifier(name) => {
                let id = NodeIdentifier::new(name.to_string());
                self.advance()?;
                Ok(id)
            }
            token => Err(ParseError::unexpected_token(
                token.clone(),
                vec!["identifier"],
                self.position(),
            )),
        }
    }

    fn parse_string_literal(&mut self) -> ParseResult<String> {
        match self.current_token()? {
            Token::StringLiteral(s) => {
                let value = s.clone();
                self.advance()?;
                Ok(value)
            }
            token => Err(ParseError::unexpected_token(
                token.clone(),
                vec!["string"],
                self.position(),
            )),
        }
    }
}
```

### 7. Error Types

```rust
// File: engram-core/src/query/parser/error.rs

use super::token::{Token, Position};
use super::ast::ValidationError;

#[derive(Debug, Clone)]
pub enum ParseError {
    UnexpectedToken {
        found: String,
        expected: Vec<String>,
        position: Position,
    },
    UnexpectedEof {
        position: Position,
    },
    ValidationError {
        message: String,
        position: Position,
    },
    TokenizeError(String),
}

impl ParseError {
    pub fn unexpected_token(
        found: Token,
        expected: Vec<impl Into<String>>,
        position: Position,
    ) -> Self {
        Self::UnexpectedToken {
            found: format!("{:?}", found),
            expected: expected.into_iter().map(Into::into).collect(),
            position,
        }
    }

    pub fn unexpected_eof(position: Position) -> Self {
        Self::UnexpectedEof { position }
    }

    pub fn validation_error(message: impl Into<String>, position: Position) -> Self {
        Self::ValidationError {
            message: message.into(),
            position,
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::UnexpectedToken { found, expected, position } => {
                write!(
                    f,
                    "Parse error at line {}, column {}:\n  \
                     Found: {}\n  \
                     Expected: {}",
                    position.line,
                    position.column,
                    found,
                    expected.join(" or ")
                )
            }
            Self::UnexpectedEof { position } => {
                write!(
                    f,
                    "Parse error at line {}, column {}: Unexpected end of query",
                    position.line, position.column
                )
            }
            Self::ValidationError { message, position } => {
                write!(
                    f,
                    "Validation error at line {}, column {}: {}",
                    position.line, position.column, message
                )
            }
            Self::TokenizeError(msg) => write!(f, "Tokenize error: {}", msg),
        }
    }
}

impl std::error::Error for ParseError {}
```

---

## Files to Create/Modify

1. **Create**: `engram-core/src/query/parser/parser.rs`
   - Parser implementation

2. **Create**: `engram-core/src/query/parser/error.rs`
   - ParseError type

3. **Modify**: `engram-core/src/query/parser/mod.rs`
   - Export parser: `pub use parser::Parser;`
   - Export error: `pub use error::ParseError;`

---

## Performance Requirements

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Parse RECALL query | <50μs | Criterion |
| Parse SPREAD query | <80μs | Criterion |
| Parse embedding literal (768 dims) | <30μs | Criterion |
| Memory allocation | Zero on hot path | Heap profiling |

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_recall_simple() {
        let query = "RECALL episode";
        let ast = Parser::parse(query).unwrap();

        match ast {
            Query::Recall(recall) => {
                assert_eq!(recall.pattern, Pattern::NodeId("episode".into()));
                assert!(recall.constraints.is_empty());
            }
            _ => panic!("Expected Recall query"),
        }
    }

    #[test]
    fn test_parse_recall_with_constraints() {
        let query = "RECALL episode WHERE confidence > 0.7";
        let ast = Parser::parse(query).unwrap();

        match ast {
            Query::Recall(recall) => {
                assert_eq!(recall.constraints.len(), 1);
                assert!(matches!(
                    recall.constraints[0],
                    Constraint::ConfidenceAbove(_)
                ));
            }
            _ => panic!("Expected Recall query"),
        }
    }

    #[test]
    fn test_parse_spread_query() {
        let query = "SPREAD FROM node_123 MAX_HOPS 5 DECAY 0.15";
        let ast = Parser::parse(query).unwrap();

        match ast {
            Query::Spread(spread) => {
                assert_eq!(spread.source, NodeIdentifier::new("node_123"));
                assert_eq!(spread.max_hops, Some(5));
                assert!((spread.decay_rate.unwrap() - 0.15).abs() < 1e-6);
            }
            _ => panic!("Expected Spread query"),
        }
    }

    #[test]
    fn test_parse_embedding_pattern() {
        let query = "RECALL [0.1, 0.2, 0.3] THRESHOLD 0.8";
        let ast = Parser::parse(query).unwrap();

        match ast {
            Query::Recall(recall) => {
                if let Pattern::Embedding { vector, threshold } = recall.pattern {
                    assert_eq!(vector.len(), 3);
                    assert!((threshold - 0.8).abs() < 1e-6);
                } else {
                    panic!("Expected embedding pattern");
                }
            }
            _ => panic!("Expected Recall query"),
        }
    }

    #[test]
    fn test_parse_error_unexpected_token() {
        let query = "RECALL WHERE";  // Missing pattern
        let result = Parser::parse(query);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ParseError::UnexpectedToken { .. }));
    }

    #[test]
    fn test_parse_error_position() {
        let query = "RECALL episode\nWHERE\n  invalid > 0.7";
        let result = Parser::parse(query);

        if let Err(ParseError::UnexpectedToken { position, .. }) = result {
            assert_eq!(position.line, 3);
        } else {
            panic!("Expected parse error with position");
        }
    }
}
```

### Integration Tests

```rust
// File: engram-core/tests/parser_integration_test.rs

#[test]
fn test_parse_all_example_queries() {
    let queries = vec![
        "RECALL episode WHERE content SIMILAR TO [0.1, 0.3] CONFIDENCE > 0.7",
        "IMAGINE episode BASED ON partial_episode NOVELTY 0.3",
        "SPREAD FROM cue_node MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1",
        "CONSOLIDATE episodes WHERE created < \"2024-10-20\" INTO semantic_memory",
        "PREDICT episode GIVEN context_embedding HORIZON 3600",
    ];

    for query in queries {
        let result = Parser::parse(query);
        assert!(result.is_ok(), "Failed to parse: {}", query);
    }
}
```

---

## Acceptance Criteria

- [ ] All cognitive operations (RECALL, SPREAD, PREDICT, IMAGINE, CONSOLIDATE) parse correctly
- [ ] Parse time <100μs for typical queries
- [ ] Error messages include position information
- [ ] Unit tests achieve >90% coverage
- [ ] Integration tests pass for all example queries
- [ ] Zero clippy warnings
- [ ] No heap allocations on hot path (verified with profiling)

---

## Integration Points

- **Previous Tasks**: Uses Tokenizer (001) and AST types (002)
- **Next Task**: Task 004 enhances error messages with suggestions
- **Query Executor**: Task 005 consumes parsed AST

---

## Notes

- **Backtracking**: Minimal backtracking needed due to LL(1) grammar design
- **Error Recovery**: Continue parsing after errors for better diagnostics (future enhancement)
- **Performance**: Use `#[inline]` on hot path functions
- **Validation**: Validate AST constraints during parsing, not execution

---

## References

- Recursive descent parsing: https://craftinginterpreters.com/parsing-expressions.html
- Rust parser patterns: https://github.com/rust-lang/rust/tree/master/compiler/rustc_parse
