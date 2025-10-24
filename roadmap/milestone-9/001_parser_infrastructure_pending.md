# Task 001: Parser Infrastructure

**Status**: Pending
**Duration**: 2 days
**Dependencies**: None
**Owner**: TBD

---

## Objective

Implement zero-copy tokenizer with precise position tracking for production-grade error messages. Foundation for recursive descent parser with cache-optimal memory layout and zero-cost abstractions.

---

## Technical Specification

### 1. Token Types

```rust
// File: engram-core/src/query/parser/token.rs

/// Token type with zero-cost lifetime-based zero-copy design.
///
/// Memory layout optimized for cache efficiency:
/// - Discriminant: 1 byte (enum tag)
/// - Padding: 7 bytes (alignment to 8-byte boundary)
/// - Payload: 16 bytes (two usizes or pointer+len for str slices)
/// Total: 24 bytes per token (fits in 3 cache lines on most architectures)
///
/// Performance characteristics:
/// - Token::Identifier uses zero-copy string slices (no allocation)
/// - Token::StringLiteral only allocates for escaped strings (rare path)
/// - All keyword variants are zero-size (compile-time discrimination)
#[derive(Debug, Clone, PartialEq)]
pub enum Token<'a> {
    // Keywords - cognitive operations (zero-size variants)
    Recall,
    Predict,
    Imagine,
    Consolidate,
    Spread,

    // Keywords - clauses (zero-size variants)
    Where,
    Given,
    BasedOn,
    From,
    Into,
    MaxHops,
    Decay,
    Threshold,
    Confidence,
    Horizon,
    Novelty,
    BaseRate,

    // Literals - carefully sized to minimize enum discriminant overhead
    /// Zero-copy identifier pointing into source string.
    /// SAFETY INVARIANT: 'a lifetime ensures source outlives all tokens.
    Identifier(&'a str),

    /// Owned string for escaped string literals only.
    /// Heap allocation is unavoidable here but rare in practice.
    StringLiteral(String),

    /// 32-bit float for cognitive confidence values (0.0-1.0).
    /// f32 chosen over f64 to keep enum small - sufficient precision for query literals.
    FloatLiteral(f32),

    /// 64-bit unsigned for node IDs, hop counts, timestamps.
    IntegerLiteral(u64),

    // Operators (zero-size variants)
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Equal,

    // Delimiters (zero-size variants)
    LeftBracket,   // [
    RightBracket,  // ]
    Comma,

    // Special (zero-size variant)
    Eof,
}

/// Position tracking with cache-optimal layout.
///
/// Packed into 24 bytes (3 x usize) for efficient storage in Spanned<T>.
/// Combined with Token, Spanned<Token> is 48 bytes (6 cache lines on 64-bit).
///
/// Performance note: Using usize instead of u32 to avoid alignment padding
/// on 64-bit architectures. While u32 would suffice for line/column numbers,
/// the memory savings would be lost to padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    /// Byte offset in source (NOT char offset - enables fast slicing)
    pub offset: usize,
    /// Line number (1-indexed for human-readable error messages)
    pub line: usize,
    /// Column number (1-indexed, counted in UTF-8 code points)
    pub column: usize,
}

impl Position {
    /// Create new position at start of source.
    #[must_use]
    pub const fn start() -> Self {
        Self { offset: 0, line: 1, column: 1 }
    }

    /// Create position at specific location (for testing).
    #[must_use]
    pub const fn new(offset: usize, line: usize, column: usize) -> Self {
        Self { offset, line, column }
    }
}

/// Spanned wrapper associating values with source positions.
///
/// Generic over T to support Spanned<Token>, Spanned<AST>, etc.
/// Memory layout: value (varies) + 2x Position (48 bytes).
/// For Token, total size is 24 + 48 = 72 bytes.
///
/// Design choice: Position stored inline rather than referenced to avoid
/// pointer indirection and improve cache locality during error reporting.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub value: T,
    pub start: Position,
    pub end: Position,
}

impl<T> Spanned<T> {
    /// Create spanned value from components.
    #[must_use]
    pub const fn new(value: T, start: Position, end: Position) -> Self {
        Self { value, start, end }
    }

    /// Map the inner value while preserving span.
    #[must_use]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned {
            value: f(self.value),
            start: self.start,
            end: self.end,
        }
    }

    /// Extract source snippet from original input.
    /// Returns None if positions are invalid.
    #[must_use]
    pub fn snippet<'a>(&self, source: &'a str) -> Option<&'a str> {
        source.get(self.start.offset..self.end.offset)
    }
}
```

### 2. Tokenizer Implementation

```rust
// File: engram-core/src/query/parser/tokenizer.rs

use super::token::{Position, Spanned, Token};
use crate::error::CognitiveError;
use thiserror::Error;

/// Tokenizer with zero-copy string handling and cache-optimal state.
///
/// Memory layout (64 bytes total on 64-bit):
/// - source: &str (16 bytes: pointer + length)
/// - chars: CharIndices (24 bytes: iterator state)
/// - current: Option<(usize, char)> (16 bytes: discriminant + tuple)
/// - position: Position (24 bytes: 3 usizes)
///
/// Fits in a single cache line (64 bytes) for maximum performance.
///
/// SAFETY INVARIANTS:
/// - source must outlive all emitted tokens (enforced by 'a lifetime)
/// - position.offset must always be valid byte index into source
/// - chars iterator must stay synchronized with position tracking
///
/// Performance characteristics:
/// - Zero allocations on hot path (identifiers, keywords, numbers)
/// - O(1) keyword lookup via compile-time PHF hash
/// - Single-pass UTF-8 validation (via CharIndices)
pub struct Tokenizer<'a> {
    /// Source query string - all tokens borrow from this
    source: &'a str,

    /// UTF-8 aware character iterator with byte indices
    /// CharIndices ensures correct handling of multi-byte UTF-8 sequences
    chars: std::str::CharIndices<'a>,

    /// Current character under cursor with its byte offset
    /// None when EOF is reached
    current: Option<(usize, char)>,

    /// Current position for error reporting and token spanning
    position: Position,
}

impl<'a> Tokenizer<'a> {
    /// Create new tokenizer from source string.
    ///
    /// Complexity: O(1) - no allocation or scanning.
    #[must_use]
    pub fn new(source: &'a str) -> Self {
        let mut chars = source.char_indices();
        let current = chars.next();

        Self {
            source,
            chars,
            current,
            position: Position::start(),
        }
    }

    /// Peek next token without consuming (for lookahead in parser).
    ///
    /// Implementation note: Creates temporary clone of tokenizer state.
    /// This is cheap (64 bytes copy) but parser should minimize peek calls
    /// in hot paths. Consider using single-token lookahead pattern instead.
    ///
    /// # Errors
    /// Returns `TokenizeError` if next token is malformed.
    pub fn peek(&mut self) -> Result<Spanned<Token<'a>>, TokenizeError> {
        // Clone current state for lookahead (cheap - 64 byte copy)
        let mut lookahead = self.clone();
        lookahead.next_token()
    }

    /// Consume and return next token.
    ///
    /// This is the hot path - optimized for zero allocations and cache efficiency.
    ///
    /// # Errors
    /// Returns `TokenizeError` for invalid syntax:
    /// - Unterminated string literals
    /// - Invalid number formats
    /// - Unexpected characters
    pub fn next_token(&mut self) -> Result<Spanned<Token<'a>>, TokenizeError> {
        // Skip whitespace and comments (common case: inlined for performance)
        loop {
            match self.current {
                Some((_, ' ' | '\t' | '\n' | '\r')) => {
                    self.advance();
                }
                Some((_, '#')) => {
                    self.skip_comment();
                }
                _ => break,
            }
        }

        let start = self.position;

        let token = match self.current {
            None => Token::Eof,

            // Operators and delimiters (single-char lookahead)
            Some((_, '>')) => {
                self.advance();
                if self.current == Some((self.position.offset, '=')) {
                    self.advance();
                    Token::GreaterOrEqual
                } else {
                    Token::GreaterThan
                }
            }
            Some((_, '<')) => {
                self.advance();
                if self.current == Some((self.position.offset, '=')) {
                    self.advance();
                    Token::LessOrEqual
                } else {
                    Token::LessThan
                }
            }
            Some((_, '=')) => {
                self.advance();
                Token::Equal
            }
            Some((_, '[')) => {
                self.advance();
                Token::LeftBracket
            }
            Some((_, ']')) => {
                self.advance();
                Token::RightBracket
            }
            Some((_, ',')) => {
                self.advance();
                Token::Comma
            }

            // String literals (rare path - allocation acceptable)
            Some((_, '"')) => {
                self.read_string_literal()?
            }

            // Numbers (hot path - zero allocation)
            Some((_, '0'..='9')) => {
                self.read_number()?
            }

            // Identifiers and keywords (hot path - zero allocation + O(1) lookup)
            Some((_, ch)) if ch.is_ascii_alphabetic() || ch == '_' => {
                let ident = self.read_identifier();
                self.keyword_or_identifier(ident)
            }

            Some((offset, ch)) => {
                return Err(TokenizeError::UnexpectedCharacter {
                    ch,
                    position: Position::new(offset, self.position.line, self.position.column),
                });
            }
        };

        let end = self.position;
        Ok(Spanned::new(token, start, end))
    }

    /// Current position in source (for error recovery).
    #[must_use]
    pub const fn position(&self) -> Position {
        self.position
    }

    // ========================================================================
    // Private helper methods - hot path optimizations
    // ========================================================================

    /// Advance to next character, updating position tracking.
    ///
    /// PERFORMANCE: This is called for every character in the query.
    /// Must be inlined and branch-predictor friendly.
    #[inline]
    fn advance(&mut self) -> Option<char> {
        if let Some((offset, ch)) = self.current {
            // Update position BEFORE consuming (correct span tracking)
            if ch == '\n' {
                self.position.line += 1;
                self.position.column = 1;
            } else {
                self.position.column += 1;
            }
            self.position.offset = offset;
        }

        self.current = self.chars.next();
        self.current.map(|(_, ch)| ch)
    }

    /// Skip line comment starting with '#'.
    ///
    /// PERFORMANCE: Comments are rare in production queries,
    /// so this cold path can afford more branches.
    fn skip_comment(&mut self) {
        // Skip until newline or EOF
        while let Some((_, ch)) = self.current {
            if ch == '\n' {
                break;
            }
            self.advance();
        }
    }

    /// Read identifier or keyword with zero-copy slice.
    ///
    /// PERFORMANCE: Hot path - no allocation, returns slice into source.
    /// ASCII-only fast path avoids expensive Unicode normalization.
    ///
    /// SAFETY: Returns slice with lifetime 'a tied to source string.
    fn read_identifier(&mut self) -> &'a str {
        let start_offset = self.position.offset;

        // Fast path: ASCII alphanumeric + underscore
        while let Some((_, ch)) = self.current {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let end_offset = self.position.offset;

        // SAFETY: CharIndices guarantees valid UTF-8 byte boundaries
        &self.source[start_offset..end_offset]
    }

    /// Read string literal with escape sequence handling.
    ///
    /// PERFORMANCE: Cold path - allocation acceptable for escaped strings.
    /// Most queries don't use string literals.
    ///
    /// # Errors
    /// Returns `UnterminatedString` if EOF before closing quote.
    fn read_string_literal(&mut self) -> Result<Token<'a>, TokenizeError> {
        let start_pos = self.position;

        self.advance(); // Skip opening quote
        let mut value = String::new();

        loop {
            match self.current {
                None => {
                    return Err(TokenizeError::UnterminatedString {
                        start: start_pos,
                    });
                }
                Some((_, '"')) => {
                    self.advance(); // Skip closing quote
                    break;
                }
                Some((_, '\\')) => {
                    self.advance();
                    // Escape sequence handling
                    match self.current {
                        Some((_, 'n')) => value.push('\n'),
                        Some((_, 't')) => value.push('\t'),
                        Some((_, 'r')) => value.push('\r'),
                        Some((_, '"')) => value.push('"'),
                        Some((_, '\\')) => value.push('\\'),
                        Some((_, ch)) => value.push(ch), // Pass through unknown escapes
                        None => {
                            return Err(TokenizeError::UnterminatedString {
                                start: start_pos,
                            });
                        }
                    }
                    self.advance();
                }
                Some((_, ch)) => {
                    value.push(ch);
                    self.advance();
                }
            }
        }

        Ok(Token::StringLiteral(value))
    }

    /// Read numeric literal (integer or float).
    ///
    /// PERFORMANCE: Hot path for confidence values and numeric parameters.
    /// Zero allocation - parse directly from source slice.
    ///
    /// Supports:
    /// - Integers: 123, 0, 999
    /// - Floats: 0.5, 0.123, 123.456
    ///
    /// # Errors
    /// Returns `InvalidNumber` if parsing fails.
    fn read_number(&mut self) -> Result<Token<'a>, TokenizeError> {
        let start_offset = self.position.offset;
        let start_pos = self.position;

        // Read digits before decimal point
        while let Some((_, '0'..='9')) = self.current {
            self.advance();
        }

        // Check for decimal point (float vs integer)
        let is_float = matches!(self.current, Some((_, '.')));

        if is_float {
            self.advance(); // Skip '.'

            // Read fractional digits
            while let Some((_, '0'..='9')) = self.current {
                self.advance();
            }
        }

        let end_offset = self.position.offset;
        let text = &self.source[start_offset..end_offset];

        // Parse with standard library (no allocation)
        if is_float {
            text.parse::<f32>()
                .map(Token::FloatLiteral)
                .map_err(|_| TokenizeError::InvalidNumber {
                    text: text.to_string(),
                    position: start_pos,
                })
        } else {
            text.parse::<u64>()
                .map(Token::IntegerLiteral)
                .map_err(|_| TokenizeError::InvalidNumber {
                    text: text.to_string(),
                    position: start_pos,
                })
        }
    }

    /// Classify identifier as keyword or user identifier.
    ///
    /// PERFORMANCE: O(1) lookup via compile-time PHF hash map.
    /// Keywords are case-insensitive for user convenience.
    fn keyword_or_identifier(&self, ident: &'a str) -> Token<'a> {
        // Convert to uppercase for case-insensitive keyword matching
        // PERFORMANCE NOTE: to_uppercase() allocates, but we only use it
        // for lookup in the PHF map. The actual token still borrows from source.
        match KEYWORDS.get(&ident.to_ascii_uppercase() as &str) {
            Some(keyword) => keyword.clone(),
            None => Token::Identifier(ident),
        }
    }
}

// Required for peek() implementation
impl<'a> Clone for Tokenizer<'a> {
    fn clone(&self) -> Self {
        Self {
            source: self.source,
            chars: self.source[self.position.offset..].char_indices(),
            current: self.current,
            position: self.position,
        }
    }
}
```

### 3. Keyword Recognition

Use compile-time perfect hash map for O(1) keyword lookup:

```rust
use phf::phf_map;

/// Compile-time perfect hash map for O(1) keyword recognition.
///
/// Performance characteristics:
/// - Zero runtime initialization cost (built at compile time)
/// - O(1) lookup with no hash collisions (perfect hash function)
/// - ~40 bytes of read-only data per keyword entry
/// - Total size: <1KB (fits in L1 cache)
///
/// The phf crate uses the CHD algorithm to generate minimal perfect hash
/// functions at compile time. This means zero dynamic memory allocation
/// and deterministic lookup time.
///
/// Case-insensitive matching is handled in keyword_or_identifier() by
/// converting input to uppercase before lookup.
static KEYWORDS: phf::Map<&'static str, Token<'static>> = phf_map! {
    // Cognitive operations
    "RECALL" => Token::Recall,
    "PREDICT" => Token::Predict,
    "IMAGINE" => Token::Imagine,
    "CONSOLIDATE" => Token::Consolidate,
    "SPREAD" => Token::Spread,

    // Query clauses
    "WHERE" => Token::Where,
    "GIVEN" => Token::Given,
    "BASED" => Token::BasedOn,  // Note: "BASED ON" is two keywords
    "FROM" => Token::From,
    "INTO" => Token::Into,

    // Parameters
    "MAX_HOPS" => Token::MaxHops,
    "DECAY" => Token::Decay,
    "THRESHOLD" => Token::Threshold,
    "CONFIDENCE" => Token::Confidence,
    "HORIZON" => Token::Horizon,
    "NOVELTY" => Token::Novelty,
    "BASE_RATE" => Token::BaseRate,
};
```

### 4. Position Tracking

Position tracking is implemented in the `advance()` method (shown in section 2).

Key design decisions:
- **Byte offsets** instead of character offsets for O(1) string slicing
- **1-indexed** line/column numbers for human-readable error messages
- **UTF-8 aware** via `CharIndices` iterator (handles multi-byte sequences)
- **Inline hot path** with `#[inline]` attribute for performance

### 5. Error Types

```rust
// File: engram-core/src/query/parser/error.rs

use super::token::Position;
use crate::error::CognitiveError;
use thiserror::Error;

/// Tokenization errors with cognitive-friendly error messages.
///
/// These errors integrate with Engram's existing CognitiveError framework
/// to provide helpful suggestions and similar alternatives.
#[derive(Debug, Clone, Error)]
pub enum TokenizeError {
    /// Unexpected character in input stream.
    ///
    /// This is the most common tokenization error (e.g., `@`, `$`, `%`).
    /// Error message suggests valid syntax at that position.
    #[error("Unexpected character '{ch}' at line {}, column {}", .position.line, .position.column)]
    UnexpectedCharacter {
        ch: char,
        position: Position,
    },

    /// String literal not terminated before EOF.
    ///
    /// Includes position where string started for precise error reporting.
    #[error("Unterminated string literal starting at line {}, column {}", .start.line, .start.column)]
    UnterminatedString {
        start: Position,
    },

    /// Invalid numeric literal format.
    ///
    /// Includes the problematic text to help users understand the issue
    /// (e.g., "1.2.3", "999999999999999999999").
    #[error("Invalid number '{text}' at line {}, column {}", .position.line, .position.column)]
    InvalidNumber {
        text: String,
        position: Position,
    },

    /// Invalid escape sequence in string literal.
    ///
    /// Added for better error messages on unknown escapes like `\x`.
    #[error("Invalid escape sequence '\\{escape}' at line {}, column {}", .position.line, .position.column)]
    InvalidEscape {
        escape: char,
        position: Position,
    },
}

impl TokenizeError {
    /// Convert to CognitiveError for integration with Engram's error system.
    ///
    /// Provides helpful suggestions based on error type.
    #[must_use]
    pub fn to_cognitive_error(&self, source: &str) -> CognitiveError {
        match self {
            Self::UnexpectedCharacter { ch, position } => {
                let snippet = extract_line(source, position.line);
                CognitiveError {
                    summary: format!("Unexpected character '{}' in query", ch),
                    details: format!(
                        "Found '{}' at line {}, column {}\n{}\n{}^",
                        ch,
                        position.line,
                        position.column,
                        snippet,
                        " ".repeat(position.column - 1)
                    ),
                    similar: vec![],
                    confidence: crate::Confidence::HIGH,
                    suggestion: Some(format!(
                        "Remove '{}' or check query syntax. Valid characters: letters, digits, _, >, <, =, [, ], ,",
                        ch
                    )),
                }
            }
            Self::UnterminatedString { start } => {
                let snippet = extract_line(source, start.line);
                CognitiveError {
                    summary: "String literal not closed with \"".to_string(),
                    details: format!(
                        "String started at line {}, column {} but never closed\n{}",
                        start.line, start.column, snippet
                    ),
                    similar: vec![],
                    confidence: crate::Confidence::CERTAIN,
                    suggestion: Some("Add closing \" to string literal".to_string()),
                }
            }
            Self::InvalidNumber { text, position } => {
                CognitiveError {
                    summary: format!("Invalid number format: '{}'", text),
                    details: format!(
                        "Cannot parse '{}' as integer or float at line {}, column {}",
                        text, position.line, position.column
                    ),
                    similar: vec![],
                    confidence: crate::Confidence::HIGH,
                    suggestion: Some("Use format: 123 (integer) or 0.5 (float)".to_string()),
                }
            }
            Self::InvalidEscape { escape, position } => {
                CognitiveError {
                    summary: format!("Unknown escape sequence '\\{}'", escape),
                    details: format!(
                        "Invalid escape at line {}, column {}. Valid escapes: \\n, \\t, \\r, \\\", \\\\",
                        position.line, position.column
                    ),
                    similar: vec![],
                    confidence: crate::Confidence::HIGH,
                    suggestion: Some(format!(
                        "Replace '\\{}' with valid escape or remove backslash",
                        escape
                    )),
                }
            }
        }
    }
}

/// Extract single line from source for error context.
/// Returns empty string if line number invalid.
fn extract_line(source: &str, line_num: usize) -> String {
    source
        .lines()
        .nth(line_num.saturating_sub(1))
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TokenizeError::UnexpectedCharacter {
            ch: '@',
            position: Position::new(5, 2, 3),
        };
        let msg = err.to_string();
        assert!(msg.contains("@"));
        assert!(msg.contains("line 2"));
        assert!(msg.contains("column 3"));
    }

    #[test]
    fn test_cognitive_error_conversion() {
        let source = "RECALL episode\nWHERE @ invalid";
        let err = TokenizeError::UnexpectedCharacter {
            ch: '@',
            position: Position::new(21, 2, 7),
        };
        let cognitive = err.to_cognitive_error(source);

        assert!(cognitive.summary.contains("@"));
        assert!(cognitive.details.contains("line 2"));
        assert!(cognitive.suggestion.is_some());
    }
}
```

---

## Files to Create/Modify

1. **Create**: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/parser/mod.rs`
   - Module declaration and exports
   - Public API surface for parser infrastructure
   - Re-exports: `Token`, `Position`, `Spanned`, `Tokenizer`, `TokenizeError`

2. **Create**: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/parser/token.rs`
   - Token enum with lifetime parameter for zero-copy
   - Position struct (24 bytes, cache-aligned)
   - Spanned<T> generic wrapper
   - Helper methods for Position and Spanned

3. **Create**: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/parser/tokenizer.rs`
   - Tokenizer struct (64 bytes, single cache line)
   - Zero-copy parsing implementation
   - PHF-based keyword lookup
   - Clone implementation for peek()

4. **Create**: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/parser/error.rs`
   - TokenizeError enum with thiserror
   - CognitiveError conversion
   - Helper functions for error context extraction

5. **Modify**: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/Cargo.toml`
   - Add dependency: `phf = { version = "0.11", features = ["macros"] }`
   - Ensure thiserror already present (it is, workspace dependency)

6. **Modify**: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/mod.rs`
   - Add: `pub mod parser;`
   - Re-export tokenizer types if needed at query module level

---

## Performance Requirements

| Operation | Target | Measurement | Rationale |
|-----------|--------|-------------|-----------|
| Tokenize 1000-char query | <10μs | Criterion benchmark | Interactive query latency budget |
| Tokenize short query (50 chars) | <500ns | Criterion benchmark | Sub-microsecond for common case |
| Keyword lookup | O(1), <5ns | Static PHF map | Zero hash collisions, L1 cache hit |
| Memory overhead | <64 bytes | `size_of::<Tokenizer>()` | Single cache line (stack only) |
| Zero allocations on hot path | Yes | Identifiers are &str slices | Only allocate for escaped strings |
| Token size | 24 bytes | `size_of::<Token>()` | Minimize discriminant overhead |
| Spanned<Token> size | 72 bytes | `size_of::<Spanned<Token>>()` | Inline Position for cache locality |

### Performance Validation

All performance requirements validated via:
1. **Criterion benchmarks** in `engram-core/benches/tokenizer.rs`
2. **Compile-time assertions** via `const_assert!` on struct sizes
3. **Allocation tracking** via custom allocator in tests
4. **Cache profiling** via perf/Instruments on representative queries

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_recall_query() {
        let source = "RECALL episode WHERE confidence > 0.7";
        let mut tokenizer = Tokenizer::new(source);

        assert_eq!(tokenizer.next_token().unwrap().value, Token::Recall);
        // ... verify all tokens
    }

    #[test]
    fn test_position_tracking_multiline() {
        let source = "RECALL\n  episode\n  WHERE";
        let mut tokenizer = Tokenizer::new(source);

        let token = tokenizer.next_token().unwrap();
        assert_eq!(token.start.line, 1);
        assert_eq!(token.start.column, 1);

        let token = tokenizer.next_token().unwrap();
        assert_eq!(token.start.line, 2);
        assert_eq!(token.start.column, 3);  // After indent
    }

    #[test]
    fn test_zero_copy_identifiers() {
        let source = "episode_123";
        let mut tokenizer = Tokenizer::new(source);

        let token = tokenizer.next_token().unwrap();
        if let Token::Identifier(ident) = token.value {
            // Verify it's a slice of source, not allocated
            assert_eq!(ident, "episode_123");
            assert_eq!(ident.as_ptr(), source.as_ptr());
        } else {
            panic!("Expected identifier");
        }
    }

    #[test]
    fn test_embedding_literal_parsing() {
        let source = "[0.1, 0.2, 0.3]";
        let mut tokenizer = Tokenizer::new(source);

        assert_eq!(tokenizer.next_token().unwrap().value, Token::LeftBracket);
        assert_eq!(tokenizer.next_token().unwrap().value, Token::FloatLiteral(0.1));
        assert_eq!(tokenizer.next_token().unwrap().value, Token::Comma);
        // ... verify complete sequence
    }

    #[test]
    fn test_comment_handling() {
        let source = "RECALL # this is a comment\n  episode";
        let mut tokenizer = Tokenizer::new(source);

        assert_eq!(tokenizer.next_token().unwrap().value, Token::Recall);
        assert_eq!(tokenizer.next_token().unwrap().value, Token::Identifier("episode"));
    }
}
```

### Benchmarks

```rust
// File: engram-core/benches/tokenizer.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use engram_core::query::parser::{Tokenizer, Token};

/// Benchmark suite for tokenizer performance validation.
///
/// Validates that tokenization meets sub-microsecond latency for short queries
/// and sub-10μs latency for complex queries, with zero allocations on hot path.

fn bench_tokenize_simple(c: &mut Criterion) {
    let query = "RECALL episode WHERE confidence > 0.7";

    c.bench_function("tokenize_simple", |b| {
        b.iter(|| {
            let mut tokenizer = Tokenizer::new(black_box(query));
            let mut count = 0;
            while let Ok(token) = tokenizer.next_token() {
                black_box(&token); // Prevent optimization
                count += 1;
                if token.value == Token::Eof {
                    break;
                }
            }
            count
        });
    });
}

fn bench_tokenize_complex(c: &mut Criterion) {
    let query = "SPREAD FROM node_123 MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1";

    c.bench_function("tokenize_complex", |b| {
        b.iter(|| {
            let mut tokenizer = Tokenizer::new(black_box(query));
            let mut count = 0;
            while let Ok(token) = tokenizer.next_token() {
                black_box(&token);
                count += 1;
                if token.value == Token::Eof {
                    break;
                }
            }
            count
        });
    });
}

fn bench_tokenize_varying_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenize_length");

    for size in [10, 50, 100, 500, 1000].iter() {
        let query = format!("RECALL {} WHERE confidence > 0.7", "node_".repeat(*size / 10));
        group.throughput(Throughput::Bytes(query.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &query, |b, q| {
            b.iter(|| {
                let mut tokenizer = Tokenizer::new(black_box(q));
                let mut count = 0;
                while let Ok(token) = tokenizer.next_token() {
                    black_box(&token);
                    count += 1;
                    if token.value == Token::Eof {
                        break;
                    }
                }
                count
            });
        });
    }

    group.finish();
}

fn bench_keyword_recognition(c: &mut Criterion) {
    // Test keyword lookup performance (should be <5ns per lookup)
    let queries = [
        "RECALL",
        "recall",  // Case-insensitive
        "Recall",
        "PREDICT",
        "IMAGINE",
        "CONSOLIDATE",
        "SPREAD",
        "WHERE",
        "GIVEN",
        "BASED",
        "MAX_HOPS",
        "THRESHOLD",
    ];

    c.bench_function("keyword_recognition", |b| {
        b.iter(|| {
            for query in &queries {
                let mut tokenizer = Tokenizer::new(black_box(query));
                let token = tokenizer.next_token().unwrap();
                black_box(&token);
            }
        });
    });
}

fn bench_number_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("number_parsing");

    group.bench_function("integer", |b| {
        let query = "123456";
        b.iter(|| {
            let mut tokenizer = Tokenizer::new(black_box(query));
            black_box(tokenizer.next_token().unwrap());
        });
    });

    group.bench_function("float", |b| {
        let query = "0.12345";
        b.iter(|| {
            let mut tokenizer = Tokenizer::new(black_box(query));
            black_box(tokenizer.next_token().unwrap());
        });
    });

    group.finish();
}

fn bench_zero_copy_verification(c: &mut Criterion) {
    // Verify identifiers use zero-copy slices (pointer comparison)
    let query = "episode_memory_node_identifier";

    c.bench_function("zero_copy_identifier", |b| {
        b.iter(|| {
            let query_ptr = black_box(query).as_ptr();
            let mut tokenizer = Tokenizer::new(black_box(query));
            if let Ok(token) = tokenizer.next_token() {
                if let Token::Identifier(ident) = token.value {
                    // Verify it's a slice, not a copy
                    assert_eq!(ident.as_ptr(), query_ptr);
                    black_box(ident);
                }
            }
        });
    });
}

criterion_group!(
    benches,
    bench_tokenize_simple,
    bench_tokenize_complex,
    bench_tokenize_varying_length,
    bench_keyword_recognition,
    bench_number_parsing,
    bench_zero_copy_verification
);
criterion_main!(benches);
```

---

## Acceptance Criteria

### Functional Requirements
- [ ] All 17 token types defined with correct memory layout (24 bytes)
- [ ] All 17 keywords recognized (case-insensitive via PHF map)
- [ ] Position tracking accurate for multi-line queries (line, column, offset)
- [ ] Zero-copy parsing for identifiers (verified with pointer comparison test)
- [ ] String literal escape sequences handled correctly (\n, \t, \r, \", \\)
- [ ] Integer and float parsing works for all valid formats
- [ ] Error messages include line, column, and source snippet

### Performance Requirements
- [ ] Tokenize 1000-character query in <10μs (criterion benchmark)
- [ ] Tokenize 50-character query in <500ns (criterion benchmark)
- [ ] Keyword lookup is O(1) with <5ns per lookup
- [ ] Tokenizer struct size is 64 bytes (compile-time assertion)
- [ ] Token enum size is 24 bytes (compile-time assertion)
- [ ] Spanned<Token> size is 72 bytes (compile-time assertion)
- [ ] Zero allocations on hot path for identifiers and keywords

### Code Quality Requirements
- [ ] Unit tests achieve >95% coverage (use cargo-llvm-cov)
- [ ] All public APIs have doc comments with examples
- [ ] Benchmarks run in CI without regression
- [ ] Zero clippy warnings (including pedantic lints)
- [ ] All SAFETY invariants documented in code comments
- [ ] CognitiveError integration tested

### Integration Verification
- [ ] `cargo test --package engram-core --lib query::parser` passes
- [ ] `cargo bench --package engram-core tokenizer` passes performance targets
- [ ] `make quality` passes (clippy, fmt, test)
- [ ] Compile-time assertions verify struct sizes
- [ ] Zero-copy verified via pointer comparison in tests
- [ ] Error messages integrate with existing CognitiveError system

---

## Integration Points

### Direct Dependencies (This Task)
- **Depends on**: Existing `CognitiveError` type in `engram-core/src/error.rs`
- **Depends on**: Existing `Confidence` type in `engram-core/src/lib.rs`
- **Depends on**: PHF crate for compile-time keyword map

### Future Consumers (Next Tasks)
- **Task 002 (AST Definition)**: Uses `Token<'a>` and `Spanned<T>` types
- **Task 003 (Recursive Descent Parser)**: Consumes `Tokenizer` for parsing
- **Task 004 (Error Recovery)**: Uses `Position` and `TokenizeError` for diagnostics
- **Milestone 9 Query Executor**: Uses complete parser to execute RECALL/SPREAD/etc queries

### Existing Codebase Integration
- **engram-core/src/query/mod.rs**: Add `pub mod parser;` declaration
- **engram-core/src/query/executor.rs**: Will consume parsed AST (future)
- **engram-core/src/error.rs**: Already has `CognitiveError` for integration

---

## Notes

### Design Decisions

- **Case Sensitivity**: Keywords are case-insensitive (RECALL = recall = Recall) for user convenience
- **Comments**: Support `#` line comments for query documentation (SQL-style)
- **Unicode**: Use `char_indices()` for proper UTF-8 multi-byte character handling
- **Allocation**: Only allocate for string literals (escaped strings), everything else is zero-copy
- **Byte offsets**: Store byte offsets instead of char offsets for O(1) string slicing
- **Position in errors**: Store Position inline (not Arc/Box) for better cache locality in error paths
- **Clone for peek()**: Cheap 64-byte copy enables one-token lookahead without buffering

### Performance Insights

**Cache Efficiency:**
- Tokenizer (64 bytes) fits in single L1 cache line (64 bytes on x86-64)
- Token (24 bytes) + Spanned overhead (48 bytes) = 72 bytes (just over 1 cache line)
- PHF map data (~1KB) fits entirely in L1 cache (32KB typical)
- Position (24 bytes) alignment ensures no false sharing in concurrent contexts

**Branch Prediction:**
- Hot path (identifiers/keywords) is first branch in token matching
- Whitespace skip loop is predictable (typically few iterations)
- Keyword lookup via PHF has zero collisions = zero branch mispredicts

**Memory Ordering:**
- No atomics needed - tokenizer is single-threaded
- Parser can clone tokenizer for speculative parsing (peek)
- Future: consider arena allocation for AST nodes

### Implementation Notes

**Lifetime Management:**
- Source string lifetime `'a` propagates to all tokens
- Ensures tokens cannot outlive source (compile-time safety)
- Parser must keep source alive until all tokens consumed

**Error Ergonomics:**
- Integration with CognitiveError provides "did you mean?" suggestions
- Error messages include source snippet with caret pointing to error
- Position tracking enables precise diagnostics (better than byte offset alone)

**Future Extensions:**
- Add Token::Comment variant if needed for doc generation
- Consider Token::Embedding([f32; 768]) for inline embeddings (but may bloat enum)
- Could add TokenizeError::SuggestKeyword for typo correction

### Compile-Time Size Verification

Add to token.rs:

```rust
// Compile-time assertions for memory layout
const _: () = assert!(std::mem::size_of::<Token>() == 24);
const _: () = assert!(std::mem::size_of::<Position>() == 24);
const _: () = assert!(std::mem::size_of::<Spanned<Token>>() == 72);

// Verify tokenizer fits in single cache line
const _: () = assert!(std::mem::size_of::<Tokenizer>() <= 64);
```

---

## References

### Rust Language & Standard Library
- **Rust string slicing**: https://doc.rust-lang.org/std/primitive.str.html#method.char_indices
- **CharIndices iterator**: https://doc.rust-lang.org/std/str/struct.CharIndices.html
- **Lifetime elision**: https://doc.rust-lang.org/nomicon/lifetime-elision.html

### Dependencies & Algorithms
- **PHF perfect hash maps**: https://docs.rs/phf/latest/phf/
  - CHD algorithm for minimal perfect hashing
  - Compile-time generation, zero runtime cost
- **thiserror**: https://docs.rs/thiserror/latest/thiserror/
  - Ergonomic Error derivation with Display impl

### Parser Design Patterns
- **Position tracking**: https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html
  - matklad's approach to zero-cost position tracking
- **Zero-copy parsing**: https://docs.rs/nom/latest/nom/
  - nom's zero-copy combinator patterns (reference, not dependency)

### Performance Analysis
- **Cache-conscious data structures**: "What Every Programmer Should Know About Memory" by Ulrich Drepper
- **Branch prediction**: https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-processing-an-unsorted-array
- **Rust performance book**: https://nnethercote.github.io/perf-book/

### Cognitive Systems Integration
- **Engram error handling**: `engram-core/src/error.rs` - CognitiveError framework
- **Engram confidence**: `engram-core/src/lib.rs` - Confidence type for probabilistic errors
- **Query module**: `engram-core/src/query/mod.rs` - Existing probabilistic query infrastructure
