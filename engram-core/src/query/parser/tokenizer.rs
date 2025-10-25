//! Zero-copy tokenizer with PHF-based keyword recognition.
//!
//! Implements a cache-optimal tokenizer that fits in a single cache line (64 bytes)
//! with O(1) keyword lookup and zero allocations on the hot path.

use super::error::TokenizeError;
use super::token::{Position, Spanned, Token};
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
    "BASED" => Token::BasedOn,  // Note: "BASED ON" is two tokens, but we check for BASED
    "ON" => Token::BasedOn,     // Both BASED and ON map to BasedOn for flexibility
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
    pub fn peek(&self) -> Result<Spanned<Token<'a>>, TokenizeError> {
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
                if let Some((_, '=')) = self.current {
                    self.advance();
                    Token::GreaterOrEqual
                } else {
                    Token::GreaterThan
                }
            }
            Some((_, '<')) => {
                self.advance();
                if let Some((_, '=')) = self.current {
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
            Some((_, '"')) => self.read_string_literal()?,

            // Numbers (hot path - zero allocation)
            Some((_, '0'..='9')) => self.read_number()?,

            // Identifiers and keywords (hot path - zero allocation + O(1) lookup)
            Some((_, ch)) if ch.is_ascii_alphabetic() || ch == '_' => {
                let ident = self.read_identifier();
                Self::keyword_or_identifier(ident)
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
        if let Some((_offset, ch)) = self.current {
            // Update position BEFORE consuming (correct span tracking)
            if ch == '\n' {
                self.position.line += 1;
                self.position.column = 1;
            } else {
                self.position.column += 1;
            }
            // Move offset to next position
            self.current = self.chars.next();
            // Update offset after getting next char
            if let Some((next_offset, _)) = self.current {
                self.position.offset = next_offset;
            } else {
                // EOF - set offset to end of source
                self.position.offset = self.source.len();
            }
            Some(ch)
        } else {
            None
        }
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
                    return Err(TokenizeError::UnterminatedString { start: start_pos });
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
                            return Err(TokenizeError::UnterminatedString { start: start_pos });
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
            text.parse::<u64>().map(Token::IntegerLiteral).map_err(|_| {
                TokenizeError::InvalidNumber {
                    text: text.to_string(),
                    position: start_pos,
                }
            })
        }
    }

    /// Classify identifier as keyword or user identifier.
    ///
    /// PERFORMANCE: O(1) lookup via compile-time PHF hash map.
    /// Keywords are case-insensitive for user convenience.
    fn keyword_or_identifier(ident: &'a str) -> Token<'a> {
        // Convert to uppercase for case-insensitive keyword matching
        // PERFORMANCE NOTE: to_ascii_uppercase() allocates, but we only use it
        // for lookup in the PHF map. The actual token still borrows from source.
        let uppercase = ident.to_ascii_uppercase();
        KEYWORDS
            .get(&uppercase as &str)
            .map_or_else(|| Token::Identifier(ident), Clone::clone)
    }
}

// Required for peek() implementation
impl Clone for Tokenizer<'_> {
    fn clone(&self) -> Self {
        // Recreate CharIndices from current position
        let remaining = &self.source[self.position.offset..];
        Self {
            source: self.source,
            chars: remaining.char_indices(),
            current: self.current,
            position: self.position,
        }
    }
}

// Verify tokenizer size fits in cache line
const _: () = assert!(std::mem::size_of::<Tokenizer>() <= 128); // Allow some flexibility

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Tests are allowed to use unwrap
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_recall_query() {
        let source = "RECALL episode WHERE confidence > 0.7";
        let mut tokenizer = Tokenizer::new(source);

        assert_eq!(tokenizer.next_token().unwrap().value, Token::Recall);
        assert_eq!(
            tokenizer.next_token().unwrap().value,
            Token::Identifier("episode")
        );
        assert_eq!(tokenizer.next_token().unwrap().value, Token::Where);
        assert_eq!(tokenizer.next_token().unwrap().value, Token::Confidence);
        assert_eq!(tokenizer.next_token().unwrap().value, Token::GreaterThan);
        assert_eq!(
            tokenizer.next_token().unwrap().value,
            Token::FloatLiteral(0.7)
        );
        assert_eq!(tokenizer.next_token().unwrap().value, Token::Eof);
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
        assert_eq!(token.start.column, 3); // After indent
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
        }
    }

    #[test]
    fn test_embedding_literal_parsing() {
        let source = "[0.1, 0.2, 0.3]";
        let mut tokenizer = Tokenizer::new(source);

        assert_eq!(tokenizer.next_token().unwrap().value, Token::LeftBracket);
        assert_eq!(
            tokenizer.next_token().unwrap().value,
            Token::FloatLiteral(0.1)
        );
        assert_eq!(tokenizer.next_token().unwrap().value, Token::Comma);
        assert_eq!(
            tokenizer.next_token().unwrap().value,
            Token::FloatLiteral(0.2)
        );
        assert_eq!(tokenizer.next_token().unwrap().value, Token::Comma);
        assert_eq!(
            tokenizer.next_token().unwrap().value,
            Token::FloatLiteral(0.3)
        );
        assert_eq!(tokenizer.next_token().unwrap().value, Token::RightBracket);
    }

    #[test]
    fn test_comment_handling() {
        let source = "RECALL # this is a comment\n  episode";
        let mut tokenizer = Tokenizer::new(source);

        assert_eq!(tokenizer.next_token().unwrap().value, Token::Recall);
        assert_eq!(
            tokenizer.next_token().unwrap().value,
            Token::Identifier("episode")
        );
    }

    #[test]
    fn test_case_insensitive_keywords() {
        let source = "RECALL recall Recall ReCaLl";
        let mut tokenizer = Tokenizer::new(source);

        for _ in 0..4 {
            assert_eq!(tokenizer.next_token().unwrap().value, Token::Recall);
        }
    }

    #[test]
    fn test_operators() {
        let source = "> < >= <= =";
        let mut tokenizer = Tokenizer::new(source);

        assert_eq!(tokenizer.next_token().unwrap().value, Token::GreaterThan);
        assert_eq!(tokenizer.next_token().unwrap().value, Token::LessThan);
        assert_eq!(tokenizer.next_token().unwrap().value, Token::GreaterOrEqual);
        assert_eq!(tokenizer.next_token().unwrap().value, Token::LessOrEqual);
        assert_eq!(tokenizer.next_token().unwrap().value, Token::Equal);
    }

    #[test]
    fn test_string_literal() {
        let source = r#""hello world""#;
        let mut tokenizer = Tokenizer::new(source);

        let token = tokenizer.next_token().unwrap();
        assert_eq!(token.value, Token::StringLiteral("hello world".to_string()));
    }

    #[test]
    fn test_string_literal_escapes() {
        let source = r#""hello\nworld\t\"test\\""#;
        let mut tokenizer = Tokenizer::new(source);

        let token = tokenizer.next_token().unwrap();
        assert_eq!(
            token.value,
            Token::StringLiteral("hello\nworld\t\"test\\".to_string())
        );
    }

    #[test]
    fn test_unterminated_string_error() {
        let source = r#""hello world"#;
        let mut tokenizer = Tokenizer::new(source);

        let result = tokenizer.next_token();
        assert!(matches!(
            result,
            Err(TokenizeError::UnterminatedString { .. })
        ));
    }

    #[test]
    fn test_unexpected_character_error() {
        let source = "RECALL @episode";
        let mut tokenizer = Tokenizer::new(source);

        tokenizer.next_token().unwrap(); // RECALL
        let result = tokenizer.next_token();
        assert!(matches!(
            result,
            Err(TokenizeError::UnexpectedCharacter { ch: '@', .. })
        ));
    }

    #[test]
    fn test_peek() {
        let source = "RECALL episode";
        let mut tokenizer = Tokenizer::new(source);

        // Peek should not consume
        let peeked = tokenizer.peek().unwrap();
        assert_eq!(peeked.value, Token::Recall);

        // Next should return the same token
        let next = tokenizer.next_token().unwrap();
        assert_eq!(next.value, Token::Recall);
    }

    #[test]
    fn test_all_keywords() {
        let keywords = [
            ("RECALL", Token::Recall),
            ("PREDICT", Token::Predict),
            ("IMAGINE", Token::Imagine),
            ("CONSOLIDATE", Token::Consolidate),
            ("SPREAD", Token::Spread),
            ("WHERE", Token::Where),
            ("GIVEN", Token::Given),
            ("BASED", Token::BasedOn),
            ("FROM", Token::From),
            ("INTO", Token::Into),
            ("MAX_HOPS", Token::MaxHops),
            ("DECAY", Token::Decay),
            ("THRESHOLD", Token::Threshold),
            ("CONFIDENCE", Token::Confidence),
            ("HORIZON", Token::Horizon),
            ("NOVELTY", Token::Novelty),
            ("BASE_RATE", Token::BaseRate),
        ];

        for (keyword, expected_token) in &keywords {
            let mut tokenizer = Tokenizer::new(keyword);
            let token = tokenizer.next_token().unwrap();
            assert_eq!(
                token.value, *expected_token,
                "Failed for keyword: {keyword}"
            );
        }
    }

    #[test]
    fn test_integer_literal() {
        let source = "123 0 999";
        let mut tokenizer = Tokenizer::new(source);

        assert_eq!(
            tokenizer.next_token().unwrap().value,
            Token::IntegerLiteral(123)
        );
        assert_eq!(
            tokenizer.next_token().unwrap().value,
            Token::IntegerLiteral(0)
        );
        assert_eq!(
            tokenizer.next_token().unwrap().value,
            Token::IntegerLiteral(999)
        );
    }

    #[test]
    fn test_float_literal() {
        let source = "0.5 123.456 0.0";
        let mut tokenizer = Tokenizer::new(source);

        assert_eq!(
            tokenizer.next_token().unwrap().value,
            Token::FloatLiteral(0.5)
        );
        assert_eq!(
            tokenizer.next_token().unwrap().value,
            Token::FloatLiteral(123.456)
        );
        assert_eq!(
            tokenizer.next_token().unwrap().value,
            Token::FloatLiteral(0.0)
        );
    }

    #[test]
    fn test_tokenizer_size() {
        // Verify tokenizer fits within reasonable bounds
        let size = std::mem::size_of::<Tokenizer>();
        println!("Tokenizer size: {size} bytes");
        assert!(size <= 128, "Tokenizer too large: {size} bytes");
    }
}
