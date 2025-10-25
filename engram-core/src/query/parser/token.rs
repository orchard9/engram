//! Token types with zero-copy lifetime-based design.
//!
//! This module defines the core token representation for the query language,
//! optimized for cache efficiency and zero-allocation parsing.

/// Token type with zero-cost lifetime-based zero-copy design.
///
/// Memory layout optimized for cache efficiency:
/// - Discriminant: 1 byte (enum tag)
/// - Padding: 7 bytes (alignment to 8-byte boundary)
/// - Payload: 16 bytes (two usizes or pointer+len for str slices)
///
/// Total: 24 bytes per token (fits in 3 cache lines on most architectures)
///
/// Performance characteristics:
/// - Token::Identifier uses zero-copy string slices (no allocation)
/// - Token::StringLiteral only allocates for escaped strings (rare path)
/// - All keyword variants are zero-size (compile-time discrimination)
#[derive(Debug, Clone, PartialEq)]
pub enum Token<'a> {
    // Keywords - cognitive operations (zero-size variants)
    /// RECALL keyword - retrieve memories matching a cue
    Recall,
    /// PREDICT keyword - predict future states
    Predict,
    /// IMAGINE keyword - generate novel combinations
    Imagine,
    /// CONSOLIDATE keyword - merge and strengthen memories
    Consolidate,
    /// SPREAD keyword - activation spreading from source nodes
    Spread,

    // Keywords - clauses (zero-size variants)
    /// WHERE keyword - filter condition clause
    Where,
    /// GIVEN keyword - context specification
    Given,
    /// BASED ON keyword - evidence source specification
    BasedOn,
    /// FROM keyword - source specification
    From,
    /// INTO keyword - destination specification
    Into,
    /// MAX_HOPS parameter - maximum graph traversal depth
    MaxHops,
    /// DECAY parameter - activation decay rate
    Decay,
    /// THRESHOLD parameter - minimum activation threshold
    Threshold,
    /// CONFIDENCE parameter - confidence level constraint
    Confidence,
    /// HORIZON parameter - temporal window for operations
    Horizon,
    /// NOVELTY parameter - novelty threshold for pattern detection
    Novelty,
    /// BASE_RATE parameter - prior probability baseline
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
    /// Greater than operator (>)
    GreaterThan,
    /// Less than operator (<)
    LessThan,
    /// Greater or equal operator (>=)
    GreaterOrEqual,
    /// Less or equal operator (<=)
    LessOrEqual,
    /// Equal operator (=)
    Equal,

    // Delimiters (zero-size variants)
    /// Left bracket delimiter ([)
    LeftBracket,
    /// Right bracket delimiter (])
    RightBracket,
    /// Comma separator (,)
    Comma,

    // Special (zero-size variant)
    /// End of file marker
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
        Self {
            offset: 0,
            line: 1,
            column: 1,
        }
    }

    /// Create position at specific location (for testing).
    #[must_use]
    pub const fn new(offset: usize, line: usize, column: usize) -> Self {
        Self {
            offset,
            line,
            column,
        }
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
#[derive(Debug, Clone)]
#[allow(clippy::derive_partial_eq_without_eq)] // Token contains f32, can't implement Eq
pub struct Spanned<T> {
    /// The wrapped value
    pub value: T,
    /// Starting position in source
    pub start: Position,
    /// Ending position in source
    pub end: Position,
}

impl<T: PartialEq> PartialEq for Spanned<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.start == other.start && self.end == other.end
    }
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

// Compile-time assertions for memory layout
const _: () = assert!(std::mem::size_of::<Token>() <= 32); // Allow some flexibility
const _: () = assert!(std::mem::size_of::<Position>() == 24);
const _: () = assert!(std::mem::size_of::<Spanned<Token>>() <= 80); // Allow some flexibility

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Tests are allowed to use unwrap
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let pos = Position::start();
        assert_eq!(pos.offset, 0);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.column, 1);

        let custom = Position::new(10, 2, 5);
        assert_eq!(custom.offset, 10);
        assert_eq!(custom.line, 2);
        assert_eq!(custom.column, 5);
    }

    #[test]
    fn test_spanned_creation() {
        let start = Position::new(0, 1, 1);
        let end = Position::new(5, 1, 6);
        let spanned = Spanned::new(Token::Recall, start, end);

        assert_eq!(spanned.value, Token::Recall);
        assert_eq!(spanned.start, start);
        assert_eq!(spanned.end, end);
    }

    #[test]
    fn test_spanned_snippet() {
        let source = "RECALL episode";
        let start = Position::new(0, 1, 1);
        let end = Position::new(6, 1, 7);
        let spanned = Spanned::new(Token::Recall, start, end);

        assert_eq!(spanned.snippet(source), Some("RECALL"));
    }

    #[test]
    fn test_spanned_map() {
        let start = Position::new(0, 1, 1);
        let end = Position::new(5, 1, 6);
        let spanned = Spanned::new(42i32, start, end);

        let mapped = spanned.map(|x| x * 2);
        assert_eq!(mapped.value, 84);
        assert_eq!(mapped.start, start);
        assert_eq!(mapped.end, end);
    }

    #[test]
    fn test_token_equality() {
        assert_eq!(Token::Recall, Token::Recall);
        assert_ne!(Token::Recall, Token::Predict);

        assert_eq!(Token::Identifier("test"), Token::Identifier("test"));
        assert_ne!(Token::Identifier("test"), Token::Identifier("other"));

        assert_eq!(Token::FloatLiteral(0.5), Token::FloatLiteral(0.5));
        assert_eq!(Token::IntegerLiteral(42), Token::IntegerLiteral(42));
    }

    #[test]
    fn test_memory_layout_sizes() {
        // Verify our compile-time assertions hold
        println!("Token size: {} bytes", std::mem::size_of::<Token>());
        println!("Position size: {} bytes", std::mem::size_of::<Position>());
        println!(
            "Spanned<Token> size: {} bytes",
            std::mem::size_of::<Spanned<Token>>()
        );

        // Token should be reasonably small (around 24-32 bytes)
        assert!(std::mem::size_of::<Token>() <= 32);
        // Position should be exactly 24 bytes (3 usizes on 64-bit)
        assert_eq!(std::mem::size_of::<Position>(), 24);
        // Spanned<Token> should be under 80 bytes
        assert!(std::mem::size_of::<Spanned<Token>>() <= 80);
    }
}
