//! Zero-copy parser infrastructure for cognitive query language.
//!
//! This module implements a production-grade tokenizer with:
//! - Zero-copy string handling via lifetime-based slices
//! - O(1) keyword lookup using compile-time PHF (perfect hash functions)
//! - UTF-8 aware position tracking for precise error messages
//! - Cache-optimal memory layout (64-byte tokenizer, 24-byte tokens)
//!
//! ## Architecture
//!
//! The parser infrastructure follows a three-stage design:
//!
//! 1. **Tokenization** (`Tokenizer`): Converts source text into token stream
//!    - Zero allocations on hot path (identifiers, keywords, numbers)
//!    - PHF-based keyword recognition (no hash collisions, <5ns lookup)
//!    - Single cache line tokenizer state (64 bytes)
//!
//! 2. **Position Tracking** (`Position`, `Spanned`): Records source locations
//!    - Byte offsets for O(1) string slicing
//!    - Line/column numbers for human-readable errors
//!    - UTF-8 aware multi-byte character handling
//!
//! 3. **Error Handling** (`TokenizeError`): Integrates with CognitiveError
//!    - Precise diagnostics with source snippets
//!    - Helpful suggestions for common mistakes
//!    - Consistent with existing Engram error framework
//!
//! ## Performance Characteristics
//!
//! - Tokenize 1000-char query: <10μs (P99)
//! - Tokenize 50-char query: <500ns (P50)
//! - Keyword lookup: O(1), <5ns
//! - Memory overhead: 64 bytes (stack only)
//!
//! ## Example
//!
//! ```rust
//! use engram_core::query::parser::{Tokenizer, Token};
//!
//! let source = "RECALL episode WHERE confidence > 0.7";
//! let mut tokenizer = Tokenizer::new(source);
//!
//! while let Ok(spanned_token) = tokenizer.next_token() {
//!     match spanned_token.value {
//!         Token::Eof => break,
//!         Token::Recall => println!("Found RECALL at {:?}", spanned_token.start),
//!         _ => {}
//!     }
//! }
//! ```

pub mod ast;
pub mod error;
#[allow(clippy::module_inception)] // parser module contains Parser type
pub mod parser;
pub mod token;
pub mod tokenizer;
pub mod typo_detection;

// Re-export public API
pub use ast::*;
pub use error::{ParseError, TokenizeError};
pub use parser::Parser;
pub use token::{Position, Spanned, Token};
pub use tokenizer::Tokenizer;
