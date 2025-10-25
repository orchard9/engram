//! Tokenization and parsing errors with cognitive-friendly error messages.
//!
//! This module implements production-grade error messages following psychological
//! research principles (Marceau et al. 2011, Becker et al. 2019):
//!
//! 1. **Clarity**: No parser jargon (avoid "AST", "token stream", "lookahead")
//! 2. **Actionability**: Every error includes concrete next step
//! 3. **Examples**: Show correct syntax (recognition easier than recall)
//! 4. **Positive Framing**: "Use X" better than "Don't use Y"
//! 5. **Stress Awareness**: Under stress, working memory capacity reduces (Arnsten 2009)
//!
//! ## Error Message Quality Standards
//!
//! - 100% of errors have actionable suggestions
//! - 100% of errors have examples
//! - Typo detection works for all 17 keywords
//! - Error messages pass the "tiredness test" (clear at 3am)
//!
//! ## References
//!
//! - Marceau, G., Fisler, K., & Krishnamurthi, S. (2011). "Mind Your Language:
//!   On Novices' Interactions with Error Messages". ACM SIGPLAN Conference.
//! - Arnsten, A. F. (2009). "Stress signalling pathways that impair prefrontal
//!   cortex structure and function". Nature Reviews Neuroscience.
//! - Becker, L. et al. (2019). "Compiler Error Messages Considered Unhelpful".
//!   ACM ICER.

use super::token::{Position, Token};
use super::typo_detection::suggest_keyword_for_token;
use crate::Confidence;
use crate::error::{CognitiveError, ErrorContext};
use std::fmt;
use thiserror::Error;

// ============================================================================
// Tokenization Errors
// ============================================================================

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
        /// The unexpected character found
        ch: char,
        /// Position where the character was found
        position: Position,
    },

    /// String literal not terminated before EOF.
    ///
    /// Includes position where string started for precise error reporting.
    #[error("Unterminated string literal starting at line {}, column {}", .start.line, .start.column)]
    UnterminatedString {
        /// Position where the unterminated string started
        start: Position,
    },

    /// Invalid numeric literal format.
    ///
    /// Includes the problematic text to help users understand the issue
    /// (e.g., "1.2.3", "999999999999999999999").
    #[error("Invalid number '{text}' at line {}, column {}", .position.line, .position.column)]
    InvalidNumber {
        /// The text that failed to parse as a number
        text: String,
        /// Position where the invalid number was found
        position: Position,
    },

    /// Invalid escape sequence in string literal.
    ///
    /// Added for better error messages on unknown escapes like `\x`.
    #[error("Invalid escape sequence '\\{escape}' at line {}, column {}", .position.line, .position.column)]
    InvalidEscape {
        /// The invalid escape character
        escape: char,
        /// Position where the invalid escape was found
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
                CognitiveError::new(
                    format!("Unexpected character '{ch}' in query"),
                    ErrorContext::new(
                        "Valid query syntax character",
                        format!("Found '{ch}' at line {}, column {}", position.line, position.column),
                    ),
                    format!(
                        "Remove '{ch}' or check query syntax. Valid characters: letters, digits, _, >, <, =, [, ], ,"
                    ),
                    "RECALL episode WHERE confidence > 0.7",
                    Confidence::HIGH,
                )
                .with_details(format!(
                    "Found '{ch}' at line {}, column {}\n{snippet}\n{}^",
                    position.line,
                    position.column,
                    " ".repeat(position.column.saturating_sub(1))
                ))
            }
            Self::UnterminatedString { start } => {
                let snippet = extract_line(source, start.line);
                CognitiveError::new(
                    "String literal not closed with \"",
                    ErrorContext::new(
                        "Closing quote for string literal",
                        format!(
                            "String started at line {}, column {} but never closed",
                            start.line, start.column
                        ),
                    ),
                    "Add closing \" to string literal",
                    r#"RECALL "my episode""#,
                    Confidence::CERTAIN,
                )
                .with_details(format!(
                    "String started at line {}, column {} but never closed\n{snippet}",
                    start.line, start.column
                ))
            }
            Self::InvalidNumber { text, position } => CognitiveError::new(
                format!("Invalid number format: '{text}'"),
                ErrorContext::new(
                    "Valid integer (123) or float (0.5)",
                    format!("Found '{text}' which cannot be parsed"),
                ),
                "Use format: 123 (integer) or 0.5 (float)",
                "WHERE confidence > 0.7",
                Confidence::HIGH,
            )
            .with_details(format!(
                "Cannot parse '{text}' as integer or float at line {}, column {}",
                position.line, position.column
            )),
            Self::InvalidEscape { escape, position } => CognitiveError::new(
                format!("Unknown escape sequence '\\{escape}'"),
                ErrorContext::new(
                    "Valid escape sequence (\\n, \\t, \\r, \\\", \\\\)",
                    format!("Found '\\{escape}' which is not recognized"),
                ),
                format!("Replace '\\{escape}' with valid escape or remove backslash"),
                r#"RECALL "line one\nline two""#,
                Confidence::HIGH,
            )
            .with_details(format!(
                "Invalid escape at line {}, column {}. Valid escapes: \\n, \\t, \\r, \\\", \\\\",
                position.line, position.column
            )),
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

// ============================================================================
// Parse Errors - Enhanced with Suggestions and Examples
// ============================================================================

/// Parse errors with position information, helpful suggestions, and examples.
///
/// ParseError represents failures during the parsing stage (converting tokens
/// to AST), as opposed to TokenizeError which represents failures during
/// tokenization (converting characters to tokens).
///
/// ## Error Message Quality Standards
///
/// Every ParseError MUST have:
/// - Precise position information (line, column, byte offset)
/// - Non-empty suggestion field (actionable next step)
/// - Non-empty example field (correct syntax demonstration)
/// - Clear, jargon-free description (no "AST", "token stream", etc.)
///
/// ## Psychological Design Principles
///
/// Following research on error message effectiveness (Marceau et al. 2011):
/// - Positive framing: "Use X" not "Don't use Y"
/// - Recognition over recall: Always show examples
/// - Stress-aware: Clear messages for tired developers (3am test)
/// - No jargon: Avoid parser terminology that users don't understand
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Type of parse error
    pub kind: ErrorKind,
    /// Position where the error occurred
    pub position: Position,
    /// Actionable suggestion for fixing the error (REQUIRED - never empty)
    pub suggestion: String,
    /// Example of correct syntax (REQUIRED - never empty)
    pub example: String,
}

/// Types of parse errors.
#[derive(Debug, Clone)]
pub enum ErrorKind {
    /// Unexpected token found during parsing.
    ///
    /// This is the most common parse error - the parser encountered a token
    /// that doesn't fit the grammar at this position.
    UnexpectedToken {
        /// Description of the token that was found
        found: String,
        /// List of expected token descriptions
        expected: Vec<String>,
    },

    /// Unknown keyword with typo suggestion.
    ///
    /// Uses Levenshtein distance to suggest similar keywords (distance ≤ 2).
    UnknownKeyword {
        /// The unrecognized keyword
        found: String,
        /// Suggested correction (if Levenshtein distance ≤ 2)
        did_you_mean: Option<String>,
    },

    /// Invalid syntax with specific message.
    ///
    /// Used for grammar violations that don't fit other categories.
    InvalidSyntax {
        /// Specific description of the syntax error
        message: String,
    },

    /// Semantic validation error during parsing.
    ///
    /// The query is syntactically valid but semantically invalid
    /// (e.g., empty embedding vector, invalid threshold range).
    ValidationError {
        /// Description of the validation failure
        message: String,
    },

    /// Unexpected end of query (missing tokens).
    ///
    /// Parser expected more tokens but reached EOF.
    UnexpectedEof,
}

impl ParseError {
    /// Create unexpected token error with suggestion and example.
    ///
    /// This is the primary constructor for ParseError. It automatically:
    /// - Detects typos using Levenshtein distance
    /// - Generates context-aware suggestions
    /// - Provides examples based on parser state
    ///
    /// # Panics
    /// Never panics - always returns a valid ParseError with suggestion and example.
    #[must_use]
    pub fn unexpected_token(
        found: &Token,
        expected: Vec<impl Into<String>>,
        position: Position,
        context: ParserContext,
    ) -> Self {
        let expected_strings: Vec<String> = expected.into_iter().map(Into::into).collect();

        // Check if this is a typo of a known keyword
        if let Some(suggestion) = suggest_keyword_for_token(found) {
            return Self::unknown_keyword(found, Some(suggestion), position, context);
        }

        let (suggestion, example) = generate_suggestion_and_example(context, &expected_strings);

        Self {
            kind: ErrorKind::UnexpectedToken {
                found: format!("{found:?}"),
                expected: expected_strings,
            },
            position,
            suggestion,
            example,
        }
    }

    /// Create unknown keyword error with typo suggestion.
    #[must_use]
    pub fn unknown_keyword(
        found: &Token,
        did_you_mean: Option<String>,
        position: Position,
        context: ParserContext,
    ) -> Self {
        let found_str = match found {
            Token::Identifier(name) => (*name).to_string(),
            _ => format!("{found:?}"),
        };

        let (suggestion, example) = did_you_mean.as_ref().map_or_else(
            || generate_suggestion_and_example(context, &[]),
            |correction| {
                (
                    format!("Use keyword {correction}"),
                    generate_example_for_keyword(correction, context),
                )
            },
        );

        Self {
            kind: ErrorKind::UnknownKeyword {
                found: found_str,
                did_you_mean,
            },
            position,
            suggestion,
            example,
        }
    }

    /// Create unexpected EOF error.
    #[must_use]
    pub fn unexpected_eof(position: Position, context: ParserContext) -> Self {
        let (suggestion, example) = generate_suggestion_and_example(context, &[]);

        Self {
            kind: ErrorKind::UnexpectedEof,
            position,
            suggestion,
            example,
        }
    }

    /// Create validation error with custom message.
    #[must_use]
    pub fn validation_error(
        message: impl Into<String>,
        position: Position,
        suggestion: impl Into<String>,
        example: impl Into<String>,
    ) -> Self {
        Self {
            kind: ErrorKind::ValidationError {
                message: message.into(),
            },
            position,
            suggestion: suggestion.into(),
            example: example.into(),
        }
    }

    /// Create invalid syntax error with custom message.
    #[must_use]
    pub fn invalid_syntax(
        message: impl Into<String>,
        position: Position,
        suggestion: impl Into<String>,
        example: impl Into<String>,
    ) -> Self {
        Self {
            kind: ErrorKind::InvalidSyntax {
                message: message.into(),
            },
            position,
            suggestion: suggestion.into(),
            example: example.into(),
        }
    }

    /// Add or override suggestion.
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = suggestion.into();
        self
    }

    /// Add or override example.
    #[must_use]
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.example = example.into();
        self
    }

    /// Convert to CognitiveError for integration with Engram's error system.
    #[must_use]
    pub fn to_cognitive_error(&self, source: &str) -> CognitiveError {
        let snippet = extract_line(source, self.position.line);
        let pointer = format!("{}^", " ".repeat(self.position.column.saturating_sub(1)));

        match &self.kind {
            ErrorKind::UnexpectedToken { found, expected } => CognitiveError::new(
                format!("Unexpected {found} in query"),
                ErrorContext::new(
                    format!("Expected: {}", expected.join(" or ")),
                    format!(
                        "Found {found} at line {}, column {}",
                        self.position.line, self.position.column
                    ),
                ),
                &self.suggestion,
                &self.example,
                Confidence::HIGH,
            )
            .with_details(format!(
                "Parse error at line {}, column {}\n{snippet}\n{pointer}",
                self.position.line, self.position.column
            )),

            ErrorKind::UnknownKeyword {
                found,
                did_you_mean,
            } => {
                let mut error = CognitiveError::new(
                    format!("Unknown keyword '{found}'"),
                    ErrorContext::new(
                        "Valid keyword",
                        did_you_mean.as_ref().map_or_else(
                            || format!("'{found}' is not recognized"),
                            |correction| format!("Did you mean '{correction}'?"),
                        ),
                    ),
                    &self.suggestion,
                    &self.example,
                    Confidence::HIGH,
                );

                if let Some(correction) = did_you_mean {
                    error = error.with_details(format!(
                        "Unknown keyword '{found}' at line {}, column {}\nDid you mean '{correction}'?\n{snippet}\n{pointer}",
                        self.position.line, self.position.column
                    ));
                } else {
                    error = error.with_details(format!(
                        "Unknown keyword '{found}' at line {}, column {}\n{snippet}\n{pointer}",
                        self.position.line, self.position.column
                    ));
                }

                error
            }

            ErrorKind::InvalidSyntax { message } => CognitiveError::new(
                format!("Invalid syntax: {message}"),
                ErrorContext::new(
                    "Valid query syntax",
                    format!(
                        "Error at line {}, column {}",
                        self.position.line, self.position.column
                    ),
                ),
                &self.suggestion,
                &self.example,
                Confidence::HIGH,
            )
            .with_details(format!(
                "Syntax error at line {}, column {}\n{snippet}\n{pointer}",
                self.position.line, self.position.column
            )),

            ErrorKind::ValidationError { message } => CognitiveError::new(
                format!("Invalid query: {message}"),
                ErrorContext::new(
                    "Valid query semantics",
                    format!(
                        "Error at line {}, column {}",
                        self.position.line, self.position.column
                    ),
                ),
                &self.suggestion,
                &self.example,
                Confidence::HIGH,
            )
            .with_details(format!(
                "Validation error at line {}, column {}\n{snippet}\n{pointer}",
                self.position.line, self.position.column
            )),

            ErrorKind::UnexpectedEof => CognitiveError::new(
                "Query ended unexpectedly",
                ErrorContext::new(
                    "Complete query syntax",
                    format!(
                        "Query ended at line {}, column {}",
                        self.position.line, self.position.column
                    ),
                ),
                &self.suggestion,
                &self.example,
                Confidence::CERTAIN,
            )
            .with_details(format!(
                "Query ended unexpectedly at line {}, column {}",
                self.position.line, self.position.column
            )),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "Parse error at line {}, column {}:",
            self.position.line, self.position.column
        )?;

        match &self.kind {
            ErrorKind::UnexpectedToken { found, expected } => {
                writeln!(f, "  Found: {found}")?;
                writeln!(f, "  Expected: {}", expected.join(" or "))?;
            }
            ErrorKind::UnknownKeyword {
                found,
                did_you_mean,
            } => {
                writeln!(f, "  Unknown keyword: '{found}'")?;
                if let Some(suggestion) = did_you_mean {
                    writeln!(f, "  Did you mean: '{suggestion}'?")?;
                }
            }
            ErrorKind::InvalidSyntax { message } => {
                writeln!(f, "  {message}")?;
            }
            ErrorKind::ValidationError { message } => {
                writeln!(f, "  Validation error: {message}")?;
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

impl std::error::Error for ParseError {}

impl From<TokenizeError> for ParseError {
    fn from(err: TokenizeError) -> Self {
        // Convert tokenize error to parse error with appropriate context
        match err {
            TokenizeError::UnexpectedCharacter { ch, position } => Self::invalid_syntax(
                format!("Unexpected character '{ch}'"),
                position,
                format!("Remove '{ch}' or check query syntax"),
                "RECALL episode WHERE confidence > 0.7",
            ),
            TokenizeError::UnterminatedString { start } => Self::invalid_syntax(
                "Unterminated string literal",
                start,
                "Add closing \" to string literal",
                r#"RECALL "my episode""#,
            ),
            TokenizeError::InvalidNumber { text, position } => Self::invalid_syntax(
                format!("Invalid number format: '{text}'"),
                position,
                "Use format: 123 (integer) or 0.5 (float)",
                "WHERE confidence > 0.7",
            ),
            TokenizeError::InvalidEscape { escape, position } => Self::invalid_syntax(
                format!("Unknown escape sequence '\\{escape}'"),
                position,
                format!("Replace '\\{escape}' with valid escape or remove backslash"),
                r#"RECALL "line one\nline two""#,
            ),
        }
    }
}

// ============================================================================
// Parser Context for Context-Aware Error Messages
// ============================================================================

/// Parser context for generating context-aware error messages.
///
/// Different parser states generate different error messages with appropriate
/// suggestions and examples. This follows the principle of providing actionable
/// guidance based on what the user is trying to accomplish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserContext {
    /// At the start of a query (expecting operation keyword)
    QueryStart,

    /// After RECALL keyword (expecting pattern)
    AfterRecall,

    /// After SPREAD keyword (expecting FROM)
    AfterSpread,

    /// After PREDICT keyword (expecting pattern)
    AfterPredict,

    /// After IMAGINE keyword (expecting pattern)
    AfterImagine,

    /// After CONSOLIDATE keyword (expecting episodes)
    AfterConsolidate,

    /// Inside WHERE clause (expecting constraint)
    InConstraints,

    /// Inside pattern specification
    InPattern,

    /// Inside embedding literal [...]
    InEmbedding,

    /// Generic context (fallback)
    Generic,
}

/// Generate context-aware suggestion and example.
///
/// This function implements the core logic for error message generation.
/// It returns (suggestion, example) tuples tailored to the parser state.
fn generate_suggestion_and_example(
    context: ParserContext,
    expected: &[String],
) -> (String, String) {
    match context {
        ParserContext::QueryStart => (
            "Query must start with a cognitive operation keyword: RECALL, PREDICT, IMAGINE, CONSOLIDATE, or SPREAD".to_string(),
            "RECALL episode WHERE confidence > 0.7".to_string(),
        ),

        ParserContext::AfterRecall => (
            "RECALL requires a pattern: node ID, embedding vector, or content match".to_string(),
            "RECALL episode_123".to_string(),
        ),

        ParserContext::AfterSpread => (
            "SPREAD requires FROM keyword followed by node identifier".to_string(),
            "SPREAD FROM node_123 MAX_HOPS 5".to_string(),
        ),

        ParserContext::AfterPredict => (
            "PREDICT requires a pattern followed by GIVEN and context nodes".to_string(),
            "PREDICT episode GIVEN context1, context2".to_string(),
        ),

        ParserContext::AfterImagine => (
            "IMAGINE requires a pattern, optionally followed by BASED ON and seed nodes".to_string(),
            "IMAGINE episode BASED ON seed1, seed2".to_string(),
        ),

        ParserContext::AfterConsolidate => (
            "CONSOLIDATE requires episodes followed by INTO and target node".to_string(),
            "CONSOLIDATE episode INTO semantic_node".to_string(),
        ),

        ParserContext::InConstraints => (
            "WHERE clause requires field name, operator, and value".to_string(),
            "WHERE confidence > 0.7".to_string(),
        ),

        ParserContext::InPattern => (
            "Pattern can be a node ID, embedding vector [...], or content string".to_string(),
            "RECALL [0.1, 0.2, 0.3] THRESHOLD 0.8".to_string(),
        ),

        ParserContext::InEmbedding => (
            "Embedding must be a list of numbers: [0.1, 0.2, 0.3]".to_string(),
            "RECALL [0.1, 0.2, 0.3]".to_string(),
        ),

        ParserContext::Generic => {
            if expected.is_empty() {
                (
                    "Check query syntax and ensure all keywords are spelled correctly".to_string(),
                    "RECALL episode WHERE confidence > 0.7".to_string(),
                )
            } else {
                (
                    format!("Use one of: {}", expected.join(", ")),
                    "RECALL episode WHERE confidence > 0.7".to_string(),
                )
            }
        }
    }
}

/// Generate example for a specific keyword.
fn generate_example_for_keyword(keyword: &str, context: ParserContext) -> String {
    match keyword {
        "RECALL" => "RECALL episode WHERE confidence > 0.7".to_string(),
        "PREDICT" => "PREDICT episode GIVEN context1, context2".to_string(),
        "IMAGINE" => "IMAGINE episode BASED ON seed1, seed2".to_string(),
        "CONSOLIDATE" => "CONSOLIDATE episode INTO semantic_node".to_string(),
        "SPREAD" => "SPREAD FROM node_123 MAX_HOPS 5".to_string(),
        "WHERE" => "WHERE confidence > 0.7".to_string(),
        "GIVEN" => "PREDICT episode GIVEN context1".to_string(),
        "FROM" => "SPREAD FROM node_123".to_string(),
        "INTO" => "CONSOLIDATE episode INTO target".to_string(),
        "CONFIDENCE" => "RECALL episode CONFIDENCE > 0.7".to_string(),
        "THRESHOLD" => "RECALL [0.1, 0.2] THRESHOLD 0.8".to_string(),
        _ => generate_suggestion_and_example(context, &[]).1,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Tests are allowed to use unwrap
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_error_display() {
        let err = TokenizeError::UnexpectedCharacter {
            ch: '@',
            position: Position::new(5, 2, 3),
        };
        let msg = err.to_string();
        assert!(msg.contains('@'));
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

        assert!(cognitive.summary.contains('@'));
        assert!(
            cognitive
                .details
                .as_ref()
                .is_some_and(|d| d.contains("line 2"))
        );
        assert!(!cognitive.suggestion.is_empty());
    }

    #[test]
    fn test_extract_line() {
        let source = "line one\nline two\nline three";
        assert_eq!(extract_line(source, 1), "line one");
        assert_eq!(extract_line(source, 2), "line two");
        assert_eq!(extract_line(source, 3), "line three");
        assert_eq!(extract_line(source, 99), "");
    }

    #[test]
    #[allow(clippy::panic)] // Tests are allowed to panic
    fn test_parse_error_with_typo_suggestion() {
        let token = Token::Identifier("RECAL");
        let err = ParseError::unexpected_token(
            &token,
            vec!["RECALL"],
            Position::new(0, 1, 1),
            ParserContext::QueryStart,
        );

        // Should detect typo and suggest RECALL
        if let ErrorKind::UnknownKeyword {
            found,
            did_you_mean,
        } = err.kind
        {
            assert_eq!(found, "RECAL");
            assert_eq!(did_you_mean, Some("RECALL".to_string()));
        } else {
            panic!("Expected UnknownKeyword error, got {:?}", err.kind);
        }

        assert!(!err.suggestion.is_empty());
        assert!(!err.example.is_empty());
    }

    #[test]
    fn test_parse_error_display() {
        let err = ParseError::unexpected_token(
            &Token::Comma,
            vec!["RECALL", "SPREAD"],
            Position::new(5, 2, 3),
            ParserContext::QueryStart,
        );

        let msg = err.to_string();
        assert!(msg.contains("line 2"));
        assert!(msg.contains("column 3"));
        assert!(msg.contains("Suggestion:"));
        assert!(msg.contains("Example:"));
    }

    #[test]
    fn test_all_errors_have_suggestions_and_examples() {
        let test_cases = [
            ParseError::unexpected_token(
                &Token::Comma,
                vec!["RECALL"],
                Position::new(0, 1, 1),
                ParserContext::QueryStart,
            ),
            ParseError::unexpected_eof(Position::new(0, 1, 1), ParserContext::AfterRecall),
            ParseError::validation_error(
                "Empty embedding",
                Position::new(0, 1, 1),
                "Provide at least one value",
                "RECALL [0.1, 0.2]",
            ),
            ParseError::unknown_keyword(
                &Token::Identifier("RECAL"),
                Some("RECALL".to_string()),
                Position::new(0, 1, 1),
                ParserContext::QueryStart,
            ),
        ];

        for err in &test_cases {
            assert!(
                !err.suggestion.is_empty(),
                "Error missing suggestion: {err:?}"
            );
            assert!(!err.example.is_empty(), "Error missing example: {err:?}");
        }
    }

    #[test]
    fn test_context_aware_messages() {
        let contexts = [
            (ParserContext::QueryStart, "RECALL"),
            (ParserContext::AfterRecall, "pattern"),
            (ParserContext::AfterSpread, "FROM"),
            (ParserContext::InConstraints, "WHERE"),
        ];

        for (context, expected_keyword) in contexts {
            let (suggestion, example) = generate_suggestion_and_example(context, &[]);
            assert!(
                !suggestion.is_empty(),
                "Empty suggestion for context {context:?}"
            );
            assert!(!example.is_empty(), "Empty example for context {context:?}");

            // Verify that suggestion mentions relevant keywords
            let suggestion_upper = suggestion.to_uppercase();
            let example_upper = example.to_uppercase();
            let keyword_upper = expected_keyword.to_uppercase();
            assert!(
                suggestion_upper.contains(&keyword_upper) || example_upper.contains(&keyword_upper),
                "Context {context:?} should mention {expected_keyword}. Suggestion: {suggestion:?}, Example: {example:?}"
            );
        }
    }

    #[test]
    fn test_error_message_no_jargon() {
        let err = ParseError::unexpected_token(
            &Token::Comma,
            vec!["identifier"],
            Position::new(0, 1, 1),
            ParserContext::QueryStart,
        );

        let msg = err.to_string();

        // Verify no parser jargon
        assert!(!msg.to_lowercase().contains("ast"));
        assert!(!msg.to_lowercase().contains("token stream"));
        assert!(!msg.to_lowercase().contains("lookahead"));
        assert!(!msg.to_lowercase().contains("parse failure"));
    }

    #[test]
    fn test_error_with_suggestion_builder() {
        let err = ParseError::validation_error(
            "Test error",
            Position::new(0, 1, 1),
            "Original suggestion",
            "Original example",
        )
        .with_suggestion("New suggestion")
        .with_example("New example");

        assert_eq!(err.suggestion, "New suggestion");
        assert_eq!(err.example, "New example");
    }
}
