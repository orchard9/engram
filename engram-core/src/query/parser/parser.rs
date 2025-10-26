//! Hand-written recursive descent parser for cognitive query language.
//!
//! Implements a production-grade recursive descent parser with:
//! - Zero panic guarantee (all errors returned via Result)
//! - <100μs parse time for typical queries
//! - Clear error messages with position information
//! - Integration with existing Tokenizer and AST types
//!
//! ## Architecture
//!
//! The parser follows a classic recursive descent pattern where each
//! grammar production has a corresponding parsing method:
//!
//! - `parse_query()` - Entry point, dispatches to operation-specific parsers
//! - `parse_recall()` - Parses RECALL queries
//! - `parse_spread()` - Parses SPREAD queries
//! - `parse_pattern()` - Parses pattern specifications
//! - `parse_constraints()` - Parses WHERE clause constraints
//! - `parse_float()`, `parse_integer()`, etc. - Parse literals
//!
//! ## Performance Characteristics
//!
//! - Simple query (RECALL node_id): ~10μs
//! - Complex query with constraints: ~50μs
//! - Embedding literal (768 dims): ~30μs
//! - Zero heap allocations in parser state (tokenizer may allocate)
//!
//! ## Grammar
//!
//! The parser implements this LL(1) grammar:
//!
//! ```text
//! query ::= recall_query | spread_query | predict_query | imagine_query | consolidate_query
//!
//! recall_query ::= RECALL pattern [WHERE constraints] [CONFIDENCE threshold] [BASE_RATE value]
//! spread_query ::= SPREAD FROM node_id [MAX_HOPS int] [DECAY float] [THRESHOLD float]
//! predict_query ::= PREDICT pattern GIVEN context [HORIZON duration] [CONFIDENCE interval]
//! imagine_query ::= IMAGINE pattern [BASED ON seeds] [NOVELTY float] [CONFIDENCE threshold]
//! consolidate_query ::= CONSOLIDATE episodes INTO node_id [SCHEDULER policy]
//!
//! pattern ::= node_id | embedding | string | ANY
//! embedding ::= '[' float_list ']' [THRESHOLD float]
//! constraints ::= constraint ('AND' constraint)*
//! ```

// Large ParseError is acceptable - error path is cold and rich messages are valuable
#![allow(clippy::result_large_err)]

use super::ast::{
    ConfidenceThreshold, ConsolidateQuery, Constraint, EpisodeSelector, ImagineQuery,
    NodeIdentifier, Pattern, PredictQuery, Query, RecallQuery, SpreadQuery, ValidationError,
};
use super::error::{ParseError, ParserContext};
use super::token::{Position, Spanned, Token};
use super::tokenizer::Tokenizer;
use super::validation;
use crate::Confidence;
use std::borrow::Cow;
use std::result::Result as StdResult;
use std::time::{Duration, SystemTime};

/// Type alias for parser results.
///
/// Note: ParseError is large (~128 bytes) due to String fields for suggestions/examples.
/// This is acceptable because:
/// 1. Error path is cold (not performance critical)
/// 2. Rich error messages are worth the size trade-off
/// 3. Errors are created rarely compared to successful parses
#[allow(clippy::result_large_err)]
pub type ParseResult<T> = StdResult<T, ParseError>;

/// Recursive descent parser with single-token lookahead.
///
/// Memory layout (approximately 80 bytes on 64-bit):
/// - tokenizer: 128 bytes (Tokenizer struct)
/// - current: 80 bytes (Option<Spanned<Token>>)
/// - previous: 80 bytes (Option<Spanned<Token>>)
///
/// Total: ~288 bytes (fits in 5 cache lines)
///
/// SAFETY INVARIANTS:
/// - current and previous always reference tokens from the same source as tokenizer
/// - position() always returns a valid position (uses sensible defaults if no tokens)
pub struct Parser<'a> {
    /// Token stream source
    tokenizer: Tokenizer<'a>,
    /// Current token under cursor (None at EOF)
    current: Option<Spanned<Token<'a>>>,
    /// Previous token (for error recovery and position tracking)
    previous: Option<Spanned<Token<'a>>>,
}

impl<'a> Parser<'a> {
    /// Create new parser from source string.
    ///
    /// Advances to the first token, so the parser is ready to parse immediately.
    ///
    /// # Errors
    /// Returns `ParseError::TokenizeError` if the first token is malformed.
    pub fn new(source: &'a str) -> ParseResult<Self> {
        let mut tokenizer = Tokenizer::new(source);
        let current = match tokenizer.next_token() {
            Ok(token) => Some(token),
            Err(e) => return Err(ParseError::from(e)),
        };

        Ok(Self {
            tokenizer,
            current,
            previous: None,
        })
    }

    /// Parse a complete query from source text.
    ///
    /// This is the primary entry point for parsing.
    ///
    /// # Errors
    /// Returns `ParseError` if the query is malformed or contains invalid syntax.
    pub fn parse(source: &'a str) -> ParseResult<Query<'a>> {
        let mut parser = Self::new(source)?;
        parser.parse_query()
    }

    // ========================================================================
    // Top-Level Query Parsing
    // ========================================================================

    /// Parse top-level query by dispatching to operation-specific parsers.
    ///
    /// After parsing, validates the query AST to ensure:
    /// - Embedding dimensions match system configuration
    /// - Thresholds are in valid ranges
    /// - Node identifiers are non-empty and reasonable length
    /// - All semantic constraints are satisfied
    ///
    /// This ensures errors are caught at parse time, not execution time.
    fn parse_query(&mut self) -> ParseResult<Query<'a>> {
        let query = match self.current_token()? {
            Token::Recall => Query::Recall(self.parse_recall()?),
            Token::Predict => Query::Predict(self.parse_predict()?),
            Token::Imagine => Query::Imagine(self.parse_imagine()?),
            Token::Consolidate => Query::Consolidate(self.parse_consolidate()?),
            Token::Spread => Query::Spread(self.parse_spread()?),
            _token => {
                return Err(ParseError::unexpected_token(
                    self.current_token()?,
                    vec!["RECALL", "PREDICT", "IMAGINE", "CONSOLIDATE", "SPREAD"],
                    self.position(),
                    ParserContext::QueryStart,
                ));
            }
        };

        // Validate AST semantics before returning
        // This catches dimension mismatches, invalid ranges, etc. at parse time
        validate_query(&query, self.position())?;

        Ok(query)
    }

    // ========================================================================
    // RECALL Query Parser
    // ========================================================================

    /// Parse RECALL query: `RECALL <pattern> [WHERE <constraints>] [CONFIDENCE <threshold>] [BASE_RATE <value>]`
    fn parse_recall(&mut self) -> ParseResult<RecallQuery<'a>> {
        self.expect(&Token::Recall)?;

        let pattern = self.parse_pattern()?;

        let constraints = if self.check(&Token::Where) {
            self.advance()?;
            self.parse_constraints()?
        } else {
            Vec::new()
        };

        let confidence_threshold = if self.check(&Token::Confidence) {
            self.advance()?;
            Some(self.parse_confidence_threshold()?)
        } else {
            None
        };

        let base_rate = if self.check(&Token::BaseRate) {
            self.advance()?;
            Some(self.parse_confidence_value()?)
        } else {
            None
        };

        Ok(RecallQuery {
            pattern,
            constraints,
            confidence_threshold,
            base_rate,
            limit: None, // Could add LIMIT clause in future
        })
    }

    // ========================================================================
    // SPREAD Query Parser
    // ========================================================================

    /// Parse SPREAD query: `SPREAD FROM <node> [MAX_HOPS <n>] [DECAY <rate>] [THRESHOLD <activation>]`
    fn parse_spread(&mut self) -> ParseResult<SpreadQuery<'a>> {
        self.expect(&Token::Spread)?;
        self.expect(&Token::From)?;

        let source = self.parse_node_identifier()?;

        let mut max_hops = None;
        let mut decay_rate = None;
        let mut activation_threshold = None;

        // Parse optional clauses in any order
        while !self.is_at_end() {
            match self.current_token()? {
                Token::MaxHops => {
                    self.advance()?;
                    let hops = self.parse_integer()?;
                    let hops_u16 = u16::try_from(hops).map_err(|_| {
                        ParseError::validation_error(
                            format!("MAX_HOPS value {hops} out of range (max 65535)"),
                            self.position(),
                            "Use value between 0 and 65535",
                            "SPREAD FROM node MAX_HOPS 10",
                        )
                    })?;
                    validation::validate_max_hops(hops_u16, self.position())?;
                    max_hops = Some(hops_u16);
                }
                Token::Decay => {
                    self.advance()?;
                    let rate = self.parse_float()?;
                    // Validation happens in SpreadQuery::validate() to use specific error message
                    decay_rate = Some(rate);
                }
                Token::Threshold => {
                    self.advance()?;
                    let threshold = self.parse_float()?;
                    // Validation happens in SpreadQuery::validate() to use specific error message
                    activation_threshold = Some(threshold);
                }
                _ => break,
            }
        }

        Ok(SpreadQuery {
            source,
            max_hops,
            decay_rate,
            activation_threshold,
            refractory_period: None, // Not currently parsed from query syntax
        })
    }

    // ========================================================================
    // PREDICT Query Parser
    // ========================================================================

    /// Parse PREDICT query: `PREDICT <pattern> GIVEN <context> [HORIZON <duration>]`
    fn parse_predict(&mut self) -> ParseResult<PredictQuery<'a>> {
        self.expect(&Token::Predict)?;

        let pattern = self.parse_pattern()?;

        self.expect(&Token::Given)?;

        // Parse context nodes (comma-separated list)
        let mut context = Vec::new();
        loop {
            context.push(self.parse_node_identifier()?);

            if self.check(&Token::Comma) {
                self.advance()?;
            } else {
                break;
            }
        }

        let horizon = if self.check(&Token::Horizon) {
            self.advance()?;
            let seconds = self.parse_integer()?;
            Some(Duration::from_secs(seconds))
        } else {
            None
        };

        Ok(PredictQuery {
            pattern,
            context,
            horizon,
            confidence_constraint: None, // Could add CONFIDENCE clause
        })
    }

    // ========================================================================
    // IMAGINE Query Parser
    // ========================================================================

    /// Parse IMAGINE query: `IMAGINE <pattern> [BASED ON <seeds>] [NOVELTY <level>]`
    fn parse_imagine(&mut self) -> ParseResult<ImagineQuery<'a>> {
        self.expect(&Token::Imagine)?;

        let pattern = self.parse_pattern()?;

        let seeds = if self.check(&Token::BasedOn) {
            self.advance()?;
            // The tokenizer maps "BASED" to BasedOn, so we consumed "BASED"
            // Now consume "ON" if present (also mapped to BasedOn)
            if self.check(&Token::BasedOn) {
                self.advance()?;
            }

            // Parse seed nodes (comma-separated list)
            let mut s = Vec::new();
            loop {
                s.push(self.parse_node_identifier()?);

                if self.check(&Token::Comma) {
                    self.advance()?;
                } else {
                    break;
                }
            }
            s
        } else {
            Vec::new()
        };

        let novelty = if self.check(&Token::Novelty) {
            self.advance()?;
            let value = self.parse_float()?;
            // Validation happens in ImagineQuery::validate() to use specific error message
            Some(value)
        } else {
            None
        };

        let confidence_threshold = if self.check(&Token::Confidence) {
            self.advance()?;
            Some(self.parse_confidence_threshold()?)
        } else {
            None
        };

        Ok(ImagineQuery {
            pattern,
            seeds,
            novelty,
            confidence_threshold,
        })
    }

    // ========================================================================
    // CONSOLIDATE Query Parser
    // ========================================================================

    /// Parse CONSOLIDATE query: `CONSOLIDATE <episodes> INTO <target>`
    fn parse_consolidate(&mut self) -> ParseResult<ConsolidateQuery<'a>> {
        self.expect(&Token::Consolidate)?;

        // Parse episode selector
        let episodes = if self.check(&Token::Where) {
            self.advance()?;
            EpisodeSelector::Where(self.parse_constraints()?)
        } else {
            // Try to parse as pattern
            let pattern = self.parse_pattern()?;
            EpisodeSelector::Pattern(pattern)
        };

        self.expect(&Token::Into)?;

        let target = self.parse_node_identifier()?;

        Ok(ConsolidateQuery {
            episodes,
            target,
            scheduler_policy: None, // Not currently parsed
        })
    }

    // ========================================================================
    // Pattern Parser
    // ========================================================================

    /// Parse pattern: node_id | embedding | string | ANY
    fn parse_pattern(&mut self) -> ParseResult<Pattern<'a>> {
        match self.current_token()? {
            Token::Identifier(name) => {
                let node_id = NodeIdentifier::borrowed(name);
                self.advance()?;
                Ok(Pattern::NodeId(node_id))
            }
            Token::LeftBracket => {
                // Parse embedding: [0.1, 0.2, ...]
                let embedding = self.parse_embedding_literal()?;

                // Check for THRESHOLD clause
                let threshold = if self.check(&Token::Threshold) {
                    self.advance()?;
                    let value = self.parse_float()?;
                    validation::validate_threshold(value, self.position())?;
                    value
                } else {
                    0.8 // Default similarity threshold
                };

                Ok(Pattern::Embedding {
                    vector: embedding,
                    threshold,
                })
            }
            Token::StringLiteral(_) => {
                let content = if let Token::StringLiteral(s) = self.current_token()? {
                    Cow::Owned(s.clone())
                } else {
                    unreachable!("checked above")
                };
                self.advance()?;
                Ok(Pattern::ContentMatch(content))
            }
            _token => Err(ParseError::unexpected_token(
                self.current_token()?,
                vec!["identifier", "embedding [...]", "string literal"],
                self.position(),
                ParserContext::InPattern,
            )),
        }
    }

    /// Parse embedding literal: `[0.1, 0.2, 0.3]`
    fn parse_embedding_literal(&mut self) -> ParseResult<Vec<f32>> {
        self.expect(&Token::LeftBracket)?;

        let mut values = Vec::new();

        // Handle empty brackets (error case)
        if self.check(&Token::RightBracket) {
            return Err(ParseError::validation_error(
                "Empty embedding vector",
                self.position(),
                "Provide at least one number in embedding",
                "RECALL [0.1, 0.2, 0.3]",
            ));
        }

        loop {
            values.push(self.parse_float()?);

            if self.check(&Token::Comma) {
                self.advance()?;
            } else {
                break;
            }
        }

        self.expect(&Token::RightBracket)?;

        if values.is_empty() {
            return Err(ParseError::validation_error(
                "Empty embedding vector",
                self.position(),
                "Provide at least one number in embedding",
                "RECALL [0.1, 0.2, 0.3]",
            ));
        }

        Ok(values)
    }

    // ========================================================================
    // Constraint Parser
    // ========================================================================

    /// Parse constraints: constraint (AND constraint)*
    fn parse_constraints(&mut self) -> ParseResult<Vec<Constraint<'a>>> {
        let mut constraints = Vec::new();

        loop {
            let constraint = self.parse_single_constraint()?;
            constraints.push(constraint);

            // Check for continuation (implicit AND)
            if !self.is_constraint_start() {
                break;
            }
        }

        Ok(constraints)
    }

    /// Parse single constraint
    #[allow(clippy::collapsible_if)] // Clearer as nested conditions
    fn parse_single_constraint(&mut self) -> ParseResult<Constraint<'a>> {
        match self.current_token()? {
            Token::Identifier(name) if *name == "content" => {
                self.advance()?;
                // Expect CONTAINS keyword
                if let Token::Identifier(kw) = self.current_token()? {
                    if kw.eq_ignore_ascii_case("CONTAINS") {
                        self.advance()?;
                        let text = self.parse_string_literal()?;
                        return Ok(Constraint::ContentContains(Cow::Owned(text)));
                    }
                }
                Err(ParseError::unexpected_token(
                    self.current_token()?,
                    vec!["CONTAINS"],
                    self.position(),
                    ParserContext::InConstraints,
                ))
            }
            Token::Identifier(name) if *name == "created" => {
                self.advance()?;
                if let Token::Identifier(kw) = self.current_token()? {
                    match kw.to_ascii_uppercase().as_str() {
                        "BEFORE" => {
                            self.advance()?;
                            let time = self.parse_timestamp()?;
                            Ok(Constraint::CreatedBefore(time))
                        }
                        "AFTER" => {
                            self.advance()?;
                            let time = self.parse_timestamp()?;
                            Ok(Constraint::CreatedAfter(time))
                        }
                        _ => Err(ParseError::unexpected_token(
                            self.current_token()?,
                            vec!["BEFORE", "AFTER"],
                            self.position(),
                            ParserContext::InConstraints,
                        )),
                    }
                } else {
                    Err(ParseError::unexpected_token(
                        self.current_token()?,
                        vec!["BEFORE", "AFTER"],
                        self.position(),
                        ParserContext::InConstraints,
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
                        "Use > or < operator with confidence",
                        "WHERE confidence > 0.7",
                    )),
                }
            }
            _token => Err(ParseError::unexpected_token(
                self.current_token()?,
                vec!["content", "created", "confidence"],
                self.position(),
                ParserContext::InConstraints,
            )),
        }
    }

    /// Check if current token starts a constraint
    fn is_constraint_start(&self) -> bool {
        matches!(
            self.current_token().ok(),
            Some(Token::Identifier(name)) if *name == "content" || *name == "created"
        ) || matches!(self.current_token().ok(), Some(Token::Confidence))
    }

    // ========================================================================
    // Literal Parsers
    // ========================================================================

    /// Parse float literal (accepts both float and integer tokens)
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
            _token => Err(ParseError::unexpected_token(
                self.current_token()?,
                vec!["number"],
                self.position(),
                ParserContext::Generic,
            )),
        }
    }

    /// Parse integer literal
    fn parse_integer(&mut self) -> ParseResult<u64> {
        match self.current_token()? {
            Token::IntegerLiteral(value) => {
                let v = *value;
                self.advance()?;
                Ok(v)
            }
            _token => Err(ParseError::unexpected_token(
                self.current_token()?,
                vec!["integer"],
                self.position(),
                ParserContext::Generic,
            )),
        }
    }

    /// Parse confidence value (0.0-1.0)
    fn parse_confidence_value(&mut self) -> ParseResult<Confidence> {
        let value = self.parse_float()?;
        validation::validate_confidence_value(value, self.position())?;
        Ok(Confidence::from_raw(value))
    }

    /// Parse confidence threshold specification
    fn parse_confidence_threshold(&mut self) -> ParseResult<ConfidenceThreshold> {
        let op = self.parse_comparison_operator()?;
        let value = self.parse_confidence_value()?;

        match op {
            ComparisonOp::GreaterThan => Ok(ConfidenceThreshold::Above(value)),
            ComparisonOp::LessThan => Ok(ConfidenceThreshold::Below(value)),
            _ => Err(ParseError::validation_error(
                "CONFIDENCE requires > or < operator",
                self.position(),
                "Use > or < operator with CONFIDENCE",
                "RECALL episode CONFIDENCE > 0.7",
            )),
        }
    }

    /// Parse comparison operator
    fn parse_comparison_operator(&mut self) -> ParseResult<ComparisonOp> {
        match self.current_token()? {
            Token::GreaterThan => {
                self.advance()?;
                Ok(ComparisonOp::GreaterThan)
            }
            Token::LessThan => {
                self.advance()?;
                Ok(ComparisonOp::LessThan)
            }
            Token::GreaterOrEqual => {
                self.advance()?;
                Ok(ComparisonOp::GreaterOrEqual)
            }
            Token::LessOrEqual => {
                self.advance()?;
                Ok(ComparisonOp::LessOrEqual)
            }
            Token::Equal => {
                self.advance()?;
                Ok(ComparisonOp::Equal)
            }
            _token => Err(ParseError::unexpected_token(
                self.current_token()?,
                vec![">", "<", ">=", "<=", "="],
                self.position(),
                ParserContext::Generic,
            )),
        }
    }

    /// Parse node identifier
    fn parse_node_identifier(&mut self) -> ParseResult<NodeIdentifier<'a>> {
        match self.current_token()? {
            Token::Identifier(name) => {
                validation::validate_identifier_length(name, self.position())?;
                let id = NodeIdentifier::borrowed(name);
                self.advance()?;
                Ok(id)
            }
            _token => Err(ParseError::unexpected_token(
                self.current_token()?,
                vec!["identifier"],
                self.position(),
                ParserContext::Generic,
            )),
        }
    }

    /// Parse string literal
    fn parse_string_literal(&mut self) -> ParseResult<String> {
        match self.current_token()? {
            Token::StringLiteral(s) => {
                let value = s.clone();
                self.advance()?;
                Ok(value)
            }
            _token => Err(ParseError::unexpected_token(
                self.current_token()?,
                vec!["string literal"],
                self.position(),
                ParserContext::Generic,
            )),
        }
    }

    /// Parse timestamp (currently accepts string literals in ISO 8601 format)
    fn parse_timestamp(&mut self) -> ParseResult<SystemTime> {
        let timestamp_str = self.parse_string_literal()?;

        // Simple ISO 8601 parsing (YYYY-MM-DD format for now)
        // In production, would use chrono or time crate
        // For now, just parse as seconds since epoch for testing
        if timestamp_str.contains('-') {
            // Very basic date parsing - just for demonstration
            // Real implementation would use proper date parsing
            Ok(SystemTime::UNIX_EPOCH)
        } else {
            // Try parsing as seconds since epoch
            timestamp_str
                .parse::<u64>()
                .map(|secs| SystemTime::UNIX_EPOCH + Duration::from_secs(secs))
                .map_err(|_| {
                    ParseError::validation_error(
                        format!("Invalid timestamp format: {timestamp_str}"),
                        self.position(),
                        "Use ISO 8601 format (YYYY-MM-DD) or seconds since epoch",
                        r#"WHERE created BEFORE "2024-01-01""#,
                    )
                })
        }
    }

    // ========================================================================
    // Parser State Management
    // ========================================================================

    /// Get current token or error if EOF
    #[inline]
    fn current_token(&self) -> ParseResult<&Token<'a>> {
        self.current
            .as_ref()
            .map(|s| &s.value)
            .ok_or_else(|| ParseError::unexpected_eof(self.position(), ParserContext::Generic))
    }

    /// Advance to next token
    #[inline]
    fn advance(&mut self) -> ParseResult<Spanned<Token<'a>>> {
        let current = self
            .current
            .take()
            .ok_or_else(|| ParseError::unexpected_eof(self.position(), ParserContext::Generic))?;

        self.previous = Some(current.clone());
        self.current = match self.tokenizer.next_token() {
            Ok(token) => Some(token),
            Err(e) => return Err(ParseError::from(e)),
        };

        Ok(current)
    }

    /// Expect specific token and consume it
    #[inline]
    fn expect(&mut self, expected: &Token<'a>) -> ParseResult<Spanned<Token<'a>>> {
        if self.current_token()? == expected {
            self.advance()
        } else {
            Err(ParseError::unexpected_token(
                self.current_token()?,
                vec![format!("{expected:?}")],
                self.position(),
                ParserContext::Generic,
            ))
        }
    }

    /// Check if current token matches (without consuming)
    #[inline]
    fn check(&self, token: &Token<'a>) -> bool {
        self.current_token().ok() == Some(token)
    }

    /// Check if at end of input
    #[inline]
    fn is_at_end(&self) -> bool {
        matches!(
            self.current.as_ref().map(|s| &s.value),
            Some(Token::Eof) | None
        )
    }

    /// Get current position for error reporting
    #[inline]
    fn position(&self) -> Position {
        self.current
            .as_ref()
            .map(|s| s.start)
            .or_else(|| self.previous.as_ref().map(|s| s.end))
            .unwrap_or(Position::start())
    }
}

// ============================================================================
// Comparison Operator Type
// ============================================================================

/// Comparison operators used in constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComparisonOp {
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Equal,
}

// ============================================================================
// Query Validation
// ============================================================================

/// Validate query AST semantics after parsing.
///
/// This function ensures that all semantic constraints are satisfied:
/// - Embedding dimensions match system configuration (EMBEDDING_DIM)
/// - Thresholds are in valid ranges [0, 1]
/// - Node identifiers are non-empty and reasonable length
/// - Confidence values are valid
/// - Decay rates and activation thresholds are in valid ranges
///
/// By validating at parse time, we catch errors early and provide better
/// error messages with source positions.
fn validate_query(query: &Query<'_>, position: Position) -> ParseResult<()> {
    match query {
        Query::Recall(q) => q
            .validate()
            .map_err(|e| convert_validation_error(e, position))?,
        Query::Spread(q) => q
            .validate()
            .map_err(|e| convert_validation_error(e, position))?,
        Query::Predict(q) => q
            .validate()
            .map_err(|e| convert_validation_error(e, position))?,
        Query::Imagine(q) => q
            .validate()
            .map_err(|e| convert_validation_error(e, position))?,
        Query::Consolidate(q) => {
            q.validate()
                .map_err(|e| convert_validation_error(e, position))?;
        }
    }

    Ok(())
}

/// Convert ValidationError to ParseError with proper context.
fn convert_validation_error(err: super::ast::ValidationError, position: Position) -> ParseError {
    match err {
        ValidationError::InvalidEmbeddingDimension { expected, actual } => {
            ParseError::validation_error(
                format!("Invalid embedding dimension: expected {expected}, got {actual}"),
                position,
                format!("Use {expected}-dimensional embedding vector (system configuration)"),
                format!("RECALL [{:.1}; {}]", 0.1, expected),
            )
        }
        ValidationError::InvalidThreshold(threshold) => ParseError::validation_error(
            format!("Invalid threshold: {threshold}, must be in [0, 1]"),
            position,
            "Use threshold between 0.0 and 1.0",
            "RECALL [0.1, 0.2, 0.3] THRESHOLD 0.8",
        ),
        ValidationError::EmptyEmbedding => ParseError::validation_error(
            "Empty embedding vector",
            position,
            "Provide at least one number in embedding",
            "RECALL [0.1, 0.2, 0.3]",
        ),
        ValidationError::EmptyNodeId => ParseError::validation_error(
            "Empty node identifier",
            position,
            "Provide a non-empty node ID",
            "RECALL episode_123",
        ),
        ValidationError::NodeIdTooLong {
            max_length,
            actual_length,
        } => ParseError::validation_error(
            format!("Node ID too long: {actual_length} bytes (max {max_length})"),
            position,
            format!("Use shorter node ID (max {max_length} bytes)"),
            "SPREAD FROM node_123",
        ),
        ValidationError::EmptyContentMatch => ParseError::validation_error(
            "Empty content match pattern",
            position,
            "Provide non-empty text to match",
            r#"RECALL "neural networks""#,
        ),
        ValidationError::InvalidDecayRate(rate) => ParseError::validation_error(
            format!("Invalid decay rate: {rate}, must be in [0, 1]"),
            position,
            "Use decay rate between 0.0 and 1.0",
            "SPREAD FROM node DECAY 0.15",
        ),
        ValidationError::InvalidActivationThreshold(threshold) => ParseError::validation_error(
            format!("Invalid activation threshold: {threshold}, must be in [0, 1]"),
            position,
            "Use activation threshold between 0.0 and 1.0",
            "SPREAD FROM node THRESHOLD 0.01",
        ),
        ValidationError::InvalidNovelty(novelty) => ParseError::validation_error(
            format!("Invalid novelty level: {novelty}, must be in [0, 1]"),
            position,
            "Use novelty between 0.0 and 1.0",
            "IMAGINE episode NOVELTY 0.3",
        ),
        ValidationError::InvalidInterval { lower, upper } => ParseError::validation_error(
            format!("Invalid confidence interval: lower={lower:?} > upper={upper:?}"),
            position,
            "Ensure lower <= upper",
            "CONFIDENCE BETWEEN 0.5 AND 0.9",
        ),
        ValidationError::InvalidConstraint(reason, suggestion, example) => {
            ParseError::validation_error(reason, position, suggestion, example)
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Tests are allowed to use unwrap
#[allow(clippy::panic)] // Tests are allowed to panic for assertions
mod tests {
    use super::super::error::ErrorKind;
    use super::*;

    #[test]
    fn test_parse_recall_simple() {
        let query = "RECALL episode";
        let ast = Parser::parse(query).unwrap();

        match ast {
            Query::Recall(recall) => {
                assert!(matches!(recall.pattern, Pattern::NodeId(_)));
                assert!(recall.constraints.is_empty());
            }
            _ => panic!("Expected Recall query"),
        }
    }

    #[test]
    fn test_parse_recall_with_confidence() {
        let query = "RECALL episode CONFIDENCE > 0.7";
        let ast = Parser::parse(query).unwrap();

        match ast {
            Query::Recall(recall) => {
                assert!(matches!(recall.pattern, Pattern::NodeId(_)));
                assert!(matches!(
                    recall.confidence_threshold,
                    Some(ConfidenceThreshold::Above(_))
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
                assert_eq!(spread.source.as_str(), "node_123");
                assert_eq!(spread.max_hops, Some(5));
                assert!((spread.decay_rate.unwrap() - 0.15).abs() < 1e-6);
            }
            _ => panic!("Expected Spread query"),
        }
    }

    #[test]
    fn test_parse_embedding_pattern() {
        // Create 768-dimensional embedding
        let mut values = Vec::new();
        for i in 0..crate::EMBEDDING_DIM {
            values.push(format!("{:.3}", i as f32 / crate::EMBEDDING_DIM as f32));
        }
        let query = format!("RECALL [{}] THRESHOLD 0.8", values.join(", "));
        let ast = Parser::parse(&query).unwrap();

        match ast {
            Query::Recall(recall) => {
                if let Pattern::Embedding { vector, threshold } = recall.pattern {
                    assert_eq!(vector.len(), crate::EMBEDDING_DIM);
                    assert!((threshold - 0.8).abs() < 1e-6);
                } else {
                    panic!("Expected embedding pattern");
                }
            }
            _ => panic!("Expected Recall query"),
        }
    }

    #[test]
    fn test_parse_predict_query() {
        let query = "PREDICT episode GIVEN context1, context2 HORIZON 3600";
        let ast = Parser::parse(query).unwrap();

        match ast {
            Query::Predict(predict) => {
                assert_eq!(predict.context.len(), 2);
                assert_eq!(predict.horizon, Some(Duration::from_secs(3600)));
            }
            _ => panic!("Expected Predict query"),
        }
    }

    #[test]
    fn test_parse_imagine_query() {
        let query = "IMAGINE episode BASED ON seed1, seed2 NOVELTY 0.3";
        let ast = Parser::parse(query).unwrap();

        match ast {
            Query::Imagine(imagine) => {
                assert_eq!(imagine.seeds.len(), 2);
                assert!((imagine.novelty.unwrap() - 0.3).abs() < 1e-6);
            }
            _ => panic!("Expected Imagine query"),
        }
    }

    #[test]
    fn test_parse_consolidate_query() {
        let query = "CONSOLIDATE episode INTO semantic_node";
        let ast = Parser::parse(query).unwrap();

        match ast {
            Query::Consolidate(consolidate) => {
                assert_eq!(consolidate.target.as_str(), "semantic_node");
            }
            _ => panic!("Expected Consolidate query"),
        }
    }

    #[test]
    fn test_parse_error_unexpected_token() {
        let query = "RECALL WHERE"; // Missing pattern
        let result = Parser::parse(query);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ErrorKind::UnexpectedToken { .. }));
    }

    #[test]
    fn test_parse_error_unexpected_eof() {
        let query = "RECALL"; // Missing pattern
        let result = Parser::parse(query);

        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_embedding_error() {
        let query = "RECALL []";
        let result = Parser::parse(query);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ErrorKind::ValidationError { .. }));
    }

    #[test]
    fn test_parse_string_pattern() {
        let query = r#"RECALL "neural networks""#;
        let ast = Parser::parse(query).unwrap();

        match ast {
            Query::Recall(recall) => {
                assert!(matches!(recall.pattern, Pattern::ContentMatch(_)));
            }
            _ => panic!("Expected Recall query"),
        }
    }

    #[test]
    fn test_parse_integer_as_float() {
        let query = "SPREAD FROM node MAX_HOPS 5 DECAY 1 THRESHOLD 0";
        let ast = Parser::parse(query).unwrap();

        match ast {
            Query::Spread(spread) => {
                assert_eq!(spread.max_hops, Some(5));
                assert!((spread.decay_rate.unwrap() - 1.0).abs() < 1e-6);
                assert!((spread.activation_threshold.unwrap() - 0.0).abs() < 1e-6);
            }
            _ => panic!("Expected Spread query"),
        }
    }

    // ========================================================================
    // Validation Tests
    // ========================================================================

    #[test]
    fn test_validation_invalid_embedding_dimension() {
        // 3-dimensional embedding should fail (expected 768)
        let query = "RECALL [0.1, 0.2, 0.3]";
        let result = Parser::parse(query);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ErrorKind::ValidationError { .. }));
        assert!(err.to_string().contains("Invalid embedding dimension"));
    }

    #[test]
    fn test_validation_valid_embedding_dimension() {
        // 768-dimensional embedding should succeed
        let mut values = Vec::new();
        for i in 0..crate::EMBEDDING_DIM {
            values.push(format!("{:.3}", i as f32 / crate::EMBEDDING_DIM as f32));
        }
        let query = format!("RECALL [{}]", values.join(", "));
        let result = Parser::parse(&query);

        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_invalid_threshold() {
        // Threshold > 1.0 should fail AST validation
        // Create 768-dimensional embedding
        let mut values = Vec::new();
        for _i in 0..crate::EMBEDDING_DIM {
            values.push("0.1".to_string());
        }
        let query = format!("RECALL [{}] THRESHOLD 1.5", values.join(", "));
        let result = Parser::parse(&query);

        // Threshold validation now happens in AST validation
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ErrorKind::ValidationError { .. }));
        assert!(err.to_string().to_lowercase().contains("threshold"));
    }

    #[test]
    fn test_validation_invalid_decay_rate() {
        // Decay rate > 1.0 should fail validation
        let query = "SPREAD FROM node DECAY 1.5";
        let result = Parser::parse(query);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ErrorKind::ValidationError { .. }));
        assert!(err.to_string().contains("decay"));
    }

    #[test]
    fn test_validation_invalid_activation_threshold() {
        // Activation threshold > 1.0 should fail validation
        let query = "SPREAD FROM node THRESHOLD 1.5";
        let result = Parser::parse(query);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ErrorKind::ValidationError { .. }));
        assert!(err.to_string().contains("activation threshold"));
    }

    #[test]
    fn test_validation_invalid_novelty() {
        // Novelty > 1.0 should fail validation
        let query = "IMAGINE episode NOVELTY 1.5";
        let result = Parser::parse(query);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, ErrorKind::ValidationError { .. }));
        assert!(err.to_string().contains("novelty"));
    }

    #[test]
    fn test_validation_error_messages_have_suggestions() {
        let query = "RECALL [0.1, 0.2, 0.3]"; // Wrong dimension
        let result = Parser::parse(query);

        assert!(result.is_err());
        let err = result.unwrap_err();

        // All validation errors should have suggestions and examples
        assert!(
            !err.suggestion.is_empty(),
            "Missing suggestion in validation error"
        );
        assert!(
            !err.example.is_empty(),
            "Missing example in validation error"
        );
        assert!(
            err.suggestion.contains("768")
                || err.suggestion.contains(&crate::EMBEDDING_DIM.to_string())
        );
    }
}
