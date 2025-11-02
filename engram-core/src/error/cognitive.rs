//! Comprehensive error handling infrastructure with cognitive principles
//!
//! Every error provides context, suggestions, and examples to support
//! developers even when System 2 reasoning is degraded (e.g., debugging at 3am).
//!
//! Includes recovery strategies for graceful degradation and automated error handling.

use crate::{Confidence, Memory};
use std::fmt;
use thiserror::Error;

/// Context information for an error: what was expected vs what actually happened
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Expected value or state
    pub expected: String,
    /// Actual value or state encountered
    pub actual: String,
}

impl ErrorContext {
    /// Create new error context with expected and actual values
    pub fn new(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self {
            expected: expected.into(),
            actual: actual.into(),
        }
    }
}

/// Recovery strategies for different types of errors
///
/// These strategies define how the system should attempt to recover from errors
/// to provide graceful degradation rather than catastrophic failure.
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry the operation with exponential backoff
    Retry {
        /// Maximum number of retry attempts
        max_attempts: u32,
        /// Backoff time in milliseconds between retries
        backoff_ms: u64,
    },
    /// Use a fallback implementation or default value
    Fallback {
        /// Description of the fallback strategy
        description: String,
    },
    /// Return partial results with reduced confidence
    PartialResult {
        /// Description of what partial results are available
        description: String,
    },
    /// Continue operation without the failed feature
    ContinueWithoutFeature,
    /// Requires manual intervention to resolve
    RequiresIntervention {
        /// Required action for manual intervention
        action: String,
    },
}

/// Production-ready error types with recovery strategies
///
/// Combines the cognitive principles of `CognitiveError` with practical
/// recovery strategies for production systems.
#[derive(Error, Debug)]
pub enum EngramError {
    #[error("Memory operation failed: {context}")]
    /// Memory-related error with context and recovery
    Memory {
        /// Context describing what memory operation failed
        context: String,
        #[source]
        /// The underlying error that caused this failure
        source: Box<dyn std::error::Error + Send + Sync>,
        /// Recovery strategy for this error
        recovery: RecoveryStrategy,
    },

    #[error("Storage error: {operation}")]
    /// Storage-related error with recovery
    Storage {
        /// Storage operation that failed
        operation: String,
        #[source]
        /// The underlying IO error
        source: std::io::Error,
        /// Recovery strategy for this error
        recovery: RecoveryStrategy,
    },

    #[error("Query failed: {reason}")]
    /// Query operation failed
    Query {
        /// Reason for query failure
        reason: String,
        /// Any partial results that were retrieved
        partial_results: Option<Vec<Memory>>,
        /// Recovery strategy for this error
        recovery: RecoveryStrategy,
    },

    #[error("WAL operation failed: {operation}")]
    /// Write-ahead log operation failed
    WriteAheadLog {
        /// WAL operation that failed
        operation: String,
        #[source]
        /// The underlying IO error
        source: std::io::Error,
        /// Recovery strategy for this error
        recovery: RecoveryStrategy,
        /// Whether the system can continue without this operation
        can_continue: bool,
    },

    #[error("Index corrupted or unavailable")]
    /// Index is corrupted or unavailable
    Index {
        #[source]
        /// The underlying error
        source: Box<dyn std::error::Error + Send + Sync>,
        /// Whether a fallback index is available
        fallback_available: bool,
        /// Recovery strategy for this error
        recovery: RecoveryStrategy,
    },

    #[error("Evidence type mismatch: expected {expected}, got {actual}")]
    /// Occurs when evidence provided doesn't match the expected type
    EvidenceTypeMismatch {
        /// The expected evidence type
        expected: String,
        /// The actual evidence type that was provided
        actual: String,
        /// Recovery strategy for this mismatch
        recovery: RecoveryStrategy,
    },

    #[error("Cue type mismatch: expected {expected}, got {actual}")]
    /// Occurs when a cue type doesn't match what was expected for the operation
    CueTypeMismatch {
        /// The expected cue type
        expected: String,
        /// The actual cue type that was provided
        actual: String,
        /// Recovery strategy for this mismatch
        recovery: RecoveryStrategy,
    },

    #[error("Pattern matching failed: {pattern}")]
    /// Occurs when pattern matching operation fails to find matches
    PatternMatch {
        /// The pattern that failed to match
        pattern: String,
        /// Recovery strategy for this failure
        recovery: RecoveryStrategy,
    },
}

impl EngramError {
    /// Get the recovery strategy for this error
    #[must_use]
    pub const fn recovery_strategy(&self) -> &RecoveryStrategy {
        match self {
            Self::Memory { recovery, .. }
            | Self::Storage { recovery, .. }
            | Self::Query { recovery, .. }
            | Self::WriteAheadLog { recovery, .. }
            | Self::Index { recovery, .. }
            | Self::EvidenceTypeMismatch { recovery, .. }
            | Self::CueTypeMismatch { recovery, .. }
            | Self::PatternMatch { recovery, .. } => recovery,
        }
    }

    /// Check if this error is recoverable
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        !matches!(
            self.recovery_strategy(),
            RecoveryStrategy::RequiresIntervention { .. }
        )
    }

    /// Create a memory error with context
    pub fn memory_error(
        context: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
        recovery: RecoveryStrategy,
    ) -> Self {
        Self::Memory {
            context: context.into(),
            source: Box::new(source),
            recovery,
        }
    }

    /// Create a storage error
    pub fn storage_error(
        operation: impl Into<String>,
        source: std::io::Error,
        recovery: RecoveryStrategy,
    ) -> Self {
        Self::Storage {
            operation: operation.into(),
            source,
            recovery,
        }
    }

    /// Create a query error with optional partial results
    pub fn query_error(
        reason: impl Into<String>,
        partial_results: Option<Vec<Memory>>,
        recovery: RecoveryStrategy,
    ) -> Self {
        Self::Query {
            reason: reason.into(),
            partial_results,
            recovery,
        }
    }

    /// Create an evidence type mismatch error
    pub fn evidence_type_mismatch(
        expected: impl Into<String>,
        actual: impl Into<String>,
        recovery: RecoveryStrategy,
    ) -> Self {
        Self::EvidenceTypeMismatch {
            expected: expected.into(),
            actual: actual.into(),
            recovery,
        }
    }

    /// Create a cue type mismatch error  
    pub fn cue_type_mismatch(
        expected: impl Into<String>,
        actual: impl Into<String>,
        recovery: RecoveryStrategy,
    ) -> Self {
        Self::CueTypeMismatch {
            expected: expected.into(),
            actual: actual.into(),
            recovery,
        }
    }

    /// Create a pattern matching error
    pub fn pattern_match_error(pattern: impl Into<String>, recovery: RecoveryStrategy) -> Self {
        Self::PatternMatch {
            pattern: pattern.into(),
            recovery,
        }
    }
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, EngramError>;

/// Comprehensive error type following cognitive principles
///
/// Every error MUST include:
/// - Context: What the system expected vs what it received
/// - Suggestion: Concrete action to resolve the issue
/// - Example: Working code snippet showing correct usage
/// - Confidence: How confident we are in the error diagnosis
#[derive(Debug)]
pub struct CognitiveError {
    /// What went wrong (one-line summary for System 1 scanning)
    pub summary: String,

    /// What was expected vs what actually happened
    pub context: ErrorContext,

    /// Concrete action to resolve the issue
    pub suggestion: String,

    /// Working code example showing correct usage
    pub example: String,

    /// How confident we are in this error diagnosis
    pub confidence: Confidence,

    /// Optional detailed explanation (for System 2 deep analysis)
    pub details: Option<String>,

    /// Optional documentation links
    pub documentation: Option<String>,

    /// Similar valid values ("did you mean?" suggestions)
    pub similar: Vec<String>,

    /// Source error if this wraps another error
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl CognitiveError {
    /// Creates a new cognitive error with all required fields
    pub fn new(
        summary: impl Into<String>,
        context: ErrorContext,
        suggestion: impl Into<String>,
        example: impl Into<String>,
        confidence: Confidence,
    ) -> Self {
        Self {
            summary: summary.into(),
            context,
            suggestion: suggestion.into(),
            example: example.into(),
            confidence,
            details: None,
            documentation: None,
            similar: Vec::new(),
            source: None,
        }
    }

    /// Add detailed explanation for System 2 analysis
    #[must_use]
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Add documentation references
    #[must_use]
    pub fn with_documentation(mut self, docs: impl Into<String>) -> Self {
        self.documentation = Some(docs.into());
        self
    }

    /// Add "did you mean?" suggestions using edit distance
    #[must_use]
    pub fn with_similar(mut self, similar: Vec<String>) -> Self {
        self.similar = similar;
        self
    }

    /// Chain with a source error
    #[must_use]
    pub fn with_source(mut self, source: Box<dyn std::error::Error + Send + Sync>) -> Self {
        self.source = Some(source);
        self
    }

    /// Calculate edit distance between two strings (Levenshtein distance)
    #[must_use]
    pub fn edit_distance(s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
            row[0] = i;
        }
        if let Some(first_row) = matrix.first_mut() {
            for (j, cell) in first_row.iter_mut().enumerate().take(len2 + 1) {
                *cell = j;
            }
        }

        for (i, c1) in s1.chars().enumerate() {
            for (j, c2) in s2.chars().enumerate() {
                let cost = usize::from(c1 != c2);
                matrix[i + 1][j + 1] = std::cmp::min(
                    std::cmp::min(
                        matrix[i][j + 1] + 1, // deletion
                        matrix[i + 1][j] + 1, // insertion
                    ),
                    matrix[i][j] + cost, // substitution
                );
            }
        }

        matrix[len1][len2]
    }

    /// Find similar strings from a list using edit distance
    #[must_use]
    pub fn find_similar(target: &str, candidates: &[String], max_distance: usize) -> Vec<String> {
        let mut similar: Vec<(String, usize)> = candidates
            .iter()
            .map(|s| (s.clone(), Self::edit_distance(target, s)))
            .filter(|(_, dist)| *dist <= max_distance)
            .collect();

        similar.sort_by_key(|(_, dist)| *dist);
        similar.into_iter().map(|(s, _)| s).take(3).collect()
    }
}

impl fmt::Display for CognitiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Level 1: One-line summary for System 1 scanning
        writeln!(
            f,
            "{} (confidence: {:.1})",
            self.summary,
            self.confidence.raw()
        )?;

        // Level 2: Context and suggestion for quick resolution
        writeln!(f, "  Expected: {}", self.context.expected)?;
        writeln!(f, "  Actual: {}", self.context.actual)?;
        writeln!(f, "  Suggestion: {}", self.suggestion)?;

        // Add "did you mean?" if we have similar values
        if !self.similar.is_empty() {
            writeln!(f, "  Did you mean: {}?", self.similar.join(", "))?;
        }

        writeln!(f, "  Example: {}", self.example)?;

        // Level 3: Optional details for deep analysis
        if let Some(ref details) = self.details {
            writeln!(f, "\n  Details: {details}")?;
        }

        if let Some(ref docs) = self.documentation {
            writeln!(f, "  Documentation: {docs}")?;
        }

        // Chain source errors if present
        if let Some(ref source) = self.source {
            write!(f, "\n  Caused by: {source}")?;
        }

        Ok(())
    }
}

impl std::error::Error for CognitiveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

/// Macro to create a cognitive error with compile-time field checking
///
/// All fields are mandatory to ensure comprehensive error messages
#[macro_export]
macro_rules! cognitive_error {
    (
        summary: $summary:expr,
        context: expected = $expected:expr, actual = $actual:expr,
        suggestion: $suggestion:expr,
        example: $example:expr,
        confidence: $confidence:expr
        $(, details: $details:expr)?
        $(, documentation: $documentation:expr)?
        $(, similar: $similar:expr)?
        $(, source: $source:expr)?
    ) => {{
        #[allow(unused_mut)]
        let mut err = $crate::error::CognitiveError::new(
            $summary,
            $crate::error::ErrorContext::new($expected, $actual),
            $suggestion,
            $example,
            $confidence,
        );

        $(err = err.with_details($details);)?
        $(err = err.with_documentation($documentation);)?
        $(err = err.with_similar($similar.into_iter().map(|s| s.to_string()).collect());)?
        $(err = err.with_source(Box::new($source));)?

        err
    }};
}

/// Builder pattern for creating cognitive errors with partial information
///
/// Ensures all required fields are provided before building
pub struct CognitiveErrorBuilder {
    summary: Option<String>,
    context: Option<ErrorContext>,
    suggestion: Option<String>,
    example: Option<String>,
    confidence: Option<Confidence>,
    details: Option<String>,
    documentation: Option<String>,
    similar: Vec<String>,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl CognitiveErrorBuilder {
    /// Create a new cognitive error builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            summary: None,
            context: None,
            suggestion: None,
            example: None,
            confidence: None,
            details: None,
            documentation: None,
            similar: Vec::new(),
            source: None,
        }
    }

    /// Set the error summary message
    #[must_use]
    pub fn summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Set error context with expected vs actual values
    #[must_use]
    pub fn context(mut self, expected: impl Into<String>, actual: impl Into<String>) -> Self {
        self.context = Some(ErrorContext::new(expected, actual));
        self
    }

    /// Set a suggestion for resolving the error
    #[must_use]
    pub fn suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Set an example of correct usage
    #[must_use]
    pub fn example(mut self, example: impl Into<String>) -> Self {
        self.example = Some(example.into());
        self
    }

    /// Set the confidence level for this error
    #[must_use]
    pub const fn confidence(mut self, confidence: Confidence) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Set additional details about the error
    #[must_use]
    pub fn details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Set documentation link for this error
    #[must_use]
    pub fn documentation(mut self, docs: impl Into<String>) -> Self {
        self.documentation = Some(docs.into());
        self
    }

    /// Set list of similar errors for context
    #[must_use]
    pub fn similar(mut self, similar: Vec<String>) -> Self {
        self.similar = similar;
        self
    }

    /// Set the underlying source error
    #[must_use]
    pub fn source(mut self, source: Box<dyn std::error::Error + Send + Sync>) -> Self {
        self.source = Some(source);
        self
    }

    /// Build the error, returning None if required fields are missing
    #[must_use]
    pub fn build(self) -> Option<CognitiveError> {
        self.build_or_error().ok()
    }

    /// Build the error, returning a descriptive error if required fields are missing.
    ///
    /// # Errors
    ///
    /// Returns a [`CognitiveErrorBuilderError`] if any mandatory fields are absent.
    pub fn build_or_error(self) -> std::result::Result<CognitiveError, CognitiveErrorBuilderError> {
        let Self {
            summary,
            context,
            suggestion,
            example,
            confidence,
            details,
            documentation,
            similar,
            source,
        } = self;

        match (summary, context, suggestion, example, confidence) {
            (Some(summary), Some(context), Some(suggestion), Some(example), Some(confidence)) => {
                Ok(CognitiveError {
                    summary,
                    context,
                    suggestion,
                    example,
                    confidence,
                    details,
                    documentation,
                    similar,
                    source,
                })
            }
            (summary, context, suggestion, example, confidence) => {
                let mut missing_fields = Vec::new();
                if summary.is_none() {
                    missing_fields.push("summary");
                }
                if context.is_none() {
                    missing_fields.push("context");
                }
                if suggestion.is_none() {
                    missing_fields.push("suggestion");
                }
                if example.is_none() {
                    missing_fields.push("example");
                }
                if confidence.is_none() {
                    missing_fields.push("confidence");
                }
                Err(CognitiveErrorBuilderError::missing(missing_fields))
            }
        }
    }
}

/// Error returned when mandatory fields are missing from a [`CognitiveErrorBuilder`].
#[derive(Debug)]
pub struct CognitiveErrorBuilderError {
    missing_fields: Vec<&'static str>,
}

impl CognitiveErrorBuilderError {
    const fn missing(missing_fields: Vec<&'static str>) -> Self {
        Self { missing_fields }
    }

    /// Fields that were missing when constructing the error.
    #[must_use]
    pub fn missing_fields(&self) -> &[&'static str] {
        &self.missing_fields
    }
}

impl fmt::Display for CognitiveErrorBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.missing_fields.is_empty() {
            write!(f, "builder missing required fields")
        } else {
            write!(
                f,
                "builder missing required fields: {}",
                self.missing_fields.join(", ")
            )
        }
    }
}

impl std::error::Error for CognitiveErrorBuilderError {}

impl Default for CognitiveErrorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for adding cognitive context to standard errors
pub trait CognitiveContext {
    /// Wrap this error with cognitive context
    fn with_cognitive_context(
        self,
        summary: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
        suggestion: impl Into<String>,
        example: impl Into<String>,
    ) -> CognitiveError;
}

impl<E: std::error::Error + Send + Sync + 'static> CognitiveContext for E {
    fn with_cognitive_context(
        self,
        summary: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
        suggestion: impl Into<String>,
        example: impl Into<String>,
    ) -> CognitiveError {
        CognitiveError::new(
            summary,
            ErrorContext::new(expected, actual),
            suggestion,
            example,
            Confidence::HIGH, // Default to high confidence when wrapping known errors
        )
        .with_source(Box::new(self))
    }
}

/// Result type for operations that return partial results with confidence
///
/// Following the principle of graceful degradation: operations should
/// return partial results with low confidence rather than failing completely
#[derive(Debug)]
pub struct PartialResult<T> {
    /// The (possibly partial) result
    pub value: T,
    /// Confidence in the result
    pub confidence: Confidence,
    /// Any errors that occurred during processing (but didn't prevent partial success)
    pub warnings: Vec<CognitiveError>,
}

impl<T> PartialResult<T> {
    /// Create a new partial result with value and confidence
    pub const fn new(value: T, confidence: Confidence) -> Self {
        Self {
            value,
            confidence,
            warnings: Vec::new(),
        }
    }

    /// Add a warning to this partial result
    #[must_use]
    pub fn with_warning(mut self, warning: CognitiveError) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Check if this result meets a confidence threshold
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence.raw() >= threshold
    }

    /// Convert to a `Result`, failing if confidence is below `threshold`
    ///
    /// # Errors
    ///
    /// Returns an error containing a boxed `CognitiveError` when the confidence is
    /// lower than `threshold`.
    pub fn require_confidence(self, threshold: f32) -> std::result::Result<T, Box<CognitiveError>> {
        if self.confidence.raw() >= threshold {
            Ok(self.value)
        } else {
            Err(Box::new(cognitive_error!(
                summary: format!(
                    "Result confidence {} below required threshold {}",
                    self.confidence.raw(),
                    threshold
                ),
                context: expected = format!("Confidence >= {threshold}"),
                         actual = format!("Confidence = {confidence}", confidence = self.confidence.raw()),
                suggestion: "Retry operation with more data or lower threshold",
                example: "let result = operation().require_confidence(0.5)?;",
                confidence: Confidence::exact(1.0)
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error_testing::{CognitiveErrorTesting, ErrorFamily};

    const EPSILON: f32 = 1.0e-6;
    type TestResult<T = ()> = std::result::Result<T, String>;

    #[test]
    fn test_cognitive_error_display() {
        let err = cognitive_error!(
            summary: "Node not found",
            context: expected = "Valid node ID", actual = "user_999",
            suggestion: "Check available nodes with graph.nodes()",
            example: "graph.get_node(\"user_123\")",
            confidence: Confidence::HIGH,
            similar: ["user_123", "user_124"]
        );

        let display = format!("{err}");
        assert!(display.contains("Node not found"));
        assert!(display.contains("confidence: 0.9"));
        assert!(display.contains("Expected: Valid node ID"));
        assert!(display.contains("user_123"));
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(CognitiveError::edit_distance("kitten", "sitting"), 3);
        assert_eq!(CognitiveError::edit_distance("saturday", "sunday"), 3);
        assert_eq!(CognitiveError::edit_distance("test", "test"), 0);
    }

    #[test]
    fn test_find_similar() {
        let candidates = vec![
            "user_123".to_string(),
            "user_124".to_string(),
            "admin_001".to_string(),
            "user_999".to_string(),
        ];

        let similar = CognitiveError::find_similar("user_125", &candidates, 2);
        assert!(similar.contains(&"user_123".to_string()));
        assert!(similar.contains(&"user_124".to_string()));
        assert!(!similar.contains(&"admin_001".to_string()));
    }

    #[test]
    fn test_error_builder() -> TestResult {
        let err = CognitiveErrorBuilder::new()
            .summary("Test error")
            .context("expected", "actual")
            .suggestion("do this")
            .example("code here")
            .confidence(Confidence::MEDIUM)
            .build_or_error()
            .map_err(|err| err.to_string())?;

        if err.summary != "Test error" {
            return Err("Error summary mismatch".to_string());
        }
        if (err.confidence.raw() - 0.5).abs() > EPSILON {
            return Err("Confidence should be 0.5".to_string());
        }
        Ok(())
    }

    #[test]
    fn test_partial_result() -> TestResult {
        let result = PartialResult::new(vec![1, 2, 3], Confidence::exact(0.7));
        if !result.is_confident(0.6) {
            return Err("Result should be confident at 0.6".to_string());
        }
        if result.is_confident(0.8) {
            return Err("Result should not be confident at 0.8".to_string());
        }

        // Test with a high confidence result
        let high_conf_result = PartialResult::new(vec![1, 2, 3], Confidence::exact(0.7));
        let value = high_conf_result
            .require_confidence(0.6)
            .map_err(|err| err.to_string())?;
        if value != vec![1, 2, 3] {
            return Err("Returned value should match input".to_string());
        }

        // Test with a low confidence result
        let low_conf_result = PartialResult::new(vec![1, 2, 3], Confidence::exact(0.7));
        let err = low_conf_result.require_confidence(0.8);
        if err.is_ok() {
            return Err("Low confidence result should error".to_string());
        }
        Ok(())
    }

    #[test]
    fn test_cognitive_error_comprehensive() {
        let mut testing_framework = CognitiveErrorTesting::new();

        let error = cognitive_error!(
            summary: "Memory node 'user_999' not found",
            context: expected = "Valid node ID from current graph",
                     actual = "user_999",
            suggestion: "Use graph.nodes() to list available nodes",
            example: "let node = graph.get_node(\"user_124\").or_insert_default();",
            confidence: Confidence::HIGH,
            similar: ["user_123", "user_124"]
        );

        let result = testing_framework.test_error_comprehensive(&error, ErrorFamily::NodeAccess);

        // Test that error passes all cognitive requirements
        assert!(
            result.passes_all_requirements(),
            "Error failed comprehensive testing: {}",
            result.detailed_report()
        );

        // Test specific aspects
        assert!(result.pattern_consistency.passes_pattern_consistency());
        assert!(result.cognitive_load.passes_cognitive_load_test());
        assert!(result.cognitive_load.fits_cognitive_limits.as_bool());
        assert!(result.cognitive_load.system1_recognizable.as_bool());
    }

    #[test]
    fn test_procedural_learning_simulation() {
        let mut testing_framework = CognitiveErrorTesting::new();

        let error = cognitive_error!(
            summary: "Invalid activation level: 1.5",
            context: expected = "Activation between 0.0 and 1.0",
                     actual = "1.5",
            suggestion: "Use activation.clamp(0.0, 1.0) to ensure valid range",
            example: "node.activate(energy.clamp(0.0, 1.0));",
            confidence: Confidence::exact(1.0)
        );

        // Simulate multiple encounters with similar validation errors
        for i in 0..5 {
            let result =
                testing_framework.test_error_comprehensive(&error, ErrorFamily::Validation);

            if i == 0 {
                // First encounter should take longer
                assert!(
                    result.resolution_time.as_millis() > 15000,
                    "First encounter should take >15s, got {:?}",
                    result.resolution_time
                );
            } else if i == 4 {
                // Later encounters should be faster due to learning
                assert!(
                    result.resolution_time.as_millis() < 25000,
                    "Fifth encounter should be faster, got {:?}",
                    result.resolution_time
                );
            }
        }

        // Check that learning occurred
        assert!(
            testing_framework.is_procedural_learning_occurring(&ErrorFamily::Validation),
            "Procedural learning should have occurred over multiple encounters"
        );
    }

    #[test]
    fn test_cognitive_load_under_fatigue() {
        use crate::error_testing::{CognitiveLoadTester, ProceduralLearningSimulator};

        let tester = CognitiveLoadTester::new();
        let mut simulator = ProceduralLearningSimulator::new().with_high_cognitive_load();

        // Test a complex error under high cognitive load
        let complex_error = cognitive_error!(
            summary: "Serialization failed for memory node with embedded NaN values in confidence scores",
            context: expected = "Valid JSON-serializable memory structure with finite confidence values",
                     actual = "MemoryNode with confidence.raw() = NaN due to division by zero",
            suggestion: "Check for NaN/Infinity values using f64::is_finite() before serialization",
            example: "if node.confidence.raw().is_finite() { serialize(node) } else { fix_confidence(node) }",
            confidence: Confidence::HIGH
        );

        let load_result = tester.test_cognitive_load_compatibility(&complex_error);

        // Should still pass cognitive load tests even when complex
        assert!(
            load_result.fits_cognitive_limits.as_bool(),
            "Error should fit cognitive limits even when detailed"
        );
        assert!(
            load_result.system1_recognizable.as_bool(),
            "Error should be recognizable by System 1 processing"
        );

        // Test simulation under high cognitive load
        let resolution_time =
            simulator.simulate_error_encounter(ErrorFamily::Serialization, &complex_error);
        assert!(
            resolution_time.as_millis() > 30000,
            "Resolution should take longer under high cognitive load"
        );
    }

    #[test]
    fn test_pattern_consistency_across_error_families() {
        let mut testing_framework = CognitiveErrorTesting::new();

        // Test node access errors follow consistent pattern
        let node_error1 = cognitive_error!(
            summary: "Memory node 'session_abc' not found",
            context: expected = "Valid node ID from active session graph",
                     actual = "session_abc",
            suggestion: "Use session.list_nodes() to see available nodes",
            example: "let node = session.get_node(\"session_123\").unwrap_or_default();",
            confidence: Confidence::HIGH
        );

        let node_error2 = cognitive_error!(
            summary: "Graph node 'temp_xyz' not found",
            context: expected = "Valid node ID from current graph context",
                     actual = "temp_xyz",
            suggestion: "Use graph.nodes() to list all available nodes",
            example: "let node = graph.get_node(\"temp_456\").or_insert_default();",
            confidence: Confidence::HIGH
        );

        let result1 =
            testing_framework.test_error_comprehensive(&node_error1, ErrorFamily::NodeAccess);
        let result2 =
            testing_framework.test_error_comprehensive(&node_error2, ErrorFamily::NodeAccess);

        // Both errors should pass pattern consistency
        assert!(
            result1.pattern_consistency.passes_pattern_consistency(),
            "Node error 1 should pass pattern consistency"
        );
        assert!(
            result2.pattern_consistency.passes_pattern_consistency(),
            "Node error 2 should pass pattern consistency"
        );

        // Both should have similar cognitive characteristics
        assert_eq!(
            result1.cognitive_load.fits_cognitive_limits,
            result2.cognitive_load.fits_cognitive_limits
        );
        assert_eq!(
            result1.cognitive_load.system1_recognizable,
            result2.cognitive_load.system1_recognizable
        );
    }

    #[test]
    fn test_performance_requirements() {
        use crate::error_testing::ErrorPerformanceMonitor;
        use std::time::Duration;

        let error = cognitive_error!(
            summary: "Validation failed",
            context: expected = "Valid input",
                     actual = "Invalid input",
            suggestion: "Fix the input",
            example: "validate_input(correct_input)",
            confidence: Confidence::MEDIUM
        );

        // Test formatting performance
        let formatting_time = ErrorPerformanceMonitor::test_formatting_performance(&error);
        assert!(
            formatting_time < Duration::from_millis(10),
            "Error formatting took {formatting_time:?}, should be <10ms"
        );

        // Test comprehension time estimation
        let comprehension_time = ErrorPerformanceMonitor::estimate_comprehension_time(&error, 1.0);
        assert!(
            comprehension_time < Duration::from_millis(30000),
            "Error comprehension estimated at {comprehension_time:?}, should be <30s"
        );

        // Test under high cognitive load
        let fatigued_comprehension =
            ErrorPerformanceMonitor::estimate_comprehension_time(&error, 3.0);
        assert!(
            fatigued_comprehension < Duration::from_millis(90000),
            "Even under high cognitive load, comprehension should be reasonable"
        );
    }

    #[test]
    fn test_context_independence() {
        use crate::error_testing::CognitiveLoadTester;

        let tester = CognitiveLoadTester::new();

        // Test error that defines its domain terms
        let context_independent_error = cognitive_error!(
            summary: "Memory activation threshold exceeded",
            context: expected = "Node activation level between 0.0 (inactive) and 1.0 (fully active)",
                     actual = "Node activation = 1.8 (exceeds maximum)",
            suggestion: "Use node.clamp_activation(0.0, 1.0) to normalize activation levels",
            example: "node.set_activation(energy.min(1.0).max(0.0));",
            confidence: Confidence::HIGH
        );

        let result = tester.test_cognitive_load_compatibility(&context_independent_error);
        assert!(
            result.context_independent.as_bool(),
            "Error should be understandable without prior context"
        );

        // Test error that assumes context
        let context_dependent_error = cognitive_error!(
            summary: "Node failed",
            context: expected = "Success",
                     actual = "Failure",
            suggestion: "Fix it",
            example: "do_fix()",
            confidence: Confidence::LOW
        );

        let bad_result = tester.test_cognitive_load_compatibility(&context_dependent_error);
        assert!(
            !bad_result.context_independent.as_bool(),
            "Vague error should fail context independence test"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::error_testing::{CognitiveErrorTesting, ErrorFamily};
    use proptest::prelude::*;

    /// Generate random but realistic error messages
    fn arb_error_components() -> impl Strategy<Value = (String, String, String, String, String)> {
        (
            prop_oneof![
                Just("not found".to_string()),
                Just("invalid".to_string()),
                Just("failed".to_string()),
                Just("exceeded".to_string()),
            ],
            prop_oneof![
                Just("Valid node ID".to_string()),
                Just("Activation between 0.0 and 1.0".to_string()),
                Just("Finite confidence value".to_string()),
            ],
            any::<String>().prop_map(|s| s.chars().take(20).collect()),
            prop_oneof![
                Just("Use graph.nodes() to list available".to_string()),
                Just("Check input bounds with clamp()".to_string()),
                Just("Validate data before processing".to_string()),
            ],
            prop_oneof![
                Just("let node = graph.get_node(\"id\").unwrap();".to_string()),
                Just("value.clamp(0.0, 1.0)".to_string()),
                Just("if value.is_finite() { process(value) }".to_string()),
            ],
        )
    }

    proptest! {
        #[test]
        fn test_all_generated_errors_pass_cognitive_requirements(
            (error_type, expected, actual, suggestion, example) in arb_error_components()
        ) {
            let mut testing_framework = CognitiveErrorTesting::new();

            let error = cognitive_error!(
                summary: format!("Memory node '{actual}' {error_type}"),
                context: expected = expected,
                         actual = actual,
                suggestion: suggestion,
                example: example,
                confidence: Confidence::HIGH
            );

            let result = testing_framework.test_error_comprehensive(&error, ErrorFamily::NodeAccess);

            // All generated errors should pass basic cognitive requirements
            prop_assert!(result.cognitive_load.fits_cognitive_limits.as_bool(),
                "Generated error should fit cognitive limits");
            prop_assert!(result.formatting_time.as_millis() < 10,
                "Generated error formatting should be <10ms");
        }

        #[test]
        fn test_procedural_learning_always_improves_over_time(
            error_components in arb_error_components()
        ) {
            let mut testing_framework = CognitiveErrorTesting::new();
            let (error_type, expected, actual, suggestion, example) = error_components;

            let error = cognitive_error!(
                summary: format!("Validation {error_type} for {actual}"),
                context: expected = expected,
                         actual = actual,
                suggestion: suggestion,
                example: example,
                confidence: Confidence::MEDIUM
            );

            let mut resolution_times = Vec::new();

            // Simulate 10 encounters with the same error family
            for _ in 0..10 {
                let result = testing_framework.test_error_comprehensive(&error, ErrorFamily::Validation);
                resolution_times.push(result.resolution_time.as_millis());
            }

            // Check that learning occurred (later encounters faster than early ones)
            let early_avg = resolution_times[0..3].iter().sum::<u128>() / 3;
            let late_avg = resolution_times[7..10].iter().sum::<u128>() / 3;

            prop_assert!(late_avg <= early_avg,
                "Learning should occur: early_avg={}, late_avg={}", early_avg, late_avg);
        }

        #[test]
        fn test_similar_suggestions_always_improve_usability(
            node_id in "user_[0-9]{1,3}",
            similar_ids in prop::collection::vec("user_[0-9]{1,3}", 1..4)
        ) {
            let mut testing_framework = CognitiveErrorTesting::new();

            let error_without_similar = cognitive_error!(
                summary: format!("Memory node '{}' not found", node_id),
                context: expected = "Valid node ID from current graph",
                         actual = node_id.clone(),
                suggestion: "Use graph.nodes() to list available nodes",
                example: "let node = graph.get_node(\"user_123\").unwrap();",
                confidence: Confidence::HIGH
            );

            let error_with_similar = cognitive_error!(
                summary: format!("Memory node '{}' not found", node_id),
                context: expected = "Valid node ID from current graph",
                         actual = node_id,
                suggestion: "Use graph.nodes() to list available nodes",
                example: "let node = graph.get_node(\"user_123\").unwrap();",
                confidence: Confidence::HIGH,
                similar: similar_ids
            );

            let result_without = testing_framework.test_error_comprehensive(&error_without_similar, ErrorFamily::NodeAccess);
            let result_with = testing_framework.test_error_comprehensive(&error_with_similar, ErrorFamily::NodeAccess);

            // Errors with similar suggestions should resolve faster
            prop_assert!(result_with.resolution_time <= result_without.resolution_time,
                "Similar suggestions should improve resolution time");
        }

        #[test]
        fn test_cognitive_load_increases_resolution_time(
            components in arb_error_components()
        ) {
            use crate::error_testing::ProceduralLearningSimulator;

            let (error_type, expected, actual, suggestion, example) = components;

            let error = cognitive_error!(
                summary: format!("Operation {error_type} for {actual}"),
                context: expected = expected,
                         actual = actual,
                suggestion: suggestion,
                example: example,
                confidence: Confidence::MEDIUM
            );

            let mut normal_simulator = ProceduralLearningSimulator::new();
            let mut fatigued_simulator = ProceduralLearningSimulator::new().with_high_cognitive_load();

            let normal_time = normal_simulator.simulate_error_encounter(ErrorFamily::NodeAccess, &error);
            let fatigued_time = fatigued_simulator.simulate_error_encounter(ErrorFamily::NodeAccess, &error);

            // High cognitive load should increase resolution time
            prop_assert!(fatigued_time >= normal_time,
                "High cognitive load should increase resolution time: normal={:?}, fatigued={:?}",
                normal_time, fatigued_time);
        }

        #[test]
        fn test_confidence_correlates_with_resolution_success(
            confidence_value in 0.1f64..1.0f64
        ) {
            use crate::error_testing::ProceduralLearningSimulator;

            let mut simulator = ProceduralLearningSimulator::new();

            let error = cognitive_error!(
                summary: "Test error with varying confidence",
                context: expected = "Valid test input",
                         actual = "Test input value",
                suggestion: "Validate the test input",
                example: "validate_test_input(input)",
                confidence: Confidence::exact(crate::numeric::saturating_f32_from_f64(confidence_value))
            );

            let resolution_time = simulator.simulate_error_encounter(ErrorFamily::Validation, &error);

            // Higher confidence should correlate with faster resolution
            // This is a heuristic: confidence > 0.9 should resolve faster than baseline
            // (The simulator adds randomness, so we need reasonable thresholds)
            if confidence_value > 0.9 {
                prop_assert!(resolution_time.as_millis() < 35000,
                    "Very high confidence errors should resolve reasonably quickly");
            }
        }
    }
}
