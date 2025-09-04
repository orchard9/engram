//! Comprehensive error handling infrastructure with cognitive principles
//!
//! Every error provides context, suggestions, and examples to support
//! developers even when System 2 reasoning is degraded (e.g., debugging at 3am).

use crate::Confidence;
use std::fmt;

/// Context information for an error: what was expected vs what actually happened
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub expected: String,
    pub actual: String,
}

impl ErrorContext {
    pub fn new(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self {
            expected: expected.into(),
            actual: actual.into(),
        }
    }
}

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
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Add documentation references
    pub fn with_documentation(mut self, docs: impl Into<String>) -> Self {
        self.documentation = Some(docs.into());
        self
    }

    /// Add "did you mean?" suggestions using edit distance
    pub fn with_similar(mut self, similar: Vec<String>) -> Self {
        self.similar = similar;
        self
    }

    /// Chain with a source error
    pub fn with_source(mut self, source: Box<dyn std::error::Error + Send + Sync>) -> Self {
        self.source = Some(source);
        self
    }

    /// Calculate edit distance between two strings (Levenshtein distance)
    pub fn edit_distance(s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
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
            self.summary, self.confidence.mean
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
            writeln!(f, "\n  Details: {}", details)?;
        }

        if let Some(ref docs) = self.documentation {
            writeln!(f, "  Documentation: {}", docs)?;
        }

        // Chain source errors if present
        if let Some(ref source) = self.source {
            write!(f, "\n  Caused by: {}", source)?;
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

    pub fn summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    pub fn context(mut self, expected: impl Into<String>, actual: impl Into<String>) -> Self {
        self.context = Some(ErrorContext::new(expected, actual));
        self
    }

    pub fn suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    pub fn example(mut self, example: impl Into<String>) -> Self {
        self.example = Some(example.into());
        self
    }

    pub fn confidence(mut self, confidence: Confidence) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    pub fn documentation(mut self, docs: impl Into<String>) -> Self {
        self.documentation = Some(docs.into());
        self
    }

    pub fn similar(mut self, similar: Vec<String>) -> Self {
        self.similar = similar;
        self
    }

    pub fn source(mut self, source: Box<dyn std::error::Error + Send + Sync>) -> Self {
        self.source = Some(source);
        self
    }

    /// Build the error, returning None if required fields are missing
    pub fn build(self) -> Option<CognitiveError> {
        Some(CognitiveError {
            summary: self.summary?,
            context: self.context?,
            suggestion: self.suggestion?,
            example: self.example?,
            confidence: self.confidence?,
            details: self.details,
            documentation: self.documentation,
            similar: self.similar,
            source: self.source,
        })
    }

    /// Build the error, panicking if required fields are missing (for use in macros)
    pub fn build_or_panic(self) -> CognitiveError {
        CognitiveError {
            summary: self.summary.expect("CognitiveError requires summary"),
            context: self.context.expect("CognitiveError requires context"),
            suggestion: self.suggestion.expect("CognitiveError requires suggestion"),
            example: self.example.expect("CognitiveError requires example"),
            confidence: self.confidence.expect("CognitiveError requires confidence"),
            details: self.details,
            documentation: self.documentation,
            similar: self.similar,
            source: self.source,
        }
    }
}

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
            Confidence::high(), // Default to high confidence when wrapping known errors
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
    pub fn new(value: T, confidence: Confidence) -> Self {
        Self {
            value,
            confidence,
            warnings: Vec::new(),
        }
    }

    pub fn with_warning(mut self, warning: CognitiveError) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Check if this result meets a confidence threshold
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence.mean >= threshold
    }

    /// Convert to a Result, failing if confidence is below threshold
    pub fn require_confidence(self, threshold: f64) -> Result<T, CognitiveError> {
        if self.confidence.mean >= threshold {
            Ok(self.value)
        } else {
            Err(cognitive_error!(
                summary: format!("Result confidence {} below required threshold {}", self.confidence.mean, threshold),
                context: expected = format!("Confidence >= {}", threshold),
                         actual = format!("Confidence = {}", self.confidence.mean),
                suggestion: "Retry operation with more data or lower threshold",
                example: "let result = operation().require_confidence(0.5)?;",
                confidence: Confidence::exact(1.0)
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_error_display() {
        let err = cognitive_error!(
            summary: "Node not found",
            context: expected = "Valid node ID", actual = "user_999",
            suggestion: "Check available nodes with graph.nodes()",
            example: "graph.get_node(\"user_123\")",
            confidence: Confidence::high(),
            similar: ["user_123", "user_124"]
        );

        let display = format!("{}", err);
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
    fn test_error_builder() {
        let err = CognitiveErrorBuilder::new()
            .summary("Test error")
            .context("expected", "actual")
            .suggestion("do this")
            .example("code here")
            .confidence(Confidence::medium())
            .build()
            .unwrap();

        assert_eq!(err.summary, "Test error");
        assert_eq!(err.confidence.mean, 0.5);
    }

    #[test]
    fn test_partial_result() {
        let result = PartialResult::new(vec![1, 2, 3], Confidence::exact(0.7));
        assert!(result.is_confident(0.6));
        assert!(!result.is_confident(0.8));

        // Test with a high confidence result
        let high_conf_result = PartialResult::new(vec![1, 2, 3], Confidence::exact(0.7));
        let value = high_conf_result.require_confidence(0.6).unwrap();
        assert_eq!(value, vec![1, 2, 3]);

        // Test with a low confidence result
        let low_conf_result = PartialResult::new(vec![1, 2, 3], Confidence::exact(0.7));
        let err = low_conf_result.require_confidence(0.8);
        assert!(err.is_err());
    }
}
