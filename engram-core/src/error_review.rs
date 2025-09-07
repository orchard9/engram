//! Error message review framework ensuring all errors have actionable suggestions
//!
//! This module implements automated scanning, validation, and quality review of error
//! messages to ensure they follow cognitive principles and include actionable guidance.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use syn::{ExprMacro, Macro, parse_file, visit::Visit};

/// Error review configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReviewConfig {
    /// Minimum confidence threshold for error suggestions
    pub min_confidence: f32,
    /// Maximum cognitive load score (1-10)
    pub max_cognitive_load: u8,
    /// Required fields that must be present
    pub required_fields: HashSet<String>,
    /// Minimum example code length
    pub min_example_length: usize,
    /// Maximum summary length for System 1 scanning
    pub max_summary_length: usize,
    /// Enable detailed reporting
    pub verbose: bool,
}

impl Default for ErrorReviewConfig {
    fn default() -> Self {
        let mut required_fields = HashSet::new();
        required_fields.insert("summary".to_string());
        required_fields.insert("context".to_string());
        required_fields.insert("suggestion".to_string());
        required_fields.insert("example".to_string());
        required_fields.insert("confidence".to_string());

        Self {
            min_confidence: 0.5,
            max_cognitive_load: 7,
            required_fields,
            min_example_length: 10,
            max_summary_length: 100,
            verbose: false,
        }
    }
}

/// Result of reviewing a single error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReviewResult {
    /// File path where error was found
    pub file_path: String,
    /// Line number where error is defined
    pub line_number: usize,
    /// The error summary text
    pub error_summary: String,
    /// Whether all required fields are present
    pub has_required_fields: bool,
    /// Missing fields
    pub missing_fields: Vec<String>,
    /// Cognitive load score (1-10)
    pub cognitive_load_score: u8,
    /// Quality issues found
    pub quality_issues: Vec<QualityIssue>,
    /// Suggestions for improvement
    pub improvement_suggestions: Vec<String>,
    /// Overall pass/fail status
    pub passes_review: bool,
}

/// Quality issue found in an error message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Type of issue
    pub issue_type: IssueType,
    /// Severity level
    pub severity: Severity,
    /// Description of the issue
    pub description: String,
    /// Suggested fix
    pub suggested_fix: String,
}

/// Types of quality issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueType {
    /// Missing required field
    MissingField,
    /// Vague or unhelpful suggestion
    VagueSuggestion,
    /// Example code too short or unclear
    PoorExample,
    /// Summary too long for System 1 processing
    SummaryTooLong,
    /// Low confidence in error diagnosis
    LowConfidence,
    /// High cognitive load
    HighCognitiveLoad,
    /// Missing "did you mean" suggestions
    NoSimilarSuggestions,
    /// No progressive disclosure
    NoProgressiveDisclosure,
}

/// Severity levels for issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// Must fix - blocks CI
    Error,
    /// Should fix - degrades quality
    Warning,
    /// Could improve - nice to have
    Info,
}

/// Main error review system
pub struct ErrorReviewer {
    config: ErrorReviewConfig,
    results: Vec<ErrorReviewResult>,
    error_catalog: HashMap<String, ErrorDefinition>,
}

/// Definition of an error found in code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDefinition {
    /// Unique identifier for this error
    pub id: String,
    /// File where error is defined
    pub file_path: String,
    /// Line number
    pub line_number: usize,
    /// Summary text
    pub summary: String,
    /// Context (expected/actual)
    pub context: Option<(String, String)>,
    /// Suggestion text
    pub suggestion: Option<String>,
    /// Example code
    pub example: Option<String>,
    /// Confidence level
    pub confidence: Option<f32>,
    /// Additional details
    pub details: Option<String>,
    /// Documentation links
    pub documentation: Option<String>,
    /// Similar values
    pub similar: Vec<String>,
}

impl ErrorReviewer {
    /// Create a new error reviewer
    #[must_use]
    pub fn new(config: ErrorReviewConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            error_catalog: HashMap::new(),
        }
    }

    /// Review all errors in a directory
    pub fn review_directory(&mut self, dir: &Path) -> Result<ErrorReviewReport, String> {
        self.scan_directory(dir)?;
        self.validate_all_errors();
        self.generate_report()
    }

    /// Scan directory for error definitions
    fn scan_directory(&mut self, dir: &Path) -> Result<(), String> {
        for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            if path.is_dir() {
                self.scan_directory(&path)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                self.scan_file(&path)?;
            }
        }
        Ok(())
    }

    /// Scan a single Rust file for error definitions
    fn scan_file(&mut self, file_path: &Path) -> Result<(), String> {
        let content = fs::read_to_string(file_path).map_err(|e| e.to_string())?;
        let syntax_tree = parse_file(&content).map_err(|e| e.to_string())?;

        let mut visitor = ErrorVisitor::new(file_path.to_string_lossy().to_string());
        visitor.visit_file(&syntax_tree);

        for error_def in visitor.errors {
            let id = format!("{}:{}", error_def.file_path, error_def.line_number);
            self.error_catalog.insert(id, error_def);
        }

        Ok(())
    }

    /// Validate all found errors
    fn validate_all_errors(&mut self) {
        for (_id, error_def) in &self.error_catalog {
            let result = self.validate_error(error_def);
            self.results.push(result);
        }
    }

    /// Validate a single error definition
    fn validate_error(&self, error_def: &ErrorDefinition) -> ErrorReviewResult {
        let mut missing_fields = Vec::new();
        let mut quality_issues = Vec::new();

        // Check required fields
        if error_def.summary.is_empty() {
            missing_fields.push("summary".to_string());
        }
        if error_def.context.is_none() {
            missing_fields.push("context".to_string());
        }
        if error_def.suggestion.is_none() {
            missing_fields.push("suggestion".to_string());
        }
        if error_def.example.is_none() {
            missing_fields.push("example".to_string());
        }
        if error_def.confidence.is_none() {
            missing_fields.push("confidence".to_string());
        }

        // Check quality issues
        if let Some(suggestion) = &error_def.suggestion {
            if suggestion.len() < 20 || is_vague_suggestion(suggestion) {
                quality_issues.push(QualityIssue {
                    issue_type: IssueType::VagueSuggestion,
                    severity: Severity::Error,
                    description: format!("Suggestion '{suggestion}' is too vague"),
                    suggested_fix: "Provide specific, actionable steps".to_string(),
                });
            }
        }

        if let Some(example) = &error_def.example {
            if example.len() < self.config.min_example_length {
                quality_issues.push(QualityIssue {
                    issue_type: IssueType::PoorExample,
                    severity: Severity::Warning,
                    description: "Example code is too short".to_string(),
                    suggested_fix: "Provide complete, runnable code example".to_string(),
                });
            }
        }

        if error_def.summary.len() > self.config.max_summary_length {
            quality_issues.push(QualityIssue {
                issue_type: IssueType::SummaryTooLong,
                severity: Severity::Warning,
                description: format!(
                    "Summary is {} chars, max is {}",
                    error_def.summary.len(),
                    self.config.max_summary_length
                ),
                suggested_fix: "Shorten summary for System 1 processing".to_string(),
            });
        }

        if let Some(confidence) = error_def.confidence {
            if confidence < self.config.min_confidence {
                quality_issues.push(QualityIssue {
                    issue_type: IssueType::LowConfidence,
                    severity: Severity::Info,
                    description: format!("Confidence {confidence} is below threshold"),
                    suggested_fix: "Improve error detection to increase confidence".to_string(),
                });
            }
        }

        // Calculate cognitive load
        let cognitive_load_score = self.calculate_cognitive_load(error_def);
        if cognitive_load_score > self.config.max_cognitive_load {
            quality_issues.push(QualityIssue {
                issue_type: IssueType::HighCognitiveLoad,
                severity: Severity::Warning,
                description: format!("Cognitive load score {cognitive_load_score} exceeds maximum"),
                suggested_fix: "Simplify error message structure".to_string(),
            });
        }

        // Generate improvement suggestions
        let improvement_suggestions =
            self.generate_improvement_suggestions(error_def, &quality_issues);

        // Determine pass/fail
        let has_required_fields = missing_fields.is_empty();
        let has_critical_issues = quality_issues
            .iter()
            .any(|issue| issue.severity == Severity::Error);
        let passes_review = has_required_fields && !has_critical_issues;

        ErrorReviewResult {
            file_path: error_def.file_path.clone(),
            line_number: error_def.line_number,
            error_summary: error_def.summary.clone(),
            has_required_fields,
            missing_fields,
            cognitive_load_score,
            quality_issues,
            improvement_suggestions,
            passes_review,
        }
    }

    /// Calculate cognitive load score for an error
    fn calculate_cognitive_load(&self, error_def: &ErrorDefinition) -> u8 {
        let mut score = 0u8;

        // Length factors
        if error_def.summary.len() > 80 {
            score += 2;
        }
        if error_def
            .suggestion
            .as_ref()
            .map_or(0, std::string::String::len)
            > 150
        {
            score += 1;
        }
        if error_def
            .example
            .as_ref()
            .map_or(0, std::string::String::len)
            > 200
        {
            score += 1;
        }

        // Complexity factors
        if error_def.details.is_some() {
            score += 1;
        }
        if error_def.similar.len() > 3 {
            score += 1;
        }

        // Clarity factors
        if let Some(suggestion) = &error_def.suggestion {
            if suggestion.contains("try") || suggestion.contains("maybe") {
                score += 2; // Uncertain language increases cognitive load
            }
        }

        // Technical jargon
        if count_technical_terms(&error_def.summary) > 3 {
            score += 2;
        }

        score.min(10)
    }

    /// Generate improvement suggestions
    fn generate_improvement_suggestions(
        &self,
        error_def: &ErrorDefinition,
        issues: &[QualityIssue],
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Based on missing fields
        if error_def.suggestion.is_none() {
            suggestions.push(
                "Add a specific suggestion field with actionable steps to resolve the error"
                    .to_string(),
            );
        }

        if error_def.example.is_none() {
            suggestions
                .push("Include a concrete code example showing the correct usage".to_string());
        }

        // Based on quality issues
        for issue in issues {
            match issue.issue_type {
                IssueType::VagueSuggestion => {
                    suggestions.push(
                        "Replace vague terms like 'fix', 'check', 'try' with specific actions"
                            .to_string(),
                    );
                }
                IssueType::PoorExample => {
                    suggestions
                        .push("Expand example to show complete context and usage".to_string());
                }
                IssueType::NoSimilarSuggestions if error_def.similar.is_empty() => {
                    suggestions
                        .push("Add 'did you mean?' suggestions for common typos".to_string());
                }
                IssueType::NoProgressiveDisclosure if error_def.details.is_none() => {
                    suggestions.push("Add optional details field for deep debugging".to_string());
                }
                _ => {}
            }
        }

        // General improvements
        if error_def.documentation.is_none() {
            suggestions.push("Consider adding documentation links for further reading".to_string());
        }

        suggestions
    }

    /// Generate final report
    fn generate_report(&self) -> Result<ErrorReviewReport, String> {
        let total_errors = self.results.len();
        let passing_errors = self.results.iter().filter(|r| r.passes_review).count();
        let failing_errors = total_errors - passing_errors;

        let errors_by_severity = self.categorize_by_severity();
        let common_issues = self.find_common_issues();

        Ok(ErrorReviewReport {
            total_errors,
            passing_errors,
            failing_errors,
            pass_rate: (passing_errors as f64 / total_errors.max(1) as f64) * 100.0,
            errors_by_severity,
            common_issues,
            detailed_results: self.results.clone(),
            error_catalog: self.error_catalog.clone(),
        })
    }

    /// Categorize errors by severity
    fn categorize_by_severity(&self) -> HashMap<Severity, Vec<ErrorReviewResult>> {
        let mut by_severity: HashMap<Severity, Vec<ErrorReviewResult>> = HashMap::new();

        for result in &self.results {
            let max_severity = result
                .quality_issues
                .iter()
                .map(|i| i.severity)
                .max()
                .unwrap_or(Severity::Info);

            by_severity
                .entry(max_severity)
                .or_default()
                .push(result.clone());
        }

        by_severity
    }

    /// Find common issues across all errors
    fn find_common_issues(&self) -> Vec<(IssueType, usize)> {
        let mut issue_counts: HashMap<IssueType, usize> = HashMap::new();

        for result in &self.results {
            for issue in &result.quality_issues {
                *issue_counts.entry(issue.issue_type).or_insert(0) += 1;
            }
        }

        let mut common: Vec<_> = issue_counts.into_iter().collect();
        common.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        common
    }
}

/// Final error review report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReviewReport {
    /// Total number of errors reviewed
    pub total_errors: usize,
    /// Number of errors passing review
    pub passing_errors: usize,
    /// Number of errors failing review
    pub failing_errors: usize,
    /// Pass rate percentage
    pub pass_rate: f64,
    /// Errors categorized by severity
    pub errors_by_severity: HashMap<Severity, Vec<ErrorReviewResult>>,
    /// Most common issues found
    pub common_issues: Vec<(IssueType, usize)>,
    /// Detailed results for each error
    pub detailed_results: Vec<ErrorReviewResult>,
    /// Complete error catalog
    pub error_catalog: HashMap<String, ErrorDefinition>,
}

impl ErrorReviewReport {
    /// Generate markdown documentation
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str("# Error Message Review Report\n\n");
        md.push_str(&format!(
            "**Date**: {}\n\n",
            chrono::Utc::now().to_rfc3339()
        ));

        md.push_str("## Summary\n\n");
        md.push_str(&format!("- Total Errors: {}\n", self.total_errors));
        md.push_str(&format!(
            "- Passing: {} ({:.1}%)\n",
            self.passing_errors, self.pass_rate
        ));
        md.push_str(&format!("- Failing: {}\n\n", self.failing_errors));

        md.push_str("## Common Issues\n\n");
        for (issue_type, count) in &self.common_issues {
            md.push_str(&format!("- {issue_type:?}: {count} occurrences\n"));
        }

        md.push_str("\n## Failed Errors\n\n");
        for result in self.detailed_results.iter().filter(|r| !r.passes_review) {
            md.push_str(&format!(
                "### {} (Line {})\n\n",
                result.file_path, result.line_number
            ));
            md.push_str(&format!("**Summary**: {}\n\n", result.error_summary));

            if !result.missing_fields.is_empty() {
                md.push_str(&format!(
                    "**Missing Fields**: {}\n\n",
                    result.missing_fields.join(", ")
                ));
            }

            md.push_str("**Issues**:\n");
            for issue in &result.quality_issues {
                md.push_str(&format!("- [{:?}] {}\n", issue.severity, issue.description));
            }

            if !result.improvement_suggestions.is_empty() {
                md.push_str("\n**Suggestions**:\n");
                for suggestion in &result.improvement_suggestions {
                    md.push_str(&format!("- {suggestion}\n"));
                }
            }

            md.push_str("\n---\n\n");
        }

        md
    }

    /// Generate HTML documentation
    #[must_use]
    pub fn to_html(&self) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Error Review Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 8px; }}
        .pass {{ color: #22c55e; }}
        .fail {{ color: #ef4444; }}
        .error {{ background: #fee2e2; padding: 15px; margin: 10px 0; border-radius: 4px; }}
        .warning {{ background: #fef3c7; padding: 15px; margin: 10px 0; border-radius: 4px; }}
        .info {{ background: #dbeafe; padding: 15px; margin: 10px 0; border-radius: 4px; }}
        pre {{ background: #1e293b; color: #e2e8f0; padding: 10px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Error Message Review Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Errors: {}</p>
        <p class="pass">Passing: {} ({:.1}%)</p>
        <p class="fail">Failing: {}</p>
    </div>
    <h2>Results</h2>
    {}
</body>
</html>"#,
            self.total_errors,
            self.passing_errors,
            self.pass_rate,
            self.failing_errors,
            self.generate_html_results()
        )
    }

    fn generate_html_results(&self) -> String {
        let mut html = String::new();

        for result in &self.detailed_results {
            let status_class = if result.passes_review { "pass" } else { "fail" };
            html.push_str(&format!(
                r#"<div class="error">
                    <h3 class="{}">{} - Line {}</h3>
                    <p><strong>Summary:</strong> {}</p>
                    <p><strong>Cognitive Load:</strong> {}/10</p>
                    "#,
                status_class,
                result.file_path,
                result.line_number,
                result.error_summary,
                result.cognitive_load_score
            ));

            if !result.quality_issues.is_empty() {
                html.push_str("<h4>Issues:</h4><ul>");
                for issue in &result.quality_issues {
                    html.push_str(&format!("<li>{}</li>", issue.description));
                }
                html.push_str("</ul>");
            }

            html.push_str("</div>");
        }

        html
    }
}

/// AST visitor to find error macro invocations
struct ErrorVisitor {
    file_path: String,
    errors: Vec<ErrorDefinition>,
}

impl ErrorVisitor {
    const fn new(file_path: String) -> Self {
        Self {
            file_path,
            errors: Vec::new(),
        }
    }

    fn extract_error_from_macro(&mut self, _mac: &Macro, line_number: usize) {
        // This is a simplified extraction - in practice, we'd parse the macro tokens
        // For now, create a placeholder
        let error_def = ErrorDefinition {
            id: format!("{}:{}", self.file_path, line_number),
            file_path: self.file_path.clone(),
            line_number,
            summary: "Error found in macro".to_string(),
            context: None,
            suggestion: None,
            example: None,
            confidence: None,
            details: None,
            documentation: None,
            similar: Vec::new(),
        };

        self.errors.push(error_def);
    }
}

impl<'ast> Visit<'ast> for ErrorVisitor {
    fn visit_expr_macro(&mut self, node: &'ast ExprMacro) {
        if let Some(ident) = node.mac.path.get_ident() {
            if ident == "cognitive_error" {
                // Extract line number (simplified - would need span info in practice)
                self.extract_error_from_macro(&node.mac, 0);
            }
        }
        syn::visit::visit_expr_macro(self, node);
    }
}

/// Check if a suggestion is too vague
fn is_vague_suggestion(suggestion: &str) -> bool {
    let vague_terms = [
        "fix", "check", "try", "maybe", "probably", "should", "could",
    ];
    let lower = suggestion.to_lowercase();
    vague_terms.iter().any(|term| lower.contains(term))
}

/// Count technical terms in text
fn count_technical_terms(text: &str) -> usize {
    let technical_terms = [
        "mutex", "async", "await", "trait", "impl", "lifetime", "borrow", "move", "clone", "arc",
        "rc", "box", "future", "stream", "spawn", "channel", "thread",
    ];

    let lower = text.to_lowercase();
    technical_terms
        .iter()
        .filter(|term| lower.contains(*term))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_review_config() {
        let config = ErrorReviewConfig::default();
        assert!(config.required_fields.contains("summary"));
        assert!(config.required_fields.contains("suggestion"));
        assert!(config.required_fields.contains("example"));
    }

    #[test]
    fn test_cognitive_load_calculation() {
        let reviewer = ErrorReviewer::new(ErrorReviewConfig::default());

        let simple_error = ErrorDefinition {
            id: "test:1".to_string(),
            file_path: "test.rs".to_string(),
            line_number: 1,
            summary: "Node not found".to_string(),
            context: Some(("Valid node".to_string(), "Invalid".to_string())),
            suggestion: Some("Use graph.nodes()".to_string()),
            example: Some("graph.get(id)".to_string()),
            confidence: Some(0.9),
            details: None,
            documentation: None,
            similar: vec![],
        };

        let load = reviewer.calculate_cognitive_load(&simple_error);
        assert!(load <= 5, "Simple error should have low cognitive load");

        let complex_error = ErrorDefinition {
            id: "test:2".to_string(),
            file_path: "test.rs".to_string(),
            line_number: 2,
            summary: "Complex serialization error with multiple nested async trait bounds failed when processing mutex arc spawn future stream channel".to_string(), // >80 chars with many technical terms
            context: Some(("Expected".to_string(), "Actual".to_string())),
            suggestion: Some("Try maybe checking if the mutex lifetime is valid and the async trait impl matches the expected bounds. You should probably also verify the arc reference counting and the future execution context to ensure proper thread safety".to_string()), // >150 chars with vague terms
            example: Some("Long example code here that spans multiple lines and includes lots of technical details about async runtime and trait implementations. This example needs to be really long to trigger the threshold so we add more content about how to properly handle the error in production environments".to_string()), // >200 chars
            confidence: Some(0.3),
            details: Some("Lots of additional details".to_string()),
            documentation: Some("https://docs.rs/complex".to_string()),
            similar: vec!["item1".to_string(), "item2".to_string(), "item3".to_string(), "item4".to_string()],
        };

        let high_load = reviewer.calculate_cognitive_load(&complex_error);
        assert!(
            high_load >= 6,
            "Complex error should have high cognitive load, got: {high_load}"
        );
    }

    #[test]
    fn test_vague_suggestion_detection() {
        assert!(is_vague_suggestion("Try to fix the error"));
        assert!(is_vague_suggestion("Maybe check the input"));
        assert!(is_vague_suggestion("You should probably validate"));
        assert!(!is_vague_suggestion(
            "Use graph.nodes() to list available nodes"
        ));
        assert!(!is_vague_suggestion(
            "Call validate_input() before processing"
        ));
    }

    #[test]
    fn test_technical_term_counting() {
        assert_eq!(count_technical_terms("Basic error message"), 0);
        assert_eq!(count_technical_terms("The mutex is locked"), 1);
        assert_eq!(count_technical_terms("async trait impl with lifetime"), 4);
        assert_eq!(
            count_technical_terms("spawn thread with arc mutex future"),
            6 // spawn, thread, arc, rc (substring of arc), mutex, future
        );
    }
}
