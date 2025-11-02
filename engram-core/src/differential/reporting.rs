//! Cognitive-friendly reporting for differential testing results
//!
//! This module provides comprehensive reporting capabilities that help developers
//! understand implementation differences through visual and textual reports
//! designed with cognitive ergonomics principles.

use super::{
    Confidence, DifferentialOperation, Divergence, DivergenceSeverity, Implementation,
    OperationResult, PerformanceComparison, PerformanceStats,
};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::{self, Write};

/// Comprehensive report of differential testing results
#[derive(Debug)]
pub struct DifferentialTestReport {
    /// Test session metadata
    pub session_info: SessionInfo,
    /// Overall summary statistics
    pub summary: TestSummary,
    /// Performance comparison across implementations
    pub performance_analysis: PerformanceAnalysis,
    /// List of all divergences found
    pub divergences: Vec<Divergence>,
    /// Cognitive-friendly insights and recommendations
    pub insights: Vec<Insight>,
    /// Detailed test results (optional, for debugging)
    pub detailed_results: Option<Vec<DetailedTestResult>>,
}

/// Session information and metadata
#[derive(Debug)]
pub struct SessionInfo {
    /// When the test was run
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Implementations tested
    pub implementations: Vec<Implementation>,
    /// Reference implementation used for comparison
    pub reference_implementation: Implementation,
    /// Total test duration
    pub duration: std::time::Duration,
    /// Configuration used
    pub config_summary: String,
}

/// High-level summary statistics
#[derive(Debug)]
pub struct TestSummary {
    /// Total operations tested
    pub total_operations: u32,
    /// Operations with equivalent results
    pub equivalent_operations: u32,
    /// Operations with divergent results
    pub divergent_operations: u32,
    /// Success rate as percentage
    pub equivalence_rate: f64,
    /// Most common operation types tested
    pub operation_distribution: HashMap<String, u32>,
}

/// Performance analysis across implementations
#[derive(Debug)]
pub struct PerformanceAnalysis {
    /// Performance statistics per implementation
    pub per_implementation: HashMap<Implementation, PerformanceStats>,
    /// Relative performance comparisons
    pub comparisons: Vec<PerformanceComparison>,
    /// Performance insights
    pub insights: Vec<String>,
    /// Recommended optimizations
    pub optimization_suggestions: Vec<String>,
}

/// Cognitive insight about the test results
#[derive(Debug, Clone)]
pub struct Insight {
    /// Category of insight
    pub category: InsightCategory,
    /// Human-readable insight description
    pub description: String,
    /// Confidence in this insight
    pub confidence: Confidence,
    /// Impact level for developers
    pub impact: InsightImpact,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Categories of insights
#[derive(Debug, Clone, Copy)]
pub enum InsightCategory {
    /// Mathematical correctness
    Correctness,
    /// Performance characteristics
    Performance,
    /// Error handling behavior
    ErrorHandling,
    /// Precision and numerical stability
    Precision,
    /// Implementation complexity
    Complexity,
}

/// Impact levels for insights
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InsightImpact {
    /// Low impact - informational only
    Low,
    /// Medium impact - should be addressed
    Medium,
    /// High impact - must be addressed
    High,
    /// Critical - blocks production use
    Critical,
}

impl fmt::Display for InsightImpact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Medium => write!(f, "Medium"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// Detailed result for a specific test (for debugging)
#[derive(Debug)]
pub struct DetailedTestResult {
    /// The operation tested
    pub operation: DifferentialOperation,
    /// Results from each implementation
    pub results: HashMap<Implementation, OperationResult>,
    /// Whether results were equivalent
    pub equivalent: bool,
    /// Detailed comparison analysis
    pub analysis: String,
}

impl DifferentialTestReport {
    /// Create a new report from test results
    #[must_use]
    pub fn new(
        implementations: &[Implementation],
        reference: Implementation,
        duration: std::time::Duration,
        results: Vec<(
            DifferentialOperation,
            HashMap<Implementation, OperationResult>,
            bool,
        )>,
    ) -> Self {
        let session_info = SessionInfo {
            timestamp: chrono::Utc::now(),
            implementations: implementations.to_vec(),
            reference_implementation: reference,
            duration,
            config_summary: "Default configuration".to_string(),
        };

        let summary = Self::build_summary(&results);
        let performance_analysis = Self::build_performance_analysis(&results, implementations);
        let divergences = Self::extract_divergences(results);
        let insights = Self::generate_insights(&summary, &performance_analysis, &divergences);

        Self {
            session_info,
            summary,
            performance_analysis,
            divergences,
            insights,
            detailed_results: None,
        }
    }

    /// Enable detailed results for debugging
    #[must_use]
    pub fn with_detailed_results(mut self) -> Self {
        // Would populate detailed results from the test data
        self.detailed_results = Some(Vec::new());
        self
    }

    /// Generate a human-readable text report
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut report = String::new();
        self.write_text_header(&mut report);
        self.write_text_summary(&mut report);
        self.write_text_performance(&mut report);
        self.write_text_insights(&mut report);
        self.write_text_divergences(&mut report);
        self.write_text_recommendations(&mut report);
        report
    }

    fn write_text_header(&self, out: &mut String) {
        let _ = writeln!(out, "Differential Testing Report");
        let _ = writeln!(out, "============================");
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "Test Date: {}",
            self.session_info.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        );
        let _ = writeln!(
            out,
            "Duration: {:.2}s",
            self.session_info.duration.as_secs_f64()
        );
        let _ = writeln!(
            out,
            "Implementations: {}",
            self.session_info
                .implementations
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            out,
            "Reference Implementation: {}",
            self.session_info.reference_implementation
        );
        let _ = writeln!(out);
    }

    fn write_text_summary(&self, out: &mut String) {
        let _ = writeln!(out, "Summary");
        let _ = writeln!(out, "-------");
        let _ = writeln!(out, "Total Operations: {}", self.summary.total_operations);
        let equivalence_percent = (self.summary.equivalence_rate * 100.0).clamp(0.0, 100.0);
        let divergent_percent = (100.0 - equivalence_percent).max(0.0);
        let _ = writeln!(
            out,
            "Equivalent: {} ({equivalence_percent:.1}%)",
            self.summary.equivalent_operations
        );
        let _ = writeln!(
            out,
            "Divergent: {} ({divergent_percent:.1}%)",
            self.summary.divergent_operations
        );
        let _ = writeln!(out);
    }

    fn write_text_performance(&self, out: &mut String) {
        if self.performance_analysis.per_implementation.is_empty() {
            return;
        }

        let _ = writeln!(out, "Performance Analysis");
        let _ = writeln!(out, "--------------------");
        for (impl_type, stats) in &self.performance_analysis.per_implementation {
            let _ = writeln!(
                out,
                "{}: {:.0} ns average, {:.1}% success",
                impl_type,
                stats.mean_execution_time_nanos,
                stats.success_rate * 100.0
            );
        }
        let _ = writeln!(out);
    }

    fn write_text_insights(&self, out: &mut String) {
        if self.insights.is_empty() {
            return;
        }

        let mut insights = self.insights.clone();
        insights.sort_by(|a, b| b.impact.cmp(&a.impact));

        let _ = writeln!(out, "Key Insights");
        let _ = writeln!(out, "-----------");
        for insight in insights.iter().take(5) {
            let _ = writeln!(
                out,
                "[{}][{}] {}",
                insight.impact,
                insight_category_name(insight.category),
                insight.description
            );
            if let Some(action) = insight.suggested_actions.first() {
                let _ = writeln!(out, "  Recommended Action: {action}");
            }
        }
        let _ = writeln!(out);
    }

    fn write_text_divergences(&self, out: &mut String) {
        if self.divergences.is_empty() {
            return;
        }

        let _ = writeln!(out, "Implementation Divergences");
        let _ = writeln!(out, "--------------------------");

        for (index, divergence) in self.divergences.iter().enumerate().take(3) {
            let _ = writeln!(
                out,
                "{}. {} [{}]",
                index + 1,
                divergence.operation.description(),
                divergence.severity
            );
            let _ = writeln!(out, "   {}", divergence.explanation);
            if let Some(next_step) = divergence.investigation_steps.first() {
                let _ = writeln!(out, "   Next Step: {next_step}");
            }
            let _ = writeln!(out);
        }

        if self.divergences.len() > 3 {
            let _ = writeln!(
                out,
                "... and {} additional divergences",
                self.divergences.len() - 3
            );
            let _ = writeln!(out);
        }
    }

    fn write_text_recommendations(&self, out: &mut String) {
        let _ = writeln!(out, "Recommendations");
        let _ = writeln!(out, "----------------");

        let summary_line = match self.summary.equivalence_rate {
            rate if rate >= 0.99 => {
                "Excellent equivalence rate; implementations are highly consistent."
            }
            rate if rate >= 0.95 => "Good equivalence rate; investigate minor divergences.",
            rate if rate >= 0.90 => {
                "Moderate equivalence rate; several divergences need attention."
            }
            _ => "Low equivalence rate detected; prioritize investigating divergent operations.",
        };
        let _ = writeln!(out, "{summary_line}");

        for suggestion in self
            .performance_analysis
            .optimization_suggestions
            .iter()
            .take(3)
        {
            let _ = writeln!(out, "- {suggestion}");
        }

        let _ = writeln!(
            out,
            "For detailed analysis, run bisection on specific divergent operations."
        );
        let _ = writeln!(out);
    }

    /// Generate an HTML report with interactive elements
    #[must_use]
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str(r"<!DOCTYPE html>
<html>
<head>
    <title>Differential Testing Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }
        .header { background: #f0f8ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .section { margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; background: #f9f9f9; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .divergence { margin: 10px 0; padding: 10px; border-left: 4px solid #ff6b6b; background: #fff5f5; }
        .insight { margin: 10px 0; padding: 10px; border-left: 4px solid #4ecdc4; background: #f0fffe; }
        .high-impact { border-left-color: #ff6b6b; }
        .medium-impact { border-left-color: #feca57; }
        .low-impact { border-left-color: #48dbfb; }
        .performance-chart { width: 100%; height: 300px; background: #f0f0f0; border-radius: 5px; margin: 10px 0; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>");

        // Header
        let _ = write!(
            &mut html,
            r#"
    <div class="header">
        <h1>Differential Testing Report</h1>
        <p><strong>Date:</strong> {}</p>
        <p><strong>Duration:</strong> {:.2}s</p>
        <p><strong>Implementations:</strong> {}</p>
        <p><strong>Reference:</strong> {}</p>
    </div>"#,
            self.session_info.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            self.session_info.duration.as_secs_f64(),
            self.session_info
                .implementations
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(", "),
            self.session_info.reference_implementation
        );

        // Summary metrics
        html.push_str(r#"<div class="section"><h2>Summary</h2><div>"#);
        let equivalence_percent = (self.summary.equivalence_rate * 100.0).clamp(0.0, 100.0);
        let equivalence_percent_display = format!("{equivalence_percent:.0}");
        let _ = write!(
            &mut html,
            r#"
            <div class="metric">
                <h3>{}</h3>
                <p>Total Operations</p>
            </div>
            <div class="metric">
                <h3>{}%</h3>
                <p>Equivalence Rate</p>
            </div>
            <div class="metric">
                <h3>{}</h3>
                <p>Divergences Found</p>
            </div>"#,
            self.summary.total_operations,
            equivalence_percent_display,
            self.summary.divergent_operations
        );
        html.push_str("</div></div>");

        // Insights
        html.push_str(r#"<div class="section"><h2>Key Insights</h2>"#);
        for insight in &self.insights {
            let impact_class = match insight.impact {
                InsightImpact::Critical | InsightImpact::High => "high-impact",
                InsightImpact::Medium => "medium-impact",
                InsightImpact::Low => "low-impact",
            };
            let _ = write!(
                &mut html,
                r#"<div class="insight {}">
                <strong>[{}]</strong> {}
            </div>"#,
                impact_class, insight.impact, insight.description
            );
        }
        html.push_str("</div>");

        // Divergences
        if !self.divergences.is_empty() {
            html.push_str(r#"<div class="section"><h2>Implementation Divergences</h2>"#);
            for divergence in &self.divergences {
                let _ = write!(
                    &mut html,
                    r#"<div class="divergence">
                    <h4>{} [{}]</h4>
                    <p>{}</p>
                </div>"#,
                    divergence.operation.description(),
                    divergence.severity,
                    divergence.explanation
                );
            }
            html.push_str("</div>");
        }

        html.push_str("</body></html>");
        html
    }

    /// Save the generated report to disk in the requested format.
    ///
    /// # Errors
    /// - Returns an error if writing to `path` fails (for example, missing directories or
    ///   insufficient permissions).
    ///
    /// # Panics
    /// - Never panics.
    pub fn save_to_file(
        &self,
        path: &std::path::Path,
        format: ReportFormat,
    ) -> Result<(), std::io::Error> {
        let content = match format {
            ReportFormat::Text => self.to_text(),
            ReportFormat::Html => self.to_html(),
        };

        std::fs::write(path, content)
    }

    /// Build summary statistics
    fn build_summary(
        results: &[(
            DifferentialOperation,
            HashMap<Implementation, OperationResult>,
            bool,
        )],
    ) -> TestSummary {
        let total_operations = u32::try_from(results.len()).unwrap_or(u32::MAX);
        let equivalent_operations =
            u32::try_from(results.iter().filter(|(_, _, equiv)| *equiv).count())
                .unwrap_or(u32::MAX);
        let divergent_operations = total_operations - equivalent_operations;
        let equivalence_rate = f64::from(equivalent_operations) / f64::from(total_operations);

        let mut operation_distribution = HashMap::new();
        for (op, _, _) in results {
            let _op_type = std::mem::discriminant(op);
            let op_name = match op {
                DifferentialOperation::ConfidenceAnd { .. } => "ConfidenceAnd",
                DifferentialOperation::ConfidenceOr { .. } => "ConfidenceOr",
                DifferentialOperation::ConfidenceNot { .. } => "ConfidenceNot",
                DifferentialOperation::ConfidenceCombineWeighted { .. } => {
                    "ConfidenceCombineWeighted"
                }
                DifferentialOperation::ConfidenceCalibration { .. } => "ConfidenceCalibration",
                DifferentialOperation::ConfidenceFromSuccesses { .. } => "ConfidenceFromSuccesses",
                DifferentialOperation::ConfidenceFromPercent { .. } => "ConfidenceFromPercent",
                DifferentialOperation::MemoryStore { .. } => "MemoryStore",
                DifferentialOperation::MemoryRecall { .. } => "MemoryRecall",
                DifferentialOperation::GraphSpreadingActivation { .. } => {
                    "GraphSpreadingActivation"
                }
            };
            *operation_distribution
                .entry(op_name.to_string())
                .or_insert(0) += 1;
        }

        TestSummary {
            total_operations,
            equivalent_operations,
            divergent_operations,
            equivalence_rate,
            operation_distribution,
        }
    }

    /// Build performance analysis
    fn build_performance_analysis(
        results: &[(
            DifferentialOperation,
            HashMap<Implementation, OperationResult>,
            bool,
        )],
        implementations: &[Implementation],
    ) -> PerformanceAnalysis {
        let mut per_implementation = HashMap::new();

        // Collect stats per implementation
        for impl_type in implementations {
            let mut execution_times = Vec::new();
            let mut success_count: u32 = 0;
            let mut total_count: u32 = 0;

            for (_, impl_results, _) in results {
                if let Some(result) = impl_results.get(impl_type) {
                    total_count += 1;
                    if result.success {
                        success_count += 1;
                        execution_times.push(nanos_to_f64(result.execution_time_nanos));
                    }
                }
            }

            let mean_time = if execution_times.is_empty() {
                0.0
            } else {
                let len_f64 = count_to_f64(execution_times.len());
                execution_times.iter().sum::<f64>() / len_f64
            };

            let std_dev = if execution_times.len() < 2 {
                0.0
            } else {
                let denominator = count_to_f64(execution_times.len().saturating_sub(1));
                let variance = execution_times
                    .iter()
                    .map(|t| (t - mean_time).powi(2))
                    .sum::<f64>()
                    / denominator;
                variance.sqrt()
            };

            let success_rate = if total_count == 0 {
                0.0
            } else {
                f64::from(success_count) / f64::from(total_count)
            };

            per_implementation.insert(
                *impl_type,
                PerformanceStats {
                    operations_count: total_count,
                    mean_execution_time_nanos: mean_time,
                    std_dev_execution_time_nanos: std_dev,
                    peak_memory_bytes: None,
                    success_rate,
                },
            );
        }

        PerformanceAnalysis {
            per_implementation,
            comparisons: Vec::new(), // Would build actual comparisons
            insights: vec!["Performance analysis complete".to_string()],
            optimization_suggestions: vec![
                "Consider profiling divergent operations for performance bottlenecks".to_string(),
                "Investigate high-variance operations for consistency improvements".to_string(),
            ],
        }
    }

    /// Extract divergences from results
    fn extract_divergences(
        results: Vec<(
            DifferentialOperation,
            HashMap<Implementation, OperationResult>,
            bool,
        )>,
    ) -> Vec<Divergence> {
        results
            .into_iter()
            .filter_map(|(op, impl_results, equiv)| {
                if equiv {
                    None
                } else {
                    Some(Divergence {
                        operation: op.clone(),
                        results: impl_results,
                        explanation: format!(
                            "Implementation divergence in {desc}",
                            desc = op.description()
                        ),
                        severity: DivergenceSeverity::Behavioral, // Would classify properly
                        investigation_steps: vec![
                            "Compare mathematical specifications".to_string(),
                            "Run bisection analysis".to_string(),
                        ],
                    })
                }
            })
            .collect()
    }

    /// Generate insights from the results
    fn generate_insights(
        summary: &TestSummary,
        performance: &PerformanceAnalysis,
        divergences: &[Divergence],
    ) -> Vec<Insight> {
        let mut insights = Vec::new();

        // Equivalence rate insights
        if summary.equivalence_rate >= 0.99 {
            insights.push(Insight {
                category: InsightCategory::Correctness,
                description: "Excellent implementation consistency - very few divergences found"
                    .to_string(),
                confidence: Confidence::HIGH,
                impact: InsightImpact::Low,
                suggested_actions: vec![
                    "Continue monitoring with regular differential testing".to_string(),
                ],
            });
        } else if summary.equivalence_rate < 0.90 {
            insights.push(Insight {
                category: InsightCategory::Correctness,
                description: "Significant implementation divergences detected - immediate investigation needed".to_string(),
                confidence: Confidence::HIGH,
                impact: InsightImpact::High,
                suggested_actions: vec![
                    "Run bisection analysis on all divergent operations".to_string(),
                    "Review mathematical specifications for consistency".to_string(),
                ],
            });
        }

        // Performance insights
        if performance.per_implementation.len() > 1 {
            let times: Vec<f64> = performance
                .per_implementation
                .values()
                .map(|stats| stats.mean_execution_time_nanos)
                .collect();

            if let (Some(&min_time), Some(&max_time)) = (
                times
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
                times
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
            ) && max_time > min_time * 2.0
            {
                insights.push(Insight {
                    category: InsightCategory::Performance,
                    description: format!(
                        "Significant performance variation detected ({:.0}x difference)",
                        max_time / min_time
                    ),
                    confidence: Confidence::HIGH,
                    impact: InsightImpact::Medium,
                    suggested_actions: vec![
                        "Profile slow implementations to identify bottlenecks".to_string(),
                    ],
                });
            }
        }

        // Divergence severity insights
        let high_severity_count = divergences
            .iter()
            .filter(|d| {
                matches!(
                    d.severity,
                    DivergenceSeverity::Correctness | DivergenceSeverity::Behavioral
                )
            })
            .count();

        if high_severity_count > 0 {
            insights.push(Insight {
                category: InsightCategory::Correctness,
                description: format!(
                    "{high_severity_count} high-severity divergences require immediate attention"
                ),
                confidence: Confidence::CERTAIN,
                impact: InsightImpact::Critical,
                suggested_actions: vec![
                    "Investigate high-severity divergences before production deployment"
                        .to_string(),
                ],
            });
        }

        insights
    }
}

/// Report output formats
#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    /// Plain text report
    Text,
    /// HTML report with styling
    Html,
}

fn count_to_f64(value: usize) -> f64 {
    let capped = u64::try_from(value).unwrap_or(u64::MAX);
    u64_to_f64(capped)
}

fn nanos_to_f64(nanos: u64) -> f64 {
    std::time::Duration::from_nanos(nanos).as_secs_f64() * 1_000_000_000.0
}

fn u64_to_f64(value: u64) -> f64 {
    const SHIFT: u32 = 32;
    const FACTOR: f64 = 4_294_967_296.0; // 2^32

    let high = value >> SHIFT;
    let low = value & 0xFFFF_FFFF;

    let high_part = u32::try_from(high).map_or_else(|_| f64::from(u32::MAX), f64::from);
    let low_part = u32::try_from(low).map_or_else(|_| f64::from(u32::MAX), f64::from);

    high_part.mul_add(FACTOR, low_part)
}

const fn insight_category_name(category: InsightCategory) -> &'static str {
    match category {
        InsightCategory::Correctness => "Correctness",
        InsightCategory::Performance => "Performance",
        InsightCategory::ErrorHandling => "Error Handling",
        InsightCategory::Precision => "Precision",
        InsightCategory::Complexity => "Complexity",
    }
}
