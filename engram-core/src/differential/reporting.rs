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
use std::fmt;
use std::io::Write;

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
        implementations: Vec<Implementation>,
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
            implementations: implementations.clone(),
            reference_implementation: reference,
            duration,
            config_summary: "Default configuration".to_string(),
        };

        let summary = Self::build_summary(&results);
        let performance_analysis = Self::build_performance_analysis(&results, &implementations);
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

        // Header
        report.push_str("🔬 Differential Testing Report\n");
        report.push_str("==============================\n\n");

        // Session info
        report.push_str(&format!(
            "📅 Test Date: {}\n",
            self.session_info.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        report.push_str(&format!(
            "⏱️  Duration: {:.2}s\n",
            self.session_info.duration.as_secs_f64()
        ));
        report.push_str(&format!(
            "🏗️  Implementations: {}\n",
            self.session_info
                .implementations
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
        report.push_str(&format!(
            "📖 Reference: {}\n\n",
            self.session_info.reference_implementation
        ));

        // Summary
        report.push_str("📊 Summary\n");
        report.push_str("----------\n");
        report.push_str(&format!(
            "Total Operations: {}\n",
            self.summary.total_operations
        ));
        report.push_str(&format!(
            "✅ Equivalent: {} ({:.1}%)\n",
            self.summary.equivalent_operations,
            self.summary.equivalence_rate * 100.0
        ));
        report.push_str(&format!(
            "❌ Divergent: {} ({:.1}%)\n\n",
            self.summary.divergent_operations,
            self.summary.equivalence_rate.mul_add(-100.0, 100.0)
        ));

        // Performance analysis
        if !self.performance_analysis.per_implementation.is_empty() {
            report.push_str("⚡ Performance Analysis\n");
            report.push_str("----------------------\n");
            for (impl_type, stats) in &self.performance_analysis.per_implementation {
                report.push_str(&format!(
                    "{}: {:.0}ns avg, {:.1}% success rate\n",
                    impl_type,
                    stats.mean_execution_time_nanos,
                    stats.success_rate * 100.0
                ));
            }
            report.push('\n');
        }

        // Insights
        if !self.insights.is_empty() {
            report.push_str("💡 Key Insights\n");
            report.push_str("---------------\n");

            // Sort insights by impact
            let mut sorted_insights = self.insights.clone();
            sorted_insights.sort_by(|a, b| b.impact.cmp(&a.impact));

            for insight in sorted_insights.iter().take(5) {
                // Top 5 insights
                let impact_emoji = match insight.impact {
                    InsightImpact::Critical => "🚨",
                    InsightImpact::High => "⚠️",
                    InsightImpact::Medium => "⚡",
                    InsightImpact::Low => "ℹ️",
                };
                report.push_str(&format!(
                    "{} [{}] {}\n",
                    impact_emoji, insight.impact, insight.description
                ));
                if !insight.suggested_actions.is_empty() {
                    report.push_str(&format!("   Action: {}\n", insight.suggested_actions[0]));
                }
            }
            report.push('\n');
        }

        // Divergences (if any)
        if !self.divergences.is_empty() {
            report.push_str("🔍 Implementation Divergences\n");
            report.push_str("-----------------------------\n");

            for (i, divergence) in self.divergences.iter().enumerate().take(3) {
                // Top 3 divergences
                report.push_str(&format!(
                    "{}. {} [{}]\n",
                    i + 1,
                    divergence.operation.description(),
                    divergence.severity
                ));
                report.push_str(&format!("   {}\n", divergence.explanation));
                if !divergence.investigation_steps.is_empty() {
                    report.push_str(&format!("   Next: {}\n", divergence.investigation_steps[0]));
                }
                report.push('\n');
            }

            if self.divergences.len() > 3 {
                report.push_str(&format!(
                    "... and {} more divergences\n\n",
                    self.divergences.len() - 3
                ));
            }
        }

        // Recommendations
        report.push_str("🎯 Recommendations\n");
        report.push_str("------------------\n");

        if self.summary.equivalence_rate >= 0.99 {
            report.push_str(
                "✅ Excellent equivalence rate! Implementations are highly consistent.\n",
            );
        } else if self.summary.equivalence_rate >= 0.95 {
            report
                .push_str("✅ Good equivalence rate. Minor divergences should be investigated.\n");
        } else if self.summary.equivalence_rate >= 0.90 {
            report.push_str("⚠️ Moderate equivalence rate. Several divergences need attention.\n");
        } else {
            report.push_str(
                "🚨 Low equivalence rate. Significant implementation differences detected.\n",
            );
        }

        // Add specific recommendations based on performance
        for suggestion in self
            .performance_analysis
            .optimization_suggestions
            .iter()
            .take(3)
        {
            report.push_str(&format!("• {suggestion}\n"));
        }

        report.push_str(
            "\n📚 For detailed analysis, run bisection on specific divergent operations.\n",
        );

        report
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
        html.push_str(&format!(
            r#"
    <div class="header">
        <h1>🔬 Differential Testing Report</h1>
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
        ));

        // Summary metrics
        html.push_str(r#"<div class="section"><h2>📊 Summary</h2><div>"#);
        html.push_str(&format!(
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
            (self.summary.equivalence_rate * 100.0) as u32,
            self.summary.divergent_operations
        ));
        html.push_str("</div></div>");

        // Insights
        html.push_str(r#"<div class="section"><h2>💡 Key Insights</h2>"#);
        for insight in &self.insights {
            let impact_class = match insight.impact {
                InsightImpact::Critical | InsightImpact::High => "high-impact",
                InsightImpact::Medium => "medium-impact",
                InsightImpact::Low => "low-impact",
            };
            html.push_str(&format!(
                r#"<div class="insight {}">
                <strong>[{}]</strong> {}
            </div>"#,
                impact_class, insight.impact, insight.description
            ));
        }
        html.push_str("</div>");

        // Divergences
        if !self.divergences.is_empty() {
            html.push_str(r#"<div class="section"><h2>🔍 Implementation Divergences</h2>"#);
            for divergence in &self.divergences {
                html.push_str(&format!(
                    r#"<div class="divergence">
                    <h4>{} [{}]</h4>
                    <p>{}</p>
                </div>"#,
                    divergence.operation.description(),
                    divergence.severity,
                    divergence.explanation
                ));
            }
            html.push_str("</div>");
        }

        html.push_str("</body></html>");
        html
    }

    /// Save report to file
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
        let total_operations = results.len() as u32;
        let equivalent_operations = results.iter().filter(|(_, _, equiv)| *equiv).count() as u32;
        let divergent_operations = total_operations - equivalent_operations;
        let equivalence_rate = f64::from(equivalent_operations) / f64::from(total_operations);

        let mut operation_distribution = HashMap::new();
        for (op, _, _) in results {
            let op_type = std::mem::discriminant(op);
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
            let mut success_count = 0;
            let mut total_count = 0;

            for (_, impl_results, _) in results {
                if let Some(result) = impl_results.get(impl_type) {
                    total_count += 1;
                    if result.success {
                        success_count += 1;
                        execution_times.push(result.execution_time_nanos as f64);
                    }
                }
            }

            let mean_time = if execution_times.is_empty() {
                0.0
            } else {
                execution_times.iter().sum::<f64>() / execution_times.len() as f64
            };

            let std_dev = if execution_times.len() < 2 {
                0.0
            } else {
                let variance = execution_times
                    .iter()
                    .map(|t| (t - mean_time).powi(2))
                    .sum::<f64>()
                    / (execution_times.len() - 1) as f64;
                variance.sqrt()
            };

            per_implementation.insert(
                *impl_type,
                PerformanceStats {
                    operations_count: total_count,
                    mean_execution_time_nanos: mean_time,
                    std_dev_execution_time_nanos: std_dev,
                    peak_memory_bytes: None,
                    success_rate: f64::from(success_count) / f64::from(total_count),
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
                        explanation: format!("Implementation divergence in {}", op.description()),
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
                times.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
                times.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
            ) {
                if max_time > min_time * 2.0 {
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
