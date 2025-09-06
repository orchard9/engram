//! CLI tool for reviewing error messages
//!
//! This tool scans the codebase for error definitions and validates they meet
//! cognitive ergonomics requirements.

use anyhow::Result;
use clap::{Parser, Subcommand};
use engram_core::error_review::{ErrorReviewConfig, ErrorReviewer, Severity};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "error-reviewer")]
#[command(about = "Review error messages for cognitive ergonomics")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Review all errors in a directory
    Review {
        /// Directory to scan for errors
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Output format
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Fail if any errors don't meet requirements
        #[arg(long)]
        strict: bool,
    },

    /// Generate error documentation
    Catalog {
        /// Directory to scan
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Output file
        #[arg(short, long, default_value = "error-catalog.md")]
        output: PathBuf,
    },

    /// Audit error messages for quality
    Audit {
        /// Directory to scan
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Maximum cognitive load (1-10)
        #[arg(long, default_value = "7")]
        max_load: u8,

        /// Minimum confidence threshold
        #[arg(long, default_value = "0.5")]
        min_confidence: f32,
    },
}

#[derive(Clone, Copy, Debug)]
enum OutputFormat {
    Text,
    Json,
    Markdown,
    Html,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(OutputFormat::Text),
            "json" => Ok(OutputFormat::Json),
            "markdown" | "md" => Ok(OutputFormat::Markdown),
            "html" => Ok(OutputFormat::Html),
            _ => Err(format!("Unknown format: {}", s)),
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Review {
            path,
            format,
            verbose,
            strict,
        } => {
            review_errors(path, format, verbose, strict)?;
        }
        Commands::Catalog { path, output } => {
            generate_catalog(path, output)?;
        }
        Commands::Audit {
            path,
            max_load,
            min_confidence,
        } => {
            audit_errors(path, max_load, min_confidence)?;
        }
    }

    Ok(())
}

fn review_errors(path: PathBuf, format: OutputFormat, verbose: bool, strict: bool) -> Result<()> {
    println!("🔍 Reviewing error messages in {:?}", path);

    let mut config = ErrorReviewConfig::default();
    config.verbose = verbose;

    let mut reviewer = ErrorReviewer::new(config);
    let report = reviewer
        .review_directory(&path)
        .map_err(|e| anyhow::anyhow!(e))?;

    // Output results in requested format
    match format {
        OutputFormat::Text => {
            print_text_report(&report);
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&report)?;
            println!("{}", json);
        }
        OutputFormat::Markdown => {
            println!("{}", report.to_markdown());
        }
        OutputFormat::Html => {
            println!("{}", report.to_html());
        }
    }

    // Exit with error if strict mode and failures found
    if strict && report.failing_errors > 0 {
        eprintln!("\n❌ {} errors failed review", report.failing_errors);
        std::process::exit(1);
    }

    Ok(())
}

fn generate_catalog(path: PathBuf, output: PathBuf) -> Result<()> {
    println!("📚 Generating error catalog from {:?}", path);

    let mut reviewer = ErrorReviewer::new(ErrorReviewConfig::default());
    let report = reviewer
        .review_directory(&path)
        .map_err(|e| anyhow::anyhow!(e))?;

    let mut catalog = String::new();
    catalog.push_str("# Engram Error Catalog\n\n");
    catalog.push_str(&format!(
        "Generated: {}\n\n",
        chrono::Utc::now().to_rfc3339()
    ));
    catalog.push_str(&format!("Total Errors: {}\n\n", report.total_errors));

    // Group errors by file
    let mut by_file: std::collections::HashMap<String, Vec<_>> = std::collections::HashMap::new();
    for (id, error_def) in &report.error_catalog {
        by_file
            .entry(error_def.file_path.clone())
            .or_insert_with(Vec::new)
            .push(error_def);
    }

    for (file, errors) in by_file {
        catalog.push_str(&format!("## {}\n\n", file));

        for error in errors {
            catalog.push_str(&format!(
                "### Line {}: {}\n\n",
                error.line_number, error.summary
            ));

            if let Some((expected, actual)) = &error.context {
                catalog.push_str(&format!(
                    "**Context**: Expected {}, got {}\n\n",
                    expected, actual
                ));
            }

            if let Some(suggestion) = &error.suggestion {
                catalog.push_str(&format!("**Suggestion**: {}\n\n", suggestion));
            }

            if let Some(example) = &error.example {
                catalog.push_str(&format!("**Example**:\n```rust\n{}\n```\n\n", example));
            }

            if let Some(confidence) = error.confidence {
                catalog.push_str(&format!("**Confidence**: {:.1}\n\n", confidence));
            }

            catalog.push_str("---\n\n");
        }
    }

    fs::write(&output, catalog)?;
    println!("✅ Catalog written to {:?}", output);

    Ok(())
}

fn audit_errors(path: PathBuf, max_load: u8, min_confidence: f32) -> Result<()> {
    println!("🔍 Auditing error messages in {:?}", path);
    println!("  Max cognitive load: {}/10", max_load);
    println!("  Min confidence: {:.1}", min_confidence);
    println!();

    let mut config = ErrorReviewConfig::default();
    config.max_cognitive_load = max_load;
    config.min_confidence = min_confidence;

    let mut reviewer = ErrorReviewer::new(config);
    let report = reviewer
        .review_directory(&path)
        .map_err(|e| anyhow::anyhow!(e))?;

    // Print audit summary
    println!("## Audit Results\n");
    println!("Total Errors: {}", report.total_errors);
    println!("Pass Rate: {:.1}%", report.pass_rate);
    println!();

    // Show common issues
    if !report.common_issues.is_empty() {
        println!("## Common Issues\n");
        for (issue_type, count) in &report.common_issues {
            println!("- {:?}: {} occurrences", issue_type, count);
        }
        println!();
    }

    // Show errors by severity
    if let Some(errors) = report.errors_by_severity.get(&Severity::Error) {
        println!("## Critical Issues ({} errors)\n", errors.len());
        for error in errors.iter().take(5) {
            println!(
                "- {} (line {}): {}",
                error.file_path,
                error.line_number,
                error
                    .quality_issues
                    .first()
                    .map(|i| i.description.as_str())
                    .unwrap_or("Unknown issue")
            );
        }
        if errors.len() > 5 {
            println!("  ... and {} more", errors.len() - 5);
        }
        println!();
    }

    // Recommendations
    println!("## Recommendations\n");

    let mut recommendations = vec![];

    if report.pass_rate < 80.0 {
        recommendations.push("Improve error message quality - many errors lack required fields");
    }

    if report
        .common_issues
        .iter()
        .any(|(t, _)| matches!(t, engram_core::error_review::IssueType::VagueSuggestion))
    {
        recommendations.push("Replace vague suggestions with specific, actionable steps");
    }

    if report
        .common_issues
        .iter()
        .any(|(t, _)| matches!(t, engram_core::error_review::IssueType::PoorExample))
    {
        recommendations.push("Expand code examples to show complete, runnable code");
    }

    if report
        .common_issues
        .iter()
        .any(|(t, _)| matches!(t, engram_core::error_review::IssueType::HighCognitiveLoad))
    {
        recommendations.push("Simplify complex error messages to reduce cognitive load");
    }

    if recommendations.is_empty() {
        println!("✅ Error messages meet quality standards!");
    } else {
        for rec in recommendations {
            println!("- {}", rec);
        }
    }

    Ok(())
}

fn print_text_report(report: &engram_core::error_review::ErrorReviewReport) {
    println!("\n📊 Error Review Report");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Total Errors: {}", report.total_errors);
    println!(
        "Passing: {} ({:.1}%)",
        report.passing_errors, report.pass_rate
    );
    println!("Failing: {}", report.failing_errors);
    println!();

    if !report.common_issues.is_empty() {
        println!("Common Issues:");
        for (issue_type, count) in &report.common_issues {
            println!("  {:?}: {} occurrences", issue_type, count);
        }
        println!();
    }

    // Show failed errors
    let failed: Vec<_> = report
        .detailed_results
        .iter()
        .filter(|r| !r.passes_review)
        .collect();

    if !failed.is_empty() {
        println!("Failed Errors:");
        for result in failed.iter().take(10) {
            println!("\n  {} (line {})", result.file_path, result.line_number);
            println!("  Summary: {}", result.error_summary);

            if !result.missing_fields.is_empty() {
                println!("  Missing: {}", result.missing_fields.join(", "));
            }

            for issue in &result.quality_issues {
                println!("  [{:?}] {}", issue.severity, issue.description);
            }
        }

        if failed.len() > 10 {
            println!("\n  ... and {} more failed errors", failed.len() - 10);
        }
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if report.pass_rate >= 100.0 {
        println!("✅ All errors pass review!");
    } else if report.pass_rate >= 80.0 {
        println!("⚠️  Most errors pass, but improvements needed");
    } else {
        println!("❌ Many errors need improvement");
    }
}
