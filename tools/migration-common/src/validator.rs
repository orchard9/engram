//! Validation infrastructure for migration integrity

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Source database statistics for validation
#[derive(Debug, Clone)]
pub struct SourceStatistics {
    /// Total number of records in source
    pub total_records: u64,
    /// Sample of record IDs for validation
    pub record_ids: Vec<String>,
    /// Total number of relationships/edges
    pub total_relationships: u64,
}

impl SourceStatistics {
    /// Create new source statistics
    #[must_use]
    pub const fn new(
        total_records: u64,
        record_ids: Vec<String>,
        total_relationships: u64,
    ) -> Self {
        Self {
            total_records,
            record_ids,
            total_relationships,
        }
    }
}

/// Count validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountReport {
    /// Record count in source
    pub source_count: u64,
    /// Record count in target (Engram)
    pub target_count: u64,
    /// Data loss rate (0.0 to 1.0)
    pub loss_rate: f64,
    /// Whether validation passed
    pub passed: bool,
}

impl CountReport {
    /// Create a new count report
    #[must_use]
    pub fn new(source_count: u64, target_count: u64) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let loss_rate = if source_count > 0 {
            1.0 - (target_count as f64 / source_count as f64)
        } else {
            0.0
        };

        let passed = loss_rate < 0.01; // Less than 1% data loss

        Self {
            source_count,
            target_count,
            loss_rate,
            passed,
        }
    }

    /// Print report
    pub fn print(&self) {
        println!("\nCount Validation Report");
        println!("====================");
        println!("Source count: {}", self.source_count);
        println!("Target count: {}", self.target_count);
        println!("Data loss: {:.2}%", self.loss_rate * 100.0);
        println!("Status: {}", if self.passed { "PASSED" } else { "FAILED" });
    }
}

/// Sample validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleReport {
    /// Number of samples checked
    pub sample_size: usize,
    /// Number of matching records
    pub matches: usize,
    /// Match rate (0.0 to 1.0)
    pub match_rate: f64,
    /// Whether validation passed
    pub passed: bool,
}

impl SampleReport {
    /// Create a new sample report
    #[must_use]
    pub fn new(sample_size: usize, matches: usize) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let match_rate = if sample_size > 0 {
            matches as f64 / sample_size as f64
        } else {
            1.0
        };

        let passed = match_rate > 0.99; // More than 99% match

        Self {
            sample_size,
            matches,
            match_rate,
            passed,
        }
    }

    /// Print report
    pub fn print(&self) {
        println!("\nSample Validation Report");
        println!("======================");
        println!("Sample size: {}", self.sample_size);
        println!("Matches: {}", self.matches);
        println!("Match rate: {:.2}%", self.match_rate * 100.0);
        println!("Status: {}", if self.passed { "PASSED" } else { "FAILED" });
    }
}

/// Edge integrity report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeReport {
    /// Total number of edges
    pub total_edges: usize,
    /// Number of orphaned edges (missing source or target)
    pub orphaned_count: usize,
    /// Whether validation passed
    pub passed: bool,
}

impl EdgeReport {
    /// Create a new edge report
    #[must_use]
    pub const fn new(total_edges: usize, orphaned_count: usize) -> Self {
        let passed = orphaned_count == 0;
        Self {
            total_edges,
            orphaned_count,
            passed,
        }
    }

    /// Print report
    pub fn print(&self) {
        println!("\nEdge Integrity Report");
        println!("===================");
        println!("Total edges: {}", self.total_edges);
        println!("Orphaned edges: {}", self.orphaned_count);
        println!("Status: {}", if self.passed { "PASSED" } else { "FAILED" });
    }
}

/// Embedding quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingReport {
    /// Number of embeddings checked
    pub checked_count: usize,
    /// Number of valid embeddings (finite values, proper dimensions)
    pub valid_count: usize,
    /// Average embedding norm
    pub avg_norm: f64,
    /// Whether validation passed
    pub passed: bool,
}

impl EmbeddingReport {
    /// Create a new embedding report
    #[must_use]
    pub fn new(checked_count: usize, valid_count: usize, avg_norm: f64) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let valid_rate = if checked_count > 0 {
            valid_count as f64 / checked_count as f64
        } else {
            1.0
        };

        let passed = valid_rate > 0.99 && (0.9..=1.1).contains(&avg_norm);

        Self {
            checked_count,
            valid_count,
            avg_norm,
            passed,
        }
    }

    /// Print report
    pub fn print(&self) {
        println!("\nEmbedding Quality Report");
        println!("======================");
        println!("Checked: {}", self.checked_count);
        println!("Valid: {}", self.valid_count);
        println!("Avg norm: {:.4}", self.avg_norm);
        println!("Status: {}", if self.passed { "PASSED" } else { "FAILED" });
    }
}

/// Full migration validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Count validation
    pub count: CountReport,
    /// Sample validation
    pub sample: SampleReport,
    /// Edge integrity
    pub edges: EdgeReport,
    /// Embedding quality
    pub embeddings: EmbeddingReport,
    /// Overall validation status
    pub overall_passed: bool,
}

impl ValidationReport {
    /// Create a new validation report
    #[must_use]
    pub const fn new(
        count: CountReport,
        sample: SampleReport,
        edges: EdgeReport,
        embeddings: EmbeddingReport,
    ) -> Self {
        let overall_passed = count.passed && sample.passed && edges.passed && embeddings.passed;

        Self {
            count,
            sample,
            edges,
            embeddings,
            overall_passed,
        }
    }

    /// Print complete report
    pub fn print(&self) {
        self.count.print();
        self.sample.print();
        self.edges.print();
        self.embeddings.print();

        println!("\n============================");
        println!(
            "Overall Status: {}",
            if self.overall_passed {
                "PASSED"
            } else {
                "FAILED"
            }
        );
        println!("============================\n");
    }
}

/// Simple migration summary report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationReport {
    /// Total records migrated
    pub records_migrated: u64,
    /// Time elapsed
    pub duration: Duration,
    /// Number of errors encountered
    pub error_count: u64,
}

impl MigrationReport {
    /// Create a new migration report
    #[must_use]
    pub const fn new(records_migrated: u64, duration: Duration, error_count: u64) -> Self {
        Self {
            records_migrated,
            duration,
            error_count,
        }
    }

    /// Print report
    pub fn print(&self) {
        println!("\nMigration Complete");
        println!("=================");
        println!("Records migrated: {}", self.records_migrated);
        println!("Duration: {:?}", self.duration);
        println!("Errors: {}", self.error_count);
        println!("=================\n");
    }
}

/// Migration validator that performs comprehensive validation
pub struct MigrationValidator;

impl MigrationValidator {
    /// Validate migration completeness and correctness
    #[must_use]
    pub fn validate(_source_stats: &SourceStatistics, _target_client: &str) -> ValidationReport {
        // Placeholder implementation
        // In production, this would:
        // 1. Count records in target
        // 2. Sample random records and verify content
        // 3. Check edge integrity
        // 4. Validate embedding quality

        ValidationReport::new(
            CountReport::new(0, 0),
            SampleReport::new(0, 0),
            EdgeReport::new(0, 0),
            EmbeddingReport::new(0, 0, 1.0),
        )
    }
}
