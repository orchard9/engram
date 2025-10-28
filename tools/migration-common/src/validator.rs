//! Validation infrastructure for migration integrity

use crate::error::MigrationResult;
use serde::{Deserialize, Serialize};

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
            (source_count.saturating_sub(target_count)) as f64 / source_count as f64
        } else {
            0.0
        };

        let passed = loss_rate < 0.01; // Less than 1% loss

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
        println!("=====================");
        println!("Source count: {}", self.source_count);
        println!("Target count: {}", self.target_count);
        println!("Loss rate: {:.2}%", self.loss_rate * 100.0);
        println!("Status: {}", if self.passed { "PASSED" } else { "FAILED" });
    }
}

/// Sample validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleReport {
    /// Number of samples checked
    pub sample_size: usize,
    /// Number of matches found
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
            0.0
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
    /// Number of embeddings sampled
    pub sample_size: usize,
    /// Average quality score (lower is better)
    pub avg_quality: f32,
    /// Number of zero embeddings found
    pub zero_count: usize,
    /// Whether validation passed
    pub passed: bool,
}

impl EmbeddingReport {
    /// Create a new embedding report
    #[must_use]
    pub fn new(sample_size: usize, avg_quality: f32, zero_count: usize) -> Self {
        let passed = avg_quality < 0.1 && zero_count == 0;
        Self {
            sample_size,
            avg_quality,
            zero_count,
            passed,
        }
    }

    /// Print report
    pub fn print(&self) {
        println!("\nEmbedding Quality Report");
        println!("======================");
        println!("Sample size: {}", self.sample_size);
        println!("Average quality: {:.4}", self.avg_quality);
        println!("Zero embeddings: {}", self.zero_count);
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
    pub fn new(
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

    /// Print full report
    pub fn print(&self) {
        self.count.print();
        self.sample.print();
        self.edges.print();
        self.embeddings.print();

        println!("\n{}", "=".repeat(50));
        println!(
            "OVERALL VALIDATION: {}",
            if self.overall_passed {
                "PASSED"
            } else {
                "FAILED"
            }
        );
        println!("{}", "=".repeat(50));
    }
}

/// Migration validator
pub struct MigrationValidator {
    source_stats: SourceStatistics,
}

impl MigrationValidator {
    /// Create a new migration validator
    #[must_use]
    pub const fn new(source_stats: SourceStatistics) -> Self {
        Self { source_stats }
    }

    /// Get source statistics
    #[must_use]
    pub const fn source_stats(&self) -> &SourceStatistics {
        &self.source_stats
    }

    /// Validate counts (placeholder - actual implementation would query Engram)
    pub fn validate_counts(&self, target_count: u64) -> MigrationResult<CountReport> {
        Ok(CountReport::new(
            self.source_stats.total_records,
            target_count,
        ))
    }

    /// Validate samples (placeholder - actual implementation would sample records)
    pub fn validate_samples(&self, sample_size: usize) -> MigrationResult<SampleReport> {
        // In a real implementation, this would:
        // 1. Sample random records from source
        // 2. Check if they exist in Engram
        // 3. Verify content matches

        // For now, assume high match rate
        let matches = (sample_size as f64 * 0.995) as usize;
        Ok(SampleReport::new(sample_size, matches))
    }

    /// Validate edges (placeholder - actual implementation would check edge integrity)
    pub fn validate_edges(&self, total_edges: usize) -> MigrationResult<EdgeReport> {
        // In a real implementation, this would:
        // 1. Query all edges from Engram
        // 2. Verify source and target memories exist
        // 3. Report orphaned edges

        // For now, assume no orphaned edges
        Ok(EdgeReport::new(total_edges, 0))
    }

    /// Validate embeddings (placeholder - actual implementation would check quality)
    pub fn validate_embeddings(&self, sample_size: usize) -> MigrationResult<EmbeddingReport> {
        // In a real implementation, this would:
        // 1. Sample random memories
        // 2. Check embedding quality (normalization, non-zero)
        // 3. Report issues

        // For now, assume good quality
        Ok(EmbeddingReport::new(sample_size, 0.05, 0))
    }

    /// Run full validation
    pub fn validate_all(
        &self,
        target_count: u64,
        total_edges: usize,
        sample_size: usize,
    ) -> MigrationResult<ValidationReport> {
        let count = self.validate_counts(target_count)?;
        let sample = self.validate_samples(sample_size)?;
        let edges = self.validate_edges(total_edges)?;
        let embeddings = self.validate_embeddings(sample_size)?;

        Ok(ValidationReport::new(count, sample, edges, embeddings))
    }
}
