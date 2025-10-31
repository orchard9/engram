//! Invariant validators for chaos testing.
//!
//! This module provides validators to check system correctness properties
//! during and after chaos testing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Error type for consistency validation failures.
#[derive(Debug, Error)]
pub enum ValidationError {
    /// Data loss detected - observations accepted but not indexed
    #[error("Data loss detected: {missing_count} observations missing from index")]
    DataLoss {
        missing_count: usize,
        missing_ids: Vec<String>,
    },

    /// Timeout waiting for eventual consistency
    #[error("Eventual consistency timeout: {remaining} observations still missing after {elapsed:?}")]
    ConsistencyTimeout {
        remaining: usize,
        elapsed: Duration,
    },

    /// Sequence number violation detected
    #[error("Sequence violation: expected monotonic sequence, found gap or duplicate")]
    SequenceViolation {
        expected: u64,
        actual: u64,
    },

    /// HNSW graph integrity violation
    #[error("HNSW integrity violation: {reason}")]
    GraphIntegrityViolation {
        reason: String,
    },
}

/// Validator for eventual consistency guarantees.
///
/// Tracks observations that were acknowledged and verifies they eventually
/// become visible in the index within bounded staleness.
pub struct EventualConsistencyValidator {
    acked_observations: HashMap<String, ObservationRecord>,
    max_staleness: Duration,
}

impl EventualConsistencyValidator {
    /// Create a new eventual consistency validator.
    ///
    /// # Arguments
    /// * `max_staleness` - Maximum time to wait for observations to become visible
    #[must_use]
    pub fn new(max_staleness: Duration) -> Self {
        Self {
            acked_observations: HashMap::new(),
            max_staleness,
        }
    }

    /// Record an observation that was acknowledged by the server.
    pub fn record_ack(&mut self, id: String, sequence: u64, acked_at: Instant) {
        self.acked_observations.insert(
            id.clone(),
            ObservationRecord {
                id,
                sequence,
                acked_at,
            },
        );
    }

    /// Validate that all acknowledged observations are present in the recalled set.
    ///
    /// # Arguments
    /// * `recalled_ids` - Set of observation IDs that were successfully recalled
    ///
    /// # Returns
    /// `Ok(())` if all acked observations are present, `Err` with details otherwise
    pub fn validate_all_present(&self, recalled_ids: &HashSet<String>) -> Result<(), ValidationError> {
        let missing: Vec<String> = self
            .acked_observations
            .keys()
            .filter(|id| !recalled_ids.contains(*id))
            .cloned()
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(ValidationError::DataLoss {
                missing_count: missing.len(),
                missing_ids: missing,
            })
        }
    }

    /// Wait for all acknowledged observations to become visible with retries.
    ///
    /// # Arguments
    /// * `recall_fn` - Async function that performs a recall and returns observation IDs
    ///
    /// # Returns
    /// `Ok(())` when all observations are visible, or timeout error
    pub async fn wait_for_consistency<F, Fut>(
        &self,
        mut recall_fn: F,
    ) -> Result<(), ValidationError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = HashSet<String>>,
    {
        let start = Instant::now();
        let mut retry_interval = Duration::from_millis(10);

        loop {
            let recalled_ids = recall_fn().await;

            let missing: Vec<String> = self
                .acked_observations
                .keys()
                .filter(|id| !recalled_ids.contains(*id))
                .cloned()
                .collect();

            if missing.is_empty() {
                return Ok(());
            }

            if start.elapsed() > self.max_staleness {
                return Err(ValidationError::ConsistencyTimeout {
                    remaining: missing.len(),
                    elapsed: start.elapsed(),
                });
            }

            tokio::time::sleep(retry_interval).await;
            retry_interval = (retry_interval * 2).min(Duration::from_secs(1));
        }
    }

    /// Get count of tracked observations.
    #[must_use]
    pub fn tracked_count(&self) -> usize {
        self.acked_observations.len()
    }

    /// Clear all tracked observations.
    pub fn clear(&mut self) {
        self.acked_observations.clear();
    }
}

/// Record of an acknowledged observation.
#[derive(Debug, Clone)]
struct ObservationRecord {
    id: String,
    sequence: u64,
    acked_at: Instant,
}

/// Validator for sequence number monotonicity.
///
/// Ensures sequence numbers within a stream session are strictly monotonic
/// (no gaps, no duplicates, no reordering).
pub struct SequenceValidator {
    last_sequence: Option<u64>,
    gaps_detected: Vec<(u64, u64)>,
    duplicates_detected: Vec<u64>,
}

impl SequenceValidator {
    /// Create a new sequence validator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            last_sequence: None,
            gaps_detected: Vec::new(),
            duplicates_detected: Vec::new(),
        }
    }

    /// Validate next sequence number in stream.
    ///
    /// # Returns
    /// `Ok(())` if sequence is valid (monotonic), `Err` if violation detected
    pub fn validate_next(&mut self, sequence: u64) -> Result<(), ValidationError> {
        match self.last_sequence {
            None => {
                // First sequence number
                self.last_sequence = Some(sequence);
                Ok(())
            }
            Some(last) => {
                if sequence == last + 1 {
                    // Expected next sequence
                    self.last_sequence = Some(sequence);
                    Ok(())
                } else if sequence <= last {
                    // Duplicate or backwards
                    self.duplicates_detected.push(sequence);
                    Err(ValidationError::SequenceViolation {
                        expected: last + 1,
                        actual: sequence,
                    })
                } else {
                    // Gap detected
                    self.gaps_detected.push((last, sequence));
                    self.last_sequence = Some(sequence);
                    Err(ValidationError::SequenceViolation {
                        expected: last + 1,
                        actual: sequence,
                    })
                }
            }
        }
    }

    /// Get count of gaps detected.
    #[must_use]
    pub fn gap_count(&self) -> usize {
        self.gaps_detected.len()
    }

    /// Get count of duplicates detected.
    #[must_use]
    pub fn duplicate_count(&self) -> usize {
        self.duplicates_detected.len()
    }

    /// Reset validator state.
    pub fn reset(&mut self) {
        self.last_sequence = None;
        self.gaps_detected.clear();
        self.duplicates_detected.clear();
    }
}

impl Default for SequenceValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validator for HNSW graph structural integrity.
///
/// Checks graph invariants like bidirectional consistency and layer structure.
pub struct GraphIntegrityValidator;

impl GraphIntegrityValidator {
    /// Validate bidirectional edge consistency.
    ///
    /// For HNSW graphs, if node A has an edge to node B, then node B must have
    /// an edge back to node A.
    pub fn validate_bidirectional<T>(
        edges: &HashMap<usize, Vec<usize>>,
    ) -> Result<(), ValidationError> {
        for (node_id, neighbors) in edges {
            for neighbor_id in neighbors {
                if let Some(neighbor_edges) = edges.get(neighbor_id) {
                    if !neighbor_edges.contains(node_id) {
                        return Err(ValidationError::GraphIntegrityViolation {
                            reason: format!(
                                "Node {} has edge to {}, but reverse edge missing",
                                node_id, neighbor_id
                            ),
                        });
                    }
                } else {
                    return Err(ValidationError::GraphIntegrityViolation {
                        reason: format!(
                            "Node {} has edge to {}, but target node not found",
                            node_id, neighbor_id
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Validate layer structure (higher layers are subsets of lower layers).
    pub fn validate_layer_hierarchy(
        layer_nodes: &[HashSet<usize>],
    ) -> Result<(), ValidationError> {
        for i in 1..layer_nodes.len() {
            let upper_layer = &layer_nodes[i];
            let lower_layer = &layer_nodes[i - 1];

            for node_id in upper_layer {
                if !lower_layer.contains(node_id) {
                    return Err(ValidationError::GraphIntegrityViolation {
                        reason: format!(
                            "Node {} in layer {} but not in layer {}",
                            node_id,
                            i,
                            i - 1
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

/// Aggregate statistics from chaos testing.
#[derive(Debug, Default, Clone)]
pub struct ChaosTestStats {
    /// Total observations sent
    pub total_sent: u64,
    /// Total observations acknowledged
    pub total_acked: u64,
    /// Total observations rejected (admission control)
    pub total_rejected: u64,
    /// Total observations recalled successfully
    pub total_recalled: u64,
    /// Sequence violations detected
    pub sequence_violations: u64,
    /// Graph integrity checks passed
    pub integrity_checks_passed: u64,
    /// Graph integrity checks failed
    pub integrity_checks_failed: u64,
    /// Test duration
    pub duration: Duration,
}

impl ChaosTestStats {
    /// Create new stats tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate data loss count.
    #[must_use]
    pub fn data_loss(&self) -> i64 {
        self.total_acked as i64 - self.total_recalled as i64
    }

    /// Calculate effective success rate (acked / sent).
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total_sent > 0 {
            self.total_acked as f64 / self.total_sent as f64
        } else {
            0.0
        }
    }

    /// Calculate rejection rate (rejected / sent).
    #[must_use]
    pub fn rejection_rate(&self) -> f64 {
        if self.total_sent > 0 {
            self.total_rejected as f64 / self.total_sent as f64
        } else {
            0.0
        }
    }

    /// Print summary report.
    pub fn print_report(&self) {
        println!("\n=== Chaos Test Report ===");
        println!("Duration: {:?}", self.duration);
        println!("Observations sent: {}", self.total_sent);
        println!("Observations acked: {}", self.total_acked);
        println!("Observations rejected: {}", self.total_rejected);
        println!("Observations recalled: {}", self.total_recalled);
        println!("Success rate: {:.2}%", self.success_rate() * 100.0);
        println!("Rejection rate: {:.2}%", self.rejection_rate() * 100.0);
        println!("Data loss: {}", self.data_loss());
        println!("Sequence violations: {}", self.sequence_violations);
        println!("Integrity checks passed: {}", self.integrity_checks_passed);
        println!("Integrity checks failed: {}", self.integrity_checks_failed);
        println!("========================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eventual_consistency_validator_tracks_observations() {
        let mut validator = EventualConsistencyValidator::new(Duration::from_secs(1));
        assert_eq!(validator.tracked_count(), 0);

        validator.record_ack("obs1".to_string(), 1, Instant::now());
        validator.record_ack("obs2".to_string(), 2, Instant::now());
        assert_eq!(validator.tracked_count(), 2);

        validator.clear();
        assert_eq!(validator.tracked_count(), 0);
    }

    #[test]
    fn eventual_consistency_validator_detects_missing() {
        let mut validator = EventualConsistencyValidator::new(Duration::from_secs(1));
        validator.record_ack("obs1".to_string(), 1, Instant::now());
        validator.record_ack("obs2".to_string(), 2, Instant::now());
        validator.record_ack("obs3".to_string(), 3, Instant::now());

        let mut recalled = HashSet::new();
        recalled.insert("obs1".to_string());
        recalled.insert("obs3".to_string());

        let result = validator.validate_all_present(&recalled);
        assert!(result.is_err());

        if let Err(ValidationError::DataLoss { missing_count, missing_ids }) = result {
            assert_eq!(missing_count, 1);
            assert!(missing_ids.contains(&"obs2".to_string()));
        } else {
            panic!("Expected DataLoss error");
        }
    }

    #[test]
    fn sequence_validator_accepts_monotonic() {
        let mut validator = SequenceValidator::new();
        assert!(validator.validate_next(1).is_ok());
        assert!(validator.validate_next(2).is_ok());
        assert!(validator.validate_next(3).is_ok());
        assert_eq!(validator.gap_count(), 0);
        assert_eq!(validator.duplicate_count(), 0);
    }

    #[test]
    fn sequence_validator_detects_gaps() {
        let mut validator = SequenceValidator::new();
        assert!(validator.validate_next(1).is_ok());
        assert!(validator.validate_next(5).is_err()); // Gap from 1 to 5
        assert_eq!(validator.gap_count(), 1);
    }

    #[test]
    fn sequence_validator_detects_duplicates() {
        let mut validator = SequenceValidator::new();
        assert!(validator.validate_next(1).is_ok());
        assert!(validator.validate_next(2).is_ok());
        assert!(validator.validate_next(2).is_err()); // Duplicate
        assert_eq!(validator.duplicate_count(), 1);
    }

    #[test]
    fn graph_integrity_validates_bidirectional() {
        let mut edges = HashMap::new();
        edges.insert(1, vec![2, 3]);
        edges.insert(2, vec![1]);
        edges.insert(3, vec![1]);

        assert!(GraphIntegrityValidator::validate_bidirectional(&edges).is_ok());
    }

    #[test]
    fn graph_integrity_detects_missing_reverse_edge() {
        let mut edges = HashMap::new();
        edges.insert(1, vec![2]);
        edges.insert(2, vec![]); // Missing reverse edge to 1

        let result = GraphIntegrityValidator::validate_bidirectional(&edges);
        assert!(result.is_err());
    }

    #[test]
    fn chaos_stats_calculations() {
        let stats = ChaosTestStats {
            total_sent: 1000,
            total_acked: 950,
            total_rejected: 50,
            total_recalled: 950,
            ..Default::default()
        };

        assert_eq!(stats.data_loss(), 0);
        assert_eq!(stats.success_rate(), 0.95);
        assert_eq!(stats.rejection_rate(), 0.05);
    }
}
