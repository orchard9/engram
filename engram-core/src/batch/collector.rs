//! Lock-free result collection and aggregation

use crate::batch::{BatchOperationResult, BatchResult};
use crossbeam_queue::SegQueue;
use std::convert::TryFrom;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Lock-free atomic result collector
pub struct AtomicResultCollector {
    /// Queue of results
    results: SegQueue<BatchResult>,
    /// Successful operations counter
    successful_ops: AtomicUsize,
    /// Failed operations counter
    failed_ops: AtomicUsize,
    /// Total processing time in microseconds
    total_latency_us: AtomicU64,
    /// SIMD operations counter
    simd_ops: AtomicUsize,
}

impl AtomicResultCollector {
    /// Create a new result collector
    #[must_use]
    pub const fn new() -> Self {
        Self {
            results: SegQueue::new(),
            successful_ops: AtomicUsize::new(0),
            failed_ops: AtomicUsize::new(0),
            total_latency_us: AtomicU64::new(0),
            simd_ops: AtomicUsize::new(0),
        }
    }

    /// Add a result to the collector
    pub fn add_result(&self, result: BatchResult) {
        // Update metrics
        self.total_latency_us
            .fetch_add(result.metadata.processing_time_us, Ordering::Relaxed);

        if result.metadata.simd_used {
            self.simd_ops.fetch_add(1, Ordering::Relaxed);
        }

        // Track success/failure
        match &result.result {
            BatchOperationResult::Store { activation, .. } => {
                if activation.is_successful() {
                    self.successful_ops.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.failed_ops.fetch_add(1, Ordering::Relaxed);
                }
            }
            BatchOperationResult::Recall(episodes) => {
                if episodes.is_empty() {
                    self.failed_ops.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.successful_ops.fetch_add(1, Ordering::Relaxed);
                }
            }
            BatchOperationResult::SimilaritySearch(results) => {
                if results.is_empty() {
                    self.failed_ops.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.successful_ops.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // Store result
        self.results.push(result);
    }

    /// Collect all results
    pub fn collect(self) -> CollectedResults {
        let mut all_results = Vec::new();
        while let Some(result) = self.results.pop() {
            all_results.push(result);
        }

        // Sort by operation ID for deterministic ordering
        all_results.sort_by_key(|r| r.operation_id);

        CollectedResults {
            results: all_results,
            successful_count: self.successful_ops.load(Ordering::Relaxed),
            failed_count: self.failed_ops.load(Ordering::Relaxed),
            total_latency_us: self.total_latency_us.load(Ordering::Relaxed),
            simd_operations: self.simd_ops.load(Ordering::Relaxed),
        }
    }

    /// Get current metrics without consuming collector
    pub fn metrics(&self) -> CollectorMetrics {
        CollectorMetrics {
            successful_ops: self.successful_ops.load(Ordering::Relaxed),
            failed_ops: self.failed_ops.load(Ordering::Relaxed),
            total_latency_us: self.total_latency_us.load(Ordering::Relaxed),
            simd_ops: self.simd_ops.load(Ordering::Relaxed),
            pending_results: self.results.len(),
        }
    }
}

/// Collected batch results with aggregated metrics
#[derive(Debug)]
pub struct CollectedResults {
    /// All batch results in order
    pub results: Vec<BatchResult>,
    /// Number of successful operations
    pub successful_count: usize,
    /// Number of failed operations
    pub failed_count: usize,
    /// Total processing time in microseconds
    pub total_latency_us: u64,
    /// Number of SIMD-accelerated operations
    pub simd_operations: usize,
}

impl CollectedResults {
    /// Get average latency per operation
    #[must_use]
    pub fn average_latency_us(&self) -> f64 {
        let total_ops = self.successful_count + self.failed_count;
        if total_ops > 0 {
            let latency = u64_to_f64(self.total_latency_us);
            let ops = usize_to_f64(total_ops);
            latency / ops
        } else {
            0.0
        }
    }

    /// Get SIMD utilization percentage
    #[must_use]
    pub fn simd_utilization(&self) -> f32 {
        let total_ops = self.successful_count + self.failed_count;
        if total_ops > 0 {
            let simd = usize_to_f64(self.simd_operations);
            let total = usize_to_f64(total_ops);
            let percentage = (simd / total) * 100.0;
            clamped_f64_to_f32(percentage, 100.0)
        } else {
            0.0
        }
    }

    /// Group results by operation type
    #[must_use]
    pub fn group_by_type(&self) -> GroupedResults {
        let mut stores = Vec::new();
        let mut recalls = Vec::new();
        let mut similarities = Vec::new();

        for result in &self.results {
            match &result.result {
                BatchOperationResult::Store { .. } => stores.push(result),
                BatchOperationResult::Recall(_) => recalls.push(result),
                BatchOperationResult::SimilaritySearch(_) => similarities.push(result),
            }
        }

        GroupedResults {
            stores,
            recalls,
            similarities,
        }
    }
}

fn usize_to_f64(value: usize) -> f64 {
    u64::try_from(value).map_or_else(|_| u64_to_f64(u64::MAX), u64_to_f64)
}

fn u64_to_f64(value: u64) -> f64 {
    let high_part = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low_part = u32::try_from(value & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    f64::from(high_part).mul_add(4_294_967_296.0, f64::from(low_part))
}

fn clamped_f64_to_f32(value: f64, default: f32) -> f32 {
    if !value.is_finite() {
        return default;
    }

    let clamped = value.clamp(-f64::from(f32::MAX), f64::from(f32::MAX));
    let sign_bit = if clamped.is_sign_negative() {
        1_u32 << 31
    } else {
        0
    };
    let abs = clamped.abs();

    if abs == 0.0 {
        return f32::from_bits(sign_bit);
    }

    let bits = abs.to_bits();
    let exponent_bits = (bits >> 52) & 0x7FF;
    let exponent = i32::try_from(exponent_bits).unwrap_or(0);
    let mut exponent_adjusted = exponent - 1023 + 127;
    if exponent_adjusted <= 0 {
        return f32::from_bits(sign_bit);
    }
    if exponent_adjusted >= 0xFF {
        exponent_adjusted = 0xFE;
    }

    let mantissa = bits & ((1_u64 << 52) - 1);
    let mantissa32 = u32::try_from(mantissa >> (52 - 23)).unwrap_or(0x007F_FFFF);
    let exponent_field = u32::try_from(exponent_adjusted).unwrap_or(0);
    let bits32 = sign_bit | (exponent_field << 23) | mantissa32;
    f32::from_bits(bits32)
}

/// Results grouped by operation type
#[derive(Debug)]
pub struct GroupedResults<'a> {
    /// Results from memory storage operations
    pub stores: Vec<&'a BatchResult>,
    /// Results from memory recall operations
    pub recalls: Vec<&'a BatchResult>,
    /// Results from similarity search operations
    pub similarities: Vec<&'a BatchResult>,
}

/// Current metrics from the collector
#[derive(Debug, Clone)]
pub struct CollectorMetrics {
    /// Number of successful operations completed
    pub successful_ops: usize,
    /// Number of failed operations
    pub failed_ops: usize,
    /// Total accumulated latency in microseconds
    pub total_latency_us: u64,
    /// Number of SIMD-accelerated operations
    pub simd_ops: usize,
    /// Number of results waiting to be processed
    pub pending_results: usize,
}

impl Default for AtomicResultCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::BatchMetadata;
    use crate::{Activation, Confidence, EpisodeBuilder};
    use chrono::Utc;

    #[test]
    fn test_result_collector() {
        let collector = AtomicResultCollector::new();

        // Add store result
        let store_result = BatchResult {
            operation_id: 0,
            result: BatchOperationResult::Store {
                activation: Activation::new(0.9),
                memory_id: "mem_0".to_string(),
            },
            metadata: BatchMetadata {
                processing_time_us: 100,
                simd_used: false,
                memory_pressure: 0.1,
            },
        };
        collector.add_result(store_result);

        // Add recall result
        let episode = EpisodeBuilder::new()
            .id("ep1".to_string())
            .when(Utc::now())
            .what("test".to_string())
            .embedding([0.1; 768])
            .confidence(Confidence::HIGH)
            .build();

        let recall_result = BatchResult {
            operation_id: 1,
            result: BatchOperationResult::Recall(vec![(episode, Confidence::HIGH)]),
            metadata: BatchMetadata {
                processing_time_us: 200,
                simd_used: true,
                memory_pressure: 0.2,
            },
        };
        collector.add_result(recall_result);

        // Check metrics
        let metrics = collector.metrics();
        assert_eq!(metrics.successful_ops, 2);
        assert_eq!(metrics.failed_ops, 0);
        assert_eq!(metrics.total_latency_us, 300);
        assert_eq!(metrics.simd_ops, 1);

        // Collect results
        let collected = collector.collect();
        assert_eq!(collected.results.len(), 2);
        assert_eq!(collected.successful_count, 2);
        assert!((collected.average_latency_us() - 150.0_f64).abs() <= 1e-6_f64);
        assert!((collected.simd_utilization() - 50.0_f32).abs() <= 1e-5_f32);
    }

    #[test]
    fn test_grouped_results() {
        let collector = AtomicResultCollector::new();

        // Add different types of results
        for i in 0..3 {
            let result = BatchResult {
                operation_id: i,
                result: if i == 0 {
                    BatchOperationResult::Store {
                        activation: Activation::new(0.9),
                        memory_id: format!("mem_{i}"),
                    }
                } else if i == 1 {
                    BatchOperationResult::Recall(vec![])
                } else {
                    BatchOperationResult::SimilaritySearch(vec![])
                },
                metadata: BatchMetadata {
                    processing_time_us: 100,
                    simd_used: false,
                    memory_pressure: 0.1,
                },
            };
            collector.add_result(result);
        }

        let collected = collector.collect();
        let grouped = collected.group_by_type();

        assert_eq!(grouped.stores.len(), 1);
        assert_eq!(grouped.recalls.len(), 1);
        assert_eq!(grouped.similarities.len(), 1);
    }
}
