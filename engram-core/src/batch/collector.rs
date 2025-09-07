//! Lock-free result collection and aggregation

use crate::batch::{BatchResult, BatchOperationResult, BatchMetadata};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crossbeam_queue::SegQueue;
use std::collections::HashMap;

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
    pub fn new() -> Self {
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
        self.total_latency_us.fetch_add(
            result.metadata.processing_time_us,
            Ordering::Relaxed
        );
        
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
            },
            BatchOperationResult::Recall(episodes) => {
                if !episodes.is_empty() {
                    self.successful_ops.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.failed_ops.fetch_add(1, Ordering::Relaxed);
                }
            },
            BatchOperationResult::SimilaritySearch(results) => {
                if !results.is_empty() {
                    self.successful_ops.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.failed_ops.fetch_add(1, Ordering::Relaxed);
                }
            },
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
    pub fn average_latency_us(&self) -> f64 {
        let total_ops = self.successful_count + self.failed_count;
        if total_ops > 0 {
            self.total_latency_us as f64 / total_ops as f64
        } else {
            0.0
        }
    }
    
    /// Get SIMD utilization percentage
    pub fn simd_utilization(&self) -> f32 {
        let total_ops = self.successful_count + self.failed_count;
        if total_ops > 0 {
            (self.simd_operations as f32 / total_ops as f32) * 100.0
        } else {
            0.0
        }
    }
    
    /// Group results by operation type
    pub fn group_by_type(&self) -> GroupedResults {
        let mut store_results = Vec::new();
        let mut recall_results = Vec::new();
        let mut similarity_results = Vec::new();
        
        for result in &self.results {
            match &result.result {
                BatchOperationResult::Store { .. } => store_results.push(result),
                BatchOperationResult::Recall(_) => recall_results.push(result),
                BatchOperationResult::SimilaritySearch(_) => similarity_results.push(result),
            }
        }
        
        GroupedResults {
            store_results,
            recall_results,
            similarity_results,
        }
    }
}

/// Results grouped by operation type
#[derive(Debug)]
pub struct GroupedResults<'a> {
    pub store_results: Vec<&'a BatchResult>,
    pub recall_results: Vec<&'a BatchResult>,
    pub similarity_results: Vec<&'a BatchResult>,
}

/// Current metrics from the collector
#[derive(Debug, Clone)]
pub struct CollectorMetrics {
    pub successful_ops: usize,
    pub failed_ops: usize,
    pub total_latency_us: u64,
    pub simd_ops: usize,
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
    use crate::{Activation, Confidence, Episode, EpisodeBuilder};
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
        assert_eq!(collected.average_latency_us(), 150.0);
        assert_eq!(collected.simd_utilization(), 50.0);
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
                        memory_id: format!("mem_{}", i),
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
        
        assert_eq!(grouped.store_results.len(), 1);
        assert_eq!(grouped.recall_results.len(), 1);
        assert_eq!(grouped.similarity_results.len(), 1);
    }
}