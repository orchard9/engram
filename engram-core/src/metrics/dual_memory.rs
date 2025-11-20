//! Dual-memory architecture specific metrics for episodic-semantic tracking
//!
//! Monitors concept formation, binding dynamics, fan effect penalties,
//! and dual-memory recall operations to provide cognitive health visibility.

use atomic_float::AtomicF32;
use crossbeam_utils::CachePadded;
use std::sync::atomic::{AtomicU64, Ordering};

/// Dual-memory architecture metrics collector
pub struct DualMemoryMetrics {
    /// Concept formation metrics
    concepts_formed_total: CachePadded<AtomicU64>,
    concept_avg_coherence: CachePadded<AtomicF32>,
    concept_avg_member_count: CachePadded<AtomicF32>,
    concept_formation_duration_ms: CachePadded<AtomicU64>,

    /// Binding operation metrics
    bindings_created_total: CachePadded<AtomicU64>,
    bindings_strengthened_total: CachePadded<AtomicU64>,
    bindings_weakened_total: CachePadded<AtomicU64>,
    bindings_pruned_total: CachePadded<AtomicU64>,
    binding_avg_strength: CachePadded<AtomicF32>,
    binding_median_age_seconds: CachePadded<AtomicU64>,

    /// Fan effect observation metrics
    node_fan_out_current: CachePadded<AtomicU64>,
    fan_effect_penalty_magnitude: CachePadded<AtomicF32>,
    high_fan_nodes_total: CachePadded<AtomicU64>,

    /// Recall pathway metrics
    episodic_recall_latency_ms: CachePadded<AtomicU64>,
    semantic_recall_latency_ms: CachePadded<AtomicU64>,
    blended_recall_latency_ms: CachePadded<AtomicU64>,
    recall_accuracy: CachePadded<AtomicF32>,
    recall_precision: CachePadded<AtomicF32>,
}

impl DualMemoryMetrics {
    /// Create new dual-memory metrics collection
    #[must_use]
    pub const fn new() -> Self {
        Self {
            // Concept formation metrics
            concepts_formed_total: CachePadded::new(AtomicU64::new(0)),
            concept_avg_coherence: CachePadded::new(AtomicF32::new(0.0)),
            concept_avg_member_count: CachePadded::new(AtomicF32::new(0.0)),
            concept_formation_duration_ms: CachePadded::new(AtomicU64::new(0)),

            // Binding operation metrics
            bindings_created_total: CachePadded::new(AtomicU64::new(0)),
            bindings_strengthened_total: CachePadded::new(AtomicU64::new(0)),
            bindings_weakened_total: CachePadded::new(AtomicU64::new(0)),
            bindings_pruned_total: CachePadded::new(AtomicU64::new(0)),
            binding_avg_strength: CachePadded::new(AtomicF32::new(0.0)),
            binding_median_age_seconds: CachePadded::new(AtomicU64::new(0)),

            // Fan effect observation metrics
            node_fan_out_current: CachePadded::new(AtomicU64::new(0)),
            fan_effect_penalty_magnitude: CachePadded::new(AtomicF32::new(0.0)),
            high_fan_nodes_total: CachePadded::new(AtomicU64::new(0)),

            // Recall pathway metrics
            episodic_recall_latency_ms: CachePadded::new(AtomicU64::new(0)),
            semantic_recall_latency_ms: CachePadded::new(AtomicU64::new(0)),
            blended_recall_latency_ms: CachePadded::new(AtomicU64::new(0)),
            recall_accuracy: CachePadded::new(AtomicF32::new(0.0)),
            recall_precision: CachePadded::new(AtomicF32::new(0.0)),
        }
    }

    /// Record a dual-memory metric
    pub fn record(&self, metric: &DualMemoryMetric) {
        match metric {
            DualMemoryMetric::ConceptFormation {
                concept_count,
                avg_coherence,
                avg_member_count,
                formation_duration_ms,
            } => {
                self.concepts_formed_total
                    .fetch_add(*concept_count, Ordering::Relaxed);
                self.concept_avg_coherence
                    .store(*avg_coherence, Ordering::Relaxed);
                self.concept_avg_member_count
                    .store(*avg_member_count, Ordering::Relaxed);
                self.concept_formation_duration_ms
                    .store(*formation_duration_ms, Ordering::Release);
            }
            DualMemoryMetric::BindingOperation { operation_type } => match operation_type {
                BindingOperationType::Created { strength } => {
                    self.bindings_created_total.fetch_add(1, Ordering::Relaxed);
                    self.update_binding_avg_strength(*strength);
                }
                BindingOperationType::Strengthened { new_strength } => {
                    self.bindings_strengthened_total
                        .fetch_add(1, Ordering::Relaxed);
                    self.update_binding_avg_strength(*new_strength);
                }
                BindingOperationType::Weakened { new_strength } => {
                    self.bindings_weakened_total.fetch_add(1, Ordering::Relaxed);
                    self.update_binding_avg_strength(*new_strength);
                }
                BindingOperationType::Pruned { count } => {
                    self.bindings_pruned_total
                        .fetch_add(*count, Ordering::Relaxed);
                }
            },
            DualMemoryMetric::FanEffect {
                node_degree,
                penalty_magnitude,
                is_high_fan,
            } => {
                self.node_fan_out_current
                    .store(*node_degree, Ordering::Release);
                self.fan_effect_penalty_magnitude
                    .store(*penalty_magnitude, Ordering::Relaxed);
                if *is_high_fan {
                    self.high_fan_nodes_total.fetch_add(1, Ordering::Relaxed);
                }
            }
            DualMemoryMetric::RecallLatency {
                pathway,
                latency_ms,
            } => match pathway {
                RecallPathway::Episodic => {
                    self.episodic_recall_latency_ms
                        .store(*latency_ms, Ordering::Release);
                }
                RecallPathway::Semantic => {
                    self.semantic_recall_latency_ms
                        .store(*latency_ms, Ordering::Release);
                }
                RecallPathway::Blended => {
                    self.blended_recall_latency_ms
                        .store(*latency_ms, Ordering::Release);
                }
            },
            DualMemoryMetric::RecallQuality {
                accuracy,
                precision,
            } => {
                self.recall_accuracy.store(*accuracy, Ordering::Relaxed);
                self.recall_precision.store(*precision, Ordering::Relaxed);
            }
        }
    }

    /// Update exponential moving average for binding strength
    fn update_binding_avg_strength(&self, new_strength: f32) {
        let current = self.binding_avg_strength.load(Ordering::Acquire);
        // Exponential moving average with alpha=0.1
        let updated = current.mul_add(0.9, new_strength * 0.1);
        self.binding_avg_strength.store(updated, Ordering::Release);
    }

    /// Get concept formation statistics
    pub fn get_concept_formation_stats(&self) -> ConceptFormationStats {
        ConceptFormationStats {
            concepts_formed: self.concepts_formed_total.load(Ordering::Acquire),
            avg_coherence: self.concept_avg_coherence.load(Ordering::Acquire),
            avg_member_count: self.concept_avg_member_count.load(Ordering::Acquire),
            formation_duration_ms: self.concept_formation_duration_ms.load(Ordering::Acquire),
        }
    }

    /// Get binding operation statistics
    pub fn get_binding_stats(&self) -> BindingStats {
        BindingStats {
            created: self.bindings_created_total.load(Ordering::Acquire),
            strengthened: self.bindings_strengthened_total.load(Ordering::Acquire),
            weakened: self.bindings_weakened_total.load(Ordering::Acquire),
            pruned: self.bindings_pruned_total.load(Ordering::Acquire),
            avg_strength: self.binding_avg_strength.load(Ordering::Acquire),
            median_age_seconds: self.binding_median_age_seconds.load(Ordering::Acquire),
        }
    }

    /// Get fan effect statistics
    pub fn get_fan_effect_stats(&self) -> FanEffectStats {
        FanEffectStats {
            current_degree: self.node_fan_out_current.load(Ordering::Acquire),
            penalty_magnitude: self.fan_effect_penalty_magnitude.load(Ordering::Acquire),
            high_fan_nodes: self.high_fan_nodes_total.load(Ordering::Acquire),
        }
    }

    /// Get recall pathway statistics
    pub fn get_recall_stats(&self) -> RecallStats {
        RecallStats {
            episodic_latency_ms: self.episodic_recall_latency_ms.load(Ordering::Acquire),
            semantic_latency_ms: self.semantic_recall_latency_ms.load(Ordering::Acquire),
            blended_latency_ms: self.blended_recall_latency_ms.load(Ordering::Acquire),
            accuracy: self.recall_accuracy.load(Ordering::Acquire),
            precision: self.recall_precision.load(Ordering::Acquire),
        }
    }
}

/// Types of dual-memory metrics
#[derive(Debug, Clone)]
pub enum DualMemoryMetric {
    /// Concept formation event
    ConceptFormation {
        /// Number of concepts formed in this cycle
        concept_count: u64,
        /// Average coherence score of formed concepts
        avg_coherence: f32,
        /// Average number of episodes per concept
        avg_member_count: f32,
        /// Time taken for concept formation (milliseconds)
        formation_duration_ms: u64,
    },
    /// Binding operation event
    BindingOperation {
        /// Type of binding operation performed
        operation_type: BindingOperationType,
    },
    /// Fan effect observation
    FanEffect {
        /// Current degree (fan-out) of the node
        node_degree: u64,
        /// Magnitude of activation penalty applied
        penalty_magnitude: f32,
        /// Whether this node exceeds high fan-out threshold
        is_high_fan: bool,
    },
    /// Recall latency measurement
    RecallLatency {
        /// Which recall pathway was used
        pathway: RecallPathway,
        /// Latency in milliseconds
        latency_ms: u64,
    },
    /// Recall quality metrics
    RecallQuality {
        /// Accuracy of recall (confidence calibration)
        accuracy: f32,
        /// Precision (false positive rate)
        precision: f32,
    },
}

/// Types of binding operations
#[derive(Debug, Clone, Copy)]
pub enum BindingOperationType {
    /// New binding created
    Created {
        /// Initial strength of binding
        strength: f32,
    },
    /// Existing binding strengthened
    Strengthened {
        /// New strength after strengthening
        new_strength: f32,
    },
    /// Existing binding weakened
    Weakened {
        /// New strength after weakening
        new_strength: f32,
    },
    /// Bindings pruned during garbage collection
    Pruned {
        /// Number of bindings pruned
        count: u64,
    },
}

/// Recall pathway type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallPathway {
    /// Episodic recall pathway (System 1)
    Episodic,
    /// Semantic recall pathway (System 2)
    Semantic,
    /// Blended recall combining both pathways
    Blended,
}

/// Concept formation statistics
#[derive(Debug, Clone)]
pub struct ConceptFormationStats {
    /// Total concepts formed
    pub concepts_formed: u64,
    /// Most recent average coherence score
    pub avg_coherence: f32,
    /// Most recent average member count
    pub avg_member_count: f32,
    /// Most recent formation duration
    pub formation_duration_ms: u64,
}

/// Binding operation statistics
#[derive(Debug, Clone)]
pub struct BindingStats {
    /// Total bindings created
    pub created: u64,
    /// Total bindings strengthened
    pub strengthened: u64,
    /// Total bindings weakened
    pub weakened: u64,
    /// Total bindings pruned
    pub pruned: u64,
    /// Exponential moving average of binding strength
    pub avg_strength: f32,
    /// Median age of bindings in seconds
    pub median_age_seconds: u64,
}

/// Fan effect observation statistics
#[derive(Debug, Clone)]
pub struct FanEffectStats {
    /// Most recent node degree observation
    pub current_degree: u64,
    /// Most recent penalty magnitude
    pub penalty_magnitude: f32,
    /// Total nodes observed with high fan-out
    pub high_fan_nodes: u64,
}

/// Recall pathway statistics
#[derive(Debug, Clone)]
pub struct RecallStats {
    /// Most recent episodic recall latency
    pub episodic_latency_ms: u64,
    /// Most recent semantic recall latency
    pub semantic_latency_ms: u64,
    /// Most recent blended recall latency
    pub blended_latency_ms: u64,
    /// Most recent recall accuracy
    pub accuracy: f32,
    /// Most recent recall precision
    pub precision: f32,
}

impl Default for DualMemoryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= f32::EPSILON * 10.0,
            "expected {expected}, got {actual} (diff {diff})"
        );
    }

    #[test]
    fn test_concept_formation_recording() {
        let metrics = DualMemoryMetrics::new();

        metrics.record(&DualMemoryMetric::ConceptFormation {
            concept_count: 5,
            avg_coherence: 0.75,
            avg_member_count: 12.5,
            formation_duration_ms: 150,
        });

        let stats = metrics.get_concept_formation_stats();
        assert_eq!(stats.concepts_formed, 5);
        assert_close(stats.avg_coherence, 0.75);
        assert_close(stats.avg_member_count, 12.5);
        assert_eq!(stats.formation_duration_ms, 150);
    }

    #[test]
    fn test_binding_creation_tracking() {
        let metrics = DualMemoryMetrics::new();

        metrics.record(&DualMemoryMetric::BindingOperation {
            operation_type: BindingOperationType::Created { strength: 0.8 },
        });

        metrics.record(&DualMemoryMetric::BindingOperation {
            operation_type: BindingOperationType::Created { strength: 0.6 },
        });

        let stats = metrics.get_binding_stats();
        assert_eq!(stats.created, 2);
        assert!(stats.avg_strength > 0.0);
        assert!(stats.avg_strength < 1.0);
    }

    #[test]
    fn test_binding_strengthening() {
        let metrics = DualMemoryMetrics::new();

        metrics.record(&DualMemoryMetric::BindingOperation {
            operation_type: BindingOperationType::Strengthened { new_strength: 0.9 },
        });

        let stats = metrics.get_binding_stats();
        assert_eq!(stats.strengthened, 1);
    }

    #[test]
    fn test_binding_weakening() {
        let metrics = DualMemoryMetrics::new();

        metrics.record(&DualMemoryMetric::BindingOperation {
            operation_type: BindingOperationType::Weakened { new_strength: 0.3 },
        });

        let stats = metrics.get_binding_stats();
        assert_eq!(stats.weakened, 1);
    }

    #[test]
    fn test_binding_pruning() {
        let metrics = DualMemoryMetrics::new();

        metrics.record(&DualMemoryMetric::BindingOperation {
            operation_type: BindingOperationType::Pruned { count: 15 },
        });

        let stats = metrics.get_binding_stats();
        assert_eq!(stats.pruned, 15);
    }

    #[test]
    fn test_fan_effect_observation() {
        let metrics = DualMemoryMetrics::new();

        metrics.record(&DualMemoryMetric::FanEffect {
            node_degree: 75,
            penalty_magnitude: 0.25,
            is_high_fan: true,
        });

        let stats = metrics.get_fan_effect_stats();
        assert_eq!(stats.current_degree, 75);
        assert_close(stats.penalty_magnitude, 0.25);
        assert_eq!(stats.high_fan_nodes, 1);
    }

    #[test]
    fn test_recall_latency_tracking() {
        let metrics = DualMemoryMetrics::new();

        metrics.record(&DualMemoryMetric::RecallLatency {
            pathway: RecallPathway::Episodic,
            latency_ms: 120,
        });

        metrics.record(&DualMemoryMetric::RecallLatency {
            pathway: RecallPathway::Semantic,
            latency_ms: 450,
        });

        metrics.record(&DualMemoryMetric::RecallLatency {
            pathway: RecallPathway::Blended,
            latency_ms: 250,
        });

        let stats = metrics.get_recall_stats();
        assert_eq!(stats.episodic_latency_ms, 120);
        assert_eq!(stats.semantic_latency_ms, 450);
        assert_eq!(stats.blended_latency_ms, 250);
    }

    #[test]
    fn test_recall_quality_metrics() {
        let metrics = DualMemoryMetrics::new();

        metrics.record(&DualMemoryMetric::RecallQuality {
            accuracy: 0.92,
            precision: 0.88,
        });

        let stats = metrics.get_recall_stats();
        assert_close(stats.accuracy, 0.92);
        assert_close(stats.precision, 0.88);
    }

    #[test]
    fn test_concurrent_metric_recording() {
        use std::sync::Arc;
        use std::thread;

        let metrics = Arc::new(DualMemoryMetrics::new());
        let mut handles = Vec::new();

        // Spawn 10 threads recording concept formations
        for _ in 0..10 {
            let m = Arc::clone(&metrics);
            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    m.record(&DualMemoryMetric::ConceptFormation {
                        concept_count: 1,
                        avg_coherence: 0.7,
                        avg_member_count: 5.0,
                        formation_duration_ms: 100,
                    });
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = metrics.get_concept_formation_stats();
        assert_eq!(stats.concepts_formed, 100); // 10 threads × 10 increments
    }

    #[test]
    fn test_binding_avg_strength_exponential_moving_average() {
        let metrics = DualMemoryMetrics::new();

        // Record several binding operations with different strengths
        metrics.record(&DualMemoryMetric::BindingOperation {
            operation_type: BindingOperationType::Created { strength: 0.8 },
        });

        metrics.record(&DualMemoryMetric::BindingOperation {
            operation_type: BindingOperationType::Created { strength: 0.6 },
        });

        metrics.record(&DualMemoryMetric::BindingOperation {
            operation_type: BindingOperationType::Created { strength: 0.9 },
        });

        let stats = metrics.get_binding_stats();
        // EMA should converge toward recent values
        assert!(stats.avg_strength > 0.0);
        assert!(stats.avg_strength < 1.0);
        assert_eq!(stats.created, 3);
    }
}
