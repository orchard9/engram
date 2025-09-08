//! Cognitive architecture specific metrics for biological plausibility tracking

use crate::Confidence;
use atomic_float::AtomicF32;
use crossbeam_utils::CachePadded;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cognitive architecture metrics collector
pub struct CognitiveMetrics {
    /// Complementary Learning Systems metrics
    cls_hippocampal_weight: CachePadded<AtomicF32>,
    cls_neocortical_weight: CachePadded<AtomicF32>,

    /// Pattern completion metrics
    pattern_completion_plausibility: CachePadded<AtomicF32>,
    false_memory_generation_rate: CachePadded<AtomicF32>,

    /// Consolidation metrics
    consolidation_transitions: CachePadded<AtomicU64>,
    current_consolidation_state: CachePadded<AtomicU64>,

    /// Spreading activation metrics
    activation_depth_sum: CachePadded<AtomicU64>,
    activation_depth_count: CachePadded<AtomicU64>,
    activation_branching_factor: CachePadded<AtomicF32>,

    /// Confidence calibration metrics
    overconfidence_corrections: CachePadded<AtomicU64>,
    base_rate_updates: CachePadded<AtomicU64>,
}

impl CognitiveMetrics {
    /// Create new cognitive metrics collection
    #[must_use]
    pub const fn new() -> Self {
        Self {
            cls_hippocampal_weight: CachePadded::new(AtomicF32::new(0.5)),
            cls_neocortical_weight: CachePadded::new(AtomicF32::new(0.5)),
            pattern_completion_plausibility: CachePadded::new(AtomicF32::new(0.0)),
            false_memory_generation_rate: CachePadded::new(AtomicF32::new(0.0)),
            consolidation_transitions: CachePadded::new(AtomicU64::new(0)),
            current_consolidation_state: CachePadded::new(AtomicU64::new(0)),
            activation_depth_sum: CachePadded::new(AtomicU64::new(0)),
            activation_depth_count: CachePadded::new(AtomicU64::new(0)),
            activation_branching_factor: CachePadded::new(AtomicF32::new(0.0)),
            overconfidence_corrections: CachePadded::new(AtomicU64::new(0)),
            base_rate_updates: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Record a cognitive metric
    #[inline(always)]
    pub fn record(&self, metric: CognitiveMetric) {
        match metric {
            CognitiveMetric::CLSContribution {
                hippocampal,
                neocortical,
            } => {
                self.cls_hippocampal_weight
                    .store(hippocampal, Ordering::Relaxed);
                self.cls_neocortical_weight
                    .store(neocortical, Ordering::Relaxed);
            }
            CognitiveMetric::PatternCompletion {
                plausibility,
                is_false_memory,
            } => {
                self.pattern_completion_plausibility
                    .store(plausibility, Ordering::Relaxed);
                if is_false_memory {
                    let current = self.false_memory_generation_rate.load(Ordering::Acquire);
                    // Exponential moving average
                    let new_rate = current.mul_add(0.95, 0.05);
                    self.false_memory_generation_rate
                        .store(new_rate, Ordering::Release);
                }
            }
            CognitiveMetric::ConsolidationTransition { from: _, to } => {
                self.consolidation_transitions
                    .fetch_add(1, Ordering::Relaxed);
                self.current_consolidation_state
                    .store(to as u64, Ordering::Release);
            }
            CognitiveMetric::SpreadingActivation {
                depth,
                branching_factor,
            } => {
                self.activation_depth_sum
                    .fetch_add(depth as u64, Ordering::Relaxed);
                self.activation_depth_count.fetch_add(1, Ordering::Relaxed);
                self.activation_branching_factor
                    .store(branching_factor, Ordering::Relaxed);
            }
            CognitiveMetric::ConfidenceCalibration { correction_type } => match correction_type {
                CalibrationCorrection::Overconfidence => {
                    self.overconfidence_corrections
                        .fetch_add(1, Ordering::Relaxed);
                }
                CalibrationCorrection::BaseRate => {
                    self.base_rate_updates.fetch_add(1, Ordering::Relaxed);
                }
            },
        }
    }

    /// Get CLS hippocampal/neocortical balance
    pub fn get_cls_balance(&self) -> (f32, f32) {
        (
            self.cls_hippocampal_weight.load(Ordering::Acquire),
            self.cls_neocortical_weight.load(Ordering::Acquire),
        )
    }

    /// Get pattern completion metrics
    pub fn get_pattern_completion_stats(&self) -> PatternCompletionStats {
        PatternCompletionStats {
            plausibility: self.pattern_completion_plausibility.load(Ordering::Acquire),
            false_memory_rate: self.false_memory_generation_rate.load(Ordering::Acquire),
        }
    }

    /// Get average activation spreading depth
    pub fn get_average_activation_depth(&self) -> f32 {
        let sum = self.activation_depth_sum.load(Ordering::Acquire);
        let count = self.activation_depth_count.load(Ordering::Acquire);

        if count > 0 {
            sum as f32 / count as f32
        } else {
            0.0
        }
    }

    /// Get confidence calibration statistics
    pub fn get_calibration_stats(&self) -> CalibrationStats {
        CalibrationStats {
            overconfidence_corrections: self.overconfidence_corrections.load(Ordering::Acquire),
            base_rate_updates: self.base_rate_updates.load(Ordering::Acquire),
        }
    }
}

/// Types of cognitive metrics
#[derive(Debug, Clone)]
pub enum CognitiveMetric {
    /// Complementary Learning Systems contribution
    CLSContribution { hippocampal: f32, neocortical: f32 },
    /// Pattern completion event
    PatternCompletion {
        plausibility: f32,
        is_false_memory: bool,
    },
    /// Memory consolidation state transition
    ConsolidationTransition {
        from: ConsolidationState,
        to: ConsolidationState,
    },
    /// Spreading activation measurement
    SpreadingActivation { depth: usize, branching_factor: f32 },
    /// Confidence calibration event
    ConfidenceCalibration {
        correction_type: CalibrationCorrection,
    },
}

/// Memory consolidation states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ConsolidationState {
    /// Recently formed memory (hippocampus-dependent)
    Recent = 0,
    /// Memory in consolidation process
    Consolidating = 1,
    /// Remote, consolidated memory (neocortex-dependent)
    Remote = 2,
}

/// Types of confidence calibration corrections
#[derive(Debug, Clone, Copy)]
pub enum CalibrationCorrection {
    /// Reduce overconfident predictions
    Overconfidence,
    /// Adjust based on base rate information
    BaseRate,
}

/// Pattern completion statistics
#[derive(Debug, Clone)]
pub struct PatternCompletionStats {
    pub plausibility: f32,
    pub false_memory_rate: f32,
}

/// Confidence calibration statistics
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    pub overconfidence_corrections: u64,
    pub base_rate_updates: u64,
}

/// Cognitive insight derived from metrics analysis
#[derive(Debug, Clone)]
pub struct CognitiveInsight {
    pub insight_type: InsightType,
    pub description: String,
    pub confidence: Confidence,
    pub biological_relevance: String,
    pub actionable_recommendation: String,
}

/// Types of cognitive insights
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsightType {
    MemorySystem,
    Performance,
    Consolidation,
    PatternCompletion,
    Confidence,
}

impl Default for CognitiveMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cls_contribution_recording() {
        let metrics = CognitiveMetrics::new();

        metrics.record(CognitiveMetric::CLSContribution {
            hippocampal: 0.7,
            neocortical: 0.3,
        });

        let (hippo, neo) = metrics.get_cls_balance();
        assert_eq!(hippo, 0.7);
        assert_eq!(neo, 0.3);
    }

    #[test]
    fn test_false_memory_rate_tracking() {
        let metrics = CognitiveMetrics::new();

        // Record some pattern completions
        for i in 0..10 {
            metrics.record(CognitiveMetric::PatternCompletion {
                plausibility: 0.8,
                is_false_memory: i % 3 == 0, // Every 3rd is false memory
            });
        }

        let stats = metrics.get_pattern_completion_stats();
        assert!(stats.false_memory_rate > 0.0);
        assert!(stats.false_memory_rate < 1.0);
    }

    #[test]
    fn test_activation_depth_averaging() {
        let metrics = CognitiveMetrics::new();

        metrics.record(CognitiveMetric::SpreadingActivation {
            depth: 3,
            branching_factor: 2.5,
        });

        metrics.record(CognitiveMetric::SpreadingActivation {
            depth: 5,
            branching_factor: 3.0,
        });

        let avg_depth = metrics.get_average_activation_depth();
        assert_eq!(avg_depth, 4.0); // (3 + 5) / 2
    }
}
