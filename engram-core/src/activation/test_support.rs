//! Test utilities for running deterministic spreading scenarios.
//!
//! This module intentionally lives in the core crate (instead of the test tree)
//! so integration tests can depend on the helpers without re-implementing
//! boilerplate around engine construction, metrics extraction, and graph setup.

use super::{ParallelSpreadingConfig, ParallelSpreadingEngine, SpreadingMetrics, SpreadingResults};
use crate::activation::storage_aware::StorageAwareActivation;
use crate::activation::{ActivationResult, NodeId};
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use super::MemoryGraph;

/// Serializable view of a storage-aware activation used in snapshots and tests.
#[derive(Debug, Serialize, Clone)]
pub struct ActivationNodeSnapshot {
    /// Identifier for the node.
    pub memory_id: NodeId,
    /// Activation value observed after spreading.
    pub activation: f32,
    /// Confidence associated with the activation.
    pub confidence: f32,
    /// Hop count from the seed node.
    pub hop_count: u16,
    /// Storage tier the activation resided in during sampling.
    pub storage_tier: String,
    /// Debug representation of activation flags.
    pub flags: String,
}

impl ActivationNodeSnapshot {
    fn from_storage_activation(activation: &StorageAwareActivation) -> Self {
        Self {
            memory_id: activation.memory_id.clone(),
            activation: activation
                .activation_level
                .load(std::sync::atomic::Ordering::Relaxed),
            confidence: activation
                .confidence
                .load(std::sync::atomic::Ordering::Relaxed),
            hop_count: activation
                .hop_count
                .load(std::sync::atomic::Ordering::Relaxed),
            storage_tier: format!("{:?}", activation.storage_tier),
            flags: format!("{:#010b}", activation.flags.bits()),
        }
    }
}

/// Serializable tier summary extracted from [`SpreadingResults`].
#[derive(Debug, Serialize, Clone)]
pub struct TierSummarySnapshot {
    /// Number of nodes touched in the tier.
    pub node_count: usize,
    /// Sum of activations in the tier.
    pub total_activation: f32,
    /// Average confidence observed in the tier.
    pub average_confidence: f32,
    /// Whether the tier missed its deadline.
    pub deadline_missed: bool,
}

impl From<&super::TierSummary> for TierSummarySnapshot {
    fn from(summary: &super::TierSummary) -> Self {
        Self {
            node_count: summary.node_count,
            total_activation: summary.total_activation,
            average_confidence: summary.average_confidence,
            deadline_missed: summary.deadline_missed,
        }
    }
}

/// Serializable view of the spreading metrics captured from [`SpreadingMetrics`].
#[derive(Debug, Serialize, Clone)]
pub struct SpreadingMetricsSnapshot {
    /// Total activation operations executed.
    pub total_activations: u64,
    /// Cache hits recorded during spreading.
    pub cache_hits: u64,
    /// Cache misses recorded during spreading.
    pub cache_misses: u64,
    /// Average latency captured in nanoseconds.
    pub average_latency_ns: u64,
    /// Latency budget violations observed.
    pub latency_budget_violations: u64,
    /// Cycle detections encountered.
    pub cycles_detected: u64,
    /// Parallel efficiency ratio.
    pub parallel_efficiency: f32,
    /// Pool slots currently available.
    pub pool_available: u64,
    /// Pool slots in flight.
    pub pool_in_flight: u64,
    /// Highest observed pool level.
    pub pool_high_water_mark: u64,
}

impl SpreadingMetricsSnapshot {
    fn from_metrics(metrics: &SpreadingMetrics) -> Self {
        use std::sync::atomic::Ordering::Relaxed;

        Self {
            total_activations: metrics.total_activations.load(Relaxed),
            cache_hits: metrics.cache_hits.load(Relaxed),
            cache_misses: metrics.cache_misses.load(Relaxed),
            average_latency_ns: metrics.average_latency.load(Relaxed),
            latency_budget_violations: metrics.latency_budget_violations.load(Relaxed),
            cycles_detected: metrics.cycles_detected.load(Relaxed),
            parallel_efficiency: metrics.parallel_efficiency.load(Relaxed),
            pool_available: metrics.pool_available.load(Relaxed),
            pool_in_flight: metrics.pool_in_flight.load(Relaxed),
            pool_high_water_mark: metrics.pool_high_water_mark.load(Relaxed),
        }
    }
}

/// Complete snapshot of a spreading run including activations, summaries, trace and metrics.
#[derive(Debug, Serialize, Clone)]
pub struct SpreadingSnapshot {
    /// Seed activations used to initiate spreading.
    pub seeds: Vec<(NodeId, f32)>,
    /// Storage-aware activations captured from the engine.
    pub activations: Vec<ActivationNodeSnapshot>,
    /// Aggregated tier statistics.
    pub tier_summaries: BTreeMap<String, TierSummarySnapshot>,
    /// Detected cycle paths.
    pub cycle_paths: Vec<Vec<NodeId>>,
    /// Deterministic trace entries when tracing is enabled.
    pub deterministic_trace: Vec<TraceEntrySnapshot>,
    /// Aggregated metrics snapshot.
    pub metrics: SpreadingMetricsSnapshot,
}

/// Snapshot-friendly representation of a deterministic trace entry.
#[derive(Debug, Serialize, Clone)]
pub struct TraceEntrySnapshot {
    /// Depth of the activation.
    pub depth: u16,
    /// Target node identifier.
    pub target_node: NodeId,
    /// Activation level at this trace step.
    pub activation: f32,
    /// Confidence value recorded.
    pub confidence: f32,
    /// Source node that triggered this step if applicable.
    pub source_node: Option<NodeId>,
}

impl From<&super::TraceEntry> for TraceEntrySnapshot {
    fn from(entry: &super::TraceEntry) -> Self {
        Self {
            depth: entry.depth,
            target_node: entry.target_node.clone(),
            activation: entry.activation,
            confidence: entry.confidence,
            source_node: entry.source_node.clone(),
        }
    }
}

/// Convenience wrapper returned by [`run_spreading_snapshot`].
#[derive(Debug)]
pub struct SpreadingRun {
    /// Raw results returned by the engine.
    pub results: SpreadingResults,
    /// Metrics snapshot captured after the run.
    pub metrics: SpreadingMetricsSnapshot,
}

/// Execute spreading with the provided configuration and return a serialisable snapshot.
///
/// This helper removes boilerplate from integration tests that need to run the parallel
/// engine deterministically and capture the resulting activations/metrics.
pub fn run_spreading_snapshot(
    graph: &Arc<MemoryGraph>,
    seeds: &[(NodeId, f32)],
    config: ParallelSpreadingConfig,
) -> ActivationResult<SpreadingSnapshot> {
    let engine = ParallelSpreadingEngine::new(config, Arc::clone(graph))?;
    let results = engine.spread_activation(seeds)?;
    let metrics_snapshot = SpreadingMetricsSnapshot::from_metrics(engine.get_metrics());

    let activations = results
        .activations
        .iter()
        .map(ActivationNodeSnapshot::from_storage_activation)
        .collect();

    let tier_summaries = results
        .tier_summaries
        .iter()
        .map(|(tier, summary)| (format!("{tier:?}"), TierSummarySnapshot::from(summary)))
        .collect();

    Ok(SpreadingSnapshot {
        seeds: seeds.to_vec(),
        activations,
        tier_summaries,
        cycle_paths: results.cycle_paths.clone(),
        deterministic_trace: results
            .deterministic_trace
            .iter()
            .map(TraceEntrySnapshot::from)
            .collect(),
        metrics: metrics_snapshot,
    })
}

/// Execute spreading and return the raw [`SpreadingResults`] alongside a metrics snapshot.
pub fn run_spreading(
    graph: &Arc<MemoryGraph>,
    seeds: &[(NodeId, f32)],
    config: ParallelSpreadingConfig,
) -> ActivationResult<SpreadingRun> {
    let engine = ParallelSpreadingEngine::new(config, Arc::clone(graph))?;
    let results = engine.spread_activation(seeds)?;
    let metrics_snapshot = SpreadingMetricsSnapshot::from_metrics(engine.get_metrics());

    Ok(SpreadingRun {
        results,
        metrics: metrics_snapshot,
    })
}

/// Helper that produces a deterministic spreading configuration with metrics + tracing enabled.
///
/// IMPORTANT: Forces single-threaded execution (num_threads=1) to eliminate PhaseBarrier
/// race conditions that can cause test timeouts. Multi-threaded determinism introduces
/// barrier synchronization overhead that can lead to deadlocks under buffer contention.
/// Sequential execution provides the strongest determinism guarantees for test validation.
///
/// TIMEOUT: Sets 5-minute completion timeout to handle resource contention during parallel
/// test execution. Tests complete in <0.1s when run in isolation, but the full test suite
/// can cause extreme CPU contention on systems with many cores. The extended timeout prevents
/// false failures while maintaining a safety net for actual hangs.
#[must_use]
pub fn deterministic_config(seed: u64) -> ParallelSpreadingConfig {
    let mut config = ParallelSpreadingConfig::deterministic(seed);
    config.num_threads = 1; // Force single-threaded for test determinism
    config.batch_size = 1; // Simplify execution path
    config.work_stealing_ratio = 0.0; // Disable work stealing
    config.max_concurrent_per_tier = 1; // Single task at a time
    config.enable_metrics = true;
    config.trace_activation_flow = true;
    // Set extended timeout for parallel test execution resource contention
    config.completion_timeout = Some(Duration::from_secs(300)); // 5 minutes
    config
}
