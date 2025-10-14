use super::{
    METRIC_SPREADING_AUTOTUNE_LAST_IMPROVEMENT, ParallelSpreadingEngine, storage_aware::StorageTier,
};
use crate::metrics;
use crate::metrics::streaming::{SpreadingSummary, TierLatencySummary};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;
use utoipa::ToSchema;

/// Record of an auto-tuner configuration change with diagnostic metadata
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AutoTuneAuditEntry {
    /// UTC timestamp when the change was applied
    pub timestamp: DateTime<Utc>,
    /// Storage tier that triggered the adjustment
    pub tier: String,
    /// Observed P95 latency in seconds that exceeded the target
    pub observed_p95_seconds: f64,
    /// Target latency threshold for the tier in seconds
    pub target_latency_seconds: f64,
    /// Batch size before the adjustment
    pub batch_size_before: usize,
    /// Batch size after the adjustment
    pub batch_size_after: usize,
    /// Max spreading depth before the adjustment
    pub max_depth_before: u16,
    /// Max spreading depth after the adjustment
    pub max_depth_after: u16,
    /// Tier timeout before adjustment in seconds
    pub timeout_before_seconds: f64,
    /// Tier timeout after adjustment in seconds
    pub timeout_after_seconds: f64,
    /// Estimated performance improvement ratio (0.0 to 1.0)
    pub estimated_improvement: f64,
    /// Human-readable explanation for the adjustment
    pub reason: String,
}

/// Automatic performance tuner for spreading configuration
pub struct SpreadingAutoTuner {
    min_improvement: f64,
    max_history: usize,
    history: Mutex<VecDeque<AutoTuneAuditEntry>>,
}

impl SpreadingAutoTuner {
    /// Create a new auto-tuner with minimum improvement threshold and history capacity
    #[must_use]
    pub fn new(min_improvement: f64, max_history: usize) -> Arc<Self> {
        Arc::new(Self {
            min_improvement,
            max_history: max_history.max(1),
            history: Mutex::new(VecDeque::with_capacity(max_history)),
        })
    }

    /// Evaluate current spreading performance and apply configuration adjustments if beneficial
    pub fn evaluate(
        &self,
        summary: &SpreadingSummary,
        engine: &ParallelSpreadingEngine,
    ) -> Option<AutoTuneAuditEntry> {
        let candidate = select_candidate(summary, self.min_improvement)?;
        let tier = tier_from_label(&candidate.tier)?;
        let tier_index = tier as usize;

        let current_config = engine.config_snapshot();

        let batch_size_before = current_config.batch_size;
        let max_depth_before = current_config.max_depth;
        let timeout_before = current_config.tier_timeouts[tier_index];
        let timeout_before_seconds = timeout_before.as_secs_f64();

        let batch_size_after = clamp_batch_size(recommend_batch_size(
            candidate.mean_seconds,
            candidate.target,
        ));
        let max_depth_after =
            clamp_max_depth(recommend_max_depth(candidate.p99_seconds, candidate.target));
        let timeout_after_seconds = clamp_timeout(
            timeout_before_seconds,
            desired_timeout_seconds(candidate.p95_seconds, candidate.target),
        );

        if batch_size_before == batch_size_after
            && max_depth_before == max_depth_after
            && (timeout_before_seconds - timeout_after_seconds).abs() < 1e-6
        {
            return None;
        }

        let mut updated_config = current_config;
        updated_config.batch_size = batch_size_after;
        updated_config.max_depth = max_depth_after;
        updated_config.tier_timeouts[tier_index] =
            Duration::from_secs_f64(timeout_after_seconds.max(0.000_001));

        engine.update_config(&updated_config);
        engine
            .get_metrics()
            .record_autotune_change(candidate.improvement);

        metrics::record_gauge(
            METRIC_SPREADING_AUTOTUNE_LAST_IMPROVEMENT,
            candidate.improvement,
        );

        let entry = AutoTuneAuditEntry {
            timestamp: Utc::now(),
            tier: candidate.tier,
            observed_p95_seconds: candidate.p95_seconds,
            target_latency_seconds: candidate.target,
            batch_size_before,
            batch_size_after,
            max_depth_before,
            max_depth_after,
            timeout_before_seconds,
            timeout_after_seconds,
            estimated_improvement: candidate.improvement,
            reason: candidate.reason,
        };

        info!(
            tier = entry.tier,
            batch_size_before = entry.batch_size_before,
            batch_size_after = entry.batch_size_after,
            max_depth_before = entry.max_depth_before,
            max_depth_after = entry.max_depth_after,
            timeout_before = entry.timeout_before_seconds,
            timeout_after = entry.timeout_after_seconds,
            improvement = entry.estimated_improvement,
            "Auto-tuner applied spreading configuration change"
        );

        {
            let mut history = self.history.lock();
            history.push_front(entry.clone());
            if history.len() > self.max_history {
                history.pop_back();
            }
        }

        Some(entry)
    }

    /// Retrieve the audit history of configuration changes made by the tuner
    #[must_use]
    pub fn history(&self) -> Vec<AutoTuneAuditEntry> {
        self.history.lock().iter().cloned().collect()
    }
}

struct Candidate {
    tier: String,
    mean_seconds: f64,
    p95_seconds: f64,
    p99_seconds: f64,
    target: f64,
    improvement: f64,
    reason: String,
}

fn select_candidate(summary: &SpreadingSummary, min_improvement: f64) -> Option<Candidate> {
    summary
        .per_tier
        .iter()
        .filter_map(|(tier, latency)| candidate_from_latency(tier, latency, min_improvement))
        .max_by(|a, b| a.improvement.total_cmp(&b.improvement))
}

fn candidate_from_latency(
    tier: &str,
    latency: &TierLatencySummary,
    min_improvement: f64,
) -> Option<Candidate> {
    if latency.p95_seconds <= 0.0 {
        return None;
    }
    let target = target_latency_for_tier(tier);
    if latency.p95_seconds <= target {
        return None;
    }
    let improvement = (latency.p95_seconds - target) / latency.p95_seconds;
    if improvement < min_improvement {
        return None;
    }
    Some(Candidate {
        tier: tier.to_string(),
        mean_seconds: latency.mean_seconds,
        p95_seconds: latency.p95_seconds,
        p99_seconds: latency.p99_seconds,
        target,
        improvement,
        reason: format!(
            "p95 latency {:.4}s exceeds target {:.4}s in {tier} tier",
            latency.p95_seconds, target
        ),
    })
}

fn tier_from_label(label: &str) -> Option<StorageTier> {
    match label {
        "hot" => Some(StorageTier::Hot),
        "warm" => Some(StorageTier::Warm),
        "cold" => Some(StorageTier::Cold),
        _ => None,
    }
}

fn target_latency_for_tier(tier: &str) -> f64 {
    match tier {
        "hot" => 0.0001,
        "warm" => 0.001,
        "cold" => 0.01,
        _ => 0.005,
    }
}

fn recommend_batch_size(mean_latency: f64, target: f64) -> usize {
    if mean_latency <= 0.0 {
        return 32;
    }
    let ratio = (mean_latency / target).clamp(1.0, 8.0);
    (64.0 / ratio).round() as usize
}

fn recommend_max_depth(p99: f64, target: f64) -> u16 {
    if p99 <= target {
        4
    } else if p99 > target * 4.0 {
        2
    } else {
        3
    }
}

fn clamp_batch_size(value: usize) -> usize {
    value.clamp(8, 128)
}

fn clamp_max_depth(value: u16) -> u16 {
    value.clamp(2, 6)
}

fn desired_timeout_seconds(p95: f64, target: f64) -> f64 {
    (p95.max(target) * 1.10).max(target * 1.05)
}

fn clamp_timeout(current_seconds: f64, desired_seconds: f64) -> f64 {
    if current_seconds <= 0.0 {
        return desired_seconds.clamp(0.000_1, 0.5);
    }
    let min = current_seconds * 0.5;
    let max = current_seconds * 2.0;
    desired_seconds.clamp(min, max)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::activation::{ParallelSpreadingConfig, create_activation_graph};

    #[test]
    fn tuner_applies_recommendation_to_engine() {
        let graph = Arc::new(create_activation_graph());
        let engine = ParallelSpreadingEngine::new(ParallelSpreadingConfig::default(), graph)
            .expect("engine creation should succeed");
        let tuner = SpreadingAutoTuner::new(0.10, 4);

        let mut summary = SpreadingSummary::default();
        summary.per_tier.insert(
            "hot".to_string(),
            TierLatencySummary {
                samples: 64,
                mean_seconds: 0.001,
                p50_seconds: 0.0008,
                p95_seconds: 0.002,
                p99_seconds: 0.003,
            },
        );

        let entry = tuner
            .evaluate(&summary, &engine)
            .expect("tuner should apply change");

        assert_eq!(entry.tier, "hot");
        assert_eq!(entry.batch_size_after, 8);
        assert_eq!(entry.max_depth_after, 2);
        assert!(entry.timeout_after_seconds >= entry.timeout_before_seconds);

        let updated_config = engine.config_snapshot();
        assert_eq!(updated_config.batch_size, 8);
        assert_eq!(updated_config.max_depth, 2);

        let history = tuner.history();
        assert_eq!(history.len(), 1);
    }
}
// *** End of File
