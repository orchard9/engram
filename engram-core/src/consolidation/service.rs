//! In-memory consolidation cache service implementation and traits.

use crate::completion::ConsolidationSnapshot;
use crate::metrics::{
    self, CONSOLIDATION_CITATION_CHURN, CONSOLIDATION_CITATIONS_CURRENT,
    CONSOLIDATION_FRESHNESS_SECONDS, CONSOLIDATION_NOVELTY_GAUGE, CONSOLIDATION_NOVELTY_VARIANCE,
    CONSOLIDATION_RUNS_TOTAL,
};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

const MAX_ALERT_LOG_ENTRIES: usize = 512;
const DEFAULT_ALERT_LOG_FILE: &str = "data/consolidation/alerts/belief_updates.jsonl";

fn default_alert_log_path() -> PathBuf {
    std::env::var("ENGRAM_CONSOLIDATION_ALERT_LOG")
        .map_or_else(|_| PathBuf::from(DEFAULT_ALERT_LOG_FILE), PathBuf::from)
}

/// Identifies where a cached consolidation snapshot originated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsolidationCacheSource {
    /// Snapshot produced by the background scheduler.
    Scheduler,
    /// Snapshot generated via on-demand API call.
    OnDemand,
}

/// Structured record describing how a consolidation run changed semantic beliefs.
#[derive(Debug, Clone, Serialize)]
pub struct BeliefUpdateRecord {
    /// Semantic pattern identifier affected by the update.
    pub pattern_id: String,
    /// Change in schema confidence relative to the prior snapshot.
    pub confidence_delta: f32,
    /// Change in citation count for the belief.
    pub citation_delta: i32,
    /// Strength delta for the consolidated belief.
    pub novelty: f32,
    /// When the update was generated.
    pub generated_at: DateTime<Utc>,
    /// Timestamp of the snapshot that triggered this update.
    pub snapshot_generated_at: DateTime<Utc>,
    /// Origin of the snapshot (scheduler vs on-demand).
    pub source: ConsolidationCacheSource,
}

/// Abstraction over consolidation snapshot caching and observability.
pub trait ConsolidationService: Send + Sync {
    /// Returns the most recently cached snapshot, if present.
    fn cached_snapshot(&self) -> Option<ConsolidationSnapshot>;
    /// Updates the cache and emits observability side effects.
    fn update_cache(&self, snapshot: &ConsolidationSnapshot, source: ConsolidationCacheSource);
    /// Overrides the alert log destination used for belief deltas.
    fn set_alert_log_path(&self, path: PathBuf);
    /// Returns the current alert log destination.
    fn alert_log_path(&self) -> PathBuf;
    /// Returns the most recent belief-update entries retained in memory.
    fn recent_updates(&self) -> Vec<BeliefUpdateRecord>;
}

/// Default in-memory implementation of the consolidation service.
pub struct InMemoryConsolidationService {
    cache: RwLock<Option<ConsolidationSnapshot>>,
    alerts: RwLock<VecDeque<BeliefUpdateRecord>>,
    alert_log_path: RwLock<PathBuf>,
}

impl Default for InMemoryConsolidationService {
    fn default() -> Self {
        Self {
            cache: RwLock::new(None),
            alerts: RwLock::new(VecDeque::with_capacity(MAX_ALERT_LOG_ENTRIES)),
            alert_log_path: RwLock::new(default_alert_log_path()),
        }
    }
}

impl InMemoryConsolidationService {
    fn compute_belief_updates(
        previous: Option<&ConsolidationSnapshot>,
        current: &ConsolidationSnapshot,
        source: ConsolidationCacheSource,
    ) -> Vec<BeliefUpdateRecord> {
        let mut updates = Vec::with_capacity(current.patterns.len());
        let mut prior_map: HashMap<String, crate::completion::SemanticPattern> = previous
            .map(|snapshot| {
                snapshot
                    .patterns
                    .iter()
                    .cloned()
                    .map(|pattern| (pattern.id.clone(), pattern))
                    .collect()
            })
            .unwrap_or_default();

        let generated_at = Utc::now();

        for pattern in &current.patterns {
            let previous_pattern = prior_map.remove(&pattern.id);
            let confidence_delta = pattern.schema_confidence.raw()
                - previous_pattern
                    .as_ref()
                    .map_or(0.0, |p| p.schema_confidence.raw());
            let citation_delta = i32::try_from(pattern.source_episodes.len()).unwrap_or(i32::MAX)
                - previous_pattern
                    .as_ref()
                    .and_then(|p| i32::try_from(p.source_episodes.len()).ok())
                    .unwrap_or(0);
            let novelty = pattern.strength - previous_pattern.as_ref().map_or(0.0, |p| p.strength);

            updates.push(BeliefUpdateRecord {
                pattern_id: pattern.id.clone(),
                confidence_delta,
                citation_delta,
                novelty,
                generated_at,
                snapshot_generated_at: current.generated_at,
                source,
            });
        }

        for (_, pattern) in prior_map {
            updates.push(BeliefUpdateRecord {
                pattern_id: pattern.id,
                confidence_delta: -pattern.schema_confidence.raw(),
                citation_delta: -i32::try_from(pattern.source_episodes.len()).unwrap_or(i32::MAX),
                novelty: -pattern.strength,
                generated_at,
                snapshot_generated_at: current.generated_at,
                source,
            });
        }

        updates
    }

    fn persist_updates(&self, updates: &[BeliefUpdateRecord]) {
        if updates.is_empty() {
            return;
        }

        {
            let mut log = self.alerts.write();
            for update in updates {
                if log.len() >= MAX_ALERT_LOG_ENTRIES {
                    log.pop_front();
                }
                log.push_back(update.clone());
            }
        }

        if let Err(error) = self.append_updates_to_disk(updates) {
            tracing::warn!(
                ?error,
                "failed to append consolidation belief updates to log"
            );
        }
    }

    fn append_updates_to_disk(&self, updates: &[BeliefUpdateRecord]) -> std::io::Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        let path = self.alert_log_path.read().clone();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        for update in updates {
            let line = serde_json::to_string(update)
                .map_err(|err| std::io::Error::other(format!("{err}")))?;
            file.write_all(line.as_bytes())?;
            file.write_all(b"\n")?;
        }

        Ok(())
    }

    fn record_metrics(
        snapshot: &ConsolidationSnapshot,
        updates: &[BeliefUpdateRecord],
        source: ConsolidationCacheSource,
    ) {
        let total_citations: usize = snapshot
            .patterns
            .iter()
            .map(|pattern| pattern.source_episodes.len())
            .sum();

        let freshness_seconds = (Utc::now() - snapshot.generated_at).num_seconds().max(0) as f64;
        let max_novelty = updates
            .iter()
            .map(|update| update.novelty.abs())
            .fold(0.0_f32, f32::max);

        // Compute novelty variance across all updates
        let novelty_variance = if updates.len() > 1 {
            let mean_novelty: f32 =
                updates.iter().map(|u| u.novelty.abs()).sum::<f32>() / updates.len() as f32;
            let variance: f32 = updates
                .iter()
                .map(|u| {
                    let diff = u.novelty.abs() - mean_novelty;
                    diff * diff
                })
                .sum::<f32>()
                / updates.len() as f32;
            variance
        } else {
            0.0
        };

        // Compute citation churn: percentage of patterns with citation changes
        let patterns_with_citation_changes =
            updates.iter().filter(|u| u.citation_delta != 0).count();
        let citation_churn = if updates.is_empty() {
            0.0
        } else {
            (patterns_with_citation_changes as f32 / updates.len() as f32) * 100.0
        };

        metrics::record_gauge(CONSOLIDATION_FRESHNESS_SECONDS, freshness_seconds);
        metrics::record_gauge(CONSOLIDATION_NOVELTY_GAUGE, f64::from(max_novelty));
        metrics::record_gauge(CONSOLIDATION_NOVELTY_VARIANCE, f64::from(novelty_variance));
        metrics::record_gauge(CONSOLIDATION_CITATION_CHURN, f64::from(citation_churn));
        metrics::record_gauge(CONSOLIDATION_CITATIONS_CURRENT, total_citations as f64);

        if matches!(source, ConsolidationCacheSource::Scheduler) {
            metrics::increment_counter(CONSOLIDATION_RUNS_TOTAL, 1);
        }
    }
}

impl ConsolidationService for InMemoryConsolidationService {
    fn cached_snapshot(&self) -> Option<ConsolidationSnapshot> {
        self.cache.read().clone()
    }

    fn update_cache(&self, snapshot: &ConsolidationSnapshot, source: ConsolidationCacheSource) {
        let updates = {
            let mut cache = self.cache.write();
            let previous = cache.clone();
            let updates = Self::compute_belief_updates(previous.as_ref(), snapshot, source);
            *cache = Some(snapshot.clone());
            updates
        };

        self.persist_updates(&updates);
        Self::record_metrics(snapshot, &updates, source);
    }

    fn set_alert_log_path(&self, path: PathBuf) {
        *self.alert_log_path.write() = path;
    }

    fn alert_log_path(&self) -> PathBuf {
        self.alert_log_path.read().clone()
    }

    fn recent_updates(&self) -> Vec<BeliefUpdateRecord> {
        self.alerts.read().iter().cloned().collect()
    }
}
