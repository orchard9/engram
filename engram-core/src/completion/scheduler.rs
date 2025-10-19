//! Consolidation scheduler for asynchronous memory consolidation.
//!
//! This module provides a background scheduler that runs consolidation processes
//! asynchronously without blocking recall operations. The scheduler is interruptible
//! and resumable, following cognitive principles of sleep-based memory consolidation.

use super::{CompletionConfig, ConsolidationEngine};
use crate::consolidation::ConsolidationCacheSource;
use crate::{Episode, MemoryStore, metrics};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Consolidation scheduler state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerState {
    /// Scheduler is idle, waiting for next consolidation window
    Idle,
    /// Consolidation is actively running
    Running,
    /// Consolidation was interrupted and can be resumed
    Interrupted,
    /// Scheduler has been stopped
    Stopped,
}

/// Statistics for consolidation operations
#[derive(Debug, Clone, Default)]
pub struct ConsolidationStats {
    /// Total consolidation runs completed
    pub total_runs: usize,
    /// Total patterns extracted
    pub patterns_extracted: usize,
    /// Total episodes consolidated
    pub episodes_consolidated: usize,
    /// Average run duration in seconds
    pub avg_run_duration_secs: f32,
    /// Last consolidation timestamp
    pub last_run: Option<chrono::DateTime<chrono::Utc>>,
}

/// Consolidation scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// How often to run consolidation (in seconds)
    pub consolidation_interval_secs: u64,
    /// Minimum episodes before triggering consolidation
    pub min_episodes_threshold: usize,
    /// Maximum episodes to process in one run
    pub max_episodes_per_run: usize,
    /// Whether consolidation is enabled
    pub enabled: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            consolidation_interval_secs: 300, // 5 minutes
            min_episodes_threshold: 10,
            max_episodes_per_run: 100,
            enabled: true,
        }
    }
}

/// Asynchronous consolidation scheduler
///
/// Runs memory consolidation in the background without blocking recall operations.
/// Supports graceful shutdown and resumption.
pub struct ConsolidationScheduler {
    /// Consolidation engine
    engine: Arc<RwLock<ConsolidationEngine>>,
    /// Scheduler configuration
    config: SchedulerConfig,
    /// Current state
    state: Arc<RwLock<SchedulerState>>,
    /// Statistics
    stats: Arc<RwLock<ConsolidationStats>>,
    /// Last processed episode ID (for resumability) - reserved for future use
    #[allow(dead_code)]
    last_processed_id: Arc<RwLock<Option<String>>>,
}

impl ConsolidationScheduler {
    /// Create a new consolidation scheduler
    #[must_use]
    pub fn new(completion_config: CompletionConfig, scheduler_config: SchedulerConfig) -> Self {
        let engine = ConsolidationEngine::new(completion_config);

        Self {
            engine: Arc::new(RwLock::new(engine)),
            config: scheduler_config,
            state: Arc::new(RwLock::new(SchedulerState::Idle)),
            stats: Arc::new(RwLock::new(ConsolidationStats::default())),
            last_processed_id: Arc::new(RwLock::new(None)),
        }
    }

    /// Run the consolidation scheduler in a background task
    ///
    /// This spawns a tokio task that runs consolidation periodically.
    /// Returns a handle that can be used to stop the scheduler.
    ///
    /// # Arguments
    ///
    /// * `store` - The memory store to consolidate
    /// * `shutdown_rx` - Channel to receive shutdown signals
    ///
    /// # Returns
    ///
    /// A `tokio::task::JoinHandle` for the background task
    pub fn spawn(
        self: Arc<Self>,
        store: Arc<MemoryStore>,
        mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            info!(
                interval_secs = self.config.consolidation_interval_secs,
                "Consolidation scheduler starting"
            );

            let mut interval =
                tokio::time::interval(Duration::from_secs(self.config.consolidation_interval_secs));

            loop {
                tokio::select! {
                    _ = shutdown_rx.changed() => {
                        info!("Consolidation scheduler received shutdown signal");

                        // Mark as interrupted if we're mid-consolidation
                        let mut state = self.state.write().await;
                        if *state == SchedulerState::Running {
                            *state = SchedulerState::Interrupted;
                            drop(state);
                            info!("Consolidation interrupted - state saved for resumption");
                        } else {
                            *state = SchedulerState::Stopped;
                            drop(state);
                        }

                        break;
                    }
                    _ = interval.tick() => {
                        if !self.config.enabled {
                            debug!("Consolidation disabled, skipping");
                            continue;
                        }

                        // Run consolidation
                        if let Err(e) = self.run_consolidation(&store).await {
                            metrics::increment_counter("engram_consolidation_failures_total", 1);
                            warn!(error = ?e, "Consolidation run failed");
                        }
                    }
                }
            }

            let final_stats = self.stats.read().await;
            let total_runs = final_stats.total_runs;
            let patterns_extracted = final_stats.patterns_extracted;
            drop(final_stats); // Release lock before logging

            info!(
                total_runs,
                patterns_extracted, "Consolidation scheduler stopped gracefully"
            );
        })
    }

    /// Run a single consolidation cycle
    async fn run_consolidation(&self, store: &Arc<MemoryStore>) -> Result<(), String> {
        // Update state to running
        {
            let mut state = self.state.write().await;
            *state = SchedulerState::Running;
        }

        let start_time = std::time::Instant::now();

        debug!("Starting consolidation run");

        // Get episodes from store
        let episodes = self.collect_episodes_for_consolidation(store);

        if episodes.len() < self.config.min_episodes_threshold {
            debug!(
                episode_count = episodes.len(),
                threshold = self.config.min_episodes_threshold,
                "Insufficient episodes for consolidation"
            );

            {
                let mut state = self.state.write().await;
                *state = SchedulerState::Idle;
            }
            return Ok(());
        }

        info!(
            episode_count = episodes.len(),
            "Running consolidation on episodes"
        );

        // Run consolidation engine and capture scheduler snapshot
        let snapshot = {
            let mut engine = self.engine.write().await;
            engine.ripple_replay(&episodes);
            engine.snapshot()
        };

        store
            .consolidation_service()
            .update_cache(&snapshot, ConsolidationCacheSource::Scheduler);

        // Update statistics
        let duration = start_time.elapsed().as_secs_f32();
        let mut consolidation_stats = self.stats.write().await;
        consolidation_stats.total_runs += 1;
        consolidation_stats.episodes_consolidated += episodes.len();
        consolidation_stats.patterns_extracted = snapshot.patterns.len();
        consolidation_stats.last_run = Some(chrono::Utc::now());

        // Update average duration
        let prev_avg = consolidation_stats.avg_run_duration_secs;
        let total_runs = consolidation_stats.total_runs as f32;
        consolidation_stats.avg_run_duration_secs =
            (prev_avg * (total_runs - 1.0) + duration) / total_runs;

        info!(
            duration_secs = duration,
            episodes_processed = episodes.len(),
            total_runs = consolidation_stats.total_runs,
            patterns_extracted = snapshot.patterns.len(),
            "Consolidation run completed"
        );

        // Mark as idle
        {
            let mut state = self.state.write().await;
            *state = SchedulerState::Idle;
        }

        Ok(())
    }

    /// Collect episodes that need consolidation
    ///
    /// This method prioritizes recent episodes with high prediction error
    /// for consolidation, following psychological principles of memory replay.
    ///
    fn collect_episodes_for_consolidation(&self, store: &Arc<MemoryStore>) -> Vec<Episode> {
        let mut episodes: Vec<Episode> = store
            .get_all_episodes()
            .map(|(_, episode)| episode)
            .collect();

        if episodes.is_empty() {
            debug!("No episodes available for consolidation");
            return Vec::new();
        }

        // Prioritize recent episodes while keeping ordering deterministic for tests
        episodes.sort_by(|a, b| b.when.cmp(&a.when));

        if self.config.max_episodes_per_run > 0 && episodes.len() > self.config.max_episodes_per_run
        {
            episodes.truncate(self.config.max_episodes_per_run);
        }

        episodes
    }

    /// Get current scheduler state
    #[must_use]
    pub async fn state(&self) -> SchedulerState {
        *self.state.read().await
    }

    /// Get consolidation statistics
    #[must_use]
    pub async fn stats(&self) -> ConsolidationStats {
        self.stats.read().await.clone()
    }

    /// Pause the scheduler
    pub async fn pause(&self) {
        let mut state = self.state.write().await;
        if *state == SchedulerState::Running {
            *state = SchedulerState::Interrupted;
            drop(state);
            info!("Consolidation scheduler paused");
        }
    }

    /// Resume the scheduler after pause
    pub async fn resume(&self) {
        let mut state = self.state.write().await;
        if *state == SchedulerState::Interrupted {
            *state = SchedulerState::Idle;
            drop(state);
            info!("Consolidation scheduler resumed");
        }
    }

    /// Enable consolidation
    pub fn enable(&mut self) {
        self.config.enabled = true;
        info!("Consolidation enabled");
    }

    /// Disable consolidation
    pub fn disable(&mut self) {
        self.config.enabled = false;
        info!("Consolidation disabled");
    }

    /// Check if consolidation is enabled
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Confidence, EpisodeBuilder, metrics, metrics::CONSOLIDATION_RUNS_TOTAL, store::MemoryStore,
    };
    use chrono::Utc;
    use std::sync::Arc;

    #[test]
    fn test_scheduler_config_defaults() {
        let config = SchedulerConfig::default();
        assert_eq!(config.consolidation_interval_secs, 300);
        assert_eq!(config.min_episodes_threshold, 10);
        assert!(config.enabled);
    }

    #[tokio::test]
    async fn test_scheduler_creation() {
        let completion_config = CompletionConfig::default();
        let scheduler_config = SchedulerConfig::default();
        let scheduler = ConsolidationScheduler::new(completion_config, scheduler_config);

        assert_eq!(scheduler.state().await, SchedulerState::Idle);
        assert!(scheduler.is_enabled());
    }

    #[tokio::test]
    async fn test_scheduler_pause_resume() {
        let completion_config = CompletionConfig::default();
        let scheduler_config = SchedulerConfig::default();
        let scheduler = ConsolidationScheduler::new(completion_config, scheduler_config);

        // Manually set to running
        {
            let mut state = scheduler.state.write().await;
            *state = SchedulerState::Running;
        }

        // Pause should transition to interrupted
        scheduler.pause().await;
        assert_eq!(scheduler.state().await, SchedulerState::Interrupted);

        // Resume should transition back to idle
        scheduler.resume().await;
        assert_eq!(scheduler.state().await, SchedulerState::Idle);
    }

    #[tokio::test]
    async fn test_scheduler_enable_disable() {
        let completion_config = CompletionConfig::default();
        let scheduler_config = SchedulerConfig::default();
        let mut scheduler = ConsolidationScheduler::new(completion_config, scheduler_config);

        assert!(scheduler.is_enabled());

        scheduler.disable();
        assert!(!scheduler.is_enabled());

        scheduler.enable();
        assert!(scheduler.is_enabled());
    }

    #[tokio::test]
    async fn test_stats_initialization() {
        let completion_config = CompletionConfig::default();
        let scheduler_config = SchedulerConfig::default();
        let scheduler = ConsolidationScheduler::new(completion_config, scheduler_config);

        let stats = scheduler.stats().await;
        assert_eq!(stats.total_runs, 0);
        assert_eq!(stats.patterns_extracted, 0);
        assert_eq!(stats.episodes_consolidated, 0);
        assert!(stats.last_run.is_none());
    }

    fn test_embedding(seed: f32) -> [f32; 768] {
        let mut embedding = [0.0f32; 768];
        for (idx, value) in embedding.iter_mut().enumerate() {
            *value = seed + idx as f32 * 0.0001;
        }
        embedding
    }

    #[tokio::test]
    async fn test_scheduler_populates_cache_and_metrics() -> Result<(), String> {
        let _ = metrics::init();

        let store = Arc::new(MemoryStore::new(64));
        for idx in 0..6 {
            let episode = EpisodeBuilder::new()
                .id(format!("ep_cache_{idx}"))
                .when(Utc::now())
                .what(format!("episode {idx}"))
                .embedding(test_embedding(idx as f32))
                .confidence(Confidence::HIGH)
                .build();
            store.store(episode);
        }

        let scheduler_config = SchedulerConfig {
            consolidation_interval_secs: 1,
            min_episodes_threshold: 1,
            max_episodes_per_run: 32,
            enabled: true,
        };
        let scheduler = ConsolidationScheduler::new(CompletionConfig::default(), scheduler_config);

        if store.cached_consolidation_snapshot().is_some() {
            return Err("cache should start empty".to_string());
        }

        scheduler
            .run_consolidation(&store)
            .await
            .map_err(|err| format!("consolidation run failed: {err}"))?;

        let cached = store
            .cached_consolidation_snapshot()
            .ok_or_else(|| "scheduler run should populate cache".to_string())?;
        let via_api = store.consolidation_snapshot(0);

        if cached.generated_at != via_api.generated_at {
            return Err("cached snapshot should match API snapshot".to_string());
        }

        let registry =
            metrics::metrics().ok_or_else(|| "metrics registry unavailable".to_string())?;
        if registry.counter_value(CONSOLIDATION_RUNS_TOTAL) < 1 {
            return Err("scheduler run counter should increment".to_string());
        }

        Ok(())
    }
}
