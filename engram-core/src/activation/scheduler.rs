//! Tier-aware scheduling for activation spreading.
//!
//! The scheduler maintains independent work queues per storage tier so that
//! latency-sensitive hot memories execute before warm/cold tiers. Each tier has
//! per-task timeouts, global deadlines, and bounded concurrency to prevent
//! cold storage from starving fast tiers. Workers request tasks directly from
//! the scheduler, which enforces these guarantees in a lock-free manner using
//! `SegQueue`.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crossbeam_queue::SegQueue;

use super::latency_budget::LatencyBudgetManager;
use super::storage_aware::StorageTier;
use super::{ActivationTask, ParallelSpreadingConfig};

/// Snapshot of scheduler state for diagnostics and result aggregation.
#[derive(Debug, Clone)]
pub struct TierQueueStateSnapshot {
    /// Tier the snapshot describes.
    pub tier: StorageTier,
    /// Number of queued tasks awaiting execution.
    pub queued: usize,
    /// Number of tasks currently being processed.
    pub in_flight: usize,
    /// Whether the tier exceeded its deadline or per-task timeout budget.
    pub deadline_missed: bool,
}

impl Default for TierQueueStateSnapshot {
    fn default() -> Self {
        Self {
            tier: StorageTier::Hot,
            queued: 0,
            in_flight: 0,
            deadline_missed: false,
        }
    }
}

/// Collection of tier snapshots captured at a moment in time.
#[derive(Debug, Clone, Default)]
pub struct SchedulerSnapshot {
    /// Per-tier scheduler state keyed by tier.
    pub tiers: HashMap<StorageTier, TierQueueStateSnapshot>,
    /// Total number of tasks across all tiers.
    pub total_pending: usize,
}

/// Tier-aware scheduler responsible for prioritising hot storage while
/// respecting warm/cold budgets.
pub struct TierAwareSpreadingScheduler {
    queues: [TierQueue; 3],
    priority_hot_tier: bool,
    bypass_cold: AtomicBool,
    latency_budget: LatencyBudgetManager,
    timing: Mutex<SchedulerTiming>,
}

impl TierAwareSpreadingScheduler {
    /// Create a scheduler aligned with the provided spreading configuration.
    #[must_use]
    pub fn from_config(config: &ParallelSpreadingConfig) -> Self {
        let latency_budget = LatencyBudgetManager::new();
        let start = Instant::now();
        let deadlines = [
            start + config.tier_timeouts[StorageTier::Hot as usize],
            start + config.tier_timeouts[StorageTier::Warm as usize],
            start + config.tier_timeouts[StorageTier::Cold as usize],
        ];

        Self {
            queues: [
                TierQueue::new(
                    StorageTier::Hot,
                    config.tier_timeouts[StorageTier::Hot as usize],
                    config.max_concurrent_per_tier,
                    config.deterministic,
                ),
                TierQueue::new(
                    StorageTier::Warm,
                    config.tier_timeouts[StorageTier::Warm as usize],
                    config.max_concurrent_per_tier,
                    config.deterministic,
                ),
                TierQueue::new(
                    StorageTier::Cold,
                    config.tier_timeouts[StorageTier::Cold as usize],
                    config.max_concurrent_per_tier,
                    config.deterministic,
                ),
            ],
            priority_hot_tier: config.priority_hot_tier,
            bypass_cold: AtomicBool::new(false),
            latency_budget,
            timing: Mutex::new(SchedulerTiming { start, deadlines }),
        }
    }

    /// Clear queued state and restart timing budgets.
    pub fn reset(&self, config: &ParallelSpreadingConfig) {
        for queue in &self.queues {
            queue.reset();
        }
        self.bypass_cold.store(false, Ordering::Relaxed);

        let mut timing = match self.timing.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        timing.start = Instant::now();
        timing.deadlines = [
            timing.start + config.tier_timeouts[StorageTier::Hot as usize],
            timing.start + config.tier_timeouts[StorageTier::Warm as usize],
            timing.start + config.tier_timeouts[StorageTier::Cold as usize],
        ];
    }

    /// Enqueue an activation task according to its storage tier metadata.
    pub fn enqueue_task(&self, mut task: ActivationTask) {
        let tier = task
            .storage_tier
            .unwrap_or_else(|| StorageTier::from_depth(task.depth));
        task.set_storage_tier(tier);
        let scheduled = ScheduledTask::new(task);
        self.queue_for(tier).push(scheduled);
    }

    /// Retrieve the next task respecting tier priorities and deadlines.
    pub fn next_task(&self) -> Option<ActivationTask> {
        let now = Instant::now();
        let deadlines = {
            let guard = match self.timing.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            guard.deadlines
        };

        let hot_deadline = deadlines[StorageTier::Hot as usize];
        let warm_deadline = deadlines[StorageTier::Warm as usize];
        let cold_deadline = deadlines[StorageTier::Cold as usize];

        if self.priority_hot_tier {
            if let Some(task) = self.queue_for(StorageTier::Hot).pop(now, hot_deadline) {
                return Some(task.into_inner());
            }
        }

        if let Some(task) = self.queue_for(StorageTier::Warm).pop(now, warm_deadline) {
            return Some(task.into_inner());
        }

        if !self.priority_hot_tier {
            if let Some(task) = self.queue_for(StorageTier::Hot).pop(now, hot_deadline) {
                return Some(task.into_inner());
            }
        }

        if self.bypass_cold.load(Ordering::Relaxed) {
            return None;
        }

        self.queue_for(StorageTier::Cold)
            .pop(now, cold_deadline)
            .map_or_else(
                || {
                    if now > cold_deadline {
                        self.bypass_cold.store(true, Ordering::Relaxed);
                        self.queue_for(StorageTier::Cold).mark_deadline_miss();
                    }
                    None
                },
                |task| Some(task.into_inner()),
            )
    }

    /// Mark tier work as complete so concurrency limits can admit new tasks.
    pub fn finish_task(&self, tier: StorageTier) {
        self.queue_for(tier).finish();
    }

    /// Whether all queues are empty and no work is in flight.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.queues.iter().all(TierQueue::is_idle)
    }

    /// Capture a snapshot of scheduler state for metrics aggregation.
    #[must_use]
    pub fn snapshot(&self) -> SchedulerSnapshot {
        let mut snapshot = SchedulerSnapshot {
            tiers: HashMap::new(),
            total_pending: 0,
        };

        for queue in &self.queues {
            let state = queue.snapshot();
            snapshot.total_pending += state.queued + state.in_flight;
            snapshot.tiers.insert(queue.tier, state);
        }

        snapshot
    }

    /// Access the latency budget manager for monitoring purposes.
    #[must_use]
    pub const fn latency_budget(&self) -> &LatencyBudgetManager {
        &self.latency_budget
    }

    const fn queue_for(&self, tier: StorageTier) -> &TierQueue {
        &self.queues[tier as usize]
    }
}

/// Internal bookkeeping for scheduler timing budgets.
struct SchedulerTiming {
    start: Instant,
    deadlines: [Instant; 3],
}

/// Wrapper around an activation task recording when it entered the queue.
struct ScheduledTask {
    task: ActivationTask,
    enqueued_at: Instant,
}

impl ScheduledTask {
    fn new(task: ActivationTask) -> Self {
        Self {
            task,
            enqueued_at: Instant::now(),
        }
    }

    fn into_inner(self) -> ActivationTask {
        self.task
    }
}

/// Tier-specific queue with bounded concurrency and deadline tracking.
struct TierQueue {
    tier: StorageTier,
    tasks: SegQueue<ScheduledTask>,
    queued: AtomicUsize,
    in_flight: AtomicUsize,
    max_in_flight: usize,
    timeout: Duration,
    deadline_missed: AtomicBool,
    /// When true, tasks are sorted by (depth, target_node, contribution) for deterministic execution
    deterministic: bool,
    /// Buffer for deterministic sorting - only used when deterministic=true
    deterministic_buffer: Mutex<Vec<ScheduledTask>>,
}

impl TierQueue {
    fn new(tier: StorageTier, timeout: Duration, max_in_flight: usize, deterministic: bool) -> Self {
        Self {
            tier,
            tasks: SegQueue::new(),
            queued: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            max_in_flight,
            timeout,
            deadline_missed: AtomicBool::new(false),
            deterministic,
            deterministic_buffer: Mutex::new(Vec::new()),
        }
    }

    fn push(&self, task: ScheduledTask) {
        self.tasks.push(task);
        self.queued.fetch_add(1, Ordering::Relaxed);
    }

    fn pop(&self, now: Instant, deadline: Instant) -> Option<ScheduledTask> {
        if now > deadline {
            self.mark_deadline_miss();
            return None;
        }

        if self.in_flight.load(Ordering::Relaxed) >= self.max_in_flight {
            return None;
        }

        // Deterministic mode: sort tasks before popping
        if self.deterministic {
            return self.pop_deterministic(now);
        }

        // Performance mode: standard FIFO popping
        while let Some(task) = self.tasks.pop() {
            self.queued.fetch_sub(1, Ordering::Relaxed);

            let waited = now.saturating_duration_since(task.enqueued_at);
            if waited > self.timeout {
                self.mark_deadline_miss();
                continue;
            }

            self.in_flight.fetch_add(1, Ordering::Relaxed);
            return Some(task);
        }

        None
    }

    /// Pop task in deterministic mode with canonical ordering
    fn pop_deterministic(&self, now: Instant) -> Option<ScheduledTask> {
        let mut buffer = match self.deterministic_buffer.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        // If buffer is empty, refill from queue with sorting
        if buffer.is_empty() {
            let mut tasks = Vec::new();
            while let Some(task) = self.tasks.pop() {
                self.queued.fetch_sub(1, Ordering::Relaxed);
                tasks.push(task);
            }

            if tasks.is_empty() {
                return None;
            }

            // Sort by (depth, target_node, contribution) for canonical ordering
            tasks.sort_by(|a, b| {
                a.task.depth
                    .cmp(&b.task.depth)
                    .then_with(|| a.task.target_node.cmp(&b.task.target_node))
                    .then_with(|| {
                        // Higher contribution first (reverse order)
                        b.task.contribution()
                            .partial_cmp(&a.task.contribution())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });

            *buffer = tasks;
        }

        // Pop from sorted buffer
        while let Some(task) = buffer.pop() {
            let waited = now.saturating_duration_since(task.enqueued_at);
            if waited > self.timeout {
                self.mark_deadline_miss();
                continue;
            }

            self.in_flight.fetch_add(1, Ordering::Relaxed);
            return Some(task);
        }

        None
    }

    fn finish(&self) {
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    fn is_idle(&self) -> bool {
        self.queued.load(Ordering::Relaxed) == 0
            && self.in_flight.load(Ordering::Relaxed) == 0
            && self.tasks.is_empty()
    }

    fn mark_deadline_miss(&self) {
        self.deadline_missed.store(true, Ordering::Relaxed);
    }

    fn snapshot(&self) -> TierQueueStateSnapshot {
        TierQueueStateSnapshot {
            tier: self.tier,
            queued: self.queued.load(Ordering::Relaxed),
            in_flight: self.in_flight.load(Ordering::Relaxed),
            deadline_missed: self.deadline_missed.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        while self.tasks.pop().is_some() {}
        self.queued.store(0, Ordering::Relaxed);
        self.in_flight.store(0, Ordering::Relaxed);
        self.deadline_missed.store(false, Ordering::Relaxed);

        // Clear deterministic buffer
        if self.deterministic {
            if let Ok(mut buffer) = self.deterministic_buffer.lock() {
                buffer.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn default_config() -> ParallelSpreadingConfig {
        ParallelSpreadingConfig {
            tier_timeouts: [
                Duration::from_millis(100),
                Duration::from_millis(200),
                Duration::from_millis(500),
            ],
            max_concurrent_per_tier: 4,
            priority_hot_tier: true,
            ..ParallelSpreadingConfig::default()
        }
    }

    fn make_task(id: &str, tier: StorageTier) -> ActivationTask {
        ActivationTask::new(id.to_string(), 1.0, 1.0, 1.0, 0, 4).with_storage_tier(tier)
    }

    #[test]
    fn scheduler_prioritizes_hot_tier() {
        let config = default_config();
        let scheduler = TierAwareSpreadingScheduler::from_config(&config);
        scheduler.reset(&config);

        scheduler.enqueue_task(make_task("cold", StorageTier::Cold));
        scheduler.enqueue_task(make_task("hot", StorageTier::Hot));

        let first = scheduler.next_task().expect("hot task");
        assert_eq!(first.target_node, "hot");
        scheduler.finish_task(StorageTier::Hot);

        let second = scheduler.next_task().expect("cold task");
        assert_eq!(second.target_node, "cold");
    }

    #[test]
    fn scheduler_respects_deadlines() {
        let config = ParallelSpreadingConfig {
            tier_timeouts: [
                Duration::from_micros(50),
                Duration::from_micros(50),
                Duration::from_millis(1),
            ],
            ..default_config()
        };
        let scheduler = TierAwareSpreadingScheduler::from_config(&config);
        scheduler.reset(&config);

        scheduler.enqueue_task(make_task("warm", StorageTier::Warm));

        // Exhaust warm deadline by sleeping beyond timeout.
        std::thread::sleep(Duration::from_micros(60));

        assert!(scheduler.next_task().is_none());
        let snapshot = scheduler.snapshot();
        assert!(
            snapshot
                .tiers
                .get(&StorageTier::Warm)
                .expect("warm tier snapshot")
                .deadline_missed
        );
    }

    #[test]
    fn scheduler_bypasses_cold_after_deadline() {
        let config = ParallelSpreadingConfig {
            tier_timeouts: [
                Duration::from_micros(50),
                Duration::from_micros(50),
                Duration::from_micros(100),
            ],
            ..default_config()
        };
        let scheduler = TierAwareSpreadingScheduler::from_config(&config);
        scheduler.reset(&config);

        scheduler.enqueue_task(make_task("cold", StorageTier::Cold));
        std::thread::sleep(Duration::from_micros(120));

        assert!(scheduler.next_task().is_none());
        let snapshot = scheduler.snapshot();
        assert!(
            snapshot
                .tiers
                .get(&StorageTier::Cold)
                .expect("cold tier snapshot")
                .deadline_missed
        );
    }
}
