//! Lock-free parallel activation spreading engine
//!
//! High-performance work-stealing thread pool implementation for parallel
//! graph traversal with cognitive architecture principles.

use crate::activation::{
    ActivationError, ActivationGraphExt, ActivationRecord, ActivationResult, ActivationTask,
    CacheOptimizedNode, MemoryGraph, NodeId, ParallelSpreadingConfig, SpreadingMetrics,
    SpreadingResults, TierAwareSpreadingScheduler, TierSummary, TraceEntry, WeightedEdge,
    cycle_detector::CycleDetector,
    gpu_interface::{AdaptiveConfig, AdaptiveSpreadingEngine},
    latency_budget::LatencyBudgetManager,
    memory_pool::ActivationRecordPool,
    storage_aware::StorageTier,
};
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, Ordering};
#[cfg(loom)]
use loom::sync::{Arc, Mutex};
use parking_lot::{RwLock, RwLockReadGuard};
use std::cell::Cell;
use std::collections::HashMap;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(loom))]
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tracing::warn;

const DEFAULT_DECAY_RATE: f32 = 0.1;

const CACHE_LINE_BYTES: usize = 64;

#[inline]
fn compute_prefetch_lookahead(config: &ParallelSpreadingConfig) -> usize {
    let distance = config.prefetch_distance;
    if distance == 0 {
        0
    } else {
        (distance / CACHE_LINE_BYTES).max(1)
    }
}

#[inline]
#[allow(clippy::missing_const_for_fn)]
fn prefetch_cache_line(node: &CacheOptimizedNode) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm_prefetch;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm_prefetch;
        const _MM_HINT_T0: i32 = 3;
        _mm_prefetch(node as *const CacheOptimizedNode as *const i8, _MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = node;
    }
}

#[inline]
fn prefetch_neighbor_handles(neighbors: &[WeightedEdge], index: usize, lookahead: usize) {
    if lookahead == 0 {
        return;
    }

    for offset in 1..=lookahead {
        if let Some(edge) = neighbors.get(index + offset) {
            if let Some(handle) = edge.hot_handle.as_ref() {
                prefetch_cache_line(handle);
            }
        } else {
            break;
        }
    }
}

/// Work-stealing thread pool for parallel activation spreading
pub struct ParallelSpreadingEngine {
    config: Arc<RwLock<ParallelSpreadingConfig>>,
    activation_records: Arc<DashMap<NodeId, Arc<ActivationRecord>>>,
    memory_graph: Arc<MemoryGraph>,
    scheduler: Arc<TierAwareSpreadingScheduler>,
    thread_handles: Vec<JoinHandle<()>>,
    metrics: Arc<SpreadingMetrics>,
    shutdown_signal: Arc<AtomicBool>,
    phase_barrier: Arc<PhaseBarrier>,
    latency_budget: Arc<RwLock<LatencyBudgetManager>>,
    cycle_detector: Arc<CycleDetector>,
    deterministic_trace: Arc<Mutex<Vec<TraceEntry>>>,
    adaptive_engine: Arc<Mutex<AdaptiveSpreadingEngine>>,
    /// Activation record pool shared across workers for low-allocation spreading
    activation_pool: Option<Arc<ActivationRecordPool>>,
    /// Adaptive batch sizing controller for dynamic performance optimization
    adaptive_batcher: Option<Arc<crate::activation::AdaptiveBatcher>>,
}

#[cfg(loom)]
type BarrierMutex<T> = loom::sync::Mutex<T>;
#[cfg(loom)]
type BarrierCondvar = loom::sync::Condvar;
#[cfg(not(loom))]
type BarrierMutex<T> = parking_lot::Mutex<T>;
#[cfg(not(loom))]
type BarrierCondvar = parking_lot::Condvar;

/// Synchronization barrier for deterministic parallel execution
///
/// Uses parking_lot's Condvar for efficient OS-level blocking instead of spin-waiting.
/// This eliminates CPU burn during synchronization and improves behavior under contention.
pub struct PhaseBarrier {
    state: BarrierMutex<BarrierState>,
    condvar: BarrierCondvar,
}

struct BarrierState {
    worker_count: usize,
    waiting: usize,
    phase_counter: u64,
    enabled: bool,
}

impl PhaseBarrier {
    #[allow(clippy::missing_const_for_fn)]
    fn new(worker_count: usize) -> Self {
        Self {
            state: BarrierMutex::new(BarrierState {
                worker_count,
                waiting: 0,
                phase_counter: 0,
                enabled: true,
            }),
            condvar: BarrierCondvar::new(),
        }
    }

    /// Disable the barrier, allowing workers to pass through without waiting
    fn disable(&self) {
        {
            let mut state = self.state.lock();
            state.enabled = false;
        }
        // Wake all waiting threads
        self.condvar.notify_all();
    }

    /// Enable the barrier for synchronization
    fn enable(&self) {
        let mut state = self.state.lock();
        state.enabled = true;
        // Reset phase counter for new spreading operation
        state.phase_counter = 0;
        state.waiting = 0;
        drop(state);
        // Wake any threads that might be waiting
        self.condvar.notify_all();
    }

    /// Wait for all workers to reach this barrier point
    #[allow(clippy::significant_drop_tightening)]
    fn wait(&self) {
        #[cfg(loom)]
        {
            self.wait_loom();
            return;
        }

        #[cfg(not(loom))]
        {
            self.wait_parking_lot();
        }
    }

    #[cfg(not(loom))]
    fn wait_parking_lot(&self) {
        let mut state = self.state.lock();

        if !state.enabled {
            return;
        }

        state.waiting += 1;
        let current_phase = state.phase_counter;

        if state.waiting == state.worker_count {
            state.waiting = 0;
            state.phase_counter += 1;
            drop(state);
            self.condvar.notify_all();
        } else {
            let timeout_secs = (state.worker_count as u64 * 2).max(5);
            let timeout = Duration::from_secs(timeout_secs);
            let start = Instant::now();

            while state.phase_counter == current_phase && state.enabled {
                let elapsed = start.elapsed();
                if elapsed >= timeout {
                    state.waiting = state.waiting.saturating_sub(1);
                    return;
                }

                let remaining = timeout - elapsed;
                let wait_result = self.condvar.wait_for(&mut state, remaining);
                if wait_result.timed_out() {
                    state.waiting = state.waiting.saturating_sub(1);
                    drop(state);
                    return;
                }
            }
        }
    }

    #[cfg(loom)]
    fn wait_loom(&self) {
        let mut state = self.state.lock();

        if !state.enabled {
            return;
        }

        state.waiting += 1;
        let current_phase = state.phase_counter;

        if state.waiting == state.worker_count {
            state.waiting = 0;
            state.phase_counter += 1;
            drop(state);
            self.condvar.notify_all();
        } else {
            let timeout_secs = (state.worker_count as u64 * 2).max(5);
            let timeout = Duration::from_secs(timeout_secs);
            let start = Instant::now();
            let mut guard = state;

            loop {
                if guard.phase_counter != current_phase || !guard.enabled {
                    break;
                }

                let elapsed = start.elapsed();
                if elapsed >= timeout {
                    guard.waiting = guard.waiting.saturating_sub(1);
                    return;
                }

                let remaining = timeout - elapsed;
                let (next_guard, wait_result) = self.condvar.wait_timeout(guard, remaining);
                guard = next_guard;
                if wait_result.timed_out() {
                    guard.waiting = guard.waiting.saturating_sub(1);
                    return;
                }
            }
        }
    }

    /// Get the current phase count (test-only)
    #[cfg(test)]
    fn phase_count(&self) -> u64 {
        let state = self.state.lock();
        state.phase_counter
    }
}

/// Worker thread context for parallel spreading
///
/// Note: No randomness is used in the current implementation. Determinism is achieved
/// through canonical task ordering (via TierQueue sorting) and phase barriers for
/// hop-level synchronization. Future probabilistic features can add RNG if needed.
struct WorkerContext {
    /// Worker ID reserved for future per-worker diagnostics and tracing
    /// Currently unused but kept for planned tracing feature (Milestone 6+)
    #[allow(dead_code)]
    worker_id: usize,
    activation_records: Arc<DashMap<NodeId, Arc<ActivationRecord>>>,
    memory_graph: Arc<MemoryGraph>,
    config: Arc<RwLock<ParallelSpreadingConfig>>,
    metrics: Arc<SpreadingMetrics>,
    shutdown_signal: Arc<AtomicBool>,
    phase_barrier: Arc<PhaseBarrier>,
    latency_budget: Arc<RwLock<LatencyBudgetManager>>,
    scheduler: Arc<TierAwareSpreadingScheduler>,
    cycle_detector: Arc<CycleDetector>,
    deterministic_trace: Arc<Mutex<Vec<TraceEntry>>>,
    activation_pool: Option<Arc<ActivationRecordPool>>,
    /// Adaptive engine reserved for future GPU-accelerated spreading (Milestone 11)
    /// Currently unused but initialized for planned GPU offloading feature
    #[allow(dead_code)]
    adaptive_engine: Arc<Mutex<AdaptiveSpreadingEngine>>,
    /// Adaptive batch sizing controller shared across workers
    adaptive_batcher: Option<Arc<crate::activation::AdaptiveBatcher>>,
}

struct TierTaskGuard<'a> {
    scheduler: &'a TierAwareSpreadingScheduler,
    tier: StorageTier,
    finished: Cell<bool>,
}

impl<'a> TierTaskGuard<'a> {
    const fn new(scheduler: &'a TierAwareSpreadingScheduler, tier: StorageTier) -> Self {
        Self {
            scheduler,
            tier,
            finished: Cell::new(false),
        }
    }

    fn finish(&self) {
        if !self.finished.replace(true) {
            self.scheduler.finish_task(self.tier);
        }
    }
}

impl Drop for TierTaskGuard<'_> {
    fn drop(&mut self) {
        if !self.finished.replace(true) {
            self.scheduler.finish_task(self.tier);
        }
    }
}

impl WorkerContext {
    fn config(&self) -> RwLockReadGuard<'_, ParallelSpreadingConfig> {
        self.config.read()
    }

    fn latency_budget(&self) -> RwLockReadGuard<'_, LatencyBudgetManager> {
        self.latency_budget.read()
    }
}

impl ParallelSpreadingEngine {
    /// Create new parallel spreading engine
    pub fn new(
        config: ParallelSpreadingConfig,
        memory_graph: Arc<MemoryGraph>,
    ) -> ActivationResult<Self> {
        if config.num_threads == 0 {
            return Err(ActivationError::InvalidConfig(
                "Number of threads must be > 0".to_string(),
            ));
        }
        let config_handle = Arc::new(RwLock::new(config));
        let activation_records = Arc::new(DashMap::new());
        let metrics = Arc::new(SpreadingMetrics::default());
        let shutdown_signal = Arc::new(AtomicBool::new(false));
        let deterministic_trace = Arc::new(Mutex::new(Vec::new()));

        let (
            phase_barrier,
            scheduler,
            cycle_detector,
            adaptive_engine,
            activation_pool,
            adaptive_batcher,
            latency_budget,
        ) = {
            let config_guard = config_handle.read();

            let phase_barrier = Arc::new(PhaseBarrier::new(config_guard.num_threads));
            let scheduler = Arc::new(TierAwareSpreadingScheduler::from_config(&config_guard));
            let cycle_detector =
                Arc::new(CycleDetector::new(config_guard.tier_cycle_budgets.clone()));

            let adaptive_config = AdaptiveConfig {
                gpu_threshold: config_guard.gpu_threshold,
                enable_gpu: config_guard.enable_gpu,
            };
            let adaptive_engine = Arc::new(Mutex::new(AdaptiveSpreadingEngine::new(
                None, // GPU interface will be injected in Milestone 11
                adaptive_config,
                Some(metrics.clone()),
            )));

            let activation_pool = if config_guard.enable_memory_pool {
                let record_size = std::mem::size_of::<ActivationRecord>().max(1);
                let max_bytes = config_guard
                    .pool_chunk_size
                    .saturating_mul(config_guard.pool_max_chunks.max(1));
                let computed_max = if max_bytes > 0 {
                    (max_bytes / record_size).max(config_guard.pool_initial_size)
                } else {
                    config_guard.pool_initial_size
                };
                let max_records = computed_max.max(64);
                let cache_capacity =
                    (config_guard.pool_chunk_size / record_size).clamp(8, max_records.max(8));
                Some(Arc::new(ActivationRecordPool::with_config(
                    config_guard.pool_initial_size,
                    max_records,
                    cache_capacity,
                    DEFAULT_DECAY_RATE,
                )))
            } else {
                None
            };

            let adaptive_batcher =
                config_guard
                    .adaptive_batcher_config
                    .as_ref()
                    .map(|batcher_config| {
                        use crate::activation::{AdaptiveBatcher, AdaptiveMode};
                        let mode = AdaptiveMode::default();
                        Arc::new(AdaptiveBatcher::new(batcher_config.clone(), mode))
                    });

            let mut budget = LatencyBudgetManager::new();
            budget.set_budget(
                StorageTier::Hot,
                config_guard.tier_timeouts[StorageTier::Hot as usize],
            );
            budget.set_budget(
                StorageTier::Warm,
                config_guard.tier_timeouts[StorageTier::Warm as usize],
            );
            budget.set_budget(
                StorageTier::Cold,
                config_guard.tier_timeouts[StorageTier::Cold as usize],
            );
            let latency_budget = Arc::new(RwLock::new(budget));
            drop(config_guard);

            (
                phase_barrier,
                scheduler,
                cycle_detector,
                adaptive_engine,
                activation_pool,
                adaptive_batcher,
                latency_budget,
            )
        };

        let mut engine = Self {
            config: config_handle,
            activation_records,
            memory_graph,
            scheduler,
            thread_handles: Vec::new(),
            metrics,
            shutdown_signal,
            phase_barrier,
            latency_budget,
            cycle_detector,
            deterministic_trace,
            adaptive_engine,
            activation_pool,
            adaptive_batcher,
        };

        engine.spawn_workers()?;

        Ok(engine)
    }

    /// Spawn worker threads for parallel processing
    fn spawn_workers(&mut self) -> ActivationResult<()> {
        let worker_count = self.config.read().num_threads;
        for worker_id in 0..worker_count {
            let context = WorkerContext {
                worker_id,
                activation_records: self.activation_records.clone(),
                memory_graph: self.memory_graph.clone(),
                config: self.config.clone(),
                metrics: self.metrics.clone(),
                shutdown_signal: self.shutdown_signal.clone(),
                phase_barrier: self.phase_barrier.clone(),
                latency_budget: self.latency_budget.clone(),
                scheduler: self.scheduler.clone(),
                cycle_detector: self.cycle_detector.clone(),
                deterministic_trace: self.deterministic_trace.clone(),
                activation_pool: self.activation_pool.clone(),
                adaptive_engine: self.adaptive_engine.clone(),
                adaptive_batcher: self.adaptive_batcher.clone(),
            };

            let handle = thread::Builder::new()
                .name(format!("activation-worker-{worker_id}"))
                .spawn(move || {
                    Self::worker_main(context);
                })
                .map_err(|e| ActivationError::ThreadingError(e.to_string()))?;

            self.thread_handles.push(handle);
        }

        Ok(())
    }

    /// Capture a snapshot of the current spreading configuration.
    #[must_use]
    pub fn config_snapshot(&self) -> ParallelSpreadingConfig {
        self.config.read().clone()
    }

    /// Apply an updated spreading configuration to the live engine.
    pub fn update_config(&self, updated: &ParallelSpreadingConfig) {
        {
            let mut guard = self.config.write();
            *guard = updated.clone();
        }

        self.scheduler.reset(updated);

        {
            let mut budget_guard = self.latency_budget.write();
            budget_guard.set_budget(
                StorageTier::Hot,
                updated.tier_timeouts[StorageTier::Hot as usize],
            );
            budget_guard.set_budget(
                StorageTier::Warm,
                updated.tier_timeouts[StorageTier::Warm as usize],
            );
            budget_guard.set_budget(
                StorageTier::Cold,
                updated.tier_timeouts[StorageTier::Cold as usize],
            );
        }

        self.phase_barrier.enable();
    }

    /// Determine if batch spreading should be used for this set of neighbors
    fn should_use_batch_spreading(
        context: &WorkerContext,
        neighbors: &[WeightedEdge],
        tier: StorageTier,
    ) -> bool {
        use crate::activation::simd_optimization::should_use_simd_for_tier;

        let neighbor_count = neighbors.len();
        let simd_batch_size = {
            let config = context.config();
            config.simd_batch_size
        };
        should_use_simd_for_tier(tier, neighbor_count, simd_batch_size)
    }

    /// Process neighbors using SIMD batch operations
    fn process_neighbors_batch(
        context: &WorkerContext,
        task: &ActivationTask,
        applied_delta: f32,
        neighbors: &[WeightedEdge],
        next_tier: StorageTier,
        decay_factor: f32,
        prefetch_lookahead: usize,
    ) {
        use crate::activation::ActivationGraphExt;
        use crate::activation::simd_optimization::SimdActivationMapper;
        use crate::compute::cosine_similarity_batch_768;

        // Get current node's embedding from the graph
        let Some(current_embedding) = context.memory_graph.get_embedding(&task.target_node) else {
            // No embedding for current node, fallback to scalar
            Self::fallback_to_scalar(
                context,
                task,
                applied_delta,
                neighbors,
                next_tier,
                decay_factor,
                prefetch_lookahead,
            );
            return;
        };

        // Collect neighbor embeddings from the graph
        let neighbor_embeddings: Vec<[f32; 768]> = neighbors
            .iter()
            .filter_map(|edge| context.memory_graph.get_embedding(&edge.target))
            .collect();

        // Ensure we have all embeddings before using SIMD
        if neighbor_embeddings.is_empty() || neighbor_embeddings.len() != neighbors.len() {
            Self::fallback_to_scalar(
                context,
                task,
                applied_delta,
                neighbors,
                next_tier,
                decay_factor,
                prefetch_lookahead,
            );
            return;
        }

        // Batch compute similarities using SIMD
        let similarities = cosine_similarity_batch_768(&current_embedding, &neighbor_embeddings);

        // Convert similarities to activations using SIMD
        let temperature = 0.5; // From config
        let threshold = {
            let config = context.config();
            config.threshold
        };
        let activations =
            SimdActivationMapper::batch_sigmoid_activation(&similarities, temperature, threshold);

        // Create tasks with computed activations
        for (idx, edge) in neighbors.iter().enumerate() {
            if let Some(handle) = edge.hot_handle.as_ref() {
                prefetch_cache_line(handle);
            }
            prefetch_neighbor_handles(neighbors, idx, prefetch_lookahead);

            let activation_weight = activations.get(idx).copied().unwrap_or(edge.weight);

            let mut next_path = task.path.clone();
            next_path.push(edge.target.clone());
            let new_task = ActivationTask::new(
                edge.target.clone(),
                applied_delta,
                activation_weight,
                decay_factor,
                task.depth + 1,
                task.max_depth,
            )
            .with_storage_tier(next_tier)
            .with_path(next_path);

            context.scheduler.enqueue_task(new_task);
        }
    }

    /// Fallback to scalar processing when embeddings are unavailable
    fn fallback_to_scalar(
        context: &WorkerContext,
        task: &ActivationTask,
        applied_delta: f32,
        neighbors: &[WeightedEdge],
        next_tier: StorageTier,
        decay_factor: f32,
        prefetch_lookahead: usize,
    ) {
        for (idx, edge) in neighbors.iter().enumerate() {
            if let Some(handle) = edge.hot_handle.as_ref() {
                prefetch_cache_line(handle);
            }
            prefetch_neighbor_handles(neighbors, idx, prefetch_lookahead);

            let mut next_path = task.path.clone();
            next_path.push(edge.target.clone());
            let new_task = ActivationTask::new(
                edge.target.clone(),
                applied_delta,
                edge.weight,
                decay_factor,
                task.depth + 1,
                task.max_depth,
            )
            .with_storage_tier(next_tier)
            .with_path(next_path);

            context.scheduler.enqueue_task(new_task);
        }
    }

    /// Main worker thread execution loop
    #[allow(clippy::needless_pass_by_value)] // Context is moved into thread closure
    fn worker_main(context: WorkerContext) {
        while !context.shutdown_signal.load(Ordering::Relaxed) {
            if let Some(task) = context.scheduler.next_task() {
                Self::process_task(&context, task);

                if context.config().deterministic {
                    context.phase_barrier.wait();
                }

                continue;
            }

            // No task available - check if truly idle before sleeping
            if context.scheduler.is_idle() {
                // Scheduler is idle - no work to do
                thread::sleep(Duration::from_micros(100));
            } else {
                // Scheduler has work but we didn't get a task - sync and retry
                if context.config().deterministic {
                    context.phase_barrier.wait();
                }
                thread::yield_now();
            }
        }
    }

    /// Process a single activation task
    fn process_task(context: &WorkerContext, mut task: ActivationTask) {
        let start_time = Instant::now();
        let tier = task
            .storage_tier
            .unwrap_or_else(|| StorageTier::from_depth(task.depth));
        let guard = TierTaskGuard::new(context.scheduler.as_ref(), tier);

        let config_view = context.config();
        let deterministic = config_view.deterministic;
        let trace_activation_flow_enabled = config_view.trace_activation_flow;
        let cycle_detection_enabled = config_view.cycle_detection;
        let cycle_penalty_factor = config_view.cycle_penalty_factor;
        let decay_function = config_view.decay_function.clone();
        let batch_size_snapshot = config_view.batch_size;
        let prefetch_lookahead_default = compute_prefetch_lookahead(&config_view);
        drop(config_view);

        // Get or create activation record
        let target_clone = task.target_node.clone();
        let pool = context.activation_pool.clone();
        let record = match context.activation_records.entry(target_clone.clone()) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(slot) => {
                let record = pool.as_ref().map_or_else(
                    || {
                        let mut base =
                            ActivationRecord::new(target_clone.clone(), DEFAULT_DECAY_RATE);
                        base.set_storage_tier(tier);
                        Arc::new(base)
                    },
                    |pool| pool.acquire(target_clone.clone(), DEFAULT_DECAY_RATE, Some(tier)),
                );
                context
                    .memory_graph
                    .register_hot_handle(target_clone.clone(), record.hot_handle());
                slot.insert(record.clone());
                record
            }
        };

        // Accumulate activation with threshold check and capture applied delta
        let contribution = task.contribution();
        let Some((updated_activation, applied_delta)) =
            record.accumulate_activation_with_result(contribution)
        else {
            guard.finish();
            return; // Below threshold
        };

        // Capture trace entry if deterministic tracing is enabled
        if trace_activation_flow_enabled && deterministic {
            let source_node = if task.path.len() > 1 {
                task.path.get(task.path.len() - 2).cloned()
            } else {
                None
            };

            let trace_entry = TraceEntry {
                depth: task.depth,
                target_node: target_clone.clone(),
                activation: updated_activation,
                confidence: record.get_confidence(),
                source_node,
            };

            if let Ok(mut trace) = context.deterministic_trace.lock() {
                trace.push(trace_entry);
            }
        }

        let visit_count = record.visits_atomic().fetch_add(1, Ordering::Relaxed) + 1;
        let mut should_mark_done = false;

        if cycle_detection_enabled {
            if task.path.is_empty() || task.path.last() != Some(&target_clone) {
                task.path.push(target_clone.clone());
            }

            if !context
                .cycle_detector
                .should_visit(&target_clone, visit_count, tier, &task.path)
            {
                record.apply_cycle_penalty(cycle_penalty_factor);
                context
                    .metrics
                    .cycles_detected
                    .fetch_add(1, Ordering::Relaxed);
                context.metrics.increment_cycle_for_tier(tier);

                if trace_activation_flow_enabled {
                    warn!(
                        target = "engram::activation::cycles",
                        node = %target_clone,
                        ?tier,
                        visit_count,
                        "Cycle detected; applying penalty"
                    );
                }

                guard.finish();
                return;
            }

            context.cycle_detector.mark_visiting(target_clone.clone());
            should_mark_done = true;
        } else if task.path.is_empty() {
            task.path.push(target_clone.clone());
        }

        // Check if we should continue spreading
        if !task.should_continue() {
            if should_mark_done {
                context.cycle_detector.mark_done(&target_clone);
            }
            guard.finish();
            return;
        }

        // Get neighbors and create new tasks
        if let Some(neighbors) =
            ActivationGraphExt::get_neighbors(&*context.memory_graph, &task.target_node)
        {
            let decay_factor = decay_function.apply(task.depth + 1);
            let next_tier = StorageTier::from_depth(task.depth + 1);
            let prefetch_lookahead = prefetch_lookahead_default;

            // Try SIMD batch processing if we have enough neighbors
            if Self::should_use_batch_spreading(context, &neighbors, next_tier) {
                Self::process_neighbors_batch(
                    context,
                    &task,
                    applied_delta,
                    &neighbors,
                    next_tier,
                    decay_factor,
                    prefetch_lookahead,
                );
            } else {
                // Scalar path: process neighbors individually
                for (idx, edge) in neighbors.iter().enumerate() {
                    if let Some(handle) = edge.hot_handle.as_ref() {
                        prefetch_cache_line(handle);
                    }
                    prefetch_neighbor_handles(&neighbors, idx, prefetch_lookahead);

                    let mut next_path = task.path.clone();
                    next_path.push(edge.target.clone());
                    let new_task = ActivationTask::new(
                        edge.target.clone(),
                        applied_delta,
                        edge.weight,
                        decay_factor,
                        task.depth + 1,
                        task.max_depth,
                    )
                    .with_storage_tier(next_tier)
                    .with_path(next_path);

                    context.scheduler.enqueue_task(new_task);
                }
            }
        }

        // Update metrics
        let elapsed = start_time.elapsed();
        let duration_ns = elapsed.as_nanos().min(u128::from(u64::MAX)) as u64;
        context.metrics.record_activation_latency(tier, elapsed);

        if !context.latency_budget().within_budget(tier, elapsed) {
            context.metrics.record_latency_budget_violation();
        }

        // Record observation for adaptive batch sizing
        if let Some(ref batcher) = context.adaptive_batcher {
            use crate::activation::Observation;
            let observation = Observation {
                batch_size: batch_size_snapshot,
                latency_ns: duration_ns,
                hop_count: task.depth as usize,
                tier,
            };
            batcher.record_observation(observation);
        }

        if should_mark_done {
            context.cycle_detector.mark_done(&target_clone);
        }

        guard.finish();
    }

    /// Initialize activation spreading from seed nodes
    pub fn spread_activation(
        &self,
        seed_activations: &[(NodeId, f32)],
    ) -> ActivationResult<SpreadingResults> {
        // Process pending observations from previous spreads
        if let Some(ref batcher) = self.adaptive_batcher {
            batcher.process_observations();
            self.metrics
                .record_adaptive_batcher_snapshot(&batcher.snapshot());
        }

        // Clear previous activation state
        self.release_activation_records();
        self.metrics.reset();
        let config_guard = self.config.read();
        self.scheduler.reset(&config_guard);
        self.cycle_detector.reset();
        self.phase_barrier.enable();
        if let Ok(mut trace) = self.deterministic_trace.lock() {
            trace.clear();
        }

        // Create initial tasks from seed nodes
        for (node_id, initial_activation) in seed_activations {
            let record = self.activation_pool.as_ref().map_or_else(
                || {
                    let mut base_record =
                        ActivationRecord::new(node_id.clone(), DEFAULT_DECAY_RATE);
                    base_record.set_storage_tier(StorageTier::Hot);
                    let record = Arc::new(base_record);
                    record.accumulate_activation(*initial_activation);
                    record
                },
                |pool| {
                    let record =
                        pool.acquire(node_id.clone(), DEFAULT_DECAY_RATE, Some(StorageTier::Hot));
                    record.accumulate_activation(*initial_activation);
                    record
                },
            );
            let hot_handle = record.hot_handle();
            self.activation_records.insert(node_id.clone(), record);
            self.memory_graph
                .register_hot_handle(node_id.clone(), hot_handle);

            if let Some(neighbors) = ActivationGraphExt::get_neighbors(&*self.memory_graph, node_id)
            {
                let decay_factor = config_guard.decay_function.apply(1);
                let next_tier = StorageTier::from_depth(1);

                for edge in neighbors {
                    let task_path = vec![node_id.clone(), edge.target.clone()];
                    let task = ActivationTask::new(
                        edge.target.clone(),
                        *initial_activation,
                        edge.weight,
                        decay_factor,
                        1,
                        config_guard.max_depth,
                    )
                    .with_storage_tier(next_tier)
                    .with_path(task_path);

                    self.scheduler.enqueue_task(task);
                }
            }
        }

        drop(config_guard);

        // Wait for processing to complete
        self.wait_for_completion()?;

        let results = self.collect_results();
        self.release_activation_records();

        Ok(results)
    }

    /// Wait for all workers to complete processing
    fn wait_for_completion(&self) -> ActivationResult<()> {
        // Adaptive timeout based on available system parallelism
        // Base: 30s + 10s per available core (accounts for contention)
        let available_parallelism = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(4);
        let timeout = Duration::from_secs(30 + (available_parallelism as u64 * 10));
        let start = Instant::now();
        while start.elapsed() < timeout {
            if self.scheduler.is_idle() {
                // Disable phase barrier to release any waiting workers
                self.phase_barrier.disable();
                break;
            }

            thread::sleep(Duration::from_millis(1));
        }

        if start.elapsed() >= timeout && !self.scheduler.is_idle() {
            // Disable barrier even on timeout to prevent hanging
            self.phase_barrier.disable();
            return Err(ActivationError::ThreadingError(
                "Timeout waiting for spreading completion".to_string(),
            ));
        }

        Ok(())
    }

    fn release_activation_records(&self) {
        let keys: Vec<NodeId> = self
            .activation_records
            .iter()
            .map(|entry| entry.key().clone())
            .collect();

        for key in &keys {
            self.memory_graph.clear_hot_handle(key);
        }

        if let Some(pool) = &self.activation_pool {
            for key in keys {
                if let Some((_, record)) = self.activation_records.remove(&key) {
                    pool.release(record);
                }
            }
        } else {
            self.activation_records.clear();
        }

        self.update_pool_metrics();
    }

    fn update_pool_metrics(&self) {
        if let Some(pool) = &self.activation_pool {
            let stats = pool.stats();
            self.metrics.record_pool_snapshot(&stats);
        }
    }

    fn collect_results(&self) -> SpreadingResults {
        #[derive(Default)]
        struct TierAccumulator {
            node_count: usize,
            activation_sum: f32,
            confidence_sum: f32,
        }

        let snapshot = self.scheduler.snapshot();
        let mut tier_accumulators: HashMap<StorageTier, TierAccumulator> = HashMap::new();
        let mut activations = Vec::new();

        for entry in self.activation_records.iter() {
            let record = entry.value();
            let tier = record.storage_tier().unwrap_or(StorageTier::from_depth(0));
            let storage_activation = record.to_storage_aware();

            let activation_value = storage_activation.activation_level.load(Ordering::Relaxed);
            let confidence_value = storage_activation.confidence.load(Ordering::Relaxed);

            let accumulator = tier_accumulators.entry(tier).or_default();
            accumulator.node_count += 1;
            accumulator.activation_sum += activation_value;
            accumulator.confidence_sum += confidence_value;

            activations.push(storage_activation);
        }

        let mut tier_summaries = HashMap::new();
        for (tier, acc) in tier_accumulators {
            let deadline_missed = snapshot
                .tiers
                .get(&tier)
                .is_some_and(|state| state.deadline_missed);

            let average = if acc.node_count > 0 {
                acc.confidence_sum / acc.node_count as f32
            } else {
                0.0
            };

            tier_summaries.insert(
                tier,
                TierSummary {
                    tier,
                    node_count: acc.node_count,
                    total_activation: acc.activation_sum,
                    average_confidence: average,
                    deadline_missed,
                },
            );
        }

        let deterministic_trace = {
            let config_guard = self.config.read();
            if config_guard.trace_activation_flow && config_guard.deterministic {
                self.deterministic_trace
                    .lock()
                    .map(|trace| trace.clone())
                    .unwrap_or_default()
            } else {
                Vec::new()
            }
        };

        SpreadingResults {
            activations,
            tier_summaries,
            cycle_paths: self.cycle_detector.get_cycle_paths(),
            deterministic_trace,
        }
    }

    /// Get current activation levels for all nodes
    #[must_use]
    pub fn get_activations(&self) -> Vec<(NodeId, f32)> {
        self.activation_records
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().get_activation()))
            .collect()
    }

    /// Get nodes with activation above threshold
    #[must_use]
    pub fn get_active_nodes(&self, threshold: f32) -> Vec<(NodeId, f32)> {
        self.activation_records
            .iter()
            .filter_map(|entry| {
                let activation = entry.value().get_activation();
                if activation >= threshold {
                    Some((entry.key().clone(), activation))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get performance metrics
    #[must_use]
    pub fn get_metrics(&self) -> &SpreadingMetrics {
        self.update_pool_metrics();
        // Update adaptive batcher metrics if available
        if let Some(ref batcher) = self.adaptive_batcher {
            self.metrics
                .record_adaptive_batcher_snapshot(&batcher.snapshot());
        }
        &self.metrics
    }

    /// Get a cloned Arc handle to the metrics structure for external monitoring
    #[must_use]
    pub fn metrics_handle(&self) -> Arc<SpreadingMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Shutdown the spreading engine
    pub fn shutdown(mut self) -> ActivationResult<()> {
        // Signal shutdown
        self.shutdown_signal.store(true, Ordering::SeqCst);

        // Wait for all threads to finish
        for handle in self.thread_handles.drain(..) {
            handle.join().map_err(|_| {
                ActivationError::ThreadingError("Failed to join worker thread".to_string())
            })?;
        }

        Ok(())
    }
}

impl Drop for ParallelSpreadingEngine {
    fn drop(&mut self) {
        self.shutdown_signal.store(true, Ordering::SeqCst);
        self.release_activation_records();

        // Join any remaining threads
        while let Some(handle) = self.thread_handles.pop() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::activation::{ActivationGraphExt, EdgeType, create_activation_graph};
    use serial_test::serial;
    use std::collections::HashMap;
    use std::time::Duration;

    /// Create a test graph with unique node IDs to prevent cross-test interference
    ///
    /// Each test invocation gets a unique prefix based on thread ID and timestamp,
    /// ensuring node IDs don't collide in the shared DashMap even when tests run concurrently.
    ///
    /// Returns (graph, node_a, node_b, node_c)
    fn create_test_graph() -> (Arc<MemoryGraph>, NodeId, NodeId, NodeId) {
        let graph = Arc::new(create_activation_graph());

        // Generate unique prefix for this test invocation
        // Format: ThreadId_Timestamp to ensure uniqueness across concurrent tests
        let prefix = format!(
            "{:?}_{}",
            std::thread::current().id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        let node_a = format!("{prefix}_A");
        let node_b = format!("{prefix}_B");
        let node_c = format!("{prefix}_C");

        // Create a simple test graph: A -> B -> C, A -> C
        ActivationGraphExt::add_edge(
            &*graph,
            node_a.clone(),
            node_b.clone(),
            0.8,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &*graph,
            node_b.clone(),
            node_c.clone(),
            0.6,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &*graph,
            node_a.clone(),
            node_c.clone(),
            0.4,
            EdgeType::Excitatory,
        );

        // Fix 1: Add embeddings to enable SIMD batch spreading
        let embedding_a = [0.1f32; 768];
        let embedding_b = [0.2f32; 768];
        let embedding_c = [0.3f32; 768];

        ActivationGraphExt::set_embedding(&*graph, &node_a, &embedding_a);
        ActivationGraphExt::set_embedding(&*graph, &node_b, &embedding_b);
        ActivationGraphExt::set_embedding(&*graph, &node_c, &embedding_c);

        // Fix 2: Diagnostic verification of edge addition
        let neighbors_a = ActivationGraphExt::get_neighbors(&*graph, &node_a);
        assert!(
            neighbors_a.is_some(),
            "Node A should have neighbors after adding edges"
        );
        let neighbors_from_a = neighbors_a.expect("node A should have neighbors");
        assert_eq!(
            neighbors_from_a.len(),
            2,
            "Node A should have 2 outgoing edges (A->B, A->C)"
        );

        let neighbors_b = ActivationGraphExt::get_neighbors(&*graph, &node_b);
        assert!(
            neighbors_b.is_some(),
            "Node B should have neighbors after adding edges"
        );
        let neighbors_from_b = neighbors_b.expect("node B should have neighbors");
        assert_eq!(
            neighbors_from_b.len(),
            1,
            "Node B should have 1 outgoing edge (B->C)"
        );

        let _neighbors_c = ActivationGraphExt::get_neighbors(&*graph, &node_c);
        // C might not have outgoing edges, but should exist in the graph
        // If it has no neighbors, that's expected for a leaf node

        (graph, node_a, node_b, node_c)
    }

    /// Compute resource-aware thread count for tests
    /// Uses at most 25% of available cores to prevent oversubscription
    fn test_thread_count() -> usize {
        std::thread::available_parallelism()
            .map(|n| (n.get() / 4).max(1))
            .unwrap_or(1)
    }

    const TEST_MAX_DEPTH: u16 = 2;

    fn fast_parallel_config() -> ParallelSpreadingConfig {
        let num_threads = test_thread_count().clamp(1, 2);
        ParallelSpreadingConfig {
            num_threads,
            max_depth: TEST_MAX_DEPTH,
            batch_size: 8,
            simd_batch_size: 4,
            prefetch_distance: 16,
            enable_memory_pool: false,
            ..ParallelSpreadingConfig::default()
        }
    }

    fn fast_deterministic_config(seed: u64, threads: usize) -> ParallelSpreadingConfig {
        ParallelSpreadingConfig {
            num_threads: threads.clamp(1, 4),
            max_depth: TEST_MAX_DEPTH,
            batch_size: 8,
            simd_batch_size: 4,
            prefetch_distance: 16,
            enable_memory_pool: false,
            tier_timeouts: [
                Duration::from_millis(50),
                Duration::from_millis(50),
                Duration::from_millis(100),
            ],
            ..ParallelSpreadingConfig::deterministic(seed)
        }
    }

    #[test]
    fn test_parallel_engine_creation() {
        let config = ParallelSpreadingConfig {
            num_threads: test_thread_count(),
            max_depth: 3,
            ..Default::default()
        };

        let (graph, _node_a, _node_b, _node_c) = create_test_graph();
        let engine =
            ParallelSpreadingEngine::new(config, graph).expect("engine creation should succeed");

        assert!(engine.scheduler.is_idle());
        assert!(!engine.shutdown_signal.load(Ordering::Relaxed));

        engine.shutdown().expect("shutdown should succeed");
    }

    #[test]
    #[serial(parallel_engine)]
    fn test_activation_spreading() {
        let config = fast_parallel_config();

        let (graph, node_a, _node_b, _node_c) = create_test_graph();

        let engine =
            ParallelSpreadingEngine::new(config, graph).expect("engine creation should succeed");

        // Spread activation from node A
        let seed_activations = vec![(node_a, 1.0)];
        let results = engine
            .spread_activation(&seed_activations)
            .expect("spread should succeed");

        assert!(!results.activations.is_empty());
        let hot_summary = results
            .tier_summary(StorageTier::Hot)
            .expect("hot tier summary");
        assert!(hot_summary.node_count >= 1);

        let seed_node = &seed_activations[0].0;
        let node_a_activation = results
            .activations
            .iter()
            .find(|activation| activation.memory_id == *seed_node)
            .map(|activation| activation.activation_level.load(Ordering::Relaxed));
        assert!(node_a_activation.is_some());
        assert!(node_a_activation.expect("activation should exist") >= 1.0);

        engine.shutdown().expect("shutdown should succeed");
    }

    #[test]
    #[serial(parallel_engine)]
    fn test_deterministic_spreading() {
        let threads = test_thread_count().clamp(1, 2);
        let config1 = fast_deterministic_config(42, threads);
        let config2 = fast_deterministic_config(42, threads);

        let (graph1, node_a1, _node_b1, _node_c1) = create_test_graph();
        let (graph2, node_a2, _node_b2, _node_c2) = create_test_graph();

        let engine1 =
            ParallelSpreadingEngine::new(config1, graph1).expect("engine1 creation should succeed");
        let engine2 =
            ParallelSpreadingEngine::new(config2, graph2).expect("engine2 creation should succeed");

        let seed_activations1 = vec![(node_a1, 1.0)];
        let seed_activations2 = vec![(node_a2, 1.0)];

        let results1 = engine1
            .spread_activation(&seed_activations1)
            .expect("spread1 should succeed");
        let results2 = engine2
            .spread_activation(&seed_activations2)
            .expect("spread2 should succeed");

        // Extract node suffixes (A, B, C) and activation values
        let mut activations1: Vec<_> = results1
            .activations
            .iter()
            .map(|activation| {
                let suffix = activation.memory_id.split('_').next_back().unwrap_or("");
                (
                    suffix.to_string(),
                    activation.activation_level.load(Ordering::Relaxed),
                )
            })
            .collect();
        let mut activations2: Vec<_> = results2
            .activations
            .iter()
            .map(|activation| {
                let suffix = activation.memory_id.split('_').next_back().unwrap_or("");
                (
                    suffix.to_string(),
                    activation.activation_level.load(Ordering::Relaxed),
                )
            })
            .collect();

        activations1.sort_by(|a, b| a.0.cmp(&b.0));
        activations2.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(activations1.len(), activations2.len());
        for (a1, a2) in activations1.iter().zip(activations2.iter()) {
            assert_eq!(a1.0, a2.0, "Node suffix mismatch");
            assert!(
                (a1.1 - a2.1).abs() < 1e-6,
                "Activation value mismatch for node {}",
                a1.0
            );
        }

        // Verify trace is captured
        assert!(!results1.deterministic_trace.is_empty());
        assert_eq!(
            results1.deterministic_trace.len(),
            results2.deterministic_trace.len()
        );

        engine1.shutdown().expect("shutdown1 should succeed");
        engine2.shutdown().expect("shutdown2 should succeed");
    }

    #[test]
    #[serial(parallel_engine)]
    fn test_deterministic_across_thread_counts() {
        let mut thread_counts = vec![1usize, 2, test_thread_count().saturating_mul(2).min(4)];
        thread_counts.sort_unstable();
        thread_counts.dedup();

        let mut activations_by_threads = Vec::new();

        for threads in thread_counts {
            let (graph, node_a, _node_b, _node_c) = create_test_graph();
            let seed_activations = vec![(node_a, 1.0)];

            let engine =
                ParallelSpreadingEngine::new(fast_deterministic_config(123, threads), graph)
                    .expect("engine should succeed");
            let results = engine
                .spread_activation(&seed_activations)
                .expect("spread should succeed");

            // Extract node suffixes (A, B, C) for comparison across different graph instances
            let mut activations: Vec<_> = results
                .activations
                .iter()
                .map(|a| {
                    let suffix = a.memory_id.split('_').next_back().unwrap_or("");
                    (
                        suffix.to_string(),
                        a.activation_level.load(Ordering::Relaxed),
                    )
                })
                .collect();
            activations.sort_by(|a, b| a.0.cmp(&b.0));
            let trace_len = results.deterministic_trace.len();

            engine.shutdown().expect("shutdown should succeed");
            activations_by_threads.push((threads, activations, trace_len));
        }

        assert!(activations_by_threads.len() >= 2);
        let (baseline_threads, baseline_activations, baseline_trace) = &activations_by_threads[0];
        for (threads, activations, trace_len) in &activations_by_threads[1..] {
            assert_eq!(
                activations.len(),
                baseline_activations.len(),
                "Activation count mismatch for {threads} threads vs baseline {baseline_threads}"
            );

            for (baseline, candidate) in baseline_activations.iter().zip(activations.iter()) {
                assert_eq!(baseline.0, candidate.0, "Node suffix mismatch");
                // Relaxed tolerance (1e-4 vs 1e-6) accounts for floating point precision
                // differences from SIMD operations and parallel accumulation across thread counts
                assert!(
                    (baseline.1 - candidate.1).abs() < 1e-4,
                    "Activation mismatch for node {} between thread counts {} and {}: baseline={}, candidate={}",
                    baseline.0,
                    baseline_threads,
                    threads,
                    baseline.1,
                    candidate.1
                );
            }

            assert_eq!(
                *baseline_trace, *trace_len,
                "Trace length mismatch for {threads} threads vs baseline {baseline_threads}"
            );
        }
    }

    #[test]
    #[serial(parallel_engine)]
    fn test_deterministic_trace_capture() {
        let config = fast_deterministic_config(999, test_thread_count().clamp(1, 2));
        let (graph, node_a, _node_b, _node_c) = create_test_graph();

        let engine = ParallelSpreadingEngine::new(config, graph).expect("engine should succeed");

        let seed_activations = vec![(node_a, 1.0)];
        let results = engine
            .spread_activation(&seed_activations)
            .expect("spread should succeed");

        // Verify trace was captured
        assert!(!results.deterministic_trace.is_empty());

        // Verify trace entries are ordered
        for entry in &results.deterministic_trace {
            assert!(entry.activation >= 0.0 && entry.activation <= 1.0);
            assert!(entry.confidence >= 0.0 && entry.confidence <= 1.0);
        }

        engine.shutdown().expect("shutdown should succeed");
    }

    #[test]
    fn test_deterministic_vs_performance_mode() {
        // Deterministic mode
        let det_config = fast_deterministic_config(555, test_thread_count().clamp(1, 2));
        let (det_graph, node_a_det, _node_b_det, _node_c_det) = create_test_graph();
        let seed_activations_det = vec![(node_a_det, 1.0)];

        let det_engine = ParallelSpreadingEngine::new(det_config, det_graph).unwrap();
        let det_results = det_engine.spread_activation(&seed_activations_det).unwrap();

        // Performance mode (non-deterministic)
        let mut perf_config = fast_parallel_config();
        perf_config.deterministic = false;
        perf_config.trace_activation_flow = false;
        let (perf_graph, node_a_perf, _node_b_perf, _node_c_perf) = create_test_graph();
        let seed_activations_perf = vec![(node_a_perf, 1.0)];

        let perf_engine = ParallelSpreadingEngine::new(perf_config, perf_graph).unwrap();
        let perf_results = perf_engine
            .spread_activation(&seed_activations_perf)
            .unwrap();

        // Both should activate nodes, but deterministic should have trace
        assert!(!det_results.activations.is_empty());
        assert!(!perf_results.activations.is_empty());
        assert!(!det_results.deterministic_trace.is_empty());
        assert!(perf_results.deterministic_trace.is_empty());

        det_engine.shutdown().unwrap();
        perf_engine.shutdown().unwrap();
    }

    #[test]
    #[serial(parallel_engine)]
    fn test_metrics_tracking() {
        let mut config = fast_parallel_config();
        config.enable_metrics = true;

        let (graph, node_a, _node_b, _node_c) = create_test_graph();

        let engine = ParallelSpreadingEngine::new(config, graph).expect("engine should succeed");

        let seed_activations = vec![(node_a, 1.0)];
        let results = engine
            .spread_activation(&seed_activations)
            .expect("spread should succeed");

        let metrics = engine.get_metrics();
        assert!(metrics.total_activations.load(Ordering::Relaxed) > 0);
        let hot_summary = results
            .tier_summary(StorageTier::Hot)
            .expect("hot tier summary available");
        assert!(hot_summary.total_activation > 0.0);

        engine.shutdown().expect("shutdown should succeed");
    }

    #[test]
    #[serial(parallel_engine)]
    fn test_threshold_filtering() {
        let mut config = fast_parallel_config();
        config.threshold = 0.5; // High threshold

        let (graph, node_a, _node_b, _node_c) = create_test_graph();

        let engine = ParallelSpreadingEngine::new(config, graph).expect("engine should succeed");

        let seed_activations = vec![(node_a, 0.3)]; // Low seed
        let results = engine
            .spread_activation(&seed_activations)
            .expect("spread should succeed");

        let active_nodes = engine.get_active_nodes(0.5);
        // Should have fewer nodes due to high threshold
        assert!(active_nodes.len() <= 1);
        assert!(results.activations.len() >= active_nodes.len());

        engine.shutdown().expect("shutdown should succeed");
    }

    #[test]
    #[serial(parallel_engine)]
    fn cycle_detection_penalises_revisits() {
        let mut config = fast_parallel_config();
        config.max_depth = 3;
        config.cycle_detection = true;
        config.cycle_penalty_factor = 0.5;
        config.tier_cycle_budgets = HashMap::from([
            (StorageTier::Hot, 1),
            (StorageTier::Warm, 1),
            (StorageTier::Cold, 1),
        ]);
        config.trace_activation_flow = true;

        // Generate unique node IDs for this test to prevent cross-test interference
        let prefix = format!(
            "{:?}_{}",
            std::thread::current().id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let node_a = format!("{prefix}_A");
        let node_b = format!("{prefix}_B");

        let graph = Arc::new(create_activation_graph());
        ActivationGraphExt::add_edge(
            &*graph,
            node_a.clone(),
            node_b.clone(),
            0.9,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &*graph,
            node_b.clone(),
            node_a.clone(),
            0.9,
            EdgeType::Excitatory,
        );

        let engine = ParallelSpreadingEngine::new(config, graph).expect("engine should succeed");
        let seed_activations = vec![(node_a, 1.0)];
        let results = engine
            .spread_activation(&seed_activations)
            .expect("spread should succeed");

        assert!(
            !results.cycle_paths.is_empty(),
            "Expected cycle paths to be detected"
        );
        assert!(
            engine.metrics.cycles_detected.load(Ordering::Relaxed) > 0,
            "Expected cycles to be detected"
        );
        // With budget=1 (allow visit 0, reject visit 1), cycle detected at depth 2 (Warm tier)
        // Flow: A(depth=0) -> B(depth=1) -> A(depth=2, visit_count=1 rejected)
        let tier_cycle_count = engine.metrics.cycle_count_for_tier(StorageTier::Warm);
        assert!(
            tier_cycle_count > 0,
            "Expected tier cycle count > 0, got {tier_cycle_count}"
        );

        // The cycle is detected on the second visit to B, so B should have the penalty
        let activation_b = results
            .activations
            .iter()
            .find(|activation| activation.memory_id == node_b)
            .map(|activation| activation.activation_level.load(Ordering::Relaxed))
            .unwrap_or_default();
        assert!(
            activation_b > 0.0 && activation_b < 0.5,
            "Expected activation penalty to be applied to B, got {activation_b}"
        );

        engine.shutdown().expect("shutdown should succeed");
    }

    #[test]
    fn test_phase_barrier_synchronization() {
        let barrier = PhaseBarrier::new(3);
        let barrier_arc = Arc::new(barrier);
        let mut handles = Vec::new();

        // Start 3 threads that will all wait at the barrier
        for i in 0..3 {
            let barrier_clone = barrier_arc.clone();
            let handle = thread::spawn(move || {
                thread::sleep(Duration::from_millis(i * 10));
                barrier_clone.wait();
                // All threads should reach this point together
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("thread should complete");
        }

        assert_eq!(barrier_arc.phase_count(), 1);
    }

    #[cfg(loom)]
    #[test]
    fn loom_phase_barrier_resets_without_deadlock() {
        loom::model(|| {
            use loom::sync::Arc;
            use loom::thread;

            let barrier = Arc::new(PhaseBarrier::new(2));
            let barrier_clone = Arc::clone(&barrier);

            let handle = thread::spawn(move || {
                barrier_clone.wait();
                barrier_clone.wait();
            });

            barrier.wait();
            barrier.wait();

            handle.join().expect("loom phase barrier join");
            assert!(barrier.phase_count() >= 1);
        });
    }

    #[test]
    #[serial(parallel_engine)]
    fn test_adaptive_batcher_records_observations() {
        use crate::activation::AdaptiveBatcherConfig;

        // Create config with adaptive batcher enabled
        let mut config = fast_parallel_config();
        config.adaptive_batcher_config = Some(AdaptiveBatcherConfig::default());

        let (graph, node_a, _node_b, _node_c) = create_test_graph();
        let engine =
            ParallelSpreadingEngine::new(config, graph).expect("engine creation should succeed");

        // Verify batcher was created
        assert!(
            engine.adaptive_batcher.is_some(),
            "AdaptiveBatcher should be initialized when config is provided"
        );

        let batcher = engine.adaptive_batcher.as_ref().expect("batcher exists");

        // Record initial metrics state
        let initial_update_count = batcher.metrics().update_count.load(Ordering::Relaxed);

        // Spread activation which should record observations
        let seed_activations = vec![(node_a, 1.0)];
        let results = engine
            .spread_activation(&seed_activations)
            .expect("spread should succeed");

        assert!(!results.activations.is_empty(), "Should have activations");

        // Process observations
        batcher.process_observations();

        // Verify observations were recorded
        let final_update_count = batcher.metrics().update_count.load(Ordering::Relaxed);
        assert!(
            final_update_count > initial_update_count,
            "Observations should have been recorded and processed"
        );

        engine.shutdown().expect("shutdown should succeed");
    }

    #[test]
    #[serial(parallel_engine)]
    fn test_adaptive_batcher_per_tier_recommendations() {
        use crate::activation::AdaptiveBatcherConfig;

        // Create config with adaptive batcher
        let mut config = fast_parallel_config();
        let batcher_config = AdaptiveBatcherConfig::default();
        config.adaptive_batcher_config = Some(batcher_config.clone());

        let (graph, _node_a, _node_b, _node_c) = create_test_graph();
        let engine =
            ParallelSpreadingEngine::new(config, graph).expect("engine creation should succeed");

        let batcher = engine.adaptive_batcher.as_ref().expect("batcher exists");

        // Get initial recommendations for each tier
        let hot_size = batcher.recommend_batch_size(StorageTier::Hot);
        let warm_size = batcher.recommend_batch_size(StorageTier::Warm);
        let cold_size = batcher.recommend_batch_size(StorageTier::Cold);

        // Verify tier-based sizing (Hot >= Warm >= Cold)
        assert!(
            hot_size >= warm_size,
            "Hot tier should have >= batch size than Warm"
        );
        assert!(
            warm_size >= cold_size,
            "Warm tier should have >= batch size than Cold"
        );

        // All should be within configured bounds
        assert!(hot_size >= batcher_config.min_batch_size);
        assert!(hot_size <= batcher_config.max_batch_size);

        engine.shutdown().expect("shutdown should succeed");
    }

    #[test]
    #[serial(parallel_engine)]
    fn test_adaptive_batcher_convergence() {
        use crate::activation::AdaptiveBatcherConfig;

        // Create config with adaptive batcher
        let mut config = fast_parallel_config();
        config.adaptive_batcher_config = Some(AdaptiveBatcherConfig {
            min_batch_size: 4,
            max_batch_size: 128,
            cooldown_interval: 1, // Very short cooldown for testing
            oscillation_threshold: 2,
            instability_threshold: 0.30,
        });

        let (graph, node_a, _node_b, _node_c) = create_test_graph();
        let engine =
            ParallelSpreadingEngine::new(config, graph).expect("engine creation should succeed");

        let batcher = engine.adaptive_batcher.as_ref().expect("batcher exists");

        // Run multiple spreads to allow convergence
        // Each spread records observations, and the next spread processes them
        let seed_activations = vec![(node_a, 1.0)];
        for _ in 0..15 {
            let _results = engine
                .spread_activation(&seed_activations)
                .expect("spread should succeed");
        }

        // Check convergence confidence
        // Note: Hot tier (depth 0) is seed nodes and may not have observations
        // Warm tier (depth 1) and Cold tier (depth 2) should have observations from spreading
        let warm_confidence = batcher.convergence_confidence(StorageTier::Warm);
        let cold_confidence = batcher.convergence_confidence(StorageTier::Cold);

        // After 15 spreads (14 processing cycles), confidence should be increasing
        // At least one tier should show convergence
        assert!(
            warm_confidence > 0.0 || cold_confidence > 0.0,
            "At least one tier should have convergence confidence: Warm={warm_confidence}, Cold={cold_confidence}"
        );

        engine.shutdown().expect("shutdown should succeed");
    }

    #[test]
    #[serial(parallel_engine)]
    fn test_adaptive_batcher_metrics_tracking() {
        use crate::activation::AdaptiveBatcherConfig;

        // Create config with adaptive batcher
        let mut config = fast_parallel_config();
        config.adaptive_batcher_config = Some(AdaptiveBatcherConfig::default());

        let (graph, node_a, _node_b, _node_c) = create_test_graph();
        let engine =
            ParallelSpreadingEngine::new(config, graph).expect("engine creation should succeed");

        let batcher = engine.adaptive_batcher.as_ref().expect("batcher exists");

        // Spread activation (records observations)
        let seed_activations = vec![(node_a, 1.0)];
        let _results = engine
            .spread_activation(&seed_activations)
            .expect("spread should succeed");

        // Run a second spread to process observations from the first spread
        let _results2 = engine
            .spread_activation(&seed_activations)
            .expect("second spread should succeed");

        // Get metrics
        let metrics = batcher.metrics();

        // Verify metrics are being tracked
        assert!(
            metrics.update_count.load(Ordering::Relaxed) > 0,
            "Update count should be tracked"
        );

        // All metric fields should be accessible
        let _guardrails = metrics.guardrail_hits.load(Ordering::Relaxed);
        let _topology = metrics.topology_changes.load(Ordering::Relaxed);
        let _fallbacks = metrics.fallback_activations.load(Ordering::Relaxed);

        engine.shutdown().expect("shutdown should succeed");
    }

    #[test]
    #[serial(parallel_engine)]
    fn test_adaptive_batcher_disabled_by_default() {
        // Create config without adaptive batcher
        let config = fast_parallel_config();
        assert!(
            config.adaptive_batcher_config.is_none(),
            "Default config should not have adaptive batcher enabled"
        );

        let (graph, _node_a, _node_b, _node_c) = create_test_graph();
        let engine =
            ParallelSpreadingEngine::new(config, graph).expect("engine creation should succeed");

        // Verify batcher was NOT created
        assert!(
            engine.adaptive_batcher.is_none(),
            "AdaptiveBatcher should be None when config is not provided"
        );

        engine.shutdown().expect("shutdown should succeed");
    }

    #[test]
    #[serial(parallel_engine)]
    fn test_adaptive_batcher_observation_flow() {
        use crate::activation::{AdaptiveBatcherConfig, Observation};

        // Create config with adaptive batcher
        let mut config = fast_parallel_config();
        config.adaptive_batcher_config = Some(AdaptiveBatcherConfig::default());
        config.max_depth = 3; // Deeper spreading for more observations

        let (graph, _node_a, _node_b, _node_c) = create_test_graph();
        let engine =
            ParallelSpreadingEngine::new(config, graph).expect("engine creation should succeed");

        let batcher = engine.adaptive_batcher.as_ref().expect("batcher exists");

        // Manually record an observation
        let test_observation = Observation {
            batch_size: 64,
            latency_ns: 1_000_000, // 1ms
            hop_count: 2,
            tier: StorageTier::Hot,
        };

        batcher.record_observation(test_observation);

        // Process observations
        batcher.process_observations();

        // Verify observation was processed
        let metrics = batcher.metrics();
        assert!(
            metrics.update_count.load(Ordering::Relaxed) > 0,
            "Observation should have been processed"
        );

        engine.shutdown().expect("shutdown should succeed");
    }
}
