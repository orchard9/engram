//! Lock-free parallel activation spreading engine
//!
//! High-performance work-stealing thread pool implementation for parallel
//! graph traversal with cognitive architecture principles.

use crate::activation::{
    cycle_detector::CycleDetector, gpu_interface::{AdaptiveConfig, AdaptiveSpreadingEngine},
    latency_budget::LatencyBudgetManager, memory_pool::ActivationMemoryPool,
    storage_aware::StorageTier, ActivationError, ActivationGraphExt, ActivationRecord,
    ActivationResult, ActivationTask, MemoryGraph, NodeId, ParallelSpreadingConfig,
    SpreadingMetrics, SpreadingResults, TierAwareSpreadingScheduler, TierSummary, TraceEntry,
    WeightedEdge,
};
use dashmap::DashMap;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tracing::warn;

/// Work-stealing thread pool for parallel activation spreading
pub struct ParallelSpreadingEngine {
    config: ParallelSpreadingConfig,
    activation_records: Arc<DashMap<NodeId, Arc<ActivationRecord>>>,
    memory_graph: Arc<MemoryGraph>,
    scheduler: Arc<TierAwareSpreadingScheduler>,
    thread_handles: Vec<JoinHandle<()>>,
    metrics: Arc<SpreadingMetrics>,
    shutdown_signal: Arc<AtomicBool>,
    phase_barrier: Arc<PhaseBarrier>,
    latency_budget: LatencyBudgetManager,
    cycle_detector: Arc<CycleDetector>,
    deterministic_trace: Arc<Mutex<Vec<TraceEntry>>>,
    adaptive_engine: Arc<Mutex<AdaptiveSpreadingEngine>>,
    memory_pool: Option<Arc<ActivationMemoryPool>>,
}

/// Synchronization barrier for deterministic parallel execution
pub struct PhaseBarrier {
    worker_count: AtomicUsize,
    phase_counter: AtomicU64,
    waiting: AtomicUsize,
    enabled: AtomicBool,
}

impl PhaseBarrier {
    fn new(worker_count: usize) -> Self {
        Self {
            worker_count: AtomicUsize::new(worker_count),
            phase_counter: AtomicU64::new(0),
            waiting: AtomicUsize::new(0),
            enabled: AtomicBool::new(true),
        }
    }

    /// Disable the barrier, allowing workers to pass through without waiting
    fn disable(&self) {
        self.enabled.store(false, Ordering::SeqCst);
    }

    /// Wait for all workers to reach this barrier point
    fn wait(&self) {
        // If barrier is disabled, return immediately
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        let total_workers = self.worker_count.load(Ordering::Relaxed);
        let current_waiting = self.waiting.fetch_add(1, Ordering::SeqCst);

        if current_waiting + 1 == total_workers {
            // Last worker to arrive - reset and advance phase
            self.waiting.store(0, Ordering::Relaxed);
            self.phase_counter.fetch_add(1, Ordering::SeqCst);
        } else {
            // Wait for phase to advance or barrier to be disabled
            let current_phase = self.phase_counter.load(Ordering::Relaxed);
            while self.phase_counter.load(Ordering::Relaxed) == current_phase
                  && self.enabled.load(Ordering::Relaxed) {
                thread::yield_now();
            }
        }
    }
}

/// Worker thread context for parallel spreading
///
/// Note: No randomness is used in the current implementation. Determinism is achieved
/// through canonical task ordering (via TierQueue sorting) and phase barriers for
/// hop-level synchronization. Future probabilistic features can add RNG if needed.
struct WorkerContext {
    worker_id: usize,
    activation_records: Arc<DashMap<NodeId, Arc<ActivationRecord>>>,
    memory_graph: Arc<MemoryGraph>,
    config: ParallelSpreadingConfig,
    metrics: Arc<SpreadingMetrics>,
    shutdown_signal: Arc<AtomicBool>,
    phase_barrier: Arc<PhaseBarrier>,
    latency_budget: LatencyBudgetManager,
    scheduler: Arc<TierAwareSpreadingScheduler>,
    cycle_detector: Arc<CycleDetector>,
    deterministic_trace: Arc<Mutex<Vec<TraceEntry>>>,
    adaptive_engine: Arc<Mutex<AdaptiveSpreadingEngine>>,
}

struct TierTaskGuard<'a> {
    scheduler: &'a TierAwareSpreadingScheduler,
    tier: StorageTier,
    finished: Cell<bool>,
}

impl<'a> TierTaskGuard<'a> {
    fn new(scheduler: &'a TierAwareSpreadingScheduler, tier: StorageTier) -> Self {
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

impl ParallelSpreadingEngine {
    /// Create new parallel spreading engine
    pub fn new(
        config: ParallelSpreadingConfig,
        memory_graph: Arc<MemoryGraph>,
    ) -> ActivationResult<Self> {
        if config.num_threads == 0 {
            return Err(ActivationError::InvalidConfig(
                "Number of threads must be > 0".to_string()
            ));
        }
        let activation_records = Arc::new(DashMap::new());
        let metrics = Arc::new(SpreadingMetrics::default());
        let shutdown_signal = Arc::new(AtomicBool::new(false));
        let phase_barrier = Arc::new(PhaseBarrier::new(config.num_threads));
        let scheduler = Arc::new(TierAwareSpreadingScheduler::from_config(&config));
        let cycle_detector = Arc::new(CycleDetector::new(config.tier_cycle_budgets.clone()));
        let deterministic_trace = Arc::new(Mutex::new(Vec::new()));

        // Initialize adaptive GPU/CPU engine
        let adaptive_config = AdaptiveConfig {
            gpu_threshold: config.gpu_threshold,
            enable_gpu: config.enable_gpu,
        };
        let adaptive_engine = Arc::new(Mutex::new(AdaptiveSpreadingEngine::new(
            None, // GPU interface will be injected in Milestone 11
            adaptive_config,
        )));

        // Initialize memory pool if configured
        let memory_pool = if config.enable_memory_pool {
            Some(Arc::new(ActivationMemoryPool::new(
                config.pool_chunk_size,
                config.pool_max_chunks,
            )))
        } else {
            None
        };

        let mut engine = Self {
            config: config.clone(),
            activation_records,
            memory_graph: memory_graph.clone(),
            scheduler,
            thread_handles: Vec::new(),
            metrics,
            shutdown_signal,
            phase_barrier,
            latency_budget: LatencyBudgetManager::new(),
            cycle_detector,
            deterministic_trace,
            adaptive_engine,
            memory_pool,
        };

        engine.spawn_workers()?;

        Ok(engine)
    }

    /// Spawn worker threads for parallel processing
    fn spawn_workers(&mut self) -> ActivationResult<()> {
        for worker_id in 0..self.config.num_threads {
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
                adaptive_engine: self.adaptive_engine.clone(),
            };

            let handle = thread::Builder::new()
                .name(format!("activation-worker-{}", worker_id))
                .spawn(move || {
                    Self::worker_main(context);
                })
                .map_err(|e| ActivationError::ThreadingError(e.to_string()))?;

            self.thread_handles.push(handle);
        }

        Ok(())
    }
    
    /// Determine if batch spreading should be used for this set of neighbors
    fn should_use_batch_spreading(
        context: &WorkerContext,
        neighbors: &[WeightedEdge],
        tier: StorageTier,
    ) -> bool {
        use crate::activation::simd_optimization::should_use_simd_for_tier;

        let neighbor_count = neighbors.len();
        should_use_simd_for_tier(tier, neighbor_count, context.config.simd_batch_size)
    }

    /// Process neighbors using SIMD batch operations
    fn process_neighbors_batch(
        context: &WorkerContext,
        task: &ActivationTask,
        record: &Arc<ActivationRecord>,
        neighbors: &[WeightedEdge],
        next_tier: StorageTier,
        decay_factor: f32,
    ) {
        use crate::activation::simd_optimization::SimdActivationMapper;
        use crate::activation::ActivationGraphExt;
        use crate::compute::cosine_similarity_batch_768;

        // Get current node's embedding from the graph
        let current_embedding = match context.memory_graph.get_embedding(&task.target_node) {
            Some(emb) => emb,
            None => {
                // No embedding for current node, fallback to scalar
                Self::fallback_to_scalar(context, task, record, neighbors, next_tier, decay_factor);
                return;
            }
        };

        // Collect neighbor embeddings from the graph
        let neighbor_embeddings: Vec<[f32; 768]> = neighbors
            .iter()
            .filter_map(|edge| context.memory_graph.get_embedding(&edge.target))
            .collect();

        // Ensure we have all embeddings before using SIMD
        if neighbor_embeddings.is_empty() || neighbor_embeddings.len() != neighbors.len() {
            Self::fallback_to_scalar(context, task, record, neighbors, next_tier, decay_factor);
            return;
        }

        // Batch compute similarities using SIMD
        let similarities = cosine_similarity_batch_768(&current_embedding, &neighbor_embeddings);

        // Convert similarities to activations using SIMD
        let mapper = SimdActivationMapper::new();
        let temperature = 0.5; // From config
        let threshold = context.config.threshold;
        let activations = mapper.batch_sigmoid_activation(&similarities, temperature, threshold);

        // Create tasks with computed activations
        for (idx, edge) in neighbors.iter().enumerate() {
            let activation_weight = activations.get(idx).copied().unwrap_or(edge.weight);

            let mut next_path = task.path.clone();
            next_path.push(edge.target.clone());
            let new_task = ActivationTask::new(
                edge.target.clone(),
                record.get_activation(),
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
        record: &Arc<ActivationRecord>,
        neighbors: &[WeightedEdge],
        next_tier: StorageTier,
        decay_factor: f32,
    ) {
        for edge in neighbors {
            let mut next_path = task.path.clone();
            next_path.push(edge.target.clone());
            let new_task = ActivationTask::new(
                edge.target.clone(),
                record.get_activation(),
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
    fn worker_main(context: WorkerContext) {
        while !context.shutdown_signal.load(Ordering::Relaxed) {
            if let Some(task) = context.scheduler.next_task() {
                Self::process_task(&context, task);

                if context.config.deterministic {
                    context.phase_barrier.wait();
                }

                continue;
            }

            if context.config.deterministic {
                context.phase_barrier.wait();
            }

            if context.scheduler.is_idle() {
                thread::sleep(Duration::from_micros(100));
            } else {
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

        // Get or create activation record
        let target_clone = task.target_node.clone();
        let record = context
            .activation_records
            .entry(target_clone.clone())
            .or_insert_with(|| {
                let mut base = ActivationRecord::new(target_clone.clone(), 0.1);
                base.set_storage_tier(tier);
                Arc::new(base)
            })
            .clone();

        // Accumulate activation with threshold check
        let contribution = task.contribution();
        if !record.accumulate_activation(contribution) {
            guard.finish();
            return; // Below threshold
        }

        // Capture trace entry if deterministic tracing is enabled
        if context.config.trace_activation_flow && context.config.deterministic {
            let source_node = if task.path.len() > 1 {
                task.path.get(task.path.len() - 2).cloned()
            } else {
                None
            };

            let trace_entry = TraceEntry {
                depth: task.depth,
                target_node: target_clone.clone(),
                activation: record.get_activation(),
                confidence: record.get_confidence(),
                source_node,
            };

            if let Ok(mut trace) = context.deterministic_trace.lock() {
                trace.push(trace_entry);
            }
        }

        let visit_count = record.visits.fetch_add(1, Ordering::Relaxed) + 1;
        let mut should_mark_done = false;

        if context.config.cycle_detection {
            if task.path.is_empty() || task.path.last() != Some(&target_clone) {
                task.path.push(target_clone.clone());
            }

            if !context
                .cycle_detector
                .should_visit(&target_clone, visit_count, tier, &task.path)
            {
                record.apply_cycle_penalty(context.config.cycle_penalty_factor);
                context
                    .metrics
                    .cycles_detected
                    .fetch_add(1, Ordering::Relaxed);
                context.metrics.increment_cycle_for_tier(tier);

                if context.config.trace_activation_flow {
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

            context
                .cycle_detector
                .mark_visiting(target_clone.clone());
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
        if let Some(neighbors) = ActivationGraphExt::get_neighbors(&*context.memory_graph, &task.target_node) {
            let decay_factor = context.config.decay_function.apply(task.depth + 1);
            let next_tier = StorageTier::from_depth(task.depth + 1);

            // Try SIMD batch processing if we have enough neighbors
            if Self::should_use_batch_spreading(context, &neighbors, next_tier) {
                Self::process_neighbors_batch(context, &task, &record, &neighbors, next_tier, decay_factor);
            } else {
                // Scalar path: process neighbors individually
                for edge in neighbors {
                    let mut next_path = task.path.clone();
                    next_path.push(edge.target.clone());
                    let new_task = ActivationTask::new(
                        edge.target.clone(),
                        record.get_activation(),
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
        let duration = start_time.elapsed().as_nanos() as u64;
        context.metrics.average_latency.store(duration, Ordering::Relaxed);
        context
            .metrics
            .total_activations
            .fetch_add(1, Ordering::Relaxed);

        if !context.latency_budget.within_budget(tier, Duration::from_nanos(duration)) {
            context
                .metrics
                .latency_budget_violations
                .fetch_add(1, Ordering::Relaxed);
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
        // Clear previous activation state
        self.activation_records.clear();
        self.metrics.reset();
        self.scheduler.reset(&self.config);
        self.cycle_detector.reset();
        self.phase_barrier.enabled.store(true, Ordering::SeqCst);
        if let Ok(mut trace) = self.deterministic_trace.lock() {
            trace.clear();
        }

        // Create initial tasks from seed nodes
        for (node_id, initial_activation) in seed_activations {
            let mut base_record = ActivationRecord::new(node_id.clone(), 0.1);
            base_record.set_storage_tier(StorageTier::Hot);
            let record = Arc::new(base_record);
            record.accumulate_activation(*initial_activation);
            self.activation_records.insert(node_id.clone(), record);

            if let Some(neighbors) = ActivationGraphExt::get_neighbors(&*self.memory_graph, node_id) {
                let decay_factor = self.config.decay_function.apply(1);
                let next_tier = StorageTier::from_depth(1);

                for edge in neighbors {
                    let task_path = vec![node_id.clone(), edge.target.clone()];
                    let task = ActivationTask::new(
                        edge.target.clone(),
                        *initial_activation,
                        edge.weight,
                        decay_factor,
                        1,
                        self.config.max_depth,
                    )
                    .with_storage_tier(next_tier)
                    .with_path(task_path);

                    self.scheduler.enqueue_task(task);
                }
            }
        }

        // Wait for processing to complete
        self.wait_for_completion()?;

        Ok(self.collect_results())
    }
    
    /// Wait for all workers to complete processing
    fn wait_for_completion(&self) -> ActivationResult<()> {
        let timeout = Duration::from_secs(30);
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
                "Timeout waiting for spreading completion".to_string()
            ));
        }

        Ok(())
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
            let tier = record
                .storage_tier()
                .unwrap_or(StorageTier::from_depth(0));
            let storage_activation = record.to_storage_aware();

            let activation_value = storage_activation
                .activation_level
                .load(Ordering::Relaxed);
            let confidence_value = storage_activation
                .confidence
                .load(Ordering::Relaxed);

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
                .map(|state| state.deadline_missed)
                .unwrap_or(false);

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

        let deterministic_trace = if self.config.trace_activation_flow && self.config.deterministic {
            self.deterministic_trace
                .lock()
                .map(|trace| trace.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        SpreadingResults {
            activations,
            tier_summaries,
            cycle_paths: self.cycle_detector.get_cycle_paths(),
            deterministic_trace,
        }
    }
    
    /// Get current activation levels for all nodes
    pub fn get_activations(&self) -> Vec<(NodeId, f32)> {
        self.activation_records
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().get_activation()))
            .collect()
    }
    
    /// Get nodes with activation above threshold
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
    pub fn get_metrics(&self) -> &SpreadingMetrics {
        &self.metrics
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
        
        // Join any remaining threads
        while let Some(handle) = self.thread_handles.pop() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{create_activation_graph, ActivationGraphExt, EdgeType};
    use std::collections::HashMap;
    use std::time::Duration;
    
    fn create_test_graph() -> Arc<MemoryGraph> {
        use crate::activation::{create_activation_graph, ActivationGraphExt};

        let graph = Arc::new(create_activation_graph());

        // Create a simple test graph: A -> B -> C, A -> C
        ActivationGraphExt::add_edge(&*graph, "A".to_string(), "B".to_string(), 0.8, EdgeType::Excitatory);
        ActivationGraphExt::add_edge(&*graph, "B".to_string(), "C".to_string(), 0.6, EdgeType::Excitatory);
        ActivationGraphExt::add_edge(&*graph, "A".to_string(), "C".to_string(), 0.4, EdgeType::Excitatory);

        graph
    }
    
    #[test]
    fn test_parallel_engine_creation() {
        let config = ParallelSpreadingConfig {
            num_threads: 2,
            max_depth: 3,
            ..Default::default()
        };
        
        let graph = create_test_graph();
        let engine = ParallelSpreadingEngine::new(config, graph).unwrap();
        
        assert!(engine.scheduler.is_idle());
        assert!(!engine.shutdown_signal.load(Ordering::Relaxed));
        
        engine.shutdown().unwrap();
    }
    
    #[test]
    fn test_activation_spreading() {
        let config = ParallelSpreadingConfig {
            num_threads: 2,
            max_depth: 2,
            threshold: 0.01,
            ..Default::default()
        };
        
        let graph = create_test_graph();
        let engine = ParallelSpreadingEngine::new(config, graph).unwrap();
        
        // Spread activation from node A
        let seed_activations = vec![("A".to_string(), 1.0)];
        let results = engine.spread_activation(&seed_activations).unwrap();

        assert!(!results.activations.is_empty());
        let hot_summary = results
            .tier_summary(StorageTier::Hot)
            .expect("hot tier summary");
        assert!(hot_summary.node_count >= 1);

        let node_a_activation = results
            .activations
            .iter()
            .find(|activation| activation.memory_id == "A")
            .map(|activation| activation.activation_level.load(Ordering::Relaxed));
        assert!(node_a_activation.is_some());
        assert!(node_a_activation.unwrap() >= 1.0);
        
        engine.shutdown().unwrap();
    }
    
    #[test]
    fn test_deterministic_spreading() {
        let config1 = ParallelSpreadingConfig::deterministic(42);
        let config2 = ParallelSpreadingConfig::deterministic(42);

        let graph1 = create_test_graph();
        let graph2 = create_test_graph();

        let engine1 = ParallelSpreadingEngine::new(config1, graph1).unwrap();
        let engine2 = ParallelSpreadingEngine::new(config2, graph2).unwrap();

        let seed_activations = vec![("A".to_string(), 1.0)];

        let results1 = engine1.spread_activation(&seed_activations).unwrap();
        let results2 = engine2.spread_activation(&seed_activations).unwrap();

        let mut activations1: Vec<_> = results1
            .activations
            .iter()
            .map(|activation| {
                (
                    activation.memory_id.clone(),
                    activation.activation_level.load(Ordering::Relaxed),
                )
            })
            .collect();
        let mut activations2: Vec<_> = results2
            .activations
            .iter()
            .map(|activation| {
                (
                    activation.memory_id.clone(),
                    activation.activation_level.load(Ordering::Relaxed),
                )
            })
            .collect();

        activations1.sort_by(|a, b| a.0.cmp(&b.0));
        activations2.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(activations1.len(), activations2.len());
        for (a1, a2) in activations1.iter().zip(activations2.iter()) {
            assert_eq!(a1.0, a2.0);
            assert!((a1.1 - a2.1).abs() < 1e-6);
        }

        // Verify trace is captured
        assert!(!results1.deterministic_trace.is_empty());
        assert_eq!(results1.deterministic_trace.len(), results2.deterministic_trace.len());

        engine1.shutdown().unwrap();
        engine2.shutdown().unwrap();
    }

    #[test]
    fn test_deterministic_across_thread_counts() {
        let seed_activations = vec![("A".to_string(), 1.0)];

        // Run with 1 thread
        let config1 = ParallelSpreadingConfig {
            num_threads: 1,
            ..ParallelSpreadingConfig::deterministic(123)
        };
        let graph1 = create_test_graph();
        let engine1 = ParallelSpreadingEngine::new(config1, graph1).unwrap();
        let results1 = engine1.spread_activation(&seed_activations).unwrap();

        // Run with 2 threads
        let config2 = ParallelSpreadingConfig {
            num_threads: 2,
            ..ParallelSpreadingConfig::deterministic(123)
        };
        let graph2 = create_test_graph();
        let engine2 = ParallelSpreadingEngine::new(config2, graph2).unwrap();
        let results2 = engine2.spread_activation(&seed_activations).unwrap();

        // Run with 4 threads
        let config3 = ParallelSpreadingConfig {
            num_threads: 4,
            ..ParallelSpreadingConfig::deterministic(123)
        };
        let graph3 = create_test_graph();
        let engine3 = ParallelSpreadingEngine::new(config3, graph3).unwrap();
        let results3 = engine3.spread_activation(&seed_activations).unwrap();

        // Compare results
        let mut act1: Vec<_> = results1.activations.iter()
            .map(|a| (a.memory_id.clone(), a.activation_level.load(Ordering::Relaxed)))
            .collect();
        let mut act2: Vec<_> = results2.activations.iter()
            .map(|a| (a.memory_id.clone(), a.activation_level.load(Ordering::Relaxed)))
            .collect();
        let mut act3: Vec<_> = results3.activations.iter()
            .map(|a| (a.memory_id.clone(), a.activation_level.load(Ordering::Relaxed)))
            .collect();

        act1.sort_by(|a, b| a.0.cmp(&b.0));
        act2.sort_by(|a, b| a.0.cmp(&b.0));
        act3.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(act1.len(), act2.len());
        assert_eq!(act2.len(), act3.len());

        for i in 0..act1.len() {
            assert_eq!(act1[i].0, act2[i].0);
            assert_eq!(act2[i].0, act3[i].0);
            assert!((act1[i].1 - act2[i].1).abs() < 1e-6,
                "Thread count 1 vs 2 mismatch for node {}", act1[i].0);
            assert!((act2[i].1 - act3[i].1).abs() < 1e-6,
                "Thread count 2 vs 4 mismatch for node {}", act2[i].0);
        }

        engine1.shutdown().unwrap();
        engine2.shutdown().unwrap();
        engine3.shutdown().unwrap();
    }

    #[test]
    fn test_deterministic_trace_capture() {
        let config = ParallelSpreadingConfig::deterministic(999);
        let graph = create_test_graph();
        let engine = ParallelSpreadingEngine::new(config, graph).unwrap();

        let seed_activations = vec![("A".to_string(), 1.0)];
        let results = engine.spread_activation(&seed_activations).unwrap();

        // Verify trace was captured
        assert!(!results.deterministic_trace.is_empty());

        // Verify trace entries are ordered
        for entry in &results.deterministic_trace {
            assert!(entry.activation >= 0.0 && entry.activation <= 1.0);
            assert!(entry.confidence >= 0.0 && entry.confidence <= 1.0);
        }

        engine.shutdown().unwrap();
    }

    #[test]
    fn test_deterministic_vs_performance_mode() {
        let seed_activations = vec![("A".to_string(), 1.0)];

        // Deterministic mode
        let det_config = ParallelSpreadingConfig::deterministic(555);
        let det_graph = create_test_graph();
        let det_engine = ParallelSpreadingEngine::new(det_config, det_graph).unwrap();
        let det_results = det_engine.spread_activation(&seed_activations).unwrap();

        // Performance mode (non-deterministic)
        let perf_config = ParallelSpreadingConfig {
            deterministic: false,
            trace_activation_flow: false,
            ..ParallelSpreadingConfig::default()
        };
        let perf_graph = create_test_graph();
        let perf_engine = ParallelSpreadingEngine::new(perf_config, perf_graph).unwrap();
        let perf_results = perf_engine.spread_activation(&seed_activations).unwrap();

        // Both should activate nodes, but deterministic should have trace
        assert!(!det_results.activations.is_empty());
        assert!(!perf_results.activations.is_empty());
        assert!(!det_results.deterministic_trace.is_empty());
        assert!(perf_results.deterministic_trace.is_empty());

        det_engine.shutdown().unwrap();
        perf_engine.shutdown().unwrap();
    }
    
    #[test]
    fn test_metrics_tracking() {
        let config = ParallelSpreadingConfig {
            num_threads: 4,
            max_depth: 3,
            ..Default::default()
        };

        let graph = create_test_graph();
        let engine = ParallelSpreadingEngine::new(config, graph).unwrap();

        let seed_activations = vec![("A".to_string(), 1.0)];
        let results = engine.spread_activation(&seed_activations).unwrap();

        let metrics = engine.get_metrics();
        assert!(metrics.total_activations.load(Ordering::Relaxed) > 0);
        let hot_summary = results
            .tier_summary(StorageTier::Hot)
            .expect("hot tier summary available");
        assert!(hot_summary.total_activation > 0.0);

        engine.shutdown().unwrap();
    }

    #[test]
    fn test_threshold_filtering() {
        let config = ParallelSpreadingConfig {
            num_threads: 2,
            max_depth: 2,
            threshold: 0.5, // High threshold
            ..Default::default()
        };
        
        let graph = create_test_graph();
        let engine = ParallelSpreadingEngine::new(config, graph).unwrap();
        
        let seed_activations = vec![("A".to_string(), 0.3)]; // Low seed
        let results = engine.spread_activation(&seed_activations).unwrap();
        
        let active_nodes = engine.get_active_nodes(0.5);
        // Should have fewer nodes due to high threshold
        assert!(active_nodes.len() <= 1);
        assert!(results.activations.len() >= active_nodes.len());
        
        engine.shutdown().unwrap();
    }

    #[test]
    fn cycle_detection_penalises_revisits() {
        let config = ParallelSpreadingConfig {
            num_threads: 1,
            max_depth: 3,
            cycle_detection: true,
            cycle_penalty_factor: 0.5,
            tier_cycle_budgets: HashMap::from([
                (StorageTier::Hot, 1),
                (StorageTier::Warm, 1),
                (StorageTier::Cold, 1),
            ]),
            trace_activation_flow: true,  // Enable tracing
            ..Default::default()
        };

        use crate::activation::{create_activation_graph, ActivationGraphExt};

        let graph = Arc::new(create_activation_graph());
        ActivationGraphExt::add_edge(&*graph, "A".to_string(), "B".to_string(), 0.9, EdgeType::Excitatory);
        ActivationGraphExt::add_edge(&*graph, "B".to_string(), "A".to_string(), 0.9, EdgeType::Excitatory);

        let engine = ParallelSpreadingEngine::new(config, graph).unwrap();
        let seed_activations = vec![("A".to_string(), 1.0)];
        let results = engine.spread_activation(&seed_activations).unwrap();

        assert!(!results.cycle_paths.is_empty(), "Expected cycle paths to be detected");
        assert!(
            engine
                .metrics
                .cycles_detected
                .load(Ordering::Relaxed)
                > 0,
            "Expected cycles to be detected"
        );
        // Cycle occurs at depth 3, which maps to Cold tier
        let tier_cycle_count = engine.metrics.cycle_count_for_tier(StorageTier::Cold);
        assert!(tier_cycle_count > 0, "Expected tier cycle count > 0, got {}", tier_cycle_count);

        // The cycle is detected on the second visit to B, so B should have the penalty
        let activation_b = results
            .activations
            .iter()
            .find(|activation| activation.memory_id == "B")
            .map(|activation| activation.activation_level.load(Ordering::Relaxed))
            .unwrap_or_default();
        assert!(activation_b > 0.0 && activation_b < 0.5, "Expected activation penalty to be applied to B, got {}", activation_b);

        engine.shutdown().unwrap();
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
            handle.join().unwrap();
        }

        assert_eq!(barrier_arc.phase_counter.load(Ordering::Relaxed), 1);
    }

}
