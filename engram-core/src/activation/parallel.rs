//! Lock-free parallel activation spreading engine
//!
//! High-performance work-stealing thread pool implementation for parallel
//! graph traversal with cognitive architecture principles.

use crate::activation::{
    ActivationRecord, ActivationTask, MemoryGraph, ParallelSpreadingConfig, 
    SpreadingMetrics, WeightedEdge, EdgeType, NodeId, ActivationResult, ActivationError
};
use crossbeam_deque::{Injector, Stealer, Worker};
use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Work-stealing thread pool for parallel activation spreading
pub struct ParallelSpreadingEngine {
    config: ParallelSpreadingConfig,
    activation_records: Arc<DashMap<NodeId, Arc<ActivationRecord>>>,
    work_queues: Vec<Worker<ActivationTask>>,
    global_queue: Arc<Injector<ActivationTask>>,
    thread_handles: Vec<JoinHandle<()>>,
    metrics: Arc<SpreadingMetrics>,
    shutdown_signal: Arc<AtomicBool>,
    active_workers: Arc<AtomicUsize>,
    phase_barrier: Arc<PhaseBarrier>,
}

/// Synchronization barrier for deterministic parallel execution
pub struct PhaseBarrier {
    worker_count: AtomicUsize,
    phase_counter: AtomicU64,
    waiting: AtomicUsize,
}

impl PhaseBarrier {
    fn new(worker_count: usize) -> Self {
        Self {
            worker_count: AtomicUsize::new(worker_count),
            phase_counter: AtomicU64::new(0),
            waiting: AtomicUsize::new(0),
        }
    }
    
    /// Wait for all workers to reach this barrier point
    fn wait(&self) {
        let total_workers = self.worker_count.load(Ordering::Relaxed);
        let current_waiting = self.waiting.fetch_add(1, Ordering::SeqCst);
        
        if current_waiting + 1 == total_workers {
            // Last worker to arrive - reset and advance phase
            self.waiting.store(0, Ordering::Relaxed);
            self.phase_counter.fetch_add(1, Ordering::SeqCst);
        } else {
            // Wait for phase to advance
            let current_phase = self.phase_counter.load(Ordering::Relaxed);
            while self.phase_counter.load(Ordering::Relaxed) == current_phase {
                thread::yield_now();
            }
        }
    }
}

/// Worker thread context for parallel spreading
struct WorkerContext {
    worker_id: usize,
    local_queue: Arc<Deque<ActivationTask>>,
    global_queue: Arc<SegQueue<ActivationTask>>,
    work_queues: Vec<Arc<Deque<ActivationTask>>>,
    activation_records: Arc<DashMap<NodeId, Arc<ActivationRecord>>>,
    memory_graph: Arc<MemoryGraph>,
    config: ParallelSpreadingConfig,
    metrics: Arc<SpreadingMetrics>,
    shutdown_signal: Arc<AtomicBool>,
    phase_barrier: Arc<PhaseBarrier>,
    rng: Option<rand::rngs::StdRng>, // For deterministic work stealing
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
        let global_queue = Arc::new(SegQueue::new());
        let metrics = Arc::new(SpreadingMetrics::default());
        let shutdown_signal = Arc::new(AtomicBool::new(false));
        let active_workers = Arc::new(AtomicUsize::new(0));
        let phase_barrier = Arc::new(PhaseBarrier::new(config.num_threads));
        
        // Create per-worker work queues
        let mut work_queues = Vec::with_capacity(config.num_threads);
        for _ in 0..config.num_threads {
            work_queues.push(Arc::new(Deque::new()));
        }
        
        let mut engine = Self {
            config: config.clone(),
            activation_records,
            work_queues,
            global_queue,
            thread_handles: Vec::new(),
            metrics,
            shutdown_signal,
            active_workers,
            phase_barrier,
        };
        
        // Start worker threads
        engine.spawn_workers(memory_graph)?;
        
        Ok(engine)
    }
    
    /// Spawn worker threads for parallel processing
    fn spawn_workers(&mut self, memory_graph: Arc<MemoryGraph>) -> ActivationResult<()> {
        for worker_id in 0..self.config.num_threads {
            let context = WorkerContext {
                worker_id,
                local_queue: self.work_queues[worker_id].clone(),
                global_queue: self.global_queue.clone(),
                work_queues: self.work_queues.clone(),
                activation_records: self.activation_records.clone(),
                memory_graph: memory_graph.clone(),
                config: self.config.clone(),
                metrics: self.metrics.clone(),
                shutdown_signal: self.shutdown_signal.clone(),
                phase_barrier: self.phase_barrier.clone(),
                rng: if self.config.deterministic {
                    use rand::{Rng, SeedableRng};
                    Some(rand::rngs::StdRng::seed_from_u64(
                        self.config.seed.unwrap_or(42) + worker_id as u64
                    ))
                } else {
                    None
                },
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
    
    /// Main worker thread execution loop
    fn worker_main(mut context: WorkerContext) {
        context.metrics.total_activations.fetch_add(1, Ordering::Relaxed);
        
        while !context.shutdown_signal.load(Ordering::Relaxed) {
            // Try to get work from local queue first
            if let Some(task) = context.local_queue.pop() {
                Self::process_task(&context, task);
            }
            // Try global queue
            else if let Some(task) = context.global_queue.pop() {
                Self::process_task(&context, task);
            }
            // Work stealing from other workers
            else if !Self::try_work_stealing(&mut context) {
                // No work available - will yield thread below
            }
            
            // Deterministic mode barrier synchronization
            if context.config.deterministic {
                context.phase_barrier.wait();
            }
            
            // No work available - yield thread
            thread::yield_now();
        }
    }
    
    /// Process a single activation task
    fn process_task(context: &WorkerContext, task: ActivationTask) {
        let start_time = Instant::now();
        
        // Get or create activation record
        let record = context
            .activation_records
            .entry(task.target_node.clone())
            .or_insert_with(|| {
                Arc::new(ActivationRecord::new(task.target_node.clone(), 0.1))
            })
            .clone();
        
        // Accumulate activation with threshold check
        let contribution = task.contribution();
        if !record.accumulate_activation(contribution) {
            return; // Below threshold
        }
        
        // Check if we should continue spreading
        if !task.should_continue() {
            return;
        }
        
        // Get neighbors and create new tasks
        if let Some(neighbors) = context.memory_graph.get_neighbors(&task.target_node) {
            let decay_factor = context.config.decay_function.apply(task.depth + 1);
            
            for edge in neighbors {
                let new_task = ActivationTask::new(
                    edge.target.clone(),
                    record.get_activation(),
                    edge.weight,
                    decay_factor,
                    task.depth + 1,
                    task.max_depth,
                );
                
                // Distribute new task (local vs global queue)
                if context.local_queue.len() < context.config.batch_size {
                    context.local_queue.push(new_task);
                } else {
                    context.global_queue.push(new_task);
                }
            }
        }
        
        // Update metrics
        let duration = start_time.elapsed().as_nanos() as u64;
        context.metrics.average_latency.store(duration, Ordering::Relaxed);
    }
    
    /// Attempt to steal work from other workers
    fn try_work_stealing(context: &mut WorkerContext) -> bool {
        let work_stealing_threshold = context.config.work_stealing_ratio;
        
        // Deterministic work stealing using seeded RNG
        let should_steal = if let Some(rng) = &mut context.rng {
            use rand::Rng;
            rng.gen_range(0.0..1.0) < work_stealing_threshold
        } else {
            use rand::Rng;
            rand::thread_rng().gen_range(0.0..1.0) < work_stealing_threshold
        };
        
        if !should_steal {
            return false;
        }
        
        // Try to steal from random other worker
        let num_workers = context.work_queues.len();
        let start_idx = if let Some(rng) = &mut context.rng {
            use rand::Rng;
            rng.gen_range(0..num_workers)
        } else {
            use rand::Rng;
            rand::thread_rng().gen_range(0..num_workers)
        };
        
        for i in 0..num_workers {
            let worker_idx = (start_idx + i) % num_workers;
            if worker_idx == context.worker_id {
                continue; // Skip own queue
            }
            
            if let Some(task) = context.work_queues[worker_idx].steal() {
                context.metrics.work_steals.fetch_add(1, Ordering::Relaxed);
                Self::process_task(context, task);
                return true;
            }
        }
        
        false
    }
    
    /// Initialize activation spreading from seed nodes
    pub fn spread_activation(
        &self,
        seed_activations: &[(NodeId, f32)],
    ) -> ActivationResult<()> {
        // Clear previous activation state
        self.activation_records.clear();
        self.metrics.reset();
        
        // Create initial tasks from seed nodes
        for (node_id, initial_activation) in seed_activations {
            // Create activation record
            let record = Arc::new(ActivationRecord::new(node_id.clone(), 0.1));
            record.accumulate_activation(*initial_activation);
            self.activation_records.insert(node_id.clone(), record);
            
            // Create spreading tasks for neighbors
            if let Some(neighbors) = self.memory_graph.get_neighbors(node_id) {
                let decay_factor = self.config.decay_function.apply(1);
                
                for edge in neighbors {
                    let task = ActivationTask::new(
                        edge.target,
                        *initial_activation,
                        edge.weight,
                        decay_factor,
                        1,
                        self.config.max_depth,
                    );
                    
                    self.global_queue.push(task);
                }
            }
        }
        
        // Wait for processing to complete
        self.wait_for_completion()?;
        
        Ok(())
    }
    
    /// Wait for all workers to complete processing
    fn wait_for_completion(&self) -> ActivationResult<()> {
        let timeout = Duration::from_secs(30);
        let start = Instant::now();
        
        while start.elapsed() < timeout {
            // Check if all queues are empty
            let mut all_empty = self.global_queue.is_empty();
            
            for queue in &self.work_queues {
                all_empty &= queue.is_empty();
            }
            
            if all_empty {
                break;
            }
            
            thread::sleep(Duration::from_millis(1));
        }
        
        if start.elapsed() >= timeout {
            return Err(ActivationError::ThreadingError(
                "Timeout waiting for spreading completion".to_string()
            ));
        }
        
        Ok(())
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
    use std::time::Duration;
    
    fn create_test_graph() -> Arc<MemoryGraph> {
        let graph = Arc::new(MemoryGraph::new());
        
        // Create a simple test graph: A -> B -> C, A -> C
        graph.add_edge("A".to_string(), "B".to_string(), 0.8, EdgeType::Excitatory);
        graph.add_edge("B".to_string(), "C".to_string(), 0.6, EdgeType::Excitatory);
        graph.add_edge("A".to_string(), "C".to_string(), 0.4, EdgeType::Excitatory);
        
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
        
        assert_eq!(engine.work_queues.len(), 2);
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
        engine.spread_activation(&seed_activations).unwrap();
        
        // Check that activation spread to other nodes
        let activations = engine.get_activations();
        assert!(!activations.is_empty());
        
        // Node A should have initial activation
        let node_a_activation = activations
            .iter()
            .find(|(id, _)| id == "A")
            .map(|(_, activation)| *activation);
        assert!(node_a_activation.is_some());
        assert!(node_a_activation.unwrap() >= 1.0);
        
        engine.shutdown().unwrap();
    }
    
    #[test]
    fn test_deterministic_spreading() {
        let config1 = ParallelSpreadingConfig {
            num_threads: 2,
            max_depth: 2,
            deterministic: true,
            seed: Some(42),
            ..Default::default()
        };
        
        let config2 = ParallelSpreadingConfig {
            deterministic: true,
            seed: Some(42),
            ..config1.clone()
        };
        
        let graph1 = create_test_graph();
        let graph2 = create_test_graph();
        
        let engine1 = ParallelSpreadingEngine::new(config1, graph1).unwrap();
        let engine2 = ParallelSpreadingEngine::new(config2, graph2).unwrap();
        
        let seed_activations = vec![("A".to_string(), 1.0)];
        
        engine1.spread_activation(&seed_activations).unwrap();
        engine2.spread_activation(&seed_activations).unwrap();
        
        let mut activations1 = engine1.get_activations();
        let mut activations2 = engine2.get_activations();
        
        // Sort for comparison
        activations1.sort_by(|a, b| a.0.cmp(&b.0));
        activations2.sort_by(|a, b| a.0.cmp(&b.0));
        
        // Results should be identical in deterministic mode
        assert_eq!(activations1.len(), activations2.len());
        for ((id1, act1), (id2, act2)) in activations1.iter().zip(activations2.iter()) {
            assert_eq!(id1, id2);
            assert!((act1 - act2).abs() < 1e-6);
        }
        
        engine1.shutdown().unwrap();
        engine2.shutdown().unwrap();
    }
    
    #[test]
    fn test_work_stealing_metrics() {
        let config = ParallelSpreadingConfig {
            num_threads: 4,
            max_depth: 3,
            work_stealing_ratio: 0.5,
            ..Default::default()
        };
        
        let graph = create_test_graph();
        let engine = ParallelSpreadingEngine::new(config, graph).unwrap();
        
        let seed_activations = vec![("A".to_string(), 1.0)];
        engine.spread_activation(&seed_activations).unwrap();
        
        let metrics = engine.get_metrics();
        assert!(metrics.total_activations.load(Ordering::Relaxed) > 0);
        
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
        engine.spread_activation(&seed_activations).unwrap();
        
        let active_nodes = engine.get_active_nodes(0.5);
        // Should have fewer nodes due to high threshold
        assert!(active_nodes.len() <= 1);
        
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