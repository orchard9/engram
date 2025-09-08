//! Simplified parallel activation spreading implementation
//!
//! This provides a working parallel spreading engine that compiles successfully
//! while we develop the full work-stealing implementation.

use crate::activation::{
    ActivationError, ActivationRecord, ActivationResult, ActivationTask, MemoryGraph, NodeId,
    ParallelSpreadingConfig, SpreadingMetrics, ActivationGraphExt,
};
use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Simplified parallel spreading engine
pub struct SimpleParallelEngine {
    config: ParallelSpreadingConfig,
    activation_records: Arc<DashMap<NodeId, Arc<ActivationRecord>>>,
    global_queue: Arc<SegQueue<ActivationTask>>,
    memory_graph: Arc<MemoryGraph>,
    thread_handles: Vec<JoinHandle<()>>,
    metrics: Arc<SpreadingMetrics>,
    shutdown_signal: Arc<AtomicBool>,
}

impl SimpleParallelEngine {
    /// Create new simplified parallel engine
    ///
    /// # Errors
    ///
    /// Returns `ActivationError::InvalidConfig` if number of threads is 0
    pub fn new(
        config: ParallelSpreadingConfig,
        memory_graph: Arc<MemoryGraph>,
    ) -> ActivationResult<Self> {
        if config.num_threads == 0 {
            return Err(ActivationError::InvalidConfig(
                "Number of threads must be > 0".to_string(),
            ));
        }

        let activation_records = Arc::new(DashMap::new());
        let global_queue = Arc::new(SegQueue::new());
        let metrics = Arc::new(SpreadingMetrics::default());
        let shutdown_signal = Arc::new(AtomicBool::new(false));

        Ok(Self {
            config,
            activation_records,
            global_queue,
            memory_graph,
            thread_handles: Vec::new(),
            metrics,
            shutdown_signal,
        })
    }

    /// Initialize activation spreading from seed nodes
    ///
    /// # Errors
    ///
    /// Currently never returns an error but maintains Result for future extensibility
    pub fn spread_activation(
        &mut self,
        seed_activations: &[(NodeId, f32)],
    ) -> ActivationResult<()> {
        // Clear previous state
        self.activation_records.clear();
        self.metrics.reset();

        // Create initial tasks
        for (node_id, initial_activation) in seed_activations {
            let record = Arc::new(ActivationRecord::new(node_id.clone(), 0.1));
            record.accumulate_activation(*initial_activation);
            self.activation_records.insert(node_id.clone(), record);

            // Create tasks for neighbors
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

        // Start worker threads
        self.spawn_workers();

        // Wait for completion
        self.wait_for_completion()?;

        // Shutdown workers
        self.shutdown_signal.store(true, Ordering::SeqCst);

        // Join threads
        while let Some(handle) = self.thread_handles.pop() {
            handle.join().map_err(|_| {
                ActivationError::ThreadingError("Failed to join thread".to_string())
            })?;
        }

        Ok(())
    }

    fn spawn_workers(&mut self) {
        for _worker_id in 0..self.config.num_threads {
            let queue = self.global_queue.clone();
            let records = self.activation_records.clone();
            let memory_graph = self.memory_graph.clone();
            let metrics = self.metrics.clone();
            let shutdown = self.shutdown_signal.clone();
            let config = self.config.clone();

            let handle = thread::spawn(move || {
                while !shutdown.load(Ordering::Relaxed) {
                    if let Some(task) = queue.pop() {
                        Self::process_task(
                            &task,
                            &records,
                            &memory_graph,
                            &queue,
                            &config,
                            &metrics,
                        );
                    } else {
                        thread::yield_now();
                    }
                }
            });

            self.thread_handles.push(handle);
        }
    }

    fn process_task(
        task: &ActivationTask,
        records: &Arc<DashMap<NodeId, Arc<ActivationRecord>>>,
        memory_graph: &Arc<MemoryGraph>,
        queue: &Arc<SegQueue<ActivationTask>>,
        config: &ParallelSpreadingConfig,
        metrics: &Arc<SpreadingMetrics>,
    ) {
        // Get or create record
        let record = records
            .entry(task.target_node.clone())
            .or_insert_with(|| Arc::new(ActivationRecord::new(task.target_node.clone(), 0.1)))
            .clone();

        // Accumulate activation
        let contribution = task.contribution();
        if !record.accumulate_activation(contribution) {
            return; // Below threshold
        }

        // Continue spreading if appropriate
        if task.should_continue() {
            if let Some(neighbors) = memory_graph.get_neighbors(&task.target_node) {
                let decay_factor = config.decay_function.apply(task.depth + 1);

                for edge in neighbors {
                    let new_task = ActivationTask::new(
                        edge.target.clone(),
                        record.get_activation(),
                        edge.weight,
                        decay_factor,
                        task.depth + 1,
                        task.max_depth,
                    );

                    queue.push(new_task);
                }
            }
        }

        metrics.total_activations.fetch_add(1, Ordering::Relaxed);
    }

    fn wait_for_completion(&self) -> ActivationResult<()> {
        // Simple busy wait - could be improved
        let timeout = Duration::from_secs(30);
        let start = std::time::Instant::now();

        while start.elapsed() < timeout {
            if self.global_queue.is_empty() {
                // Give a bit more time for final processing
                thread::sleep(Duration::from_millis(10));
                if self.global_queue.is_empty() {
                    break;
                }
            }
            thread::sleep(Duration::from_millis(1));
        }

        if start.elapsed() >= timeout {
            return Err(ActivationError::ThreadingError(
                "Timeout waiting for completion".to_string(),
            ));
        }

        Ok(())
    }

    /// Get current activation levels
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
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::EdgeType;

    fn create_test_graph() -> Arc<MemoryGraph> {
        let graph = Arc::new(MemoryGraph::new());
        graph.add_edge("A".to_string(), "B".to_string(), 0.8, EdgeType::Excitatory);
        graph.add_edge("B".to_string(), "C".to_string(), 0.6, EdgeType::Excitatory);
        graph.add_edge("A".to_string(), "C".to_string(), 0.4, EdgeType::Excitatory);
        graph
    }

    #[test]
    fn test_simple_parallel_engine() {
        let config = ParallelSpreadingConfig {
            num_threads: 2,
            max_depth: 2,
            threshold: 0.01,
            ..Default::default()
        };

        let graph = create_test_graph();
        let mut engine = SimpleParallelEngine::new(config, graph).unwrap();

        let seed_activations = vec![("A".to_string(), 1.0)];
        engine.spread_activation(&seed_activations).unwrap();

        let activations = engine.get_activations();
        assert!(!activations.is_empty());

        // Should have activated A
        let node_a = activations.iter().find(|(id, _)| id == "A");
        assert!(node_a.is_some());
    }

    #[test]
    fn test_activation_spreading() {
        let config = ParallelSpreadingConfig {
            num_threads: 1,
            max_depth: 3,
            threshold: 0.01,
            ..Default::default()
        };

        let graph = create_test_graph();
        let mut engine = SimpleParallelEngine::new(config, graph).unwrap();

        let seed_activations = vec![("A".to_string(), 0.8)];
        engine.spread_activation(&seed_activations).unwrap();

        let active_nodes = engine.get_active_nodes(0.1);
        assert!(active_nodes.len() > 1); // Should have spread to multiple nodes
    }
}
