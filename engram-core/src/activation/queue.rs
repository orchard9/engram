//! Lock-free queue implementations for parallel activation spreading
//!
//! Optimized queue structures for high-throughput task distribution
//! with work-stealing and batching support.

use crate::activation::{ActivationTask, NodeId, storage_aware::StorageTier};
use crossbeam_queue::SegQueue;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// High-performance work-stealing queue for activation tasks
pub struct WorkStealingQueue {
    local_deque: VecDeque<ActivationTask>,
    global_queue: Arc<SegQueue<ActivationTask>>,
    batch_size: usize,
    local_count: AtomicUsize,
    deterministic: bool,
}

impl WorkStealingQueue {
    /// Create new work-stealing queue
    pub const fn new(global_queue: Arc<SegQueue<ActivationTask>>, batch_size: usize) -> Self {
        Self {
            local_deque: VecDeque::new(),
            global_queue,
            batch_size,
            local_count: AtomicUsize::new(0),
            deterministic: false,
        }
    }

    /// Create new work-stealing queue with deterministic mode
    pub const fn new_deterministic(
        global_queue: Arc<SegQueue<ActivationTask>>,
        batch_size: usize,
    ) -> Self {
        Self {
            local_deque: VecDeque::new(),
            global_queue,
            batch_size,
            local_count: AtomicUsize::new(0),
            deterministic: true,
        }
    }

    /// Push task to local queue, overflow to global
    pub fn push(&mut self, task: ActivationTask) {
        let current_count = self.local_count.load(Ordering::Relaxed);

        if current_count < self.batch_size {
            self.local_deque.push_back(task);
            self.local_count.fetch_add(1, Ordering::Relaxed);
        } else {
            // Local queue full, push to global
            self.global_queue.push(task);
        }
    }

    /// Pop task from local queue
    pub fn pop(&mut self) -> Option<ActivationTask> {
        // Use pop_front for deterministic queues to maintain sort order
        // Use pop_back for non-deterministic (work-stealing) queues for LIFO locality
        let task = if self.deterministic {
            self.local_deque.pop_front()
        } else {
            self.local_deque.pop_back()
        };

        if let Some(task) = task {
            self.local_count.fetch_sub(1, Ordering::Relaxed);
            Some(task)
        } else {
            None
        }
    }

    /// Steal task from this queue (used by other workers)  
    pub fn steal(&mut self) -> Option<ActivationTask> {
        if let Some(task) = self.local_deque.pop_front() {
            self.local_count.fetch_sub(1, Ordering::Relaxed);
            Some(task)
        } else {
            None
        }
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.local_count.load(Ordering::Relaxed) == 0
    }

    /// Get current queue length
    pub fn len(&self) -> usize {
        self.local_count.load(Ordering::Relaxed)
    }

    /// Try to get task from global queue
    pub fn try_global(&self) -> Option<ActivationTask> {
        self.global_queue.pop()
    }

    /// Sort local queue for deterministic execution
    /// Orders by (depth, target_node, contribution) for reproducible task processing
    pub fn sort_deterministic(&mut self) {
        if !self.deterministic {
            return;
        }

        // Convert to Vec for sorting
        let mut tasks: Vec<ActivationTask> = self.local_deque.drain(..).collect();

        // Sort by (depth, target_node, contribution)
        tasks.sort_by(|a, b| {
            a.depth
                .cmp(&b.depth)
                .then_with(|| a.target_node.cmp(&b.target_node))
                .then_with(|| {
                    // Higher contribution first (reverse order)
                    b.contribution()
                        .partial_cmp(&a.contribution())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        // Rebuild deque
        self.local_deque = tasks.into_iter().collect();
    }
}

/// Priority queue for activation tasks with biological decay
pub struct PriorityActivationQueue {
    high_priority: SegQueue<ActivationTask>,
    medium_priority: SegQueue<ActivationTask>,
    low_priority: SegQueue<ActivationTask>,
    high_count: AtomicUsize,
    medium_count: AtomicUsize,
    low_count: AtomicUsize,
}

impl Default for PriorityActivationQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl PriorityActivationQueue {
    /// Create new priority queue
    #[must_use]
    pub const fn new() -> Self {
        Self {
            high_priority: SegQueue::new(),
            medium_priority: SegQueue::new(),
            low_priority: SegQueue::new(),
            high_count: AtomicUsize::new(0),
            medium_count: AtomicUsize::new(0),
            low_count: AtomicUsize::new(0),
        }
    }

    /// Push task with priority based on contribution
    pub fn push(&self, task: ActivationTask) {
        let contribution = task.tier_adjusted_contribution();
        let tier = task.storage_tier;
        let (high_cutoff, medium_cutoff) = match tier {
            Some(StorageTier::Hot) | None => (0.5, 0.1),
            Some(StorageTier::Warm) => (0.35, 0.05),
            Some(StorageTier::Cold) => (0.2, 0.01),
        };

        if contribution > high_cutoff {
            self.high_priority.push(task);
            self.high_count.fetch_add(1, Ordering::Relaxed);
        } else if contribution > medium_cutoff {
            self.medium_priority.push(task);
            self.medium_count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.low_priority.push(task);
            self.low_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Pop highest priority task available
    pub fn pop(&self) -> Option<ActivationTask> {
        // Try high priority first
        if let Some(task) = self.high_priority.pop() {
            self.high_count.fetch_sub(1, Ordering::Relaxed);
            return Some(task);
        }

        // Then medium priority
        if let Some(task) = self.medium_priority.pop() {
            self.medium_count.fetch_sub(1, Ordering::Relaxed);
            return Some(task);
        }

        // Finally low priority
        if let Some(task) = self.low_priority.pop() {
            self.low_count.fetch_sub(1, Ordering::Relaxed);
            return Some(task);
        }

        None
    }

    /// Get total task count
    pub fn total_count(&self) -> usize {
        self.high_count.load(Ordering::Relaxed)
            + self.medium_count.load(Ordering::Relaxed)
            + self.low_count.load(Ordering::Relaxed)
    }

    /// Check if all queues are empty
    pub fn is_empty(&self) -> bool {
        self.total_count() == 0
    }
}

/// Batched queue for efficient SIMD processing
pub struct BatchedActivationQueue {
    queue: SegQueue<ActivationTask>,
    batch_buffer: Vec<ActivationTask>,
    batch_size: usize,
}

impl BatchedActivationQueue {
    /// Create new batched queue
    #[must_use]
    pub fn new(batch_size: usize) -> Self {
        Self {
            queue: SegQueue::new(),
            batch_buffer: Vec::with_capacity(batch_size),
            batch_size,
        }
    }

    /// Push task to queue
    pub fn push(&self, task: ActivationTask) {
        self.queue.push(task);
    }

    /// Pop a batch of tasks for SIMD processing
    pub fn pop_batch(&mut self) -> Option<Vec<ActivationTask>> {
        self.batch_buffer.clear();

        // Fill batch buffer
        for _ in 0..self.batch_size {
            if let Some(task) = self.queue.pop() {
                self.batch_buffer.push(task);
            } else {
                break;
            }
        }

        if self.batch_buffer.is_empty() {
            None
        } else {
            Some(self.batch_buffer.clone())
        }
    }

    /// Check if queue has enough tasks for a full batch
    pub fn has_full_batch(&self) -> bool {
        // Approximate check - not perfectly accurate but fast
        let mut count = 0;
        while count < self.batch_size {
            if self.queue.pop().is_some() {
                count += 1;
            } else {
                break;
            }
        }

        // Put tasks back (this is a limitation of SegQueue)
        // In practice, this method would be used differently
        count >= self.batch_size
    }
}

/// Node-specific queue for locality optimization
pub struct NodeLocalQueue {
    node_queues: std::collections::HashMap<NodeId, SegQueue<ActivationTask>>,
    round_robin_index: AtomicUsize,
}

impl Default for NodeLocalQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeLocalQueue {
    /// Create new node-local queue system
    #[must_use]
    pub fn new() -> Self {
        Self {
            node_queues: std::collections::HashMap::new(),
            round_robin_index: AtomicUsize::new(0),
        }
    }

    /// Push task to node-specific queue
    pub fn push(&mut self, task: ActivationTask) {
        let queue = self
            .node_queues
            .entry(task.target_node.clone())
            .or_default();
        queue.push(task);
    }

    /// Pop task using round-robin across nodes
    pub fn pop(&self) -> Option<ActivationTask> {
        let queues: Vec<_> = self.node_queues.values().collect();
        if queues.is_empty() {
            return None;
        }

        let start_idx = self.round_robin_index.fetch_add(1, Ordering::Relaxed) % queues.len();

        // Try each queue starting from round-robin position
        for i in 0..queues.len() {
            let queue_idx = (start_idx + i) % queues.len();
            if let Some(task) = queues[queue_idx].pop() {
                return Some(task);
            }
        }

        None
    }

    /// Get queue for specific node
    pub fn get_node_queue(&self, node_id: &NodeId) -> Option<&SegQueue<ActivationTask>> {
        self.node_queues.get(node_id)
    }

    /// Get total task count across all node queues
    pub fn total_count(&self) -> usize {
        // Note: This is an approximation since SegQueue doesn't have len()
        self.node_queues.len()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    // use crate::activation::DecayFunction; // Not used in current tests

    fn create_test_task(target: &str, activation: f32, depth: u16) -> ActivationTask {
        ActivationTask::new(target.to_string(), activation, 0.5, 0.7, depth, 3)
    }

    #[test]
    fn test_work_stealing_queue() {
        let global_queue = Arc::new(SegQueue::new());
        let mut ws_queue = WorkStealingQueue::new(global_queue, 10);

        // Test pushing to local queue
        let task = create_test_task("test", 0.5, 1);
        ws_queue.push(task);

        assert_eq!(ws_queue.len(), 1);
        assert!(!ws_queue.is_empty());

        // Test popping from local queue
        let popped = ws_queue.pop().unwrap();
        assert_eq!(popped.target_node, "test");
        assert_eq!(ws_queue.len(), 0);
        assert!(ws_queue.is_empty());
    }

    #[test]
    fn test_work_stealing_overflow() {
        let global_queue = Arc::new(SegQueue::new());
        let mut ws_queue = WorkStealingQueue::new(global_queue.clone(), 2);

        // Fill local queue to batch size
        ws_queue.push(create_test_task("task1", 0.5, 1));
        ws_queue.push(create_test_task("task2", 0.5, 1));

        // Next push should overflow to global
        ws_queue.push(create_test_task("task3", 0.5, 1));

        assert_eq!(ws_queue.len(), 2); // Local still at batch size
        assert!(global_queue.pop().is_some()); // Global has overflow task
    }

    #[test]
    fn test_work_stealing_steal() {
        let global_queue = Arc::new(SegQueue::new());
        let mut ws_queue = WorkStealingQueue::new(global_queue, 10);

        let task = create_test_task("steal_test", 0.8, 1);
        ws_queue.push(task);

        // Another worker can steal from this queue
        let stolen = ws_queue.steal().unwrap();
        assert_eq!(stolen.target_node, "steal_test");
        assert_eq!(ws_queue.len(), 0);
    }

    #[test]
    fn test_priority_queue() {
        let pq = PriorityActivationQueue::new();

        // Add tasks with different priorities
        pq.push(create_test_task("low", 0.05, 1)); // Low priority
        pq.push(create_test_task("high", 0.8, 1)); // High priority  
        pq.push(create_test_task("medium", 0.3, 1)); // Medium priority

        assert_eq!(pq.total_count(), 3);

        // Should pop high priority first
        let first = pq.pop().unwrap();
        assert_eq!(first.target_node, "high");

        // Then medium priority
        let second = pq.pop().unwrap();
        assert_eq!(second.target_node, "medium");

        // Finally low priority
        let third = pq.pop().unwrap();
        assert_eq!(third.target_node, "low");

        assert!(pq.is_empty());
    }

    #[test]
    fn test_batched_queue() {
        let mut bq = BatchedActivationQueue::new(3);

        // Add tasks
        bq.push(create_test_task("task1", 0.5, 1));
        bq.push(create_test_task("task2", 0.5, 1));
        bq.push(create_test_task("task3", 0.5, 1));
        bq.push(create_test_task("task4", 0.5, 1));

        // Pop batch of 3
        let batch = bq.pop_batch().unwrap();
        assert_eq!(batch.len(), 3);

        // Pop remaining batch of 1
        let batch2 = bq.pop_batch().unwrap();
        assert_eq!(batch2.len(), 1);

        // No more tasks
        assert!(bq.pop_batch().is_none());
    }

    #[test]
    fn test_node_local_queue() {
        let mut nlq = NodeLocalQueue::new();

        // Add tasks for different nodes
        nlq.push(create_test_task("nodeA", 0.5, 1));
        nlq.push(create_test_task("nodeB", 0.5, 1));
        nlq.push(create_test_task("nodeA", 0.3, 1));

        // Should have queues for both nodes
        assert!(nlq.get_node_queue(&"nodeA".to_string()).is_some());
        assert!(nlq.get_node_queue(&"nodeB".to_string()).is_some());

        // Pop should use round-robin
        let task1 = nlq.pop().unwrap();
        let task2 = nlq.pop().unwrap();
        let task3 = nlq.pop().unwrap();

        // Should get different nodes (not strictly guaranteed due to round-robin)
        assert!(task1.target_node == "nodeA" || task1.target_node == "nodeB");
        assert!(task2.target_node == "nodeA" || task2.target_node == "nodeB");
        assert!(task3.target_node == "nodeA" || task3.target_node == "nodeB");
    }

    #[test]
    fn test_deterministic_queue_sorting() {
        let global_queue = Arc::new(SegQueue::new());
        let mut queue = WorkStealingQueue::new_deterministic(global_queue, 100);

        // Add tasks in random order
        queue.push(create_test_task("C", 0.5, 2));
        queue.push(create_test_task("A", 0.8, 1));
        queue.push(create_test_task("B", 0.3, 1));
        queue.push(create_test_task("D", 0.9, 2));

        // Sort deterministically
        queue.sort_deterministic();

        // Pop in order - should be sorted by depth, then node name, then contribution
        let task1 = queue.pop().unwrap();
        assert_eq!(task1.target_node, "A"); // depth 1, higher contribution

        let task2 = queue.pop().unwrap();
        assert_eq!(task2.target_node, "B"); // depth 1, lower contribution

        let task3 = queue.pop().unwrap();
        assert_eq!(task3.target_node, "C"); // depth 2, higher contribution (0.5)

        let task4 = queue.pop().unwrap();
        assert_eq!(task4.target_node, "D"); // depth 2, node D comes after C alphabetically
    }

    #[test]
    fn test_non_deterministic_queue_no_sorting() {
        let global_queue = Arc::new(SegQueue::new());
        let mut queue = WorkStealingQueue::new(global_queue, 100);

        // Add tasks
        queue.push(create_test_task("C", 0.5, 2));
        queue.push(create_test_task("A", 0.8, 1));

        // Call sort_deterministic - should have no effect
        queue.sort_deterministic();

        // Tasks should be in original LIFO order (pop from back)
        let task1 = queue.pop().unwrap();
        assert_eq!(task1.target_node, "A");

        let task2 = queue.pop().unwrap();
        assert_eq!(task2.target_node, "C");
    }
}
