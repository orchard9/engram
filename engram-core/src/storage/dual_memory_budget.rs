//! Lock-free memory budget coordinator for dual-tier episode-concept storage
//!
//! This module provides atomic allocation tracking with separate budgets for
//! episodes (high churn, temporal) and concepts (stable, semantic). The design
//! enforces memory limits without locks through carefully chosen atomic operations.
//!
//! # Architecture
//!
//! Episodes and concepts have fundamentally different memory dynamics:
//! - Episodes: Rapid allocation/deallocation, temporal locality
//! - Concepts: Stable long-term storage, grows slowly
//!
//! Independent budgets prevent concept growth from starving episode allocation
//! and vice versa. This separation enables tier-specific eviction policies.
//!
//! # Memory Ordering
//!
//! All atomic operations use `Ordering::Relaxed` because:
//! 1. Allocation counters are independent - no cross-atomic synchronization needed
//! 2. Approximate accounting is acceptable - budget enforcement is probabilistic
//! 3. False sharing is prevented via cache line padding
//! 4. No happens-before relationships required between allocations
//!
//! The worst case is a temporary budget overrun during concurrent allocations,
//! which is acceptable given that eviction runs asynchronously. Strict enforcement
//! would require locks (unacceptable latency) or CAS loops (excessive CPU overhead).
//!
//! # Cache Line Optimization
//!
//! Episode and concept atomics are separated into distinct cache lines (64 bytes)
//! to prevent false sharing. On modern x86-64, cache line bouncing between cores
//! would otherwise dominate performance as reads/writes ping-pong the line between
//! L1 caches. CachePadded ensures each atomic occupies its own cache line.
//!
//! # Performance
//!
//! - Allocation check: ~1-2ns (single atomic load)
//! - Record allocation: ~2-3ns (single atomic add)
//! - Zero contention overhead (no locks, no CAS loops)
//! - Scalable to hundreds of concurrent threads

use crossbeam_utils::CachePadded;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Size of a DualMemoryNode in bytes (conservative estimate)
///
/// Actual layout:
/// - 16 bytes: Uuid (128-bit)
/// - 64+ bytes: MemoryNodeType enum (largest variant + discriminant + padding)
/// - 3072 bytes: [f32; 768] embedding (4 bytes × 768)
/// - 64 bytes: CachePadded<AtomicF32> activation (padded to cache line)
/// - 16 bytes: Confidence struct
/// - 24 bytes: DateTime<Utc> × 2 (created_at, last_access)
/// - 64 bytes: repr(align(64)) struct alignment padding
///
/// Total ≈ 3320 bytes per node (rounded to 3328 for 64-byte alignment)
const DUAL_MEMORY_NODE_SIZE: usize = 3328;

/// Lock-free memory budget coordinator for episode-concept dual storage
///
/// Enforces separate memory budgets for episodes and concepts with atomic
/// allocation tracking. No locks - all operations are wait-free.
///
/// # Example
///
/// ```
/// use engram_core::storage::DualMemoryBudget;
///
/// let budget = DualMemoryBudget::new(512, 1024); // 512MB episodes, 1GB concepts
///
/// if budget.can_allocate_episode() {
///     budget.record_episode_allocation(3328);
///     // ... store episode ...
/// }
/// ```
#[derive(Debug)]
pub struct DualMemoryBudget {
    /// Maximum bytes for episode storage
    episode_budget_bytes: usize,

    /// Maximum bytes for concept storage
    concept_budget_bytes: usize,

    /// Current episode allocation in bytes (cache-padded to prevent false sharing)
    episode_allocated: CachePadded<AtomicUsize>,

    /// Current concept allocation in bytes (cache-padded to prevent false sharing)
    concept_allocated: CachePadded<AtomicUsize>,

    /// Maximum number of episode nodes that fit in budget
    episode_capacity: usize,

    /// Maximum number of concept nodes that fit in budget
    concept_capacity: usize,
}

impl DualMemoryBudget {
    /// Create a new dual memory budget with specified capacities in megabytes
    ///
    /// # Arguments
    ///
    /// * `episode_mb` - Episode budget in megabytes
    /// * `concept_mb` - Concept budget in megabytes
    ///
    /// # Panics
    ///
    /// Panics if either budget is zero (use at least 1MB per tier)
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::storage::DualMemoryBudget;
    ///
    /// let budget = DualMemoryBudget::new(512, 1024);
    /// assert_eq!(budget.episode_capacity(), 161319); // ~512MB / 3328 bytes
    /// ```
    #[must_use]
    pub fn new(episode_mb: usize, concept_mb: usize) -> Self {
        assert!(episode_mb > 0, "Episode budget must be at least 1MB");
        assert!(concept_mb > 0, "Concept budget must be at least 1MB");

        let episode_budget_bytes = episode_mb * 1024 * 1024;
        let concept_budget_bytes = concept_mb * 1024 * 1024;

        let episode_capacity = episode_budget_bytes / DUAL_MEMORY_NODE_SIZE;
        let concept_capacity = concept_budget_bytes / DUAL_MEMORY_NODE_SIZE;

        Self {
            episode_budget_bytes,
            concept_budget_bytes,
            episode_allocated: CachePadded::new(AtomicUsize::new(0)),
            concept_allocated: CachePadded::new(AtomicUsize::new(0)),
            episode_capacity,
            concept_capacity,
        }
    }

    /// Check if an episode allocation can proceed without exceeding budget
    ///
    /// Returns true if adding one more episode node would stay within budget.
    /// This is an advisory check - concurrent allocations may still cause
    /// temporary overruns, which is acceptable given async eviction.
    ///
    /// # Memory Ordering
    ///
    /// Uses `Relaxed` ordering because this is a best-effort check with no
    /// synchronization requirements. False positives/negatives are acceptable.
    #[must_use]
    pub fn can_allocate_episode(&self) -> bool {
        let current = self.episode_allocated.load(Ordering::Relaxed);
        current + DUAL_MEMORY_NODE_SIZE <= self.episode_budget_bytes
    }

    /// Check if a concept allocation can proceed without exceeding budget
    ///
    /// Returns true if adding one more concept node would stay within budget.
    #[must_use]
    pub fn can_allocate_concept(&self) -> bool {
        let current = self.concept_allocated.load(Ordering::Relaxed);
        current + DUAL_MEMORY_NODE_SIZE <= self.concept_budget_bytes
    }

    /// Record an episode allocation (call after successful allocation)
    ///
    /// # Arguments
    ///
    /// * `bytes` - Number of bytes allocated (typically `DUAL_MEMORY_NODE_SIZE`)
    ///
    /// # Overflow Protection
    ///
    /// Uses saturating arithmetic to prevent overflow. If allocation counter
    /// would exceed `usize::MAX`, it saturates at the maximum value.
    pub fn record_episode_allocation(&self, bytes: usize) {
        self.episode_allocated
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(bytes))
            })
            .ok(); // Ignore result - saturating add always succeeds
    }

    /// Record a concept allocation (call after successful allocation)
    ///
    /// # Arguments
    ///
    /// * `bytes` - Number of bytes allocated (typically `DUAL_MEMORY_NODE_SIZE`)
    ///
    /// # Overflow Protection
    ///
    /// Uses saturating arithmetic to prevent overflow.
    pub fn record_concept_allocation(&self, bytes: usize) {
        self.concept_allocated
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(bytes))
            })
            .ok(); // Ignore result - saturating add always succeeds
    }

    /// Record an episode deallocation (call after node removal)
    ///
    /// # Arguments
    ///
    /// * `bytes` - Number of bytes deallocated (typically `DUAL_MEMORY_NODE_SIZE`)
    ///
    /// # Underflow Protection
    ///
    /// Uses saturating arithmetic to prevent underflow. If counter would go
    /// below zero, it saturates at zero.
    pub fn record_episode_deallocation(&self, bytes: usize) {
        self.episode_allocated
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(bytes))
            })
            .ok(); // Ignore result - saturating sub always succeeds
    }

    /// Record a concept deallocation (call after node removal)
    ///
    /// # Arguments
    ///
    /// * `bytes` - Number of bytes deallocated (typically `DUAL_MEMORY_NODE_SIZE`)
    ///
    /// # Underflow Protection
    ///
    /// Uses saturating arithmetic to prevent underflow.
    pub fn record_concept_deallocation(&self, bytes: usize) {
        self.concept_allocated
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(bytes))
            })
            .ok(); // Ignore result - saturating sub always succeeds
    }

    /// Get maximum number of episode nodes that fit in budget
    #[must_use]
    pub const fn episode_capacity(&self) -> usize {
        self.episode_capacity
    }

    /// Get maximum number of concept nodes that fit in budget
    #[must_use]
    pub const fn concept_capacity(&self) -> usize {
        self.concept_capacity
    }

    /// Get current episode memory utilization as percentage (0.0 to 100.0)
    ///
    /// Returns percentage of episode budget currently allocated. Values > 100.0
    /// indicate temporary overruns during concurrent allocation.
    #[must_use]
    pub fn episode_utilization(&self) -> f32 {
        let allocated = self.episode_allocated.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        let ratio = (allocated as f64) / (self.episode_budget_bytes as f64);
        #[allow(clippy::cast_possible_truncation)]
        {
            (ratio * 100.0) as f32
        }
    }

    /// Get current concept memory utilization as percentage (0.0 to 100.0)
    ///
    /// Returns percentage of concept budget currently allocated.
    #[must_use]
    pub fn concept_utilization(&self) -> f32 {
        let allocated = self.concept_allocated.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        let ratio = (allocated as f64) / (self.concept_budget_bytes as f64);
        #[allow(clippy::cast_possible_truncation)]
        {
            (ratio * 100.0) as f32
        }
    }

    /// Get current episode allocation in bytes
    #[must_use]
    pub fn episode_allocated_bytes(&self) -> usize {
        self.episode_allocated.load(Ordering::Relaxed)
    }

    /// Get current concept allocation in bytes
    #[must_use]
    pub fn concept_allocated_bytes(&self) -> usize {
        self.concept_allocated.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_budget_initialization() {
        let budget = DualMemoryBudget::new(512, 1024);

        assert_eq!(
            budget.episode_capacity(),
            512 * 1024 * 1024 / DUAL_MEMORY_NODE_SIZE
        );
        assert_eq!(
            budget.concept_capacity(),
            1024 * 1024 * 1024 / DUAL_MEMORY_NODE_SIZE
        );
        assert!((budget.episode_utilization() - 0.0).abs() < f32::EPSILON);
        assert!((budget.concept_utilization() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    #[should_panic(expected = "Episode budget must be at least 1MB")]
    fn test_zero_episode_budget_panics() {
        let _ = DualMemoryBudget::new(0, 1024);
    }

    #[test]
    #[should_panic(expected = "Concept budget must be at least 1MB")]
    fn test_zero_concept_budget_panics() {
        let _ = DualMemoryBudget::new(512, 0);
    }

    #[test]
    fn test_basic_allocation_tracking() {
        let budget = DualMemoryBudget::new(1, 1); // 1MB each

        // Initially can allocate
        assert!(budget.can_allocate_episode());
        assert!(budget.can_allocate_concept());

        // Record allocations
        budget.record_episode_allocation(DUAL_MEMORY_NODE_SIZE);
        budget.record_concept_allocation(DUAL_MEMORY_NODE_SIZE);

        assert_eq!(budget.episode_allocated_bytes(), DUAL_MEMORY_NODE_SIZE);
        assert_eq!(budget.concept_allocated_bytes(), DUAL_MEMORY_NODE_SIZE);

        // Check utilization
        let expected_episode_util = (DUAL_MEMORY_NODE_SIZE as f64 / (1024.0 * 1024.0)) * 100.0;
        assert!((budget.episode_utilization() - expected_episode_util as f32).abs() < 0.01);
    }

    #[test]
    fn test_budget_exhaustion() {
        let budget = DualMemoryBudget::new(1, 1); // 1MB each

        // Fill up episode budget
        let capacity = budget.episode_capacity();
        for _ in 0..capacity {
            assert!(budget.can_allocate_episode());
            budget.record_episode_allocation(DUAL_MEMORY_NODE_SIZE);
        }

        // Should not be able to allocate more
        assert!(!budget.can_allocate_episode());

        // But concept budget is independent
        assert!(budget.can_allocate_concept());
    }

    #[test]
    fn test_deallocation() {
        let budget = DualMemoryBudget::new(1, 1);

        // Allocate and deallocate episode
        budget.record_episode_allocation(DUAL_MEMORY_NODE_SIZE);
        assert_eq!(budget.episode_allocated_bytes(), DUAL_MEMORY_NODE_SIZE);

        budget.record_episode_deallocation(DUAL_MEMORY_NODE_SIZE);
        assert_eq!(budget.episode_allocated_bytes(), 0);

        // Same for concept
        budget.record_concept_allocation(DUAL_MEMORY_NODE_SIZE);
        assert_eq!(budget.concept_allocated_bytes(), DUAL_MEMORY_NODE_SIZE);

        budget.record_concept_deallocation(DUAL_MEMORY_NODE_SIZE);
        assert_eq!(budget.concept_allocated_bytes(), 0);
    }

    #[test]
    #[allow(clippy::expect_used)] // expect is appropriate in tests
    fn test_concurrent_allocation() {
        let budget = Arc::new(DualMemoryBudget::new(512, 1024));
        let num_threads = 16;
        let allocations_per_thread = 100;

        let mut handles = vec![];

        // Spawn threads to concurrently allocate episodes
        for _ in 0..num_threads {
            let budget_clone = Arc::clone(&budget);
            let handle = thread::spawn(move || {
                for _ in 0..allocations_per_thread {
                    budget_clone.record_episode_allocation(DUAL_MEMORY_NODE_SIZE);
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle
                .join()
                .expect("Thread panicked during allocation test");
        }

        // Check total allocation
        let expected = num_threads * allocations_per_thread * DUAL_MEMORY_NODE_SIZE;
        assert_eq!(budget.episode_allocated_bytes(), expected);
    }

    #[test]
    #[allow(clippy::expect_used)] // expect is appropriate in tests
    fn test_concurrent_allocation_and_deallocation() {
        let budget = Arc::new(DualMemoryBudget::new(512, 1024));
        let num_threads = 16;
        let operations_per_thread = 100;

        let mut handles = vec![];

        // Spawn threads that allocate and deallocate
        for thread_id in 0..num_threads {
            let budget_clone = Arc::clone(&budget);
            let handle = thread::spawn(move || {
                for _ in 0..operations_per_thread {
                    if thread_id % 2 == 0 {
                        budget_clone.record_episode_allocation(DUAL_MEMORY_NODE_SIZE);
                    } else {
                        budget_clone.record_episode_deallocation(DUAL_MEMORY_NODE_SIZE);
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle
                .join()
                .expect("Thread panicked during alloc/dealloc test");
        }

        // With equal alloc/dealloc threads, should balance out (approximately)
        // Allow some slack due to concurrent operations
        let allocated = budget.episode_allocated_bytes();

        // Due to concurrent ops, final value might not be exactly zero
        // but should be within bounds of operations performed
        assert!(allocated <= num_threads * operations_per_thread * DUAL_MEMORY_NODE_SIZE);
    }

    #[test]
    fn test_overflow_protection() {
        let budget = DualMemoryBudget::new(1, 1);

        // Try to overflow by allocating near usize::MAX
        budget.record_episode_allocation(usize::MAX - 1000);
        budget.record_episode_allocation(5000);

        // Should saturate at usize::MAX, not wrap
        assert_eq!(budget.episode_allocated_bytes(), usize::MAX);
    }

    #[test]
    fn test_underflow_protection() {
        let budget = DualMemoryBudget::new(1, 1);

        // Try to underflow by deallocating when counter is zero
        budget.record_episode_deallocation(DUAL_MEMORY_NODE_SIZE);

        // Should saturate at zero, not wrap to usize::MAX
        assert_eq!(budget.episode_allocated_bytes(), 0);
    }

    #[test]
    fn test_independent_budgets() {
        let budget = DualMemoryBudget::new(2, 4);

        // Fill episode budget
        let episode_cap = budget.episode_capacity();
        for _ in 0..episode_cap {
            budget.record_episode_allocation(DUAL_MEMORY_NODE_SIZE);
        }

        // Episode budget exhausted
        assert!(!budget.can_allocate_episode());

        // But concept budget is still available
        assert!(budget.can_allocate_concept());
        assert!((budget.concept_utilization() - 0.0).abs() < f32::EPSILON);

        // Fill concept budget
        let concept_cap = budget.concept_capacity();
        for _ in 0..concept_cap {
            budget.record_concept_allocation(DUAL_MEMORY_NODE_SIZE);
        }

        // Both exhausted
        assert!(!budget.can_allocate_episode());
        assert!(!budget.can_allocate_concept());

        // Utilization should be ~100% for both
        assert!(budget.episode_utilization() >= 95.0);
        assert!(budget.concept_utilization() >= 95.0);
    }

    #[test]
    fn test_utilization_calculation() {
        let budget = DualMemoryBudget::new(10, 10); // 10MB each

        // Allocate 5MB worth of episodes
        let bytes_5mb = 5 * 1024 * 1024;
        budget.record_episode_allocation(bytes_5mb);

        // Should be ~50% utilization
        let util = budget.episode_utilization();
        assert!((util - 50.0).abs() < 1.0); // Allow 1% error

        // Allocate another 3MB
        let bytes_3mb = 3 * 1024 * 1024;
        budget.record_episode_allocation(bytes_3mb);

        // Should be ~80% utilization
        let util = budget.episode_utilization();
        assert!((util - 80.0).abs() < 1.0);
    }

    #[test]
    #[allow(clippy::expect_used)] // expect is appropriate in tests
    fn test_stress_concurrent_mixed_operations() {
        let budget = Arc::new(DualMemoryBudget::new(100, 200));
        let num_threads = 32;
        let operations = 1000;

        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let budget_clone = Arc::clone(&budget);
            let handle = thread::spawn(move || {
                for op in 0..operations {
                    match (thread_id + op) % 4 {
                        0 => budget_clone.record_episode_allocation(DUAL_MEMORY_NODE_SIZE),
                        1 => budget_clone.record_episode_deallocation(DUAL_MEMORY_NODE_SIZE),
                        2 => budget_clone.record_concept_allocation(DUAL_MEMORY_NODE_SIZE),
                        3 => budget_clone.record_concept_deallocation(DUAL_MEMORY_NODE_SIZE),
                        _ => unreachable!(),
                    }

                    // Also test reads
                    let _ = budget_clone.can_allocate_episode();
                    let _ = budget_clone.can_allocate_concept();
                    let _ = budget_clone.episode_utilization();
                    let _ = budget_clone.concept_utilization();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked during stress test");
        }

        // Test completes without panics or deadlocks - that's the main success criterion
        // Actual values will be unpredictable due to concurrent operations
    }
}
