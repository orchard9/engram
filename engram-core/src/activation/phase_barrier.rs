use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// A phase barrier that synchronizes worker threads with shutdown awareness.
///
/// This barrier ensures all threads reach the same synchronization point before
/// proceeding, while also supporting graceful shutdown by checking a shutdown
/// signal and allowing early exit.
pub struct PhaseBarrier {
    /// Number of threads that must reach the barrier
    num_threads: usize,
    /// Current generation counter
    generation: AtomicUsize,
    /// Number of threads that have reached current phase
    arrived: AtomicUsize,
    /// Whether the barrier is enabled
    enabled: AtomicBool,
    /// Shutdown signal to interrupt waiting
    shutdown: Arc<AtomicBool>,
    /// Condition variable for efficient waiting
    condvar: Condvar,
    /// Mutex protecting the condition variable
    mutex: Mutex<()>,
}

impl PhaseBarrier {
    /// Create a new phase barrier for the specified number of threads.
    ///
    /// # Arguments
    /// * `num_threads` - Number of threads that must synchronize
    /// * `shutdown` - Shared shutdown signal to check during waiting
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(num_threads: usize, shutdown: Arc<AtomicBool>) -> Self {
        Self {
            num_threads,
            generation: AtomicUsize::new(0),
            arrived: AtomicUsize::new(0),
            enabled: AtomicBool::new(true),
            shutdown,
            condvar: Condvar::new(),
            mutex: Mutex::new(()),
        }
    }

    /// Wait at the barrier until all threads have arrived or shutdown is signaled.
    ///
    /// # Returns
    /// * `true` if the barrier completed normally
    /// * `false` if shutdown was detected or barrier was disabled
    #[must_use]
    pub fn wait(&self) -> bool {
        // Fast path: check if barrier is disabled
        if !self.enabled.load(Ordering::Acquire) {
            return false;
        }

        // Record current generation before arriving
        let generation = self.generation.load(Ordering::Acquire);

        // Atomically increment arrived count
        let arrived = self.arrived.fetch_add(1, Ordering::AcqRel) + 1;

        if arrived == self.num_threads {
            // Last thread to arrive, wake everyone
            self.arrived.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
            self.condvar.notify_all();
            return true;
        }

        // Wait for barrier completion or shutdown
        let timeout = Duration::from_millis(100);
        let Ok(mut guard) = self.mutex.lock() else {
            // Mutex poisoned, treat as shutdown
            self.arrived.fetch_sub(1, Ordering::Release);
            return false;
        };

        while self.generation.load(Ordering::Acquire) == generation {
            // Check shutdown conditions
            if self.shutdown.load(Ordering::Acquire) || !self.enabled.load(Ordering::Acquire) {
                // Shutdown detected, decrement arrived count and exit
                self.arrived.fetch_sub(1, Ordering::Release);
                return false;
            }

            // Wait with timeout to periodically check shutdown
            if let Ok(result) = self.condvar.wait_timeout(guard, timeout) {
                guard = result.0;
            } else {
                // Mutex poisoned, treat as shutdown
                self.arrived.fetch_sub(1, Ordering::Release);
                return false;
            }
        }

        true
    }

    /// Disable the barrier, causing all waiting threads to return false.
    ///
    /// This is used during shutdown to unblock any threads stuck waiting.
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
        self.condvar.notify_all();
    }

    /// Enable the barrier for synchronization.
    ///
    /// This re-enables the barrier after it has been disabled.
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
        // Wake any threads that might be waiting
        self.condvar.notify_all();
    }

    /// Reset the barrier to its initial state.
    ///
    /// This should only be called when no threads are waiting.
    pub fn reset(&self) {
        self.arrived.store(0, Ordering::Release);
        self.generation.store(0, Ordering::Release);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::thread;

    #[test]
    fn test_basic_barrier_synchronization() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(PhaseBarrier::new(3, shutdown));
        let counter = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];
        for _ in 0..3 {
            let barrier = barrier.clone();
            let counter = counter.clone();

            let handle = thread::spawn(move || {
                // Increment counter before barrier
                counter.fetch_add(1, Ordering::SeqCst);

                // Wait at barrier
                assert!(barrier.wait());

                // All threads should see count == 3 after barrier
                assert_eq!(counter.load(Ordering::SeqCst), 3);
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Worker thread panicked");
        }
    }

    #[test]
    fn test_barrier_with_shutdown() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(PhaseBarrier::new(3, shutdown.clone()));

        let mut handles = vec![];

        // Start two threads that will wait
        for _ in 0..2 {
            let barrier = barrier.clone();

            let handle = thread::spawn(move || {
                // This should return false due to shutdown
                !barrier.wait()
            });

            handles.push(handle);
        }

        // Give threads time to reach barrier
        thread::sleep(Duration::from_millis(50));

        // Signal shutdown
        shutdown.store(true, Ordering::SeqCst);

        // All threads should exit with true (indicating shutdown)
        for handle in handles {
            assert!(handle.join().expect("Worker thread panicked"));
        }
    }

    #[test]
    fn test_barrier_disable() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(PhaseBarrier::new(2, shutdown));

        let barrier_clone = barrier.clone();
        let handle = thread::spawn(move || {
            // This should return false due to disable
            !barrier_clone.wait()
        });

        // Give thread time to start waiting
        thread::sleep(Duration::from_millis(50));

        // Disable the barrier
        barrier.disable();

        // Thread should exit with true (indicating disabled)
        assert!(handle.join().expect("Worker thread panicked"));
    }

    #[test]
    fn test_barrier_reset() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let barrier = PhaseBarrier::new(2, shutdown);

        // Simulate partial arrival
        barrier.arrived.store(1, Ordering::SeqCst);
        barrier.generation.store(5, Ordering::SeqCst);

        // Reset barrier
        barrier.reset();

        // Verify reset state
        assert_eq!(barrier.arrived.load(Ordering::SeqCst), 0);
        assert_eq!(barrier.generation.load(Ordering::SeqCst), 0);
    }
}
