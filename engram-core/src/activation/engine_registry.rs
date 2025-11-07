//! Global engine registry to prevent concurrent engine instances in tests
//!
//! This module provides a process-wide registry to track active ParallelSpreadingEngine
//! instances and prevent concurrent creation that can lead to deadlocks when running
//! tests with --test-threads=1.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Weak};
use std::thread::ThreadId;

/// Global registry tracking all active engines
static ENGINE_REGISTRY: LazyLock<Mutex<EngineRegistry>> =
    LazyLock::new(|| Mutex::new(EngineRegistry::new()));

/// Registry tracking active ParallelSpreadingEngine instances
struct EngineRegistry {
    /// Maps thread ID to weak references of engines created by that thread
    engines_by_thread: HashMap<ThreadId, Vec<Weak<EngineHandle>>>,
    /// Total number of active engines across all threads
    total_active: usize,
}

impl EngineRegistry {
    fn new() -> Self {
        Self {
            engines_by_thread: HashMap::new(),
            total_active: 0,
        }
    }

    /// Register a new engine and return a handle
    fn register(&mut self) -> Result<Arc<EngineHandle>, String> {
        // Clean up dead weak references first
        self.cleanup_dead_refs();

        let thread_id = std::thread::current().id();

        // Check if this thread already has an ACTIVE engine (not just existing)
        if let Some(engines) = self.engines_by_thread.get(&thread_id) {
            let has_active = engines.iter().any(|weak| {
                weak.upgrade()
                    .is_some_and(|handle| handle.active.load(Ordering::Acquire))
            });

            if has_active {
                return Err(format!(
                    "Thread {thread_id:?} already has an active engine. Multiple concurrent engines per thread not supported."
                ));
            }
        }

        // In test mode, check for active engines with minimal waiting
        if cfg!(test) && std::env::var("RUST_TEST_THREADS").as_deref() == Ok("1") {
            // Only check when running with --test-threads=1
            let active_count = self
                .engines_by_thread
                .values()
                .flat_map(|engines| engines.iter())
                .filter_map(std::sync::Weak::upgrade)
                .filter(|handle| handle.active.load(Ordering::Acquire))
                .count();

            if active_count > 0 {
                // Give a brief moment for engines to deactivate
                std::thread::sleep(std::time::Duration::from_millis(50));

                // Final check after sleep
                let active_count = self
                    .engines_by_thread
                    .values()
                    .flat_map(|engines| engines.iter())
                    .filter_map(std::sync::Weak::upgrade)
                    .filter(|handle| handle.active.load(Ordering::Acquire))
                    .count();

                if active_count > 0 {
                    // Debug: show which threads have active engines
                    let active_threads: Vec<_> = self
                        .engines_by_thread
                        .iter()
                        .filter(|(_, engines)| {
                            engines.iter().any(|w| {
                                w.upgrade()
                                    .is_some_and(|h| h.active.load(Ordering::Acquire))
                            })
                        })
                        .map(|(tid, _)| format!("{tid:?}"))
                        .collect();

                    return Err(format!(
                        "Cannot create engine: {} engine(s) still active in threads: {}. Engines must complete spreading before creating new ones.",
                        active_count,
                        active_threads.join(", ")
                    ));
                }
            }
        }

        // Create new handle (initially inactive)
        let handle = Arc::new(EngineHandle {
            id: ENGINE_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            thread_id,
            active: AtomicBool::new(false),
        });

        // Register it
        self.engines_by_thread
            .entry(thread_id)
            .or_default()
            .push(Arc::downgrade(&handle));

        self.total_active += 1;

        Ok(handle)
    }

    /// Unregister an engine when it's dropped
    fn unregister(&mut self, handle: &EngineHandle) {
        if let Some(engines) = self.engines_by_thread.get_mut(&handle.thread_id) {
            engines.retain(|weak| {
                weak.strong_count() > 0 && weak.upgrade().is_none_or(|h| h.id != handle.id)
            });

            if engines.is_empty() {
                self.engines_by_thread.remove(&handle.thread_id);
            }
        }

        self.total_active = self.total_active.saturating_sub(1);
    }

    /// Clean up dead weak references
    fn cleanup_dead_refs(&mut self) {
        let mut empty_threads = Vec::new();

        for (thread_id, engines) in &mut self.engines_by_thread {
            engines.retain(|weak| weak.strong_count() > 0);
            if engines.is_empty() {
                empty_threads.push(*thread_id);
            }
        }

        for thread_id in empty_threads {
            self.engines_by_thread.remove(&thread_id);
        }

        // Recompute total active
        self.total_active = self
            .engines_by_thread
            .values()
            .flat_map(|engines| engines.iter())
            .filter(|weak| weak.strong_count() > 0)
            .count();
    }
}

static ENGINE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Handle representing an active engine registration
pub struct EngineHandle {
    id: usize,
    thread_id: ThreadId,
    active: AtomicBool,
}

impl EngineHandle {
    /// Mark this engine as active (during spread_activation)
    pub fn activate(&self) {
        self.active.store(true, Ordering::Release);
    }

    /// Mark this engine as inactive (after spread_activation completes)
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::Release);
    }
}

impl Drop for EngineHandle {
    fn drop(&mut self) {
        // Ensure we're marked as inactive before dropping
        self.active.store(false, Ordering::Release);
        if let Ok(mut registry) = ENGINE_REGISTRY.lock() {
            registry.unregister(self);
        }
    }
}

/// Register a new engine instance
///
/// This should be called when creating a ParallelSpreadingEngine to ensure
/// proper isolation between engine instances, especially in test environments.
///
/// # Errors
/// Returns an error if another engine is already active in the current thread
/// or if running in test mode and another engine exists globally.
pub fn register_engine() -> Result<Arc<EngineHandle>, String> {
    ENGINE_REGISTRY
        .lock()
        .map_err(|_| "Engine registry lock poisoned".to_string())?
        .register()
}

/// Check if any engines are currently active
///
/// Useful for debugging and test assertions.
#[must_use]
pub fn has_active_engines() -> bool {
    ENGINE_REGISTRY
        .lock()
        .map(|mut registry| {
            registry.cleanup_dead_refs();
            registry.total_active > 0
        })
        .unwrap_or(false)
}

/// Get the number of active engines
///
/// Useful for debugging and test assertions.
#[must_use]
pub fn active_engine_count() -> usize {
    ENGINE_REGISTRY
        .lock()
        .map(|mut registry| {
            registry.cleanup_dead_refs();
            registry.total_active
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::unwrap_used)]
    fn test_single_engine_registration() {
        let handle = register_engine().unwrap();
        assert_eq!(active_engine_count(), 1);

        drop(handle);
        assert_eq!(active_engine_count(), 0);
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn test_multiple_engines_same_thread_prevented() {
        let handle1 = register_engine().unwrap();

        // Mark first engine as active (simulating spread_activation)
        handle1.activate();

        // Now second engine should fail
        let result = register_engine();
        assert!(
            result.is_err(),
            "Second engine in same thread should fail while first is active"
        );

        // Deactivate to clean up
        handle1.deactivate();
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn test_sequential_engines_allowed() {
        {
            let _handle1 = register_engine().unwrap();
            assert_eq!(active_engine_count(), 1);
        }

        // After drop, should be able to create another
        let _handle2 = register_engine().unwrap();
        assert_eq!(active_engine_count(), 1);
    }
}
