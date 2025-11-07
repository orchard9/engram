//! Test isolation utilities for parallel spreading engine tests
//!
//! Provides process-based isolation for tests that create multiple engine instances
//! to prevent interference when running with --test-threads=1.

use std::env;
use std::process::{Command, Stdio};

/// Run a test function in an isolated subprocess to prevent engine interference.
///
/// This is the industry-standard solution for tests that manage their own thread pools
/// or have complex lifecycle requirements that don't play well with sequential execution.
///
/// # Example
/// ```no_run
/// # use engram_core::activation::test_isolation::run_isolated_test;
/// #[test]
/// fn test_my_engine() {
///     run_isolated_test("test_my_engine", || {
///         // Your actual test code here
///         // This runs in a separate process
///     });
/// }
/// ```
#[allow(clippy::unwrap_used, clippy::panic)] // Test utility that should panic on failures
pub fn run_isolated_test<F>(test_name: &str, test_fn: F)
where
    F: FnOnce() + Send + 'static,
{
    // Check if we're already in isolated mode
    if env::var("ENGRAM_TEST_ISOLATED").is_ok() {
        // We're in the subprocess, run the actual test
        test_fn();
        return;
    }

    // We're in the parent process, spawn a subprocess
    let exe = env::current_exe().unwrap();

    // Build the command to run just this test
    let output = Command::new(exe)
        .env("ENGRAM_TEST_ISOLATED", "1")
        .env("RUST_TEST_THREADS", "1") // Ensure subprocess runs single-threaded
        .arg("--test")
        .arg(test_name)
        .arg("--exact")
        .arg("--nocapture")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();

    // Check if the test passed
    if !output.status.success() {
        eprintln!("Isolated test {test_name} failed");
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Isolated test {test_name} failed");
    }
}

/// Alternative: Use thread-local engine registry to detect conflicts
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

thread_local! {
    static ENGINE_ACTIVE: RefCell<bool> = const { RefCell::new(false) };
}

static GLOBAL_ENGINE_COUNT: AtomicBool = AtomicBool::new(false);

/// Guard that ensures only one engine exists at a time in tests.
///
/// This is a lighter-weight alternative to process isolation that works
/// for most cases but may still have issues with truly pathological thread
/// interaction patterns.
pub struct EngineTestGuard {
    _private: (),
}

impl EngineTestGuard {
    /// Acquire exclusive access to create an engine in tests.
    ///
    /// # Panics
    /// Panics if another engine is already active in this thread or globally.
    #[allow(clippy::panic)] // This is a test utility that should panic on misuse
    pub fn acquire() -> Self {
        // Check thread-local first
        ENGINE_ACTIVE.with(|active| {
            assert!(
                !*active.borrow(),
                "Another engine is already active in this thread"
            );
            *active.borrow_mut() = true;
        });

        // Check global
        if GLOBAL_ENGINE_COUNT.swap(true, Ordering::SeqCst) {
            // Release thread-local on failure
            ENGINE_ACTIVE.with(|active| *active.borrow_mut() = false);
            panic!("Another engine is already active globally");
        }

        Self { _private: () }
    }
}

impl Drop for EngineTestGuard {
    fn drop(&mut self) {
        // Release in reverse order
        GLOBAL_ENGINE_COUNT.store(false, Ordering::SeqCst);
        ENGINE_ACTIVE.with(|active| *active.borrow_mut() = false);

        // Give OS time to clean up threads
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_guard_prevents_concurrent_creation() {
        let _guard1 = EngineTestGuard::acquire();

        // This should panic
        let result = std::panic::catch_unwind(|| {
            let _guard2 = EngineTestGuard::acquire();
        });

        assert!(result.is_err(), "Second guard should have panicked");
    }

    #[test]
    fn test_engine_guard_allows_sequential_creation() {
        {
            let _guard1 = EngineTestGuard::acquire();
            // Guard dropped here
        }

        // Should succeed after first guard is dropped
        let _guard2 = EngineTestGuard::acquire();
    }
}
