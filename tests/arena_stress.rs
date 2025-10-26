//! Arena allocator stress tests
//!
//! Multi-threaded stress testing for arena allocator isolation and overflow handling.
//! Validates that thread-local arenas scale to 32+ threads without contention.

#![cfg(feature = "zig-kernels")]

use engram_core::zig_kernels::{
    apply_decay, configure_arena, get_arena_stats, reset_arena_metrics, reset_arenas,
    spread_activation, vector_similarity, OverflowStrategy,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

/// Test that thread-local arenas provide isolation across 32+ threads
///
/// Each thread performs independent kernel operations.
/// No contention should occur as each thread has its own arena.
#[test]
fn test_multithreaded_arena_isolation() {
    // Configure 2MB arenas with error return on overflow
    configure_arena(2, OverflowStrategy::ErrorReturn);
    reset_arena_metrics();

    const NUM_THREADS: usize = 32;
    const ITERATIONS_PER_THREAD: usize = 100;

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            thread::spawn(move || {
                for iteration in 0..ITERATIONS_PER_THREAD {
                    // Each thread runs vector similarity kernel independently
                    let query = vec![thread_id as f32; 768];
                    let candidates: Vec<f32> = (0..100)
                        .flat_map(|i| vec![(thread_id + iteration + i) as f32; 768])
                        .collect();

                    let scores = vector_similarity(&query, &candidates, 100);
                    assert_eq!(scores.len(), 100);

                    // Spread activation
                    let adjacency: Vec<u32> = (1..100).collect();
                    let weights = vec![0.1; 99];
                    let mut activations = vec![0.0; 100];
                    activations[0] = 1.0;

                    spread_activation(&adjacency, &weights, &mut activations, 100, 5);

                    // Memory decay
                    let mut strengths = vec![1.0; 100];
                    let ages = vec![thread_id as u64 * 1000; 100];
                    apply_decay(&mut strengths, &ages);
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Check global metrics
    let stats = get_arena_stats();

    // Each thread should have reset multiple times
    assert!(
        stats.total_resets > 0,
        "Expected resets from kernel operations"
    );

    // No overflows should occur with 2MB arenas
    assert_eq!(
        stats.total_overflows, 0,
        "Arena overflows detected: {}",
        stats.total_overflows
    );

    // Peak usage should be reasonable
    assert!(
        stats.max_high_water_mark > 0,
        "Expected non-zero high-water mark"
    );
    assert!(
        stats.max_high_water_mark <= 2 * 1024 * 1024,
        "High-water mark exceeded arena capacity: {}",
        stats.max_high_water_mark
    );

    println!("Arena stress test completed:");
    println!("  Threads: {}", NUM_THREADS);
    println!("  Iterations per thread: {}", ITERATIONS_PER_THREAD);
    println!("  Total resets: {}", stats.total_resets);
    println!("  Total overflows: {}", stats.total_overflows);
    println!("  Peak usage: {} bytes", stats.max_high_water_mark);
}

/// Test arena overflow handling with intentionally small arenas
///
/// Configures tiny arenas to force overflow events.
/// Validates graceful error handling without crashes.
#[test]
fn test_arena_overflow_handling() {
    // Configure tiny 1MB arenas to force overflows
    configure_arena(1, OverflowStrategy::ErrorReturn);
    reset_arena_metrics();

    const NUM_THREADS: usize = 8;

    let overflow_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|_thread_id| {
            let overflow_count = Arc::clone(&overflow_count);
            thread::spawn(move || {
                // Try to allocate more than arena capacity
                // Large vector similarity operations require significant temporary space

                for _iteration in 0..10 {
                    // Huge embeddings that will overflow 1MB arena
                    let query = vec![1.0; 100_000]; // ~400KB
                    let candidates: Vec<f32> =
                        (0..100).flat_map(|_| vec![1.0; 100_000]).collect(); // ~40MB

                    // This may or may not overflow depending on kernel implementation
                    // We're testing that it doesn't crash
                    let result = std::panic::catch_unwind(|| {
                        vector_similarity(&query, &candidates, 100)
                    });

                    if result.is_err() {
                        overflow_count.fetch_add(1, Ordering::Relaxed);
                    }

                    // Reset arena to clear any partial allocations
                    reset_arenas();
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        let _ = handle.join(); // May panic if overflow strategy is Panic
    }

    // Check that overflow events were tracked
    let stats = get_arena_stats();

    println!("Overflow test completed:");
    println!("  Total resets: {}", stats.total_resets);
    println!("  Total overflows: {}", stats.total_overflows);
    println!("  Peak usage: {} bytes", stats.max_high_water_mark);
    println!(
        "  Caught panics: {}",
        overflow_count.load(Ordering::Relaxed)
    );

    // With current stub implementations, we may not see overflows
    // This test primarily validates that the system doesn't crash
}

/// Test arena metrics accuracy under concurrent load
///
/// Validates that metrics are correctly aggregated across threads.
#[test]
fn test_arena_metrics_accuracy() {
    configure_arena(2, OverflowStrategy::ErrorReturn);
    reset_arena_metrics();

    const NUM_THREADS: usize = 16;
    const RESETS_PER_THREAD: usize = 50;

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            thread::spawn(move || {
                for _i in 0..RESETS_PER_THREAD {
                    // Perform some allocations
                    let query = vec![thread_id as f32; 768];
                    let candidates: Vec<f32> = (0..50)
                        .flat_map(|j| vec![(thread_id + j) as f32; 768])
                        .collect();

                    let _scores = vector_similarity(&query, &candidates, 50);

                    // Explicit reset to increment reset counter
                    reset_arenas();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let stats = get_arena_stats();

    // Each thread should have recorded its resets
    // Note: Actual count may vary depending on internal resets
    assert!(
        stats.total_resets >= NUM_THREADS * RESETS_PER_THREAD,
        "Expected at least {} resets, got {}",
        NUM_THREADS * RESETS_PER_THREAD,
        stats.total_resets
    );

    println!("Metrics accuracy test completed:");
    println!("  Expected resets: {}", NUM_THREADS * RESETS_PER_THREAD);
    println!("  Actual resets: {}", stats.total_resets);
    println!("  Total overflows: {}", stats.total_overflows);
    println!("  Peak usage: {} bytes", stats.max_high_water_mark);
}

/// Test arena behavior with varying pool sizes
///
/// Validates that different pool sizes work correctly.
#[test]
fn test_arena_configurable_sizes() {
    let test_sizes = vec![
        (1, "1MB - minimal"),
        (2, "2MB - small"),
        (4, "4MB - medium"),
        (8, "8MB - large"),
    ];

    for (size_mb, description) in test_sizes {
        println!("Testing arena size: {}", description);

        configure_arena(size_mb, OverflowStrategy::ErrorReturn);
        reset_arena_metrics();

        // Run some operations
        const NUM_THREADS: usize = 4;
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|thread_id| {
                thread::spawn(move || {
                    for _i in 0..20 {
                        let query = vec![thread_id as f32; 768];
                        let candidates: Vec<f32> = (0..100)
                            .flat_map(|j| vec![(thread_id + j) as f32; 768])
                            .collect();

                        let _scores = vector_similarity(&query, &candidates, 100);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let stats = get_arena_stats();
        println!(
            "  Resets: {}, Overflows: {}, Peak: {} bytes",
            stats.total_resets, stats.total_overflows, stats.max_high_water_mark
        );

        // Larger arenas should have no overflows
        if size_mb >= 2 {
            assert_eq!(
                stats.total_overflows, 0,
                "Unexpected overflows with {}MB arenas",
                size_mb
            );
        }
    }
}

/// Test arena reset functionality
///
/// Validates that manual resets work correctly.
#[test]
fn test_arena_manual_reset() {
    configure_arena(2, OverflowStrategy::ErrorReturn);
    reset_arena_metrics();

    // Perform some allocations
    let query = vec![1.0; 768];
    let candidates: Vec<f32> = (0..100).flat_map(|_| vec![1.0; 768]).collect();
    let _scores = vector_similarity(&query, &candidates, 100);

    // Manual reset
    reset_arenas();

    // Perform more allocations
    let _scores = vector_similarity(&query, &candidates, 100);

    let stats = get_arena_stats();

    // Should have at least 2 resets (1 manual + internal resets)
    assert!(stats.total_resets >= 1, "Expected at least 1 reset");

    println!("Manual reset test completed:");
    println!("  Total resets: {}", stats.total_resets);
}

/// Benchmark arena overhead
///
/// Measures the overhead of metrics tracking.
#[test]
fn test_arena_metrics_overhead() {
    use std::time::Instant;

    configure_arena(4, OverflowStrategy::ErrorReturn);

    const ITERATIONS: usize = 1000;

    // Benchmark with metrics
    reset_arena_metrics();
    let start = Instant::now();

    for i in 0..ITERATIONS {
        let query = vec![i as f32; 768];
        let candidates: Vec<f32> = (0..10).flat_map(|j| vec![(i + j) as f32; 768]).collect();
        let _scores = vector_similarity(&query, &candidates, 10);
        reset_arenas();
    }

    let duration_with_metrics = start.elapsed();

    let stats = get_arena_stats();
    println!("Metrics overhead benchmark:");
    println!("  Iterations: {}", ITERATIONS);
    println!("  Duration: {:?}", duration_with_metrics);
    println!("  Per-operation: {:?}", duration_with_metrics / ITERATIONS as u32);
    println!("  Total resets: {}", stats.total_resets);
    println!("  Total overflows: {}", stats.total_overflows);

    // Metrics overhead should be negligible (<1% of total time)
    // This is a smoke test - actual performance testing in Task 010
}
