//! Comprehensive tests for GPU OOM handling and memory pressure management
//!
//! This test suite validates that:
//! - OOM conditions are detected and handled gracefully
//! - Automatic CPU fallback maintains identical behavior
//! - Batch splitting works correctly under memory constraints
//! - Memory limits are enforced properly
//! - No crashes occur due to OOM

#[cfg(cuda_available)]
use engram_core::compute::cuda::{
    hybrid::{HybridConfig, HybridExecutor},
    memory_pressure::{MemoryPressureMonitor, get_memory_pressure_monitor},
    performance_tracker::Operation,
};

#[test]
#[cfg(cuda_available)]
fn test_memory_pressure_monitor_creation() {
    // Should create successfully if GPU available
    if let Some(monitor) = MemoryPressureMonitor::new() {
        println!(
            "Total VRAM: {:.2} GB",
            monitor.total_vram() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!(
            "VRAM limit: {:.2} GB",
            monitor.vram_limit() as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        assert!(monitor.total_vram() > 0);
        assert!(monitor.vram_limit() > 0);
        assert!(monitor.vram_limit() <= monitor.total_vram());
    } else {
        println!("GPU not available, skipping memory monitor test");
    }
}

#[test]
#[cfg(cuda_available)]
fn test_batch_size_adaptation() {
    if let Some(monitor) = MemoryPressureMonitor::new() {
        // Simulate 768-dim vectors (3KB each)
        let per_vector = 768 * std::mem::size_of::<f32>();

        // Small batch should fit
        let small_batch = 100;
        let safe_small = monitor.calculate_safe_batch_size(small_batch, per_vector);
        println!(
            "Small batch: requested {}, safe {}",
            small_batch, safe_small
        );
        // Either it fits entirely, or we're under extreme memory pressure
        assert!(safe_small > 0);

        // Huge batch should be reduced
        let huge_batch = 10_000_000; // ~30GB worth of vectors
        let safe_huge = monitor.calculate_safe_batch_size(huge_batch, per_vector);
        println!("Huge batch: requested {}, safe {}", huge_batch, safe_huge);

        assert!(safe_huge < huge_batch, "Huge batch should be reduced");
        assert!(safe_huge >= 1, "Should always allow at least 1 item");

        // Verify safe batch actually fits in VRAM limit
        let required_memory = safe_huge * per_vector;
        if let Some(available) = monitor.query_available_vram() {
            assert!(
                required_memory <= available,
                "Safe batch should fit in available VRAM: required {} MB, available {} MB",
                required_memory / (1024 * 1024),
                available / (1024 * 1024)
            );
        }
    } else {
        println!("GPU not available, skipping batch adaptation test");
    }
}

#[test]
#[cfg(cuda_available)]
fn test_chunked_processing() {
    if let Some(monitor) = MemoryPressureMonitor::new() {
        let items = vec![1.0f32; 10_000];
        let per_item = 1024; // 1KB per item

        let results = monitor.process_in_chunks(&items, per_item, |chunk| {
            // Simulate processing: double each value
            chunk.iter().map(|x| x * 2.0).collect()
        });

        assert_eq!(results.len(), items.len());
        assert!(results.iter().all(|&x| (x - 2.0).abs() < 1e-6));
        println!("Successfully processed {} items in chunks", results.len());
    } else {
        println!("GPU not available, skipping chunked processing test");
    }
}

#[test]
#[cfg(cuda_available)]
fn test_oom_safe_execution_small_batch() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping OOM safe execution test");
        return;
    }

    let executor = HybridExecutor::new(HybridConfig::default());

    // Small batch that should fit in VRAM
    let query = [1.0f32; 768];
    let targets = vec![[0.5f32; 768]; 256];

    let results = executor.execute_batch_cosine_similarity_oom_safe(&query, &targets);

    assert_eq!(results.len(), targets.len());
    for sim in &results {
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Similarity should be ~1.0, got {}",
            sim
        );
    }

    println!("Small batch executed successfully");
}

#[test]
#[cfg(cuda_available)]
fn test_oom_safe_execution_large_batch() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping large batch test");
        return;
    }

    let executor = HybridExecutor::new(HybridConfig::default());

    // Large batch that might require chunking
    let query = [1.0f32; 768];
    let targets = vec![[1.0f32; 768]; 10_000];

    let results = executor.execute_batch_cosine_similarity_oom_safe(&query, &targets);

    assert_eq!(results.len(), targets.len());
    for (i, sim) in results.iter().enumerate() {
        assert!(
            (*sim - 1.0).abs() < 1e-6,
            "Similarity at index {} should be ~1.0, got {}",
            i,
            sim
        );
    }

    println!("Large batch executed successfully (possibly chunked)");
}

#[test]
#[cfg(cuda_available)]
fn test_cpu_gpu_equivalence_with_oom_safe() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping CPU/GPU equivalence test");
        return;
    }

    // CPU-only executor
    let cpu_config = HybridConfig {
        force_cpu_mode: true,
        ..Default::default()
    };
    let cpu_executor = HybridExecutor::new(cpu_config);

    // Hybrid executor with OOM protection
    let hybrid_executor = HybridExecutor::new(HybridConfig::default());

    let query = [0.5f32; 768];
    let targets = vec![[0.75f32; 768]; 1000];

    let cpu_results = cpu_executor.execute_batch_cosine_similarity(&query, &targets);
    let gpu_results = hybrid_executor.execute_batch_cosine_similarity_oom_safe(&query, &targets);

    assert_eq!(cpu_results.len(), gpu_results.len());

    for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
        assert!(
            (cpu - gpu).abs() < 1e-6,
            "CPU/GPU divergence at index {}: cpu={}, gpu={}",
            i,
            cpu,
            gpu
        );
    }

    println!("CPU and GPU results match perfectly");
}

#[test]
#[cfg(cuda_available)]
fn test_memory_pressure_detection() {
    if let Some(monitor) = MemoryPressureMonitor::new() {
        let pressure = monitor.pressure_ratio();
        println!("Current memory pressure: {:.1}%", pressure * 100.0);

        assert!(pressure >= 0.0, "Pressure ratio should be non-negative");
        assert!(pressure <= 1.0, "Pressure ratio should not exceed 1.0");

        let under_pressure = monitor.is_under_pressure();
        println!("Under pressure: {}", under_pressure);

        // If pressure is very high, should be detected
        if pressure > 0.8 {
            assert!(under_pressure, "High pressure should be detected");
        }

        if let Some(available) = monitor.query_available_vram() {
            println!(
                "Available VRAM: {:.2} GB",
                available as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            assert!(available > 0, "Should have some available VRAM");
        }
    } else {
        println!("GPU not available, skipping memory pressure test");
    }
}

#[test]
#[cfg(cuda_available)]
fn test_allocation_tracking() {
    if let Some(monitor) = MemoryPressureMonitor::new() {
        let initial = monitor.current_allocated();

        // Track some allocations
        monitor.track_allocation(1024 * 1024); // 1MB
        assert_eq!(monitor.current_allocated(), initial + 1024 * 1024);

        monitor.track_allocation(512 * 1024); // 512KB
        assert_eq!(
            monitor.current_allocated(),
            initial + 1024 * 1024 + 512 * 1024
        );

        monitor.track_deallocation(1024 * 1024); // Free 1MB
        assert_eq!(monitor.current_allocated(), initial + 512 * 1024);

        monitor.track_deallocation(512 * 1024); // Free remaining
        assert_eq!(monitor.current_allocated(), initial);

        println!("Allocation tracking works correctly");
    } else {
        println!("GPU not available, skipping allocation tracking test");
    }
}

#[test]
#[cfg(cuda_available)]
fn test_global_monitor_singleton() {
    // Should return same instance on multiple calls
    let monitor1 = get_memory_pressure_monitor();
    let monitor2 = get_memory_pressure_monitor();

    match (monitor1, monitor2) {
        (Some(m1), Some(m2)) => {
            assert_eq!(m1.total_vram(), m2.total_vram(), "Should be same instance");
            println!("Global monitor singleton works correctly");
        }
        (None, None) => {
            println!("GPU not available, both monitors None (correct)");
        }
        _ => panic!("Inconsistent global monitor state"),
    }
}

#[test]
#[cfg(cuda_available)]
fn test_oom_event_telemetry() {
    use engram_core::compute::cuda::performance_tracker::PerformanceTracker;

    let tracker = PerformanceTracker::new(100);

    // Simulate some OOM events
    tracker.record_oom_event(Operation::CosineSimilarity);
    tracker.record_oom_event(Operation::CosineSimilarity);
    tracker.record_oom_event(Operation::ActivationSpreading);

    assert_eq!(tracker.oom_count(Operation::CosineSimilarity), 2);
    assert_eq!(tracker.oom_count(Operation::ActivationSpreading), 1);
    assert_eq!(tracker.oom_count(Operation::HnswSearch), 0);

    let telemetry = tracker.telemetry();
    println!("Telemetry output:\n{}", telemetry);

    assert!(
        telemetry.contains("OOM="),
        "Telemetry should include OOM counts"
    );
}

#[test]
#[cfg(cuda_available)]
fn test_extreme_batch_sizes() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping extreme batch test");
        return;
    }

    let executor = HybridExecutor::new(HybridConfig::default());

    // Test with single item (edge case)
    let query = [1.0f32; 768];
    let single = vec![[1.0f32; 768]; 1];
    let results = executor.execute_batch_cosine_similarity_oom_safe(&query, &single);
    assert_eq!(results.len(), 1);
    assert!((results[0] - 1.0).abs() < 1e-6);

    // Test with very large batch (stress test)
    let large = vec![[1.0f32; 768]; 50_000];
    let results = executor.execute_batch_cosine_similarity_oom_safe(&query, &large);
    assert_eq!(results.len(), 50_000);

    // All results should be correct
    for (i, sim) in results.iter().enumerate() {
        assert!(
            (*sim - 1.0).abs() < 1e-6,
            "Failed at index {}: expected 1.0, got {}",
            i,
            sim
        );
    }

    println!("Extreme batch sizes handled correctly");
}

#[test]
#[cfg(cuda_available)]
fn test_safety_margin_configuration() {
    // Test with different safety margins
    let conservative = MemoryPressureMonitor::with_safety_margin(0.5);
    let aggressive = MemoryPressureMonitor::with_safety_margin(0.9);

    match (conservative, aggressive) {
        (Some(c), Some(a)) => {
            // Conservative should have lower limit
            assert!(c.vram_limit() < a.vram_limit());

            // Conservative should recommend smaller batches
            let per_vector = 768 * 4;
            let requested = 10_000;

            let conservative_batch = c.calculate_safe_batch_size(requested, per_vector);
            let aggressive_batch = a.calculate_safe_batch_size(requested, per_vector);

            if conservative_batch < requested {
                // If conservative needs splitting, aggressive should allow more
                assert!(aggressive_batch >= conservative_batch);
            }

            println!("Safety margin configuration works correctly");
        }
        _ => {
            println!("GPU not available, skipping safety margin test");
        }
    }
}

#[test]
#[cfg(cuda_available)]
fn test_concurrent_pressure_queries() {
    use std::sync::Arc;
    use std::thread;

    if let Some(monitor) = get_memory_pressure_monitor() {
        let monitor = Arc::new(monitor);
        let mut handles = vec![];

        // Spawn multiple threads querying pressure concurrently
        for i in 0..10 {
            let mon = Arc::clone(&monitor);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let pressure = mon.pressure_ratio();
                    assert!(pressure >= 0.0 && pressure <= 1.0);

                    let safe_batch = mon.calculate_safe_batch_size(1000, 3072);
                    assert!(safe_batch > 0);
                }
                println!("Thread {} completed", i);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        println!("Concurrent pressure queries handled correctly");
    } else {
        println!("GPU not available, skipping concurrent test");
    }
}
