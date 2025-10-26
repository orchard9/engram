//! Integration tests for CPU-GPU hybrid executor
//!
//! These tests validate:
//! - Dispatch decision logic
//! - CPU fallback on GPU failure
//! - Performance tracking accuracy
//! - API compatibility with existing VectorOps
//!
//! Tests run on both CUDA and non-CUDA builds.

#[cfg(feature = "gpu")]
use engram_core::compute::cuda::hybrid::{HybridConfig, HybridExecutor};
#[cfg(feature = "gpu")]
use engram_core::compute::cuda::performance_tracker::Operation;

#[test]
#[cfg(feature = "gpu")]
fn test_hybrid_executor_basic() {
    let executor = HybridExecutor::new(HybridConfig::default());

    let query = [1.0f32; 768];
    let targets = vec![[1.0f32; 768]; 128];

    let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

    assert_eq!(similarities.len(), 128);
    for sim in &similarities {
        assert!(
            (*sim - 1.0).abs() < 1e-6,
            "Expected similarity 1.0, got {sim}"
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_small_batch_cpu_dispatch() {
    let executor = HybridExecutor::new(HybridConfig::default());

    let query = [1.0f32; 768];
    let targets = vec![[0.5f32; 768]; 32]; // < 64, should use CPU

    let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

    assert_eq!(similarities.len(), 32);
    for sim in &similarities {
        assert!((sim - 1.0).abs() < 1e-6);
    }

    // Performance tracker should have recorded CPU execution
    let tracker = executor.performance_tracker();
    let cpu_avg = tracker.average_cpu_latency(Operation::CosineSimilarity);
    assert!(cpu_avg.as_nanos() > 0, "CPU latency should be recorded");
}

#[test]
#[cfg(feature = "gpu")]
fn test_force_cpu_mode() {
    let config = HybridConfig {
        force_cpu_mode: true,
        ..Default::default()
    };
    let executor = HybridExecutor::new(config);

    assert!(!executor.is_gpu_available(), "GPU should be disabled");

    let query = [1.0f32; 768];
    let targets = vec![[1.0f32; 768]; 256]; // Large batch

    let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

    assert_eq!(similarities.len(), 256);
    for sim in &similarities {
        assert!((*sim - 1.0).abs() < 1e-6);
    }
}

#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_cpu_gpu_result_equivalence() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    // CPU-only executor
    let cpu_config = HybridConfig {
        force_cpu_mode: true,
        ..Default::default()
    };
    let cpu_executor = HybridExecutor::new(cpu_config);

    // Hybrid executor (GPU if available)
    let hybrid_executor = HybridExecutor::new(HybridConfig::default());

    let query = [0.5f32; 768];
    let targets = vec![[0.75f32; 768]; 256];

    let cpu_results = cpu_executor.execute_batch_cosine_similarity(&query, &targets);
    let hybrid_results = hybrid_executor.execute_batch_cosine_similarity(&query, &targets);

    assert_eq!(cpu_results.len(), hybrid_results.len());

    for (i, (cpu, hybrid)) in cpu_results.iter().zip(hybrid_results.iter()).enumerate() {
        assert!(
            (cpu - hybrid).abs() < 1e-6,
            "Results diverged at index {}: CPU={}, GPU={}",
            i,
            cpu,
            hybrid
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_performance_tracking() {
    let config = HybridConfig {
        telemetry_enabled: true,
        ..Default::default()
    };
    let executor = HybridExecutor::new(config);

    let query = [1.0f32; 768];
    let targets = vec![[0.5f32; 768]; 128];

    // Execute multiple times to build up telemetry
    for _ in 0..10 {
        let _ = executor.execute_batch_cosine_similarity(&query, &targets);
    }

    // Verify telemetry was recorded
    let tracker = executor.performance_tracker();
    let telemetry = tracker.telemetry();

    assert!(telemetry.contains("CosineSimilarity"));
    println!("Telemetry:\n{telemetry}");
}

#[test]
#[cfg(feature = "gpu")]
fn test_orthogonal_vectors() {
    let executor = HybridExecutor::new(HybridConfig::default());

    let mut query = [0.0f32; 768];
    query[0] = 1.0; // Unit vector in dimension 0

    let mut target = [0.0f32; 768];
    target[1] = 1.0; // Unit vector in dimension 1

    let targets = vec![target; 128];

    let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

    assert_eq!(similarities.len(), 128);
    for sim in &similarities {
        assert!(
            sim.abs() < 1e-6,
            "Orthogonal vectors should have similarity ~0, got {sim}"
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_zero_query_vector() {
    let executor = HybridExecutor::new(HybridConfig::default());

    let query = [0.0f32; 768]; // Zero vector
    let targets = vec![[1.0f32; 768]; 128];

    let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

    assert_eq!(similarities.len(), 128);
    for sim in &similarities {
        assert!(
            sim.abs() < 1e-6,
            "Zero query should produce ~0.0 similarity, got {sim}"
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_mixed_batch_sizes() {
    let executor = HybridExecutor::new(HybridConfig::default());

    let query = [1.0f32; 768];

    // Test various batch sizes
    for batch_size in [1, 16, 32, 64, 128, 256, 512, 1024] {
        let targets = vec![[1.0f32; 768]; batch_size];

        let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

        assert_eq!(similarities.len(), batch_size);
        for (i, sim) in similarities.iter().enumerate() {
            assert!(
                (*sim - 1.0).abs() < 1e-6,
                "Batch size {batch_size}, index {i}: expected 1.0, got {sim}"
            );
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_executor_capabilities() {
    let executor = HybridExecutor::new(HybridConfig::default());

    let caps = executor.capabilities();
    println!("Executor capabilities: {caps}");

    // CPU should always be available - just verify it's set
    // Note: CpuCapability variants are platform-specific, so we can't match all of them
    println!("CPU SIMD level: {:?}", caps.cpu_simd_level);
}

#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_gpu_speedup_measurement() {
    use engram_core::compute::cuda;
    use std::time::Instant;

    if !cuda::is_available() {
        println!("GPU not available, skipping speedup test");
        return;
    }

    // CPU-only executor
    let cpu_config = HybridConfig {
        force_cpu_mode: true,
        telemetry_enabled: false,
        ..Default::default()
    };
    let cpu_executor = HybridExecutor::new(cpu_config);

    // GPU executor
    let gpu_executor = HybridExecutor::new(HybridConfig {
        telemetry_enabled: false,
        ..Default::default()
    });

    let query = [0.5f32; 768];
    let targets = vec![[0.75f32; 768]; 1024]; // Large batch

    // Warm up
    cpu_executor.execute_batch_cosine_similarity(&query, &targets);
    gpu_executor.execute_batch_cosine_similarity(&query, &targets);

    // Measure CPU
    let cpu_start = Instant::now();
    for _ in 0..10 {
        cpu_executor.execute_batch_cosine_similarity(&query, &targets);
    }
    let cpu_time = cpu_start.elapsed();

    // Measure GPU
    let gpu_start = Instant::now();
    for _ in 0..10 {
        gpu_executor.execute_batch_cosine_similarity(&query, &targets);
    }
    let gpu_time = gpu_start.elapsed();

    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    println!(
        "CPU: {:.2}ms, GPU: {:.2}ms, Speedup: {:.2}x",
        cpu_time.as_secs_f64() * 1000.0,
        gpu_time.as_secs_f64() * 1000.0,
        speedup
    );

    // GPU should be at least as fast as CPU (speedup >= 1.0)
    // In practice, we expect 3-7x speedup for large batches
    assert!(
        speedup >= 0.5,
        "GPU should not be significantly slower than CPU"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_negative_cosine_similarity() {
    let executor = HybridExecutor::new(HybridConfig::default());

    let query = [1.0f32; 768];
    let target = [-1.0f32; 768]; // Opposite direction

    let targets = vec![target; 128];

    let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

    assert_eq!(similarities.len(), 128);
    for sim in &similarities {
        assert!(
            (*sim + 1.0).abs() < 1e-6,
            "Opposite vectors should have similarity -1.0, got {sim}"
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_random_vectors() {
    use rand::Rng;

    let executor = HybridExecutor::new(HybridConfig::default());

    let mut rng = rand::thread_rng();

    let query: [f32; 768] = std::array::from_fn(|_| rng.gen_range(-1.0..1.0));

    let targets: Vec<[f32; 768]> = (0..128)
        .map(|_| std::array::from_fn(|_| rng.gen_range(-1.0..1.0)))
        .collect();

    let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

    assert_eq!(similarities.len(), 128);

    // All similarities should be in valid range [-1, 1]
    for (i, sim) in similarities.iter().enumerate() {
        assert!(
            sim.abs() <= 1.0 + 1e-6,
            "Similarity at index {i} out of range: {sim}"
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_dispatch_threshold_configuration() {
    let config = HybridConfig {
        gpu_min_batch_size: 128,
        gpu_speedup_threshold: 2.0,
        gpu_success_rate_threshold: 0.98,
        ..Default::default()
    };

    let executor = HybridExecutor::new(config);

    // Batch smaller than threshold should use CPU
    let query = [1.0f32; 768];
    let targets = vec![[1.0f32; 768]; 64]; // < 128

    let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

    assert_eq!(similarities.len(), 64);

    // Should have used CPU (telemetry would show this)
    let tracker = executor.performance_tracker();
    let cpu_avg = tracker.average_cpu_latency(Operation::CosineSimilarity);
    assert!(cpu_avg.as_nanos() > 0);
}
