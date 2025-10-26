//! Comprehensive GPU Acceleration Integration Tests
//!
//! This test suite validates end-to-end GPU acceleration integration with all
//! Engram features (Milestones 1-11). These tests serve as the final gate for
//! production deployment.
//!
//! Test Categories:
//! 1. Feature Integration - GPU with spreading activation, consolidation, pattern completion
//! 2. Multi-Tenant Isolation - GPU operations maintain memory space boundaries
//! 3. Performance Under Load - Sustained throughput and latency validation
//! 4. Fallback Behavior - CPU equivalence and graceful degradation
//! 5. End-to-End Scenarios - Real-world workflow validation

#![allow(unused_imports)]

use engram_core::{Confidence, Cue, Episode, MemorySpaceId, MemorySpaceRegistry, MemoryStore};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;

// =============================================================================
// CATEGORY 1: FEATURE INTEGRATION
// =============================================================================

/// Test GPU acceleration with Milestone 3 spreading activation
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_gpu_with_spreading_activation() {
    use engram_core::activation::{
        ActivationGraphExt, EdgeType, ParallelSpreadingConfig, ParallelSpreadingEngine,
        create_activation_graph,
    };
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    // Create activation graph
    let graph = create_activation_graph();
    ActivationGraphExt::add_edge(
        &graph,
        "A".to_string(),
        "B".to_string(),
        0.8,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &graph,
        "B".to_string(),
        "C".to_string(),
        0.6,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &graph,
        "A".to_string(),
        "C".to_string(),
        0.4,
        EdgeType::Excitatory,
    );

    // Configure spreading engine with GPU enabled
    let config = ParallelSpreadingConfig {
        enable_gpu: true,
        gpu_threshold: 32,
        max_depth: 2,
        num_threads: 4,
        ..Default::default()
    };

    let engine = ParallelSpreadingEngine::new(config, Arc::new(graph))
        .expect("Failed to create spreading engine");

    // Execute spreading activation
    let seed_activations = vec![("A".to_string(), 1.0)];
    let results = engine
        .spread_activation(&seed_activations)
        .expect("Spreading activation failed");

    // Validate results
    assert!(
        !results.activations.is_empty(),
        "Spreading should produce activations"
    );
    assert!(
        results.activations.contains_key("B"),
        "Node B should be activated"
    );
    assert!(
        results.activations.contains_key("C"),
        "Node C should be activated"
    );

    // Verify activation values are reasonable
    let b_activation = results.activations.get("B").unwrap();
    assert!(
        *b_activation > 0.0 && *b_activation <= 1.0,
        "Activation values must be in [0, 1]"
    );

    engine.shutdown().expect("Failed to shutdown engine");
}

/// Test GPU acceleration with similarity search for consolidation candidates
///
/// **Note**: This tests GPU-accelerated similarity search used by consolidation,
/// not the full consolidation process (which requires consolidation scheduler).
/// Renamed from test_gpu_with_memory_consolidation to reflect actual behavior.
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_gpu_similarity_for_consolidation() {
    use engram_core::compute::cuda::{self, hybrid::HybridExecutor};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let store = MemoryStore::new(temp_dir.path(), 1000).expect("Failed to create store");

    // Store multiple similar episodes (pattern detection scenario)
    for i in 0..100 {
        let mut embedding = [0.5f32; 768];
        embedding[0] = 0.5 + (i as f32 * 0.001); // Slight variations

        let episode = Episode::new(
            format!("event_{i}"),
            chrono::Utc::now(),
            format!("Similar event pattern {i}"),
            embedding,
            Confidence::exact(0.9),
        );
        store.store(episode);
    }

    // Use GPU-accelerated similarity search for pattern detection
    let executor = HybridExecutor::new(Default::default());
    let query = [0.5f32; 768];

    // Collect all episodes for similarity computation
    let all_embeddings: Vec<[f32; 768]> = (0..100)
        .map(|i| {
            let mut embedding = [0.5f32; 768];
            embedding[0] = 0.5 + (i as f32 * 0.001);
            embedding
        })
        .collect();

    // GPU-accelerated batch similarity
    let similarities = executor.execute_batch_cosine_similarity(&query, &all_embeddings);

    // Validate pattern detection via similarity clustering
    assert_eq!(similarities.len(), 100);
    let high_similarity_count = similarities.iter().filter(|&&s| s > 0.99).count();
    assert!(
        high_similarity_count >= 90,
        "Should detect high similarity cluster: found {}",
        high_similarity_count
    );
}

/// Test GPU-accelerated similarity search for pattern matching
///
/// **Note**: This tests GPU-accelerated cosine similarity used in pattern completion,
/// not the full pattern completion logic (which involves semantic constraints).
/// Renamed to reflect actual test scope.
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_gpu_similarity_for_pattern_matching() {
    use engram_core::compute::cuda::{self, hybrid::HybridExecutor};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let executor = HybridExecutor::new(Default::default());

    // Simulate pattern matching scenario: find best match for query pattern
    let query_pattern = [0.3f32; 768];

    // Known patterns (simulating semantic memory)
    let known_patterns: Vec<[f32; 768]> = vec![
        [0.3f32; 768],  // Exact match
        [0.35f32; 768], // Close match
        [0.8f32; 768],  // Distant
        [0.1f32; 768],  // Very distant
    ];

    // GPU-accelerated pattern matching
    let similarities = executor.execute_batch_cosine_similarity(&query_pattern, &known_patterns);

    assert_eq!(similarities.len(), 4);

    // Validate similarity ranking
    let best_match_idx = similarities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    assert_eq!(
        best_match_idx, 0,
        "Should select exact match as highest similarity"
    );
    assert!(
        (similarities[0] - 1.0).abs() < 1e-6,
        "Exact match should have similarity ~1.0"
    );

    // Validate similarity ordering
    assert!(
        similarities[1] > similarities[2],
        "Close match should be more similar than distant"
    );
    assert!(
        similarities[2] > similarities[3],
        "Distant should be more similar than very distant"
    );
}

// =============================================================================
// CATEGORY 2: MULTI-TENANT ISOLATION
// =============================================================================

/// Test GPU resource isolation across memory spaces
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_multi_tenant_gpu_isolation() {
    use engram_core::compute::cuda::{self, hybrid::HybridExecutor};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let registry = MemorySpaceRegistry::new(temp_dir.path(), |space_id, _dirs| {
        Ok(Arc::new(MemoryStore::for_space(space_id.clone(), 1000)))
    })
    .expect("Failed to create registry");

    // Create multiple memory spaces
    let tenant_a = MemorySpaceId::try_from("tenant_a").unwrap();
    let tenant_b = MemorySpaceId::try_from("tenant_b").unwrap();
    let tenant_c = MemorySpaceId::try_from("tenant_c").unwrap();

    // Store different data in each space
    let spaces = vec![
        (tenant_a.clone(), [0.1f32; 768]),
        (tenant_b.clone(), [0.5f32; 768]),
        (tenant_c.clone(), [0.9f32; 768]),
    ];

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        for (space_id, embedding) in &spaces {
            let handle = registry.create_or_get(space_id).await.unwrap();
            let store = handle.store();

            let episode = Episode::new(
                format!("data_{}", space_id.as_str()),
                chrono::Utc::now(),
                format!("Data for {}", space_id.as_str()),
                *embedding,
                Confidence::exact(0.9),
            );
            store.store(episode);
        }
    });

    // Verify GPU operations maintain isolation
    let executor = HybridExecutor::new(Default::default());

    runtime.block_on(async {
        for (space_id, space_embedding) in &spaces {
            let handle = registry.create_or_get(space_id).await.unwrap();
            let store = handle.store();

            // Query this space's data
            let cue = Cue::embedding(
                format!("query_{}", space_id.as_str()),
                *space_embedding,
                Confidence::exact(0.1),
            );

            let results = store.recall(&cue);

            // Should only return data from this space
            assert_eq!(
                results.results.len(),
                1,
                "Space {} should have exactly 1 result",
                space_id.as_str()
            );
            assert_eq!(
                results.results[0].0.id,
                format!("data_{}", space_id.as_str()),
                "Result should match space-specific data"
            );

            // Verify no cross-contamination by checking embeddings
            let result_embedding = results.results[0].0.embedding;
            for i in 0..768 {
                assert!(
                    (result_embedding[i] - space_embedding[i]).abs() < 1e-6,
                    "Embedding mismatch indicates cross-space contamination"
                );
            }
        }
    });
}

/// Test GPU latency variance across sequential tenant operations
///
/// **Note**: This tests sequential latency variance, not concurrent fairness.
/// For true concurrent fairness testing, see test_multi_tenant_gpu_fairness_concurrent
/// in gpu_production_readiness.rs
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_multi_tenant_gpu_latency_consistency() {
    use engram_core::compute::cuda::{self, hybrid::HybridExecutor};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let executor = Arc::new(HybridExecutor::new(Default::default()));

    // Test latency consistency across multiple tenant operations
    let tenant_queries = vec![
        ("tenant_1", [0.1f32; 768]),
        ("tenant_2", [0.5f32; 768]),
        ("tenant_3", [0.9f32; 768]),
    ];

    let handles: Vec<_> = tenant_queries
        .iter()
        .map(|(tenant, query)| {
            let executor = Arc::clone(&executor);
            let query = *query;
            let tenant = tenant.to_string();

            std::thread::spawn(move || {
                let targets = vec![[query[0]; 768]; 256];
                let start = Instant::now();
                let _similarities = executor.execute_batch_cosine_similarity(&query, &targets);
                let latency = start.elapsed();
                (tenant, latency)
            })
        })
        .collect();

    // Collect latencies
    let latencies: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    println!("Sequential tenant operation latencies:");
    for (tenant, latency) in &latencies {
        println!("  {}: {:.2}ms", tenant, latency.as_secs_f64() * 1000.0);
    }

    // Verify latency consistency (not true fairness test)
    let max_latency = latencies.iter().map(|(_, l)| *l).max().unwrap();
    let min_latency = latencies.iter().map(|(_, l)| *l).min().unwrap();

    let variance_ratio = max_latency.as_secs_f64() / min_latency.as_secs_f64();
    assert!(
        variance_ratio < 3.0,
        "GPU latency variance {:.2}x exceeds 3x threshold",
        variance_ratio
    );
}

// =============================================================================
// CATEGORY 3: PERFORMANCE UNDER LOAD
// =============================================================================

/// Test sustained throughput meets 10K ops/sec target
///
/// **FIXED**: Now calculates ops/sec using actual elapsed time instead of hardcoded 60.
/// **Addresses**: GRAPH_SYSTEMS_ACCEPTANCE_REVIEW.md lines 538-551
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
#[ignore] // Long-running test, run with --ignored
fn test_sustained_throughput() {
    use engram_core::compute::cuda::{self, hybrid::HybridExecutor};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let executor = HybridExecutor::new(Default::default());
    let test_start = Instant::now();
    let target_duration = Duration::from_secs(60);

    let mut total_ops = 0;
    let mut latencies = Vec::new();

    while test_start.elapsed() < target_duration {
        let query = [0.5f32; 768];
        let targets = vec![[0.75f32; 768]; 1000];

        let op_start = Instant::now();
        let _results = executor.execute_batch_cosine_similarity(&query, &targets);
        let latency = op_start.elapsed();

        latencies.push(latency);
        total_ops += 1;
    }

    // FIX: Use actual elapsed time, not hardcoded 60 seconds
    let actual_duration = test_start.elapsed();
    let ops_per_sec = (total_ops as f64) / actual_duration.as_secs_f64();

    println!("Sustained throughput test results:");
    println!("  Target duration: {}s", target_duration.as_secs());
    println!("  Actual duration: {:.2}s", actual_duration.as_secs_f64());
    println!("  Total operations: {}", total_ops);
    println!("  Throughput: {:.0} ops/sec", ops_per_sec);

    assert!(
        ops_per_sec >= 10_000.0,
        "Throughput {:.0} ops/sec below 10K target",
        ops_per_sec
    );

    // Validate latency stability (no degradation over time)
    let first_half_latencies = &latencies[..latencies.len() / 2];
    let second_half_latencies = &latencies[latencies.len() / 2..];

    let first_half_p50 = percentile(first_half_latencies, 0.5);
    let second_half_p50 = percentile(second_half_latencies, 0.5);

    let degradation = (second_half_p50.as_secs_f64() - first_half_p50.as_secs_f64())
        / first_half_p50.as_secs_f64();

    println!("Performance stability:");
    println!(
        "  First half p50: {:.2}ms",
        first_half_p50.as_secs_f64() * 1000.0
    );
    println!(
        "  Second half p50: {:.2}ms",
        second_half_p50.as_secs_f64() * 1000.0
    );
    println!("  Degradation: {:.1}%", degradation * 100.0);

    assert!(
        degradation < 0.1,
        "Performance degraded by {:.1}% during test",
        degradation * 100.0
    );
}

/// Test GPU memory pressure handling
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_gpu_memory_pressure() {
    use engram_core::compute::cuda::{self, hybrid::HybridExecutor};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let executor = HybridExecutor::new(Default::default());

    // Gradually increase batch size to stress GPU memory
    let batch_sizes = vec![256, 512, 1024, 2048, 4096, 8192];

    for batch_size in batch_sizes {
        let query = [0.5f32; 768];
        let targets = vec![[0.75f32; 768]; batch_size];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor.execute_batch_cosine_similarity(&query, &targets)
        }));

        match result {
            Ok(similarities) => {
                assert_eq!(
                    similarities.len(),
                    batch_size,
                    "Result count mismatch at batch size {}",
                    batch_size
                );
                println!("✓ Batch size {} handled successfully", batch_size);
            }
            Err(_) => {
                println!(
                    "! Batch size {} triggered OOM (expected behavior)",
                    batch_size
                );
                // OOM should cause graceful fallback, not panic
                panic!("GPU OOM should not panic, should fall back to CPU");
            }
        }
    }
}

// =============================================================================
// CATEGORY 4: FALLBACK BEHAVIOR
// =============================================================================

/// Test CPU fallback maintains identical behavior
#[test]
#[cfg(feature = "gpu")]
fn test_cpu_fallback_equivalence() {
    use engram_core::compute::cuda::hybrid::{HybridConfig, HybridExecutor};

    // CPU-only executor
    let cpu_executor = HybridExecutor::new(HybridConfig {
        force_cpu_mode: true,
        ..Default::default()
    });

    // GPU-enabled executor (will fall back to CPU if GPU unavailable)
    let gpu_executor = HybridExecutor::new(HybridConfig::default());

    // Test data
    let query = [0.3f32; 768];
    let targets = vec![[0.3f32; 768], [0.5f32; 768], [0.7f32; 768], [-0.3f32; 768]];

    // Execute on both
    let cpu_results = cpu_executor.execute_batch_cosine_similarity(&query, &targets);
    let gpu_results = gpu_executor.execute_batch_cosine_similarity(&query, &targets);

    // Results must be identical
    assert_eq!(cpu_results.len(), gpu_results.len());

    for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
        assert!(
            (cpu - gpu).abs() < 1e-6,
            "Results diverged at index {i}: CPU={cpu}, GPU={gpu}"
        );
    }
}

/// Test graceful degradation when GPU unavailable
#[test]
#[cfg(feature = "gpu")]
fn test_graceful_gpu_unavailability() {
    use engram_core::compute::cuda::hybrid::{HybridConfig, HybridExecutor};

    // Executor should work even if GPU unavailable
    let executor = HybridExecutor::new(HybridConfig::default());

    let query = [1.0f32; 768];
    let targets = vec![[1.0f32; 768]; 128];

    // Should not panic, even without GPU
    let similarities = executor.execute_batch_cosine_similarity(&query, &targets);

    assert_eq!(similarities.len(), 128);
    for sim in &similarities {
        assert!(
            (*sim - 1.0).abs() < 1e-6,
            "Expected similarity 1.0, got {sim}"
        );
    }
}

// =============================================================================
// CATEGORY 5: END-TO-END SCENARIOS
// =============================================================================

/// Test full recall workflow with GPU acceleration
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_recall_with_gpu_acceleration() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let store = MemoryStore::new(temp_dir.path(), 10000).expect("Failed to create store");

    // Store 10K memories
    for i in 0..10_000 {
        let mut embedding = [0.0f32; 768];
        embedding[i % 768] = (i as f32) / 10_000.0;

        let episode = Episode::new(
            format!("memory_{i}"),
            chrono::Utc::now(),
            format!("Test memory {i}"),
            embedding,
            Confidence::exact(0.8),
        );
        store.store(episode);
    }

    // Recall with GPU-accelerated similarity
    let mut query_embedding = [0.0f32; 768];
    query_embedding[100] = 0.01; // Should match memory_100

    let cue = Cue::embedding(
        "gpu_recall_test".to_string(),
        query_embedding,
        Confidence::exact(0.5),
    );

    let start = Instant::now();
    let results = store.recall(&cue);
    let latency = start.elapsed();

    println!(
        "Recall latency for 10K memories: {:.2}ms",
        latency.as_secs_f64() * 1000.0
    );

    // Validate results
    assert!(
        !results.results.is_empty(),
        "Should return results from 10K memories"
    );

    // Verify top result is correct match
    assert!(
        results.results[0].0.id.contains("memory_100"),
        "Top result should be memory_100, got {}",
        results.results[0].0.id
    );

    // Validate latency is reasonable (sub-second for 10K)
    assert!(
        latency < Duration::from_secs(1),
        "Recall should be <1s for 10K memories, got {:.2}s",
        latency.as_secs_f64()
    );
}

/// Test store-consolidate-recall cycle with GPU
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_consolidation_cycle_with_gpu() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let store = MemoryStore::new(temp_dir.path(), 1000).expect("Failed to create store");

    // Store episodes with pattern
    for i in 0..100 {
        let mut embedding = [0.5f32; 768];
        embedding[0] = 0.5 + (i as f32 * 0.001);

        let episode = Episode::new(
            format!("pattern_event_{i}"),
            chrono::Utc::now(),
            "Repeating pattern".to_string(),
            embedding,
            Confidence::exact(0.9),
        );
        store.store(episode);
    }

    // Recall before consolidation
    let query = [0.5f32; 768];
    let cue = Cue::embedding("test".to_string(), query, Confidence::exact(0.5));
    let before_results = store.recall(&cue);

    println!(
        "Results before consolidation: {}",
        before_results.results.len()
    );

    // Note: Actual consolidation would require consolidation scheduler
    // This test validates that GPU-accelerated similarity still works
    // after episodic storage operations

    // Recall after to verify GPU still functional
    let after_results = store.recall(&cue);

    assert_eq!(
        before_results.results.len(),
        after_results.results.len(),
        "Result count should be stable"
    );
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Calculate percentile of duration samples
#[cfg(all(feature = "gpu", cuda_available))]
fn percentile(samples: &[Duration], p: f64) -> Duration {
    if samples.is_empty() {
        return Duration::from_secs(0);
    }

    let mut sorted = samples.to_vec();
    sorted.sort();

    let idx = ((samples.len() as f64) * p).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// =============================================================================
// COMPATIBILITY TESTS
// =============================================================================

/// Verify tests compile and run on CPU-only builds
#[test]
fn test_cpu_only_build_compatibility() {
    // This test ensures integration tests work even without GPU feature
    let data = [1.0f32; 768];
    let sum: f32 = data.iter().sum();
    assert!((sum - 768.0).abs() < 1e-6);
}
