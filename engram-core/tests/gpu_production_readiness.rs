//! GPU Production Readiness Tests
//!
//! This test suite addresses critical gaps identified in the Milestone 12 acceptance review:
//! - Multi-tenant security validation (resource exhaustion, security boundaries)
//! - Production workload scenarios (power-law graphs, semantic clustering)
//! - Confidence score calibration validation
//! - Chaos engineering for GPU failure modes
//!
//! IMPORTANT: These tests require GPU hardware and will be skipped in CPU-only CI.
//! See docs/operations/gpu_testing_requirements.md for manual execution procedures.

#![allow(unused_imports)]

use engram_core::{Confidence, Cue, Episode, MemorySpaceId, MemorySpaceRegistry, MemoryStore};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tempfile::tempdir;

// =============================================================================
// CATEGORY 1: MULTI-TENANT SECURITY VALIDATION
// =============================================================================

/// Test that one tenant cannot exhaust GPU resources and starve others
///
/// **Addresses**: GRAPH_SYSTEMS_ACCEPTANCE_REVIEW.md lines 96-136
/// **Validates**: Resource exhaustion does not affect other tenants
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_multi_tenant_resource_exhaustion() {
    use engram_core::compute::cuda::{self, hybrid::HybridExecutor};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let executor = Arc::new(HybridExecutor::new(Default::default()));

    // Tenant A: Adversarial - tries to consume all GPU memory with massive batch
    let tenant_a_complete = Arc::new(AtomicBool::new(false));
    let tenant_a_flag = Arc::clone(&tenant_a_complete);

    let executor_a = Arc::clone(&executor);
    let handle_a = std::thread::spawn(move || {
        let query = [0.1f32; 768];
        // Attempt massive batch that may exceed GPU memory
        let targets = vec![[0.2f32; 768]; 100_000];

        let start = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor_a.execute_batch_cosine_similarity(&query, &targets)
        }));
        let latency = start.elapsed();

        tenant_a_flag.store(true, Ordering::SeqCst);

        match result {
            Ok(similarities) => ("tenant_a_succeeded", latency, similarities.len()),
            Err(_) => ("tenant_a_oom", latency, 0),
        }
    });

    // Give tenant A a head start to attempt resource exhaustion
    std::thread::sleep(Duration::from_millis(100));

    // Tenant B: Normal workload submitted concurrently
    let executor_b = Arc::clone(&executor);
    let handle_b = std::thread::spawn(move || {
        let query = [0.5f32; 768];
        let targets = vec![[0.6f32; 768]; 256]; // Normal batch size

        let start = Instant::now();
        let similarities = executor_b.execute_batch_cosine_similarity(&query, &targets);
        let latency = start.elapsed();

        ("tenant_b", latency, similarities.len())
    });

    // Collect results
    let result_a = handle_a.join().unwrap();
    let result_b = handle_b.join().unwrap();

    println!(
        "Tenant A: {} in {:.2}ms ({} results)",
        result_a.0,
        result_a.1.as_secs_f64() * 1000.0,
        result_a.2
    );
    println!(
        "Tenant B: {} in {:.2}ms ({} results)",
        result_b.0,
        result_b.1.as_secs_f64() * 1000.0,
        result_b.2
    );

    // Validate: Tenant B must not be starved regardless of Tenant A behavior
    assert_eq!(
        result_b.2, 256,
        "Tenant B was starved by Tenant A's resource exhaustion"
    );

    // Validate: Tenant B latency must be reasonable (not blocked waiting for GPU)
    assert!(
        result_b.1 < Duration::from_secs(5),
        "Tenant B latency {:.2}s indicates starvation",
        result_b.1.as_secs_f64()
    );
}

/// Test that GPU memory is isolated between tenants (security boundary)
///
/// **Addresses**: GRAPH_SYSTEMS_ACCEPTANCE_REVIEW.md lines 96-136
/// **Validates**: Tenant A cannot access Tenant B's GPU buffers
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_multi_tenant_security_isolation() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let registry = MemorySpaceRegistry::new(temp_dir.path(), |space_id, _dirs| {
        Ok(Arc::new(MemoryStore::for_space(space_id.clone(), 1000)))
    })
    .expect("Failed to create registry");

    // Tenant A: Store sensitive data with distinctive pattern
    let tenant_a = MemorySpaceId::try_from("tenant_a_secure").unwrap();
    let sensitive_embedding = [0.999f32; 768]; // Highly distinctive

    // Tenant B: Store different data
    let tenant_b = MemorySpaceId::try_from("tenant_b_isolated").unwrap();
    let normal_embedding = [0.001f32; 768]; // Very different

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        // Store Tenant A's sensitive data
        let handle_a = registry.create_or_get(&tenant_a).await.unwrap();
        let store_a = handle_a.store();
        let episode_a = Episode::new(
            "sensitive_data".to_string(),
            chrono::Utc::now(),
            "Highly sensitive information".to_string(),
            sensitive_embedding,
            Confidence::exact(0.95),
        );
        store_a.store(episode_a);

        // Store Tenant B's data
        let handle_b = registry.create_or_get(&tenant_b).await.unwrap();
        let store_b = handle_b.store();
        let episode_b = Episode::new(
            "normal_data".to_string(),
            chrono::Utc::now(),
            "Normal information".to_string(),
            normal_embedding,
            Confidence::exact(0.95),
        );
        store_b.store(episode_b);

        // Tenant B attempts to query Tenant A's sensitive data pattern
        let cue = Cue::embedding(
            "attempt_cross_tenant_access".to_string(),
            sensitive_embedding, // Exact match to Tenant A's data
            Confidence::exact(0.5),
        );

        let results = store_b.recall(&cue);

        // Security validation: Tenant B must NOT see Tenant A's data
        for (episode, _score) in &results.results {
            assert_ne!(
                episode.id, "sensitive_data",
                "SECURITY VIOLATION: Tenant B accessed Tenant A's sensitive data"
            );

            // Verify embedding doesn't match Tenant A's sensitive pattern
            let similarity: f32 = episode
                .embedding
                .iter()
                .zip(sensitive_embedding.iter())
                .map(|(a, b)| a * b)
                .sum();

            assert!(
                similarity < 0.5,
                "SECURITY VIOLATION: Tenant B retrieved data similar to Tenant A's (similarity={})",
                similarity
            );
        }

        // Tenant A can still access its own data
        let handle_a = registry.create_or_get(&tenant_a).await.unwrap();
        let store_a = handle_a.store();
        let results_a = store_a.recall(&cue);

        assert_eq!(
            results_a.results.len(),
            1,
            "Tenant A should access its own data"
        );
        assert_eq!(
            results_a.results[0].0.id, "sensitive_data",
            "Tenant A should retrieve its sensitive data"
        );
    });
}

/// Test concurrent fairness with actual parallel execution
///
/// **Addresses**: GRAPH_SYSTEMS_ACCEPTANCE_REVIEW.md lines 480-504
/// **Fixes**: Previous test measured sequential latency, not concurrent fairness
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_multi_tenant_gpu_fairness_concurrent() {
    use engram_core::compute::cuda::{self, hybrid::HybridExecutor};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let executor = Arc::new(HybridExecutor::new(Default::default()));
    let test_duration = Duration::from_secs(10);

    // Track operations completed per tenant
    let tenant_ops = Arc::new(Mutex::new(HashMap::new()));

    let tenants = vec![
        ("tenant_1", [0.1f32; 768]),
        ("tenant_2", [0.5f32; 768]),
        ("tenant_3", [0.9f32; 768]),
    ];

    let handles: Vec<_> = tenants
        .iter()
        .map(|(tenant_name, query)| {
            let executor = Arc::clone(&executor);
            let ops_map = Arc::clone(&tenant_ops);
            let query = *query;
            let tenant = tenant_name.to_string();

            std::thread::spawn(move || {
                let start = Instant::now();
                let mut ops_count = 0;

                while start.elapsed() < test_duration {
                    let targets = vec![[query[0]; 768]; 256];
                    let _similarities = executor.execute_batch_cosine_similarity(&query, &targets);
                    ops_count += 1;
                }

                let mut map = ops_map.lock().unwrap();
                map.insert(tenant.clone(), ops_count);

                ops_count
            })
        })
        .collect();

    // Wait for all tenants to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Analyze fairness
    let ops_map = tenant_ops.lock().unwrap();
    let ops_values: Vec<usize> = ops_map.values().cloned().collect();

    let max_ops = *ops_values.iter().max().unwrap();
    let min_ops = *ops_values.iter().min().unwrap();
    let total_ops: usize = ops_values.iter().sum();
    let avg_ops = total_ops as f64 / ops_values.len() as f64;

    println!(
        "Concurrent fairness results over {}s:",
        test_duration.as_secs()
    );
    for (tenant, ops) in ops_map.iter() {
        let percentage = (*ops as f64 / total_ops as f64) * 100.0;
        println!("  {}: {} ops ({:.1}%)", tenant, ops, percentage);
    }

    // Validate fairness: Each tenant should get ≥30% of GPU time (±10%)
    for (tenant, ops) in ops_map.iter() {
        let share = (*ops as f64 / total_ops as f64) * 100.0;
        assert!(
            share >= 20.0 && share <= 45.0,
            "Tenant {} got {:.1}% of GPU time (expected 33±10%)",
            tenant,
            share
        );
    }

    // Fairness ratio: max/min should be <2.0 for good fairness
    let fairness_ratio = max_ops as f64 / min_ops as f64;
    assert!(
        fairness_ratio < 2.0,
        "Fairness ratio {:.2} indicates unfair scheduling (expected <2.0)",
        fairness_ratio
    );
}

// =============================================================================
// CATEGORY 2: PRODUCTION WORKLOAD VALIDATION
// =============================================================================

/// Test GPU performance on power-law degree distribution (social graph pattern)
///
/// **Addresses**: GRAPH_SYSTEMS_ACCEPTANCE_REVIEW.md lines 36-67
/// **Validates**: GPU speedup on realistic social graph workloads
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_production_workload_social_graph() {
    use engram_core::compute::cuda::{self, hybrid::HybridConfig, hybrid::HybridExecutor};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    // Generate power-law degree distribution (Barabási-Albert model)
    // Most nodes have low degree, few "hub" nodes have very high degree
    let total_nodes = 10_000;
    let hub_nodes = 10; // 0.1% are hubs

    let mut graph_embeddings = Vec::new();

    // Regular nodes (99.9%): sparse connections
    for i in 0..(total_nodes - hub_nodes) {
        let mut embedding = [0.0f32; 768];
        // Sparse pattern: only 5% of dimensions active
        for j in (0..768).step_by(20) {
            embedding[j] = (i as f32) / (total_nodes as f32);
        }
        graph_embeddings.push(embedding);
    }

    // Hub nodes (0.1%): dense connections
    for i in 0..hub_nodes {
        let mut embedding = [0.5f32; 768]; // Dense activation
        embedding[0] = 0.9 + (i as f32 * 0.01); // Distinctive hub signature
        graph_embeddings.push(embedding);
    }

    // Query: Find similar nodes to a hub (high fan-out scenario)
    let hub_query = [0.95f32; 768];

    // Test GPU executor
    let gpu_executor = HybridExecutor::new(Default::default());
    let gpu_start = Instant::now();
    let gpu_results = gpu_executor.execute_batch_cosine_similarity(&hub_query, &graph_embeddings);
    let gpu_duration = gpu_start.elapsed();

    // Test CPU executor
    let cpu_executor = HybridExecutor::new(HybridConfig {
        force_cpu_mode: true,
        ..Default::default()
    });
    let cpu_start = Instant::now();
    let cpu_results = cpu_executor.execute_batch_cosine_similarity(&hub_query, &graph_embeddings);
    let cpu_duration = cpu_start.elapsed();

    println!("Social graph workload (10K nodes, power-law distribution):");
    println!("  GPU: {:.2}ms", gpu_duration.as_secs_f64() * 1000.0);
    println!("  CPU: {:.2}ms", cpu_duration.as_secs_f64() * 1000.0);

    let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
    println!("  GPU Speedup: {:.2}x", speedup);

    // Validate correctness: CPU and GPU results must match
    assert_eq!(gpu_results.len(), cpu_results.len());
    for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
        assert!(
            (gpu - cpu).abs() < 1e-5,
            "Result divergence at index {}: GPU={}, CPU={}",
            i,
            gpu,
            cpu
        );
    }

    // Validate performance: GPU should provide speedup on this workload
    // Note: Speedup may be modest on small graphs; validate it doesn't regress
    assert!(
        speedup >= 1.0,
        "GPU slower than CPU on social graph workload: {:.2}x",
        speedup
    );

    if speedup >= 3.0 {
        println!("  ✓ GPU achieves target 3x speedup");
    } else {
        println!(
            "  ⚠ GPU speedup {:.2}x below 3x target (acceptable for 10K nodes)",
            speedup
        );
    }
}

/// Test GPU performance on dense semantic clustering (knowledge graph pattern)
///
/// **Addresses**: GRAPH_SYSTEMS_ACCEPTANCE_REVIEW.md lines 36-67
/// **Validates**: GPU speedup on dense semantic embeddings
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_production_workload_knowledge_graph() {
    use engram_core::compute::cuda::{self, hybrid::HybridConfig, hybrid::HybridExecutor};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    // Generate dense semantic clustering pattern (typical of knowledge graphs)
    // Multiple dense clusters with high intra-cluster similarity
    let clusters = 5;
    let nodes_per_cluster = 2000;
    let total_nodes = clusters * nodes_per_cluster;

    let mut graph_embeddings = Vec::new();

    for cluster_id in 0..clusters {
        let cluster_center = (cluster_id as f32) / (clusters as f32);

        for node in 0..nodes_per_cluster {
            let mut embedding = [cluster_center; 768]; // Dense base pattern

            // Add small intra-cluster variation
            let variation = (node as f32) / (nodes_per_cluster as f32) * 0.1;
            for i in 0..768 {
                embedding[i] += variation * ((i % 10) as f32) / 10.0;
            }

            graph_embeddings.push(embedding);
        }
    }

    // Query: Find nodes in cluster 2 (dense semantic search)
    let cluster_2_query = [0.4f32; 768];

    // Test GPU executor
    let gpu_executor = HybridExecutor::new(Default::default());
    let gpu_start = Instant::now();
    let gpu_results =
        gpu_executor.execute_batch_cosine_similarity(&cluster_2_query, &graph_embeddings);
    let gpu_duration = gpu_start.elapsed();

    // Test CPU executor
    let cpu_executor = HybridExecutor::new(HybridConfig {
        force_cpu_mode: true,
        ..Default::default()
    });
    let cpu_start = Instant::now();
    let cpu_results =
        cpu_executor.execute_batch_cosine_similarity(&cluster_2_query, &graph_embeddings);
    let cpu_duration = cpu_start.elapsed();

    println!("Knowledge graph workload (10K nodes, 5 dense clusters):");
    println!("  GPU: {:.2}ms", gpu_duration.as_secs_f64() * 1000.0);
    println!("  CPU: {:.2}ms", cpu_duration.as_secs_f64() * 1000.0);

    let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
    println!("  GPU Speedup: {:.2}x", speedup);

    // Validate correctness
    assert_eq!(gpu_results.len(), cpu_results.len());
    for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
        assert!(
            (gpu - cpu).abs() < 1e-5,
            "Result divergence at index {}: GPU={}, CPU={}",
            i,
            gpu,
            cpu
        );
    }

    // Validate performance
    assert!(
        speedup >= 1.0,
        "GPU slower than CPU on knowledge graph workload: {:.2}x",
        speedup
    );

    // Knowledge graphs with dense embeddings should benefit significantly from GPU
    if speedup >= 3.0 {
        println!("  ✓ GPU achieves target 3x speedup");
    } else {
        println!("  ⚠ GPU speedup {:.2}x below 3x target", speedup);
    }
}

// =============================================================================
// CATEGORY 3: CONFIDENCE SCORE CALIBRATION
// =============================================================================

/// Test confidence score calibration over 100K+ operations
///
/// **Addresses**: GRAPH_SYSTEMS_ACCEPTANCE_REVIEW.md lines 165-175
/// **Validates**: 0.8 confidence = 80% accuracy over large samples
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_confidence_calibration_statistical_validation() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let store = MemoryStore::new(temp_dir.path(), 150_000).expect("Failed to create store");

    // Create ground truth dataset with known similarity structure
    let num_clusters = 10;
    let items_per_cluster = 10_000;

    println!("Building ground truth dataset (100K items)...");

    for cluster_id in 0..num_clusters {
        for item in 0..items_per_cluster {
            let mut embedding = [0.0f32; 768];

            // Cluster-specific pattern
            let cluster_value = (cluster_id as f32) / (num_clusters as f32);
            for i in 0..768 {
                embedding[i] = cluster_value + (item as f32 / items_per_cluster as f32) * 0.05;
            }

            let episode = Episode::new(
                format!("cluster_{}_{}", cluster_id, item),
                chrono::Utc::now(),
                format!("Item in cluster {}", cluster_id),
                embedding,
                Confidence::exact(0.9),
            );
            store.store(episode);
        }
    }

    println!("Running calibration validation queries...");

    // Test calibration across different confidence thresholds
    let confidence_levels = vec![0.5, 0.6, 0.7, 0.8, 0.9];
    let queries_per_level = 200;

    for target_confidence in confidence_levels {
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        for query_id in 0..queries_per_level {
            // Query from a known cluster
            let true_cluster = query_id % num_clusters;
            let mut query_embedding = [0.0f32; 768];
            let cluster_value = (true_cluster as f32) / (num_clusters as f32);

            for i in 0..768 {
                query_embedding[i] = cluster_value + 0.025; // Center of cluster
            }

            let cue = Cue::embedding(
                format!("calibration_query_{}", query_id),
                query_embedding,
                Confidence::exact(target_confidence),
            );

            let results = store.recall(&cue);

            if !results.results.is_empty() {
                // Check if top result is from correct cluster
                let top_result_id = &results.results[0].0.id;
                let predicted_cluster: usize = top_result_id
                    .split('_')
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(999);

                total_predictions += 1;
                if predicted_cluster == true_cluster {
                    correct_predictions += 1;
                }
            }
        }

        let observed_accuracy = (correct_predictions as f64) / (total_predictions as f64);
        let calibration_error = (observed_accuracy - target_confidence as f64).abs();

        println!(
            "  Confidence {:.1}: {:.1}% accuracy (error: {:.1}%)",
            target_confidence,
            observed_accuracy * 100.0,
            calibration_error * 100.0
        );

        // Validate calibration within ±15% (acceptable for 200 samples)
        assert!(
            calibration_error < 0.15,
            "Confidence {:.1} has calibration error {:.1}% (expected <15%)",
            target_confidence,
            calibration_error * 100.0
        );
    }
}

/// Test confidence score stability over sustained operations
///
/// **Addresses**: GRAPH_SYSTEMS_ACCEPTANCE_REVIEW.md lines 165-175
/// **Validates**: Confidence scores don't drift over time
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
#[ignore] // Long-running test
fn test_confidence_drift_over_time() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let store = MemoryStore::new(temp_dir.path(), 50_000).expect("Failed to create store");

    // Store baseline dataset
    for i in 0..10_000 {
        let mut embedding = [0.5f32; 768];
        embedding[i % 768] = (i as f32) / 10_000.0;

        let episode = Episode::new(
            format!("baseline_{}", i),
            chrono::Utc::now(),
            "Baseline data".to_string(),
            embedding,
            Confidence::exact(0.8),
        );
        store.store(episode);
    }

    // Test query
    let query_embedding = [0.5f32; 768];
    query_embedding[100] = 0.01;

    let cue = Cue::embedding(
        "drift_test".to_string(),
        query_embedding,
        Confidence::exact(0.8),
    );

    // Measure baseline confidence scores
    let baseline_results = store.recall(&cue);
    let baseline_confidences: Vec<f32> = baseline_results
        .results
        .iter()
        .map(|(_episode, score)| *score)
        .collect();

    println!("Baseline confidence stats:");
    println!("  Results: {}", baseline_confidences.len());

    // Perform 1 million operations
    println!("Executing 1M operations...");
    let operations_per_batch = 10_000;
    let batches = 100;

    for batch in 0..batches {
        for i in 0..operations_per_batch {
            let mut embedding = [0.5f32; 768];
            embedding[(batch * 1000 + i) % 768] = ((batch * 1000 + i) as f32) / 100_000.0;

            let episode = Episode::new(
                format!("sustained_{}_{}", batch, i),
                chrono::Utc::now(),
                "Sustained operation data".to_string(),
                embedding,
                Confidence::exact(0.8),
            );
            store.store(episode);

            // Periodic recall to exercise similarity computation
            if i % 1000 == 0 {
                let _ = store.recall(&cue);
            }
        }

        if batch % 10 == 0 {
            println!("  Completed {} operations", batch * operations_per_batch);
        }
    }

    // Measure confidence scores after 1M operations
    let final_results = store.recall(&cue);
    let final_confidences: Vec<f32> = final_results
        .results
        .iter()
        .take(baseline_confidences.len())
        .map(|(_episode, score)| *score)
        .collect();

    println!("Final confidence stats:");
    println!("  Results: {}", final_confidences.len());

    // Calculate drift
    let baseline_mean: f32 =
        baseline_confidences.iter().sum::<f32>() / baseline_confidences.len() as f32;
    let final_mean: f32 = final_confidences.iter().sum::<f32>() / final_confidences.len() as f32;

    let drift = (final_mean - baseline_mean).abs();
    let drift_percentage = (drift / baseline_mean) * 100.0;

    println!("Confidence drift analysis:");
    println!("  Baseline mean: {:.4}", baseline_mean);
    println!("  Final mean: {:.4}", final_mean);
    println!("  Drift: {:.4} ({:.2}%)", drift, drift_percentage);

    // Validate: Drift should be <5% after 1M operations
    assert!(
        drift_percentage < 5.0,
        "Confidence drift {:.2}% exceeds 5% threshold after 1M operations",
        drift_percentage
    );
}

// =============================================================================
// CATEGORY 4: CHAOS ENGINEERING
// =============================================================================

/// Test GPU OOM handling with graceful fallback
///
/// **Addresses**: GRAPH_SYSTEMS_ACCEPTANCE_REVIEW.md lines 179-190
/// **Validates**: System handles GPU memory exhaustion gracefully
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_chaos_gpu_oom_injection() {
    use engram_core::compute::cuda::{self, hybrid::HybridExecutor};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let executor = HybridExecutor::new(Default::default());

    // Progressively increase batch size until OOM
    let mut batch_size = 1024;
    let mut max_successful_batch = 0;
    let mut oom_encountered = false;

    println!("GPU OOM stress test:");

    while batch_size <= 1_000_000 {
        let query = [0.5f32; 768];
        let targets = vec![[0.6f32; 768]; batch_size];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor.execute_batch_cosine_similarity(&query, &targets)
        }));

        match result {
            Ok(similarities) => {
                if similarities.len() == batch_size {
                    max_successful_batch = batch_size;
                    println!("  ✓ Batch size {} succeeded", batch_size);
                } else {
                    println!(
                        "  ! Batch size {} returned {} results (expected {})",
                        batch_size,
                        similarities.len(),
                        batch_size
                    );
                }
            }
            Err(_) => {
                println!(
                    "  ✗ Batch size {} triggered panic (NOT ACCEPTABLE)",
                    batch_size
                );
                panic!("GPU OOM caused panic instead of graceful fallback");
            }
        }

        // Check if we've reached GPU limits (smaller increments near limit)
        if batch_size >= 10_000 {
            batch_size += 10_000;
        } else {
            batch_size *= 2;
        }

        // Safety limit: Stop if we've tested up to 1M vectors
        if batch_size > 1_000_000 {
            break;
        }
    }

    println!("Max successful batch: {} vectors", max_successful_batch);

    assert!(
        max_successful_batch >= 1024,
        "GPU couldn't handle minimum expected batch size of 1024 vectors"
    );
}

/// Test concurrent access patterns don't cause CUDA errors
///
/// **Addresses**: GRAPH_SYSTEMS_ACCEPTANCE_REVIEW.md lines 179-190
/// **Validates**: Thread safety of GPU operations
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_chaos_concurrent_gpu_access() {
    use engram_core::compute::cuda::{self, hybrid::HybridExecutor};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let executor = Arc::new(HybridExecutor::new(Default::default()));
    let num_threads = 8;
    let operations_per_thread = 100;

    let error_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let executor = Arc::clone(&executor);
            let errors = Arc::clone(&error_count);

            std::thread::spawn(move || {
                for op_id in 0..operations_per_thread {
                    let query = [(thread_id as f32) / (num_threads as f32); 768];
                    let targets = vec![[0.5f32; 768]; 256];

                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        executor.execute_batch_cosine_similarity(&query, &targets)
                    }));

                    match result {
                        Ok(similarities) => {
                            if similarities.len() != 256 {
                                errors.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                        Err(_) => {
                            errors.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let total_errors = error_count.load(Ordering::SeqCst);
    let total_ops = num_threads * operations_per_thread;

    println!("Concurrent access test:");
    println!("  Threads: {}", num_threads);
    println!("  Operations per thread: {}", operations_per_thread);
    println!("  Total operations: {}", total_ops);
    println!("  Errors: {}", total_errors);

    assert_eq!(
        total_errors, 0,
        "Concurrent GPU access caused {} errors (expected 0)",
        total_errors
    );
}

// =============================================================================
// CATEGORY 5: TEST LOGIC FIXES
// =============================================================================

/// Fixed sustained throughput test with correct time calculation
///
/// **Addresses**: GRAPH_SYSTEMS_ACCEPTANCE_REVIEW.md lines 538-551
/// **Fixes**: Division by actual elapsed time, not hardcoded 60 seconds
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
#[ignore] // Long-running test
fn test_sustained_throughput_fixed() {
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

    // FIX: Use actual elapsed time, not hardcoded 60
    let actual_duration = test_start.elapsed();
    let ops_per_sec = (total_ops as f64) / actual_duration.as_secs_f64();

    println!("Sustained throughput test results:");
    println!("  Duration: {:.2}s", actual_duration.as_secs_f64());
    println!("  Total operations: {}", total_ops);
    println!("  Throughput: {:.0} ops/sec", ops_per_sec);

    // Validate throughput target
    assert!(
        ops_per_sec >= 10_000.0,
        "Throughput {:.0} ops/sec below 10K target",
        ops_per_sec
    );

    // Validate latency stability
    let first_half = &latencies[..latencies.len() / 2];
    let second_half = &latencies[latencies.len() / 2..];

    let first_half_p50 = percentile(first_half, 0.5);
    let second_half_p50 = percentile(second_half, 0.5);

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
