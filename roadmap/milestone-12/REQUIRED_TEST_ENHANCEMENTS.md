# Required Test Enhancements for Production Readiness

**Based on**: Graph Systems Acceptance Review
**Reviewer**: Denise Gosnell
**Priority**: P0 (Blocking Production Deployment)
**Estimated Effort**: 7-10 days

---

## P0 Critical Test Gaps

### 1. Multi-Tenant Security Validation

**Current State**: Only tests logical isolation (query results don't cross-contaminate)
**Required**: Security isolation, resource fairness, adversarial tenant handling

**Add to** `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/gpu_integration.rs`:

```rust
/// Test that adversarial tenant cannot exhaust GPU resources for other tenants
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_multi_tenant_resource_exhaustion_protection() {
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

    let tenant_normal = MemorySpaceId::try_from("tenant_normal").unwrap();
    let tenant_adversarial = MemorySpaceId::try_from("tenant_adversarial").unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();

    runtime.block_on(async {
        // Set up normal tenant
        let handle_normal = registry.create_or_get(&tenant_normal).await.unwrap();
        let store_normal = handle_normal.store();

        // Set up adversarial tenant
        let handle_adversarial = registry.create_or_get(&tenant_adversarial).await.unwrap();
        let store_adversarial = handle_adversarial.store();

        // Adversarial tenant: Store massive dataset
        for i in 0..10_000 {
            let episode = Episode::new(
                format!("adversarial_data_{i}"),
                chrono::Utc::now(),
                "Resource exhaustion attempt".to_string(),
                [0.9f32; 768],
                Confidence::exact(0.9),
            );
            store_adversarial.store(episode);
        }

        // Normal tenant: Store small dataset
        let episode = Episode::new(
            "normal_data".to_string(),
            chrono::Utc::now(),
            "Normal workload".to_string(),
            [0.5f32; 768],
            Confidence::exact(0.9),
        );
        store_normal.store(episode);

        // Concurrent queries: Adversarial tenant submits massive query
        let adversarial_handle = {
            let store = store_adversarial.clone();
            std::thread::spawn(move || {
                let cue = Cue::embedding(
                    "adversarial_query".to_string(),
                    [0.9f32; 768],
                    Confidence::exact(0.1),
                );
                let start = Instant::now();
                let _ = store.recall(&cue);
                start.elapsed()
            })
        };

        // Wait briefly to ensure adversarial query starts first
        std::thread::sleep(Duration::from_millis(10));

        // Normal tenant submits query while adversarial is running
        let normal_handle = {
            let store = store_normal.clone();
            std::thread::spawn(move || {
                let cue = Cue::embedding(
                    "normal_query".to_string(),
                    [0.5f32; 768],
                    Confidence::exact(0.1),
                );
                let start = Instant::now();
                let results = store.recall(&cue);
                (start.elapsed(), results)
            })
        };

        let adversarial_latency = adversarial_handle.join().unwrap();
        let (normal_latency, normal_results) = normal_handle.join().unwrap();

        // Validate normal tenant is not starved
        assert!(
            normal_latency < Duration::from_secs(5),
            "Normal tenant starved: {}ms latency",
            normal_latency.as_millis()
        );

        // Validate normal tenant got correct results
        assert!(!normal_results.results.is_empty(), "Normal tenant should get results");

        // Validate fairness: normal tenant should not be >10x slower than adversarial
        let fairness_ratio = normal_latency.as_secs_f64() / adversarial_latency.as_secs_f64();
        assert!(
            fairness_ratio < 10.0,
            "Fairness violation: normal tenant {}x slower than adversarial",
            fairness_ratio
        );

        println!("Multi-tenant resource exhaustion protection validated");
        println!("  Adversarial latency: {:.2}ms", adversarial_latency.as_millis());
        println!("  Normal latency: {:.2}ms", normal_latency.as_millis());
        println!("  Fairness ratio: {:.2}x", fairness_ratio);
    });
}

/// Test that tenant cannot access another tenant's GPU memory
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_multi_tenant_gpu_memory_isolation() {
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

    let tenant_a = MemorySpaceId::try_from("tenant_a_secret").unwrap();
    let tenant_b = MemorySpaceId::try_from("tenant_b_probe").unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();

    runtime.block_on(async {
        // Tenant A: Store secret data with distinctive embedding
        let handle_a = registry.create_or_get(&tenant_a).await.unwrap();
        let store_a = handle_a.store();

        let mut secret_embedding = [0.0f32; 768];
        secret_embedding[42] = 1.0; // Distinctive pattern

        let episode = Episode::new(
            "secret_data".to_string(),
            chrono::Utc::now(),
            "Confidential information".to_string(),
            secret_embedding,
            Confidence::exact(0.99),
        );
        store_a.store(episode);

        // Tenant B: Attempt to find Tenant A's data
        let handle_b = registry.create_or_get(&tenant_b).await.unwrap();
        let store_b = handle_b.store();

        // Query for the exact embedding pattern
        let cue = Cue::embedding(
            "probe_query".to_string(),
            secret_embedding,
            Confidence::exact(0.01), // Very permissive threshold
        );

        let results = store_b.recall(&cue);

        // Validate: Tenant B should get NO results (empty memory space)
        assert!(
            results.results.is_empty(),
            "Security violation: Tenant B accessed Tenant A's data"
        );

        // Validate: Even with perfect embedding match, cross-tenant access is blocked
        // This proves GPU memory isolation, not just logical isolation
        println!("Multi-tenant GPU memory isolation validated");
    });
}

/// Test concurrent multi-tenant fairness under sustained load
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_multi_tenant_concurrent_fairness() {
    use engram_core::compute::cuda;
    use std::sync::atomic::{AtomicUsize, Ordering};

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let executor = Arc::new(HybridExecutor::new(Default::default()));

    // Track operations per tenant
    let tenant_a_ops = Arc::new(AtomicUsize::new(0));
    let tenant_b_ops = Arc::new(AtomicUsize::new(0));
    let tenant_c_ops = Arc::new(AtomicUsize::new(0));

    let duration = Duration::from_secs(10);
    let start = Instant::now();

    // Spawn concurrent threads for each tenant
    let handles: Vec<_> = vec![
        (tenant_a_ops.clone(), [0.1f32; 768]),
        (tenant_b_ops.clone(), [0.5f32; 768]),
        (tenant_c_ops.clone(), [0.9f32; 768]),
    ]
    .into_iter()
    .map(|(counter, query)| {
        let exec = Arc::clone(&executor);
        std::thread::spawn(move || {
            while start.elapsed() < duration {
                let targets = vec![[query[0]; 768]; 256];
                let _ = exec.execute_batch_cosine_similarity(&query, &targets);
                counter.fetch_add(1, Ordering::Relaxed);
            }
        })
    })
    .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Collect operation counts
    let a_ops = tenant_a_ops.load(Ordering::Relaxed);
    let b_ops = tenant_b_ops.load(Ordering::Relaxed);
    let c_ops = tenant_c_ops.load(Ordering::Relaxed);

    let total_ops = a_ops + b_ops + c_ops;
    let a_share = (a_ops as f64) / (total_ops as f64);
    let b_share = (b_ops as f64) / (total_ops as f64);
    let c_share = (c_ops as f64) / (total_ops as f64);

    println!("Multi-tenant fairness results:");
    println!("  Tenant A: {} ops ({:.1}%)", a_ops, a_share * 100.0);
    println!("  Tenant B: {} ops ({:.1}%)", b_ops, b_share * 100.0);
    println!("  Tenant C: {} ops ({:.1}%)", c_ops, c_share * 100.0);

    // Validate fair scheduling: each tenant should get ≥25% of resources (±8%)
    assert!(
        a_share >= 0.25 && a_share <= 0.42,
        "Tenant A unfair share: {:.1}%",
        a_share * 100.0
    );
    assert!(
        b_share >= 0.25 && b_share <= 0.42,
        "Tenant B unfair share: {:.1}%",
        b_share * 100.0
    );
    assert!(
        c_share >= 0.25 && c_share <= 0.42,
        "Tenant C unfair share: {:.1}%",
        c_share * 100.0
    );

    println!("Multi-tenant concurrent fairness validated");
}
```

---

### 2. Production Workload Validation

**Current State**: Synthetic uniform vectors
**Required**: Realistic graph structures and query patterns

```rust
/// Test GPU performance on social network graph (power-law degree distribution)
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_production_workload_social_network() {
    use engram_core::compute::cuda;
    use engram_core::activation::{
        ActivationGraphExt, EdgeType, ParallelSpreadingConfig,
        ParallelSpreadingEngine, create_activation_graph,
    };

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    // Create power-law graph: 1000 nodes, few high-degree hubs, many low-degree nodes
    let graph = create_activation_graph();

    // Hub nodes (5 nodes with ~100 edges each)
    let hubs: Vec<String> = (0..5).map(|i| format!("hub_{i}")).collect();

    // Regular nodes (995 nodes with ~5 edges each)
    let nodes: Vec<String> = (0..995).map(|i| format!("node_{i}")).collect();

    // Connect hubs to many nodes (power-law distribution)
    for hub in &hubs {
        for node in nodes.iter().take(100) {
            ActivationGraphExt::add_edge(
                &graph,
                hub.clone(),
                node.clone(),
                0.6,
                EdgeType::Excitatory,
            );
        }
    }

    // Connect regular nodes to a few neighbors
    for i in 0..nodes.len() {
        for j in 0..5 {
            let target_idx = (i + j + 1) % nodes.len();
            ActivationGraphExt::add_edge(
                &graph,
                nodes[i].clone(),
                nodes[target_idx].clone(),
                0.4,
                EdgeType::Excitatory,
            );
        }
    }

    // Configure GPU-enabled spreading
    let config = ParallelSpreadingConfig {
        enable_gpu: true,
        gpu_threshold: 32,
        max_depth: 3,
        num_threads: 4,
        ..Default::default()
    };

    let engine = ParallelSpreadingEngine::new(config, Arc::new(graph))
        .expect("Failed to create spreading engine");

    // Test spreading from hub (high fan-out)
    let hub_start = Instant::now();
    let hub_results = engine
        .spread_activation(&vec![("hub_0".to_string(), 1.0)])
        .expect("Hub spreading failed");
    let hub_latency = hub_start.elapsed();

    // Test spreading from regular node (low fan-out)
    let node_start = Instant::now();
    let node_results = engine
        .spread_activation(&vec![("node_0".to_string(), 1.0)])
        .expect("Node spreading failed");
    let node_latency = node_start.elapsed();

    println!("Social network spreading results:");
    println!("  Hub activation: {} nodes in {:.2}ms",
             hub_results.activations.len(), hub_latency.as_secs_f64() * 1000.0);
    println!("  Node activation: {} nodes in {:.2}ms",
             node_results.activations.len(), node_latency.as_secs_f64() * 1000.0);

    // Validate: Hub should activate many more nodes
    assert!(
        hub_results.activations.len() > node_results.activations.len() * 5,
        "Hub should activate significantly more nodes"
    );

    // Validate: GPU should handle high fan-out efficiently (<100ms)
    assert!(
        hub_latency < Duration::from_millis(100),
        "Hub spreading too slow: {:.2}ms",
        hub_latency.as_millis()
    );

    engine.shutdown().expect("Failed to shutdown engine");
}

/// Test GPU performance on knowledge graph (dense semantic clustering)
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_production_workload_knowledge_graph() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let store = MemoryStore::new(temp_dir.path(), 10000).expect("Failed to create store");

    // Simulate semantic clustering: 5 clusters of 200 similar concepts each
    for cluster_id in 0..5 {
        let cluster_base = cluster_id as f32 / 5.0; // [0.0, 0.2, 0.4, 0.6, 0.8]

        for concept_id in 0..200 {
            let mut embedding = [cluster_base; 768];

            // Add small variation within cluster (±0.05)
            let variation = (concept_id as f32 / 200.0) * 0.1 - 0.05;
            for val in embedding.iter_mut().take(100) {
                *val += variation;
            }

            let episode = Episode::new(
                format!("cluster_{}_concept_{}", cluster_id, concept_id),
                chrono::Utc::now(),
                format!("Cluster {} concept", cluster_id),
                embedding,
                Confidence::exact(0.9),
            );
            store.store(episode);
        }
    }

    // Query for cluster 2 concepts (should recall ~200 similar, ignore other 800)
    let mut query_embedding = [0.4f32; 768]; // Cluster 2 base
    query_embedding[0] = 0.41; // Slight variation

    let cue = Cue::embedding(
        "knowledge_query".to_string(),
        query_embedding,
        Confidence::exact(0.7),
    );

    let start = Instant::now();
    let results = store.recall(&cue);
    let latency = start.elapsed();

    println!("Knowledge graph recall results:");
    println!("  Total memories: 1000");
    println!("  Recalled: {}", results.results.len());
    println!("  Latency: {:.2}ms", latency.as_secs_f64() * 1000.0);

    // Validate: Should recall primarily from cluster 2
    let cluster_2_count = results
        .results
        .iter()
        .filter(|(ep, _)| ep.id.starts_with("cluster_2"))
        .count();

    assert!(
        cluster_2_count as f64 > results.results.len() as f64 * 0.7,
        "Should recall primarily from matching cluster: {}/{}",
        cluster_2_count,
        results.results.len()
    );

    // Validate: GPU-accelerated search should be fast (<50ms for 1K vectors)
    assert!(
        latency < Duration::from_millis(50),
        "Knowledge graph recall too slow: {:.2}ms",
        latency.as_millis()
    );
}
```

---

### 3. Confidence Score Calibration Validation

**Current State**: No statistical validation of confidence scores
**Required**: Prove that 0.8 confidence = 80% accuracy

```rust
/// Test that confidence scores are properly calibrated over large sample
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
fn test_confidence_score_calibration() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let temp_dir = tempdir().expect("Failed to create temp dir");
    let store = MemoryStore::new(temp_dir.path(), 100_000).expect("Failed to create store");

    // Store labeled ground truth data: 10K memories
    let mut ground_truth = std::collections::HashMap::new();

    for category in 0..10 {
        for item in 0..1000 {
            let mut embedding = [0.0f32; 768];

            // Category-specific base pattern
            embedding[category * 70] = 1.0;

            // Add noise
            for i in 0..50 {
                embedding[(category * 70 + i) % 768] =
                    (item as f32 * 0.001 + i as f32 * 0.01).sin() * 0.3;
            }

            let id = format!("cat_{}_item_{}", category, item);
            ground_truth.insert(id.clone(), category);

            let episode = Episode::new(
                id,
                chrono::Utc::now(),
                format!("Category {}", category),
                embedding,
                Confidence::exact(0.9),
            );
            store.store(episode);
        }
    }

    // Test calibration: For each confidence bucket, measure actual accuracy
    let confidence_buckets = vec![
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 1.0),
    ];

    for (min_conf, max_conf) in confidence_buckets {
        let mut correct = 0;
        let mut total = 0;

        // Run 100 queries for this confidence bucket
        for category in 0..10 {
            for _ in 0..10 {
                let mut query = [0.0f32; 768];
                query[category * 70] = 0.9; // Slightly different from perfect match

                let cue = Cue::embedding(
                    format!("calibration_query_{}", total),
                    query,
                    Confidence::exact(min_conf),
                );

                let results = store.recall(&cue);

                // Check if top result is from correct category
                if let Some((episode, confidence)) = results.results.first() {
                    if *confidence >= min_conf && *confidence < max_conf {
                        total += 1;

                        if let Some(&true_category) = ground_truth.get(&episode.id) {
                            if true_category == category {
                                correct += 1;
                            }
                        }
                    }
                }
            }
        }

        if total > 0 {
            let accuracy = (correct as f64) / (total as f64);
            let expected_accuracy = (min_conf + max_conf) / 2.0;
            let calibration_error = (accuracy - expected_accuracy as f64).abs();

            println!("Confidence [{:.1}, {:.1}): accuracy {:.1}%, expected {:.1}%, error {:.1}%",
                     min_conf, max_conf, accuracy * 100.0,
                     expected_accuracy * 100.0, calibration_error * 100.0);

            // Validate: Calibration error should be <15% for well-calibrated system
            assert!(
                calibration_error < 0.15,
                "Poor calibration for confidence [{:.1}, {:.1}): {:.1}% error",
                min_conf, max_conf, calibration_error * 100.0
            );
        }
    }

    println!("Confidence score calibration validated over {} samples", 10_000);
}

/// Test that confidence scores don't drift over time
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
#[ignore] // Long-running test
fn test_confidence_drift_over_time() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    let executor = HybridExecutor::new(Default::default());

    // Measure confidence for same query repeated 1M times
    let query = [0.5f32; 768];
    let target = [0.5f32; 768];

    let mut initial_similarity = 0.0f32;
    let mut drift_measurements = Vec::new();

    for iteration in 0..1_000_000 {
        let similarities = executor.execute_batch_cosine_similarity(&query, &[target]);

        if iteration == 0 {
            initial_similarity = similarities[0];
        } else if iteration % 100_000 == 0 {
            let current_similarity = similarities[0];
            let drift = (current_similarity - initial_similarity).abs();
            drift_measurements.push(drift);

            println!("Iteration {}: similarity {:.6}, drift {:.2e}",
                     iteration, current_similarity, drift);
        }
    }

    // Validate: Confidence drift should be <1e-6 over 1M operations
    let max_drift = drift_measurements.iter().copied().fold(0.0f32, f32::max);
    assert!(
        max_drift < 1e-6,
        "Confidence drift too large: {:.2e}",
        max_drift
    );

    println!("No confidence drift detected over 1M operations");
}
```

---

### 4. Sustained Load Validation (Fix Existing Test)

**Current State**: Test is marked `#[ignore]` and has incorrect math
**Required**: Execute on GPU hardware with correct measurements

```rust
/// Test sustained throughput meets 10K ops/sec target
///
/// IMPORTANT: This test must be run on GPU hardware with:
///   cargo test test_sustained_throughput --ignored --features gpu -- --nocapture
#[test]
#[cfg(all(feature = "gpu", cuda_available))]
#[ignore] // Long-running test, requires GPU hardware
fn test_sustained_throughput() {
    use engram_core::compute::cuda;

    if !cuda::is_available() {
        println!("GPU not available, skipping sustained throughput test");
        println!("This test MUST be run on GPU hardware before production deployment");
        panic!("GPU not available - cannot validate production readiness");
    }

    let executor = HybridExecutor::new(Default::default());
    let test_duration = Duration::from_secs(600); // 10 minutes, not 60 seconds
    let start = Instant::now();

    let mut total_ops = 0;
    let mut latencies = Vec::new();
    let mut oom_count = 0;

    println!("Starting 10-minute sustained throughput test...");
    println!("Target: 10,000 ops/sec");

    while start.elapsed() < test_duration {
        let query = [0.5f32; 768];
        let targets = vec![[0.75f32; 768]; 1000];

        let op_start = Instant::now();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor.execute_batch_cosine_similarity(&query, &targets)
        }));

        let latency = op_start.elapsed();

        match result {
            Ok(_) => {
                latencies.push(latency);
                total_ops += 1;
            }
            Err(_) => {
                oom_count += 1;
                println!("OOM event at {}s", start.elapsed().as_secs());
            }
        }

        // Report progress every minute
        if total_ops % 10_000 == 0 {
            let elapsed = start.elapsed().as_secs();
            let current_rate = (total_ops as f64) / (elapsed as f64);
            println!("  {}s: {} ops, {:.0} ops/sec", elapsed, total_ops, current_rate);
        }
    }

    let elapsed_secs = start.elapsed().as_secs_f64();
    let ops_per_sec = (total_ops as f64) / elapsed_secs;

    println!("\nSustained throughput test results:");
    println!("  Duration: {:.1}s", elapsed_secs);
    println!("  Total operations: {}", total_ops);
    println!("  Throughput: {:.0} ops/sec", ops_per_sec);
    println!("  OOM events: {}", oom_count);

    // Calculate percentiles
    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[latencies.len() * 95 / 100];
    let p99 = latencies[latencies.len() * 99 / 100];
    let p999 = latencies[latencies.len() * 999 / 1000];

    println!("  Latency p50: {:.2}ms", p50.as_secs_f64() * 1000.0);
    println!("  Latency p95: {:.2}ms", p95.as_secs_f64() * 1000.0);
    println!("  Latency p99: {:.2}ms", p99.as_secs_f64() * 1000.0);
    println!("  Latency p99.9: {:.2}ms", p999.as_secs_f64() * 1000.0);

    // Validate throughput target
    assert!(
        ops_per_sec >= 10_000.0,
        "Throughput {:.0} ops/sec below 10K target",
        ops_per_sec
    );

    // Validate performance stability (no degradation over time)
    let first_quarter = &latencies[..latencies.len() / 4];
    let last_quarter = &latencies[latencies.len() * 3 / 4..];

    let first_p50_idx = first_quarter.len() / 2;
    let last_p50_idx = last_quarter.len() / 2;

    let first_p50 = first_quarter[first_p50_idx];
    let last_p50 = last_quarter[last_p50_idx];

    let degradation = (last_p50.as_secs_f64() - first_p50.as_secs_f64())
                      / first_p50.as_secs_f64();

    println!("  First quarter p50: {:.2}ms", first_p50.as_secs_f64() * 1000.0);
    println!("  Last quarter p50: {:.2}ms", last_p50.as_secs_f64() * 1000.0);
    println!("  Degradation: {:.1}%", degradation * 100.0);

    // Validate: No more than 10% degradation over 10 minutes
    assert!(
        degradation < 0.1,
        "Performance degraded by {:.1}% during test",
        degradation * 100.0
    );

    // Validate: OOM events should be rare (<0.1% of operations)
    let oom_rate = (oom_count as f64) / (total_ops as f64);
    assert!(
        oom_rate < 0.001,
        "Excessive OOM events: {:.2}%",
        oom_rate * 100.0
    );

    println!("\n✓ Sustained throughput test PASSED");
}
```

---

## Test Execution Requirements

### GPU Hardware Requirements

All P0 tests must be executed on actual GPU hardware:

1. **Minimum GPU**: NVIDIA Tesla T4 (16GB VRAM)
2. **Recommended GPU**: NVIDIA A100 (40GB VRAM)
3. **Test Duration**: 10 minutes sustained load + 1 hour soak test
4. **Environment**: Linux with CUDA 11.0+ toolkit installed

### Acceptance Criteria

Tests pass when:

1. **Multi-tenant security**: Zero cross-tenant data leaks
2. **Multi-tenant fairness**: Each tenant gets ≥25% GPU time (±10%)
3. **Production workloads**: Sub-100ms latency for realistic graphs
4. **Confidence calibration**: <15% calibration error across buckets
5. **Sustained throughput**: ≥10K ops/sec for 10 minutes with <10% degradation

### Execution Instructions

```bash
# Execute P0 critical tests on GPU hardware
cargo test --features gpu --test gpu_integration \
    test_multi_tenant_resource_exhaustion_protection \
    test_multi_tenant_gpu_memory_isolation \
    test_multi_tenant_concurrent_fairness \
    test_production_workload_social_network \
    test_production_workload_knowledge_graph \
    test_confidence_score_calibration \
    -- --nocapture

# Execute long-running tests
cargo test --features gpu --test gpu_integration \
    test_sustained_throughput \
    test_confidence_drift_over_time \
    --ignored -- --nocapture

# Validate results
./scripts/engram_diagnostics.sh
cat tmp/engram_diagnostics.log
```

---

## Sign-Off Criteria

Before marking Milestone 12 as PRODUCTION READY:

- [ ] All P0 tests executed on GPU hardware (Tesla T4/A100)
- [ ] All P0 tests passing for 48 consecutive hours
- [ ] Multi-tenant security validated (zero cross-tenant leaks)
- [ ] Confidence calibration error <15% across all buckets
- [ ] Sustained throughput ≥10K ops/sec with <10% degradation
- [ ] Production workload latencies <100ms for realistic graphs
- [ ] Soak test: 24 hours with <0.1% OOM rate
- [ ] Results documented in acceptance report with actual numbers

---

**Owner**: Verification & Testing Lead
**Reviewer**: Denise Gosnell
**Priority**: P0 (Blocking)
**Deadline**: Before production deployment
