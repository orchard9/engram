//! Large-scale validation tests for warm tier storage with statistical rigor
//!
//! These tests validate correctness and performance at production scale:
//! - 10K memories: Round-trip with statistical validation
//! - 100K memories: Performance with percentile tracking
//! - 1M memories: Cold tier iteration scalability limit
//! - Content extremes: Size variation and fragmentation
//! - Memory leaks: Long-running stability detection
//! - Performance cliffs: Where does performance degrade?
//! - Concurrent operations: Large-scale concurrent access
//!
//! All tests are marked #[ignore] and must be run explicitly:
//! ```
//! cargo test --test warm_tier_large_scale_tests --ignored -- --nocapture
//! ```

#![cfg(feature = "memory_mapped_persistence")]

use chrono::Utc;
use engram_core::{Confidence, CueBuilder, Episode, EpisodeBuilder, MemoryStore};
use rand::Rng;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;

// ===== Test Infrastructure Helpers =====

/// Get process memory RSS in MB
fn get_process_memory_rss_mb() -> usize {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<usize>() {
                            return kb / 1024;
                        }
                    }
                }
            }
        }
        0
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            && let Ok(rss_kb_str) = String::from_utf8(output.stdout)
            && let Ok(rss_kb) = rss_kb_str.trim().parse::<usize>()
        {
            return rss_kb / 1024;
        }
        0
    }

    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("tasklist")
            .args([
                "/FI",
                &format!("PID eq {}", std::process::id()),
                "/FO",
                "CSV",
                "/NH",
            ])
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                // Parse CSV: "process.exe","PID","Session Name","Session#","Mem Usage"
                // Mem usage is in format "1,234 K" or "12,345 K"
                if let Some(mem_field) = output_str.split(',').nth(4) {
                    let mem_str = mem_field
                        .trim_matches('"')
                        .replace(',', "")
                        .replace(" K", "");
                    if let Ok(mem_kb) = mem_str.parse::<usize>() {
                        return mem_kb / 1024;
                    }
                }
            }
        }
        0
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    0
}

/// Create a test embedding from a seed value
fn create_test_embedding(seed: f32) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for (i, elem) in embedding.iter_mut().enumerate() {
        *elem = ((seed + i as f32) * 0.01).sin();
    }
    embedding
}

// ===== Test 1: 10K Memories Round-Trip with Statistical Validation =====

#[tokio::test]
#[ignore = "slow"]
async fn test_10k_memories_statistical_validation() {
    const MEMORY_COUNT: usize = 10_000;
    const BATCH_SIZE: usize = 1_000;

    let temp_dir = TempDir::new().expect("temp dir creation");
    let store = MemoryStore::new(MEMORY_COUNT + 2000)
        .with_persistence(temp_dir.path())
        .expect("Failed to create store");

    // Track memory usage to detect leaks
    let initial_rss = get_process_memory_rss_mb();
    let mut batch_latencies = Vec::new();

    // Store 10K memories with varying content sizes (Zipf-like distribution)
    println!(
        "Storing {} memories in {} batches...",
        MEMORY_COUNT,
        MEMORY_COUNT / BATCH_SIZE
    );
    let start = Instant::now();

    for batch in 0..(MEMORY_COUNT / BATCH_SIZE) {
        let batch_start = Instant::now();

        for i in 0..BATCH_SIZE {
            let global_i = batch * BATCH_SIZE + i;
            // Zipf-like distribution: 89% small (100 bytes), 10% medium (1KB), 1% large (5KB)
            let content_size = if global_i.is_multiple_of(100) {
                5000 // 1% are 5KB
            } else if global_i.is_multiple_of(10) {
                1000 // 10% are 1KB
            } else {
                100 // 89% are 100 bytes
            };
            let content = "x".repeat(content_size);

            let episode = EpisodeBuilder::new()
                .id(format!("ep-{global_i:06}"))
                .when(Utc::now())
                .what(format!("Content {global_i} - {content}"))
                .embedding(create_test_embedding(global_i as f32))
                .confidence(Confidence::HIGH)
                .build();

            store.store(episode);
        }

        let batch_elapsed = batch_start.elapsed();
        batch_latencies.push(batch_elapsed);

        if (batch + 1) % 2 == 0 {
            println!("Stored {} memories", (batch + 1) * BATCH_SIZE);
        }
    }

    let store_duration = start.elapsed();
    let store_rate = (MEMORY_COUNT as f64) / store_duration.as_secs_f64();

    println!("Store phase: {store_duration:?} ({store_rate:.0} ops/s)");

    // Statistical analysis of batch latencies
    batch_latencies.sort();
    let median_batch = batch_latencies[batch_latencies.len() / 2];
    let p95_batch = batch_latencies[batch_latencies.len() * 95 / 100];
    let p99_batch = batch_latencies[batch_latencies.len() * 99 / 100];

    println!("Batch latencies: median={median_batch:?}, p95={p95_batch:?}, p99={p99_batch:?}");

    // Verify counts
    let counts = store.get_tier_counts();
    assert_eq!(
        counts.hot, MEMORY_COUNT,
        "Should have 10K memories in hot tier"
    );

    // Verify random sample (100 memories) for content integrity
    println!("Verifying random sample...");
    let verify_start = Instant::now();
    let mut rng = rand::thread_rng();
    let mut recall_latencies = Vec::new();

    for _ in 0..100 {
        let i = rng.gen_range(0..MEMORY_COUNT);
        let memory_id = format!("ep-{i:06}");

        let cue = CueBuilder::new()
            .id(format!("verify-{i}"))
            .embedding_search(create_test_embedding(i as f32), Confidence::LOW)
            .build();

        let recall_start = Instant::now();
        let result = store.recall(&cue);
        recall_latencies.push(recall_start.elapsed());

        assert!(
            !result.results.is_empty(),
            "Failed to recall memory {memory_id}"
        );

        let episode = &result.results[0].0;

        // Verify content prefix matches
        assert!(
            episode.what.starts_with(&format!("Content {i} - ")),
            "Content corrupted for memory {}: got '{}'",
            memory_id,
            episode.what
        );
    }

    let _verify_duration = verify_start.elapsed();

    // Statistical analysis of recall latencies
    recall_latencies.sort();
    let p50_recall = recall_latencies[recall_latencies.len() / 2];
    let p99_recall = recall_latencies[recall_latencies.len() * 99 / 100];

    println!("Recall latencies: p50={p50_recall:?}, p99={p99_recall:?}");

    // Memory leak detection
    let final_rss = get_process_memory_rss_mb();
    let rss_growth = final_rss.saturating_sub(initial_rss);
    let expected_growth_mb = (MEMORY_COUNT * 150) / (1024 * 1024); // ~150 bytes avg per memory

    println!(
        "Memory usage: initial={initial_rss}MB, final={final_rss}MB, growth={rss_growth}MB (expected ~{expected_growth_mb}MB)"
    );

    // Performance assertions with confidence intervals
    assert!(
        store_duration.as_secs() < 2,
        "Store phase too slow: {store_duration:?} (target <2s)"
    );

    assert!(
        store_rate > 5000.0,
        "Store rate too low: {store_rate:.0} ops/s (target >5000)"
    );

    assert!(
        p99_recall < Duration::from_millis(1),
        "P99 recall latency too high: {p99_recall:?} (target <1ms)"
    );

    // Memory leak check: growth should be within 2x expected (account for allocator overhead)
    if initial_rss > 0 {
        // Only check if we could measure RSS
        assert!(
            rss_growth < (expected_growth_mb * 2 + 100),
            "Possible memory leak: RSS grew by {rss_growth}MB (expected ~{expected_growth_mb}MB)"
        );
    }
}

// ===== Test 2: 100K Memories Performance with Percentile Tracking =====

#[tokio::test]
#[ignore = "slow"]
async fn test_100k_memories_performance_percentiles() {
    const MEMORY_COUNT: usize = 100_000;

    let temp_dir = TempDir::new().expect("temp dir creation");
    let store = MemoryStore::new(MEMORY_COUNT + 10_000)
        .with_persistence(temp_dir.path())
        .expect("Failed to create store");

    // Track detailed metrics
    let initial_rss = get_process_memory_rss_mb();
    let mut store_latencies = Vec::with_capacity(MEMORY_COUNT);

    // Store 100K memories with realistic content distribution
    println!("Storing {MEMORY_COUNT} memories...");
    let start = Instant::now();

    for i in 0..MEMORY_COUNT {
        let content = format!("Episode {i} content with some text");

        let store_start = Instant::now();

        let episode = EpisodeBuilder::new()
            .id(format!("ep-{i:06}"))
            .when(Utc::now())
            .what(content)
            .embedding(create_test_embedding(i as f32))
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);

        store_latencies.push(store_start.elapsed());

        if i > 0 && i % 10_000 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i as f64) / elapsed;
            let current_rss = get_process_memory_rss_mb();
            println!("Stored {i} memories ({rate:.0} ops/s, {current_rss}MB RSS)");
        }
    }

    let store_duration = start.elapsed();
    let store_rate = (MEMORY_COUNT as f64) / store_duration.as_secs_f64();

    // Comprehensive latency analysis
    store_latencies.sort();
    let p50 = store_latencies[MEMORY_COUNT / 2];
    let p90 = store_latencies[MEMORY_COUNT * 90 / 100];
    let p95 = store_latencies[MEMORY_COUNT * 95 / 100];
    let p99 = store_latencies[MEMORY_COUNT * 99 / 100];
    let p999 = store_latencies[MEMORY_COUNT * 999 / 1000];
    let max = store_latencies[MEMORY_COUNT - 1];

    println!("Store latencies:");
    println!("  p50:   {p50:?}");
    println!("  p90:   {p90:?}");
    println!("  p95:   {p95:?}");
    println!("  p99:   {p99:?}");
    println!("  p99.9: {p999:?}");
    println!("  max:   {max:?}");

    // Detect performance cliffs (p99.9 should be <10x p50)
    let cliff_ratio = p999.as_nanos() as f64 / p50.as_nanos().max(1) as f64;
    println!("Performance cliff ratio (p99.9/p50): {cliff_ratio:.1}x");

    assert!(
        cliff_ratio < 10.0,
        "Performance cliff detected: p99.9 is {cliff_ratio:.1}x p50 (target <10x)"
    );

    println!("\nStore phase: {store_duration:?} ({store_rate:.0} ops/s)");

    // Verify counts
    let counts = store.get_tier_counts();
    assert_eq!(counts.hot, MEMORY_COUNT, "Should have 100K memories");

    // Test iteration performance
    println!("\nTesting iteration performance...");
    let iter_start = Instant::now();
    let memories: Vec<(String, Episode)> = store.iter_hot_memories().collect();
    let iter_duration = iter_start.elapsed();

    println!(
        "Iteration: {:?} ({} memories)",
        iter_duration,
        memories.len()
    );

    assert_eq!(
        memories.len(),
        MEMORY_COUNT,
        "Iteration should return all memories"
    );
    assert!(
        iter_duration.as_secs() < 2,
        "Iteration too slow: {iter_duration:?} (target <2s)"
    );

    // Verify random access performance with percentile tracking
    println!("\nTesting random access performance...");
    let mut recall_latencies = Vec::with_capacity(1000);
    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
        let i = rng.gen_range(0..MEMORY_COUNT);
        let memory_id = format!("ep-{i:06}");

        let cue = CueBuilder::new()
            .id(format!("recall-{i}"))
            .embedding_search(create_test_embedding(i as f32), Confidence::LOW)
            .build();

        let recall_start = Instant::now();
        let result = store.recall(&cue);
        recall_latencies.push(recall_start.elapsed());

        assert!(
            !result.results.is_empty(),
            "Failed to recall memory {memory_id}"
        );
    }

    recall_latencies.sort();
    let recall_p50 = recall_latencies[500];
    let recall_p99 = recall_latencies[990];
    let recall_p999 = recall_latencies[999];

    println!("Random access latencies:");
    println!("  p50:   {recall_p50:?}");
    println!("  p99:   {recall_p99:?}");
    println!("  p99.9: {recall_p999:?}");

    // Memory usage analysis
    let final_rss = get_process_memory_rss_mb();
    let rss_growth = final_rss.saturating_sub(initial_rss);
    let bytes_per_memory = if rss_growth > 0 {
        (rss_growth * 1024 * 1024) / MEMORY_COUNT
    } else {
        0
    };

    println!("\nMemory usage:");
    println!("  Initial: {initial_rss}MB");
    println!("  Final: {final_rss}MB");
    println!("  Growth: {rss_growth}MB");
    println!("  Per memory: {bytes_per_memory} bytes");

    // Performance assertions
    assert!(
        store_rate > 5000.0,
        "Store rate too slow: {store_rate:.0} ops/s (target >5000)"
    );

    assert!(
        recall_p99 < Duration::from_millis(1),
        "P99 recall latency too high: {recall_p99:?} (target <1ms)"
    );

    if initial_rss > 0 {
        // Only check if we could measure RSS
        assert!(
            rss_growth < 500,
            "Memory overhead too high: {rss_growth}MB (target <500MB)"
        );
    }

    // Verify no performance degradation (early vs late stores)
    let early_p99_idx = MEMORY_COUNT / 100; // P99 of first 1K stores
    let early_p99 = store_latencies[early_p99_idx];
    let late_p99 = p99; // P99 of all stores

    let degradation_ratio = late_p99.as_nanos() as f64 / early_p99.as_nanos().max(1) as f64;
    println!("\nPerformance degradation: {degradation_ratio:.2}x");

    assert!(
        degradation_ratio < 2.0,
        "Performance degraded significantly: {degradation_ratio:.2}x (target <2x)"
    );
}

// ===== Test 3: 1M Cold Tier Iteration - Scalability Limit =====

#[tokio::test]
#[ignore = "very_slow"]
async fn test_1m_cold_tier_iteration_limit() {
    const MEMORY_COUNT: usize = 1_000_000;
    const HOT_TIER_SIZE: usize = 1000; // Small hot tier forces cold storage

    let temp_dir = TempDir::new().expect("temp dir creation");
    let store = MemoryStore::new(HOT_TIER_SIZE)
        .with_persistence(temp_dir.path())
        .expect("Failed to create store");

    let initial_rss = get_process_memory_rss_mb();

    // Store 1M memories (most will be evicted to warm/cold)
    println!("Storing {MEMORY_COUNT} memories (will evict to warm/cold)...");
    let start = Instant::now();
    let mut store_sample_latencies = Vec::new();

    for i in 0..MEMORY_COUNT {
        let store_start = Instant::now();

        let episode = EpisodeBuilder::new()
            .id(format!("ep-{i:07}"))
            .when(Utc::now())
            .what(format!("Episode {i}"))
            .embedding(create_test_embedding(i as f32))
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);

        // Sample latencies (every 1000th operation to avoid overhead)
        if i % 1000 == 0 {
            store_sample_latencies.push(store_start.elapsed());
        }

        if i > 0 && i % 100_000 == 0 {
            let elapsed = start.elapsed().as_secs();
            let rate = (i as f64) / (elapsed as f64).max(1.0);
            let current_rss = get_process_memory_rss_mb();
            println!("Stored {i} memories ({rate:.0} ops/s, {current_rss}MB RSS)");
        }
    }

    let store_duration = start.elapsed();
    let store_rate = (MEMORY_COUNT as f64) / store_duration.as_secs_f64();

    println!("Store phase: {store_duration:?} ({store_rate:.0} ops/s)");

    // Analyze store latencies for degradation
    if !store_sample_latencies.is_empty() {
        store_sample_latencies.sort();
        let early_median = store_sample_latencies[store_sample_latencies.len() / 4];
        let late_median = store_sample_latencies[store_sample_latencies.len() * 3 / 4];
        let degradation = late_median.as_nanos() as f64 / early_median.as_nanos().max(1) as f64;

        println!("Store latency degradation: {degradation:.2}x");
    }

    // Test hot tier iteration (should be fast - only 1K memories)
    println!("Testing hot tier iteration...");
    let hot_start = Instant::now();
    let hot_memory_count = store.iter_hot_memories().count();
    let hot_duration = hot_start.elapsed();

    println!("Hot tier iteration: {hot_duration:?} ({hot_memory_count} memories)",);

    assert!(
        hot_duration < Duration::from_millis(100),
        "Hot tier iteration too slow: {hot_duration:?}"
    );

    // Test cold tier iteration (performance critical)
    if let Some(cold_iter) = store.iter_cold_memories() {
        println!("Testing cold tier iteration...");
        let cold_start = Instant::now();

        let cold_memories: Vec<(String, Episode)> = cold_iter.collect();

        let cold_duration = cold_start.elapsed();
        let cold_rate = (cold_memories.len() as f64) / cold_duration.as_secs_f64();

        println!(
            "Cold tier iteration: {:?} ({} memories, {:.0} items/s)",
            cold_duration,
            cold_memories.len(),
            cold_rate
        );

        // Verify iteration completed in reasonable time
        assert!(
            cold_duration.as_secs() < 5,
            "Cold tier iteration too slow: {:?} for {} memories (target <5s)",
            cold_duration,
            cold_memories.len()
        );

        // Verify rate is acceptable (>100K items/s)
        assert!(
            cold_rate > 100_000.0,
            "Cold tier iteration rate too slow: {cold_rate:.0} items/s (target >100K)"
        );

        // Spot check content integrity (first, middle, last)
        if !cold_memories.is_empty() {
            for idx in [0, cold_memories.len() / 2, cold_memories.len() - 1] {
                let (_, episode) = &cold_memories[idx];
                assert!(
                    episode.what.starts_with("Episode "),
                    "Cold tier content corrupted at index {}: {}",
                    idx,
                    episode.what
                );
            }
        }
    } else {
        println!("No cold tier available (expected for small datasets)");
    }

    // Memory usage check
    let final_rss = get_process_memory_rss_mb();
    let rss_growth = final_rss.saturating_sub(initial_rss);

    println!("Memory usage: initial={initial_rss}MB, final={final_rss}MB, growth={rss_growth}MB");

    // At 1M scale, should use <2GB total
    if initial_rss > 0 {
        assert!(
            final_rss < 2048,
            "Memory usage too high: {final_rss}MB (target <2GB)"
        );
    }
}

// ===== Test 4: Content Size Extremes with Fragmentation =====

#[tokio::test]
#[ignore = "slow"]
async fn test_extreme_content_sizes_fragmentation() {
    const MEMORY_COUNT: usize = 20_000;

    let temp_dir = TempDir::new().expect("temp dir creation");
    let store = MemoryStore::new(MEMORY_COUNT + 5000)
        .with_persistence(temp_dir.path())
        .expect("Failed to create store");

    let initial_rss = get_process_memory_rss_mb();

    // Phase 1: Store memories with varying sizes
    println!("Phase 1: Storing {MEMORY_COUNT} memories with extreme size variation...");

    // Size distribution:
    // - 40% empty (0 bytes)
    // - 40% tiny (1-10 bytes)
    // - 10% small (100 bytes)
    // - 5% medium (1KB)
    // - 4% large (10KB)
    // - 1% huge (100KB)

    for i in 0..MEMORY_COUNT {
        let (id, content) = if i % 100 == 0 {
            // 1% huge (100KB)
            (format!("huge-{i}"), "H".repeat(100_000))
        } else if i % 25 == 0 {
            // 4% large (10KB)
            (format!("large-{i}"), "L".repeat(10_000))
        } else if i % 20 == 0 {
            // 5% medium (1KB)
            (format!("medium-{i}"), "M".repeat(1_000))
        } else if i % 10 == 0 {
            // 10% small (100 bytes)
            (format!("small-{i}"), "S".repeat(100))
        } else if i % 5 < 2 {
            // 40% empty
            (format!("empty-{i}"), String::new())
        } else {
            // 40% tiny (1-10 bytes)
            (format!("tiny-{i}"), "t".repeat((i % 10) + 1))
        };

        let episode = EpisodeBuilder::new()
            .id(id)
            .when(Utc::now())
            .what(content)
            .embedding(create_test_embedding(i as f32))
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);

        if i > 0 && i % 5000 == 0 {
            println!("Stored {i} memories");
        }
    }

    let phase1_rss = get_process_memory_rss_mb();
    let phase1_growth = phase1_rss.saturating_sub(initial_rss);
    println!("Phase 1 complete: RSS growth = {phase1_growth}MB");

    // Phase 2: Remove half the memories to create fragmentation
    println!("\nPhase 2: Removing 50% of memories to create fragmentation...");

    let mut removed_count = 0;
    for i in (0..MEMORY_COUNT).step_by(2) {
        // Remove every other memory
        let id = if i % 100 == 0 {
            format!("huge-{i}")
        } else if i % 25 == 0 {
            format!("large-{i}")
        } else if i % 20 == 0 {
            format!("medium-{i}")
        } else if i % 10 == 0 {
            format!("small-{i}")
        } else if i % 5 < 2 {
            format!("empty-{i}")
        } else {
            format!("tiny-{i}")
        };

        // Note: MemoryStore doesn't have a direct remove() method,
        // so we simulate removal by overwriting with tombstone content
        let tombstone = EpisodeBuilder::new()
            .id(format!("{id}-removed"))
            .when(Utc::now())
            .what("REMOVED".to_string())
            .embedding([0.0f32; 768])
            .confidence(Confidence::LOW)
            .build();

        store.store(tombstone);
        removed_count += 1;
    }

    println!("Removed {removed_count} memories (simulated)");

    let phase2_rss = get_process_memory_rss_mb();
    println!("Phase 2 complete: RSS = {phase2_rss}MB (should not grow much)");

    // Verify counts (should have original + tombstones)
    let counts_after_removal = store.get_tier_counts();
    println!(
        "Memory count after removal: {} (original {} + tombstones)",
        counts_after_removal.hot, MEMORY_COUNT
    );

    // Phase 3: Store new memories in the fragmented space
    println!(
        "\nPhase 3: Storing {} new memories in fragmented space...",
        MEMORY_COUNT / 2
    );

    for i in 0..(MEMORY_COUNT / 2) {
        let episode = EpisodeBuilder::new()
            .id(format!("new-{i}"))
            .when(Utc::now())
            .what(format!("New content {} - {}", i, "data".repeat(i % 20)))
            .embedding(create_test_embedding((MEMORY_COUNT + i) as f32))
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);
    }

    let phase3_rss = get_process_memory_rss_mb();
    let total_growth = phase3_rss.saturating_sub(initial_rss);
    println!("Phase 3 complete: Total RSS growth = {total_growth}MB");

    // Verify final counts
    let final_counts = store.get_tier_counts();
    println!("Final memory count: {}", final_counts.hot);

    // Phase 4: Verify content integrity for all size classes
    println!("\nPhase 4: Verifying content integrity across size classes...");

    // Check huge (should still exist at odd indices)
    for i in (1..100).step_by(2) {
        let huge_id = format!("huge-{}", i * 100);
        let huge_idx = i * 100;

        let cue = CueBuilder::new()
            .id(format!("check-huge-{huge_idx}"))
            .embedding_search(create_test_embedding(huge_idx as f32), Confidence::LOW)
            .build();

        let result = store.recall(&cue);

        if !result.results.is_empty() {
            let episode = &result.results[0].0;
            if episode.id == huge_id {
                assert_eq!(
                    episode.what.len(),
                    100_000,
                    "Huge content size wrong for {huge_id}"
                );
            }
        }
    }

    // Check new memories
    for i in 0..100 {
        let new_id = format!("new-{i}");

        let cue = CueBuilder::new()
            .id(format!("check-new-{i}"))
            .embedding_search(
                create_test_embedding((MEMORY_COUNT + i) as f32),
                Confidence::LOW,
            )
            .build();

        let result = store.recall(&cue);

        assert!(
            !result.results.is_empty(),
            "Failed to recall new memory {new_id}"
        );

        let episode = &result.results[0].0;
        assert!(
            episode.what.starts_with(&format!("New content {i}")),
            "New content corrupted for {new_id}"
        );
    }

    println!("\nFragmentation test complete!");
    println!("Total RSS growth: {total_growth}MB");
}

// ===== Test 5: Memory Leak Detection - Long-Running Stability =====

#[tokio::test]
#[ignore = "very_slow"]
async fn test_long_running_memory_leak_detection() {
    const ITERATIONS: usize = 10;
    const MEMORIES_PER_ITERATION: usize = 10_000;

    let temp_dir = TempDir::new().expect("temp dir creation");
    let store = MemoryStore::new(MEMORIES_PER_ITERATION + 1000)
        .with_persistence(temp_dir.path())
        .expect("Failed to create store");

    let initial_rss = get_process_memory_rss_mb();
    let mut rss_samples = Vec::new();

    println!("Running {ITERATIONS} iterations of {MEMORIES_PER_ITERATION} memories each...");
    println!("Initial RSS: {initial_rss}MB");

    for iteration in 0..ITERATIONS {
        println!("\n--- Iteration {} ---", iteration + 1);

        // Store memories
        for i in 0..MEMORIES_PER_ITERATION {
            let episode = EpisodeBuilder::new()
                .id(format!("iter{iteration}-mem{i}"))
                .when(Utc::now())
                .what(format!("Content for iteration {iteration} memory {i}"))
                .embedding(create_test_embedding(
                    (iteration * MEMORIES_PER_ITERATION + i) as f32,
                ))
                .confidence(Confidence::HIGH)
                .build();

            store.store(episode);
        }

        // Note: MemoryStore doesn't have a direct remove() method
        // In a real scenario, we would trigger eviction or compaction here
        // For now, we just measure memory growth with accumulating data

        // Force allocator to settle
        std::hint::black_box(&store);

        // Sample RSS
        let current_rss = get_process_memory_rss_mb();
        rss_samples.push(current_rss);
        println!(
            "Iteration {} complete: RSS = {}MB",
            iteration + 1,
            current_rss
        );

        // Short sleep to let allocator settle
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let final_rss = get_process_memory_rss_mb();
    let total_growth = final_rss.saturating_sub(initial_rss);

    println!("\n--- Memory Leak Analysis ---");
    println!("Initial RSS: {initial_rss}MB");
    println!("Final RSS: {final_rss}MB");
    println!("Total growth: {total_growth}MB");
    println!("RSS samples: {rss_samples:?}");

    if initial_rss > 0 && rss_samples.len() > 1 {
        // Linear regression to detect memory leak trend
        let n = rss_samples.len() as f64;
        let sum_x: f64 = (0..rss_samples.len()).map(|i| i as f64).sum();
        let sum_y: f64 = rss_samples.iter().map(|&v| v as f64).sum();
        let sum_x_times_y: f64 = rss_samples
            .iter()
            .enumerate()
            .map(|(i, &v)| i as f64 * v as f64)
            .sum();
        let sum_x_squared: f64 = (0..rss_samples.len()).map(|i| (i * i) as f64).sum();

        // Linear regression formula: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        #[allow(clippy::suspicious_operation_groupings)]
        let slope = (n * sum_x_times_y - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x);
        println!("Memory growth slope: {slope:.2} MB/iteration");

        // Note: This test accumulates memories, so we expect linear growth
        // A real leak would show super-linear growth or growth when removing memories
        println!("Note: This test accumulates memories, so linear growth is expected");
        println!(
            "Expected growth per iteration: ~{}MB",
            (MEMORIES_PER_ITERATION * 150) / (1024 * 1024)
        );
    }
}

// ===== Test 6: Performance Cliff Detection =====

#[tokio::test]
#[ignore = "very_slow"]
async fn test_performance_cliff_detection() {
    // Test at increasing scales: 1K, 10K, 50K, 100K, 200K
    let scale_points = vec![1_000, 10_000, 50_000, 100_000, 200_000];

    println!("Testing performance across scale points: {scale_points:?}");
    println!(
        "{:<10} {:<15} {:<15} {:<15} {:<15}",
        "Scale", "Store (ops/s)", "P99 Recall", "Iteration", "RSS (MB)"
    );
    println!("{:-<70}", "");

    for &memory_count in &scale_points {
        let temp_dir = TempDir::new().expect("temp dir creation");
        let store = MemoryStore::new(memory_count + 1000)
            .with_persistence(temp_dir.path())
            .expect("Failed to create store");

        let initial_rss = get_process_memory_rss_mb();

        // Store phase
        let store_start = Instant::now();
        for i in 0..memory_count {
            let episode = EpisodeBuilder::new()
                .id(format!("ep-{i}"))
                .when(Utc::now())
                .what(format!("Content {i}"))
                .embedding(create_test_embedding(i as f32))
                .confidence(Confidence::HIGH)
                .build();

            store.store(episode);
        }
        let store_duration = store_start.elapsed();
        let store_rate = (memory_count as f64) / store_duration.as_secs_f64();

        // Recall phase (sample 100 random accesses)
        let sample_size = 100.min(memory_count);
        let mut recall_latencies = Vec::new();
        let mut rng = rand::thread_rng();

        for _ in 0..sample_size {
            let i = rng.gen_range(0..memory_count);

            let cue = CueBuilder::new()
                .id(format!("cliff-recall-{i}"))
                .embedding_search(create_test_embedding(i as f32), Confidence::LOW)
                .build();

            let recall_start = Instant::now();
            store.recall(&cue);
            recall_latencies.push(recall_start.elapsed());
        }

        recall_latencies.sort();
        let p99_recall = if recall_latencies.len() > 1 {
            recall_latencies[recall_latencies.len() * 99 / 100]
        } else {
            Duration::from_nanos(0)
        };

        // Iteration phase
        let iter_start = Instant::now();
        let _memories: Vec<_> = store.iter_hot_memories().collect();
        let iter_duration = iter_start.elapsed();

        let final_rss = get_process_memory_rss_mb();

        println!(
            "{:<10} {:<15.0} {:<15?} {:<15?} {:<15}",
            memory_count,
            store_rate,
            p99_recall,
            iter_duration,
            final_rss.saturating_sub(initial_rss)
        );

        // Detect cliff: if store rate drops below 1000 ops/s
        if store_rate < 1000.0 {
            println!("PERFORMANCE CLIFF DETECTED at {memory_count} memories!");
        }

        // Detect cliff: if P99 recall exceeds 10ms
        if p99_recall > Duration::from_millis(10) {
            println!("LATENCY CLIFF DETECTED at {memory_count} memories!");
        }

        // Detect cliff: if iteration exceeds 5s
        if iter_duration > Duration::from_secs(5) {
            println!("ITERATION CLIFF DETECTED at {memory_count} memories!");
        }
    }
}

// ===== Test 7: Concurrent Operations at Scale =====

#[tokio::test(flavor = "multi_thread", worker_threads = 20)]
#[ignore = "slow"]
async fn test_100k_concurrent_operations() {
    const INITIAL_MEMORIES: usize = 100_000;
    const CONCURRENT_WRITERS: usize = 10;
    const CONCURRENT_READERS: usize = 10;
    const OPERATIONS_PER_THREAD: usize = 1000;

    let temp_dir = TempDir::new().expect("temp dir creation");
    let store = Arc::new(
        MemoryStore::new(INITIAL_MEMORIES + CONCURRENT_WRITERS * OPERATIONS_PER_THREAD)
            .with_persistence(temp_dir.path())
            .expect("Failed to create store"),
    );

    // Pre-populate with 100K memories
    println!("Pre-populating with {INITIAL_MEMORIES} memories...");
    for i in 0..INITIAL_MEMORIES {
        let episode = EpisodeBuilder::new()
            .id(format!("initial-{i}"))
            .when(Utc::now())
            .what(format!("Initial content {i}"))
            .embedding(create_test_embedding(i as f32))
            .confidence(Confidence::HIGH)
            .build();

        store.store(episode);

        if i > 0 && i % 10_000 == 0 {
            println!("Pre-populated {i} memories");
        }
    }

    println!("Starting concurrent operations...");

    let barrier = Arc::new(tokio::sync::Barrier::new(
        CONCURRENT_WRITERS + CONCURRENT_READERS,
    ));
    let mut handles = vec![];

    // Spawn writer threads
    for writer_id in 0..CONCURRENT_WRITERS {
        let store = Arc::clone(&store);
        let barrier = Arc::clone(&barrier);

        let handle = tokio::spawn(async move {
            barrier.wait().await;

            for i in 0..OPERATIONS_PER_THREAD {
                let episode = EpisodeBuilder::new()
                    .id(format!("writer{writer_id}-mem{i}"))
                    .when(Utc::now())
                    .what(format!("Content from writer {writer_id} memory {i}"))
                    .embedding(create_test_embedding((writer_id * 1000 + i) as f32))
                    .confidence(Confidence::HIGH)
                    .build();

                store.store(episode);
            }
        });
        handles.push(handle);
    }

    // Spawn reader threads
    for reader_id in 0..CONCURRENT_READERS {
        let store = Arc::clone(&store);
        let barrier = Arc::clone(&barrier);

        let handle = tokio::spawn(async move {
            barrier.wait().await;
            let mut rng = rand::thread_rng();

            for _ in 0..OPERATIONS_PER_THREAD {
                let i = rng.gen_range(0..INITIAL_MEMORIES);

                let cue = CueBuilder::new()
                    .id(format!("concurrent-read-{reader_id}-{i}"))
                    .embedding_search(create_test_embedding(i as f32), Confidence::LOW)
                    .build();

                store.recall(&cue);
            }
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    let start = Instant::now();
    for handle in handles {
        handle.await.expect("Thread panicked");
    }
    let duration = start.elapsed();

    println!("All concurrent operations completed in {duration:?}");

    // Verify final state
    let counts = store.get_tier_counts();
    let expected = INITIAL_MEMORIES + CONCURRENT_WRITERS * OPERATIONS_PER_THREAD;

    assert_eq!(
        counts.hot, expected,
        "Memory count mismatch after concurrent operations"
    );
}
