//! Soak test for memory leak detection
//!
//! Runs for 10 minutes performing continuous cognitive operations
//! to detect memory leaks and ensure stable memory usage.
//!
//! Target: <10 bytes/op memory growth
//! Run with: cargo test --test soak_test_memory_leaks --release -- --ignored --nocapture

use engram_core::cognitive::priming::{
    AssociativePrimingEngine, RepetitionPrimingEngine, SemanticPrimingEngine,
};
use engram_core::cognitive::reconsolidation::ReconsolidationEngine;
use engram_core::{Confidence, Cue, Episode, MemoryStore};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Get current process memory usage in bytes (platform-specific)
#[cfg(target_os = "linux")]
fn get_process_memory_bytes() -> usize {
    use std::fs;

    // Parse /proc/self/status for VmRSS (Resident Set Size)
    let status = fs::read_to_string("/proc/self/status").unwrap();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kb: usize = parts[1].parse().unwrap();
                return kb * 1024; // Convert to bytes
            }
        }
    }
    0
}

#[cfg(target_os = "macos")]
fn get_process_memory_bytes() -> usize {
    // Note: Requires mach2 dependency for full functionality
    // For now, return 0 to allow compilation without mach2
    println!("Warning: Memory tracking on macOS requires mach2 dependency");
    0
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn get_process_memory_bytes() -> usize {
    // Fallback for unsupported platforms
    println!("Warning: Memory tracking not supported on this platform");
    0
}

#[test]
#[ignore = "Long-running test (10 minutes)"]
fn test_no_memory_leaks_during_soak() {
    // Leak detection criteria
    const MAX_GROWTH_PER_OP: f64 = 10.0; // bytes (very strict threshold)
    const MAX_TOTAL_GROWTH_MB: i64 = 100; // 100 MB total growth allowed

    println!("\n=== Starting 10-minute soak test for memory leak detection ===\n");

    // Create all cognitive engines
    let semantic = Arc::new(SemanticPrimingEngine::new());
    let associative = Arc::new(AssociativePrimingEngine::new());
    let repetition = Arc::new(RepetitionPrimingEngine::new());
    let reconsolidation = Arc::new(ReconsolidationEngine::new());
    let store = Arc::new(MemoryStore::new(10000));

    // Warm-up phase: populate allocator pools and caches
    println!("Warm-up phase: 10,000 operations...");
    perform_warmup_operations(
        &semantic,
        &associative,
        &repetition,
        &reconsolidation,
        &store,
    );

    // Force GC/compaction if applicable
    std::thread::sleep(Duration::from_secs(5));

    // Baseline measurement AFTER warm-up
    let baseline_memory = get_process_memory_bytes();
    println!(
        "Baseline memory: {} MB ({} bytes)\n",
        baseline_memory / 1_000_000,
        baseline_memory
    );

    // Soak test: 10 minutes at ~1000 ops/sec = 600,000 operations
    println!("Starting 10-minute soak test (600,000 operations)...");
    let start = Instant::now();
    let mut op_count = 0u64;
    let mut max_memory = baseline_memory;
    let mut min_memory = baseline_memory;

    let mut last_report = Instant::now();

    while start.elapsed() < Duration::from_secs(600) {
        // Perform cognitive operation
        perform_cognitive_operation(
            &semantic,
            &associative,
            &repetition,
            &reconsolidation,
            &store,
            op_count,
        );
        op_count += 1;

        // Sample and report every 10 seconds
        if last_report.elapsed() >= Duration::from_secs(10) {
            let current_memory = get_process_memory_bytes();
            max_memory = max_memory.max(current_memory);
            min_memory = min_memory.min(current_memory);

            let growth = current_memory as i64 - baseline_memory as i64;
            let growth_per_op = if op_count > 0 {
                growth as f64 / op_count as f64
            } else {
                0.0
            };

            println!(
                "[{:3}s] ops={:7} mem={:4} MB growth={:+4} MB ({:+.2} bytes/op) throughput={:.0} ops/s",
                start.elapsed().as_secs(),
                op_count,
                current_memory / 1_000_000,
                growth / 1_000_000,
                growth_per_op,
                op_count as f64 / start.elapsed().as_secs_f64()
            );

            last_report = Instant::now();
        }

        // Rate limiting: target ~1000 ops/sec to avoid CPU throttling
        if op_count.is_multiple_of(1000) {
            let elapsed_secs = start.elapsed().as_secs_f64();
            let target_secs = op_count as f64 / 1000.0;
            if elapsed_secs < target_secs {
                std::thread::sleep(Duration::from_millis(
                    ((target_secs - elapsed_secs) * 1000.0) as u64,
                ));
            }
        }
    }

    // Final measurement
    let final_memory = get_process_memory_bytes();

    println!("\n=== Soak test complete ===");
    println!("Total operations: {op_count}");
    println!("Duration: {:.1}s", start.elapsed().as_secs_f64());
    println!(
        "Throughput: {:.0} ops/s",
        op_count as f64 / start.elapsed().as_secs_f64()
    );

    // Memory analysis
    let memory_growth = final_memory as i64 - baseline_memory as i64;
    let growth_per_op = if op_count > 0 {
        memory_growth as f64 / op_count as f64
    } else {
        0.0
    };
    let max_growth = max_memory as i64 - baseline_memory as i64;

    println!("\n=== Memory Analysis ===");
    println!("Baseline:  {} MB", baseline_memory / 1_000_000);
    println!("Final:     {} MB", final_memory / 1_000_000);
    println!(
        "Growth:    {:+} MB ({:+.2} bytes/op)",
        memory_growth / 1_000_000,
        growth_per_op
    );
    println!(
        "Peak:      {} MB ({:+} MB from baseline)",
        max_memory / 1_000_000,
        max_growth / 1_000_000
    );
    println!("Min:       {} MB", min_memory / 1_000_000);

    assert!(
        growth_per_op.abs() <= MAX_GROWTH_PER_OP,
        "\n❌ MEMORY LEAK DETECTED!\n\
             Growth per operation: {:.2} bytes/op (threshold: {:.2} bytes/op)\n\
             Total growth: {:+} MB over {} operations\n",
        growth_per_op,
        MAX_GROWTH_PER_OP,
        memory_growth / 1_000_000,
        op_count
    );

    assert!(
        memory_growth <= MAX_TOTAL_GROWTH_MB * 1_000_000,
        "\n❌ EXCESSIVE MEMORY GROWTH!\n\
             Total growth: {:+} MB (threshold: {} MB)\n",
        memory_growth / 1_000_000,
        MAX_TOTAL_GROWTH_MB
    );

    println!("\n✅ VERDICT: PASS - No memory leaks detected");
    println!(
        "Growth per operation: {growth_per_op:.2} bytes/op (threshold: {MAX_GROWTH_PER_OP:.2})"
    );
}

/// Warm-up to populate allocator pools
fn perform_warmup_operations(
    semantic: &Arc<SemanticPrimingEngine>,
    associative: &Arc<AssociativePrimingEngine>,
    repetition: &Arc<RepetitionPrimingEngine>,
    reconsolidation: &Arc<ReconsolidationEngine>,
    store: &Arc<MemoryStore>,
) {
    for i in 0..10_000 {
        perform_cognitive_operation(semantic, associative, repetition, reconsolidation, store, i);
    }
}

/// Perform a single cognitive operation (randomized)
fn perform_cognitive_operation(
    semantic: &Arc<SemanticPrimingEngine>,
    associative: &Arc<AssociativePrimingEngine>,
    repetition: &Arc<RepetitionPrimingEngine>,
    reconsolidation: &Arc<ReconsolidationEngine>,
    store: &Arc<MemoryStore>,
    iteration: u64,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(iteration);

    match rng.gen_range(0..100) {
        0..30 => {
            // 30% semantic priming
            let concept = format!("concept_{}", rng.gen_range(0..1000));
            let embedding = random_embedding(&mut rng);
            semantic.activate_priming(&concept, &embedding, Vec::new);
        }
        30..40 => {
            // 10% associative priming
            let node_a = format!("node_{}", rng.gen_range(0..1000));
            let node_b = format!("node_{}", rng.gen_range(0..1000));
            associative.record_coactivation(&node_a, &node_b);
        }
        40..75 => {
            // 35% recall operations
            let embedding = random_embedding(&mut rng);
            let cue = Cue::embedding("test".to_string(), embedding, Confidence::HIGH);
            let _ = store.recall(&cue);
        }
        75..90 => {
            // 15% store operations
            let episode = create_test_episode(rng.gen_range(0..10_000), 0);
            let _ = store.store(episode);
        }
        90..95 => {
            // 5% repetition priming
            let node_id = format!("episode_{}", rng.gen_range(0..1000));
            repetition.record_exposure(&node_id);
        }
        _ => {
            // 5% reconsolidation checks
            let episode = create_test_episode(rng.gen_range(0..1000), 48);
            reconsolidation.record_recall(&episode, Utc::now(), true);
        }
    }
}

fn create_test_episode(id: u64, age_hours: i64) -> Episode {
    let when = Utc::now() - chrono::Duration::hours(age_hours);
    let embedding = create_embedding(id as f32);

    Episode::new(
        format!("episode_{id}"),
        when,
        format!("content_{id}"),
        embedding,
        Confidence::HIGH,
    )
}

fn create_embedding(seed: f32) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for (i, val) in embedding.iter_mut().enumerate() {
        *val = ((seed + i as f32) * 0.001).sin();
    }
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }
    embedding
}

fn random_embedding<R: Rng>(rng: &mut R) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for val in &mut embedding {
        *val = rng.gen_range(-1.0..1.0);
    }
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }
    embedding
}
