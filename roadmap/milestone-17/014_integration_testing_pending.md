# Task 014: Integration Testing

## Objective
Design and implement comprehensive integration testing strategy for dual memory system, covering migration correctness, feature flag combinations, A/B comparison, load testing, chaos scenarios, and backwards compatibility verification with systematic differential testing between single-type and dual-type implementations.

## Background
Integration testing ensures all dual memory components work together correctly and maintain compatibility. The dual memory system introduces episodic and semantic memory types with gradual rollout capabilities (Tasks 001-013). We need systematic validation that:

1. Migration preserves data integrity and recall equivalence
2. Feature flag combinations work correctly (8 states: 2^3 flags)
3. Single-type and dual-type systems produce equivalent results
4. System handles concurrent load with consolidation running
5. Chaos scenarios (failures, OOM, timeouts) are handled gracefully
6. Backwards compatibility is maintained for existing clients

The codebase has strong testing patterns:
- Property-based testing with proptest (confidence_property_tests.rs)
- Differential testing between Rust/Zig (zig_differential/)
- Load/stress testing (query_stress_tests.rs, consolidation_load_tests.rs)
- Async integration tests with tokio::test (error_recovery_integration.rs)
- Test fixture builders (support/graph_builders.rs)
- OOM handling validation (oom_handling.rs)

## Technical Specification

### Files to Create

#### Core Integration Tests
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/dual_memory_integration.rs` - End-to-end scenarios
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/dual_memory_migration_correctness.rs` - Migration validation
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/dual_memory_feature_flags.rs` - Feature flag matrix testing
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/dual_memory_differential.rs` - A/B comparison tests
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/dual_memory_load.rs` - Concurrent load testing
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/dual_memory_chaos.rs` - Chaos engineering tests
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/dual_memory_backwards_compat.rs` - API compatibility

#### Test Support Infrastructure
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/support/dual_memory_fixtures.rs` - Test data generators
- `/Users/jordanwashburn/Workspace/orchard9/engram/tools/dual_memory_load_test.rs` - Standalone load testing tool (24h soak)

### 1. Migration Correctness Tests (`dual_memory_migration_correctness.rs`)

Validates that migration preserves data integrity and behavioral equivalence.

```rust
//! Migration correctness validation
//!
//! Systematic verification that migration from single-type Memory to dual-type
//! DualMemoryNode preserves all data and produces equivalent recall behavior.

use engram_core::{
    Memory, MemoryStore, Confidence, Episode, EpisodeBuilder,
    migration::DualMemoryMigrator,
    memory::{DualMemoryNode, MemoryNodeType},
};
use chrono::Utc;
use proptest::prelude::*;
use sha2::{Sha256, Digest};
use std::collections::HashSet;

/// Test oracle: Cryptographic hash of episode data
fn compute_episode_hash(episode: &Episode) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(&episode.id.as_bytes());
    hasher.update(&episode.embedding);
    hasher.update(&episode.encoding_confidence.raw().to_le_bytes());
    hasher.finalize().into()
}

/// Property: Migration preserves all episode data bit-for-bit
#[tokio::test]
async fn test_migration_preserves_episode_data() {
    let store = MemoryStore::new(10000);

    // Create 1000 episodes with known data
    let mut original_hashes = HashSet::new();
    for i in 0..1000 {
        let episode = create_deterministic_episode(i);
        original_hashes.insert(compute_episode_hash(&episode));
        store.store(episode);
    }

    // Migrate to dual memory
    let migrator = DualMemoryMigrator::new(store.backend());
    let dual_backend = migrator.migrate_offline().await.unwrap();

    // Verify all episodes present with identical data
    let mut migrated_hashes = HashSet::new();
    for node in dual_backend.get_all_nodes() {
        if let MemoryNodeType::Episode { episode_id, .. } = &node.node_type {
            let reconstructed_episode = dual_backend.get_episode(episode_id).unwrap();
            migrated_hashes.insert(compute_episode_hash(&reconstructed_episode));
        }
    }

    assert_eq!(original_hashes, migrated_hashes, "Migration must preserve all episode data");
}

/// Property: Migration preserves recall ordering
///
/// For any cue, the top-K recall results should be identical (within epsilon)
/// between original and migrated systems.
#[tokio::test]
async fn test_migration_preserves_recall_ordering() {
    let store = create_test_store_with_clusters(5, 100); // 5 clusters, 100 episodes each

    // Generate 50 diverse cues
    let test_cues = generate_test_cues(50, 0xDEADBEEF);

    // Collect baseline recall results
    let mut baseline_results = Vec::new();
    for cue in &test_cues {
        let result = store.recall(cue).await.unwrap();
        baseline_results.push(result.episodes.iter().map(|e| e.id.clone()).collect::<Vec<_>>());
    }

    // Migrate
    let migrator = DualMemoryMigrator::new(store.backend());
    let dual_backend = migrator.migrate_offline().await.unwrap();

    // Verify recall ordering preserved
    for (cue, baseline_order) in test_cues.iter().zip(baseline_results.iter()) {
        let migrated_result = dual_backend.recall(cue).await.unwrap();
        let migrated_order: Vec<_> = migrated_result.episodes.iter().map(|e| e.id.clone()).collect();

        // Allow minor reordering due to floating-point variance
        let overlap = compute_top_k_overlap(&baseline_order, &migrated_order, 10);
        assert!(overlap >= 0.9, "Top-10 recall overlap must be ≥90%: got {}", overlap);
    }
}

/// Property: Migration checkpoint allows resumption with no data loss
#[tokio::test]
async fn test_migration_checkpoint_resumption() {
    let store = create_test_store(10000);

    // Start migration
    let migrator = DualMemoryMigrator::new(store.backend());

    // Simulate crash after 3000 nodes migrated
    let checkpoint = migrator.migrate_with_crash_simulation(3000).await.unwrap();

    // Verify checkpoint contains exactly 3000 nodes
    assert_eq!(checkpoint.migrated_count, 3000);
    assert!(checkpoint.verify_source_integrity(store.backend()).unwrap());

    // Resume from checkpoint
    let resumed_migrator = DualMemoryMigrator::from_checkpoint(&checkpoint).unwrap();
    let final_backend = resumed_migrator.resume().await.unwrap();

    // Verify all 10000 nodes migrated
    assert_eq!(final_backend.node_count(), 10000);

    // No duplicates from checkpoint overlap
    let node_ids: HashSet<_> = final_backend.get_all_node_ids().collect();
    assert_eq!(node_ids.len(), 10000, "Checkpoint resumption must not duplicate nodes");
}

/// Property: Rollback restores exact pre-migration state
#[tokio::test]
async fn test_migration_rollback_correctness() {
    let store = create_test_store(1000);

    // Compute pre-migration hash
    let pre_migration_hash = compute_store_hash(&store);

    // Create rollback snapshot
    let migrator = DualMemoryMigrator::new(store.backend());
    let snapshot = migrator.create_rollback_snapshot().await.unwrap();

    // Perform migration
    let dual_backend = migrator.migrate_offline().await.unwrap();

    // Verify migration occurred
    assert!(dual_backend.has_dual_memory_types());

    // Rollback to snapshot
    migrator.rollback(&snapshot).await.unwrap();

    // Verify exact restoration
    let post_rollback_hash = compute_store_hash(&store);
    assert_eq!(pre_migration_hash, post_rollback_hash, "Rollback must restore exact state");
}

proptest! {
    /// Property-based test: Migration preserves embeddings for random episodes
    #[test]
    fn prop_migration_preserves_embeddings(
        num_episodes in 100_usize..1000,
        seed in 0_u64..10000,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let store = create_random_test_store(num_episodes, seed);

            // Collect all embeddings before migration
            let original_embeddings: Vec<[f32; 768]> = store
                .get_all_episodes()
                .map(|e| e.embedding)
                .collect();

            // Migrate
            let migrator = DualMemoryMigrator::new(store.backend());
            let dual_backend = migrator.migrate_offline().await.unwrap();

            // Verify embeddings unchanged
            let migrated_embeddings: Vec<[f32; 768]> = dual_backend
                .get_all_nodes()
                .filter_map(|n| if n.is_episode() { Some(n.embedding) } else { None })
                .collect();

            prop_assert_eq!(original_embeddings.len(), migrated_embeddings.len());

            for (orig, migr) in original_embeddings.iter().zip(migrated_embeddings.iter()) {
                for (o, m) in orig.iter().zip(migr.iter()) {
                    prop_assert!((o - m).abs() < 1e-6, "Embedding values must be preserved exactly");
                }
            }
        });
    }
}

/// Test data generator: Create deterministic episode from seed
fn create_deterministic_episode(seed: usize) -> Episode {
    use rand::{SeedableRng, Rng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);

    let mut embedding = [0.0f32; 768];
    for val in &mut embedding {
        *val = rng.gen_range(-1.0..1.0);
    }
    // Normalize
    let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }

    EpisodeBuilder::new()
        .id(format!("episode_{seed}"))
        .when(Utc::now())
        .what(format!("Test episode {seed}"))
        .embedding(embedding)
        .confidence(Confidence::exact(rng.gen_range(0.5..1.0)))
        .build()
}
```

### 2. Feature Flag Matrix Testing (`dual_memory_feature_flags.rs`)

Systematically tests all 8 combinations of the 3 feature flags using test-case parameterization.

```rust
//! Feature flag combination testing
//!
//! Tests all 2^3 = 8 combinations of dual memory feature flags:
//! - dual_memory_types: Enable DualMemoryNode type system
//! - concept_formation: Enable clustering and concept extraction
//! - blended_recall: Enable episodic+semantic recall fusion
//!
//! Ensures graceful degradation when features are disabled.

use engram_core::{
    MemoryStore, Confidence, EpisodeBuilder,
    config::{Config, DualMemoryFeatures},
};
use test_case::test_case;
use chrono::Utc;

/// Feature flag configuration for testing
#[derive(Debug, Clone, Copy)]
struct FeatureConfig {
    dual_memory_types: bool,
    concept_formation: bool,
    blended_recall: bool,
}

impl FeatureConfig {
    fn to_config(&self) -> Config {
        Config {
            dual_memory: DualMemoryFeatures {
                types_enabled: self.dual_memory_types,
                concept_formation_enabled: self.concept_formation,
                blended_recall_enabled: self.blended_recall,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn description(&self) -> String {
        format!(
            "types={} formation={} recall={}",
            self.dual_memory_types, self.concept_formation, self.blended_recall
        )
    }
}

// All 8 combinations of 3 boolean flags
#[test_case(false, false, false; "baseline: all disabled")]
#[test_case(true, false, false; "types only")]
#[test_case(false, true, false; "formation only (invalid: requires types)")]
#[test_case(false, false, true; "recall only (invalid: requires types)")]
#[test_case(true, true, false; "types + formation")]
#[test_case(true, false, true; "types + recall")]
#[test_case(false, true, true; "formation + recall (invalid: requires types)")]
#[test_case(true, true, true; "full dual memory")]
#[tokio::test]
async fn test_feature_flag_combination(
    dual_types: bool,
    concept_formation: bool,
    blended_recall: bool,
) {
    let features = FeatureConfig {
        dual_memory_types: dual_types,
        concept_formation,
        blended_recall,
    };

    // Invalid configurations should fail gracefully at init
    if (concept_formation || blended_recall) && !dual_types {
        let result = MemoryStore::with_config(features.to_config());
        assert!(result.is_err(), "Invalid config should be rejected: {}", features.description());
        return;
    }

    let store = MemoryStore::with_config(features.to_config()).unwrap();

    // All configurations must support basic episode storage
    for i in 0..100 {
        let episode = create_test_episode(i);
        store.store(episode).await.unwrap();
    }

    // All configurations must support recall
    let cue = create_test_cue();
    let results = store.recall(&cue).await.unwrap();
    assert!(!results.episodes.is_empty(), "Recall must work for config: {}", features.description());

    // Feature-specific validation
    if concept_formation {
        // Run consolidation
        let stats = store.consolidate().await.unwrap();

        // Should form concepts from clustered episodes
        assert!(stats.concepts_formed > 0, "Concept formation enabled but no concepts formed");
        assert!(stats.episodes_clustered > 0);

        // Verify concepts stored as DualMemoryNode with Concept type
        let concept_count = store.get_stats().concept_count;
        assert_eq!(concept_count, stats.concepts_formed);
    }

    if blended_recall {
        // Must have concepts to blend with
        if concept_formation {
            store.consolidate().await.unwrap();
        }

        // Recall should include semantic contribution
        let blended_results = store.recall(&cue).await.unwrap();

        if store.get_stats().concept_count > 0 {
            // Verify blended recall used both sources
            assert!(blended_results.metadata.used_semantic_layer);
            assert!(blended_results.metadata.used_episodic_layer);

            // Confidence should reflect blending
            assert!(blended_results.confidence.raw() > 0.0);
        }
    }

    if !dual_types {
        // Legacy mode: everything should be Memory type
        let all_nodes = store.backend().get_all_nodes();
        assert!(all_nodes.iter().all(|n| !n.is_dual_memory()));
    }
}

/// Property: Feature flags can be toggled dynamically without crashes
#[tokio::test]
async fn test_dynamic_feature_flag_transitions() {
    let mut config = Config::default();
    config.dual_memory.types_enabled = false;

    let store = MemoryStore::with_config(config.clone()).unwrap();

    // Add episodes in legacy mode
    for i in 0..100 {
        store.store(create_test_episode(i)).await.unwrap();
    }

    // Enable dual memory types mid-stream
    config.dual_memory.types_enabled = true;
    store.update_config(config.clone()).unwrap();

    // New episodes should use dual memory types
    store.store(create_test_episode(100)).await.unwrap();

    // Enable concept formation
    config.dual_memory.concept_formation_enabled = true;
    store.update_config(config.clone()).unwrap();

    // Should be able to consolidate
    let stats = store.consolidate().await.unwrap();
    assert!(stats.concepts_formed > 0);

    // Enable blended recall
    config.dual_memory.blended_recall_enabled = true;
    store.update_config(config).unwrap();

    // Should use both layers
    let results = store.recall(&create_test_cue()).await.unwrap();
    assert!(results.metadata.used_semantic_layer);
}

/// Property: Disabling features gracefully degrades to safe fallback
#[tokio::test]
async fn test_graceful_degradation_when_features_disabled() {
    // Start with full dual memory
    let mut config = Config::default();
    config.dual_memory.types_enabled = true;
    config.dual_memory.concept_formation_enabled = true;
    config.dual_memory.blended_recall_enabled = true;

    let store = MemoryStore::with_config(config.clone()).unwrap();

    // Build up dual memory state
    for i in 0..100 {
        store.store(create_test_episode(i)).await.unwrap();
    }
    store.consolidate().await.unwrap();

    // Disable blended recall - should fall back to episodic only
    config.dual_memory.blended_recall_enabled = false;
    store.update_config(config.clone()).unwrap();

    let results = store.recall(&create_test_cue()).await.unwrap();
    assert!(!results.metadata.used_semantic_layer, "Semantic layer should be disabled");
    assert!(results.metadata.used_episodic_layer, "Episodic layer should still work");

    // Disable concept formation - existing concepts remain but no new ones
    config.dual_memory.concept_formation_enabled = false;
    store.update_config(config.clone()).unwrap();

    let pre_concept_count = store.get_stats().concept_count;
    store.consolidate().await.unwrap(); // Should no-op
    let post_concept_count = store.get_stats().concept_count;

    assert_eq!(pre_concept_count, post_concept_count, "No new concepts should form");

    // Disable dual types - should fail gracefully (can't downgrade)
    config.dual_memory.types_enabled = false;
    let result = store.update_config(config);
    assert!(result.is_err(), "Cannot disable dual types after migration");
}
```

### 3. Differential Testing - A/B Comparison (`dual_memory_differential.rs`)

Validates that single-type and dual-type implementations produce equivalent results.

```rust
//! Differential testing between single-type and dual-type memory systems
//!
//! Following patterns from zig_differential/spreading_activation.rs:
//! - Property-based testing with 10,000 cases
//! - Statistical comparison of recall results
//! - Performance parity validation
//! - Numerical epsilon tolerance for floating-point operations

use engram_core::{MemoryStore, Memory, Confidence, EpisodeBuilder};
use proptest::prelude::*;
use approx::abs_diff_eq;
use std::time::Instant;

const EPSILON_RECALL: f32 = 1e-4; // Allow minor float variance
const NUM_PROPTEST_CASES: u32 = 1000; // Comprehensive coverage

/// Test oracle: Compare recall results between single-type and dual-type
fn assert_recall_equivalence(
    single_type_results: &[String],
    dual_type_results: &[String],
    k: usize,
) {
    // Top-K ordering should be identical (within epsilon for confidence scores)
    let overlap = compute_top_k_overlap(single_type_results, dual_type_results, k);

    assert!(
        overlap >= 0.95,
        "Recall equivalence violated: {:.2}% overlap (expected ≥95%)",
        overlap * 100.0
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(NUM_PROPTEST_CASES))]

    /// Property: Single-type and dual-type produce identical recall for random episode sets
    #[test]
    fn prop_recall_equivalence_random_episodes(
        num_episodes in 100_usize..1000,
        num_queries in 10_usize..50,
        seed in 0_u64..10000,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Create parallel stores: one single-type, one dual-type
            let single_store = MemoryStore::new(num_episodes);
            let dual_store = MemoryStore::with_dual_memory().unwrap();

            // Insert identical episodes into both
            let episodes = generate_random_episodes(num_episodes, seed);
            for episode in &episodes {
                single_store.store(episode.clone()).await.unwrap();
                dual_store.store(episode.clone()).await.unwrap();
            }

            // Generate diverse test cues
            let cues = generate_test_cues(num_queries, seed);

            // Compare recall results
            for cue in &cues {
                let single_results = single_store.recall(cue).await.unwrap();
                let dual_results = dual_store.recall(cue).await.unwrap();

                let single_ids: Vec<_> = single_results.episodes.iter().map(|e| e.id.clone()).collect();
                let dual_ids: Vec<_> = dual_results.episodes.iter().map(|e| e.id.clone()).collect();

                assert_recall_equivalence(&single_ids, &dual_ids, 10);

                // Confidence scores should be within epsilon
                for (single_conf, dual_conf) in single_results.confidences.iter().zip(dual_results.confidences.iter()) {
                    prop_assert!(
                        abs_diff_eq!(single_conf.raw(), dual_conf.raw(), epsilon = EPSILON_RECALL),
                        "Confidence mismatch: single={} dual={}",
                        single_conf.raw(),
                        dual_conf.raw()
                    );
                }
            }
        });
    }

    /// Property: Performance parity - dual-type should not be >10% slower
    #[test]
    fn prop_performance_parity_recall(
        num_episodes in 500_usize..1000,
        seed in 0_u64..1000,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let single_store = MemoryStore::new(num_episodes);
            let dual_store = MemoryStore::with_dual_memory().unwrap();

            let episodes = generate_random_episodes(num_episodes, seed);
            for episode in &episodes {
                single_store.store(episode.clone()).await.unwrap();
                dual_store.store(episode.clone()).await.unwrap();
            }

            let cues = generate_test_cues(100, seed);

            // Benchmark single-type recall
            let start_single = Instant::now();
            for cue in &cues {
                let _ = single_store.recall(cue).await.unwrap();
            }
            let duration_single = start_single.elapsed();

            // Benchmark dual-type recall
            let start_dual = Instant::now();
            for cue in &cues {
                let _ = dual_store.recall(cue).await.unwrap();
            }
            let duration_dual = start_dual.elapsed();

            // Allow 10% performance regression for dual-type
            let slowdown_ratio = duration_dual.as_secs_f64() / duration_single.as_secs_f64();

            prop_assert!(
                slowdown_ratio <= 1.10,
                "Dual-type recall too slow: {:.2}x slowdown (max 1.10x)",
                slowdown_ratio
            );
        });
    }
}

/// Statistical comparison: Verify recall distributions match
#[tokio::test]
async fn test_statistical_recall_distribution_equivalence() {
    use statrs::distribution::{DiscreteUniform, DiscreteCDF};

    // Create stores with 1000 episodes spanning 10 semantic clusters
    let single_store = create_clustered_store(10, 100, 0xCAFE);
    let dual_store = create_clustered_dual_store(10, 100, 0xCAFE);

    // Generate 500 cues uniformly sampling all clusters
    let cues = generate_clustered_cues(10, 50, 0xBEEF);

    let mut single_rank_distribution = vec![0u32; 1000];
    let mut dual_rank_distribution = vec![0u32; 1000];

    // Collect rank distributions
    for cue in &cues {
        let single_results = single_store.recall(cue).await.unwrap();
        let dual_results = dual_store.recall(cue).await.unwrap();

        for (rank, episode) in single_results.episodes.iter().enumerate() {
            let idx = extract_episode_index(&episode.id);
            single_rank_distribution[idx] = rank as u32;
        }

        for (rank, episode) in dual_results.episodes.iter().enumerate() {
            let idx = extract_episode_index(&episode.id);
            dual_rank_distribution[idx] = rank as u32;
        }
    }

    // Kolmogorov-Smirnov test: distributions should be statistically indistinguishable
    let ks_statistic = compute_ks_statistic(&single_rank_distribution, &dual_rank_distribution);

    assert!(
        ks_statistic < 0.05,
        "Recall distributions differ significantly: KS={:.4} (p<0.05)",
        ks_statistic
    );
}
```

### 4. Load and Stress Testing (`dual_memory_load.rs`)

Validates concurrent workload handling with realistic production patterns.

```rust
//! Load testing for dual memory system
//!
//! Simulates production workloads:
//! - Concurrent writers (episode ingestion)
//! - Concurrent readers (recall queries)
//! - Background consolidation
//! - Memory pressure scenarios
//! - Long-running soak tests
//!
//! Following patterns from query_stress_tests.rs and consolidation_load_tests.rs.

use engram_core::{MemoryStore, EpisodeBuilder, Confidence};
use tokio::task::JoinSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Instant, Duration};

/// Load test configuration
#[derive(Clone)]
struct LoadTestConfig {
    num_writer_tasks: usize,
    num_reader_tasks: usize,
    episodes_per_writer: usize,
    queries_per_reader: usize,
    consolidation_interval_sec: u64,
    test_duration_sec: u64,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            num_writer_tasks: 4,
            num_reader_tasks: 8,
            episodes_per_writer: 1000,
            queries_per_reader: 500,
            consolidation_interval_sec: 10,
            test_duration_sec: 60,
        }
    }
}

/// Load test metrics
#[derive(Default)]
struct LoadTestMetrics {
    episodes_written: AtomicU64,
    recalls_executed: AtomicU64,
    consolidations_run: AtomicU64,
    errors: AtomicU64,
}

/// Concurrent write + read + consolidation workload
#[tokio::test]
#[ignore = "Long-running load test"]
async fn test_concurrent_workload_mixed() {
    let config = LoadTestConfig::default();
    let store = Arc::new(MemoryStore::with_dual_memory().unwrap());
    let metrics = Arc::new(LoadTestMetrics::default());
    let shutdown = Arc::new(AtomicBool::new(false));

    let mut tasks = JoinSet::new();

    // Spawn writer tasks
    for writer_id in 0..config.num_writer_tasks {
        let store = Arc::clone(&store);
        let metrics = Arc::clone(&metrics);
        let shutdown = Arc::clone(&shutdown);
        let config = config.clone();

        tasks.spawn(async move {
            let mut count = 0;
            while !shutdown.load(Ordering::Relaxed) && count < config.episodes_per_writer {
                let episode = create_test_episode(writer_id * 10000 + count);

                match store.store(episode).await {
                    Ok(_) => {
                        metrics.episodes_written.fetch_add(1, Ordering::Relaxed);
                        count += 1;
                    }
                    Err(e) => {
                        metrics.errors.fetch_add(1, Ordering::Relaxed);
                        eprintln!("Writer {}: {}", writer_id, e);
                    }
                }

                // Slight backoff to simulate realistic ingestion rate
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
    }

    // Spawn reader tasks
    for reader_id in 0..config.num_reader_tasks {
        let store = Arc::clone(&store);
        let metrics = Arc::clone(&metrics);
        let shutdown = Arc::clone(&shutdown);
        let config = config.clone();

        tasks.spawn(async move {
            let mut count = 0;
            while !shutdown.load(Ordering::Relaxed) && count < config.queries_per_reader {
                let cue = create_random_cue(reader_id * 10000 + count);

                match store.recall(&cue).await {
                    Ok(_) => {
                        metrics.recalls_executed.fetch_add(1, Ordering::Relaxed);
                        count += 1;
                    }
                    Err(e) => {
                        metrics.errors.fetch_add(1, Ordering::Relaxed);
                        eprintln!("Reader {}: {}", reader_id, e);
                    }
                }

                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        });
    }

    // Spawn consolidation task
    {
        let store = Arc::clone(&store);
        let metrics = Arc::clone(&metrics);
        let shutdown = Arc::clone(&shutdown);
        let interval = Duration::from_secs(config.consolidation_interval_sec);

        tasks.spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                tokio::time::sleep(interval).await;

                match store.consolidate().await {
                    Ok(stats) => {
                        metrics.consolidations_run.fetch_add(1, Ordering::Relaxed);
                        println!("Consolidation: {} concepts formed", stats.concepts_formed);
                    }
                    Err(e) => {
                        metrics.errors.fetch_add(1, Ordering::Relaxed);
                        eprintln!("Consolidation error: {}", e);
                    }
                }
            }
        });
    }

    // Run test for configured duration
    tokio::time::sleep(Duration::from_secs(config.test_duration_sec)).await;
    shutdown.store(true, Ordering::Relaxed);

    // Wait for all tasks to complete
    while tasks.join_next().await.is_some() {}

    // Validate results
    let episodes = metrics.episodes_written.load(Ordering::Relaxed);
    let recalls = metrics.recalls_executed.load(Ordering::Relaxed);
    let consolidations = metrics.consolidations_run.load(Ordering::Relaxed);
    let errors = metrics.errors.load(Ordering::Relaxed);

    println!("\n=== Load Test Results ===");
    println!("Episodes written: {}", episodes);
    println!("Recalls executed: {}", recalls);
    println!("Consolidations run: {}", consolidations);
    println!("Errors: {}", errors);

    assert!(episodes > 0, "No episodes written");
    assert!(recalls > 0, "No recalls executed");
    assert!(consolidations > 0, "No consolidations run");
    assert!(errors == 0, "Errors occurred during load test");
}

/// Memory pressure test: Ensure graceful handling of OOM scenarios
#[tokio::test]
async fn test_memory_pressure_handling() {
    // Following patterns from oom_handling.rs

    let store = MemoryStore::with_dual_memory().unwrap();

    // Attempt to add episodes until memory pressure triggers backpressure
    let mut episodes_added = 0;
    const MAX_EPISODES: usize = 100_000; // Adjust based on test environment

    for i in 0..MAX_EPISODES {
        let episode = create_large_episode(i); // Larger than typical for stress

        match store.store(episode).await {
            Ok(_) => episodes_added += 1,
            Err(e) if e.is_memory_pressure() => {
                println!("Memory pressure detected after {} episodes", episodes_added);
                break;
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    // System should still be responsive despite memory pressure
    let cue = create_test_cue();
    let result = store.recall(&cue).await;
    assert!(result.is_ok(), "Recall should work despite memory pressure");

    // Consolidation should help free memory
    let consolidation_result = store.consolidate().await;
    assert!(consolidation_result.is_ok(), "Consolidation should succeed under pressure");
}

/// Throughput benchmark: Measure sustained write/read rates
#[tokio::test]
async fn test_sustained_throughput() {
    let store = Arc::new(MemoryStore::with_dual_memory().unwrap());

    const TARGET_WRITES_PER_SEC: u64 = 5000;
    const TEST_DURATION_SEC: u64 = 10;

    let writes = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    // Spawn 4 writers
    let mut handles = Vec::new();
    for writer_id in 0..4 {
        let store = Arc::clone(&store);
        let writes = Arc::clone(&writes);

        handles.push(tokio::spawn(async move {
            let mut count = 0;
            while start.elapsed() < Duration::from_secs(TEST_DURATION_SEC) {
                let episode = create_test_episode(writer_id * 1_000_000 + count);
                store.store(episode).await.unwrap();
                writes.fetch_add(1, Ordering::Relaxed);
                count += 1;
            }
        }));
    }

    // Wait for test duration
    for handle in handles {
        handle.await.unwrap();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total_writes = writes.load(Ordering::Relaxed);
    let writes_per_sec = (total_writes as f64) / elapsed;

    println!("\n=== Throughput Results ===");
    println!("Total writes: {}", total_writes);
    println!("Duration: {:.2}s", elapsed);
    println!("Writes/sec: {:.0}", writes_per_sec);

    assert!(
        writes_per_sec >= TARGET_WRITES_PER_SEC as f64,
        "Throughput below target: {:.0} < {}",
        writes_per_sec,
        TARGET_WRITES_PER_SEC
    );
}
```

### 5. Chaos Engineering Tests (`dual_memory_chaos.rs`)

Validates system resilience under failure scenarios and resource constraints.

```rust
//! Chaos engineering for dual memory system
//!
//! Tests resilience under:
//! - Random failures during concept formation
//! - Simulated network timeouts (prep for M14 distributed consolidation)
//! - Memory constraints and OOM scenarios
//! - Disk I/O failures during persistence
//! - Concurrent modification conflicts

use engram_core::{MemoryStore, EpisodeBuilder, Confidence};
use tokio::time::{timeout, Duration};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Chaos agent: Randomly injects failures
struct ChaosAgent {
    failure_rate: f64, // 0.0-1.0
    failure_count: AtomicU32,
}

impl ChaosAgent {
    fn new(failure_rate: f64) -> Self {
        Self {
            failure_rate,
            failure_count: AtomicU32::new(0),
        }
    }

    /// Inject failure with probability `failure_rate`
    fn maybe_inject_failure(&self) -> bool {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < self.failure_rate {
            self.failure_count.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}

/// Test: Random failures during concept formation
#[tokio::test]
async fn test_concept_formation_with_random_failures() {
    let store = MemoryStore::with_dual_memory().unwrap();
    let chaos = Arc::new(ChaosAgent::new(0.05)); // 5% failure rate

    // Add episodes
    for i in 0..500 {
        store.store(create_test_episode(i)).await.unwrap();
    }

    // Run consolidation with injected failures
    let mut successful_consolidations = 0;
    let mut failed_consolidations = 0;

    for run in 0..10 {
        if chaos.maybe_inject_failure() {
            // Simulate failure: kill consolidation mid-stream
            let result = timeout(
                Duration::from_millis(100),
                store.consolidate_with_chaos(&chaos),
            )
            .await;

            match result {
                Err(_) => failed_consolidations += 1, // Timeout
                Ok(Err(_)) => failed_consolidations += 1, // Consolidation error
                Ok(Ok(_)) => {
                    // Should not happen with chaos injection
                    println!("Warning: Consolidation succeeded despite chaos injection");
                }
            }
        } else {
            // Normal consolidation
            match store.consolidate().await {
                Ok(_) => successful_consolidations += 1,
                Err(e) => {
                    println!("Unexpected consolidation failure: {}", e);
                    failed_consolidations += 1;
                }
            }
        }
    }

    // System should remain stable despite failures
    let cue = create_test_cue();
    let recall_result = store.recall(&cue).await;
    assert!(recall_result.is_ok(), "Recall should work after chaos");

    // Verify data integrity: all episodes should still be retrievable
    let stats = store.get_stats();
    assert_eq!(stats.episode_count, 500, "No episodes should be lost");

    println!("Chaos test: {} successful, {} failed consolidations",
             successful_consolidations, failed_consolidations);
}

/// Test: Memory constraints trigger graceful degradation
#[tokio::test]
async fn test_oom_graceful_degradation() {
    // Create store with artificially low memory limit
    let mut config = Config::default();
    config.memory_limit_bytes = 50 * 1024 * 1024; // 50MB limit

    let store = MemoryStore::with_config(config).unwrap();

    let mut episodes_added = 0;
    let mut backpressure_triggered = false;

    // Add episodes until memory limit reached
    for i in 0..10_000 {
        match store.store(create_test_episode(i)).await {
            Ok(_) => episodes_added += 1,
            Err(e) if e.is_backpressure() => {
                backpressure_triggered = true;
                println!("Backpressure after {} episodes", episodes_added);
                break;
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    assert!(backpressure_triggered, "Backpressure should trigger before OOM");

    // System should still be responsive
    let cue = create_test_cue();
    let result = store.recall(&cue).await;
    assert!(result.is_ok(), "Recall should work despite backpressure");

    // Consolidation should free memory by creating concepts
    let pre_memory = store.get_memory_usage();
    store.consolidate().await.unwrap();
    let post_memory = store.get_memory_usage();

    println!("Memory: before={} MB, after={} MB",
             pre_memory / (1024 * 1024),
             post_memory / (1024 * 1024));

    // Should be able to add more episodes after consolidation
    let result = store.store(create_test_episode(episodes_added + 1)).await;
    assert!(result.is_ok(), "Should accept episodes after consolidation frees memory");
}

/// Test: Disk I/O failures during WAL writes
#[tokio::test]
async fn test_wal_write_failures() {
    use tempfile::tempdir;

    let tmp_dir = tempdir().unwrap();
    let wal_path = tmp_dir.path().join("wal");

    let store = MemoryStore::with_wal(&wal_path).unwrap();

    // Add episodes - writes should succeed initially
    for i in 0..100 {
        store.store(create_test_episode(i)).await.unwrap();
    }

    // Simulate disk full: make WAL directory read-only
    #[cfg(unix)]
    {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        let mut perms = fs::metadata(&wal_path).unwrap().permissions();
        perms.set_mode(0o444); // Read-only
        fs::set_permissions(&wal_path, perms).unwrap();
    }

    // Subsequent writes should fail gracefully
    let result = store.store(create_test_episode(100)).await;
    assert!(result.is_err(), "Write should fail when WAL is read-only");

    // System should remain queryable despite write failures
    let cue = create_test_cue();
    let recall_result = store.recall(&cue).await;
    assert!(recall_result.is_ok(), "Reads should work despite write failures");
}

/// Test: Concurrent modification conflicts during consolidation
#[tokio::test]
async fn test_concurrent_modification_conflicts() {
    let store = Arc::new(MemoryStore::with_dual_memory().unwrap());

    // Add initial episodes
    for i in 0..1000 {
        store.store(create_test_episode(i)).await.unwrap();
    }

    // Spawn concurrent tasks
    let mut handles = Vec::new();

    // Task 1: Run consolidation
    {
        let store = Arc::clone(&store);
        handles.push(tokio::spawn(async move {
            for _ in 0..5 {
                store.consolidate().await.unwrap();
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }));
    }

    // Task 2: Add new episodes during consolidation
    {
        let store = Arc::clone(&store);
        handles.push(tokio::spawn(async move {
            for i in 1000..1500 {
                store.store(create_test_episode(i)).await.unwrap();
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }));
    }

    // Task 3: Perform recalls during consolidation
    {
        let store = Arc::clone(&store);
        handles.push(tokio::spawn(async move {
            for i in 0..50 {
                let cue = create_random_cue(i);
                store.recall(&cue).await.unwrap();
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }));
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify data integrity
    let stats = store.get_stats();
    assert_eq!(stats.episode_count, 1500, "All episodes should be present");
    assert!(stats.concept_count > 0, "Concepts should be formed");

    // No corruption: all episodes should be recallable
    for i in 0..1500 {
        let result = store.get_episode(&format!("episode_{}", i)).await;
        assert!(result.is_ok(), "Episode {} should be retrievable", i);
    }
}
```

### 6. Backwards Compatibility Tests (`dual_memory_backwards_compat.rs`)

Validates that existing clients can interact with dual-memory-enabled systems.

```rust
//! Backwards compatibility validation
//!
//! Ensures that code written against the old single-type Memory API
//! continues to work correctly when dual memory is enabled.

use engram_core::{Memory, MemoryStore, Confidence, Episode, EpisodeBuilder};
use chrono::Utc;

/// Test: Old Memory API still works with dual memory backend
#[tokio::test]
async fn test_legacy_memory_api_compatibility() {
    let store = MemoryStore::with_dual_memory().unwrap();

    // Use old Memory construction (still supported for compatibility)
    let memory1 = Memory::new("legacy_001".to_string(), create_test_embedding(1), Confidence::HIGH);
    let memory2 = Memory::new("legacy_002".to_string(), create_test_embedding(2), Confidence::MEDIUM);

    // Store using legacy API
    store.store_memory(memory1.clone()).await.unwrap();
    store.store_memory(memory2.clone()).await.unwrap();

    // Retrieve using legacy API
    let retrieved = store.get_memory("legacy_001").await.unwrap();
    assert_eq!(retrieved.id, "legacy_001");
    assert_eq!(retrieved.confidence, Confidence::HIGH);

    // Old recall API should still work
    let cue = create_legacy_cue();
    let results = store.recall_legacy(&cue).await.unwrap();
    assert!(!results.is_empty());
}

/// Test: Episode API unchanged despite dual memory implementation
#[tokio::test]
async fn test_episode_api_unchanged() {
    let store_single = MemoryStore::new(1000);
    let store_dual = MemoryStore::with_dual_memory().unwrap();

    // Identical episode construction for both
    let episode = EpisodeBuilder::new()
        .id("test_123".to_string())
        .when(Utc::now())
        .what("Test episode".to_string())
        .embedding(create_test_embedding(1))
        .confidence(Confidence::HIGH)
        .build();

    // Both should accept same API
    store_single.store(episode.clone()).await.unwrap();
    store_dual.store(episode).await.unwrap();

    // Retrieval API identical
    let result_single = store_single.get_episode("test_123").await.unwrap();
    let result_dual = store_dual.get_episode("test_123").await.unwrap();

    assert_eq!(result_single.id, result_dual.id);
    assert_eq!(result_single.what, result_dual.what);
}

/// Test: Serialization format backwards compatible
#[tokio::test]
async fn test_serialization_backwards_compatibility() {
    use serde_json;

    // Serialize episode using old format
    let episode = create_test_episode(1);
    let json_old_format = serde_json::to_string(&episode).unwrap();

    // Dual memory store should deserialize old format
    let deserialized: Episode = serde_json::from_str(&json_old_format).unwrap();
    assert_eq!(deserialized.id, episode.id);

    // Serialize with dual memory enabled
    let store = MemoryStore::with_dual_memory().unwrap();
    store.store(episode.clone()).await.unwrap();

    let exported = store.export_episode("episode_1").await.unwrap();
    let json_new_format = serde_json::to_string(&exported).unwrap();

    // Old clients should be able to deserialize new format (gracefully ignore extra fields)
    let deserialized_new: Episode = serde_json::from_str(&json_new_format).unwrap();
    assert_eq!(deserialized_new.id, episode.id);
}

/// Property: Gradual rollout doesn't break existing functionality
#[tokio::test]
async fn test_gradual_rollout_compatibility() {
    // Start with dual memory disabled
    let mut config = Config::default();
    config.dual_memory.types_enabled = false;

    let store = MemoryStore::with_config(config.clone()).unwrap();

    // Old code: add episodes without knowing about dual memory
    for i in 0..100 {
        store.store(create_test_episode(i)).await.unwrap();
    }

    // System admin: enable dual memory
    config.dual_memory.types_enabled = true;
    store.update_config(config).unwrap();

    // Old code: continues to work without changes
    for i in 100..200 {
        store.store(create_test_episode(i)).await.unwrap();
    }

    let cue = create_test_cue();
    let results = store.recall(&cue).await.unwrap();

    // Should recall both old and new episodes
    let old_episode_found = results.episodes.iter().any(|e| e.id == "episode_50");
    let new_episode_found = results.episodes.iter().any(|e| e.id == "episode_150");

    assert!(old_episode_found, "Old episodes should still be recallable");
    assert!(new_episode_found, "New episodes should be recallable");
}
```

### 7. Test Data Specifications (`support/dual_memory_fixtures.rs`)

Defines realistic test data generation for integration tests.

```rust
//! Test data generation for dual memory integration tests
//!
//! Provides:
//! - Realistic episode distributions
//! - Known semantic clusters for validation
//! - Edge cases (very high fan-out, very low coherence)
//! - Deterministic seeded generation for reproducibility

use engram_core::{Episode, EpisodeBuilder, Confidence, Cue, CueBuilder};
use chrono::{Utc, Duration};
use rand::{SeedableRng, Rng, rngs::StdRng};

/// Create episode distribution with known cluster structure
///
/// Generates `num_clusters` semantic clusters, each containing `episodes_per_cluster` episodes.
/// Episodes within a cluster have similar embeddings (cosine similarity > 0.8).
pub fn create_clustered_episodes(
    num_clusters: usize,
    episodes_per_cluster: usize,
    seed: u64,
) -> Vec<Episode> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut episodes = Vec::new();

    for cluster_id in 0..num_clusters {
        // Generate cluster centroid
        let centroid = generate_normalized_embedding(&mut rng);

        for episode_id in 0..episodes_per_cluster {
            // Generate episode near centroid
            let embedding = perturb_embedding(&centroid, 0.1, &mut rng);

            let episode = EpisodeBuilder::new()
                .id(format!("cluster_{}_ep_{}", cluster_id, episode_id))
                .when(Utc::now() - Duration::hours(rng.gen_range(0..720)))
                .what(format!("Episode from cluster {}", cluster_id))
                .embedding(embedding)
                .confidence(Confidence::exact(rng.gen_range(0.7..0.95)))
                .build();

            episodes.push(episode);
        }
    }

    episodes
}

/// Generate edge case: Very high fan-out node (hub in semantic network)
///
/// Creates one central concept connected to many episodes.
/// Tests consolidation performance with large clusters.
pub fn create_high_fanout_scenario(fanout: usize, seed: u64) -> Vec<Episode> {
    let mut rng = StdRng::seed_from_u64(seed);

    let hub_embedding = generate_normalized_embedding(&mut rng);
    let mut episodes = Vec::new();

    for i in 0..fanout {
        let embedding = perturb_embedding(&hub_embedding, 0.05, &mut rng);

        let episode = EpisodeBuilder::new()
            .id(format!("fanout_ep_{}", i))
            .when(Utc::now() - Duration::hours(i as i64))
            .what(format!("Hub-connected episode {}", i))
            .embedding(embedding)
            .confidence(Confidence::HIGH)
            .build();

        episodes.push(episode);
    }

    episodes
}

/// Generate edge case: Very low coherence cluster (diffuse concept)
///
/// Episodes are loosely related (cosine similarity 0.3-0.5).
/// Tests concept formation thresholds.
pub fn create_low_coherence_cluster(size: usize, seed: u64) -> Vec<Episode> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centroid = generate_normalized_embedding(&mut rng);
    let mut episodes = Vec::new();

    for i in 0..size {
        // Large perturbation for low coherence
        let embedding = perturb_embedding(&centroid, 0.5, &mut rng);

        let episode = EpisodeBuilder::new()
            .id(format!("low_coherence_ep_{}", i))
            .when(Utc::now() - Duration::days(rng.gen_range(0..30)))
            .what(format!("Loosely related episode {}", i))
            .embedding(embedding)
            .confidence(Confidence::MEDIUM)
            .build();

        episodes.push(episode);
    }

    episodes
}

/// Generate test cues that sample semantic space uniformly
pub fn generate_test_cues(num_cues: usize, seed: u64) -> Vec<Cue> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cues = Vec::new();

    for i in 0..num_cues {
        let embedding = generate_normalized_embedding(&mut rng);

        let cue = CueBuilder::new()
            .id(format!("test_cue_{}", i))
            .embedding_search(embedding, Confidence::LOW)
            .max_results(10)
            .build();

        cues.push(cue);
    }

    cues
}

/// Helper: Generate normalized random embedding
fn generate_normalized_embedding(rng: &mut StdRng) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];

    for val in &mut embedding {
        *val = rng.gen_range(-1.0..1.0);
    }

    // Normalize to unit vector
    let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }

    embedding
}

/// Helper: Perturb embedding by adding noise
fn perturb_embedding(base: &[f32; 768], noise_scale: f32, rng: &mut StdRng) -> [f32; 768] {
    let mut result = *base;

    for val in &mut result {
        *val += rng.gen_range(-noise_scale..noise_scale);
    }

    // Re-normalize
    let norm = result.iter().map(|v| v * v).sum::<f32>().sqrt();
    for val in &mut result {
        *val /= norm;
    }

    result
}
```

## Test Execution Strategy

### CI Pipeline Integration

Tests are organized by execution time and resource requirements:

```bash
# Fast tests (< 1 second each) - run on every commit
cargo test --test dual_memory_migration_correctness
cargo test --test dual_memory_feature_flags
cargo test --test dual_memory_backwards_compat

# Medium tests (1-10 seconds) - run on PR
cargo test --test dual_memory_differential
cargo test --test dual_memory_integration

# Slow tests (10-60 seconds) - run on merge to dev
cargo test --test dual_memory_load
cargo test --test dual_memory_chaos

# Soak tests (24 hours) - run weekly
cargo run --bin dual_memory_soak_test -- --duration 86400
```

### Parallel Execution

Use `cargo-nextest` for optimal parallel test execution:

```bash
cargo nextest run --test-threads 8 --package engram-core \
  --filter-expr 'test(/dual_memory/)'
```

### Test Timing Constraints

- Unit/integration tests: max 5 seconds per test
- Load tests: max 60 seconds (use `#[ignore]` for longer)
- Property tests: 1000 cases by default (adjustable via `PROPTEST_CASES` env var)
- Soak tests: standalone binary, not part of `cargo test`

## Acceptance Criteria

- [ ] All 8 feature flag combinations tested with test-case matrix
- [ ] Migration preserves data integrity verified with cryptographic hashes
- [ ] Migration checkpoint resumption works after simulated crash
- [ ] Rollback restores exact pre-migration state (hash-verified)
- [ ] Differential testing shows ≥95% recall equivalence (single-type vs dual-type)
- [ ] Performance parity: dual-type ≤10% slower than single-type
- [ ] Statistical equivalence: KS test shows no distribution differences
- [ ] Load test sustains ≥5000 writes/sec with concurrent reads and consolidation
- [ ] Memory pressure triggers backpressure before OOM
- [ ] Chaos testing: system remains stable despite 5% random failure injection
- [ ] Concurrent modification conflicts resolved without data corruption
- [ ] Backwards compatibility: old Memory API works with dual-memory backend
- [ ] Serialization format backwards compatible (old clients can read new format)
- [ ] Gradual rollout doesn't break existing functionality
- [ ] Test coverage ≥80% for all dual memory modules
- [ ] All property tests pass 1000 cases without failures
- [ ] No memory leaks in 24-hour soak test (checked with valgrind/heaptrack)
- [ ] Zero clippy warnings in all test code

## Dependencies

- Task 001 (Dual Memory Types) - REQUIRED for type system
- Task 002 (Graph Storage Adaptation) - REQUIRED for storage layer
- Task 003 (Migration Utilities) - REQUIRED for migration testing
- Task 004 (Concept Formation) - REQUIRED for consolidation testing
- Task 005 (Binding Formation) - REQUIRED for episodic-semantic links
- Task 006 (Consolidation Integration) - REQUIRED for end-to-end tests
- Task 009 (Blended Recall) - REQUIRED for recall testing
- Task 013 (Validation Framework) - OPTIONAL, provides additional test infrastructure

## Estimated Time

5 days (revised from 3 days due to comprehensive scope)

- Day 1: Migration correctness tests + test data fixtures
- Day 2: Feature flag matrix testing + differential A/B tests
- Day 3: Load/stress testing + throughput benchmarks
- Day 4: Chaos engineering tests + backwards compatibility
- Day 5: CI integration, documentation, review and refinement

## Implementation Notes

### Following Existing Patterns

1. **Property-Based Testing**: Use proptest with 1000 cases (from confidence_property_tests.rs)
2. **Differential Testing**: Follow zig_differential/ pattern with EPSILON tolerances
3. **Load Testing**: Follow query_stress_tests.rs pattern with concurrent tasks
4. **Async Tests**: Use tokio::test pattern from error_recovery_integration.rs
5. **Test Fixtures**: Extend support/graph_builders.rs pattern for dual memory
6. **OOM Handling**: Follow oom_handling.rs graceful degradation pattern

### Test Dependencies (Cargo.toml)

Required dev-dependencies (already approved in chosen_libraries.md):

```toml
[dev-dependencies]
proptest = { workspace = true }
test-case = "3.3" # Feature flag matrix testing
approx = { workspace = true } # Floating-point comparison
sha2 = "0.10" # Cryptographic hashing for migration verification
statrs = "0.17" # Statistical tests (KS test)
tempfile = { workspace = true } # Temporary directories for chaos tests
```

### Test Organization

```
engram-core/tests/
├── dual_memory_migration_correctness.rs    (350 LOC)
├── dual_memory_feature_flags.rs            (300 LOC)
├── dual_memory_differential.rs             (400 LOC)
├── dual_memory_load.rs                     (450 LOC)
├── dual_memory_chaos.rs                    (400 LOC)
├── dual_memory_backwards_compat.rs         (250 LOC)
├── dual_memory_integration.rs              (200 LOC) - End-to-end scenarios
└── support/
    └── dual_memory_fixtures.rs             (300 LOC) - Test data generation

tools/
└── dual_memory_soak_test.rs               (150 LOC) - Standalone 24h test
```

Total: ~2800 LOC of comprehensive integration tests

### Performance Budget

- Migration speed: ≥10,000 nodes/second (offline mode)
- Migration memory overhead: <20% during checkpoint writes
- Recall equivalence: ≥95% top-K overlap (single vs dual)
- Performance regression: ≤10% slowdown for dual-type
- Throughput: ≥5000 writes/sec sustained under mixed workload
- Memory pressure detection: triggers at 80% of configured limit

### Test Metrics Tracking

Add Prometheus metrics for test observability:

- `engram_test_migration_duration_seconds`
- `engram_test_recall_equivalence_ratio`
- `engram_test_throughput_writes_per_sec`
- `engram_test_chaos_failure_injections_total`
- `engram_test_memory_pressure_triggers_total`

### Validation Against Academic Literature

Key properties validated against memory systems research:

1. **Consolidation preserves recall**: Verified via pre/post migration equivalence tests
2. **Semantic clustering emerges**: Verified via concept formation statistics
3. **Graceful degradation under load**: Verified via chaos engineering and OOM tests
4. **No false memories**: Verified via exact hash matching in migration tests
5. **Confidence calibration**: Verified via differential testing epsilon bounds

## References

### Codebase Test Patterns
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/confidence_property_tests.rs` - Property testing
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/zig_differential/spreading_activation.rs` - Differential testing
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/query_stress_tests.rs` - Load testing
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/oom_handling.rs` - Chaos/OOM testing
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/support/graph_builders.rs` - Test fixtures

### Migration Testing References
- Task 003 specification: Migration utilities with checkpoint/rollback
- SHA256 hashing for integrity verification (std::collections::hash or sha2 crate)
- Bincode serialization for checkpoint format

### Documentation
- `/Users/jordanwashburn/Workspace/orchard9/engram/chosen_libraries.md` - Approved dependencies
- Existing test suite patterns for consistency

## Risk Mitigation

### Flaky Tests

- Use deterministic seeded RNGs for all randomized tests
- Avoid wall-clock time dependencies (use tokio::time::pause in tests)
- Isolate test state (no shared global state between tests)
- Mark long-running tests with `#[ignore]` to prevent CI timeouts

### Test Maintenance

- Keep test data fixtures in separate module for reusability
- Document test invariants clearly in comments
- Use descriptive assertion messages for debugging failures
- Property test shrinking enabled for minimal counterexamples

### CI Resource Constraints

- Fast tests (<1s) run on every commit
- Medium tests (1-10s) run on PR only
- Heavy tests (>10s) run on merge to dev
- Soak tests run weekly, not in CI

This comprehensive integration testing strategy ensures the dual memory system is production-ready with systematic validation of correctness, performance, resilience, and backwards compatibility.