#![allow(missing_docs)]
#![cfg_attr(not(feature = "long_running_tests"), allow(dead_code))]

mod support;

use engram_core::activation::test_support::{SpreadingRun, deterministic_config, run_spreading};
use engram_core::activation::{ActivationRecord, ParallelSpreadingConfig};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use support::graph_builders::{GraphFixture, barabasi_albert};

fn execute_fixture(fixture: &GraphFixture, mut config: ParallelSpreadingConfig) -> SpreadingRun {
    fixture.apply_config_adjustments(&mut config);
    run_spreading(fixture.graph(), &fixture.seeds, config).expect("spreading run")
}

fn max_pool_capacity(config: &ParallelSpreadingConfig) -> u64 {
    let record_size = std::mem::size_of::<ActivationRecord>().max(1);
    let max_bytes = config
        .pool_chunk_size
        .saturating_mul(config.pool_max_chunks.max(1));
    let computed_max = if max_bytes > 0 {
        (max_bytes / record_size).max(config.pool_initial_size)
    } else {
        config.pool_initial_size
    };
    computed_max.max(64) as u64
}

fn total_activation(run: &SpreadingRun) -> f32 {
    run.results
        .activations
        .iter()
        .map(|node| node.activation_level.load(Ordering::Relaxed))
        .sum()
}

#[test]
fn scale_free_graphs_complete_within_budget() {
    let scenarios = [(5_000, 3, 0.2_f32), (20_000, 4, 0.05_f32)];

    for (nodes, attachment, activation_floor) in scenarios {
        let fixture = barabasi_albert(nodes, attachment, 0xDEAD_BEEF);
        let mut config = deterministic_config(0xACE);
        config.num_threads = 2;
        config.max_depth = 5;
        config.threshold = 0.0;

        let run = execute_fixture(&fixture, config.clone());
        assert!(
            !run.results.activations.is_empty(),
            "Expected activations for {nodes} nodes"
        );
        assert_eq!(run.metrics.latency_budget_violations, 0);
        assert!(
            run.metrics.pool_high_water_mark <= max_pool_capacity(&config),
            "Pool exceeded configured capacity for {nodes} nodes"
        );

        let total = total_activation(&run);
        assert!(
            total >= activation_floor,
            "Activation mass below floor ({total} < {activation_floor}) for {nodes} nodes"
        );
    }
}

#[ignore]
#[test]
fn million_node_scale_free_soak() {
    let fixture = barabasi_albert(1_000_000, 6, 0xBEEF_CAFE);
    let mut config = deterministic_config(0xCAFE);
    config.num_threads = 4;
    config.max_depth = 4;
    config.threshold = 0.001;
    config.pool_chunk_size = 16_384;
    config.pool_max_chunks = 256;

    let run = execute_fixture(&fixture, config.clone());
    assert!(
        !run.results.activations.is_empty(),
        "One million node soak should produce activations"
    );
    assert_eq!(run.metrics.latency_budget_violations, 0);
    assert!(
        run.metrics.pool_high_water_mark <= max_pool_capacity(&config),
        "Pool blew past configured capacity during soak"
    );
}

#[cfg(feature = "memory_mapped_persistence")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_spreading_runs_share_pools_safely() {
    use futures::future::join_all;

    let fixture = barabasi_albert(8_000, 4, 0x0BAD_5EED);
    let mut base_config = deterministic_config(0xF00D);
    base_config.num_threads = 2;
    base_config.max_depth = 5;
    base_config.threshold = 0.0;
    let pool_capacity = max_pool_capacity(&base_config);

    let tasks = (0..8).map(|offset| {
        let graph = Arc::clone(fixture.graph());
        let seeds = fixture.seeds.clone();
        let mut config = base_config.clone();
        if let Some(seed) = config.seed.as_mut() {
            *seed = seed.saturating_add((offset as u64) + 1);
        }
        tokio::spawn(async move {
            run_spreading(&graph, &seeds, config)
                .expect("spreading run")
                .metrics
        })
    });

    let results = join_all(tasks).await;
    for metrics in results {
        let metrics = metrics.expect("task join");
        assert_eq!(metrics.latency_budget_violations, 0);
        assert!(
            metrics.pool_high_water_mark <= pool_capacity,
            "Concurrent run exceeded pool capacity"
        );
    }
}

#[cfg(loom)]
mod loom_models {
    use super::*;
    use engram_core::activation::memory_pool::ActivationRecordPool;
    use engram_core::activation::storage_aware::StorageTier;
    use loom::sync::Arc;
    use loom::thread;

    #[test]
    fn activation_pool_reclaims_records_across_interleavings() {
        loom::model(|| {
            let pool = Arc::new(ActivationRecordPool::with_config(2, 8, 4, 0.1));
            let pool_clone = Arc::clone(&pool);

            let handle = thread::spawn(move || {
                let record =
                    pool_clone.acquire("node-b".to_string(), 0.25, Some(StorageTier::Warm));
                pool_clone.release(record);
            });

            let record = pool.acquire("node-a".to_string(), 0.3, Some(StorageTier::Hot));
            pool.release(record);

            handle.join().expect("loom join");
            let stats = pool.stats();
            assert!(stats.total_created <= 8);
            assert_eq!(stats.release_failures, 0);
        });
    }
}
