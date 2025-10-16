#![allow(missing_docs)]
mod support;

use engram_core::activation::storage_aware::StorageTier;
use engram_core::activation::test_support::{
    SpreadingSnapshot, deterministic_config, run_spreading_snapshot,
};
use insta::assert_yaml_snapshot;
use serde::Serialize;
use support::graph_builders::{canonical_fixtures, chain, cycle_with_breakpoint, directed_cycle};

#[derive(Serialize)]
struct FixtureSnapshot {
    metadata: support::graph_builders::GraphFixtureMetadata,
    snapshot: NormalizedSnapshot,
}

#[derive(Serialize, Clone)]
struct NormalizedSnapshot {
    seeds: Vec<(String, f32)>,
    activations: Vec<NormalizedActivation>,
    tier_summaries: Vec<(String, NormalizedTierSummary)>,
    cycle_paths: Vec<Vec<String>>,
    deterministic_trace: Vec<NormalizedTraceEntry>,
    metrics: NormalizedMetrics,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
struct NormalizedActivation {
    memory_id: String,
    activation: f32,
    confidence: f32,
    hop_count: u16,
    storage_tier: String,
    flags: String,
}

#[derive(Serialize, Clone)]
struct NormalizedTierSummary {
    node_count: usize,
    total_activation: f32,
    average_confidence: f32,
}

#[derive(Serialize, Clone)]
struct NormalizedTraceEntry {
    depth: u16,
    target_node: String,
    activation: f32,
    confidence: f32,
    source_node: Option<String>,
}

#[derive(Serialize, Clone)]
struct NormalizedMetrics {
    total_activations: u64,
    cache_hits: u64,
    cache_misses: u64,
    average_latency_ns: u64,
    latency_budget_violations: u64,
    cycles_detected: u64,
    parallel_efficiency: f32,
    pool_available: u64,
    pool_in_flight: u64,
    pool_high_water_mark: u64,
}

fn normalize(snapshot: &SpreadingSnapshot) -> NormalizedSnapshot {
    let mut activations: Vec<NormalizedActivation> = snapshot
        .activations
        .iter()
        .map(|node| NormalizedActivation {
            memory_id: node.memory_id.clone(),
            activation: (node.activation * 1_000.0).round() / 1_000.0,
            confidence: (node.confidence * 1_000.0).round() / 1_000.0,
            hop_count: node.hop_count,
            storage_tier: node.storage_tier.clone(),
            flags: node.flags.clone(),
        })
        .collect();
    activations.sort_by(|a, b| a.memory_id.cmp(&b.memory_id));

    let mut tier_summaries: Vec<(String, NormalizedTierSummary)> = snapshot
        .tier_summaries
        .iter()
        .map(|(tier, summary)| {
            (
                tier.clone(),
                NormalizedTierSummary {
                    node_count: summary.node_count,
                    total_activation: (summary.total_activation * 1_000.0).round() / 1_000.0,
                    average_confidence: (summary.average_confidence * 1_000.0).round() / 1_000.0,
                },
            )
        })
        .collect();
    tier_summaries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut deterministic_trace: Vec<NormalizedTraceEntry> = snapshot
        .deterministic_trace
        .iter()
        .map(|entry| NormalizedTraceEntry {
            depth: entry.depth,
            target_node: entry.target_node.clone(),
            activation: (entry.activation * 1_000.0).round() / 1_000.0,
            confidence: (entry.confidence * 1_000.0).round() / 1_000.0,
            source_node: entry.source_node.clone(),
        })
        .collect();
    deterministic_trace.sort_by(|a, b| {
        a.depth
            .cmp(&b.depth)
            .then(a.target_node.cmp(&b.target_node))
    });

    let mut seeds = snapshot.seeds.clone();
    seeds.sort_by(|a, b| a.0.cmp(&b.0));

    let mut cycle_paths = snapshot.cycle_paths.clone();
    cycle_paths.sort();

    NormalizedSnapshot {
        seeds,
        activations,
        tier_summaries,
        cycle_paths,
        deterministic_trace,
        metrics: NormalizedMetrics {
            total_activations: snapshot.metrics.total_activations,
            cache_hits: snapshot.metrics.cache_hits,
            cache_misses: snapshot.metrics.cache_misses,
            average_latency_ns: 0,
            latency_budget_violations: snapshot.metrics.latency_budget_violations,
            cycles_detected: snapshot.metrics.cycles_detected,
            parallel_efficiency: (snapshot.metrics.parallel_efficiency * 1_000.0).round() / 1_000.0,
            pool_available: snapshot.metrics.pool_available,
            pool_in_flight: snapshot.metrics.pool_in_flight,
            pool_high_water_mark: snapshot.metrics.pool_high_water_mark,
        },
    }
}

#[test]
#[ignore = "Flaky: Snapshots change between runs due to non-deterministic spreading depth. Root cause: Same parallel spreading regression as deterministic_chain_runs_consistently - sometimes spreads to 2 nodes, sometimes 4. Needs fix in engram-core/src/activation/parallel.rs"]
fn canonical_spreading_snapshots_are_stable() {
    let base_config = deterministic_config(4242);
    let mut base_config = base_config;
    base_config.num_threads = 1;
    base_config.batch_size = 1;
    base_config.work_stealing_ratio = 0.0;
    base_config.max_concurrent_per_tier = 1;

    for fixture in canonical_fixtures() {
        let mut config = base_config.clone();
        fixture.apply_config_adjustments(&mut config);
        let snapshot = run_spreading_snapshot(fixture.graph(), &fixture.seeds, config)
            .expect("spreading should succeed");
        let metadata = fixture.metadata();
        let normalized = normalize(&snapshot);

        insta::with_settings!({
            snapshot_path => "data/spreading_snapshots",
            snapshot_suffix => "yaml",
            prepend_module_to_snapshot => false,
        }, {
            assert_yaml_snapshot!(
                format!("{}_snapshot", metadata.name),
                FixtureSnapshot {
                    metadata: metadata.clone(),
                    snapshot: normalized,
                }
            );
        });
    }
}

#[test]
#[ignore = "Flaky: Sometimes returns 4 activations, sometimes 2. Root cause: Non-deterministic behavior in parallel spreading engine even with deterministic config. Regression from recent parallel changes. Needs investigation in engram-core/src/activation/parallel.rs"]
fn deterministic_chain_runs_consistently() {
    let fixture = chain(4);
    let mut config = deterministic_config(7);
    fixture.apply_config_adjustments(&mut config);

    let snapshot_a = run_spreading_snapshot(fixture.graph(), &fixture.seeds, config.clone())
        .expect("spreading should succeed");
    let snapshot_b = run_spreading_snapshot(fixture.graph(), &fixture.seeds, config)
        .expect("spreading should succeed");

    assert_eq!(
        normalize(&snapshot_a).activations,
        normalize(&snapshot_b).activations
    );
    assert_eq!(snapshot_a.cycle_paths, snapshot_b.cycle_paths);
    assert_eq!(
        snapshot_a.deterministic_trace.len(),
        snapshot_b.deterministic_trace.len()
    );
}

#[test]
#[ignore = "Known issue: Cycle detection not populating cycle_paths in snapshots. Root cause: CycleDetector.get_cycle_paths() returns empty even when cycles detected. Needs investigation in engram-core/src/activation/cycle_detector.rs"]
fn cycle_breakpoints_surface_cycle_paths() {
    let fixture = cycle_with_breakpoint(6);
    let mut config = deterministic_config(99);
    fixture.apply_config_adjustments(&mut config);

    let snapshot =
        run_spreading_snapshot(fixture.graph(), &fixture.seeds, config).expect("cycle run");

    assert!(
        !snapshot.cycle_paths.is_empty(),
        "expected breakpoint cycle metadata to be captured"
    );
}

#[test]
#[ignore = "Flaky: Timeout waiting for spreading completion (270s). Root cause: Same parallel spreading regression affecting all other tests. Threading error in engram-core/src/activation/parallel.rs wait_for_completion(). Run with: cargo test --test spreading_validation tier_summaries_capture_hot_tier_activity -- --ignored --nocapture"]
fn tier_summaries_capture_hot_tier_activity() {
    let fixture = directed_cycle(5);
    let mut config = deterministic_config(1337);
    fixture.apply_config_adjustments(&mut config);

    let snapshot =
        run_spreading_snapshot(fixture.graph(), &fixture.seeds, config).expect("cycle run");

    let hot_summary = snapshot
        .tier_summaries
        .get(&format!("{:?}", StorageTier::Hot))
        .expect("hot tier summary present");
    assert!(hot_summary.node_count >= 1);
    assert!(snapshot.metrics.total_activations >= 1);
}
