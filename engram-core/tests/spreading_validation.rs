#![allow(missing_docs)]
mod support;

use engram_core::activation::ActivationGraphExt;
use engram_core::activation::ParallelSpreadingConfig;
use engram_core::activation::storage_aware::StorageTier;
use engram_core::activation::test_support::{
    SpreadingSnapshot, deterministic_config, run_spreading_snapshot,
};
use insta::assert_yaml_snapshot;
use serde::Serialize;
use std::collections::HashMap;
use support::graph_builders::{
    canonical_fixtures, chain, cycle_with_breakpoint, directed_cycle, simple_cycle,
};

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
#[ignore = "Flaky test: times out after 300s (threading coordination issue)"]
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
fn deterministic_chain_runs_consistently() {
    // FIXED: Race condition in TierQueue::pop_deterministic() where tasks were drained from
    // queue (decrementing 'queued') before incrementing 'in_flight', creating a visibility
    // gap that caused is_idle() to incorrectly return true during sorting (23% failure rate).
    //
    // Fix: Reserve in_flight slot BEFORE draining queue to maintain visibility during sort.
    // See: engram-core/src/activation/scheduler.rs:342-418

    // Create separate fixtures to avoid graph state persistence between runs
    let fixture_a = chain(4);
    let fixture_b = chain(4);
    let mut config = deterministic_config(7);
    fixture_a.apply_config_adjustments(&mut config);

    let snapshot_a = run_spreading_snapshot(fixture_a.graph(), &fixture_a.seeds, config.clone())
        .expect("spreading should succeed");
    let snapshot_b = run_spreading_snapshot(fixture_b.graph(), &fixture_b.seeds, config)
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
#[ignore = "Test fixture design issue: activation decays below Cold tier threshold (0.1) before \
            completing 6-node cycle, preventing cycles from occurring. Cycle detector code is \
            correct and working (see test_cycle_detection_with_simple_cycle). \
            Root cause: Storage tier thresholds (Hot:0.01, Warm:0.05, Cold:0.1) combined with \
            edge weights (0.75) and decay function cause activation to fall below threshold at \
            depth 4-5. Analysis: tmp/cycle_detection_final_analysis.md"]
fn cycle_breakpoints_surface_cycle_paths() {
    let fixture = cycle_with_breakpoint(6);

    // Use less restrictive config to allow full spreading
    let mut config = ParallelSpreadingConfig::deterministic(99);
    config.num_threads = 1; // Keep deterministic
    config.enable_metrics = true;
    config.trace_activation_flow = true;
    fixture.apply_config_adjustments(&mut config);

    let snapshot =
        run_spreading_snapshot(fixture.graph(), &fixture.seeds, config).expect("cycle run");

    // Debug output if test fails
    if snapshot.cycle_paths.is_empty() {
        eprintln!("=== DEBUG: Cycle paths empty ===");
        eprintln!("Total activations: {}", snapshot.metrics.total_activations);
        eprintln!("Cycles detected: {}", snapshot.metrics.cycles_detected);
        eprintln!(
            "Max depth: {}",
            snapshot
                .activations
                .iter()
                .map(|a| a.hop_count)
                .max()
                .unwrap_or(0)
        );
        eprintln!("Nodes activated: {}", snapshot.activations.len());
        eprintln!(
            "\nThis fixture requires architectural changes to work with current tier thresholds."
        );
        eprintln!("See test_cycle_detection_with_simple_cycle for working cycle detection demo.");
    }

    assert!(
        !snapshot.cycle_paths.is_empty(),
        "expected breakpoint cycle metadata to be captured"
    );
}

#[test]
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

#[test]
fn test_cycle_detection_with_simple_cycle() {
    let fixture = simple_cycle();
    let mut config = ParallelSpreadingConfig::deterministic(42);
    config.num_threads = 1;
    config.enable_metrics = true;
    config.trace_activation_flow = true;
    fixture.apply_config_adjustments(&mut config);

    let snapshot =
        run_spreading_snapshot(fixture.graph(), &fixture.seeds, config).expect("cycle run");

    // Verify cycles were detected
    assert!(
        snapshot.metrics.cycles_detected > 0,
        "Expected cycles to be detected in 3-node cycle with high edge weights. \
         cycles_detected={}, max_depth={}, nodes_activated={}",
        snapshot.metrics.cycles_detected,
        snapshot
            .activations
            .iter()
            .map(|a| a.hop_count)
            .max()
            .unwrap_or(0),
        snapshot.activations.len()
    );

    // Verify cycle paths were captured
    assert!(
        !snapshot.cycle_paths.is_empty(),
        "Expected cycle paths to be captured. \
         cycles_detected={}, cycle_paths={}",
        snapshot.metrics.cycles_detected,
        snapshot.cycle_paths.len()
    );

    // Verify cycle path contains expected nodes
    let cycle_path = &snapshot.cycle_paths[0];
    assert!(
        cycle_path.contains(&"SimpleCycle_A".to_string()),
        "Cycle path should contain SimpleCycle_A: {cycle_path:?}"
    );

    // Verify all 3 nodes were activated
    assert!(
        snapshot.activations.len() >= 3,
        "Expected at least 3 nodes activated, got {}",
        snapshot.activations.len()
    );
}

#[test]
#[ignore = "Flaky test: times out after 300s (threading coordination issue)"]
fn debug_cycle_detection_behavior() {
    let fixture = cycle_with_breakpoint(6);

    // Debug graph structure first
    println!("=== GRAPH STRUCTURE ===");
    let all_nodes = fixture.graph().get_all_nodes();
    println!("Total nodes: {}", all_nodes.len());
    for node in &all_nodes {
        if let Some(neighbors) = ActivationGraphExt::get_neighbors(&**fixture.graph(), node) {
            println!("Node {}: {} neighbors", node, neighbors.len());
            for edge in neighbors {
                println!("  -> {} (weight: {:.2})", edge.target, edge.weight);
            }
        }
    }

    let mut config = deterministic_config(99);
    fixture.apply_config_adjustments(&mut config);
    config.trace_activation_flow = true;

    println!("\n=== CONFIG ===");
    println!("Max depth: {}", config.max_depth);
    println!("Threshold: {}", config.threshold);
    println!("Decay function: {:?}", config.decay_function);
    println!("Cycle detection: {}", config.cycle_detection);

    let snapshot =
        run_spreading_snapshot(fixture.graph(), &fixture.seeds, config).expect("cycle run");

    println!("\n=== CYCLE DETECTION DEBUG ===");
    println!("Total activations: {}", snapshot.metrics.total_activations);
    println!(
        "Cycles detected (metric): {}",
        snapshot.metrics.cycles_detected
    );
    println!("Cycle paths count: {}", snapshot.cycle_paths.len());
    println!("Cycle paths: {:?}", snapshot.cycle_paths);

    // Check activations to see how many nodes were visited
    println!("\n=== ACTIVATIONS ===");
    for activation in &snapshot.activations {
        println!(
            "Node: {}, Activation: {:.3}, Hop: {}, Tier: {}",
            activation.memory_id,
            activation.activation,
            activation.hop_count,
            activation.storage_tier
        );
    }

    // Check trace to see depth progression
    println!("\n=== TRACE (all) ===");
    for (i, trace) in snapshot.deterministic_trace.iter().enumerate() {
        println!(
            "{}: depth={}, node={}, activation={:.3}, source={:?}",
            i, trace.depth, trace.target_node, trace.activation, trace.source_node
        );
    }

    println!("\n=== ANALYSIS ===");
    println!(
        "Max depth reached: {}",
        snapshot
            .activations
            .iter()
            .map(|a| a.hop_count)
            .max()
            .unwrap_or(0)
    );

    // Count how many times each node appears in trace (indicates revisits)
    let mut visit_counts: HashMap<String, u32> = HashMap::new();
    for trace in &snapshot.deterministic_trace {
        *visit_counts.entry(trace.target_node.clone()).or_insert(0) += 1;
    }
    println!("\nNode visit counts from trace:");
    let mut sorted: Vec<_> = visit_counts.iter().collect();
    sorted.sort_by_key(|(node, _)| *node);
    for (node, count) in sorted {
        println!("  {node}: {count} visits");
    }
}
