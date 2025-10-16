#![allow(missing_docs)]
mod support;

use engram_core::activation::test_support::{deterministic_config, run_spreading};
use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, TestCaseError, TestCaseResult, TestRunner};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use support::graph_builders::{GraphFixture, barabasi_albert, random_graph};

#[derive(Clone, Debug)]
enum GraphScenario {
    ErdosRenyi {
        node_count: usize,
        edge_probability: f32,
        seed: u64,
    },
    BarabasiAlbert {
        node_count: usize,
        attachment_count: usize,
        seed: u64,
    },
}

fn graph_scenarios() -> impl Strategy<Value = GraphScenario> {
    let er = (3usize..16, 2u32..9u32, any::<u64>()).prop_map(|(nodes, prob, seed)| {
        GraphScenario::ErdosRenyi {
            node_count: nodes,
            edge_probability: prob as f32 / 10.0,
            seed,
        }
    });

    let ba = (5usize..24, 1usize..5, any::<u64>()).prop_map(|(nodes, attachment, seed)| {
        let adjusted_nodes = nodes.max(attachment + 3);
        GraphScenario::BarabasiAlbert {
            node_count: adjusted_nodes,
            attachment_count: attachment,
            seed,
        }
    });

    prop_oneof![er, ba]
}

fn build_fixture(scenario: &GraphScenario) -> GraphFixture {
    match scenario {
        GraphScenario::ErdosRenyi {
            node_count,
            edge_probability,
            seed,
        } => random_graph(*node_count, *edge_probability, *seed),
        GraphScenario::BarabasiAlbert {
            node_count,
            attachment_count,
            seed,
        } => barabasi_albert(*node_count, *attachment_count, *seed),
    }
}

fn run_spreading_invariants(scenario: &GraphScenario) -> TestCaseResult {
    let fixture = build_fixture(scenario);
    let mut config = deterministic_config(match scenario {
        GraphScenario::ErdosRenyi { seed, .. } | GraphScenario::BarabasiAlbert { seed, .. } => {
            *seed
        }
    });
    fixture.apply_config_adjustments(&mut config);

    // Enable cycle detection to prevent seed nodes from being re-activated via cycles
    config.cycle_detection = true;
    config.tier_cycle_budgets = std::collections::HashMap::from([
        (engram_core::activation::storage_aware::StorageTier::Hot, 0),
        (engram_core::activation::storage_aware::StorageTier::Warm, 0),
        (engram_core::activation::storage_aware::StorageTier::Cold, 0),
    ]);

    let graph = fixture.graph();

    let run = run_spreading(graph, &fixture.seeds, config.clone())
        .map_err(|err| TestCaseError::fail(format!("spreading failed: {err}")))?;

    prop_assert!(!run.results.activations.is_empty());

    let mut total_activation = 0.0f32;
    for activation in &run.results.activations {
        let value = activation.activation_level.load(Ordering::Relaxed);
        prop_assert!(value >= -f32::EPSILON);
        prop_assert!(value <= 1.0 + 1e-3);

        let confidence = activation.confidence.load(Ordering::Relaxed);
        prop_assert!(confidence >= -f32::EPSILON);
        prop_assert!(confidence <= 1.0 + 1e-3);

        let hops = activation.hop_count.load(Ordering::Relaxed);
        prop_assert!(hops <= config.max_depth);

        total_activation += value;
    }
    prop_assert!(total_activation > 0.0);

    // Validate monotonic decay: activation should generally decrease with depth
    // Track max activation at each depth level (not per-node, since multiple paths can converge)
    let mut max_activation_by_depth: HashMap<u16, f32> = HashMap::new();
    for entry in &run.results.deterministic_trace {
        max_activation_by_depth
            .entry(entry.depth)
            .and_modify(|existing| {
                if entry.activation > *existing {
                    *existing = entry.activation;
                }
            })
            .or_insert(entry.activation);
    }

    // Check that max activation generally decreases with depth (with tolerance for convergence)
    if max_activation_by_depth.len() > 1 {
        let mut depths: Vec<_> = max_activation_by_depth.keys().copied().collect();
        depths.sort_unstable();

        for window in depths.windows(2) {
            let (depth_a, depth_b) = (window[0], window[1]);
            let (max_a, max_b) = (
                max_activation_by_depth[&depth_a],
                max_activation_by_depth[&depth_b],
            );

            // Allow some increase due to multi-path convergence and decay function shape
            // but activation shouldn't explode
            prop_assert!(
                max_b <= max_a * 2.0,
                "Activation increased too much with depth (depth {}: {:.4}, depth {}: {:.4})",
                depth_a,
                max_a,
                depth_b,
                max_b
            );
        }
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1_000,
        ..ProptestConfig::default()
    })]
    #[test]
    fn spreading_activation_invariants_hold(scenario in graph_scenarios()) {
        run_spreading_invariants(&scenario)?;
    }
}

#[test]
#[ignore = "Expensive: 10k+ cases, run with: cargo test --test spreading_property_tests spreading_activation_invariants_high_volume -- --ignored"]
fn spreading_activation_invariants_high_volume() {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(10_000);

    let mut runner = TestRunner::new(ProptestConfig {
        cases,
        ..ProptestConfig::default()
    });

    runner
        .run(&graph_scenarios(), |scenario| {
            run_spreading_invariants(&scenario)
        })
        .expect("high-volume spreading invariants to hold");
}
