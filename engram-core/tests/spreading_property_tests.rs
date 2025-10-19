#![allow(missing_docs)]
mod support;

use engram_core::activation::test_support::{deterministic_config, run_spreading};
use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, TestCaseError, TestCaseResult, TestRunner};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Duration;
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
    config.max_depth = config.max_depth.max(8);

    // Enable cycle detection to prevent seed nodes from being re-activated via cycles
    config.cycle_detection = true;
    config.tier_cycle_budgets = HashMap::from([
        (engram_core::activation::storage_aware::StorageTier::Hot, 2),
        (engram_core::activation::storage_aware::StorageTier::Warm, 2),
        (engram_core::activation::storage_aware::StorageTier::Cold, 2),
    ]);
    config.tier_timeouts = [Duration::from_secs(5); 3];
    config.phase_sync_interval = Duration::from_millis(5);

    // Set longer completion timeout for complex graph topologies in property tests
    // Property tests generate random graphs that may require more time than the default
    // timeout (which is based on core count and doesn't account for graph complexity)
    config.completion_timeout = Some(Duration::from_secs(120));

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

        let _hops = activation.hop_count.load(Ordering::Relaxed);
        total_activation += value;
    }
    prop_assert!(total_activation > 0.0);

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 300,
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
