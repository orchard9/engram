#![allow(missing_docs)]
mod support;

use engram_core::activation::ActivationGraphExt;
use engram_core::activation::test_support::{deterministic_config, run_spreading};
use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, TestCaseError, TestCaseResult, TestRunner};
use std::collections::HashMap;
use std::sync::Arc;
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

fn compute_edge_weights(
    graph: &Arc<engram_core::activation::MemoryGraph>,
) -> HashMap<(String, String), f32> {
    let mut weights = HashMap::new();
    let nodes = ActivationGraphExt::get_all_nodes(&**graph);
    for node in nodes {
        if let Some(neighbors) = ActivationGraphExt::get_neighbors(&**graph, &node) {
            for edge in neighbors {
                weights.insert((node.clone(), edge.target.clone()), edge.weight);
            }
        }
    }
    weights
}

fn run_spreading_invariants(scenario: &GraphScenario) -> TestCaseResult {
    let fixture = build_fixture(scenario);
    let mut config = deterministic_config(match scenario {
        GraphScenario::ErdosRenyi { seed, .. } | GraphScenario::BarabasiAlbert { seed, .. } => {
            *seed
        }
    });
    fixture.apply_config_adjustments(&mut config);

    let graph = fixture.graph();
    let edge_weights = compute_edge_weights(graph);

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

    // Track the strongest activation observed per node depth to validate monotonic decay.
    let mut depth_activation: HashMap<(String, u16), f32> = HashMap::new();
    for entry in &run.results.deterministic_trace {
        let key = (entry.target_node.clone(), entry.depth);
        depth_activation
            .entry(key)
            .and_modify(|existing| {
                if entry.activation > *existing {
                    *existing = entry.activation;
                }
            })
            .or_insert(entry.activation);

        if let Some(source) = &entry.source_node {
            if entry.depth > 0 {
                if let Some(parent_activation) =
                    depth_activation.get(&(source.clone(), entry.depth - 1))
                {
                    if let Some(weight) =
                        edge_weights.get(&(source.clone(), entry.target_node.clone()))
                    {
                        let expected = parent_activation * weight;
                        let tolerance = expected.abs() * 0.1 + 1e-3;
                        prop_assert!(
                            entry.activation <= expected + tolerance,
                            "Activation increased beyond decay bounds (source: {}, depth: {}, activation: {:.4}, weight: {:.4}, target: {}, depth: {}, observed: {:.4})",
                            source,
                            entry.depth - 1,
                            parent_activation,
                            weight,
                            entry.target_node,
                            entry.depth,
                            entry.activation
                        );
                    }
                }
            }
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
    #[ignore] // TODO: Fix activation decay bounds violation in random graphs
    fn spreading_activation_invariants_hold(scenario in graph_scenarios()) {
        run_spreading_invariants(&scenario)?;
    }
}

#[test]
#[ignore]
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
