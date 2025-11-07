use engram_core::activation::{
    ActivationGraphExt, DecayFunction, EdgeType, MemoryGraph, ParallelSpreadingConfig,
    create_activation_graph, storage_aware::StorageTier, test_support::unique_test_id,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Metadata describing a graph fixture used in deterministic tests.
#[derive(Clone, Serialize)]
pub struct GraphFixture {
    pub name: &'static str,
    pub description: &'static str,
    pub seeds: Vec<(String, f32)>,
    #[serde(skip)]
    pub graph: Arc<MemoryGraph>,
    #[serde(skip)]
    pub config_adjuster: Option<Arc<dyn Fn(&mut ParallelSpreadingConfig) + Send + Sync>>,
}

impl GraphFixture {
    #[must_use]
    pub fn new(
        name: &'static str,
        description: &'static str,
        graph: Arc<MemoryGraph>,
        seeds: Vec<(String, f32)>,
    ) -> Self {
        Self {
            name,
            description,
            seeds,
            graph,
            config_adjuster: None,
        }
    }

    /// Convenience accessor used by tests.
    #[must_use]
    pub const fn graph(&self) -> &Arc<MemoryGraph> {
        &self.graph
    }

    /// Apply any fixture-specific configuration adjustments.
    pub fn apply_config_adjustments(&self, config: &mut ParallelSpreadingConfig) {
        if let Some(adjuster) = &self.config_adjuster {
            adjuster(config);
        }
    }

    /// Metadata summarising topology characteristics for snapshot assertions.
    #[must_use]
    pub fn metadata(&self) -> GraphFixtureMetadata {
        let mut node_ids = self.graph.get_all_nodes();
        node_ids.sort();

        let mut adjacency: BTreeMap<String, Vec<FixtureEdge>> = BTreeMap::new();
        for node in &node_ids {
            if let Some(neighbors) = self.graph.get_neighbors(node) {
                let mut edges: Vec<FixtureEdge> = neighbors
                    .into_iter()
                    .map(|edge| FixtureEdge {
                        target: edge.target,
                        weight: (edge.weight * 1_000.0).round() / 1_000.0,
                        edge_type: format!("{:?}", edge.edge_type),
                    })
                    .collect();
                edges.sort_by(|a, b| a.target.cmp(&b.target));
                adjacency.insert(node.clone(), edges);
            } else {
                adjacency.insert(node.clone(), Vec::new());
            }
        }

        let node_count = node_ids.len();
        let edge_count = adjacency.values().map(std::vec::Vec::len).sum::<usize>();

        GraphFixtureMetadata {
            name: self.name,
            description: self.description,
            seeds: self.seeds.clone(),
            node_count,
            edge_count,
            average_out_degree: if adjacency.is_empty() {
                0.0
            } else {
                adjacency.values().map(std::vec::Vec::len).sum::<usize>() as f32
                    / adjacency.len() as f32
            },
            adjacency,
        }
    }

    /// Attach a configuration adjustment closure used during deterministic runs.
    #[must_use]
    pub fn with_config_adjuster<F>(mut self, adjuster: F) -> Self
    where
        F: Fn(&mut ParallelSpreadingConfig) + Send + Sync + 'static,
    {
        self.config_adjuster = Some(Arc::new(adjuster));
        self
    }
}

/// Serializable metadata about a fixture's topology and cues.
#[derive(Clone, Serialize)]
pub struct GraphFixtureMetadata {
    pub name: &'static str,
    pub description: &'static str,
    pub seeds: Vec<(String, f32)>,
    pub node_count: usize,
    pub edge_count: usize,
    pub average_out_degree: f32,
    pub adjacency: BTreeMap<String, Vec<FixtureEdge>>,
}

/// Compact edge descriptor used when snapshotting fixture structure.
#[derive(Clone, Serialize)]
pub struct FixtureEdge {
    pub target: String,
    pub weight: f32,
    pub edge_type: String,
}

/// Chain graph: A -> B -> C -> ...
#[must_use]
pub fn chain(length: usize) -> GraphFixture {
    assert!(length >= 2, "Chain requires at least two nodes");
    let graph = Arc::new(create_activation_graph());

    let node_ids: Vec<String> = (0..length).map(|i| format!("Chain_{i}")).collect();

    for window in node_ids.windows(2) {
        let from = window[0].clone();
        let to = window[1].clone();
        ActivationGraphExt::add_edge(&*graph, from, to, 0.9, EdgeType::Excitatory);
    }

    GraphFixture::new(
        "chain",
        "Linear chain graph used for deterministic snapshotting",
        graph,
        vec![(node_ids[0].clone(), 1.0)],
    )
    .with_config_adjuster(move |config| {
        config.max_depth = length as u16;
    })
}

/// Directed cycle graph with a configurable length.
#[must_use]
pub fn directed_cycle(length: usize) -> GraphFixture {
    assert!(length >= 3, "Cycle requires at least three nodes");
    let graph = Arc::new(create_activation_graph());
    let node_ids: Vec<String> = (0..length).map(|i| format!("Cycle_{i}")).collect();

    for window in node_ids.windows(2) {
        ActivationGraphExt::add_edge(
            &*graph,
            window[0].clone(),
            window[1].clone(),
            0.85,
            EdgeType::Excitatory,
        );
    }

    ActivationGraphExt::add_edge(
        &*graph,
        node_ids[length - 1].clone(),
        node_ids[0].clone(),
        0.85,
        EdgeType::Excitatory,
    );

    GraphFixture::new(
        "cycle",
        "Directed cycle graph stressing cycle detection",
        graph,
        vec![(node_ids[0].clone(), 1.0)],
    )
    .with_config_adjuster(move |config| {
        config.max_depth = length as u16 + 2;
        config.cycle_detection = true;
    })
}

/// Simple 3-node cycle optimized for cycle detection testing.
///
/// Seeds from a separate node which activates a 3-node cycle A → B → C → A.
/// Uses high edge weights (0.99) and slow decay to ensure activation survives the cycle.
/// The cycle is triggered when node A is visited twice: once from Seed, and again from C.
///
/// Graph topology: Seed → A → B → C → A (cycle detected on second visit to A)
///
/// Expected behavior:
/// - Seed activates A (visit count = 1)
/// - A activates B, B activates C, C activates A (visit count = 2)
/// - When A is visited the second time, cycle detector triggers
///
/// This demonstrates that cycle detection works correctly when cycles actually occur.
#[must_use]
pub fn simple_cycle() -> GraphFixture {
    let graph = Arc::new(create_activation_graph());

    let seed_node = "SimpleCycle_Seed".to_string();
    let node_a = "SimpleCycle_A".to_string();
    let node_b = "SimpleCycle_B".to_string();
    let node_c = "SimpleCycle_C".to_string();

    // Seed triggers the cycle: Seed → A → B → C → A
    ActivationGraphExt::add_edge(
        &*graph,
        seed_node.clone(),
        node_a.clone(),
        0.99,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        node_a.clone(),
        node_b.clone(),
        0.99,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(&*graph, node_b, node_c.clone(), 0.99, EdgeType::Excitatory);
    ActivationGraphExt::add_edge(&*graph, node_c, node_a, 0.99, EdgeType::Excitatory);

    GraphFixture::new(
        "simple_cycle",
        "Seed node triggers 3-node cycle demonstrating cycle detection",
        graph,
        vec![(seed_node, 1.0)],
    )
    .with_config_adjuster(|config| {
        config.max_depth = 10;
        config.cycle_detection = true;
        // Use slower decay to ensure activation completes the full cycle path
        config.decay_function = DecayFunction::Exponential { rate: 0.2 };

        // Set cycle budget to 2 for all tiers to trigger detection on second visit
        config.tier_cycle_budgets = HashMap::from([
            (StorageTier::Hot, 2),
            (StorageTier::Warm, 2),
            (StorageTier::Cold, 2),
        ]);
    })
}

/// Fan graph used to validate the fan effect (one node connected to many spokes).
#[must_use]
pub fn fan(center: &str, spokes: usize) -> GraphFixture {
    assert!(spokes >= 2, "Fan graph requires at least two spokes");
    let graph = Arc::new(create_activation_graph());
    let test_id = unique_test_id();
    let center_id = format!("{test_id}_{center}");
    let weight = 1.0_f32 / spokes as f32;

    for idx in 0..spokes {
        let spoke = format!("{test_id}_{center}_spoke_{idx}");
        ActivationGraphExt::add_edge(
            &*graph,
            center_id.clone(),
            spoke.clone(),
            weight,
            EdgeType::Excitatory,
        );
    }

    GraphFixture::new(
        "fan",
        "Fan effect graph replicating Anderson (1974)",
        graph,
        vec![(center_id, 1.0)],
    )
    .with_config_adjuster(|config| {
        config.max_depth = 2;
    })
}

fn build_binary_tree_children(
    graph: &MemoryGraph,
    node: &str,
    current_depth: usize,
    max_depth: usize,
) {
    if current_depth >= max_depth {
        return;
    }

    for suffix in ["left", "right"] {
        let child = format!("{}_{}_{}", node, suffix, current_depth + 1);
        ActivationGraphExt::add_edge(
            graph,
            node.to_string(),
            child.clone(),
            0.8,
            EdgeType::Excitatory,
        );
        build_binary_tree_children(graph, &child, current_depth + 1, max_depth);
    }
}

/// Balanced binary tree of given depth (root depth = 0).
#[must_use]
pub fn binary_tree(depth: usize) -> GraphFixture {
    assert!(depth >= 1, "Binary tree requires depth >= 1");
    let graph = Arc::new(create_activation_graph());

    let root = "TreeRoot".to_string();
    build_binary_tree_children(&graph, &root, 0, depth);

    GraphFixture::new(
        "binary_tree",
        "Balanced binary tree fixture for tier coverage",
        graph,
        vec![(root, 1.0)],
    )
    .with_config_adjuster(move |config| {
        config.max_depth = depth as u16;
    })
}

/// Random graph generator used by property tests.
/// Uses an Erdős–Rényi model with the supplied probability.
#[must_use]
pub fn random_graph(node_count: usize, edge_probability: f32, seed: u64) -> GraphFixture {
    assert!(node_count >= 2, "Random graph needs at least two nodes");
    let graph = Arc::new(create_activation_graph());
    let mut rng = StdRng::seed_from_u64(seed);

    let nodes: Vec<String> = (0..node_count).map(|i| format!("Random_{i}")).collect();

    for (i, from) in nodes.iter().enumerate() {
        for (j, to) in nodes.iter().enumerate() {
            if i == j {
                continue;
            }

            let sample = rand::Rng::gen_range(&mut rng, 0.0_f32..1.0_f32);
            if sample <= edge_probability {
                ActivationGraphExt::add_edge(
                    &*graph,
                    from.clone(),
                    to.clone(),
                    0.5,
                    EdgeType::Excitatory,
                );
            }
        }
    }

    GraphFixture::new(
        "random",
        "Random Erdős–Rényi graph for property tests",
        graph,
        vec![(nodes[0].clone(), 1.0)],
    )
}

/// Complete graph (clique) used to validate high fan-in/out scenarios.
#[must_use]
pub fn clique(size: usize) -> GraphFixture {
    assert!(size >= 3, "Clique requires at least three nodes");
    let graph = Arc::new(create_activation_graph());
    let node_ids: Vec<String> = (0..size).map(|i| format!("Clique_{i}")).collect();

    for (i, from) in node_ids.iter().enumerate() {
        for (j, to) in node_ids.iter().enumerate() {
            if i == j {
                continue;
            }

            ActivationGraphExt::add_edge(
                &*graph,
                from.clone(),
                to.clone(),
                1.0 / (size.saturating_sub(1)) as f32,
                EdgeType::Excitatory,
            );
        }
    }

    GraphFixture::new(
        "clique",
        "Complete graph exploring uniform activation distribution",
        graph,
        vec![(node_ids[0].clone(), 1.0)],
    )
    .with_config_adjuster(move |config| {
        config.max_depth = 2;
    })
}

/// Cycle graph with an inhibitory breakpoint and escape hatch for deadlock detection.
#[must_use]
pub fn cycle_with_breakpoint(length: usize) -> GraphFixture {
    assert!(
        length >= 4,
        "Cycle with breakpoint requires at least four nodes"
    );
    let graph = Arc::new(create_activation_graph());
    let node_ids: Vec<String> = (0..length)
        .map(|i| format!("CycleBreakpoint_{i}"))
        .collect();

    for window in node_ids.windows(2) {
        ActivationGraphExt::add_edge(
            &*graph,
            window[0].clone(),
            window[1].clone(),
            0.75,
            EdgeType::Excitatory,
        );
    }

    // Close the loop with a slightly weaker link to encourage decay.
    ActivationGraphExt::add_edge(
        &*graph,
        node_ids[length - 1].clone(),
        node_ids[0].clone(),
        0.55,
        EdgeType::Excitatory,
    );

    // Introduce a breakpoint that siphons activation away via inhibitory and modulatory paths.
    let breakpoint = node_ids[length / 2].clone();
    let break_node = "CycleBreakpoint_break".to_string();
    let sink_node = "CycleBreakpoint_sink".to_string();
    ActivationGraphExt::add_edge(
        &*graph,
        breakpoint,
        break_node.clone(),
        0.6,
        EdgeType::Inhibitory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        break_node,
        sink_node.clone(),
        0.9,
        EdgeType::Modulatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        sink_node,
        node_ids[0].clone(),
        0.2,
        EdgeType::Inhibitory,
    );

    GraphFixture::new(
        "cycle_breakpoint",
        "Cycle with explicit breakpoint and sink to validate decay controls",
        graph,
        vec![(node_ids[0].clone(), 1.0)],
    )
    .with_config_adjuster(move |config| {
        config.max_depth = length as u16 + 2;
        config.cycle_detection = true;
        // Use power-law decay for slower falloff - allows activation to survive
        // the full 6-node cycle and return to seed, triggering cycle detection
        // PowerLaw with exponent 0.3 gives: (1+depth)^-0.3
        //   depth 3: 0.66, depth 4: 0.61, depth 5: 0.57 (all >> 0.1 Cold threshold)
        config.decay_function = DecayFunction::PowerLaw { exponent: 0.3 };
    })
}

/// Convenience accessor returning the canonical deterministic fixture set.
#[must_use]
pub fn canonical_fixtures() -> Vec<GraphFixture> {
    vec![
        chain(5),
        directed_cycle(5),
        cycle_with_breakpoint(6),
        binary_tree(3),
        fan("fan_center", 4),
        clique(4),
    ]
}

/// Barabási–Albert preferential attachment graph producing scale-free degree distributions.
#[must_use]
pub fn barabasi_albert(node_count: usize, attachment_count: usize, seed: u64) -> GraphFixture {
    assert!(
        attachment_count >= 1,
        "Attachment count must be at least one"
    );
    assert!(
        node_count > attachment_count + 1,
        "Node count must exceed attachment count"
    );

    let graph = Arc::new(create_activation_graph());
    let mut rng = StdRng::seed_from_u64(seed);

    let mut nodes: Vec<String> = (0..=attachment_count).map(|i| format!("BA_{i}")).collect();

    for i in 0..nodes.len() {
        for j in 0..nodes.len() {
            if i == j {
                continue;
            }
            ActivationGraphExt::add_edge(
                &*graph,
                nodes[i].clone(),
                nodes[j].clone(),
                1.0 / attachment_count.max(1) as f32,
                EdgeType::Excitatory,
            );
        }
    }

    let mut degrees: Vec<usize> = vec![attachment_count * 2; nodes.len()];

    while nodes.len() < node_count {
        let new_index = nodes.len();
        let new_node = format!("BA_{new_index}");
        nodes.push(new_node.clone());
        degrees.push(0);

        for _ in 0..attachment_count {
            let total_degree: usize = degrees.iter().sum();
            let target_idx = if total_degree == 0 {
                rng.gen_range(0..new_index)
            } else {
                let mut dart = rng.gen_range(0..total_degree);
                let mut chosen = 0;
                for (idx, degree) in degrees.iter().enumerate() {
                    if idx == new_index {
                        continue;
                    }
                    if dart < *degree {
                        chosen = idx;
                        break;
                    }
                    dart -= degree;
                }
                chosen
            };

            if target_idx == new_index {
                continue;
            }

            let weight = 1.0 / attachment_count.max(1) as f32;
            ActivationGraphExt::add_edge(
                &*graph,
                new_node.clone(),
                nodes[target_idx].clone(),
                weight,
                EdgeType::Excitatory,
            );
            ActivationGraphExt::add_edge(
                &*graph,
                nodes[target_idx].clone(),
                new_node.clone(),
                weight,
                EdgeType::Excitatory,
            );

            degrees[target_idx] = degrees[target_idx].saturating_add(1);
            degrees[new_index] = degrees[new_index].saturating_add(1);
        }
    }

    GraphFixture::new(
        "barabasi_albert",
        "Scale-free graph generated via Barabási–Albert preferential attachment",
        graph,
        vec![(nodes[0].clone(), 1.0)],
    )
}
