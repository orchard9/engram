//! Scaffolding for upcoming comprehensive spreading validation.
//!
//! Provides deterministic configuration helpers and lightweight graph builders
//! that future snapshot tests and property-based checks can reuse.

use engram_core::activation::{
    ActivationGraphExt, EdgeType, MemoryGraph, ParallelSpreadingConfig, create_activation_graph,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Deterministic config used when generating golden activation traces.
#[must_use]
pub fn deterministic_spreading_config() -> ParallelSpreadingConfig {
    let mut config = ParallelSpreadingConfig::deterministic(42);
    config.enable_metrics = true;
    config.trace_activation_flow = true;
    config
}

/// Build a small triangle graph with seeded embeddings for snapshot generation.
#[must_use]
pub fn triangle_fixture() -> Arc<MemoryGraph> {
    let graph = Arc::new(create_activation_graph());

    let embeddings: HashMap<_, _> = [
        ("A".to_string(), [0.1_f32; 768]),
        ("B".to_string(), [0.2_f32; 768]),
        ("C".to_string(), [0.3_f32; 768]),
    ]
    .into_iter()
    .collect();

    for (node, embedding) in &embeddings {
        ActivationGraphExt::set_embedding(&*graph, node, embedding);
    }

    ActivationGraphExt::add_edge(
        &*graph,
        "A".to_string(),
        "B".to_string(),
        0.9,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        "B".to_string(),
        "C".to_string(),
        0.8,
        EdgeType::Excitatory,
    );
    ActivationGraphExt::add_edge(
        &*graph,
        "C".to_string(),
        "A".to_string(),
        0.7,
        EdgeType::Excitatory,
    );

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn deterministic_config_enables_tracing() {
        let config = deterministic_spreading_config();
        assert!(config.deterministic);
        assert_eq!(config.seed, Some(42));
        assert!(config.trace_activation_flow);
    }

    #[test]
    fn triangle_fixture_populates_embeddings() {
        let graph = triangle_fixture();
        let expected = HashMap::from([
            ("A".to_string(), 0.1_f32),
            ("B".to_string(), 0.2_f32),
            ("C".to_string(), 0.3_f32),
        ]);

        for (node, first_component) in expected {
            let embedding = graph.get_embedding(&node).expect("embedding initialized");
            assert!((embedding[0] - first_component).abs() < f32::EPSILON * 2.0);
        }
    }
}
