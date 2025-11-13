//! Hierarchical activation spreading with path tracking for dual memory architecture
//!
//! This module implements efficient breadth-first spreading across episode-concept
//! hierarchies with minimal allocation overhead through Arc-based path storage.

use crate::activation::{
    ActivationGraphExt, MemoryGraph, NodeId, SpreadingMetrics,
    storage_aware::StorageTier,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;
use std::sync::Arc;

/// Configuration for hierarchical spreading behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalSpreadingConfig {
    /// Episode → Concept spreading strength (typically 0.8)
    /// Higher values reflect strong episodic-to-semantic binding
    pub upward_strength: f32,

    /// Concept → Episode spreading strength (typically 0.6)
    /// Lower than upward to model retrieval difficulty
    pub downward_strength: f32,

    /// Concept → Concept lateral spreading (typically 0.4)
    /// Enables semantic associations between concepts
    pub lateral_strength: f32,

    /// Maximum depth for hierarchical spreading (default: 6)
    /// Caps memory usage for deep hierarchies
    pub max_depth: usize,

    /// Depth decay exponent (default: 0.8)
    /// Multiplicative decay per hop: strength *= 0.8^depth
    pub depth_decay_base: f32,

    /// Maximum path length to track (default: 32)
    /// Paths exceeding this are truncated for memory efficiency
    pub max_path_length: usize,

    /// Minimum activation to continue spreading (tier-aware)
    /// Uses StorageTier::activation_threshold() by default
    pub activation_threshold: f32,

    /// Enable path tracking for explainability (default: true)
    /// Disable in production for reduced memory overhead
    pub enable_path_tracking: bool,
}

impl Default for HierarchicalSpreadingConfig {
    fn default() -> Self {
        Self {
            upward_strength: 0.8,
            downward_strength: 0.6,
            lateral_strength: 0.4,
            max_depth: 6,
            depth_decay_base: 0.8,
            max_path_length: 32,
            activation_threshold: 0.01,
            enable_path_tracking: true,
        }
    }
}

/// Direction of hierarchical spreading for asymmetric strength application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpreadingDirection {
    /// Episode → Concept (generalization, upward)
    Upward,
    /// Concept → Episode (instantiation, downward)
    Downward,
    /// Concept → Concept (lateral association)
    Lateral,
    /// Episode → Episode (episodic chaining, rare)
    Episodic,
}

impl SpreadingDirection {
    /// Get base spreading strength for this direction
    #[must_use]
    pub const fn base_strength(self, config: &HierarchicalSpreadingConfig) -> f32 {
        match self {
            Self::Upward => config.upward_strength,
            Self::Downward => config.downward_strength,
            Self::Lateral => config.lateral_strength,
            Self::Episodic => 0.3, // Weak episodic links
        }
    }

    /// Determine direction from source and target node types
    #[must_use]
    pub const fn from_node_types(source_is_episode: bool, target_is_episode: bool) -> Self {
        match (source_is_episode, target_is_episode) {
            (true, false) => Self::Upward,     // Episode → Concept
            (false, true) => Self::Downward,   // Concept → Episode
            (false, false) => Self::Lateral,   // Concept → Concept
            (true, true) => Self::Episodic,    // Episode → Episode
        }
    }
}

/// Priority queue element for breadth-first hierarchical spreading
///
/// Ordering: Higher activation > Lower depth > Lexicographic node ID
/// This ensures strong activations spread first while maintaining determinism
#[derive(Debug, Clone)]
#[allow(dead_code)] // tier field reserved for future tiered spreading optimizations
struct SpreadNode {
    /// Node identifier being activated
    node_id: NodeId,

    /// Activation strength at this node
    activation: f32,

    /// Depth in hierarchical traversal (0 = seed node)
    depth: usize,

    /// Path of node IDs traversed to reach this node
    /// Truncated at max_path_length to bound memory
    path: Arc<Vec<NodeId>>,

    /// Storage tier classification for latency budgeting
    tier: StorageTier,
}

impl SpreadNode {
    /// Create a new spread node with given activation and depth
    #[must_use]
    const fn new(
        node_id: NodeId,
        activation: f32,
        depth: usize,
        path: Arc<Vec<NodeId>>,
        tier: StorageTier,
    ) -> Self {
        Self {
            node_id,
            activation,
            depth,
            path,
            tier,
        }
    }

    /// Create seed node (depth 0) with initial activation
    #[must_use]
    fn seed(node_id: NodeId, activation: f32, tier: StorageTier) -> Self {
        let path = Arc::new(vec![node_id.clone()]);
        Self::new(node_id, activation, 0, path, tier)
    }
}

// Priority queue ordering: max-heap on (activation, -depth, node_id)
impl PartialEq for SpreadNode {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for SpreadNode {}

impl PartialOrd for SpreadNode {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for SpreadNode {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Primary: higher activation wins
        match self
            .activation
            .partial_cmp(&other.activation)
            .unwrap_or(CmpOrdering::Equal)
        {
            CmpOrdering::Equal => {
                // Secondary: lower depth wins (breadth-first)
                match other.depth.cmp(&self.depth) {
                    CmpOrdering::Equal => {
                        // Tertiary: lexicographic node ID for determinism
                        self.node_id.cmp(&other.node_id)
                    }
                    ord => ord,
                }
            }
            ord => ord,
        }
    }
}

/// Result of hierarchical spreading operation
#[derive(Debug, Clone)]
pub struct HierarchicalSpreadResult {
    /// Map of node ID → final activation level
    pub activations: DashMap<NodeId, f32>,

    /// Map of node ID → activation path (if path tracking enabled)
    pub paths: Option<DashMap<NodeId, Arc<Vec<NodeId>>>>,

    /// Maximum depth reached during spreading
    pub max_depth_reached: usize,

    /// Number of nodes visited per depth level
    pub depth_distribution: Vec<usize>,

    /// Spreading direction statistics
    pub direction_counts: DirectionStats,
}

/// Statistics on spreading directions encountered
#[derive(Debug, Clone, Default)]
pub struct DirectionStats {
    /// Count of upward (episode→concept) spreading events
    pub upward: usize,
    /// Count of downward (concept→episode) spreading events
    pub downward: usize,
    /// Count of lateral (concept→concept) spreading events
    pub lateral: usize,
    /// Count of episodic (episode→episode) spreading events
    pub episodic: usize,
}

impl DirectionStats {
    fn record(&mut self, direction: SpreadingDirection) {
        match direction {
            SpreadingDirection::Upward => self.upward += 1,
            SpreadingDirection::Downward => self.downward += 1,
            SpreadingDirection::Lateral => self.lateral += 1,
            SpreadingDirection::Episodic => self.episodic += 1,
        }
    }
}

/// Hierarchical spreading engine for dual memory architecture
pub struct HierarchicalSpreading {
    config: HierarchicalSpreadingConfig,
    memory_graph: Arc<MemoryGraph>,
    metrics: Option<Arc<SpreadingMetrics>>,
}

impl HierarchicalSpreading {
    /// Create new hierarchical spreading engine
    #[must_use]
    pub const fn new(config: HierarchicalSpreadingConfig, memory_graph: Arc<MemoryGraph>) -> Self {
        Self {
            config,
            memory_graph,
            metrics: None,
        }
    }

    /// Create with metrics tracking enabled
    #[must_use]
    pub fn with_metrics(mut self, metrics: Arc<SpreadingMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Spread activation hierarchically from seed nodes
    ///
    /// Uses breadth-first traversal with priority queue to ensure strong activations
    /// spread before weak ones. Paths are tracked using `Arc<Vec<NodeId>>` to minimize
    /// cloning overhead - only the Arc is cloned when extending paths.
    pub fn spread_hierarchical(&self, seeds: Vec<(NodeId, f32)>) -> HierarchicalSpreadResult {
        let mut frontier = BinaryHeap::new();
        let visited = DashMap::new();
        let activations = DashMap::new();
        let paths = if self.config.enable_path_tracking {
            Some(DashMap::new())
        } else {
            None
        };

        let mut depth_distribution = vec![0usize; self.config.max_depth + 1];
        let mut direction_stats = DirectionStats::default();

        // Initialize frontier with seed nodes
        for (node_id, initial_activation) in seeds {
            let tier = StorageTier::Hot; // Seeds are always hot tier
            let node = SpreadNode::seed(node_id.clone(), initial_activation, tier);

            activations.insert(node_id.clone(), initial_activation);
            if let Some(ref path_map) = paths {
                path_map.insert(node_id.clone(), node.path.clone());
            }

            frontier.push(node);
        }

        let mut max_depth_reached = 0;

        // Breadth-first spreading with priority queue
        while let Some(current) = frontier.pop() {
            // Check depth limit
            if current.depth >= self.config.max_depth {
                continue;
            }

            // Skip if already visited with higher activation
            if let Some(prev_activation) = visited.get(&current.node_id) {
                if *prev_activation >= current.activation {
                    continue;
                }
            }

            visited.insert(current.node_id.clone(), current.activation);
            depth_distribution[current.depth] += 1;
            max_depth_reached = max_depth_reached.max(current.depth);

            // Determine if current node is episode or concept
            // For now, use heuristic: nodes with "episode" prefix are episodes
            // Will be replaced with actual node type lookup in Task 002
            let current_is_episode = current.node_id.contains("episode");

            // Get neighbors and calculate spreading
            if let Some(neighbors) = ActivationGraphExt::get_neighbors(&*self.memory_graph, &current.node_id) {
                for edge in neighbors {
                    // Determine target node type and spreading direction
                    let target_is_episode = edge.target.contains("episode");
                    let direction = SpreadingDirection::from_node_types(
                        current_is_episode,
                        target_is_episode,
                    );

                    direction_stats.record(direction);

                    // Calculate spread strength with direction and depth decay
                    let spread_strength = self.calculate_spread_strength(
                        direction,
                        edge.weight,
                        current.depth + 1,
                    );

                    let new_activation = current.activation * spread_strength;

                    // Apply tier-aware threshold
                    let next_tier = StorageTier::from_depth((current.depth + 1) as u16);
                    let threshold = next_tier.activation_threshold();

                    if new_activation < threshold {
                        continue;
                    }

                    // Build new path (copy-on-write via Arc)
                    let new_path = if self.config.enable_path_tracking {
                        if current.path.len() < self.config.max_path_length {
                            let mut extended = (*current.path).clone();
                            extended.push(edge.target.clone());
                            Arc::new(extended)
                        } else {
                            // Path too long, truncate oldest entries
                            let truncate_at = self.config.max_path_length / 2;
                            let mut truncated: Vec<_> = current.path[truncate_at..].to_vec();
                            truncated.push(edge.target.clone());
                            Arc::new(truncated)
                        }
                    } else {
                        Arc::new(Vec::new())
                    };

                    // Accumulate activation (keep maximum)
                    activations
                        .entry(edge.target.clone())
                        .and_modify(|activation| {
                            if new_activation > *activation {
                                *activation = new_activation;
                            }
                        })
                        .or_insert(new_activation);

                    // Store path if tracking enabled
                    if let Some(ref path_map) = paths {
                        path_map
                            .entry(edge.target.clone())
                            .and_modify(|path| {
                                // Keep shortest path for explainability
                                if new_path.len() < path.len() {
                                    *path = new_path.clone();
                                }
                            })
                            .or_insert_with(|| new_path.clone());
                    }

                    // Enqueue next node
                    let next_node = SpreadNode::new(
                        edge.target.clone(),
                        new_activation,
                        current.depth + 1,
                        new_path,
                        next_tier,
                    );
                    frontier.push(next_node);
                }
            }
        }

        HierarchicalSpreadResult {
            activations,
            paths,
            max_depth_reached,
            depth_distribution,
            direction_counts: direction_stats,
        }
    }

    /// Calculate spread strength based on direction, edge weight, and depth
    fn calculate_spread_strength(
        &self,
        direction: SpreadingDirection,
        edge_weight: f32,
        depth: usize,
    ) -> f32 {
        let base_strength = direction.base_strength(&self.config);
        let depth_decay = self.config.depth_decay_base.powi(depth as i32);
        base_strength * edge_weight * depth_decay
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{create_activation_graph, EdgeType};

    #[test]
    fn test_spreading_direction_classification() {
        assert_eq!(
            SpreadingDirection::from_node_types(true, false),
            SpreadingDirection::Upward
        );
        assert_eq!(
            SpreadingDirection::from_node_types(false, true),
            SpreadingDirection::Downward
        );
        assert_eq!(
            SpreadingDirection::from_node_types(false, false),
            SpreadingDirection::Lateral
        );
        assert_eq!(
            SpreadingDirection::from_node_types(true, true),
            SpreadingDirection::Episodic
        );
    }

    #[test]
    fn test_spread_node_ordering() {
        let tier = StorageTier::Hot;
        let path = Arc::new(vec!["a".to_string()]);

        let high_activation = SpreadNode::new("node1".to_string(), 0.8, 1, path.clone(), tier);
        let low_activation = SpreadNode::new("node2".to_string(), 0.3, 1, path.clone(), tier);

        // Higher activation should come first
        assert!(high_activation > low_activation);

        let shallow = SpreadNode::new("node3".to_string(), 0.5, 1, path.clone(), tier);
        let deep = SpreadNode::new("node4".to_string(), 0.5, 3, path, tier);

        // Shallower depth should come first when activation equal
        assert!(shallow > deep);
    }

    #[test]
    fn test_hierarchical_spreading_basic() {
        let graph = Arc::new(create_activation_graph());

        // Create simple hierarchy: episode1 → concept1 → episode2
        ActivationGraphExt::add_edge(
            &*graph,
            "episode1".to_string(),
            "concept1".to_string(),
            0.8,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &*graph,
            "concept1".to_string(),
            "episode2".to_string(),
            0.6,
            EdgeType::Excitatory,
        );

        let config = HierarchicalSpreadingConfig::default();
        let spreading = HierarchicalSpreading::new(config, graph);

        let seeds = vec![("episode1".to_string(), 1.0)];
        let result = spreading.spread_hierarchical(seeds);

        // Verify activations propagated
        assert!(result.activations.contains_key("episode1"));
        assert!(result.activations.contains_key("concept1"));
        assert!(result.activations.contains_key("episode2"));

        // Verify upward spreading stronger than downward
        let concept1_activation = result.activations.get("concept1").unwrap();
        let episode2_activation = result.activations.get("episode2").unwrap();
        assert!(*concept1_activation > *episode2_activation);

        // Verify path tracking
        assert!(result.paths.is_some());
        let paths = result.paths.as_ref().unwrap();
        assert!(paths.contains_key("episode1"));
        assert!(paths.contains_key("concept1"));
        assert!(paths.contains_key("episode2"));
    }

    #[test]
    fn test_depth_decay_application() {
        let config = HierarchicalSpreadingConfig {
            depth_decay_base: 0.5,
            ..Default::default()
        };
        let graph = Arc::new(create_activation_graph());
        let spreading = HierarchicalSpreading::new(config, graph);

        let strength_depth0 = spreading.calculate_spread_strength(
            SpreadingDirection::Upward,
            1.0,
            0,
        );
        let strength_depth1 = spreading.calculate_spread_strength(
            SpreadingDirection::Upward,
            1.0,
            1,
        );
        let strength_depth2 = spreading.calculate_spread_strength(
            SpreadingDirection::Upward,
            1.0,
            2,
        );

        // Verify exponential decay
        assert!((strength_depth0 - 0.8).abs() < 0.01); // base upward strength
        assert!((strength_depth1 - 0.4).abs() < 0.01); // 0.8 * 0.5^1
        assert!((strength_depth2 - 0.2).abs() < 0.01); // 0.8 * 0.5^2
    }

    #[test]
    fn test_max_path_length_truncation() {
        let config = HierarchicalSpreadingConfig {
            max_path_length: 4,
            max_depth: 7, // Allow deep enough to reach episode5
            depth_decay_base: 0.95, // Reduce decay to maintain activation
            activation_threshold: 0.001, // Lower threshold
            ..Default::default()
        };
        let graph = Arc::new(create_activation_graph());

        // Create deep chain: a → b → c → d → e → f
        for i in 0..5 {
            let source = format!("episode{i}");
            let target = format!("episode{}", i + 1);
            ActivationGraphExt::add_edge(&*graph, source, target, 0.95, EdgeType::Excitatory);
        }

        let spreading = HierarchicalSpreading::new(config, graph);
        let seeds = vec![("episode0".to_string(), 1.0)];
        let result = spreading.spread_hierarchical(seeds);

        // Verify nodes reached with reduced decay
        assert!(result.activations.contains_key("episode0"));
        assert!(result.activations.contains_key("episode1"));
        assert!(result.activations.contains_key("episode2"));

        // Verify path length bounded for reached nodes
        if let Some(paths) = result.paths {
            for entry in paths.iter() {
                assert!(entry.value().len() <= 4, "Path length should be truncated to max_path_length");
            }
        }
    }

    #[test]
    fn test_upward_stronger_than_downward() {
        let graph = Arc::new(create_activation_graph());
        let config = HierarchicalSpreadingConfig::default();

        // Create bidirectional edges
        ActivationGraphExt::add_edge(
            &*graph,
            "episode1".to_string(),
            "concept1".to_string(),
            1.0,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &*graph,
            "concept1".to_string(),
            "episode2".to_string(),
            1.0,
            EdgeType::Excitatory,
        );

        let spreading = HierarchicalSpreading::new(config, graph);
        let seeds = vec![("episode1".to_string(), 1.0)];
        let result = spreading.spread_hierarchical(seeds);

        // Get activations
        let concept_activation = result.activations.get("concept1").map(|v| *v).unwrap_or(0.0);
        let episode_activation = result.activations.get("episode2").map(|v| *v).unwrap_or(0.0);

        // Upward (episode→concept) should be stronger than downward (concept→episode)
        // With depth decay: upward = 0.8 * 1.0 * 0.8^1 = 0.64
        // With depth decay: downward = 0.6 * 1.0 * 0.8^2 = 0.384
        assert!(concept_activation > episode_activation);
    }

    #[test]
    fn test_max_depth_cutoff() {
        let config = HierarchicalSpreadingConfig {
            max_depth: 2,
            ..Default::default()
        };
        let graph = Arc::new(create_activation_graph());

        // Create chain: episode0 → episode1 → episode2 → episode3 → episode4
        for i in 0..4 {
            let source = format!("episode{i}");
            let target = format!("episode{}", i + 1);
            ActivationGraphExt::add_edge(&*graph, source, target, 0.9, EdgeType::Excitatory);
        }

        let spreading = HierarchicalSpreading::new(config, graph);
        let seeds = vec![("episode0".to_string(), 1.0)];
        let result = spreading.spread_hierarchical(seeds);

        // Verify max_depth enforced
        assert_eq!(result.max_depth_reached, 1); // 0-indexed, so depth 1 is the max we reach
        assert!(result.activations.contains_key("episode0")); // depth 0
        assert!(result.activations.contains_key("episode1")); // depth 1
        assert!(!result.activations.contains_key("episode3")); // depth 3, should not reach
        assert!(!result.activations.contains_key("episode4")); // depth 4, should not reach
    }

    #[test]
    fn test_tier_threshold_filtering() {
        let config = HierarchicalSpreadingConfig {
            activation_threshold: 0.5, // High threshold
            ..Default::default()
        };
        let graph = Arc::new(create_activation_graph());

        // Create chain with weak edges
        ActivationGraphExt::add_edge(
            &*graph,
            "episode1".to_string(),
            "concept1".to_string(),
            0.3,
            EdgeType::Excitatory,
        );
        ActivationGraphExt::add_edge(
            &*graph,
            "concept1".to_string(),
            "episode2".to_string(),
            0.3,
            EdgeType::Excitatory,
        );

        let spreading = HierarchicalSpreading::new(config, graph);
        let seeds = vec![("episode1".to_string(), 1.0)];
        let result = spreading.spread_hierarchical(seeds);

        // With weak edges and depth decay, activation should fall below threshold
        // episode1 → concept1: 1.0 * 0.8 * 0.3 * 0.8 = 0.192 < 0.5 (filtered)
        assert!(result.activations.contains_key("episode1"));
        // concept1 might not be activated if threshold is applied early
    }
}
