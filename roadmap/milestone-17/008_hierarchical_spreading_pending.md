# Task 008: Hierarchical Spreading with Optimized Path Tracking

## Objective
Implement cache-optimal hierarchical spreading patterns that flow upward (episode to concept) and downward (concept to episode) with efficient path tracking for explainability and cycle detection, leveraging arena-allocated path storage and lock-free traversal algorithms.

## Background
Hierarchical spreading enables generalization (upward) and instantiation (downward) in dual memory networks. Path tracking provides explainability for activation flow while enabling sophisticated cycle detection across hierarchical boundaries. The challenge is to implement this without excessive allocations or cache misses during high-throughput spreading.

Based on existing spreading architecture:
- `ActivationTask` already has `path: Vec<NodeId>` (line 412, mod.rs) for cycle detection
- `ParallelSpreadingEngine` uses work-stealing deques for lock-free task distribution
- Storage tiers (Hot/Warm/Cold) provide depth-based heuristics for memory placement
- SIMD batch spreading processes 8 neighbors concurrently with embeddings
- Cycle detection uses per-tier budgets: Hot=2, Warm=3, Cold=4 hops

## Requirements
1. Design asymmetric spreading strengths for upward/downward/lateral directions
2. Implement multi-hop spreading through concept hierarchies with decay
3. Add memory-bounded path tracking for explainability (limit path depth)
4. Optimize common patterns: Episode→Concept (low fan-out), Concept→Episode (high fan-out)
5. Use binary heap priority queue for breadth-first traversal order
6. Integrate SIMD batch processing where applicable (8-neighbor batches)
7. Minimize path cloning overhead through arena allocation or copy-on-write
8. Support hierarchical cycle detection across type boundaries

## Technical Specification

### Files to Create

#### `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/activation/hierarchical.rs`
Hierarchical spreading engine with optimized path tracking:

```rust
//! Hierarchical activation spreading with path tracking for dual memory architecture
//!
//! This module implements efficient breadth-first spreading across episode-concept
//! hierarchies with minimal allocation overhead through arena-based path storage.

use crate::activation::{
    ActivationGraphExt, ActivationRecord, ActivationTask, DecayFunction, EdgeType, MemoryGraph,
    NodeId, SpreadingMetrics, WeightedEdge, storage_aware::StorageTier,
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
    pub fn from_node_types(source_is_episode: bool, target_is_episode: bool) -> Self {
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
    fn new(
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
    pub upward: usize,
    pub downward: usize,
    pub lateral: usize,
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
    pub fn new(config: HierarchicalSpreadingConfig, memory_graph: Arc<MemoryGraph>) -> Self {
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
    /// spread before weak ones. Paths are tracked using Arc<Vec<NodeId>> to minimize
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
            if let Some(neighbors) = self.memory_graph.get_neighbors(&current.node_id) {
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
        let deep = SpreadNode::new("node4".to_string(), 0.5, 3, path.clone(), tier);

        // Shallower depth should come first when activation equal
        assert!(shallow > deep);
    }

    #[test]
    fn test_hierarchical_spreading_basic() {
        let graph = Arc::new(create_activation_graph());

        // Create simple hierarchy: episode1 → concept1 → episode2
        graph.add_edge(
            "episode1".to_string(),
            "concept1".to_string(),
            0.8,
            EdgeType::Excitatory,
        );
        graph.add_edge(
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
            ..Default::default()
        };
        let graph = Arc::new(create_activation_graph());

        // Create deep chain: a → b → c → d → e → f
        for i in 0..5 {
            let source = format!("episode{i}");
            let target = format!("episode{}", i + 1);
            graph.add_edge(source, target, 0.9, EdgeType::Excitatory);
        }

        let spreading = HierarchicalSpreading::new(config, graph);
        let seeds = vec![("episode0".to_string(), 1.0)];
        let result = spreading.spread_hierarchical(seeds);

        // Verify final node reached
        assert!(result.activations.contains_key("episode5"));

        // Verify path length bounded
        if let Some(paths) = result.paths {
            if let Some(final_path) = paths.get("episode5") {
                assert!(final_path.len() <= 4, "Path length should be truncated");
            }
        }
    }
}
```

### Files to Modify

#### `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/activation/mod.rs`
Add module export (after line 77):
```rust
/// Hierarchical spreading with path tracking for dual memory architecture
pub mod hierarchical;

pub use hierarchical::{
    DirectionStats, HierarchicalSpreading, HierarchicalSpreadingConfig,
    HierarchicalSpreadResult, SpreadingDirection,
};
```

#### `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/activation/parallel.rs`
Integration with parallel spreading engine (add method to `WorkerContext`):
```rust
/// Apply hierarchical spreading to neighbors based on node types
fn apply_hierarchical_spreading(
    &self,
    source_node: &NodeId,
    neighbors: Vec<WeightedEdge>,
    hierarchical_config: &HierarchicalSpreadingConfig,
) -> Vec<WeightedEdge> {
    // Determine source node type (temporary heuristic, replaced in Task 002)
    let source_is_episode = source_node.contains("episode");

    neighbors
        .into_iter()
        .map(|mut edge| {
            // Determine target type and direction
            let target_is_episode = edge.target.contains("episode");
            let direction = SpreadingDirection::from_node_types(
                source_is_episode,
                target_is_episode,
            );

            // Apply directional strength adjustment
            let strength_multiplier = direction.base_strength(hierarchical_config);
            edge.weight *= strength_multiplier;

            edge
        })
        .collect()
}
```

## Implementation Notes

### Cache-Optimal Priority Queue
- `BinaryHeap<SpreadNode>` provides O(log n) insertion and O(1) max lookup
- `SpreadNode` sized at 48 bytes (3 cache lines) for efficient SIMD prefetching
- Priority ordering ensures determinism: activation → depth → node_id

### Path Tracking with Arc Copy-on-Write
- Paths stored as `Arc<Vec<NodeId>>` to avoid expensive clones
- Extending path: clone the Vec (not Arc), append, wrap in new Arc
- Truncation kicks in at `max_path_length` to bound memory (default: 32 nodes)
- Memory cost: ~16 bytes per Arc + Vec overhead + NodeId storage
- For 10,000 activated nodes with avg path length 8: ~1.5MB total

### Common Pattern Optimizations

#### Episode → Concept (Upward, Low Fan-out)
- Episodes typically have 1-5 concept bindings
- Strong activation (0.8 base) ensures concept activation
- SIMD batch spreading applies when ≥8 neighbors (rare for episodes)
- Scalar path with prefetching for typical 1-3 neighbors

#### Concept → Episode (Downward, High Fan-out)
- Concepts may have 50-500 episode bindings
- Lower activation (0.6 base) models retrieval difficulty
- SIMD batch spreading for neighbors ≥8 (common case)
- Priority queue ensures strongest episodes spread first

#### Concept → Concept (Lateral, Medium Fan-out)
- Semantic associations typically 5-20 neighbors
- Moderate activation (0.4 base) enables multi-hop reasoning
- SIMD batch spreading for 8+ neighbors
- Enables inference through concept graphs

### Memory Budget for Deep Hierarchies
- Max depth: 6 hops (configurable, default prevents runaway spreading)
- Max path length: 32 nodes (truncated to prevent unbounded growth)
- Visited set: DashMap for lock-free concurrent access
- Estimated peak memory: 10MB for 100,000 nodes with paths

### Integration with Existing Systems
- `ActivationTask::path` already exists, reuse for cycle detection
- Storage tiers (`StorageTier::from_depth`) provide activation thresholds
- Metrics integration: record direction counts and depth distribution
- Deterministic spreading via priority queue ordering

## Testing Approach

### Unit Tests (in hierarchical.rs)
```rust
#[test] fn test_spreading_direction_classification()
#[test] fn test_spread_node_ordering()
#[test] fn test_hierarchical_spreading_basic()
#[test] fn test_upward_stronger_than_downward()
#[test] fn test_depth_decay_application()
#[test] fn test_max_depth_cutoff()
#[test] fn test_max_path_length_truncation()
#[test] fn test_tier_threshold_filtering()
```

### Multi-Level Hierarchy Tests
```rust
#[test]
fn test_three_level_hierarchy() {
    // episode1 → concept1 → concept2 → episode2
    // Verify activation diminishes with depth
    // Verify paths tracked correctly
}

#[test]
fn test_diamond_hierarchy() {
    // episode1 → concept1, concept2 → episode2
    // Verify convergence and max activation selection
}

#[test]
fn test_fan_out_spreading() {
    // concept1 → [episode1..episode100]
    // Verify SIMD batch processing invoked
    // Verify priority queue ordering
}
```

### Path Tracking Validation
```rust
#[test]
fn test_shortest_path_selection() {
    // Multiple paths to same node
    // Verify shortest path retained
}

#[test]
fn test_path_truncation_at_max_length() {
    // Deep chain exceeding max_path_length
    // Verify truncation occurs correctly
}
```

### Performance Benchmarks
```rust
#[bench]
fn bench_hierarchical_spreading_small(b: &mut Bencher) {
    // 100 nodes, 3 levels, avg fan-out 5
    // Target: <100µs per spread operation
}

#[bench]
fn bench_hierarchical_spreading_large(b: &mut Bencher) {
    // 10,000 nodes, 6 levels, avg fan-out 20
    // Target: <5ms per spread operation
}

#[bench]
fn bench_path_tracking_overhead(b: &mut Bencher) {
    // Compare enabled vs disabled path tracking
    // Target: <15% overhead with paths enabled
}

#[bench]
fn bench_priority_queue_scaling(b: &mut Bencher) {
    // Measure heap operations with varying frontier sizes
    // Target: <50ns per insertion, <20ns per pop
}
```

### Integration Tests
```rust
#[test]
fn test_hierarchical_with_parallel_engine() {
    // Run hierarchical spreading through ParallelSpreadingEngine
    // Verify results match standalone HierarchicalSpreading
}

#[test]
fn test_hierarchical_with_cycle_detection() {
    // episode1 → concept1 → episode1 (cycle)
    // Verify cycle detector catches cross-type cycles
}

#[test]
fn test_hierarchical_with_storage_tiers() {
    // Deep hierarchy: Hot → Warm → Cold
    // Verify tier-specific thresholds applied correctly
}
```

## Acceptance Criteria

- [ ] Asymmetric spreading strengths configurable (upward, downward, lateral)
  - Default values: 0.8, 0.6, 0.4 respectively
  - Strength applied correctly based on node type detection

- [ ] Multi-hop spreading through concept hierarchies works correctly
  - Verify depth decay applied: strength *= 0.8^depth
  - Max depth enforced (default: 6 hops)

- [ ] Path tracking provides full provenance when enabled
  - Paths stored as `Arc<Vec<NodeId>>` for efficiency
  - Shortest path retained when multiple paths converge
  - Path length bounded at max_path_length (default: 32)

- [ ] Priority queue ensures breadth-first + strongest-first order
  - BinaryHeap ordering: activation → depth → node_id
  - Deterministic spreading order maintained

- [ ] Performance scales sub-linearly with depth
  - 100 nodes, 3 levels: <100µs
  - 10,000 nodes, 6 levels: <5ms
  - Path tracking overhead: <15% when enabled

- [ ] SIMD batch processing invoked for high fan-out (≥8 neighbors)
  - Concept → Episode spreading uses batch path
  - Embeddings loaded for similarity-weighted spreading

- [ ] Memory budget enforced for deep hierarchies
  - Max depth prevents runaway spreading
  - Path truncation prevents unbounded growth
  - Estimated peak: 10MB for 100,000 nodes with paths

- [ ] Integration with existing parallel spreading engine
  - `apply_hierarchical_spreading` method added to WorkerContext
  - Compatible with cycle detection and storage tiers

- [ ] Direction statistics tracked for observability
  - Counts for upward, downward, lateral, episodic
  - Depth distribution recorded per level

## Dependencies
- Task 007 (Fan Effect Spreading) - for fan-out penalty integration
- Task 005 (Binding Formation) - for episode-concept binding queries
- Task 001 (Dual Memory Types) - for MemoryNodeType discrimination (currently using heuristic)

## Estimated Time
3 days
- Day 1: Core hierarchical spreading engine and path tracking
- Day 2: Priority queue optimization and SIMD integration
- Day 3: Testing, benchmarking, and parallel engine integration

## Performance Budget
- Small hierarchy (100 nodes, 3 levels): <100µs per spread
- Large hierarchy (10,000 nodes, 6 levels): <5ms per spread
- Path tracking overhead: <15% when enabled
- Priority queue operations: <50ns insert, <20ns pop
- Memory overhead: ~10MB for 100k nodes with paths (16 bytes Arc + Vec + NodeIds)
- SIMD batch threshold: 8 neighbors (from existing config)

## Follow-up Tasks
- Task 009: Integrate with node type discrimination from Task 001/002
- Task 010: Add fan effect penalty for high fan-out concepts
- Task 011: Optimize path storage with arena allocator for zero-copy paths

## References
- Anderson, J. R. (1983). A spreading activation theory of memory. Journal of Verbal Learning and Verbal Behavior, 22(3), 261-295.
- Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. Psychological Review, 82(6), 407-428.
- Existing path tracking: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/activation/mod.rs` line 412
- Existing parallel spreading: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/activation/parallel.rs`
- Storage tier design: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/activation/storage_aware.rs`
