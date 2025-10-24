# Task 003: Network Partition Handling and Recovery

**Status**: Pending
**Estimated Duration**: 3 days
**Dependencies**: Task 001 (SWIM membership), Task 002 (Discovery)
**Owner**: TBD

## Objective

Implement graceful degradation during network partitions, local-only recall when partitioned, partition healing mechanisms, and split-brain prevention. This task ensures Engram remains available during network failures while maintaining correctness guarantees.

## Technical Specification

### Partition Detection

Network partitions manifest as:
1. SWIM failure detection marks nodes as suspect/dead
2. Inability to reach majority of cluster
3. Gossip protocol stops converging

Detection mechanism:

```rust
// engram-core/src/cluster/partition.rs

use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct PartitionDetector {
    membership: Arc<SwimMembership>,
    partition_state: Arc<RwLock<PartitionState>>,
    config: PartitionConfig,
}

#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Threshold for considering cluster partitioned (default: 0.5)
    /// If we can't reach >50% of nodes, we're partitioned
    pub majority_threshold: f64,

    /// How long to wait before declaring partition (default: 10s)
    pub detection_window: Duration,

    /// How often to check for partition (default: 5s)
    pub check_interval: Duration,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            majority_threshold: 0.5,
            detection_window: Duration::from_secs(10),
            check_interval: Duration::from_secs(5),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PartitionState {
    /// Connected to majority of cluster
    Connected {
        reachable_nodes: usize,
        total_nodes: usize,
    },

    /// Partitioned from majority
    Partitioned {
        reachable_nodes: usize,
        total_nodes: usize,
        partitioned_since: Instant,
    },

    /// Healing from partition
    Healing {
        newly_reachable: HashSet<String>,
        healing_since: Instant,
    },
}

impl PartitionDetector {
    pub fn new(
        membership: Arc<SwimMembership>,
        config: PartitionConfig,
    ) -> Self {
        Self {
            membership,
            partition_state: Arc::new(RwLock::new(PartitionState::Connected {
                reachable_nodes: 0,
                total_nodes: 0,
            })),
            config,
        }
    }

    pub async fn start_monitoring(&self) {
        let mut interval = tokio::time::interval(self.config.check_interval);

        loop {
            interval.tick().await;
            self.check_partition_status().await;
        }
    }

    async fn check_partition_status(&self) {
        let stats = self.compute_reachability();

        let reachability_ratio = stats.reachable_nodes as f64 / stats.total_nodes as f64;
        let is_partitioned = reachability_ratio < self.config.majority_threshold;

        let mut state = self.partition_state.write().await;

        match &*state {
            PartitionState::Connected { .. } => {
                if is_partitioned {
                    warn!(
                        "Network partition detected: {}/{} nodes reachable",
                        stats.reachable_nodes, stats.total_nodes
                    );

                    *state = PartitionState::Partitioned {
                        reachable_nodes: stats.reachable_nodes,
                        total_nodes: stats.total_nodes,
                        partitioned_since: Instant::now(),
                    };

                    // Trigger partition handlers
                    self.on_partition_detected().await;
                }
            },

            PartitionState::Partitioned { partitioned_since, .. } => {
                if !is_partitioned {
                    info!(
                        "Network partition healing: {}/{} nodes reachable",
                        stats.reachable_nodes, stats.total_nodes
                    );

                    *state = PartitionState::Healing {
                        newly_reachable: stats.newly_reachable,
                        healing_since: Instant::now(),
                    };

                    // Trigger healing handlers
                    self.on_partition_healing().await;
                } else if partitioned_since.elapsed() > Duration::from_secs(300) {
                    error!(
                        "Network partition persisting for {} seconds",
                        partitioned_since.elapsed().as_secs()
                    );
                }
            },

            PartitionState::Healing { healing_since, .. } => {
                if is_partitioned {
                    // Partition recurred
                    *state = PartitionState::Partitioned {
                        reachable_nodes: stats.reachable_nodes,
                        total_nodes: stats.total_nodes,
                        partitioned_since: Instant::now(),
                    };
                } else if healing_since.elapsed() > self.config.detection_window {
                    // Fully healed
                    info!("Network partition fully healed");

                    *state = PartitionState::Connected {
                        reachable_nodes: stats.reachable_nodes,
                        total_nodes: stats.total_nodes,
                    };

                    self.on_partition_healed().await;
                }
            },
        }
    }

    fn compute_reachability(&self) -> ReachabilityStats {
        let mut total = 0;
        let mut reachable = 0;
        let mut newly_reachable = HashSet::new();

        for entry in self.membership.members.iter() {
            let node = entry.value();
            total += 1;

            if node.state == NodeState::Alive {
                reachable += 1;
                newly_reachable.insert(node.id.clone());
            }
        }

        ReachabilityStats {
            total_nodes: total,
            reachable_nodes: reachable,
            newly_reachable,
        }
    }

    async fn on_partition_detected(&self) {
        // Enter read-only mode for non-local writes
        // Continue serving local reads with confidence penalty
        info!("Entering partition mode: local-only operations");

        // Increment partition counter metric
        metrics::counter!("engram.cluster.partitions_detected").increment(1);
    }

    async fn on_partition_healing(&self) {
        info!("Partition healing: preparing for sync");

        // Prepare for anti-entropy sync
        // Collect local state changes during partition
    }

    async fn on_partition_healed(&self) {
        info!("Partition healed: resuming normal operations");

        // Resume normal operations
        // Trigger anti-entropy sync

        metrics::counter!("engram.cluster.partitions_healed").increment(1);
    }

    pub async fn is_partitioned(&self) -> bool {
        matches!(
            &*self.partition_state.read().await,
            PartitionState::Partitioned { .. }
        )
    }

    pub async fn get_state(&self) -> PartitionState {
        self.partition_state.read().await.clone()
    }
}

struct ReachabilityStats {
    total_nodes: usize,
    reachable_nodes: usize,
    newly_reachable: HashSet<String>,
}
```

### Local-Only Recall

During partition, serve queries from local data only:

```rust
// engram-core/src/cluster/local_mode.rs

pub struct LocalModeGuard {
    detector: Arc<PartitionDetector>,
    local_spaces: Arc<RwLock<HashSet<String>>>,
}

impl LocalModeGuard {
    pub async fn can_serve_query(&self, space_id: &str) -> bool {
        if !self.detector.is_partitioned().await {
            // Not partitioned, can route anywhere
            return true;
        }

        // During partition, only serve if we have local data
        let local = self.local_spaces.read().await;
        if local.contains(space_id) {
            true
        } else {
            warn!(
                "Cannot serve query for space {} during partition",
                space_id
            );
            false
        }
    }

    pub async fn can_write(&self, space_id: &str) -> Result<(), ClusterError> {
        if !self.detector.is_partitioned().await {
            return Ok(());
        }

        // During partition, only write to spaces we're primary for
        let local = self.local_spaces.read().await;
        if local.contains(space_id) {
            Ok(())
        } else {
            Err(ClusterError::PartitionedFromPrimary {
                space_id: space_id.to_string(),
            })
        }
    }
}
```

### Confidence Penalty During Partition

Reduce confidence for queries during partition:

```rust
// engram-core/src/cluster/confidence.rs

pub struct PartitionAwareConfidence {
    detector: Arc<PartitionDetector>,
    base_confidence: f32,
}

impl PartitionAwareConfidence {
    pub async fn adjust_confidence(&self, base: f32) -> f32 {
        let state = self.detector.get_state().await;

        match state {
            PartitionState::Connected { .. } => base,

            PartitionState::Partitioned { reachable_nodes, total_nodes, .. } => {
                // Reduce confidence based on how many nodes we lost
                let reachability = reachable_nodes as f32 / total_nodes as f32;
                let penalty = 1.0 - reachability;

                // Apply penalty (reduce confidence by up to 50%)
                let adjusted = base * (1.0 - penalty * 0.5);

                debug!(
                    "Confidence penalty during partition: {} -> {} ({}% reachable)",
                    base, adjusted, reachability * 100.0
                );

                adjusted
            },

            PartitionState::Healing { .. } => {
                // Smaller penalty during healing
                base * 0.9
            },
        }
    }
}
```

### Split-Brain Prevention

Use vector clocks to detect concurrent primaries:

```rust
// engram-core/src/cluster/vector_clock.rs

use std::collections::HashMap;

/// Vector clock for causality tracking
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorClock {
    /// Map from node ID to logical clock value
    clocks: HashMap<String, u64>,
}

impl VectorClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment our own clock
    pub fn tick(&mut self, node_id: &str) {
        *self.clocks.entry(node_id.to_string()).or_insert(0) += 1;
    }

    /// Merge with another vector clock (on message receive)
    pub fn merge(&mut self, other: &VectorClock) {
        for (node, &clock) in &other.clocks {
            let entry = self.clocks.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(clock);
        }
    }

    /// Check causal relationship
    pub fn compare(&self, other: &VectorClock) -> Ordering {
        let mut less = false;
        let mut greater = false;

        // Get all node IDs from both clocks
        let all_nodes: HashSet<_> = self.clocks.keys()
            .chain(other.clocks.keys())
            .collect();

        for node in all_nodes {
            let self_clock = self.clocks.get(node).copied().unwrap_or(0);
            let other_clock = other.clocks.get(node).copied().unwrap_or(0);

            if self_clock < other_clock {
                less = true;
            } else if self_clock > other_clock {
                greater = true;
            }
        }

        match (less, greater) {
            (false, false) => Ordering::Equal,      // Identical
            (true, false) => Ordering::Less,        // self < other (other dominates)
            (false, true) => Ordering::Greater,     // self > other (self dominates)
            (true, true) => Ordering::Concurrent,   // Concurrent (split-brain!)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    Less,
    Equal,
    Greater,
    Concurrent, // Split-brain indicator
}

/// Detect split-brain scenario
pub struct SplitBrainDetector {
    local_clock: Arc<RwLock<VectorClock>>,
    node_id: String,
}

impl SplitBrainDetector {
    pub async fn check_for_split_brain(
        &self,
        space_id: &str,
        remote_primary_clock: &VectorClock,
    ) -> Result<(), ClusterError> {
        let local = self.local_clock.read().await;

        match local.compare(remote_primary_clock) {
            Ordering::Concurrent => {
                error!(
                    "Split-brain detected for space {}: concurrent primaries!",
                    space_id
                );

                Err(ClusterError::SplitBrain {
                    space_id: space_id.to_string(),
                    local_clock: local.clone(),
                    remote_clock: remote_primary_clock.clone(),
                })
            },
            _ => Ok(()),
        }
    }

    pub async fn on_write(&self) {
        let mut clock = self.local_clock.write().await;
        clock.tick(&self.node_id);
    }

    pub async fn on_receive(&self, remote_clock: &VectorClock) {
        let mut clock = self.local_clock.write().await;
        clock.merge(remote_clock);
    }
}
```

### Anti-Entropy Sync After Healing

```rust
// engram-core/src/cluster/anti_entropy.rs

pub struct AntiEntropySync {
    membership: Arc<SwimMembership>,
    partition_detector: Arc<PartitionDetector>,
}

impl AntiEntropySync {
    pub async fn sync_after_partition(&self) -> Result<(), ClusterError> {
        info!("Starting anti-entropy sync after partition healing");

        // 1. Identify newly reachable nodes
        let state = self.partition_detector.get_state().await;
        let newly_reachable = match state {
            PartitionState::Healing { newly_reachable, .. } => newly_reachable,
            _ => return Ok(()),
        };

        // 2. For each newly reachable node, sync state
        for node_id in newly_reachable {
            if let Some(node) = self.membership.members.get(&node_id) {
                match self.sync_with_node(&node).await {
                    Ok(()) => {
                        info!("Synced with node {}", node_id);
                    },
                    Err(e) => {
                        warn!("Failed to sync with node {}: {}", node_id, e);
                    }
                }
            }
        }

        info!("Anti-entropy sync complete");
        Ok(())
    }

    async fn sync_with_node(&self, node: &NodeInfo) -> Result<(), ClusterError> {
        // Will be implemented in Task 007 (Gossip Protocol for Consolidation)
        // For now, just log
        info!("Would sync with node {}", node.id);
        Ok(())
    }
}
```

## Files to Create

1. `engram-core/src/cluster/partition.rs` - Partition detection
2. `engram-core/src/cluster/local_mode.rs` - Local-only operations
3. `engram-core/src/cluster/confidence.rs` - Confidence adjustment
4. `engram-core/src/cluster/vector_clock.rs` - Vector clocks
5. `engram-core/src/cluster/anti_entropy.rs` - Anti-entropy sync
6. `engram-core/src/cluster/split_brain.rs` - Split-brain detection and resolution

## Files to Modify

1. `engram-core/src/cluster/mod.rs` - Export new modules
2. `engram-core/src/query/executor.rs` - Apply confidence penalty
3. `engram-cli/src/cluster.rs` - Start partition detector
4. `engram-core/src/metrics/mod.rs` - Add partition metrics

## Testing Strategy

### Unit Tests

```rust
#[tokio::test]
async fn test_partition_detection() {
    let membership = Arc::new(SwimMembership::new_test());
    let detector = PartitionDetector::new(membership.clone(), PartitionConfig::default());

    // Add 10 nodes, 6 alive, 4 dead
    for i in 0..6 {
        membership.add_node(NodeInfo {
            id: format!("node{}", i),
            state: NodeState::Alive,
            ..Default::default()
        });
    }
    for i in 6..10 {
        membership.add_node(NodeInfo {
            id: format!("node{}", i),
            state: NodeState::Dead,
            ..Default::default()
        });
    }

    detector.check_partition_status().await;

    // 6/10 = 60% alive, should be connected
    assert!(!detector.is_partitioned().await);

    // Mark more nodes dead
    for i in 3..6 {
        membership.mark_suspect(&format!("node{}", i));
    }

    detector.check_partition_status().await;

    // 3/10 = 30% alive, should be partitioned
    assert!(detector.is_partitioned().await);
}

#[tokio::test]
async fn test_vector_clock_concurrent_detection() {
    let mut clock1 = VectorClock::new();
    let mut clock2 = VectorClock::new();

    // Concurrent updates (split-brain)
    clock1.tick("node1");
    clock2.tick("node2");

    assert_eq!(clock1.compare(&clock2), Ordering::Concurrent);
}

#[test]
fn test_confidence_penalty() {
    let detector = PartitionDetector::new_test();
    let confidence = PartitionAwareConfidence::new(detector);

    // Set partition state: 3/10 nodes reachable
    detector.set_state(PartitionState::Partitioned {
        reachable_nodes: 3,
        total_nodes: 10,
        partitioned_since: Instant::now(),
    });

    let base = 0.9;
    let adjusted = confidence.adjust_confidence(base).await;

    // Should reduce confidence
    assert!(adjusted < base);
    // But not below 50% of original
    assert!(adjusted > base * 0.5);
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_partition_healing_workflow() {
    // Start 3-node cluster
    let cluster = TestCluster::new(3).await;

    // Partition node 3
    cluster.partition_node(2).await;

    tokio::time::sleep(Duration::from_secs(2)).await;

    // Nodes 1 and 2 should detect partition
    assert!(cluster.node(0).is_partitioned().await);
    assert!(cluster.node(1).is_partitioned().await);

    // Heal partition
    cluster.heal_node(2).await;

    tokio::time::sleep(Duration::from_secs(12)).await;

    // All nodes should be connected
    assert!(!cluster.node(0).is_partitioned().await);
    assert!(!cluster.node(1).is_partitioned().await);
    assert!(!cluster.node(2).is_partitioned().await);
}
```

### Chaos Tests

```rust
#[tokio::test]
#[ignore] // Long-running chaos test
async fn test_random_partition_chaos() {
    let cluster = TestCluster::new(10).await;

    // Randomly partition nodes for 5 minutes
    for _ in 0..100 {
        let node = rand::random::<usize>() % 10;
        cluster.partition_node(node).await;

        tokio::time::sleep(Duration::from_secs(3)).await;

        cluster.heal_node(node).await;

        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    // Eventually all nodes should agree on membership
    cluster.wait_for_convergence(Duration::from_secs(30)).await;
    cluster.verify_consistency().await;
}
```

## Dependencies

No new dependencies required (uses existing tokio, dashmap, serde).

## Acceptance Criteria

1. Partition detected within 10 seconds of majority unreachable
2. Local queries continue working during partition
3. Writes to non-local spaces fail fast with clear error
4. Confidence penalty applied proportionally to partition severity
5. Split-brain detected via vector clock comparison
6. Partition healing triggers anti-entropy sync
7. No data loss during partition/heal cycles
8. Metrics track partition events and duration

## Performance Targets

- Partition detection overhead: <0.1% CPU
- Confidence adjustment: <1μs per query
- Vector clock comparison: O(N) where N = cluster size
- Anti-entropy sync: completes within 1 minute for 100-node cluster

## Next Steps

After completing this task:
- Task 004 will use partition-aware routing
- Task 007 will implement full anti-entropy gossip
- Task 012 will validate partition tolerance with Jepsen
