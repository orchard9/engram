# Task 001: Cluster Membership with SWIM Protocol

**Status**: Pending
**Estimated Duration**: 3-4 days
**Dependencies**: None
**Owner**: TBD

## Objective

Implement SWIM (Scalable Weakly-consistent Infection-style Process Group Membership) protocol for cluster membership, failure detection, and gossip dissemination. This provides the foundation for all distributed coordination without requiring external services like ZooKeeper.

## Technical Specification

### SWIM Protocol Overview

SWIM provides three key services:
1. **Membership**: Maintain list of alive nodes
2. **Failure Detection**: Detect crashed/unreachable nodes
3. **Dissemination**: Gossip information across cluster

Key properties:
- Constant message load per node regardless of cluster size
- Probabilistic guarantees on detection time
- Infection-style dissemination (O(log N) convergence)

### Core Data Structures

```rust
// engram-core/src/cluster/membership.rs

use std::net::SocketAddr;
use std::time::{Duration, Instant};
use dashmap::DashMap;
use tokio::sync::RwLock;

/// Node state in the cluster membership
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    /// Node is alive and responding to pings
    Alive,
    /// Node suspected of failure (missed ping)
    Suspect,
    /// Node confirmed dead (failed indirect probe)
    Dead,
    /// Node gracefully left cluster
    Left,
}

/// Metadata about a cluster node
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Unique node identifier (UUID)
    pub id: String,
    /// Network address for SWIM protocol
    pub addr: SocketAddr,
    /// gRPC endpoint for Engram API
    pub api_addr: SocketAddr,
    /// Current state in membership
    pub state: NodeState,
    /// Incarnation number (increases on each state change)
    pub incarnation: u64,
    /// Last time we updated this node's state
    pub last_update: Instant,
    /// Memory spaces this node hosts (primary or replica)
    pub spaces: Vec<String>,
}

/// SWIM membership manager
pub struct SwimMembership {
    /// Our own node information
    local_node: NodeInfo,

    /// All known nodes in cluster (keyed by node ID)
    members: DashMap<String, NodeInfo>,

    /// Protocol timing parameters
    config: SwimConfig,

    /// Random number generator for probe selection
    rng: RwLock<rand::rngs::StdRng>,
}

#[derive(Debug, Clone)]
pub struct SwimConfig {
    /// Interval between probes (default: 1s)
    pub probe_interval: Duration,

    /// Timeout waiting for probe response (default: 500ms)
    pub probe_timeout: Duration,

    /// Number of nodes to indirect probe (default: 3)
    pub indirect_probes: usize,

    /// Suspect period before marking dead (default: 5s)
    pub suspect_timeout: Duration,

    /// Number of nodes to gossip to per interval (default: 3)
    pub gossip_fanout: usize,
}

impl Default for SwimConfig {
    fn default() -> Self {
        Self {
            probe_interval: Duration::from_secs(1),
            probe_timeout: Duration::from_millis(500),
            indirect_probes: 3,
            suspect_timeout: Duration::from_secs(5),
            gossip_fanout: 3,
        }
    }
}

/// SWIM protocol messages
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SwimMessage {
    /// Direct ping to target node
    Ping {
        from: String,
        incarnation: u64,
    },

    /// Acknowledgment of ping
    Ack {
        from: String,
        incarnation: u64,
    },

    /// Request for indirect probe
    PingReq {
        from: String,
        target: String,
        incarnation: u64,
    },

    /// Membership state updates
    Gossip {
        updates: Vec<MembershipUpdate>,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MembershipUpdate {
    pub node_id: String,
    pub addr: SocketAddr,
    pub state: NodeState,
    pub incarnation: u64,
    pub spaces: Vec<String>,
}
```

### Core Operations

#### 1. Probe Cycle (Failure Detection)

Every `probe_interval`, select a random node and probe:

```rust
impl SwimMembership {
    /// Run one probe cycle
    pub async fn probe_cycle(&self) -> Result<(), ClusterError> {
        // 1. Select random alive node
        let target = self.select_random_node(NodeState::Alive)?;

        // 2. Send direct ping
        let ping_result = tokio::time::timeout(
            self.config.probe_timeout,
            self.send_ping(&target)
        ).await;

        match ping_result {
            Ok(Ok(ack)) => {
                // Direct ack received, node is alive
                self.update_node_state(&target.id, NodeState::Alive, ack.incarnation);
            },
            _ => {
                // Direct ping failed, try indirect probes
                let indirect_result = self.indirect_probe(&target).await?;

                if !indirect_result {
                    // Mark as suspect
                    self.mark_suspect(&target.id);
                }
            }
        }

        Ok(())
    }

    /// Perform indirect probe through K random nodes
    async fn indirect_probe(&self, target: &NodeInfo) -> Result<bool, ClusterError> {
        let probers = self.select_random_nodes(self.config.indirect_probes)?;

        let mut handles = vec![];
        for prober in probers {
            let handle = tokio::spawn({
                let target = target.clone();
                async move {
                    self.send_ping_req(&prober, &target).await
                }
            });
            handles.push(handle);
        }

        // Wait for any successful indirect ack
        for handle in handles {
            if let Ok(Ok(true)) = handle.await {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Mark node as suspect, start suspect timer
    fn mark_suspect(&self, node_id: &str) {
        if let Some(mut node) = self.members.get_mut(node_id) {
            node.state = NodeState::Suspect;
            node.last_update = Instant::now();
        }

        // Schedule background task to check suspect timeout
        let suspect_timeout = self.config.suspect_timeout;
        let node_id = node_id.to_string();
        let members = self.members.clone();

        tokio::spawn(async move {
            tokio::time::sleep(suspect_timeout).await;

            if let Some(mut node) = members.get_mut(&node_id) {
                if node.state == NodeState::Suspect {
                    let elapsed = Instant::now().duration_since(node.last_update);
                    if elapsed >= suspect_timeout {
                        node.state = NodeState::Dead;
                    }
                }
            }
        });
    }
}
```

#### 2. Gossip Dissemination

Piggyback membership updates on ping messages:

```rust
impl SwimMembership {
    /// Collect recent membership updates for gossip
    fn collect_gossip(&self) -> Vec<MembershipUpdate> {
        let mut updates = vec![];

        for entry in self.members.iter() {
            let node = entry.value();

            // Include state changes from last 10 seconds
            if node.last_update.elapsed() < Duration::from_secs(10) {
                updates.push(MembershipUpdate {
                    node_id: node.id.clone(),
                    addr: node.addr,
                    state: node.state,
                    incarnation: node.incarnation,
                    spaces: node.spaces.clone(),
                });
            }
        }

        updates
    }

    /// Apply gossip updates received from other nodes
    fn apply_gossip(&self, updates: Vec<MembershipUpdate>) {
        for update in updates {
            self.merge_member_state(update);
        }
    }

    /// Merge member state update (respecting incarnation numbers)
    fn merge_member_state(&self, update: MembershipUpdate) {
        self.members.entry(update.node_id.clone()).and_modify(|node| {
            // Only apply if incarnation is newer
            if update.incarnation > node.incarnation {
                node.state = update.state;
                node.incarnation = update.incarnation;
                node.spaces = update.spaces.clone();
                node.last_update = Instant::now();
            }
        }).or_insert_with(|| {
            // New node discovered via gossip
            NodeInfo {
                id: update.node_id,
                addr: update.addr,
                api_addr: update.addr, // Will be updated when we connect
                state: update.state,
                incarnation: update.incarnation,
                last_update: Instant::now(),
                spaces: update.spaces,
            }
        });
    }
}
```

#### 3. Refutation (Incarnation Counter)

If a node hears it's been marked suspect/dead, it refutes by incrementing incarnation:

```rust
impl SwimMembership {
    /// Handle incoming gossip about ourselves
    fn handle_self_update(&self, update: MembershipUpdate) {
        if update.state == NodeState::Suspect || update.state == NodeState::Dead {
            // We're alive! Refute by incrementing incarnation
            let mut local = self.local_node.clone();
            local.incarnation = update.incarnation + 1;
            local.state = NodeState::Alive;

            // Broadcast refutation
            self.broadcast_update(MembershipUpdate {
                node_id: local.id.clone(),
                addr: local.addr,
                state: NodeState::Alive,
                incarnation: local.incarnation,
                spaces: local.spaces.clone(),
            });
        }
    }
}
```

### Network Protocol

Use UDP for SWIM messages (low overhead, connectionless):

```rust
// engram-core/src/cluster/transport.rs

use tokio::net::UdpSocket;
use bincode::{serialize, deserialize};

pub struct SwimTransport {
    socket: UdpSocket,
    max_packet_size: usize,
}

impl SwimTransport {
    pub async fn new(bind_addr: SocketAddr) -> Result<Self, ClusterError> {
        let socket = UdpSocket::bind(bind_addr).await?;
        Ok(Self {
            socket,
            max_packet_size: 1400, // Stay under MTU
        })
    }

    pub async fn send(&self, msg: &SwimMessage, to: SocketAddr) -> Result<(), ClusterError> {
        let bytes = serialize(msg)
            .map_err(|e| ClusterError::SerializationError(e.to_string()))?;

        if bytes.len() > self.max_packet_size {
            return Err(ClusterError::MessageTooLarge(bytes.len()));
        }

        self.socket.send_to(&bytes, to).await?;
        Ok(())
    }

    pub async fn recv(&self) -> Result<(SwimMessage, SocketAddr), ClusterError> {
        let mut buf = vec![0u8; self.max_packet_size];
        let (len, addr) = self.socket.recv_from(&mut buf).await?;

        let msg = deserialize(&buf[..len])
            .map_err(|e| ClusterError::DeserializationError(e.to_string()))?;

        Ok((msg, addr))
    }
}
```

### Integration Points

#### Configuration

```toml
# engram-cli/config/cluster.toml

[cluster]
enabled = false  # Single-node by default
node_id = ""     # Auto-generated UUID if empty

[cluster.swim]
bind_addr = "0.0.0.0:7946"
probe_interval_ms = 1000
probe_timeout_ms = 500
indirect_probes = 3
suspect_timeout_ms = 5000
gossip_fanout = 3

[cluster.seed_nodes]
# List of initial nodes to contact on startup
addrs = ["node1.example.com:7946", "node2.example.com:7946"]
```

#### Startup Integration

```rust
// engram-cli/src/main.rs

async fn start_cluster_mode(config: ClusterConfig) -> Result<(), Error> {
    // 1. Initialize SWIM membership
    let swim = SwimMembership::new(
        config.node_id,
        config.swim.bind_addr,
        config.swim.into(),
    ).await?;

    // 2. Join cluster via seed nodes
    for seed in config.seed_nodes.addrs {
        swim.join(seed).await?;
    }

    // 3. Start background probe/gossip loops
    tokio::spawn(async move {
        loop {
            swim.probe_cycle().await;
            tokio::time::sleep(swim.config.probe_interval).await;
        }
    });

    // 4. Start message handler
    let transport = SwimTransport::new(config.swim.bind_addr).await?;
    tokio::spawn(async move {
        loop {
            match transport.recv().await {
                Ok((msg, from)) => swim.handle_message(msg, from).await,
                Err(e) => error!("Transport error: {}", e),
            }
        }
    });

    Ok(())
}
```

## Files to Create

1. `engram-core/src/cluster/mod.rs` - Cluster module
2. `engram-core/src/cluster/membership.rs` - SWIM membership implementation
3. `engram-core/src/cluster/transport.rs` - UDP transport for SWIM
4. `engram-core/src/cluster/error.rs` - Cluster error types
5. `engram-core/src/cluster/config.rs` - Cluster configuration
6. `engram-cli/config/cluster.toml` - Cluster configuration template

## Files to Modify

1. `engram-cli/src/main.rs` - Add cluster mode startup
2. `engram-core/Cargo.toml` - Add cluster dependencies
3. `engram-core/src/lib.rs` - Export cluster module

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_swim_membership_lifecycle() {
        let swim = SwimMembership::new_test();

        // Add node
        swim.add_node(node_info("node1"));
        assert_eq!(swim.members.len(), 1);

        // Mark suspect
        swim.mark_suspect("node1");
        assert_eq!(swim.get_state("node1"), Some(NodeState::Suspect));

        // Wait for timeout
        tokio::time::sleep(Duration::from_secs(6)).await;
        assert_eq!(swim.get_state("node1"), Some(NodeState::Dead));
    }

    #[tokio::test]
    async fn test_incarnation_refutation() {
        let swim = SwimMembership::new_test();

        // Receive gossip marking us dead
        let update = MembershipUpdate {
            node_id: swim.local_node.id.clone(),
            state: NodeState::Dead,
            incarnation: 5,
            ..Default::default()
        };

        swim.handle_self_update(update);

        // Should refute with higher incarnation
        assert_eq!(swim.local_node.incarnation, 6);
        assert_eq!(swim.local_node.state, NodeState::Alive);
    }

    #[tokio::test]
    async fn test_gossip_convergence() {
        let node1 = SwimMembership::new_test();
        let node2 = SwimMembership::new_test();

        // Node 1 learns about node 3
        node1.add_node(node_info("node3"));

        // Gossip from node 1 to node 2
        let updates = node1.collect_gossip();
        node2.apply_gossip(updates);

        // Node 2 should now know about node 3
        assert!(node2.members.contains_key("node3"));
    }
}
```

### Integration Tests

```rust
// engram-core/tests/cluster_integration.rs

#[tokio::test]
async fn test_three_node_cluster() {
    // Start three nodes
    let node1 = start_test_node(7946).await;
    let node2 = start_test_node(7947).await;
    let node3 = start_test_node(7948).await;

    // Join node2 and node3 to node1
    node2.join("127.0.0.1:7946").await.unwrap();
    node3.join("127.0.0.1:7946").await.unwrap();

    // Wait for gossip convergence
    tokio::time::sleep(Duration::from_secs(5)).await;

    // All nodes should know about each other
    assert_eq!(node1.members.len(), 3);
    assert_eq!(node2.members.len(), 3);
    assert_eq!(node3.members.len(), 3);
}

#[tokio::test]
async fn test_node_failure_detection() {
    let node1 = start_test_node(7946).await;
    let node2 = start_test_node(7947).await;

    node2.join("127.0.0.1:7946").await.unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Kill node2
    node2.shutdown().await;

    // Wait for failure detection (probe + suspect timeout)
    tokio::time::sleep(Duration::from_secs(8)).await;

    // Node1 should mark node2 as dead
    assert_eq!(node1.get_state(&node2.id), Some(NodeState::Dead));
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_gossip_convergence_property(
        num_nodes in 3..10usize,
        num_updates in 1..100usize,
    ) {
        // Create N nodes
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let nodes: Vec<_> = (0..num_nodes)
            .map(|i| runtime.block_on(start_test_node(7946 + i)))
            .collect();

        // Connect them in a ring
        for i in 1..num_nodes {
            runtime.block_on(nodes[i].join(format!("127.0.0.1:{}", 7946)));
        }

        // Wait for convergence
        runtime.block_on(tokio::time::sleep(Duration::from_secs(10)));

        // All nodes should have same membership view
        let baseline = nodes[0].members.len();
        for node in &nodes[1..] {
            prop_assert_eq!(node.members.len(), baseline);
        }
    }
}
```

## Dependencies

Add to `engram-core/Cargo.toml`:

```toml
[dependencies]
# Existing dependencies...

# Cluster membership
uuid = { version = "1.6", features = ["v4"] }
rand = "0.8"

# Already have: tokio, dashmap, bincode, serde
```

## Acceptance Criteria

1. Three-node cluster forms and converges within 5 seconds
2. Node failure detected within 7 seconds (probe + suspect timeout)
3. Refutation works: marked-dead node can resurrect itself
4. Gossip converges: all nodes agree on membership within 10 seconds
5. No message loss: UDP transport handles packet loss gracefully
6. Single-node mode unaffected: cluster disabled by default
7. Configuration: all SWIM parameters tunable via config file
8. Metrics: membership size, failure detection latency, gossip lag

## Performance Targets

- Membership updates propagate in O(log N) rounds
- Constant message load per node (independent of cluster size)
- Support 100+ node clusters with <1% CPU overhead
- Memory overhead <10MB per node for membership state

## Next Steps

After completing this task:
- Task 002 will use membership for node discovery
- Task 004 will assign memory spaces to nodes based on membership
- Task 007 will layer consolidation gossip on SWIM protocol
