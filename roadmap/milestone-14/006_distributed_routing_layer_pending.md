# Task 006: Distributed Routing Layer

**Status**: Pending
**Estimated Duration**: 3 days
**Dependencies**: Task 001 (SWIM membership), Task 002 (Discovery), Task 003 (Partition handling)
**Owner**: TBD

## Objective

Implement a transparent routing layer that directs operations to the correct nodes in the cluster. The router must handle connection pooling, retry logic with exponential backoff, replica fallback on primary failure, and circuit breaking for failing nodes. This task creates the critical path between Engram's API and the distributed cluster, ensuring operations reach the right nodes with minimal latency overhead.

## Research Foundation

### Routing Patterns in Distributed Systems

Distributed routing follows several proven patterns:

**1. Token Ring Routing (Cassandra/ScyllaDB)**

Cassandra's token ring assigns each node a range of hash values. Clients hash partition keys and route to the node owning that token range. Brilliant for:
- Even distribution (virtual nodes/vnodes smooth hotspots)
- Predictable routing (client-side, no coordinator overhead)
- Seamless rebalancing (token handoff between nodes)

Key insight: Pre-compute routing table from hash ring, avoid coordinator lookup on every request.

**2. Cluster Slots (Redis Cluster)**

Redis Cluster divides keyspace into 16,384 slots. Each node owns a subset of slots. Clients cache slot→node mapping, handle redirects (MOVED/ASK). Optimizations:
- Slot ownership cached client-side
- Redirect messages teach client correct mapping
- No single point of failure for routing decisions

Problem: Redirect storms during resharding. Solution: Pause writes during slot migration, atomic ownership transfer.

**3. Consistent Hashing (Dynamo, Riak)**

Dynamo-style systems hash both nodes and keys onto a ring. Walk clockwise from key hash to find primary and N-1 replicas. Advantages:
- Minimal key movement on node add/remove (only K/N keys move)
- Replica placement is deterministic
- Natural support for rack-aware placement (walk ring, skip same rack)

We use this for Engram: MemorySpace ID → consistent hash ring → primary + replicas.

**4. Connection Pooling Best Practices**

High-performance distributed systems (gRPC, Envoy) pool connections:
- **Per-endpoint pooling**: N connections per remote node (N=4 typical)
- **Health checking**: Mark connections dead if consecutive failures exceed threshold
- **Exponential backoff**: Failed nodes get increasing backoff (1s → 2s → 4s → 8s max)
- **Jitter**: Add random jitter to avoid thundering herd on recovery
- **Circuit breaking**: Fast-fail requests to dead nodes without attempting connection

gRPC specifically: Use `tonic::transport::Channel` with connection pooling. Each channel maintains HTTP/2 connection with stream multiplexing (100s of concurrent RPCs on single TCP connection).

**5. Retry Strategies**

Proven retry patterns (Google SRE book, AWS best practices):
- **Exponential backoff**: delay = min(base * 2^attempt, max_delay)
- **Jitter**: delay = delay * (0.5 + random(0, 0.5)) // Full jitter
- **Budget-based retries**: Track retry quota, refuse retries if quota exhausted
- **Deadline propagation**: Each retry respects original deadline, avoid cascading timeouts

Our strategy: 3 retries max, exponential backoff with full jitter, 5-second total deadline.

**6. Replica Fallback**

When primary fails, Engram falls back to replicas:
- **Read fallback**: Any replica can serve reads (eventual consistency)
- **Write fallback**: Promote replica to temporary primary during partition
- **Anti-entropy sync**: When primary returns, sync writes from fallback period

Critical: Track which replica served request, include in response metadata for debugging.

### Performance Benchmarks

Measured overhead in production distributed systems:
- **Cassandra local routing**: 50-200μs (token ring lookup + connection acquisition)
- **Redis Cluster MOVED redirect**: 1-2ms (includes redirect + retry)
- **gRPC connection pool acquisition**: 10-50μs (if connection exists, 10-100ms if new)
- **Retry with backoff**: 1s → 3s → 7s (exponential growth dominates)

Our target: <1ms routing overhead for 99th percentile requests.

## Technical Specification

### Core Data Structures

```rust
// engram-core/src/cluster/router.rs

use std::sync::Arc;
use std::collections::HashMap;
use dashmap::DashMap;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

/// Routes operations to correct nodes based on memory space assignment
pub struct Router {
    /// Cluster membership (for node discovery)
    membership: Arc<SwimMembership>,

    /// Memory space assignments (space_id → primary + replicas)
    assignments: Arc<SpaceAssignmentManager>,

    /// Connection pool (node_id → gRPC channel)
    connection_pool: Arc<ConnectionPool>,

    /// Circuit breakers (node_id → breaker state)
    circuit_breakers: Arc<DashMap<String, CircuitBreaker>>,

    /// Partition detector (for local-only fallback)
    partition_detector: Arc<PartitionDetector>,

    /// Routing configuration
    config: RouterConfig,
}

#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Number of retries for failed requests (default: 3)
    pub max_retries: usize,

    /// Base delay for exponential backoff (default: 100ms)
    pub retry_base_delay: Duration,

    /// Maximum retry delay (default: 5s)
    pub retry_max_delay: Duration,

    /// Total deadline for operation including retries (default: 10s)
    pub operation_deadline: Duration,

    /// Enable replica fallback on primary failure (default: true)
    pub enable_replica_fallback: bool,

    /// Circuit breaker failure threshold (default: 5)
    pub circuit_breaker_threshold: usize,

    /// Circuit breaker timeout before half-open (default: 30s)
    pub circuit_breaker_timeout: Duration,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_base_delay: Duration::from_millis(100),
            retry_max_delay: Duration::from_secs(5),
            operation_deadline: Duration::from_secs(10),
            enable_replica_fallback: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout: Duration::from_secs(30),
        }
    }
}

/// Result of routing decision
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Primary target node
    pub primary: NodeInfo,

    /// Replica nodes (for fallback)
    pub replicas: Vec<NodeInfo>,

    /// Routing strategy used
    pub strategy: RoutingStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Route to primary node
    Primary,

    /// Route to primary, fallback to replica on failure
    PrimaryWithFallback,

    /// Scatter to multiple nodes, gather results
    ScatterGather,

    /// Local-only (during partition)
    LocalOnly,
}

impl Router {
    pub fn new(
        membership: Arc<SwimMembership>,
        assignments: Arc<SpaceAssignmentManager>,
        connection_pool: Arc<ConnectionPool>,
        partition_detector: Arc<PartitionDetector>,
        config: RouterConfig,
    ) -> Self {
        Self {
            membership,
            assignments,
            connection_pool,
            circuit_breakers: Arc::new(DashMap::new()),
            partition_detector,
            config,
        }
    }

    /// Route a store operation (write)
    pub async fn route_store(
        &self,
        space_id: &str,
    ) -> Result<RoutingDecision, RouterError> {
        // Check if we're partitioned
        if self.partition_detector.is_partitioned().await {
            return self.route_local_only(space_id).await;
        }

        // Get space assignment
        let assignment = self.assignments.get_assignment(space_id).await?;

        // Store must go to primary
        Ok(RoutingDecision {
            primary: assignment.primary,
            replicas: assignment.replicas,
            strategy: RoutingStrategy::Primary,
        })
    }

    /// Route a recall operation (read)
    pub async fn route_recall(
        &self,
        space_id: &str,
    ) -> Result<RoutingDecision, RouterError> {
        // Check if we're partitioned
        if self.partition_detector.is_partitioned().await {
            return self.route_local_only(space_id).await;
        }

        // Get space assignment
        let assignment = self.assignments.get_assignment(space_id).await?;

        // Recall can fallback to replicas
        Ok(RoutingDecision {
            primary: assignment.primary,
            replicas: assignment.replicas,
            strategy: RoutingStrategy::PrimaryWithFallback,
        })
    }

    /// Route a consolidation operation (local only)
    pub async fn route_consolidation(
        &self,
        space_id: &str,
    ) -> Result<RoutingDecision, RouterError> {
        // Consolidation is always local, synced via gossip
        self.route_local_only(space_id).await
    }

    /// Route a distributed query (scatter-gather)
    pub async fn route_distributed_query(
        &self,
        space_ids: &[String],
    ) -> Result<Vec<RoutingDecision>, RouterError> {
        let mut decisions = Vec::new();

        for space_id in space_ids {
            let assignment = self.assignments.get_assignment(space_id).await?;
            decisions.push(RoutingDecision {
                primary: assignment.primary,
                replicas: assignment.replicas,
                strategy: RoutingStrategy::ScatterGather,
            });
        }

        Ok(decisions)
    }

    async fn route_local_only(
        &self,
        space_id: &str,
    ) -> Result<RoutingDecision, RouterError> {
        let local_node = self.membership.local_node();

        // Verify we have local data for this space
        let assignment = self.assignments.get_assignment(space_id).await?;
        if assignment.primary.id != local_node.id
            && !assignment.replicas.iter().any(|r| r.id == local_node.id) {
            return Err(RouterError::NoLocalData {
                space_id: space_id.to_string(),
            });
        }

        Ok(RoutingDecision {
            primary: local_node.clone(),
            replicas: vec![],
            strategy: RoutingStrategy::LocalOnly,
        })
    }

    /// Execute operation with retry logic
    pub async fn execute_with_retry<F, Fut, T>(
        &self,
        decision: &RoutingDecision,
        operation: F,
    ) -> Result<T, RouterError>
    where
        F: Fn(NodeInfo) -> Fut,
        Fut: std::future::Future<Output = Result<T, RouterError>>,
    {
        let deadline = Instant::now() + self.config.operation_deadline;

        // Try primary first
        match self.execute_with_circuit_breaker(
            &decision.primary,
            &operation,
            deadline,
        ).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                warn!(
                    "Primary {} failed: {}, attempting replicas",
                    decision.primary.id, e
                );
            }
        }

        // Try replicas if enabled
        if self.config.enable_replica_fallback {
            for replica in &decision.replicas {
                if Instant::now() >= deadline {
                    return Err(RouterError::DeadlineExceeded);
                }

                match self.execute_with_circuit_breaker(
                    replica,
                    &operation,
                    deadline,
                ).await {
                    Ok(result) => {
                        info!(
                            "Replica {} succeeded after primary failure",
                            replica.id
                        );
                        return Ok(result);
                    },
                    Err(e) => {
                        warn!("Replica {} failed: {}", replica.id, e);
                        continue;
                    }
                }
            }
        }

        Err(RouterError::AllReplicasFailed)
    }

    async fn execute_with_circuit_breaker<F, Fut, T>(
        &self,
        node: &NodeInfo,
        operation: &F,
        deadline: Instant,
    ) -> Result<T, RouterError>
    where
        F: Fn(NodeInfo) -> Fut,
        Fut: std::future::Future<Output = Result<T, RouterError>>,
    {
        // Check circuit breaker
        let breaker = self.circuit_breakers
            .entry(node.id.clone())
            .or_insert_with(|| CircuitBreaker::new(
                self.config.circuit_breaker_threshold,
                self.config.circuit_breaker_timeout,
            ));

        if !breaker.can_attempt() {
            return Err(RouterError::CircuitBreakerOpen {
                node_id: node.id.clone(),
            });
        }

        // Execute with retry
        let mut attempt = 0;
        let mut last_error = None;

        while attempt < self.config.max_retries {
            if Instant::now() >= deadline {
                return Err(RouterError::DeadlineExceeded);
            }

            match operation(node.clone()).await {
                Ok(result) => {
                    breaker.record_success();
                    return Ok(result);
                },
                Err(e) => {
                    breaker.record_failure();
                    last_error = Some(e);

                    if attempt + 1 < self.config.max_retries {
                        let delay = self.compute_backoff(attempt);
                        tokio::time::sleep(delay).await;
                    }

                    attempt += 1;
                }
            }
        }

        Err(last_error.unwrap_or(RouterError::UnknownError))
    }

    fn compute_backoff(&self, attempt: usize) -> Duration {
        let base = self.config.retry_base_delay;
        let max = self.config.retry_max_delay;

        // Exponential backoff: delay = base * 2^attempt
        let delay = base * 2_u32.pow(attempt as u32);
        let capped = delay.min(max);

        // Add full jitter: delay * (0.5 + random(0, 0.5))
        let jitter_factor = 0.5 + (rand::random::<f64>() * 0.5);
        let jittered = Duration::from_secs_f64(
            capped.as_secs_f64() * jitter_factor
        );

        jittered
    }
}
```

### Connection Pool

```rust
// engram-core/src/cluster/connection_pool.rs

use tonic::transport::{Channel, Endpoint};
use std::collections::HashMap;
use tokio::sync::Mutex;

/// Connection pool for gRPC channels to remote nodes
pub struct ConnectionPool {
    /// Active connections (node_id → channels)
    connections: Arc<DashMap<String, ChannelPool>>,

    /// Pool configuration
    config: PoolConfig,
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of channels per node (default: 4)
    pub channels_per_node: usize,

    /// Connection timeout (default: 5s)
    pub connect_timeout: Duration,

    /// Keep-alive interval (default: 30s)
    pub keepalive_interval: Duration,

    /// Keep-alive timeout (default: 10s)
    pub keepalive_timeout: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            channels_per_node: 4,
            connect_timeout: Duration::from_secs(5),
            keepalive_interval: Duration::from_secs(30),
            keepalive_timeout: Duration::from_secs(10),
        }
    }
}

/// Pool of gRPC channels to a single node
struct ChannelPool {
    channels: Vec<Channel>,
    next_index: Arc<Mutex<usize>>, // Round-robin index
}

impl ConnectionPool {
    pub fn new(config: PoolConfig) -> Self {
        Self {
            connections: Arc::new(DashMap::new()),
            config,
        }
    }

    /// Get a channel to a node (creates if not exists)
    pub async fn get_channel(&self, node: &NodeInfo) -> Result<Channel, RouterError> {
        // Check if we have existing pool
        if let Some(pool) = self.connections.get(&node.id) {
            return Ok(pool.next_channel().await);
        }

        // Create new pool
        let pool = self.create_pool(node).await?;
        let channel = pool.next_channel().await;

        self.connections.insert(node.id.clone(), pool);

        Ok(channel)
    }

    async fn create_pool(&self, node: &NodeInfo) -> Result<ChannelPool, RouterError> {
        let endpoint = Endpoint::from_shared(format!("http://{}", node.api_addr))
            .map_err(|e| RouterError::InvalidEndpoint {
                node_id: node.id.clone(),
                error: e.to_string(),
            })?
            .connect_timeout(self.config.connect_timeout)
            .timeout(Duration::from_secs(30))
            .http2_keep_alive_interval(self.config.keepalive_interval)
            .keep_alive_timeout(self.config.keepalive_timeout)
            .keep_alive_while_idle(true);

        let mut channels = Vec::with_capacity(self.config.channels_per_node);

        for _ in 0..self.config.channels_per_node {
            let channel = endpoint.connect().await
                .map_err(|e| RouterError::ConnectionFailed {
                    node_id: node.id.clone(),
                    error: e.to_string(),
                })?;

            channels.push(channel);
        }

        Ok(ChannelPool {
            channels,
            next_index: Arc::new(Mutex::new(0)),
        })
    }

    /// Remove all connections to a node (when node fails)
    pub async fn remove_node(&self, node_id: &str) {
        self.connections.remove(node_id);
    }

    /// Health check: verify all connections are alive
    pub async fn health_check(&self) -> PoolHealthReport {
        let mut total_nodes = 0;
        let mut healthy_nodes = 0;
        let mut total_channels = 0;

        for entry in self.connections.iter() {
            total_nodes += 1;
            let pool = entry.value();

            let channels_ok = pool.channels.len();
            total_channels += channels_ok;

            if channels_ok > 0 {
                healthy_nodes += 1;
            }
        }

        PoolHealthReport {
            total_nodes,
            healthy_nodes,
            total_channels,
        }
    }
}

impl ChannelPool {
    async fn next_channel(&self) -> Channel {
        let mut index = self.next_index.lock().await;
        let channel = self.channels[*index % self.channels.len()].clone();
        *index = (*index + 1) % self.channels.len();
        channel
    }
}

#[derive(Debug)]
pub struct PoolHealthReport {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub total_channels: usize,
}
```

### Circuit Breaker

```rust
// engram-core/src/cluster/circuit_breaker.rs

use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

/// Circuit breaker for failing nodes
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    state: Arc<RwLock<BreakerState>>,
    failure_threshold: usize,
    timeout: Duration,
}

#[derive(Debug, Clone)]
enum BreakerState {
    /// Circuit closed, requests flow normally
    Closed {
        consecutive_failures: usize,
    },

    /// Circuit open, fast-fail all requests
    Open {
        opened_at: Instant,
    },

    /// Circuit half-open, allow one request to test
    HalfOpen,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: usize, timeout: Duration) -> Self {
        Self {
            state: Arc::new(RwLock::new(BreakerState::Closed {
                consecutive_failures: 0,
            })),
            failure_threshold,
            timeout,
        }
    }

    /// Check if we can attempt this request
    pub async fn can_attempt(&self) -> bool {
        let mut state = self.state.write().await;

        match &*state {
            BreakerState::Closed { .. } => true,

            BreakerState::Open { opened_at } => {
                // Check if timeout elapsed, transition to half-open
                if opened_at.elapsed() >= self.timeout {
                    info!("Circuit breaker transitioning to half-open");
                    *state = BreakerState::HalfOpen;
                    true
                } else {
                    false
                }
            },

            BreakerState::HalfOpen => true,
        }
    }

    /// Record successful request
    pub async fn record_success(&self) {
        let mut state = self.state.write().await;

        match &*state {
            BreakerState::Closed { .. } => {
                // Reset failure counter
                *state = BreakerState::Closed {
                    consecutive_failures: 0,
                };
            },

            BreakerState::HalfOpen => {
                // Success in half-open, close circuit
                info!("Circuit breaker closing after successful test");
                *state = BreakerState::Closed {
                    consecutive_failures: 0,
                };
            },

            BreakerState::Open { .. } => {
                // Should not happen, but handle gracefully
                warn!("Received success while circuit open");
            },
        }
    }

    /// Record failed request
    pub async fn record_failure(&self) {
        let mut state = self.state.write().await;

        match &*state {
            BreakerState::Closed { consecutive_failures } => {
                let new_failures = consecutive_failures + 1;

                if new_failures >= self.failure_threshold {
                    // Open circuit
                    warn!(
                        "Circuit breaker opening after {} failures",
                        new_failures
                    );
                    *state = BreakerState::Open {
                        opened_at: Instant::now(),
                    };
                } else {
                    *state = BreakerState::Closed {
                        consecutive_failures: new_failures,
                    };
                }
            },

            BreakerState::HalfOpen => {
                // Failed test, reopen circuit
                warn!("Circuit breaker reopening after failed test");
                *state = BreakerState::Open {
                    opened_at: Instant::now(),
                };
            },

            BreakerState::Open { .. } => {
                // Already open, no action needed
            },
        }
    }

    pub async fn get_state(&self) -> BreakerState {
        self.state.read().await.clone()
    }
}
```

### Router Errors

```rust
// engram-core/src/cluster/error.rs (additions)

#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("No assignment found for space: {space_id}")]
    NoAssignment {
        space_id: String,
    },

    #[error("No local data for space {space_id} during partition")]
    NoLocalData {
        space_id: String,
    },

    #[error("Invalid endpoint for node {node_id}: {error}")]
    InvalidEndpoint {
        node_id: String,
        error: String,
    },

    #[error("Connection failed to node {node_id}: {error}")]
    ConnectionFailed {
        node_id: String,
        error: String,
    },

    #[error("Circuit breaker open for node {node_id}")]
    CircuitBreakerOpen {
        node_id: String,
    },

    #[error("Operation deadline exceeded")]
    DeadlineExceeded,

    #[error("All replicas failed")]
    AllReplicasFailed,

    #[error("Unknown error")]
    UnknownError,
}
```

## Files to Create

1. `engram-core/src/cluster/router.rs` - Router implementation
2. `engram-core/src/cluster/connection_pool.rs` - gRPC connection pooling
3. `engram-core/src/cluster/circuit_breaker.rs` - Circuit breaker pattern
4. `engram-core/src/cluster/retry.rs` - Retry logic with backoff
5. `engram-core/src/cluster/routing_decision.rs` - Routing decision types

## Files to Modify

1. `engram-core/src/cluster/mod.rs` - Export router module
2. `engram-core/src/cluster/error.rs` - Add router errors
3. `engram-cli/src/cluster.rs` - Initialize router
4. `engram-core/src/metrics/mod.rs` - Add routing metrics

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_route_store_to_primary() {
        let router = Router::new_test();
        let space_id = "test_space";

        // Set up assignment
        router.assignments.assign(space_id, primary_node(), vec![replica1(), replica2()]);

        let decision = router.route_store(space_id).await.unwrap();

        assert_eq!(decision.primary.id, primary_node().id);
        assert_eq!(decision.strategy, RoutingStrategy::Primary);
    }

    #[tokio::test]
    async fn test_route_recall_with_fallback() {
        let router = Router::new_test();
        let space_id = "test_space";

        router.assignments.assign(space_id, primary_node(), vec![replica1()]);

        let decision = router.route_recall(space_id).await.unwrap();

        assert_eq!(decision.primary.id, primary_node().id);
        assert_eq!(decision.replicas.len(), 1);
        assert_eq!(decision.strategy, RoutingStrategy::PrimaryWithFallback);
    }

    #[tokio::test]
    async fn test_local_only_during_partition() {
        let router = Router::new_test();
        router.partition_detector.set_partitioned(true).await;

        let space_id = "local_space";
        router.assignments.assign(space_id, local_node(), vec![]);

        let decision = router.route_store(space_id).await.unwrap();

        assert_eq!(decision.strategy, RoutingStrategy::LocalOnly);
    }

    #[tokio::test]
    async fn test_exponential_backoff() {
        let router = Router::new_test();

        let delay0 = router.compute_backoff(0);
        let delay1 = router.compute_backoff(1);
        let delay2 = router.compute_backoff(2);

        // Verify exponential growth (accounting for jitter)
        assert!(delay1 > delay0);
        assert!(delay2 > delay1);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_after_threshold() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(30));

        assert!(breaker.can_attempt().await);

        // Record 3 failures
        breaker.record_failure().await;
        breaker.record_failure().await;
        breaker.record_failure().await;

        // Circuit should be open
        assert!(!breaker.can_attempt().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_after_timeout() {
        let breaker = CircuitBreaker::new(1, Duration::from_millis(100));

        // Open circuit
        breaker.record_failure().await;
        assert!(!breaker.can_attempt().await);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should be half-open
        assert!(breaker.can_attempt().await);
    }

    #[tokio::test]
    async fn test_connection_pool_round_robin() {
        let pool = ConnectionPool::new(PoolConfig {
            channels_per_node: 4,
            ..Default::default()
        });

        let node = test_node();

        let ch1 = pool.get_channel(&node).await.unwrap();
        let ch2 = pool.get_channel(&node).await.unwrap();
        let ch3 = pool.get_channel(&node).await.unwrap();
        let ch4 = pool.get_channel(&node).await.unwrap();
        let ch5 = pool.get_channel(&node).await.unwrap();

        // Should wrap around (ch5 == ch1)
        // Note: Channel doesn't implement Eq, so we verify pool size instead
        assert_eq!(pool.connections.get(&node.id).unwrap().channels.len(), 4);
    }
}
```

### Integration Tests

```rust
// engram-core/tests/routing_integration.rs

#[tokio::test]
async fn test_router_with_real_cluster() {
    // Start 3-node cluster
    let cluster = TestCluster::new(3).await;

    // Create space assignment
    let space_id = "test_space";
    cluster.assign_space(space_id, 0, vec![1, 2]).await;

    // Route store operation
    let router = cluster.router(0);
    let decision = router.route_store(space_id).await.unwrap();

    assert_eq!(decision.primary.id, cluster.node(0).id);
    assert_eq!(decision.replicas.len(), 2);
}

#[tokio::test]
async fn test_replica_fallback_on_primary_failure() {
    let cluster = TestCluster::new(3).await;

    let space_id = "test_space";
    cluster.assign_space(space_id, 0, vec![1]).await;

    // Kill primary
    cluster.kill_node(0).await;

    let router = cluster.router(1);

    // Execute operation with fallback
    let result = router.execute_with_retry(
        &router.route_recall(space_id).await.unwrap(),
        |node| async move {
            // Mock operation
            if node.id == cluster.node(0).id {
                Err(RouterError::ConnectionFailed {
                    node_id: node.id,
                    error: "node dead".to_string(),
                })
            } else {
                Ok("success".to_string())
            }
        },
    ).await;

    assert!(result.is_ok());
}

#[tokio::test]
async fn test_circuit_breaker_prevents_cascade() {
    let cluster = TestCluster::new(3).await;

    // Kill node 2
    cluster.kill_node(2).await;

    let router = cluster.router(0);
    let node2 = cluster.node(2);

    // Try to execute operations repeatedly
    for _ in 0..10 {
        let result = router.execute_with_circuit_breaker(
            &node2,
            &|_| async { Err(RouterError::ConnectionFailed {
                node_id: "node2".to_string(),
                error: "dead".to_string(),
            }) },
            Instant::now() + Duration::from_secs(10),
        ).await;

        assert!(result.is_err());
    }

    // Verify circuit breaker opened
    let breaker = router.circuit_breakers.get(&node2.id).unwrap();
    assert!(matches!(
        breaker.get_state().await,
        BreakerState::Open { .. }
    ));
}
```

### Performance Benchmarks

```rust
// engram-core/benches/routing_bench.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_routing_decision(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let router = runtime.block_on(async {
        Router::new_test()
    });

    c.bench_function("route_store", |b| {
        b.to_async(&runtime).iter(|| async {
            black_box(router.route_store("test_space").await)
        });
    });

    c.bench_function("route_recall", |b| {
        b.to_async(&runtime).iter(|| async {
            black_box(router.route_recall("test_space").await)
        });
    });
}

fn bench_connection_pool(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let pool = ConnectionPool::new(PoolConfig::default());
    let node = test_node();

    // Prime pool
    runtime.block_on(async {
        pool.get_channel(&node).await.unwrap();
    });

    c.bench_function("connection_pool_get_channel", |b| {
        b.to_async(&runtime).iter(|| async {
            black_box(pool.get_channel(&node).await)
        });
    });
}

criterion_group!(benches, bench_routing_decision, bench_connection_pool);
criterion_main!(benches);
```

## Dependencies

Add to `engram-core/Cargo.toml`:

```toml
[dependencies]
# gRPC client
tonic = { version = "0.12", features = ["tls", "transport"] }

# Already have: tokio, dashmap, rand
```

## Acceptance Criteria

1. Routing overhead <1ms for 99th percentile requests
2. Connection pool reuses gRPC channels (no new connections on repeat requests)
3. Exponential backoff with jitter prevents thundering herd
4. Replica fallback succeeds within 2 seconds of primary failure
5. Circuit breaker opens after configured threshold (default: 5 failures)
6. Circuit breaker half-open after timeout, closes on successful test
7. Local-only routing during partition (no remote calls)
8. Metrics track: routing latency, retry count, circuit breaker state, pool size

## Performance Targets

- Routing decision: <50μs (p99)
- Connection pool acquisition: <10μs (cached), <100ms (new connection)
- Retry with backoff: completes or fails within operation deadline (10s default)
- Circuit breaker state check: <1μs
- Memory overhead: <1MB per 100 remote nodes (connection pool)

## Integration Points

### With Task 004 (Space Assignment)

Router depends on `SpaceAssignmentManager` to determine which nodes host each memory space:

```rust
let assignment = self.assignments.get_assignment(space_id).await?;
// Returns: SpaceAssignment { primary, replicas }
```

### With Task 005 (Replication)

Router directs writes to primary, which asynchronously replicates to replicas. Router only waits for primary ack, not replica acks.

### With gRPC API

All Engram API operations flow through router:

```rust
// engram-api/src/grpc/service.rs

async fn store_memory(&self, req: Request<StoreRequest>) -> Result<Response<StoreResponse>, Status> {
    let space_id = &req.get_ref().space_id;

    // Route to correct node
    let decision = self.router.route_store(space_id).await
        .map_err(|e| Status::internal(e.to_string()))?;

    // Execute with retry
    let result = self.router.execute_with_retry(&decision, |node| {
        self.store_on_node(node, req.clone())
    }).await?;

    Ok(Response::new(result))
}
```

## Next Steps

After completing this task:
- Task 007 will use router for gossip message delivery
- Task 009 will use scatter-gather routing for distributed queries
- All Engram operations transparently route to distributed cluster
