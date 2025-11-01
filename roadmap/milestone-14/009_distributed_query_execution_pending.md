# Task 009: Distributed Query Execution (Scatter-Gather)

**Status**: Pending
**Estimated Duration**: 3-4 days
**Dependencies**: Task 001 (SWIM membership), Task 004 (Space assignment), Task 005 (Replication), Task 006 (Routing)
**Owner**: TBD

## Objective

Implement distributed query execution using scatter-gather pattern to execute activation spreading, pattern completion, and memory recall across multiple partitions. Aggregate partial results with confidence adjustment, handle timeouts gracefully, and ensure query latency remains within 2x single-node performance for intra-partition queries.

## Research Foundation

Distributed query execution is the cornerstone of any partitioned database. Three proven patterns dominate production systems:

**1. Elasticsearch's Scatter-Gather (2010-present)**

Elasticsearch pioneered practical scatter-gather for inverted indexes. Query coordinator scatters to all relevant shards, gathers partial results, merges with score re-ranking. Key insights:
- **Partial results always valid**: Each shard returns top-K results independently. Merge produces approximate top-K globally (good enough for search).
- **Timeout = partial answer**: If shard times out, use results from responding shards with reduced confidence. Better to show 80% of results than error.
- **Adaptive routing**: Track shard response times, route queries to faster replicas first.

Elasticsearch achieves <100ms p99 latency for scatter across 100s of shards by keeping shard queries simple (term matching) and results small (top-10 documents).

**2. Cassandra's Coordinator Pattern (2010-present)**

Cassandra's coordinator scatters reads to replica nodes based on consistency level (ONE, QUORUM, ALL). Brilliant design choices:
- **Consistency = user choice**: ONE = fast, ALL = consistent, QUORUM = balanced. Let application decide per-query.
- **Hinted handoff**: If replica down, coordinator stores write locally ("hint") and replays later. Availability wins over consistency.
- **Read repair**: When coordinator sees inconsistent values from replicas, repair in background. Reads become self-healing.

Cassandra achieves 99.9% availability during partial failures by never blocking on minority replicas. Quorum reads detect and repair inconsistencies automatically.

**3. MongoDB's Query Router (mongos) (2012-present)**

MongoDB's mongos router implements sophisticated query planning for sharded collections:
- **Query planner analyzes shard key**: If query contains shard key, route to single shard (targeted query). Otherwise scatter to all shards (broadcast query).
- **Cursor merging**: For sort/limit queries, merge sorted cursors from each shard using priority queue. Achieves global sort without loading all data.
- **Aggregation pipeline pushdown**: Push $match/$project stages to shards, only merge final results. Minimizes network transfer.

MongoDB achieves 10-100x speedup for targeted queries vs broadcast by aggressive query planning. Lesson: **query planning pays off when partition key is well-chosen**.

**Engram's Design Space**

Engram differs from these systems in critical ways:

1. **Queries are spreading activation, not indexed lookups**: Cannot predict which partitions will contribute to result set (spreading crosses partition boundaries). Must scatter broadly initially, learn activation patterns over time.

2. **Results are probabilistic, not deterministic**: Confidence scores naturally accommodate missing partitions. Missing 20% of partitions = 20% confidence penalty, not query failure.

3. **Cross-partition spreading is fundamental**: Unlike sharded databases where cross-shard queries are rare, Engram's semantic associations span partitions. Need efficient cross-partition message passing.

**Our Strategy**:
- **Phase 1 (this task)**: Conservative scatter-gather. Scatter to all partitions that might contribute, aggregate with confidence penalties. Optimize later.
- **Phase 2 (future)**: Adaptive routing. Learn which partitions contribute most to query patterns, route speculatively.
- **Phase 3 (future)**: Streaming scatter-gather. Start returning results while gathering, cancel slow partitions when confidence threshold met.

**Performance Model**

Single-node activation spreading: 1-5ms for 1000-node graph (M13 baselines)

Distributed overhead sources:
- Network round-trip: 1ms (same datacenter)
- Serialization/deserialization: 0.5ms per partition
- Result aggregation: 0.1ms per partition
- Confidence adjustment: 0.05ms

**Best case (intra-partition query)**: 1-5ms (same as single-node)
**Average case (3 partitions)**: 1ms + max(5ms, 5ms, 5ms) + 3×0.5ms + 0.3ms + 0.05ms = ~8ms (1.6x slowdown)
**Worst case (10 partitions)**: 1ms + max(5ms × 10) + 10×0.5ms + 1ms + 0.05ms = ~12ms (2.4x slowdown)

**Target**: <2x slowdown on average = <10ms for typical queries. Achievable if we keep scatter fanout low and parallelize aggressively.

**Timeout Handling Philosophy**

Traditional databases: timeout = error (CAP theorem's consistency choice)
Engram: timeout = reduced confidence (availability choice)

Rationale: Biological memory doesn't fail when some neurons are slow to respond. It returns best-effort answer with uncertainty. We do the same.

Implementation: 5-second default timeout (100x median query time). If partition times out, exclude from confidence calculation proportionally. Log timeout for monitoring/alerting.

## Technical Specification

### Query Planning Phase

Before scattering, determine which partitions contain relevant data:

```rust
// engram-core/src/cluster/query/planner.rs

use std::collections::HashSet;
use std::sync::Arc;

/// Query planner determines which partitions to query
pub struct QueryPlanner {
    space_assignment: Arc<SpaceAssignment>,
    membership: Arc<SwimMembership>,
    partition_detector: Arc<PartitionDetector>,
}

/// Query plan identifies target nodes and execution strategy
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Target nodes to query
    pub targets: Vec<QueryTarget>,

    /// Expected partitions (for confidence calculation)
    pub expected_partitions: usize,

    /// Query execution strategy
    pub strategy: ExecutionStrategy,

    /// Timeout per partition
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct QueryTarget {
    /// Node to query
    pub node_id: String,

    /// Node network address
    pub addr: SocketAddr,

    /// Memory spaces to query on this node
    pub spaces: Vec<String>,

    /// Whether this is primary or replica
    pub role: NodeRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    Primary,
    Replica,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStrategy {
    /// Query single partition (space explicitly specified)
    SinglePartition,

    /// Query all partitions containing specific spaces
    MultiPartition,

    /// Broadcast to all nodes (cross-partition spreading)
    Broadcast,
}

impl QueryPlanner {
    pub fn new(
        space_assignment: Arc<SpaceAssignment>,
        membership: Arc<SwimMembership>,
        partition_detector: Arc<PartitionDetector>,
    ) -> Self {
        Self {
            space_assignment,
            membership,
            partition_detector,
        }
    }

    /// Plan query execution based on query type and space IDs
    pub async fn plan_query(
        &self,
        query: &Query,
    ) -> Result<QueryPlan, ClusterError> {
        match query {
            Query::Recall { space_id, .. } => {
                // Single-partition query (space explicitly specified)
                self.plan_single_partition(space_id).await
            },

            Query::Spread { space_id, .. } => {
                // May cross partition boundaries
                self.plan_spreading_query(space_id).await
            },

            Query::Complete { space_id, .. } => {
                // Single-partition (pattern completion is local)
                self.plan_single_partition(space_id).await
            },

            Query::Consolidate { space_id, .. } => {
                // Single-partition (consolidation is local)
                self.plan_single_partition(space_id).await
            },

            Query::Imagine { spaces, .. } => {
                // Multi-partition (cross-space imagination)
                self.plan_multi_partition(spaces).await
            },
        }
    }

    /// Plan single-partition query (route to primary, fallback to replica)
    async fn plan_single_partition(
        &self,
        space_id: &str,
    ) -> Result<QueryPlan, ClusterError> {
        let assignment = self.space_assignment
            .get(space_id)
            .ok_or_else(|| ClusterError::SpaceNotFound {
                space_id: space_id.to_string(),
            })?;

        // Prefer primary, fallback to replicas if partitioned
        let is_partitioned = self.partition_detector.is_partitioned().await;

        let targets = if is_partitioned {
            // During partition, try replicas if primary unreachable
            self.build_targets_with_fallback(&assignment).await?
        } else {
            // Normal case: route to primary
            vec![self.build_target(&assignment.primary_node, &[space_id.to_string()], NodeRole::Primary)?]
        };

        Ok(QueryPlan {
            targets,
            expected_partitions: 1,
            strategy: ExecutionStrategy::SinglePartition,
            timeout: Duration::from_millis(500),
        })
    }

    /// Plan spreading query (may span multiple partitions)
    async fn plan_spreading_query(
        &self,
        space_id: &str,
    ) -> Result<QueryPlan, ClusterError> {
        // For now, spreading is intra-partition only
        // Future: detect cross-partition edges and scatter to neighbor partitions

        self.plan_single_partition(space_id).await
    }

    /// Plan multi-partition query (broadcast to specific spaces)
    async fn plan_multi_partition(
        &self,
        space_ids: &[String],
    ) -> Result<QueryPlan, ClusterError> {
        let mut targets = Vec::new();
        let mut seen_nodes = HashSet::new();

        for space_id in space_ids {
            let assignment = self.space_assignment
                .get(space_id)
                .ok_or_else(|| ClusterError::SpaceNotFound {
                    space_id: space_id.to_string(),
                })?;

            // Add primary node if not already seen
            if !seen_nodes.contains(&assignment.primary_node) {
                targets.push(self.build_target(
                    &assignment.primary_node,
                    &[space_id.to_string()],
                    NodeRole::Primary,
                )?);
                seen_nodes.insert(assignment.primary_node.clone());
            } else {
                // Node already in target list, add space to existing target
                if let Some(target) = targets.iter_mut()
                    .find(|t| t.node_id == assignment.primary_node)
                {
                    target.spaces.push(space_id.to_string());
                }
            }
        }

        Ok(QueryPlan {
            expected_partitions: targets.len(),
            targets,
            strategy: ExecutionStrategy::MultiPartition,
            timeout: Duration::from_secs(5),
        })
    }

    fn build_target(
        &self,
        node_id: &str,
        spaces: &[String],
        role: NodeRole,
    ) -> Result<QueryTarget, ClusterError> {
        let node = self.membership.members
            .get(node_id)
            .ok_or_else(|| ClusterError::NodeNotFound {
                node_id: node_id.to_string(),
            })?;

        Ok(QueryTarget {
            node_id: node_id.to_string(),
            addr: node.api_addr,
            spaces: spaces.to_vec(),
            role,
        })
    }

    async fn build_targets_with_fallback(
        &self,
        assignment: &SpaceAssignmentInfo,
    ) -> Result<Vec<QueryTarget>, ClusterError> {
        let mut targets = Vec::new();

        // Try primary first
        if let Ok(target) = self.build_target(
            &assignment.primary_node,
            &[assignment.space_id.clone()],
            NodeRole::Primary,
        ) {
            targets.push(target);
        }

        // Add reachable replicas as fallbacks
        for replica_id in &assignment.replica_nodes {
            if let Ok(target) = self.build_target(
                replica_id,
                &[assignment.space_id.clone()],
                NodeRole::Replica,
            ) {
                targets.push(target);
            }
        }

        if targets.is_empty() {
            return Err(ClusterError::NoReachableNodes {
                space_id: assignment.space_id.clone(),
            });
        }

        Ok(targets)
    }
}
```

### Scatter Phase

Dispatch queries to all target nodes in parallel:

```rust
// engram-core/src/cluster/query/scatter.rs

use futures::future;
use tokio::time::timeout;
use std::sync::Arc;

/// Scatter executor dispatches queries to multiple nodes
pub struct ScatterExecutor {
    connection_pool: Arc<ConnectionPool>,
    metrics: Arc<QueryMetrics>,
}

/// Request to scatter to a single node
#[derive(Debug, Clone)]
pub struct ScatterRequest {
    /// Target node
    pub target: QueryTarget,

    /// Query to execute
    pub query: Query,

    /// Timeout for this request
    pub timeout: Duration,
}

/// Response from a single node
#[derive(Debug, Clone)]
pub struct ScatterResponse {
    /// Node that responded
    pub node_id: String,

    /// Whether this was primary or replica
    pub role: NodeRole,

    /// Query result (or error)
    pub result: Result<QueryResult, QueryError>,

    /// Query latency
    pub latency: Duration,
}

impl ScatterExecutor {
    pub fn new(
        connection_pool: Arc<ConnectionPool>,
        metrics: Arc<QueryMetrics>,
    ) -> Self {
        Self {
            connection_pool,
            metrics,
        }
    }

    /// Scatter query to multiple nodes in parallel
    pub async fn scatter(
        &self,
        plan: &QueryPlan,
        query: Query,
    ) -> Vec<ScatterResponse> {
        let start = Instant::now();

        // Create scatter requests
        let requests: Vec<_> = plan.targets.iter()
            .map(|target| ScatterRequest {
                target: target.clone(),
                query: query.clone(),
                timeout: plan.timeout,
            })
            .collect();

        // Execute all requests in parallel
        let handles: Vec<_> = requests.into_iter()
            .map(|req| {
                let executor = self.clone();
                tokio::spawn(async move {
                    executor.scatter_single(req).await
                })
            })
            .collect();

        // Gather results (don't fail on individual errors)
        let mut responses = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(response) => responses.push(response),
                Err(e) => {
                    error!("Scatter task panicked: {}", e);
                    // Continue with partial results
                }
            }
        }

        let scatter_latency = start.elapsed();
        self.metrics.record_scatter_latency(scatter_latency, responses.len());

        responses
    }

    /// Execute query on single node
    async fn scatter_single(&self, req: ScatterRequest) -> ScatterResponse {
        let start = Instant::now();

        // Get gRPC client from pool
        let client = match self.connection_pool.get(&req.target.addr).await {
            Ok(client) => client,
            Err(e) => {
                return ScatterResponse {
                    node_id: req.target.node_id,
                    role: req.target.role,
                    result: Err(QueryError::ConnectionFailed(e.to_string())),
                    latency: start.elapsed(),
                };
            }
        };

        // Execute query with timeout
        let result = match timeout(req.timeout, self.execute_remote_query(client, &req.query)).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(e)) => Err(e),
            Err(_) => {
                warn!(
                    "Query timeout on node {} after {:?}",
                    req.target.node_id, req.timeout
                );
                Err(QueryError::Timeout {
                    node_id: req.target.node_id.clone(),
                    timeout: req.timeout,
                })
            }
        };

        let latency = start.elapsed();
        self.metrics.record_partition_latency(&req.target.node_id, latency);

        ScatterResponse {
            node_id: req.target.node_id,
            role: req.target.role,
            result,
            latency,
        }
    }

    async fn execute_remote_query(
        &self,
        mut client: EngramClient,
        query: &Query,
    ) -> Result<QueryResult, QueryError> {
        // Convert Query to gRPC request
        let grpc_request = query.to_grpc_request();

        // Execute via gRPC
        let response = client
            .execute_query(grpc_request)
            .await
            .map_err(|e| QueryError::GrpcError(e.to_string()))?;

        // Convert gRPC response to QueryResult
        QueryResult::from_grpc_response(response.into_inner())
    }
}

impl Clone for ScatterExecutor {
    fn clone(&self) -> Self {
        Self {
            connection_pool: self.connection_pool.clone(),
            metrics: self.metrics.clone(),
        }
    }
}
```

### Gather Phase

Collect partial results and merge:

```rust
// engram-core/src/cluster/query/gather.rs

use std::collections::HashMap;

/// Gather executor aggregates partial results
pub struct GatherExecutor {
    partition_detector: Arc<PartitionDetector>,
    metrics: Arc<QueryMetrics>,
}

/// Aggregated result from multiple partitions
#[derive(Debug, Clone)]
pub struct AggregatedResult {
    /// Merged query result
    pub result: QueryResult,

    /// Number of partitions that responded
    pub responding_partitions: usize,

    /// Number of partitions that timed out or failed
    pub failed_partitions: usize,

    /// Total partitions queried
    pub total_partitions: usize,

    /// Confidence penalty applied (0.0 = no penalty, 1.0 = full penalty)
    pub confidence_penalty: f32,

    /// Per-partition latencies
    pub partition_latencies: HashMap<String, Duration>,
}

impl GatherExecutor {
    pub fn new(
        partition_detector: Arc<PartitionDetector>,
        metrics: Arc<QueryMetrics>,
    ) -> Self {
        Self {
            partition_detector,
            metrics,
        }
    }

    /// Gather and aggregate partial results
    pub async fn gather(
        &self,
        plan: &QueryPlan,
        responses: Vec<ScatterResponse>,
    ) -> Result<AggregatedResult, ClusterError> {
        let total_partitions = plan.expected_partitions;

        // Separate successful and failed responses
        let (successful, failed): (Vec<_>, Vec<_>) = responses.into_iter()
            .partition(|r| r.result.is_ok());

        let responding_partitions = successful.len();
        let failed_partitions = failed.len();

        // Log failures
        for fail in &failed {
            warn!(
                "Partition {} failed: {:?}",
                fail.node_id, fail.result
            );
        }

        if successful.is_empty() {
            return Err(ClusterError::AllPartitionsFailed {
                total: total_partitions,
            });
        }

        // Extract successful results
        let partial_results: Vec<_> = successful.into_iter()
            .filter_map(|r| r.result.ok())
            .collect();

        // Merge partial results based on query type
        let merged = self.merge_results(partial_results)?;

        // Calculate confidence penalty
        let confidence_penalty = self.calculate_confidence_penalty(
            responding_partitions,
            total_partitions,
        ).await;

        // Adjust confidence scores
        let result = self.apply_confidence_penalty(merged, confidence_penalty);

        // Collect latency stats
        let partition_latencies = responses.iter()
            .map(|r| (r.node_id.clone(), r.latency))
            .collect();

        // Record metrics
        self.metrics.record_gather_stats(
            responding_partitions,
            failed_partitions,
            confidence_penalty,
        );

        Ok(AggregatedResult {
            result,
            responding_partitions,
            failed_partitions,
            total_partitions,
            confidence_penalty,
            partition_latencies,
        })
    }

    /// Merge partial results based on query type
    fn merge_results(
        &self,
        partial_results: Vec<QueryResult>,
    ) -> Result<QueryResult, ClusterError> {
        if partial_results.is_empty() {
            return Err(ClusterError::NoResults);
        }

        if partial_results.len() == 1 {
            return Ok(partial_results.into_iter().next().unwrap());
        }

        // Merge based on result type
        match &partial_results[0] {
            QueryResult::Recall(memories) => {
                self.merge_recall_results(partial_results)
            },
            QueryResult::Spread(activations) => {
                self.merge_spread_results(partial_results)
            },
            QueryResult::Complete(patterns) => {
                self.merge_complete_results(partial_results)
            },
            QueryResult::Consolidate(stats) => {
                self.merge_consolidate_results(partial_results)
            },
            QueryResult::Imagine(generated) => {
                self.merge_imagine_results(partial_results)
            },
        }
    }

    fn merge_recall_results(
        &self,
        partial_results: Vec<QueryResult>,
    ) -> Result<QueryResult, ClusterError> {
        let mut all_memories = Vec::new();

        for result in partial_results {
            if let QueryResult::Recall(memories) = result {
                all_memories.extend(memories);
            }
        }

        // Deduplicate by memory ID (same memory might exist on multiple replicas)
        let mut seen = HashMap::new();
        let mut merged = Vec::new();

        for memory in all_memories {
            let id = memory.id.clone();
            seen.entry(id.clone())
                .and_modify(|existing: &mut Memory| {
                    // Keep version with higher confidence
                    if memory.confidence.0 > existing.confidence.0 {
                        *existing = memory.clone();
                    }
                })
                .or_insert_with(|| {
                    merged.push(memory.clone());
                    memory
                });
        }

        // Sort by confidence (highest first)
        merged.sort_by(|a, b| {
            b.confidence.0.partial_cmp(&a.confidence.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(QueryResult::Recall(merged))
    }

    fn merge_spread_results(
        &self,
        partial_results: Vec<QueryResult>,
    ) -> Result<QueryResult, ClusterError> {
        let mut all_activations = HashMap::new();

        for result in partial_results {
            if let QueryResult::Spread(activations) = result {
                for (node_id, activation) in activations {
                    all_activations.entry(node_id)
                        .and_modify(|existing: &mut f32| {
                            // Sum activations from different partitions
                            *existing += activation;
                        })
                        .or_insert(activation);
                }
            }
        }

        Ok(QueryResult::Spread(all_activations))
    }

    fn merge_complete_results(
        &self,
        partial_results: Vec<QueryResult>,
    ) -> Result<QueryResult, ClusterError> {
        // Pattern completion is single-partition, shouldn't have multiple results
        // Just return first result
        Ok(partial_results.into_iter().next().unwrap())
    }

    fn merge_consolidate_results(
        &self,
        partial_results: Vec<QueryResult>,
    ) -> Result<QueryResult, ClusterError> {
        // Consolidation is single-partition, shouldn't have multiple results
        Ok(partial_results.into_iter().next().unwrap())
    }

    fn merge_imagine_results(
        &self,
        partial_results: Vec<QueryResult>,
    ) -> Result<QueryResult, ClusterError> {
        let mut all_generated = Vec::new();

        for result in partial_results {
            if let QueryResult::Imagine(generated) = result {
                all_generated.extend(generated);
            }
        }

        // Sort by confidence
        all_generated.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(QueryResult::Imagine(all_generated))
    }

    /// Calculate confidence penalty based on missing partitions
    async fn calculate_confidence_penalty(
        &self,
        responding: usize,
        total: usize,
    ) -> f32 {
        if responding == total {
            return 0.0; // No penalty
        }

        // Base penalty: proportion of missing partitions
        let missing_ratio = (total - responding) as f32 / total as f32;

        // If we're partitioned, apply additional penalty
        let partition_penalty = if self.partition_detector.is_partitioned().await {
            0.1 // Additional 10% penalty during partition
        } else {
            0.0
        };

        // Total penalty is missing ratio + partition penalty, capped at 0.5
        (missing_ratio + partition_penalty).min(0.5)
    }

    /// Apply confidence penalty to all results
    fn apply_confidence_penalty(
        &self,
        mut result: QueryResult,
        penalty: f32,
    ) -> QueryResult {
        if penalty == 0.0 {
            return result;
        }

        match &mut result {
            QueryResult::Recall(memories) => {
                for memory in memories {
                    let (lower, upper) = memory.confidence;
                    memory.confidence = (
                        lower * (1.0 - penalty),
                        upper * (1.0 - penalty),
                    );
                }
            },
            QueryResult::Spread(activations) => {
                for activation in activations.values_mut() {
                    *activation *= 1.0 - penalty;
                }
            },
            QueryResult::Complete(patterns) => {
                for pattern in patterns {
                    pattern.confidence *= 1.0 - penalty;
                }
            },
            QueryResult::Consolidate(_) => {
                // Consolidation stats don't have confidence scores
            },
            QueryResult::Imagine(generated) => {
                for item in generated {
                    item.confidence *= 1.0 - penalty;
                }
            },
        }

        result
    }
}
```

### Distributed Query Executor

High-level orchestrator:

```rust
// engram-core/src/cluster/query/distributed.rs

use std::sync::Arc;

/// Distributed query executor coordinates scatter-gather
pub struct DistributedQueryExecutor {
    planner: Arc<QueryPlanner>,
    scatter: Arc<ScatterExecutor>,
    gather: Arc<GatherExecutor>,
    metrics: Arc<QueryMetrics>,
}

impl DistributedQueryExecutor {
    pub fn new(
        planner: Arc<QueryPlanner>,
        scatter: Arc<ScatterExecutor>,
        gather: Arc<GatherExecutor>,
        metrics: Arc<QueryMetrics>,
    ) -> Self {
        Self {
            planner,
            scatter,
            gather,
            metrics,
        }
    }

    /// Execute query with scatter-gather
    pub async fn execute(
        &self,
        query: Query,
    ) -> Result<AggregatedResult, ClusterError> {
        let start = Instant::now();

        // 1. Plan query (determine target partitions)
        let plan = self.planner.plan_query(&query).await?;

        debug!(
            "Query plan: {} partitions, strategy: {:?}",
            plan.expected_partitions, plan.strategy
        );

        // 2. Scatter query to all targets
        let responses = self.scatter.scatter(&plan, query.clone()).await;

        // 3. Gather and aggregate results
        let aggregated = self.gather.gather(&plan, responses).await?;

        let total_latency = start.elapsed();

        // Record overall metrics
        self.metrics.record_distributed_query(
            plan.strategy,
            total_latency,
            aggregated.responding_partitions,
            aggregated.failed_partitions,
        );

        debug!(
            "Query complete: {} responding, {} failed, {:.1}% confidence penalty, {:?} latency",
            aggregated.responding_partitions,
            aggregated.failed_partitions,
            aggregated.confidence_penalty * 100.0,
            total_latency,
        );

        Ok(aggregated)
    }
}
```

### Connection Pool

Reuse gRPC connections:

```rust
// engram-core/src/cluster/connection_pool.rs

use dashmap::DashMap;
use tokio::sync::RwLock;
use std::net::SocketAddr;

/// Connection pool for gRPC clients
pub struct ConnectionPool {
    /// Map from address to connection
    connections: DashMap<SocketAddr, Arc<RwLock<EngramClient>>>,

    /// Pool configuration
    config: ConnectionPoolConfig,
}

#[derive(Debug, Clone)]
pub struct ConnectionPoolConfig {
    /// Maximum connections per address
    pub max_connections: usize,

    /// Connection timeout
    pub connect_timeout: Duration,

    /// Keep-alive interval
    pub keep_alive: Duration,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 4,
            connect_timeout: Duration::from_secs(5),
            keep_alive: Duration::from_secs(60),
        }
    }
}

impl ConnectionPool {
    pub fn new(config: ConnectionPoolConfig) -> Self {
        Self {
            connections: DashMap::new(),
            config,
        }
    }

    /// Get or create connection to address
    pub async fn get(&self, addr: &SocketAddr) -> Result<EngramClient, ClusterError> {
        // Try existing connection
        if let Some(conn) = self.connections.get(addr) {
            return Ok(conn.read().await.clone());
        }

        // Create new connection
        let client = self.connect(addr).await?;

        // Store in pool
        self.connections.insert(*addr, Arc::new(RwLock::new(client.clone())));

        Ok(client)
    }

    async fn connect(&self, addr: &SocketAddr) -> Result<EngramClient, ClusterError> {
        let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{}", addr))
            .map_err(|e| ClusterError::ConnectionError(e.to_string()))?
            .connect_timeout(self.config.connect_timeout)
            .keep_alive_timeout(self.config.keep_alive);

        let channel = endpoint.connect().await
            .map_err(|e| ClusterError::ConnectionError(e.to_string()))?;

        Ok(EngramClient::new(channel))
    }

    /// Remove connection from pool (on error)
    pub fn remove(&self, addr: &SocketAddr) {
        self.connections.remove(addr);
    }
}
```

### Query Metrics

Track distributed query performance:

```rust
// engram-core/src/cluster/query/metrics.rs

use metrics::{counter, histogram, gauge};
use std::time::Duration;

pub struct QueryMetrics;

impl QueryMetrics {
    pub fn new() -> Self {
        Self
    }

    pub fn record_scatter_latency(&self, latency: Duration, fanout: usize) {
        histogram!("engram.cluster.query.scatter_latency_ms")
            .record(latency.as_millis() as f64);

        histogram!("engram.cluster.query.scatter_fanout")
            .record(fanout as f64);
    }

    pub fn record_partition_latency(&self, node_id: &str, latency: Duration) {
        histogram!(
            "engram.cluster.query.partition_latency_ms",
            "node_id" => node_id.to_string()
        ).record(latency.as_millis() as f64);
    }

    pub fn record_gather_stats(
        &self,
        responding: usize,
        failed: usize,
        penalty: f32,
    ) {
        histogram!("engram.cluster.query.responding_partitions")
            .record(responding as f64);

        histogram!("engram.cluster.query.failed_partitions")
            .record(failed as f64);

        histogram!("engram.cluster.query.confidence_penalty")
            .record(penalty as f64);
    }

    pub fn record_distributed_query(
        &self,
        strategy: ExecutionStrategy,
        latency: Duration,
        responding: usize,
        failed: usize,
    ) {
        let strategy_str = match strategy {
            ExecutionStrategy::SinglePartition => "single",
            ExecutionStrategy::MultiPartition => "multi",
            ExecutionStrategy::Broadcast => "broadcast",
        };

        histogram!(
            "engram.cluster.query.total_latency_ms",
            "strategy" => strategy_str
        ).record(latency.as_millis() as f64);

        counter!(
            "engram.cluster.query.total",
            "strategy" => strategy_str
        ).increment(1);

        if failed > 0 {
            counter!(
                "engram.cluster.query.partial_results",
                "strategy" => strategy_str
            ).increment(1);
        }
    }
}
```

## Files to Create

1. `engram-core/src/cluster/query/mod.rs` - Query module exports
2. `engram-core/src/cluster/query/planner.rs` - Query planning
3. `engram-core/src/cluster/query/scatter.rs` - Scatter executor
4. `engram-core/src/cluster/query/gather.rs` - Gather executor
5. `engram-core/src/cluster/query/distributed.rs` - Distributed executor
6. `engram-core/src/cluster/query/metrics.rs` - Query metrics
7. `engram-core/src/cluster/connection_pool.rs` - gRPC connection pool

## Files to Modify

1. `engram-core/src/cluster/mod.rs` - Export query module
2. `engram-core/src/query/executor.rs` - Call distributed executor when in cluster mode
3. `engram-proto/proto/engram/v1/service.proto` - Ensure query RPCs support distributed execution
4. `engram-core/src/metrics/mod.rs` - Export query metrics

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_query_planner_single_partition() {
        let planner = create_test_planner().await;

        let query = Query::Recall {
            space_id: "space1".to_string(),
            cue: "test".to_string(),
            limit: 10,
        };

        let plan = planner.plan_query(&query).await.unwrap();

        assert_eq!(plan.strategy, ExecutionStrategy::SinglePartition);
        assert_eq!(plan.targets.len(), 1);
        assert_eq!(plan.expected_partitions, 1);
    }

    #[tokio::test]
    async fn test_query_planner_multi_partition() {
        let planner = create_test_planner().await;

        let query = Query::Imagine {
            spaces: vec!["space1".to_string(), "space2".to_string()],
            prompt: "test".to_string(),
        };

        let plan = planner.plan_query(&query).await.unwrap();

        assert_eq!(plan.strategy, ExecutionStrategy::MultiPartition);
        assert!(plan.targets.len() >= 2);
        assert_eq!(plan.expected_partitions, plan.targets.len());
    }

    #[tokio::test]
    async fn test_confidence_penalty_calculation() {
        let gather = create_test_gather().await;

        // 3 out of 5 partitions responded
        let penalty = gather.calculate_confidence_penalty(3, 5).await;

        // Should be ~40% penalty (2/5 missing)
        assert!((penalty - 0.4).abs() < 0.05);
    }

    #[tokio::test]
    async fn test_merge_recall_deduplication() {
        let gather = create_test_gather().await;

        // Same memory from two replicas
        let memory1 = Memory {
            id: "mem1".to_string(),
            confidence: (0.8, 0.9),
            ..Default::default()
        };

        let memory2 = Memory {
            id: "mem1".to_string(),
            confidence: (0.7, 0.85),
            ..Default::default()
        };

        let partial_results = vec![
            QueryResult::Recall(vec![memory1.clone()]),
            QueryResult::Recall(vec![memory2]),
        ];

        let merged = gather.merge_results(partial_results).unwrap();

        if let QueryResult::Recall(memories) = merged {
            // Should have deduplicated
            assert_eq!(memories.len(), 1);
            // Should keep higher confidence version
            assert_eq!(memories[0].confidence.0, 0.8);
        } else {
            panic!("Expected Recall result");
        }
    }

    #[tokio::test]
    async fn test_merge_spread_activation_sum() {
        let gather = create_test_gather().await;

        let mut activations1 = HashMap::new();
        activations1.insert("node1".to_string(), 0.5);
        activations1.insert("node2".to_string(), 0.3);

        let mut activations2 = HashMap::new();
        activations2.insert("node2".to_string(), 0.2);
        activations2.insert("node3".to_string(), 0.4);

        let partial_results = vec![
            QueryResult::Spread(activations1),
            QueryResult::Spread(activations2),
        ];

        let merged = gather.merge_results(partial_results).unwrap();

        if let QueryResult::Spread(activations) = merged {
            assert_eq!(activations.len(), 3);
            assert!((activations["node1"] - 0.5).abs() < 0.01);
            assert!((activations["node2"] - 0.5).abs() < 0.01); // 0.3 + 0.2
            assert!((activations["node3"] - 0.4).abs() < 0.01);
        } else {
            panic!("Expected Spread result");
        }
    }
}
```

### Integration Tests

```rust
// engram-core/tests/distributed_query_integration.rs

#[tokio::test]
async fn test_scatter_gather_basic() {
    // Start 3-node cluster
    let cluster = TestCluster::new(3).await;

    // Create memory space on node 1
    cluster.create_space("space1", "node1").await;

    // Store memories
    cluster.store("space1", "memory1", embedding1).await;
    cluster.store("space1", "memory2", embedding2).await;

    // Query from node 2 (should route to node 1)
    let result = cluster.node(1).query(Query::Recall {
        space_id: "space1".to_string(),
        cue: "test".to_string(),
        limit: 10,
    }).await.unwrap();

    assert_eq!(result.responding_partitions, 1);
    assert_eq!(result.failed_partitions, 0);
    assert_eq!(result.confidence_penalty, 0.0);
}

#[tokio::test]
async fn test_scatter_gather_with_timeout() {
    let cluster = TestCluster::new(3).await;

    // Create space on node 1
    cluster.create_space("space1", "node1").await;

    // Inject network delay on node 1
    cluster.inject_delay("node1", Duration::from_secs(10)).await;

    // Query should timeout on node 1
    let result = cluster.node(0).query(Query::Recall {
        space_id: "space1".to_string(),
        cue: "test".to_string(),
        limit: 10,
    }).await.unwrap();

    // Should have failed partition
    assert_eq!(result.failed_partitions, 1);
    assert!(result.confidence_penalty > 0.0);
}

#[tokio::test]
async fn test_multi_partition_query() {
    let cluster = TestCluster::new(3).await;

    // Create spaces on different nodes
    cluster.create_space("space1", "node1").await;
    cluster.create_space("space2", "node2").await;
    cluster.create_space("space3", "node3").await;

    // Store memories in each space
    for i in 1..=3 {
        cluster.store(&format!("space{}", i), &format!("mem{}", i), embedding(i)).await;
    }

    // Query across all spaces
    let result = cluster.node(0).query(Query::Imagine {
        spaces: vec![
            "space1".to_string(),
            "space2".to_string(),
            "space3".to_string(),
        ],
        prompt: "test".to_string(),
    }).await.unwrap();

    assert_eq!(result.responding_partitions, 3);
    assert_eq!(result.failed_partitions, 0);
    assert_eq!(result.confidence_penalty, 0.0);
}

#[tokio::test]
async fn test_replica_fallback() {
    let cluster = TestCluster::new(3).await;

    // Create space with replication factor 2
    cluster.create_space_with_replicas("space1", "node1", &["node2"]).await;
    cluster.store("space1", "memory1", embedding1).await;

    // Kill primary
    cluster.kill_node("node1").await;

    // Query should fallback to replica
    let result = cluster.node(2).query(Query::Recall {
        space_id: "space1".to_string(),
        cue: "test".to_string(),
        limit: 10,
    }).await.unwrap();

    assert_eq!(result.responding_partitions, 1);
    assert!(result.result.is_ok());
}
```

### Performance Tests

```rust
#[tokio::test]
#[ignore] // Run manually
async fn test_distributed_query_latency() {
    let cluster = TestCluster::new(5).await;

    // Create space
    cluster.create_space("space1", "node1").await;

    // Baseline: single-node query latency
    let single_node_latency = cluster.measure_query_latency("space1", 1000).await;

    println!("Single-node p50: {:?}", single_node_latency.p50);
    println!("Single-node p99: {:?}", single_node_latency.p99);

    // Distributed: query from remote node
    let distributed_latency = cluster.measure_distributed_query_latency("space1", 1000).await;

    println!("Distributed p50: {:?}", distributed_latency.p50);
    println!("Distributed p99: {:?}", distributed_latency.p99);

    // Should be <2x slowdown
    let slowdown = distributed_latency.p99.as_millis() as f64 / single_node_latency.p99.as_millis() as f64;
    assert!(slowdown < 2.0, "Distributed query too slow: {:.2}x", slowdown);
}
```

## Dependencies

All dependencies already added in previous tasks (tokio, tonic, dashmap, futures).

## Acceptance Criteria

1. Query planner correctly identifies target partitions for all query types
2. Scatter executor dispatches queries in parallel with <1ms overhead
3. Gather executor merges results correctly:
   - Recall: deduplicate by memory ID, keep highest confidence
   - Spread: sum activations from different partitions
   - Complete/Consolidate: return single result (single-partition queries)
   - Imagine: concatenate and sort by confidence
4. Confidence penalty calculated proportionally to missing partitions
5. Timeouts don't block queries (partial results returned)
6. Connection pool reuses gRPC channels (no reconnection overhead)
7. Metrics track scatter fanout, partition latency, confidence penalty
8. Integration tests pass: scatter-gather, timeout handling, replica fallback

## Performance Targets

- Query latency <2x single-node for intra-partition queries
- Scatter overhead <1ms
- Gather overhead <0.5ms per partition
- Connection pool reuse >95% (minimal reconnections)
- Timeout detection within 100ms of threshold
- Memory overhead <10MB per connection

## Next Steps

After completing this task:
- Task 010 will test partition scenarios with network simulator
- Task 011 will validate consistency with Jepsen
- Future optimization: streaming scatter-gather (return results incrementally)
- Future optimization: adaptive routing (learn which partitions contribute most)
