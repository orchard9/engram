//! Cluster-aware router that proxies Engram operations to the appropriate node.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use engram_core::cluster::{ClusterError, PartitionState, membership::NodeInfo};
use engram_core::{MemorySpaceId, metrics};
use engram_proto::engram_service_client::EngramServiceClient;
use engram_proto::{
    RecallRequest as ProtoRecallRequest, RecallResponse as ProtoRecallResponse,
    RememberRequest as ProtoRememberRequest, RememberResponse as ProtoRememberResponse,
};
use futures::future::join_all;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::time::sleep;
use tonic::{
    Request, Status,
    transport::{Channel, Endpoint},
};
use tracing::debug;

use crate::cluster::{ClusterState, RouteDecision};

const ROUTER_REQUESTS_TOTAL: &str = "engram_router_requests_total";
const ROUTER_RETRIES_TOTAL: &str = "engram_router_retries_total";
const ROUTER_BREAKERS_OPEN: &str = "engram_router_circuit_breakers_open";
const ROUTER_REPLICA_FALLBACK_TOTAL: &str = "engram_router_replica_fallback_total";

/// Shared routing helper that reuses channels and enforces circuit breaking.
pub struct Router {
    cluster: Arc<ClusterState>,
    pool: ConnectionPool,
    breakers: DashMap<String, Arc<CircuitBreaker>>,
    metrics: RouterMetrics,
    config: RouterConfig,
}

impl Router {
    /// Create a new router backed by the provided cluster state.
    #[must_use]
    pub fn new(cluster: Arc<ClusterState>, config: RouterConfig) -> Self {
        Self {
            cluster,
            pool: ConnectionPool::new(config.connection_pool_size),
            breakers: DashMap::new(),
            metrics: RouterMetrics::new(),
            config,
        }
    }

    /// Returns the configured routing parameters.
    #[must_use]
    pub const fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Produces a default deadline for router-managed operations.
    #[must_use]
    pub fn default_deadline(&self) -> Instant {
        Instant::now() + self.config.request_timeout()
    }

    /// Determine whether the local node should service the write or proxy it.
    pub fn route_write(&self, space_id: &MemorySpaceId) -> Result<RouteDecision, ClusterError> {
        self.cluster.route_for_space(space_id)
    }

    /// Compute the read routing plan for a memory space using the supplied strategy.
    pub fn route_read(
        &self,
        space_id: &MemorySpaceId,
        strategy: ReadRoutingStrategy,
    ) -> Result<ReadPlan, ClusterError> {
        let assignment = self.cluster.plan_space(space_id)?;
        let local_id = self.cluster.node_id.as_str();
        let is_local_primary = assignment.primary.id == local_id;
        let is_local_replica = assignment
            .replicas
            .iter()
            .any(|replica| replica.id == local_id);

        match strategy {
            ReadRoutingStrategy::PrimaryOnly => {
                if is_local_primary {
                    Ok(ReadPlan::Local)
                } else {
                    self.guard_partition()?;
                    Ok(ReadPlan::Remote(RemoteReadPlan::single(
                        strategy,
                        assignment.primary,
                    )))
                }
            }
            ReadRoutingStrategy::NearestReplica => {
                if is_local_primary || is_local_replica {
                    return Ok(ReadPlan::Local);
                }
                self.guard_partition()?;
                let target = self
                    .choose_best_replica(space_id, &assignment.replicas)
                    .unwrap_or_else(|| assignment.primary.clone());
                Ok(ReadPlan::Remote(RemoteReadPlan::single(strategy, target)))
            }
            ReadRoutingStrategy::ScatterGather => {
                self.guard_partition()?;
                let mut targets = Vec::with_capacity(1 + assignment.replicas.len());
                targets.push(assignment.primary);
                targets.extend(assignment.replicas);
                Ok(ReadPlan::Remote(RemoteReadPlan::scatter(strategy, targets)))
            }
        }
    }

    /// Proxy a write routed away from the local node, retrying and falling back to replicas.
    pub async fn proxy_write(
        &self,
        space: &MemorySpaceId,
        route: &RouteDecision,
        request: ProtoRememberRequest,
        deadline: Instant,
    ) -> Result<(ProtoRememberResponse, NodeInfo), RouterError> {
        let RemoteTargets { primary, replicas } = match route {
            RouteDecision::Local => {
                return Err(RouterError::LocalOnly);
            }
            RouteDecision::Remote { primary, replicas } => RemoteTargets {
                primary: primary.clone(),
                replicas: replicas.clone(),
            },
        };

        let operation = RouterOperation::WritePrimary;
        let invoke = |mut client: EngramServiceClient<Channel>, req: ProtoRememberRequest| async move {
            client
                .remember(Request::new(req))
                .await
                .map(tonic::Response::into_inner)
        };

        match self
            .dispatch_grpc(&primary, request.clone(), operation, deadline, None, invoke)
            .await
        {
            Ok(response) => Ok((response, primary)),
            Err(err) => {
                if !self.config.replica_fallback_enabled {
                    return Err(err);
                }
                let mut last_err = Some(err);
                for replica in replicas {
                    if !self.replica_is_fresh(space, &replica) {
                        continue;
                    }
                    match self
                        .dispatch_grpc(
                            &replica,
                            request.clone(),
                            RouterOperation::WriteReplica,
                            deadline,
                            None,
                            invoke,
                        )
                        .await
                    {
                        Ok(response) => {
                            self.metrics.replica_fallback();
                            return Ok((response, replica));
                        }
                        Err(err) => last_err = Some(err),
                    }
                }
                Err(last_err.unwrap())
            }
        }
    }

    /// Proxy a recall routed away from the local node.
    pub async fn proxy_read(
        &self,
        space: &MemorySpaceId,
        plan: ReadPlan,
        request: ProtoRecallRequest,
        deadline: Instant,
    ) -> Result<(ProtoRecallResponse, Option<NodeInfo>), RouterError> {
        match plan {
            ReadPlan::Local => Err(RouterError::LocalOnly),
            ReadPlan::Remote(remote) if remote.scatter => {
                self.scatter_read(space, remote, request, deadline).await
            }
            ReadPlan::Remote(remote) => {
                let target =
                    remote
                        .targets
                        .first()
                        .cloned()
                        .ok_or(RouterError::NoHealthyReplicas {
                            space: space.clone(),
                        })?;
                let response = self
                    .dispatch_grpc(
                        &target,
                        request,
                        RouterOperation::ReadPrimary,
                        deadline,
                        Some(remote.strategy),
                        |mut client, req| async move {
                            client
                                .recall(Request::new(req))
                                .await
                                .map(tonic::Response::into_inner)
                        },
                    )
                    .await?;
                Ok((response, Some(target)))
            }
        }
    }

    fn guard_partition(&self) -> Result<(), ClusterError> {
        if let PartitionState::Partitioned {
            reachable_nodes,
            total_nodes,
            ..
        } = self.cluster.partition_detector.current_state()
        {
            return Err(ClusterError::Partitioned {
                reachable_nodes,
                total_nodes,
            });
        }
        Ok(())
    }

    fn breaker_for(&self, node_id: &str) -> Arc<CircuitBreaker> {
        self.breakers
            .entry(node_id.to_string())
            .or_insert_with(|| {
                Arc::new(CircuitBreaker::new(
                    self.config.circuit_breaker_failure_threshold,
                    self.config.circuit_breaker_reset_timeout(),
                ))
            })
            .clone()
    }

    async fn dispatch_grpc<R, Resp, F, Fut>(
        &self,
        node: &NodeInfo,
        request: R,
        operation: RouterOperation,
        deadline: Instant,
        strategy: Option<ReadRoutingStrategy>,
        invoke: F,
    ) -> Result<Resp, RouterError>
    where
        R: Clone + Send + 'static,
        Resp: Send + 'static,
        F: Fn(EngramServiceClient<Channel>, R) -> Fut + Copy + Send + 'static,
        Fut: std::future::Future<Output = Result<Resp, Status>> + Send,
    {
        let strategy_label = strategy.unwrap_or(ReadRoutingStrategy::PrimaryOnly);
        self.metrics.record_request(operation, strategy_label);
        let breaker = self.breaker_for(&node.id);
        let mut attempt = 0;
        let mut delay = self.config.retry_base_delay();

        loop {
            if Instant::now() >= deadline {
                return Err(RouterError::DeadlineExceeded {
                    node_id: node.id.clone(),
                });
            }

            if let Some(retry_after) = breaker.allow() {
                return Err(RouterError::CircuitOpen {
                    node_id: node.id.clone(),
                    retry_after,
                });
            }

            let channel = self
                .pool
                .get(node)
                .await
                .map_err(|error| RouterError::Connect {
                    node_id: node.id.clone(),
                    error,
                })?;
            let client = EngramServiceClient::new(channel);

            match invoke(client, request.clone()).await {
                Ok(response) => {
                    if breaker.record_success() {
                        self.metrics.breaker_closed();
                    }
                    return Ok(response);
                }
                Err(status) => {
                    if breaker.record_failure() {
                        self.metrics.breaker_opened();
                        debug!(node = %node.id, "router breaker opened");
                    }
                    attempt += 1;
                    if attempt > self.config.max_retries {
                        return Err(RouterError::RemoteRpc {
                            node_id: node.id.clone(),
                            status,
                        });
                    }
                    self.metrics.record_retry(operation);
                    let sleep_duration = jitter(delay);
                    let next_deadline = Instant::now() + sleep_duration;
                    if next_deadline > deadline {
                        return Err(RouterError::DeadlineExceeded {
                            node_id: node.id.clone(),
                        });
                    }
                    sleep(sleep_duration).await;
                    delay = std::cmp::min(delay * 2, self.config.retry_max_delay());
                }
            }
        }
    }

    async fn scatter_read(
        &self,
        space: &MemorySpaceId,
        plan: RemoteReadPlan,
        request: ProtoRecallRequest,
        deadline: Instant,
    ) -> Result<(ProtoRecallResponse, Option<NodeInfo>), RouterError> {
        if plan.targets.is_empty() {
            return Err(RouterError::NoHealthyReplicas {
                space: space.clone(),
            });
        }

        let mut tasks = Vec::with_capacity(plan.targets.len());
        for target in plan.targets {
            let req = request.clone();
            let router = self;
            tasks.push(async move {
                let result = router
                    .dispatch_grpc(
                        &target,
                        req,
                        RouterOperation::ReadScatter,
                        deadline,
                        Some(plan.strategy),
                        |mut client, req| async move {
                            client
                                .recall(Request::new(req))
                                .await
                                .map(tonic::Response::into_inner)
                        },
                    )
                    .await;
                result.map(|response| (target, response))
            });
        }

        let mut successes = Vec::new();
        let mut last_err = None;
        for result in join_all(tasks).await {
            match result {
                Ok(pair) => successes.push(pair),
                Err(err) => last_err = Some(err),
            }
        }

        if successes.is_empty() {
            return Err(last_err.unwrap_or(RouterError::NoHealthyReplicas {
                space: space.clone(),
            }));
        }

        let mut iter = successes.into_iter();
        let (first_node, first_response) = iter.next().unwrap();
        let mut merged = first_response;
        let mut seen_ids: HashSet<String> = merged
            .memories
            .iter()
            .map(|memory| memory.id.clone())
            .collect();

        for (_, response) in iter {
            append_recall(&mut merged, response, &mut seen_ids);
        }

        Ok((merged, Some(first_node)))
    }

    fn replica_is_fresh(&self, space: &MemorySpaceId, replica: &NodeInfo) -> bool {
        #[cfg(feature = "memory_mapped_persistence")]
        {
            if let Some(metadata) = &self.cluster.replication_metadata
                && let Some(lag) = metadata.replica_lag(space, &replica.id)
            {
                return lag.sequences_behind() <= self.config.replica_fallback_lag_sequences;
            }
            false
        }

        #[cfg(not(feature = "memory_mapped_persistence"))]
        {
            let _ = (space, replica);
            false
        }
    }

    fn choose_best_replica(
        &self,
        space: &MemorySpaceId,
        replicas: &[NodeInfo],
    ) -> Option<NodeInfo> {
        if replicas.is_empty() {
            return None;
        }

        #[cfg(feature = "memory_mapped_persistence")]
        {
            if let Some(metadata) = &self.cluster.replication_metadata {
                let mut candidates: Vec<(u64, NodeInfo)> = replicas
                    .iter()
                    .filter_map(|replica| {
                        metadata
                            .replica_lag(space, &replica.id)
                            .map(|lag| (lag.sequences_behind(), replica.clone()))
                    })
                    .collect();
                candidates.sort_by_key(|(lag, _)| *lag);
                if let Some((lag, node)) = candidates.into_iter().next()
                    && lag <= self.config.replica_fallback_lag_sequences
                {
                    return Some(node);
                }
            }
        }

        replicas.first().cloned()
    }

    /// Temporary helper used by subsystems (replication) that still need raw channels.
    pub async fn client(&self, node: &NodeInfo) -> Result<Channel, tonic::transport::Error> {
        self.pool.get(node).await
    }

    /// Snapshot the router's current health for diagnostics endpoints.
    #[must_use]
    pub fn health_snapshot(&self) -> RouterHealthSnapshot {
        let mut breakers = Vec::new();
        for entry in &self.breakers {
            match entry.value().snapshot() {
                CircuitBreakerSnapshot::Closed => {}
                CircuitBreakerSnapshot::Open(retry_after) => breakers.push(RouterBreakerSnapshot {
                    node_id: entry.key().clone(),
                    state: RouterBreakerState::Open,
                    retry_after_ms: Some(retry_after.as_millis() as u64),
                }),
                CircuitBreakerSnapshot::HalfOpen => breakers.push(RouterBreakerSnapshot {
                    node_id: entry.key().clone(),
                    state: RouterBreakerState::HalfOpen,
                    retry_after_ms: None,
                }),
            }
        }

        RouterHealthSnapshot {
            requests_total: self.metrics.requests_total(),
            retries_total: self.metrics.retries_total(),
            replica_fallback_total: self.metrics.replica_fallback_total(),
            open_breakers: self.metrics.open_breakers(),
            breakers,
        }
    }
}

fn jitter(base: Duration) -> Duration {
    let mut rng = rand::thread_rng();
    let factor: f64 = rng.r#gen();
    let millis = (base.as_millis() as f64 * factor) as u64;
    Duration::from_millis(millis.max(1))
}

fn append_recall(
    target: &mut ProtoRecallResponse,
    other: ProtoRecallResponse,
    seen: &mut HashSet<String>,
) {
    for memory in other.memories {
        if seen.insert(memory.id.clone()) {
            target.memories.push(memory);
        }
    }
    target.traces.extend(other.traces);
    if target.recall_confidence.is_none() {
        target.recall_confidence = other.recall_confidence;
    }
    if target.metadata.is_none() {
        target.metadata = other.metadata;
    }
}

struct RemoteTargets {
    primary: NodeInfo,
    replicas: Vec<NodeInfo>,
}

/// Routing decision describing how reads should be handled for a memory space.
#[derive(Debug, Clone)]
pub enum ReadPlan {
    /// Execute the read locally without proxying.
    Local,
    /// Proxy the read to one or more remote nodes according to the plan.
    Remote(RemoteReadPlan),
}

/// Concrete plan describing which nodes should service a remote read.
#[derive(Debug, Clone)]
pub struct RemoteReadPlan {
    strategy: ReadRoutingStrategy,
    targets: Vec<NodeInfo>,
    scatter: bool,
}

impl RemoteReadPlan {
    fn single(strategy: ReadRoutingStrategy, target: NodeInfo) -> Self {
        Self {
            strategy,
            targets: vec![target],
            scatter: false,
        }
    }

    const fn scatter(strategy: ReadRoutingStrategy, targets: Vec<NodeInfo>) -> Self {
        Self {
            strategy,
            targets,
            scatter: true,
        }
    }
}

/// Strategy hint for read routing.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadRoutingStrategy {
    /// Always talk to the primary node.
    PrimaryOnly,
    /// Prefer a replica (based on replication lag) when possible.
    NearestReplica,
    /// Fan the request out to all placement members and merge the responses.
    ScatterGather,
}

impl ReadRoutingStrategy {
    #[must_use]
    /// Human-readable label used for metrics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PrimaryOnly => "primary_only",
            Self::NearestReplica => "nearest_replica",
            Self::ScatterGather => "scatter_gather",
        }
    }
}

impl Default for ReadRoutingStrategy {
    fn default() -> Self {
        Self::NearestReplica
    }
}

/// Externally configurable router tuning parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RouterConfig {
    /// Number of cached tonic channels per remote node.
    pub connection_pool_size: usize,
    /// Deadline (milliseconds) applied to routed operations.
    pub request_timeout_ms: u64,
    /// Maximum number of retry attempts before surfacing an error.
    pub max_retries: usize,
    /// Initial backoff delay (milliseconds) for retry loops.
    pub retry_base_delay_ms: u64,
    /// Maximum backoff delay (milliseconds) for retry loops.
    pub retry_max_delay_ms: u64,
    /// Consecutive failures required before a circuit breaker trips.
    pub circuit_breaker_failure_threshold: u32,
    /// How long a breaker stays open before transitioning to half-open.
    pub circuit_breaker_reset_ms: u64,
    /// Whether write fallbacks to replicas are enabled.
    pub replica_fallback_enabled: bool,
    /// Lag threshold (in WAL sequences) that replicas must satisfy before handling writes.
    pub replica_fallback_lag_sequences: u64,
    /// Default strategy used when read-routing call sites do not specify one explicitly.
    pub default_read_strategy: ReadRoutingStrategy,
}

impl RouterConfig {
    #[must_use]
    /// Returns the deadline applied to routed RPCs.
    pub fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.request_timeout_ms.max(1))
    }

    #[must_use]
    /// Initial backoff delay used by the retry loop.
    pub fn retry_base_delay(&self) -> Duration {
        Duration::from_millis(self.retry_base_delay_ms.max(1))
    }

    #[must_use]
    /// Maximum backoff delay allowed during the retry loop.
    pub fn retry_max_delay(&self) -> Duration {
        Duration::from_millis(self.retry_max_delay_ms.max(self.retry_base_delay_ms.max(1)))
    }

    #[must_use]
    /// Duration that circuit breakers remain open before probing again.
    pub fn circuit_breaker_reset_timeout(&self) -> Duration {
        Duration::from_millis(self.circuit_breaker_reset_ms.max(1))
    }
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            connection_pool_size: 4,
            request_timeout_ms: 3_000,
            max_retries: 3,
            retry_base_delay_ms: 50,
            retry_max_delay_ms: 500,
            circuit_breaker_failure_threshold: 5,
            circuit_breaker_reset_ms: 5_000,
            replica_fallback_enabled: true,
            replica_fallback_lag_sequences: 32,
            default_read_strategy: ReadRoutingStrategy::NearestReplica,
        }
    }
}

#[derive(Debug)]
struct RouterMetrics {
    open_breakers: AtomicUsize,
    requests: AtomicU64,
    retries: AtomicU64,
    replica_fallbacks: AtomicU64,
}

impl RouterMetrics {
    const fn new() -> Self {
        Self {
            open_breakers: AtomicUsize::new(0),
            requests: AtomicU64::new(0),
            retries: AtomicU64::new(0),
            replica_fallbacks: AtomicU64::new(0),
        }
    }

    fn record_request(&self, operation: RouterOperation, strategy: ReadRoutingStrategy) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let labels = [
            ("operation", operation.as_str().to_string()),
            ("strategy", strategy.as_str().to_string()),
        ];
        metrics::increment_counter_with_labels(ROUTER_REQUESTS_TOTAL, 1, &labels);
    }

    fn record_retry(&self, operation: RouterOperation) {
        self.retries.fetch_add(1, Ordering::Relaxed);
        let labels = [("operation", operation.as_str().to_string())];
        metrics::increment_counter_with_labels(ROUTER_RETRIES_TOTAL, 1, &labels);
    }

    fn breaker_opened(&self) {
        let open = self.open_breakers.fetch_add(1, Ordering::Relaxed) + 1;
        metrics::record_gauge(ROUTER_BREAKERS_OPEN, open as f64);
    }

    fn breaker_closed(&self) {
        let mut current = self.open_breakers.load(Ordering::Relaxed);
        while current > 0 {
            match self.open_breakers.compare_exchange(
                current,
                current - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    metrics::record_gauge(ROUTER_BREAKERS_OPEN, (current - 1) as f64);
                    return;
                }
                Err(actual) => current = actual,
            }
        }
        metrics::record_gauge(ROUTER_BREAKERS_OPEN, 0.0);
    }

    fn replica_fallback(&self) {
        self.replica_fallbacks.fetch_add(1, Ordering::Relaxed);
        metrics::increment_counter(ROUTER_REPLICA_FALLBACK_TOTAL, 1);
    }

    fn open_breakers(&self) -> usize {
        self.open_breakers.load(Ordering::Relaxed)
    }

    fn requests_total(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    fn retries_total(&self) -> u64 {
        self.retries.load(Ordering::Relaxed)
    }

    fn replica_fallback_total(&self) -> u64 {
        self.replica_fallbacks.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone, Copy)]
enum RouterOperation {
    WritePrimary,
    WriteReplica,
    ReadPrimary,
    ReadScatter,
}

impl RouterOperation {
    const fn as_str(self) -> &'static str {
        match self {
            Self::WritePrimary => "write_primary",
            Self::WriteReplica => "write_replica",
            Self::ReadPrimary => "read_primary",
            Self::ReadScatter => "read_scatter",
        }
    }
}

/// Errors surfaced while proxying operations through the router.
#[derive(Debug, Error)]
pub enum RouterError {
    /// Router was asked to handle a request that must execute locally.
    #[error("router attempted to handle request locally")]
    LocalOnly,
    /// Target node tripped its circuit breaker.
    #[error("circuit breaker for node {node_id} is open; retry after {retry_after:?}")]
    CircuitOpen {
        /// Remote node identifier.
        node_id: String,
        /// Remaining cooldown before the breaker half-opens.
        retry_after: Duration,
    },
    /// The operation exceeded the configured deadline.
    #[error("deadline exceeded before node {node_id} responded")]
    DeadlineExceeded {
        /// Remote node identifier.
        node_id: String,
    },
    /// No replica satisfied the freshness requirements for the space.
    #[error("no healthy replicas available for space {space}")]
    NoHealthyReplicas {
        /// Memory space identifier.
        space: MemorySpaceId,
    },
    /// The remote RPC failed at the application layer.
    #[error("remote rpc to node {node_id} failed: {status}")]
    RemoteRpc {
        /// Remote node identifier.
        node_id: String,
        /// Tonic status returned by the peer.
        status: Status,
    },
    /// Unable to establish a transport channel to the remote node.
    #[error("failed connecting to node {node_id}: {error}")]
    Connect {
        /// Remote node identifier.
        node_id: String,
        /// Underlying transport error.
        error: tonic::transport::Error,
    },
}

struct CircuitBreaker {
    state: Mutex<CircuitBreakerState>,
    failure_threshold: u32,
    reset_timeout: Duration,
}

impl CircuitBreaker {
    fn new(failure_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            state: Mutex::new(CircuitBreakerState::Closed { failures: 0 }),
            failure_threshold: failure_threshold.max(1),
            reset_timeout,
        }
    }

    fn allow(&self) -> Option<Duration> {
        let mut guard = self.state.lock().unwrap();
        let result = match &mut *guard {
            CircuitBreakerState::Closed { .. } => None,
            CircuitBreakerState::Open { reopen_at } => {
                let now = Instant::now();
                if now >= *reopen_at {
                    *guard = CircuitBreakerState::HalfOpen {
                        probe_in_flight: true,
                    };
                    None
                } else {
                    Some(reopen_at.saturating_duration_since(now))
                }
            }
            CircuitBreakerState::HalfOpen { probe_in_flight } => {
                if *probe_in_flight {
                    Some(Duration::from_millis(10))
                } else {
                    *probe_in_flight = true;
                    None
                }
            }
        };
        drop(guard);
        result
    }

    fn record_success(&self) -> bool {
        let mut guard = self.state.lock().unwrap();
        match &mut *guard {
            CircuitBreakerState::Closed { failures } => {
                *failures = 0;
                false
            }
            CircuitBreakerState::Open { .. } | CircuitBreakerState::HalfOpen { .. } => {
                *guard = CircuitBreakerState::Closed { failures: 0 };
                true
            }
        }
    }

    fn record_failure(&self) -> bool {
        let mut guard = self.state.lock().unwrap();
        match &mut *guard {
            CircuitBreakerState::Closed { failures } => {
                *failures += 1;
                if *failures >= self.failure_threshold {
                    *guard = CircuitBreakerState::Open {
                        reopen_at: Instant::now() + self.reset_timeout,
                    };
                    true
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen { .. } | CircuitBreakerState::Open { .. } => {
                *guard = CircuitBreakerState::Open {
                    reopen_at: Instant::now() + self.reset_timeout,
                };
                true
            }
        }
    }

    fn snapshot(&self) -> CircuitBreakerSnapshot {
        let guard = self.state.lock().unwrap();
        match &*guard {
            CircuitBreakerState::Closed { .. } => CircuitBreakerSnapshot::Closed,
            CircuitBreakerState::Open { reopen_at } => {
                let now = Instant::now();
                CircuitBreakerSnapshot::Open(reopen_at.saturating_duration_since(now))
            }
            CircuitBreakerState::HalfOpen { .. } => CircuitBreakerSnapshot::HalfOpen,
        }
    }
}

enum CircuitBreakerState {
    Closed { failures: u32 },
    Open { reopen_at: Instant },
    HalfOpen { probe_in_flight: bool },
}

enum CircuitBreakerSnapshot {
    Closed,
    Open(Duration),
    HalfOpen,
}

struct ConnectionPool {
    max_per_node: usize,
    channels: DashMap<String, Channel>,
}

impl ConnectionPool {
    fn new(max_per_node: usize) -> Self {
        Self {
            max_per_node: max_per_node.max(1),
            channels: DashMap::new(),
        }
    }

    async fn get(&self, node: &NodeInfo) -> Result<Channel, tonic::transport::Error> {
        if let Some(entry) = self.channels.get(node.id.as_str()) {
            return Ok(entry.value().clone());
        }

        let endpoint = Endpoint::from_shared(format!("http://{}", node.api_addr))?;
        let channel = endpoint.connect().await?;
        if self.channels.len() >= self.max_per_node {
            let value = self.channels.iter().next();
            if let Some(entry) = value {
                self.channels.remove(entry.key());
            }
        }
        self.channels.insert(node.id.clone(), channel.clone());
        Ok(channel)
    }
}

/// Health snapshot exposed via diagnostics endpoints.
#[derive(Debug, Clone)]
pub struct RouterHealthSnapshot {
    /// Total number of remote requests routed since startup.
    pub requests_total: u64,
    /// Total retry attempts triggered by the router.
    pub retries_total: u64,
    /// Replica fallback operations performed when primaries failed.
    pub replica_fallback_total: u64,
    /// Number of circuit breakers currently open.
    pub open_breakers: usize,
    /// Snapshot of breakers that are open or half-open.
    pub breakers: Vec<RouterBreakerSnapshot>,
}

/// Per-node breaker state summary.
#[derive(Debug, Clone)]
pub struct RouterBreakerSnapshot {
    /// Identifier of the node governed by this breaker.
    pub node_id: String,
    /// Current breaker state.
    pub state: RouterBreakerState,
    /// Optional milliseconds until the breaker attempts recovery.
    pub retry_after_ms: Option<u64>,
}

/// Current breaker state label.
#[derive(Debug, Clone, Copy)]
pub enum RouterBreakerState {
    /// Breaker is open and rejecting requests.
    Open,
    /// Breaker is probing (half-open) to determine if the node recovered.
    HalfOpen,
}

#[cfg(test)]
mod tests {
    use super::*;
    use engram_core::cluster::config::{PartitionConfig, ReplicationConfig, SwimConfig};
    use engram_core::cluster::rebalance::RebalanceCoordinator;
    use engram_core::cluster::{
        PartitionDetector, SpaceAssignmentManager, SpaceAssignmentPlanner, SplitBrainDetector,
        membership::{MembershipUpdate, NodeState, SwimMembership},
    };
    use std::sync::Arc;
    use std::thread;

    fn build_cluster_state() -> Arc<ClusterState> {
        let local = NodeInfo::new(
            "node-local",
            "127.0.0.1:7100".parse().unwrap(),
            "127.0.0.1:5200".parse().unwrap(),
            None,
            None,
        );
        let membership = Arc::new(SwimMembership::new(local.clone(), SwimConfig::default()));
        membership.apply_updates(vec![
            MembershipUpdate {
                node: NodeInfo::new(
                    "node-alpha",
                    "127.0.0.2:7100".parse().unwrap(),
                    "127.0.0.2:5200".parse().unwrap(),
                    None,
                    None,
                ),
                state: NodeState::Alive,
                incarnation: 1,
            },
            MembershipUpdate {
                node: NodeInfo::new(
                    "node-beta",
                    "127.0.0.3:7100".parse().unwrap(),
                    "127.0.0.3:5200".parse().unwrap(),
                    None,
                    None,
                ),
                state: NodeState::Alive,
                incarnation: 1,
            },
        ]);

        let replication = ReplicationConfig {
            factor: 2,
            ..ReplicationConfig::default()
        };
        let planner = Arc::new(SpaceAssignmentPlanner::new(
            Arc::clone(&membership),
            &replication,
        ));
        let assignments = Arc::new(SpaceAssignmentManager::new(
            Arc::clone(&planner),
            &replication,
        ));
        let (rebalance, _rx) =
            RebalanceCoordinator::new(Arc::clone(&assignments), Arc::clone(&membership), 4);
        let partition_detector = Arc::new(PartitionDetector::new(
            Arc::clone(&membership),
            PartitionConfig::default(),
        ));
        let split_brain = Arc::new(SplitBrainDetector::new(local.id.clone()));

        Arc::new(ClusterState {
            node_id: local.id,
            membership,
            assignments,
            replication,
            partition_detector,
            split_brain,
            rebalance,
            #[cfg(feature = "memory_mapped_persistence")]
            replication_metadata: None,
        })
    }

    fn build_router() -> Router {
        Router::new(build_cluster_state(), RouterConfig::default())
    }

    fn find_remote_space(router: &Router) -> MemorySpaceId {
        for index in 0..1024 {
            let candidate = format!("space-{index}");
            let space = MemorySpaceId::try_from(candidate.as_str()).unwrap();
            if matches!(router.route_write(&space), Ok(RouteDecision::Remote { .. })) {
                return space;
            }
        }
        panic!("unable to find remote assignment for router test");
    }

    #[test]
    fn circuit_breaker_opens_after_threshold() {
        let breaker = CircuitBreaker::new(2, Duration::from_millis(5));
        assert!(breaker.allow().is_none());
        assert!(!breaker.record_failure());
        assert!(breaker.allow().is_none());
        assert!(breaker.record_failure());
        assert!(breaker.allow().is_some());
    }

    #[test]
    fn circuit_breaker_closes_after_success() {
        let breaker = CircuitBreaker::new(1, Duration::from_millis(5));
        assert!(breaker.record_failure());
        assert!(breaker.allow().is_some());
        thread::sleep(Duration::from_millis(6));
        assert!(breaker.allow().is_none());
        assert!(breaker.record_success());
        assert!(breaker.allow().is_none());
    }

    #[test]
    fn route_read_scatter_gather_targets_multiple_nodes() {
        let router = build_router();
        let space = find_remote_space(&router);
        match router
            .route_read(&space, ReadRoutingStrategy::ScatterGather)
            .expect("scatter plan")
        {
            ReadPlan::Remote(plan) => {
                assert!(plan.scatter, "scatter flag must be set");
                assert!(plan.targets.len() >= 2);
            }
            ReadPlan::Local => panic!("expected remote scatter plan"),
        }
    }

    #[test]
    fn health_snapshot_reports_open_breakers() {
        let router = build_router();
        let breaker = router.breaker_for("node-alpha");
        for _ in 0..router.config.circuit_breaker_failure_threshold {
            if breaker.record_failure() {
                router.metrics.breaker_opened();
            }
        }
        let snapshot = router.health_snapshot();
        assert_eq!(snapshot.open_breakers, 1);
        assert_eq!(snapshot.breakers.len(), 1);
        assert_eq!(snapshot.breakers[0].node_id, "node-alpha");
    }
}
