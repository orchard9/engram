# Task 002: Node Discovery and Configuration

**Status**: Pending
**Estimated Duration**: 2 days
**Dependencies**: Task 001 (SWIM membership)
**Owner**: TBD

## Objective

Implement node discovery mechanisms for cluster formation, runtime configuration management for single-node vs cluster mode, and health monitoring integration with existing metrics system.

## Technical Specification

### Discovery Mechanisms

Support three discovery strategies:

#### 1. Static Seed List (Production)

Most reliable for production deployments:

```toml
# engram.toml
[cluster]
enabled = true
discovery = "static"

[cluster.static]
seed_nodes = [
    "engram-1.prod.example.com:7946",
    "engram-2.prod.example.com:7946",
    "engram-3.prod.example.com:7946"
]
```

#### 2. DNS SRV Records (Kubernetes/Cloud)

For dynamic environments:

```toml
[cluster]
enabled = true
discovery = "dns"

[cluster.dns]
service = "engram-cluster.default.svc.cluster.local"
port = 7946
refresh_interval_sec = 30
```

Implementation:

```rust
// engram-core/src/cluster/discovery/dns.rs

use trust_dns_resolver::TokioAsyncResolver;
use std::net::SocketAddr;

pub struct DnsDiscovery {
    resolver: TokioAsyncResolver,
    service: String,
    port: u16,
    refresh_interval: Duration,
}

impl DnsDiscovery {
    pub async fn discover(&self) -> Result<Vec<SocketAddr>, ClusterError> {
        // Resolve SRV records
        let srv_records = self.resolver
            .srv_lookup(&self.service)
            .await
            .map_err(|e| ClusterError::DnsError(e.to_string()))?;

        let mut addrs = Vec::new();
        for srv in srv_records.iter() {
            // Resolve A/AAAA records for target
            let lookup = self.resolver
                .lookup_ip(srv.target().to_utf8())
                .await
                .map_err(|e| ClusterError::DnsError(e.to_string()))?;

            for ip in lookup.iter() {
                addrs.push(SocketAddr::new(ip, srv.port()));
            }
        }

        Ok(addrs)
    }

    pub async fn watch(&self, callback: impl Fn(Vec<SocketAddr>) + Send + 'static) {
        let mut interval = tokio::time::interval(self.refresh_interval);

        loop {
            interval.tick().await;

            match self.discover().await {
                Ok(addrs) => callback(addrs),
                Err(e) => error!("DNS discovery failed: {}", e),
            }
        }
    }
}
```

#### 3. Consul/etcd Integration (Enterprise)

For environments with existing service discovery:

```rust
// engram-core/src/cluster/discovery/consul.rs

use consul::Client as ConsulClient;

pub struct ConsulDiscovery {
    client: ConsulClient,
    service_name: String,
    tag: Option<String>,
}

impl ConsulDiscovery {
    pub async fn discover(&self) -> Result<Vec<SocketAddr>, ClusterError> {
        let services = self.client
            .catalog()
            .service(&self.service_name, self.tag.as_deref())
            .await
            .map_err(|e| ClusterError::ConsulError(e.to_string()))?;

        let addrs = services
            .iter()
            .map(|s| {
                let ip = s.ServiceAddress.parse().unwrap_or(s.Address.parse().unwrap());
                SocketAddr::new(ip, s.ServicePort as u16)
            })
            .collect();

        Ok(addrs)
    }

    pub async fn register(&self, node: &NodeInfo) -> Result<(), ClusterError> {
        let registration = consul::ServiceRegistration {
            Name: self.service_name.clone(),
            ID: node.id.clone(),
            Address: node.addr.ip().to_string(),
            Port: node.addr.port() as i32,
            Tags: vec!["engram".to_string()],
            Check: Some(consul::Check {
                HTTP: format!("http://{}:{}/health", node.api_addr.ip(), node.api_addr.port()),
                Interval: "10s".to_string(),
                ..Default::default()
            }),
        };

        self.client.agent().register(registration).await
            .map_err(|e| ClusterError::ConsulError(e.to_string()))?;

        Ok(())
    }
}
```

### Configuration Management

Unified configuration supporting both modes:

```rust
// engram-core/src/cluster/config.rs

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Duration;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClusterConfig {
    /// Enable cluster mode (default: false for single-node)
    #[serde(default)]
    pub enabled: bool,

    /// Node ID (auto-generated if empty)
    #[serde(default)]
    pub node_id: String,

    /// Discovery strategy
    #[serde(default)]
    pub discovery: DiscoveryConfig,

    /// SWIM protocol settings
    #[serde(default)]
    pub swim: SwimConfig,

    /// Replication settings
    #[serde(default)]
    pub replication: ReplicationConfig,

    /// Network settings
    #[serde(default)]
    pub network: NetworkConfig,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Single-node by default
            node_id: String::new(),
            discovery: DiscoveryConfig::default(),
            swim: SwimConfig::default(),
            replication: ReplicationConfig::default(),
            network: NetworkConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DiscoveryConfig {
    Static {
        seed_nodes: Vec<String>,
    },
    Dns {
        service: String,
        port: u16,
        #[serde(default = "default_refresh_interval")]
        refresh_interval_sec: u64,
    },
    Consul {
        addr: String,
        service_name: String,
        tag: Option<String>,
    },
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self::Static {
            seed_nodes: vec![],
        }
    }
}

fn default_refresh_interval() -> u64 {
    30
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReplicationConfig {
    /// Number of replicas per memory space (default: 2)
    #[serde(default = "default_replication_factor")]
    pub factor: usize,

    /// Timeout for replication writes (default: 1s)
    #[serde(default = "default_replication_timeout")]
    pub timeout_ms: u64,

    /// Strategy for choosing replicas
    #[serde(default)]
    pub placement: PlacementStrategy,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            factor: 2,
            timeout_ms: 1000,
            placement: PlacementStrategy::Random,
        }
    }
}

fn default_replication_factor() -> usize {
    2
}

fn default_replication_timeout() -> u64 {
    1000
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PlacementStrategy {
    /// Random replica selection
    Random,
    /// Rack-aware (avoid same rack)
    RackAware,
    /// Zone-aware (avoid same availability zone)
    ZoneAware,
}

impl Default for PlacementStrategy {
    fn default() -> Self {
        Self::Random
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NetworkConfig {
    /// SWIM protocol bind address
    #[serde(default = "default_swim_addr")]
    pub swim_bind: String,

    /// gRPC API bind address
    #[serde(default = "default_api_addr")]
    pub api_bind: String,

    /// Maximum message size (bytes)
    #[serde(default = "default_max_message_size")]
    pub max_message_size: usize,

    /// Connection pool size per remote node
    #[serde(default = "default_connection_pool_size")]
    pub connection_pool_size: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            swim_bind: "0.0.0.0:7946".to_string(),
            api_bind: "0.0.0.0:50051".to_string(),
            max_message_size: 4 * 1024 * 1024, // 4MB
            connection_pool_size: 4,
        }
    }
}

fn default_swim_addr() -> String {
    "0.0.0.0:7946".to_string()
}

fn default_api_addr() -> String {
    "0.0.0.0:50051".to_string()
}

fn default_max_message_size() -> usize {
    4 * 1024 * 1024
}

fn default_connection_pool_size() -> usize {
    4
}
```

### Health Monitoring

Integrate cluster health into existing metrics system:

```rust
// engram-core/src/cluster/health.rs

use crate::metrics::health::HealthMetrics;
use std::sync::Arc;

pub struct ClusterHealth {
    membership: Arc<SwimMembership>,
    metrics: Arc<HealthMetrics>,
}

impl ClusterHealth {
    pub fn new(membership: Arc<SwimMembership>, metrics: Arc<HealthMetrics>) -> Self {
        Self { membership, metrics }
    }

    /// Start background health monitoring
    pub async fn start_monitoring(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(10));

        loop {
            interval.tick().await;
            self.collect_health_metrics().await;
        }
    }

    async fn collect_health_metrics(&self) {
        let stats = self.compute_cluster_stats();

        // Update metrics
        self.metrics.set_cluster_size(stats.total_nodes);
        self.metrics.set_alive_nodes(stats.alive_nodes);
        self.metrics.set_suspect_nodes(stats.suspect_nodes);
        self.metrics.set_dead_nodes(stats.dead_nodes);

        // Check health thresholds
        if stats.alive_ratio() < 0.5 {
            error!(
                "Cluster health critical: only {}/{} nodes alive",
                stats.alive_nodes, stats.total_nodes
            );
        } else if stats.alive_ratio() < 0.8 {
            warn!(
                "Cluster health degraded: {}/{} nodes alive",
                stats.alive_nodes, stats.total_nodes
            );
        }
    }

    fn compute_cluster_stats(&self) -> ClusterStats {
        let mut stats = ClusterStats::default();

        for entry in self.membership.members.iter() {
            stats.total_nodes += 1;

            match entry.value().state {
                NodeState::Alive => stats.alive_nodes += 1,
                NodeState::Suspect => stats.suspect_nodes += 1,
                NodeState::Dead | NodeState::Left => stats.dead_nodes += 1,
            }
        }

        stats
    }

    /// Get detailed cluster health report
    pub fn health_report(&self) -> ClusterHealthReport {
        let stats = self.compute_cluster_stats();

        ClusterHealthReport {
            status: if stats.alive_ratio() >= 0.8 {
                HealthStatus::Healthy
            } else if stats.alive_ratio() >= 0.5 {
                HealthStatus::Degraded
            } else {
                HealthStatus::Critical
            },
            total_nodes: stats.total_nodes,
            alive_nodes: stats.alive_nodes,
            suspect_nodes: stats.suspect_nodes,
            dead_nodes: stats.dead_nodes,
            nodes: self.membership.members.iter()
                .map(|e| {
                    let node = e.value();
                    NodeHealthInfo {
                        id: node.id.clone(),
                        addr: node.addr.to_string(),
                        state: node.state,
                        last_seen: node.last_update.elapsed().as_secs(),
                        spaces: node.spaces.clone(),
                    }
                })
                .collect(),
        }
    }
}

#[derive(Debug, Default)]
struct ClusterStats {
    total_nodes: usize,
    alive_nodes: usize,
    suspect_nodes: usize,
    dead_nodes: usize,
}

impl ClusterStats {
    fn alive_ratio(&self) -> f64 {
        if self.total_nodes == 0 {
            return 1.0;
        }
        self.alive_nodes as f64 / self.total_nodes as f64
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterHealthReport {
    pub status: HealthStatus,
    pub total_nodes: usize,
    pub alive_nodes: usize,
    pub suspect_nodes: usize,
    pub dead_nodes: usize,
    pub nodes: Vec<NodeHealthInfo>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Critical,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeHealthInfo {
    pub id: String,
    pub addr: String,
    pub state: NodeState,
    pub last_seen: u64, // seconds
    pub spaces: Vec<String>,
}
```

### HTTP Endpoint for Cluster Health

Add to existing HTTP API:

```rust
// engram-cli/src/http/cluster.rs

use axum::{Json, extract::State};
use std::sync::Arc;

pub async fn cluster_health(
    State(health): State<Arc<ClusterHealth>>
) -> Json<ClusterHealthReport> {
    Json(health.health_report())
}

pub async fn cluster_nodes(
    State(membership): State<Arc<SwimMembership>>
) -> Json<Vec<NodeInfo>> {
    let nodes = membership.members.iter()
        .map(|e| e.value().clone())
        .collect();
    Json(nodes)
}

// Add to router
pub fn cluster_routes() -> Router {
    Router::new()
        .route("/cluster/health", get(cluster_health))
        .route("/cluster/nodes", get(cluster_nodes))
}
```

### Startup Sequence

```rust
// engram-cli/src/cluster.rs

pub async fn initialize_cluster(config: ClusterConfig) -> Result<ClusterContext, Error> {
    if !config.enabled {
        info!("Starting in single-node mode");
        return Ok(ClusterContext::SingleNode);
    }

    info!("Starting in cluster mode");

    // 1. Generate or load node ID
    let node_id = if config.node_id.is_empty() {
        let id = uuid::Uuid::new_v4().to_string();
        info!("Generated node ID: {}", id);
        id
    } else {
        config.node_id.clone()
    };

    // 2. Parse bind addresses
    let swim_addr: SocketAddr = config.network.swim_bind.parse()?;
    let api_addr: SocketAddr = config.network.api_bind.parse()?;

    // 3. Create SWIM membership
    let membership = Arc::new(
        SwimMembership::new(node_id.clone(), swim_addr, config.swim).await?
    );

    // 4. Create discovery service
    let discovery = create_discovery(&config.discovery)?;

    // 5. Discover and join seed nodes
    let seed_addrs = discovery.discover().await?;
    for addr in seed_addrs {
        match membership.join(addr).await {
            Ok(()) => info!("Joined cluster via {}", addr),
            Err(e) => warn!("Failed to join via {}: {}", addr, e),
        }
    }

    // 6. Start health monitoring
    let health = Arc::new(ClusterHealth::new(
        membership.clone(),
        // Use existing HealthMetrics from M6
        Arc::new(HealthMetrics::global()),
    ));
    tokio::spawn({
        let health = health.clone();
        async move {
            health.start_monitoring().await;
        }
    });

    // 7. Start SWIM protocol loops
    start_swim_background_tasks(membership.clone()).await;

    // 8. Register with service discovery
    if let Some(registration) = discovery.supports_registration() {
        registration.register(&NodeInfo {
            id: node_id.clone(),
            addr: swim_addr,
            api_addr,
            state: NodeState::Alive,
            incarnation: 0,
            last_update: Instant::now(),
            spaces: vec![],
        }).await?;
    }

    Ok(ClusterContext::Distributed {
        node_id,
        membership,
        health,
        discovery,
    })
}

fn create_discovery(config: &DiscoveryConfig) -> Result<Box<dyn Discovery>, Error> {
    match config {
        DiscoveryConfig::Static { seed_nodes } => {
            Ok(Box::new(StaticDiscovery::new(seed_nodes)?))
        },
        DiscoveryConfig::Dns { service, port, refresh_interval_sec } => {
            Ok(Box::new(DnsDiscovery::new(service, *port, Duration::from_secs(*refresh_interval_sec))?))
        },
        DiscoveryConfig::Consul { addr, service_name, tag } => {
            Ok(Box::new(ConsulDiscovery::new(addr, service_name, tag.clone())?))
        },
    }
}

pub enum ClusterContext {
    SingleNode,
    Distributed {
        node_id: String,
        membership: Arc<SwimMembership>,
        health: Arc<ClusterHealth>,
        discovery: Box<dyn Discovery>,
    },
}
```

## Files to Create

1. `engram-core/src/cluster/discovery/mod.rs` - Discovery trait
2. `engram-core/src/cluster/discovery/static_discovery.rs` - Static seed list
3. `engram-core/src/cluster/discovery/dns.rs` - DNS SRV discovery
4. `engram-core/src/cluster/discovery/consul.rs` - Consul integration
5. `engram-core/src/cluster/health.rs` - Cluster health monitoring
6. `engram-core/src/cluster/config.rs` - Configuration structs
7. `engram-cli/src/cluster.rs` - Cluster initialization
8. `engram-cli/src/http/cluster.rs` - HTTP endpoints

## Files to Modify

1. `engram-cli/src/main.rs` - Call `initialize_cluster`
2. `engram-cli/src/http/mod.rs` - Add cluster routes
3. `engram-core/src/metrics/health.rs` - Add cluster metrics
4. `engram-core/Cargo.toml` - Add discovery dependencies

## Testing Strategy

### Unit Tests

```rust
#[tokio::test]
async fn test_static_discovery() {
    let discovery = StaticDiscovery::new(vec![
        "node1:7946".to_string(),
        "node2:7946".to_string(),
    ]).unwrap();

    let addrs = discovery.discover().await.unwrap();
    assert_eq!(addrs.len(), 2);
}

#[tokio::test]
async fn test_cluster_health_reporting() {
    let membership = Arc::new(SwimMembership::new_test());
    let health = ClusterHealth::new(membership.clone(), Arc::new(HealthMetrics::new()));

    // Add nodes in different states
    membership.add_node(NodeInfo { state: NodeState::Alive, ..Default::default() });
    membership.add_node(NodeInfo { state: NodeState::Suspect, ..Default::default() });
    membership.add_node(NodeInfo { state: NodeState::Dead, ..Default::default() });

    let report = health.health_report();
    assert_eq!(report.alive_nodes, 1);
    assert_eq!(report.suspect_nodes, 1);
    assert_eq!(report.dead_nodes, 1);
    assert_eq!(report.status, HealthStatus::Critical); // <50% alive
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_cluster_initialization() {
    let config = ClusterConfig {
        enabled: true,
        discovery: DiscoveryConfig::Static {
            seed_nodes: vec!["127.0.0.1:7946".to_string()],
        },
        ..Default::default()
    };

    let ctx = initialize_cluster(config).await.unwrap();

    match ctx {
        ClusterContext::Distributed { node_id, .. } => {
            assert!(!node_id.is_empty());
        },
        _ => panic!("Expected distributed context"),
    }
}
```

## Dependencies

Add to `engram-core/Cargo.toml`:

```toml
[dependencies]
# Discovery
trust-dns-resolver = { version = "0.23", optional = true }
consul = { version = "0.4", optional = true }

[features]
cluster_discovery_dns = ["trust-dns-resolver"]
cluster_discovery_consul = ["consul"]
```

## Acceptance Criteria

1. Single-node mode works with `cluster.enabled = false`
2. Static discovery connects to seed nodes
3. DNS SRV discovery resolves Kubernetes services
4. Consul integration registers and discovers nodes
5. Health monitoring tracks cluster state
6. HTTP `/cluster/health` endpoint returns accurate status
7. Configuration validated at startup with helpful errors
8. Metrics integrated with existing Prometheus/Grafana

## Performance Targets

- Cluster initialization completes within 5 seconds
- DNS discovery refresh <100ms
- Health monitoring overhead <0.1% CPU
- Configuration parsing <1ms

## Next Steps

After completing this task:
- Task 003 will add failure handling
- Task 004 will assign memory spaces based on discovered nodes
- HTTP/gRPC APIs will expose cluster status
