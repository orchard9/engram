# Task 002: Node Discovery and Configuration

**Status**: Complete — cluster discovery (static + DNS), configuration validation, HTTP inspection routes, and CLI tooling are in place. Consul/registry support remains deferred until the control-plane milestone.
**Estimated Duration**: 2 days
**Dependencies**: Task 001 (SWIM membership)
**Owner**: TBD

## Current Status (Nov 2025)
- ✅ `ClusterConfig` accepted via CLI config/defaults, including `cluster.discovery` enum.
- ✅ `ClusterContext` seeds SWIM membership from static seeds or DNS SRV (`engram-cli/src/cluster.rs`).
- ✅ Docker compose nodes ship `engram.toml` with `seed_nodes = ["engram-node1:7946", ...]` and run healthy under `docker compose up -d`.
- ⏭️ Consul/registry integration is intentionally deferred; `DiscoveryConfig::Consul` currently returns `DnsUnavailable` and should be documented as such until the control-plane milestone picks it up.
- ✅ `engram config validate` and `engram status --json` now surface discovery/feature mismatches (advertise address, DNS feature flag, Consul mode rejection) before startup.

### Next Steps
1. Documented (docs + CLI help) that DNS discovery requires the `cluster_discovery_dns` feature and that Consul is deferred.
2. Validation tooling now fails fast on discovery/advertise mismatches, and `engram status --json` reports the same issues for automation.
3. `/cluster/health` + `/cluster/nodes` routes expose membership snapshots rooted in `ClusterState`.
4. Integration tests cover hostname resolution (`build_discovery`), SWIM bootstrap (`initialize_cluster`), and HTTP cluster endpoints.

## Completion Notes

- Documentation now calls out the DNS feature flag requirement and Consul deferral, and the packaged `cluster.toml` template includes the reminder.
- `engram config validate` + `engram status --json` lint cluster fields (seed syntax, advertise overrides, feature mismatches) to catch errors before startup.
- `/cluster/health` and `/cluster/nodes` Axum routes provide operators with SWIM stats + member snapshots, backed by new API tests.
- Additional tests exercise hostname-based discovery, `initialize_cluster` bootstrap behaviour, and the HTTP inspection surface.

## Objective

Implement node discovery mechanisms for cluster formation, runtime configuration management for single-node vs cluster mode, and health monitoring integration with existing metrics system.

## Technical Specification

### Discovery Mechanisms

Support two discovery strategies (no external registry/Consul):

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

#### 3. Raft Control Plane (Authoritative State)

Deferred to the control-plane milestone. For Task 002, document that assignments still rely on the local `SpaceAssignmentPlanner` fed by SWIM membership snapshots.

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
    // Removed Consul; Raft control plane relies on static/DNS discovery.
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

Integrate SWIM/RAFT health into existing metrics system (no Consul hooks).

### HTTP Endpoint for Cluster Health

Expose `/cluster/health` and `/cluster/nodes` via `axum` so operators can query membership/health without shell access. The endpoints should serialize the `ClusterState::membership.stats()` output plus the individual `NodeInfo` snapshots. Wire them into the existing router once the handlers are implemented.

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
4. `engram-core/src/cluster/raft.rs` - Raft control plane node
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
4. Document Raft/control-plane work as future scope (no partial implementations).
5. Health monitoring tracks cluster state and is exposed through `/cluster/health`.
6. HTTP `/cluster/health` and `/cluster/nodes` endpoints return accurate status snapshots.
7. `engram config validate` (and server startup) fail fast on missing advertise addresses or discovery feature mismatches.
8. Metrics integrate with existing Prometheus/Grafana (cluster series labeled by `node_id`).

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
