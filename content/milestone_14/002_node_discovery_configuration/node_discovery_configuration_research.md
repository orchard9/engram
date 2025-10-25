# Research: Node Discovery and Configuration for Distributed Systems

## The Bootstrap Problem

You've implemented SWIM for failure detection. Nodes can now detect when peers go offline. But there's a chicken-and-egg problem: how does a node find peers in the first place?

When you start a single Engram instance, it's easy - there are no peers. But when you want to add a second node to create a cluster, that new node needs to know about at least one existing member. This is the bootstrap problem, and every distributed system solves it differently.

## Discovery Patterns in Production Systems

### Static Configuration (Simple, Brittle)

The simplest approach: hardcode peer addresses in a config file.

```toml
[cluster]
seeds = [
  "10.0.1.10:7946",
  "10.0.1.11:7946",
  "10.0.1.12:7946"
]
```

When a node starts, it contacts these seed addresses to join the cluster. This works for small, stable deployments but breaks down when:
- IP addresses change (cloud autoscaling)
- Seed nodes fail permanently
- You want dynamic cluster sizing

Static configuration is what Cassandra and early versions of Consul used. It's reliable but requires manual updates when the cluster topology changes.

### DNS-Based Discovery (Cloud-Friendly)

Use DNS SRV records to list cluster members:

```
_engram._tcp.cluster.local. 60 IN SRV 0 0 7946 node1.cluster.local.
_engram._tcp.cluster.local. 60 IN SRV 0 0 7946 node2.cluster.local.
_engram._tcp.cluster.local. 60 IN SRV 0 0 7946 node3.cluster.local.
```

Nodes query DNS to find peers. Kubernetes does this automatically with headless services. AWS also supports this via Cloud Map.

Benefits:
- Works with cloud autoscaling
- DNS handles IP changes
- Standard protocol, no custom infrastructure

Drawbacks:
- DNS caching can cause stale information
- DNS itself is a single point of failure (though highly available)
- Requires DNS infrastructure configuration

### Cloud Provider APIs (Kubernetes, AWS, Azure)

Query the cloud provider's API to find other instances:

```rust
// Kubernetes example
async fn discover_peers_k8s() -> Result<Vec<SocketAddr>> {
    let client = kube::Client::try_default().await?;
    let pods: Api<Pod> = Api::namespaced(client, "default");

    let lp = ListParams::default()
        .labels("app=engram");

    let pod_list = pods.list(&lp).await?;

    pod_list.items
        .iter()
        .filter_map(|pod| pod.status.pod_ip)
        .map(|ip| format!("{}:7946", ip).parse())
        .collect()
}
```

This is how many production systems work in Kubernetes:
- Prometheus uses the Kubernetes API for service discovery
- Consul can use various cloud provider APIs
- etcd uses Kubernetes StatefulSets for predictable naming

Benefits:
- Fully dynamic, handles autoscaling
- No additional configuration needed
- Works with cloud-native deployments

Drawbacks:
- Requires cloud provider credentials
- Different code paths for different environments
- API rate limits can be an issue

### Multicast Discovery (Local Networks Only)

Use UDP multicast to broadcast "I'm here" messages:

```rust
const MULTICAST_ADDR: &str = "239.255.42.99:7946";

async fn multicast_announce(node_addr: SocketAddr) -> Result<()> {
    let socket = UdpSocket::bind("0.0.0.0:0").await?;
    socket.join_multicast_v4(
        "239.255.42.99".parse()?,
        "0.0.0.0".parse()?,
    )?;

    let announcement = DiscoveryMessage {
        node_id: NodeId::generate(),
        addr: node_addr,
        timestamp: Instant::now(),
    };

    loop {
        socket.send_to(
            &serialize(&announcement)?,
            MULTICAST_ADDR,
        ).await?;
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
```

This is how mDNS (Bonjour, Avahi) works for local service discovery. It's great for development and small deployments, but doesn't work across data centers or cloud regions (multicast is typically blocked).

## Engram's Hybrid Approach

For Engram, we need a solution that works in multiple environments:
- Development: single machine, multiple processes
- Production: cloud deployment with autoscaling
- Edge: on-premise deployment with static IPs

The solution: layered discovery with fallback.

### Layer 1: Static Seeds (Always Available)

Every node can optionally configure seed addresses:

```toml
[cluster]
mode = "distributed"
node_id = "node-1"
listen_addr = "0.0.0.0:7946"

# Optional seed nodes for bootstrapping
seeds = [
  "engram-1.local:7946",
  "engram-2.local:7946"
]
```

If seeds are provided, the node tries them first. This handles the simple case and provides a fallback when other discovery methods fail.

### Layer 2: DNS SRV Records (Cloud-Friendly)

If `cluster.dns_discovery` is enabled, query DNS SRV records:

```toml
[cluster.discovery]
dns_srv = "_engram._tcp.cluster.local"
dns_refresh_interval = "30s"
```

The node periodically queries DNS to find new peers and detect removed ones. This integrates seamlessly with Kubernetes headless services and AWS Cloud Map.

### Layer 3: Cloud Provider API (Dynamic Environments)

For Kubernetes deployments:

```toml
[cluster.discovery]
method = "kubernetes"
namespace = "default"
label_selector = "app=engram"
```

The node uses the Kubernetes API to discover pods with matching labels. This handles autoscaling automatically.

For AWS:

```toml
[cluster.discovery]
method = "aws"
region = "us-east-1"
tag_key = "EngramCluster"
tag_value = "production"
```

Query EC2 instances with matching tags.

### Discovery Priority and Fallback

Nodes try methods in order until they find at least one peer:

```rust
pub async fn discover_peers(&self) -> Result<Vec<SocketAddr>> {
    // Try static seeds first
    if let Some(seeds) = &self.config.seeds {
        if !seeds.is_empty() {
            return Ok(seeds.clone());
        }
    }

    // Try DNS discovery
    if let Some(dns_srv) = &self.config.dns_srv {
        match query_dns_srv(dns_srv).await {
            Ok(addrs) if !addrs.is_empty() => return Ok(addrs),
            _ => {} // Fall through to next method
        }
    }

    // Try cloud provider API
    if let Some(cloud) = &self.config.cloud_discovery {
        match cloud.discover().await {
            Ok(addrs) if !addrs.is_empty() => return Ok(addrs),
            _ => {} // Fall through to next method
        }
    }

    // No peers found - start as single-node cluster
    Ok(vec![])
}
```

This provides flexibility without complexity. Most deployments will use one method, but the fallback ensures robustness.

## Configuration Design Philosophy

Engram follows a key principle: **single-node remains a first-class citizen**. Distributed mode is optional, not mandatory.

### Default: Single-Node Mode

```toml
# Minimal config - runs as single node
[storage]
path = "/var/lib/engram"

[api]
grpc_port = 8080
http_port = 8081
```

No cluster configuration means single-node mode. All the distributed machinery stays dormant.

### Opt-In: Distributed Mode

```toml
[cluster]
mode = "distributed"  # This one line enables distributed mode
node_id = "node-1"
listen_addr = "0.0.0.0:7946"
seeds = ["node-0:7946"]
```

Adding the `[cluster]` section activates distributed features. The node joins the cluster on startup.

### Environment Variables for Cloud Deployments

In Kubernetes, you often can't hardcode configuration. Support environment variables:

```bash
ENGRAM_CLUSTER_MODE=distributed
ENGRAM_CLUSTER_NODE_ID=${POD_NAME}
ENGRAM_CLUSTER_LISTEN_ADDR=0.0.0.0:7946
ENGRAM_CLUSTER_DISCOVERY_METHOD=kubernetes
ENGRAM_CLUSTER_DISCOVERY_NAMESPACE=default
ENGRAM_CLUSTER_DISCOVERY_LABEL_SELECTOR=app=engram
```

These override the config file, making Helm charts and container orchestration straightforward.

## Join Protocol: From Discovery to Membership

Discovery gives you a list of potential peers. The join protocol turns that into cluster membership.

### Step 1: Contact Seeds

```rust
pub async fn join_cluster(&mut self) -> Result<()> {
    let seeds = self.discover_peers().await?;

    if seeds.is_empty() {
        info!("No seeds found, starting as single-node cluster");
        self.state = ClusterState::SingleNode;
        return Ok(());
    }

    // Try each seed until one succeeds
    for seed in seeds {
        match self.send_join_request(seed).await {
            Ok(members) => {
                self.initialize_membership(members);
                self.state = ClusterState::Member;
                return Ok(());
            }
            Err(e) => {
                warn!("Failed to join via seed {}: {}", seed, e);
                continue;
            }
        }
    }

    Err(Error::NoSeedsReachable)
}
```

### Step 2: Receive Membership Snapshot

The seed responds with the current membership list:

```rust
struct JoinResponse {
    members: Vec<MemberInfo>,
    cluster_state: ClusterMetadata,
}

struct MemberInfo {
    node_id: NodeId,
    addr: SocketAddr,
    state: MemberState,
    sequence: u64,
}
```

This bootstraps the new node's membership table. From here, SWIM's gossip protocol keeps it up to date.

### Step 3: Announce to Cluster

The seed gossips a membership update about the new joiner:

```rust
let update = MembershipUpdate {
    node: new_node_id,
    state: MemberState::Alive,
    sequence: self.next_sequence(),
};

self.piggyback_queue.push(update);
```

Within O(log N) gossip rounds, all nodes know about the new member.

## Health Checks and Observability

Discovery isn't just about startup. Nodes need to monitor cluster health continuously.

### Discovery Health Endpoint

```rust
#[derive(Serialize)]
struct DiscoveryHealth {
    enabled: bool,
    method: String,
    seeds_reachable: usize,
    seeds_total: usize,
    last_discovery: Option<Instant>,
    next_refresh: Option<Instant>,
}

async fn discovery_health() -> Json<DiscoveryHealth> {
    // Return current discovery status
}
```

Expose this via HTTP API at `/cluster/discovery/health`. Operators can check if discovery is working and debug connectivity issues.

### Metrics

Track discovery performance:

```rust
// Prometheus metrics
discovery_attempts_total: Counter
discovery_successes_total: Counter
discovery_duration_seconds: Histogram
peers_discovered: Gauge
```

Alert when `peers_discovered` drops to zero (cluster isolation).

## Testing Discovery Logic

Discovery involves external systems (DNS, Kubernetes API), making tests challenging.

### Unit Tests with Mocks

Mock the DNS resolver and cloud provider APIs:

```rust
#[cfg(test)]
mod tests {
    struct MockDnsResolver {
        responses: HashMap<String, Vec<SocketAddr>>,
    }

    #[tokio::test]
    async fn test_dns_discovery_fallback() {
        let mut resolver = MockDnsResolver::new();
        resolver.add_response("_engram._tcp.local", vec![
            "10.0.1.1:7946".parse().unwrap(),
            "10.0.1.2:7946".parse().unwrap(),
        ]);

        let discovery = DnsDiscovery::new(resolver);
        let peers = discovery.discover().await.unwrap();

        assert_eq!(peers.len(), 2);
    }
}
```

### Integration Tests in Docker

Spin up a multi-node cluster with docker-compose:

```yaml
version: '3'
services:
  engram-1:
    image: engram:test
    environment:
      - ENGRAM_CLUSTER_MODE=distributed
      - ENGRAM_CLUSTER_NODE_ID=node-1
      - ENGRAM_CLUSTER_SEEDS=engram-2:7946

  engram-2:
    image: engram:test
    environment:
      - ENGRAM_CLUSTER_MODE=distributed
      - ENGRAM_CLUSTER_NODE_ID=node-2
```

Verify that nodes discover each other and form a cluster.

## Conclusion: Discovery as Foundation

Node discovery is the entry point to distributed operation. Get it wrong, and nodes can't join the cluster. Get it right, and the rest of the distributed machinery can build on a solid foundation.

For Engram, the hybrid approach provides flexibility:
- Static seeds for simple deployments
- DNS for cloud-native environments
- API-based discovery for dynamic scaling

Combined with SWIM's gossip protocol, this gives Engram a cluster membership layer that's both robust and cloud-friendly. Once nodes can find each other and detect failures, we can build higher-level distributed features: partitioning, replication, and distributed query execution.
