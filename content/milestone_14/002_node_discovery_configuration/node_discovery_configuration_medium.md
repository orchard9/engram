# How Distributed Systems Find Each Other: Node Discovery in Engram

Every distributed system faces the same chicken-and-egg problem: before nodes can coordinate, they need to find each other. But how do you find other nodes when you don't yet know who they are?

It's like walking into a networking event where you don't know anyone. You could wander around randomly hoping to bump into people, but that's inefficient. Instead, you might check the attendee list, ask the organizers for introductions, or look for people wearing name tags. Distributed systems use similar strategies, and each has different tradeoffs.

For Engram, our cognitive graph database, we need discovery that works everywhere: developer laptops, Kubernetes clusters, and on-premise data centers. This article walks through how we built a flexible discovery system that adapts to different environments while keeping configuration simple.

## The Discovery Challenge

Imagine you're starting a new Engram node. You've just launched the process, allocated memory, opened network sockets. Now what? To join an existing cluster, you need to know:

1. Are there any other nodes running?
2. If yes, where are they?
3. How do I contact them?

The answers depend entirely on your deployment environment.

On a developer's laptop running docker-compose, "where are they" might be localhost:7946 and localhost:7947. In Kubernetes, it's pod IPs that change every time pods restart. On AWS EC2, it's instances tagged with a specific value. On-premise, it might be hardcoded DNS names like engram-1.corp.local.

We need one discovery system that handles all these cases without requiring different code for each environment.

## Approach 1: Static Seeds (Simple and Reliable)

The simplest approach is hardcoding peer addresses in configuration:

```toml
[cluster]
mode = "distributed"
node_id = "node-1"
listen_addr = "0.0.0.0:7946"

seeds = [
  "engram-1.local:7946",
  "engram-2.local:7946",
  "engram-3.local:7946"
]
```

When the node starts, it tries to connect to each seed address in order. As soon as one succeeds, it receives the full membership list and joins the cluster.

This works beautifully for small, stable deployments. It's what Cassandra used for years, and what Consul still supports. The advantage is predictability - operators know exactly how nodes will find each other.

The downside is rigidity. If you want to add nodes, you need to update configuration files. If IP addresses change (common in cloud environments), your seeds become stale. If all seed nodes fail permanently, new nodes can't join.

Despite these limitations, static seeds remain the best fallback option. They work everywhere, require no infrastructure beyond DNS, and are trivially testable.

## Approach 2: DNS SRV Records (Cloud-Native)

DNS SRV (service) records are designed exactly for this use case. You create records that list all instances of a service:

```
_engram._tcp.cluster.local. 60 IN SRV 0 0 7946 node1.cluster.local.
_engram._tcp.cluster.local. 60 IN SRV 0 0 7946 node2.cluster.local.
_engram._tcp.cluster.local. 60 IN SRV 0 0 7946 node3.cluster.local.
```

Nodes query `_engram._tcp.cluster.local` and get back all current members. As nodes join and leave, you update DNS records.

Kubernetes makes this trivial with headless services:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: engram-cluster
spec:
  clusterIP: None  # Headless service
  selector:
    app: engram
  ports:
  - name: cluster
    port: 7946
```

Now nodes can query `_cluster._tcp.engram-cluster.default.svc.cluster.local` and automatically discover all pods.

The benefit: no manual configuration needed. Autoscaling works transparently - new pods appear in DNS automatically. The DNS server handles failover and load balancing.

The challenge: DNS caching can cause stale data. If a node fails and is removed from DNS, other nodes might not see the update for up to TTL seconds (typically 30-60s). This is usually acceptable, since SWIM's failure detection operates independently and will mark the failed node as dead anyway.

## Approach 3: Cloud Provider APIs (Fully Dynamic)

For maximum flexibility, query the cloud provider's API directly. In Kubernetes:

```rust
use kube::{Client, Api, api::ListParams};
use k8s_openapi::api::core::v1::Pod;

async fn discover_via_kubernetes() -> Result<Vec<SocketAddr>> {
    let client = Client::try_default().await?;
    let pods: Api<Pod> = Api::namespaced(client, "default");

    let lp = ListParams::default()
        .labels("app=engram");

    let pod_list = pods.list(&lp).await?;

    pod_list.items
        .iter()
        .filter_map(|pod| {
            pod.status.as_ref()?.pod_ip.as_ref()
        })
        .map(|ip| format!("{}:7946", ip).parse())
        .collect()
}
```

This queries the Kubernetes API for all pods with label `app=engram`, extracts their IPs, and returns them as discovery candidates.

The power of this approach: it works with any Kubernetes deployment, handles scaling automatically, and requires zero manual configuration beyond the label selector.

The tradeoff: it requires cloud provider credentials, adds a dependency on the cloud API being available, and different clouds need different code paths (Kubernetes, AWS EC2, Azure, GCP).

## Engram's Hybrid Strategy: Best of All Worlds

Rather than choosing one approach, Engram supports all three with automatic fallback:

```rust
pub async fn discover_peers(&self) -> Result<Vec<SocketAddr>> {
    // Try static seeds first (instant, always works)
    if let Some(seeds) = &self.config.seeds {
        if !seeds.is_empty() {
            return Ok(seeds.clone());
        }
    }

    // Try DNS discovery (fast, cloud-friendly)
    if let Some(dns_srv) = &self.config.dns_srv {
        match query_dns_srv(dns_srv).await {
            Ok(addrs) if !addrs.is_empty() => return Ok(addrs),
            _ => {} // Fall through to next method
        }
    }

    // Try cloud provider API (comprehensive, requires auth)
    if let Some(cloud) = &self.config.cloud_discovery {
        match cloud.discover().await {
            Ok(addrs) if !addrs.is_empty() => return Ok(addrs),
            _ => {} // Fall through to next method
        }
    }

    // No peers found - run as single-node cluster
    info!("No peers discovered, starting as single-node cluster");
    Ok(vec![])
}
```

Methods are tried in order of latency and reliability. Static seeds are instant and never fail (though addresses might be unreachable). DNS is fast but can return stale data. Cloud APIs are comprehensive but can be rate-limited or unavailable.

This provides a graceful degradation path. In production Kubernetes, nodes use the API for discovery. If the API is temporarily unavailable, they fall back to DNS. If DNS fails, they use static seeds as a last resort.

## Configuration Design: Simple Defaults, Powerful Options

Engram's configuration philosophy is "single-node by default, distributed opt-in." The minimal config doesn't mention clustering at all:

```toml
[storage]
path = "/var/lib/engram"

[api]
grpc_port = 8080
```

This runs as a single-node instance. To enable distributed mode, add a `[cluster]` section:

```toml
[cluster]
mode = "distributed"
node_id = "node-1"
listen_addr = "0.0.0.0:7946"
seeds = ["node-0:7946"]
```

For Kubernetes deployments, environment variables override config file settings:

```bash
export ENGRAM_CLUSTER_MODE=distributed
export ENGRAM_CLUSTER_NODE_ID=${POD_NAME}
export ENGRAM_CLUSTER_DISCOVERY_METHOD=kubernetes
export ENGRAM_CLUSTER_DISCOVERY_NAMESPACE=default
export ENGRAM_CLUSTER_DISCOVERY_LABEL_SELECTOR=app=engram
```

This makes Helm charts straightforward:

```yaml
env:
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: ENGRAM_CLUSTER_MODE
  value: "distributed"
- name: ENGRAM_CLUSTER_NODE_ID
  value: "$(POD_NAME)"
- name: ENGRAM_CLUSTER_DISCOVERY_METHOD
  value: "kubernetes"
```

The same binary works in development (with static seeds), staging (with DNS), and production (with Kubernetes API). Operators choose the discovery method that fits their infrastructure.

## Join Protocol: From Discovery to Membership

Discovery finds potential peers. The join protocol turns that into actual cluster membership.

Step 1: Contact a seed. Send a JOIN message to one of the discovered addresses:

```rust
struct JoinRequest {
    node_id: NodeId,
    listen_addr: SocketAddr,
    protocol_version: u32,
}
```

Step 2: Receive membership snapshot. The seed responds with the current cluster state:

```rust
struct JoinResponse {
    members: Vec<MemberInfo>,
    cluster_metadata: ClusterMetadata,
}

struct MemberInfo {
    node_id: NodeId,
    addr: SocketAddr,
    state: MemberState,
    incarnation: u64,
}
```

This bootstraps the new node's membership table. It now knows about all cluster members, not just the seed it contacted.

Step 3: Announce to cluster. The seed gossips a membership update about the new joiner:

```rust
let update = MembershipUpdate {
    node: new_node_id,
    state: MemberState::Alive,
    incarnation: 1,
};

self.gossip_queue.push(update);
```

Within seconds (O(log N) gossip rounds), all nodes know about the new member.

## Observability: Making Discovery Debuggable

Discovery failures are some of the most frustrating operational issues. A node can't join the cluster, but why? Wrong DNS record? Cloud API permissions? Network firewall?

Engram exposes discovery status via an HTTP endpoint:

```json
GET /cluster/discovery/health

{
  "enabled": true,
  "method": "kubernetes",
  "last_attempt": "2025-10-24T10:15:30Z",
  "last_success": "2025-10-24T10:15:30Z",
  "peers_discovered": 5,
  "discovery_methods_tried": [
    {
      "method": "static_seeds",
      "success": false,
      "error": "no seeds configured"
    },
    {
      "method": "dns",
      "success": false,
      "error": "NXDOMAIN"
    },
    {
      "method": "kubernetes",
      "success": true,
      "peers": 5,
      "duration_ms": 45
    }
  ]
}
```

This shows operators exactly what happened: static seeds weren't configured, DNS lookup failed, Kubernetes API succeeded and found 5 peers in 45ms.

Metrics track discovery performance over time:

```
discovery_attempts_total{method="kubernetes"} 1234
discovery_successes_total{method="kubernetes"} 1230
discovery_failures_total{method="kubernetes"} 4
discovery_duration_seconds{method="kubernetes",quantile="0.5"} 0.042
discovery_duration_seconds{method="kubernetes",quantile="0.99"} 0.180
```

Alert when `peers_discovered` drops to zero - that indicates the node is isolated and can't join the cluster.

## Testing Discovery Without Production Infrastructure

Testing discovery is tricky because it depends on external systems. Our approach: mock each layer independently.

For DNS, use a test DNS server:

```rust
#[tokio::test]
async fn test_dns_srv_discovery() {
    let dns_server = TestDnsServer::start().await;
    dns_server.add_srv_record("_engram._tcp.test", vec![
        (0, 0, 7946, "node1.test"),
        (0, 0, 7946, "node2.test"),
    ]);

    let discovery = DnsDiscovery::new("_engram._tcp.test");
    let peers = discovery.discover().await.unwrap();

    assert_eq!(peers.len(), 2);
}
```

For Kubernetes, use an actual test cluster or mock the API server:

```rust
#[tokio::test]
async fn test_kubernetes_discovery() {
    let k8s = TestKubernetesCluster::start().await;
    k8s.create_pod("engram-1", "10.0.1.1").await;
    k8s.create_pod("engram-2", "10.0.1.2").await;

    let discovery = KubernetesDiscovery::new("default", "app=engram");
    let peers = discovery.discover().await.unwrap();

    assert_eq!(peers.len(), 2);
}
```

Integration tests use docker-compose to spin up real multi-node clusters and verify that nodes discover and join each other automatically.

## Looking Forward

Node discovery is the entry point to distributed operation. Once nodes can find each other, the rest of the distributed machinery builds on top:

- SWIM protocol for failure detection
- Gossip for membership updates
- Partition assignment for distributing memory spaces
- Replication for durability

But all of that requires nodes to first know who their peers are. By supporting multiple discovery methods with automatic fallback, Engram works in diverse environments while keeping configuration simple.

The hybrid approach means developers can start with static seeds in docker-compose, move to DNS in staging, and use cloud provider APIs in production - all without changing code. Discovery becomes invisible infrastructure that just works.

Like neural development, distributed system formation requires multiple complementary mechanisms. Engram's discovery layer provides the initial connections that allow the cognitive graph to span multiple nodes, forming a distributed brain ready to consolidate memories at scale.
