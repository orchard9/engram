# Research: Distributed Routing in Cognitive Graph Databases

## The Routing Problem

A client sends a query: "recall memories related to coffee in Space 5." Which node processes this? The client doesn't know the cluster topology. The receiving node might not own Space 5. How does the query reach the right node without central coordination?

This is distributed routing: directing operations to the nodes that own the relevant data, transparently to the client.

## Routing Strategies

**Client-side routing**: Clients query a metadata service to find nodes, connect directly. Minimizes hops but requires clients to understand topology.

**Proxy routing**: A stateless proxy layer receives all requests, looks up owners, forwards requests. Adds hop but keeps clients simple.

**Peer-to-peer routing**: Any node can receive requests and forward to owners. No special infrastructure needed.

Engram uses peer-to-peer routing for simplicity and fault tolerance.

## Routing Table: Space to Node Mapping

Each node maintains a routing table mapping spaces to primaries and replicas:

```rust
struct RoutingTable {
    // Fast lookup: space_id -> (primary, replicas)
    assignments: DashMap<SpaceId, Assignment>,
    // Reverse index: node_id -> spaces it owns
    node_spaces: DashMap<NodeId, HashSet<SpaceId>>,
    // Versioned for staleness detection
    version: AtomicU64,
}

struct Assignment {
    primary: NodeId,
    replicas: SmallVec<[NodeId; 2]>,
    version: u64,
}
```

Assignments propagate via gossip. When a space moves (rebalancing or failover), an assignment update gossips through the cluster. Within log(N) rounds, all nodes know the new assignment.

## Routing Logic: Read vs Write

Writes must go to the primary (only primary can mutate). Reads can go to any replica:

```rust
async fn route_operation(&self, space_id: SpaceId, op: Operation) -> Result<Response> {
    let assignment = self.routing_table.get(space_id)?;

    match op {
        Operation::Write(_) => {
            // Writes go to primary
            self.send_to_node(assignment.primary, op).await
        }
        Operation::Read(_) => {
            // Reads can go to any replica, prefer local
            if self.is_local(assignment.primary) {
                self.execute_locally(op).await
            } else {
                // Try replicas, prefer closest
                self.send_to_closest(&assignment.replicas, op).await
            }
        }
    }
}
```

This balances load: writes concentrate on primaries (lower throughput), reads distribute across replicas (higher throughput).

## Handling Stale Routing Metadata

Gossip-based propagation means nodes might have stale assignments temporarily. Node A thinks Space 5 is on Node B, but it moved to Node C.

The solution: version checking and self-correcting forwarding.

When Node B receives a request for Space 5 but isn't the current owner:

```rust
fn handle_request(&self, space_id: SpaceId, request: Request) -> Response {
    let local_ownership = self.check_ownership(space_id);

    match local_ownership {
        Ownership::Primary => self.execute_request(request),
        Ownership::Replica => {
            // Can serve reads, must forward writes
            match request.operation {
                Operation::Read => self.execute_request(request),
                Operation::Write => self.forward_to_primary(space_id, request),
            }
        }
        Ownership::NotOwned { actual_owner, version } => {
            Response::WrongNode {
                actual_owner,
                version,
                hint: "update your routing table",
            }
        }
    }
}
```

The client receives WrongNode response, updates its routing table, and retries. Self-healing in one extra round trip.

## Connection Pooling for Efficiency

Opening TCP connections is expensive (3-way handshake, TLS if enabled). Routing to different nodes on every request would create connection churn.

Solution: maintain persistent connection pools:

```rust
struct ConnectionPool {
    // One gRPC channel per node
    channels: DashMap<NodeId, Arc<Channel>>,
    // Health tracking
    health: DashMap<NodeId, HealthStatus>,
}

async fn get_or_create_channel(&self, node_id: NodeId) -> Arc<Channel> {
    if let Some(channel) = self.channels.get(&node_id) {
        return channel.clone();
    }

    let addr = self.resolve_node_addr(node_id).await?;
    let channel = Channel::from_shared(addr)?
        .connect_timeout(Duration::from_secs(5))
        .connect()
        .await?;

    self.channels.insert(node_id, Arc::new(channel));
    channel
}
```

Channels are reused across requests. gRPC uses HTTP/2 multiplexing, so multiple concurrent requests share one connection.

## Retry Logic with Backoff

Network is unreliable. Requests might timeout, connections might break. Routing layer needs retry logic:

```rust
async fn send_with_retry(&self, node: NodeId, req: Request) -> Result<Response> {
    let mut backoff = ExponentialBackoff::new(10, 1000); // 10ms to 1s

    for attempt in 0..3 {
        match self.send_to_node(node, req.clone()).await {
            Ok(resp) => return Ok(resp),
            Err(e) if e.is_retriable() => {
                let delay = backoff.next();
                tokio::time::sleep(delay).await;
                continue;
            }
            Err(e) => return Err(e), // Non-retriable, fail immediately
        }
    }

    Err(Error::MaxRetriesExceeded)
}
```

Exponential backoff prevents thundering herds. If a node is overloaded, retries back off progressively.

## Academic Foundation

- **Chord**: Stoica et al. (2001) - DHT-based routing
- **Dynamo**: DeCandia et al. (2007) - gossip-based routing table propagation
- **Raft**: Ongaro & Ousterhout (2014) - routing to leader in consensus systems

Engram's approach is closest to Dynamo: eventually consistent routing tables, self-correcting on staleness.
