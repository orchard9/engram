# Research: Asynchronous Replication for Cognitive Systems

## The Durability Challenge

Memory must persist. When an Engram node stores an episodic memory, that memory should survive node failures. Replication provides durability: write to multiple nodes so that if one fails, others have the data.

The challenge: synchronous replication (wait for all replicas to acknowledge) adds latency. Asynchronous replication (don't wait) risks data loss if the primary fails before replicating.

## Asynchronous Replication Architecture

Engram chooses asynchronous replication for write latency. The protocol:

1. Client writes memory to primary node
2. Primary appends to local write-ahead log (WAL)
3. Primary acknowledges write immediately (< 10ms)
4. Background task ships WAL entries to replicas
5. Replicas apply and acknowledge asynchronously

This trades durability for latency. If the primary crashes before replication completes, recent writes may be lost. However, for cognitive systems, this matches biological reality - your brain can "forget" very recent events during trauma.

## Write-Ahead Log Design

The WAL is an append-only log of all mutations:

```rust
struct WALEntry {
    sequence: u64,
    space_id: SpaceId,
    operation: Operation,
    timestamp: f64,
    vector_clock: VectorClock,
}

enum Operation {
    StoreNode { node: MemoryNode },
    UpdateActivation { node_id: NodeId, activation: f32 },
    StoreEdge { edge: MemoryEdge },
}
```

Entries are batched and flushed to disk every 10ms or 1MB, whichever comes first. This provides crash consistency - completed flushes survive crashes.

## Replication Lag Monitoring

Track how far behind each replica is:

```rust
struct ReplicaLag {
    replica_id: NodeId,
    last_applied_sequence: u64,
    lag_bytes: u64,
    lag_time: Duration,
}
```

Alert when lag exceeds thresholds:
- lag_bytes > 100MB: replica is falling behind
- lag_time > 5s: network issues or slow replica

## Replica Promotion on Failure

When the primary fails, promote a replica to primary:

```rust
async fn promote_replica(&mut self, space_id: SpaceId) -> Result<()> {
    let replicas = self.get_replicas(space_id);

    // Choose replica with highest sequence number (most up-to-date)
    let new_primary = replicas.iter()
        .max_by_key(|r| r.last_applied_sequence)
        .ok_or(Error::NoReplicas)?;

    // Gossip assignment update
    self.update_assignment(space_id, new_primary.id, new_version).await;

    // New primary starts accepting writes
    new_primary.become_primary(space_id).await;

    Ok(())
}
```

Promotion completes in < 5 seconds, minimizing unavailability.

## Academic Foundation

- **Chain Replication**: van Renesse & Schneider (2004) - strong consistency with primary-backup
- **Dynamo**: DeCandia et al. (2007) - eventual consistency with sloppy quorums
- **Kafka**: Narkhede et al. (2011) - log-based replication at scale

Engram's approach is closest to Kafka: asynchronous log shipping with minimal write latency.
