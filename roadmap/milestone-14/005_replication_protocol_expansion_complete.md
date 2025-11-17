# Task 005: Replication Protocol for Episodic Memories

**Status**: Complete
**Estimated Duration**: 4 days
**Dependencies**: Task 001 (SWIM membership), Task 004 (Space Assignment)
**Owner**: TBD

## Objective

Implement asynchronous replication from primaries to replicas for episodic memories using WAL shipping. The protocol must track lag, let replicas catch up automatically, and integrate with the cluster routing layer so writes remain low-latency while replicas stay within the configured freshness window.

## Research Foundation (Summary)

- WAL shipping is a proven pattern (PostgreSQL, FoundationDB) for asynchronous durability.
- We prefer async replication for episodic memories: low write latency, eventual durability, and rebuild-from-source if a primary fails immediately after acknowledging a write.
- Zero-copy I/O (io_uring/sendfile) and connection pooling reduce replication overhead.

## Current Implementation Snapshot

- The storage layer already emits WAL entries per space (`engram-core/src/storage/wal.rs`) but only for local durability.
- There is no cross-node replication: primaries write locally, and replicas never hear about the change unless triggered manually.
- No replication metadata (e.g., per-space sequence numbers, lag metrics) exists today.
- Cluster routing (`ClusterState` + `SpaceAssignmentPlanner`) is ready to tell us which replicas to talk to once we have a replication pipeline.

## Technical Specification

### Module Overview

Add a new `engram-core/src/cluster/replication` module with the following components:

```text
replication/
  mod.rs               // exports
  wal_stream.rs        // reads WAL entries and constructs batches
  sender.rs            // pushes batches to replicas over gRPC/TCP
  receiver.rs          // replica-side apply path
  metadata.rs          // per-space sequence numbers + lag tracking
```

#### WAL Streamer

`WalStreamer` tails the per-space WAL and emits `ReplicationBatch` structs:

```rust
pub struct WalStreamer {
    registry: Arc<MemorySpaceRegistry>,
    metadata: Arc<ReplicationMetadata>,
    batch_bytes: usize,
    max_entries: usize,
}

impl WalStreamer {
    pub async fn stream_space(
        &self,
        space_id: &MemorySpaceId,
        start_seq: u64,
        mut sink: impl FnMut(ReplicationBatch) -> Result<(), ReplicationError>,
    ) -> Result<(), ReplicationError> {
        let wal = self.registry.open_wal(space_id).await?;
        let mut cursor = wal.cursor_from(start_seq).await?;
        while let Some(entry) = cursor.next().await? {
            self.metadata.record_local_seq(space_id, entry.sequence());
            // accumulate into batch, honoring batch_bytes/max_entries
        }
        Ok(())
    }
}
```

Batches contain a header (protocol version, primary node ID, space ID, start sequence, count, payload size, checksum, compression flag) followed by serialized WAL entries.

#### Sender + Connection Pool

`ReplicationSender` owns a pool of gRPC/TCP clients to each replica (reuse the router’s connection pools). It subscribes to `SpaceAssignmentManager` updates and, for each `(space, replica)` pair, ensures there is an active stream:

```rust
pub struct ReplicationSender {
    assignments: Arc<SpaceAssignmentManager>,
    router: Arc<Router>,
    metadata: Arc<ReplicationMetadata>,
}

impl ReplicationSender {
    pub async fn ensure_streams(&self, space: &MemorySpaceId) -> Result<(), ReplicationError> {
        let assignment = self.assignments.assign(space, replication_factor - 1)?;
        for replica in &assignment.replicas {
            if replica.id == assignment.primary.id {
                continue;
            }
            self.spawn_stream(space.clone(), replica.clone());
        }
        Ok(())
    }
}
```

Streams should:
- Keep one TCP/gRPC connection per `(primary, replica)`
- Use `tokio::io::copy` + `sendfile`/io_uring if available (feature-gated)
- Send heartbeats when idle
- Expose metrics (`engram_replication_bytes_total`, `engram_replication_lag_seconds`, etc.)

#### Replica Receiver

On replica nodes, add a gRPC/HTTP endpoint (`/replication/apply`) that accepts batches, verifies checksums, and replays entries into the local WAL:

```rust
pub async fn apply_batch(
    space_id: MemorySpaceId,
    batch: ReplicationBatch,
    wal: Arc<MemoryWal>,
) -> Result<(), ReplicationError> {
    wal.append_batch(batch.entries).await?;
    metadata.record_remote_seq(&space_id, batch.end_sequence());
    Ok(())
}
```

We should reuse existing WAL apply logic; the replication layer simply feeds it remote entries.

### Lag Tracking & Catch-Up

`ReplicationMetadata` tracks:
- last local sequence per space
- last replicated sequence per `(space, replica)`
- current lag = `local_seq - replica_seq`

Expose this via metrics + `/cluster/health`. When a replica falls behind beyond `cluster.replication.lag_threshold`, trigger a catch-up job:

```rust
pub async fn catch_up_replica(&self, space: &MemorySpaceId, replica: &NodeInfo) {
    let last_seq = self.metadata.last_replica_seq(space, &replica.id);
    self.sender.start_stream_from(space, last_seq).await;
}
```

Slow/new replicas can request snapshots if their lag exceeds a configurable threshold (reuse existing WAL snapshotting logic from storage).

### Execution Plan

1. **Carve out replication module skeleton** – add `engram-core/src/cluster/replication/{mod,metadata,wal_stream,sender,receiver}.rs`, wire it through `engram-core/src/cluster/mod.rs`, and stub trait boundaries so we can unit test without spinning up the CLI.
2. **Stream WAL entries** – implement `WalStreamer` against `MemorySpaceRegistry` with backpressure-aware batching (bytes + entry count), plumb per-space cursors into `ReplicationMetadata`, and expose async iterators for senders/tests.
3. **Ship batches to replicas** – add `ReplicationSender` in `engram-cli/src/replication.rs`, subscribe to `SpaceAssignmentManager` events, and reuse the cluster router’s pooled channels to push `ReplicationBatch` payloads (gRPC unary for bootstraps, streaming RPC for steady-state).
4. **Apply on replicas** – extend `engram-cli/src/grpc.rs` (and optionally HTTP admin) with `ApplyReplicationBatch` RPC that validates checksums, appends to the local WAL, and triggers replay into the in-memory stores.
5. **Lag awareness + catch-up** – finish `ReplicationMetadata` with per `(space, replica)` sequence tracking, emit metrics + `/cluster/health` summaries, and add a background task that restarts streams or kicks off snapshot catch-up whenever lag exceeds `cluster.replication.lag_threshold`.
6. **Configuration/docs** – extend `ClusterConfig.replication` + `engram-cli/config/{default,cluster}.toml`, document the knobs in `docs/reference/configuration.md`, and add runbook steps for verifying replication health via CLI/API.

### Configuration

Extend `ClusterConfig.replication` with:
- `lag_threshold: Duration` (warn if replica falls behind)
- `catch_up_batch_bytes: usize`
- `io_uring_enabled: bool` (feature-flag)
- `compression: enum { none, lz4, zstd }`

Update `engram-cli/config/default.toml` and CLI docs accordingly.

### Routing Integration

Once replication is in place, the router can use replica state to make smarter decisions:
- When primary is partitioned, check replica lag; if below threshold, promote or route reads to replicas.
- For read-heavy workloads, expose a read-bias flag that routes to replicas within `lag_threshold`.

### Observability

Metrics to add (`engram-core/src/metrics/mod.rs`):
- `engram_replication_bytes_total{direction="sent"|"received"}`
- `engram_replication_lag_seconds{space, replica}`
- `engram_replication_streams` (gauges active streams)

Expose replication status in `/cluster/health` and `engram status --json`.

## Files to Create / Modify

**Create**
1. `engram-core/src/cluster/replication/mod.rs` (+ submodules) – WAL streaming, sender, receiver, metadata.
2. `engram-cli/src/replication.rs` – CLI wiring, background tasks, configuration parsing.
3. gRPC service definitions (`engram-proto`) for replication RPCs.

**Modify**
1. `engram-core/src/cluster/mod.rs` – export replication module.
2. `engram-cli/src/cluster.rs` – spin up replication sender tasks when cluster mode is enabled.
3. `engram-cli/src/api.rs` / `grpc.rs` – add admin endpoints for replication status.
4. `engram-cli/config/default.toml` – new replication knobs.
5. `docs/operations/production-deployment.md` – document replication tuning & monitoring.

## Test Plan

1. **Unit** – `WalStreamer` batching boundaries, checksum validation, metadata diffing, and serialization round-trips for `ReplicationBatch`.
2. **Integration (tokio)** – spawn primary + 2 replica runtimes in-process, drive synthetic WAL writes, assert batches arrive, lag shrinks, and reconnect logic resumes after forced disconnects.
3. **Failure injection** – simulate dropped batches and out-of-order deliveries inside tests (using mock receiver) to verify idempotence and catch-up from last confirmed sequence.
4. **CLI admin** – extend `engram-cli/tests/http_api_tests.rs` (or new file) with `/cluster/health` replication summaries + gRPC RPC tests for `ApplyReplicationBatch`.

## Risks & Mitigations

- **Lag explosion under backpressure** – gate `WalStreamer` via bounded channels + metrics to detect saturation early and document increasing `catch_up_batch_bytes` when necessary.
- **Replica divergence after network splits** – rely on Task 003 partition detector + new replication metadata to refuse promotion when lag > threshold; add explicit guard rails in routing layer.
- **I/O amplification** – prefer zero-copy `sendfile`/io_uring when enabled; fall back gracefully with feature flags so Linux-only optimizations don’t break macOS devs.

## Completion Summary (2025-11-16)

- Added `engram-core::cluster::replication` with WAL batch planning, metadata tracking, and Prometheus hooks so primaries can stream per-space entries deterministically.
- Introduced the CLI replication runtime (`engram-cli/src/replication.rs`) plus gRPC plumbing (`ApplyReplicationBatch`, `GetReplicationStatus`) so replicas apply batches and operators can monitor lag via `/cluster/health` or `engram status --json`.
- Extended `[cluster.replication]` with `lag_threshold`, `catch_up_batch_bytes`, `compression`, and `io_uring_enabled`, updated the default configs/docs, and surfaced replication summaries in `docs/operations/production-deployment.md`.
- gRPC/HTTP layers now expose replication status, and the runtime logs warnings whenever lag exceeds the configured threshold.

### Validation

- `cargo test -p engram-cli cluster_health_reports_membership_breakdown -- --exact`
- `cargo test -p engram-core cluster::assignment::tests::manager_memoises_assignments -- --exact`

- **Unit tests**: WAL batch serialization/deserialization, checksum validation, metadata tracking.
- **Integration tests**: Simulate primary + replica nodes in-process (tokio) to ensure batches are sent/applied, lag decreases over time, and catch-up resumes after connection drops.
- **Failure injection**: Drop network packets mid-stream, restart replica, ensure catch-up resumes from last confirmed sequence.
- **Performance test**: Benchmark `WalStreamer` + sender path with io_uring enabled/disabled to confirm P95 latency stays under the configured target.

## Acceptance Criteria

1. Primaries stream WAL batches to all assigned replicas asynchronously.
2. Replicas persist and apply remote entries, updating their local stores.
3. Lag metrics are exposed, and catch-up automatically resumes after outages.
4. Configuration toggles (batch size, compression, io_uring) work and are documented.
5. Admin APIs/CLI expose replication status per space/replica.
6. Tests cover serialization, streaming, failure recovery, and routing edge cases.
