# Task 005: Replication Protocol for Episodic Memories

**Status**: Pending
**Estimated Duration**: 4 days
**Dependencies**: Task 001 (SWIM membership), Task 004 (Space Assignment)
**Owner**: TBD

## Objective

Implement asynchronous replication from primary to N replicas for episodic memories using WAL shipping, with lag monitoring, catchup mechanisms, and zero-copy I/O optimization. This ensures durability and availability of episodic memories across distributed nodes without blocking write operations.

## Research Foundation

### WAL-Based Replication Patterns

PostgreSQL's streaming replication (2010+) proved WAL shipping is the most reliable replication primitive. Primary writes to WAL, ships entries to standby nodes asynchronously. Standby applies entries to catch up. Key insight: WAL is already crash-consistent, so shipping it preserves all durability guarantees.

**PostgreSQL streaming replication architecture:**
- Primary writes WAL entry, fsyncs, returns to client (async mode)
- Background walsender reads WAL, streams to standby
- Standby walreceiver writes to local WAL, applies to memory
- Lag tracking via LSN (Log Sequence Number) comparison
- Catchup: standby replays from last checkpoint to current LSN

**FoundationDB's replication (deterministic simulation testing):**
- Transaction log shipping with version vectors
- Zero-copy sendfile() for log transmission on Linux
- Batching: ship 1MB chunks instead of individual entries
- Compression: zstd at level 3 (2x reduction, <1ms overhead)
- Checksums: CRC32C hardware acceleration via SSE4.2

**RocksDB replication (used by TiKV, CockroachDB):**
- Raft log IS the replication log (unified abstraction)
- Leader ships log entries to followers
- Followers apply entries in-order (strong consistency via Raft)
- Snapshot transfer for slow/new followers (fallback catchup)

**MySQL binlog replication:**
- Row-based replication (deterministic, not statement-based)
- GTID (Global Transaction ID) for position-independent catchup
- Semi-sync option: wait for 1 replica ack before commit (slower but safer)
- Delayed replication: replay lag for point-in-time recovery

### Async vs Sync Replication Tradeoffs

**Async replication (Engram choice):**
- Write latency: 1-10ms (WAL fsync only, no network wait)
- Throughput: 100K+ writes/sec per node
- Availability: writes continue during replica failures
- Durability risk: data loss if primary crashes before replication
- Use case: episodic memories (can reconstruct from source data if needed)

**Sync replication (not chosen for episodic tier):**
- Write latency: 20-100ms (wait for replica ack over network)
- Throughput: 10K writes/sec (limited by slowest replica)
- Availability: writes block if majority unavailable
- Durability guarantee: committed data on N+1 nodes before ack
- Use case: semantic memories (consolidated, irreplaceable)

### Zero-Copy I/O with io_uring (Linux 5.1+)

Traditional I/O path: read() copies from kernel to userspace, write() copies from userspace to kernel. Double copy per operation.

**io_uring advantages:**
- Single submission queue, single completion queue (lock-free ring buffer)
- Batch submissions: 100 I/O ops in one syscall
- Zero-copy: direct descriptor-to-descriptor transfer via IORING_OP_SENDMSG_ZC
- Polling mode: eliminate syscall overhead entirely (for high-throughput workloads)
- Latency: P99 write <5ms (vs 10ms with traditional I/O)

**Fallback strategy:**
- io_uring only on Linux 5.1+
- macOS: use kqueue + sendfile()
- Windows: I/O completion ports
- Generic: tokio async I/O (graceful degradation)

### Connection Pooling for Replication Streams

Problem: creating new TCP connection per WAL batch adds 1-10ms RTT latency. Connection pool amortizes handshake cost.

**Design pattern (from Apache Kafka):**
- Pre-established connections to all replicas (on cluster join)
- Multiplexing: single connection for WAL stream + heartbeats + metadata
- Keepalive: TCP_KEEPALIVE with 10s interval (detect dead connections)
- Circuit breaker: mark replica degraded after 3 consecutive failures
- Reconnection backoff: exponential 100ms → 1s → 10s → 60s

## Technical Specification

### WAL File Format for Replication

Engram already has WAL (see engram-core/src/storage/wal.rs). We extend it for distributed replication:

```rust
// engram-core/src/cluster/replication/wal_format.rs

use crate::storage::wal::{WalEntry, WalEntryHeader};
use std::io::Write;

/// Replication-specific WAL batch header
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReplicationBatchHeader {
    /// Protocol version for compatibility
    pub version: u16,

    /// Node ID of primary (UUID as 16 bytes)
    pub primary_node_id: [u8; 16],

    /// Memory space ID being replicated
    pub space_id: [u8; 16],

    /// First sequence number in this batch
    pub start_sequence: u64,

    /// Number of WAL entries in batch
    pub entry_count: u32,

    /// Total payload size (sum of all entries)
    pub total_size: u64,

    /// Compression algorithm (0=none, 1=zstd, 2=lz4)
    pub compression: u8,

    /// CRC32C checksum of entire batch (header + entries)
    pub batch_crc: u32,

    /// Reserved for future extensions
    pub reserved: [u8; 15],
}

const REPLICATION_BATCH_HEADER_SIZE: usize =
    std::mem::size_of::<ReplicationBatchHeader>();

impl ReplicationBatchHeader {
    pub fn new(
        primary_node_id: [u8; 16],
        space_id: [u8; 16],
        start_sequence: u64,
        entry_count: u32,
        total_size: u64,
    ) -> Self {
        Self {
            version: 1,
            primary_node_id,
            space_id,
            start_sequence,
            entry_count,
            total_size,
            compression: 0, // No compression initially
            batch_crc: 0,   // Computed after serialization
            reserved: [0; 15],
        }
    }

    /// Serialize header to bytes
    pub fn as_bytes(&self) -> [u8; REPLICATION_BATCH_HEADER_SIZE] {
        let mut bytes = [0u8; REPLICATION_BATCH_HEADER_SIZE];
        let mut offset = 0;

        bytes[offset..offset+2].copy_from_slice(&self.version.to_le_bytes());
        offset += 2;
        bytes[offset..offset+16].copy_from_slice(&self.primary_node_id);
        offset += 16;
        bytes[offset..offset+16].copy_from_slice(&self.space_id);
        offset += 16;
        bytes[offset..offset+8].copy_from_slice(&self.start_sequence.to_le_bytes());
        offset += 8;
        bytes[offset..offset+4].copy_from_slice(&self.entry_count.to_le_bytes());
        offset += 4;
        bytes[offset..offset+8].copy_from_slice(&self.total_size.to_le_bytes());
        offset += 8;
        bytes[offset] = self.compression;
        offset += 1;
        bytes[offset..offset+4].copy_from_slice(&self.batch_crc.to_le_bytes());
        offset += 4;
        bytes[offset..offset+15].copy_from_slice(&self.reserved);

        bytes
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8; REPLICATION_BATCH_HEADER_SIZE]) -> Self {
        let mut offset = 0;

        let version = u16::from_le_bytes([bytes[offset], bytes[offset+1]]);
        offset += 2;

        let mut primary_node_id = [0u8; 16];
        primary_node_id.copy_from_slice(&bytes[offset..offset+16]);
        offset += 16;

        let mut space_id = [0u8; 16];
        space_id.copy_from_slice(&bytes[offset..offset+16]);
        offset += 16;

        let start_sequence = u64::from_le_bytes([
            bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3],
            bytes[offset+4], bytes[offset+5], bytes[offset+6], bytes[offset+7],
        ]);
        offset += 8;

        let entry_count = u32::from_le_bytes([
            bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3],
        ]);
        offset += 4;

        let total_size = u64::from_le_bytes([
            bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3],
            bytes[offset+4], bytes[offset+5], bytes[offset+6], bytes[offset+7],
        ]);
        offset += 8;

        let compression = bytes[offset];
        offset += 1;

        let batch_crc = u32::from_le_bytes([
            bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3],
        ]);
        offset += 4;

        let mut reserved = [0u8; 15];
        reserved.copy_from_slice(&bytes[offset..offset+15]);

        Self {
            version,
            primary_node_id,
            space_id,
            start_sequence,
            entry_count,
            total_size,
            compression,
            batch_crc,
            reserved,
        }
    }

    /// Validate batch header
    pub fn validate(&self) -> Result<(), ReplicationError> {
        if self.version != 1 {
            return Err(ReplicationError::UnsupportedVersion(self.version));
        }

        if self.entry_count == 0 {
            return Err(ReplicationError::EmptyBatch);
        }

        if self.total_size == 0 {
            return Err(ReplicationError::InvalidBatchSize(self.total_size));
        }

        Ok(())
    }
}

/// Replication batch with multiple WAL entries
pub struct ReplicationBatch {
    pub header: ReplicationBatchHeader,
    pub entries: Vec<WalEntry>,
}

impl ReplicationBatch {
    pub fn new(
        primary_node_id: [u8; 16],
        space_id: [u8; 16],
        entries: Vec<WalEntry>,
    ) -> Self {
        let start_sequence = entries.first()
            .map(|e| e.header.sequence)
            .unwrap_or(0);

        let total_size = entries.iter()
            .map(|e| e.serialized_size() as u64)
            .sum();

        let header = ReplicationBatchHeader::new(
            primary_node_id,
            space_id,
            start_sequence,
            entries.len() as u32,
            total_size,
        );

        Self { header, entries }
    }

    /// Serialize batch to bytes for network transmission
    pub fn serialize(&self) -> Result<Vec<u8>, ReplicationError> {
        let mut bytes = Vec::with_capacity(
            REPLICATION_BATCH_HEADER_SIZE + self.header.total_size as usize
        );

        // Write header (CRC computed at end)
        bytes.write_all(&self.header.as_bytes())?;

        // Write each WAL entry
        for entry in &self.entries {
            bytes.write_all(&entry.header.as_bytes())?;
            bytes.write_all(&entry.payload)?;
        }

        // Compute CRC32C of entire payload
        #[cfg(feature = "memory_mapped_persistence")]
        {
            use crc32c::crc32c;
            let payload_crc = crc32c(&bytes[REPLICATION_BATCH_HEADER_SIZE..]);

            // Update CRC in header
            let crc_offset = REPLICATION_BATCH_HEADER_SIZE - 19; // Position of batch_crc
            bytes[crc_offset..crc_offset+4].copy_from_slice(&payload_crc.to_le_bytes());
        }

        Ok(bytes)
    }

    /// Deserialize batch from bytes
    pub fn deserialize(bytes: &[u8]) -> Result<Self, ReplicationError> {
        if bytes.len() < REPLICATION_BATCH_HEADER_SIZE {
            return Err(ReplicationError::TruncatedBatch);
        }

        let mut header_bytes = [0u8; REPLICATION_BATCH_HEADER_SIZE];
        header_bytes.copy_from_slice(&bytes[..REPLICATION_BATCH_HEADER_SIZE]);

        let header = ReplicationBatchHeader::from_bytes(&header_bytes);
        header.validate()?;

        // Validate CRC
        #[cfg(feature = "memory_mapped_persistence")]
        {
            use crc32c::crc32c;
            let payload_crc = crc32c(&bytes[REPLICATION_BATCH_HEADER_SIZE..]);
            if payload_crc != header.batch_crc {
                return Err(ReplicationError::ChecksumMismatch {
                    expected: header.batch_crc,
                    actual: payload_crc,
                });
            }
        }

        // Deserialize WAL entries
        let mut entries = Vec::with_capacity(header.entry_count as usize);
        let mut offset = REPLICATION_BATCH_HEADER_SIZE;

        for _ in 0..header.entry_count {
            if offset + 64 > bytes.len() {
                return Err(ReplicationError::TruncatedEntry);
            }

            let mut entry_header_bytes = [0u8; 64];
            entry_header_bytes.copy_from_slice(&bytes[offset..offset+64]);
            let entry_header = WalEntryHeader::from_bytes(&entry_header_bytes);
            offset += 64;

            let payload_size = entry_header.payload_size as usize;
            if offset + payload_size > bytes.len() {
                return Err(ReplicationError::TruncatedEntry);
            }

            let payload = bytes[offset..offset+payload_size].to_vec();
            offset += payload_size;

            entries.push(WalEntry {
                header: entry_header,
                payload,
            });
        }

        Ok(Self { header, entries })
    }
}
```

## Core Data Structures

### Replication State Tracking

```rust
// engram-core/src/cluster/replication/state.rs

use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Replication state for a memory space
#[derive(Debug, Clone)]
pub struct ReplicationState {
    /// Memory space ID
    pub space_id: String,

    /// Primary node ID
    pub primary_node_id: String,

    /// Replica node IDs
    pub replica_node_ids: Vec<String>,

    /// Last sequence number written to primary WAL
    pub primary_sequence: u64,

    /// Per-replica replication progress
    pub replica_progress: DashMap<String, ReplicaProgress>,

    /// Time of last update
    pub last_update: Instant,
}

/// Progress tracking for a single replica
#[derive(Debug, Clone)]
pub struct ReplicaProgress {
    /// Node ID of replica
    pub node_id: String,

    /// Last sequence number successfully applied by replica
    pub applied_sequence: u64,

    /// Last sequence number acknowledged by replica
    pub acked_sequence: u64,

    /// Time of last successful replication
    pub last_success: Instant,

    /// Replication lag in milliseconds
    pub lag_ms: f64,

    /// Number of consecutive failures
    pub consecutive_failures: u32,

    /// Replica health status
    pub status: ReplicaStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicaStatus {
    /// Healthy, caught up
    Healthy,

    /// Lagging but catching up
    Lagging,

    /// Severely lagged, needs catchup
    Degraded,

    /// Failed, not replicating
    Failed,
}

impl ReplicationState {
    pub fn new(
        space_id: String,
        primary_node_id: String,
        replica_node_ids: Vec<String>,
    ) -> Self {
        let replica_progress = DashMap::new();

        for replica_id in &replica_node_ids {
            replica_progress.insert(
                replica_id.clone(),
                ReplicaProgress {
                    node_id: replica_id.clone(),
                    applied_sequence: 0,
                    acked_sequence: 0,
                    last_success: Instant::now(),
                    lag_ms: 0.0,
                    consecutive_failures: 0,
                    status: ReplicaStatus::Healthy,
                },
            );
        }

        Self {
            space_id,
            primary_node_id,
            replica_node_ids,
            primary_sequence: 0,
            replica_progress,
            last_update: Instant::now(),
        }
    }

    /// Update replica progress after successful replication
    pub fn update_replica_progress(
        &self,
        replica_id: &str,
        acked_sequence: u64,
    ) {
        if let Some(mut progress) = self.replica_progress.get_mut(replica_id) {
            progress.acked_sequence = acked_sequence;
            progress.applied_sequence = acked_sequence;
            progress.last_success = Instant::now();
            progress.consecutive_failures = 0;

            // Calculate lag
            let lag_entries = self.primary_sequence.saturating_sub(acked_sequence);
            progress.lag_ms = lag_entries as f64 * 0.1; // Rough estimate: 0.1ms per entry

            // Update status
            progress.status = if lag_entries == 0 {
                ReplicaStatus::Healthy
            } else if lag_entries < 1000 {
                ReplicaStatus::Lagging
            } else {
                ReplicaStatus::Degraded
            };
        }
    }

    /// Mark replica as failed
    pub fn mark_replica_failed(&self, replica_id: &str) {
        if let Some(mut progress) = self.replica_progress.get_mut(replica_id) {
            progress.consecutive_failures += 1;

            if progress.consecutive_failures >= 3 {
                progress.status = ReplicaStatus::Failed;
            }
        }
    }

    /// Get minimum acked sequence across all healthy replicas
    pub fn min_acked_sequence(&self) -> u64 {
        self.replica_progress
            .iter()
            .filter(|entry| entry.value().status != ReplicaStatus::Failed)
            .map(|entry| entry.value().acked_sequence)
            .min()
            .unwrap_or(0)
    }

    /// Check if space is sufficiently replicated
    pub fn is_sufficiently_replicated(&self, min_replicas: usize) -> bool {
        let healthy_count = self.replica_progress
            .iter()
            .filter(|entry| {
                matches!(
                    entry.value().status,
                    ReplicaStatus::Healthy | ReplicaStatus::Lagging
                )
            })
            .count();

        healthy_count >= min_replicas
    }
}

/// Global replication coordinator
pub struct ReplicationCoordinator {
    /// Replication state per memory space
    states: Arc<DashMap<String, Arc<RwLock<ReplicationState>>>>,

    /// Replication configuration
    config: ReplicationConfig,
}

#[derive(Debug, Clone)]
pub struct ReplicationConfig {
    /// Number of replicas per space
    pub replica_count: usize,

    /// Batch size for WAL shipping (number of entries)
    pub batch_size: usize,

    /// Maximum batch delay before shipping
    pub batch_delay: Duration,

    /// Lag threshold for alerting (milliseconds)
    pub lag_alert_threshold_ms: f64,

    /// Lag threshold for marking degraded
    pub lag_degraded_threshold_ms: f64,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            replica_count: 2,
            batch_size: 100,
            batch_delay: Duration::from_millis(10),
            lag_alert_threshold_ms: 1000.0,   // 1 second
            lag_degraded_threshold_ms: 5000.0, // 5 seconds
        }
    }
}

impl ReplicationCoordinator {
    pub fn new(config: ReplicationConfig) -> Self {
        Self {
            states: Arc::new(DashMap::new()),
            config,
        }
    }

    /// Register a memory space for replication
    pub async fn register_space(
        &self,
        space_id: String,
        primary_node_id: String,
        replica_node_ids: Vec<String>,
    ) -> Arc<RwLock<ReplicationState>> {
        let state = Arc::new(RwLock::new(ReplicationState::new(
            space_id.clone(),
            primary_node_id,
            replica_node_ids,
        )));

        self.states.insert(space_id, state.clone());
        state
    }

    /// Get replication state for a space
    pub fn get_state(&self, space_id: &str) -> Option<Arc<RwLock<ReplicationState>>> {
        self.states.get(space_id).map(|entry| entry.value().clone())
    }

    /// Get all lagging replicas across all spaces
    pub async fn get_lagging_replicas(&self) -> Vec<(String, String, f64)> {
        let mut lagging = Vec::new();

        for entry in self.states.iter() {
            let space_id = entry.key().clone();
            let state = entry.value().read().await;

            for progress_entry in state.replica_progress.iter() {
                let progress = progress_entry.value();

                if progress.lag_ms > self.config.lag_alert_threshold_ms {
                    lagging.push((
                        space_id.clone(),
                        progress.node_id.clone(),
                        progress.lag_ms,
                    ));
                }
            }
        }

        lagging
    }
}
```

### WAL Shipper

```rust
// engram-core/src/cluster/replication/wal_shipper.rs

use super::{ReplicationBatch, ReplicationConfig, ReplicationState};
use crate::storage::wal::{WalEntry, WalReader};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Ships WAL entries from primary to replicas
pub struct WalShipper {
    /// Node ID of this primary
    node_id: String,

    /// WAL reader for accessing local WAL
    wal_reader: Arc<WalReader>,

    /// Replication state
    state: Arc<RwLock<ReplicationState>>,

    /// Configuration
    config: ReplicationConfig,

    /// Connection pool for replica nodes
    connection_pool: Arc<ReplicaConnectionPool>,

    /// Pending entries buffer
    pending_entries: Arc<RwLock<Vec<WalEntry>>>,

    /// Last shipped sequence
    last_shipped_sequence: Arc<RwLock<u64>>,
}

impl WalShipper {
    pub fn new(
        node_id: String,
        wal_reader: Arc<WalReader>,
        state: Arc<RwLock<ReplicationState>>,
        config: ReplicationConfig,
        connection_pool: Arc<ReplicaConnectionPool>,
    ) -> Self {
        Self {
            node_id,
            wal_reader,
            state,
            config,
            connection_pool,
            pending_entries: Arc::new(RwLock::new(Vec::new())),
            last_shipped_sequence: Arc::new(RwLock::new(0)),
        }
    }

    /// Start shipping WAL entries to replicas
    pub async fn start(&self) {
        let mut interval = tokio::time::interval(self.config.batch_delay);

        loop {
            interval.tick().await;

            if let Err(e) = self.ship_batch().await {
                error!("WAL shipping failed: {}", e);
            }
        }
    }

    /// Ship a batch of WAL entries
    async fn ship_batch(&self) -> Result<(), ReplicationError> {
        let mut pending = self.pending_entries.write().await;

        if pending.is_empty() {
            // Check for new entries from WAL
            let new_entries = self.read_new_wal_entries().await?;
            pending.extend(new_entries);
        }

        if pending.is_empty() {
            return Ok(());
        }

        // Take up to batch_size entries
        let batch_entries: Vec<_> = pending
            .drain(..pending.len().min(self.config.batch_size))
            .collect();

        if batch_entries.is_empty() {
            return Ok(());
        }

        // Create replication batch
        let state = self.state.read().await;
        let space_id_bytes = uuid::Uuid::parse_str(&state.space_id)
            .unwrap_or_default()
            .as_bytes()
            .to_owned();
        let node_id_bytes = uuid::Uuid::parse_str(&self.node_id)
            .unwrap_or_default()
            .as_bytes()
            .to_owned();

        let batch = ReplicationBatch::new(
            node_id_bytes,
            space_id_bytes,
            batch_entries.clone(),
        );

        let serialized = batch.serialize()?;

        // Ship to all replicas in parallel
        let mut handles = Vec::new();

        for replica_id in &state.replica_node_ids {
            let replica_id = replica_id.clone();
            let serialized = serialized.clone();
            let pool = self.connection_pool.clone();
            let state_clone = self.state.clone();

            let handle = tokio::spawn(async move {
                match pool.send_batch(&replica_id, &serialized).await {
                    Ok(ack) => {
                        // Update replica progress
                        state_clone.read().await.update_replica_progress(
                            &replica_id,
                            ack.applied_sequence,
                        );
                        Ok(())
                    }
                    Err(e) => {
                        state_clone.read().await.mark_replica_failed(&replica_id);
                        Err(e)
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all replicas (with timeout)
        let timeout = tokio::time::Duration::from_secs(5);
        let results = tokio::time::timeout(
            timeout,
            futures::future::join_all(handles),
        ).await;

        match results {
            Ok(results) => {
                let failures: Vec<_> = results
                    .into_iter()
                    .filter_map(|r| r.ok())
                    .filter_map(|r| r.err())
                    .collect();

                if !failures.is_empty() {
                    warn!("Some replicas failed: {:?}", failures);
                }
            }
            Err(_) => {
                warn!("Replication batch timed out");
            }
        }

        // Update last shipped sequence
        let last_sequence = batch_entries.last().unwrap().header.sequence;
        *self.last_shipped_sequence.write().await = last_sequence;

        info!(
            "Shipped batch of {} entries (sequence {}..{})",
            batch_entries.len(),
            batch_entries.first().unwrap().header.sequence,
            last_sequence,
        );

        Ok(())
    }

    /// Read new WAL entries since last shipped
    async fn read_new_wal_entries(&self) -> Result<Vec<WalEntry>, ReplicationError> {
        let last_shipped = *self.last_shipped_sequence.read().await;

        // Scan WAL for entries after last_shipped
        let all_entries = self.wal_reader.scan_all()?;

        let new_entries: Vec<_> = all_entries
            .into_iter()
            .filter(|entry| entry.header.sequence > last_shipped)
            .collect();

        Ok(new_entries)
    }

    /// Add WAL entry to pending buffer (called after write)
    pub async fn append_entry(&self, entry: WalEntry) {
        self.pending_entries.write().await.push(entry);
    }
}
```

### Replica Connection Pool

```rust
// engram-core/src/cluster/replication/connection_pool.rs

use dashmap::DashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, error, warn};

/// Pool of connections to replica nodes
pub struct ReplicaConnectionPool {
    /// Active connections (replica_id -> connection)
    connections: Arc<DashMap<String, Arc<ReplicaConnection>>>,

    /// Connection configuration
    config: ConnectionPoolConfig,
}

#[derive(Debug, Clone)]
pub struct ConnectionPoolConfig {
    /// Connection timeout
    pub connect_timeout: Duration,

    /// Read timeout
    pub read_timeout: Duration,

    /// Write timeout
    pub write_timeout: Duration,

    /// TCP keepalive interval
    pub keepalive_interval: Duration,

    /// Maximum retry attempts
    pub max_retries: usize,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(5),
            read_timeout: Duration::from_secs(10),
            write_timeout: Duration::from_secs(10),
            keepalive_interval: Duration::from_secs(10),
            max_retries: 3,
        }
    }
}

/// Single connection to a replica
pub struct ReplicaConnection {
    replica_id: String,
    replica_addr: String,
    stream: Arc<tokio::sync::Mutex<TcpStream>>,
    last_use: Arc<tokio::sync::RwLock<std::time::Instant>>,
}

impl ReplicaConnection {
    async fn new(
        replica_id: String,
        replica_addr: String,
        config: &ConnectionPoolConfig,
    ) -> Result<Self, ReplicationError> {
        let stream = tokio::time::timeout(
            config.connect_timeout,
            TcpStream::connect(&replica_addr),
        ).await??;

        // Set TCP keepalive
        let socket = socket2::Socket::from(stream.into_std()?);
        socket.set_keepalive(Some(config.keepalive_interval))?;

        let stream = TcpStream::from_std(socket.into())?;

        Ok(Self {
            replica_id,
            replica_addr,
            stream: Arc::new(tokio::sync::Mutex::new(stream)),
            last_use: Arc::new(tokio::sync::RwLock::new(std::time::Instant::now())),
        })
    }

    async fn send_batch(&self, data: &[u8]) -> Result<ReplicationAck, ReplicationError> {
        let mut stream = self.stream.lock().await;

        // Send batch size (4 bytes)
        let size = data.len() as u32;
        stream.write_all(&size.to_le_bytes()).await?;

        // Send batch data
        stream.write_all(data).await?;
        stream.flush().await?;

        // Read acknowledgment (16 bytes: sequence + timestamp)
        let mut ack_bytes = [0u8; 16];
        stream.read_exact(&mut ack_bytes).await?;

        let applied_sequence = u64::from_le_bytes([
            ack_bytes[0], ack_bytes[1], ack_bytes[2], ack_bytes[3],
            ack_bytes[4], ack_bytes[5], ack_bytes[6], ack_bytes[7],
        ]);

        let timestamp = u64::from_le_bytes([
            ack_bytes[8], ack_bytes[9], ack_bytes[10], ack_bytes[11],
            ack_bytes[12], ack_bytes[13], ack_bytes[14], ack_bytes[15],
        ]);

        *self.last_use.write().await = std::time::Instant::now();

        Ok(ReplicationAck {
            replica_id: self.replica_id.clone(),
            applied_sequence,
            timestamp,
        })
    }
}

impl ReplicaConnectionPool {
    pub fn new(config: ConnectionPoolConfig) -> Self {
        Self {
            connections: Arc::new(DashMap::new()),
            config,
        }
    }

    /// Get or create connection to replica
    async fn get_connection(
        &self,
        replica_id: &str,
        replica_addr: &str,
    ) -> Result<Arc<ReplicaConnection>, ReplicationError> {
        if let Some(conn) = self.connections.get(replica_id) {
            return Ok(conn.value().clone());
        }

        // Create new connection
        let conn = Arc::new(
            ReplicaConnection::new(
                replica_id.to_string(),
                replica_addr.to_string(),
                &self.config,
            ).await?
        );

        self.connections.insert(replica_id.to_string(), conn.clone());
        Ok(conn)
    }

    /// Send batch to replica with retries
    pub async fn send_batch(
        &self,
        replica_id: &str,
        data: &[u8],
    ) -> Result<ReplicationAck, ReplicationError> {
        // TODO: Look up replica address from membership
        let replica_addr = format!("{}:8080", replica_id); // Placeholder

        let mut last_error = None;

        for attempt in 0..self.config.max_retries {
            match self.get_connection(replica_id, &replica_addr).await {
                Ok(conn) => {
                    match conn.send_batch(data).await {
                        Ok(ack) => return Ok(ack),
                        Err(e) => {
                            warn!(
                                "Replication to {} failed (attempt {}): {}",
                                replica_id, attempt + 1, e
                            );

                            // Remove failed connection
                            self.connections.remove(replica_id);
                            last_error = Some(e);

                            // Exponential backoff
                            let backoff = Duration::from_millis(100 * 2_u64.pow(attempt as u32));
                            tokio::time::sleep(backoff).await;
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to connect to replica {}: {}", replica_id, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            ReplicationError::MaxRetriesExceeded(self.config.max_retries)
        }))
    }
}

/// Acknowledgment from replica
#[derive(Debug, Clone)]
pub struct ReplicationAck {
    pub replica_id: String,
    pub applied_sequence: u64,
    pub timestamp: u64,
}
```

### Lag Monitor

```rust
// engram-core/src/cluster/replication/lag_monitor.rs

use super::{ReplicationCoordinator, ReplicationConfig};
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

/// Monitors replication lag and triggers alerts
pub struct LagMonitor {
    coordinator: Arc<ReplicationCoordinator>,
    config: ReplicationConfig,
}

impl LagMonitor {
    pub fn new(
        coordinator: Arc<ReplicationCoordinator>,
        config: ReplicationConfig,
    ) -> Self {
        Self { coordinator, config }
    }

    /// Start monitoring loop
    pub async fn start(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(5));

        loop {
            interval.tick().await;
            self.check_lag().await;
        }
    }

    async fn check_lag(&self) {
        let lagging = self.coordinator.get_lagging_replicas().await;

        if lagging.is_empty() {
            return;
        }

        for (space_id, replica_id, lag_ms) in lagging {
            if lag_ms > self.config.lag_degraded_threshold_ms {
                error!(
                    "Replica {} for space {} severely lagged: {:.1}ms",
                    replica_id, space_id, lag_ms
                );

                // Trigger alert (integrate with metrics/alerting)
                metrics::gauge!(
                    "engram.replication.lag_ms",
                    lag_ms,
                    "space_id" => space_id.clone(),
                    "replica_id" => replica_id.clone(),
                );
            } else if lag_ms > self.config.lag_alert_threshold_ms {
                warn!(
                    "Replica {} for space {} lagging: {:.1}ms",
                    replica_id, space_id, lag_ms
                );
            }
        }
    }
}
```

## Error Types

```rust
// engram-core/src/cluster/replication/error.rs

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReplicationError {
    #[error("Unsupported replication protocol version: {0}")]
    UnsupportedVersion(u16),

    #[error("Empty replication batch")]
    EmptyBatch,

    #[error("Invalid batch size: {0}")]
    InvalidBatchSize(u64),

    #[error("Truncated batch")]
    TruncatedBatch,

    #[error("Truncated WAL entry")]
    TruncatedEntry,

    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Storage error: {0}")]
    Storage(#[from] crate::storage::StorageError),

    #[error("Connection timeout")]
    ConnectionTimeout,

    #[error("Maximum retries exceeded: {0}")]
    MaxRetriesExceeded(usize),

    #[error("Replica not found: {0}")]
    ReplicaNotFound(String),
}
```

## Files to Create

1. `engram-core/src/cluster/replication/mod.rs` - Module exports
2. `engram-core/src/cluster/replication/wal_format.rs` - Replication WAL format
3. `engram-core/src/cluster/replication/state.rs` - Replication state tracking
4. `engram-core/src/cluster/replication/wal_shipper.rs` - WAL shipping logic
5. `engram-core/src/cluster/replication/connection_pool.rs` - Connection management
6. `engram-core/src/cluster/replication/lag_monitor.rs` - Lag monitoring
7. `engram-core/src/cluster/replication/error.rs` - Error types
8. `engram-core/src/cluster/replication/receiver.rs` - Replica-side receiver
9. `engram-core/src/cluster/replication/catchup.rs` - Catchup mechanism for lagged replicas
10. `engram-core/src/cluster/replication/promotion.rs` - Replica promotion on primary failure

## Files to Modify

1. `engram-core/src/cluster/mod.rs` - Export replication module
2. `engram-core/src/storage/wal.rs` - Add replication hooks
3. `engram-core/src/registry/memory_space.rs` - Track primary/replica nodes
4. `engram-cli/src/cluster.rs` - Start replication services
5. `engram-core/src/metrics/mod.rs` - Add replication metrics
6. `engram-core/Cargo.toml` - Add dependencies (socket2, crc32c)

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replication_batch_header_size() {
        assert_eq!(
            std::mem::size_of::<ReplicationBatchHeader>(),
            64
        );
    }

    #[test]
    fn test_replication_batch_serialization() {
        let primary_id = uuid::Uuid::new_v4();
        let space_id = uuid::Uuid::new_v4();

        let entries = vec![
            WalEntry::new_checkpoint(100).unwrap(),
            WalEntry::new_checkpoint(101).unwrap(),
        ];

        let batch = ReplicationBatch::new(
            *primary_id.as_bytes(),
            *space_id.as_bytes(),
            entries,
        );

        let serialized = batch.serialize().unwrap();
        let deserialized = ReplicationBatch::deserialize(&serialized).unwrap();

        assert_eq!(batch.entries.len(), deserialized.entries.len());
        assert_eq!(
            batch.header.start_sequence,
            deserialized.header.start_sequence
        );
    }

    #[tokio::test]
    async fn test_replication_state_tracking() {
        let state = ReplicationState::new(
            "space1".to_string(),
            "primary1".to_string(),
            vec!["replica1".to_string(), "replica2".to_string()],
        );

        // Update replica progress
        state.update_replica_progress("replica1", 100);
        state.update_replica_progress("replica2", 90);

        assert_eq!(state.min_acked_sequence(), 90);
        assert!(state.is_sufficiently_replicated(2));

        // Mark one as failed
        state.mark_replica_failed("replica2");
        state.mark_replica_failed("replica2");
        state.mark_replica_failed("replica2");

        assert_eq!(state.min_acked_sequence(), 100);
        assert!(state.is_sufficiently_replicated(1));
    }
}
```

### Integration Tests

```rust
// engram-core/tests/replication_integration.rs

#[tokio::test]
async fn test_primary_to_replica_replication() {
    // Start primary node
    let primary = TestNode::start("primary1", 8001).await;

    // Start replica node
    let replica = TestNode::start("replica1", 8002).await;

    // Create memory space on primary
    let space_id = primary.create_space("test_space").await.unwrap();

    // Register replication
    primary.register_replica(&space_id, "replica1").await.unwrap();

    // Write episodes on primary
    for i in 0..100 {
        let episode = create_test_episode(i);
        primary.store_episode(&space_id, episode).await.unwrap();
    }

    // Wait for replication
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Verify replica has all episodes
    let replica_count = replica.count_episodes(&space_id).await.unwrap();
    assert_eq!(replica_count, 100);
}

#[tokio::test]
async fn test_replication_lag_recovery() {
    let primary = TestNode::start("primary1", 8001).await;
    let replica = TestNode::start("replica1", 8002).await;

    let space_id = primary.create_space("test_space").await.unwrap();
    primary.register_replica(&space_id, "replica1").await.unwrap();

    // Write many episodes
    for i in 0..1000 {
        let episode = create_test_episode(i);
        primary.store_episode(&space_id, episode).await.unwrap();
    }

    // Simulate replica delay
    replica.pause_replication().await;

    // Write more episodes while replica is paused
    for i in 1000..2000 {
        let episode = create_test_episode(i);
        primary.store_episode(&space_id, episode).await.unwrap();
    }

    // Resume replica
    replica.resume_replication().await;

    // Wait for catchup
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Verify replica caught up
    let replica_count = replica.count_episodes(&space_id).await.unwrap();
    assert_eq!(replica_count, 2000);
}
```

## Dependencies

Add to `engram-core/Cargo.toml`:

```toml
[dependencies]
# Existing dependencies...

# Replication
socket2 = "0.5"
futures = "0.3"

# Already have: tokio, dashmap, bincode, serde, crc32c, uuid
```

## Acceptance Criteria

1. Write latency <10ms (async replication, immediate return after WAL fsync)
2. Replication lag <1s under normal load (10K writes/sec)
3. Replica promotion on primary failure completes <5s
4. Batch group commit amortizes network overhead (100 entries/batch typical)
5. Connection pool reuses TCP connections (no handshake per batch)
6. Lag monitoring alerts when replica falls >1s behind
7. Catchup mechanism works for replicas 10K entries behind
8. No data loss during replica failure (writes continue on primary)
9. Checksums detect corruption in transit
10. Metrics track: replication lag per replica, batch size, throughput

## Performance Targets

- Write latency P99: <10ms (primary WAL fsync only)
- Replication throughput: 100K entries/sec per replica stream
- Network bandwidth: <100MB/sec per replica (with compression)
- Connection pool overhead: <1ms per batch send
- Lag monitoring overhead: <0.01% CPU
- Memory overhead: <100MB per replica stream (buffering)

## Out of Scope (Future Work)

- Semantic memory replication (different consistency model, Task 007)
- Strong consistency / synchronous replication (not needed for episodic tier)
- Cross-datacenter replication (requires WAN optimization)
- Encryption in transit (TLS for replication streams)
- Compression (zstd integration deferred to optimization phase)
- Zero-copy I/O with io_uring (Linux-specific, defer to M15)

## Next Steps

After completing this task:
- Task 006 (Routing Layer) will route writes to primary, trigger replication
- Task 007 (Gossip Protocol) will handle semantic memory replication
- Task 011 (Jepsen Testing) will validate replication correctness under failures
- Task 012 (Runbook) will document replica promotion procedures
