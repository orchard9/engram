use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use engram_core::cluster::config::ReplicationConfig;
use engram_core::cluster::error::ClusterError;
use engram_core::cluster::{
    NodeInfo, ReplicationBatch, ReplicationMetadata, WalStreamer, WalStreamerConfig,
};
use engram_core::{MemorySpaceId, MemorySpaceRegistry};
use engram_proto::engram_service_client::EngramServiceClient;
use engram_proto::{ApplyReplicationBatchRequest, ReplicationEntryView as ProtoReplicationEntry};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tonic::Request as TonicRequest;
use tracing::{info, warn};

use crate::router::Router;
use engram_core::cluster::SpaceAssignmentManager;

/// Background task that streams WAL entries from the local node to replicas.
pub struct ReplicationRuntime {
    node_id: String,
    assignments: Arc<SpaceAssignmentManager>,
    metadata: Arc<ReplicationMetadata>,
    wal_streamer: Arc<WalStreamer>,
    router: Arc<Router>,
    lag_threshold_sequences: u64,
}

impl ReplicationRuntime {
    /// Create a new runtime bound to the provided planner/router.
    pub fn new(
        node_id: String,
        assignments: Arc<SpaceAssignmentManager>,
        metadata: Arc<ReplicationMetadata>,
        registry: Arc<MemorySpaceRegistry>,
        router: Arc<Router>,
        config: &ReplicationConfig,
    ) -> Self {
        let wal_streamer = Arc::new(WalStreamer::new(
            registry,
            Arc::clone(&metadata),
            WalStreamerConfig {
                max_batch_bytes: config.catch_up_batch_bytes.max(128 * 1024),
                ..WalStreamerConfig::default()
            },
        ));
        let lag_threshold_sequences = config.lag_threshold.as_secs().max(1);
        Self {
            node_id,
            assignments,
            metadata,
            wal_streamer,
            router,
            lag_threshold_sequences,
        }
    }

    /// Shared replication metadata handle (used for diagnostics).
    #[must_use]
    pub fn metadata(&self) -> Arc<ReplicationMetadata> {
        Arc::clone(&self.metadata)
    }

    /// Spawn the runtime until `shutdown` is triggered.
    #[must_use]
    pub fn spawn(self: Arc<Self>, mut shutdown: watch::Receiver<bool>) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(750));
            loop {
                tokio::select! {
                    _ = shutdown.changed() => {
                        info!("Replication runtime shutting down");
                        break;
                    }
                    _ = interval.tick() => {
                        self.tick().await;
                    }
                }
            }
        })
    }

    async fn tick(&self) {
        let spaces = self.assignments.cached_spaces();
        for space in spaces {
            if let Err(err) = self.process_space(&space).await {
                warn!(space = %space, "replication update failed: {err:?}");
            }
        }
    }

    async fn process_space(
        &self,
        space: &MemorySpaceId,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let assignment = match self.assignments.assign(space) {
            Ok(assignment) => assignment,
            Err(ClusterError::NotPrimary { owner, .. }) => {
                warn!(space = %space, owner, "replication assignment not local primary");
                return Ok(());
            }
            Err(err) => {
                warn!(space = %space, "failed to fetch assignment: {err}");
                return Ok(());
            }
        };

        if assignment.primary.id != self.node_id {
            return Ok(());
        }

        for replica in assignment.replicas {
            if replica.id == self.node_id {
                continue;
            }
            self.replicate_to(space, &replica).await?;
        }

        Ok(())
    }

    async fn replicate_to(
        &self,
        space: &MemorySpaceId,
        replica: &NodeInfo,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let start_sequence = self
            .metadata
            .replica_lag(space, &replica.id)
            .map_or(0, |lag| lag.replica_sequence.saturating_add(1));

        let batches = self
            .wal_streamer
            .collect_batches(&self.node_id, space, start_sequence)
            .await?;

        if batches.is_empty() {
            return Ok(());
        }

        for batch in batches {
            self.send_batch(replica, batch).await?;
        }
        Ok(())
    }

    async fn send_batch(
        &self,
        replica: &NodeInfo,
        batch: ReplicationBatch,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let channel = self
            .router
            .client(replica)
            .await
            .with_context(|| format!("connect to replica {}", replica.id))?;
        let mut client = EngramServiceClient::new(channel);
        let request = ApplyReplicationBatchRequest {
            space_id: batch.space.to_string(),
            primary_id: batch.primary_id.clone(),
            start_sequence: batch.start_sequence,
            end_sequence: batch.end_sequence,
            checksum: batch.checksum,
            entries: batch
                .entries
                .iter()
                .map(|entry| ProtoReplicationEntry {
                    sequence: entry.sequence,
                    entry_type: entry.entry_type as u32,
                    payload: entry.payload.clone(),
                })
                .collect(),
        };

        let applied = client
            .apply_replication_batch(TonicRequest::new(request))
            .await?
            .into_inner()
            .applied_through;

        self.metadata
            .record_replica_seq(&batch.space, &replica.id, applied);

        if let Some(lag) = self.metadata.replica_lag(&batch.space, &replica.id)
            && lag.sequences_behind() > self.lag_threshold_sequences
        {
            warn!(
                space = %batch.space,
                replica = %replica.id,
                lag = lag.sequences_behind(),
                "replication lag exceeds configured threshold"
            );
        }

        #[cfg(feature = "monitoring")]
        {
            let bytes = batch.payload_bytes() as u64;
            engram_core::metrics::increment_counter_with_labels(
                engram_core::metrics::REPLICATION_BYTES_TOTAL,
                bytes,
                &[
                    ("direction", "sent".to_string()),
                    ("replica", replica.id.clone()),
                    ("space", batch.space.to_string()),
                ],
            );
        }

        Ok(())
    }
}
