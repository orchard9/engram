use std::cmp::Reverse;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::cluster::config::SwimConfig;
use crate::cluster::error::ClusterError;
use crate::cluster::transport::SwimTransport;
use dashmap::DashMap;
use parking_lot::Mutex;
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time;
use tracing::{debug, trace, warn};

/// Identifies a cluster node along with routing metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Stable logical node identifier.
    pub id: String,
    /// Gossip/SWIM endpoint.
    pub swim_addr: SocketAddr,
    /// API endpoint (gRPC/HTTP) exposed by the node.
    pub api_addr: SocketAddr,
    /// Optional rack label for topology-aware placement.
    pub rack: Option<String>,
    /// Optional zone label for topology-aware placement.
    pub zone: Option<String>,
}

impl NodeInfo {
    /// Creates a new [`NodeInfo`] with explicit routing information.
    pub fn new(
        id: impl Into<String>,
        swim_addr: SocketAddr,
        api_addr: SocketAddr,
        rack: Option<String>,
        zone: Option<String>,
    ) -> Self {
        Self {
            id: id.into(),
            swim_addr,
            api_addr,
            rack,
            zone,
        }
    }
}

#[cfg(test)]
impl Default for NodeInfo {
    fn default() -> Self {
        use std::net::{IpAddr, Ipv4Addr};

        Self {
            id: "node-0".to_string(),
            swim_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 7_946),
            api_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 50_051),
            rack: None,
            zone: None,
        }
    }
}

/// Runtime state of a member inside the SWIM protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeState {
    /// Node is healthy and participating in the cluster.
    Alive,
    /// Node failed to respond and is under suspicion.
    Suspect,
    /// Node exceeded the suspicion timeout and is treated as failed.
    Dead,
    /// Node intentionally left the cluster.
    Left,
}

impl NodeState {
    const fn is_probe_candidate(self) -> bool {
        matches!(self, Self::Alive | Self::Suspect)
    }
}

/// Observer hook used to expose SWIM telemetry to callers without coupling the
/// runtime to a specific metrics implementation.
pub trait SwimObserver: Send + Sync {
    /// Records how long a probe (direct or indirect) took to resolve.
    fn record_probe_latency(&self, latency: Duration);
}

/// Result of a direct or indirect probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeResult {
    /// The probe succeeded and includes the peer's incarnation counter.
    Ack {
        /// Incarnation counter supplied by the responding node.
        incarnation: u64,
    },
    /// The peer explicitly rejected the probe (e.g., overloaded).
    Nack,
    /// The probe timed out with no response.
    Timeout,
}

/// Planned probe for the next SWIM tick.
#[derive(Debug, Clone)]
pub struct ProbePlan {
    /// Node selected for direct probing.
    pub target: NodeInfo,
    /// Additional relay nodes for indirect ping requests.
    pub relays: Vec<NodeInfo>,
}

#[derive(Debug, Clone)]
struct MembershipRumor {
    update: MembershipUpdate,
    last_update: Instant,
}

/// Serializable membership update used for gossip dissemination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembershipUpdate {
    /// Node metadata.
    pub node: NodeInfo,
    /// Reported membership state.
    pub state: NodeState,
    /// Incarnation counter supplied by the peer.
    pub incarnation: u64,
}

/// Aggregate membership statistics for observability.
#[derive(Debug, Clone, Copy, Default)]
pub struct MembershipStats {
    /// Number of `Alive` peers.
    pub alive: usize,
    /// Number of nodes under suspicion.
    pub suspect: usize,
    /// Number of nodes considered dead.
    pub dead: usize,
    /// Nodes that gracefully left the cluster.
    pub left: usize,
}

impl MembershipStats {
    /// Total peers tracked locally.
    #[must_use]
    pub const fn total(self) -> usize {
        self.alive + self.suspect + self.dead + self.left
    }
}

/// Immutable view of a member used for diagnostics.
#[derive(Debug, Clone)]
pub struct MemberSnapshot {
    /// Node metadata.
    pub node: NodeInfo,
    /// Current membership state.
    pub state: NodeState,
    /// Last observed incarnation number.
    pub incarnation: u64,
    /// Timestamp of the last state update.
    pub last_update: Instant,
}

struct MemberRecord {
    node: NodeInfo,
    state: NodeState,
    incarnation: u64,
    last_update: Instant,
    suspect_deadline: Option<Instant>,
}

impl MemberRecord {
    fn snapshot(&self) -> MemberSnapshot {
        MemberSnapshot {
            node: self.node.clone(),
            state: self.state,
            incarnation: self.incarnation,
            last_update: self.last_update,
        }
    }
}

/// SWIM membership engine with deterministic, cache-conscious data layout.
pub struct SwimMembership {
    local: NodeInfo,
    members: DashMap<String, MemberRecord>,
    rng: Mutex<SmallRng>,
    config: SwimConfig,
    local_incarnation: AtomicU64,
    updates_tx: broadcast::Sender<MembershipUpdate>,
}

impl SwimMembership {
    /// Creates a membership engine for the supplied node.
    #[must_use]
    pub fn new(local: NodeInfo, config: SwimConfig) -> Self {
        let (updates_tx, _) = broadcast::channel(256);
        Self {
            local,
            members: DashMap::new(),
            rng: Mutex::new(SmallRng::from_entropy()),
            config,
            local_incarnation: AtomicU64::new(0),
            updates_tx,
        }
    }

    /// Returns the local node metadata used for advertisements.
    #[must_use]
    pub const fn local_node(&self) -> &NodeInfo {
        &self.local
    }

    /// Current incarnation counter for the local node.
    #[must_use]
    pub fn local_incarnation(&self) -> u64 {
        self.local_incarnation.load(Ordering::Acquire)
    }

    fn bump_local_incarnation(&self) -> u64 {
        self.local_incarnation.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Provides read-only access to the SWIM configuration.
    #[must_use]
    pub const fn config(&self) -> &SwimConfig {
        &self.config
    }

    /// Adds or updates a remote node with the provided incarnation counter.
    pub fn upsert_member(&self, node: NodeInfo, incarnation: u64, now: Instant) {
        if node.id == self.local.id {
            return;
        }

        let entry = MemberRecord {
            last_update: now,
            suspect_deadline: None,
            state: NodeState::Alive,
            node,
            incarnation,
        };
        self.members.insert(entry.node.id.clone(), entry);
    }

    /// Removes a node from membership (e.g., graceful leave).
    pub fn remove_member(&self, node_id: &str) {
        self.members.remove(node_id);
    }

    /// Produces a probe plan if any nodes require health checks.
    pub fn choose_probe(&self) -> Option<ProbePlan> {
        let mut candidates: Vec<_> = self
            .members
            .iter()
            .filter(|entry| entry.value().state.is_probe_candidate())
            .map(|entry| entry.value().node.clone())
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let mut rng = self.rng.lock();
        candidates.shuffle(&mut *rng);

        let target = candidates.pop()?;
        let relays = if self.config.indirect_probes > 0 {
            let mut relay_pool: Vec<_> = self
                .members
                .iter()
                .filter(|entry| entry.key().as_str() != target.id.as_str())
                .map(|entry| entry.value().node.clone())
                .collect();
            relay_pool.shuffle(&mut *rng);
            relay_pool
                .into_iter()
                .take(self.config.indirect_probes)
                .collect()
        } else {
            Vec::new()
        };
        drop(rng);

        Some(ProbePlan { target, relays })
    }

    /// Applies the result of a probe. Returns a rumor if the node's state changed.
    pub fn record_probe_result(
        &self,
        node_id: &str,
        result: ProbeResult,
        now: Instant,
    ) -> Option<MembershipUpdate> {
        self.members.get_mut(node_id).and_then(|mut entry| {
            let record = entry.value_mut();
            match result {
                ProbeResult::Ack { incarnation } => {
                    let changed =
                        record.state != NodeState::Alive || record.incarnation < incarnation;
                    record.state = NodeState::Alive;
                    record.incarnation = record.incarnation.max(incarnation);
                    record.last_update = now;
                    record.suspect_deadline = None;
                    changed.then(|| {
                        let update = Self::update_from(record);
                        self.publish_update(update.clone());
                        update
                    })
                }
                ProbeResult::Nack | ProbeResult::Timeout => match record.state {
                    NodeState::Alive => {
                        record.state = NodeState::Suspect;
                        record.last_update = now;
                        record.suspect_deadline = Some(now + self.config.suspicion_timeout);
                        let update = Self::update_from(record);
                        self.publish_update(update.clone());
                        Some(update)
                    }
                    NodeState::Suspect => {
                        if let Some(deadline) = record.suspect_deadline
                            && deadline <= now
                        {
                            record.state = NodeState::Dead;
                            record.last_update = now;
                            let update = Self::update_from(record);
                            self.publish_update(update.clone());
                            Some(update)
                        } else {
                            None
                        }
                    }
                    NodeState::Dead | NodeState::Left => None,
                },
            }
        })
    }

    /// Forces suspicion deadlines to be re-evaluated.
    pub fn reap_timeouts(&self, now: Instant) -> Vec<MembershipUpdate> {
        let mut rumors = Vec::new();
        for mut entry in self.members.iter_mut() {
            let record = entry.value_mut();
            if record.state == NodeState::Suspect
                && record
                    .suspect_deadline
                    .is_some_and(|deadline| deadline <= now)
            {
                record.state = NodeState::Dead;
                record.last_update = now;
                let update = Self::update_from(record);
                self.publish_update(update.clone());
                rumors.push(update);
            }
        }
        rumors
    }

    /// Samples recent membership events for gossip piggybacking.
    pub fn collect_updates(&self) -> Vec<MembershipUpdate> {
        let mut rumors: Vec<_> = self
            .members
            .iter()
            .map(|entry| Self::rumor_from(entry.value()))
            .collect();

        rumors.sort_unstable_by_key(|rumor| Reverse(rumor.last_update));
        rumors
            .into_iter()
            .take(self.config.gossip_batch)
            .map(|rumor| rumor.update)
            .collect()
    }

    /// Applies gossip updates received from other nodes.
    pub fn apply_updates(&self, updates: impl IntoIterator<Item = MembershipUpdate>) {
        let now = Instant::now();
        for update in updates {
            if self.merge_member_state(&update, now) {
                self.publish_update(update);
            }
        }
    }

    /// Returns immutable member snapshots for monitoring.
    pub fn snapshots(&self) -> Vec<MemberSnapshot> {
        self.members
            .iter()
            .map(|entry| entry.value().snapshot())
            .collect()
    }

    /// Number of tracked remote nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns true when no remote members are tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Aggregate state counts for diagnostics/metrics.
    #[must_use]
    pub fn stats(&self) -> MembershipStats {
        let mut stats = MembershipStats::default();
        for entry in &self.members {
            match entry.value().state {
                NodeState::Alive => stats.alive += 1,
                NodeState::Suspect => stats.suspect += 1,
                NodeState::Dead => stats.dead += 1,
                NodeState::Left => stats.left += 1,
            }
        }
        stats
    }

    /// Subscribe to membership updates for coordination tasks.
    pub fn subscribe(&self) -> broadcast::Receiver<MembershipUpdate> {
        self.updates_tx.subscribe()
    }

    fn publish_update(&self, update: MembershipUpdate) {
        let _ = self.updates_tx.send(update);
    }

    /// Returns all nodes currently considered alive, including the local member.
    pub fn alive_nodes(&self) -> Vec<NodeInfo> {
        let mut nodes = Vec::with_capacity(self.members.len() + 1);
        nodes.push(self.local.clone());
        nodes.extend(self.members.iter().filter_map(|entry| {
            (entry.value().state == NodeState::Alive).then(|| entry.value().node.clone())
        }));
        nodes
    }

    pub(crate) fn random_members(&self, max: usize) -> Vec<NodeInfo> {
        let mut peers: Vec<_> = self
            .members
            .iter()
            .map(|entry| entry.value().node.clone())
            .collect();
        {
            let mut rng = self.rng.lock();
            peers.shuffle(&mut *rng);
        }
        peers.truncate(max);
        peers
    }

    #[inline]
    fn update_from(record: &MemberRecord) -> MembershipUpdate {
        MembershipUpdate {
            node: record.node.clone(),
            state: record.state,
            incarnation: record.incarnation,
        }
    }

    fn rumor_from(record: &MemberRecord) -> MembershipRumor {
        MembershipRumor {
            update: Self::update_from(record),
            last_update: record.last_update,
        }
    }

    fn merge_member_state(&self, update: &MembershipUpdate, now: Instant) -> bool {
        if update.node.id == self.local.id {
            let local_incarnation = self.local_incarnation();
            if update.incarnation >= local_incarnation {
                self.bump_local_incarnation();
            }
            return false;
        }

        if self.try_relabel_member(update, now) {
            return true;
        }

        let mut changed = false;
        self.members
            .entry(update.node.id.clone())
            .and_modify(|member| {
                if update.incarnation > member.incarnation
                    || (update.incarnation == member.incarnation && member.state != update.state)
                {
                    member.node = update.node.clone();
                    member.state = update.state;
                    member.incarnation = update.incarnation;
                    member.last_update = now;
                    changed = true;
                }
            })
            .or_insert_with(|| {
                changed = true;
                MemberRecord {
                    node: update.node.clone(),
                    state: update.state,
                    incarnation: update.incarnation,
                    last_update: now,
                    suspect_deadline: None,
                }
            });
        changed
    }

    fn try_relabel_member(&self, update: &MembershipUpdate, now: Instant) -> bool {
        let placeholder = self.members.iter().find_map(|entry| {
            (entry.value().node.swim_addr == update.node.swim_addr
                && entry.key().as_str() != update.node.id.as_str())
            .then(|| entry.key().clone())
        });

        if let Some(old_id) = placeholder
            && let Some((_, mut record)) = self.members.remove(&old_id)
        {
            record.node = update.node.clone();
            record.state = update.state;
            record.incarnation = update.incarnation;
            record.last_update = now;
            record.suspect_deadline = None;
            self.members.insert(update.node.id.clone(), record);
            return true;
        }
        false
    }
}

/// SWIM protocol messages exchanged between peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwimMessage {
    /// Direct ping to target node.
    Ping {
        /// Originating node metadata.
        from: NodeInfo,
        /// Local incarnation counter.
        incarnation: u64,
        /// Membership updates piggybacked on the ping.
        rumors: Vec<MembershipUpdate>,
    },
    /// Acknowledgment of ping.
    Ack {
        /// Node acknowledging the probe.
        from: NodeInfo,
        /// Incarnation supplied by responder.
        incarnation: u64,
        /// Membership updates piggybacked on the ack.
        rumors: Vec<MembershipUpdate>,
    },
    /// Request for indirect probe.
    PingReq {
        /// Requesting node metadata.
        from: NodeInfo,
        /// Target node to probe.
        target: NodeInfo,
        /// Incarnation of the requesting node.
        incarnation: u64,
        /// Recently observed membership updates.
        rumors: Vec<MembershipUpdate>,
    },
    /// Standalone gossip message used for timeout escalation.
    Gossip {
        /// Membership updates disseminated to peers.
        rumors: Vec<MembershipUpdate>,
    },
}

/// Background runtime coordinating SWIM probes and gossip.
pub struct SwimRuntime {
    membership: Arc<SwimMembership>,
    transport: Arc<SwimTransport>,
    pending: DashMap<String, oneshot::Sender<ProbeResult>>,
    observer: Option<Arc<dyn SwimObserver>>,
}

impl SwimRuntime {
    /// Launch SWIM background tasks bound to the provided UDP address.
    pub async fn spawn(
        membership: Arc<SwimMembership>,
        bind_addr: SocketAddr,
        observer: Option<Arc<dyn SwimObserver>>,
    ) -> Result<SwimHandle, ClusterError> {
        let transport = Arc::new(SwimTransport::bind(bind_addr).await?);
        let runtime = Arc::new(Self {
            membership,
            transport,
            pending: DashMap::new(),
            observer,
        });

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut tasks = Vec::with_capacity(4);
        let probe_shutdown = shutdown_rx.clone();
        tasks.push(tokio::spawn(Self::probe_loop(
            Arc::clone(&runtime),
            probe_shutdown,
        )));
        let recv_shutdown = shutdown_rx.clone();
        tasks.push(tokio::spawn(Self::recv_loop(
            Arc::clone(&runtime),
            recv_shutdown,
        )));
        let reap_shutdown = shutdown_rx;
        tasks.push(tokio::spawn(Self::timeout_loop(
            Arc::clone(&runtime),
            reap_shutdown,
        )));
        tasks.push(tokio::spawn(Self::bootstrap(Arc::clone(&runtime))));

        Ok(SwimHandle { shutdown_tx, tasks })
    }

    async fn probe_loop(self: Arc<Self>, mut shutdown: watch::Receiver<bool>) {
        let interval = self.membership.config().ping_interval;
        let mut ticker = time::interval(interval);
        loop {
            tokio::select! {
                _ = shutdown.changed() => break,
                _ = ticker.tick() => {
                    if let Err(err) = self.run_probe_cycle().await {
                        warn!("probe cycle failed: {err}");
                    }
                }
            }
        }
    }

    async fn timeout_loop(self: Arc<Self>, mut shutdown: watch::Receiver<bool>) {
        let mut ticker = time::interval(self.membership.config().suspicion_timeout / 2);
        loop {
            tokio::select! {
                _ = shutdown.changed() => break,
                _ = ticker.tick() => {
                    let updates = self.membership.reap_timeouts(Instant::now());
                    if !updates.is_empty()
                        && let Err(err) = self.broadcast_updates(updates).await
                    {
                        warn!("timeout gossip failed: {err}");
                    }
                }
            }
        }
    }

    async fn recv_loop(self: Arc<Self>, mut shutdown: watch::Receiver<bool>) {
        loop {
            tokio::select! {
                _ = shutdown.changed() => break,
                received = self.transport.recv() => {
                    match received {
                        Ok((message, _addr)) => {
                            if let Err(err) = self.handle_message(message).await {
                                warn!("failed to handle SWIM message: {err}");
                            }
                        }
                        Err(err) => warn!("SWIM recv error: {err}"),
                    }
                }
            }
        }
    }

    async fn bootstrap(self: Arc<Self>) {
        // Give other tasks a moment to start and register receivers.
        time::sleep(Duration::from_millis(50)).await;
        for snapshot in self.membership.snapshots() {
            if let Err(err) = self.send_ping(&snapshot.node).await {
                warn!("bootstrap ping failed: {err}");
            }
        }
    }

    async fn handle_message(self: &Arc<Self>, message: SwimMessage) -> Result<(), ClusterError> {
        match message {
            SwimMessage::Ping {
                from,
                incarnation,
                rumors,
            } => {
                trace!(peer = %from.id, "received ping");
                self.membership
                    .upsert_member(from.clone(), incarnation, Instant::now());
                self.membership.apply_updates(rumors);
                self.send_ack(&from).await?;
            }
            SwimMessage::Ack {
                from,
                incarnation,
                rumors,
            } => {
                trace!(peer = %from.id, "received ack");
                self.membership
                    .upsert_member(from.clone(), incarnation, Instant::now());
                self.membership.apply_updates(rumors);
                self.complete_probe(&from.id, ProbeResult::Ack { incarnation });
            }
            SwimMessage::PingReq {
                from,
                target,
                incarnation,
                rumors,
            } => {
                self.membership
                    .upsert_member(from.clone(), incarnation, Instant::now());
                self.membership.apply_updates(rumors);
                let runtime = Arc::clone(self);
                tokio::spawn(async move {
                    if let Err(err) = runtime.process_ping_req(from, target).await {
                        warn!("ping-req handling failed: {err}");
                    }
                });
            }
            SwimMessage::Gossip { rumors } => {
                self.membership.apply_updates(rumors);
            }
        }
        Ok(())
    }

    async fn process_ping_req(
        &self,
        requester: NodeInfo,
        target: NodeInfo,
    ) -> Result<(), ClusterError> {
        trace!(from = %requester.id, target = %target.id, "handling ping-req");
        if let Some(ProbeResult::Ack { incarnation }) = self.try_direct_probe(&target).await? {
            self.membership.record_probe_result(
                &target.id,
                ProbeResult::Ack { incarnation },
                Instant::now(),
            );
            self.send_forwarded_ack(&target, incarnation, requester.swim_addr)
                .await?;
        }
        Ok(())
    }

    async fn run_probe_cycle(self: &Arc<Self>) -> Result<(), ClusterError> {
        let Some(plan) = self.membership.choose_probe() else {
            return Ok(());
        };
        debug!(target = %plan.target.id, "probing peer");
        let now = Instant::now();
        if let Some(result) = self.try_direct_probe(&plan.target).await? {
            self.apply_probe_result(&plan.target.id, result, now);
            return Ok(());
        }

        if plan.relays.is_empty() {
            self.apply_probe_result(&plan.target.id, ProbeResult::Timeout, Instant::now());
            return Ok(());
        }

        for relay in &plan.relays {
            self.send_ping_req(relay, &plan.target).await?;
        }

        let started = Instant::now();
        if let Some(result) = self.await_probe(&plan.target.id, Some(started)).await? {
            self.apply_probe_result(&plan.target.id, result, Instant::now());
        } else {
            self.apply_probe_result(&plan.target.id, ProbeResult::Timeout, Instant::now());
        }
        Ok(())
    }

    async fn try_direct_probe(
        &self,
        target: &NodeInfo,
    ) -> Result<Option<ProbeResult>, ClusterError> {
        let receiver = self.register_probe(&target.id);
        let start = Instant::now();
        self.send_ping(target).await?;
        self.await_probe_inner(&target.id, receiver, Some(start))
            .await
    }

    fn apply_probe_result(self: &Arc<Self>, node_id: &str, result: ProbeResult, now: Instant) {
        if let Some(update) = self.membership.record_probe_result(node_id, result, now) {
            debug!(node = node_id, state = ?update.state, "membership state changed");
            let runtime = Arc::clone(self);
            tokio::spawn(async move {
                if let Err(err) = runtime.broadcast_updates(vec![update]).await {
                    warn!("failed to broadcast update: {err}");
                }
            });
        }
    }

    fn register_probe(&self, node_id: &str) -> oneshot::Receiver<ProbeResult> {
        let (tx, rx) = oneshot::channel();
        self.pending.insert(node_id.to_string(), tx);
        rx
    }

    async fn await_probe(
        &self,
        node_id: &str,
        started_at: Option<Instant>,
    ) -> Result<Option<ProbeResult>, ClusterError> {
        let receiver = self.register_probe(node_id);
        self.await_probe_inner(node_id, receiver, started_at).await
    }

    async fn await_probe_inner(
        &self,
        node_id: &str,
        receiver: oneshot::Receiver<ProbeResult>,
        started_at: Option<Instant>,
    ) -> Result<Option<ProbeResult>, ClusterError> {
        let timeout = self.membership.config().ack_timeout;
        let result = time::timeout(timeout, receiver).await;
        if let (Some(observer), Some(start)) = (&self.observer, started_at) {
            observer.record_probe_latency(start.elapsed());
        }
        if let Ok(Ok(result)) = result {
            Ok(Some(result))
        } else {
            self.pending.remove(node_id);
            Ok(None)
        }
    }

    fn complete_probe(&self, node_id: &str, result: ProbeResult) {
        if let Some((_, sender)) = self.pending.remove(node_id) {
            let _ = sender.send(result);
        }
        self.membership
            .record_probe_result(node_id, result, Instant::now());
    }

    async fn send_ping(&self, target: &NodeInfo) -> Result<(), ClusterError> {
        let message = SwimMessage::Ping {
            from: self.membership.local_node().clone(),
            incarnation: self.membership.local_incarnation(),
            rumors: self.membership.collect_updates(),
        };
        self.transport.send(&message, target.swim_addr).await
    }

    async fn send_ack(&self, target: &NodeInfo) -> Result<(), ClusterError> {
        let message = SwimMessage::Ack {
            from: self.membership.local_node().clone(),
            incarnation: self.membership.local_incarnation(),
            rumors: self.membership.collect_updates(),
        };
        self.transport.send(&message, target.swim_addr).await
    }

    async fn send_forwarded_ack(
        &self,
        target: &NodeInfo,
        incarnation: u64,
        addr: SocketAddr,
    ) -> Result<(), ClusterError> {
        let message = SwimMessage::Ack {
            from: target.clone(),
            incarnation,
            rumors: self.membership.collect_updates(),
        };
        self.transport.send(&message, addr).await
    }

    async fn send_ping_req(&self, relay: &NodeInfo, target: &NodeInfo) -> Result<(), ClusterError> {
        let message = SwimMessage::PingReq {
            from: self.membership.local_node().clone(),
            target: target.clone(),
            incarnation: self.membership.local_incarnation(),
            rumors: self.membership.collect_updates(),
        };
        self.transport.send(&message, relay.swim_addr).await
    }

    async fn broadcast_updates(&self, updates: Vec<MembershipUpdate>) -> Result<(), ClusterError> {
        if updates.is_empty() {
            return Ok(());
        }
        let peers = self
            .membership
            .random_members(self.membership.config().indirect_probes.max(3));
        for peer in peers {
            let message = SwimMessage::Gossip {
                rumors: updates.clone(),
            };
            self.transport.send(&message, peer.swim_addr).await?;
        }
        Ok(())
    }
}

/// Handle used to control background SWIM tasks.
pub struct SwimHandle {
    shutdown_tx: watch::Sender<bool>,
    tasks: Vec<JoinHandle<()>>,
}

impl SwimHandle {
    /// Signals the SWIM runtime to exit.
    pub fn request_shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    /// Waits for all SWIM tasks to terminate.
    pub async fn wait(self) {
        for task in self.tasks {
            let _ = task.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn make_node(idx: u16) -> NodeInfo {
        NodeInfo {
            id: format!("node-{idx}"),
            swim_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 7_900 + idx),
            api_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5_000 + idx),
            rack: None,
            zone: None,
        }
    }

    #[test]
    fn suspect_escalates_to_dead() {
        let local = make_node(0);
        let membership = SwimMembership::new(local, SwimConfig::default());
        let now = Instant::now();

        membership.upsert_member(make_node(1), 0, now);
        assert_eq!(membership.len(), 1);

        assert!(
            membership
                .record_probe_result("node-1", ProbeResult::Timeout, now)
                .is_some(),
            "state change"
        );

        let rumors = membership.reap_timeouts(now + Duration::from_secs(3));
        assert_eq!(rumors.len(), 1);
        assert_eq!(rumors[0].state, NodeState::Dead);
    }

    #[test]
    fn ack_resets_suspicion() -> Result<(), &'static str> {
        let local = make_node(0);
        let membership = SwimMembership::new(local, SwimConfig::default());
        let now = Instant::now();

        membership.upsert_member(make_node(1), 0, now);
        if membership
            .record_probe_result("node-1", ProbeResult::Timeout, now)
            .is_none()
        {
            return Err("suspect");
        }

        let rumor = membership
            .record_probe_result(
                "node-1",
                ProbeResult::Ack { incarnation: 1 },
                now + Duration::from_millis(10),
            )
            .ok_or("back to alive")?;
        if rumor.state != NodeState::Alive {
            return Err("state");
        }
        if rumor.incarnation != 1 {
            return Err("incarnation");
        }
        Ok(())
    }

    #[test]
    fn choose_probe_skips_empty_membership() {
        let local = make_node(0);
        let membership = SwimMembership::new(local, SwimConfig::default());
        assert!(membership.choose_probe().is_none());
    }

    #[test]
    fn placeholder_seed_is_relabelled_on_first_update() {
        let local = make_node(0);
        let membership = SwimMembership::new(local, SwimConfig::default());
        let now = Instant::now();

        let mut placeholder = make_node(1);
        placeholder.id = "seed-placeholder".to_string();
        membership.upsert_member(placeholder.clone(), 0, now);
        assert_eq!(membership.len(), 1);

        let actual = make_node(1);
        let update = MembershipUpdate {
            node: actual.clone(),
            state: NodeState::Alive,
            incarnation: 4,
        };
        membership.apply_updates([update]);

        let observed: Vec<_> = membership
            .snapshots()
            .into_iter()
            .map(|snapshot| snapshot.node.id)
            .collect();
        assert!(observed.contains(&actual.id));
        assert!(!observed.contains(&placeholder.id));
    }
}
