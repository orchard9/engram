use std::net::SocketAddr;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Top-level cluster configuration shared across the CLI and core crates.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ClusterConfig {
    /// Enables distributed operation; defaults to single-node mode when `false`.
    pub enabled: bool,
    /// Optional stable node identifier (auto-generated if empty).
    pub node_id: String,
    /// Selected discovery mechanism for locating peers.
    pub discovery: DiscoveryConfig,
    /// SWIM protocol tuning parameters.
    pub swim: SwimConfig,
    /// Replication parameters for memory spaces.
    pub replication: ReplicationConfig,
    /// Network-related tuning knobs.
    pub network: NetworkConfig,
    /// Partition detection/healing parameters.
    pub partition: PartitionConfig,
}

/// Supported node discovery strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DiscoveryConfig {
    /// Uses a static seed list provided via configuration.
    Static {
        /// `[host]:[port]` endpoints contacted during startup.
        seed_nodes: Vec<String>,
    },
    /// Discovers peers via DNS SRV records (Kubernetes/cloud native).
    Dns {
        /// Service name to resolve.
        service: String,
        /// Override for the gossip port.
        port: u16,
        /// Cadence for refreshing DNS responses.
        #[serde(default = "default_refresh_interval", with = "humantime_serde")]
        refresh_interval: Duration,
    },
    /// Integrates with Consul-style service registries for discovery.
    Consul {
        /// Consul agent endpoint.
        addr: String,
        /// Service name to query/register.
        service_name: String,
        /// Optional tag filter.
        #[serde(default)]
        tag: Option<String>,
    },
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self::Static {
            seed_nodes: Vec::new(),
        }
    }
}

const fn default_refresh_interval() -> Duration {
    Duration::from_secs(30)
}

/// SWIM protocol tuning knobs loaded from configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SwimConfig {
    /// Interval between direct probes.
    #[serde(with = "humantime_serde")]
    pub ping_interval: Duration,
    /// Timeout before a probe is treated as failed.
    #[serde(with = "humantime_serde")]
    pub ack_timeout: Duration,
    /// Time allotted before suspects become dead.
    #[serde(with = "humantime_serde")]
    pub suspicion_timeout: Duration,
    /// Number of indirect probes to request.
    pub indirect_probes: usize,
    /// Maximum rumors piggybacked per message.
    pub gossip_batch: usize,
}

impl Default for SwimConfig {
    fn default() -> Self {
        Self {
            ping_interval: Duration::from_secs(1),
            ack_timeout: Duration::from_millis(600),
            suspicion_timeout: Duration::from_secs(2),
            indirect_probes: 3,
            gossip_batch: 6,
        }
    }
}

/// Replication parameters for memory spaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReplicationConfig {
    /// Desired replica count per memory space (excluding primary).
    pub factor: usize,
    /// Write timeout for replication acknowledgements.
    #[serde(with = "humantime_serde")]
    pub timeout: Duration,
    /// Strategy for choosing replica nodes.
    pub placement: PlacementStrategy,
    /// Total number of jump-hash buckets to map spaces onto before resolving to a node.
    pub jump_buckets: usize,
    /// Penalty applied when a replica shares the same rack as an existing placement.
    pub rack_penalty: f32,
    /// Penalty applied when a replica shares the same zone as an existing placement.
    pub zone_penalty: f32,
    /// Maximum acceptable lag before emitting warnings.
    #[serde(with = "humantime_serde")]
    pub lag_threshold: Duration,
    /// Target batch size (bytes) for catch-up streaming.
    pub catch_up_batch_bytes: usize,
    /// Preferred compression algorithm for replication batches.
    pub compression: ReplicationCompression,
    /// Enables io_uring fast-paths when available.
    pub io_uring_enabled: bool,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            factor: 2,
            timeout: Duration::from_secs(1),
            placement: PlacementStrategy::Random,
            jump_buckets: default_jump_buckets(),
            rack_penalty: default_rack_penalty(),
            zone_penalty: default_zone_penalty(),
            lag_threshold: Duration::from_secs(5),
            catch_up_batch_bytes: 2 * 1024 * 1024,
            compression: ReplicationCompression::None,
            io_uring_enabled: false,
        }
    }
}

/// Compression strategy for replication batches.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReplicationCompression {
    /// No compression applied.
    #[default]
    None,
    /// LZ4 frame stream compression.
    Lz4,
    /// Zstd compression.
    Zstd,
}

const fn default_jump_buckets() -> usize {
    16_384
}

const fn default_rack_penalty() -> f32 {
    0.5
}

const fn default_zone_penalty() -> f32 {
    0.35
}

/// Replica placement strategies.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PlacementStrategy {
    /// Chooses replicas uniformly at random.
    #[default]
    Random,
    /// Avoid placing more than one replica within the same rack label.
    RackAware,
    /// Avoid placing more than one replica within the same zone label.
    ZoneAware,
}

/// Network tuning knobs for cluster transports.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NetworkConfig {
    /// Address SWIM should bind to for gossip traffic.
    pub swim_bind: String,
    /// Public API bind address.
    pub api_bind: String,
    /// Maximum message size for cluster RPCs.
    pub max_message_size: usize,
    /// Connection pool size per remote node.
    pub connection_pool_size: usize,
    /// Optional externally reachable address for advertisement.
    #[serde(default)]
    pub advertise_addr: Option<SocketAddr>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            swim_bind: "0.0.0.0:7946".to_string(),
            api_bind: "0.0.0.0:50051".to_string(),
            max_message_size: 4 * 1024 * 1024,
            connection_pool_size: 4,
            advertise_addr: None,
        }
    }
}

/// Detection parameters used by the [`PartitionDetector`](crate::cluster::partition::PartitionDetector).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PartitionConfig {
    /// Minimum percentage of nodes that must be reachable to remain connected.
    pub majority_threshold: f64,
    /// Consecutive healthy window required before clearing a partition.
    #[serde(with = "humantime_serde")]
    pub detection_window: Duration,
    /// Cadence for recomputing partition status.
    #[serde(with = "humantime_serde")]
    pub check_interval: Duration,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            majority_threshold: 0.6,
            detection_window: Duration::from_secs(10),
            check_interval: Duration::from_secs(2),
        }
    }
}
