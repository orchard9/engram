use std::net::{SocketAddr, ToSocketAddrs};
use std::str::FromStr;

use crate::cluster::config::DiscoveryConfig;
use async_trait::async_trait;
use thiserror::Error;

#[cfg(feature = "cluster_discovery_dns")]
use trust_dns_resolver::TokioAsyncResolver as AsyncResolver;

/// Errors that can occur during discovery.
#[derive(Debug, Error)]
pub enum DiscoveryError {
    /// Seed entry failed to parse into a socket address.
    #[error("invalid seed node '{addr}': {source}")]
    InvalidSeed {
        /// Original string provided in configuration.
        addr: String,
        /// Parser error emitted by `SocketAddr::from_str`.
        source: std::net::AddrParseError,
    },
    /// Seed hostname could not be resolved via DNS.
    #[error("failed to resolve seed '{addr}': {source}")]
    SeedResolution {
        /// Hostname we attempted to resolve.
        addr: String,
        /// Underlying I/O error raised by the resolver.
        #[source]
        source: std::io::Error,
    },
    /// Requested discovery backend is not compiled in / available.
    #[error("dns discovery unavailable: {0}")]
    DnsUnavailable(&'static str),
    #[cfg(feature = "cluster_discovery_dns")]
    /// Underlying DNS resolver returned an error.
    #[error("dns resolution error: {0}")]
    DnsResolution(#[from] trust_dns_resolver::error::ResolveError),
}

/// Async interface implemented by discovery backends.
#[async_trait]
pub trait ClusterDiscovery: Send + Sync {
    /// Returns the current list of candidate nodes.
    async fn discover(&self) -> Result<Vec<SocketAddr>, DiscoveryError>;
}

/// Convenience trait-object alias for dynamic dispatch.
pub type DynDiscovery = Box<dyn ClusterDiscovery>;

/// Construct a discovery backend from configuration.
pub fn build_discovery(config: &DiscoveryConfig) -> Result<DynDiscovery, DiscoveryError> {
    match config {
        DiscoveryConfig::Static { seed_nodes } => {
            let discovery = StaticDiscovery::new(seed_nodes)?;
            Ok(Box::new(discovery))
        }
        DiscoveryConfig::Dns {
            service,
            port,
            refresh_interval: _,
        } => {
            #[cfg(feature = "cluster_discovery_dns")]
            {
                let resolver = AsyncResolver::tokio_from_system_conf()
                    .map_err(|_| DiscoveryError::DnsUnavailable("failed to init resolver"))?;
                Ok(Box::new(DnsDiscovery {
                    service: service.clone(),
                    port: *port,
                    resolver,
                }))
            }
            #[cfg(not(feature = "cluster_discovery_dns"))]
            {
                let _ = service;
                let _ = port;
                Err(DiscoveryError::DnsUnavailable(
                    "built without cluster_discovery_dns feature",
                ))
            }
        }
        DiscoveryConfig::Consul { .. } => Err(DiscoveryError::DnsUnavailable(
            "consul discovery not yet implemented",
        )),
    }
}

/// Discovery backend backed by a static seed list.
struct StaticDiscovery {
    seeds: Vec<SocketAddr>,
}

impl StaticDiscovery {
    fn new(seed_nodes: &[String]) -> Result<Self, DiscoveryError> {
        let mut seeds = Vec::with_capacity(seed_nodes.len());
        for addr in seed_nodes {
            match SocketAddr::from_str(addr) {
                Ok(parsed) => seeds.push(parsed),
                Err(parse_err) => {
                    let mut resolved = addr.to_socket_addrs().map_err(|source| {
                        DiscoveryError::SeedResolution {
                            addr: addr.clone(),
                            source,
                        }
                    })?;
                    if let Some(sock) = resolved.next() {
                        seeds.push(sock);
                    } else {
                        return Err(DiscoveryError::InvalidSeed {
                            addr: addr.clone(),
                            source: parse_err,
                        });
                    }
                }
            }
        }
        Ok(Self { seeds })
    }
}

#[async_trait]
impl ClusterDiscovery for StaticDiscovery {
    async fn discover(&self) -> Result<Vec<SocketAddr>, DiscoveryError> {
        Ok(self.seeds.clone())
    }
}

#[cfg(feature = "cluster_discovery_dns")]
struct DnsDiscovery {
    service: String,
    port: u16,
    resolver: AsyncResolver,
}

#[cfg(feature = "cluster_discovery_dns")]
#[async_trait]
impl ClusterDiscovery for DnsDiscovery {
    async fn discover(&self) -> Result<Vec<SocketAddr>, DiscoveryError> {
        let response = self.resolver.srv_lookup(self.service.clone()).await?;
        let mut addrs = Vec::new();
        for record in response.iter() {
            let ip_lookup = self.resolver.lookup_ip(record.target().to_utf8()).await?;
            for ip in ip_lookup {
                addrs.push(SocketAddr::new(ip, self.port));
            }
        }
        Ok(addrs)
    }
}
