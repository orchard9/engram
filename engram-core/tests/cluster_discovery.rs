#![allow(missing_docs)]

use engram_core::cluster::config::DiscoveryConfig;
use engram_core::cluster::discovery::{DiscoveryError, build_discovery};

#[tokio::test]
async fn static_discovery_resolves_hostnames() {
    let config = DiscoveryConfig::Static {
        seed_nodes: vec!["localhost:7946".to_string()],
    };

    let discovery = build_discovery(&config).expect("build static discovery");
    let seeds = discovery.discover().await.expect("discover peers");

    assert!(
        seeds.iter().any(|addr| addr.port() == 7946),
        "static discovery should resolve localhost"
    );
}

#[test]
fn consul_discovery_surfaces_not_implemented() {
    let config = DiscoveryConfig::Consul {
        addr: "http://127.0.0.1:8500".to_string(),
        service_name: "engram".to_string(),
        tag: None,
    };

    let Err(err) = build_discovery(&config) else {
        panic!("consul backend should be deferred")
    };
    match err {
        DiscoveryError::DnsUnavailable(message) => {
            assert!(message.contains("not yet implemented"));
        }
        other => panic!("unexpected error for consul discovery: {other:?}"),
    }
}
