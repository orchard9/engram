#![allow(missing_docs)]

use engram_cli::cluster::{ClusterContext, initialize_cluster};
use engram_core::cluster::config::{ClusterConfig, DiscoveryConfig};

#[tokio::test]
async fn initialize_cluster_bootstraps_static_seeds() {
    let config = ClusterConfig {
        enabled: true,
        network: engram_core::cluster::config::NetworkConfig {
            swim_bind: "127.0.0.1:7900".to_string(),
            api_bind: "127.0.0.1:5100".to_string(),
            advertise_addr: Some("127.0.0.1:7900".parse().unwrap()),
            ..Default::default()
        },
        discovery: DiscoveryConfig::Static {
            seed_nodes: vec!["localhost:7946".to_string()],
        },
        ..Default::default()
    };

    let context = initialize_cluster(&config).await.expect("cluster init");
    if let ClusterContext::Distributed {
        membership, seeds, ..
    } = context
    {
        assert_eq!(seeds.len(), 1, "discovery should return resolved hostnames");
        assert_eq!(
            membership.stats().alive,
            1,
            "resolved seed should be present in membership"
        );
    } else {
        panic!("expected distributed context");
    }
}

#[tokio::test]
async fn initialize_cluster_errors_without_advertise_on_wildcard() {
    // Test that binding to wildcard without advertise_addr and without seeds fails
    let config = ClusterConfig {
        enabled: true,
        network: engram_core::cluster::config::NetworkConfig {
            swim_bind: "0.0.0.0:7946".to_string(),
            api_bind: "127.0.0.1:50051".to_string(),
            ..Default::default()
        },
        discovery: DiscoveryConfig::Static {
            seed_nodes: vec![], // Empty seeds - can't probe for local address
        },
        ..Default::default()
    };

    let Err(err) = initialize_cluster(&config).await else {
        panic!("expected advertise address validation to fail")
    };
    let message = format!("{err}");
    assert!(message.contains("advertise address"));
}
