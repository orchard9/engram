#![allow(missing_docs)]
#![allow(clippy::missing_const_for_fn, clippy::unnecessary_cast)]

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use engram_core::cluster::config::SwimConfig;
use engram_core::cluster::{NodeInfo, SwimHandle, SwimMembership, SwimRuntime};

fn test_config() -> SwimConfig {
    SwimConfig {
        ping_interval: Duration::from_millis(75),
        ack_timeout: Duration::from_millis(50),
        suspicion_timeout: Duration::from_millis(200),
        indirect_probes: 2,
        gossip_batch: 8,
    }
}

fn make_node(idx: u16) -> NodeInfo {
    let port = 30_000 + idx;
    NodeInfo::new(
        format!("node-{idx}"),
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port),
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 40_000 + idx),
        None,
        None,
    )
}

async fn start_node(
    node: &NodeInfo,
    peers: &[NodeInfo],
    config: SwimConfig,
) -> (Arc<SwimMembership>, SwimHandle) {
    let membership = Arc::new(SwimMembership::new(node.clone(), config));
    let now = Instant::now();
    for peer in peers {
        if peer.id != node.id {
            membership.upsert_member(peer.clone(), 0, now);
        }
    }
    let handle = SwimRuntime::spawn(Arc::clone(&membership), node.swim_addr, None)
        .await
        .expect("spawn runtime");
    (membership, handle)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial]
async fn swim_cluster_converges() {
    let config = test_config();
    let nodes: Vec<_> = (0..3).map(make_node).collect();

    let mut memberships = Vec::new();
    let mut handles = Vec::new();
    for node in &nodes {
        let (membership, handle) = start_node(node, &nodes, config.clone()).await;
        memberships.push(membership);
        handles.push(handle);
    }

    tokio::time::sleep(Duration::from_secs(2)).await;

    for membership in memberships {
        let stats = membership.stats();
        assert_eq!(stats.alive, 2, "expected two remote peers alive");
        assert_eq!(stats.suspect, 0);
    }

    for handle in handles {
        handle.request_shutdown();
        handle.wait().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[serial_test::serial]
async fn swim_marks_failed_nodes_dead() {
    let config = test_config();
    let nodes: Vec<_> = (0..2).map(make_node).collect();

    let (membership_a, handle_a) = start_node(&nodes[0], &nodes, config.clone()).await;
    let (_membership_b, handle_b) = start_node(&nodes[1], &nodes, config).await;

    tokio::time::sleep(Duration::from_millis(500)).await;

    handle_b.request_shutdown();
    handle_b.wait().await;

    tokio::time::sleep(Duration::from_secs(1)).await;

    let stats = membership_a.stats();
    assert_eq!(
        stats.dead, 1,
        "failed peer should eventually be marked dead"
    );
    handle_a.request_shutdown();
    handle_a.wait().await;
}
