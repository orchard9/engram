//! Integration tests for the Engram CLI

use assert_cmd::Command;
use engram_cli::{find_available_port, is_port_available};
use predicates::prelude::*;
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
async fn test_port_availability_check() {
    // Test that we can check if a port is available
    let port = find_available_port(8000).await.unwrap();
    assert!(port >= 8000);
    assert!(port <= 8100); // Should find something in the range
}

#[tokio::test]
async fn test_find_available_port_with_occupied_port() {
    use rand::Rng;

    // Use a random port to avoid conflicts between concurrent tests
    let mut rng = rand::thread_rng();
    let base_port = rng.gen_range(20000..30000);

    // Find a port and bind to it
    let port = find_available_port(base_port).await.unwrap();
    let _listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port))
        .await
        .unwrap();

    // Port should now be occupied
    assert!(!is_port_available(port).await);

    // Finding another port should return a different port
    let new_port = find_available_port(port).await.unwrap();
    assert_ne!(port, new_port);
    assert!(new_port > port);
}

#[tokio::test]
async fn test_cli_help() {
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Engram cognitive graph database CLI",
        ));
}

#[tokio::test]
async fn test_cli_start_help() {
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.args(&["start", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Start the Engram server"))
        .stdout(predicate::str::contains("--port"));
}

#[tokio::test]
async fn test_cli_start_single_node() {
    // Test that we can start in single-node mode (but timeout quickly)
    let port = find_available_port(7500).await.unwrap();

    let mut cmd = Command::cargo_bin("engram").unwrap();
    let assert = cmd
        .args(&["start", "--port", &port.to_string(), "--single-node"])
        .timeout(Duration::from_secs(3)) // Will timeout and kill the process
        .assert();

    // The command should timeout (which means it started successfully)
    // We expect it to timeout because the server runs indefinitely
    let output = assert.get_output();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check that startup messages appeared
    assert!(
        stdout.contains("Engram") || stdout.contains("Initializing"),
        "Expected startup messages, got: {}",
        stdout
    );
}

#[tokio::test]
async fn test_port_discovery_timeout() {
    // This test ensures our port finding doesn't hang
    let result = timeout(Duration::from_secs(5), find_available_port(10000)).await;

    assert!(
        result.is_ok(),
        "Port discovery should complete within 5 seconds"
    );
    let port = result.unwrap().unwrap();
    assert!(port >= 10000);
}

#[tokio::test]
async fn test_cli_version() {
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("engram-cli"));
}

#[test]
fn test_cli_invalid_command() {
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.arg("invalid-command")
        .assert()
        .failure()
        .stderr(predicate::str::contains("unrecognized subcommand"));
}
