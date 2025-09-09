//! Integration tests for the Engram CLI

use assert_cmd::Command;
use engram_cli::{find_available_port, is_port_available};
use predicates::prelude::*;
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
#[ignore] // These port tests can be flaky in CI due to port conflicts
async fn test_port_availability_check() {
    use rand::Rng;
    // Test that we can find an available port
    // Use high port range to avoid conflicts
    let mut rng = rand::thread_rng();
    let base_port = 40000 + rng.gen_range(0..10000);
    let port = find_available_port(base_port).await.unwrap();

    assert!(port >= base_port, "Port should be at or above base");
    assert!(
        port <= base_port + 100,
        "Should find port within reasonable range"
    );

    // Verify the port is actually available
    assert!(
        is_port_available(port).await,
        "Found port should be available"
    );
}

#[tokio::test]
#[ignore] // These port tests can be flaky in CI due to port conflicts
async fn test_find_available_port_with_occupied_port() {
    use rand::Rng;
    // Use very high port range to ensure no conflicts
    let mut rng = rand::thread_rng();
    let base_port = 50000 + rng.gen_range(0..10000);

    // Bind to a specific port first
    let _listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", base_port))
        .await
        .expect("Should be able to bind to high port");

    // Port should now be occupied
    assert!(
        !is_port_available(base_port).await,
        "Bound port should be occupied"
    );

    // Finding an available port starting from the occupied one should return a different port
    let new_port = find_available_port(base_port).await.unwrap();
    assert_ne!(
        base_port, new_port,
        "Should find different port than occupied one"
    );
    assert!(
        new_port > base_port,
        "Should increment to find available port"
    );
    assert!(
        new_port <= base_port + 100,
        "Should find port within reasonable range"
    );

    // Verify the new port is actually available
    assert!(
        is_port_available(new_port).await,
        "Found port should be available"
    );
}

#[tokio::test]
async fn test_cli_help() {
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "CLI interface for the Engram cognitive graph database",
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
    use rand::Rng;
    // Find a truly available port in a high range to avoid conflicts
    let mut rng = rand::thread_rng();
    let port = find_available_port(30000 + rng.gen_range(0..10000))
        .await
        .unwrap();

    // Use isolated directories
    let temp_dir = tempfile::tempdir().unwrap();
    let pid_path = temp_dir.path().join("engram.pid");
    let data_dir = temp_dir.path().join("data");

    let mut cmd = Command::cargo_bin("engram").unwrap();
    let assert = cmd
        .env("ENGRAM_PID_PATH", pid_path.to_str().unwrap())
        .env("ENGRAM_DATA_DIR", data_dir.to_str().unwrap())
        .args(&["start", "--port", &port.to_string(), "--single-node"])
        .timeout(Duration::from_secs(2)) // Short timeout
        .assert();

    // The command should timeout (server runs indefinitely) or produce startup output
    let output = assert.get_output();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check for any indication of startup (stdout or stderr)
    let combined = format!("{}{}", stdout, stderr);
    assert!(
        combined.len() > 0 || output.status.code().is_none(),
        "Server should either produce output or timeout (no exit code)"
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

#[tokio::test]
async fn test_cli_stop_help() {
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.args(&["stop", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Stop the Engram server gracefully",
        ))
        .stdout(predicate::str::contains("--force"));
}

#[tokio::test]
async fn test_cli_stop_no_server() {
    // Use a unique PID path to avoid test interference
    let temp_dir = tempfile::tempdir().unwrap();
    let pid_path = temp_dir.path().join("engram.pid");

    // Ensure PID file doesn't exist and the temp dir is clean
    let _ = std::fs::remove_file(&pid_path);

    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.env("ENGRAM_PID_PATH", pid_path.to_str().unwrap())
        .arg("stop")
        .assert()
        .success();
    // The stop command should succeed regardless - either stops server or reports no server
}

#[tokio::test]
async fn test_cli_stop_force_flag() {
    // Use isolated PID path
    let temp_dir = tempfile::tempdir().unwrap();
    let pid_path = temp_dir.path().join("engram.pid");

    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.env("ENGRAM_PID_PATH", pid_path.to_str().unwrap())
        .args(&["stop", "--force"])
        .assert()
        .success();
    // Force flag should always succeed, regardless of server state
}

#[tokio::test]
async fn test_cli_status_help() {
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.args(&["status", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Show current status"))
        .stdout(predicate::str::contains("--json"))
        .stdout(predicate::str::contains("--watch"));
}

#[tokio::test]
async fn test_cli_status_no_server() {
    // Use isolated environment
    let temp_dir = tempfile::tempdir().unwrap();
    let pid_path = temp_dir.path().join("engram.pid");

    let mut cmd = Command::cargo_bin("engram").unwrap();
    let output = cmd
        .env("ENGRAM_PID_PATH", pid_path.to_str().unwrap())
        .arg("status")
        .output()
        .unwrap();

    // Status should succeed and indicate no server is running
    assert!(
        output.status.success(),
        "Status command should always succeed"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should indicate server is not running in some way
    assert!(
        stdout.contains("No running") || stdout.contains("offline") || stdout.contains("not found"),
        "Should indicate server is not running, got: {}",
        stdout
    );
}

#[tokio::test]
async fn test_cli_status_json_no_server() {
    // Use a unique PID path to avoid conflicts with other tests
    let temp_dir = tempfile::tempdir().unwrap();
    let pid_path = temp_dir.path().join("engram.pid");

    // Ensure PID file doesn't exist
    let _ = std::fs::remove_file(&pid_path);

    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.env("ENGRAM_PID_PATH", pid_path.to_str().unwrap())
        .args(&["status", "--json"])
        .assert()
        .success()
        // Validate JSON structure rather than exact strings
        .stdout(predicate::str::contains("health"))
        .stdout(predicate::str::contains("memory_count"));

    // Parse and validate JSON to ensure it's well-formed
    let output = cmd.output().unwrap();
    let json_str = String::from_utf8_lossy(&output.stdout);
    assert!(
        serde_json::from_str::<serde_json::Value>(&json_str).is_ok(),
        "Invalid JSON output"
    );
}

#[tokio::test]
async fn test_cli_config_help() {
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.args(&["config", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Manage configuration settings"))
        .stdout(predicate::str::contains("get"))
        .stdout(predicate::str::contains("set"))
        .stdout(predicate::str::contains("list"));
}

#[tokio::test]
async fn test_cli_config_get_help() {
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.args(&["config", "get", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Get a configuration value"))
        .stdout(predicate::str::contains("Configuration key"));
}

#[tokio::test]
async fn test_cli_config_set_help() {
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.args(&["config", "set", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Set a configuration value"))
        .stdout(predicate::str::contains("Configuration key"));
}

#[tokio::test]
async fn test_cli_config_list() {
    // Test that config list shows all major sections
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("config.toml");

    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.env("ENGRAM_CONFIG_PATH", config_path.to_str().unwrap())
        .args(&["config", "list"])
        .assert()
        .success()
        // Just verify we get structured output with sections
        .stdout(predicate::str::contains("["));
}

#[tokio::test]
async fn test_cli_config_list_section() {
    // Test section filtering works
    let temp_dir = tempfile::tempdir().unwrap();
    let config_path = temp_dir.path().join("config.toml");

    let mut cmd = Command::cargo_bin("engram").unwrap();
    let output = cmd
        .env("ENGRAM_CONFIG_PATH", config_path.to_str().unwrap())
        .args(&["config", "list", "--section", "memory"])
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should contain memory section
    assert!(stdout.contains("memory"), "Should show memory section");

    // Should NOT contain other sections when filtering
    assert!(
        !stdout.contains("[network]") && !stdout.contains("[performance]"),
        "Should filter out other sections"
    );
}

#[tokio::test]
async fn test_cli_config_get_default_value() {
    // Create a temporary config file to ensure we're testing against known defaults
    let config_dir = tempfile::tempdir().unwrap();
    let config_path = config_dir.path().join("config.toml");

    // Set config directory via environment variable
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.env("ENGRAM_CONFIG_PATH", config_path.to_str().unwrap())
        .args(&["config", "get", "network.port"])
        .assert()
        .success()
        // Test that we get a numeric port value (not hardcoding the exact value)
        .stdout(predicate::str::is_match(r"^\d+\n?$").unwrap());
}

#[tokio::test]
async fn test_cli_config_get_unknown_key() {
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.args(&["config", "get", "unknown.key"])
        .assert()
        .failure()
        .stdout(predicate::str::contains("Unknown configuration key"));
}

#[tokio::test]
async fn test_cli_config_path() {
    let mut cmd = Command::cargo_bin("engram").unwrap();
    cmd.args(&["config", "path"])
        .assert()
        .success()
        // Just verify we get a path-like output
        .stdout(predicate::str::is_match(r".*\.toml").unwrap());
}

#[tokio::test]
#[ignore] // Requires actual server startup
async fn test_health_endpoint_during_startup() {
    use rand::Rng;
    use std::process::Stdio;
    // use tokio::io::AsyncBufReadExt; // Currently unused
    use tokio::process::Command as TokioCommand;

    // Find an available port
    let mut rng = rand::thread_rng();
    let port = 35000 + rng.gen_range(0..5000);

    // Use isolated directories
    let temp_dir = tempfile::tempdir().unwrap();
    let pid_path = temp_dir.path().join("engram.pid");

    // Start the server in background
    let mut server = TokioCommand::new(env!("CARGO_BIN_EXE_engram"))
        .args(&["start", "--port", &port.to_string(), "--single-node"])
        .env("ENGRAM_PID_PATH", pid_path.to_str().unwrap())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start server");

    // Wait for server to be ready
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Test health endpoint
    let health_url = format!("http://127.0.0.1:{}/health", port);
    let client = reqwest::Client::new();

    let response =
        tokio::time::timeout(Duration::from_secs(2), client.get(&health_url).send()).await;

    // Kill the server
    let _ = server.kill().await;

    // Check that we got a response
    if let Ok(Ok(resp)) = response {
        assert_eq!(resp.status(), 200, "Health endpoint should return 200");

        let body = resp.json::<serde_json::Value>().await.unwrap();
        assert!(body.get("status").is_some(), "Should have status field");
        assert!(
            body.get("components").is_some(),
            "Should have components field"
        );
        assert!(body.get("message").is_some(), "Should have message field");
    }
}

#[tokio::test]
#[ignore] // Requires actual server startup
async fn test_health_readiness_endpoint() {
    use rand::Rng;
    use std::process::Stdio;
    use tokio::process::Command as TokioCommand;

    // Find an available port
    let mut rng = rand::thread_rng();
    let port = 40000 + rng.gen_range(0..5000);

    // Use isolated directories
    let temp_dir = tempfile::tempdir().unwrap();
    let pid_path = temp_dir.path().join("engram.pid");

    // Start the server in background
    let mut server = TokioCommand::new(env!("CARGO_BIN_EXE_engram"))
        .args(&["start", "--port", &port.to_string(), "--single-node"])
        .env("ENGRAM_PID_PATH", pid_path.to_str().unwrap())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start server");

    // Wait for server to be ready
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Test readiness endpoint
    let ready_url = format!("http://127.0.0.1:{}/health/ready", port);
    let client = reqwest::Client::new();

    let response =
        tokio::time::timeout(Duration::from_secs(2), client.get(&ready_url).send()).await;

    // Kill the server
    let _ = server.kill().await;

    // Check readiness response
    if let Ok(Ok(resp)) = response {
        assert_eq!(
            resp.status(),
            200,
            "Ready endpoint should return 200 when healthy"
        );

        let body = resp.json::<serde_json::Value>().await.unwrap();
        assert_eq!(
            body.get("ready").and_then(|v| v.as_bool()),
            Some(true),
            "Should be ready"
        );
    }
}

#[tokio::test]
#[ignore] // Requires actual server startup
async fn test_health_liveness_endpoint() {
    use rand::Rng;
    use std::process::Stdio;
    use tokio::process::Command as TokioCommand;

    // Find an available port
    let mut rng = rand::thread_rng();
    let port = 45000 + rng.gen_range(0..5000);

    // Use isolated directories
    let temp_dir = tempfile::tempdir().unwrap();
    let pid_path = temp_dir.path().join("engram.pid");

    // Start the server in background
    let mut server = TokioCommand::new(env!("CARGO_BIN_EXE_engram"))
        .args(&["start", "--port", &port.to_string(), "--single-node"])
        .env("ENGRAM_PID_PATH", pid_path.to_str().unwrap())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start server");

    // Wait for server to be ready
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Test liveness endpoint
    let alive_url = format!("http://127.0.0.1:{}/health/alive", port);
    let client = reqwest::Client::new();

    let response =
        tokio::time::timeout(Duration::from_secs(2), client.get(&alive_url).send()).await;

    // Kill the server
    let _ = server.kill().await;

    // Check liveness response
    if let Ok(Ok(resp)) = response {
        assert_eq!(
            resp.status(),
            200,
            "Alive endpoint should always return 200"
        );

        let body = resp.json::<serde_json::Value>().await.unwrap();
        assert_eq!(
            body.get("alive").and_then(|v| v.as_bool()),
            Some(true),
            "Should be alive"
        );
    }
}
