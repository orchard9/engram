//! Server status and health checking

use crate::cli::server::{is_process_running, pid_file_path, read_pid_file};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Per-space health metrics from the API
#[derive(Debug, Deserialize, Serialize)]
struct SpaceHealthMetrics {
    space: String,
    memories: u64,
    pressure: f64,
    wal_lag_ms: f64,
    consolidation_rate: f64,
}

/// Enhanced health response with per-space metrics
#[derive(Debug, Deserialize, Serialize)]
struct HealthResponse {
    status: String,
    timestamp: String,
    checks: Vec<serde_json::Value>,
    spaces: Vec<SpaceHealthMetrics>,
    cluster: Option<ClusterHealthOverview>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ClusterHealthOverview {
    node_id: String,
    alive: usize,
    suspect: usize,
    dead: usize,
    left: usize,
    total: usize,
    #[serde(default)]
    router: Option<RouterStatusOverview>,
}

#[derive(Debug, Deserialize, Serialize)]
struct RouterStatusOverview {
    requests_total: u64,
    retries_total: u64,
    replica_fallback_total: u64,
    open_breakers: usize,
    #[serde(default)]
    breakers: Vec<RouterBreakerStatus>,
}

#[derive(Debug, Deserialize, Serialize)]
struct RouterBreakerStatus {
    node_id: String,
    state: String,
    retry_after_ms: Option<u64>,
}

/// Show comprehensive server status with per-space metrics
///
/// # Errors
///
/// Returns error if status check fails
pub async fn show_status(space_filter: Option<&str>) -> Result<()> {
    println!("Engram Server Health Check");
    println!("═══════════════════════════════════════");

    let pid_path = pid_file_path();

    if !pid_path.exists() {
        println!("Status: No running server found");
        println!("Start with: engram start");
        return Ok(());
    }

    let (pid, port) = match read_pid_file() {
        Ok(info) => info,
        Err(e) => {
            println!(" Status: Server info corrupted");
            println!(" Error: {e}");
            println!(" Try: engram stop && engram start");
            return Ok(());
        }
    };

    println!("Process ID: {pid}");
    println!("HTTP Port: {port}");

    let mut process_running = is_process_running(pid);
    if !process_running && pid == std::process::id() {
        process_running = true;
    }

    if process_running {
        println!("Process Status: Running");
    } else {
        println!("Process Status: Unable to verify (PID: {pid})");
        println!(
            "Hint: containerized deployments often restrict PID probes; continuing with HTTP checks"
        );
    }

    // Check HTTP health
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    let health_url = format!("http://127.0.0.1:{port}/api/v1/system/health");
    let start_time = Instant::now();

    match client.get(&health_url).send().await {
        Ok(response) => {
            let response_time = start_time.elapsed();

            if response.status().is_success() {
                println!("HTTP Health: Responding ({response_time:?})");

                // Try to get detailed health info with per-space metrics
                if let Ok(health_data) = response.json::<HealthResponse>().await {
                    print_health_details(&health_data, space_filter);
                }
            } else {
                println!("HTTP Health: Unhealthy (status: {})", response.status());
            }
        }
        Err(e) => {
            println!("HTTP Health: Unreachable");
            println!("Error: {e}");
            if !process_running {
                println!("Cleanup needed: engram stop");
            }
        }
    }

    println!("\nUseful commands:");
    println!("  engram memory list          # List all memories");
    println!("  engram memory create \"text\" # Create a memory");
    println!("  engram status --space <id>  # Show specific space health");
    println!("  engram stop                 # Stop the server");

    Ok(())
}

/// Print detailed health information with per-space metrics table
fn print_health_details(health_data: &HealthResponse, space_filter: Option<&str>) {
    println!("\nOverall Status: {}", health_data.status);

    if let Some(cluster) = &health_data.cluster {
        println!(
            "Cluster Node: {} | alive {} | suspect {} | dead {} | left {} | total {}",
            cluster.node_id,
            cluster.alive,
            cluster.suspect,
            cluster.dead,
            cluster.left,
            cluster.total
        );

        if let Some(router) = &cluster.router {
            println!(
                "Router: requests {} | retries {} | fallbacks {} | open breakers {}",
                router.requests_total,
                router.retries_total,
                router.replica_fallback_total,
                router.open_breakers
            );
            if !router.breakers.is_empty() {
                println!("  Breakers:");
                for breaker in &router.breakers {
                    match breaker.retry_after_ms {
                        Some(ms) => println!(
                            "    - {}: {} (retry in {ms}ms)",
                            breaker.node_id, breaker.state
                        ),
                        None => println!("    - {}: {}", breaker.node_id, breaker.state),
                    }
                }
            }
        }
    }

    // Filter spaces if requested
    let spaces_to_display: Vec<&SpaceHealthMetrics> = space_filter.map_or_else(
        || health_data.spaces.iter().collect(),
        |filter| {
            health_data
                .spaces
                .iter()
                .filter(|s| s.space == filter)
                .collect()
        },
    );

    if spaces_to_display.is_empty() {
        if let Some(filter) = space_filter {
            println!("\nNo space found matching '{filter}'");
        } else {
            println!("\nNo memory spaces found");
        }
        return;
    }

    // Print per-space metrics table
    println!("\nPer-Space Metrics:");
    println!("┌────────────────────┬───────────┬──────────┬─────────────┬─────────────────┐");
    println!("│ Space              │ Memories  │ Pressure │ WAL Lag (ms)│ Consolidation   │");
    println!("├────────────────────┼───────────┼──────────┼─────────────┼─────────────────┤");

    for space in &spaces_to_display {
        println!(
            "│ {:<18} │ {:>9} │ {:>7.1}% │ {:>11.2} │ {:>13.2}/s │",
            truncate_string(&space.space, 18),
            space.memories,
            space.pressure * 100.0,
            space.wal_lag_ms,
            space.consolidation_rate
        );
    }

    println!("└────────────────────┴───────────┴──────────┴─────────────┴─────────────────┘");

    // Print health checks summary
    if !health_data.checks.is_empty() {
        println!("\nHealth Checks:");
        for check in &health_data.checks {
            let name = check
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let status = check
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let latency = check
                .get("latency_seconds")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            println!("  - {name}: {status} ({latency:.3}s)");

            if let Some(message) = check.get("message").and_then(|v| v.as_str())
                && !message.is_empty()
            {
                println!("      {message}");
            }
        }
    }
}

/// Truncate a string to a maximum length, adding ellipsis if needed
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
