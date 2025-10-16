//! Server status and health checking

use crate::cli::server::{is_process_running, pid_file_path, read_pid_file};
use anyhow::Result;
use serde_json::Value;
use std::time::{Duration, Instant};

/// Show comprehensive server status
///
/// # Errors
///
/// Returns error if status check fails
pub async fn show_status() -> Result<()> {
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

    // Check if process exists
    if !is_process_running(pid) {
        println!("Process Status: Not running (zombie PID file)");
        println!("Cleanup needed: engram stop");
        return Ok(());
    }

    println!("Process Status: Running");

    // Check HTTP health
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    let health_url = format!("http://127.0.0.1:{port}/health");
    let start_time = Instant::now();

    match client.get(&health_url).send().await {
        Ok(response) => {
            let response_time = start_time.elapsed();

            if response.status().is_success() {
                println!("HTTP Health: Responding ({response_time:?})");

                // Try to get detailed health info
                let detailed_health_url = format!("http://127.0.0.1:{port}/health");
                if let Ok(health_response) = client.get(&detailed_health_url).send().await
                    && let Ok(health_data) = health_response.json::<Value>().await
                {
                    print_health_details(&health_data);
                }
            } else {
                println!("HTTP Health: Unhealthy (status: {})", response.status());
            }
        }
        Err(e) => {
            println!("HTTP Health: Unreachable");
            println!("Error: {e}");
        }
    }

    // Check API endpoints
    println!("\nAPI Endpoints:");
    check_endpoint(
        &client,
        format!("http://127.0.0.1:{port}/api/v1/system/health"),
        "System Health API",
    )
    .await;
    check_endpoint(
        &client,
        format!("http://127.0.0.1:{port}/api/v1/memories/recall?query=test"),
        "Memory Recall API",
    )
    .await;

    println!("\nUseful commands:");
    println!("  engram memory list          # List all memories");
    println!("  engram memory create \"text\" # Create a memory");
    println!("  engram stop                 # Stop the server");

    Ok(())
}

/// Print detailed health information
fn print_health_details(health_data: &Value) {
    println!("\n Detailed Health:");

    if let Some(status) = health_data.get("status").and_then(|s| s.as_str()) {
        println!("  Overall: {status}");
    }

    if let Some(memory_total) = health_data
        .get("memory")
        .and_then(|m| m.get("total_memories"))
        .and_then(Value::as_u64)
    {
        println!("  Memories: {memory_total}");
    }

    if let Some(checks) = health_data.get("checks").and_then(Value::as_array) {
        println!("  Probes:");
        for check in checks {
            let name = check
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let status = check
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let latency = check
                .get("latency_seconds")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            println!("    - {name}: {status} ({latency:.3}s)");

            if let Some(message) = check.get("message").and_then(Value::as_str)
                && !message.is_empty()
            {
                println!("        {message}");
            }
        }
    }
}

/// Check if an API endpoint is responding
async fn check_endpoint(client: &reqwest::Client, url: String, name: &str) {
    match client.get(&url).send().await {
        Ok(response) => {
            println!("  {}: {}", name, response.status());
        }
        Err(_) => {
            println!("   {name}: Unreachable");
        }
    }
}
