//! Server status and health checking

use crate::cli::server::{is_process_running, pid_file_path, read_pid_file};
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::time::{Duration, Instant};
use tracing::{error, info};

/// Show comprehensive server status
pub async fn show_status() -> Result<()> {
    println!("🏥 Engram Server Health Check");
    println!("═══════════════════════════════════════");

    let pid_path = pid_file_path();

    if !pid_path.exists() {
        println!("❌ Status: No running server found");
        println!("💡 Start with: engram start");
        return Ok(());
    }

    let (pid, port) = match read_pid_file() {
        Ok(info) => info,
        Err(e) => {
            println!("⚠️  Status: Server info corrupted");
            println!("🔧 Error: {}", e);
            println!("💡 Try: engram stop && engram start");
            return Ok(());
        }
    };

    println!("📋 Process ID: {}", pid);
    println!("🌐 HTTP Port: {}", port);

    // Check if process exists
    if !is_process_running(pid) {
        println!("💀 Process Status: Not running (zombie PID file)");
        println!("🧹 Cleanup needed: engram stop");
        return Ok(());
    }

    println!("✅ Process Status: Running");

    // Check HTTP health
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    let health_url = format!("http://127.0.0.1:{}/health/alive", port);
    let start_time = Instant::now();

    match client.get(&health_url).send().await {
        Ok(response) => {
            let response_time = start_time.elapsed();

            if response.status().is_success() {
                println!("✅ HTTP Health: Responding ({:?})", response_time);

                // Try to get detailed health info
                let detailed_health_url = format!("http://127.0.0.1:{}/health", port);
                if let Ok(health_response) = client.get(&detailed_health_url).send().await {
                    if let Ok(health_data) = health_response.json::<Value>().await {
                        print_health_details(&health_data);
                    }
                }
            } else {
                println!("⚠️  HTTP Health: Unhealthy (status: {})", response.status());
            }
        }
        Err(e) => {
            println!("❌ HTTP Health: Unreachable");
            println!("🔍 Error: {}", e);
        }
    }

    // Check API endpoints
    println!("\n🔌 API Endpoints:");
    check_endpoint(
        &client,
        format!("http://127.0.0.1:{}/api/v1/memories", port),
        "Memories API",
    )
    .await;
    check_endpoint(
        &client,
        format!("http://127.0.0.1:{}/metrics", port),
        "Metrics",
    )
    .await;

    println!("\n💡 Useful commands:");
    println!("  engram memory list          # List all memories");
    println!("  engram memory create \"text\" # Create a memory");
    println!("  engram stop                 # Stop the server");

    Ok(())
}

/// Print detailed health information
fn print_health_details(health_data: &Value) {
    println!("\n🩺 Detailed Health:");

    if let Some(status) = health_data.get("status").and_then(|s| s.as_str()) {
        let status_emoji = match status {
            "healthy" => "✅",
            "degraded" => "⚠️ ",
            "unhealthy" => "❌",
            _ => "❓",
        };
        println!("  {} Overall: {}", status_emoji, status);
    }

    if let Some(uptime) = health_data.get("uptime_seconds").and_then(|u| u.as_f64()) {
        let hours = uptime / 3600.0;
        if hours < 1.0 {
            println!("  ⏱️  Uptime: {:.1} minutes", uptime / 60.0);
        } else {
            println!("  ⏱️  Uptime: {:.1} hours", hours);
        }
    }

    if let Some(memory_usage) = health_data.get("memory_usage_mb").and_then(|m| m.as_f64()) {
        println!("  🧠 Memory Usage: {:.1} MB", memory_usage);
    }

    if let Some(connections) = health_data
        .get("active_connections")
        .and_then(|c| c.as_u64())
    {
        println!("  🔗 Active Connections: {}", connections);
    }

    if let Some(last_activity) = health_data.get("last_activity").and_then(|l| l.as_str()) {
        if let Ok(datetime) = last_activity.parse::<DateTime<Utc>>() {
            let now = Utc::now();
            let elapsed = now.signed_duration_since(datetime);

            if elapsed.num_seconds() < 60 {
                println!("  🕐 Last Activity: {} seconds ago", elapsed.num_seconds());
            } else if elapsed.num_minutes() < 60 {
                println!("  🕐 Last Activity: {} minutes ago", elapsed.num_minutes());
            } else {
                println!("  🕐 Last Activity: {} hours ago", elapsed.num_hours());
            }
        }
    }

    if let Some(version) = health_data.get("version").and_then(|v| v.as_str()) {
        println!("  📦 Version: {}", version);
    }
}

/// Check if an API endpoint is responding
async fn check_endpoint(client: &reqwest::Client, url: String, name: &str) {
    match client.get(&url).send().await {
        Ok(response) => {
            let status_emoji = if response.status().is_success() {
                "✅"
            } else if response.status().is_client_error() {
                "⚠️ " // Client error might be expected (like 401)
            } else {
                "❌"
            };
            println!("  {} {}: {}", status_emoji, name, response.status());
        }
        Err(_) => {
            println!("  ❌ {}: Unreachable", name);
        }
    }
}
