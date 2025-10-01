//! Server management functionality

use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn};

/// Get the path to the PID file
#[must_use]
pub fn pid_file_path() -> PathBuf {
    std::env::var("ENGRAM_PID_PATH")
        .map_or_else(|_| std::env::temp_dir().join("engram.pid"), PathBuf::from)
}

/// Get the path to the state file  
#[must_use]
pub fn _state_file_path() -> PathBuf {
    std::env::temp_dir().join("engram.state")
}

/// Check if Engram server is running and get connection details
///
/// # Errors
///
/// Returns error if server is not running or PID file cannot be read
pub async fn get_server_connection() -> Result<(u16, u16)> {
    let pid_path = pid_file_path();
    if !pid_path.exists() {
        return Err(anyhow::anyhow!(
            "❌ No running Engram server found\n\
              Start a server first with: engram start\n\
             🎯 Then run your memory operations"
        ));
    }

    let (pid, port) = read_pid_file().with_context(|| {
        "Failed to read server information. The server may have crashed.\n\
         Try: engram status  # to check server health\n\
         Or:  engram start   # to start a new server"
    })?;

    // Check if server is actually responding
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;

    let health_url = format!("http://127.0.0.1:{port}/health/alive");

    match client.get(&health_url).send().await {
        Ok(response) if response.status().is_success() => {
            // Server is responding, also get gRPC port (assume default for now)
            Ok((port, 50051)) // (http_port, grpc_port)
        }
        Ok(_) => Err(anyhow::anyhow!(
            "⚠️  Server found but not responding properly (PID: {})\n\
              Try: engram stop && engram start\n\
             🔍 Check: engram status",
            pid
        )),
        Err(_) => Err(anyhow::anyhow!(
            "💔 Server found but unreachable (PID: {})\n\
             🚫 The server process may have crashed or is not listening\n\
             🔧 Try: engram stop && engram start\n\
              Or:  engram status # for detailed diagnostics",
            pid
        )),
    }
}

/// Write PID and port information to file
///
/// # Errors
///
/// Returns error if PID file cannot be written
pub fn write_pid_file(port: u16) -> Result<()> {
    let pid_path = pid_file_path();
    let content = format!("{}:{}", std::process::id(), port);
    fs::write(&pid_path, content)
        .with_context(|| format!("Failed to write PID file: {}", pid_path.display()))?;
    info!(" Server info written to {:?}", pid_path);
    Ok(())
}

/// Read PID and port from file
///
/// # Errors
///
/// Returns error if PID file cannot be read or parsed
pub fn read_pid_file() -> Result<(u32, u16)> {
    let pid_path = pid_file_path();
    let content = fs::read_to_string(&pid_path)
        .with_context(|| format!("Failed to read PID file: {}", pid_path.display()))?;

    let parts: Vec<&str> = content.trim().split(':').collect();
    if parts.len() != 2 {
        return Err(anyhow::anyhow!("Invalid PID file format"));
    }

    let pid: u32 = parts[0].parse().context("Invalid PID in file")?;
    let port: u16 = parts[1].parse().context("Invalid port in file")?;

    Ok((pid, port))
}

/// Remove PID file
///
/// # Errors
///
/// Returns error if PID file cannot be removed
pub fn remove_pid_file() -> Result<()> {
    let pid_path = pid_file_path();
    if pid_path.exists() {
        fs::remove_file(&pid_path)
            .with_context(|| format!("Failed to remove PID file: {}", pid_path.display()))?;
        info!("  Removed server info file");
    }
    Ok(())
}

/// Check if a process is running
#[must_use]
pub fn is_process_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    #[cfg(windows)]
    {
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {}", pid)])
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
            .unwrap_or(false)
    }
}

/// Stop the Engram server
///
/// # Errors
///
/// Returns error if server cannot be stopped or is not running
pub async fn stop_server() -> Result<()> {
    let pid_path = pid_file_path();

    if !pid_path.exists() {
        println!("❌ No running Engram server found");
        return Ok(());
    }

    let (pid, port) = read_pid_file().with_context(|| "Failed to read server information")?;

    if !is_process_running(pid) {
        warn!(
            "⚠️  Server process (PID: {}) not found, cleaning up files",
            pid
        );
        remove_pid_file()?;
        return Ok(());
    }

    info!(" Stopping Engram server (PID: {}, Port: {})", pid, port);

    // Try graceful shutdown first via HTTP API
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    let shutdown_url = format!("http://127.0.0.1:{port}/shutdown");
    match client.post(&shutdown_url).send().await {
        Ok(_) => {
            info!("📨 Sent graceful shutdown signal");

            // Wait for graceful shutdown
            for _ in 0..10 {
                if !is_process_running(pid) {
                    info!(" Server stopped gracefully");
                    remove_pid_file()?;
                    return Ok(());
                }
                sleep(Duration::from_millis(500)).await;
            }

            warn!("⏰ Graceful shutdown timeout, forcing stop");
        }
        Err(e) => {
            warn!("⚠️  Could not send graceful shutdown: {}", e);
        }
    }

    // Force kill if graceful shutdown failed
    #[cfg(unix)]
    {
        std::process::Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .context("Failed to send TERM signal")?;

        // Wait a bit for TERM to work
        for _ in 0..10 {
            if !is_process_running(pid) {
                info!(" Server stopped with TERM signal");
                remove_pid_file()?;
                return Ok(());
            }
            sleep(Duration::from_millis(500)).await;
        }

        // Last resort: KILL
        warn!("🔨 Using KILL signal as last resort");
        std::process::Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status()
            .context("Failed to send KILL signal")?;
    }

    #[cfg(windows)]
    {
        std::process::Command::new("taskkill")
            .args(["/F", "/PID", &pid.to_string()])
            .status()
            .context("Failed to kill process on Windows")?;
    }

    // Final cleanup
    remove_pid_file()?;
    info!(" Server stopped");
    Ok(())
}
