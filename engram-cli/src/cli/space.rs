//! Memory space management commands for the Engram CLI.

use super::server::get_server_connection;
use anyhow::{Context, Result, anyhow};
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::json;
use std::fmt::Write as _;

#[derive(Deserialize)]
struct MemorySpaceDescriptor {
    id: String,
    persistence_root: String,
    created_at: String,
}

#[derive(Deserialize)]
struct MemorySpaceListResponse {
    spaces: Vec<MemorySpaceDescriptor>,
}

fn format_error(status: StatusCode, body: &str) -> anyhow::Error {
    let mut message = format!("Server responded with status {status}");
    if !body.trim().is_empty() {
        let _ = write!(message, ": {}", body.trim());
    }
    anyhow!(message)
}

/// List all registered memory spaces via the HTTP control plane.
pub async fn list_spaces() -> Result<()> {
    let (port, _grpc_port) = get_server_connection().await?;
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/api/v1/spaces");

    let response = client
        .get(&url)
        .send()
        .await
        .context("failed to query memory spaces")?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format_error(status, &body));
    }

    let payload: MemorySpaceListResponse = response
        .json()
        .await
        .context("failed to parse memory space list response")?;

    if payload.spaces.is_empty() {
        println!("No memory spaces registered yet. Use 'engram space create <id>' to add one.");
        return Ok(());
    }

    println!("Registered memory spaces:\n");
    println!("{:<20}  {:<48}  CREATED AT", "ID", "PERSISTENCE ROOT");
    println!("{:-<20}  {:-<48}  {:-<24}", "", "", "");

    for space in payload.spaces {
        println!(
            "{:<20}  {:<48}  {}",
            space.id, space.persistence_root, space.created_at
        );
    }

    Ok(())
}

/// Create (or fetch) a memory space via the HTTP control plane.
pub async fn create_space(id: String) -> Result<()> {
    let (port, _grpc_port) = get_server_connection().await?;
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/api/v1/spaces");

    let response = client
        .post(&url)
        .json(&json!({ "id": id }))
        .send()
        .await
        .context("failed to create memory space")?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format_error(status, &body));
    }

    let descriptor: MemorySpaceDescriptor = response
        .json()
        .await
        .context("failed to parse memory space creation response")?;

    println!(
        "Memory space '{}' ready at {} (created at {})",
        descriptor.id, descriptor.persistence_root, descriptor.created_at
    );

    Ok(())
}
