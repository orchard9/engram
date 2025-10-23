//! Memory operations for the CLI

use anyhow::Result;
use engram_core::MemorySpaceId;
use serde_json::{Value, json};
use std::time::Instant;
use tracing::{error, info};

/// Print formatted memory result
pub fn print_memory_result(memory: &Value) {
    if let Some(obj) = memory.as_object() {
        if let Some(id) = obj.get("id").and_then(|v| v.as_str()) {
            println!("🧠 Memory ID: {id}");
        }

        if let Some(content) = obj.get("content").and_then(|v| v.as_str()) {
            println!("📝 Content: {content}");
        }

        if let Some(confidence) = obj.get("confidence").and_then(serde_json::Value::as_f64) {
            let confidence_bar = "█".repeat((confidence * 10.0).clamp(0.0, 10.0) as usize);
            println!(
                "🎯 Confidence: {:.1}% {}",
                confidence * 100.0,
                confidence_bar
            );
        }

        if let Some(timestamp) = obj.get("timestamp").and_then(|v| v.as_str()) {
            println!("⏰ Created: {timestamp}");
        }

        if let Some(associations) = obj.get("associations").and_then(|v| v.as_array())
            && !associations.is_empty()
        {
            println!("🔗 Associated memories: {}", associations.len());
            for (i, assoc) in associations.iter().take(3).enumerate() {
                if let Some(assoc_obj) = assoc.as_object()
                    && let Some(content) = assoc_obj.get("content").and_then(|v| v.as_str())
                {
                    let preview = if content.len() > 50 {
                        format!("{}...", &content[..47])
                    } else {
                        content.to_string()
                    };
                    println!("  {}. {}", i + 1, preview);
                }
            }
            if associations.len() > 3 {
                println!("  ... and {} more", associations.len() - 3);
            }
        }

        println!();
    } else {
        println!(
            "📄 Raw result: {}",
            serde_json::to_string_pretty(memory).unwrap_or_else(|_| "Invalid JSON".to_string())
        );
    }
}

/// Create a memory through HTTP API
///
/// # Errors
///
/// Returns error if HTTP request fails or server returns an error
pub async fn create_memory(
    port: u16,
    content: String,
    confidence: Option<f64>,
    space_id: &MemorySpaceId,
) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/api/v1/memories");

    let mut payload = json!({
        "content": content
    });

    if let Some(conf) = confidence {
        payload["confidence"] = json!(conf);
    }

    info!("🔄 Creating memory in space '{}'...", space_id.as_str());
    let start_time = Instant::now();

    let response = client
        .post(&url)
        .header("X-Engram-Memory-Space", space_id.as_str())
        .json(&payload)
        .send()
        .await?;

    let elapsed = start_time.elapsed();

    if response.status().is_success() {
        let memory: Value = response.json().await?;
        println!("✅ Memory created successfully in {elapsed:?}");
        print_memory_result(&memory);
    } else {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        error!("❌ Failed to create memory: {}", error_text);
        return Err(anyhow::anyhow!("Failed to create memory: {}", error_text));
    }

    Ok(())
}

/// Retrieve a memory by ID
///
/// # Errors
///
/// Returns error if HTTP request fails or memory not found
pub async fn get_memory(port: u16, id: String, space_id: &MemorySpaceId) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/api/v1/memories/{id}");

    info!("🔍 Retrieving memory from space '{}'...", space_id.as_str());
    let start_time = Instant::now();

    let response = client
        .get(&url)
        .header("X-Engram-Memory-Space", space_id.as_str())
        .send()
        .await?;
    let elapsed = start_time.elapsed();

    if response.status().is_success() {
        let memory: Value = response.json().await?;
        println!("✅ Memory retrieved in {elapsed:?}");
        print_memory_result(&memory);
    } else if response.status() == 404 {
        println!("❌ Memory not found with ID: {id}");
    } else {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        error!("❌ Failed to retrieve memory: {}", error_text);
        return Err(anyhow::anyhow!("Failed to retrieve memory: {}", error_text));
    }

    Ok(())
}

/// Search for memories
///
/// # Errors
///
/// Returns error if HTTP request fails or search fails
pub async fn search_memories(
    port: u16,
    query: String,
    limit: Option<usize>,
    space_id: &MemorySpaceId,
) -> Result<()> {
    let client = reqwest::Client::new();
    let mut url = format!(
        "http://127.0.0.1:{}/api/v1/memories/search?query={}",
        port,
        urlencoding::encode(&query)
    );

    if let Some(lim) = limit {
        use std::fmt::Write;
        let _ = write!(&mut url, "&limit={lim}");
    }

    info!("🔍 Searching memories in space '{}'...", space_id.as_str());
    let start_time = Instant::now();

    let response = client
        .get(&url)
        .header("X-Engram-Memory-Space", space_id.as_str())
        .send()
        .await?;
    let elapsed = start_time.elapsed();

    if response.status().is_success() {
        let results: Value = response.json().await?;

        if let Some(memories) = results.get("memories").and_then(|v| v.as_array()) {
            println!("✅ Found {} memories in {:?}", memories.len(), elapsed);

            if memories.is_empty() {
                println!("🔍 No memories found matching: '{query}'");
                println!("💡 Try a different search term or create some memories first");
            } else {
                for (i, memory) in memories.iter().enumerate() {
                    println!("📋 Result {} of {}:", i + 1, memories.len());
                    print_memory_result(memory);
                }
            }
        } else {
            println!("⚠️  Unexpected response format");
        }
    } else {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        error!("❌ Failed to search memories: {}", error_text);
        return Err(anyhow::anyhow!("Failed to search memories: {}", error_text));
    }

    Ok(())
}

/// List all memories
///
/// # Errors
///
/// Returns error if HTTP request fails or listing fails
pub async fn list_memories(port: u16, limit: Option<usize>, offset: Option<usize>) -> Result<()> {
    let client = reqwest::Client::new();
    let mut url = format!("http://127.0.0.1:{port}/api/v1/memories");

    let mut params = Vec::new();
    if let Some(lim) = limit {
        params.push(format!("limit={lim}"));
    }
    if let Some(off) = offset {
        params.push(format!("offset={off}"));
    }

    if !params.is_empty() {
        url.push('?');
        url.push_str(&params.join("&"));
    }

    info!("📋 Listing memories...");
    let start_time = Instant::now();

    let response = client.get(&url).send().await?;
    let elapsed = start_time.elapsed();

    if response.status().is_success() {
        let results: Value = response.json().await?;

        if let Some(memories) = results.get("memories").and_then(|v| v.as_array()) {
            println!("✅ Retrieved {} memories in {:?}", memories.len(), elapsed);

            if memories.is_empty() {
                println!("📭 No memories found");
                println!(
                    "💡 Create your first memory with: engram memory create \"Your thought here\""
                );
            } else {
                for (i, memory) in memories.iter().enumerate() {
                    println!("📋 Memory {} of {}:", i + 1, memories.len());
                    print_memory_result(memory);
                }
            }

            // Show pagination info if applicable
            if let Some(total) = results.get("total").and_then(serde_json::Value::as_u64) {
                let offset = offset.unwrap_or(0);
                let limit = limit.unwrap_or(10);
                if total > (offset + memories.len()) as u64 {
                    println!("📄 Showing {} of {} total memories", memories.len(), total);
                    println!(
                        "💡 Use --offset {} --limit {} to see more",
                        offset + limit,
                        limit
                    );
                }
            }
        }
    } else {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        error!("❌ Failed to list memories: {}", error_text);
        return Err(anyhow::anyhow!("Failed to list memories: {}", error_text));
    }

    Ok(())
}

/// Delete a memory by ID
///
/// # Errors
///
/// Returns error if HTTP request fails or deletion fails
pub async fn delete_memory(port: u16, id: String) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/api/v1/memories/{id}");

    info!("🗑️  Deleting memory...");

    let response = client.delete(&url).send().await?;

    if response.status().is_success() {
        println!("✅ Memory deleted successfully");
    } else if response.status() == 404 {
        println!("❌ Memory not found with ID: {id}");
    } else {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        error!("❌ Failed to delete memory: {}", error_text);
        return Err(anyhow::anyhow!("Failed to delete memory: {}", error_text));
    }

    Ok(())
}
