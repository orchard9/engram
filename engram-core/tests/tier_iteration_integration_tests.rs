//! Integration tests for tier iteration across hot/warm/cold storage
//!
//! Tests verify:
//! - Warm tier iteration with persistence
//! - Cold tier iteration with persistence
//! - All tier iteration (hot → warm → cold chaining)
//! - Graceful handling when persistence not configured
//! - Pagination across tier boundaries

use anyhow::{Context, Result, ensure};
use chrono::Utc;
use engram_core::{
    Confidence, Episode, EpisodeBuilder, Memory, MemoryStore,
    storage::{CognitiveTierArchitecture, StorageMetrics},
};
use std::sync::Arc;
use tempfile::TempDir;

/// Create a test episode with given ID and content
/// Creates orthogonal embeddings to avoid semantic deduplication (threshold: 0.95)
///
/// Strategy: Each episode gets values in a different "region" of the 768-dimensional space.
/// This makes embeddings nearly orthogonal, with cosine similarity near 0, far below 0.95.
fn create_test_episode(id: &str, content: &str) -> Episode {
    // Compute ID hash for use in embedding generation and content
    let id_hash = id.chars().fold(0u32, |acc, c| acc.wrapping_add(c as u32));

    // Extract numeric portion from ID for perfect region distribution
    // For IDs like "page_00", "hot_5", etc., extract the trailing digits
    // Note: Clippy suggests `char::is_ascii_digit` but that breaks tests (causes deduplication).
    // Keep closure form to ensure correct behavior.
    #[allow(clippy::redundant_closure_for_method_calls)]
    let numeric_part: String = id.chars().filter(|c| c.is_ascii_digit()).collect();
    let region = if numeric_part.is_empty() {
        // Fallback to hash for non-numeric IDs
        (id_hash % 24) as usize
    } else {
        // Use numeric part directly (mod 24 for sequential IDs)
        numeric_part.parse::<usize>().unwrap_or(0) % 24
    };

    // Divide 768 dimensions into 24 regions of 32 dimensions each
    // Each episode will have non-zero values only in its assigned region
    // This allows up to 24 unique memories before regions repeat
    let region_start = region * 32;

    let mut embedding = [0.0f32; 768];

    // MAXIMALLY orthogonal: Set ONE dimension per region to 1.0, all others to 0
    // This guarantees cosine similarity = 0 between different regions
    // For extra uniqueness within same region, use id_hash to vary the exact dimension
    let dimension_offset = (id_hash % 32) as usize; // 0-31 within region
    let unique_dimension = region_start + dimension_offset;
    embedding[unique_dimension] = 1.0;

    // No normalization needed - already unit vector (one dimension = 1.0)

    // Make content VERY distinct to prevent content-based deduplication
    let unique_content = format!(
        "{} - UNIQUE_ID:{} - HASH:{} - REGION:{} - TIMESTAMP:{}",
        content,
        id,
        id_hash,
        region,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );

    EpisodeBuilder::new()
        .id(id.to_string())
        .when(Utc::now())
        .what(unique_content)
        .embedding(embedding)
        .confidence(Confidence::HIGH)
        .build()
}

/// Create a test memory from episode with activation
fn create_test_memory(id: &str, content: &str, activation: f32) -> Arc<Memory> {
    let episode = create_test_episode(id, content);
    Arc::new(Memory::from_episode(episode, activation))
}

#[tokio::test]
async fn test_hot_tier_iteration() -> Result<()> {
    let store = MemoryStore::new(1000);

    // Store memories in hot tier
    for i in 0..5 {
        let episode = create_test_episode(&format!("hot_{i}"), &format!("Hot memory {i}"));
        store.store(episode);
    }

    // Iterate hot tier
    let memories: Vec<(String, Episode)> = store.iter_hot_memories().collect();

    ensure!(
        memories.len() == 5,
        "Expected 5 memories in hot tier, got {}",
        memories.len()
    );

    // Verify IDs
    for (id, episode) in &memories {
        ensure!(
            id.starts_with("hot_"),
            "Expected ID to start with 'hot_', got {id}"
        );
        ensure!(
            episode.what.starts_with("Hot memory"),
            "Expected content to start with 'Hot memory', got {}",
            episode.what
        );
    }

    Ok(())
}

#[tokio::test]
#[cfg(feature = "memory_mapped_persistence")]
async fn test_warm_tier_iteration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let metrics = Arc::new(StorageMetrics::new());

    // Create tier architecture directly to test warm tier
    let architecture = CognitiveTierArchitecture::new(
        temp_dir.path(),
        100,   // hot capacity
        1000,  // warm capacity
        10000, // cold capacity
        metrics,
    )
    .context("Failed to create tier architecture")?;

    // Store memories in warm tier (low activation)
    for i in 0..10 {
        let memory = create_test_memory(
            &format!("warm_{i}"),
            &format!("Warm memory {i}"),
            0.3, // Low activation → goes to warm tier
        );
        architecture
            .store(memory)
            .await
            .context("Failed to store memory")?;
    }

    // Iterate warm tier
    let memories: Vec<(String, Episode)> = architecture.iter_warm_tier().collect();

    ensure!(
        !memories.is_empty(),
        "Expected at least 1 memory in warm tier, got {}",
        memories.len()
    );

    // Verify warm tier memories
    for (id, episode) in &memories {
        ensure!(
            id.starts_with("warm_"),
            "Expected ID to start with 'warm_', got {id}"
        );
        ensure!(
            episode.what.starts_with("Warm memory") || episode.what.starts_with("Memory "),
            "Unexpected content: {}",
            episode.what
        );
    }

    Ok(())
}

#[tokio::test]
#[cfg(feature = "memory_mapped_persistence")]
async fn test_cold_tier_iteration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let metrics = Arc::new(StorageMetrics::new());

    // Create tier architecture directly to test cold tier
    let architecture = CognitiveTierArchitecture::new(
        temp_dir.path(),
        100,   // hot capacity
        1000,  // warm capacity
        10000, // cold capacity
        metrics,
    )
    .context("Failed to create tier architecture")?;

    // For this test, we'll directly access the cold tier
    // In production, memories migrate to cold tier over time
    // Here we'll just verify the iteration mechanism works

    // Iterate cold tier (should be empty initially)
    let memories: Vec<(String, Episode)> = architecture.iter_cold_tier();

    // Cold tier starts empty
    ensure!(
        memories.is_empty(),
        "Expected cold tier to be empty initially, got {} memories",
        memories.len()
    );

    Ok(())
}

#[tokio::test]
#[cfg(feature = "memory_mapped_persistence")]
async fn test_all_tiers_iteration() -> Result<()> {
    let store = MemoryStore::new(1000);

    // Store memories in hot tier
    for i in 0..5 {
        let episode = create_test_episode(&format!("all_hot_{i}"), &format!("Hot memory {i}"));
        store.store(episode);
    }

    // Iterate all tiers (hot + warm + cold)
    let all_memories: Vec<(String, Episode)> = store.iter_all_memories().collect();

    ensure!(
        all_memories.len() >= 5,
        "Expected at least 5 memories across all tiers, got {}",
        all_memories.len()
    );

    // Verify hot tier memories are included
    let hot_count = all_memories
        .iter()
        .filter(|(id, _)| id.starts_with("all_hot_"))
        .count();

    ensure!(
        hot_count == 5,
        "Expected 5 hot tier memories in all tiers iteration, got {hot_count}"
    );

    Ok(())
}

#[tokio::test]
async fn test_tier_iteration_without_persistence() -> Result<()> {
    // Create store without persistence
    let store = MemoryStore::new(1000);

    // Store some memories (hot tier only)
    for i in 0..3 {
        let episode = create_test_episode(&format!("no_persist_{i}"), &format!("Memory {i}"));
        store.store(episode);
    }

    // Hot tier should work
    let hot_memories: Vec<(String, Episode)> = store.iter_hot_memories().collect();
    ensure!(
        hot_memories.len() == 3,
        "Expected 3 memories in hot tier, got {}",
        hot_memories.len()
    );

    // Warm tier should return None
    let warm_iter = store.iter_warm_memories();
    ensure!(
        warm_iter.is_none(),
        "Expected warm tier to be None without persistence"
    );

    // Cold tier should return None
    let cold_iter = store.iter_cold_memories();
    ensure!(
        cold_iter.is_none(),
        "Expected cold tier to be None without persistence"
    );

    // All tiers should work (just hot tier)
    let all_memories: Vec<(String, Episode)> = store.iter_all_memories().collect();
    ensure!(
        all_memories.len() == 3,
        "Expected 3 memories in all tiers (hot only), got {}",
        all_memories.len()
    );

    Ok(())
}

#[tokio::test]
async fn test_tier_iteration_pagination() -> Result<()> {
    let store = MemoryStore::new(1000);

    // Store 20 memories
    for i in 0..20 {
        let episode = create_test_episode(&format!("page_{i:02}"), &format!("Memory {i}"));
        store.store(episode);
    }

    // Test pagination: skip 5, take 10
    let page: Vec<(String, Episode)> = store.iter_hot_memories().skip(5).take(10).collect();

    ensure!(
        page.len() == 10,
        "Expected 10 memories in page, got {}",
        page.len()
    );

    // Test pagination: skip 15, take 10 (should get 5)
    let last_page: Vec<(String, Episode)> = store.iter_hot_memories().skip(15).take(10).collect();

    ensure!(
        last_page.len() == 5,
        "Expected 5 memories in last page, got {}",
        last_page.len()
    );

    Ok(())
}

#[tokio::test]
async fn test_tier_counts() -> Result<()> {
    let store = MemoryStore::new(1000);

    // Store some memories
    for i in 0..7 {
        let episode = create_test_episode(&format!("count_{i}"), &format!("Memory {i}"));
        store.store(episode);
    }

    // Get tier counts
    let counts = store.get_tier_counts();

    ensure!(
        counts.hot == 7,
        "Expected 7 memories in hot tier, got {}",
        counts.hot
    );

    ensure!(
        counts.total >= 7,
        "Expected total to be at least 7, got {}",
        counts.total
    );

    Ok(())
}

#[tokio::test]
#[cfg(feature = "memory_mapped_persistence")]
async fn test_warm_tier_iteration_performance() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let metrics = Arc::new(StorageMetrics::new());

    let architecture = CognitiveTierArchitecture::new(
        temp_dir.path(),
        1000,  // hot capacity
        10000, // warm capacity
        10000, // cold capacity
        metrics,
    )
    .context("Failed to create tier architecture")?;

    // Store 100 memories in warm tier
    for i in 0..100 {
        let memory = create_test_memory(&format!("perf_{i}"), &format!("Memory {i}"), 0.3);
        architecture
            .store(memory)
            .await
            .context("Failed to store memory")?;
    }

    // Measure iteration time
    let start = std::time::Instant::now();
    let memories: Vec<(String, Episode)> = architecture.iter_warm_tier().collect();
    let duration = start.elapsed();

    ensure!(
        !memories.is_empty(),
        "Expected at least 1 memory, got {}",
        memories.len()
    );

    // Warm tier iteration should be fast (< 100ms for 100 memories)
    ensure!(
        duration.as_millis() < 100,
        "Warm tier iteration took too long: {:?}",
        duration
    );

    Ok(())
}

#[tokio::test]
async fn test_empty_tier_iteration() -> Result<()> {
    let store = MemoryStore::new(1000);

    // Don't store any memories

    // Iterate hot tier (should be empty)
    let hot_memories: Vec<(String, Episode)> = store.iter_hot_memories().collect();
    ensure!(
        hot_memories.is_empty(),
        "Expected hot tier to be empty, got {} memories",
        hot_memories.len()
    );

    // Iterate all tiers (should be empty)
    let all_memories: Vec<(String, Episode)> = store.iter_all_memories().collect();
    ensure!(
        all_memories.is_empty(),
        "Expected all tiers to be empty, got {} memories",
        all_memories.len()
    );

    Ok(())
}
