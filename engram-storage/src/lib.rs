//! Tiered storage system for Engram with migration policies
//!
//! Implements cognitive error handling with context, suggestions, and examples
//! for all storage operations.

use engram_core::{MemoryEdge, MemoryNode};
use std::sync::Arc;
use thiserror::Error;

/// Storage tier errors following cognitive guidance principles
#[derive(Error, Debug)]
pub enum StorageError {
    #[error(
        "Storage operation failed: {reason}\n  Context: {context}\n  Suggestion: {suggestion}\n  Example: {example}"
    )]
    OperationFailed {
        reason: String,
        context: String,
        suggestion: String,
        example: String,
    },

    #[error(
        "Node '{id}' not found in {tier} tier\n  Expected: Node stored in current or accessible tier\n  Suggestion: Check if node was migrated to {suggested_tier} tier or use storage.find_node_any_tier()\n  Example: let node = storage.get_node_or_migrate(\"{id}\").await?;"
    )]
    NodeNotFound {
        id: String,
        tier: String,
        suggested_tier: String,
    },

    #[error(
        "Migration from {from_tier} to {to_tier} tier failed: {reason}\n  Expected: Successful data transfer between tiers\n  Suggestion: {suggestion}\n  Example: storage.check_tier_capacity().await?; storage.migrate_with_retry(node).await?;"
    )]
    MigrationFailed {
        from_tier: String,
        to_tier: String,
        reason: String,
        suggestion: String,
    },

    #[error(
        "Serialization failed for {data_type}: {context}\n  Expected: Valid serializable data structure\n  Suggestion: Check for NaN/Infinity in embeddings or use validate_serializable()\n  Example: node.validate_serializable()?.serialize()"
    )]
    SerializationError {
        data_type: String,
        context: String,
        #[source]
        source: bincode::Error,
    },

    #[error(
        "IO operation failed: {operation}\n  Expected: {expected}\n  Suggestion: {suggestion}\n  Example: {example}"
    )]
    IoError {
        operation: String,
        expected: String,
        suggestion: String,
        example: String,
        #[source]
        source: std::io::Error,
    },
}

impl StorageError {
    /// Helper to create `OperationFailed` with full context
    pub fn operation_failed(
        reason: impl Into<String>,
        context: impl Into<String>,
        suggestion: impl Into<String>,
        example: impl Into<String>,
    ) -> Self {
        Self::OperationFailed {
            reason: reason.into(),
            context: context.into(),
            suggestion: suggestion.into(),
            example: example.into(),
        }
    }

    /// Helper to create `NodeNotFound` with tier hints
    pub fn node_not_found(id: impl Into<String>, current_tier: impl Into<String>) -> Self {
        let tier = current_tier.into();
        let suggested = match tier.as_str() {
            "hot" => "warm",
            "warm" => "cold",
            "cold" => "archive",
            _ => "another",
        };

        Self::NodeNotFound {
            id: id.into(),
            tier,
            suggested_tier: suggested.into(),
        }
    }
}

/// Result type for storage operations
pub type Result<T> = std::result::Result<T, StorageError>;

/// Storage tier trait for different storage layers
pub trait StorageTier: Send + Sync {
    /// Store a memory node
    fn store_node(&self, node: MemoryNode) -> impl std::future::Future<Output = Result<()>> + Send;

    /// Retrieve a memory node by ID
    fn get_node(
        &self,
        id: &str,
    ) -> impl std::future::Future<Output = Result<Option<MemoryNode>>> + Send;

    /// Store a memory edge
    fn store_edge(&self, edge: MemoryEdge) -> impl std::future::Future<Output = Result<()>> + Send;

    /// Get edges for a node
    fn get_edges(
        &self,
        node_id: &str,
    ) -> impl std::future::Future<Output = Result<Vec<MemoryEdge>>> + Send;

    /// Check if tier can accept more data
    fn can_accept(&self) -> bool;

    /// Get tier name
    fn tier_name(&self) -> &str;
}

/// Trait for migrateable data between tiers
pub trait Migratable {
    /// Migrate data to another tier
    fn migrate_to<T: StorageTier + ?Sized>(
        &self,
        target: Arc<T>,
    ) -> impl std::future::Future<Output = Result<()>> + Send;

    /// Check if migration is needed
    fn needs_migration(&self) -> bool;
}

/// Hot tier - in-memory storage with lock-free concurrent access
pub mod hot {
    use super::{MemoryEdge, MemoryNode, Result, StorageTier};
    use dashmap::DashMap;

    /// High-performance in-memory storage tier with concurrent access
    pub struct HotStorage {
        nodes: DashMap<String, MemoryNode>,
        edges: DashMap<String, Vec<MemoryEdge>>,
        max_size: usize,
    }

    impl HotStorage {
        /// Create a new hot storage tier with specified maximum capacity
        #[must_use]
        pub fn new(max_size: usize) -> Self {
            Self {
                nodes: DashMap::new(),
                edges: DashMap::new(),
                max_size,
            }
        }
    }

    impl StorageTier for HotStorage {
        async fn store_node(&self, node: MemoryNode) -> Result<()> {
            let id = node.id.clone();
            self.nodes.insert(id, node);
            Ok(())
        }

        async fn get_node(&self, id: &str) -> Result<Option<MemoryNode>> {
            Ok(self.nodes.get(id).map(|n| n.clone()))
        }

        async fn store_edge(&self, edge: MemoryEdge) -> Result<()> {
            self.edges
                .entry(edge.source.clone())
                .or_default()
                .push(edge);
            Ok(())
        }

        async fn get_edges(&self, node_id: &str) -> Result<Vec<MemoryEdge>> {
            Ok(self
                .edges
                .get(node_id)
                .map(|edges| edges.clone())
                .unwrap_or_default())
        }

        fn can_accept(&self) -> bool {
            self.nodes.len() < self.max_size
        }

        fn tier_name(&self) -> &'static str {
            "hot"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hot_storage() {
        use engram_core::MemoryNode;

        let storage = hot::HotStorage::new(100);

        // Use the backwards-compatible constructor for Active state
        let node = MemoryNode::new("test_node".to_string(), vec![1, 2, 3]);

        storage.store_node(node).await.unwrap();

        let retrieved = storage.get_node("test_node").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test_node");
    }
}
