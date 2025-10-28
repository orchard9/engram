//! Neo4j data source implementation

use migration_common::{Checkpoint, DataSource, MigrationResult, SourceRecord};
use neo4rs::{Graph, query};
use std::collections::HashMap;

/// Neo4j data source for streaming nodes
pub struct Neo4jDataSource {
    graph: Graph,
    batch_size: usize,
    current_offset: usize,
    total_count: Option<u64>,
}

impl Neo4jDataSource {
    /// Create a new Neo4j data source
    pub async fn new(
        uri: &str,
        user: &str,
        password: &str,
        batch_size: usize,
    ) -> MigrationResult<Self> {
        let graph = Graph::new(uri, user, password).await.map_err(|e| {
            migration_common::MigrationError::ConnectionError(format!(
                "Failed to connect to Neo4j: {}",
                e
            ))
        })?;

        // Query total node count
        let count_query = query("MATCH (n) RETURN count(n) as count");
        let mut result = graph.execute(count_query).await.map_err(|e| {
            migration_common::MigrationError::SourceReadError(format!(
                "Failed to query node count: {}",
                e
            ))
        })?;

        let total_count = if let Some(row) = result.next().await.map_err(|e| {
            migration_common::MigrationError::SourceReadError(format!(
                "Failed to read count result: {}",
                e
            ))
        })? {
            row.get::<i64>("count").ok().map(|c| c as u64)
        } else {
            None
        };

        Ok(Self {
            graph,
            batch_size,
            current_offset: 0,
            total_count,
        })
    }
}

impl DataSource for Neo4jDataSource {
    fn next_batch(&mut self) -> MigrationResult<Vec<SourceRecord>> {
        // In production, this would:
        // 1. Execute Cypher query with SKIP and LIMIT
        // 2. Convert Neo4j nodes to SourceRecords
        // 3. Extract properties and labels

        // For now, return empty batch to indicate completion
        Ok(Vec::new())
    }

    fn total_records(&self) -> Option<u64> {
        self.total_count
    }

    fn checkpoint(&self) -> MigrationResult<Option<Checkpoint>> {
        Ok(None)
    }

    fn resume_from(&mut self, checkpoint: &Checkpoint) -> MigrationResult<()> {
        self.current_offset = checkpoint.records_migrated as usize;
        Ok(())
    }
}
