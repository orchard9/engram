//! Query executor that routes parsed AST queries to engine operations.
//!
//! This module implements the main query executor that:
//! - Validates memory space access through registry
//! - Routes queries to appropriate handlers (MemoryStore, ActivationSpread, etc.)
//! - Constructs evidence chains from query AST
//! - Enforces timeout constraints
//!
//! # Design Philosophy
//!
//! The executor follows a clean separation of concerns:
//! 1. **Parsing**: Converts text to AST (parser module)
//! 2. **Execution**: Maps AST to engine operations (this module)
//! 3. **Results**: Constructs probabilistic results with evidence
//!
//! This allows the parser to be completely independent of execution,
//! enabling testing, optimization, and evolution of each layer separately.

use crate::query::executor::context::QueryContext;
use crate::query::parser::ast::{
    ConsolidateQuery, ImagineQuery, Pattern, PredictQuery, Query, RecallQuery, SpreadQuery,
};
use crate::query::{Evidence, EvidenceSource, MatchType, ProbabilisticQueryResult};
use crate::registry::MemorySpaceRegistry;
use crate::{Confidence, Cue};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use thiserror::Error;
use tokio::time::timeout;

/// Configuration for the AST query executor (routing and validation).
///
/// This is distinct from `QueryExecutorConfig` in the parent module, which
/// configures the probabilistic evidence aggregation executor.
#[derive(Debug, Clone)]
pub struct AstQueryExecutorConfig {
    /// Default timeout for queries without explicit timeout in context
    pub default_timeout: Duration,
    /// Whether to track evidence chains (adds overhead)
    pub track_evidence: bool,
    /// Maximum query complexity allowed (prevents resource exhaustion)
    pub max_query_cost: u64,
}

impl Default for AstQueryExecutorConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            track_evidence: true,
            max_query_cost: 100_000,
        }
    }
}

/// Main query executor that maps parsed queries to engine operations.
///
/// The executor is the bridge between the declarative query language
/// and the imperative engine operations. It ensures:
/// - Multi-tenant isolation through memory space validation
/// - Resource limits through timeout enforcement
/// - Evidence tracking for query provenance
/// - Proper error handling and reporting
///
/// # Architecture
///
/// ```text
/// Parser → AST → QueryExecutor → Engine Operations → Results
///                    ↓
///              Registry (validates memory space)
/// ```
///
/// # Example
///
/// ```ignore
/// use engram_core::query::executor::{QueryExecutor, AstQueryExecutorConfig, QueryContext};
/// use engram_core::query::parser::ast::{Query, Pattern, RecallQuery, NodeIdentifier};
/// use engram_core::registry::MemorySpaceRegistry;
/// use engram_core::MemorySpaceId;
/// use std::sync::Arc;
/// use std::time::Duration;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create registry (normally done at startup)
/// let registry = Arc::new(MemorySpaceRegistry::new(
///     "/tmp/engram",
///     |_id, _dirs| Err(engram_core::registry::MemorySpaceError::NotFound { id: _id.clone() })
/// )?);
///
/// // Create executor
/// let config = AstQueryExecutorConfig::default();
/// let executor = QueryExecutor::new(registry.clone(), config);
///
/// // Create query context
/// let context = QueryContext::with_timeout(
///     MemorySpaceId::new("user_123".to_string()).unwrap(),
///     Duration::from_secs(5),
/// );
///
/// // Build query
/// let query = Query::Recall(RecallQuery {
///     pattern: Pattern::NodeId(NodeIdentifier::from("episode_123")),
///     constraints: vec![],
///     confidence_threshold: None,
///     base_rate: None,
///     limit: Some(10),
/// });
///
/// // Execute query
/// let result = executor.execute(query, context).await?;
/// println!("Found {} episodes", result.len());
/// # Ok(())
/// # }
/// ```
pub struct QueryExecutor {
    registry: Arc<MemorySpaceRegistry>,
    config: AstQueryExecutorConfig,
}

impl QueryExecutor {
    /// Create a new query executor with the given registry and configuration.
    #[must_use]
    pub const fn new(registry: Arc<MemorySpaceRegistry>, config: AstQueryExecutorConfig) -> Self {
        Self { registry, config }
    }

    /// Execute a parsed query within the given context.
    ///
    /// This is the main entry point for query execution. It:
    /// 1. Validates the memory space exists
    /// 2. Checks query complexity against limits
    /// 3. Routes to appropriate handler based on query type
    /// 4. Enforces timeout constraints
    /// 5. Constructs evidence chain with query AST
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Memory space does not exist
    /// - Query exceeds complexity limits
    /// - Query times out
    /// - Underlying engine operation fails
    pub async fn execute(
        &self,
        query: Query<'_>,
        context: QueryContext,
    ) -> Result<ProbabilisticQueryResult, QueryExecutionError> {
        let start = Instant::now();

        // Validate memory space exists
        let space_handle = self.registry.get(&context.memory_space_id).map_err(|e| {
            QueryExecutionError::MemorySpaceNotFound {
                space_id: context.memory_space_id.clone(),
                source: Box::new(e),
            }
        })?;

        // Check query complexity
        let query_cost = query.estimated_cost();
        if query_cost > self.config.max_query_cost {
            return Err(QueryExecutionError::QueryTooComplex {
                cost: query_cost,
                limit: self.config.max_query_cost,
            });
        }

        // Determine effective timeout
        let effective_timeout = context.timeout.unwrap_or(self.config.default_timeout);

        // Execute with timeout enforcement
        match timeout(
            effective_timeout,
            self.execute_inner(query.clone(), &context, space_handle),
        )
        .await
        {
            Ok(Ok(mut result)) => {
                // Add query AST to evidence chain if tracking enabled
                if self.config.track_evidence {
                    let query_evidence = Self::create_query_evidence(&query);
                    result.evidence_chain.insert(0, query_evidence);
                }
                Ok(result)
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Err(QueryExecutionError::Timeout {
                duration: effective_timeout,
                elapsed: start.elapsed(),
            }),
        }
    }

    /// Internal execution without timeout wrapper.
    ///
    /// Routes query to appropriate handler based on type.
    ///
    /// This function is async to support timeout enforcement via tokio::time::timeout,
    /// even though the underlying handlers are currently synchronous. This allows
    /// for future async implementations without changing the public API.
    #[allow(clippy::unused_async)]
    async fn execute_inner(
        &self,
        query: Query<'_>,
        context: &QueryContext,
        space_handle: Arc<crate::registry::SpaceHandle>,
    ) -> Result<ProbabilisticQueryResult, QueryExecutionError> {
        match query {
            Query::Recall(q) => self.execute_recall(&q, context, &space_handle),
            Query::Spread(q) => Self::execute_spread(&q, context, &space_handle),
            Query::Predict(q) => Self::execute_predict(&q, context, &space_handle),
            Query::Imagine(q) => Self::execute_imagine(&q, context, &space_handle),
            Query::Consolidate(q) => Self::execute_consolidate(&q, context, &space_handle),
        }
    }

    /// Execute RECALL query - retrieve memories matching pattern.
    fn execute_recall(
        &self,
        query: &RecallQuery<'_>,
        _context: &QueryContext,
        space_handle: &Arc<crate::registry::SpaceHandle>,
    ) -> Result<ProbabilisticQueryResult, QueryExecutionError> {
        // Convert pattern to cue
        let cue = Self::pattern_to_cue(&query.pattern)?;

        // Execute recall through memory store
        let recall_result = space_handle.store().recall(&cue);
        let mut episodes = recall_result.results;

        // Apply limit if specified
        if let Some(limit) = query.limit {
            episodes.truncate(limit);
        }

        // Apply confidence threshold if specified
        let episodes = if let Some(threshold) = query.confidence_threshold {
            episodes
                .into_iter()
                .filter(|(_, conf)| threshold.matches(*conf))
                .collect()
        } else {
            episodes
        };

        // Create result with evidence
        let mut result = ProbabilisticQueryResult::from_episodes(episodes);

        // Add direct match evidence for the pattern
        if self.config.track_evidence {
            let pattern_evidence = Evidence {
                source: EvidenceSource::DirectMatch {
                    cue_id: format!("recall_pattern_{:?}", query.pattern),
                    similarity_score: 1.0,
                    match_type: MatchType::Semantic,
                },
                strength: Confidence::HIGH,
                timestamp: SystemTime::now(),
                dependencies: vec![],
            };
            result.evidence_chain.push(pattern_evidence);
        }

        Ok(result)
    }

    /// Execute SPREAD query - activation spreading from source.
    fn execute_spread(
        query: &SpreadQuery<'_>,
        context: &QueryContext,
        space_handle: &Arc<crate::registry::SpaceHandle>,
    ) -> Result<ProbabilisticQueryResult, QueryExecutionError> {
        // Use the spread executor module for SPREAD query execution
        crate::query::executor::spread::execute_spread(query, context, &space_handle.store())
            .map_err(|e| QueryExecutionError::ExecutionFailed {
                message: e.to_string(),
            })
    }

    /// Execute PREDICT query - predict future states.
    fn execute_predict(
        query: &PredictQuery<'_>,
        context: &QueryContext,
        space_handle: &Arc<crate::registry::SpaceHandle>,
    ) -> Result<ProbabilisticQueryResult, QueryExecutionError> {
        // Use the predict executor module for PREDICT query execution
        crate::query::executor::predict::execute_predict(query, context, space_handle).map_err(
            |e| QueryExecutionError::ExecutionFailed {
                message: e.to_string(),
            },
        )
    }

    /// Execute IMAGINE query - pattern completion.
    fn execute_imagine(
        query: &ImagineQuery<'_>,
        context: &QueryContext,
        space_handle: &Arc<crate::registry::SpaceHandle>,
    ) -> Result<ProbabilisticQueryResult, QueryExecutionError> {
        // Use the imagine executor module for IMAGINE query execution
        crate::query::executor::imagine::execute_imagine(query, context, space_handle).map_err(
            |e| QueryExecutionError::ExecutionFailed {
                message: e.to_string(),
            },
        )
    }

    /// Execute CONSOLIDATE query - memory consolidation.
    fn execute_consolidate(
        query: &ConsolidateQuery<'_>,
        context: &QueryContext,
        space_handle: &Arc<crate::registry::SpaceHandle>,
    ) -> Result<ProbabilisticQueryResult, QueryExecutionError> {
        // Use the consolidate executor module for CONSOLIDATE query execution
        crate::query::executor::consolidate::execute_consolidate(query, context, space_handle)
            .map_err(|e| QueryExecutionError::ExecutionFailed {
                message: e.to_string(),
            })
    }

    /// Convert AST pattern to Cue for memory store recall.
    fn pattern_to_cue(pattern: &Pattern<'_>) -> Result<Cue, QueryExecutionError> {
        match pattern {
            Pattern::NodeId(id) => {
                // Create a semantic cue using the node ID
                Ok(Cue::semantic(
                    id.to_string(),
                    id.to_string(),
                    Confidence::MEDIUM,
                ))
            }
            Pattern::Embedding { vector, threshold } => {
                // Convert Vec<f32> to fixed-size array
                if vector.len() != 768 {
                    return Err(QueryExecutionError::InvalidPattern {
                        reason: format!("Expected 768-dimensional embedding, got {}", vector.len()),
                    });
                }

                let mut embedding = [0.0f32; 768];
                embedding.copy_from_slice(vector);

                // Create an embedding cue
                Ok(Cue::embedding(
                    "embedding_pattern".to_string(),
                    embedding,
                    Confidence::from_raw(*threshold),
                ))
            }
            Pattern::ContentMatch(text) => {
                // Create a semantic cue from the content
                Ok(Cue::semantic(
                    "content_match".to_string(),
                    text.to_string(),
                    Confidence::MEDIUM,
                ))
            }
            Pattern::Any => {
                // "Any" pattern retrieves all memories with low threshold
                Ok(Cue::semantic(
                    "any_pattern".to_string(),
                    String::new(),
                    Confidence::LOW,
                ))
            }
        }
    }

    /// Create evidence entry for the query AST itself.
    ///
    /// This enables tracing query execution back to the original query.
    fn create_query_evidence(query: &Query<'_>) -> Evidence {
        Evidence {
            source: EvidenceSource::DirectMatch {
                cue_id: format!("query_ast_{}", query.category().as_str()),
                similarity_score: 1.0,
                match_type: MatchType::Semantic,
            },
            strength: Confidence::HIGH,
            timestamp: SystemTime::now(),
            dependencies: vec![],
        }
    }
}

/// Query execution errors with detailed context.
#[derive(Debug, Error)]
pub enum QueryExecutionError {
    /// Memory space does not exist in registry
    #[error("Memory space not found: {space_id}")]
    MemorySpaceNotFound {
        /// The memory space ID that was not found
        space_id: crate::MemorySpaceId,
        /// The underlying error from the registry
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Query exceeded complexity limits
    #[error("Query too complex: cost={cost}, limit={limit}")]
    QueryTooComplex {
        /// Computed cost of the query
        cost: u64,
        /// Maximum allowed cost
        limit: u64,
    },

    /// Query execution timed out
    #[error("Query timed out after {duration:?} (elapsed: {elapsed:?})")]
    Timeout {
        /// Configured timeout duration
        duration: Duration,
        /// Actual elapsed time
        elapsed: Duration,
    },

    /// Query type not yet implemented
    #[error("Query type {query_type} not implemented: {reason}")]
    NotImplemented {
        /// Type of query (SPREAD, PREDICT, etc.)
        query_type: String,
        /// Reason for not being implemented
        reason: String,
    },

    /// Invalid pattern in query
    #[error("Invalid pattern: {reason}")]
    InvalidPattern {
        /// Reason the pattern is invalid
        reason: String,
    },

    /// Generic execution error
    #[error("Query execution failed: {message}")]
    ExecutionFailed {
        /// Error message
        message: String,
    },
}

impl crate::query::parser::ast::QueryCategory {
    /// Convert category to string for logging/metrics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Recall => "RECALL",
            Self::Spread => "SPREAD",
            Self::Predict => "PREDICT",
            Self::Imagine => "IMAGINE",
            Self::Consolidate => "CONSOLIDATE",
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]
    #![allow(clippy::field_reassign_with_default)]
    #![allow(clippy::unnecessary_to_owned)]

    use super::*;
    use crate::query::parser::ast::NodeIdentifier;
    use crate::registry::MemorySpaceRegistry;
    use crate::{MemorySpaceId, MemoryStore};

    fn create_test_registry() -> Arc<MemorySpaceRegistry> {
        let registry = MemorySpaceRegistry::new("/tmp/engram_test", |_id, _dirs| {
            Ok(Arc::new(MemoryStore::new(1000)))
        })
        .expect("Failed to create registry");
        Arc::new(registry)
    }

    #[tokio::test]
    async fn test_executor_creation() {
        let registry = create_test_registry();
        let config = AstQueryExecutorConfig::default();
        let executor = QueryExecutor::new(registry, config);

        assert!(executor.config.track_evidence);
    }

    #[tokio::test]
    async fn test_memory_space_validation() {
        let registry = create_test_registry();
        let executor = QueryExecutor::new(registry, AstQueryExecutorConfig::default());

        let query = Query::Recall(RecallQuery {
            pattern: Pattern::NodeId(NodeIdentifier::from("test")),
            constraints: vec![],
            confidence_threshold: None,
            base_rate: None,
            limit: None,
        });

        let context = QueryContext::without_timeout(
            MemorySpaceId::new("nonexistent_space".to_string()).unwrap(),
        );

        let result = executor.execute(query, context).await;
        assert!(matches!(
            result,
            Err(QueryExecutionError::MemorySpaceNotFound { .. })
        ));
    }

    #[tokio::test]
    async fn test_query_complexity_limit() {
        let registry = create_test_registry();
        let mut config = AstQueryExecutorConfig::default();
        config.max_query_cost = 10; // Very low limit
        let executor = QueryExecutor::new(registry.clone(), config);

        // Create a memory space first
        let space_id = MemorySpaceId::new("test_space".to_string()).unwrap();
        registry
            .create_or_get(&space_id)
            .await
            .expect("Failed to create space");

        // Create a complex query (SPREAD with many hops = high cost)
        let query = Query::Spread(SpreadQuery {
            source: NodeIdentifier::from("source"),
            max_hops: Some(10), // This creates exponential cost
            decay_rate: None,
            activation_threshold: None,
            refractory_period: None,
        });

        let context = QueryContext::without_timeout(space_id);

        let result = executor.execute(query, context).await;
        assert!(matches!(
            result,
            Err(QueryExecutionError::QueryTooComplex { .. })
        ));
    }

    #[tokio::test]
    async fn test_timeout_enforcement() {
        let registry = create_test_registry();
        let executor = QueryExecutor::new(registry.clone(), AstQueryExecutorConfig::default());

        // Create memory space
        let space_id = MemorySpaceId::new("test_space".to_string()).unwrap();
        registry
            .create_or_get(&space_id)
            .await
            .expect("Failed to create space");

        // Create a query that's not implemented (will error before timeout in practice)
        let query = Query::Spread(SpreadQuery {
            source: NodeIdentifier::from("source"),
            max_hops: Some(3),
            decay_rate: None,
            activation_threshold: None,
            refractory_period: None,
        });

        // Very short timeout
        let context = QueryContext::with_timeout(space_id, Duration::from_millis(1));

        let result = executor.execute(query, context).await;

        // Should get either timeout or not implemented error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pattern_to_cue_conversion() {
        // Test NodeId pattern
        let pattern = Pattern::NodeId(NodeIdentifier::from("episode_123"));
        let cue = QueryExecutor::pattern_to_cue(&pattern).expect("Failed to convert");
        assert_eq!(cue.id, "episode_123");

        // Test ContentMatch pattern
        let pattern = Pattern::ContentMatch("test content".into());
        let cue = QueryExecutor::pattern_to_cue(&pattern).expect("Failed to convert");
        assert_eq!(cue.id, "content_match");

        // Test Any pattern
        let pattern = Pattern::Any;
        let cue = QueryExecutor::pattern_to_cue(&pattern).expect("Failed to convert");
        assert_eq!(cue.id, "any_pattern");
    }

    #[tokio::test]
    async fn test_invalid_embedding_dimension() {
        // Wrong dimension (not 768)
        let pattern = Pattern::Embedding {
            vector: vec![0.5; 128], // Wrong size
            threshold: 0.8,
        };

        let result = QueryExecutor::pattern_to_cue(&pattern);
        assert!(matches!(
            result,
            Err(QueryExecutionError::InvalidPattern { .. })
        ));
    }
}
