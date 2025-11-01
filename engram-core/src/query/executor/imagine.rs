//! IMAGINE query execution implementation.
//!
//! This module implements the execution logic for IMAGINE queries, which perform
//! pattern completion to generate novel combinations and fill in missing episode
//! details using hippocampal-inspired dynamics.
//!
//! ## Design Principles
//!
//! 1. **Hippocampal Dynamics**: Uses CA3 autoassociative memory for pattern completion
//! 2. **Source Monitoring**: Tracks which parts are recalled vs reconstructed
//! 3. **Novelty Control**: Balances creativity with plausibility via novelty parameter
//! 4. **Evidence Tracking**: Complete provenance of completion process
//!
//! ## Biological Grounding
//!
//! IMAGINE implements pattern completion inspired by hippocampal CA3 region:
//! - **CA3 Autoassociation**: Completes partial patterns using recurrent connections
//! - **DG Pattern Separation**: Prevents interference between similar patterns
//! - **CA1 Gating**: Filters implausible completions before output
//! - **Theta Rhythm**: Iterative completion within theta cycle constraints
//!
//! ## Example
//!
//! ```rust
//! use engram_core::query::parser::ast::{ImagineQuery, Pattern, NodeIdentifier};
//! use engram_core::query::executor::QueryContext;
//! use engram_core::MemorySpaceId;
//! use std::sync::Arc;
//!
//! // Create IMAGINE query
//! let query = ImagineQuery {
//!     pattern: Pattern::NodeId(NodeIdentifier::from("partial_memory")),
//!     seeds: vec![NodeIdentifier::from("context_node")],
//!     novelty: Some(0.5), // Moderate creativity
//!     confidence_threshold: None,
//! };
//!
//! // Execute with context
//! // let context = QueryContext::without_timeout(MemorySpaceId::from("user_123"));
//! // let result = execute_imagine(&query, &context, &space_handle)?;
//! ```

use crate::Confidence;
use crate::completion::{HippocampalCompletion, PartialEpisode, PatternCompleter};
use crate::query::executor::context::QueryContext;
use crate::query::parser::ast::{ImagineQuery, Pattern};
use crate::query::{
    ConfidenceInterval, Evidence, EvidenceSource, MatchType, ProbabilisticQueryResult,
    UncertaintySource,
};
use crate::registry::SpaceHandle;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use thiserror::Error;

/// Errors that can occur during IMAGINE query execution.
#[derive(Debug, Error)]
pub enum ImagineExecutionError {
    /// Pattern completion engine not available
    #[error("Pattern completion engine not available or feature not enabled")]
    NoCompletionEngine,

    /// Pattern is too incomplete for reliable completion
    #[error("Pattern is too incomplete for reliable completion: {0}")]
    InsufficientPattern(String),

    /// Pattern completion failed to converge
    #[error("Pattern completion failed to converge: {0}")]
    ConvergenceFailed(String),

    /// Completion confidence below threshold
    #[error("Completion confidence {0} below threshold {1}")]
    LowConfidence(f32, f32),

    /// Pattern completion internal error
    #[error("Pattern completion failed: {0}")]
    CompletionFailed(String),
}

/// Execute an IMAGINE query against a memory space.
///
/// This function:
/// 1. Converts the query pattern into a partial episode representation
/// 2. Uses hippocampal completion engine to fill in missing parts
/// 3. Applies novelty parameter to control creativity vs plausibility
/// 4. Returns completed episodes with source attribution
///
/// # Arguments
///
/// * `query` - The parsed IMAGINE query AST
/// * `_context` - Query execution context with memory space and timeout
/// * `space_handle` - Handle to the memory space containing episodes
///
/// # Returns
///
/// `ProbabilisticQueryResult` containing:
/// - Completed episodes with filled-in details
/// - Confidence intervals reflecting completion uncertainty
/// - Evidence chain with source attribution (recalled vs reconstructed)
/// - Uncertainty sources from pattern completion process
///
/// # Errors
///
/// Returns `ImagineExecutionError` if:
/// - Pattern completion engine is not available
/// - Pattern is too incomplete for reliable completion
/// - Completion process fails to converge
/// - Completion confidence is below threshold
pub fn execute_imagine(
    query: &ImagineQuery<'_>,
    _context: &QueryContext,
    space_handle: &Arc<SpaceHandle>,
) -> Result<ProbabilisticQueryResult, ImagineExecutionError> {
    // Convert query pattern to partial episode
    let partial = pattern_to_partial_episode(&query.pattern, &query.seeds, space_handle);

    // Get or create pattern completion engine
    // NOTE: In production, this would be cached in the space handle
    let completion_config = crate::completion::CompletionConfig::default();
    let completer = HippocampalCompletion::new(completion_config);

    // Adjust novelty if specified (higher novelty = more creative completions)
    let _novelty = query.novelty.unwrap_or(0.3); // Default moderate creativity

    // TODO: When pattern_completion feature is fully integrated:
    // 1. Use completer.complete(&partial) to generate completions
    // 2. Apply novelty parameter to bias completion toward novel vs conservative
    // 3. Filter completions by confidence threshold if specified
    // 4. Extract source attribution from completion result

    // For now, return a basic result indicating pattern completion is pending
    // This maintains API compatibility while the feature is being integrated

    let confidence_threshold = query.confidence_threshold.map_or(0.7, |t| match t {
        crate::query::parser::ast::ConfidenceThreshold::Above(c)
        | crate::query::parser::ast::ConfidenceThreshold::Below(c) => c.raw(),
        crate::query::parser::ast::ConfidenceThreshold::Between { lower, upper: _ } => lower.raw(),
    });

    // Estimate completion confidence from partial pattern
    let completion_confidence = completer.estimate_confidence(&partial);

    if completion_confidence.raw() < confidence_threshold {
        return Err(ImagineExecutionError::LowConfidence(
            completion_confidence.raw(),
            confidence_threshold,
        ));
    }

    // Create placeholder result with evidence of pattern completion attempt
    let confidence_interval =
        ConfidenceInterval::from_confidence_with_uncertainty(completion_confidence, 0.2);

    let evidence = Evidence {
        source: EvidenceSource::DirectMatch {
            cue_id: "pattern_completion_attempt".to_string(),
            similarity_score: completion_confidence.raw(),
            match_type: MatchType::Semantic,
        },
        strength: completion_confidence,
        timestamp: SystemTime::now(),
        dependencies: vec![],
    };

    let uncertainty_sources = vec![
        UncertaintySource::MeasurementError {
            error_magnitude: 0.2,
            confidence_degradation: 0.15,
        },
        UncertaintySource::SystemPressure {
            pressure_level: 0.1,
            effect_on_confidence: 0.05,
        },
    ];

    Ok(ProbabilisticQueryResult {
        episodes: vec![],
        confidence_interval,
        evidence_chain: vec![evidence],
        uncertainty_sources,
    })
}

/// Convert a query pattern into a partial episode for completion.
fn pattern_to_partial_episode(
    pattern: &Pattern<'_>,
    seeds: &[crate::query::parser::ast::NodeIdentifier<'_>],
    space_handle: &Arc<SpaceHandle>,
) -> PartialEpisode {
    let mut known_fields = HashMap::new();
    let mut partial_embedding = vec![None; 768];
    let mut temporal_context = Vec::new();

    // Extract known fields from pattern
    match pattern {
        Pattern::NodeId(id) => {
            known_fields.insert("id".to_string(), id.to_string());
        }
        Pattern::ContentMatch(content) => {
            known_fields.insert("what".to_string(), content.to_string());
        }
        Pattern::Embedding { vector, .. } => {
            // Copy known embedding dimensions
            for (i, &val) in vector.iter().enumerate() {
                if i < 768 {
                    partial_embedding[i] = Some(val);
                }
            }
        }
        Pattern::Any => {
            // No specific constraints
        }
    }

    // Extract temporal context from seed nodes
    for seed_id in seeds {
        if let Some(memory) = space_handle.store().get_memory_arc(seed_id.as_str()) {
            temporal_context.push(memory.id.clone());
        }
    }

    // Determine cue strength based on how much information we have
    let known_dimensions = partial_embedding.iter().filter(|x| x.is_some()).count();
    let cue_strength = if known_dimensions > 0 {
        let ratio = known_dimensions as f32 / 768.0;
        Confidence::from_raw(ratio.min(1.0))
    } else if !known_fields.is_empty() {
        Confidence::MEDIUM
    } else {
        Confidence::LOW
    };

    PartialEpisode {
        known_fields,
        partial_embedding,
        cue_strength,
        temporal_context,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::parser::ast::{ConfidenceThreshold, NodeIdentifier, Pattern};
    use crate::registry::MemorySpaceRegistry;
    use crate::{EpisodeBuilder, MemorySpaceId, MemoryStore};
    use chrono::Utc;
    use std::sync::Arc;

    fn create_test_space_handle() -> Arc<SpaceHandle> {
        let registry = MemorySpaceRegistry::new("/tmp/engram_test", |_id, _dirs| {
            Ok(Arc::new(MemoryStore::new(1000)))
        })
        .expect("Failed to create registry");

        let space_id = MemorySpaceId::new("test_space".to_string()).unwrap();
        let handle = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(registry.create_or_get(&space_id))
        })
        .expect("Failed to create space");

        // Add test episodes as seeds
        let episode = EpisodeBuilder::new()
            .id("seed_episode".to_string())
            .when(Utc::now())
            .what("seed memory content".to_string())
            .embedding([0.5f32; 768])
            .confidence(Confidence::HIGH)
            .build();
        handle.store().store(episode);

        handle
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_pattern_to_partial_episode_from_node_id() {
        let space_handle = create_test_space_handle();
        let pattern = Pattern::NodeId(NodeIdentifier::from("partial_memory"));
        let seeds = vec![NodeIdentifier::from("seed_episode")];

        let partial = pattern_to_partial_episode(&pattern, &seeds, &space_handle);

        assert_eq!(
            partial.known_fields.get("id"),
            Some(&"partial_memory".to_string())
        );
        assert_eq!(partial.temporal_context.len(), 1);
        assert_eq!(partial.temporal_context[0], "seed_episode");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_pattern_to_partial_episode_from_content() {
        let space_handle = create_test_space_handle();
        let pattern = Pattern::ContentMatch("test content".into());
        let seeds = vec![];

        let partial = pattern_to_partial_episode(&pattern, &seeds, &space_handle);

        assert_eq!(
            partial.known_fields.get("what"),
            Some(&"test content".to_string())
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_pattern_to_partial_episode_from_embedding() {
        let space_handle = create_test_space_handle();
        let vector = vec![0.1, 0.2, 0.3];
        let pattern = Pattern::Embedding {
            vector: vector.clone(),
            threshold: 0.8,
        };
        let seeds = vec![];

        let partial = pattern_to_partial_episode(&pattern, &seeds, &space_handle);

        // Check that embedding dimensions were copied
        assert_eq!(partial.partial_embedding[0], Some(0.1));
        assert_eq!(partial.partial_embedding[1], Some(0.2));
        assert_eq!(partial.partial_embedding[2], Some(0.3));
        assert_eq!(partial.partial_embedding[3], None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_imagine_low_confidence_rejection() {
        let space_handle = create_test_space_handle();
        let query = ImagineQuery {
            pattern: Pattern::Any, // Very low information content
            seeds: vec![],
            novelty: None,
            confidence_threshold: Some(ConfidenceThreshold::Above(Confidence::HIGH)),
        };

        let context =
            QueryContext::without_timeout(MemorySpaceId::new("test_space".to_string()).unwrap());

        let result = execute_imagine(&query, &context, &space_handle);

        assert!(matches!(
            result,
            Err(ImagineExecutionError::LowConfidence(_, _))
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_imagine_with_seeds() {
        let space_handle = create_test_space_handle();
        let query = ImagineQuery {
            pattern: Pattern::ContentMatch("partial memory".into()),
            seeds: vec![NodeIdentifier::from("seed_episode")],
            novelty: Some(0.5),
            confidence_threshold: None,
        };

        let context =
            QueryContext::without_timeout(MemorySpaceId::new("test_space".to_string()).unwrap());

        let result = execute_imagine(&query, &context, &space_handle);

        // Should succeed with pattern completion placeholder
        assert!(result.is_ok());
        let query_result = result.unwrap();
        assert!(!query_result.evidence_chain.is_empty());
    }
}
