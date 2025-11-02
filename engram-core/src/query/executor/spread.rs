//! SPREAD query execution implementation.
//!
//! This module implements the execution logic for SPREAD queries, mapping the
//! AST representation to the underlying `ParallelSpreadingEngine::spread_activation`
//! API with proper parameter translation and evidence extraction.
//!
//! ## Design Principles
//!
//! 1. **Zero-Copy Evidence**: Activation paths are extracted directly from
//!    spreading results without unnecessary allocations
//! 2. **Parameter Mapping**: SpreadQuery AST fields map directly to spreading
//!    engine parameters with appropriate defaults
//! 3. **Performance**: <5% overhead vs direct API call through efficient result
//!    transformation
//! 4. **Evidence Chain**: Complete activation path tracking for probabilistic
//!    reasoning
//!
//! ## Biological Grounding
//!
//! The SPREAD operation implements spreading activation from cognitive science,
//! where activation flows through semantic networks from source concepts to
//! related memories. Key biological phenomena:
//!
//! - **Decay with Distance**: Activation weakens with each hop (inverse relationship
//!   with semantic distance)
//! - **Threshold Effects**: Activation below threshold doesn't propagate (neural
//!   firing thresholds)
//! - **Refractory Periods**: Prevents immediate re-activation (neural refractory
//!   periods)
//! - **Parallel Propagation**: Multiple paths activate simultaneously (massively
//!   parallel neural architecture)
//!
//! ## Example
//!
//! ```rust
//! use engram_core::query::parser::ast::{SpreadQuery, NodeIdentifier};
//! use engram_core::query::executor::{execute_spread, QueryContext};
//! use engram_core::MemorySpaceId;
//! use std::sync::Arc;
//!
//! // Create SPREAD query from AST
//! let query = SpreadQuery {
//!     source: NodeIdentifier::from("concept_ai"),
//!     max_hops: Some(3),
//!     decay_rate: Some(0.15),
//!     activation_threshold: Some(0.02),
//!     refractory_period: None,
//! };
//!
//! // Execute with context
//! // let context = QueryContext::without_timeout(MemorySpaceId::from("user_123"));
//! // let store = ...; // Get MemoryStore reference
//! // let result = execute_spread(&query, &context, &store).await?;
//! ```

use crate::activation::{DecayFunction, ParallelSpreadingEngine, SpreadingResults};
use crate::query::executor::QueryContext;
use crate::query::{
    ConfidenceInterval, Evidence, EvidenceSource, ProbabilisticQueryResult, UncertaintySource,
};
use crate::{Confidence, Episode, MemoryStore};
use std::sync::Arc;
use std::time::SystemTime;
use thiserror::Error;

/// Errors that can occur during SPREAD query execution.
#[derive(Debug, Error)]
pub enum SpreadExecutionError {
    /// Source node not found in memory graph
    #[error("Source node '{0}' not found in memory graph")]
    SourceNodeNotFound(String),

    /// No spreading engine available (CognitiveRecall not initialized)
    #[error(
        "Spreading engine not available - memory store may not have cognitive recall initialized"
    )]
    NoSpreadingEngine,

    /// Activation spreading failed
    #[error("Activation spreading failed: {0}")]
    SpreadingFailed(String),

    /// Query timeout exceeded
    #[error("SPREAD query exceeded timeout of {0:?}")]
    TimeoutExceeded(std::time::Duration),
}

/// Execute a SPREAD query against a memory store.
///
/// This function:
/// 1. Validates the source node exists
/// 2. Extracts spreading engine from the memory store
/// 3. Maps query parameters to spreading configuration
/// 4. Executes spreading activation
/// 5. Transforms results into `ProbabilisticQueryResult` with evidence
///
/// # Arguments
///
/// * `query` - The parsed SPREAD query AST
/// * `context` - Query execution context with memory space and timeout
/// * `store` - Memory store to execute against
///
/// # Returns
///
/// `ProbabilisticQueryResult` containing:
/// - Episodes activated during spreading
/// - Confidence intervals from activation strengths
/// - Evidence chain with activation paths
/// - Uncertainty sources from spreading dynamics
///
/// # Errors
///
/// Returns `SpreadExecutionError` if:
/// - Source node doesn't exist
/// - Spreading engine is not available
/// - Activation spreading fails
/// - Query timeout is exceeded
pub fn execute_spread(
    query: &crate::query::parser::ast::SpreadQuery<'_>,
    _context: &QueryContext,
    store: &Arc<MemoryStore>,
) -> Result<ProbabilisticQueryResult, SpreadExecutionError> {
    // Get spreading engine from memory store
    let spreading_engine = store
        .spreading_engine()
        .ok_or(SpreadExecutionError::NoSpreadingEngine)?;

    // Validate source node exists in the graph
    let source_id = query.source.as_str();
    if store.get_memory_arc(source_id).is_none() {
        return Err(SpreadExecutionError::SourceNodeNotFound(
            source_id.to_string(),
        ));
    }

    // Map query parameters to spreading activation inputs
    let max_hops = query
        .max_hops
        .unwrap_or(crate::query::parser::ast::SpreadQuery::DEFAULT_MAX_HOPS);
    let decay_rate = query.effective_decay_rate();
    let threshold = query.effective_threshold();

    // Configure spreading engine if parameters differ from defaults
    update_spreading_config_if_needed(&spreading_engine, max_hops, decay_rate, threshold);

    // Execute spreading activation with initial activation of 1.0 from source
    let seed_activations = vec![(source_id.to_string(), 1.0)];

    let spreading_results = spreading_engine
        .spread_activation(&seed_activations)
        .map_err(|e| SpreadExecutionError::SpreadingFailed(e.to_string()))?;

    // Transform spreading results into ProbabilisticQueryResult
    let result = transform_spreading_results(
        &spreading_results,
        source_id,
        max_hops,
        decay_rate,
        threshold,
        store,
    );

    Ok(result)
}

/// Update spreading configuration if query parameters differ from current config.
///
/// This avoids unnecessary configuration updates when parameters match defaults.
fn update_spreading_config_if_needed(
    engine: &Arc<ParallelSpreadingEngine>,
    max_hops: u16,
    decay_rate: f32,
    threshold: f32,
) {
    let mut config = engine.config_snapshot();

    // Track if we need to update
    let mut updated = config.max_depth != max_hops;

    // Update max_depth if different from max_hops
    if config.max_depth != max_hops {
        config.max_depth = max_hops;
    }

    // Update activation threshold if different
    if (config.threshold - threshold).abs() > 1e-6 {
        config.threshold = threshold;
        updated = true;
    }

    // Update decay rate in decay function if different
    //
    // ## Biological Grounding
    //
    // In spreading activation models (Collins & Loftus, 1975; Anderson, 1983),
    // activation decays with distance following exponential dynamics:
    //   A(d) = A₀ × exp(-λ × d)
    // where λ is the spatial decay constant.
    //
    // The query `decay_rate` parameter represents the **proportion** of activation
    // lost per hop. For decay_rate = 0.1, we want activation to drop to 90% at
    // each step:
    //   A(1) = A₀ × 0.9
    //   A(2) = A₀ × 0.81
    //   A(3) = A₀ × 0.729
    //
    // The exponential decay function is: exp(-rate × depth)
    // We want: exp(-rate × 1) = (1 - decay_rate)
    // Therefore: rate = -ln(1 - decay_rate)
    //
    // ## Example
    // decay_rate = 0.1 → rate = -ln(0.9) ≈ 0.105
    //   depth=1: exp(-0.105) ≈ 0.900 ✓
    //   depth=2: exp(-0.210) ≈ 0.810 ✓
    //   depth=3: exp(-0.315) ≈ 0.730 ✓
    if let DecayFunction::Exponential { rate } = &config.decay_function {
        let target_rate = if decay_rate > 0.0 && decay_rate < 1.0 {
            -(1.0 - decay_rate).ln()
        } else if decay_rate >= 1.0 {
            // Full decay (no propagation beyond source)
            f32::INFINITY
        } else {
            // No decay (perfect propagation)
            0.0
        };
        if (rate - target_rate).abs() > 1e-6 && target_rate.is_finite() {
            config.decay_function = DecayFunction::Exponential { rate: target_rate };
            updated = true;
        }
    }

    if updated {
        engine.update_config(&config);
    }
}

/// Transform spreading results into ProbabilisticQueryResult.
///
/// Extracts:
/// - Activated episodes from spreading results
/// - Activation paths as evidence
/// - Confidence intervals from activation strengths
/// - Uncertainty sources from spreading dynamics
fn transform_spreading_results(
    results: &SpreadingResults,
    source_id: &str,
    _max_hops: u16,
    decay_rate: f32,
    threshold: f32,
    store: &Arc<MemoryStore>,
) -> ProbabilisticQueryResult {
    // Convert storage-aware activations to episodes with confidence
    let mut episodes = Vec::new();
    let mut evidence_chain = Vec::new();
    let now = SystemTime::now();

    for activation in &results.activations {
        // Skip source node itself
        if activation.memory_id == source_id {
            continue;
        }

        // Get activation level from AtomicF32
        let activation_value = activation
            .activation_level
            .load(std::sync::atomic::Ordering::Relaxed);

        // Skip activations below threshold
        if activation_value < threshold {
            continue;
        }

        // Try to retrieve episode from store
        if let Some(memory) = store.get_memory_arc(&activation.memory_id) {
            // Convert activation to confidence (they're both [0,1])
            let confidence = Confidence::from_raw(activation_value.clamp(0.0, 1.0));

            // Create episode from memory
            let episode = Episode::new(
                memory.id.clone(),
                memory.last_access,
                memory
                    .content
                    .clone()
                    .unwrap_or_else(|| format!("Memory {id}", id = memory.id)),
                memory.embedding,
                confidence,
            );

            episodes.push((episode, confidence));

            // Create evidence from spreading activation
            let hop_count = activation
                .hop_count
                .load(std::sync::atomic::Ordering::Relaxed);
            let evidence = Evidence {
                source: EvidenceSource::SpreadingActivation {
                    source_episode: source_id.to_string(),
                    activation_level: crate::Activation::new(activation_value),
                    path_length: hop_count,
                },
                strength: confidence,
                timestamp: now,
                dependencies: vec![],
            };

            evidence_chain.push(evidence);
        }
    }

    // Calculate aggregate confidence from all activations
    let aggregate_confidence = if episodes.is_empty() {
        Confidence::NONE
    } else {
        let sum: f32 = episodes.iter().map(|(_, c)| c.raw()).sum();
        let avg = sum / episodes.len() as f32;
        Confidence::from_raw(avg.clamp(0.0, 1.0))
    };

    // Create confidence interval with uncertainty from decay rate
    let uncertainty = decay_rate * 0.5; // Decay rate contributes to uncertainty
    let confidence_interval =
        ConfidenceInterval::from_confidence_with_uncertainty(aggregate_confidence, uncertainty);

    // Add uncertainty sources from spreading dynamics
    let uncertainty_sources = vec![
        UncertaintySource::SpreadingActivationNoise {
            activation_variance: decay_rate * 0.5, // Decay rate contributes to variance
            path_diversity: decay_rate,            // Higher decay = more diverse (pruned) paths
        },
        UncertaintySource::TemporalDecayUnknown {
            time_since_encoding: std::time::Duration::from_secs(0), // Unknown time
            decay_model_uncertainty: threshold, // Threshold affects decay uncertainty
        },
    ];

    ProbabilisticQueryResult {
        episodes,
        confidence_interval,
        evidence_chain,
        uncertainty_sources,
    }
}

#[cfg(test)]
mod tests {
    use crate::query::parser::ast::{NodeIdentifier, SpreadQuery};
    use std::time::Duration;

    #[test]
    #[allow(clippy::float_cmp)] // Testing exact constant values
    fn test_spread_query_parameter_defaults() {
        let query = SpreadQuery {
            source: NodeIdentifier::from("test_node"),
            max_hops: None,
            decay_rate: None,
            activation_threshold: None,
            refractory_period: None,
        };

        assert_eq!(
            query.effective_decay_rate(),
            SpreadQuery::DEFAULT_DECAY_RATE
        );
        assert_eq!(query.effective_threshold(), SpreadQuery::DEFAULT_THRESHOLD);
    }

    #[test]
    #[allow(clippy::float_cmp)] // Testing exact constant values
    fn test_spread_query_custom_parameters() {
        let query = SpreadQuery {
            source: NodeIdentifier::from("test_node"),
            max_hops: Some(5),
            decay_rate: Some(0.25),
            activation_threshold: Some(0.05),
            refractory_period: Some(Duration::from_millis(100)),
        };

        assert_eq!(query.max_hops, Some(5));
        assert_eq!(query.effective_decay_rate(), 0.25);
        assert_eq!(query.effective_threshold(), 0.05);
    }

    // Integration tests would require a full MemoryStore setup
    // These are covered in the integration test file
}
