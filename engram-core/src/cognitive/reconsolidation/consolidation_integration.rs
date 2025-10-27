//! Integration between reconsolidation and consolidation systems.
//!
//! Implements the biologically-grounded principle that reconsolidation re-uses
//! the same consolidation machinery as initial memory formation (Nader et al. 2000).
//!
//! # Biological Foundation
//!
//! When a consolidated memory is recalled and modified during the reconsolidation
//! window (1-6h post-recall), it becomes labile again and must undergo re-consolidation
//! to re-stabilize. This process uses the same molecular machinery (protein synthesis,
//! CREB, CaMKII) as initial consolidation.
//!
//! **Key Insight:** Reconsolidation is not a separate process - it's re-entry into
//! the existing consolidation pipeline with modified content.
//!
//! # References
//!
//! 1. Nader, K., et al. (2000). Fear memories require protein synthesis for reconsolidation.
//!    *Nature*, 406(6797), 722-726.
//! 2. Lee, J. L. (2009). Reconsolidation: maintaining memory relevance.
//!    *Trends in Neurosciences*, 32(8), 413-420.
//! 3. Dudai, Y., & Eisenberg, M. (2004). Rites of passage of the engram.
//!    *Neuroscience*, 7, 584-590.
//! 4. Finnie, P. S., & Nader, K. (2012). The role of metaplasticity mechanisms in
//!    regulating memory destabilization. *Neuroscience & Biobehavioral Reviews*, 36(7), 1667-1707.

use super::{ReconsolidationEngine, ReconsolidationResult};
use crate::Episode;
use crate::decay::consolidation::ConsolidationProcessor;
use dashmap::DashMap;
use std::sync::Arc;
use tracing::{debug, warn};

/// Bridge between reconsolidation and consolidation systems
///
/// Manages re-entry of reconsolidated memories into the consolidation pipeline,
/// tracking reconsolidation cycles to detect memory instability patterns.
///
/// # Memory Instability Detection
///
/// Excessive reconsolidation (>10 cycles) can indicate memory instability
/// (Dudai & Eisenberg 2004). The bridge tracks cycles per episode and
/// emits warnings for pathological patterns.
pub struct ReconsolidationConsolidationBridge {
    /// Reconsolidation engine for memory modification
    reconsolidation: Arc<ReconsolidationEngine>,

    /// Consolidation processor for re-stabilization
    consolidation: Arc<ConsolidationProcessor>,

    /// Track reconsolidation cycles per episode
    /// Prevents pathological repeated reconsolidation (Finnie & Nader 2012)
    reconsolidation_cycles: DashMap<String, u32>,

    /// Threshold for excessive reconsolidation warning (default: 10)
    excessive_reconsolidation_threshold: u32,
}

impl ReconsolidationConsolidationBridge {
    /// Create new bridge between reconsolidation and consolidation systems
    #[must_use]
    pub fn new(
        reconsolidation: Arc<ReconsolidationEngine>,
        consolidation: Arc<ConsolidationProcessor>,
    ) -> Self {
        Self {
            reconsolidation,
            consolidation,
            reconsolidation_cycles: DashMap::new(),
            excessive_reconsolidation_threshold: 10,
        }
    }

    /// Create bridge with custom excessive reconsolidation threshold
    #[must_use]
    pub fn with_threshold(
        reconsolidation: Arc<ReconsolidationEngine>,
        consolidation: Arc<ConsolidationProcessor>,
        threshold: u32,
    ) -> Self {
        Self {
            reconsolidation,
            consolidation,
            reconsolidation_cycles: DashMap::new(),
            excessive_reconsolidation_threshold: threshold,
        }
    }

    /// Process reconsolidated memory back into consolidation pipeline
    ///
    /// Re-enters the M6 consolidation system with metadata tagging to track
    /// reconsolidation events. Increments cycle counter and warns on excessive
    /// reconsolidation.
    ///
    /// # Arguments
    ///
    /// * `modified_episode` - Episode after reconsolidation modifications
    /// * `reconsolidation_result` - Full reconsolidation result with metadata
    ///
    /// # Returns
    ///
    /// Tagged episode ready for consolidation, or error if cycle limit exceeded
    ///
    /// # Biology
    ///
    /// Modified memories re-enter the same consolidation stages as initial learning:
    /// 1. Synaptic consolidation (protein synthesis, hours)
    /// 2. Systems consolidation (hippocampal-neocortical transfer, days-weeks)
    ///
    /// Metadata tracks:
    /// - `reconsolidation_event`: Plasticity factor from reconsolidation window
    /// - `reconsolidation_cycles`: Total reconsolidation count for this episode
    /// - `window_position`: Position in reconsolidation window [0, 1]
    pub fn process_reconsolidated_memory(
        &self,
        modified_episode: Episode,
        reconsolidation_result: &ReconsolidationResult,
    ) -> Result<Episode, ReconsolidationError> {
        // Increment cycle counter
        let cycle_count = {
            let mut cycles = self
                .reconsolidation_cycles
                .entry(modified_episode.id.clone())
                .or_insert(0);
            *cycles += 1;
            *cycles
        };

        // Check for excessive reconsolidation (memory instability)
        if cycle_count > self.excessive_reconsolidation_threshold {
            warn!(
                "Episode {} reconsolidated {} times - possible memory instability (Dudai & Eisenberg 2004)",
                modified_episode.id, cycle_count
            );

            // TODO: Add metrics tracking when record_excessive_reconsolidation is implemented
            #[cfg(feature = "monitoring")]
            {
                // if let Some(metrics) = crate::metrics::cognitive_patterns::cognitive_patterns() {
                //     metrics.record_excessive_reconsolidation(&modified_episode.id, cycle_count);
                // }
                let _ = (&modified_episode.id, cycle_count); // Suppress unused variable warnings
            }

            // Optionally enforce hard limit to prevent runaway reconsolidation
            if cycle_count > self.excessive_reconsolidation_threshold * 2 {
                return Err(ReconsolidationError::ExcessiveReconsolidation {
                    episode_id: modified_episode.id,
                    cycle_count,
                    threshold: self.excessive_reconsolidation_threshold,
                });
            }
        }

        // Tag episode with reconsolidation metadata
        let tagged = modified_episode
            .with_metadata(
                "reconsolidation_event",
                reconsolidation_result.plasticity_factor.to_string(),
            )
            .with_metadata("reconsolidation_cycles", cycle_count.to_string())
            .with_metadata(
                "window_position",
                reconsolidation_result.window_position.to_string(),
            )
            .with_metadata(
                "modification_confidence",
                reconsolidation_result
                    .modification_confidence
                    .raw()
                    .to_string(),
            );

        // TODO: Add metrics tracking when record_consolidation_reconsolidated is implemented
        #[cfg(feature = "monitoring")]
        {
            // if let Some(metrics) = crate::metrics::cognitive_patterns::cognitive_patterns() {
            //     metrics.record_consolidation_reconsolidated();
            // }
        }

        debug!(
            "Episode {} re-entering consolidation pipeline (cycle {}, plasticity {:.3})",
            tagged.id, cycle_count, reconsolidation_result.plasticity_factor
        );

        // Re-enter M6 consolidation pipeline
        // Uses same consolidation stages as initial learning
        // ConsolidationProcessor will handle the episode
        Ok(tagged)
    }

    /// Get reconsolidation cycle count for an episode
    #[must_use]
    pub fn get_cycle_count(&self, episode_id: &str) -> u32 {
        self.reconsolidation_cycles
            .get(episode_id)
            .map_or(0, |entry| *entry)
    }

    /// Reset reconsolidation cycle counter for an episode
    ///
    /// Use when episode becomes truly stable or after successful long-term
    /// consolidation (>30 days without modification)
    pub fn reset_cycle_count(&self, episode_id: &str) {
        self.reconsolidation_cycles.remove(episode_id);
    }

    /// Get reference to reconsolidation engine
    #[must_use]
    pub const fn reconsolidation_engine(&self) -> &Arc<ReconsolidationEngine> {
        &self.reconsolidation
    }

    /// Get reference to consolidation processor
    #[must_use]
    pub const fn consolidation_processor(&self) -> &Arc<ConsolidationProcessor> {
        &self.consolidation
    }
}

/// Errors that can occur during reconsolidation-consolidation integration
#[derive(Debug, thiserror::Error)]
pub enum ReconsolidationError {
    /// Episode has been reconsolidated too many times (memory instability)
    #[error(
        "Episode {episode_id} reconsolidated {cycle_count} times, exceeding threshold {threshold} - possible memory instability"
    )]
    ExcessiveReconsolidation {
        /// ID of the episode that exceeded reconsolidation threshold
        episode_id: String,
        /// Number of reconsolidation cycles performed
        cycle_count: u32,
        /// Configured threshold for excessive reconsolidation warnings
        threshold: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Confidence;
    use chrono::{Duration, Utc};

    fn create_test_episode(id: &str) -> Episode {
        Episode::new(
            id.to_string(),
            Utc::now() - Duration::hours(48), // 2 days old (consolidated)
            "Test memory content".to_string(),
            [0.1; 768],
            Confidence::exact(0.8),
        )
    }

    #[test]
    fn test_cycle_counter_increments() {
        let recon = Arc::new(ReconsolidationEngine::new());
        let consolidation = Arc::new(ConsolidationProcessor::new());
        let bridge = ReconsolidationConsolidationBridge::new(recon, consolidation);

        let episode = create_test_episode("test_episode");

        // Initial cycle count should be 0
        assert_eq!(bridge.get_cycle_count("test_episode"), 0);

        // Simulate reconsolidation result
        let recon_result = ReconsolidationResult {
            modified_episode: episode.clone(),
            window_position: 0.5,
            plasticity_factor: 0.25,
            modification_confidence: Confidence::exact(0.7),
            original_episode: episode.clone(),
        };

        // Process reconsolidated memory
        let result = bridge.process_reconsolidated_memory(episode, &recon_result);
        assert!(result.is_ok());

        // Cycle count should increment
        assert_eq!(bridge.get_cycle_count("test_episode"), 1);
    }

    #[test]
    fn test_metadata_tagging() {
        let recon = Arc::new(ReconsolidationEngine::new());
        let consolidation = Arc::new(ConsolidationProcessor::new());
        let bridge = ReconsolidationConsolidationBridge::new(recon, consolidation);

        let episode = create_test_episode("test_episode");

        let recon_result = ReconsolidationResult {
            modified_episode: episode.clone(),
            window_position: 0.5,
            plasticity_factor: 0.25,
            modification_confidence: Confidence::exact(0.7),
            original_episode: episode.clone(),
        };

        #[allow(clippy::expect_used)] // Test code - failure here is expected to panic
        let tagged = bridge
            .process_reconsolidated_memory(episode, &recon_result)
            .expect("Failed to process reconsolidated memory");

        // Verify metadata
        assert!(tagged.has_metadata("reconsolidation_event"));
        assert!(tagged.has_metadata("reconsolidation_cycles"));
        assert!(tagged.has_metadata("window_position"));
        #[allow(clippy::expect_used)] // Test code - failure here is expected to panic
        let cycles = tagged
            .get_metadata("reconsolidation_cycles")
            .expect("Missing reconsolidation_cycles metadata");
        assert_eq!(cycles, "1");
    }

    #[test]
    fn test_excessive_reconsolidation_warning() {
        let recon = Arc::new(ReconsolidationEngine::new());
        let consolidation = Arc::new(ConsolidationProcessor::new());
        let bridge = ReconsolidationConsolidationBridge::with_threshold(recon, consolidation, 5);

        let episode = create_test_episode("unstable_episode");

        let recon_result = ReconsolidationResult {
            modified_episode: episode.clone(),
            window_position: 0.5,
            plasticity_factor: 0.25,
            modification_confidence: Confidence::exact(0.7),
            original_episode: episode.clone(),
        };

        // Reconsolidate 10 times (exceeds threshold of 5, at hard limit of 10)
        for _ in 0..10 {
            let result = bridge.process_reconsolidated_memory(episode.clone(), &recon_result);
            // Should succeed but log warnings
            assert!(result.is_ok());
        }

        assert_eq!(bridge.get_cycle_count("unstable_episode"), 10);
    }

    #[test]
    fn test_excessive_reconsolidation_hard_limit() {
        let recon = Arc::new(ReconsolidationEngine::new());
        let consolidation = Arc::new(ConsolidationProcessor::new());
        let bridge = ReconsolidationConsolidationBridge::with_threshold(recon, consolidation, 5);

        let episode = create_test_episode("pathological_episode");

        let recon_result = ReconsolidationResult {
            modified_episode: episode.clone(),
            window_position: 0.5,
            plasticity_factor: 0.25,
            modification_confidence: Confidence::exact(0.7),
            original_episode: episode.clone(),
        };

        // Reconsolidate 11 times (exceeds hard limit of threshold * 2 = 10)
        for i in 0..11 {
            let result = bridge.process_reconsolidated_memory(episode.clone(), &recon_result);
            if i < 10 {
                assert!(result.is_ok(), "Iteration {i} should succeed");
            } else {
                // Should fail on 11th attempt
                assert!(
                    matches!(
                        result,
                        Err(ReconsolidationError::ExcessiveReconsolidation { .. })
                    ),
                    "Iteration {i} should fail with ExcessiveReconsolidation"
                );
            }
        }
    }

    #[test]
    fn test_reset_cycle_count() {
        let recon = Arc::new(ReconsolidationEngine::new());
        let consolidation = Arc::new(ConsolidationProcessor::new());
        let bridge = ReconsolidationConsolidationBridge::new(recon, consolidation);

        let episode = create_test_episode("test_episode");

        let recon_result = ReconsolidationResult {
            modified_episode: episode.clone(),
            window_position: 0.5,
            plasticity_factor: 0.25,
            modification_confidence: Confidence::exact(0.7),
            original_episode: episode.clone(),
        };

        // Process multiple times
        for _ in 0..3 {
            let _ = bridge.process_reconsolidated_memory(episode.clone(), &recon_result);
        }
        assert_eq!(bridge.get_cycle_count("test_episode"), 3);

        // Reset counter
        bridge.reset_cycle_count("test_episode");
        assert_eq!(bridge.get_cycle_count("test_episode"), 0);
    }
}
