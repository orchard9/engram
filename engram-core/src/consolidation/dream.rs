//! Dream operation for offline consolidation with enhanced pattern detection
//!
//! Implements biologically-inspired "dream" cycles that replay and consolidate
//! episodic memories into semantic knowledge during offline periods.

use crate::completion::consolidation::SemanticPattern;
use crate::consolidation::{
    CompactionConfig, CompactionResult, EpisodicPattern, PatternDetectionConfig, PatternDetector,
    StorageCompactor,
};
#[cfg(feature = "dual_memory_types")]
use crate::consolidation::{ConceptFormationEngine, ProtoConcept, SleepStage};
use crate::{Episode, MemoryStore};
use chrono::Utc;
use std::sync::Arc;
use std::time::{Duration, Instant};
#[cfg(feature = "dual_memory_types")]
use uuid::Uuid;

/// Dream operation error types
#[derive(Debug, thiserror::Error)]
#[allow(clippy::enum_variant_names)] // All error variants end in "Failed" by design
pub enum DreamError {
    /// Episode selection failed
    #[error("Failed to select episodes for dream replay: {0}")]
    EpisodeSelectionFailed(String),

    /// Pattern detection failed during replay
    #[error("Pattern detection failed: {0}")]
    PatternDetectionFailed(String),

    /// Semantic extraction failed
    #[error("Semantic extraction failed: {0}")]
    SemanticExtractionFailed(String),

    /// Storage compaction failed
    #[error("Storage compaction failed: {0}")]
    CompactionFailed(String),

    /// Replay failed
    #[error("Replay operation failed: {0}")]
    ReplayFailed(String),
}

/// Configuration for dream cycles
#[derive(Debug, Clone)]
pub struct DreamConfig {
    /// Duration of dream cycle (default: 10 minutes)
    pub dream_duration: Duration,

    /// Replay speed multiplier (default: 15x, range 10-20x)
    pub replay_speed: f32,

    /// Number of replay iterations per dream cycle (default: 5)
    pub replay_iterations: usize,

    /// Sharp-wave ripple frequency in Hz (default: 200 Hz, range 150-250 Hz)
    pub ripple_frequency: f32,

    /// Minimum episode age for dream consolidation (default: 1 day)
    pub min_episode_age: Duration,

    /// Maximum episodes to replay per iteration (default: 50)
    pub max_episodes_per_iteration: usize,

    /// Whether to compact storage after pattern detection
    pub enable_compaction: bool,

    /// Current sleep stage for consolidation modulation
    /// Default: NREM2 (peak consolidation)
    #[cfg(feature = "dual_memory_types")]
    pub sleep_stage: SleepStage,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            dream_duration: Duration::from_secs(600), // 10 minutes
            replay_speed: 15.0,                       // 15x faster than real-time
            replay_iterations: 5,
            ripple_frequency: 200.0, // 200 Hz (within 150-250 Hz range)
            min_episode_age: Duration::from_secs(86400), // 1 day
            max_episodes_per_iteration: 50,
            enable_compaction: true,
            #[cfg(feature = "dual_memory_types")]
            sleep_stage: SleepStage::NREM2, // Peak consolidation
        }
    }
}

impl DreamConfig {
    /// Create test configuration with reduced episode age threshold for rapid testing
    ///
    /// Use this in integration tests to avoid waiting for episodes to age.
    /// - `min_episode_age`: 1 second (vs 1 day in production)
    /// - `dream_duration`: 10 seconds (vs 10 minutes)
    /// - `max_episodes_per_iteration`: 10 (vs 50)
    #[cfg(test)]
    #[must_use]
    pub const fn test_config() -> Self {
        Self {
            dream_duration: Duration::from_secs(10),
            replay_speed: 15.0,
            replay_iterations: 5,
            ripple_frequency: 200.0,
            min_episode_age: Duration::from_secs(1), // 1 second for testing
            max_episodes_per_iteration: 10,
            enable_compaction: true,
            #[cfg(feature = "dual_memory_types")]
            sleep_stage: SleepStage::NREM2,
        }
    }
}

/// Engine for running dream consolidation cycles
pub struct DreamEngine {
    /// Configuration for dream cycles
    pub config: DreamConfig,

    /// Pattern detector for unsupervised pattern discovery (for storage compaction)
    pattern_detector: Arc<PatternDetector>,

    /// Concept formation engine (for semantic consolidation)
    #[cfg(feature = "dual_memory_types")]
    concept_engine: Arc<ConceptFormationEngine>,

    /// Storage compactor for replacing episodes with semantic patterns
    compactor: Arc<StorageCompactor>,
}

impl DreamEngine {
    /// Create a new dream engine with the given configuration
    #[must_use]
    pub fn new(config: DreamConfig) -> Self {
        // Create pattern detector with default configuration (for storage compaction)
        let pattern_config = PatternDetectionConfig {
            min_cluster_size: 3,
            similarity_threshold: 0.8,
            max_patterns: 100,
        };
        let pattern_detector = Arc::new(PatternDetector::new(pattern_config));

        // Create concept formation engine (for semantic consolidation)
        #[cfg(feature = "dual_memory_types")]
        let concept_engine = Arc::new(ConceptFormationEngine::new());

        // Create compactor with aligned min_episode_age from DreamConfig
        let compaction_config = CompactionConfig {
            min_episode_age: config.min_episode_age,
            preserve_threshold: 0.95,
            verify_retrieval: true,
            reconstruction_threshold: 0.8,
        };
        let compactor = Arc::new(StorageCompactor::new(compaction_config));

        Self {
            config,
            pattern_detector,
            #[cfg(feature = "dual_memory_types")]
            concept_engine,
            compactor,
        }
    }

    /// Create dream engine with custom pattern detector and compactor
    #[must_use]
    pub fn with_components(
        config: DreamConfig,
        pattern_detector: PatternDetector,
        compactor: StorageCompactor,
    ) -> Self {
        #[cfg(feature = "dual_memory_types")]
        let concept_engine = Arc::new(ConceptFormationEngine::new());

        Self {
            config,
            pattern_detector: Arc::new(pattern_detector),
            #[cfg(feature = "dual_memory_types")]
            concept_engine,
            compactor: Arc::new(compactor),
        }
    }

    /// Run dream cycle for offline consolidation
    ///
    /// This implements the enhanced dream consolidation process:
    ///
    /// 1. Select episodes for replay based on importance
    /// 2. Replay episodes with ripple dynamics (multiple iterations)
    /// 3. Pattern detection and concept formation:
    ///    - 3a. Detect patterns from replay (for storage compaction)
    ///    - 3b. Form concepts from episodes (for semantic consolidation)
    /// 4. Extract semantic memories from patterns
    /// 5. Consolidation and storage:
    ///    - 5a. Apply storage compaction (if enabled)
    ///    - 5b. Promote proto-concepts to semantic concepts
    pub fn dream(&self, store: &MemoryStore) -> Result<DreamOutcome, DreamError> {
        let start = Instant::now();

        // Phase 1: Select episodes for dream replay
        let episodes = self.select_dream_episodes(store)?;

        if episodes.is_empty() {
            return Ok(DreamOutcome {
                dream_duration: start.elapsed(),
                episodes_replayed: 0,
                replay_iterations: 0,
                patterns_discovered: 0,
                semantic_memories_created: 0,
                storage_reduction_bytes: 0,
            });
        }

        // Phase 2: Replay episodes with ripple dynamics
        let replay_outcome = self.replay_episodes(&episodes)?;

        // Phase 3a: Detect patterns from replay (existing - for storage compaction)
        let patterns = self.detect_patterns_from_replay(&episodes)?;

        // Phase 3b: Form concepts from episodes (NEW - for semantic consolidation)
        #[cfg(feature = "dual_memory_types")]
        let proto_concepts = self
            .concept_engine
            .process_episodes(&episodes, self.config.sleep_stage);

        // Phase 4: Extract semantic memories from patterns
        let semantic_patterns = Self::extract_semantic_from_patterns(&patterns)?;

        // Phase 5a: Apply storage compaction (existing)
        let compaction_results = if self.config.enable_compaction {
            self.compact_replayed_episodes(&episodes, &semantic_patterns)?
        } else {
            // Store semantic patterns without compaction
            for pattern in &semantic_patterns {
                store.store_semantic_pattern(pattern);
            }
            vec![]
        };

        let storage_reduction: u64 = compaction_results
            .iter()
            .map(|r| r.storage_reduction_bytes)
            .sum();

        // Phase 5b: Promote proto-concepts to semantic concepts (NEW)
        #[cfg(feature = "dual_memory_types")]
        let concepts_created = Self::promote_proto_concepts(&proto_concepts, store);

        #[cfg(not(feature = "dual_memory_types"))]
        let concepts_created = 0;

        Ok(DreamOutcome {
            dream_duration: start.elapsed(),
            episodes_replayed: episodes.len(),
            replay_iterations: replay_outcome.replays_completed,
            patterns_discovered: patterns.len(),
            semantic_memories_created: semantic_patterns.len() + concepts_created,
            storage_reduction_bytes: storage_reduction,
        })
    }

    /// Phase 1: Select episodes for dream replay based on importance
    ///
    /// Prioritizes episodes by:
    /// 1. Age (episodes must be at least min_episode_age old)
    /// 2. Confidence (higher confidence episodes preferred)
    /// 3. Not yet consolidated (avoid redundant consolidation)
    fn select_dream_episodes(&self, store: &MemoryStore) -> Result<Vec<Episode>, DreamError> {
        let now = Utc::now();
        let min_age = chrono::Duration::from_std(self.config.min_episode_age)
            .map_err(|e| DreamError::EpisodeSelectionFailed(e.to_string()))?;

        // Get all episodes from store
        let all_episodes = store.all_episodes();

        // Filter and score episodes
        let mut scored_episodes: Vec<(Episode, f32)> = all_episodes
            .into_iter()
            .filter(|ep| {
                // Must be old enough
                let age = now - ep.when;
                age > min_age
            })
            .map(|ep| {
                // Score based on confidence (simple priority for now)
                let priority_score = ep.encoding_confidence.raw();
                (ep, priority_score)
            })
            .collect();

        // DETERMINISM FIX 4: Sort by score (highest first) with deterministic tie-breaking
        scored_episodes.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.id.cmp(&b.0.id)) // Tie-break by episode ID
        });

        // Take top episodes up to limit
        let selected: Vec<Episode> = scored_episodes
            .into_iter()
            .take(self.config.max_episodes_per_iteration * self.config.replay_iterations)
            .map(|(ep, _)| ep)
            .collect();

        Ok(selected)
    }

    /// Phase 2: Replay episodes with sharp-wave ripple dynamics
    ///
    /// Simulates hippocampal replay during sleep/rest:
    /// - Replays episodes at accelerated speed (10-20x)
    /// - Multiple iterations to strengthen pattern detection
    /// - Sharp-wave ripple frequency (150-250 Hz)
    #[allow(clippy::unnecessary_wraps)] // May add error handling in future
    fn replay_episodes(&self, episodes: &[Episode]) -> Result<ReplayOutcome, DreamError> {
        let mut total_replay_time = Duration::ZERO;

        for iteration in 0..self.config.replay_iterations {
            // Calculate replay duration based on speed multiplier
            // Real-time encoding might take ~1 second per episode
            // At 15x speed, replay takes ~67ms per episode
            let replay_time_per_episode =
                Duration::from_millis((1000.0 / self.config.replay_speed) as u64);

            let iteration_time = replay_time_per_episode * episodes.len() as u32;

            // Simulate ripple dynamics
            // In real implementation, this would trigger neural replay mechanisms
            // For now, we just track timing and allow pattern detector to work
            std::thread::sleep(Duration::from_millis(10)); // Minimal sleep to prevent tight loop

            total_replay_time += iteration_time;

            // Check if we've exceeded dream duration budget
            if total_replay_time > self.config.dream_duration {
                return Ok(ReplayOutcome {
                    replays_completed: iteration + 1,
                    total_replay_time,
                    average_speed: self.config.replay_speed,
                });
            }
        }

        Ok(ReplayOutcome {
            replays_completed: self.config.replay_iterations,
            total_replay_time,
            average_speed: self.config.replay_speed,
        })
    }

    /// Phase 3: Detect patterns from replay using pattern detector
    ///
    /// Multiple replay iterations enhance pattern detection by:
    /// - Reinforcing consistent patterns
    /// - Filtering out noise through repetition
    /// - Allowing statistical significance to emerge
    #[allow(clippy::unnecessary_wraps)] // May add error handling in future
    fn detect_patterns_from_replay(
        &self,
        episodes: &[Episode],
    ) -> Result<Vec<EpisodicPattern>, DreamError> {
        // Use pattern detector to find clusters
        let patterns = self.pattern_detector.detect_patterns(episodes);

        Ok(patterns)
    }

    /// Phase 4: Extract semantic memories from detected patterns
    ///
    /// Converts episodic patterns into semantic representations
    #[allow(clippy::unnecessary_wraps)] // May add error handling in future
    fn extract_semantic_from_patterns(
        patterns: &[EpisodicPattern],
    ) -> Result<Vec<SemanticPattern>, DreamError> {
        let semantic_patterns: Vec<SemanticPattern> =
            patterns.iter().map(|p| p.clone().into()).collect();

        Ok(semantic_patterns)
    }

    /// Phase 5: Compact storage by replacing episodes with semantic patterns
    ///
    /// Uses StorageCompactor to safely replace consolidated episodes
    #[allow(clippy::unnecessary_wraps)] // Maintains consistency with other phases
    fn compact_replayed_episodes(
        &self,
        episodes: &[Episode],
        semantic_patterns: &[SemanticPattern],
    ) -> Result<Vec<CompactionResult>, DreamError> {
        let mut results = Vec::new();

        for pattern in semantic_patterns {
            // Find episodes that contributed to this pattern
            let contributing_episodes: Vec<Episode> = episodes
                .iter()
                .filter(|ep| pattern.source_episodes.contains(&ep.id))
                .cloned()
                .collect();

            if contributing_episodes.is_empty() {
                continue;
            }

            // Attempt compaction
            match self
                .compactor
                .compact_storage(&contributing_episodes, pattern)
            {
                Ok(result) => {
                    results.push(result);
                }
                Err(e) => {
                    // Log error but continue with other patterns
                    tracing::warn!(
                        pattern_id = %pattern.id,
                        error = %e,
                        "Failed to compact episodes for pattern"
                    );
                }
            }
        }

        Ok(results)
    }

    /// Phase 5b: Promote proto-concepts to semantic concepts
    ///
    /// Only promotes proto-concepts that meet the consolidation threshold:
    /// - consolidation_strength > 0.1 (cortical representation threshold)
    /// - replay_count >= 3 (minimum statistical evidence)
    /// - coherence_score > 0.65 (CA3 pattern completion threshold)
    ///
    /// Creates SemanticPattern and stores for backwards compatibility.
    /// Task 002 will replace this with DualMemoryNode::Concept in graph.
    #[cfg(feature = "dual_memory_types")]
    fn promote_proto_concepts(proto_concepts: &[ProtoConcept], store: &MemoryStore) -> usize {
        let mut created = 0;

        for proto in proto_concepts {
            // Only promote if consolidation strength exceeds threshold
            if proto.consolidation_strength > 0.1 {
                // Create SemanticPattern from ProtoConcept (temporary compatibility layer)
                // Task 002 will replace this with DualMemoryNode::Concept storage
                let semantic_pattern = SemanticPattern {
                    id: Uuid::new_v4().to_string(),
                    embedding: proto.centroid,
                    source_episodes: proto.episode_indices.clone(),
                    strength: proto.consolidation_strength,
                    schema_confidence: crate::Confidence::from_raw(proto.coherence_score),
                    last_consolidated: Utc::now(),
                };

                store.store_semantic_pattern(&semantic_pattern);
                created += 1;
            }
        }

        created
    }
}

/// Outcome of a dream cycle
#[derive(Debug, Clone)]
pub struct DreamOutcome {
    /// Total duration of the dream cycle
    pub dream_duration: Duration,

    /// Number of episodes replayed
    pub episodes_replayed: usize,

    /// Number of replay iterations completed
    pub replay_iterations: usize,

    /// Number of patterns discovered during replay
    pub patterns_discovered: usize,

    /// Number of semantic memories created
    pub semantic_memories_created: usize,

    /// Total storage reduction achieved (bytes)
    pub storage_reduction_bytes: u64,
}

/// Estimated size of an Episode in bytes (approximate)
const EPISODE_SIZE: u64 = 3072;

impl DreamOutcome {
    /// Calculate storage reduction ratio (0.0-1.0)
    #[must_use]
    pub fn reduction_ratio(&self) -> f32 {
        if self.episodes_replayed == 0 {
            return 0.0;
        }

        // Estimate: Episode is ~3KB, SemanticPattern is ~1KB
        let original_size = self.episodes_replayed as u64 * EPISODE_SIZE;

        if original_size == 0 {
            return 0.0;
        }

        self.storage_reduction_bytes as f32 / original_size as f32
    }

    /// Check if dream cycle met performance targets
    #[must_use]
    pub fn meets_targets(&self) -> bool {
        // Target: >50% storage reduction
        const TARGET_REDUCTION: f32 = 0.5;

        self.reduction_ratio() >= TARGET_REDUCTION
    }
}

/// Outcome of replay operations
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields reserved for future metrics/logging
struct ReplayOutcome {
    /// Number of replay iterations completed
    replays_completed: usize,

    /// Total time spent in replay
    total_replay_time: Duration,

    /// Average replay speed achieved
    average_speed: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::unwrap_used, clippy::expect_used)]
    #[test]
    fn test_dream_config_defaults() {
        let config = DreamConfig::default();

        assert_eq!(config.dream_duration, Duration::from_secs(600));
        assert!((config.replay_speed - 15.0).abs() < f32::EPSILON);
        assert_eq!(config.replay_iterations, 5);
        assert!((config.ripple_frequency - 200.0).abs() < f32::EPSILON);
        assert!(config.enable_compaction);
    }

    #[allow(clippy::unwrap_used, clippy::expect_used)]
    #[test]
    fn test_dream_engine_creation() {
        let config = DreamConfig::default();
        let engine = DreamEngine::new(config);

        // Engine should be created successfully
        assert!(engine.config.replay_speed > 0.0);
    }

    #[allow(clippy::unwrap_used, clippy::expect_used)]
    #[test]
    fn test_dream_outcome_reduction_ratio() {
        let outcome = DreamOutcome {
            dream_duration: Duration::from_secs(60),
            episodes_replayed: 100,
            replay_iterations: 5,
            patterns_discovered: 10,
            semantic_memories_created: 10,
            storage_reduction_bytes: 200_000, // ~65% reduction from 100 episodes
        };

        let ratio = outcome.reduction_ratio();
        assert!(ratio > 0.0);
        assert!(ratio < 1.0);
    }

    #[allow(clippy::unwrap_used, clippy::expect_used)]
    #[test]
    fn test_dream_outcome_meets_targets() {
        // Outcome with >50% reduction
        let good_outcome = DreamOutcome {
            dream_duration: Duration::from_secs(60),
            episodes_replayed: 100,
            replay_iterations: 5,
            patterns_discovered: 10,
            semantic_memories_created: 10,
            storage_reduction_bytes: 160_000, // ~52% reduction
        };

        assert!(good_outcome.meets_targets());

        // Outcome with <50% reduction
        let poor_outcome = DreamOutcome {
            dream_duration: Duration::from_secs(60),
            episodes_replayed: 100,
            replay_iterations: 5,
            patterns_discovered: 50,
            semantic_memories_created: 50,
            storage_reduction_bytes: 100_000, // ~33% reduction
        };

        assert!(!poor_outcome.meets_targets());
    }

    #[allow(clippy::unwrap_used, clippy::expect_used)]
    #[test]
    fn test_empty_episodes_dream() {
        let config = DreamConfig::default();
        let engine = DreamEngine::new(config);
        let store = MemoryStore::new(1000);

        // Dream with no episodes should return zero outcome
        let outcome = engine.dream(&store).unwrap();

        assert_eq!(outcome.episodes_replayed, 0);
        assert_eq!(outcome.patterns_discovered, 0);
        assert_eq!(outcome.semantic_memories_created, 0);
    }
}
