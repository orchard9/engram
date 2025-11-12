//! Consolidation service abstractions for caching, observability, and persistence.

pub mod clustering;
pub mod compaction;
pub mod concept_formation;
pub mod concept_integration;
pub mod dream;
pub mod pattern_detector;
pub mod service;
pub mod statistical_tests;

pub use clustering::{BiologicalClusterer, EpisodeCluster};
pub use compaction::{CompactionConfig, CompactionResult, StorageCompactor};
#[cfg(feature = "dual_memory_types")]
pub use concept_formation::ConceptFormationResult;
pub use concept_formation::{ConceptFormationEngine, ConceptSignature, ProtoConcept, SleepStage};
pub use concept_integration::{
    CircuitBreakerReason, ConceptFormationConfig, ConceptFormationHelper, ConceptFormationStats,
    ConsolidationCycleState, SkipReason,
};
pub use dream::{DreamConfig, DreamEngine, DreamError, DreamOutcome, ExtendedDreamOutcome};
pub use pattern_detector::{
    EpisodicPattern, PatternDetectionConfig, PatternDetector, PatternFeature,
};
pub use service::{
    BeliefUpdateRecord, ConsolidationCacheSource, ConsolidationService,
    InMemoryConsolidationService,
};
pub use statistical_tests::{SIGNIFICANCE_THRESHOLD, StatisticalFilter};
