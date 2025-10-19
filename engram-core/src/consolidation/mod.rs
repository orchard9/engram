//! Consolidation service abstractions for caching, observability, and persistence.

pub mod compaction;
pub mod pattern_detector;
pub mod service;
pub mod statistical_tests;

pub use compaction::{CompactionConfig, CompactionResult, StorageCompactor};
pub use pattern_detector::{
    EpisodicPattern, PatternDetectionConfig, PatternDetector, PatternFeature,
};
pub use service::{
    BeliefUpdateRecord, ConsolidationCacheSource, ConsolidationService,
    InMemoryConsolidationService,
};
pub use statistical_tests::{SIGNIFICANCE_THRESHOLD, StatisticalFilter};
