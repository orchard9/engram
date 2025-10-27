//! Cognitive architecture modules
//!
//! Implements biologically-inspired cognitive patterns

pub mod interference;
pub mod priming;
pub mod reconsolidation;

pub use interference::{
    InterferenceType, ProactiveInterferenceDetector, ProactiveInterferenceResult,
};
pub use priming::{PrimingStatistics, SemanticPrimingEngine};
pub use reconsolidation::{
    EpisodeModifications, ModificationType, ReconsolidationEngine, ReconsolidationResult,
    consolidation_integration::{ReconsolidationConsolidationBridge, ReconsolidationError},
};
