//! Cognitive architecture modules
//!
//! Implements biologically-inspired cognitive patterns

pub mod priming;
pub mod reconsolidation;

pub use priming::{PrimingStatistics, SemanticPrimingEngine};
pub use reconsolidation::{
    EpisodeModifications, ModificationType, ReconsolidationEngine, ReconsolidationResult,
};
