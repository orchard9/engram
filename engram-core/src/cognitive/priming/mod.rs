//! Priming effects based on spreading activation theory
//!
//! Implements Collins & Loftus (1975) spreading activation with biological
//! constraints validated against Neely (1977) temporal dynamics.

pub mod semantic;

pub use semantic::{PrimingStatistics, SemanticPrimingEngine};
