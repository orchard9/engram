//! Spacing Effect Validation Test Suite
//!
//! Integration test for spacing effect validation based on Cepeda et al. (2006).
//! This test validates that Engram's temporal dynamics replicate the spacing effect.

// Include the psychology test modules
#[path = "psychology/spacing_time_simulation.rs"]
mod spacing_time_simulation;

#[path = "psychology/spacing_effect.rs"]
mod spacing_effect;

// Note: spacing_helpers is included by spacing_effect, not directly here
