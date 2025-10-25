//! Accuracy Validation & Production Tuning
//!
//! This module provides comprehensive validation datasets and parameter tuning infrastructure
//! for pattern completion accuracy validation against ground truth.
//!
//! Components:
//! - Corrupted Episodes: Validates reconstruction accuracy at 30%, 50%, 70% corruption levels
//! - DRM Paradigm: Tests false memory formation and source attribution
//! - Serial Position: Validates biological plausibility (primacy/recency effects)
//! - Parameter Tuning: Empirical optimization via parameter sweeps and Pareto frontier analysis
//! - Isotonic Calibration: Confidence calibration with 1000+ completions

pub mod corrupted_episodes;
pub mod drm_paradigm;
pub mod isotonic_calibration;
pub mod parameter_tuning;
pub mod serial_position;
