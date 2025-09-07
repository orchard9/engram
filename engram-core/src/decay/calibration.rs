//! Confidence calibration and overconfidence correction.

use crate::Confidence;

/// Confidence calibrator for overconfidence correction
#[derive(Debug, Clone)]
pub struct ConfidenceCalibrator {
    /// Calibration strength
    pub calibration_strength: f32,
}

impl Default for ConfidenceCalibrator {
    fn default() -> Self {
        Self { calibration_strength: 0.85 }
    }
}

impl ConfidenceCalibrator {
    pub fn new() -> Self { Self::default() }
    
    pub fn calibrate_overconfidence(&self, confidence: Confidence) -> Confidence {
        confidence.calibrate_overconfidence()
    }
}