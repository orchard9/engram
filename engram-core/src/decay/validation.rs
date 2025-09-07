//! Empirical validation infrastructure.

use crate::decay::DecayResult;

/// Empirical validator for decay functions
#[derive(Debug, Clone)]
pub struct EmpiricalValidator {
    /// Target accuracy threshold
    pub accuracy_threshold: f32,
}

impl Default for EmpiricalValidator {
    fn default() -> Self {
        Self { accuracy_threshold: 0.95 }
    }
}

impl EmpiricalValidator {
    pub fn new() -> Self { Self::default() }
    
    pub fn validate_ebbinghaus_curve(&self, predicted: f32, empirical: f32) -> DecayResult<bool> {
        let error = (predicted - empirical).abs() / empirical;
        Ok(error < 0.05) // 5% error tolerance
    }
    
    pub fn validate_bahrick_permastore(&self, predicted: f32, empirical: f32) -> DecayResult<bool> {
        let error = (predicted - empirical).abs() / empirical;
        Ok(error < 0.1) // 10% error tolerance for long-term data
    }
}