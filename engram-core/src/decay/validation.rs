//! Empirical validation infrastructure.

/// Empirical validator for decay functions
#[derive(Debug, Clone)]
pub struct EmpiricalValidator {
    /// Target accuracy threshold
    pub accuracy_threshold: f32,
}

impl Default for EmpiricalValidator {
    fn default() -> Self {
        Self {
            accuracy_threshold: 0.95,
        }
    }
}

impl EmpiricalValidator {
    /// Create new empirical validator with default settings
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate predicted values against Ebbinghaus forgetting curve data
    #[must_use]
    pub fn validate_ebbinghaus_curve(&self, predicted: f32, empirical: f32) -> bool {
        if empirical == 0.0 {
            return false;
        }
        let tolerance = (1.0 - self.accuracy_threshold).max(0.05);
        let error = (predicted - empirical).abs() / empirical;
        error < tolerance
    }

    /// Validate predicted values against Bahrick permastore data
    #[must_use]
    pub fn validate_bahrick_permastore(&self, predicted: f32, empirical: f32) -> bool {
        if empirical == 0.0 {
            return false;
        }
        let base_tolerance = (1.0 - self.accuracy_threshold).max(0.05);
        let tolerance = (base_tolerance * 2.0).min(0.2);
        let error = (predicted - empirical).abs() / empirical;
        error < tolerance
    }
}
