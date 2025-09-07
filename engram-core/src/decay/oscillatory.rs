//! Oscillatory constraints for theta-gamma coupling.

/// Oscillatory constraints processor
#[derive(Debug, Clone)]
pub struct OscillatoryConstraints {
    /// Theta frequency (4-8Hz)
    pub theta_freq: f32,
    /// Gamma frequency (30-100Hz)  
    pub gamma_freq: f32,
}

impl Default for OscillatoryConstraints {
    fn default() -> Self {
        Self {
            theta_freq: 6.0,  // 6Hz theta
            gamma_freq: 60.0, // 60Hz gamma
        }
    }
}

impl OscillatoryConstraints {
    pub fn new() -> Self { Self::default() }
    
    pub fn theta_modulation(&self, phase: f32) -> f32 {
        1.0 + 0.1 * (phase * 2.0 * std::f32::consts::PI).sin()
    }
}