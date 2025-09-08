//! Decay provider abstraction for psychological decay functions
//!
//! This module provides a trait-based abstraction over decay functions,
//! allowing graceful fallback from psychological models to simple time-based decay.

use super::FeatureProvider;
use crate::{Confidence, Episode};
use std::any::Any;
use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during decay operations
#[derive(Debug, Error)]
pub enum DecayError {
    #[error("Decay calculation failed: {0}")]
    CalculationFailed(String),
    
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

/// Result type for decay operations
pub type DecayResult<T> = Result<T, DecayError>;

/// Trait for decay operations
pub trait Decay: Send + Sync {
    /// Calculate decay factor for a given time delta
    fn calculate_decay(&self, elapsed: Duration) -> f32;
    
    /// Apply decay to an episode
    fn apply_decay(&self, episode: &mut Episode, elapsed: Duration);
    
    /// Get decay parameters
    fn get_parameters(&self) -> DecayParameters;
    
    /// Update decay parameters
    fn set_parameters(&mut self, params: DecayParameters) -> DecayResult<()>;
}

/// Parameters for decay functions
#[derive(Debug, Clone)]
pub struct DecayParameters {
    /// Base decay rate
    pub base_rate: f32,
    /// Forgetting curve exponent
    pub exponent: f32,
    /// Memory strength factor
    pub strength_factor: f32,
    /// Individual difference modifier
    pub individual_modifier: f32,
}

impl Default for DecayParameters {
    fn default() -> Self {
        Self {
            base_rate: 0.1,
            exponent: 0.5,
            strength_factor: 1.0,
            individual_modifier: 1.0,
        }
    }
}

/// Provider trait for decay implementations
pub trait DecayProvider: FeatureProvider {
    /// Create a new decay instance
    fn create_decay(&self) -> Box<dyn Decay>;
    
    /// Get decay configuration
    fn get_config(&self) -> DecayConfig;
}

/// Configuration for decay operations
#[derive(Debug, Clone)]
pub struct DecayConfig {
    /// Type of decay function
    pub decay_type: DecayType,
    /// Default parameters
    pub default_params: DecayParameters,
    /// Enable consolidation effects
    pub consolidation: bool,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            decay_type: DecayType::Ebbinghaus,
            default_params: DecayParameters::default(),
            consolidation: true,
        }
    }
}

/// Types of decay functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecayType {
    /// Ebbinghaus forgetting curve
    Ebbinghaus,
    /// Power law decay
    PowerLaw,
    /// Exponential decay
    Exponential,
    /// Two-component model (fast and slow decay)
    TwoComponent,
}

/// Psychological decay provider (only available when feature is enabled)
#[cfg(feature = "psychological_decay")]
pub struct PsychologicalDecayProvider {
    config: DecayConfig,
}

#[cfg(feature = "psychological_decay")]
impl PsychologicalDecayProvider {
    pub fn new() -> Self {
        Self {
            config: DecayConfig::default(),
        }
    }
    
    pub fn with_config(config: DecayConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "psychological_decay")]
impl FeatureProvider for PsychologicalDecayProvider {
    fn is_enabled(&self) -> bool {
        true
    }
    
    fn name(&self) -> &'static str {
        "psychological_decay"
    }
    
    fn description(&self) -> &'static str {
        "Psychological decay functions based on memory research"
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(feature = "psychological_decay")]
impl DecayProvider for PsychologicalDecayProvider {
    fn create_decay(&self) -> Box<dyn Decay> {
        match self.config.decay_type {
            DecayType::Ebbinghaus => Box::new(EbbinghausDecay::new(self.config.default_params.clone())),
            DecayType::PowerLaw => Box::new(PowerLawDecay::new(self.config.default_params.clone())),
            DecayType::Exponential => Box::new(ExponentialDecay::new(self.config.default_params.clone())),
            DecayType::TwoComponent => Box::new(TwoComponentDecay::new(self.config.default_params.clone())),
        }
    }
    
    fn get_config(&self) -> DecayConfig {
        self.config.clone()
    }
}

/// Ebbinghaus forgetting curve implementation
#[cfg(feature = "psychological_decay")]
struct EbbinghausDecay {
    params: DecayParameters,
}

#[cfg(feature = "psychological_decay")]
impl EbbinghausDecay {
    fn new(params: DecayParameters) -> Self {
        Self { params }
    }
}

#[cfg(feature = "psychological_decay")]
impl Decay for EbbinghausDecay {
    fn calculate_decay(&self, elapsed: Duration) -> f32 {
        // Simple Ebbinghaus-like decay: R(t) = e^(-t/τ) where τ is strength factor
        let hours = elapsed.as_secs_f32() / 3600.0;
        (-hours / self.params.strength_factor).exp()
    }
    
    fn apply_decay(&self, episode: &mut Episode, elapsed: Duration) {
        let decay_factor = self.calculate_decay(elapsed);
        episode.decay_rate = decay_factor;
        
        // Update confidence based on decay
        let current_confidence = episode.encoding_confidence.raw();
        episode.encoding_confidence = Confidence::exact(current_confidence * decay_factor);
    }
    
    fn get_parameters(&self) -> DecayParameters {
        self.params.clone()
    }
    
    fn set_parameters(&mut self, params: DecayParameters) -> DecayResult<()> {
        self.params = params;
        Ok(())
    }
}

/// Power law decay implementation
#[cfg(feature = "psychological_decay")]
struct PowerLawDecay {
    params: DecayParameters,
}

#[cfg(feature = "psychological_decay")]
impl PowerLawDecay {
    fn new(params: DecayParameters) -> Self {
        Self { params }
    }
}

#[cfg(feature = "psychological_decay")]
impl Decay for PowerLawDecay {
    fn calculate_decay(&self, elapsed: Duration) -> f32 {
        let hours = elapsed.as_secs_f32() / 3600.0;
        (1.0 + self.params.base_rate * hours).powf(-self.params.exponent)
    }
    
    fn apply_decay(&self, episode: &mut Episode, elapsed: Duration) {
        let decay_factor = self.calculate_decay(elapsed);
        episode.decay_rate = decay_factor;
        
        let current_confidence = episode.encoding_confidence.raw();
        episode.encoding_confidence = Confidence::exact(current_confidence * decay_factor);
    }
    
    fn get_parameters(&self) -> DecayParameters {
        self.params.clone()
    }
    
    fn set_parameters(&mut self, params: DecayParameters) -> DecayResult<()> {
        self.params = params;
        Ok(())
    }
}

/// Exponential decay implementation
#[cfg(feature = "psychological_decay")]
struct ExponentialDecay {
    params: DecayParameters,
}

#[cfg(feature = "psychological_decay")]
impl ExponentialDecay {
    fn new(params: DecayParameters) -> Self {
        Self { params }
    }
}

#[cfg(feature = "psychological_decay")]
impl Decay for ExponentialDecay {
    fn calculate_decay(&self, elapsed: Duration) -> f32 {
        let hours = elapsed.as_secs_f32() / 3600.0;
        (-self.params.base_rate * hours).exp()
    }
    
    fn apply_decay(&self, episode: &mut Episode, elapsed: Duration) {
        let decay_factor = self.calculate_decay(elapsed);
        episode.decay_rate = decay_factor;
        
        let current_confidence = episode.encoding_confidence.raw();
        episode.encoding_confidence = Confidence::exact(current_confidence * decay_factor);
    }
    
    fn get_parameters(&self) -> DecayParameters {
        self.params.clone()
    }
    
    fn set_parameters(&mut self, params: DecayParameters) -> DecayResult<()> {
        self.params = params;
        Ok(())
    }
}

/// Two-component decay model
#[cfg(feature = "psychological_decay")]
struct TwoComponentDecay {
    params: DecayParameters,
}

#[cfg(feature = "psychological_decay")]
impl TwoComponentDecay {
    fn new(params: DecayParameters) -> Self {
        Self { params }
    }
}

#[cfg(feature = "psychological_decay")]
impl Decay for TwoComponentDecay {
    fn calculate_decay(&self, elapsed: Duration) -> f32 {
        // Simple two-component model: fast + slow exponential decay
        let hours = elapsed.as_secs_f32() / 3600.0;
        let fast_decay = 0.6 * (-hours / 24.0).exp(); // Fast component (24h half-life)
        let slow_decay = 0.4 * (-hours / (24.0 * 30.0)).exp(); // Slow component (30 day half-life)
        fast_decay + slow_decay
    }
    
    fn apply_decay(&self, episode: &mut Episode, elapsed: Duration) {
        let decay_factor = self.calculate_decay(elapsed);
        episode.decay_rate = decay_factor;
        
        let current_confidence = episode.encoding_confidence.raw();
        episode.encoding_confidence = Confidence::exact(current_confidence * decay_factor);
    }
    
    fn get_parameters(&self) -> DecayParameters {
        self.params.clone()
    }
    
    fn set_parameters(&mut self, params: DecayParameters) -> DecayResult<()> {
        self.params = params;
        Ok(())
    }
}