//! Sharp-wave ripple consolidation processor.

use chrono::{DateTime, Utc};

/// Consolidation processor handling sharp-wave ripple events
#[derive(Debug, Clone)]
pub struct ConsolidationProcessor {
    /// Ripple detection threshold
    pub ripple_threshold: f32,
    /// Last consolidation event
    pub last_consolidation: Option<DateTime<Utc>>,
    /// Consolidation strength multiplier
    pub strength_multiplier: f32,
}

impl Default for ConsolidationProcessor {
    fn default() -> Self {
        Self {
            ripple_threshold: 0.3,
            last_consolidation: None,
            strength_multiplier: 1.5,
        }
    }
}

impl ConsolidationProcessor {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Detects sharp-wave ripple patterns
    pub fn detect_ripple(&self, activation_pattern: &[f32]) -> bool {
        if activation_pattern.len() < 5 {
            return false;
        }
        
        let mean = activation_pattern.iter().sum::<f32>() / activation_pattern.len() as f32;
        let variance = activation_pattern.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / activation_pattern.len() as f32;
        
        mean > self.ripple_threshold && variance > 0.1 && mean < 0.7
    }
    
    /// Records consolidation event
    pub fn record_consolidation(&mut self) {
        self.last_consolidation = Some(Utc::now());
    }
    
    /// Gets consolidation boost factor
    pub fn consolidation_boost(&self) -> f32 {
        if let Some(last) = self.last_consolidation {
            let hours_since = (Utc::now() - last).num_hours() as f32;
            if hours_since < 24.0 {
                self.strength_multiplier * (1.0 - hours_since / 24.0)
            } else {
                1.0
            }
        } else {
            1.0
        }
    }
}