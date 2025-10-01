//! Sharp-wave ripple consolidation processor.

use chrono::{DateTime, Utc};
use std::convert::TryFrom;

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
    /// Create a new consolidation processor with default settings
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Detects sharp-wave ripple patterns
    #[must_use]
    pub fn detect_ripple(&self, activation_pattern: &[f32]) -> bool {
        if activation_pattern.len() < 5 {
            return false;
        }

        let Some(len) = Self::len_to_f32(activation_pattern.len()) else {
            return false;
        };
        let mean = activation_pattern.iter().sum::<f32>() / len;
        let variance = activation_pattern
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / len;

        mean > self.ripple_threshold && variance > 0.1 && mean < 0.7
    }

    /// Records consolidation event
    pub fn record_consolidation(&mut self) {
        self.last_consolidation = Some(Utc::now());
    }

    /// Gets consolidation boost factor
    #[must_use]
    pub fn consolidation_boost(&self) -> f32 {
        self.last_consolidation.map_or(1.0, |last| {
            (Utc::now() - last).to_std().map_or(1.0, |elapsed| {
                let hours_since = elapsed.as_secs_f32() / 3600.0;
                if hours_since < 24.0 {
                    self.strength_multiplier * (1.0 - hours_since / 24.0)
                } else {
                    1.0
                }
            })
        })
    }

    fn len_to_f32(len: usize) -> Option<f32> {
        u32::try_from(len).ok().map(|value| {
            let value_f64 = f64::from(value);
            #[allow(clippy::cast_possible_truncation)]
            {
                value_f64 as f32
            }
        })
    }
}
