//! Configuration for cognitive event tracing

use crate::tracing::event::EventType;
use rand::Rng;
use std::collections::{HashMap, HashSet};

/// Configuration for cognitive event tracing
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Which event types to trace
    pub enabled_events: HashSet<EventType>,

    /// Sampling rate per event type (0.0 - 1.0)
    pub sample_rates: HashMap<EventType, f32>,

    /// Ring buffer size per thread
    pub ring_buffer_size: usize,

    /// Export batch size
    pub export_batch_size: usize,

    /// Export interval (milliseconds)
    pub export_interval_ms: u64,

    /// Export format
    pub export_format: ExportFormat,
}

/// Export format for cognitive events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// JSON format for debugging and visualization
    Json,
    /// OpenTelemetry OTLP/gRPC format
    OtlpGrpc,
    /// Grafana Loki format
    Loki,
}

impl TracingConfig {
    /// Default configuration: disabled
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled_events: HashSet::new(),
            sample_rates: HashMap::new(),
            ring_buffer_size: 0,
            export_batch_size: 0,
            export_interval_ms: 0,
            export_format: ExportFormat::Json,
        }
    }

    /// Development configuration: trace everything
    #[must_use]
    pub fn development() -> Self {
        let mut config = Self::disabled();
        config.enabled_events.insert(EventType::Priming);
        config.enabled_events.insert(EventType::Interference);
        config.enabled_events.insert(EventType::Reconsolidation);
        config.enabled_events.insert(EventType::FalseMemory);

        config.sample_rates.insert(EventType::Priming, 1.0);
        config.sample_rates.insert(EventType::Interference, 1.0);
        config.sample_rates.insert(EventType::Reconsolidation, 1.0);
        config.sample_rates.insert(EventType::FalseMemory, 1.0);

        config.ring_buffer_size = 10_000;
        config.export_batch_size = 1_000;
        config.export_interval_ms = 5_000;

        config
    }

    /// Production configuration: sampled tracing
    #[must_use]
    pub fn production() -> Self {
        let mut config = Self::disabled();
        config.enabled_events.insert(EventType::Priming);
        config.enabled_events.insert(EventType::Interference);

        // Sample 1% of events in production
        config.sample_rates.insert(EventType::Priming, 0.01);
        config.sample_rates.insert(EventType::Interference, 0.01);

        config.ring_buffer_size = 10_000;
        config.export_batch_size = 5_000;
        config.export_interval_ms = 30_000; // 30 seconds

        config
    }

    /// Check if event type is enabled
    #[inline]
    #[must_use]
    pub fn is_enabled(&self, event_type: EventType) -> bool {
        self.enabled_events.contains(&event_type)
    }

    /// Check if event should be sampled (stochastic sampling)
    #[inline]
    #[must_use]
    pub fn should_sample(&self, event_type: EventType) -> bool {
        let rate = self.sample_rates.get(&event_type).copied().unwrap_or(0.0);

        if rate >= 1.0 {
            return true;
        }

        if rate <= 0.0 {
            return false;
        }

        // Use thread-local RNG for performance
        rand::thread_rng().r#gen::<f32>() < rate
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_config() {
        let config = TracingConfig::disabled();
        assert!(config.enabled_events.is_empty());
        assert_eq!(config.ring_buffer_size, 0);
    }

    #[test]
    fn test_development_config() {
        let config = TracingConfig::development();
        assert!(config.is_enabled(EventType::Priming));
        assert!(config.is_enabled(EventType::Interference));
        assert!(config.is_enabled(EventType::Reconsolidation));
        assert!(config.is_enabled(EventType::FalseMemory));
        assert_eq!(config.ring_buffer_size, 10_000);
    }

    #[test]
    fn test_production_config() {
        let config = TracingConfig::production();
        assert!(config.is_enabled(EventType::Priming));
        assert!(config.is_enabled(EventType::Interference));
        assert!(!config.is_enabled(EventType::Reconsolidation));
        assert!(!config.is_enabled(EventType::FalseMemory));
        assert_eq!(config.sample_rates.get(&EventType::Priming), Some(&0.01));
    }

    #[test]
    fn test_sampling() {
        let mut config = TracingConfig::disabled();
        config.sample_rates.insert(EventType::Priming, 0.0);
        assert!(!config.should_sample(EventType::Priming));

        config.sample_rates.insert(EventType::Priming, 1.0);
        assert!(config.should_sample(EventType::Priming));

        // Test stochastic sampling (should vary)
        config.sample_rates.insert(EventType::Priming, 0.5);
        let mut sampled = 0;
        for _ in 0..1000 {
            if config.should_sample(EventType::Priming) {
                sampled += 1;
            }
        }
        // Should be approximately 500 (allow for variance)
        assert!(sampled > 400 && sampled < 600, "sampled: {sampled}");
    }
}
