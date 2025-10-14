//! Monitoring provider abstraction for metrics and telemetry
//!
//! This module provides a trait-based abstraction over monitoring backends,
//! allowing graceful fallback from Engram's internal streaming pipeline to no-op monitoring.

use super::FeatureProvider;
use std::any::Any;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during monitoring operations
#[derive(Debug, Error)]
pub enum MonitoringError {
    /// The backend failed while recording or exporting metrics.
    #[error("Monitoring operation failed: {0}")]
    OperationFailed(String),

    /// The requested metric was not found.
    #[error("Metric not found: {0}")]
    MetricNotFound(String),

    /// The provided metric value is invalid for the requested operation.
    #[error("Invalid metric value: {0}")]
    InvalidValue(String),
}

/// Result type for monitoring operations
pub type MonitoringResult<T> = Result<T, MonitoringError>;

/// Trait for monitoring operations
pub trait Monitoring: Send + Sync {
    /// Record a counter metric
    fn record_counter(&self, name: &'static str, value: u64, labels: &[(String, String)]);

    /// Record a gauge metric
    fn record_gauge(&self, name: &'static str, value: f64, labels: &[(String, String)]);

    /// Record a histogram metric
    fn record_histogram(&self, name: &'static str, value: f64, labels: &[(String, String)]);

    /// Start a timer for measuring duration
    #[must_use]
    fn start_timer(&self, name: &'static str) -> Box<dyn Timer>;

    /// Get current metric value
    ///
    /// # Errors
    /// Returns [`MonitoringError::MetricNotFound`] when the requested metric is unknown or
    /// [`MonitoringError::OperationFailed`] when the backend cannot fulfill the query.
    fn get_metric(&self, name: &'static str) -> MonitoringResult<MetricValue>;
}

/// Timer for measuring durations
pub trait Timer: Send + Sync {
    /// Stop the timer and record the duration
    fn stop(self: Box<Self>);
}

/// Value of a metric
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Counter value representing an accumulated count.
    Counter(u64),
    /// Gauge value representing the latest measurement.
    Gauge(f64),
    /// Histogram samples collected for a metric.
    Histogram(Vec<f64>),
}

/// Provider trait for monitoring implementations
pub trait MonitoringProvider: FeatureProvider {
    /// Create a new monitoring instance
    #[must_use]
    fn create_monitoring(&self) -> Box<dyn Monitoring>;

    /// Get monitoring configuration
    #[must_use]
    fn get_config(&self) -> MonitoringConfig;
}

/// Configuration for monitoring operations
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Destination identifier for streaming metrics (e.g., log target).
    pub endpoint: String,
    /// Metric prefix
    pub prefix: String,
    /// Suggested collection interval for aggregators
    pub interval: Duration,
    /// Enable detailed metrics
    pub detailed: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            endpoint: "logs://tracing".to_string(),
            prefix: "engram".to_string(),
            interval: Duration::from_secs(60),
            detailed: false,
        }
    }
}

/// Streaming/log monitoring provider (available when monitoring feature is enabled)
#[cfg(feature = "monitoring")]
pub struct StreamingMonitoringProvider {
    config: MonitoringConfig,
}

#[cfg(feature = "monitoring")]
impl StreamingMonitoringProvider {
    /// Create a provider with the default streaming configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: MonitoringConfig::default(),
        }
    }

    /// Create a provider using caller-specified configuration.
    #[must_use]
    pub const fn with_config(config: MonitoringConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "monitoring")]
impl Default for StreamingMonitoringProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "monitoring")]
impl FeatureProvider for StreamingMonitoringProvider {
    fn is_enabled(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "monitoring"
    }

    fn description(&self) -> &'static str {
        "Internal streaming metrics and structured logs"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(feature = "monitoring")]
impl MonitoringProvider for StreamingMonitoringProvider {
    fn create_monitoring(&self) -> Box<dyn Monitoring> {
        Box::new(StreamingMonitoringImpl::new(self.config.clone()))
    }

    fn get_config(&self) -> MonitoringConfig {
        self.config.clone()
    }
}

/// Internal monitoring implementation backed by the MetricsRegistry streaming pipeline.
#[cfg(feature = "monitoring")]
struct StreamingMonitoringImpl {
    config: MonitoringConfig,
    metrics: Arc<crate::metrics::MetricsRegistry>,
}

#[cfg(feature = "monitoring")]
impl StreamingMonitoringImpl {
    fn new(config: MonitoringConfig) -> Self {
        use crate::metrics::MetricsRegistry;

        tracing::debug!(target = "engram::monitoring", destination = %config.endpoint, "initialising streaming metrics");

        let metrics = Arc::new(MetricsRegistry::new());

        Self { config, metrics }
    }

    fn maybe_log_snapshot(&self, label: &str) {
        if self.config.detailed {
            self.metrics.log_streaming_snapshot(label);
        }
    }
}

#[cfg(feature = "monitoring")]
impl Monitoring for StreamingMonitoringImpl {
    fn record_counter(&self, name: &'static str, value: u64, _labels: &[(String, String)]) {
        self.metrics.increment_counter(name, value);
        self.maybe_log_snapshot(name);
    }

    fn record_gauge(&self, name: &'static str, value: f64, _labels: &[(String, String)]) {
        self.metrics.record_gauge(name, value);
        self.maybe_log_snapshot(name);
    }

    fn record_histogram(&self, name: &'static str, value: f64, _labels: &[(String, String)]) {
        self.metrics.observe_histogram(name, value);
        self.maybe_log_snapshot(name);
    }

    fn start_timer(&self, name: &'static str) -> Box<dyn Timer> {
        Box::new(StreamingTimer {
            start: std::time::Instant::now(),
            metric_name: name,
            metrics: Arc::clone(&self.metrics),
            detailed: self.config.detailed,
        })
    }

    fn get_metric(&self, name: &'static str) -> MonitoringResult<MetricValue> {
        if let Some(gauge) = self.metrics.gauge_value(name) {
            return Ok(MetricValue::Gauge(gauge));
        }

        let counter = self.metrics.counter_value(name);
        if counter > 0 {
            return Ok(MetricValue::Counter(counter));
        }

        let quantiles = self.metrics.histogram_quantiles(name, &[0.5, 0.9, 0.99]);
        if quantiles
            .iter()
            .any(|value| value.is_finite() && *value > 0.0)
        {
            return Ok(MetricValue::Histogram(quantiles));
        }

        Err(MonitoringError::MetricNotFound(name.to_string()))
    }
}

#[cfg(feature = "monitoring")]
struct StreamingTimer {
    start: std::time::Instant,
    metric_name: &'static str,
    metrics: Arc<crate::metrics::MetricsRegistry>,
    detailed: bool,
}

#[cfg(feature = "monitoring")]
impl Timer for StreamingTimer {
    fn stop(self: Box<Self>) {
        let duration = self.start.elapsed();
        self.metrics
            .observe_histogram(self.metric_name, duration.as_secs_f64());

        if self.detailed {
            self.metrics.log_streaming_snapshot(self.metric_name);
        }
    }
}
