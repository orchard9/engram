//! Monitoring provider abstraction for metrics and telemetry
//!
//! This module provides a trait-based abstraction over monitoring backends,
//! allowing graceful fallback from Prometheus to no-op monitoring.

use super::FeatureProvider;
use std::any::Any;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

#[cfg(feature = "monitoring")]
use crate::metrics::cognitive::{CalibrationCorrection, CognitiveMetric};

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
    fn record_counter(&self, name: &str, value: u64, labels: &[(String, String)]);

    /// Record a gauge metric
    fn record_gauge(&self, name: &str, value: f64, labels: &[(String, String)]);

    /// Record a histogram metric
    fn record_histogram(&self, name: &str, value: f64, labels: &[(String, String)]);

    /// Start a timer for measuring duration
    #[must_use]
    fn start_timer(&self, name: &str) -> Box<dyn Timer>;

    /// Get current metric value
    ///
    /// # Errors
    /// Returns [`MonitoringError::MetricNotFound`] when the requested metric is unknown or
    /// [`MonitoringError::OperationFailed`] when the backend cannot fulfill the query.
    fn get_metric(&self, name: &str) -> MonitoringResult<MetricValue>;
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
    /// Prometheus endpoint
    pub endpoint: String,
    /// Metric prefix
    pub prefix: String,
    /// Collection interval
    pub interval: Duration,
    /// Enable detailed metrics
    pub detailed: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            endpoint: "0.0.0.0:9090".to_string(),
            prefix: "engram".to_string(),
            interval: Duration::from_secs(60),
            detailed: false,
        }
    }
}

/// Prometheus monitoring provider (only available when feature is enabled)
#[cfg(feature = "monitoring")]
pub struct PrometheusMonitoringProvider {
    config: MonitoringConfig,
}

#[cfg(feature = "monitoring")]
impl PrometheusMonitoringProvider {
    /// Create a provider with the default Prometheus configuration.
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
impl Default for PrometheusMonitoringProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "monitoring")]
impl FeatureProvider for PrometheusMonitoringProvider {
    fn is_enabled(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "monitoring"
    }

    fn description(&self) -> &'static str {
        "Prometheus-based monitoring and metrics collection"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(feature = "monitoring")]
impl MonitoringProvider for PrometheusMonitoringProvider {
    fn create_monitoring(&self) -> Box<dyn Monitoring> {
        Box::new(PrometheusMonitoringImpl::new(self.config.clone()))
    }

    fn get_config(&self) -> MonitoringConfig {
        self.config.clone()
    }
}

/// Actual Prometheus implementation
#[cfg(feature = "monitoring")]
struct PrometheusMonitoringImpl {
    config: MonitoringConfig,
    metrics: Arc<crate::metrics::MetricsRegistry>,
}

#[cfg(feature = "monitoring")]
impl PrometheusMonitoringImpl {
    fn new(config: MonitoringConfig) -> Self {
        use crate::metrics::MetricsRegistry;

        tracing::debug!(endpoint = %config.endpoint, "initialising Prometheus monitoring");

        let metrics = Arc::new(MetricsRegistry::new());

        Self { config, metrics }
    }

    fn should_record_metric(&self, name: &str) -> bool {
        self.config.detailed || !name.starts_with("detailed")
    }
}

#[cfg(feature = "monitoring")]
impl Monitoring for PrometheusMonitoringImpl {
    fn record_counter(&self, name: &str, value: u64, _labels: &[(String, String)]) {
        if !self.should_record_metric(name) {
            return;
        }
        // Map to MetricsRegistry counter
        // Note: MetricsRegistry uses static str, so we can't directly use dynamic names
        // This is a limitation we'll need to address in future refactoring
        self.metrics.increment_counter("custom_counter", value);
    }

    fn record_gauge(&self, name: &str, _value: f64, _labels: &[(String, String)]) {
        if !self.should_record_metric(name) {
            return;
        }
        // MetricsRegistry doesn't have direct gauge support, use cognitive metrics
        let metric = CognitiveMetric::ConfidenceCalibration {
            correction_type: CalibrationCorrection::BaseRate,
        };
        self.metrics.record_cognitive(&metric);
    }

    fn record_histogram(&self, name: &str, value: f64, _labels: &[(String, String)]) {
        if !self.should_record_metric(name) {
            return;
        }
        let normalised = value / self.config.interval.as_secs_f64().max(1.0);
        self.metrics
            .observe_histogram("custom_histogram", normalised);
    }

    fn start_timer(&self, name: &str) -> Box<dyn Timer> {
        Box::new(PrometheusTimer {
            start: std::time::Instant::now(),
            name: format!("{}.{name}", self.config.prefix),
            metrics: self.metrics.clone(),
            detailed: self.config.detailed,
        })
    }

    fn get_metric(&self, _name: &str) -> MonitoringResult<MetricValue> {
        // MetricsRegistry doesn't expose query methods yet
        if self.config.detailed {
            Ok(MetricValue::Gauge(self.config.interval.as_secs_f64()))
        } else {
            Ok(MetricValue::Counter(0))
        }
    }
}

#[cfg(feature = "monitoring")]
struct PrometheusTimer {
    start: std::time::Instant,
    name: String,
    metrics: Arc<crate::metrics::MetricsRegistry>,
    detailed: bool,
}

#[cfg(feature = "monitoring")]
impl Timer for PrometheusTimer {
    fn stop(self: Box<Self>) {
        let duration = self.start.elapsed();
        let metric_key = if self.detailed {
            "timer_detailed"
        } else {
            "timer"
        };
        tracing::debug!(timer = %self.name, elapsed = duration.as_secs_f64(), "recorded Prometheus timer");
        self.metrics
            .observe_histogram(metric_key, duration.as_secs_f64());
    }
}
