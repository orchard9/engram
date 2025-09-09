//! Monitoring provider abstraction for metrics and telemetry
//!
//! This module provides a trait-based abstraction over monitoring backends,
//! allowing graceful fallback from Prometheus to no-op monitoring.

use super::FeatureProvider;
use std::any::Any;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during monitoring operations
#[derive(Debug, Error)]
pub enum MonitoringError {
    #[error("Monitoring operation failed: {0}")]
    OperationFailed(String),
    
    #[error("Metric not found: {0}")]
    MetricNotFound(String),
    
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
    fn start_timer(&self, name: &str) -> Box<dyn Timer>;
    
    /// Get current metric value
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
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
}

/// Provider trait for monitoring implementations
pub trait MonitoringProvider: FeatureProvider {
    /// Create a new monitoring instance
    fn create_monitoring(&self) -> Box<dyn Monitoring>;
    
    /// Get monitoring configuration
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
    pub fn new() -> Self {
        Self {
            config: MonitoringConfig::default(),
        }
    }
    
    pub fn with_config(config: MonitoringConfig) -> Self {
        Self { config }
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
        
        let metrics = Arc::new(MetricsRegistry::new());
        
        Self { config, metrics }
    }
}

#[cfg(feature = "monitoring")]
impl Monitoring for PrometheusMonitoringImpl {
    fn record_counter(&self, _name: &str, value: u64, _labels: &[(String, String)]) {
        // Map to MetricsRegistry counter
        // Note: MetricsRegistry uses static str, so we can't directly use dynamic names
        // This is a limitation we'll need to address in future refactoring
        self.metrics.increment_counter("custom_counter", value);
    }
    
    fn record_gauge(&self, _name: &str, _value: f64, _labels: &[(String, String)]) {
        // MetricsRegistry doesn't have direct gauge support, use cognitive metrics
        use crate::metrics::cognitive::{CognitiveMetric, CalibrationCorrection};
        self.metrics.record_cognitive(CognitiveMetric::ConfidenceCalibration {
            correction_type: CalibrationCorrection::BaseRate,
        });
    }
    
    fn record_histogram(&self, _name: &str, value: f64, _labels: &[(String, String)]) {
        self.metrics.observe_histogram("custom_histogram", value);
    }
    
    fn start_timer(&self, name: &str) -> Box<dyn Timer> {
        Box::new(PrometheusTimer {
            start: std::time::Instant::now(),
            name: name.to_string(),
            metrics: self.metrics.clone(),
        })
    }
    
    fn get_metric(&self, _name: &str) -> MonitoringResult<MetricValue> {
        // MetricsRegistry doesn't expose query methods yet
        Ok(MetricValue::Counter(0))
    }
}

#[cfg(feature = "monitoring")]
struct PrometheusTimer {
    start: std::time::Instant,
    name: String,
    metrics: Arc<crate::metrics::MetricsRegistry>,
}

#[cfg(feature = "monitoring")]
impl Timer for PrometheusTimer {
    fn stop(self: Box<Self>) {
        let duration = self.start.elapsed();
        self.metrics.observe_histogram("timer", duration.as_secs_f64());
    }
}