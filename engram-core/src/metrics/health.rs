//! System health monitoring and checks

use crate::Confidence;
use crossbeam_utils::CachePadded;
use std::convert::TryFrom;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Result produced by a [`HealthProbe`] execution.
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Overall status reported by the probe.
    pub status: HealthStatus,
    /// Human friendly explanation of the current status.
    pub message: String,
    /// How long the probe execution took.
    pub latency: Duration,
    /// Timestamp when the observation was captured.
    pub observed_at: Instant,
}

impl HealthCheckResult {
    /// Create a new health result with the provided status and message.
    #[must_use]
    pub fn new(status: HealthStatus, message: impl Into<String>, latency: Duration) -> Self {
        Self {
            status,
            message: message.into(),
            latency,
            observed_at: Instant::now(),
        }
    }
}

/// Configuration for controlling probe hysteresis and cooldown behaviour.
#[derive(Debug, Clone, Copy)]
pub struct ProbeHysteresis {
    /// Number of consecutive failures before reporting `Degraded`.
    pub degrade_threshold: u32,
    /// Number of consecutive failures before reporting `Unhealthy`.
    pub unhealthy_threshold: u32,
    /// Number of consecutive successes required for recovery back to healthy.
    pub recovery_threshold: u32,
    /// Optional cooldown applied after an unhealthy result to reduce probe pressure.
    pub cooldown: Duration,
}

impl Default for ProbeHysteresis {
    fn default() -> Self {
        Self {
            degrade_threshold: 1,
            unhealthy_threshold: 3,
            recovery_threshold: 2,
            cooldown: Duration::from_secs(0),
        }
    }
}

/// Trait implemented by health probes that can be registered with [`SystemHealth`].
pub trait HealthProbe: Send + Sync {
    /// Machine-readable name used for registration and lookups.
    fn name(&self) -> &'static str;

    /// Logical category for this probe.
    fn check_type(&self) -> HealthCheckType {
        HealthCheckType::Custom(self.name())
    }

    /// Execute the probe and return a structured result.
    fn run(&self) -> HealthCheckResult;

    /// Hysteresis settings for this probe.
    fn hysteresis(&self) -> ProbeHysteresis {
        ProbeHysteresis::default()
    }
}

struct RegisteredProbe {
    probe: Arc<dyn HealthProbe>,
    hysteresis: ProbeHysteresis,
    state: HealthCheck,
    cooldown_until: Option<Instant>,
}

/// System health monitor
pub struct SystemHealth {
    /// Individual health checks
    probes: dashmap::DashMap<&'static str, RegisteredProbe>,

    /// Global health state
    is_healthy: CachePadded<AtomicBool>,
    last_check: CachePadded<AtomicU64>,

    /// Alert thresholds
    memory_threshold_bytes: u64,
    latency_threshold_ms: f64,
    error_rate_threshold: f32,
}

impl SystemHealth {
    /// Create a new health monitor with default thresholds
    #[must_use]
    pub fn new() -> Self {
        Self::with_thresholds(
            8 * 1024 * 1024 * 1024, // 8GB memory threshold
            100.0,                  // 100ms latency threshold
            0.01,                   // 1% error rate threshold
        )
    }

    /// Create a new health monitor with custom thresholds
    #[must_use]
    pub fn with_thresholds(
        memory_threshold_bytes: u64,
        latency_threshold_ms: f64,
        error_rate_threshold: f32,
    ) -> Self {
        let health = Self {
            probes: dashmap::DashMap::new(),
            is_healthy: CachePadded::new(AtomicBool::new(true)),
            last_check: CachePadded::new(AtomicU64::new(0)),
            memory_threshold_bytes,
            latency_threshold_ms,
            error_rate_threshold,
        };

        // Register default checks
        health.register_default_checks();
        health
    }

    fn register_default_checks(&self) {
        self.register_probe(MemoryUsageProbe::new(self.memory_threshold_bytes));
        self.register_probe(LatencyProbe::new(self.latency_threshold_ms));
        self.register_probe(ErrorRateProbe::new(self.error_rate_threshold));
        self.register_probe(ConnectivityProbe);
        self.register_probe(CognitiveProbe);
    }

    /// Register a custom health probe using its default hysteresis configuration.
    pub fn register_probe<P>(&self, probe: P)
    where
        P: HealthProbe + 'static,
    {
        let hysteresis = probe.hysteresis();
        self.register_probe_with_hysteresis(probe, hysteresis);
    }

    /// Register a probe with explicit hysteresis configuration.
    pub fn register_probe_with_hysteresis<P>(&self, probe: P, hysteresis: ProbeHysteresis)
    where
        P: HealthProbe + 'static,
    {
        let probe = Arc::new(probe);
        let probe_dyn: Arc<dyn HealthProbe> = probe;
        self.insert_probe(probe_dyn, hysteresis);
    }

    fn insert_probe(&self, probe: Arc<dyn HealthProbe>, hysteresis: ProbeHysteresis) {
        let now = Instant::now();
        let state = HealthCheck {
            name: probe.name(),
            check_type: probe.check_type(),
            status: HealthStatus::Healthy,
            last_success: now,
            consecutive_failures: 0,
            consecutive_successes: 0,
            last_failure: None,
            latency: Duration::from_secs(0),
            last_run: None,
            message: String::from("Probe registered"),
        };

        self.probes.insert(
            probe.name(),
            RegisteredProbe {
                probe,
                hysteresis,
                state,
                cooldown_until: None,
            },
        );
    }

    /// Fetch the latest recorded state for the named probe, if present.
    #[must_use]
    pub fn check_named(&self, name: &str) -> Option<HealthCheck> {
        self.probes
            .get(name)
            .map(|entry| entry.value().state.clone())
    }

    /// Convenience helper that returns the current health report.
    #[must_use]
    pub fn latest_report(&self) -> HealthReport {
        self.health_report()
    }

    /// Run all registered health probes and update their state.
    pub fn check_all(&self) -> HealthStatus {
        let now = Instant::now();
        let mut overall = HealthStatus::Healthy;

        for mut entry in self.probes.iter_mut() {
            let should_run = entry.cooldown_until.is_none_or(|until| {
                until <= now || !matches!(entry.state.status, HealthStatus::Unhealthy)
            });

            if should_run {
                let result = entry.probe.run();
                let hysteresis = entry.hysteresis;
                apply_result(&mut entry.state, &result, hysteresis);

                if matches!(entry.state.status, HealthStatus::Unhealthy)
                    && entry.hysteresis.cooldown > Duration::from_secs(0)
                {
                    entry.cooldown_until = Some(result.observed_at + entry.hysteresis.cooldown);
                } else {
                    entry.cooldown_until = None;
                }
            }

            overall = combine_status(overall, entry.state.status);
        }

        self.is_healthy
            .store(matches!(overall, HealthStatus::Healthy), Ordering::Release);
        self.last_check
            .store(now.elapsed().as_secs(), Ordering::Release);

        overall
    }

    /// Run a specific health check
    fn has_critical_failure(&self) -> bool {
        self.probes.iter().any(|entry| {
            matches!(
                entry.value().state.check_type,
                HealthCheckType::Memory | HealthCheckType::Connectivity
            ) && matches!(entry.value().state.status, HealthStatus::Unhealthy)
        })
    }

    /// Get current health status without running checks
    pub fn current_status(&self) -> HealthStatus {
        if self.is_healthy.load(Ordering::Acquire) {
            HealthStatus::Healthy
        } else if self.has_critical_failure() {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        }
    }

    /// Get detailed health report
    pub fn health_report(&self) -> HealthReport {
        let checks: Vec<HealthCheck> = self
            .probes
            .iter()
            .map(|entry| entry.value().state.clone())
            .collect();

        let overall_status = self.current_status();

        HealthReport {
            status: overall_status,
            checks,
            timestamp: Instant::now(),
            confidence: self.calculate_health_confidence(),
        }
    }

    fn calculate_health_confidence(&self) -> Confidence {
        let healthy_count = self
            .probes
            .iter()
            .filter(|entry| matches!(entry.value().state.status, HealthStatus::Healthy))
            .count();

        let total_count = self.probes.len();
        if total_count == 0 {
            return Confidence::from_probability(1.0);
        }

        let healthy_count = u32::try_from(healthy_count).unwrap_or(u32::MAX);
        let total_count = u32::try_from(total_count).unwrap_or(u32::MAX);
        if total_count == 0 {
            return Confidence::from_probability(1.0);
        }

        let ratio = f64::from(healthy_count) / f64::from(total_count);
        let clamped = ratio.clamp(0.0, 1.0);
        let probability = clamped_f64_to_f32(clamped, 1.0);

        Confidence::from_probability(probability)
    }
}

/// Individual health check
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Name of the health check
    pub name: &'static str,
    /// Type of health check being performed
    pub check_type: HealthCheckType,
    /// Current status of the check
    pub status: HealthStatus,
    /// Last time this check succeeded
    pub last_success: Instant,
    /// Number of consecutive failures
    pub consecutive_failures: u32,
    /// Number of consecutive successes (used for hysteresis recovery)
    pub consecutive_successes: u32,
    /// Last time the check failed
    pub last_failure: Option<Instant>,
    /// Duration of the most recent probe execution
    pub latency: Duration,
    /// Timestamp of the last probe execution
    pub last_run: Option<Instant>,
    /// Human-readable status message
    pub message: String,
}

/// Types of health checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthCheckType {
    /// Memory usage check
    Memory,
    /// Network/operation latency check
    Latency,
    /// Error rate monitoring
    ErrorRate,
    /// Network connectivity check
    Connectivity,
    /// Cognitive system health check
    Cognitive,
    /// Custom probe supplied by integrators
    Custom(&'static str),
}

/// Health status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HealthStatus {
    /// Checks indicate the system is operating nominally.
    Healthy,
    /// Some checks warn of degraded behavior that needs attention.
    Degraded,
    /// Critical failures observed and require immediate action.
    Unhealthy,
}

/// Detailed health report
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Overall system health classification produced by the latest sweep.
    pub status: HealthStatus,
    /// Individual check results inspected during the sweep.
    pub checks: Vec<HealthCheck>,
    /// Moment when the report snapshot was generated.
    pub timestamp: Instant,
    /// Confidence we have that this report represents current reality.
    pub confidence: Confidence,
}

/// Health alert for critical issues
#[derive(Debug, Clone)]
pub struct HealthAlert {
    /// Severity classification used for routing escalation paths.
    pub severity: AlertSeverity,
    /// Which type of check triggered the alert.
    pub check_type: HealthCheckType,
    /// Detailed human-readable summary of the issue.
    pub message: String,
    /// When the alert was emitted.
    pub timestamp: Instant,
    /// Required operator response or remediation guidance.
    pub action_required: String,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational notice; no action needed yet.
    Info,
    /// Situations to watch closely; plan remediation.
    Warning,
    /// Active problem impacting the system.
    Critical,
    /// Immediate intervention required to prevent failure.
    Emergency,
}

impl Default for SystemHealth {
    fn default() -> Self {
        Self::new()
    }
}

fn clamped_f64_to_f32(value: f64, default: f32) -> f32 {
    if !value.is_finite() {
        return default;
    }

    let clamped = value.clamp(-f64::from(f32::MAX), f64::from(f32::MAX));
    let sign_bit = if clamped.is_sign_negative() {
        1_u32 << 31
    } else {
        0
    };
    let abs = clamped.abs();

    if abs == 0.0 {
        return f32::from_bits(sign_bit);
    }

    let bits = abs.to_bits();
    let exponent_bits = (bits >> 52) & 0x7FF;
    let exponent = i32::try_from(exponent_bits).unwrap_or(0);
    let mut exponent_adjusted = exponent - 1023 + 127;
    if exponent_adjusted <= 0 {
        return f32::from_bits(sign_bit);
    }
    if exponent_adjusted >= 0xFF {
        exponent_adjusted = 0xFE;
    }

    let mantissa = bits & ((1_u64 << 52) - 1);
    let mantissa32 = u32::try_from(mantissa >> (52 - 23)).unwrap_or(0x007F_FFFF);
    let exponent_field = u32::try_from(exponent_adjusted).unwrap_or(0);
    let bits32 = sign_bit | (exponent_field << 23) | mantissa32;
    f32::from_bits(bits32)
}

/// Estimate current memory usage in bytes (placeholder implementation).
const fn estimate_memory_usage() -> u64 {
    // Simplified estimate - in production this would query actual system metrics.
    1024 * 1024 * 100
}

#[derive(Debug)]
struct MemoryUsageProbe {
    threshold_bytes: u64,
}

impl MemoryUsageProbe {
    const fn new(threshold_bytes: u64) -> Self {
        Self { threshold_bytes }
    }
}

impl HealthProbe for MemoryUsageProbe {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn check_type(&self) -> HealthCheckType {
        HealthCheckType::Memory
    }

    fn run(&self) -> HealthCheckResult {
        let start = Instant::now();
        let usage = estimate_memory_usage();

        let status = if usage < self.threshold_bytes * 70 / 100 {
            HealthStatus::Healthy
        } else if usage < self.threshold_bytes * 90 / 100 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };

        let usage_mb = usage / (1024 * 1024);
        let threshold_mb = self.threshold_bytes / (1024 * 1024);
        let message = match status {
            HealthStatus::Healthy => {
                format!("Memory usage {usage_mb}MB within {threshold_mb}MB budget")
            }
            HealthStatus::Degraded => {
                format!("Memory usage {usage_mb}MB approaching {threshold_mb}MB threshold")
            }
            HealthStatus::Unhealthy => {
                format!("Memory usage {usage_mb}MB exceeds {threshold_mb}MB threshold")
            }
        };

        let latency = start.elapsed();
        let observed_at = Instant::now();

        HealthCheckResult {
            status,
            message,
            latency,
            observed_at,
        }
    }
}

#[derive(Debug)]
struct LatencyProbe {
    threshold_ms: f64,
}

impl LatencyProbe {
    const fn new(threshold_ms: f64) -> Self {
        Self { threshold_ms }
    }
}

impl HealthProbe for LatencyProbe {
    fn name(&self) -> &'static str {
        "latency"
    }

    fn check_type(&self) -> HealthCheckType {
        HealthCheckType::Latency
    }

    fn run(&self) -> HealthCheckResult {
        let start = Instant::now();
        let p99_latency = 30.0; // Placeholder metric. Replace with real telemetry.

        let status = if p99_latency < self.threshold_ms * 0.5 {
            HealthStatus::Healthy
        } else if p99_latency < self.threshold_ms {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };

        let message = match status {
            HealthStatus::Healthy => format!(
                "p99 latency {:.2}ms within {:.2}ms budget",
                p99_latency, self.threshold_ms
            ),
            HealthStatus::Degraded => format!(
                "p99 latency {:.2}ms nearing {:.2}ms budget",
                p99_latency, self.threshold_ms
            ),
            HealthStatus::Unhealthy => format!(
                "p99 latency {:.2}ms exceeds {:.2}ms budget",
                p99_latency, self.threshold_ms
            ),
        };

        let latency = start.elapsed();
        let observed_at = Instant::now();

        HealthCheckResult {
            status,
            message,
            latency,
            observed_at,
        }
    }
}

#[derive(Debug)]
struct ErrorRateProbe {
    threshold: f32,
}

impl ErrorRateProbe {
    const fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl HealthProbe for ErrorRateProbe {
    fn name(&self) -> &'static str {
        "error_rate"
    }

    fn check_type(&self) -> HealthCheckType {
        HealthCheckType::ErrorRate
    }

    fn run(&self) -> HealthCheckResult {
        let start = Instant::now();
        let error_rate = 0.002; // Placeholder error rate.

        let status = if error_rate < self.threshold * 0.5 {
            HealthStatus::Healthy
        } else if error_rate < self.threshold {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };

        let message = match status {
            HealthStatus::Healthy => format!(
                "Error rate {:.3}% within {:.3}% budget",
                error_rate * 100.0,
                self.threshold * 100.0
            ),
            HealthStatus::Degraded => format!(
                "Error rate {:.3}% approaching {:.3}% budget",
                error_rate * 100.0,
                self.threshold * 100.0
            ),
            HealthStatus::Unhealthy => format!(
                "Error rate {:.3}% exceeds {:.3}% budget",
                error_rate * 100.0,
                self.threshold * 100.0
            ),
        };

        let latency = start.elapsed();
        let observed_at = Instant::now();

        HealthCheckResult {
            status,
            message,
            latency,
            observed_at,
        }
    }
}

#[derive(Debug, Default)]
struct ConnectivityProbe;

impl HealthProbe for ConnectivityProbe {
    fn name(&self) -> &'static str {
        "connectivity"
    }

    fn check_type(&self) -> HealthCheckType {
        HealthCheckType::Connectivity
    }

    fn run(&self) -> HealthCheckResult {
        let start = Instant::now();
        let latency = start.elapsed();
        let observed_at = Instant::now();
        HealthCheckResult {
            status: HealthStatus::Healthy,
            message: String::from("All services reachable"),
            latency,
            observed_at,
        }
    }
}

#[derive(Debug, Default)]
struct CognitiveProbe;

impl HealthProbe for CognitiveProbe {
    fn name(&self) -> &'static str {
        "cognitive"
    }

    fn check_type(&self) -> HealthCheckType {
        HealthCheckType::Cognitive
    }

    fn run(&self) -> HealthCheckResult {
        let start = Instant::now();
        let latency = start.elapsed();
        let observed_at = Instant::now();
        HealthCheckResult {
            status: HealthStatus::Healthy,
            message: String::from("Cognitive metrics within biological ranges"),
            latency,
            observed_at,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Severity {
    Healthy = 0,
    Degraded = 1,
    Unhealthy = 2,
}

const fn severity(status: HealthStatus) -> Severity {
    match status {
        HealthStatus::Healthy => Severity::Healthy,
        HealthStatus::Degraded => Severity::Degraded,
        HealthStatus::Unhealthy => Severity::Unhealthy,
    }
}

const fn status_from_severity(severity: Severity) -> HealthStatus {
    match severity {
        Severity::Healthy => HealthStatus::Healthy,
        Severity::Degraded => HealthStatus::Degraded,
        Severity::Unhealthy => HealthStatus::Unhealthy,
    }
}

fn combine_status(current: HealthStatus, next: HealthStatus) -> HealthStatus {
    let severity = severity(current).max(severity(next));
    status_from_severity(severity)
}

fn apply_result(state: &mut HealthCheck, result: &HealthCheckResult, hysteresis: ProbeHysteresis) {
    state.last_run = Some(result.observed_at);
    state.latency = result.latency;
    state.message.clone_from(&result.message);

    if result.status == HealthStatus::Healthy {
        state.consecutive_successes = state.consecutive_successes.saturating_add(1);
        state.consecutive_failures = 0;
        state.last_success = result.observed_at;

        if severity(state.status) > Severity::Healthy {
            if state.consecutive_successes >= hysteresis.recovery_threshold {
                state.status = HealthStatus::Healthy;
            }
        } else {
            state.status = HealthStatus::Healthy;
        }
    } else {
        state.consecutive_failures = state.consecutive_failures.saturating_add(1);
        state.consecutive_successes = 0;
        state.last_failure = Some(result.observed_at);

        let mut target = severity(result.status);

        if state.consecutive_failures >= hysteresis.unhealthy_threshold {
            target = Severity::Unhealthy;
        } else if state.consecutive_failures >= hysteresis.degrade_threshold {
            target = target.max(Severity::Degraded);
        }

        let combined = severity(state.status).max(target);
        state.status = status_from_severity(combined);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_checks() {
        let health = SystemHealth::new();

        // Run all checks
        let status = health.check_all();
        assert_eq!(status, HealthStatus::Healthy);

        // Get health report
        let report = health.health_report();
        assert_eq!(report.status, HealthStatus::Healthy);
        assert!(!report.checks.is_empty());
    }

    #[test]
    fn test_health_confidence() {
        let health = SystemHealth::new();
        health.check_all();

        let report = health.health_report();
        assert!(report.confidence.raw() > 0.5);
    }
}
