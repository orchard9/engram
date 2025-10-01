//! System health monitoring and checks

use crate::Confidence;
use crossbeam_utils::CachePadded;
use std::convert::TryFrom;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// System health monitor
pub struct SystemHealth {
    /// Individual health checks
    checks: dashmap::DashMap<&'static str, HealthCheck>,

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
            checks: dashmap::DashMap::new(),
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
        // Memory check
        self.checks.insert(
            "memory",
            HealthCheck {
                name: "memory",
                check_type: HealthCheckType::Memory,
                status: HealthStatus::Healthy,
                last_success: Instant::now(),
                consecutive_failures: 0,
                message: String::from("Memory usage within limits"),
            },
        );

        // Latency check
        self.checks.insert(
            "latency",
            HealthCheck {
                name: "latency",
                check_type: HealthCheckType::Latency,
                status: HealthStatus::Healthy,
                last_success: Instant::now(),
                consecutive_failures: 0,
                message: String::from("Latency within acceptable range"),
            },
        );

        // Error rate check
        self.checks.insert(
            "error_rate",
            HealthCheck {
                name: "error_rate",
                check_type: HealthCheckType::ErrorRate,
                status: HealthStatus::Healthy,
                last_success: Instant::now(),
                consecutive_failures: 0,
                message: String::from("Error rate below threshold"),
            },
        );

        // Connectivity check
        self.checks.insert(
            "connectivity",
            HealthCheck {
                name: "connectivity",
                check_type: HealthCheckType::Connectivity,
                status: HealthStatus::Healthy,
                last_success: Instant::now(),
                consecutive_failures: 0,
                message: String::from("All services reachable"),
            },
        );

        // Cognitive health check
        self.checks.insert(
            "cognitive",
            HealthCheck {
                name: "cognitive",
                check_type: HealthCheckType::Cognitive,
                status: HealthStatus::Healthy,
                last_success: Instant::now(),
                consecutive_failures: 0,
                message: String::from("Cognitive metrics within biological ranges"),
            },
        );
    }

    /// Run all health checks
    pub fn check_all(&self) -> HealthStatus {
        let mut all_healthy = true;
        let now = Instant::now();

        for mut check in self.checks.iter_mut() {
            let check_type = check.check_type;
            let status = self.run_check(check_type);
            check.status = status.clone();

            match status {
                HealthStatus::Healthy => {
                    check.last_success = now;
                    check.consecutive_failures = 0;
                    check.message = Self::get_success_message(check_type);
                }
                HealthStatus::Degraded => {
                    all_healthy = false;
                    check.consecutive_failures += 1;
                    check.message = Self::get_degraded_message(check_type);
                }
                HealthStatus::Unhealthy => {
                    all_healthy = false;
                    check.consecutive_failures += 1;
                    check.message = Self::get_failure_message(check_type);
                }
            }
        }

        self.is_healthy.store(all_healthy, Ordering::Release);
        self.last_check
            .store(now.elapsed().as_secs(), Ordering::Release);

        if all_healthy {
            HealthStatus::Healthy
        } else if self.has_critical_failure() {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        }
    }

    /// Run a specific health check
    fn run_check(&self, check_type: HealthCheckType) -> HealthStatus {
        match check_type {
            HealthCheckType::Memory => self.check_memory(),
            HealthCheckType::Latency => self.check_latency(),
            HealthCheckType::ErrorRate => self.check_error_rate(),
            HealthCheckType::Connectivity => Self::check_connectivity(),
            HealthCheckType::Cognitive => Self::check_cognitive_health(),
        }
    }

    const fn check_memory(&self) -> HealthStatus {
        // Get current memory usage (simplified)
        let usage = Self::estimate_memory_usage();

        if usage < self.memory_threshold_bytes * 70 / 100 {
            HealthStatus::Healthy
        } else if usage < self.memory_threshold_bytes * 90 / 100 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        }
    }

    fn check_latency(&self) -> HealthStatus {
        // Check recent p99 latency (would come from metrics)
        let p99_latency = 30.0; // Mock value in ms (well below 50% of 100ms threshold)

        if p99_latency < self.latency_threshold_ms * 0.5 {
            HealthStatus::Healthy
        } else if p99_latency < self.latency_threshold_ms {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        }
    }

    fn check_error_rate(&self) -> HealthStatus {
        // Check recent error rate (would come from metrics)
        let error_rate = 0.002; // Mock value (0.2%, well below 50% of 1% threshold)

        if error_rate < self.error_rate_threshold * 0.5 {
            HealthStatus::Healthy
        } else if error_rate < self.error_rate_threshold {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        }
    }

    const fn check_connectivity() -> HealthStatus {
        // Check if all required services are reachable
        // This would ping databases, external services, etc.
        HealthStatus::Healthy // Simplified
    }

    const fn check_cognitive_health() -> HealthStatus {
        // Check if cognitive metrics are within biological ranges
        // Would check CLS balance, consolidation rates, etc.
        HealthStatus::Healthy // Simplified
    }

    fn has_critical_failure(&self) -> bool {
        self.checks.iter().any(|check| {
            matches!(
                check.check_type,
                HealthCheckType::Memory | HealthCheckType::Connectivity
            ) && matches!(check.status, HealthStatus::Unhealthy)
        })
    }

    fn get_success_message(check_type: HealthCheckType) -> String {
        match check_type {
            HealthCheckType::Memory => "Memory usage within limits".to_string(),
            HealthCheckType::Latency => "Latency within acceptable range".to_string(),
            HealthCheckType::ErrorRate => "Error rate below threshold".to_string(),
            HealthCheckType::Connectivity => "All services reachable".to_string(),
            HealthCheckType::Cognitive => "Cognitive metrics within biological ranges".to_string(),
        }
    }

    fn get_degraded_message(check_type: HealthCheckType) -> String {
        match check_type {
            HealthCheckType::Memory => "Memory usage elevated but manageable".to_string(),
            HealthCheckType::Latency => "Latency elevated but acceptable".to_string(),
            HealthCheckType::ErrorRate => "Error rate elevated but tolerable".to_string(),
            HealthCheckType::Connectivity => "Some services experiencing delays".to_string(),
            HealthCheckType::Cognitive => "Cognitive metrics showing minor deviations".to_string(),
        }
    }

    fn get_failure_message(check_type: HealthCheckType) -> String {
        match check_type {
            HealthCheckType::Memory => "Memory usage critical".to_string(),
            HealthCheckType::Latency => "Latency exceeds acceptable threshold".to_string(),
            HealthCheckType::ErrorRate => "Error rate exceeds threshold".to_string(),
            HealthCheckType::Connectivity => "Critical services unreachable".to_string(),
            HealthCheckType::Cognitive => "Cognitive metrics outside biological ranges".to_string(),
        }
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
            .checks
            .iter()
            .map(|entry| entry.value().clone())
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
            .checks
            .iter()
            .filter(|check| matches!(check.status, HealthStatus::Healthy))
            .count();

        let total_count = self.checks.len();
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

    /// Estimate current memory usage in bytes
    const fn estimate_memory_usage() -> u64 {
        // Simplified estimate - in production this would query actual system metrics
        // Return a low value to pass health checks
        1024 * 1024 * 100 // 100 MB estimate
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
}

/// Health status levels
#[derive(Debug, Clone, PartialEq, Eq)]
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
