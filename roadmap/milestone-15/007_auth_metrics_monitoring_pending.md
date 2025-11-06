# Task: Authentication Metrics and Monitoring

## Objective
Add comprehensive metrics for authentication, authorization, and rate limiting to enable security monitoring and incident detection.

## Context
Security teams need visibility into authentication patterns, failed attempts, and potential security incidents. This requires detailed metrics and logging.

## Requirements

### 1. Define Security Metrics
Create metrics in `engram-core/src/metrics/security.rs`:
```rust
use prometheus::{Counter, CounterVec, Histogram, HistogramVec, Gauge, GaugeVec};

pub struct SecurityMetrics {
    /// Authentication attempts by result
    pub auth_attempts: CounterVec,
    
    /// Authentication latency
    pub auth_latency: HistogramVec,
    
    /// Active API keys
    pub active_api_keys: Gauge,
    
    /// API keys by status
    pub api_key_status: GaugeVec,
    
    /// Failed auth attempts by reason
    pub auth_failures: CounterVec,
    
    /// Rate limit hits by key
    pub rate_limit_hits: CounterVec,
    
    /// Permission denials by type
    pub permission_denials: CounterVec,
    
    /// Suspicious activity score
    pub security_score: Gauge,
    
    /// Key usage by space
    pub key_usage_by_space: CounterVec,
    
    /// Concurrent sessions per key
    pub concurrent_sessions: GaugeVec,
}

impl SecurityMetrics {
    pub fn new(registry: &Registry) -> Result<Self> {
        let auth_attempts = register_counter_vec!(
            "engram_auth_attempts_total",
            "Total authentication attempts",
            &["result", "method"]
        )?;
        
        let auth_latency = register_histogram_vec!(
            "engram_auth_latency_seconds",
            "Authentication latency in seconds",
            &["method"],
            exponential_buckets(0.0001, 2.0, 10)?
        )?;
        
        let auth_failures = register_counter_vec!(
            "engram_auth_failures_total",
            "Authentication failures by reason",
            &["reason", "source_ip"]
        )?;
        
        // ... register other metrics
        
        Ok(Self { /* fields */ })
    }
}
```

### 2. Authentication Event Logging
Create structured logging for security events:
```rust
#[derive(Serialize)]
pub struct AuthEvent {
    #[serde(rename = "type")]
    pub event_type: AuthEventType,
    pub timestamp: DateTime<Utc>,
    pub key_id: Option<String>,
    pub source_ip: String,
    pub user_agent: Option<String>,
    pub result: AuthResult,
    pub latency_ms: f64,
    pub memory_space: Option<String>,
    pub metadata: HashMap<String, Value>,
}

#[derive(Serialize)]
pub enum AuthEventType {
    ApiKeyValidation,
    PermissionCheck,
    RateLimitCheck,
    KeyCreated,
    KeyRotated,
    KeyRevoked,
    SuspiciousActivity,
}

pub fn log_auth_event(event: AuthEvent) {
    // Log as structured JSON for SIEM ingestion
    info!(
        target: "engram::security",
        event_type = %event.event_type,
        key_id = ?event.key_id,
        source_ip = %event.source_ip,
        result = ?event.result,
        latency_ms = event.latency_ms,
        "{}",
        serde_json::to_string(&event).unwrap()
    );
}
```

### 3. Update Middleware with Metrics
Add metrics collection to auth middleware:
```rust
pub async fn require_api_key_with_metrics(
    State(state): State<ApiState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let start = Instant::now();
    let source_ip = addr.ip().to_string();
    
    // Extract user agent
    let user_agent = headers
        .get(header::USER_AGENT)
        .and_then(|h| h.to_str().ok())
        .map(String::from);
    
    // Validate API key
    let auth_result = match validate_api_key(&state, &headers).await {
        Ok(context) => {
            // Record success
            state.metrics.security.auth_attempts
                .with_label_values(&["success", "api_key"])
                .inc();
            
            request.extensions_mut().insert(context.clone());
            Ok(context)
        }
        Err(e) => {
            // Record failure
            state.metrics.security.auth_attempts
                .with_label_values(&["failure", "api_key"])
                .inc();
            
            state.metrics.security.auth_failures
                .with_label_values(&[&e.to_string(), &source_ip])
                .inc();
            
            Err(e)
        }
    };
    
    // Log auth event
    let latency = start.elapsed();
    log_auth_event(AuthEvent {
        event_type: AuthEventType::ApiKeyValidation,
        timestamp: Utc::now(),
        key_id: auth_result.as_ref().ok().and_then(|c| {
            match &c.principal {
                Principal::ApiKey(id) => Some(id.clone()),
                _ => None,
            }
        }),
        source_ip,
        user_agent,
        result: auth_result.as_ref().map(|_| AuthResult::Success)
            .unwrap_or(AuthResult::Failure),
        latency_ms: latency.as_secs_f64() * 1000.0,
        memory_space: None,
        metadata: HashMap::new(),
    });
    
    // Record latency
    state.metrics.security.auth_latency
        .with_label_values(&["api_key"])
        .observe(latency.as_secs_f64());
    
    auth_result?;
    Ok(next.run(request).await)
}
```

### 4. Anomaly Detection
Implement basic anomaly detection:
```rust
pub struct SecurityMonitor {
    /// Track failed attempts per IP
    failed_attempts: Arc<DashMap<String, Vec<DateTime<Utc>>>>,
    
    /// Track key usage patterns
    key_usage: Arc<DashMap<String, KeyUsagePattern>>,
    
    /// Configuration
    config: SecurityMonitorConfig,
}

#[derive(Clone)]
pub struct SecurityMonitorConfig {
    /// Max failed attempts before flagging
    pub max_failed_attempts: usize,
    
    /// Time window for failed attempts
    pub failed_attempt_window: Duration,
    
    /// Unusual usage threshold
    pub usage_spike_threshold: f32,
}

impl SecurityMonitor {
    pub async fn check_suspicious_activity(
        &self,
        source_ip: &str,
        key_id: Option<&str>,
    ) -> Option<SecurityAlert> {
        // Check failed attempts from IP
        if let Some(attempts) = self.failed_attempts.get(source_ip) {
            let recent_attempts = attempts
                .iter()
                .filter(|&t| Utc::now() - *t < self.config.failed_attempt_window)
                .count();
            
            if recent_attempts > self.config.max_failed_attempts {
                return Some(SecurityAlert::BruteForceAttempt {
                    source_ip: source_ip.to_string(),
                    attempt_count: recent_attempts,
                });
            }
        }
        
        // Check unusual key usage patterns
        if let Some(key_id) = key_id {
            if let Some(pattern) = self.key_usage.get(key_id) {
                if pattern.is_anomalous(self.config.usage_spike_threshold) {
                    return Some(SecurityAlert::UnusualUsagePattern {
                        key_id: key_id.to_string(),
                        details: pattern.describe_anomaly(),
                    });
                }
            }
        }
        
        None
    }
}
```

### 5. Security Dashboard Queries
Provide Grafana dashboard queries:
```yaml
# Authentication success rate
rate(engram_auth_attempts_total{result="success"}[5m]) / 
rate(engram_auth_attempts_total[5m]) * 100

# Failed auth attempts by IP (top 10)
topk(10, sum by (source_ip) (
  rate(engram_auth_failures_total[5m])
))

# Rate limit hits by key
sum by (key_id) (
  rate(engram_rate_limit_hits_total[5m])
)

# API key usage heatmap
sum by (key_id, memory_space) (
  rate(engram_key_usage_by_space_total[5m])
)

# Authentication latency P99
histogram_quantile(0.99, 
  rate(engram_auth_latency_seconds_bucket[5m])
)
```

### 6. Alerting Rules
Define Prometheus alerting rules:
```yaml
groups:
  - name: security
    interval: 30s
    rules:
      - alert: HighAuthFailureRate
        expr: |
          rate(engram_auth_failures_total[5m]) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High authentication failure rate"
          description: "{{ $value }} auth failures per second"
      
      - alert: BruteForceAttempt
        expr: |
          sum by (source_ip) (
            rate(engram_auth_failures_total[1m])
          ) > 20
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Possible brute force attempt from {{ $labels.source_ip }}"
      
      - alert: ApiKeyAbuse
        expr: |
          sum by (key_id) (
            rate(engram_rate_limit_hits_total[5m])
          ) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API key {{ $labels.key_id }} hitting rate limits"
```

### 7. Security Report Generation
```rust
pub async fn generate_security_report(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> SecurityReport {
    SecurityReport {
        period: (start, end),
        total_requests: get_total_requests(start, end).await,
        auth_success_rate: calculate_success_rate(start, end).await,
        top_api_keys: get_top_keys_by_usage(start, end, 10).await,
        failed_attempts: get_failed_attempts_summary(start, end).await,
        rate_limit_violations: get_rate_limit_summary(start, end).await,
        suspicious_activities: get_security_alerts(start, end).await,
        recommendations: generate_recommendations().await,
    }
}
```

## Testing Requirements
1. Test metric collection accuracy
2. Test anomaly detection thresholds
3. Test alert generation
4. Test performance impact
5. Test log format compatibility
6. Load test metric collection

## Integration Requirements
- Export metrics to Prometheus
- Send logs to SIEM systems
- Integrate with alerting systems
- Support custom dashboards

## Acceptance Criteria
1. All auth events logged with structure
2. Metrics exported to Prometheus
3. Anomaly detection identifies patterns
4. Alerts fire correctly
5. Dashboard queries work
6. < 1ms overhead for metric collection
7. Security reports generated on demand