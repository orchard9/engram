# Task 012: Production Integration and Monitoring

## Objective
Integrate spreading engine with production monitoring and error recovery systems.

## Priority
P1 (Production Readiness)

## Effort Estimate
1 day

## Dependencies
- Task 011: Comprehensive Spreading Validation

## Technical Approach

### Implementation Details
- Add spreading metrics to existing Prometheus integration
- Implement spreading health checks and circuit breakers
- Create spreading performance dashboards
- Add automatic spreading parameter tuning based on workload characteristics

### Files to Create/Modify
- `engram-core/src/activation/metrics.rs` - Spreading-specific metrics
- `engram-core/src/activation/health_checks.rs` - Health monitoring
- `engram-core/src/activation/circuit_breaker.rs` - Error recovery
- `engram-core/src/activation/auto_tuning.rs` - Parameter optimization

### Integration Points
- Extends existing Prometheus metrics from production monitoring
- Integrates with health check infrastructure
- Uses error recovery patterns from existing codebase
- Connects to configuration management system

## Implementation Details

### Spreading Metrics
```rust
pub struct SpreadingMetrics {
    // Latency metrics
    spreading_latency: HistogramVec,
    hop_latency: HistogramVec,
    tier_access_latency: HistogramVec,

    // Throughput metrics
    spreads_per_second: CounterVec,
    activations_processed: CounterVec,
    confidence_aggregations: CounterVec,

    // Quality metrics
    average_hop_count: GaugeVec,
    cycle_detection_rate: CounterVec,
    confidence_distribution: HistogramVec,

    // Resource metrics
    memory_pool_utilization: Gauge,
    cache_hit_rate: GaugeVec,
    concurrent_spreads: Gauge,
}

impl SpreadingMetrics {
    pub fn record_spreading_latency(&self, latency: Duration, tier: StorageTier) {
        self.spreading_latency
            .with_label_values(&[tier.as_str()])
            .observe(latency.as_secs_f64());
    }

    pub fn record_spreading_result(&self, result: &SpreadingResult) {
        self.average_hop_count
            .with_label_values(&["successful"])
            .set(result.average_hop_count as f64);

        if result.cycles_detected > 0 {
            self.cycle_detection_rate
                .with_label_values(&["detected"])
                .inc_by(result.cycles_detected as u64);
        }

        for confidence in &result.confidence_scores {
            self.confidence_distribution
                .with_label_values(&["final"])
                .observe(*confidence as f64);
        }
    }
}
```

### Health Checks
```rust
pub struct SpreadingHealthChecker {
    last_successful_spread: AtomicInstant,
    consecutive_failures: AtomicUsize,
    health_threshold: Duration,
    failure_threshold: usize,
}

impl HealthChecker for SpreadingHealthChecker {
    async fn check_health(&self) -> HealthStatus {
        // Test basic spreading functionality
        let test_result = self.run_health_check_spread().await;

        match test_result {
            Ok(_) => {
                self.last_successful_spread.store(Instant::now(), Ordering::Relaxed);
                self.consecutive_failures.store(0, Ordering::Relaxed);
                HealthStatus::Healthy
            }
            Err(e) => {
                let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
                if failures >= self.failure_threshold {
                    HealthStatus::Unhealthy(format!("Spreading failures: {}", e))
                } else {
                    HealthStatus::Degraded(format!("Spreading errors: {}", e))
                }
            }
        }
    }

    async fn run_health_check_spread(&self) -> Result<(), SpreadingError> {
        // Create minimal test graph
        let test_graph = create_health_check_graph();
        let spreading_engine = create_test_spreading_engine();

        // Perform basic spreading operation
        let result = spreading_engine
            .spread_from_source("health_check_source", max_hops: 2)
            .await?;

        // Validate basic spreading properties
        if result.total_activations == 0 {
            return Err(SpreadingError::NoActivations);
        }

        if result.max_latency > Duration::from_millis(50) {
            return Err(SpreadingError::LatencyExceeded);
        }

        Ok(())
    }
}
```

### Circuit Breaker
```rust
pub struct SpreadingCircuitBreaker {
    state: AtomicU8, // 0=Closed, 1=Open, 2=HalfOpen
    failure_count: AtomicUsize,
    last_failure_time: AtomicInstant,
    failure_threshold: usize,
    timeout: Duration,
    half_open_max_calls: usize,
    half_open_calls: AtomicUsize,
}

impl SpreadingCircuitBreaker {
    pub async fn execute_with_circuit_breaker<F, T>(
        &self,
        operation: F,
    ) -> Result<T, CircuitBreakerError>
    where
        F: Future<Output = Result<T, SpreadingError>>,
    {
        match self.get_state() {
            CircuitState::Closed => {
                match operation.await {
                    Ok(result) => {
                        self.on_success();
                        Ok(result)
                    }
                    Err(e) => {
                        self.on_failure();
                        Err(CircuitBreakerError::OperationFailed(e))
                    }
                }
            }
            CircuitState::Open => {
                if self.should_attempt_reset() {
                    self.transition_to_half_open();
                    // Retry once in half-open state
                    self.execute_with_circuit_breaker(operation).await
                } else {
                    Err(CircuitBreakerError::CircuitOpen)
                }
            }
            CircuitState::HalfOpen => {
                if self.half_open_calls.load(Ordering::Relaxed) >= self.half_open_max_calls {
                    return Err(CircuitBreakerError::HalfOpenLimitExceeded);
                }

                self.half_open_calls.fetch_add(1, Ordering::Relaxed);
                match operation.await {
                    Ok(result) => {
                        self.transition_to_closed();
                        Ok(result)
                    }
                    Err(e) => {
                        self.transition_to_open();
                        Err(CircuitBreakerError::OperationFailed(e))
                    }
                }
            }
        }
    }
}
```

### Automatic Parameter Tuning
```rust
pub struct SpreadingAutoTuner {
    current_params: RwLock<SpreadingParameters>,
    performance_history: RingBuffer<PerformanceSample>,
    tuning_strategy: TuningStrategy,
    last_tuning: Instant,
    tuning_interval: Duration,
}

impl SpreadingAutoTuner {
    pub async fn tune_parameters(&self, workload_stats: &WorkloadStatistics) {
        if self.last_tuning.elapsed() < self.tuning_interval {
            return;
        }

        let current_performance = self.measure_current_performance().await;
        let recommended_params = self.tuning_strategy
            .recommend_parameters(workload_stats, &current_performance);

        if self.should_apply_recommendations(&recommended_params) {
            let mut params = self.current_params.write().await;
            *params = recommended_params;

            info!(
                "Auto-tuned spreading parameters: batch_size={}, hop_limit={}",
                params.batch_size, params.max_hop_count
            );
        }
    }

    fn should_apply_recommendations(&self, new_params: &SpreadingParameters) -> bool {
        // Only apply if significant improvement expected
        let improvement_threshold = 0.10; // 10% improvement

        let predicted_improvement = self.tuning_strategy
            .predict_improvement(new_params);

        predicted_improvement > improvement_threshold
    }
}

pub enum TuningStrategy {
    Conservative,  // Small parameter changes
    Aggressive,    // Large parameter changes for maximum performance
    Adaptive,      // Adjust strategy based on workload stability
}
```

### Performance Dashboard Configuration
```yaml
# Grafana dashboard configuration for spreading metrics
dashboard:
  title: "Engram Spreading Performance"
  panels:
    - title: "Spreading Latency P95"
      metric: "engram_spreading_latency"
      percentile: 0.95
      alert_threshold: 10ms

    - title: "Throughput by Tier"
      metric: "engram_spreads_per_second"
      group_by: ["tier"]

    - title: "Memory Pool Utilization"
      metric: "engram_memory_pool_utilization"
      alert_threshold: 90%

    - title: "Cycle Detection Rate"
      metric: "engram_cycle_detection_rate"
      rate: "5m"

    - title: "Circuit Breaker Status"
      metric: "engram_circuit_breaker_state"
      legend: ["Closed", "Open", "HalfOpen"]
```

## Acceptance Criteria
- [ ] Comprehensive metrics available for all spreading operations
- [ ] Health checks detect spreading failures within 30 seconds
- [ ] Circuit breaker prevents cascade failures during spreading errors
- [ ] Performance dashboards provide real-time spreading visibility
- [ ] Auto-tuning improves performance over baseline by >10%
- [ ] Alerting configured for critical spreading performance degradation
- [ ] Production deployment ready with monitoring integration

## Testing Approach
- Unit tests for all monitoring components
- Integration tests with existing monitoring infrastructure
- Chaos testing to validate circuit breaker behavior
- Load testing to validate metrics accuracy under stress
- Auto-tuning validation with various workload patterns

## Risk Mitigation
- **Risk**: Monitoring overhead impacts spreading performance
- **Mitigation**: Async metrics collection, sampling for high-frequency metrics
- **Testing**: Performance benchmarks with/without monitoring

- **Risk**: Circuit breaker too aggressive, blocking valid operations
- **Mitigation**: Conservative thresholds, gradual tuning based on production data
- **Monitoring**: Track circuit breaker state transitions and false positives

- **Risk**: Auto-tuning causes performance oscillations
- **Mitigation**: Damping factors, minimum tuning intervals, rollback capability
- **Validation**: Stability testing with auto-tuning enabled

## Implementation Strategy

### Phase 1: Basic Monitoring
- Implement core spreading metrics
- Add basic health checks
- Integration with existing monitoring

### Phase 2: Advanced Reliability
- Implement circuit breaker pattern
- Add comprehensive alerting
- Performance dashboard creation

### Phase 3: Intelligent Operations
- Implement auto-tuning system
- Add predictive monitoring
- Production deployment validation

## Alert Configuration
```yaml
alerts:
  - name: "spreading_latency_high"
    condition: "p95(engram_spreading_latency) > 10ms"
    duration: "2m"
    severity: "warning"

  - name: "spreading_failures"
    condition: "rate(engram_spreading_failures[5m]) > 0.1"
    duration: "1m"
    severity: "critical"

  - name: "circuit_breaker_open"
    condition: "engram_circuit_breaker_state == 1"
    duration: "30s"
    severity: "critical"
```

## Notes
This task ensures that the cognitive database is production-ready with comprehensive observability. Unlike traditional databases with well-understood performance patterns, cognitive spreading introduces novel behaviors that require specialized monitoring and recovery mechanisms. The quality of this monitoring directly impacts operational confidence in the cognitive database.