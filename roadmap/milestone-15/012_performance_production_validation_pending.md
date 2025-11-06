# Task: Performance and Production Validation

## Objective
Validate that authentication and security features meet performance requirements and are ready for production deployment at scale.

## Context
Security features must not significantly impact performance. We need to benchmark authentication overhead, validate caching effectiveness, and ensure the system can handle production load.

## Requirements

### 1. Performance Benchmarks
Create benchmarking suite:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_authentication(c: &mut Criterion) {
    let mut group = c.benchmark_group("authentication");
    
    // Setup test environment
    let rt = tokio::runtime::Runtime::new().unwrap();
    let ctx = rt.block_on(AuthTestContext::new()).unwrap();
    
    // Benchmark: Key validation (cold cache)
    group.bench_function("validate_key_cold", |b| {
        b.to_async(&rt).iter(|| async {
            ctx.validator.cache.clear();
            ctx.validator.validate(black_box(&ctx.valid_key)).await
        })
    });
    
    // Benchmark: Key validation (warm cache)
    group.bench_function("validate_key_cached", |b| {
        b.to_async(&rt).iter(|| async {
            ctx.validator.validate(black_box(&ctx.valid_key)).await
        })
    });
    
    // Benchmark: Full request with auth
    group.bench_function("request_with_auth", |b| {
        let app = ctx.app();
        b.to_async(&rt).iter(|| async {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/api/v1/system/health")
                        .header("Authorization", format!("Bearer {}", ctx.valid_key))
                        .body(Body::empty())
                        .unwrap()
                )
                .await
                .unwrap();
            black_box(response);
        })
    });
    
    // Benchmark: Request without auth (baseline)
    group.bench_function("request_no_auth", |b| {
        let app_no_auth = create_app_without_auth();
        b.to_async(&rt).iter(|| async {
            let response = app_no_auth
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/api/v1/system/health")
                        .body(Body::empty())
                        .unwrap()
                )
                .await
                .unwrap();
            black_box(response);
        })
    });
    
    group.finish();
}
```

### 2. Load Testing
Create load test scenarios:
```rust
use goose::prelude::*;

async fn authenticated_recall(user: &mut GooseUser) -> TransactionResult {
    let api_key = user.get_session_data::<String>("api_key")?;
    
    let request = user
        .get("/api/v1/memories/recall?q=test")
        .await?
        .header("Authorization", &format!("Bearer {}", api_key));
    
    let response = user.request(request).await?;
    
    if !response.status().is_success() {
        return user.set_failure(
            &format!("Recall failed: {}", response.status()),
            &mut response.request,
            None,
            None
        );
    }
    
    Ok(())
}

async fn rate_limited_requests(user: &mut GooseUser) -> TransactionResult {
    // Test rate limit behavior under load
    for _ in 0..5 {
        let _ = authenticated_recall(user).await?;
    }
    Ok(())
}

pub fn create_load_test() -> GooseAttack {
    GooseAttack::initialize()?
        .register_scenario(
            scenario!("AuthenticatedLoad")
                .set_weight(90)?
                .register_transaction(
                    transaction!(authenticated_recall)
                        .set_name("recall")
                        .set_weight(70)?
                )
                .register_transaction(
                    transaction!(authenticated_store)
                        .set_name("store")
                        .set_weight(30)?
                )
        )
        .register_scenario(
            scenario!("RateLimitTest")
                .set_weight(10)?
                .register_transaction(transaction!(rate_limited_requests))
        )
        .set_default(GooseDefault::Host, "http://localhost:9090")?
        .set_default(GooseDefault::Users, 1000)?
        .set_default(GooseDefault::HatchRate, "50")?
        .set_default(GooseDefault::RunTime, "5m")?
}
```

### 3. Cache Effectiveness Testing
```rust
pub async fn measure_cache_performance() -> CacheMetrics {
    let validator = create_test_validator().await;
    let keys = generate_test_keys(1000).await;
    
    let mut metrics = CacheMetrics::default();
    
    // Warm up cache
    for key in &keys[..100] {
        validator.validate(key).await.unwrap();
    }
    
    // Measure hit rate
    for _ in 0..10000 {
        let key = if rand::random::<f32>() < 0.8 {
            // 80% of requests use common keys
            &keys[rand::random::<usize>() % 100]
        } else {
            // 20% use random keys
            &keys[100 + rand::random::<usize>() % 900]
        };
        
        let start = Instant::now();
        let _ = validator.validate(key).await;
        let duration = start.elapsed();
        
        if duration < Duration::from_micros(100) {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }
        
        metrics.total_time += duration;
    }
    
    metrics.calculate_stats();
    metrics
}
```

### 4. Production Simulation
```yaml
# docker-compose.prod-test.yml
version: '3.8'

services:
  engram:
    image: engram:latest
    environment:
      - ENGRAM_AUTH_MODE=api_key
      - ENGRAM_RATE_LIMITING=true
    volumes:
      - ./data:/data
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
    
  load-generator:
    image: grafana/k6:latest
    volumes:
      - ./k6-scripts:/scripts
    command: run -u 1000 -d 30m /scripts/auth-load.js
    
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
    volumes:
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
```

### 5. Monitoring Dashboard
Create Grafana dashboard JSON:
```json
{
  "dashboard": {
    "title": "Engram Security Performance",
    "panels": [
      {
        "title": "Authentication Latency",
        "targets": [{
          "expr": "histogram_quantile(0.99, rate(engram_auth_latency_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [{
          "expr": "rate(engram_auth_cache_hits_total[5m]) / rate(engram_auth_attempts_total[5m]) * 100"
        }]
      },
      {
        "title": "Rate Limit Violations",
        "targets": [{
          "expr": "sum(rate(engram_rate_limit_hits_total[1m])) by (key_id)"
        }]
      },
      {
        "title": "Auth Overhead",
        "targets": [{
          "expr": "engram_request_duration_seconds{auth=\"true\"} - engram_request_duration_seconds{auth=\"false\"}"
        }]
      }
    ]
  }
}
```

### 6. Performance Requirements Validation
```rust
#[tokio::test]
async fn test_performance_requirements() {
    let metrics = run_performance_tests().await;
    
    // Requirement: Auth adds < 1ms latency
    assert!(
        metrics.auth_overhead_p99 < Duration::from_millis(1),
        "Auth overhead {} exceeds 1ms requirement",
        metrics.auth_overhead_p99.as_secs_f64() * 1000.0
    );
    
    // Requirement: Cache hit rate > 95%
    assert!(
        metrics.cache_hit_rate > 0.95,
        "Cache hit rate {} below 95% requirement",
        metrics.cache_hit_rate * 100.0
    );
    
    // Requirement: Support 10K+ API keys
    assert!(
        metrics.total_keys_tested >= 10000,
        "Failed to test with 10K+ keys"
    );
    
    // Requirement: 10K+ requests per second
    assert!(
        metrics.requests_per_second > 10000.0,
        "RPS {} below 10K requirement",
        metrics.requests_per_second
    );
}
```

### 7. Production Readiness Checklist
```rust
pub async fn validate_production_readiness() -> ValidationReport {
    let mut report = ValidationReport::new();
    
    // Check: Database indexes exist
    report.check(
        "Database indexes",
        validate_db_indexes().await
    );
    
    // Check: Connection pooling configured
    report.check(
        "Connection pooling",
        validate_connection_pools().await
    );
    
    // Check: Monitoring endpoints accessible
    report.check(
        "Monitoring endpoints",
        validate_monitoring().await
    );
    
    // Check: Log rotation configured
    report.check(
        "Log rotation",
        validate_log_rotation().await
    );
    
    // Check: Backup procedures tested
    report.check(
        "Backup procedures",
        validate_backup_restore().await
    );
    
    // Check: Graceful shutdown works
    report.check(
        "Graceful shutdown",
        validate_graceful_shutdown().await
    );
    
    report
}
```

### 8. Scalability Testing
```bash
#!/bin/bash
# Scale test script

# Start with 1 instance
docker-compose up -d --scale engram=1

# Run baseline test
echo "Testing with 1 instance..."
k6 run --vus 100 --duration 5m auth-load.js > results-1-instance.json

# Scale to 3 instances
docker-compose up -d --scale engram=3

# Test horizontal scaling
echo "Testing with 3 instances..."
k6 run --vus 300 --duration 5m auth-load.js > results-3-instances.json

# Scale to 5 instances
docker-compose up -d --scale engram=5

echo "Testing with 5 instances..."
k6 run --vus 500 --duration 5m auth-load.js > results-5-instances.json

# Generate comparison report
python3 compare_results.py results-*.json > scaling-report.html
```

## Success Metrics
- Authentication latency P99 < 1ms
- Cache hit rate > 95%
- Support 10K+ concurrent connections
- 100K+ requests per second
- Zero memory leaks over 24h
- Graceful degradation under load

## Deliverables
1. Performance benchmark results
2. Load test reports
3. Monitoring dashboard
4. Production configuration guide
5. Capacity planning document
6. Scaling recommendations

## Acceptance Criteria
1. All performance requirements met
2. No memory leaks detected
3. Scales horizontally
4. Monitoring fully functional
5. Production checklist passed
6. 24-hour stability test passed
7. Documentation complete