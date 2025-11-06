# Task: Rate Limiting Enforcement

## Objective
Implement rate limiting for API requests based on API key configuration, protecting the system from abuse while allowing legitimate usage.

## Context
The `ApiKey` struct already includes rate limit configuration. We need to enforce these limits in both HTTP and gRPC servers using the governor crate.

## Requirements

### 1. Add Rate Limiter to Auth State
Extend auth infrastructure:
```rust
use governor::{Quota, RateLimiter, state::keyed::DashMapStateStore};
use std::num::NonZeroU32;

pub struct RateLimitManager {
    /// Keyed rate limiter for per-API-key limits
    limiter: Arc<RateLimiter<String, DashMapStateStore<String>, governor::clock::DefaultClock>>,
    
    /// Global rate limiter (optional)
    global_limiter: Option<Arc<RateLimiter<governor::state::direct::NotKeyed, 
                                           governor::state::InMemoryState, 
                                           governor::clock::DefaultClock>>>,
}

impl RateLimitManager {
    pub fn new(global_rps: Option<u32>) -> Self {
        // Create global limiter if configured
        let global_limiter = global_rps.map(|rps| {
            let quota = Quota::per_second(NonZeroU32::new(rps).unwrap());
            Arc::new(RateLimiter::direct(quota))
        });
        
        // Keyed limiter uses dynamic quotas
        let limiter = Arc::new(RateLimiter::dashmap(Quota::per_second(
            NonZeroU32::new(100).unwrap() // Default quota
        )));
        
        Self {
            limiter,
            global_limiter,
        }
    }
    
    /// Check rate limit for a specific API key
    pub async fn check_limit(
        &self,
        key_id: &str,
        rate_limit: &RateLimit,
    ) -> Result<(), RateLimitError> {
        // Check global limit first
        if let Some(global) = &self.global_limiter {
            global.check()
                .map_err(|_| RateLimitError::GlobalLimitExceeded)?;
        }
        
        // Check per-key limit
        let quota = Quota::per_second(
            NonZeroU32::new(rate_limit.requests_per_second).unwrap()
        ).allow_burst(
            NonZeroU32::new(rate_limit.burst_size).unwrap()
        );
        
        // Update quota if changed
        self.limiter.check_key_n(&key_id.to_string(), 1)
            .map_err(|_| RateLimitError::KeyLimitExceeded {
                key_id: key_id.to_string(),
                limit: rate_limit.requests_per_second,
            })?;
        
        Ok(())
    }
}
```

### 2. Add Rate Limiting Middleware
Create middleware for HTTP:
```rust
pub async fn rate_limit_middleware(
    State(state): State<ApiState>,
    Extension(auth): Extension<AuthContext>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    // Skip if rate limiting disabled
    if !state.auth_config.rate_limiting {
        return Ok(next.run(request).await);
    }
    
    // Extract key ID from auth context
    let key_id = match &auth.principal {
        Principal::ApiKey(id) => id,
        _ => return Ok(next.run(request).await), // Skip for non-API key auth
    };
    
    // Check rate limit
    state.rate_limiter
        .check_limit(key_id, &auth.rate_limit)
        .await
        .map_err(|e| match e {
            RateLimitError::KeyLimitExceeded { key_id, limit } => {
                ApiError::TooManyRequests(format!(
                    "Rate limit exceeded for key {}. Limit: {} req/s",
                    key_id, limit
                ))
            }
            RateLimitError::GlobalLimitExceeded => {
                ApiError::TooManyRequests("Global rate limit exceeded".to_string())
            }
        })?;
    
    // Add rate limit headers to response
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    
    // Add standard rate limit headers
    headers.insert("X-RateLimit-Limit", auth.rate_limit.requests_per_second.to_string().parse()?);
    headers.insert("X-RateLimit-Remaining", "...".parse()?); // Calculate from governor
    headers.insert("X-RateLimit-Reset", "...".parse()?); // Calculate reset time
    
    Ok(response)
}
```

### 3. gRPC Rate Limiting
Add to gRPC interceptor:
```rust
impl AuthInterceptor {
    async fn check_rate_limit(
        &self,
        auth: &AuthContext,
    ) -> Result<(), Status> {
        if !self.rate_limiting_enabled {
            return Ok(());
        }
        
        let key_id = match &auth.principal {
            Principal::ApiKey(id) => id,
            _ => return Ok(()),
        };
        
        self.rate_limiter
            .check_limit(key_id, &auth.rate_limit)
            .await
            .map_err(|e| match e {
                RateLimitError::KeyLimitExceeded { key_id, limit } => {
                    Status::resource_exhausted(format!(
                        "Rate limit exceeded. Limit: {} req/s",
                        limit
                    ))
                }
                RateLimitError::GlobalLimitExceeded => {
                    Status::resource_exhausted("Global rate limit exceeded")
                }
            })?;
        
        Ok(())
    }
}
```

### 4. Add Rate Limit Metrics
Track rate limiting metrics:
```rust
pub struct RateLimitMetrics {
    /// Counter for rate limit hits
    pub rate_limit_exceeded: CounterVec,
    
    /// Histogram of request rates by key
    pub request_rate: HistogramVec,
    
    /// Gauge of current rate by key
    pub current_rate: GaugeVec,
}

impl RateLimitManager {
    pub fn record_request(&self, key_id: &str, allowed: bool) {
        if allowed {
            self.metrics.request_rate
                .with_label_values(&[key_id])
                .observe(1.0);
        } else {
            self.metrics.rate_limit_exceeded
                .with_label_values(&[key_id])
                .inc();
        }
    }
}
```

### 5. Configuration Options
Add rate limiting config:
```toml
[security.rate_limiting]
# Enable rate limiting
enabled = true

# Global rate limit (requests per second)
global_limit = 10000

# Default per-key limit if not specified
default_key_limit = 100

# Cleanup interval for expired rate limit entries
cleanup_interval_seconds = 300

# Rate limit response headers
include_headers = true
```

### 6. Advanced Features
```rust
/// Adaptive rate limiting based on system load
pub struct AdaptiveRateLimiter {
    base_limiter: RateLimitManager,
    cpu_threshold: f32,
    memory_threshold: f32,
    reduction_factor: f32,
}

impl AdaptiveRateLimiter {
    pub async fn check_limit_adaptive(
        &self,
        key_id: &str,
        rate_limit: &mut RateLimit,
    ) -> Result<(), RateLimitError> {
        // Get system metrics
        let cpu_usage = get_cpu_usage();
        let memory_usage = get_memory_usage();
        
        // Reduce limits if system under pressure
        if cpu_usage > self.cpu_threshold || memory_usage > self.memory_threshold {
            rate_limit.requests_per_second = 
                (rate_limit.requests_per_second as f32 * self.reduction_factor) as u32;
        }
        
        self.base_limiter.check_limit(key_id, rate_limit).await
    }
}
```

### 7. Client Retry Logic
Add retry guidance to error responses:
```rust
impl IntoResponse for RateLimitError {
    fn into_response(self) -> Response {
        let retry_after = calculate_retry_after(&self);
        
        let mut response = (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({
                "error": {
                    "code": 429,
                    "message": self.to_string(),
                    "type": "rate_limit_error",
                    "retry_after": retry_after,
                }
            }))
        ).into_response();
        
        response.headers_mut().insert(
            "Retry-After",
            retry_after.to_string().parse().unwrap()
        );
        
        response
    }
}
```

## Testing Requirements
1. Test rate limiting with various configurations
2. Test burst handling
3. Test global vs per-key limits
4. Test rate limit headers
5. Load test to verify limits
6. Test cleanup of old entries
7. Test adaptive rate limiting

## Performance Considerations
- Use efficient data structures (DashMap)
- Periodic cleanup of old entries
- Minimize lock contention
- Consider using sliding window algorithm

## Acceptance Criteria
1. Rate limits enforced per API key
2. Global rate limits work correctly
3. Proper 429 responses with retry information
4. Rate limit headers included
5. Metrics track rate limit hits
6. Performance impact < 0.5ms per request
7. Configurable via config file