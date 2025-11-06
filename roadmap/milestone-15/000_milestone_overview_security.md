# Milestone 15: API Security and Authentication

## Overview
Implement API key authentication for Engram with support for both authenticated and unauthenticated modes, enabling production deployments with proper access control.

## Goals
1. **Enable authentication middleware** for HTTP and gRPC APIs
2. **Support dual modes**: no-auth (development) and auth (production)
3. **Implement API key management** CLI commands
4. **Add rate limiting** enforcement
5. **Secure key storage** with rotation support
6. **Per-space access control** with fine-grained permissions

## Design Principles
- **Backward compatible** - existing deployments work without auth
- **Secure by default** - production configs require auth
- **Easy key rotation** - support key versioning and graceful transitions
- **Minimal overhead** - cache validation results for performance
- **Audit trail** - log all authentication events

## Technical Approach

### 1. Configuration Schema
```toml
[security]
# Authentication mode: "none", "api_key", "jwt"
auth_mode = "api_key"

# Enable rate limiting
rate_limiting = true

# API key storage backend
[security.api_keys]
# Storage backend: "file", "vault", "aws_secrets_manager"
backend = "file"
storage_path = "./data/api_keys.db"

# Key rotation settings
rotation_days = 90
warn_before_expiry_days = 14
```

### 2. Authentication Modes
- **none**: No authentication (dev/test environments)
- **api_key**: Bearer token authentication with `engram_key_*` format
- **jwt**: JWT token validation (future enhancement)

### 3. API Key Format
```
Authorization: Bearer engram_key_{key_id}_{secret}
```
- **key_id**: 24-character alphanumeric identifier
- **secret**: 32-character cryptographic secret
- Hashed with Argon2id for storage

### 4. Permissions Model
```rust
pub enum Permission {
    // Memory operations
    MemoryRead,
    MemoryWrite,
    MemoryDelete,
    
    // Space management
    SpaceCreate,
    SpaceDelete,
    SpaceList,
    
    // System operations
    SystemHealth,
    SystemMetrics,
    SystemShutdown,
    
    // Admin operations
    AdminKeyManagement,
    AdminSpaceAccess,
}
```

### 5. CLI Commands
```bash
# Generate new API key
engram auth create-key --name "production-app" \
  --spaces "default,analytics" \
  --permissions "MemoryRead,MemoryWrite" \
  --expires-in "90d"

# List API keys
engram auth list-keys

# Rotate API key
engram auth rotate-key --key-id "abc123" --grace-period "7d"

# Revoke API key
engram auth revoke-key --key-id "abc123"

# Check key status
engram auth check-key --key-id "abc123"
```

### 6. Middleware Integration

#### HTTP (Axum)
```rust
// Create auth layer based on config
let auth_layer = match config.security.auth_mode {
    AuthMode::None => None,
    AuthMode::ApiKey => Some(
        axum::middleware::from_fn_with_state(
            auth_state.clone(),
            auth::require_api_key,
        )
    ),
};

// Apply to routes that need protection
let protected_routes = Router::new()
    .route("/api/v1/memories/remember", post(remember_memory))
    .route("/api/v1/memories/recall", get(recall_memories))
    .layer(auth_layer);

// Public routes (health, metrics)
let public_routes = Router::new()
    .route("/health", get(health_check))
    .route("/metrics", get(metrics_endpoint));
```

#### gRPC (Tonic)
```rust
// Create interceptor
let auth_interceptor = match config.security.auth_mode {
    AuthMode::None => None,
    AuthMode::ApiKey => Some(auth::grpc_auth_interceptor),
};

// Apply to service
let service = MemoryServiceServer::with_interceptor(
    memory_service,
    auth_interceptor,
);
```

### 7. Key Storage Backend

#### File-based (SQLite)
```sql
CREATE TABLE api_keys (
    key_id TEXT PRIMARY KEY,
    secret_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    permissions TEXT NOT NULL, -- JSON array
    allowed_spaces TEXT NOT NULL, -- JSON array
    rate_limit_rps INTEGER,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP,
    last_used TIMESTAMP,
    revoked_at TIMESTAMP
);

CREATE INDEX idx_expires_at ON api_keys(expires_at);
CREATE INDEX idx_revoked_at ON api_keys(revoked_at);
```

### 8. Rate Limiting
```rust
// Per-key rate limits
pub struct RateLimit {
    requests_per_second: u32,
    burst_size: u32,
}

// Enforce using governor crate
let governor = Arc::new(
    RateLimiter::keyed(quota)
        .with_middleware(auth_context.key_id)
);
```

### 9. Security Best Practices
1. **Never log secrets** - use secrecy crate for handling
2. **Constant-time comparison** for secret validation
3. **Audit all auth events** - success and failures
4. **Grace period for rotation** - support old key during transition
5. **Secure headers** - add security headers to responses
6. **CORS configuration** - restrict origins in production

### 10. Monitoring and Alerts
- Track authentication failures
- Monitor key usage patterns
- Alert on keys nearing expiration
- Detect potential brute force attempts
- Log all administrative actions

## Implementation Tasks
1. Add security configuration to CLI config
2. Implement file-based API key storage
3. Create auth middleware for HTTP
4. Create auth interceptor for gRPC
5. Add CLI commands for key management
6. Implement rate limiting enforcement
7. Add authentication metrics
8. Create key rotation workflow
9. Add security documentation
10. Integration testing with auth modes
11. Performance testing with caching
12. Security audit and penetration testing

## Success Criteria
- Zero-downtime key rotation
- < 1ms auth check latency (cached)
- Support 10K+ API keys
- Graceful fallback for auth failures
- Complete audit trail
- No breaking changes for existing deployments