# Task: Integration Testing with Auth Modes

## Objective
Create comprehensive integration tests that verify authentication and authorization work correctly across all API endpoints in both authenticated and unauthenticated modes.

## Context
We need to ensure that authentication doesn't break existing functionality and that security boundaries are properly enforced. Tests should cover HTTP, gRPC, and streaming endpoints.

## Requirements

### 1. Test Infrastructure Setup
Create test helpers in `engram-core/tests/auth_helpers.rs`:
```rust
use engram_cli::{ApiState, create_api_routes};
use axum::Router;
use tower::ServiceExt;

pub struct AuthTestContext {
    pub api_state: ApiState,
    pub valid_key: String,
    pub read_only_key: String,
    pub expired_key: String,
    pub wrong_space_key: String,
}

impl AuthTestContext {
    pub async fn new() -> Result<Self> {
        // Create test key storage
        let temp_dir = tempfile::tempdir()?;
        let key_store = SqliteApiKeyStore::new(
            temp_dir.path().join("test_keys.db")
        ).await?;
        
        // Create test keys
        let validator = ApiKeyValidator::new(Arc::new(key_store));
        
        // Full access key
        let full_access = validator.generate_key(GenerateKeyRequest {
            name: "test-full-access".into(),
            allowed_spaces: vec![MemorySpaceId::default()],
            permissions: vec![
                Permission::MemoryRead,
                Permission::MemoryWrite,
                Permission::MemoryDelete,
            ],
            rate_limit: RateLimit::unlimited(),
            expires_at: None,
        }).await?;
        
        // Read-only key
        let read_only = validator.generate_key(GenerateKeyRequest {
            name: "test-read-only".into(),
            allowed_spaces: vec![MemorySpaceId::default()],
            permissions: vec![Permission::MemoryRead],
            rate_limit: RateLimit::default(),
            expires_at: None,
        }).await?;
        
        // Expired key
        let expired = validator.generate_key(GenerateKeyRequest {
            name: "test-expired".into(),
            allowed_spaces: vec![MemorySpaceId::default()],
            permissions: vec![Permission::MemoryRead],
            rate_limit: RateLimit::default(),
            expires_at: Some(Utc::now() - Duration::days(1)),
        }).await?;
        
        // Wrong space key
        let wrong_space = validator.generate_key(GenerateKeyRequest {
            name: "test-wrong-space".into(),
            allowed_spaces: vec![MemorySpaceId::try_from("other-space")?],
            permissions: vec![Permission::MemoryRead],
            rate_limit: RateLimit::default(),
            expires_at: None,
        }).await?;
        
        // Create API state with auth enabled
        let api_state = create_test_api_state(validator).await?;
        
        Ok(Self {
            api_state,
            valid_key: full_access.full_key,
            read_only_key: read_only.full_key,
            expired_key: expired.full_key,
            wrong_space_key: wrong_space.full_key,
        })
    }
    
    pub fn app(&self) -> Router {
        create_api_routes(&self.api_state.auth_config)
            .with_state(self.api_state.clone())
    }
}
```

### 2. HTTP Authentication Tests
Create `engram-core/tests/http_auth_test.rs`:
```rust
#[tokio::test]
async fn test_auth_required_endpoints() {
    let ctx = AuthTestContext::new().await.unwrap();
    let app = ctx.app();
    
    // Test endpoints that require auth
    let auth_required = vec![
        ("/api/v1/memories/remember", Method::POST),
        ("/api/v1/memories/recall", Method::GET),
        ("/api/v1/memories/1234", Method::DELETE),
        ("/api/v1/spaces", Method::POST),
        ("/shutdown", Method::POST),
    ];
    
    for (path, method) in auth_required {
        // Request without auth should fail
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        
        // Request with valid auth should pass auth check
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .header("Authorization", format!("Bearer {}", ctx.valid_key))
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap();
        
        // May fail for other reasons, but not auth
        assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    }
}

#[tokio::test]
async fn test_public_endpoints() {
    let ctx = AuthTestContext::new().await.unwrap();
    let app = ctx.app();
    
    // Public endpoints should work without auth
    let public_endpoints = vec![
        "/health",
        "/metrics",
        "/api/v1/system/health",
    ];
    
    for endpoint in public_endpoints {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(endpoint)
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn test_permission_enforcement() {
    let ctx = AuthTestContext::new().await.unwrap();
    let app = ctx.app();
    
    // Try to write with read-only key
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/v1/memories/remember")
                .header("Authorization", format!("Bearer {}", ctx.read_only_key))
                .header("Content-Type", "application/json")
                .body(Body::from(json!({
                    "content": "test memory",
                    "tags": []
                }).to_string()))
                .unwrap()
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    
    // Read should work with read-only key
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/memories/recall?q=test")
                .header("Authorization", format!("Bearer {}", ctx.read_only_key))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
}
```

### 3. gRPC Authentication Tests
Create `engram-core/tests/grpc_auth_test.rs`:
```rust
#[tokio::test]
async fn test_grpc_auth_enforcement() {
    let ctx = AuthTestContext::new().await.unwrap();
    
    // Start test gRPC server
    let addr = start_test_grpc_server(&ctx).await;
    
    // Test without auth
    let channel = Channel::from_shared(format!("http://{}", addr))
        .unwrap()
        .connect()
        .await
        .unwrap();
    
    let mut client = MemoryServiceClient::new(channel);
    
    let response = client
        .store(StoreRequest {
            content: "test".into(),
            space: Some("default".into()),
            ..Default::default()
        })
        .await;
    
    assert_eq!(
        response.unwrap_err().code(),
        Code::Unauthenticated
    );
    
    // Test with valid auth
    let channel = Channel::from_shared(format!("http://{}", addr))
        .unwrap()
        .connect()
        .await
        .unwrap();
    
    let mut client = MemoryServiceClient::with_interceptor(
        channel,
        move |mut req: Request<()>| {
            req.metadata_mut().insert(
                "authorization",
                format!("Bearer {}", ctx.valid_key).parse().unwrap()
            );
            Ok(req)
        }
    );
    
    let response = client
        .store(StoreRequest {
            content: "test".into(),
            space: Some("default".into()),
            ..Default::default()
        })
        .await;
    
    assert!(response.is_ok());
}
```

### 4. Rate Limiting Tests
```rust
#[tokio::test]
async fn test_rate_limiting() {
    let mut ctx = AuthTestContext::new().await.unwrap();
    
    // Create key with low rate limit
    let limited_key = ctx.validator.generate_key(GenerateKeyRequest {
        name: "rate-limited".into(),
        allowed_spaces: vec![MemorySpaceId::default()],
        permissions: vec![Permission::MemoryRead],
        rate_limit: RateLimit {
            requests_per_second: 2,
            burst_size: 2,
        },
        expires_at: None,
    }).await.unwrap();
    
    let app = ctx.app();
    
    // Make requests up to limit
    for i in 0..2 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/system/health")
                    .header("Authorization", format!("Bearer {}", limited_key.full_key))
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::OK);
    }
    
    // Next request should be rate limited
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/system/health")
                .header("Authorization", format!("Bearer {}", limited_key.full_key))
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    
    // Check rate limit headers
    assert!(response.headers().contains_key("X-RateLimit-Limit"));
    assert!(response.headers().contains_key("Retry-After"));
}
```

### 5. Space Access Control Tests
```rust
#[tokio::test]
async fn test_space_access_control() {
    let ctx = AuthTestContext::new().await.unwrap();
    let app = ctx.app();
    
    // Try to access wrong space
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/memories/recall?q=test")
                .header("Authorization", format!("Bearer {}", ctx.wrong_space_key))
                .header("X-Memory-Space-Id", "default")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    
    let body: Value = serde_json::from_slice(&body::to_bytes(response).await.unwrap()).unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("Access denied to space"));
}
```

### 6. Key Rotation Tests
```rust
#[tokio::test]
async fn test_key_rotation_grace_period() {
    let ctx = AuthTestContext::new().await.unwrap();
    
    // Create and rotate key
    let original = ctx.validator.generate_key(GenerateKeyRequest {
        name: "rotating-key".into(),
        // ... permissions
    }).await.unwrap();
    
    // Schedule rotation
    let rotation_manager = KeyRotationManager::new(ctx.key_store.clone());
    let plan = rotation_manager.schedule_rotation(
        &original.key_id,
        RotationSchedule {
            activate_at: Utc::now() - Duration::minutes(1),
            deprecate_at: Utc::now() + Duration::days(1),
            expire_at: Utc::now() + Duration::days(7),
            // ...
        }
    ).await.unwrap();
    
    // Execute rotation to activation phase
    rotation_manager.execute_rotation_step(&original.key_id).await.unwrap();
    
    // Both old and new keys should work
    let app = ctx.app();
    
    // Old key (deprecated but valid)
    let response = app.clone().oneshot(/* request with old key */).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    
    // New key
    let response = app.clone().oneshot(/* request with new key */).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
```

### 7. No-Auth Mode Tests
```rust
#[tokio::test]
async fn test_no_auth_mode() {
    // Create context with auth disabled
    let config = SecurityConfig {
        auth_mode: AuthMode::None,
        rate_limiting: false,
        // ...
    };
    
    let api_state = create_test_api_state_with_config(config).await.unwrap();
    let app = create_api_routes(&api_state.auth_config)
        .with_state(api_state);
    
    // All endpoints should work without auth
    let endpoints = vec![
        ("/api/v1/memories/remember", Method::POST),
        ("/api/v1/memories/recall", Method::GET),
        ("/api/v1/spaces", Method::POST),
    ];
    
    for (path, method) in endpoints {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .body(Body::empty())
                    .unwrap()
            )
            .await
            .unwrap();
        
        // Should not require auth
        assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
```

## Testing Strategy
1. Unit test each auth component
2. Integration test full request flows
3. Test both success and failure paths
4. Test edge cases (expired keys, wrong permissions)
5. Load test with concurrent auth requests
6. Test auth mode transitions

## Acceptance Criteria
1. All endpoints properly protected
2. Public endpoints remain accessible
3. Permission checks enforced
4. Rate limiting works correctly
5. Space access control validated
6. Key rotation maintains service
7. No-auth mode bypasses all checks
8. Tests pass in CI/CD pipeline