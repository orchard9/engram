# Task: HTTP Authentication Middleware

## Objective
Create Axum middleware that enforces API key authentication based on configuration, integrating the existing `ApiKeyValidator` with the HTTP server.

## Context
The HTTP server in `engram-cli/src/api.rs` currently has no authentication. We need to add middleware that can be selectively applied to routes based on the security configuration.

## Requirements

### 1. Create Auth State
Add authentication state to `ApiState`:
```rust
pub struct ApiState {
    // ... existing fields ...
    
    /// Authentication configuration
    pub auth_config: Arc<SecurityConfig>,
    
    /// API key validator (None if auth disabled)
    pub auth_validator: Option<Arc<ApiKeyValidator>>,
}
```

### 2. Implement Auth Middleware
Create middleware function in `engram-cli/src/auth/middleware.rs`:
```rust
use axum::{
    middleware::Next,
    extract::{Request, State},
    response::{Response, IntoResponse},
    http::{StatusCode, header},
};

pub async fn require_api_key(
    State(state): State<ApiState>,
    mut request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    // Skip auth if disabled
    let validator = match &state.auth_validator {
        Some(v) => v,
        None => return Ok(next.run(request).await),
    };
    
    // Extract Authorization header
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .ok_or_else(|| ApiError::Unauthorized("Missing Authorization header".into()))?;
    
    // Validate API key
    let auth_context = validator
        .validate(auth_header)
        .await
        .map_err(|e| match e {
            AuthError::InvalidApiKey => ApiError::Unauthorized("Invalid API key".into()),
            AuthError::ExpiredKey => ApiError::Unauthorized("API key expired".into()),
            _ => ApiError::SystemError(e.to_string()),
        })?;
    
    // Check space access if specified
    if let Some(space_header) = request.headers().get("X-Memory-Space-Id") {
        let space_id = MemorySpaceId::try_from(
            space_header.to_str()?
        )?;
        
        if !auth_context.allowed_spaces.contains(&space_id) {
            return Err(ApiError::Forbidden(
                format!("Access denied to space: {}", space_id)
            ));
        }
    }
    
    // Add auth context to request extensions
    request.extensions_mut().insert(auth_context);
    
    // Continue to next handler
    Ok(next.run(request).await)
}
```

### 3. Create Permission Middleware
For fine-grained access control:
```rust
pub fn require_permission(permission: Permission) -> impl Fn(Request, Next) -> impl Future {
    move |request: Request, next: Next| async move {
        // Extract auth context from extensions
        let auth_context = request
            .extensions()
            .get::<AuthContext>()
            .ok_or_else(|| ApiError::Unauthorized("No auth context".into()))?;
        
        // Check permission
        if !auth_context.permissions.contains(&permission) {
            return Err(ApiError::Forbidden(
                format!("Missing required permission: {:?}", permission)
            ));
        }
        
        Ok(next.run(request).await)
    }
}
```

### 4. Apply Middleware to Routes
Update `create_api_routes()`:
```rust
pub fn create_api_routes(auth_config: &SecurityConfig) -> Router<ApiState> {
    // Create auth layer if enabled
    let auth_layer = match auth_config.auth_mode {
        AuthMode::None => None,
        AuthMode::ApiKey => Some(middleware::from_fn(require_api_key)),
    };
    
    // Protected routes (require authentication)
    let protected_routes = Router::new()
        .route("/api/v1/memories/remember", post(remember_memory))
        .route("/api/v1/memories/recall", get(recall_memories))
        .route("/api/v1/memories/{id}", delete(delete_memory_by_id))
        .route("/api/v1/spaces", post(create_memory_space))
        .route("/shutdown", post(shutdown_server)
            .layer(middleware::from_fn(require_permission(Permission::SystemShutdown)))
        );
    
    // Apply auth layer if configured
    let protected_routes = if let Some(layer) = auth_layer {
        protected_routes.layer(layer)
    } else {
        protected_routes
    };
    
    // Public routes (no auth required)
    let public_routes = Router::new()
        .route("/health", get(simple_health))
        .route("/metrics", get(metrics_snapshot))
        .route("/api/v1/system/health", get(system_health));
    
    // Combine routes
    Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .merge(swagger_router)
}
```

### 5. Add Auth Context to Handlers
Update handlers to use auth context:
```rust
pub async fn remember_memory(
    State(state): State<ApiState>,
    Extension(auth): Extension<AuthContext>,  // Auth context from middleware
    headers: HeaderMap,
    Json(body): Json<StoreMemoryRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // Check if user has write permission for the space
    let space_id = extract_memory_space_id(...)?;
    
    if !auth.allowed_spaces.contains(&space_id) {
        return Err(ApiError::Forbidden("Access denied to space".into()));
    }
    
    // Existing handler logic...
}
```

### 6. Error Responses
Standardize auth error responses:
```rust
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg),
            ApiError::Forbidden(msg) => (StatusCode::FORBIDDEN, msg),
            // ... other cases
        };
        
        let body = json!({
            "error": {
                "code": status.as_u16(),
                "message": message,
                "type": "authentication_error"
            }
        });
        
        (status, Json(body)).into_response()
    }
}
```

### 7. Add Security Headers
Add security headers middleware:
```rust
pub async fn security_headers(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    
    response
}
```

## Testing Requirements
1. Test auth enforcement with valid/invalid keys
2. Test permission checking
3. Test space access control
4. Test auth bypass when disabled
5. Test error responses
6. Integration tests with full request flow

## Performance Considerations
- Cache validated auth contexts
- Minimize database lookups
- Add metrics for auth performance

## Acceptance Criteria
1. Middleware correctly validates API keys when enabled
2. Unauthorized requests return 401
3. Forbidden requests return 403
4. Auth can be disabled via config
5. Protected routes require auth, public routes don't
6. Auth context available to handlers
7. < 1ms overhead for cached validations