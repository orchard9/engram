//! HTTP authentication middleware for Axum router.
//!
//! Provides middleware functions for:
//! - API key validation
//! - Permission checking
//! - Security headers
//!
//! Middleware is conditionally applied based on security configuration.

use axum::{
    extract::{Request, State},
    http::{HeaderValue, header},
    middleware::Next,
    response::Response,
};

use crate::api::{ApiError, ApiState};

#[cfg(feature = "security")]
use engram_core::auth::{AuthContext, AuthError, Permission};

/// Authentication middleware that validates API keys
///
/// This middleware:
/// - Skips validation if auth is disabled (validator is None)
/// - Extracts and validates Authorization header
/// - Checks space access if X-Memory-Space-Id header present
/// - Inserts AuthContext into request extensions
///
/// # Performance
/// - < 1ms for cached validations
/// - Uses Arc for zero-cost state sharing
///
/// # Note
/// This middleware uses State<ApiState> extractor, which works with route_layer()
/// before .with_state() is called on the router.
#[cfg(feature = "security")]
pub async fn require_api_key(
    State(state): State<ApiState>,
    mut request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    // Skip auth if disabled
    let Some(validator) = &state.auth_validator else {
        return Ok(next.run(request).await);
    };

    // Extract Authorization header
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .ok_or_else(|| ApiError::Unauthorized("Missing Authorization header".to_string()))?;

    // Validate API key
    let auth_context = validator.validate(auth_header).await.map_err(|e| match e {
        AuthError::InvalidApiKey => {
            ApiError::Unauthorized("Invalid API key format or value".to_string())
        }
        AuthError::ExpiredKey => ApiError::Unauthorized("API key has expired".to_string()),
        AuthError::UnknownKey => ApiError::Unauthorized("API key not found".to_string()),
        AuthError::RevokedToken => ApiError::Unauthorized("API key has been revoked".to_string()),
        _ => ApiError::SystemError(format!("Authentication error: {e}")),
    })?;

    // Check space access if X-Memory-Space-Id header is present
    if let Some(space_header) = request.headers().get("X-Memory-Space-Id") {
        let space_str = space_header
            .to_str()
            .map_err(|_| ApiError::InvalidInput("Invalid X-Memory-Space-Id header".to_string()))?;

        let space_id = engram_core::MemorySpaceId::try_from(space_str)
            .map_err(|e| ApiError::InvalidInput(format!("Invalid space ID: {e}")))?;

        if !auth_context.allowed_spaces.contains(&space_id) {
            return Err(ApiError::Forbidden(format!(
                "Access denied to memory space: {}",
                space_id
            )));
        }
    }

    // Add auth context to request extensions for downstream handlers
    request.extensions_mut().insert(auth_context);

    // Continue to next handler
    Ok(next.run(request).await)
}

/// Stub version when security feature is disabled
#[cfg(not(feature = "security"))]
pub async fn require_api_key(request: Request, next: Next) -> Result<Response, ApiError> {
    Ok(next.run(request).await)
}

/// Permission-checking middleware factory
///
/// Returns a middleware function that checks for a specific permission
/// in the AuthContext. Must be used after `require_api_key` middleware.
///
/// # Example
/// ```ignore
/// .route("/shutdown", post(shutdown_server)
///     .layer(middleware::from_fn(require_permission(Permission::SystemShutdown)))
/// )
/// ```
#[cfg(feature = "security")]
pub fn require_permission(
    permission: Permission,
) -> impl Fn(
    Request,
    Next,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Response, ApiError>> + Send>>
+ Clone
+ Send
+ 'static {
    move |request: Request, next: Next| {
        let perm = permission.clone();
        Box::pin(async move {
            // Extract auth context from extensions (added by require_api_key)
            let auth_context = request.extensions().get::<AuthContext>().ok_or_else(|| {
                ApiError::Unauthorized("No authentication context found".to_string())
            })?;

            // Check if user has the required permission
            if !auth_context.permissions.contains(&perm) {
                return Err(ApiError::Forbidden(format!(
                    "Missing required permission: {:?}",
                    perm
                )));
            }

            Ok(next.run(request).await)
        })
    }
}

/// Stub version when security feature is disabled
#[cfg(not(feature = "security"))]
#[must_use]
pub fn require_permission(
    _permission: (),
) -> impl Fn(
    Request,
    Next,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Response, ApiError>> + Send>>
+ Clone
+ Send
+ 'static {
    move |request: Request, next: Next| Box::pin(async move { Ok(next.run(request).await) })
}

/// Security headers middleware
///
/// Adds standard security headers to all responses:
/// - X-Content-Type-Options: nosniff
/// - X-Frame-Options: DENY
/// - X-XSS-Protection: 1; mode=block
///
/// These headers provide defense-in-depth against common web vulnerabilities.
pub async fn security_headers(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    // Prevent MIME type sniffing
    headers.insert(
        "X-Content-Type-Options",
        HeaderValue::from_static("nosniff"),
    );

    // Prevent rendering in frames (clickjacking protection)
    headers.insert("X-Frame-Options", HeaderValue::from_static("DENY"));

    // Enable XSS filter in older browsers
    headers.insert(
        "X-XSS-Protection",
        HeaderValue::from_static("1; mode=block"),
    );

    response
}

// Note: Unit tests for security_headers removed as Next::new() doesn't exist in Axum 0.8.4.
// Integration tests in engram-cli/tests/auth_middleware_tests.rs provide comprehensive coverage.
