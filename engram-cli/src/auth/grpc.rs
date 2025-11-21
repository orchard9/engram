//! gRPC authentication interceptor for Tonic.
//!
//! Provides gRPC-specific authentication middleware that validates API keys
//! from metadata headers, mirroring the HTTP middleware behavior.

use std::sync::Arc;
use tonic::{Request, Status, service::Interceptor};

#[cfg(feature = "security")]
use engram_core::auth::{AuthError, api_key::ApiKeyValidator};

use crate::config::AuthMode;

/// gRPC authentication interceptor
///
/// This interceptor:
/// - Skips validation if auth is disabled (AuthMode::None)
/// - Extracts and validates authorization from gRPC metadata
/// - Validates using existing ApiKeyValidator from Task 002
/// - Handles async validation using tokio::task::block_in_place
/// - Adds AuthContext to request extensions for downstream handlers
///
/// # Performance
/// - Uses tokio::task::block_in_place for async-in-sync bridge
/// - < 1ms for cached validations
/// - Uses Arc for zero-cost validator sharing
#[cfg(feature = "security")]
#[derive(Clone)]
pub struct AuthInterceptor {
    validator: Arc<ApiKeyValidator>,
    auth_mode: AuthMode,
}

#[cfg(feature = "security")]
impl AuthInterceptor {
    /// Create new gRPC auth interceptor
    #[must_use]
    pub const fn new(validator: Arc<ApiKeyValidator>, auth_mode: AuthMode) -> Self {
        Self {
            validator,
            auth_mode,
        }
    }
}

#[cfg(feature = "security")]
impl Interceptor for AuthInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        // Skip authentication if disabled
        if matches!(self.auth_mode, AuthMode::None) {
            return Ok(request);
        }

        // Extract authorization from metadata
        let metadata = request.metadata();
        let auth_header = metadata
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| Status::unauthenticated("Missing authorization header"))?;

        // Validate API key synchronously using block_in_place
        // This bridges async validator with sync Interceptor trait
        let auth_context = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { self.validator.validate(auth_header).await })
        })
        .map_err(|e| match e {
            AuthError::InvalidApiKey => Status::unauthenticated("Invalid API key format or value"),
            AuthError::ExpiredKey => Status::unauthenticated("API key has expired"),
            AuthError::UnknownKey => Status::unauthenticated("API key not found"),
            AuthError::RevokedToken => Status::unauthenticated("API key has been revoked"),
            _ => Status::internal(format!("Authentication error: {e}")),
        })?;

        // Add auth context to request extensions for service methods
        request.extensions_mut().insert(auth_context);

        Ok(request)
    }
}

/// Stub interceptor when security feature is disabled
#[cfg(not(feature = "security"))]
#[derive(Clone)]
pub struct AuthInterceptor;

#[cfg(not(feature = "security"))]
impl AuthInterceptor {
    #[must_use]
    pub fn new(_validator: (), _auth_mode: ()) -> Self {
        Self
    }
}

#[cfg(not(feature = "security"))]
impl Interceptor for AuthInterceptor {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        Ok(request)
    }
}

// Tests moved to integration tests: engram-cli/tests/grpc_auth_tests.rs
