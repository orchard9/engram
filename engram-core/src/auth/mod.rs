//! Authentication and authorization infrastructure.
//!
//! This module provides multiple authentication mechanisms:
//! - API key authentication with Argon2 hashing
//! - JWT token validation
//! - OAuth2/OIDC integration
//!
//! Authorization is handled through fine-grained access control per memory space.

#[cfg(feature = "security")]
pub mod api_key;

#[cfg(feature = "security")]
pub mod jwt;

#[cfg(feature = "security")]
pub mod authorization;

#[cfg(feature = "security")]
pub mod sqlite_store;

#[cfg(feature = "security")]
pub use sqlite_store::{ApiKeyInfo, SqliteApiKeyStore};

use crate::MemorySpaceId;
use thiserror::Error;

/// Authentication-related errors
#[derive(Error, Debug)]
pub enum AuthError {
    /// Invalid API key
    #[error("Invalid API key")]
    InvalidApiKey,

    /// Expired API key
    #[error("API key has expired")]
    ExpiredKey,

    /// Invalid JWT token
    #[error("Invalid JWT token: {0}")]
    InvalidToken(String),

    /// Missing key ID
    #[error("Missing key ID in token header")]
    MissingKeyId,

    /// Unknown key
    #[error("Unknown key ID")]
    UnknownKey,

    /// Revoked token
    #[error("Token has been revoked")]
    RevokedToken,

    /// Space access denied
    #[error("Access to memory space denied")]
    SpaceAccessDenied,

    /// Permission denied
    #[error("Permission denied: {0:?}")]
    PermissionDenied(Permission),

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    /// Argon2 error
    #[error("Argon2 error: {0}")]
    Argon2(String),

    /// JWT error
    #[error("JWT error: {0}")]
    Jwt(String),
}

/// Authentication context for a request
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// Authenticated principal
    pub principal: Principal,

    /// Allowed memory spaces
    pub allowed_spaces: Vec<MemorySpaceId>,

    /// Granted permissions
    pub permissions: Vec<Permission>,

    /// Rate limiting configuration
    pub rate_limit: RateLimit,
}

/// Principal (authenticated entity)
#[derive(Debug, Clone)]
pub enum Principal {
    /// API key authentication
    ApiKey(String),

    /// JWT authentication
    Jwt(String),

    /// Client certificate (mTLS)
    Certificate(String),

    /// Service account
    Service(String),
}

/// Permission types
#[cfg_attr(feature = "security", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Permission {
    /// Read memory operations
    MemoryRead,
    /// Write memory operations
    MemoryWrite,
    /// Delete memory operations
    MemoryDelete,

    /// Create memory spaces
    SpaceCreate,
    /// Delete memory spaces
    SpaceDelete,
    /// List memory spaces
    SpaceList,

    /// Trigger consolidation
    ConsolidationTrigger,
    /// Monitor consolidation
    ConsolidationMonitor,

    /// System introspection
    SystemIntrospect,
    /// View system metrics
    SystemMetrics,
    /// Check system health
    SystemHealth,

    /// Administrator access (all permissions)
    AdminAll,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimit {
    /// Maximum requests per second
    pub requests_per_second: u32,

    /// Burst capacity
    pub burst_size: u32,
}

impl Default for RateLimit {
    fn default() -> Self {
        Self {
            requests_per_second: 100,
            burst_size: 200,
        }
    }
}

/// Operation types that require authorization
#[derive(Debug, Clone)]
pub enum Operation {
    /// Remember operation
    Remember,
    /// Recall operation
    Recall,
    /// Forget operation
    Forget,
    /// Consolidate operation
    Consolidate,
    /// Introspect operation
    Introspect,
}

impl Operation {
    /// Get required permission for this operation
    #[must_use]
    pub const fn required_permission(&self) -> Permission {
        match self {
            Self::Remember => Permission::MemoryWrite,
            Self::Recall => Permission::MemoryRead,
            Self::Forget => Permission::MemoryDelete,
            Self::Consolidate => Permission::ConsolidationTrigger,
            Self::Introspect => Permission::SystemIntrospect,
        }
    }
}
