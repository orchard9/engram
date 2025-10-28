//! Security infrastructure for Engram production deployments.
//!
//! This module provides comprehensive security features including:
//! - TLS/mTLS configuration for encrypted communication
//! - Secrets management integration (Vault, AWS Secrets Manager, Kubernetes)
//! - Audit logging for security events
//! - Certificate management and rotation
//!
//! All security features are optional and can be enabled via feature flags.

#[cfg(feature = "security")]
pub mod tls;

#[cfg(feature = "security")]
pub mod audit;

#[cfg(feature = "vault_secrets")]
pub mod vault;

#[cfg(feature = "aws_secrets")]
pub mod aws_secrets;

use thiserror::Error;

/// Security-related errors
#[derive(Error, Debug)]
pub enum SecurityError {
    /// TLS configuration error
    #[error("TLS configuration error: {0}")]
    TlsConfig(String),

    /// Certificate error
    #[error("Certificate error: {0}")]
    Certificate(String),

    /// Secrets management error
    #[error("Secrets management error: {0}")]
    Secrets(String),

    /// Audit logging error
    #[error("Audit logging error: {0}")]
    Audit(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Secret not found
    #[error("Secret not found: {0}")]
    SecretNotFound(String),
}
