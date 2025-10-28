//! API key authentication with Argon2 hashing.

use super::{AuthContext, AuthError, Permission, Principal, RateLimit};
use crate::MemorySpaceId;
use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
};
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use rand::Rng;
use secrecy::{ExposeSecret, SecretString};
use std::sync::Arc;

/// API key metadata
#[derive(Clone, Debug)]
pub struct ApiKeyMetadata {
    /// Human-readable name
    pub name: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Expiration timestamp
    pub expires_at: Option<DateTime<Utc>>,
    /// Last used timestamp
    pub last_used: Option<DateTime<Utc>>,
    /// Granted permissions
    pub permissions: Vec<Permission>,
}

/// API key (stored form with hashed secret)
#[derive(Clone, Debug)]
pub struct ApiKey {
    /// Unique key identifier (not the secret)
    pub key_id: String,

    /// Hashed secret (Argon2id)
    pub secret_hash: String,

    /// Associated memory spaces
    pub allowed_spaces: Vec<MemorySpaceId>,

    /// Rate limiting configuration
    pub rate_limit: RateLimit,

    /// Key metadata
    pub metadata: ApiKeyMetadata,
}

/// Cached validation result
struct CachedValidation {
    context: AuthContext,
    expires_at: DateTime<Utc>,
}

impl CachedValidation {
    fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }
}

/// API key generation request
pub struct GenerateKeyRequest {
    /// Human-readable name
    pub name: String,
    /// Allowed memory spaces
    pub allowed_spaces: Vec<MemorySpaceId>,
    /// Rate limiting configuration
    pub rate_limit: RateLimit,
    /// Expiration time
    pub expires_at: Option<DateTime<Utc>>,
    /// Granted permissions
    pub permissions: Vec<Permission>,
}

/// API key generation response
pub struct ApiKeyResponse {
    /// Key identifier
    pub key_id: String,
    /// Secret (only available at generation time)
    pub secret: String,
    /// Full key in format "engram_key_{key_id}_{secret}"
    pub full_key: String,
}

/// API key validator with caching
pub struct ApiKeyValidator {
    /// In-memory cache with TTL
    cache: Arc<DashMap<String, CachedValidation>>,

    /// Persistent storage backend
    store: Arc<dyn ApiKeyStore>,

    /// Argon2 hasher configuration
    hasher: Argon2<'static>,
}

impl ApiKeyValidator {
    /// Create a new API key validator
    #[must_use]
    pub fn new(store: Arc<dyn ApiKeyStore>) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            store,
            hasher: Argon2::default(),
        }
    }

    /// Validate API key from Authorization header
    ///
    /// # Errors
    ///
    /// Returns `AuthError` if:
    /// - Key format is invalid
    /// - Key is not found
    /// - Secret hash doesn't match
    /// - Key has expired
    pub async fn validate(&self, auth_header: &str) -> Result<AuthContext, AuthError> {
        // Parse "Bearer engram_key_..." format
        let key = parse_api_key(auth_header)?;

        // Check cache first (with TTL)
        if let Some(cached) = self.cache.get(&key.key_id) {
            if !cached.is_expired() {
                return Ok(cached.context.clone());
            }
        }

        // Load from store
        let stored = self
            .store
            .get_key(&key.key_id)
            .await
            .map_err(|_| AuthError::InvalidApiKey)?;

        // Verify secret hash (constant-time comparison)
        let parsed_hash =
            PasswordHash::new(&stored.secret_hash).map_err(|e| AuthError::Argon2(e.to_string()))?;
        self.hasher
            .verify_password(key.secret.expose_secret().as_bytes(), &parsed_hash)
            .map_err(|_| AuthError::InvalidApiKey)?;

        // Check expiration
        if stored
            .metadata
            .expires_at
            .is_some_and(|expires| Utc::now() > expires)
        {
            return Err(AuthError::ExpiredKey);
        }

        // Build auth context
        let context = AuthContext {
            principal: Principal::ApiKey(key.key_id.clone()),
            allowed_spaces: stored.allowed_spaces.clone(),
            permissions: stored.metadata.permissions.clone(),
            rate_limit: stored.rate_limit.clone(),
        };

        // Update cache
        self.cache.insert(
            key.key_id.clone(),
            CachedValidation {
                context: context.clone(),
                expires_at: Utc::now() + Duration::minutes(5),
            },
        );

        Ok(context)
    }

    /// Generate new API key
    ///
    /// # Errors
    ///
    /// Returns `AuthError` if key generation fails
    pub async fn generate_key(
        &self,
        request: GenerateKeyRequest,
    ) -> Result<ApiKeyResponse, AuthError> {
        // Generate cryptographically secure random key
        let key_id = generate_secure_id(24);
        let secret = generate_secure_secret(32);

        // Hash secret for storage
        let salt = SaltString::generate(&mut rand::thread_rng());
        let secret_hash = self
            .hasher
            .hash_password(secret.as_bytes(), &salt)
            .map_err(|e| AuthError::Argon2(e.to_string()))?
            .to_string();

        // Store key
        let api_key = ApiKey {
            key_id: key_id.clone(),
            secret_hash,
            allowed_spaces: request.allowed_spaces,
            rate_limit: request.rate_limit,
            metadata: ApiKeyMetadata {
                name: request.name,
                created_at: Utc::now(),
                expires_at: request.expires_at,
                last_used: None,
                permissions: request.permissions,
            },
        };

        self.store
            .create_key(api_key)
            .await
            .map_err(|_| AuthError::Argon2("Failed to store key".to_string()))?;

        // Return key only once (never stored in plaintext)
        Ok(ApiKeyResponse {
            key_id: key_id.clone(),
            secret: secret.clone(),
            full_key: format!("engram_key_{key_id}_{secret}"),
        })
    }
}

/// Parsed API key
struct ParsedApiKey {
    key_id: String,
    secret: SecretString,
}

/// Parse API key from Authorization header
fn parse_api_key(auth_header: &str) -> Result<ParsedApiKey, AuthError> {
    let key = auth_header
        .strip_prefix("Bearer ")
        .ok_or(AuthError::InvalidApiKey)?;

    let key = key
        .strip_prefix("engram_key_")
        .ok_or(AuthError::InvalidApiKey)?;

    let parts: Vec<&str> = key.split('_').collect();
    if parts.len() != 2 {
        return Err(AuthError::InvalidApiKey);
    }

    Ok(ParsedApiKey {
        key_id: parts[0].to_string(),
        secret: SecretString::new(parts[1].to_string().into()),
    })
}

/// Generate cryptographically secure ID
fn generate_secure_id(length: usize) -> String {
    let mut rng = rand::thread_rng();
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..62);
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                .chars()
                .nth(idx)
                .unwrap_or('a')
        })
        .collect()
}

/// Generate cryptographically secure secret
fn generate_secure_secret(length: usize) -> String {
    generate_secure_id(length)
}

/// API key storage backend trait
#[async_trait::async_trait]
pub trait ApiKeyStore: Send + Sync {
    /// Get API key by ID
    async fn get_key(&self, key_id: &str) -> Result<ApiKey, String>;

    /// Create new API key
    async fn create_key(&self, key: ApiKey) -> Result<(), String>;

    /// Update last used timestamp
    async fn update_last_used(&self, key_id: &str) -> Result<(), String>;

    /// Delete API key
    async fn delete_key(&self, key_id: &str) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_api_key() {
        let header = "Bearer engram_key_abc123_secret456";
        let parsed = parse_api_key(header).unwrap();
        assert_eq!(parsed.key_id, "abc123");
        assert_eq!(parsed.secret.expose_secret(), "secret456");
    }

    #[test]
    fn test_secure_id_generation() {
        let id = generate_secure_id(24);
        assert_eq!(id.len(), 24);
        assert!(id.chars().all(|c| c.is_alphanumeric()));
    }
}
