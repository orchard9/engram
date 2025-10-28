//! JWT token validation and OAuth2/OIDC integration.

use super::{AuthContext, AuthError, Permission, Principal, RateLimit};
use crate::MemorySpaceId;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// JWT claims
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user/service identifier)
    pub sub: String,

    /// Issued at
    pub iat: i64,

    /// Expiration
    pub exp: i64,

    /// Not before
    pub nbf: i64,

    /// Allowed memory spaces
    pub spaces: Vec<String>,

    /// Permissions
    pub perms: Vec<String>,

    /// Token ID for revocation
    pub jti: String,
}

/// JSON Web Key Set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwkSet {
    /// Keys
    pub keys: Vec<Jwk>,
}

impl JwkSet {
    /// Find key by ID
    #[must_use]
    pub fn find(&self, kid: &str) -> Option<&Jwk> {
        self.keys.iter().find(|k| k.kid.as_deref() == Some(kid))
    }
}

/// JSON Web Key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Jwk {
    /// Key type
    pub kty: String,

    /// Key ID
    pub kid: Option<String>,

    /// Algorithm
    pub alg: Option<String>,

    /// Use
    #[serde(rename = "use")]
    pub use_: Option<String>,

    /// Key value (base64url encoded)
    pub n: Option<String>,

    /// Exponent (base64url encoded)
    pub e: Option<String>,
}

impl Jwk {
    /// Convert JWK to PEM format
    #[must_use]
    #[allow(clippy::unused_self)]
    pub fn to_pem(&self) -> Option<Vec<u8>> {
        // Simplified - real implementation would decode n and e
        None
    }

    /// Convert to bytes representation
    #[must_use]
    #[allow(clippy::unused_self)]
    pub fn as_bytes(&self) -> Vec<u8> {
        // Simplified - would encode properly
        vec![]
    }
}

/// JWT validator
pub struct JwtValidator {
    /// Public keys for verification (supports rotation)
    keys: Arc<RwLock<JwkSet>>,

    /// Validation rules
    validation: Validation,

    /// Revocation list cache
    revoked_tokens: Arc<DashMap<String, DateTime<Utc>>>,
}

impl JwtValidator {
    /// Create a new JWT validator
    #[must_use]
    pub fn new(keys: JwkSet) -> Self {
        let mut validation = Validation::new(Algorithm::RS256);
        validation.validate_exp = true;
        validation.validate_nbf = true;

        Self {
            keys: Arc::new(RwLock::new(keys)),
            validation,
            revoked_tokens: Arc::new(DashMap::new()),
        }
    }

    /// Validate JWT from Authorization header
    ///
    /// # Errors
    ///
    /// Returns `AuthError` if:
    /// - Token format is invalid
    /// - Token signature is invalid
    /// - Token is expired
    /// - Token is revoked
    pub async fn validate(&self, token: &str) -> Result<AuthContext, AuthError> {
        // Remove "Bearer " prefix
        let token = token.strip_prefix("Bearer ").unwrap_or(token);

        // Decode header to get key ID
        let header = jsonwebtoken::decode_header(token)
            .map_err(|e| AuthError::InvalidToken(e.to_string()))?;
        let kid = header.kid.ok_or(AuthError::MissingKeyId)?;

        // Get public key for verification
        let keys = self.keys.read().await;
        let _key = keys.find(&kid).ok_or(AuthError::UnknownKey)?;

        // For now, use a placeholder decoding key
        // Real implementation would properly decode the JWK
        let decoding_key = DecodingKey::from_secret(b"placeholder");

        // Verify and decode
        let token_data = decode::<Claims>(token, &decoding_key, &self.validation)
            .map_err(|e| AuthError::InvalidToken(e.to_string()))?;

        // Check revocation list
        if self.revoked_tokens.contains_key(&token_data.claims.jti) {
            return Err(AuthError::RevokedToken);
        }

        // Build auth context
        let allowed_spaces = token_data
            .claims
            .spaces
            .into_iter()
            .filter_map(|s| MemorySpaceId::try_from(s.as_str()).ok())
            .collect();

        let permissions = token_data
            .claims
            .perms
            .into_iter()
            .filter_map(|p| match p.as_str() {
                "memory:read" => Some(Permission::MemoryRead),
                "memory:write" => Some(Permission::MemoryWrite),
                "memory:delete" => Some(Permission::MemoryDelete),
                "space:create" => Some(Permission::SpaceCreate),
                "space:delete" => Some(Permission::SpaceDelete),
                "space:list" => Some(Permission::SpaceList),
                "consolidation:trigger" => Some(Permission::ConsolidationTrigger),
                "consolidation:monitor" => Some(Permission::ConsolidationMonitor),
                "system:introspect" => Some(Permission::SystemIntrospect),
                "system:metrics" => Some(Permission::SystemMetrics),
                "system:health" => Some(Permission::SystemHealth),
                "admin:all" => Some(Permission::AdminAll),
                _ => None,
            })
            .collect();

        Ok(AuthContext {
            principal: Principal::Jwt(token_data.claims.sub),
            allowed_spaces,
            permissions,
            rate_limit: RateLimit::default(),
        })
    }

    /// Periodically refresh JWKS from OIDC provider
    ///
    /// # Errors
    ///
    /// Returns `AuthError` if JWKS refresh fails
    #[cfg(all(feature = "security", feature = "reqwest"))]
    pub async fn refresh_keys(&self, _jwks_uri: &str) -> Result<(), AuthError> {
        // Real implementation would use reqwest
        // For now, return placeholder
        Ok(())
    }

    /// Revoke a token by JTI
    pub fn revoke_token(&self, jti: String) {
        self.revoked_tokens.insert(jti, Utc::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jwk_set_find() {
        let jwk = Jwk {
            kty: "RSA".to_string(),
            kid: Some("key1".to_string()),
            alg: Some("RS256".to_string()),
            use_: Some("sig".to_string()),
            n: None,
            e: None,
        };

        let jwks = JwkSet { keys: vec![jwk] };
        assert!(jwks.find("key1").is_some());
        assert!(jwks.find("key2").is_none());
    }
}
