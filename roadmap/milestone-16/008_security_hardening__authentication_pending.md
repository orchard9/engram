# Task 008: Security Hardening & Authentication — pending

**Priority:** P1
**Estimated Effort:** 3 days
**Dependencies:** Task 001 (Container & Orchestration Deployment)

## Objective

Implement comprehensive security hardening for Engram's production deployment, covering TLS/mTLS encryption, authentication mechanisms, authorization for multi-tenant isolation, secrets management, and compliance validation. This task establishes the security foundation required for handling sensitive memory data in production environments.

## Architecture Overview

### Security Layers

```
┌─────────────────────────────────────────────────────────┐
│                    External Clients                      │
└─────────────────┬───────────────┬───────────────────────┘
                  │               │
         ┌────────▼────────┐  ┌──▼──────────┐
         │  TLS Termination│  │ mTLS Gateway│
         │   (HTTP/REST)   │  │   (gRPC)    │
         └────────┬────────┘  └──┬──────────┘
                  │               │
         ┌────────▼───────────────▼──────────┐
         │   Authentication Middleware        │
         │  (API Keys / JWT / OAuth2)        │
         └────────────────┬──────────────────┘
                          │
         ┌────────────────▼──────────────────┐
         │    Authorization Engine           │
         │  (Memory Space Access Control)    │
         └────────────────┬──────────────────┘
                          │
         ┌────────────────▼──────────────────┐
         │    Audit Logging Pipeline         │
         │   (Security Event Tracking)       │
         └────────────────┬──────────────────┘
                          │
         ┌────────────────▼──────────────────┐
         │   Memory Space Registry           │
         │  (Tenant Isolation Boundary)      │
         └───────────────────────────────────┘
```

### Security Model

1. **Transport Security**: All communication encrypted with TLS 1.3+
2. **Authentication**: Multiple mechanisms for different use cases
3. **Authorization**: Fine-grained access control per memory space
4. **Secrets Management**: Zero plaintext secrets in configuration
5. **Audit Trail**: Complete security event logging for compliance

## Key Deliverables

### 1. TLS/mTLS Configuration

#### TLS for HTTP/REST Endpoints

**Implementation Details:**
```rust
// engram-core/src/security/tls.rs
pub struct TlsConfig {
    /// Path to server certificate chain (PEM format)
    pub cert_chain_path: PathBuf,

    /// Path to server private key (PEM format)
    pub private_key_path: PathBuf,

    /// Optional CA bundle for client certificate validation
    pub ca_bundle_path: Option<PathBuf>,

    /// TLS protocol versions (min: TLS 1.3)
    pub min_protocol_version: TlsVersion,

    /// Cipher suites (AEAD only)
    pub cipher_suites: Vec<CipherSuite>,

    /// OCSP stapling configuration
    pub ocsp_stapling: bool,

    /// Session resumption settings
    pub session_cache_size: usize,

    /// Certificate rotation signal handler
    pub rotation_signal: Option<RotationSignal>,
}

impl TlsConfig {
    /// Load and validate TLS configuration
    pub fn load() -> Result<Self, SecurityError> {
        // Validate certificate chain
        // Verify private key matches certificate
        // Check certificate expiration (warn if <30 days)
        // Validate cipher suite availability
    }

    /// Hot-reload certificates without downtime
    pub async fn reload_certificates(&mut self) -> Result<(), SecurityError> {
        // Atomic certificate swap
        // Graceful connection draining
        // Update OCSP stapling
    }
}
```

**Certificate Generation Script:**
```bash
#!/bin/bash
# deployments/tls/generate_certs.sh

# Generate CA for development/testing
generate_ca() {
    openssl genrsa -out ca.key 4096
    openssl req -new -x509 -days 3650 -key ca.key \
        -out ca.crt -subj "/CN=Engram CA"
}

# Generate server certificate
generate_server_cert() {
    openssl genrsa -out server.key 2048
    openssl req -new -key server.key \
        -out server.csr -subj "/CN=$1" \
        -addext "subjectAltName=DNS:$1,IP:$2"
    openssl x509 -req -days 365 -in server.csr \
        -CA ca.crt -CAkey ca.key -CAcreateserial \
        -out server.crt -extfile <(printf "subjectAltName=DNS:$1,IP:$2")
}

# Generate client certificate for mTLS
generate_client_cert() {
    openssl genrsa -out client.key 2048
    openssl req -new -key client.key \
        -out client.csr -subj "/CN=$1"
    openssl x509 -req -days 365 -in client.csr \
        -CA ca.crt -CAkey ca.key -CAcreateserial \
        -out client.crt
}
```

#### mTLS for gRPC Endpoints

**Implementation:**
```rust
// engram-core/src/security/mtls.rs
pub struct MutualTlsConfig {
    /// Server identity
    pub server_identity: Identity,

    /// Client CA certificates for validation
    pub client_ca_certs: Vec<Certificate>,

    /// Client certificate validation mode
    pub validation_mode: ClientCertMode,

    /// Certificate pinning for high-security deployments
    pub pinned_certificates: Option<Vec<Fingerprint>>,

    /// Dynamic certificate validation callback
    pub custom_validator: Option<Arc<dyn CertificateValidator>>,
}

pub enum ClientCertMode {
    /// Reject connections without client cert
    Required,

    /// Accept but validate if provided
    Optional,

    /// Validate against specific CN patterns
    PatternMatch(Vec<String>),
}

impl MutualTlsConfig {
    /// Configure tonic server with mTLS
    pub fn configure_server(&self) -> ServerTlsConfig {
        ServerTlsConfig::new()
            .identity(self.server_identity.clone())
            .client_ca_root(self.client_ca_certs.clone())
            .client_auth_optional(matches!(self.validation_mode, ClientCertMode::Optional))
    }
}
```

### 2. Authentication Mechanisms

#### API Key Authentication

**Implementation:**
```rust
// engram-core/src/auth/api_key.rs
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use chrono::{DateTime, Utc};

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

#[derive(Clone, Debug)]
pub struct ApiKeyMetadata {
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub last_used: Option<DateTime<Utc>>,
    pub permissions: Vec<Permission>,
}

pub struct ApiKeyValidator {
    /// In-memory cache with TTL
    cache: Arc<DashMap<String, CachedValidation>>,

    /// Persistent storage backend
    store: Arc<dyn ApiKeyStore>,

    /// Argon2 configuration
    hasher: Argon2<'static>,
}

impl ApiKeyValidator {
    /// Validate API key from Authorization header
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
        let stored = self.store.get_key(&key.key_id).await?;

        // Verify secret hash (constant-time comparison)
        let parsed_hash = PasswordHash::new(&stored.secret_hash)?;
        self.hasher.verify_password(key.secret.as_bytes(), &parsed_hash)?;

        // Check expiration
        if let Some(expires) = stored.metadata.expires_at {
            if Utc::now() > expires {
                return Err(AuthError::ExpiredKey);
            }
        }

        // Build auth context
        let context = AuthContext {
            principal: Principal::ApiKey(key.key_id.clone()),
            allowed_spaces: stored.allowed_spaces.clone(),
            permissions: stored.metadata.permissions.clone(),
            rate_limit: stored.rate_limit.clone(),
        };

        // Update cache
        self.cache.insert(key.key_id.clone(), CachedValidation {
            context: context.clone(),
            expires_at: Utc::now() + Duration::minutes(5),
        });

        // Update last_used asynchronously
        tokio::spawn(async move {
            let _ = self.store.update_last_used(&key.key_id).await;
        });

        Ok(context)
    }

    /// Generate new API key
    pub async fn generate_key(&self, request: GenerateKeyRequest) -> Result<ApiKeyResponse, AuthError> {
        // Generate cryptographically secure random key
        let key_id = format!("engram_key_{}", generate_secure_id(24));
        let secret = generate_secure_secret(32);

        // Hash secret for storage
        let salt = SaltString::generate(&mut OsRng);
        let secret_hash = self.hasher
            .hash_password(secret.as_bytes(), &salt)?
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

        self.store.create_key(api_key).await?;

        // Return key only once (never stored in plaintext)
        Ok(ApiKeyResponse {
            key_id,
            secret,  // Only time secret is available
            full_key: format!("engram_key_{}_{}", key_id, secret),
        })
    }
}
```

#### JWT Token Authentication

**Implementation:**
```rust
// engram-core/src/auth/jwt.rs
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};

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

pub struct JwtValidator {
    /// Public keys for verification (supports rotation)
    pub keys: Arc<RwLock<JwkSet>>,

    /// Validation rules
    pub validation: Validation,

    /// Revocation list cache
    pub revoked_tokens: Arc<DashMap<String, DateTime<Utc>>>,
}

impl JwtValidator {
    /// Validate JWT from Authorization header
    pub async fn validate(&self, token: &str) -> Result<AuthContext, AuthError> {
        // Remove "Bearer " prefix
        let token = token.strip_prefix("Bearer ").unwrap_or(token);

        // Decode header to get key ID
        let header = jsonwebtoken::decode_header(token)?;
        let kid = header.kid.ok_or(AuthError::MissingKeyId)?;

        // Get public key for verification
        let keys = self.keys.read().await;
        let key = keys.find(&kid).ok_or(AuthError::UnknownKey)?;

        // Verify and decode
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_rsa_pem(key.as_bytes())?,
            &self.validation
        )?;

        // Check revocation list
        if self.revoked_tokens.contains_key(&token_data.claims.jti) {
            return Err(AuthError::RevokedToken);
        }

        // Build auth context
        Ok(AuthContext {
            principal: Principal::Jwt(token_data.claims.sub),
            allowed_spaces: token_data.claims.spaces
                .into_iter()
                .map(MemorySpaceId::from)
                .collect(),
            permissions: token_data.claims.perms
                .into_iter()
                .map(Permission::from_str)
                .collect::<Result<Vec<_>, _>>()?,
            rate_limit: RateLimit::default(),  // Can be customized per JWT
        })
    }

    /// Periodically refresh JWKS from OIDC provider
    pub async fn refresh_keys(&self, jwks_uri: &str) -> Result<(), AuthError> {
        let response = reqwest::get(jwks_uri).await?;
        let jwks: JwkSet = response.json().await?;

        let mut keys = self.keys.write().await;
        *keys = jwks;

        Ok(())
    }
}
```

#### OAuth2 Integration

**Configuration:**
```rust
// engram-core/src/auth/oauth2.rs
pub struct OAuth2Config {
    /// OIDC discovery endpoint
    pub issuer: String,

    /// Client credentials
    pub client_id: String,
    pub client_secret: SecretString,

    /// Redirect URI for authorization code flow
    pub redirect_uri: String,

    /// Required scopes
    pub scopes: Vec<String>,

    /// Custom claim mappings
    pub claim_mappings: ClaimMappings,
}

pub struct ClaimMappings {
    /// Map OIDC claim to memory spaces
    pub spaces_claim: String,  // e.g., "engram_spaces"

    /// Map OIDC claim to permissions
    pub permissions_claim: String,  // e.g., "engram_perms"

    /// Custom transformer functions
    pub transformers: HashMap<String, Box<dyn ClaimTransformer>>,
}
```

### 3. Authorization Model

#### Memory Space Access Control

**Implementation:**
```rust
// engram-core/src/auth/authorization.rs
use std::collections::HashSet;

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Permission {
    // Memory operations
    MemoryRead,
    MemoryWrite,
    MemoryDelete,

    // Space management
    SpaceCreate,
    SpaceDelete,
    SpaceList,

    // Consolidation operations
    ConsolidationTrigger,
    ConsolidationMonitor,

    // System operations
    SystemIntrospect,
    SystemMetrics,
    SystemHealth,

    // Admin operations
    AdminAll,
}

pub struct AuthorizationEngine {
    /// Policy evaluator
    evaluator: Arc<PolicyEvaluator>,

    /// Permission cache
    cache: Arc<DashMap<String, CachedPermissions>>,
}

impl AuthorizationEngine {
    /// Check if operation is authorized
    pub async fn authorize(
        &self,
        ctx: &AuthContext,
        space_id: &MemorySpaceId,
        operation: &Operation,
    ) -> Result<(), AuthError> {
        // Check space access
        if !ctx.allowed_spaces.contains(space_id) {
            // Check wildcard access
            if !ctx.allowed_spaces.iter().any(|s| s.as_str() == "*") {
                return Err(AuthError::SpaceAccessDenied);
            }
        }

        // Check permission
        let required_perm = operation.required_permission();
        if !ctx.permissions.contains(&required_perm) {
            // Check admin override
            if !ctx.permissions.contains(&Permission::AdminAll) {
                return Err(AuthError::PermissionDenied(required_perm));
            }
        }

        // Apply rate limiting
        self.apply_rate_limit(ctx).await?;

        // Log authorization decision
        audit_log::info!({
            "event": "authorization",
            "principal": format!("{:?}", ctx.principal),
            "space": space_id.as_str(),
            "operation": format!("{:?}", operation),
            "result": "allowed"
        });

        Ok(())
    }

    /// Apply rate limiting
    async fn apply_rate_limit(&self, ctx: &AuthContext) -> Result<(), AuthError> {
        // Implementation using token bucket or sliding window
        // Track per principal + operation
        Ok(())
    }
}

#[derive(Debug)]
pub enum Operation {
    Remember,
    Recall,
    Forget,
    Consolidate,
    Introspect,
}

impl Operation {
    fn required_permission(&self) -> Permission {
        match self {
            Operation::Remember => Permission::MemoryWrite,
            Operation::Recall => Permission::MemoryRead,
            Operation::Forget => Permission::MemoryDelete,
            Operation::Consolidate => Permission::ConsolidationTrigger,
            Operation::Introspect => Permission::SystemIntrospect,
        }
    }
}
```

### 4. Secrets Management Integration

#### HashiCorp Vault Integration

**Implementation:**
```rust
// engram-core/src/security/vault.rs
use vaultrs::{client::VaultClient, error::ClientError};

pub struct VaultSecretsManager {
    client: VaultClient,
    mount_path: String,
    role_id: String,
    cache: Arc<DashMap<String, CachedSecret>>,
}

impl VaultSecretsManager {
    /// Initialize with AppRole authentication
    pub async fn new(config: VaultConfig) -> Result<Self, SecurityError> {
        let client = VaultClient::new(config.address)?;

        // Authenticate with AppRole
        let auth_response = client
            .auth()
            .approle(&config.role_id, &config.secret_id)
            .await?;

        client.set_token(&auth_response.client_token);

        Ok(Self {
            client,
            mount_path: config.mount_path,
            role_id: config.role_id,
            cache: Arc::new(DashMap::new()),
        })
    }

    /// Retrieve secret with caching
    pub async fn get_secret(&self, key: &str) -> Result<SecretString, SecurityError> {
        // Check cache
        if let Some(cached) = self.cache.get(key) {
            if !cached.is_expired() {
                return Ok(cached.value.clone());
            }
        }

        // Fetch from Vault
        let secret = self.client
            .kv2(&self.mount_path)
            .read(key)
            .await?;

        // Cache with TTL
        self.cache.insert(key.to_string(), CachedSecret {
            value: SecretString::from(secret.data["value"].as_str().unwrap()),
            expires_at: Utc::now() + Duration::minutes(5),
        });

        Ok(secret.into())
    }

    /// Rotate database credentials
    pub async fn rotate_db_credentials(&self) -> Result<DatabaseCredentials, SecurityError> {
        let response = self.client
            .database()
            .generate_credentials("engram-db")
            .await?;

        Ok(DatabaseCredentials {
            username: response.username,
            password: SecretString::from(response.password),
            expires_at: response.lease_duration.map(|d| Utc::now() + Duration::seconds(d as i64)),
        })
    }
}
```

#### AWS Secrets Manager Integration

**Implementation:**
```rust
// engram-core/src/security/aws_secrets.rs
use aws_sdk_secretsmanager::{Client, Error};

pub struct AwsSecretsManager {
    client: Client,
    cache: Arc<DashMap<String, CachedSecret>>,
}

impl AwsSecretsManager {
    pub async fn get_secret(&self, secret_id: &str) -> Result<SecretString, SecurityError> {
        // Check cache
        if let Some(cached) = self.cache.get(secret_id) {
            if !cached.is_expired() {
                return Ok(cached.value.clone());
            }
        }

        // Fetch from AWS
        let response = self.client
            .get_secret_value()
            .secret_id(secret_id)
            .send()
            .await?;

        let secret = SecretString::from(
            response.secret_string()
                .ok_or(SecurityError::SecretNotFound)?
        );

        // Cache with TTL
        self.cache.insert(secret_id.to_string(), CachedSecret {
            value: secret.clone(),
            expires_at: Utc::now() + Duration::minutes(5),
        });

        Ok(secret)
    }
}
```

#### Kubernetes Secrets Integration

**Configuration:**
```yaml
# deployments/kubernetes/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: engram-auth-secrets
  namespace: engram
type: Opaque
data:
  api-key-salt: <base64>
  jwt-public-key: <base64>
  oauth-client-secret: <base64>
---
apiVersion: v1
kind: Secret
metadata:
  name: engram-tls
  namespace: engram
type: kubernetes.io/tls
data:
  tls.crt: <base64>
  tls.key: <base64>
```

### 5. Security Hardening Checklist

#### System Hardening

```bash
#!/bin/bash
# scripts/security_hardening.sh

# OS-level hardening
harden_os() {
    # Disable unnecessary services
    systemctl disable avahi-daemon
    systemctl disable cups

    # Kernel parameters
    cat >> /etc/sysctl.d/99-engram.conf <<EOF
# Network hardening
net.ipv4.tcp_syncookies = 1
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.tcp_timestamps = 0

# File system hardening
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
kernel.randomize_va_space = 2
kernel.yama.ptrace_scope = 1
EOF

    sysctl -p /etc/sysctl.d/99-engram.conf
}

# Container hardening
harden_container() {
    # Run as non-root user
    useradd -r -s /bin/false engram

    # Set file permissions
    chown -R engram:engram /app
    chmod 750 /app
    chmod 640 /app/config/*

    # Remove unnecessary packages
    apt-get remove --purge -y wget curl netcat
    apt-get autoremove -y

    # Clear package cache
    apt-get clean
    rm -rf /var/lib/apt/lists/*
}

# Application hardening
harden_application() {
    # Validate configuration
    /app/engram validate-config --strict

    # Check for known vulnerabilities
    /app/engram security-scan

    # Initialize secure defaults
    /app/engram init-security \
        --min-tls-version=1.3 \
        --require-auth=true \
        --enable-audit-log=true
}
```

#### Security Scanning

```toml
# .cargo/audit.toml
[advisories]
ignore = []
informational_warnings = ["unmaintained"]

[output]
deny = ["warnings"]
format = "json"
```

### 6. Audit Logging

**Implementation:**
```rust
// engram-core/src/security/audit.rs
use serde_json::json;

#[derive(Debug, Serialize)]
pub struct AuditEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Event type
    pub event_type: AuditEventType,

    /// Principal performing action
    pub principal: Option<String>,

    /// Target resource
    pub resource: Option<String>,

    /// Operation details
    pub operation: String,

    /// Result (success/failure)
    pub result: AuditResult,

    /// Additional metadata
    pub metadata: serde_json::Value,

    /// Request correlation ID
    pub correlation_id: String,

    /// Source IP address
    pub source_ip: Option<String>,
}

#[derive(Debug, Serialize)]
pub enum AuditEventType {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    Configuration,
    SystemAccess,
}

#[derive(Debug, Serialize)]
pub enum AuditResult {
    Success,
    Failure(String),
    PartialSuccess(String),
}

pub struct AuditLogger {
    /// Structured logger
    logger: slog::Logger,

    /// Event buffer for batching
    buffer: Arc<Mutex<Vec<AuditEvent>>>,

    /// Remote audit sink (e.g., SIEM)
    remote_sink: Option<Arc<dyn AuditSink>>,
}

impl AuditLogger {
    /// Log security event
    pub async fn log_event(&self, event: AuditEvent) {
        // Log locally with structured format
        slog::info!(self.logger, "AUDIT";
            "timestamp" => event.timestamp.to_rfc3339(),
            "event_type" => format!("{:?}", event.event_type),
            "principal" => event.principal.as_ref().unwrap_or(&"anonymous".to_string()),
            "resource" => event.resource.as_ref().unwrap_or(&"".to_string()),
            "operation" => &event.operation,
            "result" => format!("{:?}", event.result),
            "correlation_id" => &event.correlation_id,
            "source_ip" => event.source_ip.as_ref().unwrap_or(&"".to_string()),
            "metadata" => event.metadata.to_string()
        );

        // Buffer for remote sink
        if let Some(sink) = &self.remote_sink {
            let mut buffer = self.buffer.lock().await;
            buffer.push(event);

            // Flush if buffer is full
            if buffer.len() >= 100 {
                let events = std::mem::take(&mut *buffer);
                tokio::spawn(async move {
                    let _ = sink.send_batch(events).await;
                });
            }
        }
    }
}

// Macros for common audit events
#[macro_export]
macro_rules! audit_auth {
    ($logger:expr, $principal:expr, $result:expr) => {
        $logger.log_event(AuditEvent {
            timestamp: Utc::now(),
            event_type: AuditEventType::Authentication,
            principal: Some($principal.to_string()),
            resource: None,
            operation: "authenticate".to_string(),
            result: $result,
            metadata: json!({}),
            correlation_id: uuid::Uuid::new_v4().to_string(),
            source_ip: None,
        }).await
    };
}
```

## Implementation Plan

### Phase 1: TLS/mTLS Setup (Day 1)

1. **Morning (4 hours)**:
   - Implement TLS configuration module
   - Create certificate generation scripts
   - Configure HTTP/REST endpoints with TLS
   - Test with curl and openssl s_client

2. **Afternoon (4 hours)**:
   - Implement mTLS for gRPC endpoints
   - Configure client certificate validation
   - Create certificate rotation mechanism
   - Test with grpcurl and custom client

### Phase 2: Authentication Implementation (Day 2)

1. **Morning (4 hours)**:
   - Implement API key generation and validation
   - Create JWT token validation
   - Set up OAuth2/OIDC integration
   - Build authentication middleware

2. **Afternoon (4 hours)**:
   - Integrate with secrets management
   - Implement credential caching
   - Add authentication to all endpoints
   - Create authentication test suite

### Phase 3: Authorization & Hardening (Day 3)

1. **Morning (4 hours)**:
   - Implement authorization engine
   - Configure memory space access control
   - Set up rate limiting
   - Create permission evaluation logic

2. **Afternoon (4 hours)**:
   - Implement audit logging
   - Run security hardening script
   - Perform vulnerability scanning
   - Create compliance documentation

## Files Created/Modified

### New Files
- `/engram-core/src/security/mod.rs` - Security module root
- `/engram-core/src/security/tls.rs` - TLS configuration
- `/engram-core/src/security/mtls.rs` - mTLS implementation
- `/engram-core/src/auth/mod.rs` - Authentication module root
- `/engram-core/src/auth/api_key.rs` - API key authentication
- `/engram-core/src/auth/jwt.rs` - JWT token validation
- `/engram-core/src/auth/oauth2.rs` - OAuth2 integration
- `/engram-core/src/auth/authorization.rs` - Authorization engine
- `/engram-core/src/security/vault.rs` - Vault integration
- `/engram-core/src/security/aws_secrets.rs` - AWS Secrets Manager
- `/engram-core/src/security/audit.rs` - Audit logging
- `/deployments/tls/generate_certs.sh` - Certificate generation
- `/deployments/tls/README.md` - TLS setup documentation
- `/scripts/security_hardening.sh` - Hardening script
- `/scripts/security_scan.sh` - Vulnerability scanning

### Modified Files
- `/engram-cli/src/config.rs` - Add security configuration
- `/engram-cli/src/cli/server.rs` - Integrate authentication
- `/proto/engram/v1/service.proto` - Add auth metadata fields
- `/deployments/kubernetes/*.yaml` - Add security contexts
- `/deployments/docker/Dockerfile` - Apply hardening

## Testing & Validation

### Security Test Suite

```rust
#[cfg(test)]
mod security_tests {
    use super::*;

    #[tokio::test]
    async fn test_tls_configuration() {
        // Test TLS 1.3 enforcement
        // Test cipher suite restrictions
        // Test certificate validation
    }

    #[tokio::test]
    async fn test_api_key_authentication() {
        // Test key generation
        // Test validation with valid key
        // Test rejection with invalid key
        // Test expiration handling
    }

    #[tokio::test]
    async fn test_jwt_validation() {
        // Test valid JWT acceptance
        // Test expired token rejection
        // Test signature verification
        // Test claim extraction
    }

    #[tokio::test]
    async fn test_authorization() {
        // Test space access control
        // Test permission evaluation
        // Test rate limiting
        // Test audit logging
    }

    #[tokio::test]
    async fn test_secrets_management() {
        // Test secret retrieval
        // Test caching behavior
        // Test rotation handling
    }
}
```

### Penetration Testing

```bash
#!/bin/bash
# scripts/pentest.sh

# TLS/SSL testing
testssl() {
    docker run --rm -ti drwetter/testssl.sh https://localhost:8443
}

# OWASP ZAP scanning
owasp_scan() {
    docker run -t owasp/zap2docker-stable zap-baseline.py \
        -t https://localhost:8443 -r security-report.html
}

# Authentication bypass attempts
auth_bypass_test() {
    # Test without auth header
    curl -k https://localhost:8443/api/recall

    # Test with malformed token
    curl -k -H "Authorization: Bearer invalid" https://localhost:8443/api/recall

    # Test with expired token
    curl -k -H "Authorization: Bearer $EXPIRED_TOKEN" https://localhost:8443/api/recall
}

# SQL injection attempts (for API parameters)
injection_test() {
    curl -k -H "Authorization: Bearer $TOKEN" \
        "https://localhost:8443/api/recall?space=test' OR '1'='1"
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] TLS 1.3+ enforced on all endpoints
- [ ] mTLS available for gRPC connections
- [ ] API key authentication working
- [ ] JWT validation integrated
- [ ] OAuth2/OIDC flow functional
- [ ] Memory space access control enforced
- [ ] Rate limiting active
- [ ] Audit logging capturing all security events
- [ ] Secrets never stored in plaintext
- [ ] Certificate rotation without downtime

### Security Requirements
- [ ] Pass OWASP ZAP baseline scan
- [ ] Pass testssl.sh with A+ rating
- [ ] No high/critical vulnerabilities in cargo audit
- [ ] All OWASP Top 10 addressed
- [ ] Security headers configured (HSTS, CSP, etc.)
- [ ] Input validation on all endpoints
- [ ] SQL/NoSQL injection prevention
- [ ] XSS protection for web interfaces
- [ ] CSRF tokens for state-changing operations

### Performance Requirements
- [ ] Authentication adds <10ms latency
- [ ] Authorization checks <5ms
- [ ] TLS handshake <50ms
- [ ] Audit logging doesn't block operations
- [ ] Secret caching reduces external calls by >90%

### Operational Requirements
- [ ] Security configuration documented
- [ ] Credential rotation procedures defined
- [ ] Incident response playbook created
- [ ] Compliance mapping (SOC2, GDPR) documented
- [ ] Security monitoring dashboards configured

## Documentation

### Security Configuration Guide
- TLS/mTLS setup instructions
- Authentication configuration
- Authorization rules definition
- Secrets management integration
- Hardening checklist walkthrough

### API Authentication Guide
- API key generation and usage
- JWT token format and claims
- OAuth2 flow documentation
- Authentication examples in multiple languages

### Security Operations Runbook
- Certificate rotation procedures
- Credential management
- Security monitoring
- Incident response procedures
- Compliance validation steps

## Follow-Up Tasks

1. **Security Monitoring Enhancement** (P2)
   - Implement intrusion detection
   - Add behavioral analytics
   - Create security dashboards

2. **Advanced Authentication** (P2)
   - Add WebAuthn/FIDO2 support
   - Implement risk-based authentication
   - Add session management

3. **Compliance Automation** (P3)
   - Automated compliance scanning
   - Policy-as-code implementation
   - Compliance report generation

## Risk Mitigation

### Implementation Risks
- **Certificate Management Complexity**: Mitigate with automation and clear procedures
- **Performance Impact**: Mitigate with caching and optimized crypto operations
- **Breaking Changes**: Mitigate with gradual rollout and backwards compatibility

### Security Risks
- **Zero-Day Vulnerabilities**: Mitigate with defense in depth and regular updates
- **Credential Compromise**: Mitigate with rotation, monitoring, and least privilege
- **Insider Threats**: Mitigate with audit logging and separation of duties

## Notes

This security implementation follows defense-in-depth principles with multiple layers of protection. The architecture supports both simple deployments (API keys) and enterprise requirements (OAuth2/OIDC, mTLS). All security events are audited for compliance and forensics.

The implementation prioritizes:
1. **Zero Trust**: Never trust, always verify
2. **Least Privilege**: Minimal permissions by default
3. **Defense in Depth**: Multiple security layers
4. **Observability**: Complete audit trail
5. **Automation**: Reduce human error through automation

Production operators should start with the security hardening checklist and gradually enable additional features based on their threat model and compliance requirements.