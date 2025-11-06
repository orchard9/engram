# Task: File-based API Key Storage Implementation

## Objective
Implement a SQLite-based storage backend for API keys that implements the existing `ApiKeyStore` trait.

## Context
The `engram-core/src/auth/api_key.rs` file defines an `ApiKeyStore` trait that needs a concrete implementation. We'll use SQLite for simple, file-based storage that's easy to backup and manage.

## Requirements

### 1. Create SQLite Storage Implementation
Implement `ApiKeyStore` trait:
```rust
use sqlx::{SqlitePool, sqlite::SqlitePoolOptions};

pub struct SqliteApiKeyStore {
    pool: SqlitePool,
}

impl SqliteApiKeyStore {
    pub async fn new(path: &Path) -> Result<Self, AuthError> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(
                SqliteConnectOptions::new()
                    .filename(path)
                    .create_if_missing(true)
            )
            .await?;
            
        // Run migrations
        sqlx::migrate!("./migrations/api_keys")
            .run(&pool)
            .await?;
            
        Ok(Self { pool })
    }
}
```

### 2. Database Schema
Create migration in `migrations/api_keys/001_initial.sql`:
```sql
CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    secret_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    permissions TEXT NOT NULL, -- JSON array
    allowed_spaces TEXT NOT NULL, -- JSON array
    rate_limit_rps INTEGER NOT NULL,
    rate_limit_burst INTEGER NOT NULL,
    created_at TEXT NOT NULL, -- ISO 8601
    expires_at TEXT, -- ISO 8601
    last_used TEXT, -- ISO 8601
    revoked_at TEXT -- ISO 8601
);

CREATE INDEX idx_expires_at ON api_keys(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_revoked_at ON api_keys(revoked_at) WHERE revoked_at IS NOT NULL;
CREATE INDEX idx_last_used ON api_keys(last_used);
```

### 3. Implement ApiKeyStore Methods
```rust
#[async_trait::async_trait]
impl ApiKeyStore for SqliteApiKeyStore {
    async fn get_key(&self, key_id: &str) -> Result<ApiKey, String> {
        // Query key and check not revoked/expired
        let row = sqlx::query!(
            r#"
            SELECT * FROM api_keys 
            WHERE key_id = ? 
                AND (revoked_at IS NULL)
                AND (expires_at IS NULL OR expires_at > datetime('now'))
            "#,
            key_id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "Key not found or expired".to_string())?;
        
        // Deserialize and return
        Ok(ApiKey {
            key_id: row.key_id,
            secret_hash: row.secret_hash,
            allowed_spaces: serde_json::from_str(&row.allowed_spaces)?,
            rate_limit: RateLimit {
                requests_per_second: row.rate_limit_rps as u32,
                burst_size: row.rate_limit_burst as u32,
            },
            metadata: ApiKeyMetadata {
                name: row.name,
                created_at: DateTime::parse_from_rfc3339(&row.created_at)?.into(),
                expires_at: row.expires_at.map(|s| DateTime::parse_from_rfc3339(&s)?.into()),
                last_used: row.last_used.map(|s| DateTime::parse_from_rfc3339(&s)?.into()),
                permissions: serde_json::from_str(&row.permissions)?,
            },
        })
    }
    
    async fn create_key(&self, key: ApiKey) -> Result<(), String> {
        // Insert new key
    }
    
    async fn update_last_used(&self, key_id: &str) -> Result<(), String> {
        // Update last_used timestamp
    }
    
    async fn delete_key(&self, key_id: &str) -> Result<(), String> {
        // Soft delete by setting revoked_at
    }
}
```

### 4. Add Management Methods
Extend with additional management functionality:
```rust
impl SqliteApiKeyStore {
    /// List all active keys
    pub async fn list_keys(&self) -> Result<Vec<ApiKeyInfo>, String>;
    
    /// Get keys expiring soon
    pub async fn get_expiring_keys(&self, days: u32) -> Result<Vec<ApiKeyInfo>, String>;
    
    /// Revoke a key with reason
    pub async fn revoke_key(&self, key_id: &str, reason: &str) -> Result<(), String>;
    
    /// Clean up expired/revoked keys
    pub async fn cleanup_old_keys(&self) -> Result<u64, String>;
}
```

### 5. Connection Pool Management
- Use connection pooling for concurrent access
- Configure appropriate pool size
- Handle connection errors gracefully
- Add retry logic for transient failures

### 6. Testing
- Unit tests for all CRUD operations
- Test expiration handling
- Test concurrent access
- Test migration execution
- Test cleanup operations

## Dependencies
- Add `sqlx` with sqlite and runtime-tokio-native-tls features
- Add `sqlx-cli` as dev dependency for migrations

## Implementation Notes
- Use prepared statements for all queries
- Ensure thread-safe access
- Add proper error handling and logging
- Consider WAL mode for better concurrency
- Add database backup functionality

## Acceptance Criteria
1. SQLite storage implementation passes all ApiKeyStore trait tests
2. Migrations create proper schema with indexes
3. Concurrent access works correctly
4. Expired keys are filtered out automatically
5. Soft delete functionality works
6. Performance: < 5ms for key lookups