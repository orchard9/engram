//! SQLite-based API key storage implementation.
//!
//! This module provides a production-ready SQLite storage backend for API keys,
//! implementing the `ApiKeyStore` trait with features including:
//!
//! - Automatic expiration filtering at the database level
//! - Soft delete with audit trail via `revoked_at` timestamps
//! - WAL mode for concurrent read/write access
//! - Connection pooling for multi-threaded applications
//! - Comprehensive key management methods beyond the core trait
//!
//! # Feature Flag
//!
//! This module requires the `security` feature flag to be enabled:
//!
//! ```toml
//! [dependencies]
//! engram-core = { version = "0.1.0", features = ["security"] }
//! ```
//!
//! # Example
//!
//! ```no_run
//! use engram_core::auth::SqliteApiKeyStore;
//! use std::path::Path;
//!
//! # async fn example() -> Result<(), String> {
//! // Create a new store
//! let store = SqliteApiKeyStore::new(Path::new("data/api_keys.db")).await?;
//!
//! // List all active keys
//! let keys = store.list_keys().await?;
//! println!("Found {} active API keys", keys.len());
//!
//! // Get keys expiring within 7 days
//! let expiring = store.get_expiring_keys(7).await?;
//! for key in expiring {
//!     println!("Key '{}' expires soon", key.name);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Performance
//!
//! The implementation is optimized for < 5ms key lookup latency with:
//! - Indexed queries on `expires_at`, `revoked_at`, and `last_used` columns
//! - Prepared statements via sqlx for all database operations
//! - Connection pooling (5 connections by default)
//! - WAL journaling mode for better concurrency

use super::api_key::{ApiKey, ApiKeyMetadata, ApiKeyStore};
use super::{Permission, RateLimit};
use crate::MemorySpaceId;
use chrono::{DateTime, Utc};
use sqlx::{
    Row, SqlitePool,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions, SqliteRow},
};
use std::path::Path;

/// SQLite-based API key storage
pub struct SqliteApiKeyStore {
    pool: SqlitePool,
}

/// Lightweight API key info for listing operations
#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    /// Key identifier
    pub key_id: String,
    /// Human-readable name
    pub name: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Expiration timestamp
    pub expires_at: Option<DateTime<Utc>>,
    /// Last used timestamp
    pub last_used: Option<DateTime<Utc>>,
    /// Whether the key is revoked
    pub revoked: bool,
}

/// Helper function to parse timestamps from SQLite rows
///
/// # Errors
///
/// Returns error if the timestamp string is not valid RFC3339 format
fn parse_timestamp(row: &SqliteRow, column: &str) -> Result<DateTime<Utc>, String> {
    let timestamp_str: String = row
        .try_get(column)
        .map_err(|e| format!("Failed to get {column}: {e}"))?;
    DateTime::parse_from_rfc3339(&timestamp_str)
        .map(Into::into)
        .map_err(|e| format!("Invalid {column} timestamp '{timestamp_str}': {e}"))
}

/// Helper function to parse optional timestamps from SQLite rows
fn parse_optional_timestamp(
    row: &SqliteRow,
    column: &str,
) -> Result<Option<DateTime<Utc>>, String> {
    let timestamp_str: Option<String> = row
        .try_get(column)
        .map_err(|e| format!("Failed to get {column}: {e}"))?;

    timestamp_str
        .map(|s| {
            DateTime::parse_from_rfc3339(&s)
                .map(Into::into)
                .map_err(|e| format!("Invalid {column} timestamp '{s}': {e}"))
        })
        .transpose()
}

impl SqliteApiKeyStore {
    /// Create a new SQLite API key store
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Cannot create parent directory
    /// - Cannot connect to database
    /// - Cannot run migrations
    #[must_use = "the SqliteApiKeyStore should be used to access the API key storage"]
    pub async fn new(path: &Path) -> Result<Self, String> {
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {e}"))?;
        }

        // Configure connection with WAL mode for better concurrency
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(
                SqliteConnectOptions::new()
                    .filename(path)
                    .create_if_missing(true)
                    .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal),
            )
            .await
            .map_err(|e| format!("Failed to connect to database: {e}"))?;

        // Run migrations
        sqlx::migrate!("../migrations/api_keys")
            .run(&pool)
            .await
            .map_err(|e| format!("Failed to run migrations: {e}"))?;

        Ok(Self { pool })
    }

    /// List all active (non-revoked, non-expired) keys
    ///
    /// # Errors
    ///
    /// Returns error if database query fails
    #[must_use = "iterator result should be checked for errors and processed"]
    pub async fn list_keys(&self) -> Result<Vec<ApiKeyInfo>, String> {
        let rows = sqlx::query(
            r"
            SELECT key_id, name, created_at, expires_at, last_used, revoked_at
            FROM api_keys
            WHERE revoked_at IS NULL
                AND (expires_at IS NULL OR datetime(expires_at) > datetime('now'))
            ORDER BY created_at DESC
            ",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("Failed to list keys: {e}"))?;

        rows.into_iter()
            .map(|row: SqliteRow| {
                Ok(ApiKeyInfo {
                    key_id: row
                        .try_get("key_id")
                        .map_err(|e| format!("Failed to get key_id: {e}"))?,
                    name: row
                        .try_get("name")
                        .map_err(|e| format!("Failed to get name: {e}"))?,
                    created_at: parse_timestamp(&row, "created_at")?,
                    expires_at: parse_optional_timestamp(&row, "expires_at")?,
                    last_used: parse_optional_timestamp(&row, "last_used")?,
                    revoked: {
                        let revoked_str: Option<String> = row
                            .try_get("revoked_at")
                            .map_err(|e| format!("Failed to get revoked_at: {e}"))?;
                        revoked_str.is_some()
                    },
                })
            })
            .collect()
    }

    /// Get keys expiring within specified number of days
    ///
    /// # Errors
    ///
    /// Returns error if database query fails
    #[must_use = "expiring keys should be checked to alert administrators"]
    pub async fn get_expiring_keys(&self, days: u32) -> Result<Vec<ApiKeyInfo>, String> {
        let rows = sqlx::query(
            r"
            SELECT key_id, name, created_at, expires_at, last_used, revoked_at
            FROM api_keys
            WHERE revoked_at IS NULL
                AND expires_at IS NOT NULL
                AND datetime(expires_at) <= datetime('now', '+' || ? || ' days')
                AND datetime(expires_at) > datetime('now')
            ORDER BY expires_at ASC
            ",
        )
        .bind(days)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("Failed to get expiring keys: {e}"))?;

        rows.into_iter()
            .map(|row: SqliteRow| {
                Ok(ApiKeyInfo {
                    key_id: row
                        .try_get("key_id")
                        .map_err(|e| format!("Failed to get key_id: {e}"))?,
                    name: row
                        .try_get("name")
                        .map_err(|e| format!("Failed to get name: {e}"))?,
                    created_at: parse_timestamp(&row, "created_at")?,
                    expires_at: parse_optional_timestamp(&row, "expires_at")?,
                    last_used: parse_optional_timestamp(&row, "last_used")?,
                    revoked: {
                        let revoked_str: Option<String> = row
                            .try_get("revoked_at")
                            .map_err(|e| format!("Failed to get revoked_at: {e}"))?;
                        revoked_str.is_some()
                    },
                })
            })
            .collect()
    }

    /// Revoke a key with audit reason
    ///
    /// # Errors
    ///
    /// Returns error if database update fails
    pub async fn revoke_key(&self, key_id: &str, reason: &str) -> Result<(), String> {
        let now = Utc::now().to_rfc3339();
        let result = sqlx::query(
            r"
            UPDATE api_keys
            SET revoked_at = ?, revocation_reason = ?
            WHERE key_id = ? AND revoked_at IS NULL
            ",
        )
        .bind(&now)
        .bind(reason)
        .bind(key_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("Failed to revoke key: {e}"))?;

        if result.rows_affected() == 0 {
            return Err("Key not found or already revoked".to_string());
        }

        Ok(())
    }

    /// Clean up old revoked and expired keys
    ///
    /// Returns the number of keys deleted
    ///
    /// # Errors
    ///
    /// Returns error if database delete fails
    #[must_use = "cleanup count should be logged for audit purposes"]
    pub async fn cleanup_old_keys(&self) -> Result<u64, String> {
        let result = sqlx::query(
            r"
            DELETE FROM api_keys
            WHERE (revoked_at IS NOT NULL AND datetime(revoked_at) < datetime('now', '-30 days'))
                OR (expires_at IS NOT NULL AND datetime(expires_at) < datetime('now', '-30 days'))
            ",
        )
        .execute(&self.pool)
        .await
        .map_err(|e| format!("Failed to cleanup old keys: {e}"))?;

        Ok(result.rows_affected())
    }
}

#[async_trait::async_trait]
impl ApiKeyStore for SqliteApiKeyStore {
    async fn get_key(&self, key_id: &str) -> Result<ApiKey, String> {
        let row = sqlx::query(
            r"
            SELECT key_id, secret_hash, name, permissions, allowed_spaces,
                   rate_limit_rps, rate_limit_burst, created_at, expires_at,
                   last_used
            FROM api_keys
            WHERE key_id = ?
                AND revoked_at IS NULL
                AND (expires_at IS NULL OR datetime(expires_at) > datetime('now'))
            ",
        )
        .bind(key_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| format!("Database error: {e}"))?
        .ok_or_else(|| "Key not found or expired".to_string())?;

        // Deserialize JSON fields
        let permissions_str: String = row
            .try_get("permissions")
            .map_err(|e| format!("Failed to get permissions: {e}"))?;
        let permissions: Vec<Permission> = serde_json::from_str(&permissions_str)
            .map_err(|e| format!("Failed to deserialize permissions: {e}"))?;

        let allowed_spaces_str: String = row
            .try_get("allowed_spaces")
            .map_err(|e| format!("Failed to get allowed_spaces: {e}"))?;
        let allowed_spaces: Vec<MemorySpaceId> = serde_json::from_str(&allowed_spaces_str)
            .map_err(|e| format!("Failed to deserialize allowed_spaces: {e}"))?;

        // Parse timestamps
        let created_at = parse_timestamp(&row, "created_at")?;
        let expires_at = parse_optional_timestamp(&row, "expires_at")?;
        let last_used = parse_optional_timestamp(&row, "last_used")?;

        let rate_limit_rps: i64 = row
            .try_get("rate_limit_rps")
            .map_err(|e| format!("Failed to get rate_limit_rps: {e}"))?;
        let rate_limit_burst: i64 = row
            .try_get("rate_limit_burst")
            .map_err(|e| format!("Failed to get rate_limit_burst: {e}"))?;

        Ok(ApiKey {
            key_id: row
                .try_get("key_id")
                .map_err(|e| format!("Failed to get key_id: {e}"))?,
            secret_hash: row
                .try_get("secret_hash")
                .map_err(|e| format!("Failed to get secret_hash: {e}"))?,
            allowed_spaces,
            rate_limit: RateLimit {
                requests_per_second: rate_limit_rps
                    .try_into()
                    .map_err(|_| {
                        format!(
                            "Invalid rate_limit_rps: value {rate_limit_rps} exceeds u32::MAX (4294967295). \
                             Suggestion: Verify database integrity with 'SELECT * FROM api_keys WHERE rate_limit_rps > 4294967295'. \
                             Expected range: 1-4294967295 requests per second"
                        )
                    })?,
                burst_size: rate_limit_burst
                    .try_into()
                    .map_err(|_| {
                        format!(
                            "Invalid rate_limit_burst: value {rate_limit_burst} exceeds u32::MAX (4294967295). \
                             Suggestion: Verify database integrity with 'SELECT * FROM api_keys WHERE rate_limit_burst > 4294967295'. \
                             Expected range: 1-4294967295 burst size"
                        )
                    })?,
            },
            metadata: ApiKeyMetadata {
                name: row
                    .try_get("name")
                    .map_err(|e| format!("Failed to get name: {e}"))?,
                created_at,
                expires_at,
                last_used,
                permissions,
            },
        })
    }

    async fn create_key(&self, key: &ApiKey) -> Result<(), String> {
        // Serialize JSON fields
        let permissions = serde_json::to_string(&key.metadata.permissions)
            .map_err(|e| format!("Failed to serialize permissions: {e}"))?;

        let allowed_spaces = serde_json::to_string(&key.allowed_spaces)
            .map_err(|e| format!("Failed to serialize allowed_spaces: {e}"))?;

        // Convert timestamps to RFC3339
        let created_at = key.metadata.created_at.to_rfc3339();
        let expires_at = key.metadata.expires_at.map(|dt| dt.to_rfc3339());
        let last_used = key.metadata.last_used.map(|dt| dt.to_rfc3339());

        // Convert rate limits to i64 for SQLite
        let rate_limit_rps: i64 = key.rate_limit.requests_per_second.into();
        let rate_limit_burst: i64 = key.rate_limit.burst_size.into();

        sqlx::query(
            r"
            INSERT INTO api_keys (
                key_id, secret_hash, name, permissions, allowed_spaces,
                rate_limit_rps, rate_limit_burst, created_at, expires_at, last_used
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ",
        )
        .bind(&key.key_id)
        .bind(&key.secret_hash)
        .bind(&key.metadata.name)
        .bind(&permissions)
        .bind(&allowed_spaces)
        .bind(rate_limit_rps)
        .bind(rate_limit_burst)
        .bind(&created_at)
        .bind(expires_at.as_ref())
        .bind(last_used.as_ref())
        .execute(&self.pool)
        .await
        .map_err(|e| format!("Failed to create key: {e}"))?;

        Ok(())
    }

    async fn update_last_used(&self, key_id: &str) -> Result<(), String> {
        let now = Utc::now().to_rfc3339();
        let result = sqlx::query(
            r"
            UPDATE api_keys
            SET last_used = ?
            WHERE key_id = ?
            ",
        )
        .bind(&now)
        .bind(key_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("Failed to update last_used: {e}"))?;

        if result.rows_affected() == 0 {
            return Err("Key not found".to_string());
        }

        Ok(())
    }

    async fn delete_key(&self, key_id: &str) -> Result<(), String> {
        let now = Utc::now().to_rfc3339();
        let result = sqlx::query(
            r"
            UPDATE api_keys
            SET revoked_at = ?, revocation_reason = ?
            WHERE key_id = ? AND revoked_at IS NULL
            ",
        )
        .bind(&now)
        .bind("Soft delete via delete_key")
        .bind(key_id)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("Failed to delete key: {e}"))?;

        if result.rows_affected() == 0 {
            return Err("Key not found or already revoked".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
// Per coding guidelines: unwrap() is allowed in tests (prefer over expect())
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::auth::Permission;
    use tempfile::NamedTempFile;

    async fn create_test_store() -> (SqliteApiKeyStore, NamedTempFile) {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        let store = SqliteApiKeyStore::new(path).await.unwrap();
        (store, temp_file)
    }

    fn create_test_key(key_id: &str) -> ApiKey {
        ApiKey {
            key_id: key_id.to_string(),
            secret_hash: "test_hash".to_string(),
            allowed_spaces: vec![crate::MemorySpaceId::default()],
            rate_limit: RateLimit {
                requests_per_second: 100,
                burst_size: 200,
            },
            metadata: ApiKeyMetadata {
                name: "Test Key".to_string(),
                created_at: Utc::now(),
                expires_at: None,
                last_used: None,
                permissions: vec![Permission::MemoryRead, Permission::MemoryWrite],
            },
        }
    }

    #[tokio::test]
    async fn test_create_and_get_key() {
        let (store, _temp) = create_test_store().await;
        let key = create_test_key("test_key_1");

        // Create key
        store.create_key(&key).await.unwrap();

        // Retrieve key
        let retrieved = store.get_key("test_key_1").await.unwrap();

        assert_eq!(retrieved.key_id, key.key_id);
        assert_eq!(retrieved.secret_hash, key.secret_hash);
        assert_eq!(retrieved.metadata.name, key.metadata.name);
        assert_eq!(retrieved.allowed_spaces, key.allowed_spaces);
        assert_eq!(
            retrieved.rate_limit.requests_per_second,
            key.rate_limit.requests_per_second
        );
    }

    #[tokio::test]
    async fn test_update_last_used() {
        let (store, _temp) = create_test_store().await;
        let key = create_test_key("test_key_2");

        store.create_key(&key).await.unwrap();

        // Update last_used
        store.update_last_used("test_key_2").await.unwrap();

        // Verify update
        let retrieved = store.get_key("test_key_2").await.unwrap();
        assert!(retrieved.metadata.last_used.is_some());
    }

    #[tokio::test]
    async fn test_delete_key_soft_delete() {
        let (store, _temp) = create_test_store().await;
        let key = create_test_key("test_key_3");

        store.create_key(&key).await.unwrap();

        // Soft delete
        store.delete_key("test_key_3").await.unwrap();

        // Should not be retrievable
        let result = store.get_key("test_key_3").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_expired_key_not_returned() {
        let (store, _temp) = create_test_store().await;
        let mut key = create_test_key("test_key_4");
        key.metadata.expires_at = Some(Utc::now() - chrono::Duration::hours(1));

        store.create_key(&key).await.unwrap();

        // Should not be retrievable (expired)
        let result = store.get_key("test_key_4").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_keys() {
        let (store, _temp) = create_test_store().await;

        // Create multiple keys
        for i in 1..=3 {
            let key = create_test_key(&format!("list_key_{i}"));
            store.create_key(&key).await.unwrap();
        }

        // List all keys
        let keys = store.list_keys().await.unwrap();
        assert_eq!(keys.len(), 3);
    }

    #[tokio::test]
    async fn test_get_expiring_keys() {
        let (store, _temp) = create_test_store().await;

        // Create key expiring in 2 days
        let mut key = create_test_key("expiring_key");
        key.metadata.expires_at = Some(Utc::now() + chrono::Duration::days(2));
        store.create_key(&key).await.unwrap();

        // Get keys expiring within 7 days
        let keys = store.get_expiring_keys(7).await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].key_id, "expiring_key");
    }

    #[tokio::test]
    async fn test_revoke_key() {
        let (store, _temp) = create_test_store().await;
        let key = create_test_key("revoke_key");

        store.create_key(&key).await.unwrap();

        // Revoke key
        store
            .revoke_key("revoke_key", "Testing revocation")
            .await
            .unwrap();

        // Should not be retrievable
        let result = store.get_key("revoke_key").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cleanup_old_keys() {
        let (store, _temp) = create_test_store().await;

        // Create an expired key (would need to be >30 days old to be cleaned up)
        // For testing, we just verify the method works
        let count = store.cleanup_old_keys().await.unwrap();
        assert_eq!(count, 0); // No keys old enough to clean
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        // Create multiple connections to same database
        let store1 = SqliteApiKeyStore::new(&path).await.unwrap();
        let store2 = SqliteApiKeyStore::new(&path).await.unwrap();

        // Write from store1
        let key = create_test_key("concurrent_key");
        store1.create_key(&key).await.unwrap();

        // Read from store2
        let retrieved = store2.get_key("concurrent_key").await.unwrap();
        assert_eq!(retrieved.key_id, "concurrent_key");
    }
}
