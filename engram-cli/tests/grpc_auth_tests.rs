//! Integration tests for gRPC authentication interceptor.
//!
//! Tests cover:
//! - Auth disabled mode
//! - Missing auth header
//! - Invalid API key format
//!
//! Note: Full end-to-end tests with valid API keys would require
//! spinning up a real gRPC server, which is covered by manual testing
//! and will be added in future integration test improvements.

#[cfg(feature = "security")]
mod grpc_auth_tests {
    use engram_cli::auth::AuthInterceptor;
    use engram_cli::config::AuthMode;
    use engram_core::auth::SqliteApiKeyStore;
    use engram_core::auth::api_key::ApiKeyValidator;
    use std::sync::Arc;
    use tonic::{Request, service::Interceptor};

    #[tokio::test]
    async fn test_auth_disabled() {
        // Create minimal store for validator (won't be used when auth disabled)
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let db_path = temp_dir.path().join("test_keys.db");
        let store = SqliteApiKeyStore::new(&db_path)
            .await
            .expect("create store");

        let validator = Arc::new(ApiKeyValidator::new(Arc::new(store)));
        let mut interceptor = AuthInterceptor::new(validator, AuthMode::None);

        // Create request without auth header
        let request = Request::new(());

        // Should pass even without auth when disabled
        let result = interceptor.call(request);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_missing_auth_header() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let db_path = temp_dir.path().join("test_keys.db");
        let store = SqliteApiKeyStore::new(&db_path)
            .await
            .expect("create store");

        let validator = Arc::new(ApiKeyValidator::new(Arc::new(store)));
        let mut interceptor = AuthInterceptor::new(validator, AuthMode::ApiKey);

        // Create request without auth header
        let request = Request::new(());

        // Should reject
        let result = interceptor.call(request);
        assert!(result.is_err());

        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
        assert!(status.message().contains("Missing authorization header"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_invalid_api_key_format() {
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let db_path = temp_dir.path().join("test_keys.db");
        let store = SqliteApiKeyStore::new(&db_path)
            .await
            .expect("create store");

        let validator = Arc::new(ApiKeyValidator::new(Arc::new(store)));
        let mut interceptor = AuthInterceptor::new(validator, AuthMode::ApiKey);

        // Create request with invalid auth format
        let mut request = Request::new(());
        request.metadata_mut().insert(
            "authorization",
            "invalid_format".parse().expect("parse metadata"),
        );

        // Should reject
        let result = interceptor.call(request);
        assert!(result.is_err());

        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }
}

// Stub tests when security feature is disabled
#[cfg(not(feature = "security"))]
mod grpc_auth_tests {
    #[test]
    fn security_feature_disabled() {
        // No-op test when security feature is not enabled
    }
}
