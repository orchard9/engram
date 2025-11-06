# Task: Security Configuration Schema

## Objective
Add security configuration to the CLI config system to support authentication modes, API key storage settings, and rate limiting configuration.

## Context
The current `CliConfig` struct in `engram-cli/src/config.rs` needs to be extended with security settings. This configuration will control whether authentication is enabled and how it behaves.

## Requirements

### 1. Extend Configuration Schema
Add new security section to `CliConfig`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Authentication mode: "none", "api_key"
    pub auth_mode: AuthMode,
    
    /// Enable rate limiting
    pub rate_limiting: bool,
    
    /// API key storage configuration
    pub api_keys: ApiKeyStorageConfig,
    
    /// CORS settings for production
    pub cors: CorsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMode {
    None,
    ApiKey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyStorageConfig {
    /// Storage backend type
    pub backend: StorageBackend,
    
    /// Path for file-based storage
    pub storage_path: PathBuf,
    
    /// Key rotation settings
    pub rotation_days: u32,
    pub warn_before_expiry_days: u32,
}
```

### 2. Default Configuration
Update `default.toml` with security defaults:
```toml
[security]
# Default to no auth for backward compatibility
auth_mode = "none"
rate_limiting = false

[security.api_keys]
backend = "file"
storage_path = "./data/api_keys.db"
rotation_days = 90
warn_before_expiry_days = 14

[security.cors]
# Allow all origins in development
allowed_origins = ["*"]
allowed_methods = ["GET", "POST", "DELETE", "OPTIONS"]
max_age_seconds = 3600
```

### 3. Production Configuration Template
Create `production.toml` example:
```toml
[security]
auth_mode = "api_key"
rate_limiting = true

[security.api_keys]
backend = "file"
storage_path = "/var/lib/engram/api_keys.db"
rotation_days = 30
warn_before_expiry_days = 7

[security.cors]
allowed_origins = ["https://app.example.com"]
allowed_methods = ["GET", "POST", "DELETE", "OPTIONS"]
max_age_seconds = 3600
```

### 4. Environment Variable Overrides
Support environment variables for security settings:
- `ENGRAM_AUTH_MODE` - Override auth mode
- `ENGRAM_API_KEY_PATH` - Override storage path
- `ENGRAM_RATE_LIMIT` - Enable/disable rate limiting

## Implementation Notes
- Ensure backward compatibility - default to no auth
- Validate configuration on load
- Support merging configs (like existing merge functionality)
- Add configuration documentation

## Acceptance Criteria
1. Security config section added to CliConfig
2. Default configuration maintains backward compatibility
3. Environment variables can override security settings
4. Configuration validates correctly on load
5. Production example configuration provided
6. Tests for config loading and validation