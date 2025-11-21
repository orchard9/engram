use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use engram_core::{MemorySpaceId, cluster::config::ClusterConfig};
use serde::{Deserialize, Serialize};

use crate::router::RouterConfig;

const DEFAULT_CONFIG: &str = include_str!("../config/default.toml");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    #[serde(default)]
    pub feature_flags: FeatureFlags,
    #[serde(default)]
    pub memory_spaces: MemorySpacesConfig,
    #[serde(default)]
    pub persistence: PersistenceConfig,
    #[serde(default)]
    pub cluster: ClusterConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub router: RouterConfig,
    #[serde(default)]
    pub security: SecurityConfig,
}

impl Default for CliConfig {
    fn default() -> Self {
        toml::from_str(DEFAULT_CONFIG).expect("default CLI config to parse")
    }
}

impl CliConfig {
    #[allow(clippy::missing_const_for_fn)]
    pub fn merge(&mut self, other: &Self) {
        self.feature_flags.merge(&other.feature_flags);
        self.memory_spaces.merge(&other.memory_spaces);
        self.persistence.merge(&other.persistence);
        self.cluster = other.cluster.clone();
        self.server.merge(&other.server);
        self.router = other.router.clone();
        self.security.merge(&other.security);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureFlags {
    pub spreading_api_beta: bool,
}

impl FeatureFlags {
    #[allow(clippy::missing_const_for_fn)]
    fn merge(&mut self, other: &Self) {
        self.spreading_api_beta = other.spreading_api_beta;
    }
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            spreading_api_beta: default_spreading_api_beta(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemorySpacesConfig {
    #[serde(default = "default_memory_space_id")]
    pub default_space: MemorySpaceId,
    #[serde(default = "default_bootstrap_spaces")]
    pub bootstrap_spaces: Vec<MemorySpaceId>,
}

impl MemorySpacesConfig {
    #[allow(clippy::missing_const_for_fn)]
    fn merge(&mut self, other: &Self) {
        self.default_space = other.default_space.clone();
        if !other.bootstrap_spaces.is_empty() {
            self.bootstrap_spaces.clone_from(&other.bootstrap_spaces);
        }
    }
}

impl Default for MemorySpacesConfig {
    fn default() -> Self {
        Self {
            default_space: default_memory_space_id(),
            bootstrap_spaces: default_bootstrap_spaces(),
        }
    }
}

fn default_memory_space_id() -> MemorySpaceId {
    MemorySpaceId::default()
}

fn default_bootstrap_spaces() -> Vec<MemorySpaceId> {
    vec![MemorySpaceId::default()]
}

const fn default_spreading_api_beta() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistenceConfig {
    pub data_root: String,
    pub hot_capacity: usize,
    pub warm_capacity: usize,
    pub cold_capacity: usize,
}

impl PersistenceConfig {
    #[allow(clippy::missing_const_for_fn)]
    fn merge(&mut self, other: &Self) {
        if !other.data_root.is_empty() {
            self.data_root.clone_from(&other.data_root);
        }
        if other.hot_capacity > 0 {
            self.hot_capacity = other.hot_capacity;
        }
        if other.warm_capacity > 0 {
            self.warm_capacity = other.warm_capacity;
        }
        if other.cold_capacity > 0 {
            self.cold_capacity = other.cold_capacity;
        }
    }
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            data_root: "~/.local/share/engram".to_string(),
            hot_capacity: 100_000,
            warm_capacity: 1_000_000,
            cold_capacity: 10_000_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub http_bind: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            http_bind: "127.0.0.1".to_string(),
        }
    }
}

impl ServerConfig {
    #[allow(clippy::missing_const_for_fn)]
    fn merge(&mut self, other: &Self) {
        if !other.http_bind.is_empty() {
            self.http_bind.clone_from(&other.http_bind);
        }
    }
}

pub struct ConfigManager {
    path: PathBuf,
    config: CliConfig,
}

impl ConfigManager {
    pub fn load() -> Result<Self> {
        let path = default_config_path()?;
        Self::load_with_path(path)
    }

    pub fn load_with_path<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let path = path.into();
        let mut config = CliConfig::default();

        if path.exists() {
            let contents = fs::read_to_string(&path)
                .with_context(|| format!("failed to read config at {}", path.display()))?;
            let user_config: CliConfig = toml::from_str(&contents)
                .with_context(|| format!("invalid config at {}", path.display()))?;
            config.merge(&user_config);
        }

        Ok(Self { path, config })
    }

    #[must_use]
    pub const fn config(&self) -> &CliConfig {
        &self.config
    }

    #[allow(dead_code)]
    pub const fn config_mut(&mut self) -> &mut CliConfig {
        &mut self.config
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create config directory {}", parent.display())
            })?;
        }
        let toml = toml::to_string_pretty(&self.config)?;
        fs::write(&self.path, toml)
            .with_context(|| format!("failed to write config to {}", self.path.display()))?;
        Ok(())
    }

    #[must_use]
    pub fn get(&self, key: &str) -> Option<String> {
        match key {
            "feature_flags.spreading_api_beta" => {
                Some(self.config.feature_flags.spreading_api_beta.to_string())
            }
            "memory_spaces.default_space" => {
                Some(self.config.memory_spaces.default_space.to_string())
            }
            "memory_spaces.bootstrap_spaces" => Some(
                self.config
                    .memory_spaces
                    .bootstrap_spaces
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            "server.http_bind" => Some(self.config.server.http_bind.clone()),
            _ => None,
        }
    }

    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "feature_flags.spreading_api_beta" => {
                let parsed = parse_bool(value)?;
                self.config.feature_flags.spreading_api_beta = parsed;
                Ok(())
            }
            "memory_spaces.default_space" => {
                let id = MemorySpaceId::try_from(value)?;
                self.config.memory_spaces.default_space = id;
                Ok(())
            }
            "memory_spaces.bootstrap_spaces" => {
                let spaces = value
                    .split(',')
                    .filter(|segment| !segment.trim().is_empty())
                    .map(|segment| MemorySpaceId::try_from(segment.trim()))
                    .collect::<Result<Vec<_>, _>>()?;
                self.config.memory_spaces.bootstrap_spaces = if spaces.is_empty() {
                    default_bootstrap_spaces()
                } else {
                    spaces
                };
                Ok(())
            }
            "server.http_bind" => {
                if value.trim().is_empty() {
                    return Err(anyhow!("server.http_bind cannot be empty"));
                }
                self.config.server.http_bind = value.trim().to_string();
                Ok(())
            }
            _ => Err(anyhow!("unknown configuration key: {key}")),
        }
    }
}

fn parse_bool(value: &str) -> Result<bool> {
    match value.to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(anyhow!("expected boolean value, received '{value}'")),
    }
}

fn default_config_path() -> Result<PathBuf> {
    let base =
        dirs::config_dir().ok_or_else(|| anyhow!("unable to determine configuration directory"))?;
    Ok(base.join("engram").join("config.toml"))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SecurityConfig {
    pub auth_mode: AuthMode,
    pub rate_limiting: bool,
    pub api_keys: ApiKeyStorageConfig,
    pub cors: CorsConfig,
}

impl SecurityConfig {
    #[allow(clippy::missing_const_for_fn)]
    fn merge(&mut self, other: &Self) {
        self.auth_mode = other.auth_mode.clone();
        self.rate_limiting = other.rate_limiting;
        self.api_keys.merge(&other.api_keys);
        self.cors.merge(&other.cors);
    }

    pub fn validate(&self) -> Result<()> {
        // Validate API key storage path is not empty
        if self.api_keys.storage_path.as_os_str().is_empty() {
            return Err(anyhow!("api_keys.storage_path cannot be empty"));
        }

        // Validate rotation settings
        if self.api_keys.rotation_days == 0 {
            return Err(anyhow!("api_keys.rotation_days must be greater than 0"));
        }

        if self.api_keys.warn_before_expiry_days >= self.api_keys.rotation_days {
            return Err(anyhow!(
                "api_keys.warn_before_expiry_days ({}) must be less than rotation_days ({})",
                self.api_keys.warn_before_expiry_days,
                self.api_keys.rotation_days
            ));
        }

        // Validate CORS settings
        if self.cors.allowed_origins.is_empty() {
            return Err(anyhow!("cors.allowed_origins cannot be empty"));
        }

        if self.cors.allowed_methods.is_empty() {
            return Err(anyhow!("cors.allowed_methods cannot be empty"));
        }

        if self.cors.max_age_seconds == 0 {
            return Err(anyhow!("cors.max_age_seconds must be greater than 0"));
        }

        Ok(())
    }

    #[must_use]
    pub fn with_env_overrides(&self) -> Self {
        let mut config = self.clone();

        // Override auth mode
        if let Ok(mode) = std::env::var("ENGRAM_AUTH_MODE")
            && let Ok(auth_mode) = AuthMode::try_from(mode.as_str()) {
            config.auth_mode = auth_mode;
        }

        // Override API key path
        if let Ok(path) = std::env::var("ENGRAM_API_KEY_PATH") {
            config.api_keys.storage_path = PathBuf::from(path);
        }

        // Override rate limiting
        if let Ok(rate_limit) = std::env::var("ENGRAM_RATE_LIMIT")
            && let Ok(enabled) = parse_bool(&rate_limit) {
            config.rate_limiting = enabled;
        }

        config
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            auth_mode: AuthMode::None,
            rate_limiting: false,
            api_keys: ApiKeyStorageConfig::default(),
            cors: CorsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMode {
    None,
    ApiKey,
}

impl TryFrom<&str> for AuthMode {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self> {
        match value.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "api_key" | "apikey" => Ok(Self::ApiKey),
            _ => Err(anyhow!(
                "invalid auth mode: '{}', expected 'none' or 'api_key'",
                value
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiKeyStorageConfig {
    pub backend: StorageBackend,
    pub storage_path: PathBuf,
    pub rotation_days: u32,
    pub warn_before_expiry_days: u32,
}

impl ApiKeyStorageConfig {
    #[allow(clippy::missing_const_for_fn)]
    fn merge(&mut self, other: &Self) {
        self.backend = other.backend.clone();
        if !other.storage_path.as_os_str().is_empty() {
            self.storage_path.clone_from(&other.storage_path);
        }
        if other.rotation_days > 0 {
            self.rotation_days = other.rotation_days;
        }
        if other.warn_before_expiry_days > 0 {
            self.warn_before_expiry_days = other.warn_before_expiry_days;
        }
    }
}

impl Default for ApiKeyStorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::File,
            storage_path: PathBuf::from("./data/api_keys.db"),
            rotation_days: 90,
            warn_before_expiry_days: 14,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageBackend {
    File,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CorsConfig {
    pub allowed_origins: Vec<String>,
    pub allowed_methods: Vec<String>,
    pub max_age_seconds: u32,
}

impl CorsConfig {
    #[allow(clippy::missing_const_for_fn)]
    fn merge(&mut self, other: &Self) {
        if !other.allowed_origins.is_empty() {
            self.allowed_origins.clone_from(&other.allowed_origins);
        }
        if !other.allowed_methods.is_empty() {
            self.allowed_methods.clone_from(&other.allowed_methods);
        }
        if other.max_age_seconds > 0 {
            self.max_age_seconds = other.max_age_seconds;
        }
    }
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "DELETE".to_string(),
                "OPTIONS".to_string(),
            ],
            max_age_seconds: 3600,
        }
    }
}

#[must_use]
pub fn format_feature_flags(flags: &FeatureFlags) -> Vec<String> {
    vec![format!("spreading_api_beta={}", flags.spreading_api_beta)]
}

#[must_use]
pub fn format_memory_spaces(cfg: &MemorySpacesConfig) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("default_space=\"{}\"", cfg.default_space));
    let bootstrap = cfg
        .bootstrap_spaces
        .iter()
        .map(|id| format!("\"{id}\""))
        .collect::<Vec<_>>()
        .join(", ");
    lines.push(format!("bootstrap_spaces=[{bootstrap}]"));
    lines
}

#[must_use]
pub fn format_sections(config: &CliConfig) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push("[feature_flags]".to_string());
    lines.extend(format_feature_flags(&config.feature_flags));
    lines.push(String::new());
    lines.push("[memory_spaces]".to_string());
    lines.extend(format_memory_spaces(&config.memory_spaces));
    lines
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Global mutex to serialize environment variable tests
    // This prevents race conditions when tests run in parallel
    static ENV_TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_default_security_config() {
        let config = SecurityConfig::default();

        assert_eq!(config.auth_mode, AuthMode::None);
        assert!(!config.rate_limiting);
        assert_eq!(config.api_keys.backend, StorageBackend::File);
        assert_eq!(config.api_keys.storage_path, PathBuf::from("./data/api_keys.db"));
        assert_eq!(config.api_keys.rotation_days, 90);
        assert_eq!(config.api_keys.warn_before_expiry_days, 14);
        assert_eq!(config.cors.allowed_origins, vec!["*"]);
        assert_eq!(config.cors.allowed_methods, vec!["GET", "POST", "DELETE", "OPTIONS"]);
        assert_eq!(config.cors.max_age_seconds, 3600);
    }

    #[test]
    fn test_security_config_validation_success() {
        let config = SecurityConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_security_config_validation_empty_storage_path() {
        let mut config = SecurityConfig::default();
        config.api_keys.storage_path = PathBuf::from("");

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("storage_path cannot be empty"));
    }

    #[test]
    fn test_security_config_validation_zero_rotation_days() {
        let mut config = SecurityConfig::default();
        config.api_keys.rotation_days = 0;

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rotation_days must be greater than 0"));
    }

    #[test]
    fn test_security_config_validation_invalid_expiry_warning() {
        let mut config = SecurityConfig::default();
        config.api_keys.rotation_days = 30;
        config.api_keys.warn_before_expiry_days = 30;

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be less than rotation_days"));
    }

    #[test]
    fn test_security_config_validation_empty_origins() {
        let mut config = SecurityConfig::default();
        config.cors.allowed_origins = vec![];

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("allowed_origins cannot be empty"));
    }

    #[test]
    fn test_security_config_validation_empty_methods() {
        let mut config = SecurityConfig::default();
        config.cors.allowed_methods = vec![];

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("allowed_methods cannot be empty"));
    }

    #[test]
    fn test_security_config_validation_zero_max_age() {
        let mut config = SecurityConfig::default();
        config.cors.max_age_seconds = 0;

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_age_seconds must be greater than 0"));
    }

    #[test]
    fn test_auth_mode_from_str() {
        assert_eq!(AuthMode::try_from("none").unwrap(), AuthMode::None);
        assert_eq!(AuthMode::try_from("None").unwrap(), AuthMode::None);
        assert_eq!(AuthMode::try_from("NONE").unwrap(), AuthMode::None);
        assert_eq!(AuthMode::try_from("api_key").unwrap(), AuthMode::ApiKey);
        assert_eq!(AuthMode::try_from("API_KEY").unwrap(), AuthMode::ApiKey);
        assert_eq!(AuthMode::try_from("apikey").unwrap(), AuthMode::ApiKey);

        let result = AuthMode::try_from("invalid");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid auth mode"));
    }

    #[test]
    fn test_security_config_from_toml() {
        // Parse SecurityConfig directly without [security] header
        let toml = r#"
            auth_mode = "api_key"
            rate_limiting = true

            [api_keys]
            backend = "file"
            storage_path = "/var/lib/engram/api_keys.db"
            rotation_days = 30
            warn_before_expiry_days = 7

            [cors]
            allowed_origins = ["https://example.com"]
            allowed_methods = ["GET", "POST"]
            max_age_seconds = 7200
        "#;

        let config: SecurityConfig = toml::from_str(toml).unwrap();

        assert_eq!(config.auth_mode, AuthMode::ApiKey);
        assert!(config.rate_limiting);
        assert_eq!(config.api_keys.storage_path, PathBuf::from("/var/lib/engram/api_keys.db"));
        assert_eq!(config.api_keys.rotation_days, 30);
        assert_eq!(config.api_keys.warn_before_expiry_days, 7);
        assert_eq!(config.cors.allowed_origins, vec!["https://example.com"]);
        assert_eq!(config.cors.allowed_methods, vec!["GET", "POST"]);
        assert_eq!(config.cors.max_age_seconds, 7200);
    }

    #[test]
    fn test_security_config_env_override_auth_mode() {
        let _guard = ENV_TEST_MUTEX.lock().unwrap();

        // Clear any existing env vars first
        unsafe {
            std::env::remove_var("ENGRAM_AUTH_MODE");
            std::env::remove_var("ENGRAM_API_KEY_PATH");
            std::env::remove_var("ENGRAM_RATE_LIMIT");
            std::env::set_var("ENGRAM_AUTH_MODE", "api_key");
        }

        let config = SecurityConfig::default();
        let overridden = config.with_env_overrides();

        assert_eq!(overridden.auth_mode, AuthMode::ApiKey);

        unsafe {
            std::env::remove_var("ENGRAM_AUTH_MODE");
        }
    }

    #[test]
    fn test_security_config_env_override_api_key_path() {
        let _guard = ENV_TEST_MUTEX.lock().unwrap();

        // Clear any existing env vars first
        unsafe {
            std::env::remove_var("ENGRAM_AUTH_MODE");
            std::env::remove_var("ENGRAM_API_KEY_PATH");
            std::env::remove_var("ENGRAM_RATE_LIMIT");
            std::env::set_var("ENGRAM_API_KEY_PATH", "/custom/path/keys.db");
        }

        let config = SecurityConfig::default();
        let overridden = config.with_env_overrides();

        assert_eq!(overridden.api_keys.storage_path, PathBuf::from("/custom/path/keys.db"));

        unsafe {
            std::env::remove_var("ENGRAM_API_KEY_PATH");
        }
    }

    #[test]
    fn test_security_config_env_override_rate_limit() {
        let _guard = ENV_TEST_MUTEX.lock().unwrap();

        // Clear any existing env vars first
        unsafe {
            std::env::remove_var("ENGRAM_AUTH_MODE");
            std::env::remove_var("ENGRAM_API_KEY_PATH");
            std::env::remove_var("ENGRAM_RATE_LIMIT");
            std::env::set_var("ENGRAM_RATE_LIMIT", "true");
        }

        let config = SecurityConfig::default();
        let overridden = config.with_env_overrides();

        assert!(overridden.rate_limiting);

        unsafe {
            std::env::remove_var("ENGRAM_RATE_LIMIT");
        }
    }

    #[test]
    fn test_security_config_env_override_invalid_auth_mode() {
        let _guard = ENV_TEST_MUTEX.lock().unwrap();

        // Clear any existing env vars first
        unsafe {
            std::env::remove_var("ENGRAM_AUTH_MODE");
            std::env::remove_var("ENGRAM_API_KEY_PATH");
            std::env::remove_var("ENGRAM_RATE_LIMIT");
            std::env::set_var("ENGRAM_AUTH_MODE", "invalid");
        }

        let config = SecurityConfig::default();
        let overridden = config.with_env_overrides();

        // Should keep original auth_mode when invalid
        assert_eq!(overridden.auth_mode, AuthMode::None);

        unsafe {
            std::env::remove_var("ENGRAM_AUTH_MODE");
        }
    }

    #[test]
    fn test_security_config_env_override_invalid_rate_limit() {
        let _guard = ENV_TEST_MUTEX.lock().unwrap();

        // Clear any existing env vars first
        unsafe {
            std::env::remove_var("ENGRAM_AUTH_MODE");
            std::env::remove_var("ENGRAM_API_KEY_PATH");
            std::env::remove_var("ENGRAM_RATE_LIMIT");
            std::env::set_var("ENGRAM_RATE_LIMIT", "invalid");
        }

        let config = SecurityConfig::default();
        let overridden = config.with_env_overrides();

        // Should keep original value when invalid
        assert!(!overridden.rate_limiting);

        unsafe {
            std::env::remove_var("ENGRAM_RATE_LIMIT");
        }
    }

    #[test]
    fn test_security_config_merge() {
        let mut base = SecurityConfig::default();

        let other = SecurityConfig {
            auth_mode: AuthMode::ApiKey,
            rate_limiting: true,
            api_keys: ApiKeyStorageConfig {
                backend: StorageBackend::File,
                storage_path: PathBuf::from("/custom/path"),
                rotation_days: 60,
                warn_before_expiry_days: 10,
            },
            cors: CorsConfig {
                allowed_origins: vec!["https://example.com".to_string()],
                allowed_methods: vec!["GET".to_string()],
                max_age_seconds: 7200,
            },
        };

        base.merge(&other);

        assert_eq!(base.auth_mode, AuthMode::ApiKey);
        assert!(base.rate_limiting);
        assert_eq!(base.api_keys.storage_path, PathBuf::from("/custom/path"));
        assert_eq!(base.api_keys.rotation_days, 60);
        assert_eq!(base.api_keys.warn_before_expiry_days, 10);
        assert_eq!(base.cors.allowed_origins, vec!["https://example.com"]);
        assert_eq!(base.cors.allowed_methods, vec!["GET"]);
        assert_eq!(base.cors.max_age_seconds, 7200);
    }

    #[test]
    fn test_cli_config_with_security() {
        let config = CliConfig::default();

        // Should have default security config
        assert_eq!(config.security.auth_mode, AuthMode::None);
        assert!(!config.security.rate_limiting);
    }

    #[test]
    fn test_cli_config_merge_security() {
        let mut base = CliConfig::default();

        let mut other = CliConfig::default();
        other.security.auth_mode = AuthMode::ApiKey;
        other.security.rate_limiting = true;

        base.merge(&other);

        assert_eq!(base.security.auth_mode, AuthMode::ApiKey);
        assert!(base.security.rate_limiting);
    }

    #[test]
    fn test_default_config_includes_security() {
        // Test that the default.toml includes security config and parses correctly
        let config = CliConfig::default();

        // Validate security section
        assert!(config.security.validate().is_ok());
        assert_eq!(config.security.auth_mode, AuthMode::None);
        assert!(!config.security.rate_limiting);
    }

    #[test]
    fn test_api_key_storage_config_merge_partial() {
        let mut base = ApiKeyStorageConfig::default();

        // Only override rotation_days, keep other values
        let other = ApiKeyStorageConfig {
            backend: StorageBackend::File,
            storage_path: PathBuf::from(""),  // Empty path should not override
            rotation_days: 60,
            warn_before_expiry_days: 0,  // Zero should not override
        };

        base.merge(&other);

        assert_eq!(base.storage_path, PathBuf::from("./data/api_keys.db"));  // Kept original
        assert_eq!(base.rotation_days, 60);  // Updated
        assert_eq!(base.warn_before_expiry_days, 14);  // Kept original
    }

    #[test]
    fn test_cors_config_merge_partial() {
        let mut base = CorsConfig::default();

        // Only override allowed_origins, keep other values
        let other = CorsConfig {
            allowed_origins: vec!["https://example.com".to_string()],
            allowed_methods: vec![],  // Empty should not override
            max_age_seconds: 0,  // Zero should not override
        };

        base.merge(&other);

        assert_eq!(base.allowed_origins, vec!["https://example.com"]);  // Updated
        assert_eq!(base.allowed_methods, vec!["GET", "POST", "DELETE", "OPTIONS"]);  // Kept original
        assert_eq!(base.max_age_seconds, 3600);  // Kept original
    }
}
