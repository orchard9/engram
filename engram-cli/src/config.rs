use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use engram_core::MemorySpaceId;
use serde::{Deserialize, Serialize};

const DEFAULT_CONFIG: &str = include_str!("../config/default.toml");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    #[serde(default)]
    pub feature_flags: FeatureFlags,
    #[serde(default)]
    pub memory_spaces: MemorySpacesConfig,
    #[serde(default)]
    pub persistence: PersistenceConfig,
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
