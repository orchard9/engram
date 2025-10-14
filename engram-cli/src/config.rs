use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

const DEFAULT_CONFIG: &str = include_str!("../config/default.toml");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    #[serde(default)]
    pub feature_flags: FeatureFlags,
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

const fn default_spreading_api_beta() -> bool {
    true
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
pub fn format_sections(config: &CliConfig) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push("[feature_flags]".to_string());
    lines.extend(format_feature_flags(&config.feature_flags));
    lines
}
