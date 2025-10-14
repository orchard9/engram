#![allow(missing_docs)]
use engram_cli::config::{CliConfig, ConfigManager};

#[test]
fn default_spreading_flag_is_enabled() {
    let config = CliConfig::default();
    assert!(config.feature_flags.spreading_api_beta);
}

#[test]
fn toggling_spreading_flag_persists() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let config_path = temp_dir.path().join("engram").join("config.toml");

    // Initially loads default (true)
    let mut manager = ConfigManager::load_with_path(&config_path).expect("load default config");
    assert!(manager.config().feature_flags.spreading_api_beta);

    manager
        .set("feature_flags.spreading_api_beta", "false")
        .expect("set flag");
    manager.save().expect("save config");

    // Reload and ensure the flag stayed disabled
    let reloaded = ConfigManager::load_with_path(&config_path).expect("reload config");
    assert!(!reloaded.config().feature_flags.spreading_api_beta);
}
