//! Build script for generating shell completion files
//!
//! This script runs during the build process to generate shell completion
//! scripts for bash, zsh, and fish. The completion scripts are written to
//! the completions/ directory in the project root.

use clap::CommandFactory;
use clap_complete::{Shell, generate_to};
use std::env;
use std::path::PathBuf;

// Import the CLI structure from the binary crate
include!("src/cli/commands.rs");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only generate completions during release builds to speed up dev builds
    let _profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    // Get the output directory for completions
    let out_dir = env::var_os("OUT_DIR").map(PathBuf::from);

    // Use project root completions directory for easier installation
    let project_root = env::var("CARGO_MANIFEST_DIR")?;
    let completions_dir = PathBuf::from(&project_root)
        .parent()
        .unwrap()
        .join("completions");

    // Create completions directory if it doesn't exist
    std::fs::create_dir_all(&completions_dir)?;

    println!(
        "cargo:warning=Generating shell completions to {}",
        completions_dir.display()
    );

    let mut cmd = Cli::command();

    // Generate completion scripts for each supported shell
    for shell in [Shell::Bash, Shell::Zsh, Shell::Fish] {
        let path = generate_to(shell, &mut cmd, "engram", &completions_dir)?;
        println!(
            "cargo:warning=Generated {shell:?} completion: {}",
            path.display()
        );
    }

    // Also generate to OUT_DIR if available (for packaging)
    if let Some(out) = out_dir {
        for shell in [Shell::Bash, Shell::Zsh, Shell::Fish] {
            generate_to(shell, &mut cmd, "engram", &out)?;
        }
    }

    Ok(())
}
