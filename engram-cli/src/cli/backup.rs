//! Backup command implementations

#![allow(missing_docs)]

use crate::output::{OperationProgress, TableBuilder, format_bytes, spinner};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    pub backup_type: String,
    pub space_id: String,
    pub timestamp: String,
    pub size_bytes: u64,
    pub verification_status: String,
    pub file_path: String,
}

/// Create a backup (full or incremental)
pub fn create_backup(
    backup_type: &str,
    space: &str,
    output: Option<PathBuf>,
    compression: u8,
    show_progress: bool,
) -> Result<()> {
    let script_path = match backup_type {
        "full" => "/usr/local/bin/backup_full.sh",
        "incremental" => "/usr/local/bin/backup_incremental.sh",
        _ => anyhow::bail!("Invalid backup type: {}", backup_type),
    };

    // Check if script exists, fall back to local scripts directory
    let script_path = if Path::new(script_path).exists() {
        script_path.to_string()
    } else {
        let local_script = format!(
            "{}/scripts/backup_{}.sh",
            std::env::var("ENGRAM_ROOT").unwrap_or_else(|_| ".".to_string()),
            backup_type
        );
        if !Path::new(&local_script).exists() {
            anyhow::bail!(
                "Backup script not found at {} or {}",
                script_path,
                local_script
            );
        }
        local_script
    };

    println!("Creating {} backup for space: {}", backup_type, space);

    let mut cmd = Command::new("bash");
    cmd.arg(&script_path);

    // Set environment variables
    cmd.env("ENGRAM_SPACE_ID", space);
    if let Some(output_dir) = output {
        cmd.env("BACKUP_DIR", output_dir);
    }
    cmd.env("ZSTD_LEVEL", compression.to_string());

    if show_progress {
        let progress = OperationProgress::new("Backup", 100);
        progress.set_message("Running backup script");

        let output = cmd.output().context("Failed to execute backup script")?;

        progress.finish("Backup complete");

        return process_backup_output(&output, backup_type);
    }

    let output = cmd.output().context("Failed to execute backup script")?;
    process_backup_output(&output, backup_type)
}

fn process_backup_output(output: &std::process::Output, _backup_type: &str) -> Result<()> {
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse output for backup location and size
        let backup_file = stdout
            .lines()
            .find(|line| line.contains("Location:"))
            .and_then(|line| line.split(':').nth(1))
            .map(str::trim);

        let backup_size = stdout
            .lines()
            .find(|line| line.contains("Size:"))
            .and_then(|line| line.split(':').nth(1))
            .map(str::trim);

        println!("Backup created successfully");
        if let Some(location) = backup_file {
            println!("  Location: {}", location);
        }
        if let Some(size) = backup_size {
            println!("  Size: {}", size);
        }

        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Backup failed: {}", stderr)
    }
}

/// List available backups
pub fn list_backups(backup_type: Option<&str>, space: Option<&str>, format: &str) -> Result<()> {
    let backup_dir =
        std::env::var("BACKUP_DIR").unwrap_or_else(|_| "/var/backups/engram".to_string());

    // Read manifest files
    let manifests = discover_backups(&backup_dir, backup_type, space)?;

    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&manifests)?);
        }
        "table" => {
            let mut table = TableBuilder::new(vec![
                "Type".to_string(),
                "Space".to_string(),
                "Timestamp".to_string(),
                "Size".to_string(),
                "Status".to_string(),
            ]);

            for manifest in manifests {
                table.add_row(vec![
                    manifest.backup_type.clone(),
                    manifest.space_id.clone(),
                    manifest.timestamp.clone(),
                    format_bytes(manifest.size_bytes),
                    manifest.verification_status.clone(),
                ]);
            }

            table.render(&mut std::io::stdout())?;
        }
        "compact" => {
            for manifest in manifests {
                println!(
                    "{} {} {} {}",
                    manifest.backup_type,
                    manifest.space_id,
                    manifest.timestamp,
                    format_bytes(manifest.size_bytes)
                );
            }
        }
        _ => anyhow::bail!("Invalid format: {}", format),
    }

    Ok(())
}

/// Verify backup integrity
pub fn verify_backup(backup_file: &Path, level: &str, verbose: bool) -> Result<()> {
    println!("Verifying backup: {}", backup_file.display());
    println!(
        "Verification level: {} ({})",
        level,
        level_description(level)
    );

    let verify_script = "/usr/local/bin/verify_backup.sh";
    let verify_script = if Path::new(verify_script).exists() {
        verify_script.to_string()
    } else {
        let local_script = format!(
            "{}/scripts/verify_backup.sh",
            std::env::var("ENGRAM_ROOT").unwrap_or_else(|_| ".".to_string())
        );
        if !Path::new(&local_script).exists() {
            anyhow::bail!("Verification script not found");
        }
        local_script
    };

    let spinner_obj = spinner("Verifying backup integrity");

    let output = Command::new("bash")
        .arg(&verify_script)
        .arg(backup_file)
        .arg(level)
        .output()
        .context("Failed to execute verification script")?;

    spinner_obj.finish_with_message("Verification complete");

    if verbose {
        println!("\n{}", String::from_utf8_lossy(&output.stdout));
    }

    if output.status.success() {
        println!("Verification PASSED");
        Ok(())
    } else {
        println!("Verification FAILED");
        if !verbose {
            println!("{}", String::from_utf8_lossy(&output.stderr));
        }
        anyhow::bail!("Backup verification failed")
    }
}

/// Prune old backups according to retention policy
pub fn prune_backups(
    dry_run: bool,
    daily: usize,
    weekly: usize,
    monthly: usize,
    auto_confirm: bool,
) -> Result<()> {
    println!("Backup pruning with retention policy:");
    println!("  Daily: {} days", daily);
    println!("  Weekly: {} weeks", weekly);
    println!("  Monthly: {} months", monthly);

    if dry_run {
        println!("\nDRY RUN - No files will be deleted\n");
    } else if !auto_confirm {
        use crate::interactive::confirm;
        if !confirm("This will permanently delete old backups. Continue?", false)? {
            println!("Aborted.");
            return Ok(());
        }
    }

    let prune_script = "/usr/local/bin/prune_backups.sh";
    let prune_script = if Path::new(prune_script).exists() {
        prune_script.to_string()
    } else {
        // For now, implement simple pruning logic
        println!("Prune script not found, using built-in logic");
        return prune_backups_builtin(dry_run, daily, weekly, monthly);
    };

    let output = Command::new("bash")
        .arg(&prune_script)
        .env("DRY_RUN", if dry_run { "true" } else { "false" })
        .env("BACKUP_RETENTION_DAILY", daily.to_string())
        .env("BACKUP_RETENTION_WEEKLY", weekly.to_string())
        .env("BACKUP_RETENTION_MONTHLY", monthly.to_string())
        .output()
        .context("Failed to execute prune script")?;

    println!("{}", String::from_utf8_lossy(&output.stdout));

    if output.status.success() {
        Ok(())
    } else {
        anyhow::bail!(
            "Pruning failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
    }
}

fn level_description(level: &str) -> &'static str {
    match level {
        "L1" => "Quick manifest check",
        "L2" => "File checksums",
        "L3" => "Deep structure validation",
        "L4" => "Full restore test",
        _ => "Unknown",
    }
}

fn discover_backups(
    backup_dir: &str,
    backup_type: Option<&str>,
    space: Option<&str>,
) -> Result<Vec<BackupManifest>> {
    let mut manifests = Vec::new();

    // Check if backup directory exists
    let dir_path = Path::new(backup_dir);
    if !dir_path.exists() {
        return Ok(manifests);
    }

    // Read directory entries
    for entry in std::fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        // Look for backup files (tar.zst)
        if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
            if filename.ends_with(".tar.zst") || filename.ends_with(".manifest.json") {
                // Try to parse manifest
                if let Some(manifest) = parse_backup_manifest(&path)? {
                    // Apply filters
                    if let Some(bt) = backup_type {
                        if manifest.backup_type != bt {
                            continue;
                        }
                    }
                    if let Some(sp) = space {
                        if manifest.space_id != sp {
                            continue;
                        }
                    }
                    manifests.push(manifest);
                }
            }
        }
    }

    // Sort by timestamp (newest first)
    manifests.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    Ok(manifests)
}

fn parse_backup_manifest(path: &Path) -> Result<Option<BackupManifest>> {
    // If it's a manifest file, read it directly
    if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
        if filename.ends_with(".manifest.json") {
            let content = std::fs::read_to_string(path)?;
            let manifest: BackupManifest = serde_json::from_str(&content)?;
            return Ok(Some(manifest));
        }

        // If it's a backup file, construct manifest from filename
        if filename.ends_with(".tar.zst") {
            // Example: engram-full-default-20240101-120000.tar.zst
            let parts: Vec<&str> = filename.split('-').collect();
            if parts.len() >= 5 {
                let backup_type = parts[1].to_string();
                let space_id = parts[2].to_string();
                let timestamp = format!("{}-{}", parts[3], parts[4].trim_end_matches(".tar.zst"));

                let size_bytes = std::fs::metadata(path)?.len();

                return Ok(Some(BackupManifest {
                    backup_type,
                    space_id,
                    timestamp,
                    size_bytes,
                    verification_status: "unknown".to_string(),
                    file_path: path.to_string_lossy().to_string(),
                }));
            }
        }
    }

    Ok(None)
}

fn prune_backups_builtin(dry_run: bool, daily: usize, weekly: usize, monthly: usize) -> Result<()> {
    let backup_dir =
        std::env::var("BACKUP_DIR").unwrap_or_else(|_| "/var/backups/engram".to_string());
    let manifests = discover_backups(&backup_dir, None, None)?;

    println!("Found {} backups", manifests.len());
    println!(
        "Would keep: {} daily, {} weekly, {} monthly",
        daily, weekly, monthly
    );

    // Simple logic: keep the most recent N backups
    let keep_count = daily + weekly + monthly;
    let to_delete: Vec<_> = manifests.iter().skip(keep_count).collect();

    if to_delete.is_empty() {
        println!("No backups to prune");
        return Ok(());
    }

    println!("\nBackups to delete: {}", to_delete.len());
    for manifest in &to_delete {
        println!(
            "  - {} ({})",
            manifest.file_path,
            format_bytes(manifest.size_bytes)
        );
    }

    if !dry_run {
        for manifest in to_delete {
            println!("Deleting: {}", manifest.file_path);
            std::fs::remove_file(&manifest.file_path)?;
        }
        println!("Pruning complete");
    }

    Ok(())
}
