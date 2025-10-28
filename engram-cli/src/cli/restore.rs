//! Restore command implementations

use crate::output::{OperationProgress, spinner};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Restore from full backup
pub fn restore_full(
    backup_file: &Path,
    target_dir: Option<PathBuf>,
    show_progress: bool,
) -> Result<()> {
    println!("Restoring from full backup: {}", backup_file.display());

    if !backup_file.exists() {
        anyhow::bail!("Backup file not found: {}", backup_file.display());
    }

    let restore_script = get_restore_script()?;

    let mut cmd = Command::new("bash");
    cmd.arg(&restore_script);
    cmd.arg(backup_file);

    if let Some(target) = target_dir {
        cmd.env("RESTORE_TARGET", target);
    }

    if show_progress {
        let progress = OperationProgress::new("Restore", 100);
        progress.set_message("Extracting backup archive");

        let output = cmd.output().context("Failed to execute restore script")?;

        progress.finish("Restore complete");

        if output.status.success() {
            println!("Restore completed successfully");
            println!("{}", String::from_utf8_lossy(&output.stdout));
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Restore failed: {}", stderr)
    }

    let output = cmd.output().context("Failed to execute restore script")?;

    if output.status.success() {
        println!("Restore completed successfully");
        println!("{}", String::from_utf8_lossy(&output.stdout));
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Restore failed: {}", stderr)
    }
}

/// Apply incremental backup
pub fn restore_incremental(backup_file: &Path, show_progress: bool) -> Result<()> {
    println!("Applying incremental backup: {}", backup_file.display());

    if !backup_file.exists() {
        anyhow::bail!("Backup file not found: {}", backup_file.display());
    }

    let restore_script = get_restore_script()?;

    let mut cmd = Command::new("bash");
    cmd.arg(&restore_script);
    cmd.arg(backup_file);
    cmd.env("RESTORE_MODE", "incremental");

    if show_progress {
        let progress = OperationProgress::new("Restore", 100);
        progress.set_message("Applying incremental changes");

        let output = cmd.output().context("Failed to execute restore script")?;

        progress.finish("Incremental restore complete");

        if output.status.success() {
            println!("Incremental restore completed successfully");
            println!("{}", String::from_utf8_lossy(&output.stdout));
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Incremental restore failed: {}", stderr)
    }

    let output = cmd.output().context("Failed to execute restore script")?;

    if output.status.success() {
        println!("Incremental restore completed successfully");
        println!("{}", String::from_utf8_lossy(&output.stdout));
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Incremental restore failed: {}", stderr)
    }
}

/// Point-in-time recovery
pub fn restore_pitr(timestamp: &str, target_dir: Option<PathBuf>) -> Result<()> {
    println!("Point-in-time recovery to: {}", timestamp);

    // Validate timestamp format (YYYY-MM-DD HH:MM:SS or similar)
    if !is_valid_timestamp(timestamp) {
        anyhow::bail!("Invalid timestamp format. Expected: YYYY-MM-DD HH:MM:SS or YYYYMMDD-HHMMSS");
    }

    let pitr_script = get_pitr_script()?;

    let spinner_obj = spinner("Searching for backups and transaction logs");

    let mut cmd = Command::new("bash");
    cmd.arg(&pitr_script);
    cmd.arg(timestamp);

    if let Some(target) = target_dir {
        cmd.env("RESTORE_TARGET", target);
    }

    let output = cmd.output().context("Failed to execute PITR script")?;

    spinner_obj.finish_with_message("PITR complete");

    if output.status.success() {
        println!("Point-in-time recovery completed successfully");
        println!("{}", String::from_utf8_lossy(&output.stdout));
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("PITR failed: {}", stderr)
    }
}

/// Verify restore without actually applying it
pub fn verify_restore(backup_file: &Path) -> Result<()> {
    println!("Verifying restore (dry-run): {}", backup_file.display());

    if !backup_file.exists() {
        anyhow::bail!("Backup file not found: {}", backup_file.display());
    }

    let restore_script = get_restore_script()?;

    let spinner_obj = spinner("Testing restore process");

    let output = Command::new("bash")
        .arg(&restore_script)
        .arg(backup_file)
        .env("DRY_RUN", "true")
        .output()
        .context("Failed to execute restore verification")?;

    spinner_obj.finish_with_message("Verification complete");

    if output.status.success() {
        println!("Restore verification PASSED");
        println!("The backup can be restored successfully");
        println!("\n{}", String::from_utf8_lossy(&output.stdout));
        Ok(())
    } else {
        println!("Restore verification FAILED");
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Verification failed: {}", stderr)
    }
}

fn get_restore_script() -> Result<String> {
    let script_path = "/usr/local/bin/restore.sh";
    if Path::new(script_path).exists() {
        return Ok(script_path.to_string());
    }

    let local_script = format!(
        "{}/scripts/restore.sh",
        std::env::var("ENGRAM_ROOT").unwrap_or_else(|_| ".".to_string())
    );
    if Path::new(&local_script).exists() {
        return Ok(local_script);
    }

    anyhow::bail!(
        "Restore script not found at {} or {}",
        script_path,
        local_script
    )
}

fn get_pitr_script() -> Result<String> {
    let script_path = "/usr/local/bin/restore_pitr.sh";
    if Path::new(script_path).exists() {
        return Ok(script_path.to_string());
    }

    let local_script = format!(
        "{}/scripts/restore_pitr.sh",
        std::env::var("ENGRAM_ROOT").unwrap_or_else(|_| ".".to_string())
    );
    if Path::new(&local_script).exists() {
        return Ok(local_script);
    }

    anyhow::bail!(
        "PITR script not found at {} or {}",
        script_path,
        local_script
    )
}

fn is_valid_timestamp(timestamp: &str) -> bool {
    // Accept formats: YYYY-MM-DD HH:MM:SS, YYYYMMDD-HHMMSS, or similar
    let patterns = [
        // YYYY-MM-DD HH:MM:SS
        regex::Regex::new(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").ok(),
        // YYYYMMDD-HHMMSS
        regex::Regex::new(r"^\d{8}-\d{6}$").ok(),
        // ISO 8601
        regex::Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}").ok(),
    ];

    patterns
        .iter()
        .any(|pattern| pattern.as_ref().is_some_and(|p| p.is_match(timestamp)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_validation() {
        // Valid formats
        assert!(is_valid_timestamp("2024-01-15 12:30:45"));
        assert!(is_valid_timestamp("20240115-123045"));
        assert!(is_valid_timestamp("2024-01-15T12:30:45"));

        // Invalid formats
        assert!(!is_valid_timestamp("invalid"));
        assert!(!is_valid_timestamp("2024-01-15")); // Missing time
        assert!(!is_valid_timestamp("12:30:45")); // Missing date

        // Note: This function validates format only, not date semantics
        // So "2024-13-01" matches the format pattern even though month 13 is invalid
    }
}
