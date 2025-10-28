//! Diagnostic command implementations

use crate::output::spinner;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Run comprehensive health check
pub fn run_health_check(output_file: Option<&PathBuf>, strict_mode: bool) -> Result<()> {
    println!("Running comprehensive health diagnostic...\n");

    let diagnose_script = get_diagnose_script()?;

    let spinner_obj = spinner("Checking system health");

    let output_path = output_file
        .and_then(|p| p.to_str())
        .unwrap_or("/dev/stdout");

    let output = Command::new("bash")
        .arg(&diagnose_script)
        .arg(output_path)
        .output()
        .context("Failed to execute health check script")?;

    spinner_obj.finish_and_clear();

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);

    // Parse diagnostic results
    let has_critical = stdout.contains("CRITICAL") || stdout.contains("✗");
    let has_warnings = stdout.contains("WARNING") || stdout.contains("⚠");

    if has_critical {
        anyhow::bail!("Health check found CRITICAL issues")
    }

    if strict_mode && has_warnings {
        anyhow::bail!("Health check found warnings (strict mode)")
    }

    println!("\nHealth check complete");
    if let Some(path) = output_file {
        println!("Report saved to: {}", path.display());
    }

    Ok(())
}

/// Collect debug bundle for support
pub fn collect_debug_bundle(include_dumps: bool, log_lines: usize) -> Result<()> {
    println!("Collecting debug information...");
    println!("This may take 30-60 seconds\n");

    let collect_script = get_collect_script();

    let spinner_obj = spinner("Gathering system information");

    let mut cmd = Command::new("bash");
    cmd.arg(&collect_script);

    if include_dumps {
        cmd.env("INCLUDE_DUMPS", "true");
    }
    cmd.env("LOG_LINES", log_lines.to_string());

    let output = cmd
        .output()
        .context("Failed to execute debug collection script")?;

    spinner_obj.finish_with_message("Debug bundle created");

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Extract bundle filename
        let bundle_file = stdout
            .lines()
            .find(|line| line.contains("Debug bundle created:") || line.contains("Bundle:"))
            .and_then(|line| line.split(':').nth(1))
            .map(str::trim);

        if let Some(bundle) = bundle_file {
            println!("\nDebug bundle: {}", bundle);
            println!("\nNext steps:");
            println!("1. Review the bundle for sensitive information");
            println!("2. Upload to support via secure channel");
            println!("3. Reference your support ticket number");
        } else {
            println!("\n{}", stdout);
        }

        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Debug collection failed: {}", stderr)
    }
}

/// Analyze logs for patterns and errors
pub fn analyze_logs(
    log_file: Option<&PathBuf>,
    time_window: &str,
    severity_filter: Option<&str>,
) -> Result<()> {
    println!("Analyzing logs for patterns and errors...");
    println!("Time window: {}", time_window);
    if let Some(severity) = severity_filter {
        println!("Severity filter: {}", severity);
    }
    println!();

    let analyze_script = get_analyze_logs_script();

    let spinner_obj = spinner("Parsing log entries");

    let mut cmd = Command::new("bash");
    cmd.arg(&analyze_script);

    if let Some(file) = log_file {
        cmd.arg(file);
    }
    cmd.env("TIME_WINDOW", time_window);
    if let Some(sev) = severity_filter {
        cmd.env("SEVERITY_FILTER", sev);
    }

    let output = cmd
        .output()
        .context("Failed to execute log analysis script")?;

    spinner_obj.finish_and_clear();

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Log analysis failed: {}", stderr)
    }
}

/// Emergency recovery procedures
pub fn emergency_recovery(scenario: &str, auto_mode: bool) -> Result<()> {
    let scenario_name = match scenario {
        "corruption" => "data corruption",
        "oom" => "out of memory",
        "deadlock" => "process deadlock",
        "disk-full" => "disk full",
        _ => scenario,
    };

    println!("EMERGENCY RECOVERY MODE");
    println!("Scenario: {}", scenario_name);
    println!();

    if !auto_mode {
        use crate::interactive::confirm;
        println!("This will perform automated recovery steps that may:");
        println!("- Restart the Engram service");
        println!("- Restore from backup");
        println!("- Delete temporary files");
        println!();

        if !confirm("Continue with emergency recovery?", false)? {
            println!("Aborted.");
            return Ok(());
        }
    }

    let recovery_script = get_emergency_recovery_script()?;

    let output = Command::new("bash")
        .arg(&recovery_script)
        .arg(scenario)
        .env("AUTO_MODE", if auto_mode { "true" } else { "false" })
        .output()
        .context("Failed to execute emergency recovery script")?;

    println!("{}", String::from_utf8_lossy(&output.stdout));

    if output.status.success() {
        println!("\nEmergency recovery completed");
        println!("Verify service health with: engram diagnose health");
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Emergency recovery failed: {}", stderr)
    }
}

fn get_diagnose_script() -> Result<String> {
    let script_path = "/usr/local/bin/diagnose_health.sh";
    if Path::new(script_path).exists() {
        return Ok(script_path.to_string());
    }

    let local_script = format!(
        "{}/scripts/diagnose_health.sh",
        std::env::var("ENGRAM_ROOT").unwrap_or_else(|_| ".".to_string())
    );
    if Path::new(&local_script).exists() {
        return Ok(local_script);
    }

    anyhow::bail!(
        "Diagnostic script not found at {} or {}",
        script_path,
        local_script
    )
}

fn get_collect_script() -> String {
    let script_path = "/usr/local/bin/collect_debug_info.sh";
    if Path::new(script_path).exists() {
        return script_path.to_string();
    }

    let local_script = format!(
        "{}/scripts/collect_debug_info.sh",
        std::env::var("ENGRAM_ROOT").unwrap_or_else(|_| ".".to_string())
    );
    if Path::new(&local_script).exists() {
        return local_script;
    }

    // Fallback: create a minimal debug collection
    println!("Warning: Debug collection script not found, using minimal collection");
    "echo".to_string() // Will output minimal info
}

fn get_analyze_logs_script() -> String {
    let script_path = "/usr/local/bin/analyze_logs.sh";
    if Path::new(script_path).exists() {
        return script_path.to_string();
    }

    let local_script = format!(
        "{}/scripts/analyze_logs.sh",
        std::env::var("ENGRAM_ROOT").unwrap_or_else(|_| ".".to_string())
    );
    if Path::new(&local_script).exists() {
        return local_script;
    }

    // Fallback: use grep for basic log analysis
    println!("Warning: Log analysis script not found, using basic grep");
    "grep".to_string()
}

fn get_emergency_recovery_script() -> Result<String> {
    let script_path = "/usr/local/bin/emergency_recovery.sh";
    if Path::new(script_path).exists() {
        return Ok(script_path.to_string());
    }

    let local_script = format!(
        "{}/scripts/emergency_recovery.sh",
        std::env::var("ENGRAM_ROOT").unwrap_or_else(|_| ".".to_string())
    );
    if Path::new(&local_script).exists() {
        return Ok(local_script);
    }

    anyhow::bail!(
        "Emergency recovery script not found at {} or {}",
        script_path,
        local_script
    )
}
