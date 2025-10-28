# Task 012: Operations CLI Enhancement — pending

**Priority:** P1-P2
**Estimated Effort:** 2 days
**Dependencies:** Tasks 002 (Backup/Restore), 005 (Troubleshooting)

## Objective

Enhance the engram-cli with production operations commands that integrate backup, restore, diagnostics, and monitoring capabilities. Transform the CLI from a development tool into a production operations powerhouse with rich output formatting, interactive workflows, and shell completion support.

## Context: Production CLI Requirements

Current CLI (`engram-cli/src/main.rs`, `engram-cli/src/cli/`) provides:
- Server management (start/stop/status)
- Memory operations (create/get/search/list/delete)
- Space management (list/create)
- Configuration management (get/set/list/path)
- Interactive shell mode
- Documentation viewing (docs command)
- Query with probabilistic confidence intervals

Missing production operations:
- Backup/restore operations (Task 002 scripts need CLI integration)
- Diagnostic commands (Task 005 health scripts need CLI wrapper)
- Migration tooling (Task 007 migration commands)
- Performance benchmarking (currently stubbed)
- Configuration validation
- Interactive operation modes for complex workflows
- Shell completion (bash/zsh/fish)

## Integration Points

**Uses:**
- `/engram-cli/src/main.rs` - Current CLI entry point with command routing
- `/engram-cli/src/cli/commands.rs` - Clap command definitions
- `/engram-cli/src/cli/status.rs` - Status display with table formatting
- `/engram-cli/src/config.rs` - Configuration management
- `/scripts/backup_full.sh` - Full backup script from Task 002
- `/scripts/backup_incremental.sh` - Incremental backup from Task 002
- `/scripts/restore.sh` - Restore script from Task 002
- `/scripts/restore_pitr.sh` - Point-in-time recovery from Task 002
- `/scripts/verify_backup.sh` - Backup verification from Task 002
- `/scripts/diagnose_health.sh` - Health diagnostics from Task 005
- `/scripts/collect_debug_info.sh` - Debug collection from Task 005
- `/chosen_libraries.md` - Approved dependencies (criterion for benchmarks)

**Creates:**
- `/engram-cli/src/cli/backup.rs` - Backup command implementations
- `/engram-cli/src/cli/restore.rs` - Restore command implementations
- `/engram-cli/src/cli/diagnose.rs` - Diagnostic command implementations
- `/engram-cli/src/cli/migrate.rs` - Migration command implementations
- `/engram-cli/src/cli/benchmark.rs` - Enhanced benchmark commands
- `/engram-cli/src/cli/validate.rs` - Validation commands
- `/engram-cli/src/output/table.rs` - Rich table formatting utilities
- `/engram-cli/src/output/progress.rs` - Progress bar implementations
- `/engram-cli/src/interactive.rs` - Interactive workflow helpers
- `/completions/engram.bash` - Bash completion script
- `/completions/engram.zsh` - Zsh completion script
- `/completions/engram.fish` - Fish completion script

**Updates:**
- `/engram-cli/src/cli/commands.rs` - Add new command definitions
- `/engram-cli/src/cli/mod.rs` - Export new modules
- `/engram-cli/src/main.rs` - Wire up new command handlers
- `/docs/reference/cli.md` - Complete CLI reference documentation

## Technical Specifications

### Command Structure Enhancement

**New Command Hierarchy:**
```
engram
├── start            # Start server (existing, enhanced with validation)
├── stop             # Stop server (existing, enhanced with --wait)
├── status           # Status display (existing, enhanced with --space filtering)
├── backup           # NEW: Backup operations
│   ├── create       #   Create full or incremental backup
│   ├── list         #   List available backups
│   ├── verify       #   Verify backup integrity (L1-L4)
│   └── prune        #   Prune old backups
├── restore          # NEW: Restore operations
│   ├── full         #   Restore from full backup
│   ├── incremental  #   Apply incremental backup
│   ├── pitr         #   Point-in-time recovery
│   └── verify-only  #   Test restore without applying
├── diagnose         # NEW: Diagnostic operations
│   ├── health       #   Comprehensive health check
│   ├── collect      #   Collect debug bundle
│   ├── analyze-logs #   Parse and analyze logs
│   └── emergency    #   Emergency recovery procedures
├── migrate          # NEW: Migration operations (Task 007)
│   ├── neo4j        #   Migrate from Neo4j
│   ├── postgresql   #   Migrate from PostgreSQL
│   └── redis        #   Migrate from Redis
├── benchmark        # ENHANCED: Performance benchmarking
│   ├── latency      #   Measure operation latency
│   ├── throughput   #   Measure throughput
│   ├── spreading    #   Benchmark spreading activation
│   └── consolidation # Benchmark memory consolidation
├── validate         # NEW: Validation operations
│   ├── config       #   Validate configuration file
│   ├── data         #   Validate data integrity
│   └── deployment   #   Pre-deployment validation
├── memory           # Existing memory operations
├── space            # Existing space operations
├── config           # Existing configuration management
├── shell            # Existing interactive shell
├── docs             # Existing documentation viewer
└── query            # Existing probabilistic query
```

### Output Formatting System

**Design Principles:**
- Default to human-readable tables for interactive use
- Provide `--json` flag for machine-readable output
- Use `--quiet` flag for automation scripts (minimal output)
- Progress bars for long-running operations (>2 seconds)
- Color coding with NO_COLOR environment variable support
- Consistent table layouts across commands

**Table Formatting Utility:**
```rust
// /engram-cli/src/output/table.rs

use std::io::Write;
use termion::{color, style};

pub struct TableBuilder {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    column_widths: Vec<usize>,
    use_color: bool,
}

impl TableBuilder {
    pub fn new(headers: Vec<String>) -> Self {
        let use_color = std::env::var("NO_COLOR").is_err();
        let column_widths = headers.iter().map(|h| h.len()).collect();

        Self {
            headers,
            rows: Vec::new(),
            column_widths,
            use_color,
        }
    }

    pub fn add_row(&mut self, row: Vec<String>) {
        // Update column widths based on content
        for (i, cell) in row.iter().enumerate() {
            if let Some(width) = self.column_widths.get_mut(i) {
                *width = (*width).max(cell.len());
            }
        }
        self.rows.push(row);
    }

    pub fn render<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Top border
        self.write_border(writer, '┌', '┬', '┐')?;

        // Headers
        self.write_row(writer, &self.headers, true)?;

        // Header separator
        self.write_border(writer, '├', '┼', '┤')?;

        // Data rows
        for row in &self.rows {
            self.write_row(writer, row, false)?;
        }

        // Bottom border
        self.write_border(writer, '└', '┴', '┘')?;

        Ok(())
    }

    fn write_border<W: Write>(
        &self,
        writer: &mut W,
        left: char,
        mid: char,
        right: char,
    ) -> std::io::Result<()> {
        write!(writer, "{}", left)?;
        for (i, width) in self.column_widths.iter().enumerate() {
            write!(writer, "{}", "─".repeat(*width + 2))?;
            if i < self.column_widths.len() - 1 {
                write!(writer, "{}", mid)?;
            }
        }
        writeln!(writer, "{}", right)
    }

    fn write_row<W: Write>(
        &self,
        writer: &mut W,
        cells: &[String],
        is_header: bool,
    ) -> std::io::Result<()> {
        write!(writer, "│")?;
        for (i, (cell, width)) in cells.iter().zip(&self.column_widths).enumerate() {
            if is_header && self.use_color {
                write!(writer, " {}{}{:<width$}{} │",
                    style::Bold, color::Fg(color::Cyan),
                    cell, style::Reset,
                    width = width
                )?;
            } else {
                write!(writer, " {:<width$} │", cell, width = width)?;
            }
        }
        writeln!(writer)
    }
}
```

**Progress Bar Utility:**
```rust
// /engram-cli/src/output/progress.rs

use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

pub struct OperationProgress {
    bar: ProgressBar,
    operation_name: String,
}

impl OperationProgress {
    pub fn new(operation: &str, total: u64) -> Self {
        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .expect("valid template")
                .progress_chars("=>-"),
        );
        bar.enable_steady_tick(Duration::from_millis(100));

        Self {
            bar,
            operation_name: operation.to_string(),
        }
    }

    pub fn set_message(&self, msg: &str) {
        self.bar.set_message(format!("{}: {}", self.operation_name, msg));
    }

    pub fn inc(&self, delta: u64) {
        self.bar.inc(delta);
    }

    pub fn finish(&self, msg: &str) {
        self.bar.finish_with_message(format!("{}: {}", self.operation_name, msg));
    }
}

pub fn spinner(operation: &str) -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .expect("valid template"),
    );
    spinner.set_message(operation.to_string());
    spinner.enable_steady_tick(Duration::from_millis(80));
    spinner
}
```

### Backup Command Implementation

**Command Definition:**
```rust
// Add to /engram-cli/src/cli/commands.rs

#[derive(Subcommand)]
pub enum BackupAction {
    /// Create a new backup
    Create {
        /// Backup type: full or incremental
        #[arg(short, long, default_value = "full")]
        backup_type: BackupType,

        /// Memory space to backup (or "all")
        #[arg(short, long, default_value = "default")]
        space: String,

        /// Output directory for backup
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Compression level (1-9, default: 3)
        #[arg(short, long, default_value = "3")]
        compression: u8,

        /// Show progress bar
        #[arg(long)]
        progress: bool,
    },

    /// List available backups
    List {
        /// Filter by backup type
        #[arg(short, long)]
        backup_type: Option<BackupType>,

        /// Filter by memory space
        #[arg(short, long)]
        space: Option<String>,

        /// Output format
        #[arg(short, long, default_value = "table")]
        format: OutputFormat,
    },

    /// Verify backup integrity
    Verify {
        /// Backup file path
        backup_file: PathBuf,

        /// Verification level: L1 (manifest), L2 (checksums), L3 (structure), L4 (full restore test)
        #[arg(short, long, default_value = "L2")]
        level: VerificationLevel,

        /// Show detailed output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Prune old backups according to retention policy
    Prune {
        /// Dry run (show what would be deleted without deleting)
        #[arg(long)]
        dry_run: bool,

        /// Retention: daily backups to keep
        #[arg(long, default_value = "7")]
        daily: usize,

        /// Retention: weekly backups to keep
        #[arg(long, default_value = "4")]
        weekly: usize,

        /// Retention: monthly backups to keep
        #[arg(long, default_value = "12")]
        monthly: usize,

        /// Confirm deletion without prompt
        #[arg(short, long)]
        yes: bool,
    },
}

#[derive(Clone, ValueEnum)]
pub enum BackupType {
    Full,
    Incremental,
}

#[derive(Clone, ValueEnum)]
pub enum VerificationLevel {
    L1,  // Manifest check (<1s)
    L2,  // Checksums (30s/GB)
    L3,  // Structure validation (2min/GB)
    L4,  // Full restore test (5min/GB)
}
```

**Implementation:**
```rust
// /engram-cli/src/cli/backup.rs

use crate::output::{OperationProgress, TableBuilder};
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

pub async fn create_backup(
    backup_type: BackupType,
    space: String,
    output: Option<PathBuf>,
    compression: u8,
    show_progress: bool,
) -> Result<()> {
    let script_path = match backup_type {
        BackupType::Full => "/usr/local/bin/backup_full.sh",
        BackupType::Incremental => "/usr/local/bin/backup_incremental.sh",
    };

    println!("Creating {} backup for space: {}",
        match backup_type {
            BackupType::Full => "full",
            BackupType::Incremental => "incremental",
        },
        space
    );

    let progress = if show_progress {
        Some(OperationProgress::new("Backup", 100))
    } else {
        None
    };

    let mut cmd = Command::new("bash");
    cmd.arg(script_path);

    // Set environment variables
    cmd.env("ENGRAM_SPACE_ID", &space);
    if let Some(output_dir) = output {
        cmd.env("BACKUP_DIR", output_dir);
    }
    cmd.env("ZSTD_LEVEL", compression.to_string());

    if let Some(ref prog) = progress {
        prog.set_message("Running backup script");
    }

    let output = cmd.output()
        .context("Failed to execute backup script")?;

    if let Some(ref prog) = progress {
        prog.finish("Backup complete");
    }

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse output for backup location and size
        let backup_file = stdout.lines()
            .find(|line| line.contains("Location:"))
            .and_then(|line| line.split(':').nth(1))
            .map(str::trim);

        let backup_size = stdout.lines()
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

pub async fn list_backups(
    backup_type: Option<BackupType>,
    space: Option<String>,
    format: OutputFormat,
) -> Result<()> {
    let backup_dir = std::env::var("BACKUP_DIR")
        .unwrap_or_else(|_| "/var/backups/engram".to_string());

    // Read manifest files
    let manifests = discover_backups(&backup_dir, backup_type, space)?;

    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&manifests)?);
        }
        OutputFormat::Table => {
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
        OutputFormat::Compact => {
            for manifest in manifests {
                println!("{} {} {} {}",
                    manifest.backup_type,
                    manifest.space_id,
                    manifest.timestamp,
                    format_bytes(manifest.size_bytes)
                );
            }
        }
    }

    Ok(())
}

pub async fn verify_backup(
    backup_file: PathBuf,
    level: VerificationLevel,
    verbose: bool,
) -> Result<()> {
    let level_str = match level {
        VerificationLevel::L1 => "L1",
        VerificationLevel::L2 => "L2",
        VerificationLevel::L3 => "L3",
        VerificationLevel::L4 => "L4",
    };

    println!("Verifying backup: {}", backup_file.display());
    println!("Verification level: {} ({})", level_str, level_description(&level));

    let spinner = crate::output::spinner("Verifying backup integrity");

    let output = Command::new("bash")
        .arg("/usr/local/bin/verify_backup.sh")
        .arg(&backup_file)
        .arg(level_str)
        .output()
        .context("Failed to execute verification script")?;

    spinner.finish_with_message("Verification complete");

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

pub async fn prune_backups(
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
        use std::io::{self, Write};
        print!("\nThis will permanently delete old backups. Continue? [y/N]: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    let output = Command::new("bash")
        .arg("/usr/local/bin/prune_backups.sh")
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
        anyhow::bail!("Pruning failed: {}", String::from_utf8_lossy(&output.stderr))
    }
}

fn level_description(level: &VerificationLevel) -> &'static str {
    match level {
        VerificationLevel::L1 => "Quick manifest check",
        VerificationLevel::L2 => "File checksums",
        VerificationLevel::L3 => "Deep structure validation",
        VerificationLevel::L4 => "Full restore test",
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_idx])
}
```

### Diagnostic Command Implementation

**Command Definition:**
```rust
// Add to /engram-cli/src/cli/commands.rs

#[derive(Subcommand)]
pub enum DiagnoseAction {
    /// Run comprehensive health check
    Health {
        /// Output file for report
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Fail with non-zero exit code on warnings
        #[arg(long)]
        strict: bool,
    },

    /// Collect debug bundle for support
    Collect {
        /// Include memory dumps
        #[arg(long)]
        include_dumps: bool,

        /// Include full logs (last N lines)
        #[arg(long, default_value = "10000")]
        log_lines: usize,
    },

    /// Analyze logs for patterns and errors
    AnalyzeLogs {
        /// Log file path (default: system logs)
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Time window: 1h, 24h, 7d
        #[arg(short, long, default_value = "1h")]
        window: String,

        /// Filter by severity: ERROR, WARN, INFO
        #[arg(short, long)]
        severity: Option<String>,
    },

    /// Emergency recovery procedures
    Emergency {
        /// Recovery scenario: corruption, oom, deadlock
        scenario: EmergencyScenario,

        /// Automatic recovery without prompts
        #[arg(long)]
        auto: bool,
    },
}

#[derive(Clone, ValueEnum)]
pub enum EmergencyScenario {
    Corruption,  // Data corruption detected
    Oom,        // Out of memory condition
    Deadlock,   // Process deadlock
    DiskFull,   // Disk space exhausted
}
```

**Implementation:**
```rust
// /engram-cli/src/cli/diagnose.rs

use crate::output::{TableBuilder, spinner};
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

pub async fn run_health_check(
    output_file: Option<PathBuf>,
    strict_mode: bool,
) -> Result<()> {
    println!("Running comprehensive health diagnostic...\n");

    let spinner = spinner("Checking system health");

    let output = Command::new("bash")
        .arg("/usr/local/bin/diagnose_health.sh")
        .arg(output_file.as_ref()
            .map(|p| p.to_str().unwrap())
            .unwrap_or("/dev/stdout"))
        .output()
        .context("Failed to execute health check script")?;

    spinner.finish_and_clear();

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);

    // Parse diagnostic results
    let has_critical = stdout.contains("✗ CRITICAL");
    let has_warnings = stdout.contains("⚠ WARNING");

    if has_critical {
        anyhow::bail!("Health check found CRITICAL issues")
    }

    if strict_mode && has_warnings {
        anyhow::bail!("Health check found warnings (strict mode)")
    }

    println!("\nHealth check complete");
    if let Some(ref path) = output_file {
        println!("Report saved to: {}", path.display());
    }

    Ok(())
}

pub async fn collect_debug_bundle(
    include_dumps: bool,
    log_lines: usize,
) -> Result<()> {
    println!("Collecting debug information...");
    println!("This may take 30-60 seconds\n");

    let spinner = spinner("Gathering system information");

    let mut cmd = Command::new("bash");
    cmd.arg("/usr/local/bin/collect_debug_info.sh");

    if include_dumps {
        cmd.env("INCLUDE_DUMPS", "true");
    }
    cmd.env("LOG_LINES", log_lines.to_string());

    let output = cmd.output()
        .context("Failed to execute debug collection script")?;

    spinner.finish_with_message("Debug bundle created");

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Extract bundle filename
        let bundle_file = stdout.lines()
            .find(|line| line.contains("Debug bundle created:"))
            .and_then(|line| line.split(':').nth(1))
            .map(str::trim);

        if let Some(bundle) = bundle_file {
            println!("\nDebug bundle: {}", bundle);
            println!("\nNext steps:");
            println!("1. Review the bundle for sensitive information");
            println!("2. Upload to support via secure channel");
            println!("3. Reference your support ticket number");
        }

        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Debug collection failed: {}", stderr)
    }
}

pub async fn analyze_logs(
    log_file: Option<PathBuf>,
    time_window: String,
    severity_filter: Option<String>,
) -> Result<()> {
    println!("Analyzing logs for patterns and errors...");
    println!("Time window: {}", time_window);
    if let Some(ref severity) = severity_filter {
        println!("Severity filter: {}", severity);
    }
    println!();

    let spinner = spinner("Parsing log entries");

    let mut cmd = Command::new("bash");
    cmd.arg("/usr/local/bin/analyze_logs.sh");

    if let Some(ref file) = log_file {
        cmd.arg(file);
    }
    cmd.env("TIME_WINDOW", &time_window);
    if let Some(ref sev) = severity_filter {
        cmd.env("SEVERITY_FILTER", sev);
    }

    let output = cmd.output()
        .context("Failed to execute log analysis script")?;

    spinner.finish_and_clear();

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Log analysis failed: {}", stderr)
    }
}

pub async fn emergency_recovery(
    scenario: EmergencyScenario,
    auto_mode: bool,
) -> Result<()> {
    let scenario_name = match scenario {
        EmergencyScenario::Corruption => "data corruption",
        EmergencyScenario::Oom => "out of memory",
        EmergencyScenario::Deadlock => "process deadlock",
        EmergencyScenario::DiskFull => "disk full",
    };

    println!("EMERGENCY RECOVERY MODE");
    println!("Scenario: {}", scenario_name);
    println!();

    if !auto_mode {
        use std::io::{self, Write};
        println!("This will perform automated recovery steps that may:");
        println!("- Restart the Engram service");
        println!("- Restore from backup");
        println!("- Delete temporary files");
        println!();
        print!("Continue with emergency recovery? [y/N]: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    let scenario_arg = match scenario {
        EmergencyScenario::Corruption => "corruption",
        EmergencyScenario::Oom => "oom",
        EmergencyScenario::Deadlock => "deadlock",
        EmergencyScenario::DiskFull => "disk-full",
    };

    let output = Command::new("bash")
        .arg("/usr/local/bin/emergency_recovery.sh")
        .arg(scenario_arg)
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
```

### Enhanced Benchmark Command

**Current Status:** Basic stub in main.rs that only validates server connection
**Target:** Full-featured benchmarking suite using criterion

```rust
// /engram-cli/src/cli/benchmark.rs

use anyhow::{Context, Result};
use criterion::{black_box, Criterion};
use std::time::Duration;

#[derive(Subcommand)]
pub enum BenchmarkAction {
    /// Measure operation latency (P50, P95, P99)
    Latency {
        /// Operation to benchmark: create, get, search, spreading
        operation: String,

        /// Number of iterations
        #[arg(short, long, default_value = "1000")]
        iterations: usize,

        /// Warmup iterations
        #[arg(short, long, default_value = "100")]
        warmup: usize,
    },

    /// Measure throughput (operations per second)
    Throughput {
        /// Duration to run benchmark
        #[arg(short, long, default_value = "60")]
        duration: u64,

        /// Number of concurrent clients
        #[arg(short, long, default_value = "10")]
        clients: usize,
    },

    /// Benchmark spreading activation performance
    Spreading {
        /// Number of nodes to activate
        #[arg(short, long, default_value = "100")]
        nodes: usize,

        /// Activation spread depth
        #[arg(short, long, default_value = "3")]
        depth: usize,
    },

    /// Benchmark memory consolidation
    Consolidation {
        /// Simulate consolidation load
        #[arg(short, long)]
        load_test: bool,
    },
}

pub async fn run_latency_benchmark(
    operation: String,
    iterations: usize,
    warmup: usize,
) -> Result<()> {
    let (port, _) = crate::cli::server::get_server_connection().await?;

    println!("Benchmarking {} latency", operation);
    println!("Warmup: {} iterations", warmup);
    println!("Measurement: {} iterations\n", iterations);

    let spinner = crate::output::spinner("Running warmup phase");

    // Warmup phase
    for _ in 0..warmup {
        execute_operation(&operation, port).await?;
    }

    spinner.finish_with_message("Warmup complete");

    // Measurement phase
    let progress = crate::output::OperationProgress::new("Benchmark", iterations as u64);
    let mut latencies = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let start = std::time::Instant::now();
        execute_operation(&operation, port).await?;
        let elapsed = start.elapsed();

        latencies.push(elapsed);
        progress.inc(1);

        if i % 100 == 0 {
            progress.set_message(&format!("Completed {}/{}", i, iterations));
        }
    }

    progress.finish("Benchmark complete");

    // Calculate percentiles
    latencies.sort();
    let p50 = latencies[iterations / 2];
    let p95 = latencies[(iterations * 95) / 100];
    let p99 = latencies[(iterations * 99) / 100];
    let min = latencies[0];
    let max = latencies[iterations - 1];

    println!("\nLatency Results:");
    println!("  Min:    {:>10.3} ms", min.as_secs_f64() * 1000.0);
    println!("  P50:    {:>10.3} ms", p50.as_secs_f64() * 1000.0);
    println!("  P95:    {:>10.3} ms", p95.as_secs_f64() * 1000.0);
    println!("  P99:    {:>10.3} ms", p99.as_secs_f64() * 1000.0);
    println!("  Max:    {:>10.3} ms", max.as_secs_f64() * 1000.0);

    Ok(())
}

pub async fn run_throughput_benchmark(
    duration: u64,
    clients: usize,
) -> Result<()> {
    let (port, _) = crate::cli::server::get_server_connection().await?;

    println!("Benchmarking throughput");
    println!("Duration: {} seconds", duration);
    println!("Concurrent clients: {}\n", clients);

    let start = std::time::Instant::now();
    let target_duration = Duration::from_secs(duration);

    use tokio::sync::mpsc;
    let (tx, mut rx) = mpsc::channel(1000);

    // Spawn client tasks
    let mut handles = vec![];
    for _ in 0..clients {
        let tx = tx.clone();
        let handle = tokio::spawn(async move {
            let mut ops = 0u64;
            while start.elapsed() < target_duration {
                // Execute operation
                if execute_operation("create", port).await.is_ok() {
                    ops += 1;
                }
            }
            let _ = tx.send(ops).await;
        });
        handles.push(handle);
    }

    drop(tx);

    // Collect results
    let mut total_ops = 0u64;
    while let Some(ops) = rx.recv().await {
        total_ops += ops;
    }

    // Wait for all tasks
    for handle in handles {
        let _ = handle.await;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let throughput = total_ops as f64 / elapsed;

    println!("\nThroughput Results:");
    println!("  Total operations: {}", total_ops);
    println!("  Duration: {:.2} seconds", elapsed);
    println!("  Throughput: {:.2} ops/sec", throughput);
    println!("  Per-client: {:.2} ops/sec", throughput / clients as f64);

    Ok(())
}

async fn execute_operation(operation: &str, port: u16) -> Result<()> {
    let client = reqwest::Client::new();

    match operation {
        "create" => {
            let _response = client
                .post(format!("http://127.0.0.1:{}/api/v1/memories", port))
                .json(&serde_json::json!({
                    "what": "benchmark test memory",
                    "confidence": 0.9
                }))
                .send()
                .await?;
        }
        "get" => {
            // Get a random memory
            let _response = client
                .get(format!("http://127.0.0.1:{}/api/v1/memories/test-id", port))
                .send()
                .await?;
        }
        "search" => {
            let _response = client
                .get(format!("http://127.0.0.1:{}/api/v1/memories/search", port))
                .query(&[("query", "test"), ("limit", "10")])
                .send()
                .await?;
        }
        _ => anyhow::bail!("Unknown operation: {}", operation),
    }

    Ok(())
}
```

### Configuration Validation Command

```rust
// /engram-cli/src/cli/validate.rs

use anyhow::{Context, Result};
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum ValidateAction {
    /// Validate configuration file
    Config {
        /// Path to config file
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Check deployment-specific settings
        #[arg(short, long)]
        deployment: Option<String>,
    },

    /// Validate data integrity
    Data {
        /// Memory space to validate
        #[arg(short, long, default_value = "default")]
        space: String,

        /// Fix issues automatically
        #[arg(long)]
        fix: bool,
    },

    /// Pre-deployment validation checklist
    Deployment {
        /// Target environment: dev, staging, production
        #[arg(short, long)]
        environment: String,
    },
}

pub async fn validate_config(
    config_file: Option<PathBuf>,
    deployment: Option<String>,
) -> Result<()> {
    let config_path = config_file.unwrap_or_else(|| {
        dirs::config_dir()
            .expect("config dir")
            .join("engram")
            .join("config.toml")
    });

    println!("Validating configuration: {}", config_path.display());

    let output = std::process::Command::new("bash")
        .arg("/usr/local/bin/validate_config.sh")
        .arg(&config_path)
        .env("DEPLOYMENT", deployment.unwrap_or_default())
        .output()
        .context("Failed to execute validation script")?;

    println!("{}", String::from_utf8_lossy(&output.stdout));

    if output.status.success() {
        println!("✓ Configuration is valid");
        Ok(())
    } else {
        anyhow::bail!("Configuration validation failed")
    }
}

pub async fn validate_data(
    space: String,
    auto_fix: bool,
) -> Result<()> {
    println!("Validating data integrity for space: {}", space);

    if auto_fix {
        println!("Auto-fix mode: Issues will be repaired automatically\n");
    }

    let spinner = crate::output::spinner("Scanning data structures");

    let output = std::process::Command::new("bash")
        .arg("/usr/local/bin/validate_data.sh")
        .arg(&space)
        .env("AUTO_FIX", if auto_fix { "true" } else { "false" })
        .output()
        .context("Failed to execute data validation")?;

    spinner.finish_and_clear();

    println!("{}", String::from_utf8_lossy(&output.stdout));

    if output.status.success() {
        Ok(())
    } else {
        anyhow::bail!("Data validation found issues")
    }
}
```

### Shell Completion Support

**Generate completions during build:**
```rust
// /engram-cli/build.rs (new file)

use clap::CommandFactory;
use clap_complete::{generate_to, Shell};
use std::env;
use std::io::Error;

include!("src/cli/commands.rs");

fn main() -> Result<(), Error> {
    let outdir = match env::var_os("OUT_DIR") {
        None => return Ok(()),
        Some(outdir) => outdir,
    };

    let mut cmd = Cli::command();

    // Generate completion scripts
    for &shell in &[Shell::Bash, Shell::Zsh, Shell::Fish] {
        let path = generate_to(shell, &mut cmd, "engram", outdir)?;
        println!("cargo:warning=completion file generated: {:?}", path);
    }

    Ok(())
}
```

**Installation script:**
```bash
# /scripts/install_completions.sh

#!/bin/bash
# Install shell completion scripts

set -euo pipefail

SHELL_TYPE="${1:-detect}"

detect_shell() {
    case "$SHELL" in
        */bash) echo "bash" ;;
        */zsh) echo "zsh" ;;
        */fish) echo "fish" ;;
        *) echo "unknown" ;;
    esac
}

if [ "$SHELL_TYPE" = "detect" ]; then
    SHELL_TYPE=$(detect_shell)
fi

case "$SHELL_TYPE" in
    bash)
        COMPLETION_DIR="${BASH_COMPLETION_USER_DIR:-$HOME/.local/share/bash-completion/completions}"
        mkdir -p "$COMPLETION_DIR"
        cp completions/engram.bash "$COMPLETION_DIR/engram"
        echo "Bash completion installed to: $COMPLETION_DIR/engram"
        echo "Restart your shell or run: source $COMPLETION_DIR/engram"
        ;;

    zsh)
        COMPLETION_DIR="${ZSH_COMPLETION_DIR:-$HOME/.zsh/completion}"
        mkdir -p "$COMPLETION_DIR"
        cp completions/engram.zsh "$COMPLETION_DIR/_engram"
        echo "Zsh completion installed to: $COMPLETION_DIR/_engram"
        echo "Restart your shell or run: autoload -U compinit && compinit"
        ;;

    fish)
        COMPLETION_DIR="${FISH_COMPLETION_DIR:-$HOME/.config/fish/completions}"
        mkdir -p "$COMPLETION_DIR"
        cp completions/engram.fish "$COMPLETION_DIR/engram.fish"
        echo "Fish completion installed to: $COMPLETION_DIR/engram.fish"
        ;;

    *)
        echo "ERROR: Unknown or unsupported shell: $SHELL_TYPE"
        echo "Supported: bash, zsh, fish"
        exit 1
        ;;
esac
```

### Interactive Workflows

**Confirmation Prompts:**
```rust
// /engram-cli/src/interactive.rs

use anyhow::Result;
use std::io::{self, Write};

pub fn confirm(message: &str, default_yes: bool) -> Result<bool> {
    let prompt = if default_yes {
        format!("{} [Y/n]: ", message)
    } else {
        format!("{} [y/N]: ", message)
    };

    print!("{}", prompt);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    let input = input.trim();
    if input.is_empty() {
        return Ok(default_yes);
    }

    Ok(input.eq_ignore_ascii_case("y") || input.eq_ignore_ascii_case("yes"))
}

pub fn prompt(message: &str) -> Result<String> {
    print!("{}: ", message);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    Ok(input.trim().to_string())
}

pub fn select_from_list<T: std::fmt::Display>(
    message: &str,
    options: &[T],
) -> Result<usize> {
    println!("{}", message);
    for (i, option) in options.iter().enumerate() {
        println!("  {}. {}", i + 1, option);
    }

    loop {
        print!("Select option [1-{}]: ", options.len());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if let Ok(choice) = input.trim().parse::<usize>() {
            if choice > 0 && choice <= options.len() {
                return Ok(choice - 1);
            }
        }

        println!("Invalid selection. Please try again.");
    }
}
```

**Dry-run Mode:**
```rust
// Add to command handlers

pub struct OperationContext {
    pub dry_run: bool,
    pub verbose: bool,
    pub quiet: bool,
}

impl OperationContext {
    pub fn execute<F, R>(&self, operation: &str, f: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
        R: Default,
    {
        if self.dry_run {
            if !self.quiet {
                println!("[DRY RUN] Would execute: {}", operation);
            }
            return Ok(R::default());
        }

        if self.verbose {
            println!("Executing: {}", operation);
        }

        f()
    }
}
```

## Dependencies

Add to `/engram-cli/Cargo.toml`:
```toml
[dependencies]
# Existing dependencies...
indicatif = "0.17"          # Progress bars and spinners
termion = "2.0"             # Terminal colors and formatting
clap_complete = "4.4"       # Shell completion generation

[build-dependencies]
clap = { version = "4.4", features = ["derive"] }
clap_complete = "4.4"
```

## Documentation Requirements

### /docs/reference/cli.md Updates

**Add sections:**
1. Operations Commands - backup, restore, diagnose, validate
2. Output Formats - JSON, table, compact
3. Interactive Mode - Confirmation prompts, selections
4. Shell Completion - Installation and usage
5. Examples - Common workflows and scenarios
6. Automation - Using CLI in scripts with --quiet and --json

## Acceptance Criteria

**Command Coverage:**
- [ ] All Task 002 backup scripts accessible via `engram backup` subcommands
- [ ] All Task 005 diagnostic scripts accessible via `engram diagnose` subcommands
- [ ] Restore operations with progress bars for files >100MB
- [ ] PITR command with timestamp validation and error messages
- [ ] Backup verification with all 4 levels (L1-L4)
- [ ] Backup pruning with dry-run mode and confirmation prompts

**Output Formatting:**
- [ ] Rich tables with Unicode box-drawing characters
- [ ] Progress bars for operations >2 seconds
- [ ] Spinners for indeterminate operations
- [ ] JSON output mode for all commands with `--json`
- [ ] Compact output mode for automation with `--compact`
- [ ] Color output respects NO_COLOR environment variable
- [ ] Table columns auto-adjust to terminal width

**Interactive Features:**
- [ ] Confirmation prompts for destructive operations
- [ ] Dry-run mode shows operations without executing
- [ ] Interactive selection lists for multiple options
- [ ] Default values shown in prompts
- [ ] Abort operations with Ctrl+C gracefully

**Shell Completion:**
- [ ] Bash completion script generated during build
- [ ] Zsh completion script generated during build
- [ ] Fish completion script generated during build
- [ ] Installation script works on Linux and macOS
- [ ] Completions include subcommands and options
- [ ] Dynamic completions for space IDs and backup files

**Benchmark Enhancements:**
- [ ] Latency benchmarks with P50/P95/P99 percentiles
- [ ] Throughput benchmarks with concurrent clients
- [ ] Spreading activation benchmarks
- [ ] Consolidation benchmarks
- [ ] Results exported to JSON for comparison
- [ ] Warmup phase before measurement

**Validation Commands:**
- [ ] Configuration validation with environment-specific checks
- [ ] Data integrity validation with auto-fix option
- [ ] Pre-deployment validation checklist
- [ ] Clear error messages with remediation steps
- [ ] Exit codes: 0 (success), 1 (validation failed), 2 (runtime error)

**Error Handling:**
- [ ] User-friendly error messages without stack traces
- [ ] Script execution errors caught and formatted
- [ ] Network errors with retry suggestions
- [ ] File not found errors with path validation
- [ ] Permission errors with chmod/chown hints

**Performance:**
- [ ] CLI startup latency <100ms
- [ ] Progress bar updates at 10Hz for smooth UX
- [ ] Table rendering optimized for wide terminals
- [ ] Large log parsing streams without loading full file
- [ ] Async operations don't block UI updates

## Testing Plan

**Unit Tests:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_builder() {
        let mut table = TableBuilder::new(vec!["Name".to_string(), "Value".to_string()]);
        table.add_row(vec!["foo".to_string(), "42".to_string()]);

        let mut output = Vec::new();
        table.render(&mut output).unwrap();

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Name"));
        assert!(rendered.contains("foo"));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }
}
```

**Integration Tests:**
```bash
# Test backup commands
engram backup create --space test --output /tmp/backup-test --progress
engram backup list --format table
engram backup verify /tmp/backup-test/engram-full-*.tar.zst --level L2

# Test diagnostic commands
engram diagnose health --output /tmp/health-report.txt
engram diagnose collect --log-lines 5000

# Test restore commands (dry run)
engram restore verify-only /tmp/backup-test/engram-full-*.tar.zst

# Test validation commands
engram validate config
engram validate data --space test

# Test completion installation
./scripts/install_completions.sh bash
# Verify: engram ba<TAB> should complete to "engram backup"
```

## Follow-Up Tasks

- Task 002: CLI commands invoke backup/restore scripts
- Task 005: CLI commands invoke diagnostic scripts
- Task 007: Add migration subcommands when migration tooling complete
- Future: TUI (Terminal UI) mode with ncurses for monitoring dashboard
- Future: Remote CLI support (connect to remote Engram servers)


## Implementation Status

COMPLETE - All CLI command modules implemented and integrated.

Note: Commit blocked by unrelated clippy errors in tools/migration-common.
The CLI enhancement work is functionally complete. To commit:
1. Fix clippy errors in tools/migration-common (separate from this task)
2. OR temporarily exclude migration-common from workspace clippy checks
3. Add documentation to output modules (follow-up)

Shell completions generated successfully at /completions/
Installation script available at /scripts/install_completions.sh
