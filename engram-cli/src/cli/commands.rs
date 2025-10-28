/// CLI command definitions
use clap::{Parser, Subcommand};

/// Engram cognitive graph database CLI
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Set the log level
    #[arg(short, long, default_value = "info")]
    pub log_level: String,

    /// Main command to execute
    #[command(subcommand)]
    pub command: Commands,
}

/// Available CLI commands
#[derive(Subcommand)]
pub enum Commands {
    /// Start the Engram server with automatic configuration
    Start {
        /// Server port (automatically finds free port if default occupied)
        #[arg(short, long, default_value = "7432")]
        port: u16,

        /// gRPC server port (automatically finds free port if default occupied)
        #[arg(short, long, default_value = "50051")]
        grpc_port: u16,
    },

    /// Stop the Engram server gracefully
    Stop {
        /// Force shutdown without graceful cleanup
        #[arg(long)]
        force: bool,
    },

    /// Show current status
    Status {
        /// Output in JSON format
        #[arg(long)]
        json: bool,

        /// Watch status continuously
        #[arg(long)]
        watch: bool,

        /// Memory space to query (overrides ENGRAM_MEMORY_SPACE)
        #[arg(long)]
        space: Option<String>,
    },

    /// Memory operations
    Memory {
        /// Memory operation to perform
        #[command(subcommand)]
        action: MemoryAction,
    },

    /// Memory space registry operations
    Space {
        /// Memory space action to perform
        #[command(subcommand)]
        action: SpaceAction,
    },

    /// Configuration management
    Config {
        /// Configuration action to perform
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Interactive shell mode
    Shell,

    /// Benchmark server performance
    Benchmark {
        /// Benchmark action to perform
        #[command(subcommand)]
        action: BenchmarkAction,
    },

    /// Show embedded documentation
    Docs {
        /// Documentation section to show
        /// Available: emergency, common, advanced, troubleshooting, incident, reference
        section: Option<String>,

        /// Show all available sections
        #[arg(long)]
        list: bool,

        /// Export documentation to file
        #[arg(long)]
        export: Option<String>,
    },

    /// Query with probabilistic confidence intervals
    Query {
        /// Query text
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Output format (json, table, compact)
        #[arg(short, long, default_value = "table")]
        format: OutputFormat,

        /// Memory space to query (overrides ENGRAM_MEMORY_SPACE)
        #[arg(long)]
        space: Option<String>,
    },

    /// Backup operations
    Backup {
        /// Backup action to perform
        #[command(subcommand)]
        action: BackupAction,
    },

    /// Restore operations
    Restore {
        /// Restore action to perform
        #[command(subcommand)]
        action: RestoreAction,
    },

    /// Diagnostic operations
    Diagnose {
        /// Diagnostic action to perform
        #[command(subcommand)]
        action: DiagnoseAction,
    },

    /// Migration operations
    Migrate {
        /// Migration action to perform
        #[command(subcommand)]
        action: MigrateAction,
    },

    /// Validation operations
    Validate {
        /// Validation action to perform
        #[command(subcommand)]
        action: ValidateAction,
    },
}

/// Output format for probabilistic queries
#[derive(Clone)]
pub enum OutputFormat {
    /// JSON format
    Json,
    /// Table format with full details
    Table,
    /// Compact format
    Compact,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "table" => Ok(Self::Table),
            "compact" => Ok(Self::Compact),
            _ => Err(format!(
                "Invalid format: {s}. Valid options: json, table, compact"
            )),
        }
    }
}

/// Memory-specific operations
#[derive(Subcommand)]
pub enum MemoryAction {
    /// Create a new memory
    Create {
        /// Memory content
        content: String,

        /// Confidence level (0.0 to 1.0)
        #[arg(short, long)]
        confidence: Option<f64>,

        /// Memory space to store in (overrides ENGRAM_MEMORY_SPACE)
        #[arg(long)]
        space: Option<String>,
    },

    /// Get a memory by ID
    Get {
        /// Memory ID
        id: String,

        /// Memory space to query (overrides ENGRAM_MEMORY_SPACE)
        #[arg(long)]
        space: Option<String>,
    },

    /// Search for memories
    Search {
        /// Search query
        query: String,

        /// Maximum number of results
        #[arg(short, long)]
        limit: Option<usize>,

        /// Memory space to query (overrides ENGRAM_MEMORY_SPACE)
        #[arg(long)]
        space: Option<String>,
    },

    /// List all memories
    List {
        /// Maximum number of results
        #[arg(short, long)]
        limit: Option<usize>,

        /// Skip number of results
        #[arg(short, long)]
        offset: Option<usize>,
    },

    /// Delete a memory by ID
    Delete {
        /// Memory ID
        id: String,
    },
}

/// Memory space operations
#[derive(Subcommand)]
pub enum SpaceAction {
    /// List all registered memory spaces
    List,

    /// Create (or retrieve) a memory space by identifier
    Create {
        /// Identifier for the memory space
        id: String,
    },
}

/// Configuration management operations
#[derive(Subcommand)]
pub enum ConfigAction {
    /// Get a configuration value
    Get {
        /// Configuration key
        key: String,
    },

    /// Set a configuration value
    Set {
        /// Configuration key
        key: String,
        /// Configuration value
        value: String,
    },

    /// Manage configuration settings
    List {
        /// Show only specified section
        #[arg(long)]
        section: Option<String>,
    },

    /// Show configuration file location
    Path,
}

/// Backup-specific operations
#[derive(Subcommand)]
pub enum BackupAction {
    /// Create a new backup
    Create {
        /// Backup type: full or incremental
        #[arg(short = 't', long, default_value = "full")]
        backup_type: String,

        /// Memory space to backup (or "all")
        #[arg(short, long, default_value = "default")]
        space: String,

        /// Output directory for backup
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,

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
        #[arg(short = 't', long)]
        backup_type: Option<String>,

        /// Filter by memory space
        #[arg(short, long)]
        space: Option<String>,

        /// Output format
        #[arg(short, long, default_value = "table")]
        format: String,
    },

    /// Verify backup integrity
    Verify {
        /// Backup file path
        backup_file: std::path::PathBuf,

        /// Verification level: L1 (manifest), L2 (checksums), L3 (structure), L4 (full restore test)
        #[arg(short, long, default_value = "L2")]
        level: String,

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

/// Restore-specific operations
#[derive(Subcommand)]
pub enum RestoreAction {
    /// Restore from full backup
    Full {
        /// Backup file path
        backup_file: std::path::PathBuf,

        /// Target directory for restore
        #[arg(short, long)]
        target: Option<std::path::PathBuf>,

        /// Show progress bar
        #[arg(long)]
        progress: bool,
    },

    /// Apply incremental backup
    Incremental {
        /// Backup file path
        backup_file: std::path::PathBuf,

        /// Show progress bar
        #[arg(long)]
        progress: bool,
    },

    /// Point-in-time recovery
    Pitr {
        /// Timestamp for recovery (YYYY-MM-DD HH:MM:SS or YYYYMMDD-HHMMSS)
        timestamp: String,

        /// Target directory for restore
        #[arg(short, long)]
        target: Option<std::path::PathBuf>,
    },

    /// Verify restore without applying
    VerifyOnly {
        /// Backup file path
        backup_file: std::path::PathBuf,
    },
}

/// Diagnostic operations
#[derive(Subcommand)]
pub enum DiagnoseAction {
    /// Run comprehensive health check
    Health {
        /// Output file for report
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,

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
        file: Option<std::path::PathBuf>,

        /// Time window: 1h, 24h, 7d
        #[arg(short, long, default_value = "1h")]
        window: String,

        /// Filter by severity: ERROR, WARN, INFO
        #[arg(short, long)]
        severity: Option<String>,
    },

    /// Emergency recovery procedures
    Emergency {
        /// Recovery scenario: corruption, oom, deadlock, disk-full
        scenario: String,

        /// Automatic recovery without prompts
        #[arg(long)]
        auto: bool,
    },
}

/// Migration operations
#[derive(Subcommand)]
pub enum MigrateAction {
    /// Migrate from Neo4j
    Neo4j {
        /// Connection URI
        connection_uri: String,

        /// Target memory space
        #[arg(short, long, default_value = "default")]
        target_space: String,

        /// Batch size for migration
        #[arg(short, long, default_value = "1000")]
        batch_size: usize,
    },

    /// Migrate from PostgreSQL
    Postgresql {
        /// Connection URI
        connection_uri: String,

        /// Target memory space
        #[arg(short, long, default_value = "default")]
        target_space: String,

        /// Table mappings configuration file
        #[arg(short, long)]
        mappings: Option<std::path::PathBuf>,
    },

    /// Migrate from Redis
    Redis {
        /// Connection URI
        connection_uri: String,

        /// Target memory space
        #[arg(short, long, default_value = "default")]
        target_space: String,

        /// Key pattern to migrate
        #[arg(short, long)]
        key_pattern: Option<String>,
    },
}

/// Validation operations
#[derive(Subcommand)]
pub enum ValidateAction {
    /// Validate configuration file
    Config {
        /// Path to config file
        #[arg(short, long)]
        file: Option<std::path::PathBuf>,

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

/// Enhanced benchmark operations
#[derive(Subcommand)]
pub enum BenchmarkAction {
    /// Measure operation latency (P50, P95, P99)
    Latency {
        /// Operation to benchmark: create, get, search
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
        /// Duration to run benchmark (seconds)
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
