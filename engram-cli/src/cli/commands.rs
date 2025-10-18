//! CLI command definitions

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
    },

    /// Memory operations
    Memory {
        /// Memory operation to perform
        #[command(subcommand)]
        action: MemoryAction,
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
        /// Number of operations to perform
        #[arg(short, long, default_value = "1000")]
        operations: usize,

        /// Number of concurrent connections
        #[arg(short, long, default_value = "10")]
        concurrent: usize,

        /// Use hyperfine for benchmarking (requires hyperfine to be installed)
        #[arg(long)]
        hyperfine: bool,

        /// Memory operation to benchmark (create, get, search)
        #[arg(short, long, default_value = "create")]
        operation: String,
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
    },

    /// Get a memory by ID
    Get {
        /// Memory ID
        id: String,
    },

    /// Search for memories
    Search {
        /// Search query
        query: String,

        /// Maximum number of results
        #[arg(short, long)]
        limit: Option<usize>,
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
