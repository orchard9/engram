//! CLI command definitions

use clap::{Parser, Subcommand};

/// Engram cognitive graph database CLI
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Set the log level
    #[arg(short, long, default_value = "info")]
    pub log_level: String,

    #[command(subcommand)]
    pub command: Commands,
}

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
        #[command(subcommand)]
        action: MemoryAction,
    },

    /// Configuration management
    Config {
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
}

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
