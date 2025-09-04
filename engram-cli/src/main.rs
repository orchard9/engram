//! Engram CLI - Command-line interface for the Engram cognitive graph database

use anyhow::Result;
use clap::{Parser, Subcommand};
use engram_core::MemoryNode;
use engram_storage::{StorageTier, hot::HotStorage};
use std::sync::Arc;
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

/// Engram cognitive graph database CLI
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Set the log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Store a memory node
    Store {
        /// Node ID
        #[arg(short, long)]
        id: String,

        /// Content as text
        #[arg(short, long)]
        content: String,

        /// Initial activation level (0.0 to 1.0)
        #[arg(short, long, default_value = "0.5")]
        activation: f64,
    },

    /// Retrieve a memory node
    Get {
        /// Node ID to retrieve
        #[arg(short, long)]
        id: String,
    },

    /// Activate a memory node
    Activate {
        /// Node ID
        #[arg(short, long)]
        id: String,

        /// Activation energy to add
        #[arg(short, long, default_value = "0.1")]
        energy: f64,
    },

    /// Show storage statistics
    Stats,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let level = match cli.log_level.as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder().with_max_level(level).finish();

    tracing::subscriber::set_global_default(subscriber)?;

    // Create storage
    let storage = Arc::new(HotStorage::new(10000));

    match cli.command {
        Commands::Store {
            id,
            content,
            activation,
        } => {
            // Create a new active memory node using the backwards-compatible constructor
            let mut node = MemoryNode::new(id.clone(), content.into_bytes());
            node.activation = activation; // Set the custom activation level

            storage.store_node(node).await?;
            info!("Stored node: {}", id);
        }

        Commands::Get { id } => match storage.get_node(&id).await? {
            Some(node) => {
                println!("Node ID: {}", node.id);
                println!("Activation: {}", node.activation);
                println!("Content: {}", String::from_utf8_lossy(&node.content));
                println!("Confidence: {:?}", node.confidence);
            }
            None => {
                println!("Node not found: {}", id);
            }
        },

        Commands::Activate { id, energy } => match storage.get_node(&id).await? {
            Some(mut node) => {
                use engram_core::Activatable;
                node.activate(energy);
                let new_activation = node.activation_level();
                storage.store_node(node).await?;
                info!(
                    "Activated node {} with energy {}, new level: {}",
                    id, energy, new_activation
                );
            }
            None => {
                println!("Node not found: {}", id);
            }
        },

        Commands::Stats => {
            println!("Storage tier: {}", storage.tier_name());
            println!("Can accept more: {}", storage.can_accept());
        }
    }

    Ok(())
}
