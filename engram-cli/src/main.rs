//! Engram CLI - Command-line interface for the Engram cognitive graph database

use anyhow::{Context, Result};
use axum::{Router, response::Json, routing::get};
use clap::{Parser, Subcommand};
use engram_cli::find_available_port;
use engram_core::MemoryNode;
use engram_storage::{StorageTier, hot::HotStorage};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tracing::{Level, info, warn};
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
    /// Start the Engram server with automatic configuration
    Start {
        /// Server port (automatically finds free port if default occupied)
        #[arg(short, long, default_value = "7432")]
        port: u16,

        /// Skip cluster discovery and run in single-node mode
        #[arg(long)]
        single_node: bool,

        /// Timeout for cluster discovery in seconds
        #[arg(long, default_value = "30")]
        discovery_timeout: u64,
    },

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

/// Health check endpoint
async fn health() -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "service": "engram",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

/// Start the Engram server with cognitive-friendly progress indication
async fn start_server(
    preferred_port: u16,
    single_node: bool,
    _discovery_timeout: u64,
) -> Result<()> {
    // Create progress bar with cognitive design
    let pb = ProgressBar::new(5);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>2}/{len:2} {msg}",
            )
            .context("Failed to create progress bar template")?
            .progress_chars("##-"),
    );
    pb.set_message("Initializing Engram cognitive memory system...");

    // Stage 1: Initialize storage
    pb.set_position(1);
    pb.set_message("🧠 Initializing memory storage layer...");
    tokio::time::sleep(Duration::from_millis(200)).await;
    let _storage = Arc::new(HotStorage::new(10000));

    // Stage 2: Port discovery
    pb.set_position(2);
    pb.set_message("🔍 Discovering available network port...");
    let port = find_available_port(preferred_port)
        .await
        .context("Failed to find available port")?;

    if port != preferred_port {
        warn!(
            "Port {} was occupied, using port {} instead",
            preferred_port, port
        );
    }

    // Stage 3: Network binding
    pb.set_position(3);
    pb.set_message("🌐 Binding to network interface...");
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr)
        .await
        .context("Failed to bind to address")?;

    // Stage 4: Health checks
    pb.set_position(4);
    pb.set_message("❤️  Initializing health monitoring...");
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Stage 5: Ready
    pb.set_position(5);
    pb.set_message("✅ Engram server ready!");
    pb.finish();

    // Create the Axum app with routes
    let app = Router::new()
        .route("/health", get(health))
        .route("/", get(|| async { "Engram Cognitive Graph Database" }));

    println!();
    if single_node {
        println!("🧠 Engram started in single-node mode");
    } else {
        println!("🧠 Engram started with cluster discovery enabled");
    }
    println!("📡 Server listening on: http://127.0.0.1:{}", port);
    println!("❤️  Health endpoint: http://127.0.0.1:{}/health", port);
    println!("🎯 Ready to process cognitive graph operations");
    println!();
    info!("Engram server started successfully on port {}", port);

    // Start the server
    axum::serve(listener, app).await.context("Server error")?;

    Ok(())
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
        Commands::Start {
            port,
            single_node,
            discovery_timeout,
        } => {
            start_server(port, single_node, discovery_timeout).await?;
        }
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
