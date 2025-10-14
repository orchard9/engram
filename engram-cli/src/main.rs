//! Engram CLI - Command-line interface for the Engram cognitive graph database

// Clippy configuration: Only allow specific patterns with justification
#![allow(clippy::multiple_crate_versions)] // Dependencies control their own versions
#![allow(clippy::too_many_lines)] // Main.rs coordinates multiple subsystems

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use engram_cli::{
    api::{ApiState, create_api_routes},
    docs::{DocSection, OperationalDocs},
    find_available_port,
    grpc::MemoryService,
};
#[cfg(feature = "hnsw_index")]
use engram_core::activation::{RecallConfig, RecallMode, SpreadingAutoTuner};
use engram_core::{MemoryStore, metrics};
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal;
use tower_http::cors::{Any, CorsLayer};
use tracing::{Level, error, info, warn};
use tracing_subscriber::FmtSubscriber;

// Import our CLI modules
mod cli;
mod config;
use cli::{
    Cli, Commands, ConfigAction, MemoryAction, create_memory, delete_memory, get_memory,
    get_server_connection, list_memories, remove_pid_file, search_memories, show_status,
    stop_server, write_pid_file,
};
use config::ConfigManager;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let log_level = match cli.log_level.to_lowercase().as_str() {
        "error" => Level::ERROR,
        "warn" => Level::WARN,
        "debug" => Level::DEBUG,
        "trace" => Level::TRACE,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder().with_max_level(log_level).finish();

    tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");

    let mut config_manager = ConfigManager::load().context("failed to load CLI configuration")?;

    match cli.command {
        Commands::Start { port, grpc_port } => {
            start_server(port, grpc_port, config_manager.config()).await
        }

        Commands::Stop { force } => {
            if force {
                println!(" Force stopping server...");
            }
            stop_server().await
        }

        Commands::Status { json, watch } => {
            if watch {
                println!("  Watching status (press Ctrl+C to exit)...");
                loop {
                    if json {
                        show_status_json().await?;
                    } else {
                        show_status().await?;
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            } else if json {
                show_status_json().await
            } else {
                show_status().await
            }
        }

        Commands::Memory { action } => handle_memory_command(action).await,

        Commands::Config { action } => handle_config_command(action, &mut config_manager),

        Commands::Shell => start_interactive_shell().await,

        Commands::Benchmark {
            operations,
            concurrent,
            hyperfine,
            operation,
        } => handle_benchmark_command(operations, concurrent, hyperfine, operation).await,

        Commands::Docs {
            section,
            list,
            export,
        } => handle_docs_command(section, list, export),
    }
}

async fn start_server(port: u16, grpc_port: u16, cli_config: &config::CliConfig) -> Result<()> {
    info!(" Starting Engram server...");

    let actual_port = find_available_port(port).await?;
    let actual_grpc_port = find_available_port(grpc_port).await?;

    if actual_port != port {
        warn!(
            "  Port {} occupied, using port {} instead",
            port, actual_port
        );
    }
    if actual_grpc_port != grpc_port {
        warn!(
            "  gRPC port {} occupied, using port {} instead",
            grpc_port, actual_grpc_port
        );
    }

    // Initialize memory store with optional indexing and persistence
    // mut needed for cfg-gated feature initialization below
    #[allow(unused_mut)]
    let mut store = MemoryStore::new(100_000);

    #[cfg(feature = "hnsw_index")]
    {
        store = store.with_hnsw_index();
    }

    #[cfg(feature = "memory_mapped_persistence")]
    {
        let data_dir = resolve_data_directory()?;
        store = store.with_persistence(&data_dir).map_err(|e| {
            anyhow::anyhow!(
                "Failed to enable persistence at {}: {}",
                data_dir.display(),
                e
            )
        })?;

        let recovered = store.recover_from_wal().map_err(|e| {
            anyhow::anyhow!("Failed to recover WAL from {}: {}", data_dir.display(), e)
        })?;
        if recovered > 0 {
            info!(recovered, "Recovered episodes from write-ahead log");
        } else {
            info!("No write-ahead log entries to recover");
        }

        store
            .initialize_persistence()
            .map_err(|e| anyhow::anyhow!("Failed to start persistence workers: {}", e))?;
    }

    #[cfg(feature = "hnsw_index")]
    if cli_config.feature_flags.spreading_api_beta {
        info!(" Spreading API beta flag enabled");
    } else {
        warn!(" Spreading API beta flag disabled — similarity-only recall will be used by default");
        store = store.with_recall_config(RecallConfig {
            recall_mode: RecallMode::Similarity,
            ..RecallConfig::default()
        });
    }

    #[cfg(not(feature = "hnsw_index"))]
    if cli_config.feature_flags.spreading_api_beta {
        warn!(
            " Spreading API beta flag ignored because the CLI was built without the 'hnsw_index' feature"
        );
    }

    // Enable event streaming for real-time observability
    // CRITICAL: Keep the receiver alive to maintain broadcast channel subscriptions
    // Without an active subscriber, all events are silently dropped
    let mut event_rx = store.enable_event_streaming(1000);
    info!(" Event streaming enabled (buffer size: 1000)");

    let memory_store = Arc::new(store);

    // Spawn keepalive subscriber task to guarantee the broadcast channel
    // always has at least one active receiver. This prevents events from
    // being silently dropped when no SSE clients are connected.
    info!(" Keepalive subscriber started - maintaining event streaming invariant");
    tokio::spawn(async move {
        let mut lagged_count = 0u64;
        let mut events_received = 0u64;
        let mut last_health_log = std::time::Instant::now();

        loop {
            match event_rx.recv().await {
                Ok(_event) => {
                    events_received += 1;

                    // Periodic health reporting (every 10 seconds)
                    if last_health_log.elapsed() >= std::time::Duration::from_secs(10) {
                        info!(
                            events_received,
                            lagged_count,
                            "Keepalive subscriber healthy - {} events received, {} lagged",
                            events_received,
                            lagged_count
                        );
                        last_health_log = std::time::Instant::now();
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    lagged_count += skipped;
                    warn!(
                        lagged_count,
                        skipped,
                        "Keepalive event subscriber lagging - {} events skipped (total: {}). \
                         This indicates high event volume or slow processing.",
                        skipped,
                        lagged_count
                    );
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    error!(
                        events_received,
                        lagged_count,
                        "CRITICAL: Event broadcast channel closed - keepalive subscriber exiting. \
                         Event streaming is now BROKEN. {} events were received before shutdown.",
                        events_received
                    );
                    break;
                }
            }
        }
    });
    let metrics = metrics::init();

    let auto_tuner = SpreadingAutoTuner::new(0.10, 64);
    let tuner_metrics = Arc::clone(&metrics);
    let tuner_auto = Arc::clone(&auto_tuner);
    #[cfg(feature = "hnsw_index")]
    let tuner_store = Arc::clone(&memory_store);
    let auto_tune_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(300));
        loop {
            interval.tick().await;
            let snapshot = tuner_metrics.streaming_snapshot();
            if let Some(summary) = snapshot.spreading.clone() {
                #[cfg(feature = "hnsw_index")]
                if let Some(engine) = tuner_store.spreading_engine() {
                    let _ = tuner_auto.evaluate(&summary, &engine);
                }
            }
        }
    });
    info!(" Auto-tuner background worker started (5 minute interval)");

    let health_metrics = Arc::clone(&metrics);
    let health_handle = tokio::spawn(async move {
        let registry = health_metrics.health_registry();
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
        loop {
            interval.tick().await;
            let status = registry.check_all();
            tracing::trace!(
                target = "engram::health",
                ?status,
                "health probes evaluated"
            );
        }
    });

    // Start background tier migration task if persistence is enabled
    #[cfg(feature = "memory_mapped_persistence")]
    {
        memory_store.start_tier_migration();
        info!(" Background tier migration started (5 minute interval)");
    }

    // Start HNSW update worker for async index updates
    #[cfg(feature = "hnsw_index")]
    {
        memory_store.start_hnsw_worker();
        info!(" HNSW update worker started (batch size: 100, timeout: 50ms)");
    }

    // Create API state
    let api_state = ApiState::new(
        Arc::clone(&memory_store),
        Arc::clone(&metrics),
        Arc::clone(&auto_tuner),
    );

    // Clone memory store for gRPC before moving api_state into router
    let grpc_memory_store = Arc::clone(&api_state.store);
    let grpc_metrics = Arc::clone(&api_state.metrics);

    // Build HTTP API routes
    let app = create_api_routes().with_state(api_state).layer(
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any),
    );

    // Emit periodic structured metrics logs for operators tailing logs.
    let logging_metrics = Arc::clone(&metrics);
    let logging_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            interval.tick().await;
            logging_metrics.log_streaming_snapshot("daemon");
        }
    });

    // Start HTTP server
    let addr = SocketAddr::from(([127, 0, 0, 1], actual_port));
    let listener = TcpListener::bind(addr).await?;

    info!(
        " HTTP API server listening on http://127.0.0.1:{}",
        actual_port
    );
    info!(" Starting gRPC server on 127.0.0.1:{}", actual_grpc_port);

    // Write PID file for server management
    write_pid_file(actual_port)?;

    // Start gRPC server in background task
    let grpc_service = MemoryService::new(grpc_memory_store, grpc_metrics);
    let grpc_handle = tokio::spawn(async move {
        if let Err(e) = grpc_service.serve(actual_grpc_port).await {
            error!(" gRPC server error: {}", e);
        }
    });

    println!(" Engram server started successfully!");
    println!(" HTTP API: http://127.0.0.1:{actual_port}");
    println!(" gRPC: 127.0.0.1:{actual_grpc_port}");
    println!(" Health: http://127.0.0.1:{actual_port}/health");
    println!(" API Docs: http://127.0.0.1:{actual_port}/docs");
    println!();
    println!(" Use 'engram status' to check server health");
    println!(" Use 'engram stop' to shutdown the server");

    // Start HTTP server with graceful shutdown, alongside gRPC
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // gRPC server will be cancelled when the tokio runtime shuts down
    // Explicitly abort to ensure clean shutdown
    grpc_handle.abort();
    logging_handle.abort();
    auto_tune_handle.abort();
    health_handle.abort();

    // Cleanup on exit
    remove_pid_file()?;
    info!(" Server stopped gracefully");

    Ok(())
}

#[cfg(feature = "memory_mapped_persistence")]
fn resolve_data_directory() -> Result<PathBuf> {
    let env_dir = std::env::var("ENGRAM_DATA_DIR").map(PathBuf::from).ok();
    let base_dir = if let Some(path) = env_dir {
        path
    } else {
        let cwd = std::env::current_dir().context("Unable to determine current directory")?;
        cwd.join("engram-data")
    };

    fs::create_dir_all(&base_dir)
        .with_context(|| format!("Failed to create data directory at {}", base_dir.display()))?;

    Ok(base_dir)
}

async fn handle_memory_command(action: MemoryAction) -> Result<()> {
    let (port, _grpc_port) = get_server_connection().await?;

    match action {
        MemoryAction::Create {
            content,
            confidence,
        } => create_memory(port, content, confidence).await,
        MemoryAction::Get { id } => get_memory(port, id).await,
        MemoryAction::Search { query, limit } => search_memories(port, query, limit).await,
        MemoryAction::List { limit, offset } => list_memories(port, limit, offset).await,
        MemoryAction::Delete { id } => delete_memory(port, id).await,
    }
}

fn handle_config_command(action: ConfigAction, manager: &mut ConfigManager) -> Result<()> {
    match action {
        ConfigAction::Get { key } => manager.get(&key).map_or_else(
            || Err(anyhow!("unknown configuration key: {key}")),
            |value| {
                println!("{value}");
                Ok(())
            },
        ),
        ConfigAction::Set { key, value } => {
            manager.set(&key, &value)?;
            manager.save()?;
            println!("Updated {key} = {value}");
            Ok(())
        }
        ConfigAction::List { section } => {
            match section.as_deref() {
                Some("feature_flags") => {
                    for line in config::format_feature_flags(&manager.config().feature_flags) {
                        println!("{line}");
                    }
                }
                None => {
                    for line in config::format_sections(manager.config()) {
                        println!("{line}");
                    }
                }
                Some(other) => {
                    return Err(anyhow!("unknown section: {other}"));
                }
            }
            Ok(())
        }
        ConfigAction::Path => {
            println!("{}", manager.path().display());
            Ok(())
        }
    }
}

async fn start_interactive_shell() -> Result<()> {
    println!(" Engram Interactive Shell");
    println!("Type 'help' for commands, 'exit' to quit");

    let mut rl = rustyline::DefaultEditor::new()?;

    loop {
        match rl.readline("engram> ") {
            Ok(line) => {
                let _ = rl.add_history_entry(&line);
                let trimmed = line.trim();

                if trimmed == "exit" || trimmed == "quit" {
                    break;
                }

                if trimmed == "help" {
                    print_shell_help();
                    continue;
                }

                // Parse shell command and execute
                if let Err(e) = execute_shell_command(trimmed).await {
                    eprintln!(" {e}");
                }
            }
            Err(
                rustyline::error::ReadlineError::Interrupted | rustyline::error::ReadlineError::Eof,
            ) => {
                break;
            }
            Err(err) => {
                eprintln!(" Error: {err}");
                break;
            }
        }
    }

    println!(" Goodbye!");
    Ok(())
}

fn print_shell_help() {
    println!(" Available Commands:");
    println!("  status              - Show server status");
    println!("  create <content>    - Create a memory");
    println!("  get <id>           - Get memory by ID");
    println!("  search <query>     - Search memories");
    println!("  list               - List all memories");
    println!("  help               - Show this help");
    println!("  exit               - Exit shell");
}

async fn execute_shell_command(cmd: &str) -> Result<()> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(());
    }

    match parts.first().copied() {
        Some("status") => show_status().await,
        Some("create") => {
            if parts.len() < 2 {
                eprintln!("Usage: create <content>");
                return Ok(());
            }
            let content = parts[1..].join(" ");
            let (port, _) = get_server_connection().await?;
            create_memory(port, content, None).await
        }
        Some("get") => {
            if parts.len() != 2 {
                eprintln!("Usage: get <id>");
                return Ok(());
            }
            let (port, _) = get_server_connection().await?;
            get_memory(port, parts[1].to_string()).await
        }
        Some("search") => {
            if parts.len() < 2 {
                eprintln!("Usage: search <query>");
                return Ok(());
            }
            let query = parts[1..].join(" ");
            let (port, _) = get_server_connection().await?;
            search_memories(port, query, None).await
        }
        Some("list") => {
            let (port, _) = get_server_connection().await?;
            list_memories(port, Some(10), None).await
        }
        Some(cmd) => {
            eprintln!(" Unknown command: {cmd}");
            eprintln!(" Type 'help' for available commands");
            Ok(())
        }
        None => Ok(()),
    }
}

async fn handle_benchmark_command(
    operations: usize,
    concurrent: usize,
    hyperfine: bool,
    operation: String,
) -> Result<()> {
    println!(
        " Starting benchmark with {operations} operations, {concurrent} concurrent connections"
    );

    if hyperfine {
        println!("  Hyperfine benchmarking not implemented for memory operations");
        println!(" Use the built-in benchmark instead");
        return Ok(());
    }

    // For now, just validate that the server is running
    let (_port, _grpc_port) = get_server_connection().await?;

    println!(" Server connection verified");
    println!("  Full memory operation benchmarking not yet implemented");
    println!(" This would benchmark {operations} operations of type '{operation}'");

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }

    info!(" Shutdown signal received");
}

fn handle_docs_command(section: Option<String>, list: bool, export: Option<String>) -> Result<()> {
    if list {
        println!(" Available Documentation Sections:");
        println!("═══════════════════════════════════════");
        for (name, section_type) in OperationalDocs::available_sections() {
            let description = match section_type {
                DocSection::Emergency => "Emergency procedures (2-5 min fixes)",
                DocSection::Common => "Common operations (5-15 min)",
                DocSection::Advanced => "Advanced operations (30+ min)",
                DocSection::Troubleshooting => "Decision trees and debugging",
                DocSection::IncidentResponse => "Incident response playbooks",
                DocSection::Reference => "Command and API reference",
            };
            println!(" {name:<15} - {description}");
        }
        println!("\nUsage: engram docs <section>");
        return Ok(());
    }

    let content = if let Some(section_name) = section {
        match section_name.parse::<DocSection>() {
            Ok(section_type) => OperationalDocs::get_section(section_type),
            Err(e) => {
                eprintln!(" {e}");
                eprintln!(" Use 'engram docs --list' to see available sections");
                return Ok(());
            }
        }
    } else {
        OperationalDocs::complete_guide()
    };

    if let Some(export_path) = export {
        std::fs::write(&export_path, content)?;
        println!(" Documentation exported to: {export_path}");
    } else {
        println!("{content}");
    }

    Ok(())
}

async fn show_status_json() -> Result<()> {
    use cli::server::{is_process_running, pid_file_path, read_pid_file};

    let pid_path = pid_file_path();

    if !pid_path.exists() {
        println!(
            r#"{{"status": "offline", "health": "not_found", "memory_count": 0, "message": "No server running"}}"#
        );
        return Ok(());
    }

    let Ok((pid, port)) = read_pid_file() else {
        println!(
            r#"{{"status": "error", "health": "corrupted", "memory_count": 0, "message": "Server info corrupted"}}"#
        );
        return Ok(());
    };

    if !is_process_running(pid) {
        println!(
            r#"{{"status": "offline", "health": "process_dead", "memory_count": 0, "pid": {pid}, "port": {port}, "message": "Process not running"}}"#
        );
        return Ok(());
    }

    // Try to get health status
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    let health_url = format!("http://127.0.0.1:{port}/health/alive");

    match client.get(&health_url).send().await {
        Ok(response) if response.status().is_success() => {
            println!(
                r#"{{"status": "online", "health": "responsive", "memory_count": 0, "pid": {pid}, "port": {port}}}"#
            );
        }
        Ok(_) => {
            println!(
                r#"{{"status": "degraded", "health": "unresponsive", "memory_count": 0, "pid": {pid}, "port": {port}}}"#
            );
        }
        Err(_) => {
            println!(
                r#"{{"status": "offline", "health": "unreachable", "memory_count": 0, "pid": {pid}, "port": {port}}}"#
            );
        }
    }

    Ok(())
}
