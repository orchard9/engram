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
use engram_core::{
    MemorySpaceError, MemorySpaceId, MemorySpaceRegistry, MemoryStore, SpaceDirectories, metrics,
};
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal;
use tower_http::cors::{Any, CorsLayer};
use tracing::{Level, error, info, warn};
use tracing_subscriber::FmtSubscriber;

// Import from our library
use engram_cli::cli::{
    BackupAction, BenchmarkAction, Cli, Commands, ConfigAction, DiagnoseAction, MemoryAction,
    MigrateAction, OutputFormat, RestoreAction, SpaceAction, ValidateAction, create_memory,
    create_space, delete_memory, get_memory, get_server_connection, list_memories, list_spaces,
    remove_pid_file, search_memories, show_status, stop_server, write_pid_file,
};
use engram_cli::config::ConfigManager;

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

        Commands::Status { json, watch, space } => {
            // Task 006b: Per-space metrics now supported
            if watch {
                println!("  Watching status (press Ctrl+C to exit)...");
                loop {
                    if json {
                        show_status_json().await?;
                    } else {
                        show_status(space.as_deref()).await?;
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            } else if json {
                show_status_json().await
            } else {
                show_status(space.as_deref()).await
            }
        }

        Commands::Memory { action } => handle_memory_command(action, config_manager.config()).await,

        Commands::Space { action } => handle_space_command(action).await,

        Commands::Config { action } => handle_config_command(action, &mut config_manager),

        Commands::Shell => start_interactive_shell().await,

        Commands::Benchmark { action } => handle_benchmark_command(action).await,

        Commands::Docs {
            section,
            list,
            export,
        } => handle_docs_command(section, list, export),

        Commands::Query {
            query,
            limit,
            format,
            space,
        } => handle_query_command(query, limit, format, space, config_manager.config()).await,

        Commands::Backup { action } => handle_backup_command(action),

        Commands::Restore { action } => handle_restore_command(action),

        Commands::Diagnose { action } => handle_diagnose_command(action),

        Commands::Migrate { action } => handle_migrate_command(action),

        Commands::Validate { action } => handle_validate_command(action),
    }
}

/// Resolve memory space ID from CLI flag, environment variable, or config default
///
/// Priority order:
/// 1. CLI --space flag (explicit, highest priority)
/// 2. ENGRAM_MEMORY_SPACE environment variable
/// 3. Config default_space (fallback)
///
/// # Errors
///
/// Returns error if memory space ID validation fails
fn resolve_memory_space(
    cli_space: Option<String>,
    config_default: &MemorySpaceId,
) -> Result<MemorySpaceId> {
    // Priority 1: CLI flag
    if let Some(space_str) = cli_space {
        return MemorySpaceId::try_from(space_str.as_str())
            .map_err(|e| anyhow::anyhow!("Invalid memory space ID: {e}"));
    }

    // Priority 2: Environment variable
    if let Ok(env_space) = std::env::var("ENGRAM_MEMORY_SPACE")
        && !env_space.trim().is_empty()
    {
        return MemorySpaceId::try_from(env_space.as_str())
            .map_err(|e| anyhow::anyhow!("Invalid ENGRAM_MEMORY_SPACE: {e}"));
    }

    // Priority 3: Config default
    Ok(config_default.clone())
}

/// Wait for gRPC server to be ready to accept connections
///
/// # Errors
///
/// Returns error if gRPC server doesn't become ready within timeout
async fn wait_for_grpc_ready(port: u16) -> Result<()> {
    use std::time::Duration;
    use tokio::net::TcpStream;

    let max_attempts = 20; // 20 * 500ms = 10 seconds
    let retry_interval = Duration::from_millis(500);

    for attempt in 1..=max_attempts {
        match TcpStream::connect(format!("127.0.0.1:{port}")).await {
            Ok(_) => {
                info!(" gRPC server ready (attempt {attempt})");
                return Ok(());
            }
            Err(_) => {
                if attempt < max_attempts {
                    tokio::time::sleep(retry_interval).await;
                }
            }
        }
    }

    Err(anyhow::anyhow!(
        "❌ gRPC server did not become ready within 10 seconds\\n\\\
         🔍 The server may have failed to start or is taking too long to initialize.\\n\\\
         💡 Check logs for gRPC server errors"
    ))
}

async fn start_server(
    port: u16,
    grpc_port: u16,
    cli_config: &engram_cli::config::CliConfig,
) -> Result<()> {
    info!(" Starting Engram server...");

    // Create shutdown signal channel for graceful termination
    // All background tasks subscribe to shutdown_rx to know when to exit
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let shutdown_tx = Arc::new(shutdown_tx);

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

    let data_root = resolve_data_directory()?;
    info!(root = %data_root.display(), "Using memory space data root");

    let feature_flags = cli_config.feature_flags.clone();
    let registry_flags = feature_flags.clone();

    let registry = Arc::new(MemorySpaceRegistry::new(
        &data_root,
        move |space_id, directories| {
            build_memory_space_store(space_id, directories, &registry_flags)
        },
    )?);

    // Recover all existing memory spaces from WAL logs
    #[cfg(feature = "memory_mapped_persistence")]
    {
        info!("Scanning persistence root for existing memory spaces");
        match registry.recover_all().await {
            Ok(reports) => {
                if reports.is_empty() {
                    info!("No existing memory spaces found - starting with clean slate");
                } else {
                    info!(
                        spaces_recovered = reports.len(),
                        "Completed WAL recovery for all memory spaces"
                    );
                    for report in reports {
                        info!(
                            recovered = report.recovered_entries,
                            corrupted = report.corrupted_entries,
                            duration_ms = report.recovery_duration.as_millis(),
                            "Recovery report"
                        );
                    }
                }
            }
            Err(e) => {
                error!(error = ?e, "Failed to recover memory spaces - continuing with bootstrap");
            }
        }
    }

    registry
        .ensure_spaces(cli_config.memory_spaces.bootstrap_spaces.clone())
        .await?;

    let default_space_id = cli_config.memory_spaces.default_space.clone();
    let default_handle = registry.create_or_get(&default_space_id).await?;
    info!(
        space = %default_space_id,
        root = %default_handle.directories().root.display(),
        "Default memory space initialised"
    );

    let memory_store = default_handle.store();

    // Enable event streaming for real-time observability. Without an active
    // subscriber the broadcast channel drops events, so we immediately attach
    // a keepalive receiver after enabling the stream within the registry.
    let mut event_rx = memory_store.subscribe_to_events().ok_or_else(|| {
        anyhow!("Event streaming not initialised for memory space {default_space_id}")
    })?;
    info!(space = %default_space_id, " Event streaming enabled (buffer size: 1000)");

    info!(" Keepalive subscriber started - maintaining event streaming invariant");
    let mut keepalive_shutdown = shutdown_rx.clone();
    let keepalive_space = default_space_id.clone();
    let keepalive_handle = tokio::spawn(async move {
        let mut lagged_count = 0u64;
        let mut events_received = 0u64;
        let mut last_health_log = std::time::Instant::now();

        loop {
            tokio::select! {
                _ = keepalive_shutdown.changed() => {
                    info!(
                        space = %keepalive_space,
                        events_received,
                        lagged_count,
                        "Keepalive subscriber shutting down gracefully - {} events received, {} lagged",
                        events_received,
                        lagged_count
                    );
                    break;
                }
                result = event_rx.recv() => {
                    match result {
                        Ok(_event) => {
                            events_received += 1;

                            // Periodic health reporting (every 10 seconds)
                            if last_health_log.elapsed() >= std::time::Duration::from_secs(10) {
                                info!(
                                    space = %keepalive_space,
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
                                space = %keepalive_space,
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
                                space = %keepalive_space,
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
            }
        }
    });

    #[cfg(feature = "hnsw_index")]
    if feature_flags.spreading_api_beta {
        info!(" Spreading API beta flag enabled");
    } else {
        warn!(" Spreading API beta flag disabled — similarity-only recall will be used by default");
    }

    #[cfg(not(feature = "hnsw_index"))]
    if feature_flags.spreading_api_beta {
        warn!(
            " Spreading API beta flag ignored because the CLI was built without the 'hnsw_index' feature"
        );
    }

    let metrics = metrics::init();

    let auto_tuner = SpreadingAutoTuner::new(0.10, 64);
    let tuner_metrics = Arc::clone(&metrics);
    let tuner_auto = Arc::clone(&auto_tuner);
    #[cfg(feature = "hnsw_index")]
    let tuner_store = Arc::clone(&memory_store);
    let mut autotuner_shutdown = shutdown_rx.clone();
    let auto_tune_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(300));
        loop {
            tokio::select! {
                _ = autotuner_shutdown.changed() => {
                    info!("Auto-tuner shutting down gracefully");
                    break;
                }
                _ = interval.tick() => {
                    let snapshot = tuner_metrics.streaming_snapshot();
                    if let Some(summary) = snapshot.spreading.clone() {
                        #[cfg(feature = "hnsw_index")]
                        if let Some(engine) = tuner_store.spreading_engine() {
                            let _ = tuner_auto.evaluate(&summary, &engine);
                        }
                    }
                }
            }
        }
    });
    info!(" Auto-tuner background worker started (5 minute interval)");

    let health_metrics = Arc::clone(&metrics);
    let mut health_shutdown = shutdown_rx.clone();
    let health_handle = tokio::spawn(async move {
        let registry = health_metrics.health_registry();
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
        loop {
            tokio::select! {
                _ = health_shutdown.changed() => {
                    info!("Health monitor shutting down gracefully");
                    break;
                }
                _ = interval.tick() => {
                    let status = registry.check_all();
                    tracing::trace!(
                        target = "engram::health",
                        ?status,
                        "health probes evaluated"
                    );
                }
            }
        }
    });

    // Start background tier migration task if persistence is enabled
    #[cfg(feature = "memory_mapped_persistence")]
    {
        memory_store.start_tier_migration();
        info!(
            space = %default_space_id,
            " Background tier migration started (5 minute interval)"
        );
    }

    // Start HNSW update worker for async index updates
    #[cfg(feature = "hnsw_index")]
    {
        memory_store.start_hnsw_worker();
        info!(
            space = %default_space_id,
            " HNSW update worker started (batch size: 100, timeout: 50ms)"
        );
    }

    // Create API state
    let api_state = ApiState::new(
        Arc::clone(&memory_store),
        Arc::clone(&registry),
        default_space_id.clone(),
        Arc::clone(&metrics),
        Arc::clone(&auto_tuner),
        Arc::clone(&shutdown_tx),
    );

    // Clone memory store for gRPC before moving api_state into router
    #[allow(deprecated)] // TODO: Migrate to registry pattern once gRPC service is updated
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
    let mut logging_shutdown = shutdown_rx.clone();
    let logging_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
        loop {
            tokio::select! {
                _ = logging_shutdown.changed() => {
                    info!("Metrics logging shutting down gracefully");
                    break;
                }
                _ = interval.tick() => {
                    logging_metrics.log_streaming_snapshot("daemon");
                }
            }
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
    let grpc_service = MemoryService::new(
        grpc_memory_store,
        grpc_metrics,
        Arc::clone(&registry),
        default_space_id.clone(),
    );
    let grpc_handle = tokio::spawn(async move {
        if let Err(e) = grpc_service.serve(actual_grpc_port).await {
            error!(" gRPC server error: {}", e);
        }
    });

    // Wait for gRPC server to be ready before announcing success
    wait_for_grpc_ready(actual_grpc_port).await?;

    println!(" Engram server started successfully!");
    println!(" HTTP API: http://127.0.0.1:{actual_port}");
    println!(" gRPC: 127.0.0.1:{actual_grpc_port}");
    println!(" Health: http://127.0.0.1:{actual_port}/health");
    println!(" API Docs: http://127.0.0.1:{actual_port}/docs");
    println!();
    println!(" Use 'engram status' to check server health");
    println!(" Use 'engram stop' to shutdown the server");

    // Start HTTP server with graceful shutdown, alongside gRPC
    let shutdown_signal_tx = Arc::clone(&shutdown_tx);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown_signal_tx))
        .await?;

    info!(" HTTP server shutdown complete, cleaning up background tasks");

    // Gracefully shutdown all background tasks with timeout
    let shutdown_timeout = std::time::Duration::from_secs(3);
    let shutdown_result = tokio::time::timeout(shutdown_timeout, async {
        // Wait for all background tasks to exit gracefully
        // Tasks will see the shutdown signal and exit their loops
        let _ = tokio::join!(
            keepalive_handle,
            auto_tune_handle,
            health_handle,
            logging_handle,
        );
        info!(" All background tasks stopped");
    })
    .await;

    if shutdown_result.is_err() {
        warn!(" Background task shutdown timeout, aborting remaining tasks");
    }

    // Abort gRPC server (doesn't have shutdown signal integration yet)
    grpc_handle.abort();

    // Cleanup on exit
    remove_pid_file()?;
    info!(" Server stopped gracefully");

    Ok(())
}

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

fn build_memory_space_store(
    space_id: &MemorySpaceId,
    directories: &SpaceDirectories,
    feature_flags: &engram_cli::config::FeatureFlags,
) -> Result<Arc<MemoryStore>, MemorySpaceError> {
    #[allow(unused_mut)]
    let mut store = MemoryStore::for_space(space_id.clone(), 100_000);

    #[cfg(feature = "hnsw_index")]
    {
        store = store.with_hnsw_index();
        if feature_flags.spreading_api_beta {
            tracing::info!(space = %space_id, "Spreading API beta enabled");
        } else {
            tracing::warn!(
                space = %space_id,
                "Spreading API beta disabled — falling back to similarity recall"
            );
            store = store.with_recall_config(RecallConfig {
                recall_mode: RecallMode::Similarity,
                ..RecallConfig::default()
            });
        }
    }

    #[cfg(not(feature = "hnsw_index"))]
    let _ = feature_flags;

    #[cfg(feature = "memory_mapped_persistence")]
    {
        // NOTE: Using legacy with_persistence for now.
        // TODO: Migrate to with_persistence_handle() from registry in future milestone.
        // Each space still gets isolated directories (wal/, hot/, warm/, cold/).
        store = store.with_persistence(&directories.root).map_err(|err| {
            MemorySpaceError::StoreInit {
                id: space_id.clone(),
                source: err,
            }
        })?;

        let recovered = store
            .recover_from_wal()
            .map_err(|err| MemorySpaceError::StoreInit {
                id: space_id.clone(),
                source: Box::new(err),
            })?;
        if recovered > 0 {
            tracing::info!(
                space = %space_id,
                recovered,
                "Recovered episodes from write-ahead log"
            );
        }

        store
            .initialize_persistence()
            .map_err(|err| MemorySpaceError::StoreInit {
                id: space_id.clone(),
                source: err,
            })?;
    }

    let _ = store.enable_event_streaming(1000);

    Ok(Arc::new(store))
}

async fn handle_memory_command(
    action: MemoryAction,
    config: &engram_cli::config::CliConfig,
) -> Result<()> {
    let (port, _grpc_port) = get_server_connection().await?;

    match action {
        MemoryAction::Create {
            content,
            confidence,
            space,
        } => {
            let space_id = resolve_memory_space(space, &config.memory_spaces.default_space)?;
            create_memory(port, content, confidence, &space_id).await
        }
        MemoryAction::Get { id, space } => {
            let space_id = resolve_memory_space(space, &config.memory_spaces.default_space)?;
            get_memory(port, id, &space_id).await
        }
        MemoryAction::Search {
            query,
            limit,
            space,
        } => {
            let space_id = resolve_memory_space(space, &config.memory_spaces.default_space)?;
            search_memories(port, query, limit, &space_id).await
        }
        MemoryAction::List { limit, offset } => list_memories(port, limit, offset).await,
        MemoryAction::Delete { id } => delete_memory(port, id).await,
    }
}

async fn handle_space_command(action: SpaceAction) -> Result<()> {
    match action {
        SpaceAction::List => list_spaces().await,
        SpaceAction::Create { id } => create_space(id).await,
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
                    for line in
                        engram_cli::config::format_feature_flags(&manager.config().feature_flags)
                    {
                        println!("{line}");
                    }
                }
                None => {
                    for line in engram_cli::config::format_sections(manager.config()) {
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
        Some("status") => show_status(None).await,
        Some("create") => {
            if parts.len() < 2 {
                eprintln!("Usage: create <content>");
                return Ok(());
            }
            let content = parts[1..].join(" ");
            let (port, _) = get_server_connection().await?;
            let default_space = MemorySpaceId::default();
            create_memory(port, content, None, &default_space).await
        }
        Some("get") => {
            if parts.len() != 2 {
                eprintln!("Usage: get <id>");
                return Ok(());
            }
            let (port, _) = get_server_connection().await?;
            let default_space = MemorySpaceId::default();
            get_memory(port, parts[1].to_string(), &default_space).await
        }
        Some("search") => {
            if parts.len() < 2 {
                eprintln!("Usage: search <query>");
                return Ok(());
            }
            let query = parts[1..].join(" ");
            let (port, _) = get_server_connection().await?;
            let default_space = MemorySpaceId::default();
            search_memories(port, query, None, &default_space).await
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

async fn handle_benchmark_command(action: BenchmarkAction) -> Result<()> {
    match action {
        BenchmarkAction::Latency {
            operation,
            iterations,
            warmup,
        } => {
            engram_cli::cli::benchmark_ops::run_latency_benchmark(&operation, iterations, warmup)
                .await
        }
        BenchmarkAction::Throughput { duration, clients } => {
            engram_cli::cli::benchmark_ops::run_throughput_benchmark(duration, clients).await
        }
        BenchmarkAction::Spreading { nodes, depth } => {
            engram_cli::cli::benchmark_ops::run_spreading_benchmark(nodes, depth).await
        }
        BenchmarkAction::Consolidation { load_test } => {
            engram_cli::cli::benchmark_ops::run_consolidation_benchmark(load_test).await
        }
    }
}

async fn shutdown_signal(shutdown_tx: Arc<tokio::sync::watch::Sender<bool>>) {
    let mut shutdown_rx = shutdown_tx.subscribe();

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

    let api_shutdown = async {
        let _ = shutdown_rx.changed().await;
    };

    tokio::select! {
        () = ctrl_c => {
            info!(" Ctrl+C received");
        },
        () = terminate => {
            info!(" TERM signal received");
        },
        () = api_shutdown => {
            info!(" API shutdown endpoint called");
        },
    }

    info!(" Shutdown signal received");

    // Notify all background tasks to shutdown
    shutdown_tx.send(true).ok();
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
    use engram_cli::cli::server::{is_process_running, pid_file_path, read_pid_file};

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

async fn handle_query_command(
    query: String,
    limit: usize,
    format: OutputFormat,
    space: Option<String>,
    config: &engram_cli::config::CliConfig,
) -> Result<()> {
    let (port, _grpc_port) = get_server_connection().await?;
    let space_id = resolve_memory_space(space, &config.memory_spaces.default_space)?;

    // Make HTTP request
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let url = format!("http://127.0.0.1:{port}/api/v1/query/probabilistic");

    let response = client
        .get(&url)
        .query(&[("query", &query), ("limit", &limit.to_string())])
        .header("X-Engram-Memory-Space", space_id.as_str())
        .send()
        .await
        .context("Failed to send request to server")?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        anyhow::bail!("Server returned error {status}: {error_text}");
    }

    let result: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse server response")?;

    // Format and print output
    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        OutputFormat::Table => {
            print_probabilistic_table(&result);
        }
        OutputFormat::Compact => {
            print_probabilistic_compact(&result);
        }
    }

    Ok(())
}

fn print_probabilistic_table(result: &serde_json::Value) {
    // Parse response fields
    let confidence_interval = &result["confidence_interval"];
    let point = confidence_interval["point"].as_f64().unwrap_or(0.0);
    let lower = confidence_interval["lower"].as_f64().unwrap_or(0.0);
    let upper = confidence_interval["upper"].as_f64().unwrap_or(0.0);
    let is_successful = result["is_successful"].as_bool().unwrap_or(false);

    println!("\nProbabilistic Query Results\n");
    println!(
        "Confidence: {:.2}% [{:.2}% - {:.2}%]",
        point * 100.0,
        lower * 100.0,
        upper * 100.0
    );
    println!(
        "Status: {}\n",
        if is_successful {
            "Successful"
        } else {
            "Low Confidence"
        }
    );

    // Print episodes table
    if let Some(episodes) = result["episodes"].as_array() {
        for (i, ep_conf) in episodes.iter().enumerate() {
            let confidence = ep_conf["confidence"].as_f64().unwrap_or(0.0);
            let what = ep_conf["episode"]["what"].as_str().unwrap_or("");
            println!("{:2}. [{}] {}", i + 1, format_confidence(confidence), what);
        }
    }

    // Print evidence chain
    if let Some(evidence_chain) = result["evidence_chain"].as_array()
        && !evidence_chain.is_empty()
    {
        println!("\nEvidence Chain:");
        for evidence in evidence_chain {
            let source = evidence["source"].as_str().unwrap_or("unknown");
            let confidence_pct = evidence["confidence"].as_f64().unwrap_or(0.0) * 100.0;
            println!("  - {source}: {confidence_pct:.2}%");
        }
    }

    // Print uncertainty sources
    if let Some(uncertainty_sources) = result["uncertainty_sources"].as_array()
        && !uncertainty_sources.is_empty()
    {
        println!("\nUncertainty Sources:");
        for source in uncertainty_sources {
            let description = source["description"].as_str().unwrap_or("unknown");
            println!("  - {description}");
        }
    }
}

fn print_probabilistic_compact(result: &serde_json::Value) {
    let confidence_interval = &result["confidence_interval"];
    let point_pct = confidence_interval["point"].as_f64().unwrap_or(0.0) * 100.0;

    println!("Confidence: {point_pct:.1}%");

    if let Some(episodes) = result["episodes"].as_array() {
        for (i, ep_conf) in episodes.iter().enumerate() {
            let confidence_pct = ep_conf["confidence"].as_f64().unwrap_or(0.0) * 100.0;
            let what = ep_conf["episode"]["what"].as_str().unwrap_or("");
            println!("{}. [{confidence_pct:.1}%] {what}", i + 1);
        }
    }
}

fn format_confidence(confidence: f64) -> String {
    let percentage = confidence * 100.0;
    format!("{percentage:.1}%")
}

// ============================================================================
// New Operations CLI Handlers
// ============================================================================

fn handle_backup_command(action: BackupAction) -> Result<()> {
    match action {
        BackupAction::Create {
            backup_type,
            space,
            output,
            compression,
            progress,
        } => engram_cli::cli::backup::create_backup(
            &backup_type,
            &space,
            output,
            compression,
            progress,
        ),
        BackupAction::List {
            backup_type,
            space,
            format,
        } => {
            engram_cli::cli::backup::list_backups(backup_type.as_deref(), space.as_deref(), &format)
        }
        BackupAction::Verify {
            backup_file,
            level,
            verbose,
        } => engram_cli::cli::backup::verify_backup(&backup_file, &level, verbose),
        BackupAction::Prune {
            dry_run,
            daily,
            weekly,
            monthly,
            yes,
        } => engram_cli::cli::backup::prune_backups(dry_run, daily, weekly, monthly, yes),
    }
}

fn handle_restore_command(action: RestoreAction) -> Result<()> {
    match action {
        RestoreAction::Full {
            backup_file,
            target,
            progress,
        } => engram_cli::cli::restore::restore_full(&backup_file, target, progress),
        RestoreAction::Incremental {
            backup_file,
            progress,
        } => engram_cli::cli::restore::restore_incremental(&backup_file, progress),
        RestoreAction::Pitr { timestamp, target } => {
            engram_cli::cli::restore::restore_pitr(&timestamp, target)
        }
        RestoreAction::VerifyOnly { backup_file } => {
            engram_cli::cli::restore::verify_restore(&backup_file)
        }
    }
}

fn handle_diagnose_command(action: DiagnoseAction) -> Result<()> {
    match action {
        DiagnoseAction::Health { output, strict } => {
            engram_cli::cli::diagnose::run_health_check(output.as_ref(), strict)
        }
        DiagnoseAction::Collect {
            include_dumps,
            log_lines,
        } => engram_cli::cli::diagnose::collect_debug_bundle(include_dumps, log_lines),
        DiagnoseAction::AnalyzeLogs {
            file,
            window,
            severity,
        } => engram_cli::cli::diagnose::analyze_logs(file.as_ref(), &window, severity.as_deref()),
        DiagnoseAction::Emergency { scenario, auto } => {
            engram_cli::cli::diagnose::emergency_recovery(&scenario, auto)
        }
    }
}

fn handle_migrate_command(action: MigrateAction) -> Result<()> {
    match action {
        MigrateAction::Neo4j {
            connection_uri,
            target_space,
            batch_size,
        } => {
            engram_cli::cli::migrate::migrate_from_neo4j(&connection_uri, &target_space, batch_size)
        }
        MigrateAction::Postgresql {
            connection_uri,
            target_space,
            mappings,
        } => engram_cli::cli::migrate::migrate_from_postgresql(
            &connection_uri,
            &target_space,
            mappings.as_ref(),
        ),
        MigrateAction::Redis {
            connection_uri,
            target_space,
            key_pattern,
        } => engram_cli::cli::migrate::migrate_from_redis(
            &connection_uri,
            &target_space,
            key_pattern.as_deref(),
        ),
    }
}

fn handle_validate_command(action: ValidateAction) -> Result<()> {
    match action {
        ValidateAction::Config { file, deployment } => {
            engram_cli::cli::validate::validate_config(file, deployment.as_deref())
        }
        ValidateAction::Data { space, fix } => {
            engram_cli::cli::validate::validate_data(&space, fix)
        }
        ValidateAction::Deployment { environment } => {
            engram_cli::cli::validate::validate_deployment(&environment)
        }
    }
}
