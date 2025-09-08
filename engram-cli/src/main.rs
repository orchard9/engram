//! Engram CLI - Command-line interface for the Engram cognitive graph database

use anyhow::Result;
use clap::Parser;
use engram_cli::{
    api::{ApiState, create_api_routes},
    docs::{DocSection, OperationalDocs},
    find_available_port,
};
use engram_core::graph::MemoryGraph;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal;
use tower_http::cors::{Any, CorsLayer};
use tracing::{Level, info, warn};
use tracing_subscriber::FmtSubscriber;

// Import our CLI modules
mod cli;
use cli::{
    Cli, Commands, ConfigAction, MemoryAction, create_memory, delete_memory, get_memory,
    get_server_connection, list_memories, remove_pid_file, search_memories, show_status,
    stop_server, write_pid_file,
};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    let log_level = match cli.log_level.to_lowercase().as_str() {
        "error" => Level::ERROR,
        "warn" => Level::WARN,
        "info" => Level::INFO,
        "debug" => Level::DEBUG,
        "trace" => Level::TRACE,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder().with_max_level(log_level).finish();

    tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");

    match cli.command {
        Commands::Start { port, grpc_port } => start_server(port, grpc_port).await,

        Commands::Stop { force } => {
            if force {
                println!("🔨 Force stopping server...");
                stop_server().await
            } else {
                stop_server().await
            }
        }

        Commands::Status { json, watch } => {
            if watch {
                println!("👁️  Watching status (press Ctrl+C to exit)...");
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

        Commands::Config { action } => handle_config_command(action).await,

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
        } => handle_docs_command(section, list, export).await,
    }
}

async fn start_server(port: u16, grpc_port: u16) -> Result<()> {
    info!("🚀 Starting Engram server...");

    let actual_port = find_available_port(port).await?;
    let actual_grpc_port = find_available_port(grpc_port).await?;

    if actual_port != port {
        warn!(
            "⚠️  Port {} occupied, using port {} instead",
            port, actual_port
        );
    }
    if actual_grpc_port != grpc_port {
        warn!(
            "⚠️  gRPC port {} occupied, using port {} instead",
            grpc_port, actual_grpc_port
        );
    }

    // Initialize memory graph
    let memory_graph = Arc::new(tokio::sync::RwLock::new(MemoryGraph::new()));

    // Create API state
    let api_state = ApiState::new(memory_graph.clone());

    // Build HTTP API routes
    let app = create_api_routes().with_state(api_state).layer(
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any),
    );

    // Start HTTP server
    let addr = SocketAddr::from(([127, 0, 0, 1], actual_port));
    let listener = TcpListener::bind(addr).await?;

    info!(
        "🌐 HTTP API server listening on http://127.0.0.1:{}",
        actual_port
    );
    info!(
        "🔌 gRPC server would listen on 127.0.0.1:{}",
        actual_grpc_port
    );

    // Write PID file for server management
    write_pid_file(actual_port)?;

    println!("✅ Engram server started successfully!");
    println!("🌐 HTTP API: http://127.0.0.1:{}", actual_port);
    println!("🔌 gRPC: 127.0.0.1:{}", actual_grpc_port);
    println!("🩺 Health: http://127.0.0.1:{}/health", actual_port);
    println!("📖 API Docs: http://127.0.0.1:{}/docs", actual_port);
    println!("");
    println!("💡 Use 'engram status' to check server health");
    println!("🛑 Use 'engram stop' to shutdown the server");

    // Start server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Cleanup on exit
    remove_pid_file()?;
    info!("👋 Server stopped gracefully");

    Ok(())
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

async fn handle_config_command(action: ConfigAction) -> Result<()> {
    match action {
        ConfigAction::Get { key } => match key.as_str() {
            "network.port" => println!("7432"),
            "network.grpc_port" => println!("50051"),
            _ => {
                println!("Unknown configuration key: {}", key);
                std::process::exit(1);
            }
        },
        ConfigAction::Set { key, value } => {
            println!("Setting {} = {}", key, value);
            println!("⚠️  Configuration setting not yet implemented");
        }
        ConfigAction::List { section } => match section.as_deref() {
            Some("memory") => {
                println!("memory.cache_size=100MB");
                println!("memory.gc_threshold=0.7");
            }
            Some("network") => {
                println!("network.port=7432");
                println!("network.grpc_port=50051");
            }
            None => {
                println!("[network]");
                println!("port=7432");
                println!("grpc_port=50051");
                println!();
                println!("[memory]");
                println!("cache_size=100MB");
                println!("gc_threshold=0.7");
            }
            Some(s) => {
                println!("Unknown section: {}", s);
            }
        },
        ConfigAction::Path => {
            println!(
                "{}/config.toml",
                cli::server::pid_file_path().parent().unwrap().display()
            );
        }
    }
    Ok(())
}

async fn start_interactive_shell() -> Result<()> {
    println!("🐚 Engram Interactive Shell");
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
                    eprintln!("❌ {}", e);
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                break;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("❌ Error: {}", err);
                break;
            }
        }
    }

    println!("👋 Goodbye!");
    Ok(())
}

fn print_shell_help() {
    println!("📋 Available Commands:");
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

    match parts[0] {
        "status" => show_status().await,
        "create" => {
            if parts.len() < 2 {
                eprintln!("Usage: create <content>");
                return Ok(());
            }
            let content = parts[1..].join(" ");
            let (port, _) = get_server_connection().await?;
            create_memory(port, content, None).await
        }
        "get" => {
            if parts.len() != 2 {
                eprintln!("Usage: get <id>");
                return Ok(());
            }
            let (port, _) = get_server_connection().await?;
            get_memory(port, parts[1].to_string()).await
        }
        "search" => {
            if parts.len() < 2 {
                eprintln!("Usage: search <query>");
                return Ok(());
            }
            let query = parts[1..].join(" ");
            let (port, _) = get_server_connection().await?;
            search_memories(port, query, None).await
        }
        "list" => {
            let (port, _) = get_server_connection().await?;
            list_memories(port, Some(10), None).await
        }
        _ => {
            eprintln!("❌ Unknown command: {}", parts[0]);
            eprintln!("💡 Type 'help' for available commands");
            Ok(())
        }
    }
}

async fn handle_benchmark_command(
    operations: usize,
    concurrent: usize,
    hyperfine: bool,
    operation: String,
) -> Result<()> {
    println!(
        "🚀 Starting benchmark with {} operations, {} concurrent connections",
        operations, concurrent
    );

    if hyperfine {
        println!("⚠️  Hyperfine benchmarking not implemented for memory operations");
        println!("💡 Use the built-in benchmark instead");
        return Ok(());
    }

    // For now, just validate that the server is running
    let (_port, _grpc_port) = get_server_connection().await?;

    println!("✅ Server connection verified");
    println!("⚠️  Full memory operation benchmarking not yet implemented");
    println!(
        "💡 This would benchmark {} operations of type '{}'",
        operations, operation
    );

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
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    info!("🛑 Shutdown signal received");
}

async fn handle_docs_command(
    section: Option<String>,
    list: bool,
    export: Option<String>,
) -> Result<()> {
    if list {
        println!("📚 Available Documentation Sections:");
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
            println!("📖 {:<15} - {}", name, description);
        }
        println!("\nUsage: engram docs <section>");
        return Ok(());
    }

    let content = if let Some(section_name) = section {
        match section_name.parse::<DocSection>() {
            Ok(section_type) => OperationalDocs::get_section(section_type),
            Err(e) => {
                eprintln!("❌ {}", e);
                eprintln!("💡 Use 'engram docs --list' to see available sections");
                return Ok(());
            }
        }
    } else {
        OperationalDocs::complete_guide()
    };

    if let Some(export_path) = export {
        std::fs::write(&export_path, content)?;
        println!("📝 Documentation exported to: {}", export_path);
    } else {
        println!("{}", content);
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

    let (pid, port) = match read_pid_file() {
        Ok(info) => info,
        Err(_) => {
            println!(
                r#"{{"status": "error", "health": "corrupted", "memory_count": 0, "message": "Server info corrupted"}}"#
            );
            return Ok(());
        }
    };

    if !is_process_running(pid) {
        println!(
            r#"{{"status": "offline", "health": "process_dead", "memory_count": 0, "pid": {}, "port": {}, "message": "Process not running"}}"#,
            pid, port
        );
        return Ok(());
    }

    // Try to get health status
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    let health_url = format!("http://127.0.0.1:{}/health/alive", port);

    match client.get(&health_url).send().await {
        Ok(response) if response.status().is_success() => {
            println!(
                r#"{{"status": "online", "health": "responsive", "memory_count": 0, "pid": {}, "port": {}}}"#,
                pid, port
            );
        }
        Ok(_) => {
            println!(
                r#"{{"status": "degraded", "health": "unresponsive", "memory_count": 0, "pid": {}, "port": {}}}"#,
                pid, port
            );
        }
        Err(_) => {
            println!(
                r#"{{"status": "offline", "health": "unreachable", "memory_count": 0, "pid": {}, "port": {}}}"#,
                pid, port
            );
        }
    }

    Ok(())
}
