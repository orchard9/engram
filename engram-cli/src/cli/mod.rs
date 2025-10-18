//! CLI module organization

pub mod commands;
pub mod memory;
pub mod server;
pub mod status;

pub use commands::{Cli, Commands, ConfigAction, MemoryAction, OutputFormat};
pub use memory::{create_memory, delete_memory, get_memory, list_memories, search_memories};
pub use server::{get_server_connection, remove_pid_file, stop_server, write_pid_file};
pub use status::show_status;
