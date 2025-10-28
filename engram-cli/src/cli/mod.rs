//! CLI module organization

#![allow(missing_docs)]

pub mod backup;
pub mod benchmark_ops;
pub mod commands;
pub mod diagnose;
pub mod memory;
pub mod migrate;
pub mod restore;
pub mod server;
pub mod space;
pub mod status;
pub mod validate;

pub use commands::{
    BackupAction, BenchmarkAction, Cli, Commands, ConfigAction, DiagnoseAction, MemoryAction,
    MigrateAction, OutputFormat, RestoreAction, SpaceAction, ValidateAction,
};
pub use memory::{create_memory, delete_memory, get_memory, list_memories, search_memories};
pub use server::{get_server_connection, remove_pid_file, stop_server, write_pid_file};
pub use space::{create_space, list_spaces};
pub use status::show_status;
