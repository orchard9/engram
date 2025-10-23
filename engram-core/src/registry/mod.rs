//! Memory space registry orchestrating multi-tenant control plane state.

pub mod error;
pub mod memory_space;

pub use error::MemorySpaceError;
pub use memory_space::{MemorySpaceRegistry, SpaceDirectories, SpaceHandle, SpaceSummary};
