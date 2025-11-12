//! Backend implementations for memory storage

mod dashmap;
mod hashmap;
mod infallible;

#[cfg(feature = "dual_memory_types")]
mod dual_dashmap;

pub use dashmap::DashMapBackend;
pub use hashmap::HashMapBackend;
pub use infallible::InfallibleBackend;

#[cfg(feature = "dual_memory_types")]
pub use dual_dashmap::{DUAL_MEMORY_NODE_SIZE, DualDashMapBackend};
