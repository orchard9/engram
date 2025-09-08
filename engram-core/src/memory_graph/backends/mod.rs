//! Backend implementations for memory storage

mod dashmap;
mod hashmap;
mod infallible;

pub use dashmap::DashMapBackend;
pub use hashmap::HashMapBackend;
pub use infallible::InfallibleBackend;