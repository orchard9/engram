//! Error handling module with recovery strategies
//!
//! Re-exports core error types and recovery utilities

pub mod recovery;

pub use recovery::{ErrorRecovery, ResultExt, unreachable_pattern};