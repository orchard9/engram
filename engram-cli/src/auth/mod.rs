//! HTTP and gRPC authentication middleware for Engram CLI.
//!
//! This module provides authentication enforcement for the HTTP API server,
//! integrating with the core auth module for API key validation.

pub mod middleware;

pub use middleware::{require_api_key, require_permission, security_headers};
