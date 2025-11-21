//! HTTP and gRPC authentication middleware for Engram CLI.
//!
//! This module provides authentication enforcement for both HTTP and gRPC servers,
//! integrating with the core auth module for API key validation.

pub mod grpc;
pub mod middleware;

pub use grpc::AuthInterceptor;
pub use middleware::{require_api_key, require_permission, security_headers};
