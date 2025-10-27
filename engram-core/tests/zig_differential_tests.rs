//! Differential testing harness for Zig kernels vs. Rust baseline implementations.
//!
//! This test suite uses property-based testing to validate that Zig SIMD kernels
//! produce numerically identical results to the reference Rust implementations.
//!
//! # Test Organization
//!
//! Tests are organized in the `zig_differential/` module directory:
//! - `mod.rs` - Test utilities and helpers
//! - `vector_similarity.rs` - Cosine similarity differential tests
//! - `spreading_activation.rs` - Graph spreading differential tests
//! - `decay_functions.rs` - Memory decay differential tests
//!
//! # Running Tests
//!
//! Run all differential tests:
//! ```bash
//! cargo test --package engram-core --test zig_differential_tests
//! ```
//!
//! Run with Zig kernels enabled (for full differential testing):
//! ```bash
//! cargo test --package engram-core --test zig_differential_tests --features zig-kernels
//! ```
//!
//! # Expected Behavior
//!
//! Currently (Task 002), Zig kernels are stubs:
//! - `vector_similarity` returns all zeros
//! - `spread_activation` is a no-op
//!
//! ALL TESTS IN THIS FILE ARE EXPECTED TO FAIL until Zig kernel implementations
//! are complete. Tests are kept to document expected behavior.

#![cfg_attr(not(feature = "zig-kernels"), allow(dead_code, unused_imports))]
//! - `apply_decay` is a no-op
//!
//! These tests will FAIL until Tasks 005-007 implement actual kernels.
//! This is expected and validates the differential testing framework.

// Allow test-specific clippy lints
#![allow(clippy::float_cmp)]
#![allow(clippy::unseparated_literal_suffix)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::trivially_copy_pass_by_ref)]

#[path = "zig_differential/mod.rs"]
#[cfg(feature = "zig-kernels")] // Only run tests when Zig kernels are implemented
mod zig_differential;
